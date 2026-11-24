"""End-to-end distillation-free training for MemoChain (no Memorilla teacher).

Trains the 22 KB Q14 Delta-rule writer + memory reader directly through a
frozen open causal LLM (e.g., Qwen/Qwen3-0.6B or Qwen/Qwen3-8B) using pure
language-modeling cross-entropy loss on target answers:

  docs [B, N, 2560] -> MemoChainStudent.fold_sequence (STE Q14 int16) -> 22 KB state
  (state, question_embed) -> MemoChainStudent.read_tokens -> [B, M, hidden_size]
  replace <|memory|> token embeddings inside frozen LLM prompt -> LM loss on answer

Because gradients flow straight from the frozen LLM's cross-entropy loss into
the reader and through the Straight-Through Estimator into the Delta-rule
writer, MemoChain learns its own 22 KB memory representations from scratch
with zero dependency on Memorilla's teacher weights or architecture.

Any model trained here exports directly to:
  - student.pt         (PyTorch checkpoint, same format as MemoChainStudent)
  - weights.bin        (3.6 MB Q14 int16 writer for Rust/SP1 zkVM proving)
  - np_*.npy           (torch-free numpy writer arrays)
  - reader.npz         (torch-free numpy reader arrays)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .eval_qa import MEMORY_TOKEN, PERSONA_PROMPT, build_memory_messages
from .student import MemoChainStudent, StudentConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_OK = False


def _require_torch() -> None:
    if not TORCH_OK:
        raise ImportError("PyTorch is required for end-to-end training.")


def build_training_example(
    tokenizer: Any,
    question: str,
    answer: str,
    num_memories: int,
    memory_token_id: int,
    max_prompt_tokens: int = 384,
    max_answer_tokens: int = 128,
    system_prompt: str = PERSONA_PROMPT,
) -> tuple[list[int], list[int]]:
    """Build (input_ids, labels) with prompt masked to -100 and answer supervised.

    Guarantees all ``num_memories`` <|memory|> placeholder tokens are preserved
    even if the question is truncated.
    """
    messages = build_memory_messages(question, num_memories, system_prompt=system_prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_tokens:
        # Keep the prefix containing all <|memory|> tokens intact, trim only trailing question text.
        mem_positions = [i for i, tid in enumerate(prompt_ids) if tid == memory_token_id]
        min_keep = (mem_positions[-1] + 2) if mem_positions else 0
        cut = max(min_keep, max_prompt_tokens)
        prompt_ids = prompt_ids[:cut]

    ans_text = str(answer).strip()
    eos = tokenizer.eos_token or ""
    ans_ids = tokenizer.encode(ans_text + eos, add_special_tokens=False)[:max_answer_tokens]
    if not ans_ids:
        ans_ids = [tokenizer.eos_token_id]

    input_ids = prompt_ids + ans_ids
    labels = [-100] * len(prompt_ids) + ans_ids
    return input_ids, labels


def inject_memory_embeddings(
    base_embeds: "torch.Tensor",
    memory_tokens: "torch.Tensor",
    input_ids: "torch.Tensor",
    memory_token_id: int,
) -> "torch.Tensor":
    """Differentiably replace <|memory|> embeddings with student memory_tokens.

    base_embeds:   [B, T, H] (from frozen LLM embed_tokens)
    memory_tokens: [B, M, H] (from MemoChainStudent.read_tokens; keeps grad!)
    input_ids:     [B, T]
    """
    bsz, seqlen, hidden = base_embeds.shape
    num_mem = memory_tokens.shape[1]
    flat_base = base_embeds.reshape(bsz * seqlen, hidden)
    flat_mem = memory_tokens.reshape(bsz * num_mem, hidden).to(dtype=base_embeds.dtype)
    positions = (input_ids.reshape(-1) == memory_token_id).nonzero(as_tuple=False).squeeze(1)
    expected = bsz * num_mem
    if positions.numel() != expected:
        raise ValueError(f"Expected {expected} <|memory|> tokens, found {positions.numel()}")
    # Out-of-place scatter so autograd flows cleanly into memory_tokens.
    out = flat_base.clone()
    out.index_copy_(0, positions, flat_mem)
    return out.view(bsz, seqlen, hidden)


def collate_qa_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
    device: "torch.device",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Right-pad (input_ids, attention_mask, labels) for causal LM forward."""
    max_len = max(len(ids) for ids, _ in examples)
    bsz = len(examples)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, labs) in enumerate(examples):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(labs, dtype=torch.long)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


class E2EDatasetBatches:
    """Samples (doc_embeddings, doc_padding, question_embeddings, question_texts, answers)."""

    def __init__(
        self,
        data_dir: str | Path,
        datasets: str | list[str],
        split: str,
        max_docs: int,
        seed: int = 0,
    ) -> None:
        import pyarrow.parquet as pq

        from .teacher import collection_embeds, load_embedding_store

        self.max_docs = max_docs
        self.rng = np.random.default_rng(seed)
        self.datasets = [datasets] if isinstance(datasets, str) else list(datasets)

        root = Path(data_dir)
        # (ds, local_row, cid, doc_end_idx_or_None, question, answer)
        self.rows: list[tuple[str, int, int, int | None, str, str]] = []
        self.rows_per_ds: dict[str, list[int]] = {}
        self.collection_docs: dict[tuple[str, int], np.ndarray] = {}
        self.stores: dict[str, dict] = {}
        self.embed_dim: int = 2560

        for ds in self.datasets:
            files = sorted((root / "data" / ds).glob(f"{split}-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No {ds}/{split} parquet under {root / 'data' / ds}")
            store = load_embedding_store(ds, split, root)
            self.stores[ds] = store
            self.embed_dim = int(store["questions"].shape[1])

            ds_indices: list[int] = []
            local_row = 0
            for f in files:
                schema_cols = set(pq.read_schema(f).names)
                cols = ["collection_id", "question", "answer"]
                has_doc_end = "doc_end_idx" in schema_cols
                if has_doc_end:
                    cols.append("doc_end_idx")
                tbl = pq.read_table(f, columns=cols)
                cids = tbl["collection_id"].to_pylist()
                qs = tbl["question"].to_pylist()
                anss = tbl["answer"].to_pylist()
                doc_ends = tbl["doc_end_idx"].to_pylist() if has_doc_end else [None] * len(cids)
                for cid_raw, q, a, d_end in zip(cids, qs, anss, doc_ends):
                    cid = int(cid_raw)
                    a_str = str(a).strip()
                    q_str = str(q).strip()
                    end_int = int(d_end) if d_end is not None else None
                    if a_str and q_str:
                        idx = len(self.rows)
                        self.rows.append((ds, local_row, cid, end_int, q_str, a_str))
                        ds_indices.append(idx)
                        key = (ds, cid)
                        if key not in self.collection_docs:
                            self.collection_docs[key] = collection_embeds(store, cid)
                    local_row += 1
            self.rows_per_ds[ds] = ds_indices

    def sample(
        self, batch_size: int, device: "torch.device"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", list[str], list[str]]:
        chosen: list[int] = []
        for i in range(batch_size):
            ds = self.datasets[i % len(self.datasets)]
            pool = self.rows_per_ds[ds]
            chosen.append(int(pool[int(self.rng.integers(0, len(pool)))]))

        doc_lists: list[np.ndarray] = []
        q_vecs: list[np.ndarray] = []
        q_texts: list[str] = []
        a_texts: list[str] = []
        max_n = 0

        for idx in chosen:
            ds, local_row, cid, doc_end, q_str, a_str = self.rows[idx]
            docs = self.collection_docs[(ds, cid)]
            if doc_end is not None:
                docs = docs[: max(1, doc_end)]
            if self.max_docs > 0 and docs.shape[0] > self.max_docs:
                # Preserve chronological order when subsampling long histories.
                pick = np.sort(self.rng.choice(docs.shape[0], size=self.max_docs, replace=False))
                docs = docs[pick]
            doc_lists.append(docs)
            max_n = max(max_n, docs.shape[0])
            q_vecs.append(self.stores[ds]["questions"][local_row])
            q_texts.append(q_str)
            a_texts.append(a_str)

        bsz = len(chosen)
        docs_arr = np.zeros((bsz, max_n, self.embed_dim), dtype=np.float32)
        pad_arr = np.ones((bsz, max_n), dtype=bool)
        for i, d in enumerate(doc_lists):
            n = d.shape[0]
            docs_arr[i, :n] = d.astype(np.float32)
            pad_arr[i, :n] = False

        q_arr = np.stack([q.astype(np.float32) for q in q_vecs], axis=0)
        return (
            torch.from_numpy(docs_arr).to(device),
            torch.from_numpy(pad_arr).to(device),
            torch.from_numpy(q_arr).to(device),
            q_texts,
            a_texts,
        )


def compute_e2e_loss(
    student: MemoChainStudent,
    lm: "nn.Module",
    embed_tokens: "nn.Module",
    tokenizer: Any,
    memory_token_id: int,
    docs: "torch.Tensor",
    padding: "torch.Tensor",
    q_embeds: "torch.Tensor",
    q_texts: list[str],
    a_texts: list[str],
    *,
    quantize: bool = True,
    max_prompt_tokens: int = 384,
    max_answer_tokens: int = 128,
    zero_memory: bool = False,
    system_prompt: str = PERSONA_PROMPT,
) -> "torch.Tensor":
    """Forward docs -> 22 KB state -> memory tokens -> frozen LM cross-entropy loss."""
    device = docs.device
    if zero_memory:
        mem_tokens = torch.zeros(
            (docs.shape[0], student.config.num_memories, student.config.output_dim),
            device=device,
            dtype=torch.float32,
        )
    else:
        state = student.fold_sequence(docs, padding=padding, quantize=quantize)
        mem_tokens = student.read_tokens(state, q_embeds)  # [B, M, H]

    examples = [
        build_training_example(
            tokenizer,
            q,
            a,
            student.config.num_memories,
            memory_token_id,
            max_prompt_tokens=max_prompt_tokens,
            max_answer_tokens=max_answer_tokens,
            system_prompt=system_prompt,
        )
        for q, a in zip(q_texts, a_texts)
    ]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids, attention_mask, labels = collate_qa_batch(examples, pad_id, device)

    with torch.no_grad():
        clamped = input_ids.clamp(max=embed_tokens.num_embeddings - 1)
        base_embeds = embed_tokens(clamped)

    inputs_embeds = inject_memory_embeddings(base_embeds, mem_tokens, input_ids, memory_token_id)
    out = lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return out.loss


@torch.no_grad()
def evaluate_validation_nll(
    student: MemoChainStudent,
    lm: "nn.Module",
    embed_tokens: "nn.Module",
    tokenizer: Any,
    memory_token_id: int,
    val_batches: E2EDatasetBatches,
    *,
    num_batches: int = 20,
    batch_size: int = 4,
    device: "torch.device",
    max_prompt_tokens: int = 384,
    max_answer_tokens: int = 128,
    system_prompt: str = PERSONA_PROMPT,
) -> dict[str, float]:
    """Compare validation answer NLL with 22 KB memory vs zero-memory baseline."""
    was_training = student.training
    student.eval()
    # Use deterministic seed for repeatable validation slices.
    saved_state = val_batches.rng.bit_generator.state
    val_batches.rng = np.random.default_rng(12345)

    mem_losses: list[float] = []
    zero_losses: list[float] = []
    for _ in range(num_batches):
        docs, pad, q_emb, q_txt, a_txt = val_batches.sample(batch_size, device)
        l_mem = compute_e2e_loss(
            student, lm, embed_tokens, tokenizer, memory_token_id,
            docs, pad, q_emb, q_txt, a_txt,
            quantize=True,
            max_prompt_tokens=max_prompt_tokens,
            max_answer_tokens=max_answer_tokens,
            zero_memory=False,
            system_prompt=system_prompt,
        )
        l_zero = compute_e2e_loss(
            student, lm, embed_tokens, tokenizer, memory_token_id,
            docs, pad, q_emb, q_txt, a_txt,
            quantize=True,
            max_prompt_tokens=max_prompt_tokens,
            max_answer_tokens=max_answer_tokens,
            zero_memory=True,
            system_prompt=system_prompt,
        )
        mem_losses.append(float(l_mem.item()))
        zero_losses.append(float(l_zero.item()))

    val_batches.rng.bit_generator.state = saved_state
    if was_training:
        student.train()
    mem_mean = float(np.mean(mem_losses))
    zero_mean = float(np.mean(zero_losses))
    return {
        "val_nll": round(mem_mean, 4),
        "val_zero_nll": round(zero_mean, 4),
        "val_nll_gain": round(zero_mean - mem_mean, 4),
    }


def export_student_artifacts(student: MemoChainStudent, save_dir: Path) -> dict[str, Any]:
    """Save student.pt plus torch-free numpy/SP1 Q14 artifacts.

    weights.bin/np_*.npy are only exported for the canonical 11x32x32 layout
    (the SP1 wire format is fixed); other capacity ablation shapes still get
    student.pt + reader.npz and a recorded state_bytes figure.
    """
    from .state import KEY_DIM as CANON_K, N_HEADS as CANON_H, VAL_DIM as CANON_V

    save_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    ckpt = {
        "config": student.config.__dict__,
        "state_dict": cpu_state,
        "training_mode": "e2e_no_teacher",
    }
    torch.save(ckpt, save_dir / "student.pt")

    cfg = student.config
    state_bytes = cfg.num_heads * cfg.key_dim * cfg.val_dim * 2
    info: dict[str, Any] = {"state_bytes": state_bytes}

    if (cfg.num_heads, cfg.key_dim, cfg.val_dim) == (CANON_H, CANON_K, CANON_V):
        from .student_int import IntStudentWriter

        writer = IntStudentWriter.from_checkpoint(save_dir / "student.pt")
        weights_raw = writer.to_bytes()
        (save_dir / "weights.bin").write_bytes(weights_raw)
        np.save(save_dir / "np_key_w.npy", writer.w_key)
        np.save(save_dir / "np_key_b.npy", writer.b_key)
        np.save(save_dir / "np_val_w.npy", writer.w_val)
        np.save(save_dir / "np_val_b.npy", writer.b_val)
        np.save(save_dir / "np_beta.npy", writer.beta)
        np.save(save_dir / "np_decay.npy", writer.decay)
        info["weights_digest"] = writer.weights_digest()
        info["weights_bytes"] = len(weights_raw)

    # Export numpy reader head
    np.savez(
        save_dir / "reader.npz",
        norm_w=student.reader_head_norm.weight.detach().cpu().float().numpy(),
        norm_b=student.reader_head_norm.bias.detach().cpu().float().numpy(),
        slots_w=student.reader_slots.weight.detach().cpu().float().numpy(),
        slots_b=student.reader_slots.bias.detach().cpu().float().numpy(),
        out_w=student.reader_out.weight.detach().cpu().float().numpy(),
        out_b=student.reader_out.bias.detach().cpu().float().numpy(),
    )
    return info


def train_e2e(args: argparse.Namespace) -> dict[str, Any]:
    """End-to-end training loop: frozen causal LM loss -> 22 KB student."""
    _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[e2e] loading frozen decoder {args.decoder} on {device} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [MEMORY_TOKEN]})
    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)

    lm = AutoModelForCausalLM.from_pretrained(args.decoder, torch_dtype=dtype).to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    if args.gradient_checkpointing and hasattr(lm, "gradient_checkpointing_enable"):
        lm.gradient_checkpointing_enable()
        if hasattr(lm, "enable_input_require_grads"):
            lm.enable_input_require_grads()

    embed_tokens = lm.get_input_embeddings()
    hidden_size = int(lm.config.hidden_size)
    # Scale initial reader_out so injected memory tokens match the decoder's embedding norm.
    with torch.no_grad():
        emb_std = float(embed_tokens.weight.float().std().item())

    train_batches = E2EDatasetBatches(
        args.data_dir, args.dataset, "train", max_docs=args.num_docs, seed=args.seed
    )
    val_batches = None
    try:
        val_batches = E2EDatasetBatches(
            args.data_dir, args.dataset, "validation", max_docs=args.num_docs, seed=args.seed + 1
        )
    except FileNotFoundError:
        pass

    scfg = StudentConfig(
        embedding_dim=train_batches.embed_dim,
        output_dim=hidden_size,
        num_memories=args.num_memories,
        num_heads=args.num_heads,
        key_dim=args.key_dim,
        val_dim=args.key_dim,
        initial_beta=args.beta,
        initial_decay=args.decay,
    )
    student = MemoChainStudent(scfg).to(device)
    if args.init:
        ckpt = torch.load(Path(args.init) / "student.pt", map_location=device, weights_only=False)
        student.load_state_dict(ckpt["state_dict"])
        print(f"[e2e] warm-started from {args.init}", flush=True)
    else:
        with torch.no_grad():
            student.reader_out.weight.mul_(emb_std / max(float(student.reader_out.weight.std().item()), 1e-4))
            student.reader_out.bias.zero_()

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    state_kb = scfg.num_heads * scfg.key_dim * scfg.val_dim * 2 / 1024.0
    print(
        f"[e2e] student params={n_params/1e6:.2f}M | embed_dim={scfg.embedding_dim} "
        f"| output_dim={scfg.output_dim} | num_memories={scfg.num_memories} "
        f"| state={scfg.num_heads}x{scfg.key_dim}x{scfg.val_dim} ({state_kb:.1f} KiB) "
        f"| emb_std={emb_std:.4f} | train_rows={len(train_batches.rows)}",
        flush=True,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(100, args.steps // 10))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    save_dir = Path(args.save) if args.save else None
    best_val_nll = float("inf")
    t0 = time.perf_counter()
    running_loss = 0.0
    history: list[dict[str, Any]] = []

    sys_prompt = getattr(args, "system_prompt", PERSONA_PROMPT)
    # Baseline validation before training
    if val_batches is not None and args.eval_every > 0:
        v0 = evaluate_validation_nll(
            student, lm, embed_tokens, tokenizer, memory_token_id, val_batches,
            num_batches=args.eval_batches, batch_size=args.batch, device=device,
            max_prompt_tokens=args.max_prompt_tokens, max_answer_tokens=args.max_answer_tokens,
            system_prompt=sys_prompt,
        )
        print(
            f"[e2e] step 0 | val_nll={v0['val_nll']:.4f} | zero_nll={v0['val_zero_nll']:.4f} "
            f"| gain={v0['val_nll_gain']:+.4f}",
            flush=True,
        )
        history.append({"step": 0, **v0})

    student.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            docs, pad, q_emb, q_txt, a_txt = train_batches.sample(args.batch, device)
            loss = compute_e2e_loss(
                student, lm, embed_tokens, tokenizer, memory_token_id,
                docs, pad, q_emb, q_txt, a_txt,
                quantize=not args.no_quant,
                max_prompt_tokens=args.max_prompt_tokens,
                max_answer_tokens=args.max_answer_tokens,
                system_prompt=sys_prompt,
            )
            (loss / args.grad_accum).backward()
            step_loss += float(loss.item()) / args.grad_accum

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        running_loss += step_loss

        if step % args.log_every == 0 or step == args.steps:
            avg_loss = running_loss / (args.log_every if step % args.log_every == 0 else (step % args.log_every))
            running_loss = 0.0
            elapsed = time.perf_counter() - t0
            b_mean = float(student.beta.mean().item())
            d_mean = float(student.decay.mean().item())
            vram = (
                f" vram={torch.cuda.max_memory_allocated(device)/1048576:.0f}MiB"
                if device.type == "cuda"
                else ""
            )
            print(
                f"[e2e] step {step:5d}/{args.steps} | loss={avg_loss:.4f} "
                f"| beta={b_mean:.3f} decay={d_mean:.4f} | lr={sched.get_last_lr()[0]:.2e} "
                f"| {elapsed/step:.3f}s/step{vram}",
                flush=True,
            )

        if val_batches is not None and args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            val = evaluate_validation_nll(
                student, lm, embed_tokens, tokenizer, memory_token_id, val_batches,
                num_batches=args.eval_batches, batch_size=args.batch, device=device,
                max_prompt_tokens=args.max_prompt_tokens, max_answer_tokens=args.max_answer_tokens,
                system_prompt=sys_prompt,
            )
            print(
                f"[e2e] EVAL step {step:5d} | val_nll={val['val_nll']:.4f} "
                f"| zero_nll={val['val_zero_nll']:.4f} | gain={val['val_nll_gain']:+.4f}",
                flush=True,
            )
            history.append({"step": step, **val})
            if save_dir is not None and val["val_nll"] < best_val_nll:
                best_val_nll = val["val_nll"]
                export_student_artifacts(student, save_dir)
                print(f"[e2e] saved new best checkpoint to {save_dir} (val_nll={best_val_nll:.4f})", flush=True)

    meta: dict[str, Any] = {
        "training_mode": "e2e_no_teacher",
        "decoder": args.decoder,
        "datasets": args.dataset,
        "steps": args.steps,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "num_docs": args.num_docs,
        "num_memories": args.num_memories,
        "num_heads": scfg.num_heads,
        "key_dim": scfg.key_dim,
        "val_dim": scfg.val_dim,
        "state_bytes": scfg.num_heads * scfg.key_dim * scfg.val_dim * 2,
        "fact_slots": scfg.num_heads * scfg.key_dim,
        "lr": args.lr,
        "beta_learned": [round(float(x), 4) for x in student.beta.detach().cpu().tolist()],
        "decay_learned": [round(float(x), 5) for x in student.decay.detach().cpu().tolist()],
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "history": history,
    }
    if save_dir is not None:
        if best_val_nll == float("inf"):
            exp = export_student_artifacts(student, save_dir)
        else:
            # Ensure weights_digest is reported for the saved best checkpoint.
            best_ckpt = torch.load(save_dir / "student.pt", map_location="cpu", weights_only=False)
            best_student = MemoChainStudent(StudentConfig(**best_ckpt["config"]))
            best_student.load_state_dict(best_ckpt["state_dict"])
            exp = export_student_artifacts(best_student, save_dir)
        meta.update(exp)
        (save_dir / "train_e2e_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end distillation-free training for MemoChain (no Memorilla teacher)"
    )
    p.add_argument("--decoder", default="Qwen/Qwen3-0.6B", help="Frozen causal LM decoder")
    p.add_argument("--system-prompt", default=PERSONA_PROMPT, help="System prompt preceding the memory slots")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", nargs="+", default=["pv4"])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-docs", type=int, default=64)
    p.add_argument("--num-memories", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=11, help="state heads (capacity ablation)")
    p.add_argument("--key-dim", type=int, default=32, help="key/value dims (capacity ablation; val_dim == key_dim)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--decay", type=float, default=0.999)
    p.add_argument("--max-prompt-tokens", type=int, default=384)
    p.add_argument("--max-answer-tokens", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--no-quant", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--init", default=None)
    p.add_argument("--save", default="runs/e2e_qwen06b_pv4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_e2e(args)


if __name__ == "__main__":
    main()
