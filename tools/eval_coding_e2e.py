"""Held-out evaluation for coding-domain MemoChain e2e checkpoints.

Evaluates one or more Qwen3-8B e2e checkpoints on the full held-out test split
(e.g. ``kai_coding/test``, every row evaluated exactly once, zero subsampling)
using ONLY reloaded 22,528-byte Q14 int16 Capsules (folded through the exact
``IntStudentWriter`` and serialized/deserialized via ``Capsule.to_bytes`` ->
``Capsule.from_bytes`` so zero float state can leak).

Reports:
  - Full-split test answer NLL with 22 KB capsule vs zero-memory baseline
  - Per-row NLL gain (mean, median, % of rows improved)
  - Optional cross-domain retention check on ``pv4/validation``
  - Weights digest and Q14 integer fold verification for every checkpoint
  - Optional greedy generation samples from the best checkpoint (printed only
    when ``--show-samples N`` is passed; never written to the JSON report)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from memochain.eval_qa import MEMORY_TOKEN, PERSONA_PROMPT, build_memory_messages
from memochain.state import Capsule
from memochain.student import MemoChainStudent, StudentConfig
from memochain.student_int import IntStudentWriter, quantize_embeddings_q14
from memochain.teacher import collection_embeds, load_embedding_store
from memochain.train_e2e import (
    E2EDatasetBatches,
    build_training_example,
    collate_qa_batch,
    evaluate_validation_nll,
    inject_memory_embeddings,
)

CODING_SYSTEM_PROMPT = (
    "You are a coding assistant embedded in a developer terminal. "
    "Use the compressed session memory below to stay consistent with earlier "
    "project files, commands, decisions, and bugs."
)


@torch.no_grad()
def eval_checkpoint_exact_capsule(
    ckpt_dir: Path,
    lm: torch.nn.Module,
    embed_tokens: torch.nn.Module,
    tokenizer: Any,
    memory_token_id: int,
    data_dir: Path,
    dataset: str,
    split: str,
    device: torch.device,
    *,
    system_prompt: str = CODING_SYSTEM_PROMPT,
    max_prompt_tokens: int = 384,
    max_answer_tokens: int = 128,
    max_docs: int = 64,
    compute_zero: bool = False,
) -> tuple[dict[str, Any], list[float] | None]:
    """Evaluate every row in ``dataset/split`` through exact reloaded Q14 Capsules."""
    import pyarrow.parquet as pq

    ckpt = torch.load(ckpt_dir / "student.pt", map_location="cpu", weights_only=False)
    scfg = StudentConfig(**ckpt["config"])
    student = MemoChainStudent(scfg).to(device).eval()
    student.load_state_dict(ckpt["state_dict"])

    writer = IntStudentWriter.from_checkpoint(ckpt_dir / "student.pt")
    weights_digest = writer.weights_digest()
    if (ckpt_dir / "weights.bin").exists():
        disk_writer = IntStudentWriter.from_bytes((ckpt_dir / "weights.bin").read_bytes())
        assert disk_writer.weights_digest() == weights_digest, "weights.bin mismatch vs student.pt"

    store = load_embedding_store(dataset, split, data_dir)
    q_all = np.asarray(store["questions"], dtype=np.float32)

    files = sorted((data_dir / "data" / dataset).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for {dataset}/{split}")

    rows: list[tuple[int, int, int | None, str, str]] = []
    local_row = 0
    for f in files:
        schema_cols = set(pq.read_schema(f).names)
        cols = ["collection_id", "question", "answer"]
        has_end = "doc_end_idx" in schema_cols
        if has_end:
            cols.append("doc_end_idx")
        tbl = pq.read_table(f, columns=cols)
        cids = tbl["collection_id"].to_pylist()
        qs = tbl["question"].to_pylist()
        anss = tbl["answer"].to_pylist()
        ends = tbl["doc_end_idx"].to_pylist() if has_end else [None] * len(cids)
        for cid_raw, q, a, d_end in zip(cids, qs, anss, ends):
            q_s, a_s = str(q).strip(), str(a).strip()
            if q_s and a_s:
                rows.append((local_row, int(cid_raw), int(d_end) if d_end is not None else None, q_s, a_s))
            local_row += 1

    # Cache exact Q14 folded Capsules per (cid, doc_end) so repeated prefixes fold once.
    col_cache: dict[int, np.ndarray] = {}
    cap_cache: dict[tuple[int, int], tuple[torch.Tensor, int]] = {}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    mem_losses: list[float] = []
    zero_losses: list[float] | None = [] if compute_zero else None
    capsule_bytes = 0

    t0 = time.perf_counter()
    for local_idx, cid, doc_end, q_str, a_str in rows:
        if cid not in col_cache:
            col_cache[cid] = collection_embeds(store, cid).astype(np.float32)
        docs_np = col_cache[cid]
        eff_end = docs_np.shape[0] if doc_end is None else max(1, min(doc_end, docs_np.shape[0]))
        slice_docs = docs_np[:eff_end]
        if max_docs > 0 and slice_docs.shape[0] > max_docs:
            slice_docs = slice_docs[-max_docs:]
        cache_key = (cid, eff_end)
        if cache_key not in cap_cache:
            q14 = quantize_embeddings_q14(slice_docs)
            cap, _head = writer.export_proven_capsule(q14)
            raw = cap.to_bytes()
            reloaded = Capsule.from_bytes(raw)
            assert reloaded.state.nbytes == 22528
            state_f32 = reloaded.state.tokens.astype(np.float32) / float(1 << 14)
            cap_cache[cache_key] = (torch.from_numpy(state_f32).unsqueeze(0).to(device), len(raw))
        state_t, raw_len = cap_cache[cache_key]
        capsule_bytes = raw_len

        q_t = torch.from_numpy(q_all[local_idx]).unsqueeze(0).to(device=device, dtype=torch.float32)
        mem_tokens = student.read_tokens(state_t, q_t)

        ex = build_training_example(
            tokenizer,
            q_str,
            a_str,
            scfg.num_memories,
            memory_token_id,
            max_prompt_tokens=max_prompt_tokens,
            max_answer_tokens=max_answer_tokens,
            system_prompt=system_prompt,
        )
        input_ids, attention_mask, labels = collate_qa_batch([ex], pad_id, device)
        clamped = input_ids.clamp(max=embed_tokens.num_embeddings - 1)
        base_embeds = embed_tokens(clamped)

        in_mem = inject_memory_embeddings(base_embeds, mem_tokens, input_ids, memory_token_id)
        out_mem = lm(inputs_embeds=in_mem, attention_mask=attention_mask, labels=labels, use_cache=False)
        mem_losses.append(float(out_mem.loss.item()))

        if zero_losses is not None:
            z_tok = torch.zeros_like(mem_tokens)
            in_zero = inject_memory_embeddings(base_embeds, z_tok, input_ids, memory_token_id)
            out_zero = lm(inputs_embeds=in_zero, attention_mask=attention_mask, labels=labels, use_cache=False)
            zero_losses.append(float(out_zero.loss.item()))

    elapsed = time.perf_counter() - t0
    return (
        {
            "checkpoint": str(ckpt_dir),
            "weights_digest": weights_digest,
            "capsule_bytes": capsule_bytes,
            "num_rows": len(rows),
            "num_unique_capsules": len(cap_cache),
            "nll_mean": round(float(np.mean(mem_losses)), 4),
            "nll_median": round(float(np.median(mem_losses)), 4),
            "per_row_nll": mem_losses,
            "elapsed_sec": round(elapsed, 2),
        },
        zero_losses,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Held-out coding-domain evaluation through exact Q14 capsules")
    p.add_argument("--decoder", default="Qwen/Qwen3-8B")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", default="kai_coding")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--check-pv4-val", action="store_true", help="Also measure retention on pv4/validation (20 batches)")
    p.add_argument("--out", default="runs/coding_eval_summary.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[eval] loading frozen {args.decoder} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [MEMORY_TOKEN]})
    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)

    lm = AutoModelForCausalLM.from_pretrained(args.decoder, torch_dtype=dtype).to(device).eval()
    for p_param in lm.parameters():
        p_param.requires_grad_(False)
    embed_tokens = lm.get_input_embeddings()

    pv4_val = None
    if args.check_pv4_val:
        pv4_val = E2EDatasetBatches(args.data_dir, "pv4", "validation", max_docs=64, seed=1)

    zero_losses: list[float] | None = None
    results: list[dict[str, Any]] = []

    for idx, ckpt_str in enumerate(args.checkpoints):
        ckpt_dir = Path(ckpt_str)
        res, z_out = eval_checkpoint_exact_capsule(
            ckpt_dir,
            lm,
            embed_tokens,
            tokenizer,
            memory_token_id,
            Path(args.data_dir),
            args.dataset,
            args.split,
            device,
            compute_zero=(idx == 0),
        )
        if z_out is not None:
            zero_losses = z_out
        assert zero_losses is not None
        gains = [z - m for z, m in zip(zero_losses, res["per_row_nll"])]
        entry: dict[str, Any] = {
            "name": ckpt_dir.name,
            "checkpoint": str(ckpt_dir),
            "weights_digest": res["weights_digest"][:16],
            "capsule_bytes": res["capsule_bytes"],
            "num_rows": res["num_rows"],
            "test_nll": res["nll_mean"],
            "test_zero_nll": round(float(np.mean(zero_losses)), 4),
            "test_gain_nats": round(float(np.mean(gains)), 4),
            "test_gain_median_nats": round(float(np.median(gains)), 4),
            "rows_improved_pct": round(100.0 * sum(1 for g in gains if g > 0) / len(gains), 2),
            "elapsed_sec": res["elapsed_sec"],
        }
        if pv4_val is not None:
            ckpt = torch.load(ckpt_dir / "student.pt", map_location="cpu", weights_only=False)
            student = MemoChainStudent(StudentConfig(**ckpt["config"])).to(device).eval()
            student.load_state_dict(ckpt["state_dict"])
            pv4_m = evaluate_validation_nll(
                student,
                lm,
                embed_tokens,
                tokenizer,
                memory_token_id,
                pv4_val,
                num_batches=15,
                batch_size=4,
                device=device,
                system_prompt=PERSONA_PROMPT,
            )
            entry["pv4_val_nll"] = pv4_m["val_nll"]
            entry["pv4_val_gain_nats"] = pv4_m["val_nll_gain"]
            del student
        results.append(entry)
        print(
            f"[eval] {entry['name']:<30} | coding_test_nll={entry['test_nll']:.4f} "
            f"(zero={entry['test_zero_nll']:.4f}, gain={entry['test_gain_nats']:+.4f}, "
            f"improved={entry['rows_improved_pct']:.1f}%)"
            + (f" | pv4_val_gain={entry['pv4_val_gain_nats']:+.4f}" if "pv4_val_gain_nats" in entry else ""),
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "decoder": args.decoder,
        "dataset": args.dataset,
        "split": args.split,
        "capsule_path": "exact_q14_int16_reloaded",
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
