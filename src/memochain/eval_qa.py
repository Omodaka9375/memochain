"""End-to-end Qwen3-8B QA evaluation: closed-book vs Memorilla teacher vs 22 KB capsule.

Implements Memorilla's exact evaluation protocol (snap-stanford/memorilla
evaluate.py + evaluate_baselines.py + memorilla/metrics.py):

  - Decoder: Qwen/Qwen3-8B (bfloat16, greedy temperature=0, max_new_tokens=256,
    enable_thinking=False)
  - Prompt:  PERSONA_PROMPT + "\\nMemory" + "<|memory|>"*16 + "Memory" for
    memory methods; HELPFUL_PROMPT with no memory tokens for closed-book
  - Scorer:  Qwen/Qwen3-Embedding-4B nearest-neighbor cosine over each row's
    answer choices (reports accuracy, accuracy_hard, and teacher agreement)

For MemoChain, every collection is folded into a real Q14 int16 Capsule,
serialized to bytes (22,528-byte state payload), reloaded from bytes, and read
solely from that reloaded capsule so zero side-channel float state can leak.

Usage (on a GPU machine with vLLM, e.g. DGX Spark):
  PYTHONPATH=src python -m memochain.eval_qa \\
      --checkpoint runs/pv4_10k --teacher data/stage3 \\
      --dataset pv4 --split test --data-dir data --extra-doc-caps 64 \\
      --out runs/pv4_10k/qa_pv4_test.json
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

MEMORY_TOKEN = "<|memory|>"
PERSONA_PROMPT = (
    "You are a personalized AI assistant. Answer the question about the user "
    "based on your understanding of the user."
)
HELPFUL_PROMPT = "You are a helpful assistant."
DEFAULT_DECODER = "Qwen/Qwen3-8B"
DEFAULT_ENCODER = "Qwen/Qwen3-Embedding-4B"
CHAT_TOKEN_RE = re.compile(r"<\|(?:im_end|im_start|endoftext|im_sep)\|>")


def clean_generation(text: str) -> str:
    """Strip Qwen chat control tokens and surrounding whitespace."""
    return CHAT_TOKEN_RE.sub("", text).strip()


def memory_placeholder(num_memories: int) -> str:
    """Format the 16-slot memory placeholder block appended to the system prompt."""
    return f"\nMemory{MEMORY_TOKEN * num_memories}Memory"


def build_memory_messages(
    question: str, num_memories: int, system_prompt: str = PERSONA_PROMPT, suffix: str = ""
) -> list[dict[str, str]]:
    """Build chat messages with `<|memory|>` placeholders in the system prompt."""
    return [
        {"role": "system", "content": system_prompt + memory_placeholder(num_memories)},
        {"role": "user", "content": str(question).strip() + suffix},
    ]


def build_closed_book_messages(
    question: str, system_prompt: str = HELPFUL_PROMPT, suffix: str = ""
) -> list[dict[str, str]]:
    """Build text-only closed-book chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(question).strip() + suffix},
    ]


def choice_index(choices: Sequence[str], reference: str) -> int:
    """Locate the gold answer among a row's choices."""
    stripped = [c.strip() for c in choices]
    target = reference.strip()
    if target not in stripped:
        raise ValueError(f"Reference {reference!r} not found in choices {list(choices)}")
    return stripped.index(target)


def score_choices_from_vectors(
    pred_vecs: np.ndarray,
    choice_vecs_per_row: Sequence[np.ndarray],
    choices_per_row: Sequence[Sequence[str]],
    references: Sequence[str],
    hard_flags: Sequence[bool] | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Nearest-neighbor choice scoring matching memorilla.metrics.choice_accuracy.

    Returns (metrics_dict_in_0_to_1, selected_choice_indices).
    """
    selected = np.empty(len(references), dtype=np.int64)
    for idx, (pvec, cmat, texts) in enumerate(
        zip(pred_vecs, choice_vecs_per_row, choices_per_row)
    ):
        pnorm = pvec.astype(np.float32)
        pnorm = pnorm / (np.linalg.norm(pnorm) + 1e-8)
        cnorm = cmat.astype(np.float32)
        cnorm = cnorm / (np.linalg.norm(cnorm, axis=-1, keepdims=True) + 1e-8)
        sims = np.full(len(texts), -np.inf, dtype=np.float32)
        valid_positions = [i for i, t in enumerate(texts) if t.strip()]
        sims[valid_positions] = pnorm @ cnorm.T
        selected[idx] = int(np.argmax(sims))

    gold = np.array(
        [choice_index(row, ref) for row, ref in zip(choices_per_row, references)],
        dtype=np.int64,
    )
    correct = selected == gold
    out: dict[str, float] = {"accuracy": float(correct.mean())}
    if hard_flags is not None:
        mask = np.asarray(hard_flags, dtype=bool)
        if mask.any():
            out["accuracy_hard"] = float(correct[mask].mean())
    return out, selected


def _load_split_rows(
    data_dir: Path,
    dataset: str,
    split: str,
    max_rows: int = 0,
    collection_id: int | None = None,
) -> dict:
    import pyarrow.parquet as pq

    files = sorted((data_dir / "data" / dataset).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for {dataset}/{split} under {data_dir / 'data' / dataset}")
    cols = ["collection_id", "question", "answer", "choices"]
    schema_names = set(pq.read_schema(files[0]).names)
    if "hard" in schema_names:
        cols.append("hard")

    tables = [pq.read_table(f, columns=cols) for f in files]
    cids, questions, answers, choices, hard = [], [], [], [], []
    has_hard = "hard" in cols
    for t in tables:
        cids.extend(int(x) for x in t["collection_id"].to_pylist())
        questions.extend(str(x) for x in t["question"].to_pylist())
        answers.extend(str(x) for x in t["answer"].to_pylist())
        choices.extend([list(map(str, row)) for row in t["choices"].to_pylist()])
        if has_hard:
            hard.extend(bool(x) for x in t["hard"].to_pylist())

    row_indices = list(range(len(questions)))
    if collection_id is not None:
        keep = [i for i, c in enumerate(cids) if c == collection_id]
        if not keep:
            raise ValueError(f"collection_id {collection_id} not found in {dataset}/{split}")
        row_indices = [row_indices[i] for i in keep]
        cids = [cids[i] for i in keep]
        questions = [questions[i] for i in keep]
        answers = [answers[i] for i in keep]
        choices = [choices[i] for i in keep]
        if has_hard:
            hard = [hard[i] for i in keep]

    if max_rows and max_rows < len(questions):
        row_indices = row_indices[:max_rows]
        cids = cids[:max_rows]
        questions = questions[:max_rows]
        answers = answers[:max_rows]
        choices = choices[:max_rows]
        if has_hard:
            hard = hard[:max_rows]

    return {
        "row_indices": row_indices,
        "collection_ids": cids,
        "questions": questions,
        "answers": answers,
        "choices": choices,
        "hard": hard if has_hard and any(hard) else None,
    }


def _load_input_embeddings(model_path: str, device: "torch.device") -> "torch.nn.Embedding":
    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    p = Path(model_path)
    if not p.exists():
        p = Path(snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors", "*.json"]))
    keys = ("model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight")
    for shard in sorted(p.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device=str(device)) as fh:
            avail = set(fh.keys())
            for k in keys:
                if k in avail:
                    weight = fh.get_tensor(k).to(dtype=torch.bfloat16)
                    return nn.Embedding.from_pretrained(weight, freeze=True).eval()
    raise FileNotFoundError(f"Could not find input embedding matrix in {p}")


def _inject_memory(
    inputs_embeds: "torch.Tensor",
    memory_tokens: "torch.Tensor",
    input_ids: "torch.Tensor",
    memory_token_id: int,
) -> "torch.Tensor":
    hidden = inputs_embeds.size(-1)
    flat_embeds = inputs_embeds.reshape(-1, hidden)
    flat_mem = memory_tokens.reshape(-1, hidden).to(dtype=inputs_embeds.dtype)
    positions = (input_ids.reshape(-1) == memory_token_id).nonzero(as_tuple=False).squeeze(1)
    expected = inputs_embeds.size(0) * memory_tokens.size(1)
    if positions.numel() != expected:
        raise ValueError(f"Expected {expected} <|memory|> positions, found {positions.numel()}")
    delta = flat_mem - flat_embeds.index_select(0, positions)
    flat_embeds = flat_embeds.scatter_add(0, positions.unsqueeze(1).expand(-1, hidden), delta)
    return flat_embeds.view_as(inputs_embeds)


def _exit_worker_cleanly(engine: object) -> None:
    """Stop vLLM EngineCore subprocesses fully before leaving this OS process.

    In vLLM V1, EngineCore runs in a child process and only shuts down cleanly
    when `llm_engine.engine_core.shutdown()` is called (otherwise `del engine`
    leaves a reference cycle and `os._exit` orphans the EngineCore child).
    """
    try:
        llm_engine = getattr(engine, "llm_engine", None)
        core = getattr(llm_engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception:
        pass
    del engine
    gc.collect()
    try:
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        pass
    import torch

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        deadline = time.time() + 30
        while time.time() < deadline:
            free, _ = torch.cuda.mem_get_info()
            if free > 0.85 * total:
                break
            time.sleep(0.5)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _generate_worker(cfg: dict) -> None:
    """Spawned worker 1: precompute capsules + teacher tokens, then decode all methods in vLLM."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from .export import fold_student_state, provenance_head
    from .state import Capsule
    from .student import MemoChainStudent, StudentConfig
    from .teacher import MemorillaTeacher, collection_embeds, load_embedding_store

    device = torch.device("cuda")
    data_dir = Path(cfg["data_dir"])
    rows = _load_split_rows(
        data_dir, cfg["dataset"], cfg["split"], cfg["max_rows"], cfg.get("collection_id")
    )
    store = load_embedding_store(cfg["dataset"], cfg["split"], data_dir)
    q_all = np.asarray(store["questions"], dtype=np.float32)[rows["row_indices"]]
    unique_cids = sorted(set(rows["collection_ids"]))
    n_rows = len(rows["questions"])

    print(f"[generate] {cfg['dataset']}/{cfg['split']}: {n_rows} rows across {len(unique_cids)} collections")

    ckpt = torch.load(Path(cfg["checkpoint"]) / "student.pt", map_location="cpu", weights_only=False)
    scfg = StudentConfig(**ckpt["config"])
    student = MemoChainStudent(scfg).to(device).eval()
    student.load_state_dict(ckpt["state_dict"])

    use_teacher = bool(cfg.get("teacher")) and str(cfg["teacher"]).lower() != "none"
    teacher = (
        MemorillaTeacher.from_pretrained(cfg["teacher"]).to(device=device, dtype=torch.bfloat16).eval()
        if use_teacher
        else None
    )

    rows_by_cid: dict[int, list[int]] = {cid: [] for cid in unique_cids}
    for r_idx, cid in enumerate(rows["collection_ids"]):
        rows_by_cid[cid].append(r_idx)

    def build_student_bank(doc_cap: int) -> tuple[torch.Tensor, int, int, float]:
        t0 = time.perf_counter()
        reloaded_states: dict[int, torch.Tensor] = {}
        sample_total_bytes = 0
        counts: list[int] = []
        with torch.inference_mode():
            for cid in unique_cids:
                docs_np = collection_embeds(store, cid).astype(np.float32)
                if doc_cap and docs_np.shape[0] > doc_cap:
                    docs_np = docs_np[-doc_cap:]
                counts.append(int(docs_np.shape[0]))
                mem_state, hashes = fold_student_state(student, docs_np, device)
                cap = Capsule(state=mem_state, step=docs_np.shape[0], head=provenance_head(hashes, mem_state.digest()))
                raw = cap.to_bytes()
                reloaded = Capsule.from_bytes(raw)
                assert reloaded.state.nbytes == 22528
                sample_total_bytes = len(raw)
                state_f32 = reloaded.state.tokens.astype(np.float32) / float(1 << 14)
                reloaded_states[cid] = torch.from_numpy(state_f32).unsqueeze(0).to(device)

            bank_cpu = torch.empty((n_rows, scfg.num_memories, scfg.output_dim), dtype=torch.bfloat16)
            for cid, r_indices in rows_by_cid.items():
                q_f32 = torch.from_numpy(q_all[r_indices]).to(device=device, dtype=torch.float32)
                st_state = reloaded_states[cid].expand(len(r_indices), -1, -1, -1)
                s_tok = student.read_tokens(st_state, q_f32).to(torch.bfloat16)
                bank_cpu[torch.tensor(r_indices, dtype=torch.long)] = s_tok.cpu()
        return bank_cpu, int(np.median(counts)), sample_total_bytes, time.perf_counter() - t0

    # Teacher bank (optional — omitted for standalone Memorilla-free models)
    teacher_mem_cpu = None
    teacher_kv_bytes_median = 0
    fold_timings: dict[str, float] = {}
    if teacher is not None:
        t_teacher0 = time.perf_counter()
        teacher_mem_cpu = torch.empty((n_rows, scfg.num_memories, scfg.output_dim), dtype=torch.bfloat16)
        with torch.inference_mode():
            kv_sizes: list[int] = []
            for cid, r_indices in rows_by_cid.items():
                docs_np = collection_embeds(store, cid).astype(np.float32)
                if cfg["max_docs"] and docs_np.shape[0] > cfg["max_docs"]:
                    docs_np = docs_np[-cfg["max_docs"] :]
                dt_bf16 = torch.from_numpy(docs_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                t_cache = teacher.cache_keys_values(dt_bf16, None)
                kv_sizes.append(sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in t_cache))
                q_bf16 = torch.from_numpy(q_all[r_indices]).to(device=device, dtype=torch.bfloat16)
                b = len(r_indices)
                expanded = [(k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1)) for k, v in t_cache]
                t_tok = teacher.read_cache(expanded, None, q_bf16).to(torch.bfloat16)
                teacher_mem_cpu[torch.tensor(r_indices, dtype=torch.long)] = t_tok.cpu()
            teacher_kv_bytes_median = int(np.median(kv_sizes))
        fold_timings["teacher_cache_prep"] = round(time.perf_counter() - t_teacher0, 3)

    # Student banks: primary (max_docs, default 0 = full collection) + any extra_doc_caps
    student_banks: dict[str, torch.Tensor] = {}
    token_cosines: dict[str, float] = {}

    primary_bank, median_docs, cap_total_bytes, fold_secs = build_student_bank(cfg["max_docs"])
    student_banks["memochain_22kb"] = primary_bank
    fold_timings["memochain_22kb_fold"] = round(fold_secs, 3)

    for extra_cap in cfg["extra_doc_caps"]:
        label = f"memochain_22kb_{extra_cap}d"
        ebank, _, _, esecs = build_student_bank(extra_cap)
        student_banks[label] = ebank
        fold_timings[f"{label}_fold"] = round(esecs, 3)

    if cfg.get("proven_capsule"):
        if len(unique_cids) != 1:
            raise ValueError("--proven-capsule requires --collection to select a single user")
        cap_raw = Path(cfg["proven_capsule"]).read_bytes()
        proven_cap = Capsule.from_bytes(cap_raw)
        assert proven_cap.state.nbytes == 22528
        print(
            f"[generate] loaded SP1-proven capsule {cfg['proven_capsule']} "
            f"(step={proven_cap.step}, digest={proven_cap.state.digest()[:16]}..., "
            f"head={proven_cap.head[:16]}...)"
        )
        p_state_f32 = proven_cap.state.tokens.astype(np.float32) / float(1 << 14)
        p_state_t = torch.from_numpy(p_state_f32).unsqueeze(0).to(device)
        with torch.inference_mode():
            q_f32 = torch.from_numpy(q_all).to(device=device, dtype=torch.float32)
            p_tok = student.read_tokens(p_state_t.expand(n_rows, -1, -1, -1), q_f32).to(torch.bfloat16)
        student_banks["memochain_sp1_proven"] = p_tok.cpu()

    if teacher_mem_cpu is not None:
        for label, sbank in student_banks.items():
            cos = float(F.cosine_similarity(sbank.float(), teacher_mem_cpu.float(), dim=-1).mean())
            token_cosines[label] = round(cos, 4)
            print(f"[generate] {label} vs teacher token cosine: {cos:.4f}")

    del student
    if teacher is not None:
        del teacher
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(cfg["decoder"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [MEMORY_TOKEN]})
    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)

    embed_tokens = _load_input_embeddings(cfg["decoder"], device)
    engine = LLM(
        model=cfg["decoder"],
        dtype="bfloat16",
        max_model_len=cfg["max_length"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prompt_embeds=True,
    )
    sampling = SamplingParams(max_tokens=cfg["max_new_tokens"], temperature=0.0, top_p=1.0)

    mem_prompts = [
        tokenizer.apply_chat_template(
            build_memory_messages(q, scfg.num_memories, system_prompt=cfg["system_prompt"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in rows["questions"]
    ]
    cb_token_ids = [
        tokenizer.encode(
            tokenizer.apply_chat_template(
                build_closed_book_messages(q),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            add_special_tokens=False,
        )
        for q in rows["questions"]
    ]

    def decode_memory_method(label: str, mem_bank_cpu: torch.Tensor) -> tuple[list[str], float]:
        t_start = time.perf_counter()
        preds: list[str] = []
        bs = cfg["batch_size"]
        with torch.inference_mode():
            for start in range(0, n_rows, bs):
                end = min(start + bs, n_rows)
                enc = tokenizer(
                    mem_prompts[start:end],
                    truncation=True,
                    max_length=cfg["max_length"],
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device).bool()
                clamped_ids = input_ids.clamp(max=embed_tokens.num_embeddings - 1)
                base_embeds = embed_tokens(clamped_ids)
                mem_slice = mem_bank_cpu[start:end].to(device=device, dtype=torch.bfloat16)
                injected = _inject_memory(base_embeds, mem_slice, input_ids, memory_token_id)
                reqs = [
                    {"prompt_embeds": row[mask].detach().cpu()}
                    for row, mask in zip(injected, attn_mask)
                ]
                outs = engine.generate(reqs, sampling_params=sampling, use_tqdm=False)
                preds.extend(clean_generation(o.outputs[0].text) for o in outs)
        elapsed = time.perf_counter() - t_start
        print(f"[generate] {label}: {len(preds)} completions in {elapsed:.1f}s", flush=True)
        return preds, elapsed

    t_cb = time.perf_counter()
    cb_outs = engine.generate([{"prompt_token_ids": ids} for ids in cb_token_ids], sampling_params=sampling, use_tqdm=False)
    cb_preds = [clean_generation(o.outputs[0].text) for o in cb_outs]
    cb_secs = time.perf_counter() - t_cb
    print(f"[generate] closed_book: {len(cb_preds)} completions in {cb_secs:.1f}s", flush=True)

    predictions: dict[str, list[str]] = {"closed_book": cb_preds}
    gen_timings: dict[str, float] = {**fold_timings, "closed_book_generate": round(cb_secs, 2)}

    if teacher_mem_cpu is not None:
        teacher_preds, teacher_secs = decode_memory_method("memorilla_teacher", teacher_mem_cpu)
        predictions["memorilla_teacher"] = teacher_preds
        gen_timings["memorilla_teacher_generate"] = round(teacher_secs, 2)

    for label, sbank in student_banks.items():
        spreds, ssecs = decode_memory_method(label, sbank)
        predictions[label] = spreds
        gen_timings[f"{label}_generate"] = round(ssecs, 2)

    raw_path = Path(cfg["raw_predictions_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": cfg["dataset"],
        "split": cfg["split"],
        "num_examples": n_rows,
        "num_collections": len(unique_cids),
        "median_docs_per_collection": median_docs,
        "teacher_kv_cache_bytes_median": teacher_kv_bytes_median,
        "capsule_state_bytes": 22528,
        "capsule_total_bytes": cap_total_bytes,
        "compression_vs_teacher_kv_x": (
            round(teacher_kv_bytes_median / 22528.0, 1) if teacher_kv_bytes_median else None
        ),
        "token_cosine_mean": token_cosines,
        "timing_sec": gen_timings,
        "predictions": predictions,
    }
    raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _exit_worker_cleanly(engine)


def _score_worker(cfg: dict) -> None:
    """Spawned worker 2: embed predictions + choices with Qwen3-Embedding-4B and score."""
    from vllm import LLM

    data_dir = Path(cfg["data_dir"])
    rows = _load_split_rows(
        data_dir, cfg["dataset"], cfg["split"], cfg["max_rows"], cfg.get("collection_id")
    )
    raw = json.loads(Path(cfg["raw_predictions_path"]).read_text(encoding="utf-8"))
    preds_by_method: dict[str, list[str]] = raw["predictions"]

    unique_texts: list[str] = []
    text_to_idx: dict[str, int] = {}

    def register(t: str) -> int:
        s = t.strip()
        if s not in text_to_idx:
            text_to_idx[s] = len(unique_texts)
            unique_texts.append(s)
        return text_to_idx[s]

    for ch_row in rows["choices"]:
        for c in ch_row:
            if c.strip():
                register(c)
    for m_preds in preds_by_method.values():
        for p in m_preds:
            register(p)

    print(f"[score] embedding {len(unique_texts)} unique strings with {cfg['encoder']} ...", flush=True)
    encoder = LLM(
        model=cfg["encoder"],
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=cfg["scorer_gpu_memory_utilization"],
    )
    vecs_list = []
    chunk = 4096
    for start in range(0, len(unique_texts), chunk):
        outs = encoder.embed(unique_texts[start : start + chunk], use_tqdm=False)
        for o in outs:
            vecs_list.append(o.outputs.embedding)
    all_vecs = np.asarray(vecs_list, dtype=np.float32)

    choice_vecs_per_row = [
        all_vecs[[register(c) for c in ch_row if c.strip()]] for ch_row in rows["choices"]
    ]

    results: dict[str, dict] = {}
    selected_by_method: dict[str, np.ndarray] = {}
    for method, m_preds in preds_by_method.items():
        p_vecs = all_vecs[[register(p) for p in m_preds]]
        scores, selected = score_choices_from_vectors(
            p_vecs,
            choice_vecs_per_row,
            rows["choices"],
            rows["answers"],
            rows["hard"],
        )
        selected_by_method[method] = selected
        results[method] = {k: round(100.0 * v, 2) for k, v in scores.items()}

    cb_acc = results["closed_book"]["accuracy"]
    agreements: dict[str, float] = {}
    gain_retained: dict[str, float | None] = {}
    delta_vs_closed_book: dict[str, float] = {}

    has_teacher = "memorilla_teacher" in selected_by_method
    teacher_sel = selected_by_method["memorilla_teacher"] if has_teacher else None
    teacher_acc = results["memorilla_teacher"]["accuracy"] if has_teacher else None

    for method in preds_by_method:
        if method.startswith("memochain_"):
            mc_acc = results[method]["accuracy"]
            delta_vs_closed_book[method] = round(mc_acc - cb_acc, 2)
            if has_teacher and teacher_sel is not None and teacher_acc is not None:
                agr = float((selected_by_method[method] == teacher_sel).mean())
                agreements[method] = round(100.0 * agr, 2)
                gain_retained[method] = (
                    round(100.0 * (mc_acc - cb_acc) / (teacher_acc - cb_acc), 2)
                    if teacher_acc > cb_acc
                    else None
                )

    report = {
        "dataset": raw["dataset"],
        "split": raw["split"],
        "decoder": cfg["decoder"],
        "encoder": cfg["encoder"],
        "checkpoint": cfg["checkpoint"],
        "max_docs": cfg["max_docs"],
        "extra_doc_caps": cfg["extra_doc_caps"],
        "num_examples": raw["num_examples"],
        "num_collections": raw["num_collections"],
        "median_docs_per_collection": raw["median_docs_per_collection"],
        "teacher_kv_cache_bytes_median": raw["teacher_kv_cache_bytes_median"],
        "capsule_state_bytes": raw["capsule_state_bytes"],
        "capsule_total_bytes": raw["capsule_total_bytes"],
        "compression_vs_teacher_kv_x": raw["compression_vs_teacher_kv_x"],
        "token_cosine_mean": raw["token_cosine_mean"],
        "metrics_pct": results,
        "delta_vs_closed_book_pp": delta_vs_closed_book,
        "teacher_choice_agreement_pct": agreements,
        "memory_gain_retained_pct": gain_retained,
        "timing_sec": raw["timing_sec"],
    }
    out_path = Path(cfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[score] wrote {out_path}", flush=True)
    _exit_worker_cleanly(encoder)


def _run_spawned(fn, cfg: dict) -> None:
    """Run one stage in a fully isolated OS subprocess (via re-exec with --stage).

    A fresh subprocess guarantees the prior vLLM engine (including its EngineCore
    children, CUDA contexts and NCCL state) is completely torn down before the
    next stage starts. Skipping CPython static teardown with os._exit(0) inside
    each worker avoids the ZMQ 'terminate called without an active exception'
    abort that fires when a spawned multiprocessing child finalizes.
    """
    stage = "generate" if fn is _generate_worker else "score"
    run_cfg = json.dumps(cfg, sort_keys=True)
    proc = subprocess.run(
        [sys.executable, "-m", "memochain.eval_qa", "--stage", stage, "--stage-cfg", run_cfg],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)},
    )
    if proc.returncode != 0:
        raise SystemExit(f"{fn.__name__} exited with code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Qwen3-8B QA evaluation for MemoChain")
    parser.add_argument(
        "--checkpoint",
        required="--stage" not in sys.argv,
        help="Student run dir containing student.pt",
    )
    parser.add_argument("--teacher", default="data/stage3", help="Teacher dir containing memory.pt + config.json")
    parser.add_argument("--dataset", default="pv4", choices=["pv4", "pmv2"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--system-prompt", default=PERSONA_PROMPT)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-docs", type=int, default=0, help="Cap docs folded per collection (0 = all)")
    parser.add_argument(
        "--extra-doc-caps",
        type=int,
        nargs="*",
        default=[],
        help="Optional additional doc caps to evaluate in the same vLLM pass (e.g. 64)",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Cap evaluated rows for smoke runs (0 = all)")
    parser.add_argument("--collection", type=int, default=None, help="Filter evaluation to one collection_id")
    parser.add_argument(
        "--proven-capsule",
        default=None,
        help="Path to an SP1-proven Capsule binary file (requires --collection)",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--scorer-gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--skip-generate", action="store_true", help="Score existing raw predictions file")
    parser.add_argument("--out", default="runs/pv4_10k/qa_eval.json")
    parser.add_argument("--stage", choices=["generate", "score"], help=argparse.SUPPRESS)
    parser.add_argument("--stage-cfg", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Stage re-entry: a child subprocess runs exactly one worker function and exits.
    if args.stage and args.stage_cfg:
        cfg = json.loads(args.stage_cfg)
        if args.stage == "generate":
            _generate_worker(cfg)
        else:
            _score_worker(cfg)
        return

    out_path = Path(args.out)
    raw_preds_path = out_path.with_suffix(".raw_predictions.json")
    cfg = {
        "checkpoint": args.checkpoint,
        "teacher": args.teacher,
        "dataset": args.dataset,
        "split": args.split,
        "data_dir": args.data_dir,
        "decoder": args.decoder,
        "encoder": args.encoder,
        "system_prompt": args.system_prompt,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "max_docs": args.max_docs,
        "extra_doc_caps": list(args.extra_doc_caps),
        "max_rows": args.max_rows,
        "collection_id": args.collection,
        "proven_capsule": args.proven_capsule,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "scorer_gpu_memory_utilization": args.scorer_gpu_memory_utilization,
        "out": str(out_path),
        "raw_predictions_path": str(raw_preds_path),
    }

    if not args.skip_generate:
        _run_spawned(_generate_worker, cfg)
    _run_spawned(_score_worker, cfg)


if __name__ == "__main__":
    main()
