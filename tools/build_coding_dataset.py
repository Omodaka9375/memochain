"""Build a MemoChain-compatible coding-domain dataset from local KAI sessions.

Reads KAI session JSON exports (``kai-sessions-*.json``), extracts chronological
coding memory documents (user prompts, assistant replies, and terse tool-action
summaries such as files read/edited and commands run), and produces causal
training pairs:

  For every later turn t in a session:
    history docs = session_docs[0 : doc_end_idx]   (strictly BEFORE turn t!)
    question     = user message at turn t
    answer       = assistant reply at turn t (with tool-call actions summarized)

Zero future leakage: ``doc_end_idx`` is written into the parquet table so
``train_e2e.py`` only folds documents strictly prior to each question.

Also scrubs common secret/token patterns before emitting text. Raw session data
stays strictly under ``data/`` (gitignored) and is never committed.

Output layout (matches MemoChain / Memorilla-Data store conventions):
  {out_dir}/data/{name}/{split}-00000-of-00001.parquet
      columns: collection_id, doc_end_idx, question, answer
  {out_dir}/embeddings/qwen3-embedding-4b/{name}/{split}/
      index.safetensors         (collection_ids, shard, start, length)
      documents-00000.safetensors   float16 [total_docs, 2560]
      questions.safetensors         float16 [num_rows, 2560]
  {out_dir}/data/{name}/docs_{split}.jsonl  (optional raw doc texts for inspection)

Embeddings can be computed either via a running OpenAI-compatible sidecar
(``--embedding-url http://localhost:8001/v1``) or locally with vLLM
(``--local-vllm``).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}", re.I),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^\s'\"`]{12,}"),
]
ENV_BLOCK_RE = re.compile(r"<env>.*?</env>", re.S)
FENCE_MARKER_RE = re.compile(r"\[(?:start|end)\s+(?:tool|web|watch|mcp)_[A-Za-z0-9]+\]")


def scrub_text(text: str) -> str:
    """Remove env headers, fence markers, and obvious credential strings."""
    s = ENV_BLOCK_RE.sub("", text)
    s = FENCE_MARKER_RE.sub("", s)
    for pat in SECRET_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def summarize_tool_part(part: dict[str, Any], max_output_chars: int = 240) -> str:
    """Format one KAI tool part into a compact coding memory line."""
    ptype = str(part.get("type", ""))
    tool_name = str(part.get("toolName") or ptype.removeprefix("tool-")).strip()
    inp = part.get("input") if isinstance(part.get("input"), dict) else {}
    out = part.get("output") if isinstance(part.get("output"), dict) else {}

    args_bits: list[str] = []
    for k in ("path", "pattern", "command", "query", "url", "root"):
        if k in inp and inp[k]:
            val = str(inp[k]).replace("\n", " ").strip()
            args_bits.append(f"{k}={val[:140]}")
    if "edits" in inp and isinstance(inp["edits"], list):
        paths = sorted({str(e.get("path", "")) for e in inp["edits"] if isinstance(e, dict) and e.get("path")})
        if paths:
            args_bits.append(f"paths={','.join(paths[:4])}")

    status_bits: list[str] = []
    if "exit_code" in out and out["exit_code"] is not None:
        status_bits.append(f"exit={out['exit_code']}")
    if out.get("ok") is True:
        status_bits.append("ok")
    if out.get("error"):
        status_bits.append(f"err={str(out['error'])[:100]}")
    elif part.get("errorText"):
        status_bits.append(f"err={str(part['errorText'])[:100]}")
    elif isinstance(out.get("stdout"), str) and out["stdout"].strip():
        snippet = scrub_text(out["stdout"]).replace("\n", " ")[:max_output_chars]
        if snippet:
            status_bits.append(f"out={snippet}")

    head = f"[tool:{tool_name or 'call'}]"
    body = " ".join(args_bits)
    tail = (" -> " + " ".join(status_bits)) if status_bits else ""
    return scrub_text(f"{head} {body}{tail}".strip())


def extract_message_text(msg: dict[str, Any], *, include_tools: bool = True, max_chars: int = 1600) -> str:
    """Extract clean text (and optional compact tool actions) from one KAI message."""
    chunks: list[str] = []
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        chunks.append(scrub_text(content))

    parts = msg.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if not isinstance(p, dict):
                continue
            pt = str(p.get("type", ""))
            if pt == "text":
                t = scrub_text(str(p.get("text") or ""))
                if t:
                    chunks.append(t)
            elif include_tools and (pt.startswith("tool-") or pt == "dynamic-tool"):
                line = summarize_tool_part(p)
                if line:
                    chunks.append(line)

    joined = "\n".join(c for c in chunks if c)
    return joined[:max_chars].strip()


@dataclass
class CodingSession:
    session_key: str
    docs: list[str]
    qa_rows: list[tuple[int, str, str]]  # (doc_end_idx, question, answer)


def extract_session(
    session_key: str,
    messages: list[dict[str, Any]],
    *,
    min_history_docs: int = 2,
    min_q_chars: int = 12,
    min_a_chars: int = 20,
    max_doc_chars: int = 1600,
    max_answer_chars: int = 800,
) -> CodingSession | None:
    """Convert one KAI message list into chronological memory docs + causal Q/A pairs.

    Each turn appends its text/tool steps into ``docs``. Any user turn that has
    at least ``min_history_docs`` prior documents AND a substantive assistant
    reply becomes one causal QA example pointing at ``doc_end_idx`` (the exact
    number of docs that existed BEFORE this user turn).
    """
    docs: list[str] = []
    qa_rows: list[tuple[int, str, str]] = []

    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if not isinstance(m, dict):
            i += 1
            continue
        role = str(m.get("role", ""))
        if role == "user":
            q_text = extract_message_text(m, include_tools=False, max_chars=max_doc_chars)
            # Skip synthetic watch-firing injections or tiny acknowledgements as questions
            is_watch = "Watch fired" in q_text or "[source: watch" in q_text
            doc_end_idx = len(docs)

            # Check if followed by an assistant response
            a_text = ""
            if i + 1 < n and isinstance(messages[i + 1], dict) and messages[i + 1].get("role") == "assistant":
                a_text = extract_message_text(
                    messages[i + 1], include_tools=False, max_chars=max_answer_chars
                )
                a_doc = extract_message_text(
                    messages[i + 1], include_tools=True, max_chars=max_doc_chars
                )
                if (
                    not is_watch
                    and doc_end_idx >= min_history_docs
                    and len(q_text) >= min_q_chars
                    and len(a_text) >= min_a_chars
                ):
                    qa_rows.append((doc_end_idx, q_text, a_text))
                if q_text:
                    docs.append(f"User: {q_text}")
                if a_doc:
                    docs.append(f"Assistant: {a_doc}")
                i += 2
                continue

            if q_text:
                docs.append(f"User: {q_text}")
            i += 1
        elif role == "assistant":
            a_doc = extract_message_text(m, include_tools=True, max_chars=max_doc_chars)
            if a_doc:
                docs.append(f"Assistant: {a_doc}")
            i += 1
        else:
            i += 1

    if not qa_rows or len(docs) < min_history_docs + 1:
        return None
    return CodingSession(session_key=session_key, docs=docs, qa_rows=qa_rows)


def load_kai_sessions(
    kai_dir: str | Path,
    *,
    min_history_docs: int = 2,
    max_sessions: int = 0,
) -> list[CodingSession]:
    """Scan all non-backup kai-sessions-*.json files under ``kai_dir``."""
    root = Path(kai_dir)
    files = sorted(
        f
        for f in glob.glob(str(root / "kai-sessions-*.json"))
        if "backup" not in os.path.basename(f) and "snapshot" not in os.path.basename(f)
    )
    out: list[CodingSession] = []
    seen_keys: set[str] = set()
    for f in files:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for k, v in sorted(data.items()):
            if not (k.startswith("messages:") and isinstance(v, list) and len(v) >= 4):
                continue
            skey = f"{Path(f).stem}:{k}"
            if skey in seen_keys:
                continue
            seen_keys.add(skey)
            sess = extract_session(skey, v, min_history_docs=min_history_docs)
            if sess is not None:
                out.append(sess)
                if max_sessions > 0 and len(out) >= max_sessions:
                    return out
    return out


def split_sessions(
    sessions: list[CodingSession],
    val_frac: float = 0.12,
    test_frac: float = 0.12,
    seed: int = 0,
) -> dict[str, list[CodingSession]]:
    """Deterministic session-level split (zero session leakage across splits)."""
    idx = np.arange(len(sessions))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(sessions)
    n_test = max(1, int(round(n * test_frac))) if n >= 5 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    test_ids = set(idx[:n_test].tolist())
    val_ids = set(idx[n_test : n_test + n_val].tolist())
    splits: dict[str, list[CodingSession]] = {"train": [], "validation": [], "test": []}
    for i, s in enumerate(sessions):
        if i in test_ids:
            splits["test"].append(s)
        elif i in val_ids:
            splits["validation"].append(s)
        else:
            splits["train"].append(s)
    return splits


def write_split_store(
    out_dir: Path,
    dataset_name: str,
    split: str,
    sessions: list[CodingSession],
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    embed_batch: int = 64,
) -> dict[str, Any]:
    """Write parquet + safetensors embedding store for one split."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from safetensors.numpy import save_file

    data_dir = out_dir / "data" / dataset_name
    emb_dir = out_dir / "embeddings" / "qwen3-embedding-4b" / dataset_name / split
    data_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[str] = []
    col_ids: list[int] = []
    starts: list[int] = []
    lengths: list[int] = []

    row_cids: list[int] = []
    row_doc_ends: list[int] = []
    row_questions: list[str] = []
    row_answers: list[str] = []

    for cid, sess in enumerate(sessions):
        start = len(all_docs)
        all_docs.extend(sess.docs)
        col_ids.append(cid)
        starts.append(start)
        lengths.append(len(sess.docs))
        for doc_end_idx, q, a in sess.qa_rows:
            row_cids.append(cid)
            row_doc_ends.append(doc_end_idx)
            row_questions.append(q)
            row_answers.append(a)

    # Write parquet
    tbl = pa.table(
        {
            "collection_id": pa.array(row_cids, type=pa.int64()),
            "doc_end_idx": pa.array(row_doc_ends, type=pa.int64()),
            "question": pa.array(row_questions, type=pa.string()),
            "answer": pa.array(row_answers, type=pa.string()),
        }
    )
    pq.write_table(tbl, data_dir / f"{split}-00000-of-00001.parquet")

    def embed_all(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 2560), dtype=np.float16)
        vecs: list[np.ndarray] = []
        for s in range(0, len(texts), embed_batch):
            batch = texts[s : s + embed_batch]
            arr = np.asarray(embed_fn(batch), dtype=np.float32)
            # L2-normalize to match Qwen3-Embedding-4B store convention
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs.append((arr / norms).astype(np.float16))
        return np.concatenate(vecs, axis=0)

    print(f"[build] {split}: embedding {len(all_docs)} docs + {len(row_questions)} questions ...", flush=True)
    doc_vecs = embed_all(all_docs)
    q_vecs = embed_all(row_questions)

    save_file(
        {
            "collection_ids": np.asarray(col_ids, dtype=np.int64),
            "shard": np.zeros(len(col_ids), dtype=np.int64),
            "start": np.asarray(starts, dtype=np.int64),
            "length": np.asarray(lengths, dtype=np.int64),
        },
        str(emb_dir / "index.safetensors"),
    )
    save_file({"embeddings": doc_vecs}, str(emb_dir / "documents-00000.safetensors"))
    save_file({"embeddings": q_vecs}, str(emb_dir / "questions.safetensors"))

    return {
        "split": split,
        "sessions": len(sessions),
        "docs": len(all_docs),
        "qa_rows": len(row_questions),
        "embed_dim": int(doc_vecs.shape[1]) if doc_vecs.size else 2560,
    }


def make_http_embed_fn(base_url: str, model: str = "Qwen/Qwen3-Embedding-4B") -> Callable[[list[str]], np.ndarray]:
    import httpx

    client = httpx.Client(base_url=base_url, timeout=180.0)

    def _embed(texts: list[str]) -> np.ndarray:
        resp = client.post("/embeddings", json={"model": model, "input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda row: row["index"])
        return np.asarray([row["embedding"] for row in data], dtype=np.float32)

    return _embed


def make_vllm_embed_fn(model: str = "Qwen/Qwen3-Embedding-4B", gpu_mem: float = 0.25) -> Callable[[list[str]], np.ndarray]:
    from vllm import LLM

    engine = LLM(
        model=model,
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=gpu_mem,
    )

    def _embed(texts: list[str]) -> np.ndarray:
        outs = engine.embed(texts, use_tqdm=False)
        return np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)

    return _embed


def export_raw_jsonl(sessions: list[CodingSession], path: Path) -> None:
    """Export scrubbed sessions to a portable JSONL file (e.g. to embed on a remote GPU host)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(
                json.dumps(
                    {
                        "id": hashlib.sha256(s.session_key.encode()).hexdigest()[:16],
                        "docs": s.docs,
                        "qa_rows": s.qa_rows,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_raw_jsonl(path: Path) -> list[CodingSession]:
    out: list[CodingSession] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                CodingSession(
                    session_key=str(obj["id"]),
                    docs=list(obj["docs"]),
                    qa_rows=[(int(r[0]), str(r[1]), str(r[2])) for r in obj["qa_rows"]],
                )
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build MemoChain coding dataset from KAI sessions")
    p.add_argument(
        "--kai-dir",
        default=os.path.expandvars(r"%APPDATA%\app.kai.terminal"),
        help="Directory with kai-sessions-*.json files",
    )
    p.add_argument("--from-jsonl", default=None, help="Read pre-extracted scrubbed sessions JSONL instead of --kai-dir")
    p.add_argument("--export-jsonl", default=None, help="Only extract & scrub sessions to JSONL (no embedding)")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--name", default="kai_coding")
    p.add_argument("--embedding-url", default=None, help="OpenAI-compatible /v1 embeddings URL")
    p.add_argument("--local-vllm", action="store_true", help="Embed locally with vLLM Qwen3-Embedding-4B")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--embed-batch", type=int, default=64)
    p.add_argument("--min-history-docs", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--test-frac", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.from_jsonl:
        sessions = load_raw_jsonl(Path(args.from_jsonl))
    else:
        sessions = load_kai_sessions(args.kai_dir, min_history_docs=args.min_history_docs)

    print(
        f"[build] loaded {len(sessions)} coding sessions "
        f"({sum(len(s.docs) for s in sessions)} docs, {sum(len(s.qa_rows) for s in sessions)} causal QA rows)",
        flush=True,
    )

    if args.export_jsonl:
        export_raw_jsonl(sessions, Path(args.export_jsonl))
        print(f"[build] exported scrubbed JSONL to {args.export_jsonl}", flush=True)
        return

    if args.local_vllm:
        embed_fn = make_vllm_embed_fn(args.embed_model)
    elif args.embedding_url:
        embed_fn = make_http_embed_fn(args.embedding_url, args.embed_model)
    else:
        raise SystemExit("Pass --embedding-url, --local-vllm, or --export-jsonl")

    splits = split_sessions(sessions, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    summary = {}
    for sp, sp_sessions in splits.items():
        if not sp_sessions:
            continue
        summary[sp] = write_split_store(
            Path(args.out_dir),
            args.name,
            sp,
            sp_sessions,
            embed_fn,
            embed_batch=args.embed_batch,
        )
    meta_path = Path(args.out_dir) / "data" / args.name / "dataset_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
