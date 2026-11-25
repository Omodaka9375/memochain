"""Dynamic session folding: hot window + 22 KB capsule memory per session.

The proxy receives the FULL conversation history on every request (standard
OpenAI chat semantics). Per session it tracks which prefix of that history has
already been folded into the session's Capsule (by content hash), drops that
prefix from the outgoing request, folds any newly-overflowing old turns into
the 22 KB Q14 state (through the exact integer student writer), and injects a
compact memory digest into the system prompt.

Eviction is group-safe: an assistant message carrying ``tool_calls`` is never
split from the ``tool`` result messages that answer it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from ..state import STATE_BYTES, Capsule, MemoryState
from ..verify import ChainHead

EmbedFn = Callable[[list[str]], np.ndarray]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token); good enough for windowing."""
    return max(1, len(text) // 4)


def message_text(msg: dict[str, Any]) -> str:
    """Extract comparable text from one chat message (OpenAI + Anthropic blocks)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif btype == "tool_use":
                name = block.get("name") or "tool"
                inp = block.get("input")
                parts.append(
                    f"[tool_use {name}: {json.dumps(inp, sort_keys=True, default=str)}]"
                    if inp is not None
                    else f"[tool_use {name}]"
                )
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            parts.append(sub["text"])
        return "\n".join(parts)
    return ""


def _anthropic_tool_use_ids(msg: dict[str, Any]) -> set[str]:
    content = msg.get("content")
    if not isinstance(content, list):
        return set()
    return {
        str(b["id"])
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
    }


def _anthropic_tool_result_ids(msg: dict[str, Any]) -> set[str]:
    content = msg.get("content")
    if not isinstance(content, list):
        return set()
    return {
        str(b["tool_use_id"])
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")
    }


def message_hash(msg: dict[str, Any]) -> str:
    """Stable content hash of one message (role, text, tool ids)."""
    payload = json.dumps(
        {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "tool_call_id": msg.get("tool_call_id"),
            "name": msg.get("name"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eviction_groups(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Group message indices into atomic eviction units.

    An assistant message carrying OpenAI ``tool_calls`` or Anthropic
    ``tool_use`` blocks groups together with the following ``tool`` /
    ``tool_result`` message(s) that answer it, so tool calls and results
    can never be split across a fold boundary.
    """
    groups: list[tuple[int, int]] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            call_ids = {tc.get("id") for tc in m["tool_calls"] if isinstance(tc, dict)}
            while j < n and messages[j].get("role") == "tool" and messages[j].get("tool_call_id") in call_ids:
                j += 1
            groups.append((i, j))
            i = j
        elif m.get("role") == "assistant" and (use_ids := _anthropic_tool_use_ids(m)):
            j = i + 1
            if j < n and messages[j].get("role") == "user" and (_anthropic_tool_result_ids(messages[j]) & use_ids):
                j += 1
            groups.append((i, j))
            i = j
        else:
            groups.append((i, i + 1))
            i += 1
    return groups


class FoldWriter(Protocol):
    """The narrow writer interface SessionMemory needs."""

    def fold_with_receipts(
        self,
        docs_q14: "np.ndarray",
        initial_state: "MemoryState | None" = None,
        initial_head: "ChainHead | None" = None,
    ) -> tuple[MemoryState, ChainHead, list[dict[str, Any]]]: ...

    def weights_digest(self) -> str: ...


class MemoryReader(Protocol):
    """Turns a fresh capsule state + query embedding into a text digest."""

    def read(self, state: MemoryState, query_embed: "np.ndarray", prev_summary: str) -> str: ...


class ChunkStore:
    """SQLite-backed verbatim chunk store + vector/lexical hybrid search (Option B).

    Each folded message is saved with its role, text, message hash, and a
    per-vector int8-quantized unit embedding (4x smaller than float32, with
    transparent fallback when reading legacy float32 rows). Optional
    ``max_chunks`` caps table size via FIFO eviction so disk and scan cost
    stay bounded per session.
    """

    def __init__(self, db_path: str | Path, max_chunks: int = 2000) -> None:
        self.db_path = Path(db_path)
        self.max_chunks = max(0, int(max_chunks))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    msg_hash TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL,
                    scale REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "scale" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN scale REAL NOT NULL DEFAULT 0.0")

    @staticmethod
    def _quantize_i8(unit_vec: "np.ndarray") -> tuple[bytes, float]:
        max_abs = float(np.max(np.abs(unit_vec)))
        if max_abs <= 0.0:
            return np.zeros(unit_vec.shape, dtype=np.int8).tobytes(), 0.0
        scale = max_abs / 127.0
        q = np.clip(np.round(unit_vec / scale), -127, 127).astype(np.int8)
        return q.tobytes(), scale

    @staticmethod
    def _dequantize(blob: bytes, dim: int, scale: float) -> "np.ndarray | None":
        if len(blob) == dim:  # int8 quantized
            if scale <= 0.0:
                return np.zeros(dim, dtype=np.float32)
            return np.frombuffer(blob, dtype=np.int8).astype(np.float32) * float(scale)
        if len(blob) == dim * 4:  # legacy float32 row
            return np.frombuffer(blob, dtype="<f4")
        return None

    def add(self, step: int, entries: list[tuple[str, str, str]], embeds: "np.ndarray") -> None:
        """Store (msg_hash, role, text) rows aligned with embeds [N, D]."""
        arr = np.asarray(embeds, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        with sqlite3.connect(self.db_path) as conn:
            for i, (mhash, role, text) in enumerate(entries):
                blob, scale = self._quantize_i8(arr[i])
                conn.execute(
                    "INSERT OR IGNORE INTO chunks (step, msg_hash, role, text, dim, vec, scale) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (step, mhash, role, text, int(arr.shape[1]), blob, scale),
                )
            if self.max_chunks > 0:
                conn.execute(
                    "DELETE FROM chunks WHERE id NOT IN (SELECT id FROM chunks ORDER BY id DESC LIMIT ?)",
                    (self.max_chunks,),
                )

    def count(self) -> int:
        if not self.db_path.exists():
            return 0
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_./:-]{2,}", text.lower()))

    def search(self, query_text: str, query_embed: "np.ndarray", top_k: int = 4) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT step, role, text, dim, vec, scale FROM chunks ORDER BY id ASC"
            ).fetchall()
        if not rows:
            return []
        q = np.asarray(query_embed, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        q_toks = self._tokens(query_text)

        scored: list[tuple[float, int, str, str]] = []
        for step, role, text, dim, blob, scale in rows:
            cos = 0.0
            if dim == q.size:
                v = self._dequantize(blob, dim, float(scale or 0.0))
                if v is not None:
                    cos = float(np.dot(q, v))
            lex = 0.0
            if q_toks:
                c_toks = self._tokens(text)
                if c_toks:
                    lex = len(q_toks & c_toks) / max(1, len(q_toks))
            score = 0.65 * cos + 0.35 * lex
            scored.append((score, int(step), str(role), str(text)))

        scored.sort(key=lambda r: r[0], reverse=True)
        return [
            {"score": round(score, 4), "step": step, "role": role, "text": text}
            for score, step, role, text in scored[:top_k]
        ]


@dataclass
class FoldStats:
    folded_messages: int
    folded_tokens: int
    step: int
    state_digest: str
    capsule_bytes: int


class SessionMemory:
    """One session's hot window + persistent 22 KB capsule memory."""

    def __init__(
        self,
        session_id: str,
        store_dir: str | Path,
        writer: FoldWriter,
        embed_fn: EmbedFn,
        reader: MemoryReader,
        hot_limit: int = 80000,
        keep_min: int = 40000,
        max_doc_chars: int = 2000,
        proof_queue: Any | None = None,
        recall_top_k: int = 0,
        max_chunks: int = 2000,
    ) -> None:
        self.session_id = session_id
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.embed_fn = embed_fn
        self.reader = reader
        self.hot_limit = hot_limit
        self.keep_min = keep_min
        self.max_doc_chars = max_doc_chars
        self.proof_queue = proof_queue
        self.recall_top_k = recall_top_k
        self.max_chunks = max_chunks
        self.chunk_store = (
            ChunkStore(self.dir / "chunks.db", max_chunks=max_chunks) if recall_top_k > 0 else None
        )

        self.folded_hashes: list[str] = []
        self._folded_set: set[str] = set()
        self.summary = ""
        self.state: MemoryState | None = None
        self.head: ChainHead | None = None
        self.tampered = False
        self._load()

    # ---- persistence ----------------------------------------------------

    @property
    def capsule_path(self) -> Path:
        return self.dir / "capsule.bin"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def receipts_path(self) -> Path:
        return self.dir / "receipts.jsonl"

    def _load(self) -> None:
        if not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.folded_hashes = list(meta.get("folded_hashes", []))
        self._folded_set = set(self.folded_hashes)
        self.summary = meta.get("summary", "")
        if not (self.capsule_path.exists() and meta.get("state_digest")):
            return
        try:
            cap = Capsule.from_bytes(self.capsule_path.read_bytes())
        except ValueError:
            # corrupted / tampered capsule bytes fail the envelope hash check
            self.tampered = True
            return
        state_digest_matches = cap.state.digest() == meta["state_digest"]
        head_matches = cap.head == meta.get("accumulator")
        if state_digest_matches and head_matches:
            self.state = cap.state
            self.head = ChainHead(step=cap.step, accumulator=cap.head, state_digest=cap.state.digest())
        else:
            # capsule/meta mismatch: keep the session unusable rather than
            # silently continuing on bad state.
            self.tampered = True

    def _save(self) -> None:
        meta = {
            "session_id": self.session_id,
            "step": self.head.step if self.head else 0,
            "state_digest": self.head.state_digest if self.head else "",
            "accumulator": self.head.accumulator if self.head else "",
            "folded_hashes": self.folded_hashes,
            "summary": self.summary,
            "weights_digest": self.writer.weights_digest(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def _append_receipts(self, rows: list[dict[str, Any]]) -> None:
        with open(self.receipts_path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    # ---- memory operations ----------------------------------------------

    def _fold_messages(self, msgs: list[dict[str, Any]], model_hint: str | None = None) -> FoldStats:
        entries: list[tuple[str, str, str]] = []
        texts: list[str] = []
        for m in msgs:
            t = message_text(m).strip()
            if t:
                clipped = t[: self.max_doc_chars]
                texts.append(clipped)
                entries.append((message_hash(m), str(m.get("role") or "user"), clipped))
        if not texts:
            texts = ["(empty tool exchange)"]

        docs_f = self.embed_fn(texts).astype(np.float32)
        if docs_f.ndim == 1:
            docs_f = docs_f[None, :]
        q14 = np.clip(np.round(docs_f * (1 << 14)), -32768, 32767).astype("<i2")

        state, head, receipt_rows = self.writer.fold_with_receipts(
            q14, initial_state=self.state, initial_head=self.head
        )
        self.state, self.head = state, head
        if self.chunk_store is not None and entries:
            self.chunk_store.add(head.step, entries, docs_f[: len(entries)])
        if self.proof_queue is not None and q14.shape[1] == 2560:
            # raw Q14 little-endian i16 docs for the SP1 IVC queue
            self.proof_queue.submit(self.session_id, q14.tobytes())

        cap = Capsule(state=state, step=head.step, head=head.accumulator)
        raw = cap.to_bytes()
        tmp = self.capsule_path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, self.capsule_path)

        assert self.state.nbytes == STATE_BYTES
        self._append_receipts(receipt_rows)

        # query embedding for the digest reader: normalized mean of folded docs
        q = docs_f.mean(axis=0)
        norm = float(np.linalg.norm(q)) or 1.0
        if hasattr(self.reader, "read_fold"):
            try:
                self.summary = self.reader.read_fold(
                    state, q / norm, self.summary, entries, model_hint=model_hint
                )
            except TypeError:
                self.summary = self.reader.read_fold(state, q / norm, self.summary, entries)
        else:
            self.summary = self.reader.read(state, q / norm, self.summary)
        self._save()
        return FoldStats(
            folded_messages=len(msgs),
            folded_tokens=sum(estimate_tokens(message_text(m)) for m in msgs),
            step=head.step,
            state_digest=head.state_digest,
            capsule_bytes=len(raw),
        )

    def process(
        self,
        messages: list[dict[str, Any]],
        model_hint: str | None = None,
        inject_memory: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fold overflow and return (rewritten_messages, stats).

        Invariants preserved on the rewritten message list:
        - Leading system messages stay intact (never evicted into the capsule).
        - At least one user message remains in the hot tail whenever the input
          contains a user message (chat templates such as Qwen3 reject prompts
          with no user turn).
        - Assistant tool_calls stay grouped with their tool results.
        """
        # Keep leading system prompt(s) out of folding so instructions/tools
        # stay verbatim and we inject [Durable memory] into the real system msg.
        sys_end = 0
        while sys_end < len(messages) and messages[sys_end].get("role") == "system":
            sys_end += 1
        sys_msgs = messages[:sys_end]
        conv = messages[sys_end:]

        hashes = [message_hash(m) for m in conv]

        # longest already-folded prefix of the non-system conversation
        prefix = 0
        while prefix < len(hashes) and hashes[prefix] in self._folded_set:
            prefix += 1

        # Never leave the hot tail without a user turn if the input had one.
        if not any(m.get("role") == "user" for m in conv[prefix:]):
            last_user = next(
                (i for i in range(len(conv) - 1, -1, -1) if conv[i].get("role") == "user"),
                None,
            )
            if last_user is not None:
                prefix = last_user

        hot = conv[prefix:]
        sys_tokens = sum(estimate_tokens(message_text(m)) for m in sys_msgs)
        conv_tokens = sum(estimate_tokens(message_text(m)) for m in hot)
        stats = {
            "session_id": self.session_id,
            "already_folded": prefix,
            "hot_messages": len(sys_msgs) + len(hot),
            "system_tokens": sys_tokens,
            "hot_tokens": conv_tokens,
            "fold": None,
        }

        groups = eviction_groups(hot)
        if conv_tokens > self.hot_limit and len(groups) > 1:
            keep_min = min(self.keep_min, self.hot_limit)
            evict_end = 0  # exclusive index into hot
            remaining = conv_tokens
            last_user_idx = next(
                (i for i in range(len(hot) - 1, -1, -1) if hot[i].get("role") == "user"),
                None,
            )
            for gi, (start, end) in enumerate(groups[:-1]):  # never evict the last group
                # Never evict the last remaining user message.
                if last_user_idx is not None and end > last_user_idx:
                    break
                group_tokens = sum(estimate_tokens(message_text(m)) for m in hot[start:end])
                if remaining - group_tokens < keep_min:
                    break
                remaining -= group_tokens
                evict_end = end
            if evict_end > 0:
                to_fold = hot[:evict_end]
                fold_stats = self._fold_messages(to_fold, model_hint=model_hint)
                for m in to_fold:
                    h = message_hash(m)
                    self.folded_hashes.append(h)
                    self._folded_set.add(h)
                self._save()
                hot = hot[evict_end:]
                stats["fold"] = {
                    "messages": fold_stats.folded_messages,
                    "tokens": fold_stats.folded_tokens,
                    "step": fold_stats.step,
                    "capsule_bytes": fold_stats.capsule_bytes,
                }
                stats["hot_messages"] = len(sys_msgs) + len(hot)
                stats["hot_tokens"] = sum(estimate_tokens(message_text(m)) for m in hot)

        recalled: list[dict[str, Any]] = []
        if self.chunk_store is not None and self.chunk_store.count() > 0:
            last_user_text = next(
                (
                    message_text(m).strip()
                    for m in reversed(hot)
                    if m.get("role") == "user" and message_text(m).strip()
                ),
                "",
            )
            if last_user_text:
                q_vec = self.embed_fn([last_user_text])[0]
                recalled = self.chunk_store.search(
                    last_user_text, q_vec, top_k=self.recall_top_k
                )
                stats["recalled_chunks"] = len(recalled)

        memory_block = self.format_memory_block(recalled=recalled)
        if memory_block:
            stats["memory_block"] = memory_block
        rewritten = (
            self._inject_memory(list(sys_msgs) + list(hot), memory_block=memory_block)
            if inject_memory
            else list(sys_msgs) + list(hot)
        )
        return rewritten, stats

    def format_memory_block(self, recalled: list[dict[str, Any]] | None = None) -> str:
        if not self.summary and not recalled:
            return ""
        sections: list[str] = [
            "[Durable memory]",
            "Compressed notes from earlier parts of this session (older than what you can see here):",
        ]
        if self.summary:
            sections.append(self.summary)
        if recalled:
            sections.append("")
            sections.append("Relevant verbatim turns recalled for the current question:")
            for item in recalled:
                snippet = item["text"].replace("\n", " ").strip()
                if len(snippet) > 600:
                    snippet = snippet[:600] + "..."
                sections.append(f"- [{item['role']}] {snippet}")
        return "\n".join(sections)

    def _inject_memory(
        self,
        hot: list[dict[str, Any]],
        recalled: list[dict[str, Any]] | None = None,
        memory_block: str | None = None,
    ) -> list[dict[str, Any]]:
        rendered = memory_block if memory_block is not None else self.format_memory_block(recalled=recalled)
        if not rendered:
            return list(hot)
        block = "\n\n" + rendered
        out = [dict(m) for m in hot]
        if out and out[0].get("role") == "system":
            first = dict(out[0])
            c = first.get("content")
            first["content"] = (c + block) if isinstance(c, str) else block.lstrip()
            out[0] = first
        else:
            out.insert(0, {"role": "system", "content": block.strip()})
        return out

    # ---- integrity --------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """Recompute the receipt chain from receipts.jsonl and check the capsule."""
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "capsule_present": self.capsule_path.exists(),
            "ok": False,
        }
        if self.tampered:
            result["error"] = "capsule does not match meta (tamper detected on load)"
            return result
        if not self.capsule_path.exists() or not self.meta_path.exists():
            result["error"] = "missing capsule or meta"
            return result
        cap = Capsule.from_bytes(self.capsule_path.read_bytes())
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        result["step"] = cap.step
        result["capsule_bytes"] = self.capsule_path.stat().st_size
        result["state_digest"] = cap.state.digest()
        result["meta_state_digest"] = meta.get("state_digest")
        result["state_matches_meta"] = cap.state.digest() == meta.get("state_digest")
        result["head_matches_meta"] = cap.head == meta.get("accumulator")

        # replay the binding chain from stored per-doc receipts
        acc = "0" * 64
        chain_ok = True
        n_rows = 0
        if self.receipts_path.exists():
            with open(self.receipts_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    n_rows += 1
                    binding = hashlib.sha256(
                        f"{row['step']}|{acc}|{row['doc_hash']}|{row['state_digest']}".encode()
                    ).hexdigest()
                    acc = hashlib.sha256(f"{acc}{binding}".encode()).hexdigest()
                    if binding != row["binding"] or acc != row["accumulator"]:
                        chain_ok = False
                        break
        result["receipt_rows"] = n_rows
        result["receipt_chain_ok"] = chain_ok
        result["receipt_head_matches"] = (n_rows == 0) or (acc == cap.head)
        result["ok"] = bool(
            result["state_matches_meta"]
            and result["head_matches_meta"]
            and result["receipt_chain_ok"]
            and result["receipt_head_matches"]
        )
        return result
