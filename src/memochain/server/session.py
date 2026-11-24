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
    """Extract comparable text from one chat message (text parts joined)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


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

    An assistant message with ``tool_calls`` groups together with every
    following ``tool`` message that answers one of its call ids, so a tool
    result can never be evicted without its request (or vice versa).
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

    def _fold_messages(self, msgs: list[dict[str, Any]]) -> FoldStats:
        texts = []
        for m in msgs:
            t = message_text(m).strip()
            if t:
                texts.append(t[: self.max_doc_chars])
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
        self.summary = self.reader.read(state, q / norm, self.summary)
        self._save()
        return FoldStats(
            folded_messages=len(msgs),
            folded_tokens=sum(estimate_tokens(message_text(m)) for m in msgs),
            step=head.step,
            state_digest=head.state_digest,
            capsule_bytes=len(raw),
        )

    def process(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                fold_stats = self._fold_messages(to_fold)
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

        rewritten = self._inject_memory(list(sys_msgs) + list(hot))
        return rewritten, stats

    def _inject_memory(self, hot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.summary:
            return list(hot)
        block = (
            "\n\n[Durable memory]\n"
            "Compressed notes from earlier parts of this session (older than "
            "what you can see here):\n" + self.summary
        )
        out = [dict(m) for m in hot]
        if out and out[0].get("role") == "system":
            first = dict(out[0])
            c = first.get("content")
            first["content"] = (c + block) if isinstance(c, str) else block
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
