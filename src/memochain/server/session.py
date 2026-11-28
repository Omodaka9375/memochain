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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

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
    """SQLite-backed verbatim chunk store + FTS5/vector RRF hybrid search.

    Each folded message is saved with its role, text, message hash, and a
    per-vector int8-quantized unit embedding (4x smaller than float32, with
    transparent fallback when reading legacy float32 rows). Supports:
    - FTS5 BM25 + cosine Reciprocal Rank Fusion (RRF) retrieval
    - insert-time near-duplicate merging (bumps hit counter instead of duplicating)
    - hit-based priority eviction when ``max_chunks`` is exceeded
    - append-only ``facts`` ledger that accumulates durable key/value facts across folds
    """

    def __init__(
        self,
        db_path: str | Path,
        max_chunks: int = 2000,
        dedupe_threshold: float = 0.96,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_chunks = max(0, int(max_chunks))
        self.dedupe_threshold = float(dedupe_threshold)
        self.fts_enabled = False
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                pass
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
                    scale REAL NOT NULL DEFAULT 0.0,
                    hits INTEGER NOT NULL DEFAULT 0,
                    last_hit_step INTEGER NOT NULL DEFAULT 0,
                    pinned INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "scale" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN scale REAL NOT NULL DEFAULT 0.0")
            if "hits" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN hits INTEGER NOT NULL DEFAULT 0")
            if "last_hit_step" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN last_hit_step INTEGER NOT NULL DEFAULT 0")
            if "pinned" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")

            # Append-only durable facts ledger (supersedes on matching normalized key)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    step INTEGER NOT NULL DEFAULT 0,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # FTS5 lexical index with identifier-friendly tokenchars (_./:-)
            try:
                existed = (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
                    ).fetchone()
                    is not None
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        text,
                        content='chunks',
                        content_rowid='id',
                        tokenize="porter unicode61 tokenchars '_./-'"
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text)
                        VALUES ('delete', old.id, old.text);
                    END
                    """
                )
                if not existed:
                    # Backfill legacy rows created before FTS5 existed
                    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False

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

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            t.strip("._:/-")
            for t in re.findall(r"[a-z0-9_./:-]{2,}", text.lower())
            if len(t.strip("._:/-")) >= 2
        }

    @staticmethod
    def _fts_query(query_text: str) -> str:
        raw_tokens = re.findall(r"[a-zA-Z0-9_./:-]{2,}", query_text.lower())
        if not raw_tokens:
            return ""
        # Strip outer punctuation and split on ':' (since FTS5 separates on ':' so
        # 'TK-1005:' indexes as 'tk-1005' and 'host:5433' indexes as 'host' + '5433').
        seen: set[str] = set()
        ident: list[str] = []
        plain: list[str] = []
        for raw in raw_tokens:
            for piece in raw.split(":"):
                tok = piece.strip("._/-")
                if len(tok) < 2 or tok in seen:
                    continue
                seen.add(tok)
                if any(ch.isdigit() or ch in "_./-" for ch in tok):
                    ident.append(tok)
                else:
                    plain.append(tok)
        ordered = (ident + plain)[:24]
        return " OR ".join(f'"{t.replace(chr(34), "")}"' for t in ordered if t)

    def _find_duplicate_id(
        self,
        conn: sqlite3.Connection,
        role: str,
        text: str,
        unit_vec: "np.ndarray",
    ) -> int | None:
        if self.dedupe_threshold <= 0.0:
            return None
        norm_text = " ".join(text.split())
        if not norm_text:
            return None
        new_toks = self._tokens(norm_text)
        rows = conn.execute(
            "SELECT id, text, dim, vec, scale FROM chunks WHERE role = ? ORDER BY id DESC LIMIT 256",
            (role,),
        ).fetchall()
        for cid, existing_text, dim, blob, scale in rows:
            if " ".join(str(existing_text).split()) == norm_text:
                return int(cid)
            if dim != unit_vec.size:
                continue
            ev = self._dequantize(blob, dim, float(scale or 0.0))
            if ev is None:
                continue
            cos = float(np.dot(unit_vec, ev))
            if cos >= self.dedupe_threshold:
                e_toks = self._tokens(str(existing_text))
                union = new_toks | e_toks
                jaccard = (len(new_toks & e_toks) / len(union)) if union else 1.0
                if jaccard >= 0.85:
                    return int(cid)
        return None

    def add(
        self,
        step: int,
        entries: list[tuple[str, str, str]],
        embeds: "np.ndarray",
        *,
        pinned: bool = False,
    ) -> int:
        """Store (msg_hash, role, text) rows aligned with embeds [N, D].

        Returns the number of newly inserted rows (excluding near-duplicates
        that merged into an existing chunk by incrementing its hit counter).
        """
        if not entries:
            return 0
        arr = np.asarray(embeds, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        inserted = 0
        with self._connect() as conn:
            for i, (mhash, role, text) in enumerate(entries):
                dup_id = self._find_duplicate_id(conn, role, text, arr[i])
                if dup_id is not None:
                    conn.execute(
                        "UPDATE chunks SET hits = hits + 1, last_hit_step = MAX(last_hit_step, ?), pinned = MAX(pinned, ?) WHERE id = ?",
                        (int(step), 1 if pinned else 0, dup_id),
                    )
                    continue
                blob, scale = self._quantize_i8(arr[i])
                cur = conn.execute(
                    "INSERT OR IGNORE INTO chunks (step, msg_hash, role, text, dim, vec, scale, pinned) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (int(step), mhash, role, text, int(arr.shape[1]), blob, scale, 1 if pinned else 0),
                )
                if cur.rowcount and cur.rowcount > 0:
                    inserted += 1
            if self.max_chunks > 0:
                # Hit-based priority eviction:
                # - pinned chunks survive
                # - repeatedly useful chunks (hits >= 2) outlive unretrieved churn
                # - untouched / single-hit chunks evict by recency
                conn.execute(
                    """
                    DELETE FROM chunks WHERE id NOT IN (
                        SELECT id FROM chunks
                        ORDER BY
                            pinned DESC,
                            CASE WHEN hits >= 2 THEN 1 ELSE 0 END DESC,
                            CASE WHEN hits >= 2 THEN (hits * 50 + MAX(step, last_hit_step)) ELSE step END DESC,
                            id DESC
                        LIMIT ?
                    )
                    """,
                    (self.max_chunks,),
                )
        return inserted

    def count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0

    def upsert_facts(
        self,
        facts: list[tuple[str, str, str]],
        step: int,
        *,
        pinned: bool = False,
    ) -> int:
        """Append/supersede structured (key, value, category) facts."""
        if not facts:
            return 0
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        changed = 0
        with self._connect() as conn:
            for raw_key, raw_val, category in facts:
                key = re.sub(r"\s+", " ", str(raw_key).strip().lower())[:120]
                val = re.sub(r"\s+", " ", str(raw_val).strip())[:320]
                cat = str(category or "fact").strip()[:32]
                if not key or not val:
                    continue
                conn.execute(
                    """
                    INSERT INTO facts (key, value, category, step, pinned, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        category = excluded.category,
                        step = MAX(facts.step, excluded.step),
                        pinned = MAX(facts.pinned, excluded.pinned),
                        updated_at = excluded.updated_at
                    """,
                    (key, val, cat, int(step), 1 if pinned else 0, now),
                )
                changed += 1
        return changed

    def list_facts(self, limit: int = 80) -> list[dict[str, Any]]:
        if not self.db_path.exists() or limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value, category, step, pinned
                FROM facts
                ORDER BY pinned DESC, step DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "key": str(k),
                "value": str(v),
                "category": str(c),
                "step": int(s),
                "pinned": bool(p),
            }
            for k, v, c, s, p in rows
        ]

    def facts_count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM facts").fetchone()
            return int(row[0]) if row else 0

    def search(
        self,
        query_text: str,
        query_embed: "np.ndarray",
        top_k: int = 4,
        *,
        current_step: int | None = None,
        record_hits: bool = True,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, step, role, text, dim, vec, scale, hits FROM chunks ORDER BY id ASC"
            ).fetchall()
            if not rows:
                return []

            fts_ranks: dict[int, int] = {}
            if self.fts_enabled:
                fts_q = self._fts_query(query_text)
                if fts_q:
                    try:
                        # Prefer exact-identifier matches first so rare codes/paths/keys
                        # (e.g. TK-1005, svc_node_05) are never outranked by common words.
                        fts_rows = conn.execute(
                            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) ASC LIMIT ?",
                            (fts_q, max(64, top_k * 16)),
                        ).fetchall()
                        for rank_idx, (rid,) in enumerate(fts_rows, start=1):
                            fts_ranks[int(rid)] = rank_idx
                    except sqlite3.OperationalError:
                        fts_ranks = {}

            q = np.asarray(query_embed, dtype=np.float32).reshape(-1)
            q_norm = float(np.linalg.norm(q))
            if q_norm > 0:
                q = q / q_norm
            q_toks = self._tokens(query_text)
            q_ident_toks = {t for t in q_toks if any(ch.isdigit() or ch in "_./:-" for ch in t)}

            cos_scores: dict[int, float] = {}
            lex_scores: dict[int, float] = {}
            ident_scores: dict[int, float] = {}
            row_map: dict[int, tuple[int, str, str, int]] = {}
            for cid, step, role, text, dim, blob, scale, hits in rows:
                cid_i = int(cid)
                row_map[cid_i] = (int(step), str(role), str(text), int(hits or 0))
                cos = 0.0
                if dim == q.size:
                    v = self._dequantize(blob, dim, float(scale or 0.0))
                    if v is not None:
                        cos = float(np.dot(q, v))
                cos_scores[cid_i] = cos
                lex = 0.0
                ident = 0.0
                if q_toks:
                    c_toks = self._tokens(str(text))
                    if c_toks:
                        lex = len(q_toks & c_toks) / max(1, len(q_toks))
                        if q_ident_toks:
                            ident = len(q_ident_toks & c_toks) / len(q_ident_toks)
                lex_scores[cid_i] = lex
                ident_scores[cid_i] = ident

            # Dense ranks (only positive cosine scores contribute)
            dense_sorted = sorted(
                (cid for cid, c in cos_scores.items() if c > 0.0),
                key=lambda cid: cos_scores[cid],
                reverse=True,
            )
            dense_ranks = {cid: idx for idx, cid in enumerate(dense_sorted, start=1)}

            # Lexical fallback ranks when FTS5 had no match (or is unavailable)
            if not fts_ranks and q_toks:
                lex_sorted = sorted(
                    (cid for cid, lx in lex_scores.items() if lx > 0.0),
                    key=lambda cid: lex_scores[cid],
                    reverse=True,
                )
                fts_ranks = {cid: idx for idx, cid in enumerate(lex_sorted, start=1)}

            rrf_k = 60.0
            rrf_max = 2.0 / (rrf_k + 1.0)
            scored: list[tuple[float, int]] = []
            for cid in row_map:
                rrf = 0.0
                if cid in dense_ranks:
                    rrf += 1.0 / (rrf_k + dense_ranks[cid])
                if cid in fts_ranks:
                    rrf += 1.0 / (rrf_k + fts_ranks[cid])
                rrf_norm = rrf / rrf_max
                cos_pos = max(0.0, cos_scores[cid])
                lex = lex_scores[cid]
                ident = ident_scores[cid]
                if q_ident_toks:
                    score = 0.35 * rrf_norm + 0.25 * cos_pos + 0.10 * lex + 0.30 * ident
                else:
                    score = 0.50 * rrf_norm + 0.35 * cos_pos + 0.15 * lex
                scored.append((score, cid))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            top = scored[:top_k]

            if record_hits and top:
                hit_step = int(current_step) if current_step is not None else 0
                for _, cid in top:
                    conn.execute(
                        "UPDATE chunks SET hits = hits + 1, last_hit_step = MAX(last_hit_step, ?) WHERE id = ?",
                        (hit_step, cid),
                    )

            out: list[dict[str, Any]] = []
            for score, cid in top:
                step, role, text, hits = row_map[cid]
                out.append(
                    {
                        "score": round(score, 4),
                        "step": step,
                        "role": role,
                        "text": text,
                        "hits": hits + (1 if record_hits else 0),
                    }
                )
            return out


@dataclass
class FoldStats:
    folded_messages: int
    folded_tokens: int
    step: int
    state_digest: str
    capsule_bytes: int


_READ_TOOL_NAMES = frozenset(
    {"read_file", "view_file", "open_file", "get_file", "cat", "read"}
)
_ENDPOINT_RE = re.compile(
    r"\b([a-zA-Z0-9][a-zA-Z0-9._-]*\.(?:internal|local|lan|svc|cluster|com|net|io|org|dev)(?::\d{2,5})?|[a-zA-Z0-9._-]+:\d{4,5})\b"
)
_CODE_TOKEN_RE = re.compile(r"\b([A-Z]{3,12}-\d{2,6})\b")
_BULLET_KV_RE = re.compile(r"^\s*[-*•]\s*([^:\n]{2,60}):\s*(.+?)\s*$")


def extract_facts_from_summary(summary: str) -> list[tuple[str, str, str]]:
    """Parse structured `- key: value` bullets from a fold summary."""
    out: list[tuple[str, str, str]] = []
    if not summary:
        return out
    for line in summary.splitlines():
        m = _BULLET_KV_RE.match(line)
        if m:
            k, v = m.group(1).strip(), m.group(2).strip()
            if k and v:
                out.append((k, v, "summary"))
    return out


def extract_deterministic_facts(
    entries: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Zero-LLM extraction of high-signal concrete anchors (endpoints, codes,
    explicit key: value lines) so critical entities survive even if an LLM
    summary call degrades or drops older bullets."""
    facts: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for _, role, text in entries:
        if not text:
            continue
        for ep in _ENDPOINT_RE.findall(text):
            low = ep.lower()
            if low.startswith(("http", "https")):
                continue
            k = f"endpoint:{low}"
            if k not in seen:
                seen.add(k)
                facts.append((k, ep, "endpoint"))
        for code in _CODE_TOKEN_RE.findall(text):
            k = f"code:{code.lower()}"
            if k not in seen:
                seen.add(k)
                facts.append((k, code, "code"))
        if role == "user":
            for line in text.splitlines():
                m = _BULLET_KV_RE.match(line)
                if m:
                    k, v = m.group(1).strip(), m.group(2).strip()
                    lk = k.lower()
                    if lk not in seen and len(v) <= 200:
                        seen.add(lk)
                        facts.append((k, v, "user"))
    return facts


def _parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {}


def _extract_read_path(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name.lower() not in _READ_TOOL_NAMES:
        return None
    for k in ("path", "file_path", "filepath", "file"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().replace("\\", "/")
    return None


def _extract_tool_call_meta(messages: list[dict[str, Any]]) -> dict[str, tuple[str, str | None]]:
    """Map tool_call_id / tool_use.id -> (tool_name, read_file_path_or_None)."""
    meta: dict[str, tuple[str, str | None]] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name") or tc.get("name") or "tool")
            args = _parse_tool_args(fn.get("arguments"))
            if tc_id:
                meta[str(tc_id)] = (name, _extract_read_path(name, args))
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                    name = str(b.get("name") or "tool")
                    args = _parse_tool_args(b.get("input"))
                    meta[str(b["id"])] = (name, _extract_read_path(name, args))
    return meta


def _make_archived_tool_stub(
    tool_name: str, text: str, *, superseded_path: str | None = None
) -> str:
    if superseded_path:
        return (
            f"[superseded {tool_name}: {superseded_path} "
            f"({len(text)} chars archived in MemoChain; see newer read below)]"
        )
    head = text[:240].replace("\n", " ").strip()
    tail = text[-160:].replace("\n", " ").strip() if len(text) > 440 else ""
    mid = f"... [archived {len(text)} chars of {tool_name} output in MemoChain chunk store] ..."
    return f"{head}\n{mid}\n{tail}" if tail else f"{head}\n{mid}"


def _apply_tool_stub_to_message(msg: dict[str, Any], stub: Any) -> dict[str, Any]:
    """Replace tool message content with a persisted deterministic stub."""
    out = dict(msg)
    if out.get("role") == "tool" and isinstance(stub, str):
        out["content"] = stub
        return out
    content = out.get("content")
    if isinstance(content, list) and isinstance(stub, dict):
        new_blocks: list[Any] = []
        for b in content:
            if (
                isinstance(b, dict)
                and b.get("type") == "tool_result"
                and str(b.get("tool_use_id") or "") in stub
            ):
                nb = dict(b)
                nb["content"] = stub[str(b["tool_use_id"])]
                new_blocks.append(nb)
            else:
                new_blocks.append(b)
        out["content"] = new_blocks
    return out


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
        async_fold: bool = False,
        sync_fold_ceiling_ratio: float = 1.5,
        compact_tool_outputs: bool = True,
        tool_archive_min_chars: int = 1500,
        tool_keep_recent_groups: int = 4,
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
        self.async_fold = bool(async_fold)
        self.sync_fold_ceiling_ratio = max(1.0, float(sync_fold_ceiling_ratio))
        self.compact_tool_outputs = bool(compact_tool_outputs)
        self.tool_archive_min_chars = max(200, int(tool_archive_min_chars))
        self.tool_keep_recent_groups = max(1, int(tool_keep_recent_groups))
        self.chunk_store = (
            ChunkStore(self.dir / "chunks.db", max_chunks=max_chunks) if recall_top_k > 0 else None
        )

        self._lock = threading.RLock()
        self._fold_thread: threading.Thread | None = None
        self._fold_in_progress = False
        self._last_fold_error: str | None = None

        self.folded_hashes: list[str] = []
        self._folded_set: set[str] = set()
        self.compacted_stubs: dict[str, Any] = {}
        self.summary = ""
        self.fold_fallback_count = 0
        self.fold_shrink_guard_count = 0
        self.last_fold_fallback = False
        self.state: MemoryState | None = None
        self.head: ChainHead | None = None
        self.tampered = False
        self._meta_mtime_ns: int = -1
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
        with self._lock:
            if not self.meta_path.exists():
                return
            self._meta_mtime_ns = self.meta_path.stat().st_mtime_ns
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.folded_hashes = list(meta.get("folded_hashes", []))
            self._folded_set = set(self.folded_hashes)
            self.compacted_stubs = dict(meta.get("compacted_stubs") or {})
            self.summary = meta.get("summary", "")
            self.fold_fallback_count = int(meta.get("fold_fallback_count", 0))
            self.fold_shrink_guard_count = int(meta.get("fold_shrink_guard_count", 0))
            self.last_fold_fallback = bool(meta.get("last_fold_fallback", False))
            self.tampered = False
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

    def sync_from_disk(self) -> None:
        """Reload meta/capsule if another proxy process updated this session."""
        with self._lock:
            if self._fold_in_progress or not self.meta_path.exists():
                return
            mtime_ns = self.meta_path.stat().st_mtime_ns
            if mtime_ns != self._meta_mtime_ns:
                self._load()

    def _save(self) -> None:
        with self._lock:
            # Cap persisted compacted_stubs to newest 500 entries so meta.json stays small.
            if len(self.compacted_stubs) > 500:
                keys = list(self.compacted_stubs.keys())[-500:]
                self.compacted_stubs = {k: self.compacted_stubs[k] for k in keys}
            meta = {
                "session_id": self.session_id,
                "step": self.head.step if self.head else 0,
                "state_digest": self.head.state_digest if self.head else "",
                "accumulator": self.head.accumulator if self.head else "",
                "folded_hashes": self.folded_hashes,
                "compacted_stubs": self.compacted_stubs,
                "summary": self.summary,
                "fold_fallback_count": self.fold_fallback_count,
                "fold_shrink_guard_count": self.fold_shrink_guard_count,
                "last_fold_fallback": self.last_fold_fallback,
                "weights_digest": self.writer.weights_digest(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            tmp = self.meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.meta_path)
            self._meta_mtime_ns = self.meta_path.stat().st_mtime_ns

    def _append_receipts(self, rows: list[dict[str, Any]]) -> None:
        with open(self.receipts_path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def wait_for_pending_fold(self, timeout: float | None = 30.0) -> bool:
        """Block until any in-flight background fold finishes. Returns True if idle."""
        thread = self._fold_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        return not (self._fold_thread is not None and self._fold_thread.is_alive())

    # ---- memory operations ----------------------------------------------

    def _apply_persisted_tool_stubs(
        self, hot: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.compacted_stubs:
            return list(hot)
        out: list[dict[str, Any]] = []
        for m in hot:
            h = message_hash(m)
            stub = self.compacted_stubs.get(h)
            out.append(_apply_tool_stub_to_message(m, stub) if stub is not None else m)
        return out

    def _compact_hot_tool_outputs(
        self, hot: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Cache-safe Phase B compaction: called ONLY when conv_tokens > hot_limit.

        1. Supersedes earlier `read_file` outputs when the exact same file path
           was read again later in `hot`.
        2. Archives old large tool outputs (older than `tool_keep_recent_groups`
           and >= `tool_archive_min_chars`) into `chunk_store` and replaces them
           with deterministic compact stubs persisted in `self.compacted_stubs`.
        Returns (compacted_hot, compacted_count, tokens_saved).
        """
        if not self.compact_tool_outputs or len(hot) <= 2:
            return list(hot), 0, 0

        groups = eviction_groups(hot)
        if len(groups) <= self.tool_keep_recent_groups:
            return list(hot), 0, 0
        protected_start = groups[-self.tool_keep_recent_groups][0]

        tool_meta = _extract_tool_call_meta(hot)

        # Find the latest message index where each file path was read
        latest_read_idx: dict[str, int] = {}
        for idx, m in enumerate(hot):
            if m.get("role") == "tool":
                tc_id = str(m.get("tool_call_id") or "")
                _, rpath = tool_meta.get(tc_id, ("tool", None))
                if rpath:
                    latest_read_idx[rpath] = idx
            elif m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tid = str(b.get("tool_use_id") or "")
                        _, rpath = tool_meta.get(tid, ("tool", None))
                        if rpath:
                            latest_read_idx[rpath] = idx

        out = list(hot)
        archive_entries: list[tuple[str, str, str]] = []
        compacted_count = 0
        tokens_saved = 0
        step_tag = self.head.step if self.head else 0

        for idx in range(protected_start):
            m = out[idx]
            orig_hash = message_hash(m)
            if orig_hash in self.compacted_stubs:
                continue

            # OpenAI role=="tool"
            if m.get("role") == "tool":
                raw_text = message_text(m)
                if not raw_text:
                    continue
                tc_id = str(m.get("tool_call_id") or "")
                tool_name, rpath = tool_meta.get(tc_id, (str(m.get("name") or "tool"), None))
                superseded = bool(rpath and latest_read_idx.get(rpath, -1) > idx)
                if not superseded and len(raw_text) < self.tool_archive_min_chars:
                    continue
                stub = _make_archived_tool_stub(
                    tool_name, raw_text, superseded_path=rpath if superseded else None
                )
                if len(stub) >= len(raw_text):
                    continue
                archive_entries.append((f"tool:{orig_hash}", "tool", raw_text[: self.max_doc_chars]))
                self.compacted_stubs[orig_hash] = stub
                out[idx] = _apply_tool_stub_to_message(m, stub)
                compacted_count += 1
                tokens_saved += max(0, estimate_tokens(raw_text) - estimate_tokens(stub))
                continue

            # Anthropic user message containing tool_result blocks
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                block_stubs: dict[str, str] = {}
                for b in m["content"]:
                    if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                        continue
                    tid = str(b.get("tool_use_id") or "")
                    inner = b.get("content")
                    if isinstance(inner, str):
                        raw_text = inner
                    elif isinstance(inner, list):
                        raw_text = "\n".join(
                            str(sub.get("text", ""))
                            for sub in inner
                            if isinstance(sub, dict) and isinstance(sub.get("text"), str)
                        )
                    else:
                        continue
                    if not raw_text:
                        continue
                    tool_name, rpath = tool_meta.get(tid, ("tool", None))
                    superseded = bool(rpath and latest_read_idx.get(rpath, -1) > idx)
                    if not superseded and len(raw_text) < self.tool_archive_min_chars:
                        continue
                    stub = _make_archived_tool_stub(
                        tool_name, raw_text, superseded_path=rpath if superseded else None
                    )
                    if len(stub) >= len(raw_text):
                        continue
                    block_stubs[tid] = stub
                    archive_entries.append(
                        (f"tool:{orig_hash}:{tid}", "tool", raw_text[: self.max_doc_chars])
                    )
                    tokens_saved += max(0, estimate_tokens(raw_text) - estimate_tokens(stub))
                if block_stubs:
                    self.compacted_stubs[orig_hash] = block_stubs
                    out[idx] = _apply_tool_stub_to_message(m, block_stubs)
                    compacted_count += len(block_stubs)

        if archive_entries and self.chunk_store is not None:
            texts = [e[2] for e in archive_entries]
            vecs = self.embed_fn(texts).astype(np.float32)
            self.chunk_store.add(step_tag, archive_entries, vecs)
        if compacted_count > 0:
            self._save()
        return out, compacted_count, tokens_saved

    def _fold_messages(
        self,
        msgs: list[dict[str, Any]],
        model_hint: str | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> FoldStats:
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

        with self._lock:
            init_state = self.state
            init_head = self.head
            prev_summary = self.summary

        state, head, receipt_rows = self.writer.fold_with_receipts(
            q14, initial_state=init_state, initial_head=init_head
        )
        if self.chunk_store is not None and entries:
            self.chunk_store.add(head.step, entries, docs_f[: len(entries)])
        if self.proof_queue is not None and q14.shape[1] == 2560:
            # raw Q14 little-endian i16 docs for the SP1 IVC queue
            self.proof_queue.submit(self.session_id, q14.tobytes())

        cap = Capsule(state=state, step=head.step, head=head.accumulator)
        raw = cap.to_bytes()
        assert state.nbytes == STATE_BYTES

        # query embedding for the digest reader: normalized mean of folded docs
        q = docs_f.mean(axis=0)
        norm = float(np.linalg.norm(q)) or 1.0
        if hasattr(self.reader, "read_fold"):
            try:
                candidate = self.reader.read_fold(
                    state,
                    q / norm,
                    prev_summary,
                    entries,
                    model_hint=model_hint,
                    auth_headers=auth_headers,
                )
            except TypeError:
                try:
                    candidate = self.reader.read_fold(
                        state, q / norm, prev_summary, entries, model_hint=model_hint
                    )
                except TypeError:
                    candidate = self.reader.read_fold(state, q / norm, prev_summary, entries)
        else:
            candidate = self.reader.read(state, q / norm, prev_summary)

        fallback_used = bool(getattr(self.reader, "last_call_fallback", False))
        candidate = (candidate or "").strip()

        # Shrinkage guard: refuse catastrophic summary amnesia when a fold model
        # drops more than 55% of an established summary (e.g. reasoning-token exhaustion).
        shrink_guarded = False
        if (
            len(prev_summary.strip()) >= 120
            and (not candidate or len(candidate) < int(0.45 * len(prev_summary.strip())))
        ):
            final_summary = prev_summary
            shrink_guarded = True
        else:
            final_summary = candidate or prev_summary

        # Append-only structured facts ledger (deterministic + summary bullets)
        if self.chunk_store is not None:
            facts = extract_deterministic_facts(entries) + extract_facts_from_summary(final_summary)
            if facts:
                self.chunk_store.upsert_facts(facts, step=head.step)

        with self._lock:
            self.state, self.head = state, head
            self.summary = final_summary
            self.last_fold_fallback = fallback_used
            if fallback_used:
                self.fold_fallback_count += 1
            if shrink_guarded:
                self.fold_shrink_guard_count += 1
            tmp = self.capsule_path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, self.capsule_path)
            self._append_receipts(receipt_rows)
            for m in msgs:
                h = message_hash(m)
                if h not in self._folded_set:
                    self.folded_hashes.append(h)
                    self._folded_set.add(h)
            self._save()

        return FoldStats(
            folded_messages=len(msgs),
            folded_tokens=sum(estimate_tokens(message_text(m)) for m in msgs),
            step=head.step,
            state_digest=head.state_digest,
            capsule_bytes=len(raw),
        )

    def _schedule_background_fold(
        self,
        to_fold: list[dict[str, Any]],
        model_hint: str | None,
        auth_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._fold_in_progress:
                return {"status": "in_progress"}
            self._fold_in_progress = True
            self._last_fold_error = None
            snapshot = [dict(m) for m in to_fold]
            hdrs = dict(auth_headers) if auth_headers else None
            tokens = sum(estimate_tokens(message_text(m)) for m in snapshot)

            def _worker() -> None:
                try:
                    self._fold_messages(snapshot, model_hint=model_hint, auth_headers=hdrs)
                except Exception as exc:  # pragma: no cover - defensive
                    with self._lock:
                        self._last_fold_error = str(exc)
                finally:
                    with self._lock:
                        self._fold_in_progress = False

            t = threading.Thread(
                target=_worker,
                name=f"memochain-fold-{self.session_id[:20]}",
                daemon=True,
            )
            self._fold_thread = t
            t.start()
            return {
                "status": "scheduled",
                "messages": len(snapshot),
                "tokens": tokens,
            }

    def process(
        self,
        messages: list[dict[str, Any]],
        model_hint: str | None = None,
        inject_memory: bool = True,
        auth_headers: dict[str, str] | None = None,
        passthrough: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fold overflow and return (rewritten_messages, stats).

        Invariants preserved on the rewritten message list:
        - Leading system messages stay intact (never evicted into the capsule).
        - At least one user message remains in the hot tail whenever the input
          contains a user message (chat templates such as Qwen3 reject prompts
          with no user turn).
        - Assistant tool_calls stay grouped with their tool results.

        When ``passthrough`` is True the session is forwarded unchanged: no
        fold, no memory injection. Used for cache-heavy providers where
        caching already covers the hot window.
        """
        if passthrough:
            sys_end = 0
            while sys_end < len(messages) and messages[sys_end].get("role") == "system":
                sys_end += 1
            sys_msgs = messages[:sys_end]
            conv = messages[sys_end:]
            return list(messages), {
                "session_id": self.session_id,
                "already_folded": 0,
                "hot_messages": len(messages),
                "system_tokens": sum(estimate_tokens(message_text(m)) for m in sys_msgs),
                "hot_tokens": sum(estimate_tokens(message_text(m)) for m in conv),
                "fold": None,
                "passthrough": True,
            }
        # Keep leading system prompt(s) out of folding so instructions/tools
        # stay verbatim and we inject [Durable memory] into the real system msg.
        sys_end = 0
        while sys_end < len(messages) and messages[sys_end].get("role") == "system":
            sys_end += 1
        sys_msgs = messages[:sys_end]
        conv = messages[sys_end:]

        hashes = [message_hash(m) for m in conv]

        with self._lock:
            folded_set = set(self._folded_set)

        # longest already-folded prefix of the non-system conversation
        prefix = 0
        while prefix < len(hashes) and hashes[prefix] in folded_set:
            prefix += 1

        # Never leave the hot tail without a user turn if the input had one.
        if not any(m.get("role") == "user" for m in conv[prefix:]):
            last_user = next(
                (i for i in range(len(conv) - 1, -1, -1) if conv[i].get("role") == "user"),
                None,
            )
            if last_user is not None:
                prefix = last_user

        raw_hot = conv[prefix:]
        # Re-apply any previously compacted tool stubs so the hot prefix stays
        # 100% byte-identical across turns after a compaction boundary.
        with self._lock:
            hot = self._apply_persisted_tool_stubs(raw_hot)
        sys_tokens = sum(estimate_tokens(message_text(m)) for m in sys_msgs)
        conv_tokens = sum(estimate_tokens(message_text(m)) for m in hot)
        stats: dict[str, Any] = {
            "session_id": self.session_id,
            "already_folded": prefix,
            "hot_messages": len(sys_msgs) + len(hot),
            "system_tokens": sys_tokens,
            "hot_tokens": conv_tokens,
            "fold": None,
        }

        # Phase B: when overflowing hot_limit, first compact superseded file
        # reads and old oversized tool outputs (cache-safe: only triggers at
        # overflow boundaries and persists deterministic stubs).
        if conv_tokens > self.hot_limit and self.compact_tool_outputs:
            with self._lock:
                hot, compacted_n, tokens_saved = self._compact_hot_tool_outputs(hot)
            if compacted_n > 0:
                conv_tokens = sum(estimate_tokens(message_text(m)) for m in hot)
                stats["hot_tokens"] = conv_tokens
                stats["tool_compactions"] = {
                    "count": compacted_n,
                    "tokens_saved": tokens_saved,
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
                # Fold the original (un-stubbed) messages so hashes match future
                # client resends and the capsule + chunk store capture full text.
                to_fold = raw_hot[:evict_end]
                sync_ceiling = int(self.hot_limit * self.sync_fold_ceiling_ratio)
                if self.async_fold and conv_tokens <= sync_ceiling:
                    stats["fold"] = self._schedule_background_fold(
                        to_fold, model_hint=model_hint, auth_headers=auth_headers
                    )
                else:
                    if self._fold_in_progress:
                        self.wait_for_pending_fold()
                    fold_stats = self._fold_messages(
                        to_fold, model_hint=model_hint, auth_headers=auth_headers
                    )
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
                cur_step = self.head.step if self.head else 0
                recalled = self.chunk_store.search(
                    last_user_text,
                    q_vec,
                    top_k=self.recall_top_k,
                    current_step=cur_step,
                )
                stats["recalled_chunks"] = len(recalled)

        static_block, dynamic_block = self.format_memory_sections(recalled=recalled)
        memory_block = self.format_memory_block(recalled=recalled)
        if static_block:
            stats["static_memory_block"] = static_block
        if dynamic_block:
            stats["dynamic_memory_block"] = dynamic_block
        if memory_block:
            stats["memory_block"] = memory_block
        rewritten = (
            self._inject_memory(
                list(sys_msgs) + list(hot),
                static_block=static_block,
                dynamic_block=dynamic_block,
            )
            if inject_memory
            else list(sys_msgs) + list(hot)
        )
        return rewritten, stats

    def format_memory_sections(
        self, recalled: list[dict[str, Any]] | None = None
    ) -> tuple[str, str]:
        """Split durable memory into two cache-aware pieces:

        1. ``static_block``: the folded summary / fact sheet + any append-only
           verified facts not already covered by the summary. Byte-identical
           across turns between folds, so injecting it into the system prompt
           preserves provider KV-cache prefixes (Anthropic, OpenAI, Gemini,
           local vLLM APC / llama.cpp slot cache).
        2. ``dynamic_block``: per-turn top-K recalled chunks. Changes every
           turn with the user's latest query, so injecting it at the tail
           (never into ``messages[0]``) avoids invalidating the cached prefix.
        """
        with self._lock:
            summary = self.summary
        extra_facts: list[str] = []
        if self.chunk_store is not None:
            summary_lower = summary.lower()
            for row in self.chunk_store.list_facts(limit=40):
                val = row["value"]
                if val.lower() in summary_lower:
                    continue
                extra_facts.append(f"- {row['key']}: {val}")
                if len(extra_facts) >= 20:
                    break

        static_block = ""
        if summary or extra_facts:
            parts = [
                "[Durable memory]",
                "Compressed notes from earlier parts of this session (older than what you can see here):",
            ]
            if summary:
                parts.append(summary)
            if extra_facts:
                parts.append("")
                parts.append("Verified facts ledger (append-only across folds):")
                parts.extend(extra_facts)
            static_block = "\n".join(parts)

        dynamic_block = ""
        if recalled:
            lines = [
                "[Recalled memory]",
                "Relevant verbatim turns recalled for the current question:",
            ]
            for item in recalled:
                snippet = item["text"].replace("\n", " ").strip()
                if len(snippet) > 600:
                    snippet = snippet[:600] + "..."
                lines.append(f"- [{item['role']}] {snippet}")
            dynamic_block = "\n".join(lines)
        return static_block, dynamic_block

    def format_memory_block(self, recalled: list[dict[str, Any]] | None = None) -> str:
        """Combined single-string view of durable memory (kept for back-compat)."""
        static_block, dynamic_block = self.format_memory_sections(recalled=recalled)
        if static_block and dynamic_block:
            return f"{static_block}\n\n{dynamic_block}"
        return static_block or dynamic_block

    @staticmethod
    def inject_dynamic_recall(
        messages: list[dict[str, Any]], dynamic_block: str
    ) -> list[dict[str, Any]]:
        """Attach ``dynamic_block`` onto the last user message so the leading
        prefix (system prompt + fact sheet + earlier turns) stays immutable
        for KV-cache reuse. Works for both OpenAI and Anthropic message shapes
        (preserves alternating user/assistant turns and structured content lists).
        """
        if not dynamic_block:
            return list(messages)
        out = [dict(m) for m in messages]
        last_user = next(
            (i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"),
            None,
        )
        if last_user is None:
            return out
        msg = dict(out[last_user])
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = f"{dynamic_block}\n\n{content}" if content else dynamic_block
        elif isinstance(content, list):
            # Prepend a text block (or append after tool_result blocks so
            # Anthropic's tool_result-first requirement stays valid).
            blocks = list(content)
            insert_at = 0
            while insert_at < len(blocks) and isinstance(blocks[insert_at], dict) and blocks[insert_at].get("type") == "tool_result":
                insert_at += 1
            blocks.insert(insert_at, {"type": "text", "text": dynamic_block})
            msg["content"] = blocks
        else:
            msg["content"] = dynamic_block
        out[last_user] = msg
        return out

    def _inject_memory(
        self,
        hot: list[dict[str, Any]],
        recalled: list[dict[str, Any]] | None = None,
        memory_block: str | None = None,
        static_block: str | None = None,
        dynamic_block: str | None = None,
    ) -> list[dict[str, Any]]:
        if static_block is None and dynamic_block is None:
            if memory_block is not None:
                static_block = memory_block
                dynamic_block = ""
            else:
                static_block, dynamic_block = self.format_memory_sections(recalled=recalled)
        out = [dict(m) for m in hot]
        if static_block:
            block = "\n\n" + static_block
            if out and out[0].get("role") == "system":
                first = dict(out[0])
                c = first.get("content")
                if isinstance(c, str):
                    first["content"] = c + block
                elif isinstance(c, list):
                    first["content"] = list(c) + [{"type": "text", "text": static_block}]
                else:
                    first["content"] = static_block
                out[0] = first
            else:
                out.insert(0, {"role": "system", "content": static_block})
        if dynamic_block:
            out = self.inject_dynamic_recall(out, dynamic_block)
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
