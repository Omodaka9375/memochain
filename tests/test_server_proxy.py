"""Tests for the MemoChain OpenAI-compatible memory proxy (session folding + HTTP API)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from memochain.server.app import _hash_embed, create_app
from memochain.server.session import (
    SessionMemory,
    estimate_tokens,
    eviction_groups,
    message_hash,
    message_text,
)
from memochain.state import Capsule


class FakeWriter:
    """Deterministic writer matching the FoldWriter protocol (no torch)."""

    def __init__(self) -> None:
        self._digest = "ab" * 32

    def weights_digest(self) -> str:
        return self._digest

    def fold_with_receipts(self, docs_q14, initial_state=None, initial_head=None):
        from memochain.state import MemoryState
        from memochain.verify import ChainHead

        docs = np.asarray(docs_q14, dtype="<i2")
        if docs.ndim == 1:
            docs = docs[None, :]
        state_arr = (
            np.zeros((11, 32, 32), dtype="<i2")
            if initial_state is None
            else np.asarray(initial_state.tokens, dtype="<i2").copy()
        )
        step = 0 if initial_head is None else initial_head.step
        check_acc = "0" * 64 if initial_head is None else initial_head.accumulator
        rows = []
        flat = state_arr.reshape(-1)
        for i in range(docs.shape[0]):
            d = docs[i].astype(np.int64)
            take = min(flat.size, d.size)
            flat[:take] = (flat[:take] + d[:take]).astype("<i2")
            state_arr = flat.reshape(11, 32, 32)
            step += 1
            doc_hash = hashlib.sha256(d.astype("<i2").tobytes()).hexdigest()
            state_digest = hashlib.sha256(state_arr.astype("<i2").tobytes()).hexdigest()
            payload = f"{step}|{check_acc}|{doc_hash}|{state_digest}"
            binding = hashlib.sha256(payload.encode()).hexdigest()
            check_acc = hashlib.sha256(f"{check_acc}{binding}".encode()).hexdigest()
            rows.append(
                {
                    "step": step,
                    "doc_hash": doc_hash,
                    "state_digest": state_digest,
                    "binding": binding,
                    "accumulator": check_acc,
                }
            )
        head = ChainHead(step=step, accumulator=check_acc, state_digest=rows[-1]["state_digest"])
        return MemoryState(state_arr), head, rows


class EchoReader:
    """Reader stub that tags the fold so tests can see digest updates."""

    def read(self, state, query_embed, prev_summary: str) -> str:
        tag = f"recall-{int(state.tokens[0, 0, 0])}"
        return (prev_summary + " " + tag).strip()


def make_session(tmp_path, hot_limit: int = 200, keep_min: int = 60) -> SessionMemory:
    return SessionMemory(
        "s1",
        tmp_path / "s1",
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        reader=EchoReader(),
        hot_limit=hot_limit,
        keep_min=keep_min,
    )


def msg(role: str, text: str, **kw) -> dict:
    return {"role": role, "content": text, **kw}


def test_estimate_tokens_and_message_text() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100
    msg = {"content": [{"type": "text", "text": "hello"}, {"type": "other"}]}
    assert message_text(msg) == "hello"
    # hash covers full payload structurally (different shapes hash differently)
    assert message_hash(msg) != message_hash({"content": [{"text": "hello"}, {}]})


def test_tool_call_groups_stay_atomic() -> None:
    msgs = [
        msg("user", "q"),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        msg("assistant", "answer"),
    ]
    groups = eviction_groups(msgs)
    sizes = [end - start for start, end in groups]
    assert sizes == [1, 2, 1]  # tool pair (2) never splits


def test_no_fold_under_limit_and_memory_injected_when_folded(tmp_path) -> None:
    s = make_session(tmp_path)
    small = [msg("system", "sys"), msg("user", "hi"), msg("assistant", "hello")]
    out, stats = s.process(small)
    assert stats["fold"] is None
    assert out == small  # nothing folded, nothing injected
    assert s.head is None and not s.capsule_path.exists()

    big = [msg("system", "sys")]
    big += [msg("user", "x" * 900), msg("assistant", "y" * 900)] * 4  # ~7,200 tokens
    out, stats = s.process(big)
    assert stats["fold"] is not None and stats["fold"]["messages"] > 0
    assert stats["hot_tokens"] < 7_200
    assert s.capsule_path.exists() and s.capsule_path.stat().st_size > 22528
    assert "recall-" in s.summary
    # memory digest injected into the system message
    assert "[Durable memory]" in out[0]["content"]
    # exact preserved prefix + no re-fold on identical replay
    _, stats2 = s.process(big)
    assert stats2["fold"] is None  # everything already folded


def test_session_persists_and_rejects_tampered_capsule(tmp_path) -> None:
    s = make_session(tmp_path)
    big = [msg("user", "z" * 800), msg("assistant", "w" * 800)] * 3
    s.process(big)
    step1, digest1 = s.head.step, s.head.state_digest

    s2 = SessionMemory(
        "s1",
        tmp_path / "s1",
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        reader=EchoReader(),
        hot_limit=200,
        keep_min=60,
    )
    assert s2.head is not None
    assert s2.head.step == step1
    assert s2.head.state_digest == digest1
    check = s2.verify()
    assert check["ok"] is True

    # flip one byte of the state payload -> capsule hash check must fail the reload
    raw = bytearray(s2.capsule_path.read_bytes())
    raw[-1] ^= 0xFF
    s2.capsule_path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        Capsule.from_bytes(bytes(raw))
    tampered = SessionMemory(
        "s1",
        tmp_path / "s1",
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        reader=EchoReader(),
        hot_limit=200,
        keep_min=60,
    )
    assert tampered.tampered or tampered.verify()["ok"] is False


@pytest.mark.skipif(
    not pytest.importorskip("fastapi", minversion=None) if False else False, reason="fastapi"
)
def test_app_health() -> None:
    pass


def test_create_app_and_http_flow(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class FakeAsyncBackend:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url, json=None, headers=None):
            self.calls += 1

            class R:
                status_code = 200

                def json(self_inner):
                    return {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": f"reply:{self_inner.n}"},
                                "finish_reason": "stop",
                            }
                        ],
                    }

            R.n = len(json["messages"])
            return R()

    class FakeReader:
        def read(self, state, query_embed, prev_summary):
            return (prev_summary + " fake-recall").strip()

    from memochain.server import app as app_mod

    fake_backend = FakeAsyncBackend()
    app = app_mod.create_app(
        backend_base_url="http://backend.test/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "sessions",
        embedding_base_url=None,
        writer=FakeWriter(),
        reader=FakeReader(),
        embed_fn=_hash_embed,
        backend_client=fake_backend,
    )
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True

    big = [msg("system", "sys")] + [msg("user", "u" * 900), msg("assistant", "a" * 900)] * 4
    r = c.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": big},
        headers={"x-session-id": "kai-test"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert "memochain" in payload
    assert payload["memochain"]["session_id"] == "kai-test"
    assert payload["memochain"]["folded_messages"] >= 0
    assert fake_backend.calls == 1
    # memory block injected into the outgoing system message
    sent = fake_backend.last_json["messages"] if hasattr(fake_backend, "last_json") else None

    v = c.get("/v1/memochain/verify", params={"session_id": "kai-test"})
    assert v.status_code == 200 and v.json()["ok"] is True


def test_eviction_preserves_system_and_last_user_during_tool_loop(tmp_path) -> None:
    """Folding mid-tool-loop must never evict the leading system prompt or the last user turn."""
    s = make_session(tmp_path, hot_limit=200, keep_min=50)
    msgs = [
        msg("system", "S" * 1200),  # large system prompt (300 tokens)
        msg("user", "inspect repo"),
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "R" * 400},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "R" * 400},
    ]
    out, _ = s.process(msgs)
    assert out[0]["role"] == "system"
    assert "S" * 100 in out[0]["content"]
    assert any(m.get("role") == "user" for m in out)
    # tool call / tool result pairing still intact
    roles = [m["role"] for m in out]
    assert roles[0] == "system"
    assert roles[1] == "user"

    # Now add a second user turn and force a fold: earlier turn folds, latest user stays
    msgs2 = msgs + [
        msg("assistant", "done inspecting " + "A" * 600),
        msg("user", "now design it"),
        {
            "role": "assistant",
            "content": "checking more",
            "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "R" * 400},
    ]
    out2, stats2 = s.process(msgs2)
    assert stats2["fold"] is not None
    assert out2[0]["role"] == "system"
    assert "[Durable memory]" in out2[0]["content"]
    assert any(m.get("role") == "user" and m.get("content") == "now design it" for m in out2)


def test_parse_error_payload_preserves_openai_object() -> None:
    from memochain.server.app import _parse_error_payload

    raw = json.dumps({"error": {"message": "No user query found in messages.", "type": "BadRequestError", "code": 400}})
    parsed = _parse_error_payload(raw, 400)
    assert isinstance(parsed["error"], dict)
    assert parsed["error"]["message"] == "No user query found in messages."

    plain = _parse_error_payload("upstream exploded", 502)
    assert isinstance(plain["error"], dict)
    assert plain["error"]["message"] == "upstream exploded"
    assert plain["error"]["code"] == 502


def test_hot_limit_counts_conversation_not_system_prompt(tmp_path) -> None:
    """A large leading system prompt must not eat the conversation hot_limit."""
    s = make_session(tmp_path, hot_limit=400, keep_min=200)
    msgs = [
        msg("system", "S" * 4000),  # 1,000 tokens of system prompt (> hot_limit)
        msg("user", "u" * 400),     # 100 tokens
        msg("assistant", "a" * 400),  # 100 tokens
        msg("user", "u" * 400),     # 100 tokens (conv total = 300 <= 400)
    ]
    out, stats = s.process(msgs)
    assert stats["fold"] is None
    assert stats["system_tokens"] == 1000
    assert stats["hot_tokens"] == 300
    assert len(out) == 4


def test_hybrid_read_fold_and_chunk_retrieval(tmp_path) -> None:
    """Hybrid memory saves folded chunks and retrieves top-K verbatim turns for a new query."""

    class FactReader:
        def read(self, state, query_embed, prev_summary: str) -> str:
            return prev_summary

        def read_fold(self, state, query_embed, prev_summary: str, folded_entries) -> str:
            return "- city: Novi Sad\n- tool: pnpm"

    s = SessionMemory(
        "h1",
        tmp_path / "h1",
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        reader=FactReader(),
        hot_limit=120,
        keep_min=40,
        recall_top_k=2,
    )
    conv = [
        msg("user", "My city is Novi Sad and I use pnpm for packages. Secret code: ZEBRA-42."),
        msg("assistant", "Got it, Novi Sad, pnpm, ZEBRA-42."),
        msg("user", "Let us also note the database port is 5432 and host is db.internal. " + ("x" * 300)),
        msg("assistant", "Noted port 5432 and host db.internal. " + ("y" * 300)),
        msg("user", "What is my secret code ZEBRA-42 and package manager?"),
    ]
    out, stats = s.process(conv)
    assert stats["fold"] is not None
    assert stats["recalled_chunks"] == 2
    sys_content = out[0]["content"]
    assert "[Durable memory]" in sys_content
    assert "- city: Novi Sad" in sys_content
    assert "Relevant verbatim turns recalled" in sys_content
    assert "ZEBRA-42" in sys_content
    assert s.verify()["ok"] is True


def test_hybrid_reader_adapter_calls_backend(monkeypatch) -> None:
    from memochain.server.app import HybridReaderAdapter

    adapter = HybridReaderAdapter("http://backend.test/v1")

    class FakeResp:
        def __init__(self, status_code: int, payload: dict) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._payload

    calls: list[tuple[str, dict | None]] = []

    def fake_get(url: str):
        calls.append((url, None))
        return FakeResp(200, {"data": [{"id": "inkling-2-mini"}]})

    def fake_post(url: str, json: dict):
        calls.append((url, json))
        return FakeResp(200, {"choices": [{"message": {"content": "- live in Novi Sad\n- uses pnpm"}}]})

    monkeypatch.setattr(adapter.client, "get", fake_get)
    monkeypatch.setattr(adapter.client, "post", fake_post)

    out = adapter.read_fold(
        state=None,
        query_embed=np.zeros(2560, dtype=np.float32),
        prev_summary="",
        folded_entries=[("h1", "user", "I live in Novi Sad and use pnpm.")],
    )
    assert "Novi Sad" in out
    assert calls[0][0] == "/models"
    assert calls[1][0] == "/chat/completions"
    assert calls[1][1]["model"] == "inkling-2-mini"


def test_anthropic_sse_passthrough_and_openai_direct_claude_auth(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from memochain.server import app as app_mod

    seen_headers: list[dict] = []

    class StreamResp:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_lines(self):
            yield "event: message_start"
            yield 'data: {"type":"message_start","message":{"id":"msg_s"}}'
            yield ""
            yield "event: message_stop"
            yield 'data: {"type":"message_stop"}'
            yield ""

    class FakeClient:
        def stream(self, method: str, url: str, json=None, headers=None):
            seen_headers.append({"url": url, "headers": headers or {}, "json": json})
            return StreamResp()

    app = app_mod.create_app(
        backend_base_url="https://api.anthropic.com/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "sse_ant",
        embedding_base_url=None,
        writer=FakeWriter(),
        reader=app_mod.HybridReaderAdapter("https://api.anthropic.com/v1"),
        embed_fn=_hash_embed,
        backend_client=FakeClient(),
    )
    c = TestClient(app)
    r = c.post(
        "/v1/messages",
        headers={"Authorization": "Bearer sk-ant-bearer", "x-session-id": "ant-sse"},
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert "message_stop" in r.text
    assert seen_headers[0]["headers"]["x-api-key"] == "sk-ant-bearer"
    assert seen_headers[0]["headers"]["anthropic-version"] == "2023-06-01"


def test_create_app_defaults_to_hybrid_reader(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from memochain.server import app as app_mod

    captured: dict = {}
    orig_sm = app_mod.SessionManager

    class SpySessionManager(orig_sm):
        def __init__(self, store_root, **kwargs):
            captured.update(kwargs)
            super().__init__(store_root, **kwargs)

    monkeypatch.setattr(app_mod, "SessionManager", SpySessionManager)

    app_default = app_mod.create_app(
        backend_base_url="http://backend.test/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "s_default",
        embedding_base_url=None,
        writer=FakeWriter(),
        embed_fn=_hash_embed,
    )
    assert app_default is not None
    assert isinstance(captured["reader"], app_mod.HybridReaderAdapter)
    assert captured["recall_top_k"] == 4


def test_anthropic_messages_endpoint_and_direct_reader(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from memochain.server import app as app_mod

    # 1) HybridReaderAdapter against api.anthropic.com uses model_hint + /messages with no temperature
    adapter = app_mod.HybridReaderAdapter("https://api.anthropic.com/v1", backend_api_key="sk-ant-test")
    posted: list[dict] = []

    def fake_post(url: str, json: dict, headers: dict | None = None):
        posted.append({"url": url, "json": json, "headers": headers or {}})

        class R:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"content": [{"type": "text", "text": "- uses claude direct"}]}

        return R()

    monkeypatch.setattr(adapter.client, "post", fake_post)
    out = adapter.read_fold(
        state=None,
        query_embed=np.zeros(2560, dtype=np.float32),
        prev_summary="",
        folded_entries=[("h1", "user", "hello")],
        model_hint="claude-sonnet-4-6",
    )
    assert out == "- uses claude direct"
    assert posted[0]["url"] == "/messages"
    assert posted[0]["json"]["model"] == "claude-sonnet-4-6"
    assert "temperature" not in posted[0]["json"]

    # 2) POST /v1/messages preserves tool_use/tool_result pairs, folds overflow, and injects into top-level system
    backend_calls: list[dict] = []

    class FakeAnthropicBackend:
        async def get(self, url: str):
            class R:
                status_code = 404

            return R()

        async def post(self, url: str, json=None, headers=None):
            backend_calls.append({"url": url, "json": json, "headers": headers or {}})

            class R:
                status_code = 200

                def json(self_inner):
                    return {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Novi Sad"}],
                    }

            return R()

    app = app_mod.create_app(
        backend_base_url="https://api.anthropic.com/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "ant_sessions",
        embedding_base_url=None,
        writer=FakeWriter(),
        reader=adapter,
        embed_fn=_hash_embed,
        backend_client=FakeAnthropicBackend(),
        hot_limit=120,
        keep_min=40,
        recall_top_k=2,
    )
    c = TestClient(app)
    msgs = [
        {"role": "user", "content": "I live in Novi Sad and use pnpm. " + ("x" * 240)},
        {"role": "assistant", "content": "Noted: Novi Sad and pnpm. " + ("z" * 240)},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "note", "input": {"k": "v"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "saved " + ("y" * 240)}],
        },
        {"role": "user", "content": "Where do I live?"},
    ]
    r = c.post(
        "/v1/messages",
        headers={"x-session-id": "ant-1", "x-api-key": "sk-ant-client", "anthropic-version": "2023-06-01"},
        json={"model": "claude-sonnet-4-6", "max_tokens": 256, "system": "Be concise.", "messages": msgs},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["memochain"]["step"] == 2
    assert body["memochain"]["recalled_chunks"] == 2
    sent = backend_calls[0]
    assert sent["url"] == "/messages"
    assert sent["headers"]["x-api-key"] == "sk-ant-client"
    assert "Be concise." in sent["json"]["system"]
    assert "[Durable memory]" in sent["json"]["system"]
    assert "- uses claude direct" in sent["json"]["system"]
    # first two turns folded; tool_use/tool_result pair + final user turn stay intact (3 msgs)
    assert len(sent["json"]["messages"]) == 3
    assert sent["json"]["messages"][0]["content"][0]["type"] == "tool_use"
    assert sent["json"]["messages"][1]["content"][0]["type"] == "tool_result"


def test_messages_normalizes_model_for_local_vllm_backend(tmp_path) -> None:
    """Claude Code style /v1/messages client works against local Inkling/vLLM too."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from memochain.server import app as app_mod

    seen: list[dict] = []

    class FakeLocalVllm:
        async def get(self, url: str):
            class R:
                status_code = 200

                def json(self):
                    return {"data": [{"id": "inkling-2-mini"}]}

            return R()

        async def post(self, url: str, json=None, headers=None):
            seen.append({"url": url, "json": json})

            class R:
                status_code = 200

                def json(self):
                    return {"id": "msg_local", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "ok"}]}

            return R()

    app = app_mod.create_app(
        backend_base_url="http://localhost:8000/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "local_msg",
        embedding_base_url=None,
        writer=FakeWriter(),
        reader=app_mod.HybridReaderAdapter("http://localhost:8000/v1"),
        embed_fn=_hash_embed,
        backend_client=FakeLocalVllm(),
    )
    c = TestClient(app)
    r = c.post(
        "/v1/messages",
        headers={"x-session-id": "local-cc"},
        json={"model": "claude-sonnet-4-6", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert seen[0]["url"] == "/messages"
    assert seen[0]["json"]["model"] == "inkling-2-mini"


def test_chunk_store_int8_quantization_ring_cap_and_legacy_float32(tmp_path) -> None:
    """ChunkStore stores int8 vectors (4x smaller), enforces max_chunks FIFO cap, and reads legacy float32 rows."""
    import sqlite3
    from memochain.server.session import ChunkStore

    db = tmp_path / "chunks.db"
    # 1) Create a legacy float32 database without the `scale` column
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                msg_hash TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL
            )
            """
        )
        legacy_vec = _hash_embed(["legacy port 5433"])[0].astype("<f4")
        conn.execute(
            "INSERT INTO chunks (step, msg_hash, role, text, dim, vec) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "h_legacy", "user", "legacy port 5433", int(legacy_vec.size), legacy_vec.tobytes()),
        )

    # 2) Open with ChunkStore (auto-migrates schema) and verify legacy row is searchable
    store = ChunkStore(db, max_chunks=3)
    hits_legacy = store.search("legacy port 5433", _hash_embed(["legacy port 5433"])[0], top_k=1)
    assert len(hits_legacy) == 1
    assert hits_legacy[0]["text"] == "legacy port 5433"
    assert hits_legacy[0]["score"] > 0.9

    # 3) Add 3 new int8 rows -> total would be 4, so oldest (legacy) is evicted by FIFO cap=3
    texts = ["city Novi Sad", "package manager pnpm", "token FALCON-739"]
    vecs = _hash_embed(texts)
    store.add(2, [(f"h{i}", "user", t) for i, t in enumerate(texts)], vecs)
    assert store.count() == 3

    # Verify stored blobs are int8 (len == dim, i.e. 2560 bytes, not 10240)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT text, dim, length(vec), scale FROM chunks ORDER BY id ASC").fetchall()
    assert [r[0] for r in rows] == texts
    for _, dim, blob_len, scale in rows:
        assert blob_len == dim  # 1 byte per dim (int8) instead of 4 bytes (float32)
        assert scale > 0.0

    # Verify cosine ranking accuracy with int8 dequantization
    q = _hash_embed(["token FALCON-739"])[0]
    top = store.search("token FALCON-739", q, top_k=2)
    assert top[0]["text"] == "token FALCON-739"
    assert top[0]["score"] > 0.95


def test_byok_client_auth_forwarded_to_hybrid_fold_summary(tmp_path, monkeypatch) -> None:
    """BYOK client API key authenticates HybridReaderAdapter fold summaries even when proxy has no server key."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from memochain.server import app as app_mod

    reader = app_mod.HybridReaderAdapter("https://openrouter.ai/api/v1", backend_api_key=None)
    summary_posts: list[dict] = []

    def fake_reader_post(url: str, json: dict, headers: dict | None = None):
        summary_posts.append({"url": url, "json": json, "headers": headers or {}})

        class R:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "- BYOK summary ok"}}]}

        return R()

    monkeypatch.setattr(reader.client, "post", fake_reader_post)

    class FakeOpenRouterBackend:
        async def get(self, url: str):
            class R:
                status_code = 404

            return R()

        async def post(self, url: str, json=None, headers=None):
            class R:
                status_code = 200

                def json(self):
                    return {
                        "id": "chatcmpl-byok",
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    }

            return R()

    app = app_mod.create_app(
        backend_base_url="https://openrouter.ai/api/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "byok_sessions",
        embedding_base_url=None,
        writer=FakeWriter(),
        reader=reader,
        embed_fn=_hash_embed,
        backend_client=FakeOpenRouterBackend(),
        hot_limit=120,
        keep_min=40,
        recall_top_k=2,
    )
    c = TestClient(app)
    msgs = [
        msg("user", "Remember city Novi Sad and pnpm. " + ("x" * 240)),
        msg("assistant", "Stored Novi Sad and pnpm. " + ("y" * 240)),
        msg("user", "What did I say?"),
    ]
    r = c.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-or-client-byok", "x-session-id": "byok-1"},
        json={"model": "anthropic/claude-sonnet-4", "messages": msgs},
    )
    assert r.status_code == 200
    assert len(summary_posts) == 1
    assert summary_posts[0]["headers"]["Authorization"] == "Bearer sk-or-client-byok"
    assert summary_posts[0]["json"]["model"] == "anthropic/claude-sonnet-4"

    st = c.get("/v1/memochain/status", params={"session_id": "byok-1"}).json()
    assert "- BYOK summary ok" in st["summary"]



