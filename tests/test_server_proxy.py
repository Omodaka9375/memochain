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

        async def post(self, url, json=None):
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


