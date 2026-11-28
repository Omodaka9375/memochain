from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memochain.server.app import (
    _cache_hit_ratio,
    _hash_embed,
    create_app,
)
from memochain.server.session import SessionMemory
from test_server_proxy import FakeWriter, msg


def test_cache_hit_ratio_across_providers() -> None:
    """_cache_hit_ratio parses OpenAI, Anthropic, DeepSeek, OpenRouter, Gemini, and SSE events."""
    # OpenAI / OpenRouter structured prompt_tokens_details
    assert _cache_hit_ratio(
        {"usage": {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 800}}}
    ) == pytest.approx(0.8)
    # DeepSeek flat key
    assert _cache_hit_ratio(
        {"usage": {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 600}}
    ) == pytest.approx(0.6)
    # Anthropic direct: total prompt = input_tokens + cache_read + cache_creation
    assert _cache_hit_ratio(
        {
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 100,
            }
        }
    ) == pytest.approx(0.8)
    # Anthropic SSE message_start nests usage under message.usage
    assert _cache_hit_ratio(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {
                    "input_tokens": 200,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    ) == pytest.approx(0.8)
    # Gemini usageMetadata
    assert _cache_hit_ratio(
        {"usageMetadata": {"promptTokenCount": 2000, "cachedContentTokenCount": 1500}}
    ) == pytest.approx(0.75)
    # Unreported usage returns None
    assert _cache_hit_ratio({"usage": {"prompt_tokens": 1000}}) is None
    assert _cache_hit_ratio({}) is None


def test_split_injection_preserves_exact_prefix_between_folds(tmp_path) -> None:
    """Across turns with different queries between folds, messages[0..-1] stay
    100% byte-identical so cloud + local KV prefix caches stay hot."""

    class FactReader:
        def read(self, state, q, prev):
            return prev

        def read_fold(self, state, q, prev, entries):
            return "- city: Lisbon\n- port: 5432\n- code: ZEBRA-42"

    s = SessionMemory(
        "prefix-inv",
        tmp_path / "prefix-inv",
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        reader=FactReader(),
        hot_limit=150,
        keep_min=40,
        recall_top_k=2,
    )
    base = [
        msg("system", "You are a helpful coding assistant."),
        msg("user", "My city is Lisbon and code is ZEBRA-42. " + ("x" * 300)),
        msg("assistant", "Stored Lisbon and ZEBRA-42. " + ("y" * 300)),
        msg("user", "Database runs on port 5432 at db.internal. " + ("w" * 300)),
        msg("assistant", "Stored port 5432 and db.internal. " + ("z" * 300)),
    ]
    turn1 = base + [msg("user", "What is my secret code ZEBRA-42?")]
    out1, stats1 = s.process(turn1)
    assert stats1["fold"] is not None

    # Turn 2: different query -> different recalled chunks, NO new fold
    turn2 = turn1 + [
        msg("assistant", "ZEBRA-42"),
        msg("user", "What database port is db.internal using?"),
    ]
    out2, stats2 = s.process(turn2)
    assert stats2["fold"] is None

    # Turn 3: another different query -> different recalled chunks, NO new fold
    turn3 = turn2 + [
        msg("assistant", "5432"),
        msg("user", "Which city do I live in?"),
    ]
    out3, stats3 = s.process(turn3)
    assert stats3["fold"] is None

    # System prompt + static fact sheet MUST be byte-identical across all 3 turns
    assert out1[0] == out2[0] == out3[0]
    assert "[Durable memory]" in out1[0]["content"]
    assert "[Recalled memory]" not in out1[0]["content"]

    # Every earlier hot message (prefix before the newest user turn) stays byte-identical
    assert out2[:-1][: len(out1) - 1] == out1[:-1]
    assert out3[:-1][: len(out2) - 1] == out2[:-1]

    # Dynamic recall landed on each turn's tail user message and tracks the query
    assert "ZEBRA-42" in out1[-1]["content"]
    assert "5432" in out2[-1]["content"]
    assert "Lisbon" in out3[-1]["content"]


def test_cache_passthrough_skips_folding_when_backend_reports_high_cache_hit_rate(tmp_path) -> None:
    """When the backend reports a persistently high cache-hit ratio (streak=2),
    the next request enters passthrough; a low-hit response exits passthrough."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from memochain.server import app as app_mod

    class FakeCachingBackend:
        def __init__(self):
            self.calls = 0
            self.cached_sequence = [700, 700, 700, 100, 700]

        async def post(self, url, json=None, headers=None):
            idx = min(self.calls, len(self.cached_sequence) - 1)
            cached = self.cached_sequence[idx]
            self.calls += 1

            class R:
                status_code = 200

                def json(self_inner):
                    return {
                        "id": "chatcmpl-cache",
                        "object": "chat.completion",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 1000,
                            "prompt_tokens_details": {"cached_tokens": cached},
                            "completion_tokens": 5,
                        },
                    }

            return R()

        async def get(self, url):
            class R:
                status_code = 404

            return R()

    class NotFoldReader:
        def read(self, state, q, prev):
            return prev or "fallback"

        def read_fold(self, *a, **k):
            return "- folded-fact"

    backend = FakeCachingBackend()
    app = app_mod.create_app(
        backend_base_url="http://backend.test/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "sessions",
        embedding_base_url=None,
        writer=FakeWriter(),
        reader=NotFoldReader(),
        embed_fn=_hash_embed,
        backend_client=backend,
        hot_limit=50,
        keep_min=10,
        cache_passthrough=True,
        cache_passthrough_threshold=0.5,
        cache_passthrough_streak=2,
    )

    c = TestClient(app)
    big = [msg("system", "sys")] + [msg("user", "u" * 2000), msg("assistant", "a" * 2000)] * 3

    # Req 1: streak=0 before call -> not passthrough; resp reports 0.7 -> streak becomes 1
    r1 = c.post("/v1/chat/completions", json={"model": "test", "messages": big}, headers={"x-session-id": "cache-bypass"})
    assert r1.status_code == 200
    assert r1.json()["memochain"].get("passthrough") is not True
    assert r1.json()["memochain"]["cache_hit_ratio"] == pytest.approx(0.7)

    # Req 2: streak=1 before call -> still not passthrough; resp reports 0.7 -> streak becomes 2
    r2 = c.post("/v1/chat/completions", json={"model": "test", "messages": big}, headers={"x-session-id": "cache-bypass"})
    assert r2.status_code == 200
    assert r2.json()["memochain"].get("passthrough") is not True

    # Req 3: streak=2 before call -> passthrough active! Resp reports 0.7 -> streak 3
    r3 = c.post("/v1/chat/completions", json={"model": "test", "messages": big}, headers={"x-session-id": "cache-bypass"})
    assert r3.status_code == 200
    assert r3.json()["memochain"].get("passthrough") is True

    # Req 4: still passthrough, but backend reports low cache hit (0.1) -> resets streak to 0
    r4 = c.post("/v1/chat/completions", json={"model": "test", "messages": big}, headers={"x-session-id": "cache-bypass"})
    assert r4.status_code == 200
    assert r4.json()["memochain"].get("passthrough") is True

    # Req 5: streak was reset to 0 -> exits passthrough immediately
    r5 = c.post("/v1/chat/completions", json={"model": "test", "messages": big}, headers={"x-session-id": "cache-bypass"})
    assert r5.status_code == 200
    assert r5.json()["memochain"].get("passthrough") is not True


def test_streaming_sse_updates_cache_hit_telemetry_openai_and_anthropic(tmp_path) -> None:
    """Streaming SSE responses update cache-hit streak on both /v1/chat/completions
    and /v1/messages, and OpenAI/OpenRouter stream requests get stream_options.include_usage."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from memochain.server import app as app_mod

    openai_seen: list[dict] = []

    class OpenAIStreamResp:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aiter_lines(self):
            yield 'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}'
            yield 'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":1000,"prompt_tokens_details":{"cached_tokens":900},"completion_tokens":2}}'
            yield "data: [DONE]"

    class FakeOpenAIClient:
        async def get(self, url):
            class R:
                status_code = 404

            return R()

        def stream(self, method, url, json=None, headers=None):
            openai_seen.append(json)
            return OpenAIStreamResp()

    app_oai = create_app(
        backend_base_url="https://api.openai.com/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "oai_sse",
        embedding_base_url=None,
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        backend_client=FakeOpenAIClient(),
        cache_passthrough=True,
        cache_passthrough_threshold=0.5,
        cache_passthrough_streak=2,
    )
    c_oai = TestClient(app_oai)
    for _ in range(3):
        r = c_oai.post(
            "/v1/chat/completions",
            headers={"x-session-id": "oai-stream-1"},
            json={"model": "gpt-4o", "stream": True, "messages": [msg("user", "hello")]},
        )
        assert r.status_code == 200

    # Verify OpenAI cache hints were attached
    assert openai_seen[0]["prompt_cache_key"] == "memochain:oai-stream-1"
    assert openai_seen[0]["stream_options"]["include_usage"] is True
    # Third streaming turn should have entered passthrough after two 0.9 hits
    chunks = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    assert chunks[-1]["memochain"].get("passthrough") is True
    assert chunks[-1]["memochain"]["cache_hit_ratio"] == pytest.approx(0.9)

    # Now verify Anthropic SSE `message_start` usage updates passthrough streak
    ant_seen: list[dict] = []

    class AntStreamResp:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aiter_lines(self):
            yield "event: message_start"
            yield 'data: {"type":"message_start","message":{"id":"m1","usage":{"input_tokens":100,"cache_read_input_tokens":900,"cache_creation_input_tokens":0}}}'
            yield ""
            yield "event: message_stop"
            yield 'data: {"type":"message_stop"}'
            yield ""

    class FakeAntClient:
        async def get(self, url):
            class R:
                status_code = 404

            return R()

        def stream(self, method, url, json=None, headers=None):
            ant_seen.append(json)
            return AntStreamResp()

    app_ant = create_app(
        backend_base_url="https://api.anthropic.com/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "ant_sse_cache",
        embedding_base_url=None,
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        backend_client=FakeAntClient(),
        hot_limit=50,
        keep_min=10,
        cache_passthrough=True,
        cache_passthrough_threshold=0.5,
        cache_passthrough_streak=2,
    )
    c_ant = TestClient(app_ant)
    big_ant = [
        {"role": "user", "content": "first " + ("a" * 500)},
        {"role": "assistant", "content": "reply " + ("b" * 500)},
        {"role": "user", "content": "latest"},
    ]
    for _ in range(3):
        ra = c_ant.post(
            "/v1/messages",
            headers={"x-session-id": "ant-stream-1", "x-api-key": "sk-test"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 64, "stream": True, "messages": big_ant},
        )
        assert ra.status_code == 200
    # Turn 1 folded (sent fewer messages); turn 3 entered passthrough (sent all 3 unmodified messages)
    assert len(ant_seen[0]["messages"]) < len(big_ant)
    assert len(ant_seen[2]["messages"]) == len(big_ant)


def test_openrouter_anthropic_route_receives_cache_control_breakpoints(tmp_path) -> None:
    """OpenRouter requests targeting anthropic/* models get Anthropic cache_control
    breakpoints on the static system prompt and pre-turn history boundary."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    seen: list[dict] = []

    class FakeORBackend:
        async def get(self, url):
            class R:
                status_code = 404

            return R()

        async def post(self, url, json=None, headers=None):
            seen.append(json)

            class R:
                status_code = 200

                def json(self_inner):
                    return {
                        "id": "chatcmpl-or",
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    }

            return R()

    app = create_app(
        backend_base_url="https://openrouter.ai/api/v1",
        backend_api_key=None,
        checkpoint="unused",
        store_root=tmp_path / "or_cache",
        embedding_base_url=None,
        writer=FakeWriter(),
        embed_fn=_hash_embed,
        backend_client=FakeORBackend(),
    )
    c = TestClient(app)
    r = c.post(
        "/v1/chat/completions",
        headers={"x-session-id": "or-ant-1"},
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [
                msg("system", "You are helpful."),
                msg("user", "Turn 1"),
                msg("assistant", "Reply 1"),
                msg("user", "Turn 2"),
            ],
        },
    )
    assert r.status_code == 200
    sent_msgs = seen[0]["messages"]
    assert sent_msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent_msgs[-2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Latest user turn stays un-annotated (dynamic tail)
    assert sent_msgs[-1]["content"] == "Turn 2"
