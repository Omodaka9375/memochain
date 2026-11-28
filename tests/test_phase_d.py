"""Tests for Phase D: cloud/local auto-profiles, Gemini support, cheap fold-model routing,
and per-session economics telemetry."""

from __future__ import annotations

import numpy as np
import pytest

from memochain.server.app import (
    HybridReaderAdapter,
    IntWriterAdapter,
    _hash_embed,
    create_app,
    resolve_profile_config,
    suggest_cheap_fold_model,
)


def test_resolve_profile_config_auto_cloud_and_local() -> None:
    """Auto-profile selects tuned cloud vs local defaults from backend URL."""
    c_ant = resolve_profile_config("https://api.anthropic.com/v1", "auto")
    assert c_ant["profile"] == "cloud"
    assert c_ant["backend_kind"] == "anthropic"
    assert c_ant["hot_limit"] == 96000
    assert c_ant["keep_min"] == 32000
    assert c_ant["cheap_fold_routing"] is True
    assert c_ant["cache_hints"] is True

    c_gem = resolve_profile_config("https://generativelanguage.googleapis.com/v1beta/openai", "auto")
    assert c_gem["profile"] == "cloud"
    assert c_gem["backend_kind"] == "gemini"

    c_or = resolve_profile_config("https://openrouter.ai/api/v1", "auto")
    assert c_or["profile"] == "cloud"
    assert c_or["backend_kind"] == "openrouter"

    c_loc = resolve_profile_config("http://localhost:8000/v1", "auto")
    assert c_loc["profile"] == "local"
    assert c_loc["backend_kind"] == "local"
    assert c_loc["hot_limit"] == 28000
    assert c_loc["keep_min"] == 12000
    assert c_loc["cache_passthrough"] is False
    assert c_loc["cheap_fold_routing"] is False

    # Explicit override works in both directions
    forced_local = resolve_profile_config("https://openrouter.ai/api/v1", "local")
    assert forced_local["profile"] == "local"
    assert forced_local["hot_limit"] == 28000


def test_suggest_cheap_fold_model_maps_frontier_to_fast_tier() -> None:
    assert suggest_cheap_fold_model("https://api.anthropic.com/v1", "claude-opus-4-6") == "claude-haiku-4-5"
    assert suggest_cheap_fold_model("https://api.anthropic.com/v1", "claude-sonnet-4-6") == "claude-haiku-4-5"
    assert suggest_cheap_fold_model("https://api.anthropic.com/v1", "claude-haiku-4-5") is None

    assert suggest_cheap_fold_model("https://api.openai.com/v1", "gpt-5") == "gpt-4o-mini"
    assert suggest_cheap_fold_model("https://api.openai.com/v1", "gpt-4o-mini") is None

    assert suggest_cheap_fold_model("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro") == "gemini-2.5-flash"
    assert suggest_cheap_fold_model("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash") is None

    assert suggest_cheap_fold_model("https://openrouter.ai/api/v1", "anthropic/claude-opus-4") == "anthropic/claude-haiku-4.5"
    assert suggest_cheap_fold_model("https://openrouter.ai/api/v1", "google/gemini-2.5-pro") == "google/gemini-2.5-flash"
    assert suggest_cheap_fold_model("http://localhost:8000/v1", "Qwen/Qwen3-32B") is None


def test_hybrid_reader_cheap_fold_routing_and_fallback_to_active_model(monkeypatch) -> None:
    """Cheap fold routing uses the fast model first and transparently falls back to
    model_hint if the cheap model call fails."""
    adapter = HybridReaderAdapter(
        "https://openrouter.ai/api/v1",
        cheap_fold_routing=True,
    )
    tried: list[str] = []

    def fake_post(url: str, json: dict, headers=None):
        tried.append(json["model"])
        if json["model"] == "anthropic/claude-haiku-4.5":
            raise RuntimeError("model unavailable")

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "- city: Zurich"}}]}

        return R()

    monkeypatch.setattr(adapter.client, "post", fake_post)
    out = adapter.read_fold(
        state=None,
        query_embed=np.zeros(2560, dtype=np.float32),
        prev_summary="",
        folded_entries=[("h1", "user", "My city is Zurich.")],
        model_hint="anthropic/claude-sonnet-4",
    )
    assert out == "- city: Zurich"
    assert tried == ["anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4"]
    assert adapter.last_used_model == "anthropic/claude-sonnet-4"
    assert adapter.last_call_fallback is False


def test_healthz_profile_and_session_economics_telemetry(tmp_path) -> None:
    """create_app(profile='auto') exposes resolved profile in /healthz and tracks
    cumulative token savings + cache hit % + reported USD cost in /v1/memochain/status."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class FakeCloudBackend:
        def __init__(self) -> None:
            self.seen: list[dict] = []

        async def get(self, url: str):
            class R:
                status_code = 404

            return R()

        async def post(self, url: str, json: dict, headers=None):
            self.seen.append(json)

            class R:
                status_code = 200
                text = ""

                def json(self_inner):
                    return {
                        "id": "chatcmpl-econ",
                        "object": "chat.completion",
                        "model": json["model"],
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 120,
                            "prompt_tokens_details": {"cached_tokens": 90},
                            "completion_tokens": 5,
                            "cost": 0.0012,
                        },
                    }

            return R()

        async def aclose(self) -> None:
            pass

    class StubReader:
        def read_fold(self, state, q, prev, entries, model_hint=None, auth_headers=None) -> str:
            return "- city: Zurich"

    backend = FakeCloudBackend()
    app = create_app(
        backend_base_url="https://openrouter.ai/api/v1",
        backend_api_key=None,
        checkpoint="",
        store_root=tmp_path / "sessions",
        embedding_base_url=None,
        writer=IntWriterAdapter(None),
        reader=StubReader(),
        embed_fn=_hash_embed,
        backend_client=backend,
        profile="auto",
        hot_limit=80,
        keep_min=30,
        async_fold=False,
        cache_passthrough=False,
    )
    client = TestClient(app)

    hz = client.get("/healthz").json()
    assert hz["profile"] == "cloud"
    assert hz["backend_kind"] == "openrouter"
    assert hz["cheap_fold_routing"] is True
    assert hz["hot_limit"] == 80  # explicit override beats profile default

    big_msgs = [
        {"role": "system", "content": "Helper."},
        {"role": "user", "content": "Remember Zurich " + ("x" * 600)},
        {"role": "assistant", "content": "Saved " + ("y" * 600)},
        {"role": "user", "content": "Where do I live?"},
    ]
    r = client.post(
        "/v1/chat/completions",
        headers={"x-session-id": "econ-1"},
        json={"model": "google/gemini-2.5-pro", "messages": big_msgs},
    )
    assert r.status_code == 200
    econ = r.json()["memochain"]["economics"]
    assert econ["turns"] == 1
    assert econ["raw_input_tokens"] > econ["proxied_input_tokens"]
    assert econ["tokens_saved"] > 0
    assert econ["token_savings_pct"] > 0.0
    assert econ["cache_hit_pct"] == pytest.approx(75.0)
    assert econ["reported_cost_usd"] == pytest.approx(0.0012)

    st = client.get("/v1/memochain/status", params={"session_id": "econ-1"}).json()
    assert st["profile"] == "cloud"
    assert st["economics"]["turns"] == 1
    assert st["economics"]["reported_cost_usd"] == pytest.approx(0.0012)
