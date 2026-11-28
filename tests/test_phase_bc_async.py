"""Tests for Phase B (tool compaction), Phase C (FTS5/RRF/dedupe/hits/facts/guards),
and Async Background Folding."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from memochain.server.app import HybridReaderAdapter, IntWriterAdapter, _hash_embed, create_app
from memochain.server.session import ChunkStore, SessionMemory


def _stub_writer() -> IntWriterAdapter:
    return IntWriterAdapter(None)


# ---------------------------------------------------------------------------
# Phase C: ChunkStore FTS5 + RRF + semantic dedupe + hit-based eviction + facts
# ---------------------------------------------------------------------------


def test_chunk_store_fts5_rrf_finds_exact_code_identifiers_with_zero_dense_signal(tmp_path) -> None:
    """FTS5 BM25 + RRF retrieves rare exact identifiers even when embeddings are orthogonal."""
    store = ChunkStore(tmp_path / "chunks.db", max_chunks=100)
    assert store.fts_enabled is True

    texts = [
        "Configured PostgreSQL replica at db-staging.internal:5433 for failover.",
        "The release ticket is FALCON-739 and deploys via src/memochain/server/session.py.",
        "General brainstorming about colors, weather, and lunch plans today.",
    ]
    # Give every chunk the exact same flat embedding so dense cosine provides zero discrimination
    flat = np.ones((len(texts), 64), dtype=np.float32)
    store.add(1, [(f"h{i}", "user", t) for i, t in enumerate(texts)], flat)

    q_flat = np.ones(64, dtype=np.float32)
    res_ticket = store.search("Where is FALCON-739 implemented in session.py?", q_flat, top_k=1)
    assert len(res_ticket) == 1
    assert "FALCON-739" in res_ticket[0]["text"]

    res_host = store.search("What port is db-staging.internal:5433?", q_flat, top_k=1)
    assert len(res_host) == 1
    assert "db-staging.internal:5433" in res_host[0]["text"]


def test_chunk_store_semantic_dedupe_merges_and_bumps_hits(tmp_path) -> None:
    """Duplicate or near-identical chunks merge into one row and increment its hit counter."""
    store = ChunkStore(tmp_path / "chunks.db", max_chunks=100, dedupe_threshold=0.96)
    text = "Deploy target is db-staging.internal:5433 using pnpm."
    vec = _hash_embed([text])

    n1 = store.add(1, [("h1", "user", text)], vec)
    n2 = store.add(2, [("h2", "user", "Deploy   target is db-staging.internal:5433 using pnpm.")], vec)
    assert n1 == 1
    assert n2 == 0
    assert store.count() == 1

    # Merged row now has hits >= 1 before even being searched
    hits = store.search("db-staging.internal:5433", vec[0], top_k=1, record_hits=False)
    assert hits[0]["hits"] >= 1


def test_chunk_store_hit_based_priority_eviction_keeps_useful_old_chunks(tmp_path) -> None:
    """Old chunks recalled >=2 times survive max_chunks eviction while untouched churn is evicted."""
    store = ChunkStore(tmp_path / "chunks.db", max_chunks=3)
    anchor_text = "Critical architecture rule: always use pnpm and port 5433."
    anchor_vec = _hash_embed([anchor_text])
    store.add(1, [("h_anchor", "user", anchor_text)], anchor_vec)

    # Recall the anchor twice so hits == 2
    store.search("pnpm port 5433", anchor_vec[0], top_k=1, current_step=2)
    store.search("pnpm port 5433", anchor_vec[0], top_k=1, current_step=3)

    # Flood with 5 brand-new unretrieved chunks (exceeding max_chunks=3)
    churn = [f"Unrelated filler turn number {i} about topic_{i}" for i in range(5)]
    store.add(4, [(f"hc{i}", "user", t) for i, t in enumerate(churn)], _hash_embed(churn))
    assert store.count() == 3

    # The repeatedly recalled anchor must still be present
    res = store.search("pnpm port 5433", anchor_vec[0], top_k=3, record_hits=False)
    assert any("5433" in r["text"] for r in res)


def test_chunk_store_append_only_facts_ledger_supersedes_by_key(tmp_path) -> None:
    """Facts accumulate across folds and supersede old values cleanly on the same key."""
    store = ChunkStore(tmp_path / "chunks.db")
    store.upsert_facts(
        [
            ("city", "Zurich", "summary"),
            ("db_port", "5432", "summary"),
            ("package_manager", "pnpm", "summary"),
        ],
        step=1,
    )
    store.upsert_facts([("db_port", "5433", "summary"), ("ticket", "FALCON-739", "summary")], step=2)

    facts = {f["key"]: f["value"] for f in store.list_facts()}
    assert facts["city"] == "Zurich"
    assert facts["package_manager"] == "pnpm"
    assert facts["db_port"] == "5433"
    assert facts["ticket"] == "FALCON-739"
    assert store.facts_count() == 4


# ---------------------------------------------------------------------------
# Phase C: Shrinkage guard, fallback telemetry, and append-only ledger injection
# ---------------------------------------------------------------------------


def test_shrinkage_guard_and_append_only_facts_prevent_amnesia(tmp_path) -> None:
    """If a fold LLM drops >55% of an established summary (or falls back), MemoChain keeps the
    previous summary, increments telemetry counters, and preserves facts in static_block."""

    class _DegradingReader:
        def __init__(self) -> None:
            self.calls = 0
            self.last_call_fallback = False

        def read_fold(self, state, q, prev_summary, entries, model_hint=None, auth_headers=None) -> str:
            self.calls += 1
            if self.calls == 1:
                self.last_call_fallback = False
                return (
                    "- city: Zurich\n"
                    "- package_manager: pnpm\n"
                    "- staging_db: db-staging.internal:5433\n"
                    "- ticket: FALCON-739\n"
                    "- architecture: rust core with python proxy and sqlite chunks"
                )
            # Second fold simulates a bad/truncated LLM output (catastrophic shrinkage + fallback)
            self.last_call_fallback = True
            return "- short"

    reader = _DegradingReader()
    mem = SessionMemory(
        session_id="guard-test",
        store_dir=tmp_path / "sess",
        writer=_stub_writer(),
        embed_fn=_hash_embed,
        reader=reader,
        hot_limit=80,
        keep_min=30,
        recall_top_k=3,
    )

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "City is Zurich, ticket FALCON-739, host db-staging.internal:5433 " + ("a" * 240)},
        {"role": "assistant", "content": "Noted all settings " + ("b" * 240)},
        {"role": "user", "content": "Next question"},
    ]
    _, stats1 = mem.process(msgs)
    assert stats1["fold"] is not None
    assert "Zurich" in mem.summary
    assert mem.fold_shrink_guard_count == 0
    assert mem.fold_fallback_count == 0

    # Push enough new turns to trigger a second fold where the reader shrinks catastrophically
    msgs.extend(
        [
            {"role": "assistant", "content": "Working on ALPHA-99 rollout " + ("c" * 260)},
            {"role": "user", "content": "Also check prod-cache.internal:6379 " + ("d" * 260)},
            {"role": "assistant", "content": "Done checking " + ("e" * 260)},
            {"role": "user", "content": "Final question"},
        ]
    )
    _, stats2 = mem.process(msgs)
    assert stats2["fold"] is not None
    # Shrinkage guard preserved the full established summary instead of replacing with "- short"
    assert "Zurich" in mem.summary
    assert "FALCON-739" in mem.summary
    assert mem.fold_shrink_guard_count == 1
    assert mem.fold_fallback_count == 1
    assert mem.last_fold_fallback is True

    # Deterministic fact extraction captured ALPHA-99 and prod-cache.internal:6379 into the ledger
    # even though the LLM fold summary degraded!
    static_block = stats2["static_memory_block"]
    assert "ALPHA-99" in static_block
    assert "prod-cache.internal:6379" in static_block


# ---------------------------------------------------------------------------
# Async Background Folding: zero-latency fold turn, next turn uses folded state
# ---------------------------------------------------------------------------


def test_async_background_fold_does_not_block_turn_and_applies_next_turn(tmp_path) -> None:
    """With async_fold=True, the fold turn returns immediately without blocking on a slow LLM
    reader, and the next turn picks up the folded summary, capsule, and trimmed hot window."""

    class _SlowReader:
        def __init__(self) -> None:
            self.calls = 0

        def read_fold(self, state, q, prev_summary, entries, model_hint=None, auth_headers=None) -> str:
            self.calls += 1
            time.sleep(0.25)
            return "- city: Zurich\n- ticket: FALCON-739"

    reader = _SlowReader()
    mem = SessionMemory(
        session_id="async-test",
        store_dir=tmp_path / "sess",
        writer=_stub_writer(),
        embed_fn=_hash_embed,
        reader=reader,
        hot_limit=120,
        keep_min=40,
        recall_top_k=2,
        async_fold=True,
        sync_fold_ceiling_ratio=2.5,
    )

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "System instructions."},
        {"role": "user", "content": "City Zurich ticket FALCON-739 " + ("x" * 260)},
        {"role": "assistant", "content": "Got it " + ("y" * 260)},
        {"role": "user", "content": "Continue please"},
    ]

    t0 = time.perf_counter()
    _, stats1 = mem.process(msgs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Should return immediately (< 150ms despite 250ms reader sleep) with status="scheduled"
    assert stats1["fold"] is not None
    assert stats1["fold"]["status"] == "scheduled"
    assert elapsed_ms < 200.0

    # Wait for background fold to finish cleanly
    assert mem.wait_for_pending_fold(timeout=5.0) is True
    assert reader.calls == 1
    assert mem.verify()["ok"] is True

    # Next turn automatically sees the already-folded prefix, trimmed hot window, and static memory
    msgs.append({"role": "assistant", "content": "Sure."})
    msgs.append({"role": "user", "content": "What was the ticket?"})
    rewritten2, stats2 = mem.process(msgs)
    assert stats2["already_folded"] >= 1
    assert "FALCON-739" in stats2["static_memory_block"]
    assert "FALCON-739" in rewritten2[0]["content"]


# ---------------------------------------------------------------------------
# Phase B: Age-gated tool-output archiving & file-read superseding (cache-safe)
# ---------------------------------------------------------------------------


def test_phase_b_file_read_superseding_and_old_tool_archiving_cache_safe(tmp_path) -> None:
    """At overflow, superseded file reads and old huge tool outputs compact into deterministic
    stubs, archive full text to ChunkStore, and stay byte-identical on subsequent turns."""

    class _NoopReader:
        def read_fold(self, state, q, prev_summary, entries, model_hint=None, auth_headers=None) -> str:
            return "- tool compaction test"

    mem = SessionMemory(
        session_id="phase-b-test",
        store_dir=tmp_path / "sess",
        writer=_stub_writer(),
        embed_fn=_hash_embed,
        reader=_NoopReader(),
        hot_limit=600,
        keep_min=300,
        recall_top_k=2,
        compact_tool_outputs=True,
        tool_archive_min_chars=800,
        tool_keep_recent_groups=2,
    )

    old_file_body = "OLD_VERSION_CONTENT_MARKER " + ("line_old\n" * 120)
    new_file_body = "NEW_VERSION_CONTENT_MARKER " + ("line_new\n" * 30)
    huge_bash_out = "BASH_UNIQUE_ERROR_CODE_9912 " + ("trace_line\n" * 150)

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "Coding agent."},
        {"role": "user", "content": "Read config.py and run tests."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "src/config.py"}'},
                },
                {
                    "id": "call_bash_1",
                    "type": "function",
                    "function": {"name": "bash_run", "arguments": '{"command": "pytest"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_read_1", "name": "read_file", "content": old_file_body},
        {"role": "tool", "tool_call_id": "call_bash_1", "name": "bash_run", "content": huge_bash_out},
        {"role": "assistant", "content": "I edited config.py."},
        {"role": "user", "content": "Read config.py again to verify."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "src/config.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read_2", "name": "read_file", "content": new_file_body},
        {"role": "assistant", "content": "Verified."},
        {"role": "user", "content": "What was the error code earlier?"},
    ]

    rewritten1, stats1 = mem.process(msgs)
    assert stats1.get("tool_compactions", {}).get("count", 0) >= 2
    assert stats1["tool_compactions"]["tokens_saved"] > 400

    # Find the compacted tool messages
    tool_msgs = [m for m in rewritten1 if m.get("role") == "tool"]
    read1 = next(m for m in tool_msgs if m.get("tool_call_id") == "call_read_1")
    bash1 = next(m for m in tool_msgs if m.get("tool_call_id") == "call_bash_1")
    read2 = next(m for m in tool_msgs if m.get("tool_call_id") == "call_read_2")

    # Older read_file was superseded; older bash_run was archived; newer read_file stayed verbatim
    assert "[superseded read_file: src/config.py" in read1["content"]
    assert "[archived " in bash1["content"]
    assert "NEW_VERSION_CONTENT_MARKER" in read2["content"]

    # Archived bash output is searchable via ChunkStore FTS5!
    assert mem.chunk_store is not None
    recalled = mem.chunk_store.search(
        "BASH_UNIQUE_ERROR_CODE_9912",
        _hash_embed(["BASH_UNIQUE_ERROR_CODE_9912"])[0],
        top_k=2,
        record_hits=False,
    )
    assert any("BASH_UNIQUE_ERROR_CODE_9912" in r["text"] for r in recalled)

    # Cache-safety invariant: on the next turn, the earlier compacted prefix is byte-identical!
    msgs2 = list(msgs) + [
        {"role": "assistant", "content": "It was 9912."},
        {"role": "user", "content": "Great, proceed."},
    ]
    rewritten2, _ = mem.process(msgs2)
    # Every non-tail message up to the previous assistant turn must match byte-for-byte
    prefix_len = len(rewritten1) - 1  # exclude old last user msg (which carried dynamic recall)
    assert rewritten2[:prefix_len] == rewritten1[:prefix_len]


def test_status_endpoint_exposes_phase_c_telemetry(tmp_path) -> None:
    """/v1/memochain/status reports fold_fallback_count, shrinkage guard, fts_enabled, and facts_count."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class _FailReader(HybridReaderAdapter):
        def _call_chat_completions(self, model, prompt, auth_headers=None) -> str:
            raise RuntimeError("simulated timeout")

    class _FakeBackend:
        async def get(self, url: str):
            class R:
                status_code = 200

                def json(self):
                    return {"data": [{"id": "demo-model"}]}

            return R()

        async def post(self, url: str, json: dict, headers: dict | None = None):
            class R:
                status_code = 200
                text = ""

                def json(self):
                    return {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "model": json["model"],
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 200, "completion_tokens": 5, "total_tokens": 205},
                    }

            return R()

        async def aclose(self) -> None:
            pass

    app = create_app(
        backend_base_url="http://local.test/v1",
        backend_api_key=None,
        checkpoint="",
        store_root=tmp_path / "sessions",
        embedding_base_url=None,
        hot_limit=80,
        keep_min=30,
        writer=_stub_writer(),
        reader=_FailReader("http://local.test/v1", reader_model="demo-model"),
        embed_fn=_hash_embed,
        backend_client=_FakeBackend(),
        recall_top_k=3,
        cache_passthrough=False,
    )
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        headers={"x-session-id": "telemetry-sess"},
        json={
            "model": "demo-model",
            "messages": [
                {"role": "system", "content": "Helper."},
                {"role": "user", "content": "Remember ticket FALCON-739 and host db.internal:5433 " + ("x" * 240)},
                {"role": "assistant", "content": "Got it " + ("y" * 240)},
                {"role": "user", "content": "Now summarize."},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["memochain"]["fold_fallback_count"] == 1

    st = client.get("/v1/memochain/status", params={"session_id": "telemetry-sess"}).json()
    assert st["fold_fallback_count"] == 1
    assert st["last_fold_fallback"] is True
    assert st["fts_enabled"] is True
    assert st["facts_count"] >= 2
    assert st["chunks_count"] >= 1
