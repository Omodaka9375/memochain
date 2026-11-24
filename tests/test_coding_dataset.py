"""Tests for tools/build_coding_dataset.py and causal prefix slicing in E2EDatasetBatches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from build_coding_dataset import (  # noqa: E402
    extract_session,
    scrub_text,
    split_sessions,
    write_split_store,
)


def test_scrub_text_redacts_secrets_and_env() -> None:
    raw = (
        "<env>\nos: windows\n</env>\n"
        "[start tool_abc123]key=sk-1234567890abcdefghijklmnop "
        "token: hf_abcdefghijklmnop1234[end tool_abc123]"
    )
    cleaned = scrub_text(raw)
    assert "<env>" not in cleaned
    assert "sk-1234" not in cleaned
    assert "hf_abcd" not in cleaned
    assert "[REDACTED]" in cleaned
    assert "[start tool_" not in cleaned


def test_extract_session_causal_prefix_no_leakage() -> None:
    msgs = [
        {"role": "user", "parts": [{"type": "text", "text": "Build a FastAPI service in app.py"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "tool-write_file", "input": {"path": "app.py"}, "output": {"ok": True}},
                {"type": "text", "text": "Created app.py with FastAPI healthz route."},
            ],
        },
        {"role": "user", "parts": [{"type": "text", "text": "Add a postgres pool to app.py"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "tool-edit", "input": {"path": "app.py"}, "output": {"ok": True}},
                {"type": "text", "text": "Added asyncpg connection pool in app.py."},
            ],
        },
        {"role": "user", "parts": [{"type": "text", "text": "Which database driver did we pick?"}]},
        {
            "role": "assistant",
            "parts": [{"type": "text", "text": "We picked asyncpg for the postgres pool in app.py."}],
        },
    ]
    sess = extract_session("s1", msgs, min_history_docs=2)
    assert sess is not None
    # 6 messages -> 6 docs; 2 later turns (turn 2 sees 2 docs, turn 3 sees 4 docs)
    assert len(sess.docs) == 6
    assert len(sess.qa_rows) == 2
    assert sess.qa_rows[0][0] == 2
    assert "postgres" in sess.qa_rows[0][1]
    assert sess.qa_rows[1][0] == 4
    assert "database driver" in sess.qa_rows[1][1]
    assert "asyncpg" in sess.qa_rows[1][2]


def test_write_split_store_and_causal_e2e_loader(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    pytest.importorskip("safetensors")

    from memochain.train_e2e import E2EDatasetBatches

    msgs = [
        {"role": "user", "parts": [{"type": "text", "text": "Initialize project alpha with rust"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "Initialized cargo workspace alpha."}]},
        {"role": "user", "parts": [{"type": "text", "text": "Add sqlite storage layer to alpha"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "Added rusqlite module to alpha."}]},
        {"role": "user", "parts": [{"type": "text", "text": "What storage crate is alpha using?"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "Alpha uses rusqlite for local storage."}]},
    ]
    sess = extract_session("s0", msgs, min_history_docs=2)
    assert sess is not None

    # Distinct deterministic embeddings per text index so we can assert prefix truncation.
    counter = {"n": 0}

    def fake_embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 2560), dtype=np.float32)
        for i in range(len(texts)):
            out[i, counter["n"] % 2560] = 1.0
            counter["n"] += 1
        return out

    info = write_split_store(tmp_path, "coding_test", "train", [sess], fake_embed)
    assert info["sessions"] == 1
    assert info["docs"] == 6
    assert info["qa_rows"] == 2

    batches = E2EDatasetBatches(tmp_path, "coding_test", "train", max_docs=64, seed=0)
    assert len(batches.rows) == 2
    # Every row carries doc_end_idx < 6, so sampled docs must have <= 4 non-padded entries
    docs, pad, q_emb, q_txt, a_txt = batches.sample(4, torch.device("cpu"))
    assert docs.shape[1] <= 4
    assert int((~pad).sum()) <= 4 * 4
    # Confirm session split keeps splits disjoint
    many = [extract_session(f"s{i}", msgs, min_history_docs=2) for i in range(10)]
    splits = split_sessions([s for s in many if s is not None], val_frac=0.2, test_frac=0.2, seed=0)
    keys = [set(s.session_key for s in splits[k]) for k in ("train", "validation", "test")]
    assert len(keys[0] & keys[1]) == 0
    assert len(keys[0] & keys[2]) == 0
    assert len(keys[1] & keys[2]) == 0
