"""Tests for exact Q14 integer student writer (src/memochain/student_int.py)."""

from __future__ import annotations

import numpy as np

from memochain.student_int import (
    EMBED_DIM,
    KEY_PART,
    WEIGHTS_BYTES,
    IntStudentWriter,
    quantize_embeddings_q14,
)
from memochain.state import Capsule, N_HEADS


def _synthetic_writer(seed: int = 0) -> IntStudentWriter:
    rng = np.random.default_rng(seed)
    return IntStudentWriter(
        w_key=rng.integers(-2000, 2000, size=(KEY_PART, EMBED_DIM), dtype="<i2"),
        b_key=rng.integers(-500, 500, size=(KEY_PART,), dtype="<i2"),
        w_val=rng.integers(-2000, 2000, size=(KEY_PART, EMBED_DIM), dtype="<i2"),
        b_val=rng.integers(-500, 500, size=(KEY_PART,), dtype="<i2"),
        beta=rng.integers(5000, 7500, size=(N_HEADS,), dtype="<i2"),
        decay=np.full((N_HEADS,), 16350, dtype="<i2"),
    )


def test_weights_binary_roundtrip_and_digest() -> None:
    w = _synthetic_writer(1)
    raw = w.to_bytes()
    assert len(raw) == WEIGHTS_BYTES
    w2 = IntStudentWriter.from_bytes(raw)
    assert w.weights_digest() == w2.weights_digest()
    assert np.array_equal(w.w_key, w2.w_key)
    assert np.array_equal(w.w_val, w2.w_val)
    assert np.array_equal(w.beta, w2.beta)
    assert np.array_equal(w.decay, w2.decay)


def test_incremental_split_matches_monolithic_fold_bit_exact() -> None:
    """Folding docs[0..5] then +1 doc[5] must equal folding docs[0..6] bit-for-bit."""
    w = _synthetic_writer(2)
    rng = np.random.default_rng(3)
    docs_f = rng.standard_normal((6, EMBED_DIM)).astype(np.float32) * 0.02
    docs_q = quantize_embeddings_q14(docs_f)

    full_state, full_head = w.fold_sequence(docs_q)

    s5, h5 = w.fold_sequence(docs_q[:5])
    s6, h6 = w.fold_sequence(docs_q[5:6], initial_state=s5.tokens, initial_head=h5)

    assert np.array_equal(full_state.tokens, s6.tokens)
    assert full_head == h6

    # Capsule roundtrip preserves exact 22,528-byte state and accumulator head.
    cap, head = w.export_proven_capsule(docs_q)
    reloaded = Capsule.from_bytes(cap.to_bytes())
    assert reloaded.step == 6
    assert reloaded.head == head.accumulator
    assert reloaded.state.digest() == head.state_digest
    assert np.array_equal(reloaded.state.tokens, full_state.tokens)
