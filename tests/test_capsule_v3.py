"""Capsule v3 tests: embedding hot tail (student world) + kind handling. Pure numpy."""

from __future__ import annotations

import numpy as np
import pytest

from memochain.state import (
    HOT_DOC_DIM,
    HOT_KIND_EMBED,
    HOT_KIND_RAW,
    FORMAT_VERSION,
    MemoryState,
    Capsule,
)


def _tokens(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, (11, 32, 32)).astype(np.int16)


def test_v3_embedding_hot_tail_roundtrip(tmp_path) -> None:
    state = MemoryState(_tokens(0))
    embs = [np.random.default_rng(i).standard_normal(2560).astype(np.float16) for i in range(4)]
    capsule = Capsule(
        state=state, step=12, head="e" * 64, hot=tuple(embs), hot_kind=HOT_KIND_EMBED
    )
    path = tmp_path / "capsule.bin"
    path.write_bytes(capsule.to_bytes())
    restored = Capsule.from_bytes(path.read_bytes())

    assert FORMAT_VERSION == 3
    assert restored.hot_kind == HOT_KIND_EMBED
    assert len(restored.hot) == 4
    for i, h in enumerate(restored.hot):
        assert h.dtype == np.float16
        assert np.array_equal(h, embs[i])
    assert restored.state.digest() == state.digest()


def test_v3_raw_hot_tail_matches_v2_semantics() -> None:
    state = MemoryState(_tokens(1))
    docs = [np.full(HOT_DOC_DIM, 7, dtype=np.int16)]
    capsule = Capsule(state=state, step=1, head="", hot=tuple(docs), hot_kind=HOT_KIND_RAW)
    restored = Capsule.from_bytes(capsule.to_bytes())
    assert restored.hot_kind == HOT_KIND_RAW
    assert np.array_equal(restored.hot[0], docs[0])


def test_v3_mixed_dims_rejected() -> None:
    state = MemoryState(_tokens(2))
    bad = (np.zeros(2560, dtype=np.float16), np.zeros(100, dtype=np.float16))
    with pytest.raises(ValueError):
        Capsule(state=state, step=1, head="", hot=bad, hot_kind=HOT_KIND_EMBED).to_bytes()


def test_v3_unknown_kind_rejected() -> None:
    state = MemoryState(_tokens(3))
    with pytest.raises(ValueError):
        Capsule(state=state, step=1, head="", hot=(np.zeros(8),), hot_kind=5).to_bytes()


def test_v3_embedding_tail_corruption_detected(tmp_path) -> None:
    embs = [np.random.default_rng(9).standard_normal(2560).astype(np.float16)]
    capsule = Capsule(
        state=MemoryState(_tokens(4)), step=3, head="f" * 64,
        hot=tuple(embs), hot_kind=HOT_KIND_EMBED,
    )
    raw = bytearray(capsule.to_bytes())
    raw[-3] ^= 0x01
    with pytest.raises(ValueError):
        Capsule.from_bytes(bytes(raw))
