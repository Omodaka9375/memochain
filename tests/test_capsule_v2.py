"""Capsule v2 tests: hot-tail roundtrip, corruption, hybrid integration. Pure numpy."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from memochain.hybrid import HybridMemory
from memochain.state import (
    FORMAT_VERSION,
    HOT_DOC_DIM,
    STATE_BYTES,
    Capsule,
    MemoryState,
)
from memochain.verify import write_with_receipt
from memochain.fold import Folder, DOC_DIM


def _tokens(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, (11, 32, 32)).astype(np.int16)


def test_v2_roundtrip_with_hot_tail(tmp_path) -> None:
    state = MemoryState(_tokens(0))
    hot = [np.full(DOC_DIM, i, dtype=np.int16) for i in range(3)]
    capsule = Capsule(state=state, step=7, head="a" * 64, hot=tuple(hot))
    assert FORMAT_VERSION == 3

    path = tmp_path / "capsule.bin"
    path.write_bytes(capsule.to_bytes())
    restored = Capsule.from_bytes(path.read_bytes())

    assert restored.step == 7
    assert restored.state.digest() == state.digest()
    assert len(restored.hot) == 3
    for i, h in enumerate(restored.hot):
        assert np.array_equal(h, hot[i])


def test_v2_hot_tail_corruption_detected(tmp_path) -> None:
    state = MemoryState(_tokens(1))
    hot = [np.arange(DOC_DIM, dtype=np.int16)]
    capsule = Capsule(state=state, step=2, head="b" * 64, hot=tuple(hot))
    raw = bytearray(capsule.to_bytes())

    # Flip a byte inside the hot section: must fail loudly.
    raw[-1] ^= 0x01
    with pytest.raises(ValueError):
        Capsule.from_bytes(bytes(raw))

    # Wrong size: rejected.
    good = capsule.to_bytes()
    with pytest.raises(ValueError):
        Capsule.from_bytes(good[:-2])


def test_v1_files_still_load() -> None:
    """A v1 capsule (state only) parses with hot=()."""
    payload = _tokens(2).tobytes()
    header = struct.pack(
        "<4sHQ32s32s", b"MCN1", 1, 42, bytes.fromhex("c" * 64), hashlib.sha256(payload).digest()
    )
    capsule = Capsule.from_bytes(header + payload)
    assert capsule.step == 42
    assert capsule.head == "c" * 64
    assert capsule.hot == ()
    assert capsule.state.digest() == MemoryState(_tokens(2)).digest()


def test_hot_entries_must_match_doc_dim() -> None:
    state = MemoryState(_tokens(3))
    bad = [np.zeros(10, dtype=np.int16)]
    with pytest.raises(ValueError):
        Capsule(state=state, step=1, head="", hot=tuple(bad)).to_bytes()


def test_hybrid_memory_capsule_roundtrip(tmp_path) -> None:
    folder = Folder(seed=0)
    rng = np.random.default_rng(5)
    mem = HybridMemory(folder, hot_size=3)

    needle = np.zeros(DOC_DIM, dtype=np.int16)
    needle[5] = 16000
    for i in range(20):
        noise = rng.integers(-200, 200, size=DOC_DIM).astype(np.int16)
        mem.write((needle if i == 0 else np.zeros(DOC_DIM, dtype=np.int16)) + noise)
    mem.write(needle)  # ends in the hot tail

    capsule = mem.to_capsule()
    assert capsule.step >= 18  # evictions folded into the cold state
    assert len(capsule.hot) == 3

    path = tmp_path / "hybrid.bin"
    path.write_bytes(capsule.to_bytes())
    restored = HybridMemory.from_capsule(folder, Capsule.from_bytes(path.read_bytes()))

    assert restored.head is not None and restored.head.step == capsule.step
    result = restored.read(needle)
    assert any(np.array_equal(h, needle) for h in result.hot)
    assert result.cold.shape == (352,)  # one recalled value per key dim
    assert restored.total_bytes <= 3 * DOC_DIM * 2 + STATE_BYTES
