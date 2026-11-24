"""Capsule export tests: student fold -> capsule -> verify -> read. Skips without torch."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from memochain.export import (  # noqa: E402
    doc_hash,
    fold_student_state,
    load_capsule,
    provenance_head,
    save_capsule,
    verify_capsule,
)
from memochain.state import STATE_BYTES, Capsule  # noqa: E402
from memochain.student import MemoChainStudent, StudentConfig  # noqa: E402


def _small_student() -> MemoChainStudent:
    cfg = StudentConfig(
        embedding_dim=64,
        output_dim=128,
        num_memories=4,
        num_heads=11,
        key_dim=32,
        val_dim=32,
        reader_hidden=64,
    )
    student = MemoChainStudent(cfg)
    student.eval()
    return student


def test_provenance_head_is_order_sensitive() -> None:
    hashes = [doc_hash(np.full(8, i, dtype=np.float32)) for i in range(3)]
    head = provenance_head(hashes, "a" * 64)
    assert head != provenance_head(list(reversed(hashes)), "a" * 64)
    assert head != provenance_head(hashes, "b" * 64)
    assert head == provenance_head(hashes, "a" * 64)  # deterministic


def test_export_roundtrip_and_verify(tmp_path) -> None:
    student = _small_student()
    rng = np.random.default_rng(0)
    docs = rng.standard_normal((50, 64)).astype(np.float32)

    state, hashes = fold_student_state(student, docs, torch.device("cpu"))
    assert state.nbytes == STATE_BYTES
    assert len(hashes) == 50

    capsule = Capsule(state=state, step=50, head=provenance_head(hashes, state.digest()))
    path = tmp_path / "capsule.bin"
    save_capsule(capsule, path)

    reloaded = load_capsule(path)
    assert reloaded.step == 50
    assert reloaded.state.digest() == state.digest()

    report = verify_capsule(reloaded, hashes)
    assert report["payload_intact"] is True
    assert report["head_matches_docs"] is True

    # A dropped or reordered document breaks the head.
    assert verify_capsule(reloaded, hashes[:-1])["head_matches_docs"] is False
    assert verify_capsule(reloaded, list(reversed(hashes)))["head_matches_docs"] is False


def test_quantized_state_is_exact_int16_grid() -> None:
    student = _small_student()
    rng = np.random.default_rng(1)
    docs = rng.standard_normal((10, 64)).astype(np.float32)
    state, _ = fold_student_state(student, docs, torch.device("cpu"))
    grid = state.tokens.astype(np.int64)
    assert grid.max() <= 32767 and grid.min() >= -32768


def test_read_from_reloaded_state_matches_original() -> None:
    """Reading the reloaded (serialized) state gives identical tokens."""
    student = _small_student()
    rng = np.random.default_rng(2)
    docs = rng.standard_normal((24, 64)).astype(np.float32)
    question = rng.standard_normal((1, 64)).astype(np.float32)

    state, hashes = fold_student_state(student, docs, torch.device("cpu"))
    capsule = Capsule(state=state, step=24, head=provenance_head(hashes, state.digest()))

    # tokens from the in-memory state
    q = torch.from_numpy(question)
    s_mem = torch.from_numpy(state.tokens.astype(np.float32) / float(1 << 14)).unsqueeze(0)
    with torch.no_grad():
        t1 = student.read_tokens(s_mem, q)

    # tokens from a state that went through serialization
    s_disk = torch.from_numpy(capsule.state.tokens.astype(np.float32) / float(1 << 14)).unsqueeze(0)
    with torch.no_grad():
        t2 = student.read_tokens(s_disk, q)

    assert torch.allclose(t1, t2, atol=0.0)  # bit-identical
