"""Capacity ablation: parameterized state shapes fold and export correctly."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from memochain.student import MemoChainStudent, StudentConfig
from memochain.train_e2e import export_student_artifacts

CAPACITIES = [
    (1, 32, 32),    # 2 KiB floor
    (5, 32, 32),    # 10 KiB
    (11, 32, 32),   # 22 KiB canonical
    (22, 32, 32),   # 44 KiB
    (11, 64, 64),   # 88 KiB wide
]


@pytest.mark.parametrize("h,k,v", CAPACITIES)
def test_capacity_folds_and_exports(h: int, k: int, v: int, tmp_path) -> None:
    cfg = StudentConfig(embedding_dim=2560, output_dim=64, num_memories=4, num_heads=h, key_dim=k, val_dim=v)
    student = MemoChainStudent(cfg)
    assert student.initial_state(2).shape == (2, h, k, v)
    expected_bytes = h * k * v * 2

    docs = torch.randn(2, 6, 2560) * 0.05
    pad = torch.zeros(2, 6, dtype=torch.bool)
    state = student.fold_sequence(docs, padding=pad, quantize=True)
    assert state.shape == (2, h, k, v)

    q = torch.randn(2, 2560) * 0.05
    tokens = student.read_tokens(state, q)
    assert tokens.shape == (2, 4, 64)
    assert torch.isfinite(tokens).all()

    info = export_student_artifacts(student, tmp_path / "ck")
    assert info["state_bytes"] == expected_bytes
    assert (tmp_path / "ck" / "student.pt").exists()
    assert (tmp_path / "ck" / "reader.npz").exists()
    if (h, k, v) == (11, 32, 32):
        # canonical shape: SP1 wire export present
        assert (tmp_path / "ck" / "weights.bin").exists()
        assert (tmp_path / "ck" / "np_key_w.npy").exists()
        assert len(info["weights_digest"]) == 64
    else:
        # ablation shapes: no fixed-format weights.bin
        assert not (tmp_path / "ck" / "weights.bin").exists()


def test_canonical_state_bytes_unchanged() -> None:
    from memochain.state import STATE_BYTES

    cfg = StudentConfig()  # defaults = 11x32x32
    assert cfg.num_heads * cfg.key_dim * cfg.val_dim * 2 == STATE_BYTES
