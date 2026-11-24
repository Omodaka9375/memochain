"""Parity test: NumpyReader vs MemoChainStudent.read_tokens (torch)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from memochain.reader_np import NumpyReader

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


def test_numpy_reader_matches_torch(tmp_path) -> None:
    from memochain.student import MemoChainStudent, StudentConfig

    torch.manual_seed(0)
    cfg = StudentConfig()
    student = MemoChainStudent(cfg).eval()
    # random but deterministic weights
    for p in student.parameters():
        torch.nn.init.normal_(p, std=0.05)

    state = torch.from_numpy(
        (np.random.default_rng(7).normal(0, 0.02, (11, 32, 32)) * (1 << 14))
        .round()
        .clip(-32768, 32767)
        .astype("<i2")
    )
    q = torch.from_numpy(np.random.default_rng(8).normal(0, 0.1, 2560).astype(np.float32))

    with torch.inference_mode():
        expected = student.read_tokens((state[None].float() / (1 << 14)), q[None])[0].numpy()

    # export the same weights to npz
    npz = tmp_path / "reader.npz"
    torch.save({}, npz)  # placeholder replace below
    np.savez(
        npz,
        norm_w=student.reader_head_norm.weight.detach().numpy(),
        norm_b=student.reader_head_norm.bias.detach().numpy(),
        slots_w=student.reader_slots.weight.detach().numpy(),
        slots_b=student.reader_slots.bias.detach().numpy(),
        out_w=student.reader_out.weight.detach().numpy(),
        out_b=student.reader_out.bias.detach().numpy(),
    )

    from memochain.student_int import KEY_PART

    # key_proj weights from the student (shared between writer and reader)
    kw = student.key_proj.weight.detach().numpy()  # [H*K, D]
    kb = student.key_proj.bias.detach().numpy()

    reader = NumpyReader(npz, kw, kb)
    got = reader.read_tokens(state.numpy(), q.numpy())

    assert got.shape == expected.shape
    cos = float(
        np.sum(got * expected) / (np.linalg.norm(got) * np.linalg.norm(expected) + 1e-9)
    )
    assert cos > 0.999, f"cosine {cos}"
