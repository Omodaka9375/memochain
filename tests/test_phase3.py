"""Tests for Phase 3: student, teacher, hybrid. Skipped without the torch extra."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from memochain.hybrid import HybridMemory  # noqa: E402
from memochain.state import STATE_BYTES, MemoryState  # noqa: E402
from memochain.student import (  # noqa: E402
    MemoChainStudent,
    StudentConfig,
    export_to_memory_state,
    quantize_state_int16,
)
from memochain.teacher import SyntheticTeacher  # noqa: E402
from memochain.distill import token_loss  # noqa: E402

from memochain.fold import DOC_DIM, KEY_PART, VAL_PART  # noqa: E402


def _small_config() -> StudentConfig:
    return StudentConfig(
        embedding_dim=64,
        output_dim=128,
        num_memories=4,
        num_heads=11,
        key_dim=32,
        val_dim=32,
        reader_hidden=64,
    )


def _synthetic_teacher_small(seed: int = 0) -> SyntheticTeacher:
    return SyntheticTeacher(embedding_dim=64, output_dim=128, num_memories=4, seed=seed)


def test_student_state_shape_is_22kb() -> None:
    student = MemoChainStudent(_small_config())
    docs = torch.randn(2, 16, 64)
    state = student.fold_sequence(docs)
    assert state.shape == (2, 11, 32, 32)
    flat = state[0].detach().numpy().astype(np.int16)
    assert flat.nbytes == STATE_BYTES == 22_528


def test_student_keys_are_unit_norm() -> None:
    student = MemoChainStudent(_small_config())
    docs = torch.randn(3, 64)
    keys, vals = student.project(docs)
    norms = keys.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_student_delta_rule_replaces_fact() -> None:
    """Writing the same key twice: recalled value should track the SECOND write."""
    torch.manual_seed(0)
    student = MemoChainStudent(_small_config())
    with torch.no_grad():
        student.beta_logits.fill_(3.0)  # beta ~ 0.95, fast write
        student.decay_logits.fill_(6.0)

    # Two docs whose key parts are similar but value parts differ.
    key_dir = torch.randn(1, 64)
    v1 = 3.0 * key_dir
    v2 = -3.0 * key_dir  # opposite value, same direction

    state = student.initial_state(1)
    state = student.fold_step(state, v1)
    r1 = student.read_tokens(state, key_dir)
    state = student.fold_step(state, v2)
    r2 = student.read_tokens(state, key_dir)

    # recall must change after the second write (not stay blended 50/50)
    diff = (r1 - r2).abs().max().item()
    assert diff > 1e-3


def test_student_q14_quantization_rounds_to_grid() -> None:
    x = torch.tensor([[0.0001234, -0.0002, 1.5]])
    q = quantize_state_int16(x)
    grid = (torch.round(x * (1 << 14)) / (1 << 14)).abs()
    assert torch.allclose(q.abs(), grid, atol=1e-9)


def test_student_exports_to_memory_state() -> None:
    student = MemoChainStudent(_small_config())
    docs = torch.randn(1, 8, 64)
    state = student.fold_sequence(docs)
    ms = export_to_memory_state(student, state)
    assert isinstance(ms, MemoryState)
    assert ms.nbytes == 22_528


def test_teacher_synthetic_matches_interface() -> None:
    teacher = _synthetic_teacher_small()
    docs = torch.randn(2, 8, 64)
    question = torch.randn(2, 64)
    cache = teacher.cache_keys_values(docs, None)
    tokens = teacher.read_cache(cache, None, question)
    assert tokens.shape == (2, 4, 128)


def test_distill_loss_decreases_on_synthetic() -> None:
    """A few gradient steps must improve student-teacher token agreement."""
    torch.manual_seed(0)
    cfg = _small_config()
    student = MemoChainStudent(cfg)
    teacher = _synthetic_teacher_small()

    docs = torch.randn(2, 12, 64)
    questions = torch.randn(2, 64)
    with torch.no_grad():
        cache = teacher.cache_keys_values(docs, None)
        targets = teacher.read_cache(cache, None, questions)

    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        state = student.fold_sequence(docs)
        tokens = student.read_tokens(state, questions)
        loss, _cos, _mse = token_loss(tokens, targets)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first, f"loss did not decrease: {first} -> {loss.item()}"


def test_hybrid_memory_o1_footprint_and_recall() -> None:
    np_rng = np.random.default_rng(0)
    folder = _tiny_folder()
    mem = HybridMemory(folder, hot_size=3)

    needle = _doc(np_rng, 5, planted=True)
    for i in range(20):
        mem.write(_doc(np_rng, i))
    mem.write(needle)  # planted needle stays in the hot tail

    # total footprint must stay bounded: hot 3 docs + 22 KB cold state
    assert mem.total_bytes <= 3 * DOC_DIM * 2 + STATE_BYTES
    assert mem.head is not None and mem.head.step >= 17  # 20 + 3 - 3 evictions
    result = mem.read(needle)
    assert any(np.array_equal(h, needle) for h in result.hot)
    assert result.cold.shape == (352,)  # one recalled value per key dim


def _tiny_folder():
    from memochain.fold import Folder

    return Folder(seed=0)


def _doc(rng: np.random.Generator, i: int, planted: bool = False) -> np.ndarray:
    key = np.zeros(KEY_PART, dtype=np.int16)
    key[i % KEY_PART] = 16000
    value = np.zeros(VAL_PART, dtype=np.int16)
    if planted:
        value[:32] = 9000
    noise = rng.integers(-200, 200, size=DOC_DIM).astype(np.int16)
    return (np.concatenate([key, value]) + noise).astype(np.int16)
