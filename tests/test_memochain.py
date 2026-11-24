"""Tests for the MemoChain core: state format, fold, read, verification.

Document layout in these tests: doc[0:176] = key part (addresses the
memory), doc[176:352] = value part (the stored content).
"""

from __future__ import annotations

import numpy as np
import pytest

from memochain.fold import (
    DOC_DIM,
    KEY_PART,
    VAL_PART,
    Folder,
    make_doc,
    query_doc,
    read,
    read_baseline,
)
from memochain.state import (
    FORMAT_VERSION,
    KEY_DIM,
    N_HEADS,
    STATE_BYTES,
    VAL_DIM,
    Capsule,
    MemoryState,
)
from memochain.verify import ChainHead, verify_chain, verify_head, write_with_receipt


def _distractor(rng: np.random.Generator, i: int) -> np.ndarray:
    """A distractor doc: a one-hot key signal plus small noise everywhere."""
    key = np.zeros(KEY_PART, dtype=np.int16)
    key[i % KEY_PART] = 16000
    value = np.zeros(VAL_PART, dtype=np.int16)
    noise = rng.integers(-200, 200, size=DOC_DIM).astype(np.int16)
    return (np.concatenate([key, value]) + noise).astype(np.int16)


def test_state_is_exactly_22kb() -> None:
    state = MemoryState.zero()
    assert state.nbytes == STATE_BYTES == 22_528
    assert (N_HEADS, KEY_DIM, VAL_DIM) == (11, 32, 32)


def test_state_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        MemoryState(np.zeros((4, 4)))


def test_state_roundtrip_bytes() -> None:
    tokens = np.random.default_rng(0).integers(-1000, 1000, (N_HEADS, KEY_DIM, VAL_DIM))
    state = MemoryState(tokens.astype(np.int16))
    assert MemoryState.from_bytes(state.to_bytes()).digest() == state.digest()


def test_state_digest_is_full_256bit() -> None:
    d = MemoryState.zero().digest()
    assert len(d) == 64


def test_capsule_roundtrip_and_corruption() -> None:
    tokens = np.random.default_rng(1).integers(-1000, 1000, (N_HEADS, KEY_DIM, VAL_DIM))
    state = MemoryState(tokens.astype(np.int16))
    head = ChainHead(step=7, accumulator="a" * 64, state_digest=state.digest())
    capsule = Capsule(state=state, step=7, head=head.accumulator)

    raw = capsule.to_bytes()
    restored = Capsule.from_bytes(raw)
    assert restored.state.digest() == state.digest()
    assert restored.step == 7

    # Flip one bit: the capsule must fail loudly, not load silently.
    bad = bytearray(raw)
    bad[-1] ^= 0x01
    with pytest.raises(ValueError):
        Capsule.from_bytes(bytes(bad))

    # Wrong magic: rejected.
    bad_magic = bytearray(raw)
    bad_magic[0] = ord("X")
    with pytest.raises(ValueError):
        Capsule.from_bytes(bytes(bad_magic))


def test_fold_is_deterministic_bitwise() -> None:
    """Same inputs, two Folder instances: bit-identical states."""
    rng = np.random.default_rng(2)
    docs = [_distractor(rng, i) for i in range(50)]
    f1, f2 = Folder(seed=0), Folder(seed=0)
    s1, s2 = MemoryState.zero(), MemoryState.zero()
    for doc in docs:
        s1, _r1, _h1 = write_with_receipt(f1, s1, doc)
        s2, _r2, _h2 = write_with_receipt(f2, s2, doc)
    assert s1.digest() == s2.digest()


def test_fold_is_o1_in_history() -> None:
    """Fold cost does not depend on how much history is already folded."""
    folder = Folder(seed=0)
    doc = make_doc(np.full(KEY_PART, 100, dtype=np.int16), np.full(VAL_PART, 100, dtype=np.int16))

    s_small = MemoryState.zero()
    for _ in range(100):
        s_small = folder.fold(s_small, doc)
    s_big = MemoryState.zero()
    for _ in range(10_000):
        s_big = folder.fold(s_big, doc)

    assert s_small.nbytes == s_big.nbytes == STATE_BYTES
    a = folder.fold(s_small, doc)
    b = folder.fold(s_big, doc)
    assert a.nbytes == b.nbytes


def test_associative_recall_finds_planted_needle() -> None:
    """Plant a fact, fold 500 distractors, read it back by its key."""
    folder = Folder(seed=0, beta=0.1)
    rng = np.random.default_rng(3)

    key = np.zeros(KEY_PART, dtype=np.int16)
    key[5] = 16000
    value = np.zeros(VAL_PART, dtype=np.int16)
    value[:32] = 9000  # head 0 carries the planted payload
    needle = make_doc(key, value)
    val_n = folder.project(needle)[1][0].astype(np.float64)

    state = MemoryState.zero()
    state = folder.fold(state, needle)
    for i in range(500):
        state = folder.fold(state, _distractor(rng, 100 + i))

    recalled = read(folder, state, query_doc(key))[:32].astype(np.float64)
    corr = float(
        np.dot(recalled, val_n) / (np.linalg.norm(recalled) * np.linalg.norm(val_n) + 1e-9)
    )
    # After 500 noisy distractor writes the needle must still be recoverable.
    # Measured on this configuration (beta=0.1): corr = 0.96 at 100 distractors,
    # 0.76 at 500, then falls off; distractor noise drives the degradation.
    assert corr > 0.5, f"needle corrupted after distractors: corr={corr:.3f}"


def test_delta_rule_updates_facts() -> None:
    """Writing a new value for the same key replaces the old one."""
    folder = Folder(seed=0)

    key = np.zeros(KEY_PART, dtype=np.int16)
    key[5] = 16000

    v1 = np.zeros(VAL_PART, dtype=np.int16)
    v1[:16] = 12000   # old fact: strong in the first 16 value dims
    v2 = np.zeros(VAL_PART, dtype=np.int16)
    v2[16:32] = 12000  # new fact: strong in the next 16 dims

    state = MemoryState.zero()
    state = folder.fold(state, make_doc(key, v1))
    state = folder.fold(state, make_doc(key, v2))

    recalled = read(folder, state, query_doc(key))
    # With random dense W_val, v1 and v2 project to non-orthogonal value
    # vectors, so "erasure" is measured as dominance, not zero. Verify the
    # exact least-squares fit: recalled ~= decay*(1-beta)*val1 + beta*val2,
    # meaning the new fact dominates the old one by roughly 4x.
    val1 = folder.project(make_doc(key, v1))[1][0].astype(np.float64)
    val2 = folder.project(make_doc(key, v2))[1][0].astype(np.float64)
    A = np.stack([val1, val2], axis=1)
    recalled0 = recalled[:VAL_DIM].astype(np.float64)
    (coef_old, coef_new), *_ = np.linalg.lstsq(A, recalled0, rcond=None)
    assert coef_new > 0.6, f"new value not recalled: coef={coef_new:.3f}"
    assert coef_old < 0.30, f"old value not suppressed: coef={coef_old:.3f}"
    assert coef_new > 2.0 * coef_old, (
        f"new fact must dominate: old={coef_old:.3f}, new={coef_new:.3f}"
    )


def test_read_baseline_attention_picks_best_doc() -> None:
    rng = np.random.default_rng(4)
    docs = rng.integers(-500, 500, (8, DOC_DIM)).astype(np.int16)
    target = rng.integers(-500, 500, DOC_DIM).astype(np.int16)
    docs[3] = target  # exact match should win dot-product attention
    got = read_baseline(docs, target)
    assert np.array_equal(got, target)


def test_receipt_chain_head_is_o1() -> None:
    """The light-client head stays 3 small fields, no matter the history."""
    folder = Folder(seed=0)
    rng = np.random.default_rng(5)
    state = MemoryState.zero()
    head: ChainHead | None = None
    for i in range(1_000):
        doc = _distractor(rng, i)
        state, _r, head = write_with_receipt(folder, state, doc, head)

    assert head is not None and head.step == 1_000
    assert len(head.accumulator) == 64
    assert verify_head(state, head) is True


def test_full_audit_replays_and_verifies() -> None:
    folder = Folder(seed=0)
    rng = np.random.default_rng(6)
    docs = [_distractor(rng, i) for i in range(30)]

    state = MemoryState.zero()
    head: ChainHead | None = None
    for doc in docs:
        state, _r, head = write_with_receipt(folder, state, doc, head)

    assert verify_chain(folder, docs, head) is True


def test_tamper_detected_everywhere() -> None:
    folder = Folder(seed=0)
    rng = np.random.default_rng(7)
    docs = [_distractor(rng, i) for i in range(10)]

    state = MemoryState.zero()
    head: ChainHead | None = None
    for doc in docs:
        state, _r, head = write_with_receipt(folder, state, doc, head)
    assert head is not None

    # Tamper with one document: full audit fails.
    bad_docs = list(docs)
    bad_docs[5] = bad_docs[5].copy()
    bad_docs[5][7] += 3
    assert verify_chain(folder, bad_docs, head) is False

    # Tamper with state bytes: digest changes and head check fails.
    raw = bytearray(state.to_bytes())
    raw[0] ^= 0x01
    tampered = MemoryState.from_bytes(bytes(raw))
    assert tampered.digest() != state.digest()
    assert verify_head(tampered, head) is False


def test_format_version_is_stable() -> None:
    assert FORMAT_VERSION == 3  # v3: hot tail + kind byte (v1/v2 readers OK)
