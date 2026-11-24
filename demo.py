"""End-to-end demo: fold a history, recall a planted fact, verify the
receipt chain, show the tamper check, and run the constant-cost benchmark.

Run: uv run demo.py
"""

from __future__ import annotations

import numpy as np

from memochain.bench import run as bench
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
from memochain.state import Capsule, MemoryState
from memochain.verify import verify_chain, verify_head, write_with_receipt


def main() -> None:
    rng = np.random.default_rng(42)

    print("== MemoChain demo: 22 KB memory, O(1) writes ==")
    print()

    folder = Folder(seed=0, beta=0.1)

    # 1. Fold a "chat history": 5,000 distractors plus one planted fact.
    n_docs = 5_000
    key = np.zeros(KEY_PART, dtype=np.int16)
    key[5] = 16000  # the fact's address
    value = np.zeros(VAL_PART, dtype=np.int16)
    value[:32] = 9000  # the fact's payload
    needle = make_doc(key, value)

    state = MemoryState.zero()
    head = None
    for i in range(n_docs):
        doc = needle if i == 0 else _distractor(rng, i)
        state, _receipt, head = write_with_receipt(folder, state, doc, head)

    print(f"folded {n_docs:,} documents -> state is {state}")
    print(f"receipt chain head: step={head.step}, accumulator={head.accumulator[:16]}...")
    print()

    # 2. Recall the planted fact by its key: O(1), independent of history.
    recalled = read(folder, state, query_doc(key))[:32].astype(np.float64)
    val_n = folder.project(needle)[1][0].astype(np.float64)
    corr = float(
        np.dot(recalled, val_n) / (np.linalg.norm(recalled) * np.linalg.norm(val_n) + 1e-9)
    )
    print(f"planted fact recall after {n_docs:,} distractors: corr={corr:.3f}")
    print()

    # 3. Light verification (O(1)) and full audit (replay).
    print(f"light head check (O(1)): {verify_head(state, head)}")
    tampered_docs = [_distractor(rng, i) for i in range(6)]
    print(f"full audit replay with wrong history fails: {verify_chain(folder, tampered_docs, head)}")
    print()

    # 4. Export as a versioned capsule, corrupt one byte, reload fails loudly.
    capsule = Capsule(state=state, step=head.step, head=head.accumulator)
    raw = bytearray(capsule.to_bytes())
    raw[-1] ^= 0x01
    try:
        Capsule.from_bytes(bytes(raw))
        print("capsule corruption NOT detected (bug!)")
    except ValueError as exc:
        print(f"corrupted capsule rejected: {exc}")
    ok_capsule = Capsule.from_bytes(capsule.to_bytes())
    print(f"intact capsule reloads: digest matches = "
          f"{ok_capsule.state.digest() == state.digest()}")
    print()

    # 5. Benchmark: constant cost vs the growing baseline.
    print("== Benchmark ==")
    bench()


def _distractor(rng: np.random.Generator, i: int) -> np.ndarray:
    key = np.zeros(KEY_PART, dtype=np.int16)
    key[i % KEY_PART] = 16000
    noise = rng.integers(-200, 200, size=DOC_DIM).astype(np.int16)
    return (np.concatenate([key, np.zeros(VAL_PART, dtype=np.int16)]) + noise).astype(np.int16)


if __name__ == "__main__":
    main()
