"""Latency and memory benchmark: MemoChain vs the growing baseline.

Shows the two claims that matter:
  1. fold latency and memory are constant in history size
  2. the growing baseline pays O(n) on every query
"""

from __future__ import annotations

import time

import numpy as np

from .fold import DOC_DIM, Folder, read, read_baseline
from .state import STATE_BYTES, MemoryState
from .verify import write_with_receipt


def _time(fn, *args, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best * 1000.0  # ms


def run(sizes: tuple[int, ...] = (100, 1_000, 10_000, 100_000)) -> None:
    rng = np.random.default_rng(42)
    folder = Folder(seed=0)

    header = (
        f"{'docs':>8} | {'fold ms/doc':>11} | {'memo read ms':>12} | "
        f"{'baseline read ms':>17} | {'memo RAM':>10} | {'baseline RAM':>12}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        docs = rng.integers(-500, 500, (n, DOC_DIM)).astype(np.int16)
        query = rng.integers(-500, 500, DOC_DIM).astype(np.int16)

        # Write all documents into the fixed state, timing the fold.
        state = MemoryState.zero()
        start = time.perf_counter()
        for doc in docs:
            state = folder.fold(state, doc)
        fold_ms = (time.perf_counter() - start) / n * 1000.0

        memo_ms = _time(read, folder, state, query)
        base_ms = _time(read_baseline, docs, query)

        baseline_ram = n * DOC_DIM * 2  # int16 raw-document cache

        print(
            f"{n:>8} | {fold_ms:>11.3f} | {memo_ms:>12.3f} | "
            f"{base_ms:>17.3f} | {_fmt(state.nbytes):>10} | {_fmt(baseline_ram):>12}"
        )

    print()
    print(f"MemoChain state: {STATE_BYTES:,} bytes, constant.")
    print(f"Baseline cache: grows linearly (int16, {DOC_DIM} dims per doc).")


def _fmt(nbytes: int) -> str:
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MiB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KiB"
    return f"{nbytes} B"


if __name__ == "__main__":
    run()
