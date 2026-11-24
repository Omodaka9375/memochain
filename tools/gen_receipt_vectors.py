"""Rust-Python receipt parity: sha256 chain over the fold must match exactly.

Runs the Rust chain via a tiny cargo test that reads the same parity vectors
as the fold, and compares against Python's verify.py receipts. The cross-check
runs the Python chain over the vector docs and writes expected receipts to a
temp file; the Rust test asserts they match. Simpler: generate the receipt
vectors here and assert in Rust via a second parity file.

This test file itself only generates and validates the Python side; the Rust
side lives in crates/memochain-core/tests/parity_receipts.rs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from memochain.fold import DOC_DIM, Folder
from memochain.state import MemoryState
from memochain.verify import write_with_receipt

OUT = (
    Path(__file__).parent.parent
    / "crates" / "memochain-core" / "tests" / "data" / "parity_receipts.txt"
)


def main() -> None:
    folder = Folder(seed=0)
    rng = np.random.default_rng(11)
    docs = rng.integers(-1000, 1000, (6, DOC_DIM)).astype(np.int16)

    state = MemoryState.zero()
    head = None
    lines = [f"N_DOCS 1\n{len(docs)}"]

    receipts = []
    for doc in docs:
        state, receipt, head = write_with_receipt(folder, state, doc, head)
        receipts.append(receipt)

    # docs for the rust replay
    lines = []
    lines.append(f"DOC_DIM 1\n{DOC_DIM}")
    lines.append(f"N_DOCS 1\n{len(docs)}")
    lines.append(f"DOCS {len(docs) * DOC_DIM}")
    lines.append(" ".join(str(int(v)) for v in docs.reshape(-1)))
    lines.append(f"HEAD_STEP 1\n{head.step}")
    lines.append(f"HEAD_ACC 1\n{head.accumulator}")
    lines.append(f"HEAD_STATE 1\n{head.state_digest}")
    for i, r in enumerate(receipts):
        lines.append(f"RECEIPT_{i} 1\n{r.digest}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
