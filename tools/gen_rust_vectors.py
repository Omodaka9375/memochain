"""Generate bit-exact parity vectors for the Rust fold implementation.

Writes crates/memochain-core/tests/data/parity.txt with the Folder(seed=0)
weights and a deterministic document sequence, plus the exact int16 state
after each fold. The Rust test folds the same docs and asserts every state
matches, proving the two implementations agree bit for bit.

Run from repo root:  uv run python tools/gen_rust_vectors.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from memochain.fold import DOC_DIM, Folder
from memochain.state import MemoryState
from memochain.verify import write_with_receipt

OUT = Path(__file__).parent.parent / "crates" / "memochain-core" / "tests" / "data" / "parity.txt"


def main() -> None:
    folder = Folder(seed=0)  # default beta=0.75, decay=0.999
    rng = np.random.default_rng(7)
    docs = rng.integers(-1000, 1000, (5, DOC_DIM)).astype(np.int16)

    states = []
    accs = []
    state = MemoryState.zero()
    head = None
    for doc in docs:
        state, _receipt, head = write_with_receipt(folder, state, doc, head)
        states.append(state.tokens.reshape(-1).astype(np.int64).tolist())
        accs.append(head.accumulator)

    sections: list[tuple[str, list[int]]] = [
        ("W_KEY", folder.w_key.reshape(-1).astype(np.int64).tolist()),
        ("W_VAL", folder.w_val.reshape(-1).astype(np.int64).tolist()),
        ("BETA", [folder.beta]),
        ("DECAY", [folder.decay]),
        ("N_DOCS", [len(docs)]),
        ("DOC_DIM", [DOC_DIM]),
        ("DOCS", docs.reshape(-1).astype(np.int64).tolist()),
    ]
    for i, s in enumerate(states):
        sections.append((f"STATE_{i}", s))
    # accumulators are hex strings; emit as a separate receipts file with raw
    # hex tokens (the chain-logic parser reads whitespace tokens as strings,
    # so leading zeros are preserved).
    acc_path = OUT.parent / "parity_accumulators.txt"
    acc_lines = [f"N {len(accs)}"]
    acc_lines += [a for a in accs]
    acc_path.write_text("\n".join(acc_lines) + "\n", encoding="ascii")

    lines = []
    for name, values in sections:
        lines.append(f"{name} {len(values)}")
        lines.append(" ".join(str(v) for v in values))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
