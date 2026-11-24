"""Generate a ChainInput JSON for the native runner and SP1 host.

Emits the parity-vector collection (Folder seed 0, 5 docs) as the exact JSON
the guest consumes, so the native runner, the zkVM guest, and Python all
process identical inputs. Optionally emits N extra random docs for a
heavier cycle estimate.

Run: uv run python tools/gen_chain_input.py [--docs 5] [--out chain_input.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memochain.fold import DOC_DIM, Folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=5, help="number of documents")
    parser.add_argument("--out", default="chain_input.json")
    args = parser.parse_args()

    folder = Folder(seed=0)
    rng = np.random.default_rng(7)
    docs = rng.integers(-1000, 1000, (args.docs, DOC_DIM)).astype(np.int16)

    payload = {
        "w_key": [int(x) for x in folder.w_key.reshape(-1)],
        "w_val": [int(x) for x in folder.w_val.reshape(-1)],
        "beta": int(folder.beta),
        "decay": int(folder.decay),
        "n_docs": int(args.docs),
        "docs_buf": [int(x) for doc in docs for x in doc],
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {args.docs} docs)")


if __name__ == "__main__":
    main()
