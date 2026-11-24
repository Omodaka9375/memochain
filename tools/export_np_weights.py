"""Export the trained student writer to torch-free numpy arrays for the proxy.

Usage:
  python tools/export_np_weights.py [checkpoint_dir] [out_dir]

Writes np_key_w/np_key_b/np_val_w/np_val_b/np_beta/np_decay as Q14 int16 arrays
(the exact same quantization as IntStudentWriter.from_checkpoint / weights.bin),
plus prints the canonical weights digest for cross-checking against weights.bin
(sha256) and SP1 proof metadata.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memochain.student_int import IntStudentWriter  # noqa: E402


def main() -> None:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/pv4_10k")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt
    w = IntStudentWriter.from_checkpoint(ckpt)
    arrays = {
        "np_key_w": w.w_key,
        "np_key_b": w.b_key,
        "np_val_w": w.w_val,
        "np_val_b": w.b_val,
        "np_beta": w.beta,
        "np_decay": w.decay,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        assert arr.dtype == np.dtype("<i2"), f"{name} not int16: {arr.dtype}"
        np.save(out / name, arr)
    print(f"exported {len(arrays)} Q14 int16 arrays to {out}")
    print(f"weights_digest: {w.weights_digest()}")


if __name__ == "__main__":
    main()
