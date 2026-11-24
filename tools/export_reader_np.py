"""Export the trained student reader head to numpy arrays for torch-free hosts.

Writes reader.npz (LayerNorm + slots + out linear weights) next to the existing
np_key_*.npy writer export (key_proj is shared between writer and reader).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memochain.student import MemoChainStudent, StudentConfig  # noqa: E402


def main() -> None:
    ckpt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/pv4_10k")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt_dir
    import torch

    ckpt = torch.load(ckpt_dir / "student.pt", map_location="cpu", weights_only=False)
    cfg = StudentConfig(**ckpt["config"])
    student = MemoChainStudent(cfg)
    student.load_state_dict(ckpt["state_dict"])

    np.savez(
        out / "reader.npz",
        norm_w=student.reader_head_norm.weight.detach().numpy(),
        norm_b=student.reader_head_norm.bias.detach().numpy(),
        slots_w=student.reader_slots.weight.detach().numpy(),
        slots_b=student.reader_slots.bias.detach().numpy(),
        out_w=student.reader_out.weight.detach().numpy(),
        out_b=student.reader_out.bias.detach().numpy(),
    )
    print(f"wrote {out / 'reader.npz'}")


if __name__ == "__main__":
    main()
