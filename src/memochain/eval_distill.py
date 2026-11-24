"""Evaluate a distilled student checkpoint against the teacher on held-out data.

Loads runs/<name>/student.pt, replays the validation split, and reports
cosine agreement between student and teacher memory tokens, plus the same
statistic for two trivial baselines so the number has meaning:

  zeros baseline:   a student that outputs zeros (cosine undefined -> 0)
  raw-doc baseline: tokens computed from a single "bag of docs" mean vector
                    passed through the teacher's projection, i.e. what you get
                    without any query-conditioned memory at all

Usage:
  python -m memochain.eval_distill --checkpoint runs/pv4 --teacher data/stage3 --dataset pv4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_OK = False


def _require_torch() -> None:
    if not TORCH_OK:
        raise ImportError("PyTorch required: uv sync --extra train")


def main() -> None:
    _require_torch()
    parser = argparse.ArgumentParser(description="Evaluate distilled student vs teacher")
    parser.add_argument("--checkpoint", default="runs/pv4", help="dir containing student.pt")
    parser.add_argument("--teacher", default="data/stage3")
    parser.add_argument("--dataset", default="pv4")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-docs", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=100, help="cap eval batches (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(args.checkpoint) / "student.pt", map_location="cpu", weights_only=False)
    cfg_kwargs = ckpt["config"]
    from .student import MemoChainStudent, StudentConfig

    cfg = StudentConfig(**cfg_kwargs)
    student = MemoChainStudent(cfg).to(device)
    student.load_state_dict(ckpt["state_dict"])
    student.eval()
    print(f"loaded checkpoint: {args.checkpoint} (train cosine {ckpt.get('final_cosine')})")

    from .teacher import MemorillaTeacher

    teacher = MemorillaTeacher.from_pretrained(args.teacher).to(device)
    teacher.eval()

    from .distill import RealDatasetBatches, token_loss

    batches = RealDatasetBatches(
        args.data_dir, args.dataset, args.split, cfg.embedding_dim, args.num_docs, device
    )
    n_rows = len(batches.row_collection)
    n_batches = (n_rows + args.batch - 1) // args.batch
    if args.max_batches:
        n_batches = min(n_batches, args.max_batches)
    print(f"eval: {args.dataset}/{args.split}, {n_rows} rows, {n_batches} batches")

    stats = defaultdict(list)
    with torch.no_grad():
        for b in range(n_batches):
            docs, padding, questions = batches.next_batch(args.batch)
            cache = teacher.cache_keys_values(docs, padding)
            targets = teacher.read_cache(cache, padding, questions)

            state = student.fold_sequence(docs, padding, quantize=True)
            tokens = student.read_tokens(state, questions)
            _loss, cosine, mse = token_loss(tokens, targets)
            stats["student_cos"].append(cosine)
            stats["student_mse"].append(mse)

            # float-state reference: same weights, no Q14 quantization
            state_f = student.fold_sequence(docs, padding, quantize=False)
            tokens_f = student.read_tokens(state_f, questions)
            _l, cosine_f, _m = token_loss(tokens_f, targets)
            stats["student_cos_float"].append(cosine_f)

            # raw-doc baseline: mean of doc embeddings -> teacher projection
            mask = (~padding).float().unsqueeze(-1)
            mean_docs = (docs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            raw = teacher.memory_projection(teacher.layer_norm(mean_docs))
            raw = raw.unsqueeze(1).expand(-1, targets.shape[1], -1)
            _l, raw_cos, _m = token_loss(raw, targets)
            stats["rawdoc_cos"].append(float(raw_cos))

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "batches": n_batches,
        "student_cosine_mean": float(np.mean(stats["student_cos"])),
        "student_cosine_p25": float(np.percentile(stats["student_cos"], 25)),
        "student_cosine_p75": float(np.percentile(stats["student_cos"], 75)),
        "student_cosine_float_state": float(np.mean(stats["student_cos_float"])),
        "student_mse_mean": float(np.mean(stats["student_mse"])),
        "rawdoc_baseline_cosine": float(np.mean(stats["rawdoc_cos"])),
        "train_final_cosine": ckpt.get("final_cosine"),
    }
    print(json.dumps(report, indent=2))
    out = Path(args.checkpoint) / "eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
