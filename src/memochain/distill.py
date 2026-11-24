"""Distillation trainer: teach the 22 KB student to act like Memorilla.

The student folds a document collection into the fixed 11x32x32 state, then
reads memory tokens for a question; the teacher (Memorilla's MemoryModule)
produces the target tokens from the same collection and question. Loss is
cosine + MSE over [B, num_memories, output_dim].

Designed to run on a single consumer GPU (tested target: RTX 3060 12 GB):
the teacher is only the 388 MB memory module, embeddings are precomputed,
and the student is a few million parameters. `--synthetic` runs offline with
no downloads; `--teacher data/stage3 --dataset pv4` trains against the real
teacher on the real 382 MB pv4 embedding store (downloads on first use).

Usage:
  python -m memochain.distill --steps 200
  python -m memochain.distill --teacher data/stage3 --dataset pv4 --data-dir data --steps 5000
"""

from __future__ import annotations

import argparse
import math
import time
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


def synthetic_batch(
    batch: int, num_docs: int, embed_dim: int, device, seed: int = 0
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Random collections + questions for offline smoke training."""
    rng = np.random.default_rng(seed)
    docs = torch.from_numpy(
        rng.standard_normal((batch, num_docs, embed_dim)).astype(np.float32) * 0.5
    ).to(device)
    questions = torch.from_numpy(
        rng.standard_normal((batch, embed_dim)).astype(np.float32) * 0.5
    ).to(device)
    return docs, questions, docs  # reuse docs as targets source


def token_loss(
    student_tokens: "torch.Tensor", teacher_tokens: "torch.Tensor"
) -> tuple["torch.Tensor", float, float]:
    """Cosine + MSE over [B, M, D]; returns (loss, cosine_mean, mse)."""
    s = student_tokens.float()
    t = teacher_tokens.float()
    cosine = F.cosine_similarity(s, t, dim=-1).mean()
    mse = F.mse_loss(s, t)
    return 1.0 - cosine + 0.1 * mse, float(cosine), float(mse)


class RealDatasetBatches:
    """Rotating batches from one or more Memorilla-Data embedding stores.

    With multiple datasets, each batch draws datasets round-robin so every
    recipe contributes every step (matches memorilla's config concatenation
    and shuffling, but deterministic).

    Layout (memorilla docs/data.md):
      {data_dir}/data/{dataset}/{split}-XXXXX-of-YYYYY.parquet  (collection_id, ...)
      {data_dir}/embeddings/qwen3-embedding-4b/{dataset}/{split}/
        index.safetensors, documents-*.safetensors, questions.safetensors

    questions.safetensors is row-aligned with the parquet shards in sorted
    filename order. Each batch packs rows that may come from different
    collections; document embeddings are padded per batch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset: str | list[str],
        split: str,
        embed_dim: int,
        max_docs: int,
        device,
    ) -> None:
        import pyarrow.parquet as pq

        self.device = device
        self.max_docs = max_docs
        self.rng = np.random.default_rng(0)
        self.datasets = [dataset] if isinstance(dataset, str) else list(dataset)

        from .teacher import collection_embeds, load_embedding_store

        root = Path(data_dir)
        self.row_collection: list[tuple[str, int]] = []  # (dataset, global_row)
        self.rows_per_dataset: dict[str, list[int]] = {}
        self.collection_docs: dict[tuple[str, int], np.ndarray] = {}
        self.stores: dict[str, dict] = {}
        self.embed_dim = embed_dim

        for ds in self.datasets:
            parquet_files = sorted((root / "data" / ds).glob(f"{split}-*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"no {ds}/{split} parquet under {root / 'data' / ds}; "
                    "download Memorilla-Data first (see DESIGN.md)"
                )
            offset = 0
            row_ids: list[int] = []
            for f in parquet_files:
                table = pq.read_table(f, columns=["collection_id"])
                cids = table.column("collection_id").to_pylist()
                for local_row, cid in enumerate(cids):
                    self.row_collection.append((ds, cid))
                    row_ids.append(offset + local_row)
                offset += len(cids)

            store = load_embedding_store(ds, split, root)
            self.stores[ds] = store
            questions = store["questions"]
            if questions is None or len(questions) < offset:
                raise ValueError(
                    f"{ds}: questions.safetensors has {0 if questions is None else len(questions)} rows, "
                    f"expected >= {offset} (parquet rows)"
                )
            self.rows_per_dataset[ds] = row_ids

            for cid in sorted({c for (d, c) in self.row_collection if d == ds}):
                docs = collection_embeds(store, cid)
                if docs.shape[0] > max_docs:
                    keep = self.rng.choice(docs.shape[0], size=max_docs, replace=False)
                    docs = docs[np.sort(keep)]
                self.collection_docs[(ds, cid)] = docs.astype(np.float32)

        self._global_rows = list(range(len(self.row_collection)))
        # interleaved round-robin order over datasets
        self._order: list[int] = []
        cursors = {ds: 0 for ds in self.datasets}
        any_left = True
        while any_left:
            any_left = False
            for ds in self.datasets:
                rows = self.rows_per_dataset[ds]
                if cursors[ds] < len(rows):
                    # map (dataset, local row) -> global row index
                    global_idx = self._dataset_row_offset(ds) + cursors[ds]
                    self._order.append(global_idx)
                    cursors[ds] += 1
                    any_left = True
        self._order = [self._order[i] for i in self.rng.permutation(len(self._order))]
        self._cursor = 0
        self.epoch = 0

    def _dataset_row_offset(self, dataset: str) -> int:
        """First global row index of this dataset (rows are appended per ds)."""
        # row_collection is filled per dataset in order; find the first entry
        for i, (ds, _cid) in enumerate(self.row_collection):
            if ds == dataset:
                # first occurrence marks the block start only if datasets
                # were appended contiguously - they are (loop above).
                # find the contiguous block for this dataset
                start = i
                end = start
                while end < len(self.row_collection) and self.row_collection[end][0] == dataset:
                    end += 1
                if end - start == len(self.rows_per_dataset[dataset]):
                    return start
        raise KeyError(f"dataset {dataset} has no rows")

    def _next_rows(self, batch: int) -> list[int]:
        out = []
        while len(out) < batch:
            if self._cursor >= len(self._order):
                self._order = [
                    self._order[i] for i in self.rng.permutation(len(self._order))
                ]
                self._cursor = 0
                self.epoch += 1
            idx = self._order[self._cursor]
            self._cursor += 1
            out.append(idx)
        return out

    def next_batch(self, batch: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        rows = self._next_rows(batch)
        doc_lists = [self.collection_docs[self.row_collection[r]] for r in rows]
        max_n = max(d.shape[0] for d in doc_lists)
        docs = np.zeros((len(rows), max_n, self.embed_dim), dtype=np.float32)
        padding = np.ones((len(rows), max_n), dtype=bool)
        for i, d in enumerate(doc_lists):
            docs[i, : d.shape[0]] = d
            padding[i, : d.shape[0]] = False
        questions = np.stack(
            [
                np.asarray(
                    self.stores[self.row_collection[r][0]]["questions"][
                        r - self._dataset_row_offset(self.row_collection[r][0])
                    ],
                    dtype=np.float32,
                )
                for r in rows
            ]
        )
        return (
            torch.from_numpy(docs).to(self.device),
            torch.from_numpy(padding).to(self.device),
            torch.from_numpy(questions).to(self.device),
        )


def run(args: argparse.Namespace) -> None:
    _require_torch()
    from .student import MemoChainStudent, StudentConfig
    from .teacher import MemorillaTeacher, SyntheticTeacher

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)

    cfg = StudentConfig()
    student = MemoChainStudent(cfg).to(device)
    student.beta_logits.data.fill_(float(np.log(args.beta / (1 - args.beta))))

    if args.init:
        init_ckpt = torch.load(Path(args.init) / "student.pt", map_location="cpu", weights_only=False)
        student.load_state_dict(init_ckpt["state_dict"])
        print(f"warm-started from {args.init} (train cosine {init_ckpt.get('final_cosine')})")

    batches = None
    if args.dataset:
        if not args.teacher:
            raise SystemExit("--dataset needs a real teacher: pass --teacher data/stage3")
        ds_names = [args.dataset] if isinstance(args.dataset, str) else list(args.dataset)
        print(f"loading {ds_names} embedding stores from {args.data_dir} ...")
        batches = RealDatasetBatches(
            args.data_dir, ds_names, args.split, cfg.embedding_dim, args.num_docs, device
        )
        n_collections = len(batches.collection_docs)
        print(f"loaded {len(batches.row_collection)} rows over {n_collections} collections")

    if args.teacher:
        print(f"loading Memorilla teacher from {args.teacher} ...")
        teacher = MemorillaTeacher.from_pretrained(args.teacher).to(device)
    else:
        print("using synthetic teacher (offline mode)")
        teacher = SyntheticTeacher(
            embedding_dim=cfg.embedding_dim, output_dim=cfg.output_dim,
            num_memories=cfg.num_memories, seed=args.seed,
        )

    student_params = sum(p.numel() for p in student.parameters())
    if hasattr(teacher, "parameters"):
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(f"teacher params: {teacher_params:,}  student params: {student_params:,}")
    else:
        print(f"teacher: synthetic  student params: {student_params:,}")
    print(f"state per collection: {cfg.num_heads * cfg.key_dim * cfg.val_dim * 2:,} bytes")

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        vram_total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"device: {torch.cuda.get_device_name(0)} ({vram_total:.1f} GiB total)")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    # Synthetic mode holds ONE fixed collection set: the synthetic teacher's
    # mixing is a fixed random projection, so rotating collections every step
    # makes the target a moving average that cannot be fit. Real datasets
    # rotate rows every step (RealDatasetBatches) and do converge.
    rng = np.random.default_rng(args.seed)
    fixed_docs = torch.from_numpy(
        rng.standard_normal((args.batch, args.num_docs, cfg.embedding_dim)).astype(np.float32) * 0.5
    ).to(device)
    fixed_questions = torch.from_numpy(
        rng.standard_normal((args.batch, cfg.embedding_dim)).astype(np.float32) * 0.5
    ).to(device)

    warmup = max(1, args.steps // 10)
    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log_every = max(1, args.steps // 20)
    start = time.perf_counter()
    running = defaultdict(list)
    for step in range(args.steps):
        if batches is not None:
            docs, padding, questions = batches.next_batch(args.batch)
        else:
            docs, questions = fixed_docs, fixed_questions
            padding = None

        with torch.no_grad():
            if hasattr(teacher, "cache_keys_values"):
                cache = teacher.cache_keys_values(docs, padding)
                targets = teacher.read_cache(cache, padding, questions)
            else:
                targets = teacher(docs, padding, questions)

        state = student.fold_sequence(docs, padding, quantize=args.quantize)
        tokens = student.read_tokens(state, questions)
        loss, cosine, mse = token_loss(tokens, targets)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        running["loss"].append(loss.item())
        running["cos"].append(cosine)
        if step % log_every == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - start
            vram = ""
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated() / 2**20
                vram = f"  peak VRAM {peak:.0f} MiB"
            print(
                f"step {step:>6}  loss {loss.item():.4f}  cosine {cosine:.4f}  "
                f"mse {mse:.5f}  ({elapsed:.1f}s){vram}",
                flush=True,
            )

    print(f"training complete: final cosine {running['cos'][-1]:.4f}, "
          f"mean last-100 {np.mean(running['cos'][-100:]):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "student.pt"
    torch.save(
        {
            "config": vars(cfg),
            "state_dict": student.state_dict(),
            "args": vars(args),
            "final_cosine": running["cos"][-1],
        },
        ckpt_path,
    )
    print(f"checkpoint saved: {ckpt_path}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"final peak VRAM: {peak:.0f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoChain distillation trainer")
    parser.add_argument("--teacher", default="", help="path to stage3 dir (memory.pt + config.json); empty = synthetic")
    parser.add_argument("--dataset", default="", nargs="+", help="real dataset config(s) (e.g. pv4 or pv4 quail drop); empty = synthetic mode")
    parser.add_argument("--data-dir", default="data", help="Memorilla-Data local mirror root")
    parser.add_argument("--split", default="train", help="dataset split to train on")
    parser.add_argument("--out", default="runs/distill", help="checkpoint output dir")
    parser.add_argument("--init", default="", help="warm-start from a checkpoint dir (student.pt)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-docs", type=int, default=32, help="max docs per collection in a batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="initial write strength (0..1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA exists")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false", help="disable Q14 STE quantization")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
