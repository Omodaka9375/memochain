"""Exact Q14 fixed-point writer for the distilled MemoChainStudent.

Converts a trained MemoChainStudent's write path (key_proj, val_proj,
per-head beta[11], per-head decay[11]) into pure int16/int64 arithmetic so
the exact 22,528-byte capsule read by Qwen3-8B is bit-identical in Python,
native Rust, and the SP1 zkVM guest.

Binary wire formats (all little-endian, matching crates/memochain-core/src/student.rs):
  weights.bin (3,605,932 bytes):
    w_key [352, 2560] i16 | b_key [352] i16 | w_val [352, 2560] i16 |
    b_val [352] i16 | beta [11] i16 | decay [11] i16
    weights_digest = sha256(weights.bin).hexdigest()
  docs.bin (n_docs * 5,120 bytes):
    n_docs rows of 2560 i16 values (Q14 quantized Qwen3-Embedding-4B vectors)
  proj.bin (optional, n_docs * 5,632 bytes):
    dot_k [n_docs, 352] i64 | dot_v [n_docs, 352] i64 (exact pre-shift products
    for Fiat-Shamir Freivalds verification inside the zkVM when n_docs > 4)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .state import Capsule, KEY_DIM, N_HEADS, STATE_BYTES, VAL_DIM, MemoryState
from .verify import ChainHead, Receipt

EMBED_DIM = 2560
KEY_PART = N_HEADS * KEY_DIM  # 352
Q_SHIFT = 14
Q_ONE = 1 << Q_SHIFT  # 16384
I16_MIN = -32768
I16_MAX = 32767

WEIGHTS_BYTES = (
    KEY_PART * EMBED_DIM * 2  # w_key
    + KEY_PART * 2            # b_key
    + KEY_PART * EMBED_DIM * 2  # w_val
    + KEY_PART * 2            # b_val
    + N_HEADS * 2             # beta
    + N_HEADS * 2             # decay
)  # 3,605,932 bytes

GENESIS_ACCUMULATOR = "0" * 64


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantize_embeddings_q14(docs: np.ndarray) -> np.ndarray:
    """Quantize float32/float16 [N, 2560] embeddings to Q14 int16."""
    arr = np.asarray(docs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != EMBED_DIM:
        raise ValueError(f"expected embedding dim {EMBED_DIM}, got {arr.shape[1]}")
    return np.clip(np.round(arr * Q_ONE), I16_MIN, I16_MAX).astype("<i2")


def doc_hash_2560(doc_q14: np.ndarray) -> str:
    """SHA-256 hex of one Q14 int16 document vector (little-endian bytes)."""
    arr = np.asarray(doc_q14, dtype="<i2").reshape(-1)
    if arr.size != EMBED_DIM:
        raise ValueError(f"expected {EMBED_DIM} i16 values, got {arr.size}")
    return _sha256_hex(arr.tobytes())


def state_digest_i16(state: np.ndarray) -> str:
    """SHA-256 hex of the 22,528-byte int16 state (little-endian bytes)."""
    arr = np.asarray(state, dtype="<i2").reshape(N_HEADS, KEY_DIM, VAL_DIM)
    return _sha256_hex(arr.tobytes())


def unit_key_i64(k: np.ndarray) -> np.ndarray:
    """Scale a 32-d integer vector to Q14 unit norm using integer math only."""
    vals = [int(x) for x in k]
    norm = math.isqrt(sum(x * x for x in vals))
    if norm == 0:
        return np.zeros(len(vals), dtype=np.int64)
    out = np.empty(len(vals), dtype=np.int64)
    for i, x in enumerate(vals):
        q = (x * Q_ONE) // norm  # floor division, matches Rust i64::div_euclid
        out[i] = max(I16_MIN, min(I16_MAX, q))
    return out


@dataclass
class IntStudentWriter:
    """Fixed-point Q14 writer extracted from a trained MemoChainStudent."""

    w_key: np.ndarray  # [352, 2560] <i2
    b_key: np.ndarray  # [352] <i2
    w_val: np.ndarray  # [352, 2560] <i2
    b_val: np.ndarray  # [352] <i2
    beta: np.ndarray   # [11] <i2
    decay: np.ndarray  # [11] <i2

    def __post_init__(self) -> None:
        self.w_key = np.ascontiguousarray(self.w_key, dtype="<i2")
        self.b_key = np.ascontiguousarray(self.b_key, dtype="<i2")
        self.w_val = np.ascontiguousarray(self.w_val, dtype="<i2")
        self.b_val = np.ascontiguousarray(self.b_val, dtype="<i2")
        self.beta = np.ascontiguousarray(self.beta, dtype="<i2")
        self.decay = np.ascontiguousarray(self.decay, dtype="<i2")
        if self.w_key.shape != (KEY_PART, EMBED_DIM):
            raise ValueError(f"w_key shape {self.w_key.shape} != {(KEY_PART, EMBED_DIM)}")
        if self.b_key.shape != (KEY_PART,):
            raise ValueError(f"b_key shape {self.b_key.shape} != {(KEY_PART,)}")
        if self.w_val.shape != (KEY_PART, EMBED_DIM):
            raise ValueError(f"w_val shape {self.w_val.shape} != {(KEY_PART, EMBED_DIM)}")
        if self.b_val.shape != (KEY_PART,):
            raise ValueError(f"b_val shape {self.b_val.shape} != {(KEY_PART,)}")
        if self.beta.shape != (N_HEADS,) or self.decay.shape != (N_HEADS,):
            raise ValueError("beta and decay must have shape (11,)")

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> IntStudentWriter:
        """Quantize a trained MemoChainStudent checkpoint (student.pt) to Q14."""
        import torch

        path = Path(checkpoint_dir)
        ckpt_path = path / "student.pt" if path.is_dir() else path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]

        def q14(tensor: torch.Tensor, lo: int = I16_MIN, hi: int = I16_MAX) -> np.ndarray:
            arr = tensor.detach().cpu().float().numpy()
            return np.clip(np.round(arr * Q_ONE), lo, hi).astype("<i2")

        return cls(
            w_key=q14(sd["key_proj.weight"]),
            b_key=q14(sd["key_proj.bias"]),
            w_val=q14(sd["val_proj.weight"]),
            b_val=q14(sd["val_proj.bias"]),
            beta=q14(torch.sigmoid(sd["beta_logits"]), 0, Q_ONE),
            decay=q14(torch.sigmoid(sd["decay_logits"]), 0, Q_ONE),
        )

    def to_bytes(self) -> bytes:
        """Pack weights into the canonical little-endian binary blob (3,605,932 B)."""
        raw = b"".join(
            [
                self.w_key.tobytes(),
                self.b_key.tobytes(),
                self.w_val.tobytes(),
                self.b_val.tobytes(),
                self.beta.tobytes(),
                self.decay.tobytes(),
            ]
        )
        assert len(raw) == WEIGHTS_BYTES
        return raw

    @classmethod
    def from_bytes(cls, raw: bytes) -> IntStudentWriter:
        """Unpack canonical little-endian binary weights."""
        if len(raw) != WEIGHTS_BYTES:
            raise ValueError(f"weights length {len(raw)} != {WEIGHTS_BYTES}")
        pos = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal pos
            count = int(np.prod(shape))
            end = pos + count * 2
            arr = np.frombuffer(raw[pos:end], dtype="<i2").reshape(shape).copy()
            pos = end
            return arr

        w_key = take((KEY_PART, EMBED_DIM))
        b_key = take((KEY_PART,))
        w_val = take((KEY_PART, EMBED_DIM))
        b_val = take((KEY_PART,))
        beta = take((N_HEADS,))
        decay = take((N_HEADS,))
        return cls(w_key=w_key, b_key=b_key, w_val=w_val, b_val=b_val, beta=beta, decay=decay)

    def weights_digest(self) -> str:
        """SHA-256 hex commitment over the exact quantized writer weights."""
        return _sha256_hex(self.to_bytes())

    def exact_projections(self, docs_q14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact int64 pre-shift products (dot_k, dot_v) for [N, 2560] i16 docs."""
        x = np.asarray(docs_q14, dtype=np.int64)
        if x.ndim == 1:
            x = x[None, :]
        dot_k = x @ self.w_key.astype(np.int64).T
        dot_v = x @ self.w_val.astype(np.int64).T
        return np.ascontiguousarray(dot_k, dtype="<i8"), np.ascontiguousarray(dot_v, dtype="<i8")

    def fold_from_dots(
        self, state: np.ndarray, dot_k: np.ndarray, dot_v: np.ndarray
    ) -> np.ndarray:
        """Advance state by one doc given exact pre-shift dot products [352], [352]."""
        raw_k = ((dot_k.astype(np.int64) >> Q_SHIFT) + self.b_key.astype(np.int64)).reshape(
            N_HEADS, KEY_DIM
        )
        raw_v = np.clip(
            (dot_v.astype(np.int64) >> Q_SHIFT) + self.b_val.astype(np.int64),
            I16_MIN,
            I16_MAX,
        ).reshape(N_HEADS, VAL_DIM)

        out = np.asarray(state, dtype=np.int64).reshape(N_HEADS, KEY_DIM, VAL_DIM).copy()
        for h in range(N_HEADS):
            uk = unit_key_i64(raw_k[h])
            if not np.any(uk):
                continue
            recalled = (uk @ out[h]) >> Q_SHIFT
            delta = (int(self.beta[h]) * (raw_v[h] - recalled)) >> Q_SHIFT
            decayed = (int(self.decay[h]) * out[h]) >> Q_SHIFT
            update = np.outer(uk, delta) >> Q_SHIFT
            out[h] = np.clip(decayed + update, I16_MIN, I16_MAX)
        return out.astype("<i2")

    def fold_one(self, state: np.ndarray, doc_q14: np.ndarray) -> np.ndarray:
        """Fold one Q14 int16 [2560] document into state [11, 32, 32] int16."""
        dot_k, dot_v = self.exact_projections(doc_q14)
        return self.fold_from_dots(state, dot_k[0], dot_v[0])

    def write_with_receipt(
        self,
        state: np.ndarray,
        doc_q14: np.ndarray,
        prev: ChainHead | None = None,
        dot_k: np.ndarray | None = None,
        dot_v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Receipt, ChainHead]:
        """Fold one Q14 document and advance the SHA-256 receipt chain."""
        step = 1 if prev is None else prev.step + 1
        prev_acc = GENESIS_ACCUMULATOR if prev is None else prev.accumulator
        d_hash = doc_hash_2560(doc_q14)
        if dot_k is None or dot_v is None:
            new_state = self.fold_one(state, doc_q14)
        else:
            new_state = self.fold_from_dots(state, dot_k, dot_v)
        new_digest = state_digest_i16(new_state)

        binding = _sha256_hex(f"{step}|{prev_acc}|{d_hash}|{new_digest}".encode("ascii"))
        accumulator = _sha256_hex(f"{prev_acc}{binding}".encode("ascii"))
        receipt = Receipt(
            step=step,
            prev_head=prev_acc,
            doc_digest=d_hash,
            new_head=new_digest,
            digest=binding,
        )
        head = ChainHead(step=step, accumulator=accumulator, state_digest=new_digest)
        return new_state, receipt, head

    def fold_sequence(
        self,
        docs_q14: np.ndarray,
        initial_state: np.ndarray | None = None,
        initial_head: ChainHead | None = None,
    ) -> tuple[MemoryState, ChainHead]:
        """Fold a sequence of Q14 [N, 2560] docs, returning (MemoryState, ChainHead)."""
        docs = np.asarray(docs_q14, dtype="<i2")
        if docs.ndim == 1:
            docs = docs[None, :]
        state = (
            np.zeros((N_HEADS, KEY_DIM, VAL_DIM), dtype="<i2")
            if initial_state is None
            else np.asarray(initial_state, dtype="<i2").reshape(N_HEADS, KEY_DIM, VAL_DIM).copy()
        )
        head = initial_head
        dot_k, dot_v = self.exact_projections(docs)
        for i in range(docs.shape[0]):
            state, _r, head = self.write_with_receipt(
                state, docs[i], prev=head, dot_k=dot_k[i], dot_v=dot_v[i]
            )
        assert head is not None
        mem_state = MemoryState(state)
        assert mem_state.nbytes == STATE_BYTES
        return mem_state, head

    def export_proven_capsule(self, docs_q14: np.ndarray) -> tuple[Capsule, ChainHead]:
        """Fold docs into a Capsule whose head is the exact receipt-chain accumulator."""
        mem_state, head = self.fold_sequence(docs_q14)
        return Capsule(state=mem_state, step=head.step, head=head.accumulator), head


def pack_docs_bin(docs_q14: np.ndarray) -> bytes:
    """Serialize [N, 2560] Q14 int16 docs to little-endian raw bytes."""
    arr = np.ascontiguousarray(docs_q14, dtype="<i2")
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise ValueError(f"expected [N, {EMBED_DIM}], got {arr.shape}")
    return arr.tobytes()


def pack_projections_bin(dot_k: np.ndarray, dot_v: np.ndarray) -> bytes:
    """Serialize exact int64 [N, 352] pre-shift projections for Freivalds zkVM checks."""
    dk = np.ascontiguousarray(dot_k, dtype="<i8")
    dv = np.ascontiguousarray(dot_v, dtype="<i8")
    if dk.shape != dv.shape or dk.ndim != 2 or dk.shape[1] != KEY_PART:
        raise ValueError(f"bad projection shapes: {dk.shape}, {dv.shape}")
    return dk.tobytes() + dv.tobytes()


def main() -> None:
    """CLI: export quantized writer weights + user docs for the SP1 IVC prover."""
    import argparse
    import json

    from .teacher import collection_embeds, load_embedding_store

    parser = argparse.ArgumentParser(description="Export Q14 student writer + docs for SP1 IVC")
    parser.add_argument("--checkpoint", required=True, help="dir with student.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="pv4")
    parser.add_argument("--split", default="test")
    parser.add_argument("--collection", type=int, required=True, help="collection_id")
    parser.add_argument(
        "--split-at",
        type=int,
        default=199,
        help="Split docs into base [0:split_at] and recursive step [split_at:]",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    writer = IntStudentWriter.from_checkpoint(args.checkpoint)
    weights_bytes = writer.to_bytes()
    (out / "weights.bin").write_bytes(weights_bytes)

    store = load_embedding_store(args.dataset, args.split, args.data_dir)
    docs_f = collection_embeds(store, args.collection).astype(np.float32)
    docs_q = quantize_embeddings_q14(docs_f)

    n = docs_q.shape[0]
    if not (1 <= args.split_at < n):
        raise SystemExit(f"split-at={args.split_at} must be in [1, {n - 1}]")

    docs_base = docs_q[: args.split_at]
    docs_step = docs_q[args.split_at :]
    (out / "docs_base.bin").write_bytes(pack_docs_bin(docs_base))
    (out / "docs_step.bin").write_bytes(pack_docs_bin(docs_step))

    state_base, head_base = writer.fold_sequence(docs_base)
    state_final, head_final = writer.fold_sequence(
        docs_step, initial_state=state_base.tokens, initial_head=head_base
    )

    # Save Python reference capsule for bit-exact comparison against SP1 output.
    cap_ref = Capsule(state=state_final, step=head_final.step, head=head_final.accumulator)
    (out / "python_ref.capsule").write_bytes(cap_ref.to_bytes())

    expected = {
        "dataset": args.dataset,
        "split": args.split,
        "collection_id": args.collection,
        "total_docs": int(n),
        "split_at": int(args.split_at),
        "step_docs": int(docs_step.shape[0]),
        "weights_bytes": len(weights_bytes),
        "weights_digest": writer.weights_digest(),
        "base": {
            "step": head_base.step,
            "state_digest": head_base.state_digest,
            "accumulator": head_base.accumulator,
        },
        "final": {
            "step": head_final.step,
            "state_digest": head_final.state_digest,
            "accumulator": head_final.accumulator,
        },
    }
    (out / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()

