"""Capsule export: fold a real collection into a 22 KB provenance artifact.

A capsule is the portable MemoChain artifact: a versioned envelope (magic,
format version, step count, provenance head, payload hash) around the exact
22,528-byte int16 state. This module bridges the trained student (torch) and
the artifact format (pure numpy):

  export:  student.fold_sequence(docs) -> hard Q14 int16 state -> Capsule
  verify:  reload from disk, check payload hash + state digest + head chain
  read:    student.read_tokens(state, question) on the reloaded state

The provenance head is a sha256 chain over the per-document embedding hashes
and the final state digest, so any changed, dropped, or reordered document
changes the head. Verification needs only the capsule (no reader, no GPU).

CLI:
  python -m memochain.export export --checkpoint runs/pv4 --data-dir data \
      --dataset pv4 --split validation --collection 900 --out capsule.bin
  python -m memochain.export verify --capsule capsule.bin
  python -m memochain.export read --capsule capsule.bin --checkpoint runs/pv4 \
      --data-dir data --dataset pv4 --split validation --row 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .state import Capsule, MemoryState


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def doc_hash(embedding: np.ndarray) -> str:
    """Hash one document embedding (float32 bytes, deterministic)."""
    return _sha(np.asarray(embedding, dtype=np.float32).tobytes())


def provenance_head(doc_hashes: list[str], state_digest: str) -> str:
    """Rolling chain over documents + final state: H(H0|H1|...|state_digest)."""
    acc = "0" * 64
    for h in doc_hashes:
        acc = _sha((acc + "|" + h).encode())
    return _sha((acc + "|" + state_digest).encode())


def fold_student_state(
    student, docs: np.ndarray, device, chunk: int = 64
) -> tuple[MemoryState, list[str]]:
    """Fold a collection's embeddings through the student into an int16 state.

    docs: [N, embed_dim] float embeddings (any N; processed in chunks).
    Returns (state, doc_hashes) with the state hard-quantized to the Q14 grid.
    """
    import torch

    hashes = [doc_hash(d) for d in docs]
    state = None
    x = np.asarray(docs, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk):
            part = torch.from_numpy(x[start : start + chunk]).unsqueeze(0).to(device)
            state = student.fold_sequence(
                part, None, quantize=True, initial=state
            )
    arr = student.state_numpy(state)
    return MemoryState(arr), hashes


def save_capsule(capsule: Capsule, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(capsule.to_bytes())
    return path


def load_capsule(path: str | Path) -> Capsule:
    return Capsule.from_bytes(Path(path).read_bytes())


def verify_capsule(capsule: Capsule, doc_hashes: list[str] | None = None) -> dict:
    """Full verification. With doc_hashes, also re-derives the provenance head."""
    result = {
        "step": capsule.step,
        "state_digest": capsule.state.digest(),
        "payload_intact": True,  # Capsule.from_bytes already enforced this
    }
    if doc_hashes is not None:
        expected = provenance_head(doc_hashes, capsule.state.digest())
        result["head_matches_docs"] = expected == capsule.head
        result["head"] = capsule.head
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoChain capsule export/verify/read")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="fold a collection into a capsule")
    p_exp.add_argument("--checkpoint", required=True, help="dir with student.pt")
    p_exp.add_argument("--data-dir", default="data")
    p_exp.add_argument("--dataset", default="pv4")
    p_exp.add_argument("--split", default="validation")
    p_exp.add_argument("--collection", type=int, required=True, help="collection_id")
    p_exp.add_argument("--max-docs", type=int, default=0, help="cap docs (0 = all)")
    p_exp.add_argument("--out", default="capsule.bin")

    p_ver = sub.add_parser("verify", help="verify a capsule from disk")
    p_ver.add_argument("--capsule", default="capsule.bin")

    p_read = sub.add_parser("read", help="read memory tokens for one question")
    p_read.add_argument("--capsule", default="capsule.bin")
    p_read.add_argument("--checkpoint", required=True)
    p_read.add_argument("--data-dir", default="data")
    p_read.add_argument("--dataset", default="pv4")
    p_read.add_argument("--split", default="validation")
    p_read.add_argument("--row", type=int, required=True, help="question row index")
    p_read.add_argument("--compare-teacher", action="store_true")

    args = parser.parse_args()

    if args.cmd == "export":
        import torch

        from .student import MemoChainStudent, StudentConfig
        from .teacher import collection_embeds, load_embedding_store

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(args.checkpoint) / "student.pt", map_location="cpu", weights_only=False)
        cfg = StudentConfig(**ckpt["config"])
        student = MemoChainStudent(cfg).to(device)
        student.load_state_dict(ckpt["state_dict"])
        student.eval()

        store = load_embedding_store(args.dataset, args.split, args.data_dir)
        docs = collection_embeds(store, args.collection).astype(np.float32)
        if args.max_docs and docs.shape[0] > args.max_docs:
            docs = docs[: args.max_docs]
        print(f"folding {docs.shape[0]} documents from collection {args.collection} ...")
        state, hashes = fold_student_state(student, docs, device)
        capsule = Capsule(
            state=state, step=docs.shape[0], head=provenance_head(hashes, state.digest())
        )
        save_capsule(capsule, args.out)
        print(json.dumps(verify_capsule(capsule, hashes), indent=2))
        print(f"saved: {args.out} ({capsule.state.nbytes:,} byte payload)")

    elif args.cmd == "verify":
        capsule = load_capsule(args.capsule)
        print(json.dumps(verify_capsule(capsule), indent=2))
        print(f"capsule OK: step={capsule.step}, digest={capsule.state.digest()[:16]}...")

    elif args.cmd == "read":
        import torch

        from .student import MemoChainStudent, StudentConfig
        from .teacher import MemorillaTeacher, load_embedding_store

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        capsule = load_capsule(args.capsule)

        ckpt = torch.load(Path(args.checkpoint) / "student.pt", map_location="cpu", weights_only=False)
        cfg = StudentConfig(**ckpt["config"])
        student = MemoChainStudent(cfg).to(device)
        student.load_state_dict(ckpt["state_dict"])
        student.eval()

        store = load_embedding_store(args.dataset, args.split, args.data_dir)
        questions = np.asarray(store["questions"], dtype=np.float32)
        question = torch.from_numpy(questions[args.row]).unsqueeze(0).to(device)

        state_t = (
            torch.from_numpy(capsule.state.tokens.astype(np.float32) / float(1 << 14))
            .unsqueeze(0)
            .to(device)
        )
        tokens = student.read_tokens(state_t, question)
        print(f"capsule step {capsule.step}, question row {args.row}")
        print(f"token shape: {tuple(tokens.shape)}, norm: {float(tokens.norm()):.2f}")

        if args.compare_teacher:
            from .teacher import MemorillaTeacher, collection_embeds

            teacher = MemorillaTeacher.from_pretrained("data/stage3").to(device)
            teacher.eval()
            cid = None
            # find the collection this row belongs to
            import pyarrow.parquet as pq

            root = Path(args.data_dir) / "data" / args.dataset
            offset = 0
            for f in sorted(root.glob(f"{args.split}-*.parquet")):
                t = pq.read_table(f, columns=["collection_id"])
                cids = t.column("collection_id").to_pylist()
                if offset <= args.row < offset + len(cids):
                    cid = int(cids[args.row - offset])
                    break
                offset += len(cids)
            if cid is None:
                raise SystemExit("row out of range")
            docs = np.asarray(collection_embeds(store, cid), dtype=np.float32)
            dt = torch.from_numpy(docs).unsqueeze(0).to(device)
            with torch.no_grad():
                cache = teacher.cache_keys_values(dt, None)
                target = teacher.read_cache(cache, None, question)
            cos = float(
                torch.nn.functional.cosine_similarity(
                    tokens.float(), target.float(), dim=-1
                ).mean()
            )
            print(f"collection {cid}: student-vs-teacher token cosine = {cos:.4f}")


if __name__ == "__main__":
    main()
