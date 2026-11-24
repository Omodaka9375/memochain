"""Receipt chain: tamper detection for the memory state.

Each write emits a receipt binding (previous head, document hash, new head)
into a rolling accumulator, so the verifier state stays O(1): it needs only
the latest 22 KB state, the step count, and one 256-bit accumulator hash.
Full audits can still replay every document; light verification cannot,
which is the Mina light-client property.

All digests are full 256-bit SHA-256. Documents are hashed as their raw
int16 bytes, so no float non-determinism can enter the chain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .fold import Folder
from .state import MemoryState


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _doc_hash(doc: np.ndarray) -> str:
    return _sha(np.asarray(doc, dtype=np.int16).tobytes())


@dataclass
class Receipt:
    step: int
    prev_head: str
    doc_digest: str
    new_head: str
    digest: str  # hash binding the whole transition


@dataclass
class ChainHead:
    """The O(1) verification artifact: what a light client holds."""

    step: int
    accumulator: str  # rolling hash over every receipt so far
    state_digest: str

    def to_dict(self) -> dict[str, int | str]:
        return {"step": self.step, "accumulator": self.accumulator, "state": self.state_digest}


def write_with_receipt(
    folder: Folder, state: MemoryState, doc: np.ndarray, prev: ChainHead | None = None
) -> tuple[MemoryState, Receipt, ChainHead]:
    """Fold one document and emit a receipt, advancing the O(1) chain head."""
    step = (prev.step if prev else 0) + 1
    prev_head = prev.accumulator if prev else "0" * 64
    doc_digest = _doc_hash(doc)
    new_state = folder.fold(state, doc)
    new_head_state = new_state.digest()

    binding = _sha(
        f"{step}|{prev_head}|{doc_digest}|{new_head_state}".encode()
    )
    accumulator = _sha((prev_head + binding).encode())

    receipt = Receipt(
        step=step,
        prev_head=prev_head,
        doc_digest=doc_digest,
        new_head=new_head_state,
        digest=binding,
    )
    head = ChainHead(step=step, accumulator=accumulator, state_digest=new_head_state)
    return new_state, receipt, head


def verify_head(state: MemoryState, head: ChainHead) -> bool:
    """Light check: does this stored state match the chain head's digest?"""
    return state.digest() == head.state_digest


def verify_chain(
    folder: Folder, docs: list[np.ndarray], head: ChainHead
) -> bool:
    """Full audit: replay every document and confirm the final chain head.

    A single flipped bit anywhere in the state, or any inserted, dropped,
    duplicated, or reordered document, changes the head and fails here.
    """
    state = MemoryState.zero()
    running: ChainHead | None = None
    for doc in docs:
        state, _receipt, running = write_with_receipt(folder, state, doc, running)
    if running is None or running.step != head.step:
        return False
    return running.accumulator == head.accumulator and running.state_digest == head.state_digest
