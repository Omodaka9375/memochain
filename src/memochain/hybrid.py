"""Bounded hot-tail hybrid: exact recent recall over an O(1) memory.

Keeps the last W document embeddings verbatim in a fixed ring buffer (the
"hot tail") and folds older documents into the 22 KB Delta-rule cold state.
Total memory stays constant forever: W raw embeddings + 22,528 bytes.
Reading uses both paths: exact recent matches from the tail, associative
recall from the cold state.

This is the Mina full-node/light-client split applied to memory: recent
blocks are kept raw, old history is compressed into the verified constant
state. The receipt chain (ChainHead) tracks the cold state only, so the
light-client property is preserved.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .fold import DOC_DIM, Folder, read
from .state import Capsule, MemoryState
from .verify import ChainHead, write_with_receipt


@dataclass
class HybridRead:
    """Result of a hybrid read."""

    cold: np.ndarray      # associative recall from the 22 KB state
    hot: list[np.ndarray] # verbatim recent documents matching the query
    step: int             # number of documents folded into the cold state


class HybridMemory:
    """W-slot hot ring buffer in front of the 22 KB cold state."""

    def __init__(
        self,
        folder: Folder,
        hot_size: int = 8,
        state: MemoryState | None = None,
        head: ChainHead | None = None,
    ) -> None:
        if hot_size < 0:
            raise ValueError("hot_size must be >= 0")
        self.folder = folder
        self.hot_size = hot_size
        self.hot: deque[np.ndarray] = deque(maxlen=hot_size)
        self.cold = state if state is not None else MemoryState.zero()
        self.head = head

    def write(self, doc: np.ndarray) -> None:
        """Add one document: evicted hot docs fold into the cold state."""
        doc = np.asarray(doc, dtype=np.int16).reshape(-1)
        if doc.shape != (DOC_DIM,):
            raise ValueError(f"expected doc of shape ({DOC_DIM},), got {doc.shape}")
        if len(self.hot) == self.hot.maxlen and self.hot_size > 0:
            evicted = self.hot.popleft()
            self.cold, _receipt, self.head = write_with_receipt(
                self.folder, self.cold, evicted, self.head
            )
        if self.hot_size > 0:
            self.hot.append(doc)
        else:
            self.cold, _receipt, self.head = write_with_receipt(
                self.folder, self.cold, doc, self.head
            )

    def read(self, query: np.ndarray) -> HybridRead:
        """Recall from both paths. Hot hits are exact; cold hits are folded."""
        q = np.asarray(query, dtype=np.int16).reshape(-1)
        if q.shape != (DOC_DIM,):
            raise ValueError(f"expected query of shape ({DOC_DIM},), got {q.shape}")
        cold_vec = read(self.folder, self.cold, q)
        hot_hits = [doc for doc in self.hot if int(np.dot(doc.astype(np.int64), q)) != 0]
        return HybridRead(cold=cold_vec, hot=hot_hits, step=self.head.step if self.head else 0)

    @property
    def total_bytes(self) -> int:
        return self.cold.nbytes + len(self.hot) * DOC_DIM * 2

    def to_capsule(self) -> Capsule:
        """Serialize to a v2 capsule: cold state + verbatim hot tail."""
        from .verify import ChainHead

        return Capsule(
            state=self.cold,
            step=self.head.step if self.head else 0,
            head=self.head.accumulator if self.head else "",
            hot=tuple(self.hot),
        )

    @classmethod
    def from_capsule(cls, folder: Folder, capsule: Capsule) -> HybridMemory:
        """Restore a hybrid memory from a capsule (hot_size = stored count)."""
        from .verify import ChainHead

        head = None
        if capsule.step > 0:
            head = ChainHead(
                step=capsule.step,
                accumulator=capsule.head or "0" * 64,
                state_digest=capsule.state.digest(),
            )
        mem = cls(folder, hot_size=len(capsule.hot), state=capsule.state, head=head)
        mem.hot.extend(capsule.hot)
        return mem
