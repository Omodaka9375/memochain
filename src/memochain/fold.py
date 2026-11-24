"""The O(1) fold: write one document per step, constant cost forever.

The state is an associative fast-weight memory: 11 independent heads, each a
32 x 32 int16 matrix (the whole state is exactly 22,528 bytes).

Document layout (704 int16 values):
  doc[0:352]    key part    (11 heads x 32: what addresses this memory)
  doc[352:704]  value part  (11 heads x 32: what gets stored)

Head h binds key_part[h*32:(h+1)*32] to value_part[h*32:(h+1)*32] through
learnable projections W_key[h] and W_val[h]. A write uses the Delta rule: it
recalls through the incoming key, subtracts, and writes the correction, so
re-writing a key with new content replaces the old value instead of blending
("moved from Paris to Berlin" just updates). A query carries only the key
part (value part zeros) and reads back the bound values.

All state-transition arithmetic is pure integer (multiply, add, shift, plus
an integer square root for key normalization). The same inputs produce
bit-identical states on any machine, and the fold maps directly onto ZK
circuits (SP1 / Nova) later.
"""

from __future__ import annotations

import numpy as np

from .state import KEY_DIM, N_HEADS, VAL_DIM, MemoryState

KEY_PART = N_HEADS * KEY_DIM  # 352: first half of a doc
VAL_PART = N_HEADS * VAL_DIM  # 352: second half of a doc
DOC_DIM = KEY_PART + VAL_PART  # 704

# Q14 fixed point for keys, beta, and decay.
Q_SHIFT = 14
Q_ONE = 1 << Q_SHIFT  # 16384
INT16_MAX = 32767
INT16_MIN = -32768


def _rng_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((rows, cols)) * 11.3  # ~N(0, 11^2) entries
    return np.round(m).astype(np.int16)


def _clip16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, INT16_MIN, INT16_MAX).astype(np.int16)


def _isqrt(n: int) -> int:
    """Deterministic integer square root (no float rounding anywhere)."""
    if n <= 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def _unit_key(k: np.ndarray) -> np.ndarray:
    """Scale a key to Q14 unit norm using integer math only."""
    norm = _isqrt(int(np.dot(k.astype(np.int64), k.astype(np.int64))))
    if norm == 0:
        return np.zeros_like(k)
    scaled = (k.astype(np.int64) * Q_ONE) // norm
    return _clip16(scaled)


def make_doc(key_part: np.ndarray, value_part: np.ndarray) -> np.ndarray:
    """Assemble a doc from its key part (176) and value part (176)."""
    k = np.asarray(key_part, dtype=np.int16).reshape(-1)
    v = np.asarray(value_part, dtype=np.int16).reshape(-1)
    if k.shape != (KEY_PART,) or v.shape != (VAL_PART,):
        raise ValueError(f"key part must be {KEY_PART}, value part {VAL_PART}")
    return np.concatenate([k, v]).astype(np.int16)


def query_doc(key_part: np.ndarray) -> np.ndarray:
    """Build a query (key part only, zero value part)."""
    return make_doc(key_part, np.zeros(VAL_PART, dtype=np.int16))


class Folder:
    """Folds document embeddings into the fixed state, one per step."""

    def __init__(self, seed: int = 0, beta: float = 0.75, decay: float = 0.999) -> None:
        self.w_key = np.stack([_rng_matrix(KEY_DIM, KEY_DIM, seed + h) for h in range(N_HEADS)])
        self.w_val = np.stack(
            [_rng_matrix(VAL_DIM, KEY_DIM, seed + 100 + h) for h in range(N_HEADS)]
        )
        self.beta = int(beta * Q_ONE)    # Q14 write strength
        self.decay = int(decay * Q_ONE)  # Q14 per-write decay

    def project(self, doc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project a doc into per-head unit keys (Q14) and bounded values.

        Returns (keys, values), each shape (N_HEADS, 32).
        """
        x = np.asarray(doc, dtype=np.int64).reshape(-1)
        if x.shape != (DOC_DIM,):
            raise ValueError(f"expected doc of shape ({DOC_DIM},), got {x.shape}")
        keys = np.zeros((N_HEADS, KEY_DIM), dtype=np.int64)
        vals = np.zeros((N_HEADS, VAL_DIM), dtype=np.int64)
        for h in range(N_HEADS):
            k_slice = x[h * KEY_DIM : (h + 1) * KEY_DIM]
            v_slice = x[KEY_PART + h * VAL_DIM : KEY_PART + (h + 1) * VAL_DIM]
            keys[h] = (self.w_key[h].astype(np.int64) @ k_slice) >> 4
            vals[h] = (self.w_val[h].astype(np.int64) @ v_slice) >> 8
        keys = np.stack([_unit_key(keys[h]) for h in range(N_HEADS)])
        return keys, _clip16(vals)

    def fold(self, state: MemoryState, doc: np.ndarray) -> MemoryState:
        """Return the new state after absorbing one document.

        doc: int16 vector, length 704. Cost: 11 small integer matvecs plus
        up to 11 rank-1 updates, O(1) in history size.
        """
        keys, vals = self.project(doc)
        tokens = state.tokens.astype(np.int64).copy()
        for h in range(N_HEADS):
            k = keys[h]
            if not k.any():
                continue  # zero key: nothing addressable, leave the head alone
            s = tokens[h]                                  # (KEY_DIM, VAL_DIM)
            recalled = (k @ s) >> Q_SHIFT                  # (VAL_DIM,)
            d = (self.beta * (vals[h] - recalled)) >> Q_SHIFT
            update = (k[:, None] * d[None, :]) >> Q_SHIFT  # (KEY, VAL)
            decayed = (self.decay * s) >> Q_SHIFT
            tokens[h] = decayed + update
        return MemoryState(_clip16(tokens))

    def fold_all(self, docs: np.ndarray) -> MemoryState:
        """Fold a whole collection, document by document."""
        state = MemoryState.zero()
        for doc in np.asarray(docs, dtype=np.int16):
            state = self.fold(state, doc)
        return state


def read(folder: Folder, state: MemoryState, query: np.ndarray) -> np.ndarray:
    """Associative recall: address the state with the query's key part.

    query: int16 vector, length 704 (use query_doc to build one). Returns
    the concatenated recalled values, shape (704,). O(1) in history size.
    The folder's projections must be the same ones used at write time.
    """
    q = np.asarray(query, dtype=np.int64).reshape(-1)
    if q.shape != (DOC_DIM,):
        raise ValueError(f"expected query of shape ({DOC_DIM},), got {q.shape}")
    keys, _ = folder.project(q)
    parts = []
    for h in range(N_HEADS):
        recalled = (keys[h] @ state.tokens[h].astype(np.int64)) >> Q_SHIFT
        parts.append(recalled)
    return _clip16(np.concatenate(parts))


def read_baseline(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Growing-cache baseline: dot-product attention over ALL documents.

    This is what RAG-style or full-context methods pay per query: O(n) in
    history size, for every single query. Returns the best-matching doc.
    """
    docs_arr = np.asarray(docs, dtype=np.int64)
    q = np.asarray(query, dtype=np.int64).reshape(-1)
    scores = docs_arr @ q
    return docs_arr[int(np.argmax(scores))].astype(np.int16)
