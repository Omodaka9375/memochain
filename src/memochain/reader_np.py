"""Torch-free numpy read path for the distilled MemoChain student reader.

Reimplements MemoChainStudent.read_tokens with plain numpy so the proxy host
(exported weights only, no torch) can produce the 16 x 4096 memory tokens.

Weight layout in `reader.npz`:
    norm_w, norm_b           LayerNorm(reader_in = H*V + D)  (reader_head_norm)
    slots_w, slots_b         Linear(reader_in -> M * D)      (reader_slots)
    out_w, out_b             Linear(D -> output_dim)         (reader_out)
plus the writer's key projection np_key_w/np_key_b (np_key_w is [K*H, D],
shared with IntStudentWriter.w_key).

Read path (matches MemoChainStudent.read_tokens):
    q_keys = normalize(key_proj(q))            [H, K]
    recalled = einsum('hk,hkv->hv', q_keys, state_float)   state scaled Q14 -> float
    z = concat(recalled.flatten(), q)           [H*V + D]
    z = LayerNorm(z; norm_w, norm_b)
    slots = Linear_slots(z)                     [M, D]
    tokens = Linear_out(slots)                  [M, output_dim]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Q14 = 1 << 14


class NumpyReader:
    """Numpy reimplementation of the student reader head."""

    def __init__(self, npz_path: str | Path, key_w: np.ndarray, key_b: np.ndarray) -> None:
        z = np.load(npz_path)
        self.norm_w = z["norm_w"].astype(np.float32)
        self.norm_b = z["norm_b"].astype(np.float32)
        self.slots_w = z["slots_w"].astype(np.float32)  # [M*D, in]
        self.slots_b = z["slots_b"].astype(np.float32)
        self.out_w = z["out_w"].astype(np.float32)  # [output_dim, D]
        self.out_b = z["out_b"].astype(np.float32)
        self.key_w = np.asarray(key_w, dtype=np.float32)  # [H*K, D]
        self.key_b = np.asarray(key_b, dtype=np.float32)  # [H*K]
        self.num_memories = self.out_w.shape[1] // self.slots_w.shape[1] if False else None
        self.num_memories = self.out_w.shape[0] // self.slots_w.shape[0] if False else None
        # slots_w: out_features = M*D rows; D = out_w in_features
        self.slot_dim = self.out_w.shape[1]  # D
        self.num_memories = self.slots_w.shape[0] // self.slot_dim

    def read_tokens(self, state_i16: np.ndarray, query: np.ndarray) -> np.ndarray:
        """state [11,32,32] int16 Q14, query [D] -> tokens [M, output_dim]."""
        state = np.asarray(state_i16, dtype=np.float32) / Q14
        q = np.asarray(query, dtype=np.float32).reshape(-1)

        proj = q @ self.key_w.T + self.key_b  # [H*K]
        keys = proj.reshape(-1, 32)  # [H, K]
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = keys / norms

        recalled = np.einsum("hk,hkv->hv", unit, state)  # [H, V]
        z = np.concatenate([recalled.reshape(-1), q], axis=0)  # [H*V + D]

        mean = z.mean()
        var = np.var(z)
        z = (z - mean) / np.sqrt(var + 1e-5)
        z = z * self.norm_w + self.norm_b

        slots = z @ self.slots_w.T + self.slots_b  # [M*D]
        slots = slots.reshape(self.num_memories, self.slot_dim)
        tokens = slots @ self.out_w.T + self.out_b  # [M, output_dim]
        return tokens.astype(np.float32)
