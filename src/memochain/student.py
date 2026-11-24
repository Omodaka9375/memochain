"""The differentiable 22 KB student: writes the Delta-rule state, reads it back.

Write path: project a document embedding (2560-d, Qwen3-Embedding-4B) into
per-head unit keys and bounded values, then update the state exactly like
src/memochain/fold.py's integer fold, but in float32/bfloat16 with a
Straight-Through Estimator toward the Q14 int16 grid so trained weights
survive quantization with no accuracy cliff.

Read path: address the state with query keys, then refine Memorilla-shaped
memory tokens (num_memories x output_dim, default 16 x 4096) from the
recalled values plus the question embedding.

The state has exactly the MemoryState shape (11, 32, 32): whatever the
student learns folds into the same 22,528-byte capsule the Rust/ZK path uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_OK = False

from .state import KEY_DIM, N_HEADS, VAL_DIM

Q14 = 1 << 14
INT16_MAX = 32767
INT16_MIN = -32768


def _require_torch() -> None:
    if not TORCH_OK:
        raise ImportError(
            "PyTorch is required for the student. Install with: uv sync --extra train"
        )


@dataclass
class StudentConfig:
    embedding_dim: int = 2560    # Qwen3-Embedding-4B width
    output_dim: int = 4096       # Qwen3-8B soft-token width
    num_memories: int = 16       # teacher's token budget
    num_heads: int = N_HEADS     # 11: the 22 KB state layout
    key_dim: int = KEY_DIM       # 32
    val_dim: int = VAL_DIM       # 32
    initial_beta: float = 0.5
    initial_decay: float = 0.999
    reader_hidden: int = 1024


if TORCH_OK:

    def quantize_q14_ste(x: "torch.Tensor") -> "torch.Tensor":
        """Straight-through Q14 int16 quantizer: forward = quantized, backward = identity."""
        q = torch.clamp(
            torch.round(x * Q14), INT16_MIN, INT16_MAX
        ) / Q14
        return x + (q - x).detach()

    def quantize_state_int16(state: "torch.Tensor") -> "torch.Tensor":
        """Hard int16 Q14 quantization for export/eval (no STE)."""
        return torch.clamp(torch.round(state * Q14), INT16_MIN, INT16_MAX) / Q14


    class MemoChainStudent(nn.Module):
        """22 KB Delta-rule memory with a Memorilla-shaped read head."""

        def __init__(self, config: StudentConfig | None = None) -> None:
            super().__init__()
            cfg = config or StudentConfig()
            self.config = cfg
            h, kd, vd = cfg.num_heads, cfg.key_dim, cfg.val_dim

            self.key_proj = nn.Linear(cfg.embedding_dim, h * kd)
            self.val_proj = nn.Linear(cfg.embedding_dim, h * vd)
            self.beta_logits = nn.Parameter(torch.zeros(h))
            self.decay_logits = nn.Parameter(torch.full((h,), 6.0))  # sigmoid(6) ~ 0.998

            nn.init.xavier_uniform_(self.key_proj.weight)
            nn.init.xavier_uniform_(self.val_proj.weight)
            with torch.no_grad():
                # Start near the configured operating point.
                self.beta_logits.fill_(float(np.log(cfg.initial_beta / (1 - cfg.initial_beta))))

            # Reader: recalled values (h*vd) + question (embedding_dim) -> tokens.
            # Small bottleneck: pool into memory slots with a shared projection,
            # then per-slot refine with a low-rank map. Memorilla's own head is
            # a projection, not a wide MLP.
            reader_in = h * vd + cfg.embedding_dim
            self.reader_head_norm = nn.LayerNorm(reader_in)
            self.reader_slots = nn.Linear(reader_in, cfg.num_memories * cfg.embedding_dim)
            self.reader_out = nn.Linear(cfg.embedding_dim, cfg.output_dim)
            nn.init.xavier_uniform_(self.reader_slots.weight)
            nn.init.xavier_uniform_(self.reader_out.weight)

        @property
        def beta(self) -> "torch.Tensor":
            return torch.sigmoid(self.beta_logits)

        @property
        def decay(self) -> "torch.Tensor":
            return torch.sigmoid(self.decay_logits)

        def initial_state(self, batch: int, device=None, dtype=None) -> "torch.Tensor":
            shape = (batch, self.config.num_heads, self.config.key_dim, self.config.val_dim)
            return torch.zeros(shape, device=device, dtype=dtype or torch.float32)

        def project(self, docs: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """docs [B, D] -> unit keys [B, H, K] and values [B, H, V]."""
            keys = self.key_proj(docs).view(-1, self.config.num_heads, self.config.key_dim)
            vals = self.val_proj(docs).view(-1, self.config.num_heads, self.config.val_dim)
            keys = F.normalize(keys, dim=-1, eps=1e-6)
            return keys, vals

        def fold_step(
            self, state: "torch.Tensor", docs: "torch.Tensor", quantize: bool = False
        ) -> "torch.Tensor":
            """One Delta-rule step: state [B,H,K,V], docs [B,D] -> new state.

            Mirrors fold.py exactly: decayed state plus rank-1 key/delta update.
            """
            keys, vals = self.project(docs)
            recalled = torch.einsum("bhk,bhkv->bhv", keys, state)
            beta = self.beta.view(1, -1, 1)
            delta = beta * (vals - recalled)
            update = torch.einsum("bhk,bhv->bhkv", keys, delta)
            decay = self.decay.view(1, -1, 1, 1)
            new_state = decay * state + update
            if quantize:
                new_state = quantize_q14_ste(new_state)
            return new_state

        def fold_sequence(
            self,
            docs: "torch.Tensor",
            padding: "torch.Tensor | None" = None,
            quantize: bool = True,
            initial: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """Fold a padded batch of collections: docs [B, N, D] -> state [B,H,K,V].

            padding [B, N] True = padded position (skipped).
            initial: optional starting state [B,H,K,V] (for chunked folding);
            defaults to zeros.
            """
            if initial is not None:
                state = initial
            else:
                state = self.initial_state(
                    docs.shape[0], device=docs.device, dtype=docs.dtype
                )
            n = docs.shape[1]
            for t in range(n):
                if padding is not None:
                    keep = (~padding[:, t]).float().unsqueeze(1)  # [B, 1]
                else:
                    keep = torch.ones(docs.shape[0], 1, device=docs.device, dtype=docs.dtype)
                # Skip padded steps entirely for speed when the batch is aligned.
                if bool(keep.all()):
                    state = self.fold_step(state, docs[:, t], quantize=quantize)
                else:
                    # Correctness path: fold real docs, hold padded rows.
                    stepped = self.fold_step(state, docs[:, t], quantize=quantize)
                    state = keep.unsqueeze(-1).unsqueeze(-1) * stepped + (
                        1 - keep
                    ).unsqueeze(-1).unsqueeze(-1) * state
            return state

        def read_tokens(
            self,
            state: "torch.Tensor",
            question: "torch.Tensor",
            quantized_state: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """state [B,H,K,V] + question [B,D] -> memory tokens [B, M, output_dim]."""
            q_keys, _ = self.project(question)
            recalled = torch.einsum("bhk,bhkv->bhv", q_keys, state)
            flat = recalled.reshape(recalled.shape[0], -1)
            z = torch.cat([flat, question], dim=-1)
            z = self.reader_head_norm(z)
            slots = self.reader_slots(z)  # [B, M * embedding_dim]
            slots = slots.view(-1, self.config.num_memories, self.config.embedding_dim)
            tokens = self.reader_out(slots)  # [B, M, output_dim]
            return tokens

        def state_numpy(self, state: "torch.Tensor", index: int = 0) -> "np.ndarray":
            """Export one batch element's state as the 22,528-byte int16 grid."""
            s = quantize_state_int16(state[index : index + 1]).detach().cpu().numpy()
            return np.round(s[0] * Q14).astype(np.int16)


def export_to_memory_state(
    student: MemoChainStudent, state: "torch.Tensor"
) -> "MemoryState":
    """Convert a student state batch of one into a real MemoryState."""
    from .state import MemoryState

    _require_torch()
    arr = student.state_numpy(state)
    return MemoryState(arr)
