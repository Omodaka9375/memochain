"""Standalone Memorilla teacher: query-conditioned compression for distillation.

Reimplements Memorilla's MemoryModule (snap-stanford/memorilla,
Apache-2.0, paper arXiv:2610.21375) faithfully in pure PyTorch, WITHOUT
vllm / the Qwen3-8B decoder / Linux. The teacher's job in MemoChain Phase 3
is to produce the target memory tokens ``read_cache(cache, question)``
that our 22 KB student learns to reproduce.

Only two artifacts are needed:
  - memorilla/Memorilla-Qwen3-8B/stage3/memory.pt  (388 MB)
  - memorilla/Memorilla-Data precomputed embeddings (pv4 = 382 MB, pmv2 = 1.15 GB)

Both fit comfortably in a 12 GB RTX 3060; no 80 GB H100 is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # torch is an optional [train] extra; core package stays pure NumPy.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_OK = False

TEACHER_REPO = "memorilla/Memorilla-Qwen3-8B"
DATA_REPO = "memorilla/Memorilla-Data"

FFN_EXPANSION = 4


def _require_torch() -> None:
    if not TORCH_OK:
        raise ImportError(
            "PyTorch is required for the distillation path. "
            "Install with: uv sync --extra train"
        )


@dataclass(frozen=True)
class TeacherConfig:
    """Mirror of Memorilla's MemoryConfig (stage3 defaults)."""

    embedding_dim: int = 2560
    output_dim: int = 4096
    num_memories: int = 16
    num_heads: int = 8
    num_self_attn_layers: int = 0
    num_cross_attn_layers: int = 2
    dropout: float = 0.0  # teacher runs in eval mode; dropout disabled
    retrieval_init: bool = False


if TORCH_OK:

    class MemorillaTeacher(nn.Module):
        """Memorilla's memory module, standalone, in eval mode.

        forward(docs, padding, question) == memorilla.MemoryModule.forward
        """

        def __init__(self, config: TeacherConfig | None = None) -> None:
            super().__init__()
            cfg = config or TeacherConfig()
            self.config = cfg
            d = cfg.embedding_dim

            self.memory_queries = nn.Parameter(torch.empty(1, cfg.num_memories, d))
            self.query_cond_attn = nn.MultiheadAttention(
                d, cfg.num_heads, dropout=0.0, batch_first=True
            )
            self.query_cond_norm = nn.LayerNorm(d)
            self.doc_norm = nn.LayerNorm(d)
            self.cross_attn_norms = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(cfg.num_cross_attn_layers)]
            )
            self.cross_attn_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(d, cfg.num_heads, dropout=0.0, batch_first=True)
                    for _ in range(cfg.num_cross_attn_layers)
                ]
            )
            self.ffn_norms = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(cfg.num_cross_attn_layers)]
            )
            self.ffn_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d, d * FFN_EXPANSION),
                        nn.GELU(),
                        nn.Dropout(0.0),
                        nn.Linear(d * FFN_EXPANSION, d),
                        nn.Dropout(0.0),
                    )
                    for _ in range(cfg.num_cross_attn_layers)
                ]
            )
            self.layer_norm = nn.LayerNorm(d)
            self.memory_projection = nn.Linear(d, cfg.output_dim)

        @classmethod
        def from_pretrained(cls, path: str | Path) -> MemorillaTeacher:
            """Load stage3/memory.pt + config.json from a local dir or HF hub.

            A local directory must contain memory.pt and config.json. A bare
            name triggers a huggingface_hub snapshot download of only those
            two files (about 388 MB).
            """
            _require_torch()
            root = Path(path)
            if not (root / "memory.pt").exists():
                from huggingface_hub import hf_hub_download

                root = Path(
                    hf_hub_download(TEACHER_REPO, "stage3/memory.pt")
                ).parent
                config_path = hf_hub_download(TEACHER_REPO, "stage3/config.json")
            else:
                config_path = root / "config.json"

            with open(config_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            cfg = TeacherConfig(
                embedding_dim=raw["embedding_dim"],
                output_dim=raw["output_dim"],
                num_memories=raw["num_memories"],
                num_heads=raw["num_heads"],
                num_self_attn_layers=raw["num_self_attn_layers"],
                num_cross_attn_layers=raw["num_cross_attn_layers"],
                dropout=0.0,
                retrieval_init=raw.get("retrieval_init", False),
            )
            module = cls(cfg)
            state = torch.load(root / "memory.pt", map_location="cpu", weights_only=True)
            module.load_state_dict(state, strict=True)
            module.eval()
            return module

        def initial_slots(self, docs: "torch.Tensor", padding: "torch.Tensor | None") -> "torch.Tensor":
            batch = docs.shape[0]
            return self.memory_queries.expand(batch, -1, -1)

        def forward(
            self,
            docs: "torch.Tensor",  # [batch, num_docs, embedding_dim]
            padding: "torch.Tensor | None",  # [batch, num_docs] True = pad
            question: "torch.Tensor",  # [batch, embedding_dim]
        ) -> "torch.Tensor":
            """Return memory tokens [batch, num_memories, output_dim]."""
            slots = self.initial_slots(docs, padding)
            q = question.unsqueeze(1)
            conditioned, _ = self.query_cond_attn(slots, q, q)
            slots = self.query_cond_norm(slots + conditioned)

            docs = self.doc_norm(docs)
            for norm, attn, fnorm, ffn in zip(
                self.cross_attn_norms, self.cross_attn_layers, self.ffn_norms, self.ffn_layers
            ):
                attended, _ = attn(
                    norm(slots), docs, docs, key_padding_mask=padding, need_weights=False
                )
                slots = slots + attended
                slots = slots + ffn(fnorm(slots))

            return self.memory_projection(self.layer_norm(slots))

        def cache_keys_values(
            self, docs: "torch.Tensor", padding: "torch.Tensor | None"
        ) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
            """Question-independent K/V per cross-attention layer (the write side).

            keys/values: [batch, heads, num_docs, head_dim].
            """
            d = self.config.embedding_dim
            docs_n = self.doc_norm(docs)  # memorilla normalizes docs in project_documents
            layers = []
            for attn in self.cross_attn_layers:
                w = attn.in_proj_weight
                b = attn.in_proj_bias
                keys = F.linear(docs_n, w[d : 2 * d], b[d : 2 * d])
                values = F.linear(docs_n, w[2 * d :], b[2 * d :])
                bsz, n, dim = keys.shape
                heads = self.config.num_heads
                hd = dim // heads
                keys = keys.view(bsz, n, heads, hd).transpose(1, 2)
                values = values.view(bsz, n, heads, hd).transpose(1, 2)
                layers.append((keys, values))
            return layers

        def read_cache(
            self,
            cache: list[tuple["torch.Tensor", "torch.Tensor"]],
            padding: "torch.Tensor | None",
            question: "torch.Tensor",
        ) -> "torch.Tensor":
            """Produce memory tokens from a cached collection (the read side)."""
            batch = question.shape[0]
            slots = self.memory_queries.expand(batch, -1, -1)
            q = question.unsqueeze(1)
            conditioned, _ = self.query_cond_attn(slots, q, q)
            slots = self.query_cond_norm(slots + conditioned)

            for (keys, values), norm, attn, fnorm, ffn in zip(
                cache,
                self.cross_attn_norms,
                self.cross_attn_layers,
                self.ffn_norms,
                self.ffn_layers,
            ):
                attended = self._attend_cached(attn, norm(slots), keys, values, padding)
                slots = slots + attended
                slots = slots + ffn(fnorm(slots))

            return self.memory_projection(self.layer_norm(slots))

        @staticmethod
        def _attend_cached(
            attn, slots, keys, values, padding
        ) -> "torch.Tensor":
            """Cross-attention of ``attn`` from slots over cached K/V.

            keys: [B, heads, num_docs, head_dim], values: [B, heads, num_docs, head_dim].
            """
            batch, n_slots, dim = slots.shape
            heads = attn.num_heads
            hd = dim // heads
            w = attn.in_proj_weight
            b = attn.in_proj_bias
            queries = F.linear(slots, w[:dim], b[:dim]).to(keys.dtype)
            queries = queries.view(batch, n_slots, heads, hd).transpose(1, 2)
            scores = torch.matmul(queries, keys.transpose(-1, -2)).float() * hd**-0.5
            if padding is not None:
                scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
            attended = torch.matmul(
                torch.softmax(scores, dim=-1).to(values.dtype), values
            )
            return attn.out_proj(
                attended.transpose(1, 2).reshape(batch, n_slots, dim).to(slots.dtype)
            )


class SyntheticTeacher:
    """Offline stand-in for MemorillaTeacher: deterministic, no weights.

    Produces memory tokens via a fixed random attention over document
    embeddings, so distillation pipelines can be smoke-tested end to end
    with zero downloads. Same interface as MemorillaTeacher.read_cache.
    """

    def __init__(
        self, embedding_dim: int = 2560, output_dim: int = 4096, num_memories: int = 16, seed: int = 0
    ) -> None:
        _require_torch()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_memories = num_memories
        rng = np.random.default_rng(seed)
        self._proj = torch.from_numpy(
            rng.standard_normal((embedding_dim, output_dim)).astype(np.float32) * 0.02
        )
        self._mix = torch.from_numpy(
            rng.standard_normal((num_memories, embedding_dim)).astype(np.float32) / np.sqrt(embedding_dim)
        )

    def cache_keys_values(self, docs: "torch.Tensor", padding=None):
        # [batch, num_memories, embedding_dim] mixing weights, cached per collection
        mix = self._mix.to(docs.dtype)  # [m, d]
        return [
            (mix.unsqueeze(0).expand(docs.shape[0], -1, -1).clone(), docs)
        ]

    def read_cache(self, cache, padding, question: "torch.Tensor") -> "torch.Tensor":
        (mix, docs) = cache[0]  # single layer: [b, m, d] mixing + raw docs
        mix = mix.to(question.device)
        # attend documents with the mixing rows, weighted by the question
        scores = torch.einsum("bmd,bd->bm", mix, question)  # [b, m]
        weights = torch.softmax(scores, dim=-1)  # [b, m]
        pooled = torch.einsum("bm,bnd->bmd", weights, docs)  # [b, m, d]
        return pooled @ self._proj.to(docs.device, docs.dtype)  # [b, m, output_dim]


def load_embedding_store(
    config: str,
    split: str,
    data_dir: str | Path,
) -> dict:
    """Load a Memorilla-Data embedding store: index + documents + questions.

    Layout (from memorilla docs/data.md):
      embeddings/qwen3-embedding-4b/{config}/{split}/index.safetensors
      .../documents-XXXXX.safetensors   float16 [total_docs, 2560]
      .../questions.safetensors         float16 [num_rows, 2560]

    Returns a dict with numpy arrays and the index records.
    """
    from safetensors import safe_open

    root = Path(data_dir) / "embeddings" / "qwen3-embedding-4b" / config / split
    index_path = root / "index.safetensors"
    if not index_path.exists():
        raise FileNotFoundError(
            f"embedding store not found under {root}; download Memorilla-Data/{config}/{split} first"
        )

    index = {}
    with safe_open(index_path, framework="numpy") as fh:
        for key in fh.keys():
            index[key] = fh.get_tensor(key)

    documents = []
    for path in sorted(root.glob("documents-*.safetensors")):
        with safe_open(path, framework="numpy") as fh:
            for key in fh.keys():
                documents.append(fh.get_tensor(key))
    docs_arr = np.concatenate(documents, axis=0) if documents else np.zeros((0, 2560), np.float16)

    questions = None
    qpath = root / "questions.safetensors"
    if qpath.exists():
        with safe_open(qpath, framework="numpy") as fh:
            for key in fh.keys():
                questions = fh.get_tensor(key)
                break

    return {"index": index, "documents": docs_arr, "questions": questions}


def collection_embeds(store: dict, collection_id: int) -> np.ndarray:
    """Slice one collection's document embeddings out of the store."""
    idx = store["index"]
    cids = np.asarray(idx["collection_ids"])
    shards = np.asarray(idx["shard"])
    starts = np.asarray(idx["start"])
    lengths = np.asarray(idx["length"])
    matches = np.nonzero(cids == collection_id)[0]
    if len(matches) == 0:
        raise KeyError(f"collection_id {collection_id} not in index")
    # collections may span shards; take the first record per (shard, collection)
    out = []
    for i in matches:
        docs = store["documents"]
        start = int(starts[i])
        length = int(lengths[i])
        out.append(docs[start : start + length])
    return np.concatenate(out, axis=0)
