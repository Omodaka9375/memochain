"""MemoChain OpenAI-compatible memory proxy.

Sits between any OpenAI-compatible client (KAI, IDEs, agent harnesses) and
any OpenAI-compatible backend (vLLM, llama.cpp server, cloud APIs). Per
session it keeps recent turns verbatim (hot window), folds evicted old turns
into the 22 KB Q14 Capsule through the exact integer student writer, and
injects a compact recalled digest into the system prompt.

Run:
  uv run --extra proxy python -m memochain.server \
      --backend-base-url http://localhost:8000/v1 \
      --host 0.0.0.0 --port 9000

Endpoints:
  POST /v1/chat/completions   OpenAI chat completions with memory folding (+SSE)
  POST /v1/messages           Anthropic Messages API with memory folding (+SSE)
  GET  /v1/memochain/status   memory stats for a session
  GET  /v1/memochain/verify   receipt-chain + capsule integrity for a session
  GET  /healthz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import numpy as np

from .session import SessionMemory, estimate_tokens, message_hash, message_text

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    FASTAPI_OK = True
except ImportError:  # pragma: no cover
    FASTAPI_OK = False


# --------------------------------------------------------------------------
# Writer adapter: exact integer student writer (no torch needed at request time)
# --------------------------------------------------------------------------

class IntWriterAdapter:
    """Adapts IntStudentWriter to the FoldWriter protocol with per-doc receipts.

    ``checkpoint`` may be a training dir with student.pt (needs torch) or a
    dir of pre-exported numpy arrays (np_key_w.npy etc., torch-free, for
    torch-free proxy hosts).
    """

    def __init__(self, checkpoint: str | Path) -> None:
        from ..student_int import IntStudentWriter

        path = Path(checkpoint)
        np_key_w = path / "np_key_w.npy"
        if np_key_w.exists():  # torch-free numpy checkpoint
            arrays = {
                "w_key": np.load(path / "np_key_w.npy"),
                "b_key": np.load(path / "np_key_b.npy"),
                "w_val": np.load(path / "np_val_w.npy"),
                "b_val": np.load(path / "np_val_b.npy"),
                "beta": np.load(path / "np_beta.npy"),
                "decay": np.load(path / "np_decay.npy"),
            }
            for name, arr in arrays.items():
                if arr.dtype != np.dtype("<i2"):
                    raise ValueError(
                        f"{path}/{name} is {arr.dtype}, expected int16 Q14 "
                        "(export with tools/export_np_weights.py, NOT raw floats)"
                    )
            self.inner = IntStudentWriter(**arrays)
        else:
            self.inner = IntStudentWriter.from_checkpoint(path)
        self._digest = self.inner.weights_digest()

    def weights_digest(self) -> str:
        return self._digest

    def fold_with_receipts(self, docs_q14, initial_state=None, initial_head=None):
        from ..state import MemoryState
        from ..verify import ChainHead

        docs = np.asarray(docs_q14, dtype="<i2")
        if docs.ndim == 1:
            docs = docs[None, :]
        state = (
            np.zeros((11, 32, 32), dtype="<i2")
            if initial_state is None
            else np.asarray(initial_state.tokens, dtype="<i2").copy()
        )
        head = initial_head
        rows: list[dict[str, Any]] = []
        dot_k, dot_v = self.inner.exact_projections(docs)
        for i in range(docs.shape[0]):
            state, receipt, head = self.inner.write_with_receipt(
                state, docs[i], prev=head, dot_k=dot_k[i], dot_v=dot_v[i]
            )
            rows.append(
                {
                    "step": receipt.step,
                    "doc_hash": receipt.doc_digest,
                    "state_digest": receipt.new_head,
                    "binding": receipt.digest,
                    "accumulator": head.accumulator,
                }
            )
        return MemoryState(state), head, rows


# --------------------------------------------------------------------------
# Reader adapter: query-conditioned digest via the distilled student reader
# --------------------------------------------------------------------------

class ReaderSidecarAdapter:
    """Calls the reader sidecar (e.g. localhost:8002) for natural-language memory notes.

    Falls back to the local numpy profile digest when the sidecar is
    unreachable, so the proxy keeps working standalone.
    """

    def __init__(self, base_url: str, max_words: int = 160, timeout: float = 300.0) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.max_words = max_words

    def read(self, state, query_embed, prev_summary: str) -> str:
        import base64

        try:
            state_b64 = base64.b64encode(
                np.asarray(state.tokens, dtype="<i2").tobytes()
            ).decode()
            resp = self.client.post(
                "/read",
                json={
                    "state_b64": state_b64,
                    "query_embedding": np.asarray(query_embed, dtype=float).tolist()
                    if query_embed is not None
                    else None,
                    "max_words": self.max_words,
                },
            )
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
            if text and text.lower() != "no salient memory":
                return text
            return prev_summary
        except Exception:
            pass
        # fallback: local deterministic digest
        return _profile_digest(state, query_embed, prev_summary)


def _profile_digest(state, query_embed: np.ndarray, prev_summary: str) -> str:
    """Torch-free digest: numpy head-recall profile over the Q14 state."""
    state_arr = np.asarray(state.tokens, dtype=np.float32) / float(1 << 14)
    q = np.asarray(query_embed if query_embed is not None else np.zeros(2560), dtype=np.float32).reshape(2560)
    keys = q[: 11 * 32].reshape(11, 32)
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = keys / norms
    recalled = np.einsum("hk,hkv->hv", unit, state_arr)
    recall_norms = np.linalg.norm(recalled, axis=-1)
    profile = ", ".join(f"h{i}={n:.2f}" for i, n in enumerate(recall_norms))
    parts = []
    if prev_summary:
        parts.append(prev_summary)
    parts.append(
        f"Earlier conversation compressed into session memory; recall profile: {profile}. "
        "Stay consistent with earlier decisions and preferences."
    )
    return "\n".join(parts)[:1600]


class StudentReaderAdapter:
    """Runs the distilled student's read head on GPU/CPU to produce a digest.

    The reader turns (22 KB state, query embedding, previous digest) into a
    short text block injected into the system prompt. We decode the student's
    memory tokens with a small local LLM (the reader LLM the student was
    distilled against) or fall back to a deterministic similarity digest.
    """

    def __init__(self, checkpoint: str | Path, decoder: str, device: str = "cpu", max_words: int = 220) -> None:
        try:
            import torch

            from ..student import MemoChainStudent, StudentConfig

            self.torch = torch
            self.device = torch.device(device)
            self.max_words = max_words
            ckpt = torch.load(Path(checkpoint) / "student.pt", map_location="cpu", weights_only=False)
            self.student = MemoChainStudent(StudentConfig(**ckpt["config"])).to(self.device).eval()
            self.student.load_state_dict(ckpt["state_dict"])

            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(decoder)
            self._llm = None
        except ImportError:  # torch-free host: fall back to profile digest only
            self.torch = None
            self.student = None
            self.max_words = max_words
            self._llm = None

    def read(self, state, query_embed, prev_summary: str) -> str:
        if self.torch is None or self.student is None:
            return self._numpy_digest(state, query_embed, prev_summary)
        torch = self.torch
        with torch.inference_mode():
            state_t = torch.from_numpy(
                np.asarray(state.tokens, dtype=np.float32) / float(1 << 14)
            ).unsqueeze(0).to(self.device)
            q = torch.from_numpy(np.asarray(query_embed, dtype=np.float32)).unsqueeze(0).to(self.device)
            tokens = self.student.read_tokens(state_t, q)[0]  # [M, output_dim]

        if self._llm is not None:
            try:
                return self._llm.summarize(tokens, prev_summary, self.max_words)
            except Exception:
                pass
        return self._similarity_digest(state, query_embed, prev_summary)

    def _numpy_digest(self, state, query_embed: np.ndarray, prev_summary: str) -> str:
        """Torch-free digest: numpy head-recall profile over the Q14 state."""
        state_arr = np.asarray(state.tokens, dtype=np.float32) / float(1 << 14)
        q = np.asarray(query_embed, dtype=np.float32).reshape(2560)
        # distribute query across 11 heads (32+32 chunks)
        keys = q[: 11 * 32].reshape(11, 32)
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = keys / norms
        recalled = np.einsum("hk,hkv->hv", unit, state_arr)  # [11, 32]
        recall_norms = np.linalg.norm(recalled, axis=-1)
        profile = ", ".join(f"h{i}={n:.2f}" for i, n in enumerate(recall_norms))
        parts = []
        if prev_summary:
            parts.append(prev_summary)
        parts.append(
            f"[Folded session memory, step {state_digest[:8] if False else ''}]: earlier "
            f"conversation compressed; recall profile: {profile}. Stay consistent with "
            "earlier decisions and preferences."
        )
        return " ".join(parts)[: self.max_words * 8]

    def _similarity_digest(self, state, query_embed: np.ndarray, prev_summary: str) -> str:
        """Deterministic, dependency-free digest: head-recall profile of the query."""
        torch = self.torch
        with torch.inference_mode():
            state_t = torch.from_numpy(
                np.asarray(state.tokens, dtype=np.float32) / float(1 << 14)
            ).unsqueeze(0).to(self.device)
            q = torch.from_numpy(np.asarray(query_embed, dtype=np.float32)).unsqueeze(0).to(self.device)
            keys, _ = self.student.project(q)
            recalled = torch.einsum("bhk,bhkv->bhv", keys, state_t)[0]  # [H, V]
            recall_norms = recalled.norm(dim=-1).tolist()
            beta = self.student.beta.tolist()

        parts = []
        if prev_summary:
            parts.append(prev_summary)
        profile = ", ".join(
            f"head{i}={n:.2f}@b={b:.2f}" for i, (n, b) in enumerate(zip(recall_norms, beta))
        )
        parts.append(
            "Earlier conversation folded (session memory state v"
            f"{int(state.tokens.size)}-slot); recall profile: {profile}. "
            "Continue consistently with earlier decisions and preferences."
        )
        return " ".join(parts)[: self.max_words * 8]


class TransformersReaderAdapter:
    """In-process reader: numpy read head + a small HF decoder, no vLLM.

    Loads the decoder once with transformers; each fold decodes the 16 memory
    tokens with greedy generate (a few seconds on GPU). Use with the joint or
    pv4 e2e checkpoint whose reader.npz matches the decoder's hidden size.
    """

    def __init__(self, checkpoint: str | Path, decoder: str, device: str = "cuda", max_words: int = 160) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..eval_qa import MEMORY_TOKEN
        from ..reader_np import NumpyReader

        path = Path(checkpoint)
        self.reader = NumpyReader(
            path / "reader.npz", np.load(path / "np_key_w.npy"), np.load(path / "np_key_b.npy")
        )
        import torch

        self.torch = torch
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.max_words = max_words
        self.tokenizer = AutoTokenizer.from_pretrained(decoder)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [MEMORY_TOKEN]})
        self.memory_token_id = self.tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(decoder, dtype=torch.bfloat16)
        except TypeError:  # transformers < 5
            self.model = AutoModelForCausalLM.from_pretrained(decoder, torch_dtype=torch.bfloat16)
        self.model = self.model.to(self.device).eval()
        self.embed = self.model.get_input_embeddings()

    def read(self, state, query_embed, prev_summary: str) -> str:
        from .reader_service import SUMMARY_QUESTION
        from ..eval_qa import (
            PERSONA_PROMPT,
            build_memory_messages,
            clean_generation,
        )

        torch = self.torch
        t = self.reader.read_tokens(np.asarray(state.tokens), np.asarray(query_embed, dtype=np.float32))
        question = SUMMARY_QUESTION.format(words=self.max_words)
        messages = build_memory_messages(question, t.shape[0], system_prompt=PERSONA_PROMPT)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        positions = [i for i, tid in enumerate(ids) if tid == self.memory_token_id]
        if len(positions) != t.shape[0]:
            raise RuntimeError(f"expected {t.shape[0]} memory tokens in prompt, found {len(positions)}")
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        base_embeds = self.embed(ids_t.clamp(max=self.embed.num_embeddings - 1)).clone()
        mem = torch.from_numpy(t).to(device=base_embeds.device, dtype=base_embeds.dtype)
        for row, pos in enumerate(positions):
            base_embeds[0, pos] = mem[row]
        with torch.inference_mode():
            out = self.model.generate(
                inputs_embeds=base_embeds, max_new_tokens=256, do_sample=False
            )
        text = clean_generation(self.tokenizer.decode(out[0], skip_special_tokens=True)).strip()
        if text and text.lower() != "no salient memory":
            return text
        return prev_summary


def _backend_auth_headers(backend_base_url: str, backend_api_key: str | None) -> dict[str, str]:
    """Build auth headers for OpenAI-style, OpenRouter, or direct Anthropic backends."""
    if not backend_api_key:
        return {}
    if "api.anthropic.com" in backend_base_url:
        return {"x-api-key": backend_api_key, "Authorization": f"Bearer {backend_api_key}"}
    return {"Authorization": f"Bearer {backend_api_key}"}


def _merge_anthropic_system(system_field: Any, memory_block: str) -> Any:
    """Append the memory block to Anthropic's top-level ``system`` (string or blocks)."""
    if not memory_block:
        return system_field
    if system_field is None:
        return memory_block
    if isinstance(system_field, str):
        return f"{system_field}\n\n{memory_block}" if system_field else memory_block
    if isinstance(system_field, list):
        return list(system_field) + [{"type": "text", "text": memory_block}]
    return memory_block


class HybridReaderAdapter:
    """Reader-free Option A digest: asks the active backend model to update a
    compact structured fact sheet from newly folded turns.

    Runs inside the slim numpy-only container with zero extra GPU memory and
    zero reader-model boot. Supports OpenAI ``/chat/completions`` and Anthropic
    ``/messages`` backends (direct Anthropic API or OpenRouter). Falls back to
    the deterministic profile digest if the backend call fails.
    """

    def __init__(
        self,
        backend_base_url: str,
        backend_api_key: str | None = None,
        max_words: int = 220,
        timeout: float = 60.0,
        reader_model: str | None = None,
    ) -> None:
        headers = _backend_auth_headers(backend_base_url, backend_api_key)
        self.backend_base_url = backend_base_url
        self.client = httpx.Client(base_url=backend_base_url, timeout=timeout, headers=headers)
        self.max_words = max_words
        self._model_id: str | None = reader_model
        self._prefer_messages = "api.anthropic.com" in backend_base_url

    def _pick_model(
        self, model_hint: str | None = None, auth_headers: dict[str, str] | None = None
    ) -> str | None:
        if self._model_id:
            return self._model_id
        if model_hint:
            self._model_id = model_hint
            return self._model_id
        try:
            r = self.client.get("/models", headers=auth_headers) if auth_headers else self.client.get("/models")
            if r.status_code == 200:
                ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
                if ids:
                    self._model_id = ids[0]
        except Exception:
            pass
        return self._model_id

    def read(self, state, query_embed, prev_summary: str) -> str:
        return prev_summary or _profile_digest(state, query_embed, prev_summary)

    def read_fold(
        self,
        state,
        query_embed,
        prev_summary: str,
        folded_entries: list[tuple[str, str, str]],
        model_hint: str | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> str:
        if not folded_entries:
            return prev_summary
        model = self._pick_model(model_hint=model_hint, auth_headers=auth_headers)
        if not model:
            return _profile_digest(state, query_embed, prev_summary)

        turns = "\n".join(f"[{role}] {text}" for _, role, text in folded_entries)
        prompt = (
            f"Merge the existing notes and the new transcript lines below into one clean bullet list "
            f"(max {self.max_words} words). Keep concrete entities, names, cities, ports, hosts, URLs, "
            f"package/tool names, file paths, decisions, and preferences verbatim. "
            f"Output only the bullet points with no preamble or commentary.\n\n"
            f"Existing notes:\n{prev_summary or '(none)'}\n\n"
            f"Transcript lines:\n{turns}"
        )
        try:
            if self._prefer_messages:
                text = self._call_messages(model, prompt, auth_headers=auth_headers)
            else:
                text = self._call_chat_completions(model, prompt, auth_headers=auth_headers)
            if text:
                return text
        except Exception:
            pass
        return prev_summary or _profile_digest(state, query_embed, prev_summary)

    def _call_chat_completions(
        self, model: str, prompt: str, auth_headers: dict[str, str] | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1536,
        }
        if "api.anthropic.com" not in self.backend_base_url:
            payload["temperature"] = 0.0
        resp = (
            self.client.post("/chat/completions", json=payload, headers=auth_headers)
            if auth_headers
            else self.client.post("/chat/completions", json=payload)
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return (msg.get("content") or "").strip()

    def _call_messages(
        self, model: str, prompt: str, auth_headers: dict[str, str] | None = None
    ) -> str:
        hdrs = {"anthropic-version": "2023-06-01"}
        if auth_headers:
            hdrs.update(auth_headers)
        resp = self.client.post(
            "/messages",
            headers=hdrs,
            json={
                "model": model,
                "max_tokens": 1536,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        blocks = resp.json().get("content") or []
        parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()


class EmbeddingSidecar:
    """Embeds folded messages via an OpenAI-compatible /v1/embeddings backend."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.post(
            "/embeddings", json={"model": self.model, "input": texts}, headers=self.headers
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return np.asarray([row["embedding"] for row in data], dtype=np.float32)


class LocalEmbeddingFn:
    """Embeds via the local Qwen3-Embedding sidecar vLLM process."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.post(
            "/embeddings", json={"model": self.model, "input": texts}, headers=self.headers
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return np.asarray([row["embedding"] for row in data], dtype=np.float32)


# --------------------------------------------------------------------------
# Session manager
# --------------------------------------------------------------------------

class SessionManager:
    def __init__(self, store_root: str | Path, **session_kwargs: Any) -> None:
        self.store_root = Path(store_root)
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, SessionMemory] = {}
        self.session_kwargs = session_kwargs

    def get(self, session_id: str | None, messages: list[dict[str, Any]]) -> SessionMemory:
        if not session_id:
            seed = hashlib.sha256(
                json.dumps(messages[:3], sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            session_id = f"anon-{seed}"
        sid = "".join(c if c.isalnum() or c in "-_" else "-" for c in session_id)[:80]
        if sid not in self.sessions:
            self.sessions[sid] = SessionMemory(sid, self.store_root / sid, **self.session_kwargs)
        else:
            self.sessions[sid].sync_from_disk()
        return self.sessions[sid]

    def get_by_id(self, session_id: str) -> SessionMemory:
        """Lookup for proof-status callbacks; creates the dir lazily if missing."""
        sid = "".join(c if c.isalnum() or c in "-_" else "-" for c in session_id)[:80]
        if sid not in self.sessions:
            self.sessions[sid] = SessionMemory(sid, self.store_root / sid, **self.session_kwargs)
        else:
            self.sessions[sid].sync_from_disk()
        return self.sessions[sid]


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

def create_app(
    backend_base_url: str,
    backend_api_key: str | None,
    checkpoint: str | Path,
    store_root: str | Path,
    embedding_base_url: str | None,
    embedding_model: str = "Qwen/Qwen3-Embedding-4B",
    embedding_api_key: str | None = None,
    reader_decoder: str = "Qwen/Qwen3-8B",
    reader_device: str = "cpu",
    hot_limit: int = 80000,
    keep_min: int = 40000,
    writer: Any | None = None,
    reader: Any | None = None,
    embed_fn: Any | None = None,
    backend_client: Any | None = None,
    proof_mode: str | None = None,
    ivc_bin: str | Path = "crates/sp1/script/target/release/ivc",
    proof_weights: str | Path | None = None,
    proof_work_root: str | Path = "runs/proofs",
    reader_sidecar_url: str | None = None,
    reader_backend: str = "hybrid",
    recall_top_k: int | None = None,
    max_chunks: int = 2000,
    reader_model: str | None = None,
) -> "FastAPI":
    if not FASTAPI_OK:  # pragma: no cover
        raise ImportError("fastapi is required for the proxy: uv sync --extra proxy")

    writer = writer or IntWriterAdapter(checkpoint)
    if reader is None:
        if reader_sidecar_url:
            reader = ReaderSidecarAdapter(reader_sidecar_url)
        elif reader_backend == "transformers":
            reader = TransformersReaderAdapter(checkpoint, reader_decoder, device=reader_device)
        elif reader_backend == "profile":
            reader = StudentReaderAdapter(checkpoint, reader_decoder, device=reader_device)
        else:
            reader = HybridReaderAdapter(
                backend_base_url,
                backend_api_key=backend_api_key,
                reader_model=reader_model,
            )
    if recall_top_k is None:
        recall_top_k = 4 if reader_backend == "hybrid" else 0
    if embed_fn is None:
        embed_fn = (
            LocalEmbeddingFn(embedding_base_url, embedding_model, embedding_api_key)
            if embedding_base_url
            else _hash_embed
        )

    sessions = SessionManager(
        store_root,
        writer=writer,
        embed_fn=embed_fn,
        reader=reader,
        hot_limit=hot_limit,
        keep_min=keep_min,
        recall_top_k=recall_top_k,
        max_chunks=max_chunks,
    )

    backend_headers = _backend_auth_headers(backend_base_url, backend_api_key)
    app = FastAPI(title="MemoChain memory proxy", version="0.1.0")
    client = backend_client or httpx.AsyncClient(
        base_url=backend_base_url, timeout=None, headers=backend_headers
    )

    proofs = None
    if proof_mode:
        from .proof import ProofQueue

        proofs = ProofQueue(
            mode=proof_mode,
            ivc_bin=ivc_bin,
            weights_path=proof_weights or Path(checkpoint) / "weights.bin",
            work_root=proof_work_root,
            session_lookup=sessions.get_by_id,
            expected_weights_digest=writer.weights_digest(),
        )

    sessions.session_kwargs["proof_queue"] = proofs

    # backend served-model cache (for model-name normalization + /v1/models)
    _model_cache: dict[str, Any] = {"ids": None, "ts": 0.0}

    async def _backend_model_ids() -> list[str] | None:
        import time as _time

        if _model_cache["ids"] is not None and _time.time() - _model_cache["ts"] < 300:
            return _model_cache["ids"]
        try:
            resp = await client.get("/models")
            if resp.status_code == 200:
                ids = [m.get("id") for m in resp.json().get("data", []) if m.get("id")]
                _model_cache["ids"] = ids
                _model_cache["ts"] = _time.time()
                return ids
        except Exception:
            pass
        return _model_cache["ids"]

    @app.get("/v1/models")
    async def list_models() -> Any:
        ids = await _backend_model_ids()
        if ids is None:
            return JSONResponse({"error": "backend unavailable"}, status_code=502)
        return {
            "object": "list",
            "data": [
                {"id": i, "object": "model", "owned_by": "memochain-proxy"} for i in ids
            ],
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "backend": backend_base_url,
            "sessions": len(sessions.sessions),
            "proof_mode": proof_mode,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("X-Session-Id")
            or body.get("session_id")
            or body.get("user")
        )
        messages: list[dict[str, Any]] = body.get("messages") or []
        mem = sessions.get(session_id, messages)

        # normalize any requested model name to the backend's served model,
        # so clients with a stale/alias model id still work
        requested = body.get("model")
        backend_ids = await _backend_model_ids()
        if backend_ids and requested not in backend_ids:
            body["model"] = backend_ids[0]

        req_headers = _forward_auth_headers(request, backend_base_url)
        rewritten, stats = mem.process(
            messages, model_hint=body.get("model"), auth_headers=req_headers
        )
        out_body = {k: v for k, v in body.items() if k not in ("session_id",)}
        out_body["messages"] = rewritten
        stream = bool(out_body.get("stream"))

        if not stream:
            resp = await client.post("/chat/completions", json=out_body, headers=req_headers)
            if resp.status_code != 200:
                return _error_json_response(resp.text, resp.status_code)
            payload = resp.json()
            payload["memochain"] = _stats_block(mem, stats)
            return JSONResponse(payload)

        # SSE passthrough; attach stats to the LAST real chunk (an extra key on
        # a valid chunk — clients strip unknown fields, but reject chunks that
        # lack `choices`, so we must never emit a stats-only chunk).
        # If the backend fails before streaming (e.g. 400/500), return a proper
        # non-200 OpenAI error JSON response instead of a malformed 200 SSE chunk.
        async def gen() -> AsyncIterator[bytes]:
            async with client.stream(
                "POST", "/chat/completions", json=out_body, headers=req_headers
            ) as resp:
                if resp.status_code != 200:
                    err = (await resp.aread()).decode("utf-8", "replace")
                    yield f"data: {json.dumps(_parse_error_payload(err, resp.status_code))}\n\n".encode()
                    return
                last_chunk: dict[str, Any] | None = None
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            if last_chunk is not None:
                                last_chunk["memochain"] = _stats_block(mem, stats)
                                yield f"data: {json.dumps(last_chunk)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            return
                        yield f"data: {data}\n\n".encode()
                        try:
                            parsed = json.loads(data)
                            if isinstance(parsed, dict):
                                last_chunk = parsed
                        except ValueError:
                            pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        body = await request.json()
        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("X-Session-Id")
            or (body.get("metadata") or {}).get("user_id")
            or body.get("session_id")
        )
        messages: list[dict[str, Any]] = body.get("messages") or []
        mem = sessions.get(session_id, messages)

        requested = body.get("model")
        backend_ids = await _backend_model_ids()
        if backend_ids and requested not in backend_ids:
            body["model"] = backend_ids[0]

        req_headers = _forward_anthropic_headers(request, backend_base_url)
        rewritten, stats = mem.process(
            messages,
            model_hint=body.get("model"),
            inject_memory=False,
            auth_headers=req_headers,
        )
        out_body = {k: v for k, v in body.items() if k not in ("session_id",)}
        out_body["messages"] = rewritten
        memory_block = stats.get("memory_block", "")
        if memory_block:
            out_body["system"] = _merge_anthropic_system(out_body.get("system"), memory_block)

        stream = bool(out_body.get("stream"))

        if not stream:
            resp = await client.post("/messages", json=out_body, headers=req_headers)
            if resp.status_code != 200:
                return _anthropic_error_response(resp.text, resp.status_code)
            payload = resp.json()
            payload["memochain"] = _stats_block(mem, stats)
            return JSONResponse(payload)

        async def gen_anthropic() -> AsyncIterator[bytes]:
            async with client.stream(
                "POST", "/messages", json=out_body, headers=req_headers
            ) as resp:
                if resp.status_code != 200:
                    err = (await resp.aread()).decode("utf-8", "replace")
                    yield f"event: error\ndata: {json.dumps(_parse_error_payload(err, resp.status_code))}\n\n".encode()
                    return
                async for line in resp.aiter_lines():
                    yield f"{line}\n".encode()

        return StreamingResponse(gen_anthropic(), media_type="text/event-stream")

    @app.get("/v1/memochain/status")
    async def status(session_id: str) -> dict[str, Any]:
        mem = sessions.get_by_id(session_id)
        st = mem.verify()
        st["summary"] = mem.summary
        st["folded_messages"] = len(mem.folded_hashes)
        if proofs is not None:
            st["proof"] = proofs.status(mem.session_id)
        return st

    @app.get("/v1/memochain/verify")
    async def verify(session_id: str) -> dict[str, Any]:
        mem = sessions.get_by_id(session_id)
        result = mem.verify()
        # a freshly created (never-folded) session has nothing to verify yet
        if not mem.capsule_path.exists():
            result["ok"] = True
            result["note"] = "no capsule yet (nothing folded)"
        if proofs is not None:
            result["proof"] = proofs.status(mem.session_id)
        return result

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await client.aclose()

    return app


def _forward_auth_headers(request: Request, backend_base_url: str) -> dict[str, str]:
    """Pass through client auth headers only when present (preserves BYOK keys)."""
    headers: dict[str, str] = {}
    auth = request.headers.get("authorization")
    x_api_key = request.headers.get("x-api-key")
    if auth:
        headers["Authorization"] = auth
        if "api.anthropic.com" in backend_base_url and not x_api_key and auth.lower().startswith("bearer "):
            headers["x-api-key"] = auth[7:].strip()
    if x_api_key:
        headers["x-api-key"] = x_api_key
    workspace = request.headers.get("anthropic-workspace-id")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    return headers


def _forward_anthropic_headers(request: Request, backend_base_url: str) -> dict[str, str]:
    headers = _forward_auth_headers(request, backend_base_url)
    headers["anthropic-version"] = request.headers.get("anthropic-version") or "2023-06-01"
    beta = request.headers.get("anthropic-beta")
    if beta:
        headers["anthropic-beta"] = beta
    return headers


def _anthropic_error_response(raw_text: str, status_code: int) -> "JSONResponse":
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            return JSONResponse(parsed, status_code=status_code)
    except ValueError:
        pass
    return _error_json_response(raw_text, status_code)


def _parse_error_payload(raw_text: str, status_code: int) -> dict[str, Any]:
    """Normalize an upstream error body into OpenAI's {"error": {...}} object."""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            return parsed
        if isinstance(parsed, dict):
            return {
                "error": {
                    "message": str(parsed.get("message") or parsed.get("detail") or raw_text),
                    "type": str(parsed.get("type") or "upstream_error"),
                    "code": status_code,
                }
            }
    except ValueError:
        pass
    return {
        "error": {
            "message": raw_text or f"Upstream HTTP {status_code}",
            "type": "upstream_error",
            "code": status_code,
        }
    }


def _error_json_response(raw_text: str, status_code: int) -> "JSONResponse":
    return JSONResponse(_parse_error_payload(raw_text, status_code), status_code=status_code)


def _stats_block(mem: SessionMemory, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": stats.get("session_id"),
        "step": mem.head.step if mem.head else 0,
        "capsule_bytes": mem.capsule_path.stat().st_size if mem.capsule_path.exists() else 0,
        "state_digest": mem.head.state_digest[:16] if mem.head else "",
        "folded_messages": len(mem.folded_hashes),
        "fold_now": stats.get("fold"),
        "hot_messages": stats.get("hot_messages"),
        "hot_tokens": stats.get("hot_tokens"),
        "recalled_chunks": stats.get("recalled_chunks", 0),
    }


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic offline embedding fallback (feature hash -> 2560-d, L2-normed)."""
    dim = 2560
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in message_text({"content": text}).lower().split():
            h = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            out[i, idx] += sign
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _env(name: str, default: str | None = None) -> str | None:
    import os

    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoChain OpenAI-compatible memory proxy")
    _backend_url = _env("MEMOCHAIN_BACKEND_BASE_URL")
    parser.add_argument(
        "--backend-base-url",
        default=_backend_url,
        required=_backend_url is None,
        help="e.g. http://localhost:8000/v1 (or MEMOCHAIN_BACKEND_BASE_URL)",
    )
    parser.add_argument("--backend-api-key", default=_env("MEMOCHAIN_BACKEND_API_KEY"))
    parser.add_argument("--checkpoint", default=_env("MEMOCHAIN_CHECKPOINT", "runs/pv4_10k"), help="dir with student.pt")
    parser.add_argument("--store-root", default=_env("MEMOCHAIN_STORE_ROOT", "runs/sessions"))
    parser.add_argument(
        "--embedding-base-url",
        default=_env("MEMOCHAIN_EMBEDDING_BASE_URL"),
        help="OpenAI-compatible /v1 embeddings endpoint for real Qwen3-Embedding-4B; omit for offline hash-embed fallback",
    )
    parser.add_argument("--embedding-model", default=_env("MEMOCHAIN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-api-key", default=_env("MEMOCHAIN_EMBEDDING_API_KEY"))
    parser.add_argument("--reader-decoder", default=_env("MEMOCHAIN_READER_DECODER", "Qwen/Qwen3-8B"))
    parser.add_argument("--reader-device", default=_env("MEMOCHAIN_READER_DEVICE", "cpu"))
    parser.add_argument(
        "--reader-backend",
        default=_env("MEMOCHAIN_READER_BACKEND", "hybrid"),
        choices=["hybrid", "transformers", "profile"],
        help="reader mode: hybrid (default, backend fact sheet + top-K chunk recall, no reader LLM), transformers (in-process decoder LM), or profile (numpy recall profile only)",
    )
    _rk = _env("MEMOCHAIN_RECALL_TOP_K")
    parser.add_argument(
        "--recall-top-k",
        type=int,
        default=int(_rk) if _rk is not None else None,
        help="verbatim old chunks retrieved per query (default 4 for hybrid, 0 otherwise)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=int(_env("MEMOCHAIN_MAX_CHUNKS", "2000")),
        help="max stored SQLite chunks per session (ring buffer cap, 0 = unlimited, default 2000)",
    )
    parser.add_argument(
        "--reader-sidecar-url",
        default=_env("MEMOCHAIN_READER_SIDECAR_URL"),
        help="reader sidecar base URL (e.g. http://localhost:8002) for natural-language memory notes",
    )
    parser.add_argument(
        "--hot-limit",
        type=int,
        default=int(_env("MEMOCHAIN_HOT_LIMIT", "80000")),
        help="conversation hot-window token limit (excluding system prompt)",
    )
    parser.add_argument(
        "--keep-min",
        type=int,
        default=int(_env("MEMOCHAIN_KEEP_MIN", "40000")),
        help="min conversation tokens kept hot after folding",
    )
    parser.add_argument(
        "--proof-mode",
        default=_env("MEMOCHAIN_PROOF_MODE"),
        choices=[None, "execute", "prove"],
        help="background SP1 verification: execute (fast guest replay check) or prove (recursive STARK)",
    )
    parser.add_argument("--ivc-bin", default="crates/sp1/script/target/release/ivc")
    parser.add_argument("--proof-weights", default=None, help="weights.bin for the prover (default: <checkpoint>/weights.bin)")
    parser.add_argument("--proof-work-root", default="runs/proofs")
    parser.add_argument(
        "--reader-model",
        default=_env("MEMOCHAIN_READER_MODEL"),
        help="model id for hybrid fold summaries (defaults to backend /models or last chat model)",
    )
    parser.add_argument("--host", default=_env("MEMOCHAIN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env("MEMOCHAIN_PORT", "9000")))
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        backend_base_url=args.backend_base_url,
        backend_api_key=args.backend_api_key,
        checkpoint=args.checkpoint,
        store_root=args.store_root,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        reader_decoder=args.reader_decoder,
        reader_device=args.reader_device,
        hot_limit=args.hot_limit,
        keep_min=args.keep_min,
        proof_mode=args.proof_mode,
        ivc_bin=args.ivc_bin,
        proof_weights=args.proof_weights,
        proof_work_root=args.proof_work_root,
        reader_sidecar_url=args.reader_sidecar_url,
        reader_backend=args.reader_backend,
        recall_top_k=args.recall_top_k,
        max_chunks=args.max_chunks,
        reader_model=args.reader_model,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
