"""MemoChain OpenAI-compatible memory proxy.

Sits between any OpenAI-compatible or Anthropic client (Claude Code, IDEs,
agent harnesses) and any backend (vLLM, llama.cpp server, Ollama, LM Studio,
OpenRouter, Anthropic). Per session it keeps recent turns verbatim (hot
window), folds evicted old turns into the 22 KB Q14 Capsule through the exact
integer student writer, and injects a compact recalled digest into the system
prompt.

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

    ``checkpoint`` may be:
      - a ``weights.bin`` file or a directory containing ``weights.bin``
        (canonical 3.6 MB Q14 binary, torch-free),
      - a directory of pre-exported numpy arrays (``np_key_w.npy`` etc.,
        torch-free),
      - a training directory with ``student.pt`` (requires torch), or
      - ``None`` / ``"default"`` / a missing path when ``allow_deterministic_fallback``
        is enabled: falls back to a fixed-seed deterministic Q14 writer so
        first-time users can boot the proxy with zero external downloads
        (still producing exact 22 KB capsules and valid SHA-256 receipts).
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        allow_deterministic_fallback: bool = True,
    ) -> None:
        from ..student_int import IntStudentWriter

        self.weights_source = "checkpoint"
        if checkpoint is None or str(checkpoint).strip() in ("", "none", "default"):
            if not allow_deterministic_fallback:
                raise FileNotFoundError("No MemoChain checkpoint specified")
            self.inner = _default_deterministic_writer()
            self.weights_source = "deterministic-fallback"
            self._digest = self.inner.weights_digest()
            return

        path = Path(checkpoint)
        weights_bin = path if path.is_file() and path.name.endswith(".bin") else path / "weights.bin"
        np_key_w = path / "np_key_w.npy"
        student_pt = path if path.is_file() and path.name.endswith(".pt") else path / "student.pt"

        if weights_bin.is_file():
            self.inner = IntStudentWriter.from_bytes(weights_bin.read_bytes())
            self.weights_source = str(weights_bin)
        elif np_key_w.exists():  # torch-free numpy checkpoint
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
            self.weights_source = str(path)
        elif student_pt.is_file():
            self.inner = IntStudentWriter.from_checkpoint(student_pt)
            self.weights_source = str(student_pt)
        elif allow_deterministic_fallback:
            import logging

            logging.getLogger("memochain.server").warning(
                "Checkpoint %s not found; using built-in deterministic Q14 writer "
                "(capsules and SHA-256 receipts remain exact; mount weights.bin or "
                "np_*.npy under MEMOCHAIN_CHECKPOINT for trained neural-reader weights).",
                path,
            )
            self.inner = _default_deterministic_writer()
            self.weights_source = "deterministic-fallback"
        else:
            raise FileNotFoundError(
                f"MemoChain checkpoint not found at {path} "
                "(expected weights.bin, np_key_w.npy, or student.pt)"
            )
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


def _default_deterministic_writer():
    """Built-in fixed-seed Q14 writer for zero-download hybrid proxy starts.

    Produces canonical 22,528-byte capsules and SHA-256 receipt chains that
    verify identically in Python and Rust without requiring a pre-trained
    checkpoint file. For trained neural-reader decoding, supply ``weights.bin``,
    ``np_*.npy``, or ``student.pt`` via ``--checkpoint`` / ``MEMOCHAIN_CHECKPOINT``.
    """
    from ..student_int import EMBED_DIM, KEY_PART, N_HEADS, Q_ONE, IntStudentWriter

    rng = np.random.default_rng(20260417)
    scale = Q_ONE / np.sqrt(EMBED_DIM)
    w_key = np.clip(np.round(rng.standard_normal((KEY_PART, EMBED_DIM)) * scale), -32768, 32767).astype("<i2")
    b_key = np.zeros(KEY_PART, dtype="<i2")
    w_val = np.clip(np.round(rng.standard_normal((KEY_PART, EMBED_DIM)) * scale), -32768, 32767).astype("<i2")
    b_val = np.zeros(KEY_PART, dtype="<i2")
    beta = np.full(N_HEADS, int(round(0.25 * Q_ONE)), dtype="<i2")
    decay = np.full(N_HEADS, int(round(0.95 * Q_ONE)), dtype="<i2")
    return IntStudentWriter(w_key=w_key, b_key=b_key, w_val=w_val, b_val=b_val, beta=beta, decay=decay)


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


def _merge_anthropic_system(
    system_field: Any, memory_block: str, *, cache_control: bool = False
) -> Any:
    """Append the static memory block to Anthropic's top-level ``system``.

    When ``cache_control=True``, returns structured text blocks with an
    ``ephemeral`` cache breakpoint on the static prefix so Anthropic caches
    system prompt + durable fact sheet across turns between folds.
    """
    if not memory_block:
        return system_field
    if not cache_control:
        if system_field is None:
            return memory_block
        if isinstance(system_field, str):
            return f"{system_field}\n\n{memory_block}" if system_field else memory_block
        if isinstance(system_field, list):
            return list(system_field) + [{"type": "text", "text": memory_block}]
        return memory_block

    blocks: list[dict[str, Any]] = []
    if isinstance(system_field, str) and system_field:
        blocks.append({"type": "text", "text": system_field})
    elif isinstance(system_field, list):
        for item in system_field:
            if isinstance(item, dict):
                blocks.append(dict(item))
            elif isinstance(item, str) and item:
                blocks.append({"type": "text", "text": item})
    blocks.append(
        {"type": "text", "text": memory_block, "cache_control": {"type": "ephemeral"}}
    )
    return blocks


def _is_anthropic_target(backend_base_url: str, model: str | None = None) -> bool:
    url = (backend_base_url or "").lower()
    if "api.anthropic.com" in url:
        return True
    if "openrouter.ai" in url and isinstance(model, str) and "anthropic/" in model.lower():
        return True
    return False


def _is_openai_direct_target(backend_base_url: str) -> bool:
    return "api.openai.com" in (backend_base_url or "").lower()


def _is_gemini_target(backend_base_url: str, model: str | None = None) -> bool:
    url = (backend_base_url or "").lower()
    if "generativelanguage.googleapis.com" in url or "aiplatform.googleapis.com" in url:
        return True
    if "openrouter.ai" in url and isinstance(model, str) and "google/" in model.lower():
        return True
    return False


def _detect_backend_kind(backend_base_url: str) -> str:
    """Classify backend URL into a known provider kind (cloud vs local)."""
    url = (backend_base_url or "").lower()
    if "api.anthropic.com" in url:
        return "anthropic"
    if "api.openai.com" in url:
        return "openai"
    if "openrouter.ai" in url:
        return "openrouter"
    if "generativelanguage.googleapis.com" in url or "aiplatform.googleapis.com" in url:
        return "gemini"
    if any(h in url for h in ("api.deepseek.com", "api.x.ai", "api.groq.com", "api.together.xyz", "api.fireworks.ai")):
        return "cloud_openai"
    return "local"


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "cloud": {
        "hot_limit": 96000,
        "keep_min": 32000,
        "cache_passthrough": True,
        "cache_hints": True,
        "compact_tool_outputs": True,
        "async_fold": True,
        "cheap_fold_routing": True,
    },
    "local": {
        "hot_limit": 28000,
        "keep_min": 12000,
        "cache_passthrough": False,
        "cache_hints": False,
        "compact_tool_outputs": True,
        "async_fold": True,
        "cheap_fold_routing": False,
    },
}


def resolve_profile_config(
    backend_base_url: str,
    profile: str = "auto",
) -> dict[str, Any]:
    """Resolve effective profile ('cloud' or 'local') and tuned defaults."""
    kind = _detect_backend_kind(backend_base_url)
    prof = (profile or "auto").strip().lower()
    if prof == "auto":
        effective = "local" if kind == "local" else "cloud"
    elif prof in _PROFILE_DEFAULTS:
        effective = prof
    else:
        raise ValueError(f"Unknown profile {profile!r}; expected 'auto', 'cloud', or 'local'")
    cfg = dict(_PROFILE_DEFAULTS[effective])
    cfg["profile"] = effective
    cfg["backend_kind"] = kind
    return cfg


def suggest_cheap_fold_model(backend_base_url: str, request_model: str | None) -> str | None:
    """Suggest a 10-20x cheaper, fast non-reasoning model on the SAME backend
    for hybrid fold fact-sheet extraction when the main chat uses a frontier model.

    Returns None when the request model is already cheap/fast, unknown, or local.
    """
    if not request_model:
        return None
    kind = _detect_backend_kind(backend_base_url)
    m = request_model.strip().lower()
    if kind == "local":
        return None
    # Already a cheap/mini/flash/haiku tier -> keep as-is (token-bounded so 'gemini' != 'mini')
    import re as _re

    if _re.search(r"(?:^|[-_/.])(haiku|mini|nano|flash|flash-lite|8b|7b)(?:$|[-_/.])", m):
        return None
    if kind == "anthropic":
        if "opus" in m or "sonnet" in m:
            return "claude-haiku-4-5"
    elif kind == "openai":
        if any(tok in m for tok in ("gpt-5", "gpt-4o", "gpt-4.1", "o1", "o3", "o4")):
            return "gpt-4o-mini"
    elif kind == "gemini":
        if "pro" in m:
            return "gemini-2.5-flash"
    elif kind == "openrouter":
        if m.startswith("anthropic/") and ("opus" in m or "sonnet" in m):
            return "anthropic/claude-haiku-4.5"
        if m.startswith("openai/") and any(tok in m for tok in ("gpt-5", "gpt-4o", "gpt-4.1", "o1", "o3", "o4")):
            return "openai/gpt-4o-mini"
        if m.startswith("google/") and "pro" in m:
            return "google/gemini-2.5-flash"
    return None


def _apply_anthropic_message_cache_breakpoint(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place an Anthropic ``cache_control`` breakpoint on the last stable
    pre-turn non-system message (the second-to-last message, i.e. right before
    the newest user turn) when no explicit cache breakpoint is already present
    on any non-system message.
    """
    if len(messages) < 2:
        return messages
    # Respect caller-supplied cache_control breakpoints on non-system messages.
    for m in messages:
        if m.get("role") == "system":
            continue
        c = m.get("content")
        if isinstance(c, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in c
        ):
            return messages
    out = [dict(m) for m in messages]
    idx = len(out) - 2
    if out[idx].get("role") == "system":
        return out
    target = dict(out[idx])
    content = target.get("content")
    if isinstance(content, str) and content:
        target["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
        out[idx] = target
    elif isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        last_block = blocks[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = {"type": "ephemeral"}
            target["content"] = blocks
            out[idx] = target
    return out


def _apply_openai_cache_hints(
    out_body: dict[str, Any],
    backend_base_url: str,
    session_id: str,
    *,
    stream: bool,
    cache_hints: bool,
) -> None:
    """Attach safe provider cache hints on OpenAI-compatible requests."""
    if not cache_hints:
        return
    if _is_openai_direct_target(backend_base_url):
        out_body.setdefault("prompt_cache_key", f"memochain:{session_id}")
        if stream:
            opts = dict(out_body.get("stream_options") or {})
            opts.setdefault("include_usage", True)
            out_body["stream_options"] = opts
    elif _is_anthropic_target(backend_base_url, out_body.get("model")):
        # OpenRouter Anthropic routes honor Anthropic cache_control on system
        # and message content blocks.
        msgs = out_body.get("messages")
        if isinstance(msgs, list) and msgs:
            out_msgs = [dict(m) for m in msgs]
            if out_msgs[0].get("role") == "system":
                sys_c = out_msgs[0].get("content")
                if isinstance(sys_c, str) and sys_c:
                    out_msgs[0]["content"] = [
                        {
                            "type": "text",
                            "text": sys_c,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(sys_c, list) and sys_c:
                    blocks = [dict(b) if isinstance(b, dict) else b for b in sys_c]
                    if isinstance(blocks[-1], dict) and "cache_control" not in blocks[-1]:
                        blocks[-1]["cache_control"] = {"type": "ephemeral"}
                    out_msgs[0]["content"] = blocks
            out_body["messages"] = _apply_anthropic_message_cache_breakpoint(out_msgs)
        if stream:
            opts = dict(out_body.get("stream_options") or {})
            opts.setdefault("include_usage", True)
            out_body["stream_options"] = opts
    elif _is_gemini_target(backend_base_url, out_body.get("model")) and stream:
        # Gemini OpenAI-compat endpoint returns usage in SSE when requested;
        # prefix caching activates implicitly on stable leading blocks.
        opts = dict(out_body.get("stream_options") or {})
        opts.setdefault("include_usage", True)
        out_body["stream_options"] = opts
    elif "openrouter.ai" in (backend_base_url or "").lower() and stream:
        opts = dict(out_body.get("stream_options") or {})
        opts.setdefault("include_usage", True)
        out_body["stream_options"] = opts


class HybridReaderAdapter:
    """Reader-free Option A digest: asks the active backend model to update a
    compact structured fact sheet from newly folded turns.

    Runs inside the slim numpy-only container with zero extra GPU memory and
    zero reader-model boot. Supports OpenAI ``/chat/completions`` and Anthropic
    ``/messages`` backends (direct Anthropic API, OpenAI, Gemini OpenAI-compat,
    OpenRouter, or local servers). Optionally routes fold calls to a cheaper
    fast tier on the same backend (with automatic fallback to ``model_hint``).
    """

    def __init__(
        self,
        backend_base_url: str,
        backend_api_key: str | None = None,
        max_words: int = 220,
        timeout: float = 60.0,
        reader_model: str | None = None,
        max_tokens: int = 4096,
        cheap_fold_routing: bool = False,
    ) -> None:
        headers = _backend_auth_headers(backend_base_url, backend_api_key)
        self.backend_base_url = backend_base_url
        self.client = httpx.Client(base_url=backend_base_url, timeout=timeout, headers=headers)
        self.max_words = max_words
        self.max_tokens = max_tokens
        self._pinned_model: str | None = reader_model
        self._model_id: str | None = reader_model
        self._prefer_messages = "api.anthropic.com" in backend_base_url
        self.cheap_fold_routing = bool(cheap_fold_routing)
        self.last_call_fallback: bool = False
        self.last_used_model: str | None = None
        self.fallback_count: int = 0

    def _pick_model(
        self, model_hint: str | None = None, auth_headers: dict[str, str] | None = None
    ) -> str | None:
        if self._pinned_model:
            return self._pinned_model
        if model_hint:
            if self.cheap_fold_routing:
                cheap = suggest_cheap_fold_model(self.backend_base_url, model_hint)
                if cheap:
                    return cheap
            return model_hint
        if self._model_id:
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

    def _invoke_model(
        self, model: str, prompt: str, auth_headers: dict[str, str] | None = None
    ) -> str:
        if self._prefer_messages:
            return self._call_messages(model, prompt, auth_headers=auth_headers)
        return self._call_chat_completions(model, prompt, auth_headers=auth_headers)

    def read_fold(
        self,
        state,
        query_embed,
        prev_summary: str,
        folded_entries: list[tuple[str, str, str]],
        model_hint: str | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> str:
        self.last_call_fallback = False
        self.last_used_model = None
        if not folded_entries:
            return prev_summary
        model = self._pick_model(model_hint=model_hint, auth_headers=auth_headers)
        if not model:
            self.last_call_fallback = True
            self.fallback_count += 1
            return prev_summary or _profile_digest(state, query_embed, prev_summary)

        turns = "\n".join(f"[{role}] {text}" for _, role, text in folded_entries)
        prompt = (
            f"Update the bullet-point fact sheet below using the new transcript lines "
            f"(max {self.max_words} words). Preserve ALL existing concrete facts, names, "
            f"cities, ports, hosts, URLs, codes, package/tool names, file paths, decisions, "
            f"and preferences verbatim unless explicitly superseded by a newer turn. "
            f"Output only `- key: value` bullet points with no preamble or commentary.\n\n"
            f"Existing notes:\n{prev_summary or '(none)'}\n\n"
            f"Transcript lines:\n{turns}"
        )
        candidates = [model]
        if model_hint and model != model_hint and not self._pinned_model:
            # If the auto-routed cheap model is unavailable/rejected, fall back to active model_hint
            candidates.append(model_hint)
        for cand in candidates:
            try:
                text = self._invoke_model(cand, prompt, auth_headers=auth_headers)
                if text:
                    self.last_used_model = cand
                    return text
            except Exception:
                continue
        self.last_call_fallback = True
        self.fallback_count += 1
        return prev_summary or _profile_digest(state, query_embed, prev_summary)

    def _call_chat_completions(
        self, model: str, prompt: str, auth_headers: dict[str, str] | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
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
                "max_tokens": self.max_tokens,
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
# Cache-aware telemetry & passthrough
# --------------------------------------------------------------------------

_CACHE_HIT_KEYS = (
    "cache_read_input_tokens",
    "cached_tokens",
    "prompt_cache_hit_tokens",
    "cache_read_tokens",
    "cachedContentTokenCount",
)


def _extract_usage_from_payload(payload: Any) -> dict[str, Any] | None:
    """Extract a usage dict from OpenAI, Anthropic, Gemini, or SSE event payloads."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if isinstance(usage, dict) and usage:
        return usage
    # Anthropic SSE `message_start` nests usage under `message.usage`
    msg = payload.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("usage"), dict) and msg["usage"]:
        return msg["usage"]
    # Gemini native responses expose `usageMetadata`
    umeta = payload.get("usageMetadata")
    if isinstance(umeta, dict) and umeta:
        return umeta
    return None


def _cache_hit_ratio(payload: Any) -> float | None:
    """Return (cached / total_input) for OpenAI, Anthropic, DeepSeek, OpenRouter,
    or Gemini responses (non-streaming or SSE event), or None if unreported."""
    usage = _extract_usage_from_payload(payload)
    if not isinstance(usage, dict):
        return None

    # OpenAI / OpenRouter structured prompt_tokens_details
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    cached: float | None = None
    if isinstance(details, dict):
        for k in ("cached_tokens", "cached_tokens_count", "cache_read_tokens"):
            v = details.get(k)
            if isinstance(v, (int, float)):
                cached = float(v)
                break
    if cached is None:
        for key in _CACHE_HIT_KEYS:
            v = usage.get(key)
            if isinstance(v, (int, float)):
                cached = float(v)
                break
    if cached is None:
        return None

    # Anthropic's `input_tokens` excludes cache_read + cache_creation tokens;
    # sum all three so the ratio is cached / total_prompt_tokens.
    if "cache_read_input_tokens" in usage and "prompt_tokens" not in usage:
        inp = (
            float(usage.get("input_tokens") or 0)
            + float(usage.get("cache_read_input_tokens") or 0)
            + float(usage.get("cache_creation_input_tokens") or 0)
        )
    else:
        inp = float(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )
    if inp <= 0:
        return None
    return min(1.0, max(0.0, cached / inp))


def _record_cache_hit(
    session_state: dict[str, Any],
    hit_ratio: float | None,
    threshold: float,
) -> None:
    """Update per-session cache telemetry and consecutive high-hit streak."""
    session_state["last_cache_hit"] = hit_ratio
    if hit_ratio is not None and hit_ratio >= threshold:
        session_state["passthrough_streak"] = session_state.get("passthrough_streak", 0) + 1
    else:
        session_state["passthrough_streak"] = 0


def _is_passthrough(session_state: dict[str, Any], streak_needed: int) -> bool:
    return session_state.get("passthrough_streak", 0) >= max(1, int(streak_needed))


def _reset_passthrough(session_state: dict[str, Any]) -> None:
    session_state["passthrough_streak"] = 0
    session_state["last_cache_hit"] = None


def _extract_token_counts(payload: Any) -> tuple[int, int, float | None]:
    """Return (total_input_tokens, cached_input_tokens, provider_cost_usd_or_None)."""
    usage = _extract_usage_from_payload(payload)
    if not isinstance(usage, dict):
        return 0, 0, None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    cached = 0
    if isinstance(details, dict):
        for k in ("cached_tokens", "cached_tokens_count", "cache_read_tokens"):
            v = details.get(k)
            if isinstance(v, (int, float)):
                cached = int(v)
                break
    if cached == 0:
        for key in _CACHE_HIT_KEYS:
            v = usage.get(key)
            if isinstance(v, (int, float)):
                cached = int(v)
                break
    if "cache_read_input_tokens" in usage and "prompt_tokens" not in usage:
        inp = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
        )
    else:
        inp = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )
    raw_cost = usage.get("cost")
    cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
    return max(0, inp), max(0, min(inp, cached)), cost


def _record_session_economics(
    session_state: dict[str, Any],
    raw_prompt_tokens: int,
    sent_prompt_tokens: int,
    usage_payload: Any,
) -> dict[str, Any]:
    """Accumulate per-session token, cache, and cost-savings telemetry."""
    econ = session_state.setdefault(
        "economics",
        {
            "turns": 0,
            "raw_input_tokens": 0,
            "proxied_input_tokens": 0,
            "cached_input_tokens": 0,
            "reported_cost_usd": 0.0,
            "has_reported_cost": False,
        },
    )
    inp, cached, cost = _extract_token_counts(usage_payload)
    actual_in = inp if inp > 0 else sent_prompt_tokens
    econ["turns"] += 1
    econ["raw_input_tokens"] += max(actual_in, int(raw_prompt_tokens))
    econ["proxied_input_tokens"] += actual_in
    econ["cached_input_tokens"] += cached
    if cost is not None:
        econ["reported_cost_usd"] += cost
        econ["has_reported_cost"] = True
    return _economics_snapshot(econ)


def _economics_snapshot(econ: dict[str, Any]) -> dict[str, Any]:
    raw_in = max(1, int(econ.get("raw_input_tokens") or 0))
    prox_in = int(econ.get("proxied_input_tokens") or 0)
    cached = int(econ.get("cached_input_tokens") or 0)
    uncached = max(0, prox_in - cached)
    saved_tokens = max(0, raw_in - prox_in)
    # Effective uncached-equivalent tokens (assuming standard 0.1x cache read price)
    eff_prox = uncached + 0.1 * cached
    out: dict[str, Any] = {
        "turns": int(econ.get("turns") or 0),
        "raw_input_tokens": int(econ.get("raw_input_tokens") or 0),
        "proxied_input_tokens": prox_in,
        "cached_input_tokens": cached,
        "tokens_saved": saved_tokens,
        "token_savings_pct": round(100.0 * saved_tokens / raw_in, 1),
        "cache_hit_pct": round(100.0 * cached / max(1, prox_in), 1),
        "effective_input_tokens": int(round(eff_prox)),
    }
    if econ.get("has_reported_cost"):
        out["reported_cost_usd"] = round(float(econ["reported_cost_usd"]), 6)
    return out


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
    hot_limit: int | None = None,
    keep_min: int | None = None,
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
    reader_max_tokens: int = 4096,
    max_doc_chars: int = 2000,
    profile: str | None = None,
    cache_passthrough: bool | None = None,
    cache_passthrough_threshold: float = 0.5,
    cache_passthrough_streak: int = 2,
    cache_hints: bool | None = None,
    async_fold: bool | None = None,
    compact_tool_outputs: bool | None = None,
    tool_archive_min_chars: int = 1500,
    tool_keep_recent_groups: int = 4,
    cheap_fold_routing: bool | None = None,
) -> "FastAPI":
    if not FASTAPI_OK:  # pragma: no cover
        raise ImportError("fastapi is required for the proxy: uv sync --extra proxy")

    prof_cfg = resolve_profile_config(backend_base_url, profile or "auto")
    active_profile = prof_cfg["profile"]
    backend_kind = prof_cfg["backend_kind"]

    # When profile is explicitly passed ("auto"/"cloud"/"local"), profile defaults
    # fill any unset knobs; when profile is None (direct programmatic create_app),
    # keep legacy defaults so existing tests/callers behave identically.
    use_profile = profile is not None
    eff_hot_limit = (
        hot_limit
        if hot_limit is not None
        else (prof_cfg["hot_limit"] if use_profile else 80000)
    )
    eff_keep_min = (
        keep_min
        if keep_min is not None
        else (prof_cfg["keep_min"] if use_profile else 40000)
    )
    eff_cache_passthrough = (
        cache_passthrough
        if cache_passthrough is not None
        else (prof_cfg["cache_passthrough"] if use_profile else True)
    )
    eff_cache_hints = (
        cache_hints
        if cache_hints is not None
        else (prof_cfg["cache_hints"] if use_profile else True)
    )
    eff_async_fold = (
        async_fold
        if async_fold is not None
        else (prof_cfg["async_fold"] if use_profile else False)
    )
    eff_compact_tools = (
        compact_tool_outputs
        if compact_tool_outputs is not None
        else (prof_cfg["compact_tool_outputs"] if use_profile else True)
    )
    eff_cheap_fold = (
        cheap_fold_routing
        if cheap_fold_routing is not None
        else (prof_cfg["cheap_fold_routing"] if use_profile else False)
    )

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
                max_tokens=reader_max_tokens,
                cheap_fold_routing=eff_cheap_fold,
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
        hot_limit=eff_hot_limit,
        keep_min=eff_keep_min,
        recall_top_k=recall_top_k,
        max_chunks=max_chunks,
        max_doc_chars=max_doc_chars,
        async_fold=eff_async_fold,
        compact_tool_outputs=eff_compact_tools,
        tool_archive_min_chars=tool_archive_min_chars,
        tool_keep_recent_groups=tool_keep_recent_groups,
    )

    backend_headers = _backend_auth_headers(backend_base_url, backend_api_key)
    app = FastAPI(title="MemoChain memory proxy", version="0.2.0")
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

    # per-session cache passthrough state
    _passthrough_state: dict[str, dict[str, Any]] = {}

    def _session_state(sid: str | None) -> dict[str, Any]:
        return _passthrough_state.setdefault(sid or "__default__", {})

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
            "backend_kind": backend_kind,
            "profile": active_profile,
            "hot_limit": eff_hot_limit,
            "keep_min": eff_keep_min,
            "async_fold": eff_async_fold,
            "cheap_fold_routing": eff_cheap_fold,
            "compact_tool_outputs": eff_compact_tools,
            "cache_hints": eff_cache_hints,
            "cache_passthrough": eff_cache_passthrough,
            "sessions": len(sessions.sessions),
            "proof_mode": proof_mode,
            "weights_source": getattr(writer, "weights_source", "custom"),
            "weights_digest": writer.weights_digest()[:16],
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
        raw_prompt_tokens = sum(estimate_tokens(message_text(m)) for m in messages)
        mem = sessions.get(session_id, messages)

        # normalize any requested model name to the backend's served model,
        # so clients with a stale/alias model id still work
        requested = body.get("model")
        backend_ids = await _backend_model_ids()
        if backend_ids and requested not in backend_ids:
            body["model"] = backend_ids[0]

        req_headers = _forward_auth_headers(request, backend_base_url)
        sess_state = _session_state(mem.session_id)
        cache_hit = (
            _is_passthrough(sess_state, cache_passthrough_streak)
            if eff_cache_passthrough
            else False
        )
        rewritten, stats = mem.process(
            messages,
            model_hint=body.get("model"),
            auth_headers=req_headers,
            passthrough=cache_hit,
        )
        sent_prompt_tokens = sum(estimate_tokens(message_text(m)) for m in rewritten)
        if sess_state.get("last_cache_hit") is not None:
            stats["last_cache_hit_ratio"] = round(float(sess_state["last_cache_hit"]), 4)
        out_body = {k: v for k, v in body.items() if k not in ("session_id",)}
        out_body["messages"] = rewritten
        stream = bool(out_body.get("stream"))
        _apply_openai_cache_hints(
            out_body,
            backend_base_url,
            mem.session_id,
            stream=stream,
            cache_hints=eff_cache_hints,
        )

        if not stream:
            resp = await client.post("/chat/completions", json=out_body, headers=req_headers)
            if resp.status_code != 200:
                return _error_json_response(resp.text, resp.status_code)
            payload = resp.json()
            hit = _cache_hit_ratio(payload)
            if eff_cache_passthrough:
                _record_cache_hit(sess_state, hit, cache_passthrough_threshold)
            elif hit is not None:
                sess_state["last_cache_hit"] = hit
            if hit is not None:
                stats["cache_hit_ratio"] = round(hit, 4)
            stats["economics"] = _record_session_economics(
                sess_state, raw_prompt_tokens, sent_prompt_tokens, payload
            )
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
                usage_chunk: dict[str, Any] | None = None
                stream_hit: float | None = None
                saw_usage = False
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            if eff_cache_passthrough and saw_usage:
                                _record_cache_hit(sess_state, stream_hit, cache_passthrough_threshold)
                            elif stream_hit is not None:
                                sess_state["last_cache_hit"] = stream_hit
                            if stream_hit is not None:
                                stats["cache_hit_ratio"] = round(stream_hit, 4)
                            stats["economics"] = _record_session_economics(
                                sess_state, raw_prompt_tokens, sent_prompt_tokens, usage_chunk
                            )
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
                                if _extract_usage_from_payload(parsed) is not None:
                                    saw_usage = True
                                    usage_chunk = parsed
                                    hit = _cache_hit_ratio(parsed)
                                    if hit is not None:
                                        stream_hit = hit
                        except ValueError:
                            pass
                if eff_cache_passthrough and saw_usage:
                    _record_cache_hit(sess_state, stream_hit, cache_passthrough_threshold)
                _record_session_economics(
                    sess_state, raw_prompt_tokens, sent_prompt_tokens, usage_chunk
                )

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
        raw_prompt_tokens = sum(estimate_tokens(message_text(m)) for m in messages)
        mem = sessions.get(session_id, messages)

        requested = body.get("model")
        backend_ids = await _backend_model_ids()
        if backend_ids and requested not in backend_ids:
            body["model"] = backend_ids[0]

        req_headers = _forward_anthropic_headers(request, backend_base_url)
        sess_state = _session_state(mem.session_id)
        cache_hit = (
            _is_passthrough(sess_state, cache_passthrough_streak)
            if eff_cache_passthrough
            else False
        )
        rewritten, stats = mem.process(
            messages,
            model_hint=body.get("model"),
            inject_memory=False,
            auth_headers=req_headers,
            passthrough=cache_hit,
        )
        if sess_state.get("last_cache_hit") is not None:
            stats["last_cache_hit_ratio"] = round(float(sess_state["last_cache_hit"]), 4)
        out_body = {k: v for k, v in body.items() if k not in ("session_id",)}
        # Cache-safe split injection for Anthropic /v1/messages:
        #   - static fact sheet -> top-level `system` (with ephemeral cache_control when targeting Anthropic)
        #   - dynamic per-turn recall -> tail user message (never invalidates system/prefix cache)
        static_block = stats.get("static_memory_block", "")
        dynamic_block = stats.get("dynamic_memory_block", "")
        if dynamic_block:
            rewritten = SessionMemory.inject_dynamic_recall(rewritten, dynamic_block)
        ant_cache = bool(eff_cache_hints and _is_anthropic_target(backend_base_url, body.get("model")))
        if static_block:
            out_body["system"] = _merge_anthropic_system(
                out_body.get("system"), static_block, cache_control=ant_cache
            )
        if ant_cache:
            rewritten = _apply_anthropic_message_cache_breakpoint(rewritten)
        out_body["messages"] = rewritten
        sent_prompt_tokens = sum(estimate_tokens(message_text(m)) for m in rewritten)

        stream = bool(out_body.get("stream"))

        if not stream:
            resp = await client.post("/messages", json=out_body, headers=req_headers)
            if resp.status_code != 200:
                return _anthropic_error_response(resp.text, resp.status_code)
            payload = resp.json()
            hit = _cache_hit_ratio(payload)
            if eff_cache_passthrough:
                _record_cache_hit(sess_state, hit, cache_passthrough_threshold)
            elif hit is not None:
                sess_state["last_cache_hit"] = hit
            if hit is not None:
                stats["cache_hit_ratio"] = round(hit, 4)
            stats["economics"] = _record_session_economics(
                sess_state, raw_prompt_tokens, sent_prompt_tokens, payload
            )
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
                stream_hit: float | None = None
                usage_chunk: dict[str, Any] | None = None
                saw_usage = False
                async for line in resp.aiter_lines():
                    yield f"{line}\n".encode()
                    if line.startswith("data: "):
                        try:
                            parsed = json.loads(line[6:])
                            if isinstance(parsed, dict) and _extract_usage_from_payload(parsed) is not None:
                                saw_usage = True
                                usage_chunk = parsed
                                hit = _cache_hit_ratio(parsed)
                                if hit is not None:
                                    stream_hit = hit
                        except ValueError:
                            pass
                if eff_cache_passthrough and saw_usage:
                    _record_cache_hit(sess_state, stream_hit, cache_passthrough_threshold)
                elif stream_hit is not None:
                    sess_state["last_cache_hit"] = stream_hit
                _record_session_economics(
                    sess_state, raw_prompt_tokens, sent_prompt_tokens, usage_chunk
                )

        return StreamingResponse(gen_anthropic(), media_type="text/event-stream")

    @app.get("/v1/memochain/status")
    async def status(session_id: str) -> dict[str, Any]:
        mem = sessions.get_by_id(session_id)
        st = mem.verify()
        st["profile"] = active_profile
        st["backend_kind"] = backend_kind
        st["summary"] = mem.summary
        st["folded_messages"] = len(mem.folded_hashes)
        st["fold_fallback_count"] = mem.fold_fallback_count
        st["fold_shrink_guard_count"] = mem.fold_shrink_guard_count
        st["last_fold_fallback"] = mem.last_fold_fallback
        st["fold_in_progress"] = mem._fold_in_progress
        st["compacted_tools_count"] = len(mem.compacted_stubs)
        sess_state = _session_state(mem.session_id)
        if "economics" in sess_state:
            st["economics"] = _economics_snapshot(sess_state["economics"])
        if mem.chunk_store is not None:
            st["chunks_count"] = mem.chunk_store.count()
            st["facts_count"] = mem.chunk_store.facts_count()
            st["fts_enabled"] = mem.chunk_store.fts_enabled
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
        for mem in list(sessions.sessions.values()):
            mem.wait_for_pending_fold(timeout=15.0)
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
    out = {
        "session_id": stats.get("session_id"),
        "step": mem.head.step if mem.head else 0,
        "capsule_bytes": mem.capsule_path.stat().st_size if mem.capsule_path.exists() else 0,
        "state_digest": mem.head.state_digest[:16] if mem.head else "",
        "folded_messages": len(mem.folded_hashes),
        "fold_now": stats.get("fold"),
        "hot_messages": stats.get("hot_messages"),
        "hot_tokens": stats.get("hot_tokens"),
        "recalled_chunks": stats.get("recalled_chunks", 0),
        "fold_fallback_count": mem.fold_fallback_count,
        "fold_shrink_guard_count": mem.fold_shrink_guard_count,
        **({"tool_compactions": stats["tool_compactions"]} if stats.get("tool_compactions") else {}),
        **({"passthrough": True} if stats.get("passthrough") else {}),
    }
    if "cache_hit_ratio" in stats:
        out["cache_hit_ratio"] = stats["cache_hit_ratio"]
    elif "last_cache_hit_ratio" in stats:
        out["last_cache_hit_ratio"] = stats["last_cache_hit_ratio"]
    if "economics" in stats:
        out["economics"] = stats["economics"]
    return out


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
    parser.add_argument(
        "--checkpoint",
        default=_env("MEMOCHAIN_CHECKPOINT", ""),
        help="path to weights.bin, np_*.npy dir, or student.pt dir (omit for built-in deterministic Q14 writer)",
    )
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
        "--max-doc-chars",
        type=int,
        default=int(_env("MEMOCHAIN_MAX_DOC_CHARS", "2000")),
        help="chars kept per stored chunk before embedding (default 2000; raise for long notes so facts deep in a document stay recallable)",
    )
    parser.add_argument(
        "--reader-sidecar-url",
        default=_env("MEMOCHAIN_READER_SIDECAR_URL"),
        help="reader sidecar base URL (e.g. http://localhost:8002) for natural-language memory notes",
    )
    parser.add_argument(
        "--profile",
        default=_env("MEMOCHAIN_PROFILE", "auto"),
        choices=["auto", "cloud", "local"],
        help="tuning profile: auto (detect from backend URL), cloud (96k/32k, cache hints + cheap fold model), or local (28k/12k, VRAM/TTFT optimized)",
    )
    _hl = _env("MEMOCHAIN_HOT_LIMIT")
    parser.add_argument(
        "--hot-limit",
        type=int,
        default=int(_hl) if _hl is not None else None,
        help="conversation hot-window token limit (defaults from --profile: 96000 cloud / 28000 local)",
    )
    _km = _env("MEMOCHAIN_KEEP_MIN")
    parser.add_argument(
        "--keep-min",
        type=int,
        default=int(_km) if _km is not None else None,
        help="min conversation tokens kept hot after folding (defaults from --profile: 32000 cloud / 12000 local)",
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
        help="model id for hybrid fold summaries (defaults to active request model or backend /models)",
    )
    parser.add_argument(
        "--reader-max-tokens",
        type=int,
        default=int(_env("MEMOCHAIN_READER_MAX_TOKENS", "4096")),
        help="max_tokens for hybrid fold-summary call (default 4096 to give reasoning models headroom)",
    )
    _cp = _env("MEMOCHAIN_CACHE_PASSTHROUGH")
    parser.add_argument(
        "--cache-passthrough",
        action=argparse.BooleanOptionalAction,
        default=(_cp.lower() not in ("0", "false", "no", "off")) if _cp is not None else None,
        help="stop folding/injecting when backend reports high prompt-cache hit rate (default from --profile)",
    )
    parser.add_argument(
        "--cache-passthrough-threshold",
        type=float,
        default=float(_env("MEMOCHAIN_CACHE_PASSTHROUGH_THRESHOLD", "0.5")),
        help="cache hit ratio required to enter passthrough (default 0.5)",
    )
    parser.add_argument(
        "--cache-passthrough-streak",
        type=int,
        default=int(_env("MEMOCHAIN_CACHE_PASSTHROUGH_STREAK", "2")),
        help="consecutive high-hit responses required before passthrough (default 2)",
    )
    _ch = _env("MEMOCHAIN_CACHE_HINTS")
    parser.add_argument(
        "--cache-hints",
        action=argparse.BooleanOptionalAction,
        default=(_ch.lower() not in ("0", "false", "no", "off")) if _ch is not None else None,
        help="attach provider cache hints (Anthropic cache_control, OpenAI prompt_cache_key, stream usage; default from --profile)",
    )
    _af = _env("MEMOCHAIN_ASYNC_FOLD")
    parser.add_argument(
        "--async-fold",
        action=argparse.BooleanOptionalAction,
        default=(_af.lower() not in ("0", "false", "no", "off")) if _af is not None else None,
        help="run fold summary calls in a background thread so fold turns add zero latency (default from --profile: enabled)",
    )
    _tc = _env("MEMOCHAIN_TOOL_COMPACTION")
    parser.add_argument(
        "--tool-compaction",
        action=argparse.BooleanOptionalAction,
        default=(_tc.lower() not in ("0", "false", "no", "off")) if _tc is not None else None,
        help="compact superseded file reads and old oversized tool outputs at overflow boundaries (default from --profile: enabled)",
    )
    parser.add_argument(
        "--tool-archive-min-chars",
        type=int,
        default=int(_env("MEMOCHAIN_TOOL_ARCHIVE_MIN_CHARS", "1500")),
        help="min char size before an old tool output is archived into ChunkStore at overflow boundaries (default 1500)",
    )
    _cfr = _env("MEMOCHAIN_CHEAP_FOLD_ROUTING")
    parser.add_argument(
        "--cheap-fold-routing",
        action=argparse.BooleanOptionalAction,
        default=(_cfr.lower() not in ("0", "false", "no", "off")) if _cfr is not None else None,
        help="route cloud fold summaries to a fast cheap model on the same provider (default: enabled on cloud profile)",
    )
    parser.add_argument("--host", default=_env("MEMOCHAIN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env("MEMOCHAIN_PORT", "9000")))
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        backend_base_url=args.backend_base_url,
        backend_api_key=args.backend_api_key,
        checkpoint=args.checkpoint or None,
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
        max_doc_chars=args.max_doc_chars,
        reader_model=args.reader_model,
        reader_max_tokens=args.reader_max_tokens,
        profile=args.profile,
        cache_passthrough=args.cache_passthrough,
        cache_passthrough_threshold=args.cache_passthrough_threshold,
        cache_passthrough_streak=args.cache_passthrough_streak,
        cache_hints=args.cache_hints,
        async_fold=args.async_fold,
        compact_tool_outputs=args.tool_compaction,
        tool_archive_min_chars=args.tool_archive_min_chars,
        cheap_fold_routing=args.cheap_fold_routing,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
