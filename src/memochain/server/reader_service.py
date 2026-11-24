"""MemoChain reader sidecar: capsule state + query -> memory-token text digest.

POST /tokens  -> 16x4096 memory tokens (numpy reader only, no LLM).
POST /read    -> text notes decoded by Qwen3-8B (vLLM offline engine with
                 enable-prompt-embeds injecting the memory tokens into <|memory|>
                 slots), or the raw token profile when no LLM is configured.
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from memochain.eval_qa import (
    MEMORY_TOKEN,
    PERSONA_PROMPT,
    build_memory_messages,
    clean_generation,
)
from memochain.reader_np import NumpyReader

SUMMARY_QUESTION = (
    "List the concrete facts, user preferences, project details, and decisions "
    "you know from memory as concise bullet points (max {words} words). "
    "Do not repeat instructions. If you have no concrete facts, reply only 'no salient memory'."
)

app = FastAPI(title="MemoChain reader sidecar")

reader: NumpyReader | None = None
engine = None  # vLLM LLM with enable_prompt_embeds
tokenizer = None
memory_token_id: int | None = None
embed_matrix = None


class TokensRequest(BaseModel):
    state_b64: str
    query_embedding: list[float] | None = None
    question: str | None = None


class ReadRequest(TokensRequest):
    max_words: int = 160


def _decode_state(state_b64: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(state_b64), dtype="<i2")
    if raw.size != 11 * 32 * 32:
        raise ValueError(f"state must be 22528 bytes, got {raw.size * 2}")
    return raw.reshape(11, 32, 32)


def _query_vec(req) -> np.ndarray:
    if req.query_embedding:
        return np.asarray(req.query_embedding, dtype=np.float32)
    return np.zeros(2560, dtype=np.float32)


@app.post("/tokens")
def tokens(req: TokensRequest) -> dict:
    t = reader.read_tokens(_decode_state(req.state_b64), _query_vec(req))
    return {"tokens": t.astype(float).tolist()}


@app.post("/read")
def read(req: ReadRequest) -> dict:
    t = reader.read_tokens(_decode_state(req.state_b64), _query_vec(req))  # [M, 4096]
    if engine is None:
        return {"text": "", "mode": "tokens-only"}
    import torch
    from vllm import SamplingParams

    question = (req.question or "").strip() or SUMMARY_QUESTION.format(words=req.max_words)
    messages = build_memory_messages(question, t.shape[0], system_prompt=PERSONA_PROMPT)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    positions = [i for i, t_id in enumerate(ids) if t_id == memory_token_id]
    if len(positions) != t.shape[0]:
        raise RuntimeError(f"expected {t.shape[0]} memory tokens in prompt, found {len(positions)}")
    ids_t = torch.tensor([ids], dtype=torch.long)
    base_embeds = embed_matrix(ids_t.clamp(max=embed_matrix.num_embeddings - 1)).clone()
    mem = torch.from_numpy(t).to(dtype=base_embeds.dtype)
    for row, pos in enumerate(positions):
        base_embeds[0, pos] = mem[row]

    outs = engine.generate(
        [{"prompt_embeds": base_embeds[0].to(torch.bfloat16)}],
        sampling_params=SamplingParams(max_tokens=256, temperature=0.0),
        use_tqdm=False,
    )
    text = clean_generation(outs[0].outputs[0].text)
    return {"text": text, "mode": "qwen3-decoded"}


def main() -> None:
    global reader, engine, tokenizer, memory_token_id, embed_matrix
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/pv4_10k")
    parser.add_argument("--decoder", default="Qwen/Qwen3-8B")
    parser.add_argument("--with-llm", action="store_true", help="load Qwen3-8B for text decoding")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    reader = NumpyReader(
        path / "reader.npz", np.load(path / "np_key_w.npy"), np.load(path / "np_key_b.npy")
    )

    if args.with_llm:
        from transformers import AutoTokenizer
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(args.decoder)
        tokenizer.add_special_tokens({"additional_special_tokens": [MEMORY_TOKEN]})
        memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
        engine = LLM(
            model=args.decoder,
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prompt_embeds=True,
        )
        _load_embeddings(args.decoder)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


def _load_embeddings(decoder: str) -> None:
    """Load the decoder's input embedding matrix as a frozen nn.Embedding."""
    global embed_matrix
    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download

    p = Path(snapshot_download(decoder))
    from safetensors import safe_open

    keys = ("model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight")
    for shard in sorted(p.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            avail = set(fh.keys())
            for k in keys:
                if k in avail:
                    w = fh.get_tensor(k).to(torch.bfloat16)
                    embed_matrix = nn.Embedding.from_pretrained(w, freeze=True).eval()
                    return
    raise FileNotFoundError(f"no embedding matrix for {decoder}")


if __name__ == "__main__":
    main()
