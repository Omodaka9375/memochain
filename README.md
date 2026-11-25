# MemoChain

Mina-style constant-size memory for LLMs: the whole history of a document
collection or chat log collapses into a fixed 22 KB state, updated one
document at a time in O(1), with a receipt chain (or a full SP1 zkVM proof)
that makes every memory write verifiable.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24497714.svg)](https://doi.org/10.5281/zenodo.24497714)

[Website](https://omodaka9375.github.io/memochain/) | [Paper (PDF)](paper/main.pdf) | [Paper (Markdown)](PAPER.md) | [LaTeX Source](paper/main.tex) | [Blog](https://omodaka9375.github.io/portfolio/#/blog/memochain-22kb-memory-with-a-proof) | [Experimental Log](DESIGN.md) | [License (Apache-2.0)](LICENSE) | [Citation](CITATION.cff)

```text
Mina:      new_chain = fold(old_chain, block)  + recursive zk proof
MemoChain: new_state = fold(old_state, doc)   + receipt / zk proof of the write
```

The state is exactly 22,528 bytes (11 heads x 32 keys x 32 values, Q14
fixed-point int16) and never grows, no matter how much history is folded in.
All fold arithmetic is integer-only, so the same inputs produce bit-identical
state digests in Python, native Rust, and the SP1 zkVM.

## What is here

- **Core memory engine** (`src/memochain/`): Delta-rule fold, associative
  read, versioned capsule envelope, SHA-256 receipt chain with O(1) light
  head and O(n) full audit.
- **Trainable student** (`student.py`): the same 22 KB state, made
  differentiable with a straight-through Q14 quantizer. Two training paths:
  - **End-to-end, distillation-free** (`train_e2e.py`): gradients flow from a
    frozen open LLM's cross-entropy loss through memory-token injection, the
    reader, and the integer writer. No teacher weights needed anywhere.
  - Distillation (`distill.py`): optional, matches a Memorilla teacher's
    memory tokens (kept for comparison only).
- **Exact integer writer** (`student_int.py` + `crates/memochain-core`):
  trained weights quantize to a 3,605,932-byte binary that folds
  bit-identically in Python, Rust, and SP1.
- **ZK proofs** (`crates/sp1/`): Mina-style recursive IVC. A compressed
  STARK proves the whole 200-doc fold (1.24 MiB, ~24 min prove on DGX
  Spark) and a +1-doc recursive step costs 33.8M cycles; anyone verifies
  in ~28 ms.
- **OpenAI + Anthropic memory proxy** (`server/`): single ~55 MB RAM,
  zero-GPU container sitting in front of any local or cloud backend
  (`/v1/chat/completions` + native `/v1/messages`, BYOK key pass-through).
  Folds overflow into the 22 KB capsule + SHA-256 receipt chain, pairs it
  with hybrid recall (rolling fact sheet + int8 FIFO-capped SQLite chunks),
  shares sessions across local and cloud containers via one `/data` volume,
  and runs the optional SP1 proof queue in the background.

## Quickstart

```bash
uv sync --extra dev
uv run pytest              # 76 Python tests (torch tests skip if absent) + 14 Rust tests
uv run demo.py             # fold, read, verify, tamper-check, benchmark
```

## Training the 22 KB memory (no teacher)

End-to-end training through a frozen Qwen3-0.6B (or any causal LM), using
the LLM's own language-modeling loss on real answers:

```bash
uv sync --extra train
PYTHONPATH=src python -m memochain.train_e2e \
    --decoder Qwen/Qwen3-0.6B --data-dir data --dataset pv4 \
    --steps 3000 --batch 4 --grad-accum 2 --num-docs 64 \
    --save runs/e2e_qwen06b_pv4
```

Requires a Memorilla-Data-format store under `data/` (parquet questions +
precomputed Qwen3-Embedding-4B safetensors). From scratch, 3,000 steps,
0.37 s/step, 6.4 GiB VRAM:

```text
validation answer NLL: 5.242 (no memory) -> 3.535 (22 KB capsule)   +1.71 nats
held-out pv4/test QA (4,237 q, 30 unseen users, 200 docs each):
  closed-book 50.77% -> memochain_22kb 57.02%  (+6.25 pp, hard +3.18 pp)
```

Exports the Q14 `weights.bin` (SP1-provable), torch-free numpy arrays, and
`reader.npz` for the proxy.

## Evaluation

```bash
PYTHONPATH=src python -m memochain.eval_qa \
    --checkpoint runs/e2e_qwen06b_pv4 --teacher none \
    --dataset pv4 --split test --data-dir data \
    --decoder Qwen/Qwen3-0.6B --out runs/e2e_qwen06b_pv4/qa_pv4_test.json
```

`--teacher none` runs the standalone pipeline (no Memorilla). With a teacher
dir it also scores the Memorilla baseline for comparison. Distilled-model
reference numbers (Qwen3-8B decoder, pv4/test): 22 KB capsule 64.67% vs
Memorilla teacher 64.76% (4.10 MB KV cache) vs closed-book 58.58% - 98.5% of
the teacher's memory gain at 181.8x compression. See DESIGN.md Phase 10.

## Memory proxy (OpenAI + Anthropic, BYOK, shared sessions)

By default the proxy runs in **hybrid reader-free mode** (`--reader-backend hybrid`, `--recall-top-k 4`): zero extra GPU or reader sidecar, using the active backend model for a rolling fact sheet plus an int8-quantized, FIFO-capped per-session SQLite chunk store (default 2,000 chunks, ~7 MB max/session) alongside the exact 22 KB Q14 capsule and SHA-256 receipt chain.

```bash
# Docker (recommended: starts :9000 local + :9001 OpenRouter on a shared /data volume)
docker compose up -d

# Or directly via Python
uv sync --extra proxy
PYTHONPATH=src python -m memochain.server \
    --backend-base-url http://localhost:8000/v1 \
    --checkpoint runs/e2e_qwen06b_pv4 \
    --port 9000
```

Point any OpenAI client at `http://localhost:9000/v1` (local) or
`http://localhost:9001/v1` (OpenRouter), or any Anthropic `/v1/messages`
client such as Claude Code at `http://localhost:9000`. With the same
`x-session-id` header, a session can start on a local model and continue
on a cloud model with the same capsule and history via the shared volume.
Client API keys (`Authorization` / `x-api-key`) pass through verbatim for
both chat and fold-summary calls (BYOK); `MEMOCHAIN_BACKEND_API_KEY` is
only an optional fallback.

Supported backends via `MEMOCHAIN_BACKEND_BASE_URL` / `--backend-base-url`:

- **Local OpenAI-compatible (default)**: vLLM (e.g. `http://localhost:8000/v1`),
  llama.cpp server, Ollama (`http://localhost:11434/v1`), LM Studio.
- **OpenRouter**: `https://openrouter.ai/api/v1` (BYOK pass-through or
  `MEMOCHAIN_BACKEND_API_KEY`).
- **Direct Anthropic**: `https://api.anthropic.com/v1`.

Sessions start with zero memory overhead; once the conversation exceeds
`--hot-limit` tokens (default 80,000, conversation only; folding down to
40,000) the oldest turns fold into the 22 KB capsule, recalled facts +
top-K verbatim chunks are injected into the system prompt, and the
optional SP1 queue replays the fold in the background
(`/v1/memochain/verify` reports capsule, receipt chain, and proof status).
Optional flags: `--reader-backend transformers` (in-process neural reader),
`--reader-backend profile` (numpy recall profile only), `--reader-sidecar-url`,
`--embedding-base-url`, `--max-chunks`, and `--proof-mode execute`.

Measured savings (`tools/benchmark_savings.py`, 120-turn identical session,
backend-metered `usage.prompt_tokens`): 7,086,360 input tokens direct vs
674,016 through the proxy (**90% reduction**, final-turn prompt 117,125 to
6,746 tokens). See `tools/bench_results/savings_report.md`.

## ZK proofs (Linux / SP1)

```bash
cd crates/sp1/ivc-program && cargo prove build
cd ../script && cargo build --release --bin ivc
./target/release/ivc prove --weights weights.bin --docs docs.bin \
    --out-proof proof.bin --out-capsule capsule.bin
./target/release/ivc verify --proof proof.bin   # ~28 ms
```

## Status

Phases 1-21 complete: integer Delta-rule fold (bit-identical across
machines), O(1) receipt chain, versioned capsules, distilled student +
end-to-end distillation-free training (Qwen3-8B e2e matches the Memorilla
teacher's held-out accuracy at 64.69% vs 64.76% from the 22 KB capsule with
zero teacher weights), exact Q14 writer, Mina-style recursive SP1 IVC proofs,
capacity scaling ablation (gain saturates by 10 KiB; the 22 KiB Mina-sized
state is past the knee), coding-domain transfer on real sessions, slim
single-container hybrid proxy (OpenAI + Anthropic `/v1/messages`, BYOK
pass-through, int8 FIFO-capped SQLite recall, cross-container shared
sessions), and a 120-turn A/B benchmark measuring 90% input-token
reduction. Everything except the Qwen3-8B benchmarks runs on a 12 GB GPU
or CPU only. See [DESIGN.md](DESIGN.md) for the phase-by-phase results
and negative results.

Known limits (detailed in DESIGN.md and PAPER.md Section 6): the 22 KB
state holds gist and preferences well (~350 orthogonal fact slots) but
cannot store verbatim long lists (which is why the default live proxy pairs
it with bounded SQLite hybrid recall, outside the ZK circuit); capsules
are bound to one embedding geometry (Qwen3-Embedding-4B) plus a reader
checkpoint; SP1 prove times are minutes on CPU-class hosts.
