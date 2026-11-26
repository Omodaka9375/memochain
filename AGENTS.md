# MemoChain AGENTS.md

This file is for agents and AI assistants that work on the MemoChain repository.

## What this is

Constant-size (22 KB) memory for LLM agents. A fixed 22,528-byte state is
updated in O(1) per document via an integer-only Delta-rule fold, with a
SHA-256 receipt chain. Bit-identical across Python, Rust, and SP1 zkVM.

## How to install and run

See [docs/llms.txt](docs/llms.txt) for the agent-friendly install path.

Quick summary (details and commands in docs/llms.txt):

- Docker (recommended): `docker run -d --name memochain -p 9000:9000 \
  --add-host=host.docker.internal:host-gateway \
  -v memochain-data:/data \
  -e MEMOCHAIN_BACKEND_BASE_URL=http://host.docker.internal:8000/v1 \
  ghcr.io/omodaka9375/memochain:latest`
- From source: `git clone https://github.com/Omodaka9375/memochain && \
  cd memochain && uv sync --extra proxy && \
  PYTHONPATH=src python -m memochain.server \
    --backend-base-url http://localhost:8000/v1 --port 9000`
- Client URL: `http://localhost:9000/v1` (local) or
  `http://localhost:9001/v1` (OpenRouter).

## Repo layout

```text
src/memochain/        core memory engine (fold, receipt chain, capsule)
server/               FastAPI OpenAI + Anthropic proxy
crates/memochain-core Rust native mirror of the integer writer
crates/sp1/           Mina-style recursive SP1 IVC proofs
tests/                74 Python tests (torch tests skip if absent)
tools/                benchmarks, weight export, A/B harness
docs/                 landing page (GitHub Pages) + llms.txt
```

## Tests

- Python: `uv run pytest` (55 pass, 6 skip on CPU-only hosts; 74 on a
  torch-enabled host)
- Rust: `cargo test --manifest-path crates/memochain-core/Cargo.toml`
  (10 tests) and
  `cargo test --manifest-path crates/chain-logic/Cargo.toml` (4 tests)

## Conventions

- Integer-only arithmetic for the fold; never introduce floats into the
  state.
- Receipt chain is append-only; do not rewrite history.
- Do not commit hostnames, IPs, local paths, or personal data. Everything
  tracked in this repo must be safe to publish.
- Never push to `origin` without explicit user approval.

## Trained weights (optional, not in repo)

Published as a GitHub release asset on the v0.1.0 release. Expected digest:
`626e2089fe38bc593ebf8421873739438aa47b86408db10bafc4eb424f82d67b`
URL: `https://github.com/Omodaka9375/memochain/releases/download/v0.1.0/weights.bin`
