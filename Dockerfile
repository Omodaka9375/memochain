# MemoChain memory proxy - single tight container (v0.2.0).
#
# Default build: numpy-only hybrid proxy (~57 MB RSS, 0 extra GPU):
#   - exact Q14 22 KB capsule + sha256 receipt chain (zero-download built-in
#     deterministic writer; optionally mount trained weights.bin or np_*.npy)
#   - cache-safe split injection (static fact sheet in system prompt, dynamic
#     recall in tail user message) + provider cache hints + passthrough
#   - SQLite FTS5 + dense vector RRF search, append-only facts ledger,
#     near-dupe suppression, hit-based eviction, overflow tool compaction
#   - zero-latency async background folding + auto cloud/local profile tuning
#   - works out of the box with OpenAI-compatible, OpenRouter, Anthropic, and Gemini backends
#
# Optional neural-reader build:
#   docker build --build-arg WITH_READER=1 .
# adds CPU torch + transformers (~1 GB image) for --reader-backend transformers.

FROM python:3.12-slim

ARG WITH_READER=0

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[proxy]" \
 && if [ "$WITH_READER" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
     && pip install --no-cache-dir "transformers>=4.46" safetensors huggingface-hub; \
    fi

# all runtime config flows through MEMOCHAIN_* env vars
# MEMOCHAIN_PROFILE=auto picks cloud (96k/32k, cache hints, cheap fold routing)
# vs local (28k/12k, VRAM/TTFT tuned) from MEMOCHAIN_BACKEND_BASE_URL.
ENV MEMOCHAIN_HOST=0.0.0.0 \
    MEMOCHAIN_PORT=9000 \
    MEMOCHAIN_BACKEND_BASE_URL=http://host.docker.internal:8000/v1 \
    MEMOCHAIN_PROFILE=auto \
    MEMOCHAIN_CHECKPOINT= \
    MEMOCHAIN_STORE_ROOT=/data/sessions \
    MEMOCHAIN_READER_BACKEND=hybrid \
    MEMOCHAIN_READER_MAX_TOKENS=4096 \
    MEMOCHAIN_RECALL_TOP_K=4 \
    MEMOCHAIN_MAX_CHUNKS=2000

VOLUME ["/data"]
EXPOSE 9000

CMD ["python", "-m", "memochain.server"]
