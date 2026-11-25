# MemoChain memory proxy - single tight container.
#
# Default build: numpy-only hybrid proxy (~57 MB RSS, 0 extra GPU):
#   - exact Q14 22 KB capsule + sha256 receipt chain
#   - backend fact-sheet digest + SQLite top-K verbatim chunk recall
#   - works out of the box with any OpenAI-compatible backend
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
ENV MEMOCHAIN_HOST=0.0.0.0 \
    MEMOCHAIN_PORT=9000 \
    MEMOCHAIN_BACKEND_BASE_URL=http://backend:8000/v1 \
    MEMOCHAIN_CHECKPOINT=/checkpoints/student \
    MEMOCHAIN_STORE_ROOT=/data/sessions \
    MEMOCHAIN_READER_BACKEND=hybrid \
    MEMOCHAIN_RECALL_TOP_K=4 \
    MEMOCHAIN_MAX_CHUNKS=2000

VOLUME ["/data"]
EXPOSE 9000

CMD ["python", "-m", "memochain.server"]
