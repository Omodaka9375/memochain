#!/usr/bin/env bash
# Coding-domain e2e training follow-ups (Qwen3-8B frozen decoder).
# Run 1 (from scratch) peaked at step 250 (val_nll=3.1532, +1.55 nats) and
# overfit past step 300 on 1,091 rows; its best checkpoint is saved in
# runs/e2e_qwen8b_coding_scratch.
#
# Run 2: warm-start transfer from e2e_qwen8b_pv4 (low LR 5e-5, 400 steps)
# Run 3: joint multi-domain pv4 + kai_coding from scratch (800 steps)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
PROMPT="You are a coding assistant embedded in a developer terminal. Use the compressed session memory below to stay consistent with earlier project files, commands, decisions, and bugs."

echo "=== RUN 2: kai_coding warm-started from e2e_qwen8b_pv4 (lr 5e-5, 400 steps) ==="
PYTHONPATH=src $PY -m memochain.train_e2e \
    --decoder Qwen/Qwen3-8B \
    --system-prompt "$PROMPT" \
    --data-dir data --dataset kai_coding \
    --steps 400 --batch 4 --grad-accum 2 --num-docs 64 \
    --lr 5e-5 --weight-decay 0.05 \
    --eval-every 50 --eval-batches 20 --log-every 25 \
    --gradient-checkpointing --init runs/e2e_qwen8b_pv4 \
    --save runs/e2e_qwen8b_coding_transfer --seed 0

echo "=== RUN 3: joint pv4 + kai_coding (Qwen3-8B, 800 steps) ==="
PYTHONPATH=src $PY -m memochain.train_e2e \
    --decoder Qwen/Qwen3-8B \
    --system-prompt "$PROMPT" \
    --data-dir data --dataset pv4 kai_coding \
    --steps 800 --batch 4 --grad-accum 2 --num-docs 64 \
    --lr 2e-4 --weight-decay 0.05 \
    --eval-every 100 --eval-batches 20 --log-every 50 \
    --gradient-checkpointing --save runs/e2e_qwen8b_joint_pv4_coding --seed 0

echo "=== CODING_TRAIN_DONE ==="
