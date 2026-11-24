#!/usr/bin/env bash
# Capacity scaling ablation: train identical recipes at 4 state sizes.
# Usage: bash tools/run_capacity_ablation.sh [steps] [data_dir]
set -euo pipefail
STEPS="${1:-1500}"
DATA="${2:-data}"
VENV_PY="${PYTHON:-python3}"
COMMON="--decoder Qwen/Qwen3-0.6B --data-dir $DATA --dataset pv4 --steps $STEPS --batch 4 --grad-accum 2 --num-docs 64 --lr 3e-4 --eval-every 250 --eval-batches 30 --log-every 50 --gradient-checkpointing --seed 0"

cd "$(dirname "$0")/.."
mkdir -p runs/capacity_ablation
for CFG in "5 32 32" "11 32 32" "22 32 32" "44 32 32"; do
  set -- $CFG
  H=$1; K=$2; V=$3
  KB=$(python3 -c "print($H*$K*$V*2/1024)")
  OUT=runs/capacity_ablation/h${H}_k${K}
  echo "=== capacity H=$H K=$K V=$V (${KB} KiB) ==="
  PYTHONPATH=src $VENV_PY -m memochain.train_e2e $COMMON \
      --num-heads $H --key-dim $K --save $OUT
done
echo "=== ablation complete ==="
