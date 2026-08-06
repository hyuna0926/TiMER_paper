#!/usr/bin/env bash
set -euo pipefail

MTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${MTE_DIR}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "${MTE_DIR}/train_gft_varlen.py" \
  --train_parquet "${REPO_ROOT}/data/TSAQA/train.parquet" \
  --val_parquet "${REPO_ROOT}/data/TSAQA/val.parquet" \
  --save_path "${MTE_DIR}/checkpoint/gft_varlen_best_forecast.pt" \
  --batch_size 256 \
  --epochs 30
