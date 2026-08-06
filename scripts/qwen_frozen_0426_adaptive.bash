#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/TSAQA}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/checkpoint_qwen06}"
MTE_CKPT="${MTE_CKPT:-${REPO_ROOT}/MTE/checkpoint/gft_varlen_best_forecast.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${REPO_ROOT}"

# Ours(Top-3, cls router)
# checkpoint_qwen06
python train2_adaptive.py \
    --model_name          "Qwen/Qwen3-0.6B" \
    --train_path          "${DATA_DIR}/train.parquet" \
    --val_path            "${DATA_DIR}/val.parquet" \
    --gft_ckpt            "${MTE_CKPT}" \
    --output_dir          "${OUTPUT_DIR}" \
    --num_epochs          30 \
    --batch_size          8 \
    --grad_accum          4 \
    --max_seq_len         4096 \
    --max_ts_len          512 \
    --lr                  1e-4 \
    --num_qformer_queries 25 \
    --early_stopping_patience 3 \
    --top_k 3 \
    --frozen_llm

python eval2_adaptive.py \
    --checkpoint_dir "${OUTPUT_DIR}/best" \
    --gft_ckpt       "${MTE_CKPT}" \
    --test_path      "${DATA_DIR}/test.parquet" \
    --output_path    "${OUTPUT_DIR}/eval_results.csv" \
    --num_qformer_queries 25 \
    --top_k 3 \
    --max_seq_len 4096 \
    --model_name "Qwen/Qwen3-0.6B" \
    --frozen_llm
