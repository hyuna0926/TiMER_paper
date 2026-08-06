# TiMER_paper

### Adaptive Multi-Granularity Routing for Time-Series Question Answering

Official implementation of **TiMER**, a time-series question answering framework
that combines a frozen **Multi-Granularity Time Encoder (MTE)**, a
**Query-Adaptive Granularity Router (QGR)**, temporal experts, and a frozen LLM.

The repository provides self-contained training and evaluation entry points for
both frozen-LLM and LoRA settings.

## Architecture

**[View the full TiMER architecture figure (PDF)](Figure/TiMER_Overview.pdf)**

TiMER consists of four main stages, following the terminology in the overview:

1. **Multi-Granularity Time Encoder (MTE)** extracts fine- and coarse-grained
   temporal representations.
2. **Query-Adaptive Granularity Router (QGR)** predicts query-conditioned
   routing weights for the local, global, and cross-granularity experts.
3. **Granularity-aware Temporal Experts Aggregation** applies adaptive average
   pooling and combines the routed expert representations.
4. **Time-to-Language Alignment** aligns the resulting time tokens with the
   language-model representation before passing them to the frozen LLM.


## Repository Structure

```text
TiMER_paper/
├── train2_adaptive.py     # frozen-LLM training
├── eval2_adaptive.py      # frozen-LLM evaluation
├── MTE/                   # Multi-Granularity Time Encoder and checkpoint
├── QGR/                   # Query-Adaptive Granularity Router and experts
├── Figure/                # TiMER architecture figure
├── scripts/               # example shell scripts
└── requirements.txt
```

## Installation

Python 3.11 and a CUDA-enabled PyTorch environment are recommended.

```bash
git clone <REPOSITORY_URL>
cd TiMER_paper

conda create -n timer python=3.11 -y
conda activate timer

pip install torch transformers peft accelerate pandas pyarrow numpy tqdm
```

The full reference environment is recorded in `requirements.txt`.

## Dataset

We use **[TSAQA: Time Series Analysis Question And Answering Benchmark](https://huggingface.co/datasets/TSAQA/TSAQA-Benchmark)**.

TSAQA contains train, validation, and test splits for six time-series analysis
tasks and three closed-ended question formats.

The following example downloads the Parquet splits into `data/TSAQA/`:

Expected data layout:

```text
data/TSAQA/
├── train.parquet
├── val.parquet
└── test.parquet
```

## MTE Checkpoint

The pretrained MTE checkpoint used by the examples is located at:

```text
MTE/checkpoint/gft_varlen_best_forecast.pt
```

## Training

Set the paths once:

```bash
DATA_DIR="./data/TSAQA"
MTE_CKPT="./MTE/checkpoint/gft_varlen_best_forecast.pt"
OUTPUT_DIR="./checkpoints/qwen3_0.6b"
```

### Frozen LLM

This is the recommended default configuration:

```bash
python train2_adaptive.py \
    --frozen_llm \
    --model_name "Qwen/Qwen3-0.6B" \
    --train_path "${DATA_DIR}/train.parquet" \
    --val_path "${DATA_DIR}/val.parquet" \
    --gft_ckpt "${MTE_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs 30 \
    --batch_size 8 \
    --grad_accum 4 \
    --max_seq_len 4096 \
    --max_ts_len 512 \
    --lr 1e-4 \
    --num_qformer_queries 25 \
    --top_k 3 \
    --early_stopping_patience 3
```

The best checkpoint is saved to `${OUTPUT_DIR}/best`.


## Evaluation

Use the same mode and architecture settings used during training.

### Frozen LLM

```bash
python eval2_adaptive.py \
    --frozen_llm \
    --model_name "Qwen/Qwen3-0.6B" \
    --checkpoint_dir "${OUTPUT_DIR}/best" \
    --gft_ckpt "${MTE_CKPT}" \
    --test_path "${DATA_DIR}/test.parquet" \
    --output_path "${OUTPUT_DIR}/eval_results.csv" \
    --max_seq_len 4096 \
    --num_qformer_queries 25 \
    --top_k 3
```

Evaluation writes per-sample predictions, accuracy, and MoE routing statistics
to the output CSV.
