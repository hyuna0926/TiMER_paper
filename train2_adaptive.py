"""
Unified adaptive GFT + MoE trainer for TSAQA.

This file contains both LoRA and frozen-LLM training paths. The adaptive
temporal pooling signal path is used in both modes.

Usage:
    python train2_adaptive.py [LoRA options]
    python train2_adaptive.py --frozen_llm [frozen options]
"""

from __future__ import annotations

import os
import sys
import math
import ast
import re
import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs

# ---------------------------------------------------------------------------
# Path setup — MTE and QGR modules
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "MTE"))
sys.path.insert(0, os.path.join(_HERE, "QGR"))

from gft_varlen_forecast import GFTVarLenConfig, GFTVarLenForecaster  # noqa: E402
from granularity2 import (  # noqa: E402
    GranularityMoE,
    QFormerConnector,
    QuestionConditionedRouter,
    get_text_representative,
    masked_mean,
    compute_router_reg_loss,
    format_expert_selection,
)

# ---------------------------------------------------------------------------
# Config defaults  (all overridable via argparse)
# ---------------------------------------------------------------------------
MODEL_NAME    = "Qwen/Qwen3-0.6B"
TRAIN_PATH    = os.path.join(_HERE, "data", "TSAQA", "train.parquet")
VAL_PATH      = os.path.join(_HERE, "data", "TSAQA", "val.parquet")
GFT_CKPT      = os.path.join(_HERE, "MTE", "checkpoint", "gft_varlen_best_forecast.pt")
OUTPUT_DIR    = os.path.join(_HERE, "checkpoints", "checkpoint_lora")

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


LORA_RANK     = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 30
BATCH_SIZE    = 4       # per device
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 1024    # max text token length
MAX_TS_LEN    = 512     # max time-series length (TSAQA actual max = 512 points)
WARMUP_RATIO  = 0.05
WEIGHT_DECAY  = 0.01
SAVE_STEPS    = 1000
LOG_STEPS     = 100

# GFT / MoE architecture (matches GFT checkpoint d_model=256)
TS_HIDDEN_DIM       = 256
NUM_QFORMER_QUERIES = 8
NUM_QFORMER_LAYERS  = 2
TOP_K               = 2
ROUTER_ENTROPY_W    = 0.0    # 기본 off — 필요 시 sharpen 방향으로 약하게
ROUTER_BALANCE_W    = 0.001  # collapse만 방지, routing 자유도 확보

SYSTEM_PROMPT = ""
ANSWER_PATTERN = re.compile(r"^(?:[A-D]|[TF]|[A-D](?:,[A-D]){1,})$")

# ---------------------------------------------------------------------------
# input_ts parsing  (reused from GFT training code)
# ---------------------------------------------------------------------------

def load_llm_for_model_name(model_name: str, **model_kwargs):
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    model_kwargs["config"] = config
    if config.__class__.__name__ == "Mistral3Config":
        return AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_llm_hidden_dim(llm) -> int:
    configs = [getattr(llm, "config", None)]
    base_model = getattr(llm, "base_model", None)
    configs.append(getattr(base_model, "config", None))
    configs.append(getattr(getattr(base_model, "model", None), "config", None))
    for config in configs:
        if config is None:
            continue
        hidden_size = getattr(config, "hidden_size", None)
        text_config = getattr(config, "text_config", None)
        if hidden_size is None and text_config is not None:
            hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
    raise ValueError("Could not infer LLM hidden size from model config.")


def _to_python_obj(v):
    if isinstance(v, str):
        return ast.literal_eval(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return v
    raise TypeError(f"Unsupported input_ts type: {type(v)}")


def _is_nested(seq) -> bool:
    if not isinstance(seq, (list, tuple)) or len(seq) == 0:
        return False
    return any(isinstance(x, (list, tuple, np.ndarray)) for x in seq)


def parse_ts_field(v) -> List[List[float]]:
    """Parse input_ts column into a list of 1-D float lists (one per series)."""
    seq = _to_python_obj(v)

    if not _is_nested(seq):
        arr = np.asarray(seq, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return [arr.tolist()] if len(arr) > 0 else [[0.0]]

    series_list = []
    for row in seq:
        r = np.asarray(row, dtype=np.float32).reshape(-1)
        r = r[np.isfinite(r)]
        if len(r) > 0:
            series_list.append(r.tolist())

    return series_list if series_list else [[0.0]]


def normalize_answer_text(answer: str) -> str:
    s = str(answer).strip().strip('"').strip("'")
    s = re.sub(r"\s*,\s*", ",", s)
    up = s.upper()
    if up == "TRUE":
        return "T"
    if up == "FALSE":
        return "F"
    if ANSWER_PATTERN.fullmatch(up):
        return up
    return s


def contiguous_spans(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    spans: List[Tuple[int, int]] = []
    start = prev = indices[0]
    for pos in indices[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        spans.append((start, prev))
        start = prev = pos
    spans.append((start, prev))
    return spans


def print_tokenizer_answer_debug(tokenizer) -> None:
    samples = ["A", "B", "T", "F", "A,B,D,C", "B,A,D,C", "A, B, D, C", "\"A,B,D,C\""]
    print("[Tokenizer answer-token debug]")
    for s in samples:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {s!r} -> ids={ids} tokens={toks}")


FALLBACK_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ 'Question: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Answer: ' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Answer: ' }}{% endif %}"""


def ensure_tokenizer_chat_template(tokenizer) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    return True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TSAQAWithTSDataset(Dataset):
    """Load parquet; for each row expose ts tensor + tokenised question/answer."""

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_text_len: int,
        max_ts_len: int,
        debug_label_samples: int = 0,
        split_name: str = "train",
    ):
        available_cols = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        self.has_task_col = "task" in available_cols
        read_cols = ["input_ts", "question", "answer"] + (["task"] if self.has_task_col else [])
        table = pq.read_table(parquet_path, columns=read_cols)
        self.df          = table.to_pandas().reset_index(drop=True)
        self.tokenizer   = tokenizer
        self.max_text_len = max_text_len
        self.max_ts_len   = max_ts_len
        self.debug_label_samples = max(0, int(debug_label_samples))
        self.split_name = split_name
        self._debug_printed = 0

        ans_norm = self.df["answer"].astype(str).map(normalize_answer_text)
        self.answer_value_counts = ans_norm.value_counts(dropna=False)
        bad_mask = ~ans_norm.str.fullmatch(ANSWER_PATTERN.pattern, na=False)
        self.answer_bad_count = int(bad_mask.sum())
        self.answer_bad_examples = ans_norm[bad_mask].head(10).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        question = str(row["question"])
        answer_raw = str(row["answer"])
        answer_norm = normalize_answer_text(answer_raw)
        task = "other"

        # --- time series ---
        try:
            series_list = parse_ts_field(row["input_ts"])
        except Exception:
            series_list = [[0.0]]
        if not series_list:
            series_list = [[0.0]]

        ts_tensors = []
        for s in series_list:
            s = s[-self.max_ts_len:]
            if not s:
                s = [0.0]
            ts_tensors.append(torch.tensor(s, dtype=torch.float32).unsqueeze(-1))  # [L_i, 1]

        # --- keep prompt template, but supervise only the raw answer text ---
        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )

        prefix_enc = self.tokenizer(
            prefix_text,
            truncation=False,
            padding=False, return_tensors=None,
        )
        answer_ids = self.tokenizer(
            answer_raw,
            add_special_tokens=False,
            truncation=False,
            padding=False, return_tensors=None,
        )["input_ids"]
        if not answer_ids:
            answer_ids = self.tokenizer(
                answer_norm,
                add_special_tokens=False,
                truncation=False,
                padding=False, return_tensors=None,
            )["input_ids"]

        prefix_ids = prefix_enc["input_ids"]
        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        keep_prefix = max(0, self.max_text_len - len(answer_ids) - len(eos_ids))
        if len(prefix_ids) > keep_prefix:
            prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []

        input_ids = prefix_ids + answer_ids + eos_ids
        labels = ([-100] * len(prefix_ids)) + answer_ids + ([-100] * len(eos_ids))
        if len(input_ids) > self.max_text_len:
            input_ids = input_ids[-self.max_text_len:]
            labels = labels[-self.max_text_len:]

        if self.debug_label_samples > 0 and self._debug_printed < self.debug_label_samples:
            label_positions = [i for i, l in enumerate(labels) if l != -100]
            label_token_ids = [tid for tid, l in zip(input_ids, labels) if l != -100]
            label_decoded = self.tokenizer.decode(label_token_ids, skip_special_tokens=True).strip()
            final_input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            spans = contiguous_spans(label_positions)

            print(f"\n[LabelDebug][{self.split_name}] idx={idx}")
            print(f"  answer_raw={answer_raw!r}")
            print(f"  answer_norm={answer_norm!r}")
            print("  final_input_text:")
            print(final_input_text)
            print(f"  label_positions={spans}")
            print(f"  label_decode={label_decoded!r}")
            if label_token_ids != answer_ids:
                print("  WARNING: label tokens do not match the raw answer tokens.")
            self._debug_printed += 1

        # --- question-only tokens (for router conditioning) ---
        q_enc = self.tokenizer(
            question,
            truncation=True, max_length=self.max_text_len,
            padding=False, return_tensors=None,
        )

        # task = get_task_label(question)
        if self.has_task_col and "task" in row and not pd.isna(row["task"]):
            t = str(row["task"]).strip()
            if t:
                task = t

        return {
            "ts_list":    ts_tensors,           # List of [L_i, 1] tensors, one per series
            "num_series": len(ts_tensors),      # int
            "input_ids":  input_ids,            # list[int]
            "labels":     labels,               # list[int]
            "q_ids":      q_enc["input_ids"],   # list[int]  (question only)
            "task":       task,                 # str task label
        }


def collate_fn(batch, pad_token_id: int) -> dict:
    B = len(batch)

    # --- pad time series: [B, S, L, 1] ---
    max_num_series = max(b["num_series"] for b in batch)
    all_lengths = [t.size(0) for b in batch for t in b["ts_list"]]
    max_ts = max(all_lengths) if all_lengths else 1

    ts_pad      = torch.zeros(B, max_num_series, max_ts, 1)
    ts_len_mask = torch.zeros(B, max_num_series, max_ts, dtype=torch.bool)
    series_mask = torch.zeros(B, max_num_series, dtype=torch.bool)

    for i, b in enumerate(batch):
        for s_idx, ts_tensor in enumerate(b["ts_list"]):
            l = ts_tensor.size(0)
            ts_pad[i, s_idx, :l]      = ts_tensor
            ts_len_mask[i, s_idx, :l] = True
            series_mask[i, s_idx]     = True

    # --- pad full conversation ---
    max_text = max(len(b["input_ids"]) for b in batch)
    ids_list, lab_list, attn_list = [], [], []
    for b in batch:
        pad = max_text - len(b["input_ids"])
        ids_list.append(b["input_ids"] + [pad_token_id] * pad)
        lab_list.append(b["labels"]    + [-100]          * pad)
        attn_list.append([1] * len(b["input_ids"]) + [0] * pad)

    # --- pad question-only ids ---
    max_q = max(len(b["q_ids"]) for b in batch)
    q_ids_list, q_attn_list = [], []
    for b in batch:
        pad = max_q - len(b["q_ids"])
        q_ids_list.append(b["q_ids"]  + [pad_token_id] * pad)
        q_attn_list.append([1] * len(b["q_ids"]) + [0] * pad)

    return {
        "ts":             ts_pad,           # [B, S, L, 1]
        "ts_len_mask":    ts_len_mask,      # [B, S, L]  bool — valid time steps
        "series_mask":    series_mask,      # [B, S]     bool — real vs padding series
        "input_ids":      torch.tensor(ids_list,   dtype=torch.long),
        "attention_mask": torch.tensor(attn_list,  dtype=torch.long),
        "labels":         torch.tensor(lab_list,   dtype=torch.long),
        "q_ids":          torch.tensor(q_ids_list, dtype=torch.long),
        "q_mask":         torch.tensor(q_attn_list, dtype=torch.long),
        "tasks":          [b["task"] for b in batch],   # list[str]
    }


# ---------------------------------------------------------------------------
# GFTSignalLLM  — main model
# ---------------------------------------------------------------------------

class GFTSignalLLM(nn.Module):
    """
    Pipeline:
      1. GFT encoder (frozen)  →  fine_tokens / coarse_tokens  [B, N, 256]
      2. Question embedding pool  →  q_vec  [B, 256]
      3. QuestionConditionedRouter  →  sparse_gates
      4. GranularityMoE  →  expert outputs
      5. QFormerConnector  →  signal_embeds  [B, Q, llm_dim]
      6. Prepend signal tokens to LLM input  →  LM loss
    """

    def __init__(
        self,
        gft_config: GFTVarLenConfig,
        gft_state_dict: dict,
        llm,                       # PEFT-wrapped LLM
        llm_hidden_dim: int,
        ts_hidden_dim: int = 256,
        num_qformer_queries: int = 8,
        num_qformer_layers: int = 2,
        num_experts: int = 3,
        top_k: int = TOP_K,
        num_heads: int = 4,
        dropout: float = 0.1,
        router_entropy_weight: float = 0.01,
        router_balance_weight: float = 0.01,
        topk_warmup_steps:     int   = 0,    # 이 step 이후부터 top-k routing 활성 (0=처음부터)
    ):
        super().__init__()
        self.ts_hidden_dim         = ts_hidden_dim
        self.llm_hidden_dim        = llm_hidden_dim
        self.router_entropy_weight = router_entropy_weight
        self.router_balance_weight = router_balance_weight
        self.topk_warmup_steps     = topk_warmup_steps

        # ---- GFT encoder (frozen) ----
        self.gft = GFTVarLenForecaster(gft_config)
        self.gft.load_state_dict(gft_state_dict)
        for p in self.gft.parameters():
            p.requires_grad = False

        # ---- LLM (LoRA applied externally) ----
        self.llm = llm

        # ---- Question projection: llm_dim → ts_hidden_dim ----
        self.q_proj = nn.Linear(llm_hidden_dim, ts_hidden_dim)

        # ---- Router & MoE ----
        self.router = QuestionConditionedRouter(
            hidden_dim=ts_hidden_dim,
            router_hidden_dim=ts_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_router_heads=num_heads,
            dropout=dropout,
        )
        self.moe = GranularityMoE(
            hidden_dim=ts_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ---- Pre-QFormer projection + QFormer ----
        self.pre_qformer_proj = nn.Linear(ts_hidden_dim, ts_hidden_dim)
        self.qformer = QFormerConnector(
            input_dim=ts_hidden_dim,
            output_dim=llm_hidden_dim,
            num_queries=num_qformer_queries,
            num_layers=num_qformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ---- Learnable signal boundary tokens ----
        self.signal_start = nn.Parameter(torch.randn(1, 1, llm_hidden_dim) * 0.02)
        self.signal_end   = nn.Parameter(torch.randn(1, 1, llm_hidden_dim) * 0.02)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return LLM token embeddings (frozen weights, float32 output)."""
        embed_fn = self.llm.get_input_embeddings()
        with torch.no_grad():
            emb = embed_fn(input_ids)   # bfloat16 or float32 depending on LLM
        return emb.float()

    def _encode_question(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Question representation for router conditioning (causal LM-safe):
          1) LLM last hidden states
          2) last valid token pooling (default in granularity2)
          3) projection to ts_hidden_dim
        """
        with torch.no_grad():
            # CausalLM output does not always expose `last_hidden_state`,
            # so use hidden_states[-1] with output_hidden_states=True.
            out = self.llm(
                input_ids=q_ids,
                attention_mask=q_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]  # [B, Tq, llm_dim]

        q_repr = get_text_representative(
            last_hidden.float(),
            q_mask,
            strategy="last_token",
        )  # [B, llm_dim]
        return self.q_proj(q_repr)  # [B, ts_hidden_dim]

# ---------------------------------------------------------------------------
# Save helper (LoRA adapter + custom modules only, no full model)
# ---------------------------------------------------------------------------

def save_checkpoint(
    tag: str,
    model: nn.Module,
    tokenizer,
    accelerator: Accelerator,
    output_dir: str,
) -> None:
    path = os.path.join(output_dir, tag)
    os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    unwrapped: GFTSignalLLM = accelerator.unwrap_model(model)

    # 1) LoRA adapter only
    lora_path = os.path.join(path, "lora_adapter")
    unwrapped.llm.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # 2) Custom trainable modules (everything except GFT + LLM)
    custom_state = {
        k: v for k, v in unwrapped.state_dict().items()
        if not k.startswith("gft.") and not k.startswith("llm.")
    }
    torch.save(custom_state, os.path.join(path, "custom_modules.pt"))

    print(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_lora() -> None:
    parser = argparse.ArgumentParser(description="GFT + MoE + LLM LoRA trainer")
    parser.add_argument("--model_name",          default=MODEL_NAME)
    parser.add_argument("--train_path",          default=TRAIN_PATH)
    parser.add_argument("--val_path",            default=VAL_PATH)
    parser.add_argument("--gft_ckpt",            default=GFT_CKPT)
    parser.add_argument("--output_dir",          default=OUTPUT_DIR)
    parser.add_argument("--batch_size",          type=int,   default=BATCH_SIZE)
    parser.add_argument("--grad_accum",          type=int,   default=GRAD_ACCUM)
    parser.add_argument("--max_seq_len",         type=int,   default=MAX_SEQ_LEN)
    parser.add_argument("--max_ts_len",          type=int,   default=MAX_TS_LEN)
    parser.add_argument("--num_epochs",          type=int,   default=NUM_EPOCHS)
    parser.add_argument("--lr",                  type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_rank",           type=int,   default=LORA_RANK)
    parser.add_argument("--num_qformer_queries", type=int,   default=NUM_QFORMER_QUERIES)
    parser.add_argument("--top_k",               type=int,   default=TOP_K)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop if val_loss does not improve for N epochs. 0 = disabled.")
    parser.add_argument("--topk_warmup_steps", type=int, default=0,
                        help="Steps of soft routing before switching to masked top-k. 0 = top-k from start.")
    parser.add_argument("--debug_label_samples", type=int, default=0,
                        help="Print first N samples with label-mask debug information.")
    args = parser.parse_args()
    set_seed(args.seed)

    # Adaptive ablation variants intentionally skip subsets of trainable modules
    # (for example router/MoE or raw-signal projection), so DDP must tolerate
    # unused parameters on each step.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[ddp_kwargs],
    )

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added_fallback_chat_template = ensure_tokenizer_chat_template(tokenizer)
    if accelerator.is_main_process:
        if added_fallback_chat_template:
            print(
                "Tokenizer has no chat_template; using fallback Question/Answer template."
            )
        print_tokenizer_answer_debug(tokenizer)

    # ---- Datasets ----
    if accelerator.is_main_process:
        print("Loading datasets …")

    train_dataset = TSAQAWithTSDataset(
        args.train_path, tokenizer, args.max_seq_len, args.max_ts_len,
        debug_label_samples=args.debug_label_samples,
        split_name="train",
    )
    val_dataset = TSAQAWithTSDataset(
        args.val_path, tokenizer, args.max_seq_len, args.max_ts_len,
        debug_label_samples=0,
        split_name="val",
    )

    if accelerator.is_main_process:
        print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")
        print(f"[train] normalized answer unique={len(train_dataset.answer_value_counts):,}")
        print(train_dataset.answer_value_counts.head(20).to_string())
        print(f"[train] non-canonical answers={train_dataset.answer_bad_count:,}")
        if train_dataset.answer_bad_examples:
            print(f"[train] bad examples={train_dataset.answer_bad_examples}")
        print(f"[val] normalized answer unique={len(val_dataset.answer_value_counts):,}")
        print(val_dataset.answer_value_counts.head(20).to_string())
        print(f"[val] non-canonical answers={val_dataset.answer_bad_count:,}")
        if val_dataset.answer_bad_examples:
            print(f"[val] bad examples={val_dataset.answer_bad_examples}")

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=4, pin_memory=True,
    )

    # ---- GFT checkpoint ----
    if accelerator.is_main_process:
        print(f"Loading GFT checkpoint: {args.gft_ckpt}")

    gft_ckpt   = torch.load(args.gft_ckpt, map_location="cpu")
    gft_config = GFTVarLenConfig(**gft_ckpt["config"])
    gft_state  = gft_ckpt["model"]

    # ---- LLM + LoRA ----
    if accelerator.is_main_process:
        print(f"Loading LLM: {args.model_name}")

    llm = load_llm_for_model_name(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    llm = get_peft_model(llm, lora_config)

    if accelerator.is_main_process:
        llm.print_trainable_parameters()

    llm_hidden_dim = get_llm_hidden_dim(llm)

    # ---- Full model ----
    model = AdaptiveGFTSignalLLM(
        gft_config=gft_config,
        gft_state_dict=gft_state,
        llm=llm,
        llm_hidden_dim=llm_hidden_dim,
        ts_hidden_dim=TS_HIDDEN_DIM,
        num_qformer_queries=args.num_qformer_queries,
        num_qformer_layers=NUM_QFORMER_LAYERS,
        top_k=args.top_k,
        router_entropy_weight=ROUTER_ENTROPY_W,
        router_balance_weight=ROUTER_BALANCE_W,
        topk_warmup_steps=args.topk_warmup_steps,
    )

    # ---- Optimizer (trainable params only; GFT is frozen) ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)

    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.num_epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- Accelerate prepare ----
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ---- Training loop ----
    global_step   = 0
    best_val_loss = float("inf")
    no_improve    = 0
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train.log")
    def write_log(msg: str):
        if accelerator.is_main_process:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line)
            with open(log_path, "a") as f:
                f.write(line + "\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = running_lm = running_reg = 0.0
        running_gates   = None   # [num_experts] cumulative sum for gate logging
        running_topk_counts = None  # [num_experts] expert 선택 횟수 누적

        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                out = model(
                    ts=batch["ts"],
                    ts_len_mask=batch["ts_len_mask"],
                    series_mask=batch["series_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    q_ids=batch["q_ids"],
                    q_mask=batch["q_mask"],
                    global_step=global_step,
                )
                loss = out["loss"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            running_lm   += out["lm_loss"].item()
            running_reg  += out["router_reg_loss"].item()
            # gates: [B*S, E] → real-series-weighted mean → 누적
            gates_flat  = out["gates"].float().cpu()           # [B*S, E]
            smask_flat  = batch["series_mask"].view(-1).float().cpu()  # [B*S]
            real_count  = smask_flat.sum().clamp_min(1.0)
            batch_gate  = (gates_flat * smask_flat.unsqueeze(-1)).sum(0) / real_count  # [E]
            running_gates = batch_gate if running_gates is None else running_gates + batch_gate
            # topk_indices: [B*S, K] → expert별 선택 횟수 누적 (real series만)
            E = batch_gate.numel()
            batch_counts = torch.zeros(E)
            topk_cpu = out["topk_indices"].cpu()  # [B*S, K]
            for e in range(E):
                batch_counts[e] = ((topk_cpu == e).float() * smask_flat.unsqueeze(-1)).sum()
            running_topk_counts = batch_counts if running_topk_counts is None \
                                               else running_topk_counts + batch_counts

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % LOG_STEPS == 0:
                    avg_loss    = running_loss / step
                    avg_lm      = running_lm   / step
                    avg_reg     = running_reg  / step
                    avg_gates   = running_gates / step   # [num_experts]
                    gate_str    = " ".join(f"{g:.3f}" for g in avg_gates.tolist())
                    expert_str  = format_expert_selection(running_topk_counts)
                    lr_now      = scheduler.get_last_lr()[0]
                    warmup_flag = "" if global_step >= accelerator.unwrap_model(model).topk_warmup_steps \
                                     else " [soft]"
                    write_log(
                        f"Epoch {epoch} | Step {global_step}{warmup_flag} | "
                        f"loss={avg_loss:.4f} (lm={avg_lm:.4f} reg={avg_reg:.4f}) | "
                        f"gates=[{gate_str}] | {expert_str} | lr={lr_now:.2e}"
                    )

                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(
                        f"step-{global_step}", model, tokenizer, accelerator, args.output_dir
                    )
                    write_log(f"Checkpoint saved → step-{global_step}")

        # ---- Validation ----
        model.eval()
        val_loss_sum, val_acc_sum, val_steps = 0.0, 0.0, 0
        # task별 gate 누적: {task: [sum_gates, count]}
        from collections import defaultdict
        task_gate_sums   = defaultdict(lambda: torch.zeros(3))
        task_gate_counts = defaultdict(int)

        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    ts=batch["ts"],
                    ts_len_mask=batch["ts_len_mask"],
                    series_mask=batch["series_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    q_ids=batch["q_ids"],
                    q_mask=batch["q_mask"],
                )
                val_loss_sum += out["loss"].item()
                val_acc_sum  += out["token_acc"].item()
                val_steps    += 1

                # task별 gate 누적 (per-sample mean across real series)
                S_val      = out["num_series"]
                E_val      = out["gates"].size(-1)
                gates_bs   = out["gates"].float().cpu().view(-1, S_val, E_val)  # [B, S, E]
                smask_bs   = batch["series_mask"].cpu()                          # [B, S]
                for i, task in enumerate(batch["tasks"]):
                    real_g = gates_bs[i][smask_bs[i]]   # [num_real_series, E]
                    task_gate_sums[task]   += real_g.mean(0)
                    task_gate_counts[task] += 1

        val_loss   = val_loss_sum / max(val_steps, 1)
        val_acc    = val_acc_sum  / max(val_steps, 1)

        val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
        val_acc_tensor  = torch.tensor(val_acc,  device=accelerator.device)
        val_loss_tensor = accelerator.reduce(val_loss_tensor, reduction="mean")
        val_acc_tensor  = accelerator.reduce(val_acc_tensor,  reduction="mean")
        val_loss = val_loss_tensor.item()
        val_acc  = val_acc_tensor.item()

        write_log(f"Epoch {epoch} done | Val loss: {val_loss:.4f} | Val token acc: {val_acc*100:.2f}%")
        if accelerator.is_main_process:
            print("  [Task-level gate avg]  (local | global | cross)")
            gate_lines = []
            for task in sorted(task_gate_sums.keys()):
                cnt  = task_gate_counts[task]
                avg  = task_gate_sums[task] / cnt
                gstr = " ".join(f"{v:.3f}" for v in avg.tolist())
                line = f"    {task:12s}: [{gstr}]  (n={cnt})"
                print(line)
                gate_lines.append(line)
            print()
            with open(log_path, "a") as f:
                f.write("  [Task-level gate avg]  (local | global | cross)\n")
                f.write("\n".join(gate_lines) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint("best", model, tokenizer, accelerator, args.output_dir)
            write_log(f"Best model saved → {args.output_dir}/best  (val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%)")
        else:
            no_improve += 1
            write_log(f"No improvement for {no_improve} epoch(s).")
            if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
                write_log(f"Early stopping triggered (patience={args.early_stopping_patience}).")
                break

    # ---- Final save ----
    save_checkpoint("final", model, tokenizer, accelerator, args.output_dir)
    write_log(f"Training complete. Final → {args.output_dir}/final")


# ---------------------------------------------------------------------------
# Frozen-LLM training support
# ---------------------------------------------------------------------------

FROZEN_OUTPUT_DIR = os.path.join(_HERE, "checkpoints", "checkpoint_frozen")
TASK_NAME = None


class FilteredTSAQAWithTSDataset(TSAQAWithTSDataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_text_len: int,
        max_ts_len: int,
        debug_label_samples: int = 0,
        split_name: str = "train",
        task_name: Optional[str] = None,
    ):
        self.task_name = self._normalize_task_name(task_name)
        super().__init__(
            parquet_path=parquet_path,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_ts_len=max_ts_len,
            debug_label_samples=debug_label_samples,
            split_name=split_name,
        )

        raw_tasks = []
        if self.has_task_col:
            raw_tasks = (
                self.df["task"]
                .fillna("")
                .astype(str)
                .map(str.strip)
                .tolist()
            )
        self.available_tasks = sorted({t for t in raw_tasks if t})

        if self.task_name is not None:
            if not self.has_task_col:
                raise ValueError(
                    f"Requested task={self.task_name!r}, but `{parquet_path}` has no `task` column."
                )
            task_series = self.df["task"].fillna("").astype(str).map(str.strip)
            self.df = self.df.loc[task_series == self.task_name].reset_index(drop=True)
            if len(self.df) == 0:
                available = ", ".join(self.available_tasks) if self.available_tasks else "(none)"
                raise ValueError(
                    f"No rows found for task={self.task_name!r} in {parquet_path}. "
                    f"Available tasks: {available}"
                )

        self.selected_task = self.task_name or "all"
        self._refresh_answer_stats()

    @staticmethod
    def _normalize_task_name(task_name: Optional[str]) -> Optional[str]:
        if task_name is None:
            return None
        normalized = str(task_name).strip()
        if not normalized or normalized.lower() == "all":
            return None
        return normalized

    def _refresh_answer_stats(self) -> None:
        ans_norm = self.df["answer"].astype(str).map(normalize_answer_text)
        self.answer_value_counts = ans_norm.value_counts(dropna=False)
        bad_mask = ~ans_norm.str.fullmatch(ANSWER_PATTERN.pattern, na=False)
        self.answer_bad_count = int(bad_mask.sum())
        self.answer_bad_examples = ans_norm[bad_mask].head(10).tolist()


def print_trainable_parameters(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / max(total, 1)
    print(
        f"[Trainable params] trainable={trainable:,} | total={total:,} | ratio={ratio:.4f}%"
    )


def save_frozen_checkpoint(
    tag: str,
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
    output_dir: str,
    task_name: Optional[str],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    no_improve: int,
    epoch_completed: bool,
    step_in_epoch: int = 0,
) -> None:
    path = os.path.join(output_dir, tag)
    os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    unwrapped: FrozenAdaptiveLLMGFTSignalLLM = accelerator.unwrap_model(model)

    tokenizer_path = os.path.join(path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    custom_state = {
        k: v for k, v in unwrapped.state_dict().items()
        if not k.startswith("gft.") and not k.startswith("llm.")
    }
    torch.save(custom_state, os.path.join(path, "custom_modules.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))

    trainer_state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "no_improve": no_improve,
        "epoch_completed": epoch_completed,
        "step_in_epoch": step_in_epoch,
    }
    torch.save(trainer_state, os.path.join(path, "trainer_state.pt"))

    metadata = {
        "checkpoint_tag": tag,
        "llm_frozen": True,
        "task_name": task_name or "all",
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "no_improve": no_improve,
        "epoch_completed": epoch_completed,
        "step_in_epoch": step_in_epoch,
        "resume_behavior": (
            "optimizer/scheduler/trainer state included; "
            "mid-epoch checkpoints restart that epoch from the beginning."
        ),
    }
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Checkpoint saved -> {path}")


def load_resume_state(
    resume_from: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    if not os.path.isdir(resume_from):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")

    custom_state_path = os.path.join(resume_from, "custom_modules.pt")
    trainer_state_path = os.path.join(resume_from, "trainer_state.pt")
    optimizer_state_path = os.path.join(resume_from, "optimizer.pt")
    scheduler_state_path = os.path.join(resume_from, "scheduler.pt")

    for required_path in (
        custom_state_path,
        trainer_state_path,
        optimizer_state_path,
        scheduler_state_path,
    ):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing resume file: {required_path}")

    custom_state = torch.load(custom_state_path, map_location="cpu")
    unwrapped: FrozenAdaptiveLLMGFTSignalLLM = accelerator.unwrap_model(model)
    incompatible = unwrapped.load_state_dict(custom_state, strict=False)
    missing_keys = [
        k for k in incompatible.missing_keys
        if not k.startswith("gft.") and not k.startswith("llm.")
    ]
    unexpected_keys = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Resume checkpoint is incompatible. "
            f"missing_keys={missing_keys} unexpected_keys={unexpected_keys}"
        )

    optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
    scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu"))

    trainer_state = torch.load(trainer_state_path, map_location="cpu")
    trainer_state.setdefault("epoch", 0)
    trainer_state.setdefault("global_step", 0)
    trainer_state.setdefault("best_val_loss", float("inf"))
    trainer_state.setdefault("no_improve", 0)
    trainer_state.setdefault("epoch_completed", False)
    trainer_state.setdefault("step_in_epoch", 0)
    return trainer_state


def peek_resume_trainer_state(resume_from: str) -> Dict[str, Any]:
    trainer_state_path = os.path.join(resume_from, "trainer_state.pt")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"Missing resume file: {trainer_state_path}")

    trainer_state = torch.load(trainer_state_path, map_location="cpu")
    trainer_state.setdefault("epoch", 0)
    trainer_state.setdefault("global_step", 0)
    trainer_state.setdefault("best_val_loss", float("inf"))
    trainer_state.setdefault("no_improve", 0)
    trainer_state.setdefault("epoch_completed", False)
    trainer_state.setdefault("step_in_epoch", 0)
    return trainer_state


def describe_dataset(split_name: str, dataset: FilteredTSAQAWithTSDataset) -> None:
    print(
        f"{split_name}: {len(dataset):,} rows | task={dataset.selected_task} "
        f"| available_tasks={dataset.available_tasks if dataset.available_tasks else ['(missing task col)']}"
    )
    print(f"[{split_name}] normalized answer unique={len(dataset.answer_value_counts):,}")
    print(dataset.answer_value_counts.head(20).to_string())
    print(f"[{split_name}] non-canonical answers={dataset.answer_bad_count:,}")
    if dataset.answer_bad_examples:
        print(f"[{split_name}] bad examples={dataset.answer_bad_examples}")


def main_frozen() -> None:
    parser = argparse.ArgumentParser(description="GFT + MoE + frozen LLM trainer")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_path", default=TRAIN_PATH)
    parser.add_argument("--val_path", default=VAL_PATH)
    parser.add_argument("--gft_ckpt", default=GFT_CKPT)
    parser.add_argument("--output_dir", default=FROZEN_OUTPUT_DIR)
    parser.add_argument("--resume_from", default=None,
                        help="Checkpoint directory to resume from, e.g. /path/to/final or /path/to/step-1000")
    parser.add_argument("--task_name", default=TASK_NAME,
                        help="Single task to train/evaluate. Use 'all' or empty for no filtering.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max_ts_len", type=int, default=MAX_TS_LEN)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_qformer_queries", type=int, default=NUM_QFORMER_QUERIES)
    parser.add_argument(
        "--qformer_layer",
        "--num_qformer_layers",
        dest="num_qformer_layers",
        type=int,
        default=NUM_QFORMER_LAYERS,
    )
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop if val_loss does not improve for N epochs. 0 = disabled.")
    parser.add_argument("--topk_warmup_steps", type=int, default=0,
                        help="Steps of soft routing before switching to masked top-k. 0 = top-k from start.")
    parser.add_argument("--debug_label_samples", type=int, default=0,
                        help="Print first N samples with label-mask debug information.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    # Adaptive ablation variants intentionally skip subsets of trainable modules
    # (for example router/MoE or raw-signal projection), so DDP must tolerate
    # unused parameters on each step.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[ddp_kwargs],
    )
    normalized_task_name = FilteredTSAQAWithTSDataset._normalize_task_name(args.task_name)

    tokenizer_source = args.model_name
    if args.resume_from:
        resume_tokenizer_dir = os.path.join(args.resume_from, "tokenizer")
        if os.path.isdir(resume_tokenizer_dir):
            tokenizer_source = resume_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added_fallback_chat_template = ensure_tokenizer_chat_template(tokenizer)
    if accelerator.is_main_process:
        if args.resume_from:
            print(f"Tokenizer source: {tokenizer_source}")
        if added_fallback_chat_template:
            print(
                "Tokenizer has no chat_template; using fallback Question/Answer template."
            )
        print_tokenizer_answer_debug(tokenizer)

    if accelerator.is_main_process:
        print("Loading datasets ...")

    train_dataset = FilteredTSAQAWithTSDataset(
        args.train_path,
        tokenizer,
        args.max_seq_len,
        args.max_ts_len,
        debug_label_samples=args.debug_label_samples,
        split_name="train",
        task_name=normalized_task_name,
    )
    val_dataset = FilteredTSAQAWithTSDataset(
        args.val_path,
        tokenizer,
        args.max_seq_len,
        args.max_ts_len,
        debug_label_samples=0,
        split_name="val",
        task_name=normalized_task_name,
    )

    if accelerator.is_main_process:
        describe_dataset("train", train_dataset)
        describe_dataset("val", val_dataset)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"Loading GFT checkpoint: {args.gft_ckpt}")

    gft_ckpt = torch.load(args.gft_ckpt, map_location="cpu")
    gft_config = GFTVarLenConfig(**gft_ckpt["config"])
    gft_state = gft_ckpt["model"]

    if accelerator.is_main_process:
        print(f"Loading frozen LLM: {args.model_name}")

    llm = load_llm_for_model_name(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    llm_hidden_dim = get_llm_hidden_dim(llm)
    model = FrozenAdaptiveLLMGFTSignalLLM(
        gft_config=gft_config,
        gft_state_dict=gft_state,
        llm=llm,
        llm_hidden_dim=llm_hidden_dim,
        ts_hidden_dim=TS_HIDDEN_DIM,
        num_qformer_queries=args.num_qformer_queries,
        num_qformer_layers=args.num_qformer_layers,
        top_k=args.top_k,
        router_entropy_weight=ROUTER_ENTROPY_W,
        router_balance_weight=ROUTER_BALANCE_W,
        topk_warmup_steps=args.topk_warmup_steps,
    )

    if accelerator.is_main_process:
        print_trainable_parameters(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found. Expected custom modules to remain trainable.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)

    resume_state_preview = None
    if args.resume_from:
        resume_state_preview = peek_resume_trainer_state(args.resume_from)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    resume_global_step = int(resume_state_preview["global_step"]) if resume_state_preview else 0
    total_steps = max(1, resume_global_step + steps_per_epoch * args.num_epochs)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    best_val_loss = float("inf")
    no_improve = 0
    start_epoch = 1
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train.log")

    def write_log(msg: str):
        if accelerator.is_main_process:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if args.resume_from:
        resume_state = load_resume_state(
            args.resume_from, model, optimizer, scheduler, accelerator
        )
        global_step = int(resume_state["global_step"])
        best_val_loss = float(resume_state["best_val_loss"])
        no_improve = int(resume_state["no_improve"])
        saved_epoch = int(resume_state["epoch"])
        epoch_completed = bool(resume_state["epoch_completed"])
        step_in_epoch = int(resume_state["step_in_epoch"])
        start_epoch = saved_epoch + 1 if epoch_completed else max(1, saved_epoch)
        if accelerator.is_main_process:
            resume_note = (
                f"Resuming from {args.resume_from} | saved_epoch={saved_epoch} "
                f"| start_epoch={start_epoch} | global_step={global_step}"
            )
            if not epoch_completed and step_in_epoch > 0:
                resume_note += (
                    f" | note: checkpoint was saved at step {step_in_epoch} within the epoch, "
                    "so this epoch will restart from the beginning."
                )
            write_log(resume_note)

    end_epoch = start_epoch + args.num_epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = running_lm = running_reg = 0.0
        running_gates = None
        running_topk_counts = None

        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                out = model(
                    ts=batch["ts"],
                    ts_len_mask=batch["ts_len_mask"],
                    series_mask=batch["series_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    q_ids=batch["q_ids"],
                    q_mask=batch["q_mask"],
                    global_step=global_step,
                )
                loss = out["loss"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            running_lm += out["lm_loss"].item()
            running_reg += out["router_reg_loss"].item()
            gates_flat = out["gates"].float().cpu()
            smask_flat = batch["series_mask"].view(-1).float().cpu()
            real_count = smask_flat.sum().clamp_min(1.0)
            batch_gate = (gates_flat * smask_flat.unsqueeze(-1)).sum(0) / real_count
            running_gates = batch_gate if running_gates is None else running_gates + batch_gate
            num_experts = batch_gate.numel()
            batch_counts = torch.zeros(num_experts)
            topk_cpu = out["topk_indices"].cpu()
            for e in range(num_experts):
                batch_counts[e] = ((topk_cpu == e).float() * smask_flat.unsqueeze(-1)).sum()
            running_topk_counts = (
                batch_counts if running_topk_counts is None else running_topk_counts + batch_counts
            )

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % LOG_STEPS == 0:
                    avg_loss = running_loss / step
                    avg_lm = running_lm / step
                    avg_reg = running_reg / step
                    avg_gates = running_gates / step
                    gate_str = " ".join(f"{g:.3f}" for g in avg_gates.tolist())
                    expert_str = format_expert_selection(running_topk_counts)
                    lr_now = scheduler.get_last_lr()[0]
                    warmup_flag = "" if global_step >= accelerator.unwrap_model(model).topk_warmup_steps else " [soft]"
                    write_log(
                        f"Epoch {epoch} | Step {global_step}{warmup_flag} | "
                        f"loss={avg_loss:.4f} (lm={avg_lm:.4f} reg={avg_reg:.4f}) | "
                        f"gates=[{gate_str}] | {expert_str} | lr={lr_now:.2e}"
                    )

                if global_step % SAVE_STEPS == 0:
                    save_frozen_checkpoint(
                        f"step-{global_step}",
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        accelerator,
                        args.output_dir,
                        normalized_task_name,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        no_improve=no_improve,
                        epoch_completed=False,
                        step_in_epoch=step,
                    )
                    write_log(f"Checkpoint saved -> step-{global_step}")

        model.eval()
        val_loss_sum, val_acc_sum, val_steps = 0.0, 0.0, 0
        task_gate_sums = defaultdict(lambda: torch.zeros(3))
        task_gate_counts = defaultdict(int)

        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    ts=batch["ts"],
                    ts_len_mask=batch["ts_len_mask"],
                    series_mask=batch["series_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    q_ids=batch["q_ids"],
                    q_mask=batch["q_mask"],
                )
                val_loss_sum += out["loss"].item()
                val_acc_sum += out["token_acc"].item()
                val_steps += 1

                num_series = out["num_series"]
                num_experts = out["gates"].size(-1)
                gates_bs = out["gates"].float().cpu().view(-1, num_series, num_experts)
                smask_bs = batch["series_mask"].cpu()
                for i, task in enumerate(batch["tasks"]):
                    real_gates = gates_bs[i][smask_bs[i]]
                    task_gate_sums[task] += real_gates.mean(0)
                    task_gate_counts[task] += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        val_acc = val_acc_sum / max(val_steps, 1)

        val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
        val_acc_tensor = torch.tensor(val_acc, device=accelerator.device)
        val_loss_tensor = accelerator.reduce(val_loss_tensor, reduction="mean")
        val_acc_tensor = accelerator.reduce(val_acc_tensor, reduction="mean")
        val_loss = val_loss_tensor.item()
        val_acc = val_acc_tensor.item()

        write_log(f"Epoch {epoch} done | Val loss: {val_loss:.4f} | Val token acc: {val_acc*100:.2f}%")
        if accelerator.is_main_process:
            print("  [Task-level gate avg]  (local | global | cross)")
            gate_lines = []
            for task in sorted(task_gate_sums.keys()):
                cnt = task_gate_counts[task]
                avg = task_gate_sums[task] / cnt
                gstr = " ".join(f"{v:.3f}" for v in avg.tolist())
                line = f"    {task:12s}: [{gstr}]  (n={cnt})"
                print(line)
                gate_lines.append(line)
            print()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("  [Task-level gate avg]  (local | global | cross)\n")
                f.write("\n".join(gate_lines) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_frozen_checkpoint(
                "best",
                model,
                tokenizer,
                optimizer,
                scheduler,
                accelerator,
                args.output_dir,
                normalized_task_name,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                no_improve=no_improve,
                epoch_completed=True,
            )
            write_log(
                f"Best model saved -> {args.output_dir}/best  "
                f"(val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%)"
            )
        else:
            no_improve += 1
            write_log(f"No improvement for {no_improve} epoch(s).")
            if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
                write_log(f"Early stopping triggered (patience={args.early_stopping_patience}).")
                break

    save_frozen_checkpoint(
        "final",
        model,
        tokenizer,
        optimizer,
        scheduler,
        accelerator,
        args.output_dir,
        normalized_task_name,
        epoch=end_epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        no_improve=no_improve,
        epoch_completed=True,
    )
    write_log(f"Training complete. Final -> {args.output_dir}/final")


# ---------------------------------------------------------------------------
# Adaptive temporal pooling
# ---------------------------------------------------------------------------

LOCAL_SUMMARY_TOKENS = 8
GLOBAL_SUMMARY_TOKENS = 4
CROSS_SUMMARY_TOKENS = 8
QFORMER_INPUT_TOKENS = LOCAL_SUMMARY_TOKENS + GLOBAL_SUMMARY_TOKENS + CROSS_SUMMARY_TOKENS + 1
ABLATION_MODES = (
    "raw",
    "fine",
    "coarse",
    "fused",
    "moe",
    "moe_without_routing",
    "signal_moe",
)


def temporal_adaptive_pool(
    tokens: torch.Tensor,
    num_tokens: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool a temporal token sequence into a fixed number of summary tokens.

    Args:
        tokens: [B, T, D]
        num_tokens: pooled token count
        mask: optional [B, T] bool mask for valid time positions

    Returns:
        [B, num_tokens, D]
    """
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens with shape [B, T, D], got {tuple(tokens.shape)}")
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    if mask is None:
        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), num_tokens)
        return pooled.transpose(1, 2)

    if mask.dim() != 2 or mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"Expected mask with shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}"
        )

    mask = mask.to(device=tokens.device, dtype=torch.bool)
    bsz, _, hidden_dim = tokens.shape

    # GFT/time-series masks are left-aligned valid tokens followed by padding.
    # Pool by each row's valid length without CUDA boolean advanced indexing.
    lengths = mask.sum(dim=1).clamp_min(1).to(torch.long)
    masked_tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
    prefix = torch.cat(
        [tokens.new_zeros(bsz, 1, hidden_dim), masked_tokens.cumsum(dim=1)],
        dim=1,
    )

    out_idx = torch.arange(num_tokens, device=tokens.device, dtype=torch.long)
    starts = torch.div(
        out_idx.unsqueeze(0) * lengths.unsqueeze(1),
        num_tokens,
        rounding_mode="floor",
    )
    ends = torch.div(
        (out_idx + 1).unsqueeze(0) * lengths.unsqueeze(1) + num_tokens - 1,
        num_tokens,
        rounding_mode="floor",
    )
    ends = torch.maximum(ends, starts + 1)

    gather_shape = (-1, -1, hidden_dim)
    start_sum = prefix.gather(1, starts.unsqueeze(-1).expand(*gather_shape))
    end_sum = prefix.gather(1, ends.unsqueeze(-1).expand(*gather_shape))
    denom = (ends - starts).clamp_min(1).unsqueeze(-1).to(dtype=tokens.dtype)
    return (end_sum - start_sum) / denom


def _scale_summary_tokens(summary_tokens: torch.Tensor, gate_values: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sample gate values onto [B, N, D] summary tokens."""
    gate = gate_values.to(dtype=summary_tokens.dtype, device=summary_tokens.device)
    if gate.dim() == 1:
        gate = gate.unsqueeze(-1)
    if gate.dim() != 2 or gate.size(1) != 1:
        raise ValueError(
            f"Expected gate values with shape [B] or [B, 1], got {tuple(gate_values.shape)}"
        )
    return summary_tokens * gate.unsqueeze(-1)


def build_single_source_qformer_input(
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_tokens: int = QFORMER_INPUT_TOKENS,
) -> torch.Tensor:
    """Pool one signal source into a fixed-length Q-Former input sequence."""
    return temporal_adaptive_pool(tokens, num_tokens=num_tokens, mask=mask)


def build_adaptive_qformer_input(
    moe_out: dict,
    gates: torch.Tensor,
    fine_mask: Optional[torch.Tensor] = None,
    coarse_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create Q-Former inputs with multi-token temporal summaries.

    Output shape is [B, 21, D] = [B, 8 + 4 + 8 + 1, D].
    """
    local_summary = temporal_adaptive_pool(
        moe_out["local_tokens"],
        LOCAL_SUMMARY_TOKENS,
        mask=fine_mask,
    )
    global_summary = temporal_adaptive_pool(
        moe_out["global_tokens"],
        GLOBAL_SUMMARY_TOKENS,
        mask=coarse_mask,
    )
    cross_summary = temporal_adaptive_pool(
        moe_out["cross_tokens"],
        CROSS_SUMMARY_TOKENS,
        mask=fine_mask,
    )

    local_summary = _scale_summary_tokens(local_summary, gates[:, 0])
    global_summary = _scale_summary_tokens(global_summary, gates[:, 1])
    cross_summary = _scale_summary_tokens(cross_summary, gates[:, 2])

    routed = moe_out["routed_state"].unsqueeze(1)
    return torch.cat([local_summary, global_summary, cross_summary, routed], dim=1)


# ---------------------------------------------------------------------------
# Adaptive model and unified entry point
# ---------------------------------------------------------------------------

_WRAPPER_ABLATION_MODE = "moe"


class AdaptiveGFTSignalLLM(GFTSignalLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_mode = _WRAPPER_ABLATION_MODE
        self.raw_signal_proj = None
        if self.ablation_mode == "raw":
            self.raw_signal_proj = nn.Linear(1, self.ts_hidden_dim)
        self.direct_signal_proj = None
        if self.ablation_mode == "signal_moe":
            self.direct_signal_proj = nn.Linear(self.ts_hidden_dim, self.llm_hidden_dim)
            self._freeze_signal_moe_unused_modules()
        self.debug_signal_shapes = os.environ.get("SIGNALLLM_DEBUG_SIGNAL_SHAPES") == "1"
        self._signal_shape_debug_printed = False

    def _freeze_signal_moe_unused_modules(self) -> None:
        """Keep the signal_moe ablation to GFT -> MoE -> direct LLM projection."""
        for module in (self.q_proj, self.router, self.pre_qformer_proj, self.qformer):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _num_experts(self) -> int:
        return getattr(self.router, "num_experts", 3)

    def _empty_topk_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)

    def _zero_router_gates(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self._num_experts(), device=device, dtype=dtype)

    def _uniform_router_gates(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_experts = self._num_experts()
        return torch.full(
            (batch_size, num_experts),
            fill_value=1.0 / float(num_experts),
            device=device,
            dtype=dtype,
        )

    def _maybe_log_signal_shapes(
        self,
        source_name: str,
        source_tokens: torch.Tensor,
        connector_tokens: torch.Tensor,
    ) -> None:
        if self.debug_signal_shapes and not self._signal_shape_debug_printed:
            print(
                "[SignalTokenDebug] "
                f"mode={self.ablation_mode} "
                f"source={source_name} "
                f"source_shape={tuple(source_tokens.shape)} "
                f"connector_input={tuple(connector_tokens.shape)}"
            )
            self._signal_shape_debug_printed = True

    def _build_single_source_tokens(
        self,
        source_name: str,
        source_tokens: torch.Tensor,
        source_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tokens = build_single_source_qformer_input(source_tokens, mask=source_mask)
        self._maybe_log_signal_shapes(source_name, source_tokens, tokens)
        return self.pre_qformer_proj(tokens)

    def _build_raw_signal_tokens(
        self,
        ts_flat: torch.Tensor,
        mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        if self.raw_signal_proj is None:
            raise RuntimeError("raw_signal_proj is only available when ablation_mode='raw'.")
        raw_hidden = self.raw_signal_proj(ts_flat.float())
        return self._build_single_source_tokens("raw", raw_hidden, mask_flat)

    def _build_signal_tokens(
        self,
        moe_out: dict,
        gates: torch.Tensor,
        fine_mask: Optional[torch.Tensor] = None,
        coarse_mask: Optional[torch.Tensor] = None,
        apply_pre_proj: bool = True,
    ) -> torch.Tensor:
        # [B, 21, D] = [B, 8(local) + 4(global) + 8(cross) + 1(routed), D]
        tokens = build_adaptive_qformer_input(
            moe_out,
            gates,
            fine_mask=fine_mask,
            coarse_mask=coarse_mask,
        )
        self._maybe_log_signal_shapes("moe", moe_out["local_tokens"], tokens)
        if apply_pre_proj:
            tokens = self.pre_qformer_proj(tokens)
        return tokens

    def _direct_signal_tokens_to_llm(self, signal_tokens: torch.Tensor) -> torch.Tensor:
        if self.direct_signal_proj is None:
            raise RuntimeError(
                "direct_signal_proj is only available when ablation_mode='signal_moe'."
            )
        return self.direct_signal_proj(signal_tokens)

    def forward(
        self,
        ts: torch.Tensor,
        ts_len_mask: torch.Tensor,
        series_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_series, ts_len, _ = ts.shape
        device = ts.device

        ts_flat = ts.view(bsz * num_series, ts_len, 1)
        mask_flat = ts_len_mask.view(bsz * num_series, ts_len)
        batch_series = bsz * num_series
        router_reg = torch.zeros((), device=device)

        if self.ablation_mode == "raw":
            signal_tokens = self._build_raw_signal_tokens(ts_flat, mask_flat)
            gates = self._zero_router_gates(batch_series, device=device, dtype=signal_tokens.dtype)
            topk_indices = self._empty_topk_indices(batch_series, device=device)
        else:
            with torch.no_grad():
                encoded = self.gft.encode_multigranularity(ts_flat.float(), input_mask=mask_flat)
            h_f = encoded["fine_tokens"].float()
            h_c = encoded["coarse_tokens"].float()
            h_fc = encoded["fused_tokens"].float()
            fine_mask = encoded["fine_mask"]
            coarse_mask = encoded["coarse_mask"]

            if self.ablation_mode == "fine":
                signal_tokens = self._build_single_source_tokens("fine", h_f, fine_mask)
                gates = self._zero_router_gates(batch_series, device=device, dtype=signal_tokens.dtype)
                topk_indices = self._empty_topk_indices(batch_series, device=device)
            elif self.ablation_mode == "coarse":
                signal_tokens = self._build_single_source_tokens("coarse", h_c, coarse_mask)
                gates = self._zero_router_gates(batch_series, device=device, dtype=signal_tokens.dtype)
                topk_indices = self._empty_topk_indices(batch_series, device=device)
            elif self.ablation_mode == "fused":
                signal_tokens = self._build_single_source_tokens("fused", h_fc, coarse_mask)
                gates = self._zero_router_gates(batch_series, device=device, dtype=signal_tokens.dtype)
                topk_indices = self._empty_topk_indices(batch_series, device=device)
            elif self.ablation_mode == "moe_without_routing":
                gates = self._uniform_router_gates(batch_series, device=device, dtype=h_f.dtype)
                topk_indices = self._empty_topk_indices(batch_series, device=device)
                moe_out = self.moe(h_f, h_c, gates, coarse_mask)
                signal_tokens = self._build_signal_tokens(
                    moe_out,
                    gates,
                    fine_mask=fine_mask,
                    coarse_mask=coarse_mask,
                )
            elif self.ablation_mode == "signal_moe":
                gates = self._uniform_router_gates(batch_series, device=device, dtype=h_f.dtype)
                topk_indices = self._empty_topk_indices(batch_series, device=device)
                moe_out = self.moe(h_f, h_c, gates, coarse_mask)
                signal_tokens = self._build_signal_tokens(
                    moe_out,
                    gates,
                    fine_mask=fine_mask,
                    coarse_mask=coarse_mask,
                    apply_pre_proj=False,
                )
            elif self.ablation_mode == "moe":
                q_vec_single = self._encode_question(q_ids, q_mask)
                q_vec = q_vec_single.unsqueeze(1).expand(-1, num_series, -1).reshape(batch_series, -1)

                use_topk = global_step >= self.topk_warmup_steps
                gates, topk_indices, router_aux = self.router(
                    q_vec,
                    h_f,
                    h_c,
                    coarse_mask=coarse_mask,
                    use_topk=use_topk,
                    fine_mask=fine_mask,
                )

                moe_out = self.moe(h_f, h_c, gates, coarse_mask)
                signal_tokens = self._build_signal_tokens(
                    moe_out,
                    gates,
                    fine_mask=fine_mask,
                    coarse_mask=coarse_mask,
                )
                router_reg = compute_router_reg_loss(
                    original_gates=router_aux["original_gates"],
                    masked_gates=router_aux["masked_gates"],
                    balance_weight=self.router_balance_weight,
                    entropy_weight=self.router_entropy_weight,
                )
            else:
                raise ValueError(f"Unsupported ablation_mode={self.ablation_mode}")

        if self.ablation_mode == "signal_moe":
            signal_embeds = self._direct_signal_tokens_to_llm(signal_tokens)
        else:
            signal_embeds = self.qformer(signal_tokens)

        num_queries, hidden_dim = signal_embeds.size(1), signal_embeds.size(2)
        signal_embeds = signal_embeds.view(bsz, num_series, num_queries, hidden_dim)
        smask = series_mask.unsqueeze(-1).unsqueeze(-1).float()
        signal_embeds = signal_embeds * smask
        all_sig = signal_embeds.view(bsz, num_series * num_queries, hidden_dim)

        text_embeds = self._embed_ids(input_ids)

        target_dtype = next(self.llm.parameters()).dtype
        sig_start = self.signal_start.to(dtype=target_dtype).expand(bsz, -1, -1)
        sig_end = self.signal_end.to(dtype=target_dtype).expand(bsz, -1, -1)
        all_sig = all_sig.to(dtype=target_dtype)
        txt_emb = text_embeds.to(dtype=target_dtype)

        fused_embeds = torch.cat([sig_start, all_sig, sig_end, txt_emb], dim=1)

        prefix_len = 1 + num_series * num_queries + 1
        sig_token_mask = series_mask.unsqueeze(-1).expand(-1, -1, num_queries).reshape(
            bsz, num_series * num_queries
        )
        boundary_ones = torch.ones(bsz, 1, device=device, dtype=attention_mask.dtype)
        sig_attn = torch.cat(
            [boundary_ones, sig_token_mask.to(attention_mask.dtype), boundary_ones],
            dim=1,
        )
        fused_mask = torch.cat([sig_attn, attention_mask], dim=1)

        full_labels = torch.full(
            (bsz, prefix_len + labels.size(1)),
            fill_value=-100,
            dtype=labels.dtype,
            device=device,
        )
        full_labels[:, prefix_len:] = labels

        lm_out = self.llm(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            labels=full_labels,
            return_dict=True,
        )
        lm_loss = lm_out.loss
        total_loss = lm_loss + router_reg

        shift_logits = lm_out.logits[:, :-1].float()
        shift_labels = full_labels[:, 1:]
        ans_mask = shift_labels != -100
        token_acc = torch.tensor(0.0, device=device)
        if ans_mask.any():
            preds = shift_logits.argmax(-1)
            token_acc = ((preds == shift_labels) & ans_mask).sum().float() / ans_mask.sum().float()

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "router_reg_loss": router_reg,
            "token_acc": token_acc.detach(),
            "gates": gates.detach(),
            "topk_indices": topk_indices.detach(),
            "num_series": num_series,
        }


class FrozenAdaptiveLLMGFTSignalLLM(AdaptiveGFTSignalLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.llm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self


def _parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frozen_llm",
        action="store_true",
        help="Use the frozen-LLM adaptive trainer instead of the LoRA trainer.",
    )
    parser.add_argument(
        "--ablation_mode",
        choices=ABLATION_MODES,
        default="moe",
        help="Signal path to feed into the Q-Former.",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    global _WRAPPER_ABLATION_MODE
    wrapper_args, remaining_argv = _parse_wrapper_args(sys.argv[1:])
    _WRAPPER_ABLATION_MODE = wrapper_args.ablation_mode
    sys.argv = [sys.argv[0], *remaining_argv]
    print(f"[AdaptiveTrainer] ablation_mode={_WRAPPER_ABLATION_MODE}")

    if wrapper_args.frozen_llm:
        main_frozen()
        return

    main_lora()


if __name__ == "__main__":
    main()
