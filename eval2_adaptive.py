"""
Unified adaptive GFT + MoE evaluator for TSAQA.

This file contains both LoRA and frozen-LLM evaluation paths. The adaptive
temporal pooling signal path is used in both modes.

Usage:
    python eval2_adaptive.py [LoRA options]
    python eval2_adaptive.py --frozen_llm [frozen options]
"""

from __future__ import annotations

import os
import re
import sys
import ast
import argparse
import random
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
from peft import PeftModel
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "MTE"))
sys.path.insert(0, os.path.join(_HERE, "QGR"))

from gft_varlen_forecast import GFTVarLenConfig, GFTVarLenForecaster  # noqa: E402
from granularity2 import (  # noqa: E402
    GranularityMoE,
    QFormerConnector,
    QuestionConditionedRouter,
    get_text_representative,
)

# ---------------------------------------------------------------------------
# Defaults (can all be overridden via argparse)
# ---------------------------------------------------------------------------
MODEL_NAME       = "Qwen/Qwen3-0.6B"
CHECKPOINT_DIR   = os.path.join(_HERE, "checkpoints", "checkpoint_lora", "best")
GFT_CKPT         = os.path.join(_HERE, "MTE", "checkpoint", "gft_varlen_best_forecast.pt")
TEST_PATH        = os.path.join(_HERE, "data", "TSAQA", "test.parquet")
OUTPUT_PATH      = os.path.join(_HERE, "results", "eval_results.csv")

BATCH_SIZE       = 16
MAX_SEQ_LEN      = 1024
MAX_TS_LEN       = 512
MAX_NEW_TOKENS   = 16

TS_HIDDEN_DIM       = 256
NUM_QFORMER_QUERIES = 8
NUM_QFORMER_LAYERS  = 2
TOP_K               = 3

SYSTEM_PROMPT = ""

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


FALLBACK_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ 'Question: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Answer: ' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Answer: ' }}{% endif %}"""


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


def ensure_tokenizer_chat_template(tokenizer) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    return True


EXPERT_NAMES = ("local", "global", "cross")
ANSWER_PATTERN = re.compile(r"^(?:[A-D]|[TF]|[A-D](?:,[A-D]){1,})$")


# ---------------------------------------------------------------------------
# Answer extraction & scoring
# ---------------------------------------------------------------------------

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


def clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        text = lines[0]
    for stop in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text.strip()


def extract_answer(text: str, question_type: str) -> str:
    text = clean_generated_text(text)

    if question_type == "ordering":
        m = re.search(r"[A-D](?:\s*,\s*[A-D]){3}", text, re.IGNORECASE)
        if m:
            return normalize_answer_text(m.group())
        letters = re.findall(r"\b[A-D]\b", text.upper())
        if letters:
            seen, unique = set(), []
            for l in letters:
                if l not in seen:
                    seen.add(l)
                    unique.append(l)
            return normalize_answer_text(",".join(unique[:4]))
        return normalize_answer_text(text[:15])

    elif question_type == "true_or_false":
        upper = text.upper()
        if upper.startswith("TRUE") or upper == "T":
            return "T"
        if upper.startswith("FALSE") or upper == "F":
            return "F"
        m = re.search(r"\b(T|F|TRUE|FALSE)\b", upper)
        if m:
            token = m.group(1)
            return "T" if token.startswith("T") else "F"
        return normalize_answer_text(text[:1])

    else:  # multiple_choices
        m = re.search(r"\b([A-D])\b", text.upper())
        if m:
            return m.group(1)
        m = re.search(r"[A-D]", text.upper())
        if m:
            return m.group(0)
        return normalize_answer_text(text[:1])


def score(ground_truth: str, prediction: str, question_type: str) -> float:
    gt   = normalize_answer_text(ground_truth)
    pred = normalize_answer_text(prediction)

    if question_type == "ordering":
        gt_list   = [x.strip().upper() for x in gt.split(",")]
        pred_list = [x.strip().upper() for x in pred.split(",")]
        pred_list  = pred_list[:len(gt_list)]
        pred_list += [""] * (len(gt_list) - len(pred_list))
        matches = sum(g == p for g, p in zip(gt_list, pred_list))
        return matches / len(gt_list)
    else:
        return 1.0 if gt == pred else 0.0


# ---------------------------------------------------------------------------
# TS parsing (same as train.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eval Dataset — no label masking needed; keep ground-truth for comparison
# ---------------------------------------------------------------------------

class TSAQAEvalDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_text_len: int, max_ts_len: int):
        cols  = ["input_ts", "question", "answer", "question_type", "task", "domain"]
        table = pq.read_table(parquet_path, columns=cols)
        self.df           = table.to_pandas().reset_index(drop=True)
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.max_ts_len   = max_ts_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        question = str(row["question"])
        answer   = str(row["answer"]).strip()

        # time series (multi-series aware; same policy as train2)
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

        # prefix only (question → generate answer)
        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        prefix_enc = self.tokenizer(
            prefix_text,
            truncation=True, max_length=self.max_text_len,
            padding=False, return_tensors=None,
        )

        # question-only tokens for router conditioning
        q_enc = self.tokenizer(
            question,
            truncation=True, max_length=self.max_text_len,
            padding=False, return_tensors=None,
        )

        return {
            "ts_list":       ts_tensors,            # List[[L_i,1]]
            "num_series":    len(ts_tensors),
            "input_ids":     prefix_enc["input_ids"],
            "q_ids":         q_enc["input_ids"],
            "answer":        answer,
            "question_type": str(row["question_type"]),
            "task":          str(row["task"]),
            "domain":        str(row["domain"]),
        }


def collate_eval(batch, pad_token_id: int) -> dict:
    B = len(batch)

    # pad time series: [B, S, L, 1]
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

    # pad prefix input_ids (left-pad for generation)
    max_text = max(len(b["input_ids"]) for b in batch)
    ids_list, attn_list = [], []
    for b in batch:
        pad = max_text - len(b["input_ids"])
        ids_list.append([pad_token_id] * pad + b["input_ids"])
        attn_list.append([0] * pad + [1] * len(b["input_ids"]))

    # pad question-only ids
    max_q = max(len(b["q_ids"]) for b in batch)
    q_ids_list, q_attn_list = [], []
    for b in batch:
        pad = max_q - len(b["q_ids"])
        q_ids_list.append(b["q_ids"] + [pad_token_id] * pad)
        q_attn_list.append([1] * len(b["q_ids"]) + [0] * pad)

    return {
        "ts":             ts_pad,       # [B, S, L, 1]
        "ts_len_mask":    ts_len_mask,  # [B, S, L]
        "series_mask":    series_mask,  # [B, S]
        "input_ids":      torch.tensor(ids_list,    dtype=torch.long),
        "attention_mask": torch.tensor(attn_list,   dtype=torch.long),
        "q_ids":          torch.tensor(q_ids_list,  dtype=torch.long),
        "q_mask":         torch.tensor(q_attn_list, dtype=torch.long),
        "answers":        [b["answer"]        for b in batch],
        "question_types": [b["question_type"] for b in batch],
        "tasks":          [b["task"]          for b in batch],
        "domains":        [b["domain"]        for b in batch],
    }


# ---------------------------------------------------------------------------
# LoRAEvalGFTSignalLLM (eval version — generate() support added)
# ---------------------------------------------------------------------------

class LoRAEvalGFTSignalLLM(nn.Module):
    def __init__(
        self,
        gft_config: GFTVarLenConfig,
        gft_state_dict: dict,
        llm,
        llm_hidden_dim: int,
        ts_hidden_dim: int = 256,
        num_qformer_queries: int = 8,
        num_qformer_layers: int = 2,
        num_experts: int = 3,
        top_k: int = TOP_K,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ts_hidden_dim  = ts_hidden_dim
        self.llm_hidden_dim = llm_hidden_dim

        # GFT encoder (frozen)
        self.gft = GFTVarLenForecaster(gft_config)
        self.gft.load_state_dict(gft_state_dict)
        for p in self.gft.parameters():
            p.requires_grad = False

        self.llm = llm

        self.q_proj = nn.Linear(llm_hidden_dim, ts_hidden_dim)

        self.router = QuestionConditionedRouter(
            hidden_dim=ts_hidden_dim,
            router_hidden_dim=ts_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.moe = GranularityMoE(
            hidden_dim=ts_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.pre_qformer_proj = nn.Linear(ts_hidden_dim, ts_hidden_dim)
        self.qformer = QFormerConnector(
            input_dim=ts_hidden_dim,
            output_dim=llm_hidden_dim,
            num_queries=num_qformer_queries,
            num_layers=num_qformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.signal_start = nn.Parameter(torch.randn(1, 1, llm_hidden_dim) * 0.02)
        self.signal_end   = nn.Parameter(torch.randn(1, 1, llm_hidden_dim) * 0.02)

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_fn = self.llm.get_input_embeddings()
        with torch.no_grad():
            emb = embed_fn(input_ids)
        return emb.float()

    def _encode_question(self, q_ids: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
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
        )
        return self.q_proj(q_repr)

    @torch.inference_mode()
    def generate(
        self,
        ts: torch.Tensor,
        ts_len_mask: torch.Tensor,
        series_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        max_new_tokens: int = 16,
        return_router_stats: bool = False,
        **gen_kwargs,
    ) -> torch.Tensor:
        if return_router_stats:
            fused_embeds, fused_mask, router_stats = self._build_fused_inputs(
                ts, ts_len_mask, series_mask, input_ids, attention_mask, q_ids, q_mask,
                return_router_stats=True,
            )
        else:
            fused_embeds, fused_mask = self._build_fused_inputs(
                ts, ts_len_mask, series_mask, input_ids, attention_mask, q_ids, q_mask
            )

        output_ids = self.llm.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            **gen_kwargs,
        )
        if return_router_stats:
            return output_ids, router_stats
        return output_ids


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_lora_model(args, device: torch.device) -> tuple:
    print(f"Loading tokenizer from {args.checkpoint_dir}/lora_adapter …")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.checkpoint_dir, "lora_adapter"),
        trust_remote_code=True,
        padding_side="left",   # left-pad for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added_fallback_chat_template = ensure_tokenizer_chat_template(tokenizer)
    if added_fallback_chat_template:
        print("Tokenizer has no chat_template; using fallback Question/Answer template.")

    print(f"Loading GFT checkpoint: {args.gft_ckpt}")
    gft_ckpt   = torch.load(args.gft_ckpt, map_location="cpu")
    gft_config = GFTVarLenConfig(**gft_ckpt["config"])
    gft_state  = gft_ckpt["model"]

    print(f"Loading base LLM: {args.model_name}")
    base_llm = load_llm_for_model_name(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    lora_adapter_path = os.path.join(args.checkpoint_dir, "lora_adapter")
    print(f"Loading LoRA adapter: {lora_adapter_path}")
    llm = PeftModel.from_pretrained(base_llm, lora_adapter_path)
    llm.eval()

    llm_hidden_dim = get_llm_hidden_dim(llm)

    model = AdaptiveEvalGFTSignalLLM(
        gft_config=gft_config,
        gft_state_dict=gft_state,
        llm=llm,
        llm_hidden_dim=llm_hidden_dim,
        ts_hidden_dim=TS_HIDDEN_DIM,
        num_qformer_queries=args.num_qformer_queries,
        num_qformer_layers=NUM_QFORMER_LAYERS,
        top_k=args.top_k,
    )

    custom_path = os.path.join(args.checkpoint_dir, "custom_modules.pt")
    print(f"Loading custom modules: {custom_path}")
    custom_state = torch.load(custom_path, map_location="cpu")

    # infer num_qformer_queries from checkpoint to avoid shape mismatch
    if "qformer.query_tokens" in custom_state:
        ckpt_nq = custom_state["qformer.query_tokens"].shape[1]
        if ckpt_nq != args.num_qformer_queries:
            print(f"  INFO — num_qformer_queries: {args.num_qformer_queries} → {ckpt_nq} (from checkpoint)")
            args.num_qformer_queries = ckpt_nq
            # rebuild model with correct query count
            model = AdaptiveEvalGFTSignalLLM(
                gft_config=gft_config,
                gft_state_dict=gft_state,
                llm=llm,
                llm_hidden_dim=llm_hidden_dim,
                ts_hidden_dim=TS_HIDDEN_DIM,
                num_qformer_queries=ckpt_nq,
                num_qformer_layers=NUM_QFORMER_LAYERS,
                top_k=args.top_k,
            )

    missing, unexpected = model.load_state_dict(custom_state, strict=False)

    if missing:
        # only llm.* and gft.* should be missing (loaded separately)
        non_llm_missing = [k for k in missing if not k.startswith("llm.") and not k.startswith("gft.")]
        if non_llm_missing:
            print(f"  WARNING — missing keys: {non_llm_missing}")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_lora() -> None:
    parser = argparse.ArgumentParser(description="GFT + MoE + LLM evaluator")
    parser.add_argument("--model_name",          default=MODEL_NAME)
    parser.add_argument("--checkpoint_dir",      default=CHECKPOINT_DIR)
    parser.add_argument("--gft_ckpt",            default=GFT_CKPT)
    parser.add_argument("--test_path",           default=TEST_PATH)
    parser.add_argument("--output_path",         default=OUTPUT_PATH)
    parser.add_argument("--batch_size",          type=int,   default=BATCH_SIZE)
    parser.add_argument("--max_seq_len",         type=int,   default=MAX_SEQ_LEN)
    parser.add_argument("--max_ts_len",          type=int,   default=MAX_TS_LEN)
    parser.add_argument("--max_new_tokens",      type=int,   default=MAX_NEW_TOKENS)
    parser.add_argument("--num_qformer_queries", type=int,   default=NUM_QFORMER_QUERIES)
    parser.add_argument("--top_k",               type=int,   default=TOP_K)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--gpu",                 default="0")
    args = parser.parse_args()
    set_seed(args.seed)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_lora_model(args, device)

    print(f"Loading test data: {args.test_path}")
    dataset = TSAQAEvalDataset(
        args.test_path, tokenizer, args.max_seq_len, args.max_ts_len
    )
    print(f"Test samples: {len(dataset):,}")

    _collate = lambda b: collate_eval(b, tokenizer.pad_token_id)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=4, pin_memory=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    results = []
    task_gate_sums   = defaultdict(lambda: torch.zeros(3))
    task_gate_counts = defaultdict(int)
    task_topk_counts = defaultdict(lambda: torch.zeros(3))

    for batch in tqdm(loader, desc="Evaluating"):
        ts             = batch["ts"].to(device)
        ts_len_mask    = batch["ts_len_mask"].to(device)
        series_mask    = batch["series_mask"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        q_ids          = batch["q_ids"].to(device)
        q_mask         = batch["q_mask"].to(device)

        output_ids, router_stats = model.generate(
            ts=ts,
            ts_len_mask=ts_len_mask,
            series_mask=series_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            q_ids=q_ids,
            q_mask=q_mask,
            max_new_tokens=args.max_new_tokens,
            return_router_stats=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gates_cpu = router_stats["gates"].float().cpu()               # [B, S, E]
        topk_cpu  = router_stats["topk_indices"].long().cpu()         # [B, S, K]
        smask_cpu = router_stats["series_mask"].bool().cpu()          # [B, S]

        # output_ids contains only generated tokens (inputs_embeds path)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print(decoded)
        for i, (gt, raw_pred, qtype, task, domain) in enumerate(zip(
            batch["answers"], decoded,
            batch["question_types"], batch["tasks"], batch["domains"],
        )):
            pred = extract_answer(raw_pred, qtype)
            acc  = score(gt, pred, qtype)

            real_gates = gates_cpu[i][smask_cpu[i]]
            if real_gates.numel() == 0:
                real_gates = torch.zeros(1, len(EXPERT_NAMES))
            gate_mean = real_gates.mean(0)

            task_gate_sums[task]   += gate_mean
            task_gate_counts[task] += 1

            real_topk = topk_cpu[i][smask_cpu[i]]
            for expert_idx in range(len(EXPERT_NAMES)):
                task_topk_counts[task][expert_idx] += (real_topk == expert_idx).sum().item()

            results.append({
                "ground_truth":  gt,
                "prediction":    pred,
                "acc":           acc,
                "question_type": qtype,
                "task":          task,
                "domain":        domain,
                "gate_local":    float(gate_mean[0].item()),
                "gate_global":   float(gate_mean[1].item()),
                "gate_cross":    float(gate_mean[2].item()),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved → {args.output_path}")

    overall_acc = results_df["acc"].mean()
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"{'='*50}")

    print("\n[By question_type]")
    for qtype, grp in results_df.groupby("question_type"):
        print(f"  {qtype:<20}: {grp['acc'].mean():.4f}  (n={len(grp):,})")

    print("\n[By task]")
    for task, grp in results_df.groupby("task"):
        print(f"  {task:<25}: {grp['acc'].mean():.4f}  (n={len(grp):,})")

    print("\n[By task, question_type]")
    for (task, ques), grp in results_df.groupby(["task", "question_type"]):
        print(f"  {task}, {ques:<25}: {grp['acc'].mean():.4f}  (n={len(grp):,})")

    print("\n[Task-level MoE gate avg]  (local | global | cross)")
    for task in sorted(task_gate_sums.keys()):
        cnt = task_gate_counts[task]
        avg = task_gate_sums[task] / max(cnt, 1)
        gstr = " ".join(f"{v:.3f}" for v in avg.tolist())

        sel = task_topk_counts[task]
        sel_total = sel.sum().clamp_min(1.0)
        sel_str = " ".join(
            f"{EXPERT_NAMES[e]}={sel[e] / sel_total * 100:.1f}%"
            for e in range(len(EXPERT_NAMES))
        )
        print(f"  {task:<25}: [{gstr}]  (n={cnt:,}) | topk {sel_str}")


# ---------------------------------------------------------------------------
# Frozen-LLM evaluation support
# ---------------------------------------------------------------------------

FROZEN_MODEL_NAME = "Qwen/Qwen3-0.6B"
FROZEN_CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints", "checkpoint_frozen", "best")
FROZEN_GFT_CKPT = os.path.join(_HERE, "MTE", "checkpoint", "gft_varlen_best_forecast.pt")
FROZEN_TEST_PATH = os.path.join(_HERE, "data", "TSAQA", "test.parquet")
FROZEN_OUTPUT_PATH = os.path.join(_HERE, "results", "eval_results_frozen.csv")
TASK_NAME = None

FROZEN_BATCH_SIZE = 16
FROZEN_MAX_SEQ_LEN = 1024
FROZEN_MAX_TS_LEN = 512
FROZEN_MAX_NEW_TOKENS = 16

FROZEN_TS_HIDDEN_DIM = 256
FROZEN_NUM_QFORMER_QUERIES = 8
FROZEN_NUM_QFORMER_LAYERS = 2
FROZEN_TOP_K = 3

FROZEN_SYSTEM_PROMPT = ""

FROZEN_FALLBACK_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ 'Question: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Answer: ' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Answer: ' }}{% endif %}"""


def set_frozen_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_frozen_llm_for_model_name(model_name: str, **model_kwargs):
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    model_kwargs["config"] = config
    if config.__class__.__name__ == "Mistral3Config":
        return AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_frozen_llm_hidden_dim(llm) -> int:
    config = getattr(llm, "config", None)
    hidden_size = getattr(config, "hidden_size", None) if config is not None else None
    text_config = getattr(config, "text_config", None) if config is not None else None
    if hidden_size is None and text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not infer LLM hidden size from model config.")
    return hidden_size


def ensure_frozen_tokenizer_chat_template(tokenizer) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = FROZEN_FALLBACK_CHAT_TEMPLATE
    return True


FROZEN_EXPERT_NAMES = ("local", "global", "cross")
FROZEN_ANSWER_PATTERN = re.compile(r"^(?:[A-D]|[TF]|[A-D](?:,[A-D]){1,})$")


def default_frozen_log_path(output_path: str) -> str:
    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs) or "."
    return os.path.join(output_dir, "eval.log")


def build_frozen_logger(log_path: str) -> Callable[[str], None]:
    log_dir = os.path.dirname(os.path.abspath(log_path)) or "."
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    def _log(message: str = "") -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    return _log


def format_frozen_expert_weights(weights: torch.Tensor) -> str:
    values = weights.detach().float().tolist()
    return ", ".join(
        f"{name}={value:.4f}" for name, value in zip(FROZEN_EXPERT_NAMES, values)
    )


def format_frozen_series_gate_details(gates: torch.Tensor) -> str:
    if gates.numel() == 0:
        return "none"
    chunks = []
    for idx, weights in enumerate(gates.detach().float()):
        chunks.append(f"s{idx}: {format_frozen_expert_weights(weights)}")
    return " | ".join(chunks)


def format_frozen_topk_counts(topk_indices: torch.Tensor) -> str:
    if topk_indices.numel() == 0:
        return "none"
    flat = topk_indices.detach().long().reshape(-1)
    counts = [(flat == expert_idx).sum().item() for expert_idx in range(len(FROZEN_EXPERT_NAMES))]
    total = max(sum(counts), 1)
    return ", ".join(
        f"{FROZEN_EXPERT_NAMES[expert_idx]}={counts[expert_idx] / total * 100:.1f}%"
        for expert_idx in range(len(FROZEN_EXPERT_NAMES))
    )


def normalize_frozen_answer_text(answer: str) -> str:
    s = str(answer).strip().strip('"').strip("'")
    s = re.sub(r"\s*,\s*", ",", s)
    up = s.upper()
    if up == "TRUE":
        return "T"
    if up == "FALSE":
        return "F"
    if FROZEN_ANSWER_PATTERN.fullmatch(up):
        return up
    return s


def clean_frozen_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        text = lines[0]
    for stop in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text.strip()


def extract_frozen_answer(text: str, question_type: str) -> str:
    text = clean_frozen_generated_text(text)

    if question_type == "ordering":
        m = re.search(r"[A-D](?:\s*,\s*[A-D]){3}", text, re.IGNORECASE)
        if m:
            return normalize_frozen_answer_text(m.group())
        letters = re.findall(r"\b[A-D]\b", text.upper())
        if letters:
            seen, unique = set(), []
            for letter in letters:
                if letter not in seen:
                    seen.add(letter)
                    unique.append(letter)
            return normalize_frozen_answer_text(",".join(unique[:4]))
        return normalize_frozen_answer_text(text[:15])

    if question_type == "true_or_false":
        upper = text.upper()
        if upper.startswith("TRUE") or upper == "T":
            return "T"
        if upper.startswith("FALSE") or upper == "F":
            return "F"
        m = re.search(r"\b(T|F|TRUE|FALSE)\b", upper)
        if m:
            token = m.group(1)
            return "T" if token.startswith("T") else "F"
        return normalize_frozen_answer_text(text[:1])

    m = re.search(r"\b([A-D])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"[A-D]", text.upper())
    if m:
        return m.group(0)
    return normalize_frozen_answer_text(text[:1])


def score_frozen(ground_truth: str, prediction: str, question_type: str) -> float:
    gt = normalize_frozen_answer_text(ground_truth)
    pred = normalize_frozen_answer_text(prediction)

    if question_type == "ordering":
        gt_list = [x.strip().upper() for x in gt.split(",")]
        pred_list = [x.strip().upper() for x in pred.split(",")]
        pred_list = pred_list[: len(gt_list)]
        pred_list += [""] * (len(gt_list) - len(pred_list))
        matches = sum(g == p for g, p in zip(gt_list, pred_list))
        return matches / len(gt_list)

    return 1.0 if gt == pred else 0.0


def _frozen_to_python_obj(v):
    if isinstance(v, str):
        return ast.literal_eval(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return v
    raise TypeError(f"Unsupported input_ts type: {type(v)}")


def _frozen_is_nested(seq) -> bool:
    if not isinstance(seq, (list, tuple)) or len(seq) == 0:
        return False
    return any(isinstance(x, (list, tuple, np.ndarray)) for x in seq)


def parse_frozen_ts_field(v) -> List[List[float]]:
    seq = _frozen_to_python_obj(v)

    if not _frozen_is_nested(seq):
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


class TSAQAEvalFrozenDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_text_len: int,
        max_ts_len: int,
        task_name: Optional[str] = None,
    ):
        available_cols = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        self.has_task_col = "task" in available_cols

        base_cols = ["input_ts", "question", "answer", "question_type", "domain"]
        cols = base_cols + (["task"] if self.has_task_col else [])
        table = pq.read_table(parquet_path, columns=cols)
        self.df = table.to_pandas().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_ts_len = max_ts_len

        self.task_name = self._normalize_task_name(task_name)
        raw_tasks = []
        if self.has_task_col:
            raw_tasks = self.df["task"].fillna("").astype(str).map(str.strip).tolist()
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

    @staticmethod
    def _normalize_task_name(task_name: Optional[str]) -> Optional[str]:
        if task_name is None:
            return None
        normalized = str(task_name).strip()
        if not normalized or normalized.lower() == "all":
            return None
        return normalized

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"]).strip()

        try:
            series_list = parse_frozen_ts_field(row["input_ts"])
        except Exception:
            series_list = [[0.0]]
        if not series_list:
            series_list = [[0.0]]

        ts_tensors = []
        for series in series_list:
            series = series[-self.max_ts_len :]
            if not series:
                series = [0.0]
            ts_tensors.append(torch.tensor(series, dtype=torch.float32).unsqueeze(-1))

        prefix_messages = [
            {"role": "system", "content": FROZEN_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        prefix_enc = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.max_text_len,
            padding=False,
            return_tensors=None,
        )

        q_enc = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_text_len,
            padding=False,
            return_tensors=None,
        )

        task = "other"
        if self.has_task_col and "task" in row and not pd.isna(row["task"]):
            task_value = str(row["task"]).strip()
            if task_value:
                task = task_value

        return {
            "ts_list": ts_tensors,
            "num_series": len(ts_tensors),
            "input_ids": prefix_enc["input_ids"],
            "q_ids": q_enc["input_ids"],
            "answer": answer,
            "question_type": str(row["question_type"]),
            "task": task,
            "domain": str(row["domain"]),
        }


def collate_frozen_eval(batch, pad_token_id: int) -> dict:
    batch_size = len(batch)

    max_num_series = max(b["num_series"] for b in batch)
    all_lengths = [t.size(0) for b in batch for t in b["ts_list"]]
    max_ts = max(all_lengths) if all_lengths else 1

    ts_pad = torch.zeros(batch_size, max_num_series, max_ts, 1)
    ts_len_mask = torch.zeros(batch_size, max_num_series, max_ts, dtype=torch.bool)
    series_mask = torch.zeros(batch_size, max_num_series, dtype=torch.bool)
    for i, b in enumerate(batch):
        for s_idx, ts_tensor in enumerate(b["ts_list"]):
            length = ts_tensor.size(0)
            ts_pad[i, s_idx, :length] = ts_tensor
            ts_len_mask[i, s_idx, :length] = True
            series_mask[i, s_idx] = True

    max_text = max(len(b["input_ids"]) for b in batch)
    ids_list, attn_list = [], []
    for b in batch:
        pad = max_text - len(b["input_ids"])
        ids_list.append([pad_token_id] * pad + b["input_ids"])
        attn_list.append([0] * pad + [1] * len(b["input_ids"]))

    max_q = max(len(b["q_ids"]) for b in batch)
    q_ids_list, q_attn_list = [], []
    for b in batch:
        pad = max_q - len(b["q_ids"])
        q_ids_list.append(b["q_ids"] + [pad_token_id] * pad)
        q_attn_list.append([1] * len(b["q_ids"]) + [0] * pad)

    return {
        "ts": ts_pad,
        "ts_len_mask": ts_len_mask,
        "series_mask": series_mask,
        "input_ids": torch.tensor(ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        "q_ids": torch.tensor(q_ids_list, dtype=torch.long),
        "q_mask": torch.tensor(q_attn_list, dtype=torch.long),
        "answers": [b["answer"] for b in batch],
        "question_types": [b["question_type"] for b in batch],
        "tasks": [b["task"] for b in batch],
        "domains": [b["domain"] for b in batch],
    }


class FrozenEvalGFTSignalLLM(nn.Module):
    def __init__(
        self,
        gft_config: GFTVarLenConfig,
        gft_state_dict: dict,
        llm,
        llm_hidden_dim: int,
        ts_hidden_dim: int = 256,
        num_qformer_queries: int = 8,
        num_qformer_layers: int = 2,
        num_experts: int = 3,
        top_k: int = FROZEN_TOP_K,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ts_hidden_dim = ts_hidden_dim
        self.llm_hidden_dim = llm_hidden_dim

        self.gft = GFTVarLenForecaster(gft_config)
        self.gft.load_state_dict(gft_state_dict)
        for p in self.gft.parameters():
            p.requires_grad = False

        self.llm = llm
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        self.q_proj = nn.Linear(llm_hidden_dim, ts_hidden_dim)

        self.router = QuestionConditionedRouter(
            hidden_dim=ts_hidden_dim,
            router_hidden_dim=ts_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.moe = GranularityMoE(
            hidden_dim=ts_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.pre_qformer_proj = nn.Linear(ts_hidden_dim, ts_hidden_dim)
        self.qformer = QFormerConnector(
            input_dim=ts_hidden_dim,
            output_dim=llm_hidden_dim,
            num_queries=num_qformer_queries,
            num_layers=num_qformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.signal_start = nn.Parameter(torch.randn(1, 1, llm_hidden_dim) * 0.02)
        self.signal_end = nn.Parameter(torch.randn(1, 1, llm_hidden_dim) * 0.02)

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_fn = self.llm.get_input_embeddings()
        with torch.no_grad():
            emb = embed_fn(input_ids)
        return emb.float()

    def _encode_question(self, q_ids: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.llm(
                input_ids=q_ids,
                attention_mask=q_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]

        q_repr = get_text_representative(
            last_hidden.float(),
            q_mask,
            strategy="last_token",
        )
        return self.q_proj(q_repr)

    @torch.inference_mode()
    def generate(
        self,
        ts: torch.Tensor,
        ts_len_mask: torch.Tensor,
        series_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        max_new_tokens: int = 16,
        return_router_stats: bool = False,
        **gen_kwargs,
    ) -> torch.Tensor:
        if return_router_stats:
            fused_embeds, fused_mask, router_stats = self._build_fused_inputs(
                ts,
                ts_len_mask,
                series_mask,
                input_ids,
                attention_mask,
                q_ids,
                q_mask,
                return_router_stats=True,
            )
        else:
            fused_embeds, fused_mask = self._build_fused_inputs(
                ts, ts_len_mask, series_mask, input_ids, attention_mask, q_ids, q_mask
            )

        output_ids = self.llm.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            **gen_kwargs,
        )
        if return_router_stats:
            return output_ids, router_stats
        return output_ids


def load_frozen_model(args, device: torch.device, log: Callable[[str], None] = print) -> tuple:
    tokenizer_path = os.path.join(args.checkpoint_dir, "tokenizer")
    if not os.path.isdir(tokenizer_path):
        tokenizer_path = args.checkpoint_dir
    log(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added_fallback_chat_template = ensure_frozen_tokenizer_chat_template(tokenizer)
    if added_fallback_chat_template:
        log("Tokenizer has no chat_template; using fallback Question/Answer template.")

    log(f"Loading GFT checkpoint: {args.gft_ckpt}")
    gft_ckpt = torch.load(args.gft_ckpt, map_location="cpu")
    gft_config = GFTVarLenConfig(**gft_ckpt["config"])
    gft_state = gft_ckpt["model"]

    log(f"Loading frozen base LLM: {args.model_name}")
    llm = load_frozen_llm_for_model_name(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    llm_hidden_dim = get_frozen_llm_hidden_dim(llm)

    model = AdaptiveFrozenEvalGFTSignalLLM(
        gft_config=gft_config,
        gft_state_dict=gft_state,
        llm=llm,
        llm_hidden_dim=llm_hidden_dim,
        ts_hidden_dim=FROZEN_TS_HIDDEN_DIM,
        num_qformer_queries=args.num_qformer_queries,
        num_qformer_layers=args.num_qformer_layers,
        top_k=args.top_k,
    )

    custom_path = os.path.join(args.checkpoint_dir, "custom_modules.pt")
    log(f"Loading custom modules: {custom_path}")
    custom_state = torch.load(custom_path, map_location="cpu")

    if "qformer.query_tokens" in custom_state:
        ckpt_nq = custom_state["qformer.query_tokens"].shape[1]
        if ckpt_nq != args.num_qformer_queries:
            log(
                f"  INFO - num_qformer_queries: {args.num_qformer_queries} -> {ckpt_nq} (from checkpoint)"
            )
            args.num_qformer_queries = ckpt_nq
            model = AdaptiveFrozenEvalGFTSignalLLM(
                gft_config=gft_config,
                gft_state_dict=gft_state,
                llm=llm,
                llm_hidden_dim=llm_hidden_dim,
                ts_hidden_dim=FROZEN_TS_HIDDEN_DIM,
                num_qformer_queries=ckpt_nq,
                num_qformer_layers=args.num_qformer_layers,
                top_k=args.top_k,
            )

    missing, unexpected = model.load_state_dict(custom_state, strict=False)

    if missing:
        non_backbone_missing = [
            k for k in missing if not k.startswith("llm.") and not k.startswith("gft.")
        ]
        if non_backbone_missing:
            log(f"  WARNING - missing keys: {non_backbone_missing}")
    if unexpected:
        log(f"  WARNING - unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    return model, tokenizer


def main_frozen() -> None:
    parser = argparse.ArgumentParser(description="GFT + MoE + frozen LLM evaluator")
    parser.add_argument("--model_name", default=FROZEN_MODEL_NAME)
    parser.add_argument("--checkpoint_dir", default=FROZEN_CHECKPOINT_DIR)
    parser.add_argument("--gft_ckpt", default=FROZEN_GFT_CKPT)
    parser.add_argument("--test_path", default=FROZEN_TEST_PATH)
    parser.add_argument("--output_path", default=FROZEN_OUTPUT_PATH)
    parser.add_argument("--log_path", default=None)
    parser.add_argument(
        "--task_name",
        default=TASK_NAME,
        help="Single task to evaluate. Use 'all' or empty for no filtering.",
    )
    parser.add_argument("--batch_size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=FROZEN_MAX_SEQ_LEN)
    parser.add_argument("--max_ts_len", type=int, default=FROZEN_MAX_TS_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=FROZEN_MAX_NEW_TOKENS)
    parser.add_argument("--num_qformer_queries", type=int, default=FROZEN_NUM_QFORMER_QUERIES)
    parser.add_argument(
        "--qformer_layer",
        "--num_qformer_layers",
        dest="num_qformer_layers",
        type=int,
        default=FROZEN_NUM_QFORMER_LAYERS,
    )
    parser.add_argument("--top_k", type=int, default=FROZEN_TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    set_frozen_seed(args.seed)

    if not args.log_path:
        args.log_path = default_frozen_log_path(args.output_path)
    log = build_frozen_logger(args.log_path)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"eval.log -> {args.log_path}")

    model, tokenizer = load_frozen_model(args, device, log=log)

    log(f"Loading test data: {args.test_path}")
    dataset = TSAQAEvalFrozenDataset(
        args.test_path,
        tokenizer,
        args.max_seq_len,
        args.max_ts_len,
        task_name=args.task_name,
    )
    log(
        f"Test samples: {len(dataset):,} | task={dataset.selected_task} "
        f"| available_tasks={dataset.available_tasks if dataset.available_tasks else ['(missing task col)']}"
    )

    _collate = lambda b: collate_frozen_eval(b, tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    results = []
    task_gate_sums = defaultdict(lambda: torch.zeros(3))
    task_gate_counts = defaultdict(int)
    task_topk_counts = defaultdict(lambda: torch.zeros(3))

    for batch in tqdm(loader, desc="Evaluating"):
        ts = batch["ts"].to(device)
        ts_len_mask = batch["ts_len_mask"].to(device)
        series_mask = batch["series_mask"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)

        output_ids, router_stats = model.generate(
            ts=ts,
            ts_len_mask=ts_len_mask,
            series_mask=series_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            q_ids=q_ids,
            q_mask=q_mask,
            max_new_tokens=args.max_new_tokens,
            return_router_stats=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gates_cpu = router_stats["gates"].float().cpu()
        topk_cpu = router_stats["topk_indices"].long().cpu()
        smask_cpu = router_stats["series_mask"].bool().cpu()

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for i, (gt, raw_pred, qtype, task, domain) in enumerate(
            zip(
                batch["answers"],
                decoded,
                batch["question_types"],
                batch["tasks"],
                batch["domains"],
            )
        ):
            pred = extract_frozen_answer(raw_pred, qtype)
            acc = score_frozen(gt, pred, qtype)

            real_gates = gates_cpu[i][smask_cpu[i]]
            if real_gates.numel() == 0:
                real_gates = torch.zeros(1, len(FROZEN_EXPERT_NAMES))
            gate_mean = real_gates.mean(0)

            task_gate_sums[task] += gate_mean
            task_gate_counts[task] += 1

            real_topk = topk_cpu[i][smask_cpu[i]]
            for expert_idx in range(len(FROZEN_EXPERT_NAMES)):
                task_topk_counts[task][expert_idx] += (real_topk == expert_idx).sum().item()

            gate_mean_str = format_frozen_expert_weights(gate_mean)
            series_gate_str = format_frozen_series_gate_details(real_gates)
            topk_str = format_frozen_topk_counts(real_topk)
            log(
                f"[sample {len(results):05d}] task={task} | domain={domain} | qtype={qtype} "
                f"| gt={gt} | pred={pred} | acc={acc:.4f}"
            )
            log(f"  mean_moe_weights: {gate_mean_str}")
            log(f"  series_moe_weights: {series_gate_str}")
            log(f"  topk_usage: {topk_str}")

            results.append(
                {
                    "ground_truth": gt,
                    "prediction": pred,
                    "acc": acc,
                    "question_type": qtype,
                    "task": task,
                    "domain": domain,
                    "gate_local": float(gate_mean[0].item()),
                    "gate_global": float(gate_mean[1].item()),
                    "gate_cross": float(gate_mean[2].item()),
                    "moe_weight_mean": gate_mean_str,
                    "moe_weight_series": series_gate_str,
                    "moe_topk_usage": topk_str,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    log(f"\nResults saved -> {args.output_path}")

    overall_acc = results_df["acc"].mean()
    log(f"\n{'=' * 50}")
    log(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    log(f"{'=' * 50}")

    log("\n[By question_type]")
    for qtype, grp in results_df.groupby("question_type"):
        log(f"  {qtype:<20}: {grp['acc'].mean():.4f}  (n={len(grp):,})")

    log("\n[By task]")
    for task, grp in results_df.groupby("task"):
        log(f"  {task:<25}: {grp['acc'].mean():.4f}  (n={len(grp):,})")

    log("\n[By task, question_type]")
    for (task, ques), grp in results_df.groupby(["task", "question_type"]):
        log(f"  {task}, {ques:<25}: {grp['acc'].mean():.4f}  (n={len(grp):,})")

    log("\n[Task-level MoE gate avg]  (local | global | cross)")
    for task in sorted(task_gate_sums.keys()):
        cnt = task_gate_counts[task]
        avg = task_gate_sums[task] / max(cnt, 1)
        gstr = " ".join(f"{v:.3f}" for v in avg.tolist())

        sel = task_topk_counts[task]
        sel_total = sel.sum().clamp_min(1.0)
        sel_str = " ".join(
            f"{FROZEN_EXPERT_NAMES[e]}={sel[e] / sel_total * 100:.1f}%"
            for e in range(len(FROZEN_EXPERT_NAMES))
        )
        log(f"  {task:<25}: [{gstr}]  (n={cnt:,}) | topk {sel_str}")


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
# Adaptive evaluator and unified entry point
# ---------------------------------------------------------------------------

_SAMPLE_LOG_RE = re.compile(r"^\[sample\s+(\d+)\]")
_WRAPPER_ABLATION_MODE = "moe"


def _make_sample_throttled_logger_builder(base_builder, sample_log_interval: int):
    interval = int(sample_log_interval)
    if interval <= 1:
        return base_builder

    def _builder(log_path: str):
        base_log = base_builder(log_path)
        emit_sample_details = False

        def _log(message: str = "") -> None:
            nonlocal emit_sample_details

            match = _SAMPLE_LOG_RE.match(message)
            if match is not None:
                sample_idx = int(match.group(1))
                emit_sample_details = (sample_idx % interval) == 0
                if emit_sample_details:
                    base_log(message)
                return

            if message.startswith("  mean_moe_weights:") or message.startswith("  series_moe_weights:") or message.startswith("  topk_usage:"):
                if emit_sample_details:
                    base_log(message)
                return

            emit_sample_details = False
            base_log(message)

        return _log

    return _builder


class _AdaptiveEvalMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_mode = _WRAPPER_ABLATION_MODE
        self.raw_signal_proj = None
        if self.ablation_mode == "raw":
            self.raw_signal_proj = nn.Linear(1, self.ts_hidden_dim)
        self.direct_signal_proj = None
        if self.ablation_mode == "signal_moe":
            self.direct_signal_proj = nn.Linear(self.ts_hidden_dim, self.llm_hidden_dim)
        self.debug_signal_shapes = os.environ.get("SIGNALLLM_DEBUG_SIGNAL_SHAPES") == "1"
        self._signal_shape_debug_printed = False

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
        fine_mask: torch.Tensor,
        coarse_mask: torch.Tensor,
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

    def _build_fused_inputs(
        self,
        ts: torch.Tensor,
        ts_len_mask: torch.Tensor,
        series_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        return_router_stats: bool = False,
    ):
        bsz, num_series, ts_len, _ = ts.shape
        device = ts.device

        ts_flat = ts.view(bsz * num_series, ts_len, 1)
        mask_flat = ts_len_mask.view(bsz * num_series, ts_len)
        batch_series = bsz * num_series

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
                    fine_mask,
                    coarse_mask,
                )
            elif self.ablation_mode == "signal_moe":
                gates = self._uniform_router_gates(batch_series, device=device, dtype=h_f.dtype)
                topk_indices = self._empty_topk_indices(batch_series, device=device)
                moe_out = self.moe(h_f, h_c, gates, coarse_mask)
                signal_tokens = self._build_signal_tokens(
                    moe_out,
                    gates,
                    fine_mask,
                    coarse_mask,
                    apply_pre_proj=False,
                )
            elif self.ablation_mode == "moe":
                q_vec_single = self._encode_question(q_ids, q_mask)
                q_vec = q_vec_single.unsqueeze(1).expand(-1, num_series, -1).reshape(batch_series, -1)
                gates, topk_indices, _ = self.router(
                    q_vec,
                    h_f,
                    h_c,
                    coarse_mask=coarse_mask,
                    use_topk=True,
                    fine_mask=fine_mask,
                )
                moe_out = self.moe(h_f, h_c, gates, coarse_mask)
                signal_tokens = self._build_signal_tokens(
                    moe_out,
                    gates,
                    fine_mask,
                    coarse_mask,
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
        sig_emb = all_sig.to(dtype=target_dtype)
        txt_emb = text_embeds.to(dtype=target_dtype)

        fused_embeds = torch.cat([sig_start, sig_emb, sig_end, txt_emb], dim=1)

        sig_token_mask = series_mask.unsqueeze(-1).expand(-1, -1, num_queries).reshape(
            bsz, num_series * num_queries
        )
        boundary_ones = torch.ones(bsz, 1, device=device, dtype=attention_mask.dtype)
        sig_attn = torch.cat(
            [boundary_ones, sig_token_mask.to(attention_mask.dtype), boundary_ones],
            dim=1,
        )
        fused_mask = torch.cat([sig_attn, attention_mask], dim=1)

        if return_router_stats:
            num_experts = gates.size(-1)
            top_k = topk_indices.size(-1)
            router_stats = {
                "gates": gates.view(bsz, num_series, num_experts).detach(),
                "topk_indices": topk_indices.view(bsz, num_series, top_k).detach(),
                "series_mask": series_mask.detach(),
            }
            return fused_embeds, fused_mask, router_stats

        return fused_embeds, fused_mask


class AdaptiveEvalGFTSignalLLM(_AdaptiveEvalMixin, LoRAEvalGFTSignalLLM):
    pass


class AdaptiveFrozenEvalGFTSignalLLM(_AdaptiveEvalMixin, FrozenEvalGFTSignalLLM):
    pass


def _parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frozen_llm",
        action="store_true",
        help="Use the frozen-LLM adaptive evaluator instead of the LoRA evaluator.",
    )
    parser.add_argument(
        "--ablation_mode",
        choices=ABLATION_MODES,
        default="moe",
        help="Signal path to feed into the Q-Former.",
    )
    parser.add_argument(
        "--sample_log_interval",
        type=int,
        default=1000,
        help="Log per-sample MoE details every N samples (default: 1000, use 1 to log all).",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    global _WRAPPER_ABLATION_MODE, build_frozen_logger
    wrapper_args, remaining_argv = _parse_wrapper_args(sys.argv[1:])
    _WRAPPER_ABLATION_MODE = wrapper_args.ablation_mode
    sys.argv = [sys.argv[0], *remaining_argv]
    print(f"[AdaptiveEval] ablation_mode={_WRAPPER_ABLATION_MODE}")

    build_frozen_logger = _make_sample_throttled_logger_builder(
        build_frozen_logger,
        wrapper_args.sample_log_interval,
    )

    if wrapper_args.frozen_llm:
        main_frozen()
        return

    main_lora()


if __name__ == "__main__":
    main()
