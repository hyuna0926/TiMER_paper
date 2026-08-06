from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================
@dataclass
class SignalLLMConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    ts_input_dim: int = 1
    ts_hidden_dim: int = 256
    router_hidden_dim: int = 256
    llm_hidden_dim: int = 2048
    num_qformer_queries: int = 8
    num_qformer_layers: int = 2
    num_experts: int = 3
    top_k: int = 2                          # [FIX 5] sparse routing용 top-k
    num_router_heads: int = 4
    num_ts_heads: int = 4
    dropout: float = 0.1
    freeze_llm: bool = True
    max_text_len: int = 256
    router_entropy_weight: float = 0.01     # [FIX 7] RouterRegularizer 통합
    router_balance_weight: float = 0.01
    gradient_checkpointing: bool = False


# ============================================================
# Utilities
# ============================================================
def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.float()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim).clamp_min(1e-6)
    return (x * mask).sum(dim=dim) / denom


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def masked_mean_pooling(
    hidden_states: torch.Tensor,               # [B, T, D]
    attention_mask: Optional[torch.Tensor],   # [B, T], 1=True(valid)
) -> torch.Tensor:
    """
    Mean-pool over valid tokens only.
    """
    return masked_mean(hidden_states, attention_mask, dim=1)


def last_token_pooling(
    hidden_states: torch.Tensor,               # [B, T, D]
    attention_mask: Optional[torch.Tensor],   # [B, T], 1=True(valid)
) -> torch.Tensor:
    """
    Causal LM용 대표 벡터:
      - attention_mask 기준 "마지막 유효 토큰" hidden state를 선택
      - right/left padding 모두 안전하게 동작
      - 전부 padding인 샘플은 mean pooling으로 안전 fallback
    """
    if attention_mask is None:
        return hidden_states[:, -1, :]

    bsz, tlen, _ = hidden_states.shape
    device = hidden_states.device

    valid = attention_mask.to(torch.bool)  # [B, T]
    positions = torch.arange(tlen, device=device).unsqueeze(0).expand(bsz, -1)  # [B, T]
    last_idx = torch.where(valid, positions, positions.new_zeros(1)).max(dim=1).values  # [B]

    batch_idx = torch.arange(bsz, device=device)
    pooled = hidden_states[batch_idx, last_idx]  # [B, D]

    # fully-masked row 보호: NaN/쓰레기 인덱스 대신 mean fallback
    has_any = valid.any(dim=1)  # [B]
    mean_fallback = masked_mean_pooling(hidden_states, attention_mask)  # [B, D]
    pooled = torch.where(has_any.unsqueeze(-1), pooled, mean_fallback)
    return pooled


def get_text_representative(
    hidden_states: torch.Tensor,               # [B, T, D]
    attention_mask: Optional[torch.Tensor],   # [B, T]
    strategy: Literal["last_token", "mean"] = "last_token",
) -> torch.Tensor:
    """
    텍스트 대표 벡터 추출 helper.
      - "last_token": causal LM에서 CLS 대체로 권장
      - "mean": baseline/ablation 용
    """
    if strategy == "last_token":
        return last_token_pooling(hidden_states, attention_mask)
    if strategy == "mean":
        return masked_mean_pooling(hidden_states, attention_mask)
    raise ValueError(f"Unknown text pooling strategy: {strategy}")


# ============================================================
# Positional Encoding
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ============================================================
# Multi-Granularity Encoder
# ============================================================
class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(residual + self.dropout(y))


class FineEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = SinusoidalPositionalEncoding(hidden_dim)
        self.block1 = TemporalConvBlock(hidden_dim, dropout)
        self.block2 = TemporalConvBlock(hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.block1(h)
        h = self.block2(h)
        return h


class CoarseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, pool_kernel: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pool_kernel = pool_kernel
        self.pos = SinusoidalPositionalEncoding(hidden_dim)
        self.block1 = TemporalConvBlock(hidden_dim, dropout)
        self.block2 = TemporalConvBlock(hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        ts_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [FIX 8] 패딩 영역을 마스킹 후 pooling
        h = self.proj(x)  # [B, T, D]

        if ts_mask is not None:
            # 패딩 위치를 0으로 만들어 pooling에서 제외
            h = h * ts_mask.float().unsqueeze(-1)

        h_pooled = F.avg_pool1d(
            h.transpose(1, 2),
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
            ceil_mode=True,
        ).transpose(1, 2)  # [B, Tc, D]

        # 마스크도 동일하게 pooling (pooling된 위치에 유효한 값이 하나라도 있으면 True)
        coarse_mask = None
        if ts_mask is not None:
            coarse_mask = F.max_pool1d(
                ts_mask.float().unsqueeze(1),
                kernel_size=self.pool_kernel,
                stride=self.pool_kernel,
                ceil_mode=True,
            ).squeeze(1).bool()  # [B, Tc]

        h_pooled = self.pos(h_pooled)
        h_pooled = self.block1(h_pooled)
        h_pooled = self.block2(h_pooled)
        return h_pooled, coarse_mask


class MultiGranularityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fine_encoder = FineEncoder(input_dim, hidden_dim, dropout)
        self.coarse_encoder = CoarseEncoder(input_dim, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        ts_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h_f = self.fine_encoder(x)
        h_c, coarse_mask = self.coarse_encoder(x, ts_mask)
        return h_f, h_c, coarse_mask


# ============================================================
# Question Encoder
# ============================================================
# [FIX 1] LLM backbone을 외부에서 주입받아 이중 로드 방지
class QuestionEncoder(nn.Module):
    def __init__(
        self,
        shared_llm_backbone: nn.Module,  # SignalLLM의 self.llm.model 공유
        llm_hidden_dim: int,
        target_dim: int,
        max_text_len: int = 256,
        pooling_strategy: Literal["last_token", "mean"] = "last_token",
    ):
        super().__init__()
        self.backbone = shared_llm_backbone  # 가중치 공유, 별도 파라미터 없음
        self.proj = nn.Linear(llm_hidden_dim, target_dim)
        self.max_text_len = max_text_len
        self.pooling_strategy = pooling_strategy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        text_repr = get_text_representative(
            last_hidden,
            attention_mask,
            strategy=self.pooling_strategy,
        )  # [B, llm_hidden_dim]
        pooled = self.proj(text_repr)          # [B, target_dim]
        return {
            "token_embeddings": last_hidden,
            "pooled": pooled,
            "text_repr": text_repr,
        }


# ============================================================
# Router Utility: Masked Top-K Gating
# ============================================================

# Expert 이름 매핑 (로깅용)
EXPERT_NAMES = {0: "local", 1: "global", 2: "cross"}


def compute_topk_masked_gates(
    logits: torch.Tensor,   # [B, E]  라우터 출력 logits
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Softmax gate 계산 후 top-k만 남기고 renormalize.

    기존 re-softmax 방식과의 차이:
      - 기존: top-k logits에 다시 softmax → 원본 확률 분포 왜곡
      - 신규: softmax 먼저 → top-k 마스킹 → sum=1 재정규화 → 원본 분포 유지

    Returns:
        original_gates: [B, E]  softmax(logits) — 원본 전체 분포
        masked_gates:   [B, E]  top-k만 살리고 합=1로 재정규화, 나머지 0
        topk_indices:   [B, K]  선택된 expert index
        topk_values:    [B, K]  masked_gates 기준 top-k 위치의 값
    """
    B, E = logits.shape
    k = min(top_k, E)                                            # expert 수 초과 방지

    # 1) softmax → 확률 분포
    original_gates = torch.softmax(logits, dim=-1)               # [B, E]

    # 2) original_gates 기준으로 top-k index 선택
    _, topk_indices = torch.topk(original_gates, k=k, dim=-1)    # [B, K]

    # 3) top-k 위치의 확률값 수집
    topk_probs = original_gates.gather(1, topk_indices)          # [B, K]

    # 4) 마스크 텐서 생성 후 scatter: top-k 외 나머지는 0
    masked = torch.zeros_like(original_gates)                    # [B, E]
    masked.scatter_(1, topk_indices, topk_probs)                 # [B, E]

    # 5) renormalize: 남은 값의 합이 1이 되도록 (zero division 방지)
    gate_sum     = masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, 1]
    masked_gates = masked / gate_sum                                  # [B, E]

    # 6) masked_gates 기준 topk 값 (renormalize 이후)
    topk_values = masked_gates.gather(1, topk_indices)           # [B, K]

    return original_gates, masked_gates, topk_indices, topk_values


def compute_router_reg_loss(
    original_gates:         torch.Tensor,   # [B, E]  softmax(logits)
    masked_gates:           torch.Tensor,   # [B, E]  renormalized top-k
    balance_weight:         float = 0.001,
    entropy_weight:         float = 0.0,
    use_masked_for_entropy: bool  = False,
    sharpen_entropy:        bool  = False,  # True: routing을 sharp하게 (H 최소화)
                                            # False: routing을 다양하게 (H 최대화)
) -> torch.Tensor:
    """
    라우터 정규화 손실: load balancing + optional entropy.

    [Load balancing]
      배치 평균 masked gate와 uniform([1/E,...,1/E]) 간의 MSE.
      특정 expert 과집중을 약하게 억제. balance_weight=0 이면 비활성.
      기본값 0.001: collapse 방지만 하고 routing 자유도는 유지.

    [Entropy regularization]
      H = -Σ p·log(p)  (클수록 균등, 작을수록 sharp)
      sharpen_entropy=False (기본): loss - weight×H  → H 최대화 → 균등 유도
      sharpen_entropy=True:         loss + weight×H  → H 최소화 → sharp routing 유도
      entropy_weight=0 이면 비활성(기본값).
      use_masked_for_entropy=True 면 masked_gates, False 면 original_gates 기준.
    """
    loss = original_gates.new_tensor(0.0)

    # --- Load balancing ---
    if balance_weight > 0.0:
        mean_gate = masked_gates.mean(dim=0)                       # [E]
        E         = mean_gate.numel()
        uniform   = torch.full_like(mean_gate, 1.0 / E)
        loss      = loss + balance_weight * F.mse_loss(mean_gate, uniform)

    # --- Entropy regularization ---
    if entropy_weight > 0.0:
        g    = masked_gates if use_masked_for_entropy else original_gates
        H    = -(g * torch.log(g.clamp_min(1e-8))).sum(-1).mean()  # scalar > 0
        sign = 1.0 if sharpen_entropy else -1.0   # +H: sharp,  -H: diverse
        loss = loss + sign * entropy_weight * H

    return loss


def format_expert_selection(topk_indices_or_counts: torch.Tensor) -> str:
    """
    expert 선택 빈도를 문자열로 반환.

    입력 두 가지 모두 허용:
      - topk_indices: [B, K]  — 배치의 expert index (raw selection)
      - counts:       [E]     — 누적 선택 횟수 벡터 (running sum)

    예: "local=55% global=0% cross=45%"
    """
    E = len(EXPERT_NAMES)
    t = topk_indices_or_counts
    if t.dim() == 1 and t.numel() == E:
        # [E] 누적 count 벡터
        counts = t.float()
    else:
        # [B, K] index 텐서 → count로 변환
        counts = torch.zeros(E, dtype=torch.float)
        for e in range(E):
            counts[e] = (t == e).float().sum().item()
    total = counts.sum().clamp_min(1.0)
    return " ".join(f"{EXPERT_NAMES[e]}={counts[e]/total*100:.0f}%" for e in range(E))


# ============================================================
# Question-Conditioned Sparse Router
# ============================================================
class QuestionAwareSignalSummary(nn.Module):
    """
    Lightweight question-conditioned summary.

    Inputs:
      - text_repr: [B, D]
      - signal_tokens: [B, T, D]
      - key_padding_mask: [B, T], True=valid
    Output:
      - ctx: [B, D]
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim({hidden_dim}) must be divisible by num_heads({num_heads}).")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        text_repr: torch.Tensor,                     # [B, D]
        signal_tokens: torch.Tensor,                # [B, T, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T], True=valid
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tlen, _ = signal_tokens.shape
        h = self.num_heads
        d = self.head_dim

        # [B, D] -> [B, H, 1, Dh]
        q = self.q_proj(text_repr).view(bsz, h, d).unsqueeze(2)
        # [B, T, D] -> [B, H, T, Dh]
        k = self.k_proj(signal_tokens).view(bsz, tlen, h, d).permute(0, 2, 1, 3)
        v = self.v_proj(signal_tokens).view(bsz, tlen, h, d).permute(0, 2, 1, 3)

        # [B, H, 1, T]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        valid = None
        if key_padding_mask is not None:
            valid = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn_logits = attn_logits.masked_fill(~valid, -1e4)

        # softmax 후 마스크 재정규화: fully-masked row도 안정적으로 0 벡터 처리
        attn_probs = torch.softmax(attn_logits, dim=-1)  # [B, H, 1, T]
        if valid is not None:
            attn_probs = attn_probs * valid.to(attn_probs.dtype)
            attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_probs = self.attn_dropout(attn_probs)

        # [B, H, 1, Dh] -> [B, D]
        ctx = torch.matmul(attn_probs, v).squeeze(2).reshape(bsz, self.hidden_dim)
        ctx = self.out_proj(ctx)

        if return_attn:
            return ctx, attn_probs.squeeze(2)  # [B, H, T]
        return ctx, None


class QuestionConditionedRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        router_hidden_dim: int,
        num_experts: int = 3,
        top_k: int = 2,
        num_router_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.summary = QuestionAwareSignalSummary(
            hidden_dim=hidden_dim,
            num_heads=num_router_heads,
            attn_dropout=dropout,
        )
        self.router_in_norm = nn.LayerNorm(hidden_dim * 3)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 3, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, num_experts),
        )

    def forward(
        self,
        q:           torch.Tensor,                       # [B, D]  질문 벡터
        h_f:         torch.Tensor,                       # [B, Tf, D]  fine tokens
        h_c:         torch.Tensor,                       # [B, Tc, D]  coarse tokens
        coarse_mask: Optional[torch.Tensor] = None,      # [B, Tc]  bool
        use_topk:    bool = True,                        # False=soft routing (warm-up용)
        fine_mask:   Optional[torch.Tensor] = None,      # [B, Tf]  bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Router input pipeline:
          text_repr q [B, D]
            -> fine_ctx   = Attn(q, h_f, h_f) [B, D]
            -> coarse_ctx = Attn(q, h_c, h_c) [B, D]
            -> concat([q; fine_ctx; coarse_ctx]) [B, 3D]
            -> MLP logits [B, E]

        Returns:
            gates:        [B, E]  use_topk=True  → masked_gates (sparse)
                                  use_topk=False → original_gates (soft, warm-up)
            topk_indices: [B, K]  선택된 expert index
            aux:          dict    원본/마스킹 gates, logits, 요약 벡터
        """
        # text-guided fine/coarse summary
        fine_ctx, fine_attn = self.summary(
            text_repr=q,
            signal_tokens=h_f,
            key_padding_mask=fine_mask,
            return_attn=True,
        )  # fine_ctx: [B, D], fine_attn: [B, H, Tf]
        coarse_ctx, coarse_attn = self.summary(
            text_repr=q,
            signal_tokens=h_c,
            key_padding_mask=coarse_mask,
            return_attn=True,
        )  # coarse_ctx: [B, D], coarse_attn: [B, H, Tc]

        # router 입력: [text_repr | fine_ctx | coarse_ctx]
        router_in = torch.cat([q, fine_ctx, coarse_ctx], dim=-1)   # [B, 3D]
        router_in = self.router_in_norm(router_in)
        logits = self.router(router_in)                             # [B, E]

        # masked top-k gates 계산
        original_gates, masked_gates, topk_indices, topk_values = \
            compute_topk_masked_gates(logits, self.top_k)

        # warm-up 분기: 초반에는 soft routing, 이후 sparse top-k
        gates = masked_gates if use_topk else original_gates

        aux = {
            "router_logits":  logits,
            "original_gates": original_gates,   # [B, E]  reg loss / 로깅용
            "masked_gates":   masked_gates,     # [B, E]  실제 sparse 분포
            "topk_indices":   topk_indices,     # [B, K]
            "topk_values":    topk_values,      # [B, K]
            "router_input":   router_in,        # [B, 3D]
            "text_repr":      q,                # [B, D]
            "fine_ctx":       fine_ctx,         # [B, D]
            "coarse_ctx":     coarse_ctx,       # [B, D]
            "fine_attn":      fine_attn,        # [B, H, Tf]
            "coarse_attn":    coarse_attn,      # [B, H, Tc]
            # 하위 호환 키
            "fine_summary":   fine_ctx,
            "coarse_summary": coarse_ctx,
            # 하위 호환: full_gates 키 유지
            "full_gates":     original_gates,
        }
        return gates, topk_indices, aux


# ============================================================
# Experts
# ============================================================
class LocalDetailExpert(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h_f: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(h_f, h_f, h_f, need_weights=False)
        h = self.norm1(h_f + y)
        z = self.ffn(h)
        return self.norm2(h + z)  # [B, Tf, D]


class GlobalTrendExpert(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h_c: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(h_c, h_c, h_c, need_weights=False)
        h = self.norm1(h_c + y)
        z = self.ffn(h)
        return self.norm2(h + z)  # [B, Tc, D]


class CrossScaleInteractionExpert(nn.Module):
    # [FIX 6] 단일 토큰이 아닌 fine 길이 기준 토큰 시퀀스 반환
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fine_to_coarse = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.coarse_to_fine = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_f: torch.Tensor, h_c: torch.Tensor) -> torch.Tensor:
        # fine이 coarse를 참조: global context 주입
        fine_enriched, _ = self.fine_to_coarse(h_f, h_c, h_c, need_weights=False)
        # coarse를 fine 길이로 업샘플해서 local evidence 주입
        coarse_enriched, _ = self.coarse_to_fine(h_f, h_c, h_c, need_weights=False)

        # gated fusion: 두 방향 정보를 동적으로 혼합
        gate = self.gate_proj(torch.cat([fine_enriched, coarse_enriched], dim=-1))
        fused = gate * fine_enriched + (1 - gate) * coarse_enriched  # [B, Tf, D]
        fused = self.norm(self.out_proj(fused))
        return fused  # [B, Tf, D] — fine 시퀀스 길이 유지


# ============================================================
# Sparse MoE Aggregation
# ============================================================
class GranularityMoE(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.local_expert = LocalDetailExpert(hidden_dim, num_heads, dropout)
        self.global_expert = GlobalTrendExpert(hidden_dim, num_heads, dropout)
        self.cross_expert = CrossScaleInteractionExpert(hidden_dim, num_heads, dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_f:         torch.Tensor,                       # [B, Tf, D]
        h_c:         torch.Tensor,                       # [B, Tc, D]
        gates:       torch.Tensor,                       # [B, E]  masked_gates 또는 original_gates
        coarse_mask: Optional[torch.Tensor] = None,      # [B, Tc]  bool
    ) -> Dict[str, torch.Tensor]:
        """
        모든 expert를 항상 forward하되, aggregation에만 gates 적용.

        Expert:
          0: LocalDetailExpert  — fine 중심 (단기·국소 패턴)
          1: GlobalTrendExpert  — coarse 중심 (장기·전역 추세)
          2: CrossScaleExpert   — fine+coarse 상호작용
        """
        # 1) 3개 expert 모두 forward (계산 안정성 유지)
        e_local  = self.local_expert(h_f)           # [B, Tf, D]
        e_global = self.global_expert(h_c)          # [B, Tc, D]
        e_cross  = self.cross_expert(h_f, h_c)     # [B, Tf, D]

        # 2) 각 expert 대표 벡터 (time dim mean-pool)
        z_local  = e_local.mean(dim=1)                           # [B, D]
        z_global = masked_mean(e_global, coarse_mask, dim=1)    # [B, D]
        z_cross  = e_cross.mean(dim=1)                          # [B, D]

        # 3) [B, 3, D] 로 stack
        expert_states = torch.stack([z_local, z_global, z_cross], dim=1)  # [B, 3, D]

        # 4) masked top-k gates로 weighted sum
        #    gates: [B, 3] → unsqueeze: [B, 3, 1] → broadcast → sum over dim=1
        routed = torch.sum(expert_states * gates.unsqueeze(-1), dim=1)  # [B, D]
        routed = self.norm(self.out_proj(routed))

        return {
            "local_tokens": e_local,    # [B, Tf, D]
            "global_tokens": e_global,  # [B, Tc, D]
            "cross_tokens":  e_cross,         # [B, Tf, D]
            "expert_states": expert_states,   # [B, 3, D]  (이전: stacked_expert_states)
            "routed_state":  routed,          # [B, D]
        }


# ============================================================
# Q-Former Connector
# ============================================================
class QFormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        y, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + y)
        y, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.norm2(q + y)
        y = self.ffn(q)
        q = self.norm3(q + y)
        return q


class QFormerConnector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        self.layers = nn.ModuleList([
            QFormerBlock(input_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, signal_tokens: torch.Tensor) -> torch.Tensor:
        bsz = signal_tokens.size(0)
        q = self.query_tokens.expand(bsz, -1, -1)
        for layer in self.layers:
            q = layer(q, signal_tokens)
        return self.out_proj(q)


# ============================================================
# Signal LLM
# ============================================================
class SignalLLMGranularityRouting(nn.Module):
    def __init__(self, cfg: SignalLLMConfig):
        super().__init__()
        self.cfg = cfg

        # [FIX 1] LLM 먼저 로드
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name)
        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        if cfg.gradient_checkpointing and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()

        # [FIX 1] LLM backbone(self.llm.model) 공유 — 이중 로드 완전 제거
        llm_hidden_dim = self.llm.config.hidden_size
        self.question_encoder = QuestionEncoder(
            shared_llm_backbone=self.llm.model,
            llm_hidden_dim=llm_hidden_dim,
            target_dim=cfg.ts_hidden_dim,
            max_text_len=cfg.max_text_len,
            pooling_strategy="last_token",  # causal LM 기본 권장
        )

        self.signal_encoder = MultiGranularityEncoder(
            input_dim=cfg.ts_input_dim,
            hidden_dim=cfg.ts_hidden_dim,
            dropout=cfg.dropout,
        )
        self.router = QuestionConditionedRouter(
            hidden_dim=cfg.ts_hidden_dim,
            router_hidden_dim=cfg.router_hidden_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            num_router_heads=cfg.num_router_heads,
            dropout=cfg.dropout,
        )
        self.moe = GranularityMoE(
            hidden_dim=cfg.ts_hidden_dim,
            num_heads=cfg.num_ts_heads,
            dropout=cfg.dropout,
        )
        self.pre_qformer_proj = nn.Linear(cfg.ts_hidden_dim, cfg.ts_hidden_dim)
        self.qformer = QFormerConnector(
            input_dim=cfg.ts_hidden_dim,
            output_dim=cfg.llm_hidden_dim,
            num_queries=cfg.num_qformer_queries,
            num_layers=cfg.num_qformer_layers,
            num_heads=cfg.num_ts_heads,
            dropout=cfg.dropout,
        )

        self.signal_start = nn.Parameter(torch.randn(1, 1, cfg.llm_hidden_dim) * 0.02)
        self.signal_end = nn.Parameter(torch.randn(1, 1, cfg.llm_hidden_dim) * 0.02)

    # --------------------------------------------------------
    # [FIX 3] expert 출력 토큰들로 signal_tokens 구성
    # --------------------------------------------------------
    def build_signal_tokens(
        self,
        moe_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        routed = moe_out["routed_state"].unsqueeze(1)       # [B, 1, D]
        local_summary = moe_out["local_tokens"].mean(dim=1, keepdim=True)   # [B, 1, D]
        global_summary = moe_out["global_tokens"].mean(dim=1, keepdim=True) # [B, 1, D]
        cross_summary = moe_out["cross_tokens"].mean(dim=1, keepdim=True)   # [B, 1, D]

        # expert 출력 + routing 결과 모두 Q-Former에 노출
        signal_tokens = torch.cat([
            local_summary,
            global_summary,
            cross_summary,
            routed,
        ], dim=1)  # [B, 4, D]
        return self.pre_qformer_proj(signal_tokens)

    # --------------------------------------------------------
    def fuse_embeddings(
        self,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        signal_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = text_embeds.size(0)
        signal_start = self.signal_start.expand(bsz, -1, -1)
        signal_end = self.signal_end.expand(bsz, -1, -1)

        fused_embeds = torch.cat([signal_start, signal_embeds, signal_end, text_embeds], dim=1)
        signal_mask = torch.ones(
            bsz,
            1 + signal_embeds.size(1) + 1,
            device=text_mask.device,
            dtype=text_mask.dtype,
        )
        fused_mask = torch.cat([signal_mask, text_mask], dim=1)
        return fused_embeds, fused_mask

    # --------------------------------------------------------
    # [FIX 7] Router regularization loss를 내부에서 계산
    # --------------------------------------------------------
    def _router_reg_loss(self, full_gates: torch.Tensor) -> torch.Tensor:
        loss = full_gates.new_tensor(0.0)
        cfg = self.cfg

        if cfg.router_entropy_weight > 0:
            entropy = -(full_gates * torch.log(full_gates.clamp_min(1e-8))).sum(dim=-1).mean()
            loss = loss + cfg.router_entropy_weight * entropy

        if cfg.router_balance_weight > 0:
            mean_gate = full_gates.mean(dim=0)
            uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
            loss = loss + cfg.router_balance_weight * F.mse_loss(mean_gate, uniform)

        return loss

    # --------------------------------------------------------
    def _tokenize_questions(
        self, questions: List[str], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        toks = self.llm_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
        ).to(device)
        return toks

    def _build_supervised_text_batch(
        self,
        questions: List[str],
        answers: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        pad_token_id = self.llm_tokenizer.pad_token_id
        eos_ids = [self.llm_tokenizer.eos_token_id] if self.llm_tokenizer.eos_token_id is not None else []

        input_id_rows: List[List[int]] = []
        label_rows: List[List[int]] = []

        for question, answer in zip(questions, answers):
            prefix_text = f"Question: {question}\nAnswer:"
            prefix_ids = self.llm_tokenizer(
                prefix_text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors=None,
            )["input_ids"]

            answer_text = str(answer)
            answer_ids = self.llm_tokenizer(
                answer_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            if not answer_ids:
                answer_ids = self.llm_tokenizer(
                    answer_text.strip(),
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )["input_ids"]

            keep_prefix = max(0, self.cfg.max_text_len - len(answer_ids) - len(eos_ids))
            if len(prefix_ids) > keep_prefix:
                prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []

            input_ids = prefix_ids + answer_ids + eos_ids
            labels = ([-100] * len(prefix_ids)) + answer_ids + ([-100] * len(eos_ids))

            if len(input_ids) > self.cfg.max_text_len:
                input_ids = input_ids[-self.cfg.max_text_len:]
                labels = labels[-self.cfg.max_text_len:]

            input_id_rows.append(input_ids)
            label_rows.append(labels)

        max_text = max(len(ids) for ids in input_id_rows)
        padded_ids: List[List[int]] = []
        padded_labels: List[List[int]] = []
        padded_mask: List[List[int]] = []

        for input_ids, labels in zip(input_id_rows, label_rows):
            pad = max_text - len(input_ids)
            padded_ids.append(input_ids + [pad_token_id] * pad)
            padded_labels.append(labels + [-100] * pad)
            padded_mask.append([1] * len(input_ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long, device=device),
            "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
        }

    # --------------------------------------------------------
    def forward(
        self,
        input_ts: torch.Tensor,
        questions: List[str],
        answers: Optional[List[str]] = None,
        ts_mask: Optional[torch.Tensor] = None,  # [FIX 8] 가변 길이 지원
    ) -> Dict[str, torch.Tensor]:
        device = input_ts.device

        # 1) 시계열 → fine / coarse 표현
        h_f, h_c, coarse_mask = self.signal_encoder(input_ts, ts_mask)

        # 2) 질문 인코딩 (LLM backbone 공유)
        q_toks = self._tokenize_questions(questions, device)
        q_out = self.question_encoder(q_toks["input_ids"], q_toks["attention_mask"])
        q_vec = q_out["pooled"]

        # 3) Sparse routing
        sparse_gates, topk_idx, router_aux = self.router(q_vec, h_f, h_c, coarse_mask)

        # 4) Granularity MoE
        moe_out = self.moe(h_f, h_c, sparse_gates, coarse_mask)

        # 5) Signal tokens → Q-Former → LLM 차원 정렬
        signal_tokens = self.build_signal_tokens(moe_out)   # [FIX 3]
        signal_embeds = self.qformer(signal_tokens)

        # 6) Prompt 구성 및 텍스트 임베딩
        if answers is None:
            prompts = [f"Question: {q}\nAnswer:" for q in questions]
            text_toks = self.llm_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_text_len,
            ).to(device)
        else:
            text_toks = self._build_supervised_text_batch(questions, answers, device)

        text_input_ids = text_toks["input_ids"]
        text_attention_mask = text_toks["attention_mask"]
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        fused_embeds, fused_mask = self.fuse_embeddings(
            text_embeds, text_attention_mask, signal_embeds
        )

        # 7) prefix 길이 (signal 구간)
        prefix_len = 1 + signal_embeds.size(1) + 1  # start + queries + end

        result = {
            "gates": sparse_gates,
            "topk_idx": topk_idx,
            "signal_embeds": signal_embeds,
            "fine_states": h_f,
            "coarse_states": h_c,
            "routed_state": moe_out["routed_state"],
            "router_logits": router_aux["router_logits"],
        }

        if answers is not None:
            # full sequence 길이에 맞게 -100 패딩
            full_labels = torch.full(
                (text_toks["labels"].size(0), prefix_len + text_toks["labels"].size(1)),
                fill_value=-100,
                dtype=text_toks["labels"].dtype,
                device=device,
            )
            # answer-only supervision: prefix/question part is masked out.
            full_labels[:, prefix_len:] = text_toks["labels"]

            # [FIX 2] LLM forward 한 번만 호출 (loss 포함)
            train_outputs = self.llm(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=full_labels,
                return_dict=True,
            )
            result["lm_loss"] = train_outputs.loss
            result["logits"] = train_outputs.logits

            # [FIX 7] router regularization loss를 forward에서 함께 반환
            result["router_reg_loss"] = self._router_reg_loss(router_aux["full_gates"])
            result["loss"] = train_outputs.loss + result["router_reg_loss"]
        else:
            # inference 시에는 logits만
            outputs = self.llm(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                return_dict=True,
            )
            result["logits"] = outputs.logits

        return result

    # --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ts: torch.Tensor,
        questions: List[str],
        ts_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, object]:
        device = input_ts.device

        h_f, h_c, coarse_mask = self.signal_encoder(input_ts, ts_mask)
        q_toks = self._tokenize_questions(questions, device)
        q_out = self.question_encoder(q_toks["input_ids"], q_toks["attention_mask"])
        sparse_gates, topk_idx, _ = self.router(q_out["pooled"], h_f, h_c, coarse_mask)
        moe_out = self.moe(h_f, h_c, sparse_gates, coarse_mask)
        signal_tokens = self.build_signal_tokens(moe_out)
        signal_embeds = self.qformer(signal_tokens)

        prompts = [f"Question: {q}\nAnswer:" for q in questions]
        text_toks = self.llm_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
        ).to(device)

        text_embeds = self.llm.get_input_embeddings()(text_toks.input_ids)
        fused_embeds, fused_mask = self.fuse_embeddings(
            text_embeds, text_toks.attention_mask, signal_embeds
        )

        gen_ids = self.llm.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )

        texts = self.llm_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return {
            "texts": texts,
            "gates": sparse_gates,
            "topk_idx": topk_idx,
        }
