from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


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


__all__ = [
    "ABLATION_MODES",
    "CROSS_SUMMARY_TOKENS",
    "GLOBAL_SUMMARY_TOKENS",
    "LOCAL_SUMMARY_TOKENS",
    "QFORMER_INPUT_TOKENS",
    "build_adaptive_qformer_input",
    "build_single_source_qformer_input",
    "temporal_adaptive_pool",
]
