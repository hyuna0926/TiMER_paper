from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def build_padding_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    """Return bool mask with True at valid positions.

    Args:
        lengths: [B], each entry is valid length.
        max_len: optional maximum length. If None, use lengths.max().

    Returns:
        mask: [B, max_len] bool
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def rotate_half(x: Tensor) -> Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


def apply_rope(x: Tensor, base: float = 10000.0) -> Tensor:
    """RoPE on token dimension.

    Args:
        x: [B, N, D]

    Returns:
        [B, N, D]
    """
    _, n_tokens, d_model = x.shape
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for RoPE, got {d_model}")

    device = x.device
    half = d_model // 2

    pos = torch.arange(n_tokens, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    freqs = torch.outer(pos, inv_freq)  # [N, D/2]

    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # [N, D]
    cos = emb.cos().to(dtype=x.dtype).unsqueeze(0)
    sin = emb.sin().to(dtype=x.dtype).unsqueeze(0)

    return x * cos + rotate_half(x) * sin


def fft_period_from_series(x: Tensor, sampling: str = "median", k: int = 10) -> int:
    """Estimate period from one sample.

    Args:
        x: [L, C]
    Returns:
        period: int >= 1
    """
    length = int(x.size(0))
    if length < 4:
        return 1

    xf = torch.fft.rfft(x, dim=0)  # [L_rfft, C]
    amp = xf.abs().mean(dim=-1)  # [L_rfft]
    if amp.numel() <= 1:
        return 1
    amp[0] = 0

    top_k = min(k, int(amp.numel() - 1))
    if top_k <= 0:
        return 1

    _, top_idx = torch.topk(amp, k=top_k)

    if sampling == "median":
        freq = int(torch.median(top_idx).item())
    elif sampling == "max":
        freq = int(torch.max(top_idx).item())
    elif sampling == "min":
        freq = int(torch.min(top_idx).item())
    else:
        raise ValueError(f"Unsupported sampling={sampling}")

    freq = max(freq, 1)
    period = max(length // freq, 1)
    return period


class moving_avg(nn.Module):
    """GFT-style moving average."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class MaskedSeriesDecomposition(nn.Module):
    """GFT-style decomposition + validity mask control."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, c_in = x.shape
        trend = x.new_zeros(bsz, seq_len, c_in)

        # Keep padded tail values out of the moving-average window so the
        # valid prefix matches original GFT decomposition semantics.
        for b in range(bsz):
            valid_len = int(mask[b].sum().item())
            if valid_len <= 0:
                continue
            trend[b, :valid_len, :] = self.moving_avg(x[b : b + 1, :valid_len, :])[0]

        fine = x - trend
        m = mask.unsqueeze(-1).float()
        return fine * m, trend * m


class FineGrainedBlock(nn.Module):
    """Same structure as original GFT fine-grained block."""

    def __init__(
        self,
        enc_in: int,
        d_model: int,
        kernel_num: int,
        scale_factor: int = 3,
        input_kernel_size: int = 3,
        mid_kernel_size: int = 3,
        output_kernel_size: int = 3,
        input_padding: Optional[int] = None,
        mid_padding: Optional[int] = None,
        output_padding: Optional[int] = None,
    ):
        super().__init__()

        input_padding = input_kernel_size // 2 if input_padding is None else input_padding
        mid_padding = mid_kernel_size // 2 if mid_padding is None else mid_padding
        output_padding = output_kernel_size // 2 if output_padding is None else output_padding

        self.input_layer = nn.Sequential(
            nn.Conv2d(enc_in, d_model, kernel_size=input_kernel_size, padding=input_padding),
            nn.PReLU(d_model),
        )

        mid_layers = [
            nn.Conv2d(d_model, d_model // 2, kernel_size=mid_kernel_size, padding=mid_padding),
            nn.PReLU(d_model // 2),
        ]
        for _ in range(kernel_num):
            mid_layers.extend(
                [
                    nn.Conv2d(d_model // 2, d_model // 2, kernel_size=mid_kernel_size, padding=mid_padding),
                    nn.PReLU(d_model // 2),
                ]
            )
        mid_layers.extend(
            [
                nn.Conv2d(d_model // 2, d_model, kernel_size=mid_kernel_size, padding=mid_padding),
                nn.PReLU(d_model),
            ]
        )
        self.mid_layers = nn.Sequential(*mid_layers)

        self.output_layer = nn.ConvTranspose2d(
            d_model,
            enc_in,
            kernel_size=output_kernel_size,
            stride=(1, scale_factor),
            padding=output_padding,
            output_padding=(0, scale_factor - 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x


class MaskedPatchEmbedding(nn.Module):
    """Patch embedding with validity mask, close to GFT PatchEmbedding behavior."""

    def __init__(
        self,
        c_in: int,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float = 0.1,
        min_valid_ratio: float = 0.5,
    ):
        super().__init__()
        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.min_valid_ratio = min_valid_ratio

        self.value_embedding = nn.Linear(c_in * patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _pad_len(self, seq_len: int) -> int:
        if seq_len < self.patch_len:
            return self.patch_len - seq_len
        rem = (seq_len - self.patch_len) % self.stride
        return 0 if rem == 0 else self.stride - rem

    def _sinusoidal_pos(self, n_tokens: int, d_model: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = torch.arange(n_tokens, device=device, dtype=torch.float32).unsqueeze(1)
        half = d_model // 2
        div = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
        pe = torch.zeros(1, n_tokens, d_model, device=device, dtype=torch.float32)
        pe[:, :, 0::2] = torch.sin(pos * div)
        pe[:, :, 1::2] = torch.cos(pos * div)
        return pe.to(dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Args:
            x: [B, L, C]
            mask: [B, L] bool

        Returns:
            tokens: [B, N, D]
            token_mask: [B, N] bool
        """
        bsz, seq_len, c_in = x.shape
        if c_in != self.c_in:
            raise ValueError(f"Expected c_in={self.c_in}, got {c_in}")

        x_t = x.transpose(1, 2)  # [B, C, L]
        m_t = mask.float().unsqueeze(1)  # [B, 1, L]

        pad_len = self._pad_len(seq_len)
        x_pad = F.pad(x_t, (0, pad_len), mode="replicate")
        m_pad = F.pad(m_t, (0, pad_len), value=0.0)

        patches = x_pad.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, P]
        patch_mask = m_pad.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, 1, N, P]

        n_tokens = int(patches.size(2))
        flat = patches.permute(0, 2, 1, 3).contiguous().view(bsz, n_tokens, c_in * self.patch_len)
        token_mask = patch_mask.squeeze(1).mean(dim=-1) >= self.min_valid_ratio

        has_any = mask.any(dim=1)
        if n_tokens > 0:
            token_mask[has_any, 0] = True

        tokens = self.value_embedding(flat)
        tokens = tokens + self._sinusoidal_pos(n_tokens, tokens.size(-1), tokens.device, tokens.dtype)
        tokens = self.dropout(tokens)
        tokens = tokens * token_mask.unsqueeze(-1).float()
        return tokens, token_mask


class GFTCrossFusionLayer(nn.Module):
    """Patch-wise cross attention fusion (coarse query, fine key/value)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, coarse: Tensor, fine: Tensor, coarse_mask: Tensor, fine_mask: Tensor) -> Tensor:
        attn_out, _ = self.cross_attn(
            query=coarse,
            key=fine,
            value=fine,
            key_padding_mask=~fine_mask,
            need_weights=False,
        )
        mixed = self.norm1(coarse + self.dropout(attn_out))
        mixed = self.norm2(mixed + self.dropout(self.ffn(mixed)))

        gate = torch.sigmoid(self.gate(torch.cat([coarse, mixed], dim=-1)))
        fused = gate * mixed + (1.0 - gate) * coarse
        fused = fused * coarse_mask.unsqueeze(-1).float()
        return fused


@dataclass
class GFTVarLenConfig:
    c_in: int = 1
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1

    decomp_kernel_size: int = 5

    fine_patch_len: int = 4
    fine_stride: int = 2
    coarse_patch_len: int = 12
    coarse_stride: int = 6
    min_valid_ratio: float = 0.5

    up_scale: int = 3
    num_fine_grained_conv: int = 2
    sampling: str = "median"
    fft_top_k: int = 10

    fine_layers: int = 2
    coarse_layers: int = 2
    fusion_layers: int = 2

    decoder_layers: int = 2
    max_horizon: int = 80

    rope_base_fine: float = 10000.0
    rope_base_coarse: float = 10000.0

    short_len_threshold: int = 8
    use_rope: bool = True
    use_revin_affine: bool = True


class GFTVarLenForecaster(nn.Module):
    """Variable-length GFT-based forecaster.

    GFT-anchored flow:
    1) RevIN-like normalization
    2) series decomposition -> fine/coarse
    3) fine periodic reshape + FineGrainedBlock (FFT period)
    4) coarse up-sampling
    5) fine/coarse patch embedding
    6) coarse<-fine cross fusion
    7) horizon-query decoder
    """

    def __init__(self, config: GFTVarLenConfig):
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError("d_model must be even when RoPE is enabled.")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.config = config
        self.eps = 1e-6

        if config.use_revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, config.c_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, config.c_in))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        self.decomposition = MaskedSeriesDecomposition(kernel_size=config.decomp_kernel_size)
        self.fine_grained_block = FineGrainedBlock(
            enc_in=config.c_in,
            d_model=config.d_model,
            kernel_num=config.num_fine_grained_conv,
            scale_factor=config.up_scale,
        )

        self.fine_embed = MaskedPatchEmbedding(
            c_in=config.c_in,
            patch_len=config.fine_patch_len,
            stride=config.fine_stride,
            d_model=config.d_model,
            dropout=config.dropout,
            min_valid_ratio=config.min_valid_ratio,
        )
        self.coarse_embed = MaskedPatchEmbedding(
            c_in=config.c_in,
            patch_len=config.coarse_patch_len,
            stride=config.coarse_stride,
            d_model=config.d_model,
            dropout=config.dropout,
            min_valid_ratio=config.min_valid_ratio,
        )

        self.fine_encoder = self._build_encoder(config.fine_layers)
        self.coarse_encoder = self._build_encoder(config.coarse_layers)

        self.fusion_blocks = nn.ModuleList(
            [
                GFTCrossFusionLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.fusion_layers)
            ]
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=config.decoder_layers)

        self.horizon_queries = nn.Parameter(torch.randn(config.max_horizon, config.d_model) * 0.02)
        self.out_proj = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.c_in))

    def _build_encoder(self, n_layers: int) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=n_layers)

    def _masked_revin_norm(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)

        mean = (x * m).sum(dim=1, keepdim=True) / denom
        var = ((x - mean).pow(2) * m).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var + self.eps)

        x_norm = (x - mean) / std
        if self.affine_weight is not None:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        x_norm = x_norm * m
        return x_norm, mean, std

    def _masked_revin_denorm(self, y: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        if self.affine_weight is not None:
            y = (y - self.affine_bias) / (self.affine_weight + self.eps)
        return y * std + mean

    def _build_multigrain_sequences(
        self,
        fine: Tensor,
        coarse: Tensor,
        input_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Create GFT-style fine/coarse sequences before patching.

        Returns:
            fine_seq: [B, Lf_max, C]
            fine_mask: [B, Lf_max]
            coarse_seq: [B, Lc_max, C]
            coarse_mask: [B, Lc_max]
        """
        bsz, _, c_in = fine.shape

        fine_list = []
        coarse_list = []
        fine_lens = []
        coarse_lens = []

        for b in range(bsz):
            valid_len = int(input_mask[b].sum().item())
            if valid_len <= 0:
                valid_len = 1

            fine_i = fine[b, :valid_len, :]  # [Li, C]
            coarse_i = coarse[b, :valid_len, :]  # [Li, C]

            period = fft_period_from_series(
                fine_i,
                sampling=self.config.sampling,
                k=self.config.fft_top_k,
            )
            period = max(1, min(period, valid_len))

            pad_len = (period - (valid_len % period)) % period
            fine_pad = F.pad(
                fine_i.transpose(0, 1).unsqueeze(0),
                (0, pad_len),
                mode="replicate",
            ).squeeze(0).transpose(0, 1)  # [Li+pad, C]

            lp = int(fine_pad.size(0))
            fdim = lp // period

            # [1, C, F, P]
            fine_2d = fine_pad.view(fdim, period, c_in).permute(2, 0, 1).unsqueeze(0)
            fine_up_2d = self.fine_grained_block(fine_2d)
            fine_up = fine_up_2d.squeeze(0).permute(1, 2, 0).reshape(-1, c_in)

            target_len = max(int(fine_up.size(0)), 1)
            coarse_up = F.interpolate(
                coarse_i.transpose(0, 1).unsqueeze(0),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)

            fine_list.append(fine_up)
            coarse_list.append(coarse_up)
            fine_lens.append(target_len)
            coarse_lens.append(target_len)

        max_f = max(fine_lens)
        max_c = max(coarse_lens)

        fine_seq = fine.new_zeros(bsz, max_f, c_in)
        coarse_seq = coarse.new_zeros(bsz, max_c, c_in)
        fine_mask = torch.zeros(bsz, max_f, dtype=torch.bool, device=fine.device)
        coarse_mask = torch.zeros(bsz, max_c, dtype=torch.bool, device=fine.device)

        for b in range(bsz):
            lf = fine_lens[b]
            lc = coarse_lens[b]
            fine_seq[b, :lf] = fine_list[b]
            coarse_seq[b, :lc] = coarse_list[b]
            fine_mask[b, :lf] = True
            coarse_mask[b, :lc] = True

        return fine_seq, fine_mask, coarse_seq, coarse_mask

    def encode_multigranularity(self, x: Tensor, input_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if input_mask is None:
            input_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

        x_norm, mean, std = self._masked_revin_norm(x, input_mask)
        fine, coarse = self.decomposition(x_norm, input_mask)

        fine_seq, fine_seq_mask, coarse_seq, coarse_seq_mask = self._build_multigrain_sequences(
            fine=fine,
            coarse=coarse,
            input_mask=input_mask,
        )

        fine_tokens, fine_mask = self.fine_embed(fine_seq, fine_seq_mask)
        coarse_tokens, coarse_mask = self.coarse_embed(coarse_seq, coarse_seq_mask)

        if self.config.use_rope:
            fine_tokens = apply_rope(fine_tokens, base=self.config.rope_base_fine)
            coarse_tokens = apply_rope(coarse_tokens, base=self.config.rope_base_coarse)

        fine_tokens = self.fine_encoder(fine_tokens, src_key_padding_mask=~fine_mask)
        coarse_tokens = self.coarse_encoder(coarse_tokens, src_key_padding_mask=~coarse_mask)

        fused = coarse_tokens
        for block in self.fusion_blocks:
            fused = block(fused, fine_tokens, coarse_mask, fine_mask)

        # Length-aware trust for very short input.
        lengths = input_mask.sum(dim=1).float()
        reliability = torch.sigmoid((lengths - float(self.config.short_len_threshold)) / 2.0).view(-1, 1, 1)
        fused = reliability * fused + (1.0 - reliability) * coarse_tokens
        fused = fused * coarse_mask.unsqueeze(-1).float()

        return {
            "fine_tokens": fine_tokens,
            "coarse_tokens": coarse_tokens,
            "fused_tokens": fused,
            "fine_mask": fine_mask,
            "coarse_mask": coarse_mask,
            "norm_mean": mean,
            "norm_std": std,
        }

    # Backward-friendly alias.
    def MGblock(self, x_enc: Tensor, input_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        out = self.encode_multigranularity(x_enc, input_mask=input_mask)
        return out["fine_tokens"], out["coarse_tokens"], out["fused_tokens"], out["fine_mask"], out["coarse_mask"]

    def _resolve_horizon(
        self,
        batch_size: int,
        device: torch.device,
        horizon: Optional[Union[int, Tensor]],
    ) -> Tuple[Tensor, Tensor, int]:
        if horizon is None:
            horizon_len = torch.full((batch_size,), self.config.max_horizon, device=device, dtype=torch.long)
        elif isinstance(horizon, int):
            horizon_len = torch.full((batch_size,), horizon, device=device, dtype=torch.long)
        else:
            horizon_len = horizon.to(device=device, dtype=torch.long)
            if horizon_len.ndim != 1 or horizon_len.size(0) != batch_size:
                raise ValueError("horizon tensor must be shape [B]")

        max_h = int(horizon_len.max().item())
        if max_h > self.config.max_horizon:
            raise ValueError(f"Requested horizon {max_h} exceeds max_horizon={self.config.max_horizon}.")

        horizon_mask = build_padding_mask(horizon_len, max_len=max_h)
        return horizon_len, horizon_mask, max_h

    def forward(
        self,
        x: Tensor,
        input_mask: Optional[Tensor] = None,
        horizon: Optional[Union[int, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Forecast future values.

        Args:
            x: [B, L, C]
            input_mask: [B, L] bool (True=valid). If None, all valid.
            horizon: int or [B] tensor. If None, uses config.max_horizon.

        Returns:
            dict:
                pred: [B, H, C]
                horizon_mask: [B, H] bool
                fine_mask: [B, Nf] bool
                coarse_mask: [B, Nc] bool
                fused_tokens: [B, Nc, D]
        """
        bsz, _, _ = x.shape
        if input_mask is None:
            input_mask = torch.ones(bsz, x.size(1), dtype=torch.bool, device=x.device)

        encoded = self.encode_multigranularity(x, input_mask=input_mask)
        fused = encoded["fused_tokens"]
        fused_mask = encoded["coarse_mask"]

        _, horizon_mask, max_h = self._resolve_horizon(bsz, x.device, horizon)
        queries = self.horizon_queries[:max_h].unsqueeze(0).expand(bsz, -1, -1)

        dec_out = self.decoder(
            tgt=queries,
            memory=fused,
            tgt_key_padding_mask=~horizon_mask,
            memory_key_padding_mask=~fused_mask,
        )
        pred_norm = self.out_proj(dec_out)

        pred = self._masked_revin_denorm(pred_norm, encoded["norm_mean"], encoded["norm_std"])
        pred = pred * horizon_mask.unsqueeze(-1).float()

        return {
            "pred": pred,
            "horizon_mask": horizon_mask,
            "fine_mask": encoded["fine_mask"],
            "coarse_mask": encoded["coarse_mask"],
            "fused_tokens": fused,
        }


def masked_forecast_loss(
    pred: Tensor,
    target: Tensor,
    target_mask: Tensor,
    mse_weight: float = 0.5,
    mae_weight: float = 0.5,
) -> Tensor:
    """Hybrid masked loss for variable horizon.

    Args:
        pred: [B, H, C]
        target: [B, H, C]
        target_mask: [B, H] bool
    """
    m = target_mask.unsqueeze(-1).float()
    diff = pred - target

    mse = (diff.pow(2) * m).sum() / m.sum().clamp_min(1.0)
    mae = (diff.abs() * m).sum() / m.sum().clamp_min(1.0)
    return mse_weight * mse + mae_weight * mae


if __name__ == "__main__":
    cfg = GFTVarLenConfig(c_in=1, d_model=128, n_heads=4, d_ff=256, max_horizon=40)
    model = GFTVarLenForecaster(cfg)

    x = torch.randn(3, 60, 1)
    lengths = torch.tensor([60, 37, 9])
    mask = build_padding_mask(lengths, max_len=60)
    horizon = torch.tensor([20, 12, 3])

    out = model(x, input_mask=mask, horizon=horizon)
    print("pred:", out["pred"].shape)
    print("horizon_mask:", out["horizon_mask"].shape)
    print("fine_mask:", out["fine_mask"].shape)
    print("coarse_mask:", out["coarse_mask"].shape)
