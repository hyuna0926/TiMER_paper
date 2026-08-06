from __future__ import annotations

import argparse
import ast
import math
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from gft_varlen_forecast import (
    GFTVarLenConfig,
    GFTVarLenForecaster,
    masked_forecast_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_python_obj(v: Any) -> Any:
    if isinstance(v, str):
        return ast.literal_eval(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return v
    raise TypeError(f"Unsupported input_ts type: {type(v)}")


def _is_nested_sequence(v: Any) -> bool:
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        return False
    return any(isinstance(x, (list, tuple, np.ndarray)) for x in v)


def _channel_mean_from_rows(rows: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(r) for r in rows)
    mat = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return np.nanmean(mat, axis=0)


def _nested_rows_to_arrays(seq: Any) -> List[np.ndarray]:
    rows: List[np.ndarray] = []
    for row in seq:
        if isinstance(row, np.ndarray):
            row_arr = row.astype(np.float32).reshape(-1)
        else:
            row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
        row_arr = row_arr[np.isfinite(row_arr)]
        if len(row_arr) > 0:
            rows.append(row_arr)
    return rows


def parse_ts_field(v: Any, nested_policy: str = "auto") -> List[float]:
    """Parse input_ts into a single 1D float sequence.

    nested_policy:
        - auto: if nested rows have equal length -> channel_mean, else concat
        - concat: flatten row-by-row
        - channel_mean: align rows by time index and average across rows
        - separate: not supported here; use `parse_ts_candidates`
    """
    seq = _to_python_obj(v)

    if not _is_nested_sequence(seq):
        arr = np.asarray(seq, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return arr.tolist()

    rows = _nested_rows_to_arrays(seq)

    if len(rows) == 0:
        return []

    if nested_policy not in {"auto", "concat", "channel_mean", "separate"}:
        raise ValueError(f"Unsupported nested_policy: {nested_policy}")
    if nested_policy == "separate":
        raise ValueError("Use parse_ts_candidates(..., nested_policy='separate') for nested samples.")

    if nested_policy == "concat":
        arr = np.concatenate(rows, axis=0)
    elif nested_policy == "channel_mean":
        arr = _channel_mean_from_rows(rows)
    else:
        lengths = [len(r) for r in rows]
        same_length = len(set(lengths)) == 1
        arr = _channel_mean_from_rows(rows) if same_length else np.concatenate(rows, axis=0)

    arr = arr[np.isfinite(arr)]
    return arr.tolist()


def parse_ts_candidates(v: Any, nested_policy: str = "separate") -> List[List[float]]:
    """Parse input_ts into one or multiple candidate 1D sequences.

    nested_policy:
        - separate: nested rows are treated as independent series (recommended)
        - auto/concat/channel_mean: nested rows are merged into one sequence
    """
    seq = _to_python_obj(v)
    is_nested = _is_nested_sequence(seq)

    if not is_nested:
        one = parse_ts_field(seq, nested_policy="auto")
        return [one] if len(one) > 0 else []

    rows = _nested_rows_to_arrays(seq)
    if len(rows) == 0:
        return []

    if nested_policy == "separate":
        return [r.tolist() for r in rows]

    merged = parse_ts_field(seq, nested_policy=nested_policy)
    return [merged] if len(merged) > 0 else []


class Split82ForecastDataset(Dataset):
    """Create 8:2 forecasting pairs from `input_ts`.

    Each sample:
        input  = first 80% points
        target = last 20% points
    """

    def __init__(
        self,
        parquet_path: str,
        split_ratio: float = 0.8,
        min_total_len: int = 2,
        max_input_len: int | None = None,
        max_samples: int | None = None,
        nested_policy: str = "separate",
    ):
        super().__init__()
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio must be between 0 and 1")
        if nested_policy not in {"separate", "auto", "concat", "channel_mean"}:
            raise ValueError("nested_policy must be one of: separate, auto, concat, channel_mean")
        self.nested_policy = nested_policy

        try:
            df = pd.read_parquet(parquet_path, columns=["input_ts"])
        except Exception as e:
            raise RuntimeError(
                f"Failed to read parquet: {parquet_path}\n"
                f"Reason: {e}\n"
                "Try upgrading pyarrow/pandas or converting parquet once in your environment."
            )

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.n_nested = 0
        self.n_nested_separate_rows = 0
        self.n_nested_channel_mean = 0
        self.n_nested_concat = 0

        for raw in df["input_ts"].tolist():
            try:
                parsed_obj = _to_python_obj(raw)
                is_nested = _is_nested_sequence(parsed_obj)
                candidates = parse_ts_candidates(parsed_obj, nested_policy=self.nested_policy)
            except Exception:
                continue

            if is_nested:
                self.n_nested += 1
                if self.nested_policy == "separate":
                    self.n_nested_separate_rows += len(candidates)
                elif self.nested_policy == "concat":
                    self.n_nested_concat += 1
                elif self.nested_policy == "channel_mean":
                    self.n_nested_channel_mean += 1
                else:
                    # auto policy accounting
                    rows = _nested_rows_to_arrays(parsed_obj)
                    lengths = [len(r) for r in rows]
                    if len(lengths) > 0 and len(set(lengths)) == 1:
                        self.n_nested_channel_mean += 1
                    else:
                        self.n_nested_concat += 1

            for seq in candidates:
                if len(seq) < min_total_len:
                    continue

                total_len = len(seq)
                split_idx = int(math.floor(total_len * split_ratio))
                split_idx = max(1, min(split_idx, total_len - 1))

                x = np.asarray(seq[:split_idx], dtype=np.float32)
                y = np.asarray(seq[split_idx:], dtype=np.float32)

                if max_input_len is not None and len(x) > max_input_len:
                    # Keep the most recent context.
                    x = x[-max_input_len:]

                self.samples.append((x, y))

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found after parsing input_ts.")

        self.max_target_len = max(len(y) for _, y in self.samples)
        if self.n_nested > 0:
            print(
                f"[dataset] nested input_ts handled: total={self.n_nested}, "
                f"separate_rows={self.n_nested_separate_rows}, "
                f"channel_mean={self.n_nested_channel_mean}, concat={self.n_nested_concat}, "
                f"policy={self.nested_policy}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        # x: [L_in] -> [L_in, 1]
        # y: [L_out] -> [L_out, 1]
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)


def collate_varlen(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length input/target and build masks.

    Returns:
        x: [B, L_max, 1]
        x_mask: [B, L_max] bool
        y: [B, H_max, 1]
        y_mask: [B, H_max] bool
        horizon_len: [B]
    """
    bsz = len(batch)
    x_lens = torch.tensor([item[0].size(0) for item in batch], dtype=torch.long)
    y_lens = torch.tensor([item[1].size(0) for item in batch], dtype=torch.long)

    l_max = int(x_lens.max().item())
    h_max = int(y_lens.max().item())

    x_pad = torch.zeros(bsz, l_max, 1, dtype=torch.float32)
    y_pad = torch.zeros(bsz, h_max, 1, dtype=torch.float32)
    x_mask = torch.zeros(bsz, l_max, dtype=torch.bool)
    y_mask = torch.zeros(bsz, h_max, dtype=torch.bool)

    for i, (x, y) in enumerate(batch):
        lx = x.size(0)
        ly = y.size(0)
        x_pad[i, :lx] = x
        y_pad[i, :ly] = y
        x_mask[i, :lx] = True
        y_mask[i, :ly] = True

    return {
        "x": x_pad,
        "x_mask": x_mask,
        "y": y_pad,
        "y_mask": y_mask,
        "horizon_len": y_lens,
    }


@torch.no_grad()
def evaluate(model: GFTVarLenForecaster, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses, maes, mses = [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        x_mask = batch["x_mask"].to(device)
        y = batch["y"].to(device)
        y_mask = batch["y_mask"].to(device)
        horizon = batch["horizon_len"].to(device)

        out = model(x, input_mask=x_mask, horizon=horizon)
        pred = out["pred"]  # [B, H, 1]

        loss = masked_forecast_loss(pred, y, y_mask)
        losses.append(loss.item())

        m = y_mask.unsqueeze(-1).float()
        diff = pred - y
        mae = (diff.abs() * m).sum() / m.sum().clamp_min(1.0)
        mse = (diff.pow(2) * m).sum() / m.sum().clamp_min(1.0)

        maes.append(mae.item())
        mses.append(mse.item())

    return {
        "loss": float(np.mean(losses)),
        "mae": float(np.mean(maes)),
        "mse": float(np.mean(mses)),
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_parquet",
        type=str,
        default=os.path.join(repo_root, "data", "TSAQA", "train.parquet"),
    )
    parser.add_argument(
        "--val_parquet",
        type=str,
        default=os.path.join(repo_root, "data", "TSAQA", "val.parquet"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_input_len", type=int, default=None)
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(here, "checkpoint", "gft_varlen_best_forecast.pt"),
    )
    parser.add_argument(
        "--nested_policy",
        type=str,
        default="separate",
        choices=["separate", "auto", "concat", "channel_mean"],
        help="how to handle nested input_ts like [[...], [...]]",
    )

    # Model hparams
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fine_patch_len", type=int, default=4)
    parser.add_argument("--fine_stride", type=int, default=2)
    parser.add_argument("--coarse_patch_len", type=int, default=12)
    parser.add_argument("--coarse_stride", type=int, default=6)
    parser.add_argument("--decomp_kernel_size", type=int, default=5)
    parser.add_argument("--up_scale", type=int, default=3)
    parser.add_argument("--num_fine_grained_conv", type=int, default=2)
    parser.add_argument("--sampling", type=str, default="median", choices=["median", "max", "min"])
    parser.add_argument("--fft_top_k", type=int, default=10)
    parser.add_argument("--fine_layers", type=int, default=2)
    parser.add_argument("--coarse_layers", type=int, default=2)
    parser.add_argument("--fusion_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_dataset = Split82ForecastDataset(
        parquet_path=args.train_parquet,
        split_ratio=0.8,
        min_total_len=2,
        max_input_len=args.max_input_len,
        max_samples=args.max_train_samples,
        nested_policy=args.nested_policy,
    )
    val_dataset = Split82ForecastDataset(
        parquet_path=args.val_parquet,
        split_ratio=0.8,
        min_total_len=2,
        max_input_len=args.max_input_len,
        max_samples=args.max_val_samples,
        nested_policy=args.nested_policy,
    )
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(
        f"max target len train/val: "
        f"{train_dataset.max_target_len}/{val_dataset.max_target_len}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_varlen,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_varlen,
        pin_memory=True,
    )

    config = GFTVarLenConfig(
        c_in=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        decomp_kernel_size=args.decomp_kernel_size,
        fine_patch_len=args.fine_patch_len,
        fine_stride=args.fine_stride,
        coarse_patch_len=args.coarse_patch_len,
        coarse_stride=args.coarse_stride,
        up_scale=args.up_scale,
        num_fine_grained_conv=args.num_fine_grained_conv,
        sampling=args.sampling,
        fft_top_k=args.fft_top_k,
        fine_layers=args.fine_layers,
        coarse_layers=args.coarse_layers,
        fusion_layers=args.fusion_layers,
        decoder_layers=args.decoder_layers,
        max_horizon=max(train_dataset.max_target_len, val_dataset.max_target_len),
    )
    print("config:", asdict(config))

    model = GFTVarLenForecaster(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            y_mask = batch["y_mask"].to(device)
            horizon = batch["horizon_len"].to(device)

            out = model(x, input_mask=x_mask, horizon=horizon)
            pred = out["pred"]  # [B, H, 1]

            loss = masked_forecast_loss(pred, y, y_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))

        val_metrics = evaluate(model, val_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | lr {lr_now:.6e} | "
            f"train_loss {train_loss:.6f} | "
            f"val_loss {val_metrics['loss']:.6f} | val_mae {val_metrics['mae']:.6f} | val_mse {val_metrics['mse']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            payload = {
                "model": model.state_dict(),
                "config": asdict(config),
                "best_val_loss": best_val,
                "epoch": epoch,
            }
            torch.save(payload, args.save_path)
            print(f"  saved best checkpoint -> {args.save_path}")

    print(f"done. best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
