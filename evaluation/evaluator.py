from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from evaluation.metrics import directional_accuracy, rmse, mae
from utils.device import assert_same_device, transfer_kwargs


@torch.no_grad()
def evaluate_model_with_outputs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    target_scaler=None,
) -> Dict[str, object]:
    model.eval()
    to_kwargs = transfer_kwargs(device)
    was_training = bool(getattr(loss_fn, "training", False))
    if hasattr(loss_fn, "eval"):
        loss_fn.eval()

    try:
        weighted_loss_sum = 0.0
        total_samples = 0
        pred_scaled_list = []
        true_scaled_list = []
        true_raw_list = []
        prev_raw_list = []

        for batch in dataloader:
            x = batch["x"].to(device, **to_kwargs)
            y = batch["y"].to(device, **to_kwargs)
            y_prev = batch["y_prev"].to(device, **to_kwargs)
            assert_same_device(x, y, y_prev)

            y_pred = model(x)
            assert_same_device(y_pred, y, y_prev)
            loss = loss_fn(y_pred, y, y_prev)
            assert_same_device(loss, y_pred)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite evaluation loss detected: {loss.item()}")
            if not torch.isfinite(y_pred).all():
                raise RuntimeError("Non-finite predictions detected during evaluation.")

            batch_size = int(x.size(0))
            weighted_loss_sum += float(loss.item()) * batch_size
            total_samples += batch_size

            pred_scaled_list.append(y_pred.detach().cpu().numpy())
            true_scaled_list.append(y.detach().cpu().numpy())

            true_raw_list.append(batch["y_raw"].cpu().numpy())
            prev_raw_list.append(batch["y_prev_raw"].cpu().numpy())

        y_pred_scaled = np.concatenate(pred_scaled_list, axis=0).reshape(-1, 1)
        y_true_scaled = np.concatenate(true_scaled_list, axis=0).reshape(-1, 1)
        y_true_raw = np.concatenate(true_raw_list, axis=0).reshape(-1, 1)
        y_prev_raw = np.concatenate(prev_raw_list, axis=0).reshape(-1, 1)

        if target_scaler is not None:
            y_pred_raw = target_scaler.inverse_transform(y_pred_scaled)
        else:
            y_pred_raw = y_pred_scaled

        metrics = {
            "loss": float(weighted_loss_sum / max(1, total_samples)),
            "rmse": rmse(y_pred_raw, y_true_raw),
            "mae": mae(y_pred_raw, y_true_raw),
            "directional_accuracy": directional_accuracy(
                y_pred=y_pred_raw,
                y_true=y_true_raw,
                y_prev=y_prev_raw,
                ignore_zero_change=True,
            ),
        }
        return {
            "metrics": metrics,
            "y_pred": y_pred_raw.reshape(-1),
            "y_true": y_true_raw.reshape(-1),
            "y_prev": y_prev_raw.reshape(-1),
        }
    finally:
        if hasattr(loss_fn, "train") and was_training:
            loss_fn.train()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    target_scaler=None,
) -> Dict[str, float]:
    out = evaluate_model_with_outputs(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        target_scaler=target_scaler,
    )
    return out["metrics"]
