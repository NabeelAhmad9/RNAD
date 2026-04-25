from __future__ import annotations

import numpy as np



def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))



def directional_accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_prev: np.ndarray,
    ignore_zero_change: bool = True,
) -> float:
    """Directional Accuracy based on sign(y_t+1 - y_t)."""
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prev = np.asarray(y_prev, dtype=np.float64).reshape(-1)

    pred_sign = np.sign(y_pred - y_prev)
    true_sign = np.sign(y_true - y_prev)

    if ignore_zero_change:
        mask = true_sign != 0
        if mask.sum() == 0:
            return float("nan")
        acc = np.mean(pred_sign[mask] == true_sign[mask])
    else:
        acc = np.mean(pred_sign == true_sign)

    return float(acc * 100.0)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error between predictions and true values."""
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true)))
