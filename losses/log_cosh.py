from __future__ import annotations

import torch
from torch import nn


class LogCoshLoss(nn.Module):
    """Log-cosh loss for regression.

    log(cosh(x)) ≈ (x²)/2 for small |x| and |x| - log(2) for large |x|.
    Approximately equal to MSE for small errors and MAE for large errors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = y_pred - y_true
        loss = torch.mean(torch.log(torch.cosh(error)))
        return loss
