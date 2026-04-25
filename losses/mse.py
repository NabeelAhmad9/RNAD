from __future__ import annotations

import torch
from torch import nn


class MSELossModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._loss(y_pred, y_true)
