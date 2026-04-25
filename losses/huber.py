from __future__ import annotations

import torch
from torch import nn


class HuberLossModule(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = float(delta)
        self._loss = nn.HuberLoss(delta=self.delta, reduction="mean")

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._loss(y_pred, y_true)
