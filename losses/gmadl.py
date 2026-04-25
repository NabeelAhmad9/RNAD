from __future__ import annotations

import torch
from torch import nn


class GMADL(nn.Module):
    """Generalized Magnitude-Aware Directional Loss (GMADL).

    A differentiable hybrid objective:
      - Magnitude term: robust smooth L1-like error magnitude
      - Direction term: smooth directional alignment via tanh-projected deltas

    Formula:
      mag = sqrt((y_pred - y_true)^2 + eps)
      d_true = tanh(beta * (y_true - y_prev))
      d_pred = tanh(beta * (y_pred - y_prev))
      dir_penalty = 1 - d_true * d_pred
      loss = alpha * mag + (1 - alpha) * dir_penalty
    """

    def __init__(self, alpha: float = 0.5, beta: float = 5.0, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt((y_pred - y_true) ** 2 + self.eps)

        smooth_true = torch.tanh(self.beta * (y_true - y_prev))
        smooth_pred = torch.tanh(self.beta * (y_pred - y_prev))
        directional_penalty = 1.0 - smooth_true * smooth_pred

        return (self.alpha * magnitude + (1.0 - self.alpha) * directional_penalty).mean()
