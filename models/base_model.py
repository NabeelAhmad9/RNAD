from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn


class BaseForecastModel(nn.Module, ABC):
    """Base interface for forecasting models.

    Supports optional model metadata and hyperparameter registration so
    architecture-specific models (e.g., official PatchTST) can expose
    their native configuration without adapter boilerplate.
    """

    def __init__(self, model_name: str | None = None, **hyperparams: Any) -> None:
        super().__init__()
        self.model_name = model_name or self.__class__.__name__
        self.hparams: Dict[str, Any] = dict(hyperparams)

        # Provide common attribute-style access expected by some model
        # implementations (e.g., self.window_size, self.horizon, ...).
        for key, value in hyperparams.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch, seq_len, n_features]

        Returns:
            Tensor of shape [batch, horizon]
        """

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
