from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNAD(nn.Module):
    """
    RNAD Loss: Regime-Normalized Adaptive Directional Loss
    for Financial Time Series Forecasting.

    Overview:
        RNAD is a unified objective designed for heteroskedastic,
        heavy-tailed, and noisy financial time series.

    Components:
        1. Regime-normalized magnitude loss (scale invariant)
        2. Tail-adaptive robust loss (L2 core + Charbonnier tails)
        3. Noise-aware directional calibration

    Objective:
        L = L_mag + lambda_dir * L_dir

    Design Properties:
        - Scale invariance via volatility normalization
        - Robustness to heavy-tailed errors
        - Suppression of noisy directional signals
        - Alignment with directional forecasting performance

    Args:
        lambda_dir: Weight for directional component
        tail_tau: Threshold for tail transition
        tail_temp: Smoothness of tail transition
        charb_eps: Charbonnier smoothing constant
        beta: Slope for directional encoding
        kappa: Margin sharpness for directional loss
        noise_level: Threshold for noise suppression
        noise_temp: Smoothness of noise gating
        vol_momentum: EMA momentum for volatility estimation
        eps: Numerical stability constant
    """

    def __init__(
        self,
        lambda_dir: float = 0.5,
        tail_tau: float = 1.1,
        tail_temp: float = 0.30,
        charb_eps: float = 1e-3,
        beta: float = 4.5,
        kappa: float = 4.5,
        noise_level: float = 0.10,
        noise_temp: float = 0.04,
        vol_momentum: float = 0.985,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.lambda_dir = lambda_dir
        self.tail_tau = tail_tau
        self.tail_temp = tail_temp
        self.charb_eps = charb_eps

        self.beta = beta
        self.kappa = kappa

        self.noise_level = noise_level
        self.noise_temp = noise_temp

        self.vol_momentum = vol_momentum
        self.eps = eps

        # EMA volatility (regime memory)
        self.register_buffer("ema_vol", torch.tensor(1.0))

    # ---------------------------------------------------------
    # Regime normalization
    # ---------------------------------------------------------
    def _compute_scale(self, delta_true: torch.Tensor) -> torch.Tensor:
        """
        Compute volatility scale using batch + EMA blending.
        Ensures stability under regime shifts.
        """
        batch_vol = torch.sqrt(delta_true.pow(2).mean() + self.eps)

        if self.training:
            with torch.no_grad():
                self.ema_vol.copy_(
                    self.vol_momentum * self.ema_vol
                    + (1.0 - self.vol_momentum) * batch_vol
                )

        # Geometric blend for smooth regime adaptation
        sigma = torch.sqrt(batch_vol * self.ema_vol + self.eps)
        return sigma.clamp_min(self.eps)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            y_prev: Previous timestep values (for returns)

        Returns:
            Scalar loss
        """

        # --- basic quantities ---
        error = y_pred - y_true
        delta_true = y_true - y_prev
        delta_pred = y_pred - y_prev

        # --- regime normalization ---
        sigma = self._compute_scale(delta_true)

        u = error / sigma
        v_true = delta_true / sigma
        v_pred = delta_pred / sigma

        # =====================================================
        # 1. Tail-adaptive magnitude loss
        # =====================================================

        tail_gate = torch.sigmoid(
            (u.abs() - self.tail_tau) / (self.tail_temp + self.eps)
        )

        # L2 core (efficient near center)
        l2 = 0.5 * u.pow(2)

        # Charbonnier loss (robust to outliers)
        charb = torch.sqrt(u.pow(2) + self.charb_eps**2) - self.charb_eps

        L_mag = ((1.0 - tail_gate) * l2 + tail_gate * charb).mean()

        # =====================================================
        # 2. Noise-aware directional loss
        # =====================================================

        noise_gate = torch.sigmoid(
            (v_true.abs() - self.noise_level) / (self.noise_temp + self.eps)
        )

        # Smooth directional encoding
        t = torch.tanh(self.beta * v_true)
        p = torch.tanh(self.beta * v_pred)

        # Margin-based directional penalty
        direction_loss = F.softplus(-self.kappa * t * p)

        L_dir = (noise_gate * direction_loss).mean()

        # =====================================================
        # Final objective
        # =====================================================

        loss = L_mag + self.lambda_dir * L_dir
        return loss