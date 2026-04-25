from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNAD(nn.Module):
    """
    Deterministic RNAD loss with explicit ablation switches.

    Ablation mapping:
        - full     -> use_direction=True,  use_tail=True,  use_noise_gate=True
        - no_dir   -> use_direction=False
        - no_tail  -> use_tail=False
        - no_noise -> use_noise_gate=False
    """

    def __init__(
        self,
        use_direction: bool = True,
        use_tail: bool = True,
        use_noise_gate: bool = True,
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

        self.use_direction = bool(use_direction)
        self.use_tail = bool(use_tail)
        self.use_noise_gate = bool(use_noise_gate)

        self.lambda_dir = float(lambda_dir)
        self.tail_tau = float(tail_tau)
        self.tail_temp = float(tail_temp)
        self.charb_eps = float(charb_eps)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.noise_level = float(noise_level)
        self.noise_temp = float(noise_temp)
        self.vol_momentum = float(vol_momentum)
        self.eps = float(eps)

        # EMA volatility (regime memory) - stateful buffer, not a trainable parameter.
        self.register_buffer("ema_vol", torch.tensor(1.0))

    def _compute_scale(self, delta_true: torch.Tensor) -> torch.Tensor:
        """Compute volatility scale using batch + EMA blending."""
        batch_vol = torch.sqrt(delta_true.pow(2).mean() + self.eps)

        if self.training:
            ema_vol = (
                self.vol_momentum * self.ema_vol.detach()
                + (1.0 - self.vol_momentum) * batch_vol
            )

            with torch.no_grad():
                self.ema_vol.copy_(ema_vol.detach())
        else:
            ema_vol = self.ema_vol.detach()

        sigma = torch.sqrt(batch_vol * ema_vol + self.eps)
        return sigma.clamp_min(self.eps)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_prev: torch.Tensor,
    ) -> torch.Tensor:
        error = y_pred - y_true
        delta_true = y_true - y_prev
        delta_pred = y_pred - y_prev

        sigma = self._compute_scale(delta_true)
        u = error / sigma
        v_true = delta_true / sigma
        v_pred = delta_pred / sigma

        l2 = 0.5 * u.pow(2)
        if not self.use_tail:
            L_mag = l2.mean()
        else:
            tail_gate = torch.sigmoid(
                (u.abs() - self.tail_tau) / (self.tail_temp + self.eps)
            )
            charb = torch.sqrt(u.pow(2) + self.charb_eps**2) - self.charb_eps
            L_mag = ((1.0 - tail_gate) * l2 + tail_gate * charb).mean()

        if not self.use_direction:
            return L_mag

        if self.use_noise_gate:
            noise_gate = torch.sigmoid(
                (v_true.abs() - self.noise_level) / (self.noise_temp + self.eps)
            )
        else:
            noise_gate = 1.0

        t = torch.tanh(self.beta * v_true)
        p = torch.tanh(self.beta * v_pred)

        direction_loss = F.softplus(-self.kappa * t * p)
        L_dir = (noise_gate * direction_loss).mean()

        return L_mag + self.lambda_dir * L_dir
