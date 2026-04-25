from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RNAD", "rnad_loss"]


def _compute_rnad_loss(
	y_pred: torch.Tensor,
	y_true: torch.Tensor,
	y_prev: torch.Tensor,
	*,
	lambda_dir: float,
	tail_tau: float,
	tail_temp: float,
	charb_eps: float,
	beta: float,
	kappa: float,
	noise_level: float,
	noise_temp: float,
	vol_momentum: float,
	eps: float,
	ema_vol: Optional[torch.Tensor] = None,
	training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Compute the RNAD objective and return the updated EMA volatility."""

	y_true = y_true.to(dtype=y_pred.dtype, device=y_pred.device)
	y_prev = y_prev.to(dtype=y_pred.dtype, device=y_pred.device)

	if ema_vol is None:
		ema_vol = torch.tensor(1.0, device=y_pred.device, dtype=y_pred.dtype)
	else:
		ema_vol = ema_vol.to(device=y_pred.device, dtype=y_pred.dtype)

	error = y_pred - y_true
	delta_true = y_true - y_prev
	delta_pred = y_pred - y_prev

	batch_vol = torch.sqrt(delta_true.pow(2).mean() + eps)
	updated_ema = (
		vol_momentum * ema_vol + (1.0 - vol_momentum) * batch_vol
		if training
		else ema_vol
	)

	sigma = torch.sqrt(batch_vol * updated_ema + eps).clamp_min(eps)

	u = error / sigma
	v_true = delta_true / sigma
	v_pred = delta_pred / sigma

	tail_gate = torch.sigmoid((u.abs() - tail_tau) / (tail_temp + eps))
	l2 = 0.5 * u.pow(2)
	charb = torch.sqrt(u.pow(2) + charb_eps**2) - charb_eps
	magnitude_loss = ((1.0 - tail_gate) * l2 + tail_gate * charb).mean()

	noise_gate = torch.sigmoid((v_true.abs() - noise_level) / (noise_temp + eps))
	t = torch.tanh(beta * v_true)
	p = torch.tanh(beta * v_pred)
	direction_loss = F.softplus(-kappa * t * p)
	directional_loss = (noise_gate * direction_loss).mean()

	loss = magnitude_loss + lambda_dir * directional_loss
	return loss, updated_ema


def rnad_loss(
	y_pred: torch.Tensor,
	y_true: torch.Tensor,
	y_prev: torch.Tensor,
	*,
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
) -> torch.Tensor:
	"""Functional RNAD loss.

	Args:
		y_pred: Predicted values.
		y_true: Ground-truth target values.
		y_prev: Previous-step target values used to compute return-like deltas.
		lambda_dir: Weight of the directional component.
		tail_tau: Threshold controlling the L2-to-Charbonnier transition.
		tail_temp: Smoothness of the tail gate.
		charb_eps: Charbonnier smoothing constant.
		beta: Slope used in directional encoding.
		kappa: Margin sharpness for the directional penalty.
		noise_level: Threshold for suppressing noisy directional signals.
		noise_temp: Smoothness of the noise gate.
		vol_momentum: EMA momentum for volatility tracking.
		eps: Numerical stability constant.

	Returns:
		A scalar tensor containing the RNAD loss.
	"""

	loss, _ = _compute_rnad_loss(
		y_pred,
		y_true,
		y_prev,
		lambda_dir=lambda_dir,
		tail_tau=tail_tau,
		tail_temp=tail_temp,
		charb_eps=charb_eps,
		beta=beta,
		kappa=kappa,
		noise_level=noise_level,
		noise_temp=noise_temp,
		vol_momentum=vol_momentum,
		eps=eps,
		ema_vol=None,
		training=False,
	)
	return loss


class RNAD(nn.Module):
	"""Regime-Normalized Adaptive Directional Loss (RNAD).

	RNAD combines three ideas for noisy financial time series forecasting:

	1. A regime-normalized magnitude term that rescales errors by a volatility
	   estimate.
	2. A tail-adaptive robust penalty that behaves like L2 near zero and a
	   Charbonnier-style loss in the tails.
	3. A noise-aware directional term that calibrates sign matching on
	   non-trivial return moves.

	Args:
		lambda_dir: Weight for the directional component.
		tail_tau: Threshold for switching toward the robust tail regime.
		tail_temp: Smoothness of the tail transition.
		charb_eps: Charbonnier smoothing constant.
		beta: Slope used in directional encoding.
		kappa: Margin sharpness for the directional penalty.
		noise_level: Threshold below which directional signals are suppressed.
		noise_temp: Smoothness of the noise gate.
		vol_momentum: EMA momentum for volatility estimation.
		eps: Numerical stability constant.

	Note:
		The loss expects ``y_prev`` to contain the previous-step values needed
		to compute return-like deltas.
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
	) -> None:
		super().__init__()

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

		# EMA volatility state (updated during training calls).
		self.register_buffer("ema_vol", torch.tensor(1.0))

	def forward(
		self,
		y_pred: torch.Tensor,
		y_true: torch.Tensor,
		y_prev: torch.Tensor,
	) -> torch.Tensor:
		"""Compute the RNAD loss.

		Args:
			y_pred: Model predictions.
			y_true: Ground-truth target values.
			y_prev: Previous-step target values used to form deltas/returns.

		Returns:
			A scalar tensor containing the batch loss.
		"""

		loss, updated_ema = _compute_rnad_loss(
			y_pred,
			y_true,
			y_prev,
			lambda_dir=self.lambda_dir,
			tail_tau=self.tail_tau,
			tail_temp=self.tail_temp,
			charb_eps=self.charb_eps,
			beta=self.beta,
			kappa=self.kappa,
			noise_level=self.noise_level,
			noise_temp=self.noise_temp,
			vol_momentum=self.vol_momentum,
			eps=self.eps,
			ema_vol=self.ema_vol,
			training=self.training,
		)

		if self.training:
			self.ema_vol.copy_(updated_ema.detach())

		return loss
