from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = ["RNAD", "rnad_loss"]

try:
	import tensorflow as tf
	_TF_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency guard
	tf = None  # type: ignore[assignment]
	_TF_IMPORT_ERROR = exc


def _missing_tf(*_: Any, **__: Any) -> None:
	"""Raise a clear error when TensorFlow is unavailable."""

	raise ImportError(
		"TensorFlow is required for this module but is not installed in the current "
		"environment. Install tensorflow to use RNADTensorFlow."
	) from _TF_IMPORT_ERROR


if tf is not None:

	def _as_tensor(value: Any, dtype: tf.dtypes.DType) -> tf.Tensor:
		"""Convert ``value`` to a tensor with the requested dtype."""

		return tf.cast(tf.convert_to_tensor(value), dtype)


	def _unpack_targets(y_true: Any) -> tuple[Any, Any]:
		"""Extract ``(y_true, y_prev)`` from structured targets.

		Accepted formats:
			- ``(y_true, y_prev)`` or ``[y_true, y_prev]``
			- ``{"y_true": ..., "y_prev": ...}``
		"""

		if isinstance(y_true, Mapping):
			if "y_true" not in y_true or "y_prev" not in y_true:
				raise ValueError(
					"When passing a mapping, provide both 'y_true' and 'y_prev'."
				)
			return y_true["y_true"], y_true["y_prev"]

		if isinstance(y_true, Sequence) and len(y_true) == 2 and not tf.is_tensor(y_true):
			return y_true[0], y_true[1]

		raise ValueError(
			"RNAD expects structured targets containing both current and previous values, "
			"e.g. (y_true, y_prev) or {'y_true': ..., 'y_prev': ...}."
		)


	def _compute_rnad_loss(
		y_pred: tf.Tensor,
		y_true: tf.Tensor,
		y_prev: tf.Tensor,
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
		ema_vol: tf.Variable | None = None,
		training: bool = False,
	) -> tuple[tf.Tensor, tf.Tensor]:
		"""Compute RNAD and return the loss plus updated EMA volatility."""

		y_pred = tf.convert_to_tensor(y_pred)
		dtype = y_pred.dtype
		y_true = _as_tensor(y_true, dtype)
		y_prev = _as_tensor(y_prev, dtype)

		if ema_vol is None:
			ema_vol_value = tf.cast(1.0, dtype)
		else:
			ema_vol_value = tf.cast(ema_vol, dtype)

		error = y_pred - y_true
		delta_true = y_true - y_prev
		delta_pred = y_pred - y_prev

		batch_vol = tf.sqrt(tf.reduce_mean(tf.square(delta_true)) + tf.cast(eps, dtype))
		if training:
			updated_ema = (
				tf.cast(vol_momentum, dtype) * ema_vol_value
				+ (1.0 - tf.cast(vol_momentum, dtype)) * batch_vol
			)
		else:
			updated_ema = ema_vol_value

		sigma = tf.sqrt(batch_vol * updated_ema + tf.cast(eps, dtype))
		sigma = tf.maximum(sigma, tf.cast(eps, dtype))

		u = error / sigma
		v_true = delta_true / sigma
		v_pred = delta_pred / sigma

		tail_gate = tf.math.sigmoid(
			(tf.abs(u) - tf.cast(tail_tau, dtype))
			/ (tf.cast(tail_temp, dtype) + tf.cast(eps, dtype))
		)
		l2 = 0.5 * tf.square(u)
		charb = tf.sqrt(tf.square(u) + tf.cast(charb_eps, dtype) ** 2) - tf.cast(charb_eps, dtype)
		magnitude_loss = tf.reduce_mean((1.0 - tail_gate) * l2 + tail_gate * charb)

		noise_gate = tf.math.sigmoid(
			(tf.abs(v_true) - tf.cast(noise_level, dtype))
			/ (tf.cast(noise_temp, dtype) + tf.cast(eps, dtype))
		)
		t = tf.tanh(tf.cast(beta, dtype) * v_true)
		p = tf.tanh(tf.cast(beta, dtype) * v_pred)
		direction_loss = tf.nn.softplus(-tf.cast(kappa, dtype) * t * p)
		directional_loss = tf.reduce_mean(noise_gate * direction_loss)

		loss = magnitude_loss + tf.cast(lambda_dir, dtype) * directional_loss
		return loss, updated_ema


	def rnad_loss(
		y_pred: tf.Tensor,
		y_true: tf.Tensor,
		y_prev: tf.Tensor,
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
	) -> tf.Tensor:
		"""Functional RNAD loss.

		Args:
			y_pred: Predicted values.
			y_true: Ground-truth values.
			y_prev: Previous-step values used to compute return-like deltas.
			lambda_dir: Weight for the directional component.
			tail_tau: Threshold controlling the robust tail transition.
			tail_temp: Smoothness of the tail gate.
			charb_eps: Charbonnier smoothing constant.
			beta: Slope used in the directional encoding.
			kappa: Margin sharpness for the directional penalty.
			noise_level: Threshold below which directional signals are suppressed.
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


	class RNAD(tf.keras.losses.Loss):
		"""Regime-Normalized Adaptive Directional Loss (RNAD).

		This TensorFlow/Keras implementation mirrors the PyTorch version and uses
		the same three components:

		1. Regime-normalized magnitude loss.
		2. Tail-adaptive robust penalty.
		3. Noise-aware directional calibration.

		Usage examples:
			Direct helper:
				``loss = rnad_loss(y_pred, y_true, y_prev)``

			Stateful class:
				``loss_fn = RNAD()``
				``loss = loss_fn((y_true, y_prev), y_pred)``

		The class accepts structured targets so it remains compatible with the
		Keras loss API while still receiving the previous-step values required by
		RNAD.
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
			name: str = "rnad",
			reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
		) -> None:
			super().__init__(name=name, reduction=reduction)

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

			self.ema_vol = self.add_weight(
				name="ema_vol",
				shape=(),
				dtype=tf.float32,
				initializer=tf.keras.initializers.Constant(1.0),
				trainable=False,
			)

		def call(self, y_true: Any, y_pred: tf.Tensor) -> tf.Tensor:
			"""Compute the RNAD loss.

			Args:
				y_true: Either a tuple/list ``(target, previous_target)`` or a
					mapping with keys ``'y_true'`` and ``'y_prev'``.
				y_pred: Model predictions.

			Returns:
				A scalar tensor containing the batch loss.
			"""

			target, previous = _unpack_targets(y_true)
			loss, updated_ema = _compute_rnad_loss(
				y_pred,
				target,
				previous,
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
				training=True,
			)

			self.ema_vol.assign(tf.cast(updated_ema, self.ema_vol.dtype))
			return loss
else:

	def rnad_loss(*args: Any, **kwargs: Any) -> None:
		"""Placeholder that raises a clear error when TensorFlow is unavailable."""

		_missing_tf()

	class RNAD:  # type: ignore[too-many-ancestors]
		"""Placeholder RNAD class for environments without TensorFlow."""

		def __init__(self, *args: Any, **kwargs: Any) -> None:
			_missing_tf()

		def __call__(self, *args: Any, **kwargs: Any) -> None:
			_missing_tf()
