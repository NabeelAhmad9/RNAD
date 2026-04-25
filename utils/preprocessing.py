from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TECHNICAL_INDICATOR_COLUMNS = [
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "ema_9",
    "bb_middle",
    "bb_upper",
    "bb_lower",
]


@dataclass
class NumpyStandardScaler:
    """Simple standard scaler with numpy for reproducible preprocessing."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float64)
        return x * self.std_ + self.mean_


def _relative_strength_index(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder-style RSI using only present and historical observations."""

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    no_movement = avg_gain.eq(0.0) & avg_loss.eq(0.0)
    only_gains = avg_loss.eq(0.0) & avg_gain.gt(0.0)
    only_losses = avg_gain.eq(0.0) & avg_loss.gt(0.0)

    rsi = rsi.where(~no_movement, 50.0)
    rsi = rsi.where(~only_gains, 100.0)
    rsi = rsi.where(~only_losses, 0.0)

    return rsi


def compute_technical_indicators(
    df: pd.DataFrame,
    price_column: str = "close",
    *,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    ema_window: int = 9,
    bb_window: int = 20,
    bb_std_dev: float = 2.0,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Append common technical indicators to a time-series frame.

    The indicators are computed causally from the selected price column, so they
    only depend on present and past values. Warm-up rows naturally contain NaNs;
    by default these rows are removed after the indicator columns are created.
    """

    if price_column not in df.columns:
        raise ValueError(
            f"Price column '{price_column}' not found. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    price = pd.to_numeric(out[price_column], errors="coerce").astype(np.float64)

    ema_fast = price.ewm(span=macd_fast, adjust=False, min_periods=macd_fast).mean()
    ema_slow = price.ewm(span=macd_slow, adjust=False, min_periods=macd_slow).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(
        span=macd_signal, adjust=False, min_periods=macd_signal
    ).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["rsi_14"] = _relative_strength_index(price, period=rsi_period)
    out["ema_9"] = price.ewm(span=ema_window, adjust=False, min_periods=ema_window).mean()

    bb_rolling = price.rolling(window=bb_window, min_periods=bb_window)
    out["bb_middle"] = bb_rolling.mean()
    bb_std = bb_rolling.std()
    out["bb_upper"] = out["bb_middle"] + (bb_std_dev * bb_std)
    out["bb_lower"] = out["bb_middle"] - (bb_std_dev * bb_std)

    if drop_na:
        out = out.dropna(subset=TECHNICAL_INDICATOR_COLUMNS).copy()

    return out



def time_series_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    n = len(df)
    if n < 10:
        raise ValueError("Dataset is too small for reliable splitting.")

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the time-based splits is empty. Adjust split ratios.")

    return train_df, val_df, test_df



def build_sliding_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    horizon: int = 1,
) -> Dict[str, np.ndarray]:
    if window_size <= 1:
        raise ValueError("window_size must be > 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)

    max_target_idx = len(targets) - horizon
    if max_target_idx < window_size:
        raise ValueError(
            "Not enough samples to create windows. Increase data size or reduce window_size/horizon."
        )

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    y_prev_list: List[np.ndarray] = []
    target_indices: List[int] = []

    for end_idx in range(window_size, len(features) - horizon + 1):
        y_start = end_idx
        y_end = end_idx + horizon
        y_idx = y_end - 1

        y_seq = targets[y_start:y_end]
        y_prev_seq = np.empty_like(y_seq)
        y_prev_seq[0] = targets[y_start - 1]
        if horizon > 1:
            y_prev_seq[1:] = targets[y_start : y_end - 1]

        x_list.append(features[end_idx - window_size : end_idx])
        y_list.append(y_seq)
        y_prev_list.append(y_prev_seq)
        target_indices.append(y_idx)

    return {
        "x": np.asarray(x_list, dtype=np.float32),
        "y": np.asarray(y_list, dtype=np.float32),
        "y_prev": np.asarray(y_prev_list, dtype=np.float32),
        "target_indices": np.asarray(target_indices, dtype=np.int64),
    }



def split_scale_and_window(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    window_size: int,
    horizon: int = 1,
) -> Dict[str, object]:
    train_df, val_df, test_df = time_series_split(df, train_ratio, val_ratio, test_ratio)

    x_scaler = NumpyStandardScaler().fit(train_df[feature_columns].to_numpy(dtype=np.float64))
    y_scaler = NumpyStandardScaler().fit(train_df[[target_column]].to_numpy(dtype=np.float64))

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    windowed: Dict[str, Dict[str, np.ndarray]] = {}

    for split_name, split_df in split_frames.items():
        x_raw = split_df[feature_columns].to_numpy(dtype=np.float64)
        y_raw = split_df[target_column].to_numpy(dtype=np.float64).reshape(-1, 1)

        x_scaled = x_scaler.transform(x_raw).astype(np.float32)
        y_scaled = y_scaler.transform(y_raw).astype(np.float32).reshape(-1)

        split_windows = build_sliding_windows(
            features=x_scaled,
            targets=y_scaled,
            window_size=window_size,
            horizon=horizon,
        )

        target_indices = split_windows["target_indices"]
        raw_targets = y_raw.reshape(-1)
        y_raw_windows: List[np.ndarray] = []
        y_prev_raw_windows: List[np.ndarray] = []

        for y_last_idx in target_indices:
            y_start = int(y_last_idx) - horizon + 1
            y_end = int(y_last_idx) + 1

            y_seq_raw = raw_targets[y_start:y_end]
            y_prev_seq_raw = np.empty_like(y_seq_raw)
            y_prev_seq_raw[0] = raw_targets[y_start - 1]
            if horizon > 1:
                y_prev_seq_raw[1:] = raw_targets[y_start : y_end - 1]

            y_raw_windows.append(y_seq_raw)
            y_prev_raw_windows.append(y_prev_seq_raw)

        y_raw_windows_arr = np.asarray(y_raw_windows, dtype=np.float32)
        y_prev_raw_windows_arr = np.asarray(y_prev_raw_windows, dtype=np.float32)

        split_windows["y_raw"] = y_raw_windows_arr
        split_windows["y_prev_raw"] = y_prev_raw_windows_arr
        split_windows["timestamps"] = split_df.index.to_numpy()[target_indices]

        windowed[split_name] = split_windows

    return {
        "windowed": windowed,
        "feature_scaler": x_scaler,
        "target_scaler": y_scaler,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }



def save_processed_pickle(path: str | Path, payload: Dict[str, object]) -> None:
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(payload, f)
