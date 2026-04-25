from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.preprocessing import (
    TECHNICAL_INDICATOR_COLUMNS,
    compute_technical_indicators,
    save_processed_pickle,
    split_scale_and_window,
)


class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, split_payload: Dict[str, object]) -> None:
        self.x = torch.tensor(split_payload["x"], dtype=torch.float32)
        self.y = torch.tensor(split_payload["y"], dtype=torch.float32)
        self.y_prev = torch.tensor(split_payload["y_prev"], dtype=torch.float32)
        self.y_raw = torch.tensor(split_payload["y_raw"], dtype=torch.float32)
        self.y_prev_raw = torch.tensor(split_payload["y_prev_raw"], dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "y_prev": self.y_prev[idx],
            "y_raw": self.y_raw[idx],
            "y_prev_raw": self.y_prev_raw[idx],
        }


@dataclass
class DataPipelineConfig:
    csv_path: str
    datetime_column: str
    target_column: str
    feature_columns: List[str] | None
    train_ratio: float
    val_ratio: float
    test_ratio: float
    window_size: int
    horizon: int
    batch_size: int
    num_workers: int = 0
    pin_memory: bool | None = None
    processed_output_path: str | None = None


def _append_technical_indicators(feature_columns: List[str], df: pd.DataFrame) -> List[str]:
    ordered_columns = list(dict.fromkeys(feature_columns))
    for column in TECHNICAL_INDICATOR_COLUMNS:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)
    return ordered_columns



def load_csv_time_series(csv_path: str, datetime_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if datetime_column not in df.columns:
        raise ValueError(
            f"Datetime column '{datetime_column}' not found in CSV. Available columns: {list(df.columns)}"
        )

    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
    df = df.dropna(subset=[datetime_column]).sort_values(datetime_column)
    df = df.set_index(datetime_column)
    df = df.dropna()

    if len(df) < 100:
        raise ValueError("Dataset is too small after cleaning for robust experimentation.")

    return df



def infer_feature_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    feature_columns = [
        c
        for c in df.columns
        if c != target_column and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns available after excluding target column.")
    return feature_columns



def prepare_datasets(config: DataPipelineConfig) -> Dict[str, object]:
    df = load_csv_time_series(config.csv_path, config.datetime_column)

    if config.target_column not in df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' not found. Available: {list(df.columns)}"
        )

    price_column = "close" if "close" in df.columns else config.target_column
    df = compute_technical_indicators(df, price_column=price_column)

    feature_columns = config.feature_columns or infer_feature_columns(df, config.target_column)
    if config.feature_columns is not None:
        feature_columns = _append_technical_indicators(feature_columns, df)

    pipeline = split_scale_and_window(
        df=df,
        feature_columns=feature_columns,
        target_column=config.target_column,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        window_size=config.window_size,
        horizon=config.horizon,
    )

    if config.processed_output_path:
        save_processed_pickle(
            config.processed_output_path,
            {
                "processed_dataframe": df,
                "windowed": pipeline["windowed"],
                "feature_columns": feature_columns,
                "target_column": config.target_column,
                "feature_scaler_mean": pipeline["feature_scaler"].mean_.tolist(),
                "feature_scaler_std": pipeline["feature_scaler"].std_.tolist(),
                "target_scaler_mean": pipeline["target_scaler"].mean_.tolist(),
                "target_scaler_std": pipeline["target_scaler"].std_.tolist(),
            },
        )

    train_ds = WindowedTimeSeriesDataset(pipeline["windowed"]["train"])
    val_ds = WindowedTimeSeriesDataset(pipeline["windowed"]["val"])
    test_ds = WindowedTimeSeriesDataset(pipeline["windowed"]["test"])

    pin_memory = bool(config.pin_memory) if config.pin_memory is not None else False

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "feature_columns": feature_columns,
        "target_column": config.target_column,
        "target_scaler": pipeline["target_scaler"],
        "window_size": config.window_size,
        "n_features": len(feature_columns),
    }



def default_processed_output(csv_path: str, processed_dir: str = "data/processed") -> str:
    src = Path(csv_path)
    stem = src.stem
    out_path = Path(processed_dir) / f"{stem}.pkl"
    return str(out_path)
