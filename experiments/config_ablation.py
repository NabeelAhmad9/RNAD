from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]

ABLATED_MODEL = "patchtst"
ABLATION_VARIANTS: Sequence[str] = (
    "RNAD_FULL",
    "RNAD_NO_DIR",
    "RNAD_NO_TAIL",
    "RNAD_NO_NOISE",
)

DEFAULT_ABLATION_DATASETS: Sequence[str] = (
    "data/raw/AUDUSD.csv",
    "data/raw/BTCUSD.csv",
    "data/raw/US500.csv",
    "data/raw/XAUUSD.csv",
)


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _normalize_dataset_entry(dataset_path: str | Path) -> str:
    p = _resolve_path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Ablation dataset not found: {p}")

    try:
        return str(p.relative_to(ROOT).as_posix())
    except ValueError:
        return str(p)


def default_ablation_datasets() -> List[str]:
    return [_normalize_dataset_entry(path) for path in DEFAULT_ABLATION_DATASETS]


def list_dataset_csvs(data_raw_dir: str = "data/raw") -> List[str]:
    data_dir = _resolve_path(data_raw_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    csvs = sorted(data_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not csvs:
        raise RuntimeError(f"No CSV files found in dataset directory: {data_dir}")

    rel_paths: List[str] = []
    for p in csvs:
        try:
            rel_paths.append(str(p.resolve().relative_to(ROOT).as_posix()))
        except ValueError:
            rel_paths.append(str(p.resolve()))
    return rel_paths


def load_base_config(config_path: str = "experiments/config.yaml") -> Dict[str, Any]:
    path = _resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_ablation_config(
    base_config: Dict[str, Any],
    fixed_seed: int | None = None,
    datasets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("experiment", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("training", {})

    selected_seed = int(
        fixed_seed
        if fixed_seed is not None
        else (cfg.get("experiment", {}).get("seeds", [42]) or [42])[0]
    )

    cfg["experiment"]["models"] = [ABLATED_MODEL]
    cfg["experiment"]["losses"] = list(ABLATION_VARIANTS)
    cfg["experiment"]["seeds"] = [selected_seed]
    cfg["experiment"]["datasets"] = (
        [_normalize_dataset_entry(path) for path in datasets]
        if datasets is not None
        else default_ablation_datasets()
    )

    # Ensure ablation runs preserve the standard resume/checkpoint behavior.
    cfg["training"]["resume"] = True
    cfg["training"]["checkpoint_every"] = int(cfg["training"].get("checkpoint_every", 1))

    # Route all run artifacts to ablation directories.
    cfg["paths"]["runs_root"] = "results/ablations"
    cfg["paths"]["include_seed_subdir"] = False

    return cfg


def save_config(config: Dict[str, Any], output_path: str | Path) -> Path:
    path = _resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path
