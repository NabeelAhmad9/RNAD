from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config_ablation import (
    ABLATED_MODEL,
    ABLATION_VARIANTS,
    build_ablation_config,
    load_base_config,
    save_config,
)
from scripts.run_single import run_single_pipeline


def _normalize_dataset_path(dataset: str) -> str:
    value = str(dataset).strip()
    p = Path(value)

    if p.suffix.lower() != ".csv":
        p = Path("data/raw") / f"{p.stem}.csv"

    if not p.is_absolute():
        abs_path = (ROOT / p).resolve()
        try:
            return str(abs_path.relative_to(ROOT).as_posix())
        except ValueError:
            return str(abs_path)

    return str(p)


def _normalize_variant(variant: str) -> str:
    normalized = str(variant).strip().upper()
    if normalized not in set(ABLATION_VARIANTS):
        raise ValueError(
            f"Invalid variant '{variant}'. Expected one of: {list(ABLATION_VARIANTS)}"
        )
    return normalized


def run_single_ablation(
    base_config_path: str,
    dataset: str,
    variant: str,
    seed: int | None = None,
    resume_mode: str = "auto",
) -> Dict[str, Any]:
    dataset_path = _normalize_dataset_path(dataset)
    variant_name = _normalize_variant(variant)

    base_cfg = load_base_config(base_config_path)
    ablation_cfg = build_ablation_config(
        base_config=base_cfg,
        fixed_seed=seed,
        datasets=[dataset_path],
    )
    selected_seed = int((ablation_cfg.get("experiment", {}).get("seeds", [42]) or [42])[0])

    runtime_cfg_path = save_config(
        ablation_cfg,
        ROOT / "results" / "logs" / "run_ablation_runtime_config.yaml",
    )

    result = run_single_pipeline(
        config=ablation_cfg,
        model_name=ABLATED_MODEL,
        loss_name=variant_name,
        dataset_path=dataset_path,
        seed=selected_seed,
        resume_mode=resume_mode,
    )

    return {
        "runtime_config": str(runtime_cfg_path),
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single deterministic RNAD ablation job.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file path or dataset stem.")
    parser.add_argument("--variant", type=str, required=True, choices=list(ABLATION_VARIANTS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="auto",
        choices=["auto", "resumed", "restarted"],
    )
    args = parser.parse_args()

    out = run_single_ablation(
        base_config_path=args.config,
        dataset=args.dataset,
        variant=args.variant,
        seed=args.seed,
        resume_mode=args.resume_mode,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
