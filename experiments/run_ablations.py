from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from experiments.run_experiment import aggregate_results
from scripts.run_all import run_all
from utils.logging import ensure_dir, save_json, save_rows_csv

REQUIRED_ARTIFACTS: Sequence[str] = (
    "metrics.json",
    "predictions.csv",
    "val_predictions.csv",
    "model_final.pt",
    "model_last.pt",
    "model_last_best.pt",
    "checkpoint.pt",
    "checkpoint_best.pt",
    "checkpoint_meta.json",
    "training_history.json",
    "training_log.csv",
    "loss_curve.png",
    "val_metrics.png",
)


def _resolve_runs_root(runs_root: str) -> Path:
    p = Path(runs_root)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _run_dir_for(runs_root: Path, model: str, dataset: str, loss: str) -> Path:
    dataset_name = Path(dataset).stem
    return runs_root / model / dataset_name / loss


def _missing_artifacts(run_dir: Path) -> List[str]:
    missing: List[str] = []
    for filename in REQUIRED_ARTIFACTS:
        path = run_dir / filename
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def _collect_per_run_rows(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    runs_root = _resolve_runs_root(str(config.get("paths", {}).get("runs_root", "results/ablations")))
    datasets = list(config.get("experiment", {}).get("datasets", []))
    models = list(config.get("experiment", {}).get("models", [ABLATED_MODEL]))
    losses = list(config.get("experiment", {}).get("losses", list(ABLATION_VARIANTS)))
    seeds = list(config.get("experiment", {}).get("seeds", [42]))
    seed_value = int(seeds[0]) if seeds else 42

    rows: List[Dict[str, Any]] = []
    missing_reports: List[Dict[str, Any]] = []

    for dataset in datasets:
        dataset_tag = Path(dataset).stem
        for model in models:
            for loss in losses:
                run_dir = _run_dir_for(runs_root=runs_root, model=model, dataset=dataset, loss=loss)
                missing = _missing_artifacts(run_dir)
                if missing:
                    missing_reports.append(
                        {
                            "dataset": dataset_tag,
                            "model": model,
                            "loss": loss,
                            "run_dir": str(run_dir),
                            "missing_artifacts": missing,
                        }
                    )
                    continue

                metrics_path = run_dir / "metrics.json"
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics_payload = json.load(f)

                rows.append(
                    {
                        "run_name": f"{model}/{dataset_tag}/{loss}",
                        "dataset": dataset_tag,
                        "model": model,
                        "loss": loss,
                        "seed": int(metrics_payload.get("seed", seed_value)),
                        "test_rmse": float(metrics_payload["test_rmse"]),
                        "test_directional_accuracy": float(metrics_payload["test_directional_accuracy"]),
                        "test_loss": float(metrics_payload.get("test_loss", metrics_payload.get("test_rmse", 0.0))),
                        "val_rmse": float(metrics_payload.get("val_rmse", 0.0)),
                        "val_mae": float(metrics_payload.get("val_mae", 0.0)),
                        "val_directional_accuracy": float(metrics_payload.get("val_directional_accuracy", 0.0)),
                    }
                )

    return rows, missing_reports


def run_all_ablations(
    base_config_path: str = "experiments/config.yaml",
    workers: int = 3,
    force_rerun: bool = False,
    dry_run: bool = False,
    fixed_seed: int | None = None,
    datasets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if int(workers) != 3:
        raise ValueError("Ablation runner requires exactly 3 workers.")

    base_cfg = load_base_config(base_config_path)
    ablation_cfg = build_ablation_config(base_cfg, fixed_seed=fixed_seed, datasets=datasets)

    runtime_cfg_path = save_config(ablation_cfg, ROOT / "results" / "logs" / "run_all_ablations_runtime_config.yaml")

    run_summary = run_all(
        config_path=str(runtime_cfg_path),
        workers=3,
        force_rerun=bool(force_rerun),
        dry_run=bool(dry_run),
    )

    summary_payload: Dict[str, Any] = {
        "workers": 3,
        "runtime_config": str(runtime_cfg_path),
        "ablation_model": ABLATED_MODEL,
        "ablation_variants": list(ABLATION_VARIANTS),
        "datasets": list(ablation_cfg.get("experiment", {}).get("datasets", [])),
        "run_all": run_summary,
    }

    runs_root = _resolve_runs_root(str(ablation_cfg.get("paths", {}).get("runs_root", "results/ablations")))
    ensure_dir(runs_root)

    if dry_run:
        save_json(runs_root / "ablation_summary.json", summary_payload)
        return summary_payload

    if int(run_summary.get("failed", 0)) > 0 or int(run_summary.get("locked", 0)) > 0:
        summary_payload["status"] = "incomplete"
        save_json(runs_root / "ablation_summary.json", summary_payload)
        raise RuntimeError(
            "Ablation run finished with failures/locks. Inspect run_all summary and worker logs for details."
        )

    rows, missing_reports = _collect_per_run_rows(ablation_cfg)
    if missing_reports:
        summary_payload["status"] = "artifact_missing"
        summary_payload["missing_reports"] = missing_reports
        save_json(runs_root / "ablation_summary.json", summary_payload)
        raise RuntimeError(
            "Some ablation runs are missing required artifacts. "
            f"First missing report: {missing_reports[0]}"
        )

    if not rows:
        summary_payload["status"] = "no_rows"
        save_json(runs_root / "ablation_summary.json", summary_payload)
        raise RuntimeError("No ablation metric rows were collected.")

    aggregates = aggregate_results(rows)

    save_rows_csv(runs_root / "ablation_per_run_results.csv", rows)
    save_json(runs_root / "ablation_per_run_results.json", {"results": rows})
    save_rows_csv(runs_root / "ablation_aggregated_results.csv", aggregates)
    save_json(runs_root / "ablation_aggregated_results.json", {"aggregated": aggregates})

    summary_payload.update(
        {
            "status": "complete",
            "runs_expected": len(ablation_cfg.get("experiment", {}).get("datasets", [])) * len(ABLATION_VARIANTS),
            "runs_collected": len(rows),
            "per_run_csv": str(runs_root / "ablation_per_run_results.csv"),
            "per_run_json": str(runs_root / "ablation_per_run_results.json"),
            "aggregated_csv": str(runs_root / "ablation_aggregated_results.csv"),
            "aggregated_json": str(runs_root / "ablation_aggregated_results.json"),
        }
    )
    save_json(runs_root / "ablation_summary.json", summary_payload)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic RNAD ablations with fixed hyperparameters.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    summary = run_all_ablations(
        base_config_path=args.config,
        workers=3,
        force_rerun=bool(args.force_rerun),
        dry_run=bool(args.dry_run),
        fixed_seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
