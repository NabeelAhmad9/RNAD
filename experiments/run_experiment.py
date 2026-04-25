from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import run_single_experiment
from utils.logging import ensure_dir, save_json, save_rows_csv



def aggregate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (row["dataset"], row["model"], row["loss"])
        grouped[key].append(row)

    aggregates: List[Dict[str, Any]] = []
    for (dataset, model, loss), items in grouped.items():
        rmse_vals = np.array([r["test_rmse"] for r in items], dtype=np.float64)
        da_vals = np.array([r["test_directional_accuracy"] for r in items], dtype=np.float64)

        aggregates.append(
            {
                "dataset": dataset,
                "model": model,
                "loss": loss,
                "runs": len(items),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals, ddof=0)),
                "directional_accuracy_mean": float(np.mean(da_vals)),
                "directional_accuracy_std": float(np.std(da_vals, ddof=0)),
            }
        )

    return sorted(aggregates, key=lambda x: (x["dataset"], x["model"], x["loss"]))



def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    exp_cfg = config["experiment"]
    seeds = exp_cfg.get("seeds", [42])
    datasets = exp_cfg.get("datasets", [config["dataset"]["path"]])
    models = exp_cfg["models"]
    losses = exp_cfg["losses"]

    results_rows: List[Dict[str, Any]] = []

    total_runs = len(datasets) * len(models) * len(losses) * len(seeds)
    run_count = 0

    for dataset_path in datasets:
        if not Path(dataset_path).exists():
            print(f"[SKIP] Dataset not found: {dataset_path}")
            continue

        dataset_tag = Path(dataset_path).stem

        for model_name in models:
            for loss_name in losses:
                for seed in seeds:
                    run_count += 1
                    print(
                        f"[{run_count}/{total_runs}] Running model={model_name}, loss={loss_name}, seed={seed}, dataset={dataset_path}"
                    )

                    run_cfg = copy.deepcopy(config)
                    run_cfg["dataset"]["path"] = dataset_path
                    run_cfg["dataset"]["processed_output_path"] = str(
                        Path("data/processed") / f"{dataset_tag}.pkl"
                    )

                    # Keep per-dataset artifacts isolated.
                    run_cfg.setdefault("paths", {})
                    # Ensure base path respects the new nested structure conceptually, 
                    # but actually run_single_experiment will overwrite logs_dir/plots_dir anyway.
                    # We can just pass the base directories.
                    run_cfg["paths"]["logs_dir"] = config["paths"].get("logs_dir", "results/logs")
                    run_cfg["paths"]["plots_dir"] = config["paths"].get("plots_dir", "results/plots")
                    run_cfg["paths"]["checkpoints_dir"] = config["paths"].get("checkpoints_dir", "results/checkpoints")

                    result = run_single_experiment(
                        config=run_cfg,
                        model_name=model_name,
                        loss_name=loss_name,
                        seed=int(seed),
                        resume=bool(run_cfg.get("training", {}).get("resume", False)),
                    )
                    result["dataset"] = dataset_tag
                    results_rows.append(result)

    if not results_rows:
        raise RuntimeError("No experiments were executed. Check dataset paths and config.")

    aggregates = aggregate_results(results_rows)

    metrics_dir = ensure_dir(config["paths"]["metrics_dir"])

    per_run_csv = metrics_dir / "per_run_results.csv"
    per_run_json = metrics_dir / "per_run_results.json"
    agg_csv = metrics_dir / "aggregated_results.csv"
    agg_json = metrics_dir / "aggregated_results.json"

    save_rows_csv(per_run_csv, results_rows)
    save_json(per_run_json, {"results": results_rows})

    save_rows_csv(agg_csv, aggregates)
    save_json(agg_json, {"aggregated": aggregates})

    return {
        "runs_executed": len(results_rows),
        "per_run_csv": str(per_run_csv),
        "per_run_json": str(per_run_json),
        "aggregated_csv": str(agg_csv),
        "aggregated_json": str(agg_json),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled loss-function experiments.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
