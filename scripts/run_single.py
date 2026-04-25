from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import DataPipelineConfig, prepare_datasets
from evaluation.evaluator import evaluate_model_with_outputs
from training.seed_utils import set_global_seed
from training.train import build_loss, build_model, resolve_loss_config
from training.trainer import Trainer
from utils.device import dataloader_pin_memory, resolve_device
from utils.logging import ensure_dir, save_json



def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stable_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _dataset_fingerprint(dataset_path: str) -> Dict[str, Any]:
    p = Path(dataset_path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()

    if not p.exists():
        return {
            "path": str(p),
            "exists": False,
            "size": -1,
            "mtime_ns": -1,
            "sha256": "",
        }

    stat = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(p),
    }


def _resolve_runs_root(runs_root: str) -> Path:
    p = Path(runs_root)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _run_dir_for(
    model: str,
    loss: str,
    dataset: str,
    seed: int,
    runs_root: str = "results/single_runs",
    include_seed_subdir: bool = True,
) -> Path:
    root = _resolve_runs_root(runs_root)
    dataset_name = Path(dataset).stem

    if include_seed_subdir:
        new_dir = root / model / dataset_name / loss / f"seed{seed}"
        legacy_dir = root / f"{model}_{loss}_{dataset_name}_seed{seed}"
    else:
        new_dir = root / model / dataset_name / loss
        legacy_dir = root / f"{model}_{loss}_{dataset_name}"

    if new_dir.exists():
        return new_dir
    if legacy_dir.exists():
        return legacy_dir
    return new_dir


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "model_last.pt",
        run_dir / "checkpoint.pt",  # legacy
    ]
    existing = [p for p in candidates if p.exists() and p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _count_csv_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if row and any((v or "").strip() != "" for v in row.values()):
                    count += 1
            return count
    except Exception:
        return 0


def _history_epoch_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return 0
        epoch_list = payload.get("epoch")
        if isinstance(epoch_list, list):
            return len(epoch_list)
        train_loss = payload.get("train_loss")
        if isinstance(train_loss, list):
            return len(train_loss)
        val_rmse = payload.get("val_rmse")
        if isinstance(val_rmse, list):
            return len(val_rmse)
    except Exception:
        return 0
    return 0


def _infer_resume_epoch(run_dir: Path, model_name: str, loss_name: str, seed: int) -> int:
    history_path = run_dir / "training_history.json"
    history_epochs = _history_epoch_count(history_path)

    train_log = run_dir / "training_log.csv"
    legacy_train_log = run_dir / f"{model_name}_{loss_name}_seed{seed}_training_log.csv"
    log_path = train_log if train_log.exists() else legacy_train_log
    log_epochs = _count_csv_data_rows(log_path)

    return max(history_epochs, log_epochs)


def _cleanup_for_restart(run_dir: Path) -> None:
    stale_paths = [
        run_dir / "metrics.json",
        run_dir / "predictions.csv",
        run_dir / "val_predictions.csv",
        run_dir / "model_final.pt",
        run_dir / "model_last.pt",
        run_dir / "checkpoint.pt",  # legacy
        run_dir / "model_last_best.pt",
        run_dir / "checkpoint_best.pt",  # legacy
        run_dir / "checkpoint_meta.json",
        run_dir / "training_history.json",
        run_dir / "training_log.csv",
        run_dir / "loss_curve.png",
        run_dir / "val_metrics.png",
    ]
    for p in stale_paths:
        p.unlink(missing_ok=True)


def _atomic_write_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _mirror_legacy_checkpoint_aliases(run_dir: Path) -> None:
    alias_pairs = [
        ("model_last.pt", "checkpoint.pt"),
        ("model_last_best.pt", "checkpoint_best.pt"),
    ]
    for src_name, dst_name in alias_pairs:
        src = run_dir / src_name
        dst = run_dir / dst_name
        if src.exists() and src.is_file() and src.stat().st_size > 0:
            shutil.copy2(src, dst)



def _plot_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def _plot_val_metrics(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["val_rmse"], marker="o")
    axes[0].set_title("Validation RMSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")

    axes[1].plot(history_df["epoch"], history_df["val_directional_accuracy"], marker="o")
    axes[1].set_title("Validation Directional Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Directional Accuracy (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def run_single_pipeline(
    config: Dict[str, Any],
    model_name: str,
    loss_name: str,
    dataset_path: str,
    seed: int,
    resume_mode: str = "auto",
    resume_epoch_hint: int = 0,
    expected_signature_hash: str = "",
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)

    cfg["dataset"]["path"] = dataset_path
    dataset_name = Path(dataset_path).stem
    cfg["dataset"]["processed_output_path"] = str(Path("data/processed") / f"{dataset_name}.pkl")

    paths_cfg = cfg.get("paths", {})
    runs_root = str(paths_cfg.get("runs_root", "results/single_runs"))
    include_seed_subdir = bool(paths_cfg.get("include_seed_subdir", True))

    run_dir = ensure_dir(
        _run_dir_for(
            model_name,
            loss_name,
            dataset_path,
            seed,
            runs_root=runs_root,
            include_seed_subdir=include_seed_subdir,
        )
    )
    
    if not run_dir.exists():
        raise RuntimeError(f"Failed to resolve and create run directory: {run_dir}")

    cfg["paths"]["logs_dir"] = str(run_dir)
    cfg["paths"]["plots_dir"] = str(run_dir)
    cfg["paths"]["checkpoints_dir"] = str(run_dir)

    set_global_seed(int(seed), deterministic=bool(cfg["reproducibility"].get("deterministic", True)))

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    device = resolve_device(train_cfg.get("device", "auto"))

    data_bundle = prepare_datasets(
        DataPipelineConfig(
            csv_path=dataset_cfg["path"],
            datetime_column=dataset_cfg.get("datetime_column", "datetime"),
            target_column=dataset_cfg.get("target_column", "close"),
            feature_columns=dataset_cfg.get("feature_columns", None),
            train_ratio=float(dataset_cfg.get("train_ratio", 0.7)),
            val_ratio=float(dataset_cfg.get("val_ratio", 0.15)),
            test_ratio=float(dataset_cfg.get("test_ratio", 0.15)),
            window_size=int(dataset_cfg.get("window_size", 48)),
            horizon=int(dataset_cfg.get("horizon", 1)),
            batch_size=int(train_cfg["batch_size"]),
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=dataloader_pin_memory(device),
            processed_output_path=dataset_cfg.get("processed_output_path"),
        )
    )

    model = build_model(
        model_name=model_name,
        n_features=int(data_bundle["n_features"]),
        window_size=int(data_bundle["window_size"]),
        horizon=int(dataset_cfg.get("horizon", 1)),
        model_cfg=cfg.get("model_configs", {}),
        feature_columns=data_bundle.get("feature_columns"),
        target_column=data_bundle.get("target_column"),
    )
    loss_fn = build_loss(loss_name, cfg.get("loss_configs", {}))

    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError("Controlled experiment constraint violation: optimizer must remain Adam.")
    optim_params = list(model.parameters())
    if str(loss_name).lower() in {
        "rnad",
        "rnad_full",
        "rnad_no_dir",
        "rnad_no_tail",
        "rnad_no_noise",
    }:
        loss_trainable_params = [p for p in loss_fn.parameters() if p.requires_grad]
        if loss_trainable_params:
            optim_params.extend(loss_trainable_params)

    optimizer = torch.optim.Adam(optim_params, lr=float(train_cfg["learning_rate"]))

    dataset_fp = _dataset_fingerprint(dataset_path)
    run_signature_payload = {
        "model": str(model_name),
        "loss": str(loss_name),
        "seed": int(seed),
        "dataset": str(Path(dataset_path)),
        "dataset_fingerprint": dataset_fp,
        "training": cfg.get("training", {}),
        "loss_configs": resolve_loss_config(loss_name, cfg.get("loss_configs", {})),
        "model_config": cfg.get("model_configs", {}).get(model_name, {}),
        "dataset_config": {
            k: v for k, v in cfg.get("dataset", {}).items() if k != "processed_output_path"
        },
    }
    run_signature_hash = _stable_hash(run_signature_payload)

    resolved_resume_mode = str(resume_mode or "auto").strip().lower()
    if resolved_resume_mode not in {"auto", "resumed", "restarted"}:
        raise ValueError(f"Invalid resume_mode '{resume_mode}'. Expected one of: auto, resumed, restarted.")

    if expected_signature_hash and expected_signature_hash != run_signature_hash:
        print(
            "[WARN] Expected run signature hash does not match current resolved signature. "
            "For safety, forcing restart mode."
        )
        resolved_resume_mode = "restarted"

    latest_checkpoint = _find_latest_checkpoint(run_dir)
    checkpoint_path = run_dir / "model_last.pt"
    checkpoint_meta_path = run_dir / "checkpoint_meta.json"
    history_path = run_dir / "training_history.json"

    if resolved_resume_mode == "restarted":
        _cleanup_for_restart(run_dir)
        latest_checkpoint = None
    elif latest_checkpoint is not None and latest_checkpoint.name != "model_last.pt":
        # Migrate legacy checkpoint naming without losing compatibility.
        shutil.copy2(latest_checkpoint, checkpoint_path)
        latest_checkpoint = checkpoint_path

    if latest_checkpoint is None:
        latest_checkpoint = _find_latest_checkpoint(run_dir)

    resume_cfg_flag = bool(train_cfg.get("resume", False))
    resume_flag = bool(resume_cfg_flag and latest_checkpoint is not None and resolved_resume_mode != "restarted")
    if resolved_resume_mode == "resumed" and latest_checkpoint is None:
        print("[WARN] Resume requested but no valid checkpoint was found. Restarting from scratch.")

    resume_status = "resumed" if resume_flag else "restarted"
    inferred_resume_epoch = _infer_resume_epoch(run_dir, model_name, loss_name, seed)
    effective_resume_epoch_hint = int(max(0, resume_epoch_hint, inferred_resume_epoch))

    run_name = f"{model_name}/{dataset_name}/{loss_name}/seed{seed}"

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
    train_out = trainer.fit(
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        epochs=int(train_cfg["epochs"]),
        target_scaler=data_bundle["target_scaler"],
        run_name=run_name,
        log_dir=str(run_dir),
        plot_dir=None,
        checkpoint_path=str(checkpoint_path),
        checkpoint_meta_path=str(checkpoint_meta_path),
        history_path=str(history_path),
        resume=resume_flag,
        resume_epoch_hint=effective_resume_epoch_hint,
        resume_status=resume_status,
        checkpoint_every=int(train_cfg.get("checkpoint_every", 1)),
        checkpoint_metadata={
            "model": model_name,
            "loss": loss_name,
            "seed": int(seed),
            "dataset": dataset_path,
            "run_signature_hash": run_signature_hash,
            "dataset_fingerprint": dataset_fp,
            "resume_mode": resolved_resume_mode,
            "resume_epoch_hint": int(effective_resume_epoch_hint),
        },
        use_tqdm=bool(train_cfg.get("progress_bar", True)),
    )

    _mirror_legacy_checkpoint_aliases(run_dir)

    eval_out = evaluate_model_with_outputs(
        model=model,
        dataloader=data_bundle["test_loader"],
        loss_fn=loss_fn,
        device=device,
        target_scaler=data_bundle["target_scaler"],
    )

    metrics = eval_out["metrics"]
    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]

    history = train_out["history"]
    history_df = pd.DataFrame(history)

    training_history_payload = {
        "epoch": [int(v) for v in history_df["epoch"].tolist()],
        "train_loss": [float(v) for v in history_df["train_loss"].tolist()],
        "val_loss": [float(v) for v in history_df["val_loss"].tolist()],
        "val_rmse": [float(v) for v in history_df["val_rmse"].tolist()],
        "val_mae": [float(v) for v in history_df["val_mae"].tolist()],
        "val_directional_accuracy": [float(v) for v in history_df["val_directional_accuracy"].tolist()],
    }

    # Best validation metrics from history
    best_val_idx = history_df["val_rmse"].idxmin()
    best_val_record = history[best_val_idx]

    # Get validation predictions with the best model (which is already loaded in 'model')
    val_out = evaluate_model_with_outputs(
        model=model,
        dataloader=data_bundle["val_loader"],
        loss_fn=loss_fn,
        device=device,
        target_scaler=data_bundle["target_scaler"],
    )

    metrics_payload = {
        "model": model_name,
        "loss": loss_name,
        "dataset": dataset_path,
        "seed": int(seed),
        "device": str(device),
        "val_rmse": float(best_val_record["val_rmse"]),
        "val_mae": float(best_val_record["val_mae"]),
        "val_directional_accuracy": float(best_val_record["val_directional_accuracy"]),
        "test_rmse": float(metrics["rmse"]),
        "test_mae": float(metrics["mae"]),
        "test_directional_accuracy": float(metrics["directional_accuracy"]),
        "run_signature_hash": run_signature_hash,
        "dataset_fingerprint": dataset_fp,
        "resume_status": str(train_out.get("resume_status", resume_status)),
        "resumed_from_epoch": int(train_out.get("resumed_from_epoch", 0)),
        "final_epoch": int(train_out.get("final_epoch", len(history))),
    }

    save_json(run_dir / "metrics.json", metrics_payload)
    save_json(run_dir / "training_history.json", training_history_payload)

    _plot_loss_curve(history_df, run_dir / "loss_curve.png")
    _plot_val_metrics(history_df, run_dir / "val_metrics.png")

    _atomic_write_dataframe_csv(pd.DataFrame({"y_true": y_true, "y_pred": y_pred}), run_dir / "predictions.csv")
    _atomic_write_dataframe_csv(
        pd.DataFrame({"y_true": val_out["y_true"], "y_pred": val_out["y_pred"]}),
        run_dir / "val_predictions.csv",
    )

    # Optional useful artifacts
    save_json(
        run_dir / "config_snapshot.json",
        {
            "saved_at": float(time.time()),
            "run_signature_hash": run_signature_hash,
            "run_signature_payload": run_signature_payload,
            "config": cfg,
        },
    )
    _atomic_torch_save(run_dir / "model_final.pt", {"model_state_dict": model.state_dict()})

    save_json(
        checkpoint_meta_path,
        {
            "last_epoch": int(train_out.get("final_epoch", len(history))),
            "timestamp": float(time.time()),
            "metadata": {
                "model": model_name,
                "loss": loss_name,
                "seed": int(seed),
                "dataset": dataset_path,
                "run_signature_hash": run_signature_hash,
                "dataset_fingerprint": dataset_fp,
            },
        },
    )

    print(
        f"Model={model_name} | Loss={loss_name} | RMSE={metrics_payload['test_rmse']:.6f} "
        f"| DA={metrics_payload['test_directional_accuracy']:.2f}%"
    )

    return {
        "run_dir": str(run_dir),
        "metrics": metrics_payload,
        "training_log_path": train_out["log_path"],
        "checkpoint_path": train_out.get("checkpoint_path"),
        "best_model_path": train_out.get("best_model_path"),
        "resume_status": str(train_out.get("resume_status", resume_status)),
        "resumed_from_epoch": int(train_out.get("resumed_from_epoch", 0)),
        "final_epoch": int(train_out.get("final_epoch", len(history))),
        "run_signature_hash": run_signature_hash,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run single model/loss/dataset training-evaluation pipeline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-mode", type=str, default="auto", choices=["auto", "resumed", "restarted"])
    parser.add_argument("--resume-epoch-hint", type=int, default=0)
    parser.add_argument("--expected-signature-hash", type=str, default="")
    args = parser.parse_args()

    base_cfg = _load_config(args.config)
    result = run_single_pipeline(
        config=base_cfg,
        model_name=args.model,
        loss_name=args.loss,
        dataset_path=args.dataset,
        seed=args.seed,
        resume_mode=args.resume_mode,
        resume_epoch_hint=int(args.resume_epoch_hint),
        expected_signature_hash=str(args.expected_signature_hash),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
