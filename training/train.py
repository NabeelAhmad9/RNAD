from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import DataPipelineConfig, default_processed_output, prepare_datasets
from evaluation.evaluator import evaluate_model
from losses.gmadl import GMADL
from losses.huber import HuberLossModule
from losses.log_cosh import LogCoshLoss
from losses.mse import MSELossModule
from losses.rnad import RNAD as RNADBase
from losses.rnad_ablation import RNAD as RNADAblation
from models.itransformer import iTransformerForecaster
from models.nhits import NHiTSForecaster
from models.patchtst import PatchTST
from models.timexer import TimeXerForecaster
from training.seed_utils import set_global_seed
from training.trainer import Trainer
from utils.device import dataloader_pin_memory, resolve_device



def _resolve_target_idx(
    cfg: Dict[str, Any],
    feature_columns: Optional[List[str]],
    target_column: Optional[str],
    model_key: str,
) -> int:
    if "target_idx" in cfg:
        return int(cfg["target_idx"])

    if feature_columns and target_column and target_column in feature_columns:
        return int(feature_columns.index(target_column))

    raise ValueError(
        f"{model_key} requires the forecast target to map to an input channel. "
        f"Set model_configs.{model_key}.target_idx explicitly, or include target_column in feature_columns."
    )


def _int_list(values: Any) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(v) for v in values]


def _optional_int_list(values: Any) -> Optional[List[Optional[int]]]:
    if values is None:
        return None
    return [None if v is None else int(v) for v in values]


def resolve_loss_config(
    loss_name: str,
    loss_cfg: Dict[str, Any],
    fallback_key: str = "rnad",
) -> Dict[str, Any]:
    stripped_name = str(loss_name).strip()
    normalized_name = stripped_name.lower()

    candidate_keys = []
    for candidate in (stripped_name, normalized_name):
        if candidate and candidate not in candidate_keys:
            candidate_keys.append(candidate)

    if normalized_name != fallback_key and normalized_name.startswith(f"{fallback_key}_"):
        candidate_keys.append(fallback_key)

    for candidate in candidate_keys:
        candidate_cfg = loss_cfg.get(candidate)
        if isinstance(candidate_cfg, dict):
            return candidate_cfg

    return {}


def build_model(
    model_name: str,
    n_features: int,
    window_size: int,
    horizon: int,
    model_cfg: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> torch.nn.Module:
    model_name = model_name.lower()

    if model_name == "patchtst":
        cfg = model_cfg.get("patchtst", {})
        target_idx = _resolve_target_idx(
            cfg=cfg,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key="patchtst",
        )

        # Keep backward compatibility for prior config keys while prioritizing
        # official PatchTST argument names.
        e_layers = int(cfg.get("e_layers", cfg.get("num_layers", 2)))
        d_ff = int(cfg.get("d_ff", cfg.get("ff_dim", 128)))

        return PatchTST(
            input_size=n_features,
            window_size=window_size,
            horizon=horizon,
            patch_len=int(cfg.get("patch_len", 16)),
            stride=int(cfg.get("stride", 8)),
            d_model=int(cfg.get("d_model", 64)),
            n_heads=int(cfg.get("n_heads", 4)),
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=float(cfg.get("dropout", 0.1)),
            fc_dropout=float(cfg.get("fc_dropout", 0.0)),
            head_dropout=float(cfg.get("head_dropout", 0.0)),
            individual=bool(cfg.get("individual", False)),
            padding_patch=cfg.get("padding_patch", None),
            revin=bool(cfg.get("revin", True)),
            affine=bool(cfg.get("affine", True)),
            subtract_last=bool(cfg.get("subtract_last", False)),
            decomposition=bool(cfg.get("decomposition", False)),
            kernel_size=int(cfg.get("kernel_size", 25)),
            target_idx=target_idx,
            max_seq_len=int(cfg.get("max_seq_len", 1024)),
            d_k=cfg.get("d_k", None),
            d_v=cfg.get("d_v", None),
            norm=str(cfg.get("norm", "BatchNorm")),
            attn_dropout=float(cfg.get("attn_dropout", 0.0)),
            act=str(cfg.get("act", "gelu")),
            key_padding_mask=cfg.get("key_padding_mask", "auto"),
            padding_var=cfg.get("padding_var", None),
            attn_mask=cfg.get("attn_mask", None),
            res_attention=bool(cfg.get("res_attention", True)),
            pre_norm=bool(cfg.get("pre_norm", False)),
            store_attn=bool(cfg.get("store_attn", False)),
            pe=str(cfg.get("pe", "zeros")),
            learn_pe=bool(cfg.get("learn_pe", True)),
            pretrain_head=bool(cfg.get("pretrain_head", False)),
            head_type=str(cfg.get("head_type", "flatten")),
            verbose=bool(cfg.get("verbose", False)),
        )

    if model_name == "itransformer":
        cfg = model_cfg.get("itransformer", {})
        target_idx = _resolve_target_idx(
            cfg=cfg,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key="itransformer",
        )

        # Keep backward compatibility for legacy naming conventions.
        e_layers = int(cfg.get("e_layers", cfg.get("num_layers", 2)))
        d_ff = int(cfg.get("d_ff", cfg.get("ff_dim", 256)))

        return iTransformerForecaster(
            input_size=n_features,
            window_size=window_size,
            horizon=horizon,
            d_model=int(cfg.get("d_model", 128)),
            d_ff=d_ff,
            e_layers=e_layers,
            n_heads=int(cfg.get("n_heads", 8)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "gelu")),
            target_idx=target_idx,
        )

    if model_name == "nhits":
        cfg = model_cfg.get("nhits", {})
        target_idx = _resolve_target_idx(
            cfg=cfg,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key="nhits",
        )

        return NHiTSForecaster(
            input_size=n_features,
            window_size=window_size,
            horizon=horizon,
            n_stacks=int(cfg.get("n_stacks", 3)),
            n_blocks=int(cfg.get("n_blocks", 1)),
            n_layers=int(cfg.get("n_layers", 2)),
            n_hidden=int(cfg.get("n_hidden", 512)),
            n_pool_kernel_size=_int_list(cfg.get("n_pool_kernel_size", [2, 2, 2])),
            n_freq_downsample=_int_list(cfg.get("n_freq_downsample", [4, 2, 1])),
            pooling_mode=str(cfg.get("pooling_mode", "max")),
            interpolation_mode=str(cfg.get("interpolation_mode", "linear")),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "ReLU")),
            batch_normalization=bool(cfg.get("batch_normalization", False)),
            shared_weights=bool(cfg.get("shared_weights", False)),
            target_idx=target_idx,
        )

    if model_name == "timexer":
        cfg = model_cfg.get("timexer", {})
        target_idx = _resolve_target_idx(
            cfg=cfg,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key="timexer",
        )

        return TimeXerForecaster(
            input_size=n_features,
            window_size=window_size,
            horizon=horizon,
            patch_len=int(cfg.get("patch_len", 16)),
            d_model=int(cfg.get("d_model", 128)),
            d_ff=int(cfg.get("d_ff", 256)),
            e_layers=int(cfg.get("e_layers", 2)),
            n_heads=int(cfg.get("n_heads", 8)),
            dropout=float(cfg.get("dropout", 0.1)),
            factor=int(cfg.get("factor", 5)),
            activation=str(cfg.get("activation", "gelu")),
            use_norm=bool(cfg.get("use_norm", True)),
            target_idx=target_idx,
        )

    raise ValueError(f"Unknown model: {model_name}")



def build_loss(loss_name: str, loss_cfg: Dict[str, Any]) -> torch.nn.Module:
    key = str(loss_name).strip().lower()

    if key == "mse":
        return MSELossModule()
    if key == "huber":
        return HuberLossModule(delta=float(loss_cfg.get("huber", {}).get("delta", 1.0)))
    if key == "log_cosh":
        return LogCoshLoss()
    if key == "rnad":
        cfg = resolve_loss_config(loss_name, loss_cfg)
        return RNADBase(
            lambda_dir=float(cfg.get("lambda_dir", 0.5)),
            tail_tau=float(cfg.get("tail_tau", cfg.get("tau", 1.1))),
            tail_temp=float(cfg.get("tail_temp", 0.30)),
            charb_eps=float(cfg.get("charb_eps", 1e-3)),
            beta=float(cfg.get("beta", 4.5)),
            kappa=float(cfg.get("kappa", 4.5)),
            noise_level=float(cfg.get("noise_level", 0.10)),
            noise_temp=float(cfg.get("noise_temp", 0.04)),
            vol_momentum=float(cfg.get("vol_momentum", 0.985)),
            eps=float(cfg.get("eps", 1e-8)),
        )
    if key in {
        "rnad_full",
        "rnad_no_dir",
        "rnad_no_tail",
        "rnad_no_noise",
    }:
        cfg = resolve_loss_config(loss_name, loss_cfg)
        flags_map = {
            "rnad_full": {
                "use_direction": True,
                "use_tail": True,
                "use_noise_gate": True,
            },
            "rnad_no_dir": {
                "use_direction": False,
                "use_tail": True,
                "use_noise_gate": True,
            },
            "rnad_no_tail": {
                "use_direction": True,
                "use_tail": False,
                "use_noise_gate": True,
            },
            "rnad_no_noise": {
                "use_direction": True,
                "use_tail": True,
                "use_noise_gate": False,
            },
        }
        return RNADAblation(
            **flags_map[key],
            lambda_dir=float(cfg.get("lambda_dir", 0.5)),
            tail_tau=float(cfg.get("tail_tau", cfg.get("tau", 1.1))),
            tail_temp=float(cfg.get("tail_temp", 0.30)),
            charb_eps=float(cfg.get("charb_eps", 1e-3)),
            beta=float(cfg.get("beta", 4.5)),
            kappa=float(cfg.get("kappa", 4.5)),
            noise_level=float(cfg.get("noise_level", 0.10)),
            noise_temp=float(cfg.get("noise_temp", 0.04)),
            vol_momentum=float(cfg.get("vol_momentum", 0.985)),
            eps=float(cfg.get("eps", 1e-8)),
        )
    if key == "gmadl":
        cfg = loss_cfg.get("gmadl", {})
        return GMADL(alpha=float(cfg.get("alpha", 0.5)), beta=float(cfg.get("beta", 5.0)))

    raise ValueError(f"Unknown loss function: {loss_name}")



def run_single_experiment(
    config: Dict[str, Any],
    model_name: str,
    loss_name: str,
    seed: int,
    resume: bool | None = None,
) -> Dict[str, Any]:
    set_global_seed(int(seed), deterministic=bool(config["reproducibility"].get("deterministic", True)))

    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    device = resolve_device(train_cfg.get("device", "auto"))

    processed_output = dataset_cfg.get("processed_output_path") or default_processed_output(
        dataset_cfg["path"], processed_dir="data/processed"
    )

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
            processed_output_path=processed_output,
        )
    )

    model = build_model(
        model_name=model_name,
        n_features=int(data_bundle["n_features"]),
        window_size=int(data_bundle["window_size"]),
        horizon=int(dataset_cfg.get("horizon", 1)),
        model_cfg=config.get("model_configs", {}),
        feature_columns=data_bundle.get("feature_columns"),
        target_column=data_bundle.get("target_column"),
    )

    loss_fn = build_loss(loss_name, config.get("loss_configs", {}))

    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(
            "Controlled experiment constraint violation: optimizer must remain Adam across runs."
        )

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

    dataset_path = dataset_cfg.get("path", "")
    dataset_name = Path(dataset_path).stem if dataset_path else "unknown_dataset"
    run_name = f"{model_name}/{dataset_name}/{loss_name}/seed{seed}"

    # Overwrite the directories to be exactly this structure
    base_dir = Path("results/single_runs") / model_name / dataset_name / loss_name / f"seed{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanity check validation
    if not base_dir.exists():
        raise RuntimeError(f"Failed to create run directory: {base_dir}")
        
    checkpoints_dir = str(base_dir)
    config.setdefault("paths", {})
    config["paths"]["logs_dir"] = str(base_dir)
    config["paths"]["plots_dir"] = str(base_dir)
    config["paths"]["checkpoints_dir"] = str(base_dir)
    
    resume_flag = bool(train_cfg.get("resume", False)) if resume is None else bool(resume)
    use_tqdm = bool(train_cfg.get("progress_bar", True))

    checkpoint_path = base_dir / "checkpoint.pt"

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
    train_out = trainer.fit(
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        epochs=int(train_cfg["epochs"]),
        target_scaler=data_bundle["target_scaler"],
        run_name=run_name,
        log_dir=config["paths"]["logs_dir"],
        plot_dir=config["paths"].get("plots_dir", None),
        checkpoint_path=str(checkpoint_path),
        resume=resume_flag,
        checkpoint_metadata={
            "model": model_name,
            "loss": loss_name,
            "seed": int(seed),
            "dataset": dataset_cfg.get("path"),
            "learning_rate": float(train_cfg["learning_rate"]),
            "epochs": int(train_cfg["epochs"]),
            "batch_size": int(train_cfg["batch_size"]),
        },
        use_tqdm=use_tqdm,
    )

    test_metrics = evaluate_model(
        model=model,
        dataloader=data_bundle["test_loader"],
        loss_fn=loss_fn,
        device=device,
        target_scaler=data_bundle["target_scaler"],
    )

    return {
        "run_name": run_name,
        "model": model_name,
        "loss": loss_name,
        "seed": int(seed),
        "test_rmse": float(test_metrics["rmse"]),
        "test_directional_accuracy": float(test_metrics["directional_accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "best_val_rmse": float(train_out["best_val_rmse"]),
        "training_log_path": str(train_out["log_path"]),
        "n_features": int(data_bundle["n_features"]),
        "window_size": int(data_bundle["window_size"]),
        "epochs": int(train_cfg["epochs"]),
        "batch_size": int(train_cfg["batch_size"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "device": str(device),
        "resume_enabled": resume_flag,
        "resumed_from_epoch": int(train_out.get("resumed_from_epoch", 0)),
        "checkpoint_path": train_out.get("checkpoint_path"),
        "best_model_path": train_out.get("best_model_path"),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single controlled model/loss/seed experiment.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = run_single_experiment(config, args.model, args.loss, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
