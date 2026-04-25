from __future__ import annotations

import copy
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from evaluation.evaluator import evaluate_model
from utils.device import assert_same_device, transfer_kwargs
from utils.logging import save_json, save_rows_csv


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self._to_kwargs = transfer_kwargs(device)

        self.model.to(self.device)

    @staticmethod
    def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp_name)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _history_payload(history: List[Dict[str, float]]) -> Dict[str, List[float]]:
        epochs = [int(h.get("epoch", 0)) for h in history]
        train_loss = [float(h.get("train_loss", 0.0)) for h in history]
        val_loss = [float(h.get("val_loss", 0.0)) for h in history]
        val_rmse = [float(h.get("val_rmse", 0.0)) for h in history]
        val_mae = [float(h.get("val_mae", 0.0)) for h in history]
        val_da = [float(h.get("val_directional_accuracy", 0.0)) for h in history]
        return {
            "epoch": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_directional_accuracy": val_da,
        }

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        def _move_value(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {k: _move_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_move_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_move_value(v) for v in value)
            return value

        for state in optimizer.state.values():
            for key, value in list(state.items()):
                state[key] = _move_value(value)

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_meta_path: Path | None,
        epoch: int,
        best_val_rmse: float,
        history: List[Dict[str, float]],
        checkpoint_metadata: Dict[str, Any] | None,
        best_model_state_dict: Dict[str, Any] | None,
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": int(epoch),
            "best_val_rmse": float(best_val_rmse),
            "history": history,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": checkpoint_metadata or {},
        }
        if best_model_state_dict is not None:
            payload["best_model_state_dict"] = best_model_state_dict

        self._atomic_torch_save(checkpoint_path, payload)

        if checkpoint_meta_path is not None:
            meta_payload = {
                "last_epoch": int(epoch),
                "timestamp": float(time.time()),
                "best_val_rmse": float(best_val_rmse),
                "metadata": checkpoint_metadata or {},
            }
            save_json(checkpoint_meta_path, meta_payload)

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Dict[str, Any] | None:
        if not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
                self._move_optimizer_state_to_device(self.optimizer, self.device)

            return checkpoint
        except Exception as exc:
            print(
                f"[WARN] Could not load checkpoint '{checkpoint_path}': {exc}. "
                "Starting fresh training."
            )
            return None

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        total_epochs: int,
        use_tqdm: bool,
    ) -> float:
        self.model.train()
        self.loss_fn.train()
        running_loss_sum = 0.0
        count_samples = 0

        bar_enabled = bool(use_tqdm and tqdm is not None)
        iterator = train_loader
        if bar_enabled:
            iterator = tqdm(
                train_loader,
                desc=f"Epoch [{epoch}/{total_epochs}]",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )

        for batch_idx, batch in enumerate(iterator, start=1):
            x = batch["x"].to(self.device, **self._to_kwargs)
            y = batch["y"].to(self.device, **self._to_kwargs)
            y_prev = batch["y_prev"].to(self.device, **self._to_kwargs)
            assert_same_device(x, y, y_prev)

            self.optimizer.zero_grad(set_to_none=True)
            y_pred = self.model(x)
            assert_same_device(y_pred, y, y_prev)
            loss = self.loss_fn(y_pred, y, y_prev)
            assert_same_device(loss, y_pred)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss detected at epoch {epoch}, batch {batch_idx}: {loss.item()}"
                )

            loss.backward()
            self.optimizer.step()

            batch_size = int(x.size(0))
            running_loss_sum += float(loss.item()) * batch_size
            count_samples += batch_size

            if bar_enabled:
                avg_loss = running_loss_sum / max(1, count_samples)
                iterator.set_postfix_str(f"Loss: {avg_loss:.4f}", refresh=False)

        return running_loss_sum / max(1, count_samples)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        target_scaler,
        run_name: str,
        log_dir: str,
        plot_dir: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_meta_path: str | None = None,
        history_path: str | None = None,
        resume: bool = False,
        resume_epoch_hint: int = 0,
        resume_status: str | None = None,
        checkpoint_every: int = 1,
        checkpoint_metadata: Dict[str, Any] | None = None,
        use_tqdm: bool = True,
    ) -> Dict[str, object]:
        history: List[Dict[str, float]] = []

        best_rmse = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        start_epoch = 0
        resumed_from_epoch = 0

        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
        checkpoint_meta_file = Path(checkpoint_meta_path) if checkpoint_meta_path else None
        best_model_path: Path | None = None
        if checkpoint_file is not None:
            best_model_path = checkpoint_file.with_name(f"{checkpoint_file.stem}_best.pt")

        if checkpoint_file is not None and resume:
            loaded = self._load_checkpoint(checkpoint_file)
            if loaded is not None:
                start_epoch = int(loaded.get("epoch", 0))
                resumed_from_epoch = start_epoch
                history = list(loaded.get("history", []))
                best_rmse = float(loaded.get("best_val_rmse", float("inf")))
                best_state = copy.deepcopy(
                    loaded.get("best_model_state_dict", self.model.state_dict())
                )
                
                if start_epoch >= epochs:
                    print(f"Model already trained for {start_epoch}/{epochs} epochs. Skipping training.")
                    return {
                        "epoch": start_epoch,
                        "best_val_rmse": best_rmse,
                        "history": history,
                        "resumed_from_epoch": resumed_from_epoch,
                        "resume_status": "already_completed",
                        "log_path": str(log_path),
                        "checkpoint_path": str(checkpoint_file) if checkpoint_file is not None else None,
                        "best_model_path": str(best_model_path) if best_model_path is not None else None,
                    }
                
                print(f"Resuming from epoch {start_epoch}")
            elif resume_epoch_hint > 0:
                print(
                    f"[WARN] Resume requested at/after epoch {resume_epoch_hint} but no valid checkpoint was loaded; restarting from epoch 0."
                )
        elif checkpoint_file is not None and checkpoint_file.exists():
            # Start fresh when resume is disabled.
            checkpoint_file.unlink(missing_ok=True)
            if best_model_path is not None:
                best_model_path.unlink(missing_ok=True)
            if checkpoint_meta_file is not None:
                checkpoint_meta_file.unlink(missing_ok=True)

        checkpoint_every = max(1, int(checkpoint_every))
        log_path = Path(log_dir) / "training_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        history_json_path = Path(history_path) if history_path else (Path(log_dir) / "training_history.json")
        history_json_path.parent.mkdir(parents=True, exist_ok=True)

        if history:
            save_rows_csv(log_path, history)
            save_json(history_json_path, self._history_payload(history))

        epoch_range = range(start_epoch + 1, epochs + 1)
        epoch_bar_enabled = bool(use_tqdm and tqdm is not None)
        epoch_iterator = epoch_range
        if epoch_bar_enabled:
            epoch_iterator = tqdm(
                epoch_range,
                desc=f"Epoch [{start_epoch}/{epochs}]",
                total=max(0, epochs - start_epoch),
                leave=True,
                position=0,
                dynamic_ncols=True,
            )

        for epoch in epoch_iterator:
            train_loss = self.train_epoch(
                train_loader=train_loader,
                epoch=epoch,
                total_epochs=epochs,
                use_tqdm=use_tqdm,
            )
            val_metrics = evaluate_model(
                model=self.model,
                dataloader=val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                target_scaler=target_scaler,
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_directional_accuracy": val_metrics["directional_accuracy"],
            }
            history.append(record)

            if val_metrics["rmse"] < best_rmse:
                best_rmse = val_metrics["rmse"]
                best_state = copy.deepcopy(self.model.state_dict())

                if best_model_path is not None:
                    best_model_path.parent.mkdir(parents=True, exist_ok=True)
                    self._atomic_torch_save(
                        best_model_path,
                        {
                            "epoch": int(epoch),
                            "best_val_rmse": float(best_rmse),
                            "model_state_dict": best_state,
                            "metadata": checkpoint_metadata or {},
                        },
                    )

            if checkpoint_file is not None and (epoch % checkpoint_every == 0 or epoch == epochs):
                self._save_checkpoint(
                    checkpoint_path=checkpoint_file,
                    checkpoint_meta_path=checkpoint_meta_file,
                    epoch=epoch,
                    best_val_rmse=best_rmse,
                    history=history,
                    checkpoint_metadata=checkpoint_metadata,
                    best_model_state_dict=best_state,
                )

            save_rows_csv(log_path, history)
            save_json(history_json_path, self._history_payload(history))

            if epoch_bar_enabled:
                epoch_iterator.set_description(f"Epoch [{epoch}/{epochs}]")
                epoch_iterator.set_postfix_str(
                    f"Loss: {train_loss:.4f} | RMSE: {val_metrics['rmse']:.4f}",
                    refresh=False,
                )

        self.model.load_state_dict(best_state)

        # Persist final state after best model restoration.
        save_rows_csv(log_path, history)
        save_json(history_json_path, self._history_payload(history))

        if plot_dir is not None:
            self._plot_history(history=history, run_name=run_name, plot_dir=plot_dir)

        return {
            "history": history,
            "best_val_rmse": best_rmse,
            "log_path": str(log_path),
            "checkpoint_path": str(checkpoint_file) if checkpoint_file is not None else None,
            "best_model_path": str(best_model_path) if best_model_path is not None else None,
            "resumed_from_epoch": resumed_from_epoch,
            "final_epoch": int(history[-1]["epoch"]) if history else int(start_epoch),
            "resume_status": str(resume_status or ("resumed" if resumed_from_epoch > 0 else "started")),
        }

    @staticmethod
    def _plot_history(history: List[Dict[str, float]], run_name: str, plot_dir: str) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if not history:
            return

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # Ensure title doesn't look weird if run_name has slashes
        clean_run_name = run_name.replace("/", " - ") if run_name else "Unknown"
        plt.title(f"Training Curves: {clean_run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(plot_dir) / "loss_curve.png", dpi=150)
        plt.close()
