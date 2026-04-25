from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import resolve_loss_config


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_runtime_config(base_cfg: Dict[str, Any]) -> Path:
    runtime_cfg = copy.deepcopy(base_cfg)
    runtime_cfg.setdefault("training", {})
    runtime_cfg["training"]["resume"] = True
    runtime_cfg["training"]["checkpoint_every"] = int(runtime_cfg["training"].get("checkpoint_every", 1))

    runtime_cfg_path = ROOT / "results" / "logs" / "run_all_runtime_config.yaml"
    runtime_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    return runtime_cfg_path


def _resolve_dataset_path(dataset_path: str) -> Path:
    p = Path(dataset_path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _resolve_runs_root_path(runs_root: str) -> Path:
    p = Path(runs_root)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


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
    p = _resolve_dataset_path(dataset_path)
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


def _run_signature_payload(cfg: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    model = str(job["model"])
    loss = str(job["loss"])
    seed = int(job["seed"])
    dataset = str(job["dataset"])

    return {
        "model": model,
        "loss": loss,
        "seed": seed,
        "dataset": str(Path(dataset)),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "training": cfg.get("training", {}),
        "loss_configs": resolve_loss_config(loss, cfg.get("loss_configs", {})),
        "model_config": cfg.get("model_configs", {}).get(model, {}),
        "dataset_config": {
            k: v for k, v in cfg.get("dataset", {}).items() if k != "processed_output_path"
        },
    }


def _expected_signature_hash(cfg: Dict[str, Any], job: Dict[str, Any]) -> str:
    return _stable_hash(_run_signature_payload(cfg, job))


def _run_dir_for(
    model: str,
    loss: str,
    dataset: str,
    seed: int,
    runs_root: str = "results/single_runs",
    include_seed_subdir: bool = True,
) -> Path:
    root = _resolve_runs_root_path(runs_root)
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


def _training_log_path(run_dir: Path, model: str, loss: str, seed: int) -> Path:
    new_path = run_dir / "training_log.csv"
    legacy_path = run_dir / f"{model}_{loss}_seed{seed}_training_log.csv"
    return legacy_path if not new_path.exists() and legacy_path.exists() else new_path


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

        lengths: List[int] = []
        for key in ("epoch", "train_loss", "val_loss", "val_rmse", "val_mae", "val_directional_accuracy"):
            value = payload.get(key)
            if isinstance(value, list):
                lengths.append(len(value))

        if not lengths:
            return 0

        # Reject malformed mixed-length lists (partial write, manual corruption, etc).
        if len(set(lengths)) > 1:
            return 0

        return int(lengths[0])
    except Exception:
        return 0


def _metrics_looks_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return False

        required = {"test_rmse", "test_mae", "test_directional_accuracy"}
        if not required.issubset(payload.keys()):
            return False

        for key in required:
            val = payload.get(key)
            if not isinstance(val, (int, float)):
                return False

        return True
    except Exception:
        return False


def _csv_file_has_data(path: Path) -> bool:
    return _count_csv_data_rows(path) > 0


def _model_file_exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "model_last.pt",
        run_dir / "checkpoint.pt",  # legacy
    ]
    existing = [p for p in candidates if p.exists() and p.is_file() and p.stat().st_size > 0]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _checkpoint_looks_valid(path: Path | None) -> bool:
    if path is None:
        return False
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return False
        return "model_state_dict" in payload
    except Exception:
        return False


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _read_recorded_signature_hash(run_dir: Path) -> str:
    meta_payload = _load_json(run_dir / "checkpoint_meta.json")
    if meta_payload:
        metadata = meta_payload.get("metadata")
        if isinstance(metadata, dict):
            h = metadata.get("run_signature_hash")
            if isinstance(h, str) and h:
                return h

    metrics_payload = _load_json(run_dir / "metrics.json")
    if metrics_payload:
        h = metrics_payload.get("run_signature_hash")
        if isinstance(h, str) and h:
            return h

    snapshot_payload = _load_json(run_dir / "config_snapshot.json")
    if snapshot_payload:
        h = snapshot_payload.get("run_signature_hash")
        if isinstance(h, str) and h:
            return h

    return ""


def _is_signature_compatible(run_dir: Path, expected_signature_hash: str) -> Tuple[bool, str]:
    if not expected_signature_hash:
        return True, "expected signature missing; compatibility check skipped"

    recorded_hash = _read_recorded_signature_hash(run_dir)
    if not recorded_hash:
        return True, "legacy run without recorded signature; compatibility check skipped"

    if recorded_hash != expected_signature_hash:
        return False, f"signature mismatch (expected={expected_signature_hash[:10]}..., found={recorded_hash[:10]}...)"

    return True, "signature match"


def _is_run_complete(
    job: Dict[str, Any],
    expected_epochs: int,
    expected_signature_hash: str = "",
    runs_root: str = "results/single_runs",
    include_seed_subdir: bool = True,
) -> Tuple[bool, str]:
    model = str(job["model"])
    loss = str(job["loss"])
    dataset = str(job["dataset"])
    seed = int(job["seed"])

    run_dir = _run_dir_for(
        model,
        loss,
        dataset,
        seed,
        runs_root=runs_root,
        include_seed_subdir=include_seed_subdir,
    )
    if not run_dir.exists():
        return False, "run directory not found"

    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.json"
    predictions_path = run_dir / "predictions.csv"
    val_predictions_path = run_dir / "val_predictions.csv"
    model_final_path = run_dir / "model_final.pt"
    train_log_path = _training_log_path(run_dir, model, loss, seed)

    required_files = [model_final_path, metrics_path, predictions_path, val_predictions_path]
    missing = [p.name for p in required_files if not p.exists()]
    if missing:
        return False, f"missing outputs: {', '.join(missing)}"

    if not _model_file_exists_nonempty(model_final_path):
        return False, "model_final.pt missing or empty"

    if not _metrics_looks_complete(metrics_path):
        return False, "metrics.json missing, malformed, or incomplete"

    if not _csv_file_has_data(predictions_path):
        return False, "predictions.csv is empty, malformed, or header-only"

    if not _csv_file_has_data(val_predictions_path):
        return False, "val_predictions.csv is empty, malformed, or header-only"

    history_epochs = _history_epoch_count(history_path)
    log_epochs = _count_csv_data_rows(train_log_path)
    completed_epochs = max(history_epochs, log_epochs)

    if expected_epochs > 0 and completed_epochs < expected_epochs:
        return False, f"incomplete epochs: {completed_epochs}/{expected_epochs}"

    signature_ok, signature_reason = _is_signature_compatible(run_dir, expected_signature_hash)
    if not signature_ok:
        return False, signature_reason

    return True, "complete"


def _inspect_run_state(
    cfg: Dict[str, Any],
    job: Dict[str, Any],
    expected_epochs: int,
    force_rerun: bool,
    runs_root: str,
    include_seed_subdir: bool,
) -> Dict[str, Any]:
    model = str(job["model"])
    loss = str(job["loss"])
    dataset = str(job["dataset"])
    seed = int(job["seed"])

    run_dir = _run_dir_for(
        model,
        loss,
        dataset,
        seed,
        runs_root=runs_root,
        include_seed_subdir=include_seed_subdir,
    )
    signature_hash = _expected_signature_hash(cfg, job)

    if force_rerun:
        return {
            "resume_status": "restarted",
            "reason": "force rerun enabled",
            "resume_epoch": 0,
            "run_dir": str(run_dir),
            "signature_hash": signature_hash,
        }

    complete, reason = _is_run_complete(
        job=job,
        expected_epochs=expected_epochs,
        expected_signature_hash=signature_hash,
        runs_root=runs_root,
        include_seed_subdir=include_seed_subdir,
    )
    if complete:
        return {
            "resume_status": "complete",
            "reason": reason,
            "resume_epoch": expected_epochs,
            "run_dir": str(run_dir),
            "signature_hash": signature_hash,
        }

    history_path = run_dir / "training_history.json"
    train_log_path = _training_log_path(run_dir, model, loss, seed)
    completed_epochs = max(_history_epoch_count(history_path), _count_csv_data_rows(train_log_path))

    signature_ok, signature_reason = _is_signature_compatible(run_dir, signature_hash)
    if not signature_ok:
        return {
            "resume_status": "restarted",
            "reason": signature_reason,
            "resume_epoch": 0,
            "run_dir": str(run_dir),
            "signature_hash": signature_hash,
        }

    checkpoint_path = _find_latest_checkpoint(run_dir)
    if _checkpoint_looks_valid(checkpoint_path):
        return {
            "resume_status": "resumed",
            "reason": f"partial run with valid checkpoint at epoch~{completed_epochs}",
            "resume_epoch": int(completed_epochs),
            "run_dir": str(run_dir),
            "signature_hash": signature_hash,
        }

    return {
        "resume_status": "restarted",
        "reason": f"{reason}; no valid checkpoint for safe resume",
        "resume_epoch": 0,
        "run_dir": str(run_dir),
        "signature_hash": signature_hash,
    }


def _thread_limited_env(base_env: Dict[str, str], gpu_id: str, cpu_threads: int) -> Dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    thread_value = str(max(1, int(cpu_threads)))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[var] = thread_value

    return env


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _read_lock_payload(lock_path: Path) -> Dict[str, Any] | None:
    if not lock_path.exists() or lock_path.stat().st_size == 0:
        return None
    try:
        with lock_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _acquire_run_lock(lock_path: Path, worker_id: int, stale_seconds: int) -> Tuple[bool, str]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": int(os.getpid()),
        "worker_id": int(worker_id),
        "acquired_ts": float(time.time()),
    }

    def _try_create() -> bool:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
        return True

    try:
        _try_create()
        return True, "lock acquired"
    except FileExistsError:
        existing = _read_lock_payload(lock_path)
        if existing is None:
            # Corrupted lock file: treat as stale and replace.
            lock_path.unlink(missing_ok=True)
            try:
                _try_create()
                return True, "replaced corrupted lock"
            except Exception as exc:
                return False, f"failed to replace corrupted lock: {exc}"

        lock_pid = int(existing.get("pid", -1))
        lock_ts = float(existing.get("acquired_ts", 0.0))
        age_s = max(0.0, time.time() - lock_ts)

        stale = (not _pid_is_alive(lock_pid)) or (stale_seconds > 0 and age_s > float(stale_seconds))
        if stale:
            lock_path.unlink(missing_ok=True)
            try:
                _try_create()
                return True, "replaced stale lock"
            except Exception as exc:
                return False, f"failed to acquire lock after stale cleanup: {exc}"

        return False, f"active lock held by pid={lock_pid} age_s={age_s:.1f}"
    except Exception as exc:
        return False, f"lock acquisition error: {exc}"


def _release_run_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def _classify_failure(return_code: int, stdout_text: str, stderr_text: str) -> str:
    text = f"{stdout_text}\n{stderr_text}".lower()

    non_recoverable_markers = [
        "config error",
        "dataset not found",
        "controlled experiment constraint violation",
        "unknown model",
        "unknown loss",
        "filenotfounderror",
        "modulenotfounderror",
        "valueerror",
        "typeerror",
    ]
    if any(marker in text for marker in non_recoverable_markers):
        return "non-recoverable"

    recoverable_markers = [
        "could not load checkpoint",
        "cuda out of memory",
        "connection reset",
        "timeout",
        "killed",
    ]
    if any(marker in text for marker in recoverable_markers):
        return "recoverable"

    if return_code in {-9, 137, 143}:
        return "recoverable"

    return "recoverable"


def _worker_main(
    worker_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    runtime_config_path: str,
    python_executable: str,
    gpu_id: str,
    cpu_threads: int,
    max_retries: int,
    lock_stale_seconds: int,
) -> None:
    base_env = _thread_limited_env(os.environ, gpu_id=gpu_id, cpu_threads=cpu_threads)

    while True:
        job = job_queue.get()
        if job is None:
            result_queue.put({"type": "worker_done", "worker_id": worker_id})
            return

        model = str(job["model"])
        loss = str(job["loss"])
        dataset = str(job["dataset"])
        seed = int(job["seed"])
        resume_status = str(job.get("resume_status", "auto"))
        resume_epoch = int(job.get("resume_epoch", 0))
        signature_hash = str(job.get("signature_hash", ""))

        run_dir_hint = str(job.get("run_dir", "")).strip()
        run_dir = Path(run_dir_hint) if run_dir_hint else _run_dir_for(model, loss, dataset, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = run_dir / "RUNNING.lock"
        orchestrator_log_path = run_dir / "run_all_worker.log"

        lock_ok, lock_reason = _acquire_run_lock(lock_path, worker_id=worker_id, stale_seconds=lock_stale_seconds)
        if not lock_ok:
            result_queue.put(
                {
                    "type": "job_result",
                    "status": "locked",
                    "worker_id": worker_id,
                    "model": model,
                    "loss": loss,
                    "dataset": dataset,
                    "seed": seed,
                    "elapsed_s": 0.0,
                    "return_code": -1,
                    "stderr_tail": lock_reason,
                    "failure_class": "recoverable",
                    "attempts": 0,
                    "resume_status": resume_status,
                }
            )
            continue

        attempts = 0
        final_record: Dict[str, Any] | None = None
        try:
            while attempts <= int(max_retries):
                attempts += 1
                start = time.time()
                cmd = [
                    python_executable,
                    str(ROOT / "scripts" / "run_single.py"),
                    "--config",
                    runtime_config_path,
                    "--model",
                    model,
                    "--loss",
                    loss,
                    "--dataset",
                    dataset,
                    "--seed",
                    str(seed),
                    "--resume-mode",
                    resume_status if resume_status in {"resumed", "restarted"} else "auto",
                    "--resume-epoch-hint",
                    str(max(0, resume_epoch)),
                    "--expected-signature-hash",
                    signature_hash,
                ]

                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        env=base_env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    elapsed_s = time.time() - start

                    failure_class = _classify_failure(proc.returncode, proc.stdout or "", proc.stderr or "")
                    status = "done" if proc.returncode == 0 else "failed"

                    with orchestrator_log_path.open("a", encoding="utf-8") as f:
                        f.write("\n" + "=" * 100 + "\n")
                        f.write(
                            f"worker={worker_id} attempt={attempts} model={model} loss={loss} seed={seed} "
                            f"dataset={dataset} resume_status={resume_status} resume_epoch={resume_epoch} "
                            f"elapsed_s={elapsed_s:.2f}\n"
                        )
                        f.write(f"lock_status={lock_reason}\n")
                        f.write(f"return_code={proc.returncode} failure_class={failure_class}\n")
                        f.write("--- STDOUT ---\n")
                        f.write(proc.stdout or "")
                        f.write("\n--- STDERR ---\n")
                        f.write(proc.stderr or "")
                        f.write("\n")

                    final_record = {
                        "type": "job_result",
                        "status": status,
                        "worker_id": worker_id,
                        "model": model,
                        "loss": loss,
                        "dataset": dataset,
                        "seed": seed,
                        "elapsed_s": elapsed_s,
                        "return_code": proc.returncode,
                        "stderr_tail": (proc.stderr or "")[-1500:],
                        "failure_class": failure_class,
                        "attempts": attempts,
                        "resume_status": resume_status,
                    }

                    if status == "done":
                        break

                    can_retry = failure_class == "recoverable" and attempts <= int(max_retries)
                    if not can_retry:
                        break

                    # Retry as resume after any partial-progress failure.
                    resume_status = "resumed"
                    time.sleep(min(10.0, 1.0 + attempts * 1.5))
                except Exception as exc:
                    elapsed_s = time.time() - start
                    final_record = {
                        "type": "job_result",
                        "status": "failed",
                        "worker_id": worker_id,
                        "model": model,
                        "loss": loss,
                        "dataset": dataset,
                        "seed": seed,
                        "elapsed_s": elapsed_s,
                        "return_code": -1,
                        "stderr_tail": str(exc),
                        "failure_class": "recoverable",
                        "attempts": attempts,
                        "resume_status": resume_status,
                    }

                    if attempts > int(max_retries):
                        break
                    time.sleep(min(10.0, 1.0 + attempts * 1.5))

            if final_record is None:
                final_record = {
                    "type": "job_result",
                    "status": "failed",
                    "worker_id": worker_id,
                    "model": model,
                    "loss": loss,
                    "dataset": dataset,
                    "seed": seed,
                    "elapsed_s": 0.0,
                    "return_code": -1,
                    "stderr_tail": "unknown worker state",
                    "failure_class": "recoverable",
                    "attempts": attempts,
                    "resume_status": resume_status,
                }

            result_queue.put(final_record)
        finally:
            _release_run_lock(lock_path)


def _matrix_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    exp_cfg = cfg.get("experiment", {})

    models = list(exp_cfg.get("models", []))
    losses = list(exp_cfg.get("losses", []))
    datasets = list(exp_cfg.get("datasets", []))
    seeds = list(exp_cfg.get("seeds", [42]))

    if not models:
        raise ValueError("Config error: experiment.models is empty.")
    if not losses:
        raise ValueError("Config error: experiment.losses is empty.")
    if not datasets:
        fallback = cfg.get("dataset", {}).get("path")
        if fallback:
            datasets = [fallback]
    if not datasets:
        raise ValueError("Config error: experiment.datasets is empty and dataset.path fallback is missing.")

    jobs: List[Dict[str, Any]] = []
    for dataset, model, loss, seed in product(datasets, models, losses, seeds):
        jobs.append(
            {
                "dataset": str(dataset),
                "model": str(model),
                "loss": str(loss),
                "seed": int(seed),
            }
        )

    return jobs


def run_all(
    config_path: str,
    workers: int = 3,
    gpu_id: str = "0",
    cpu_threads: int = 1,
    force_rerun: bool = False,
    dry_run: bool = False,
    max_retries: int = 2,
    lock_stale_seconds: int = 6 * 3600,
) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    runtime_config_path = _save_runtime_config(cfg)

    # Use the runtime config for all downstream decisions so that
    # the signature/hash computed here matches what the child
    # `run_single.py` process will compute (runtime config sets
    # training.resume = True and may tweak checkpointing).
    try:
        cfg = _load_config(str(runtime_config_path))
    except Exception:
        # Fallback to the original config if loading the runtime
        # config fails for any reason.
        pass

    expected_epochs = int(cfg.get("training", {}).get("epochs", 0))
    paths_cfg = cfg.get("paths", {})
    runs_root = str(paths_cfg.get("runs_root", "results/single_runs"))
    include_seed_subdir = bool(paths_cfg.get("include_seed_subdir", True))
    runs_root_path = _resolve_runs_root_path(runs_root)

    all_jobs = _matrix_from_config(cfg)
    pending_jobs: List[Dict[str, Any]] = []
    skipped_jobs: List[Dict[str, Any]] = []
    invalid_jobs: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    for job in all_jobs:
        dataset_abs = _resolve_dataset_path(str(job["dataset"]))
        if not dataset_abs.exists():
            invalid_jobs.append({**job, "reason": f"dataset not found: {dataset_abs}"})
            decisions.append({
                **job,
                "resume_status": "invalid",
                "reason": f"dataset not found: {dataset_abs}",
            })
            continue

        state = _inspect_run_state(
            cfg=cfg,
            job=job,
            expected_epochs=expected_epochs,
            force_rerun=force_rerun,
            runs_root=runs_root,
            include_seed_subdir=include_seed_subdir,
        )
        decision_row = {
            **job,
            "resume_status": state["resume_status"],
            "reason": state["reason"],
            "resume_epoch": int(state.get("resume_epoch", 0)),
            "run_dir": state.get("run_dir", ""),
            "signature_hash": state.get("signature_hash", ""),
        }
        decisions.append(decision_row)

        if state["resume_status"] == "complete":
            skipped_jobs.append({**job, "reason": state["reason"]})
            continue

        pending_jobs.append({
            **job,
            "resume_status": state["resume_status"],
            "resume_epoch": int(state.get("resume_epoch", 0)),
            "signature_hash": str(state.get("signature_hash", "")),
            "run_dir": str(state.get("run_dir", "")),
        })

    resumed_jobs = [j for j in pending_jobs if j.get("resume_status") == "resumed"]
    restarted_jobs = [j for j in pending_jobs if j.get("resume_status") == "restarted"]

    summary: Dict[str, Any] = {
        "total_matrix_jobs": len(all_jobs),
        "invalid_jobs": len(invalid_jobs),
        "skipped_completed": len(skipped_jobs),
        "scheduled": len(pending_jobs),
        "scheduled_resumed": len(resumed_jobs),
        "scheduled_restarted": len(restarted_jobs),
        "workers": int(workers),
        "gpu_id": str(gpu_id),
        "cpu_threads": int(cpu_threads),
        "max_retries": int(max_retries),
        "lock_stale_seconds": int(lock_stale_seconds),
        "runtime_config": str(runtime_config_path),
        "runs_root": str(runs_root_path),
        "include_seed_subdir": bool(include_seed_subdir),
    }

    print(json.dumps(summary, indent=2))

    if invalid_jobs:
        print("\n[WARN] Invalid jobs (missing dataset files):")
        for item in invalid_jobs:
            print(
                f"  - model={item['model']} loss={item['loss']} seed={item['seed']} "
                f"dataset={item['dataset']} | {item['reason']}"
            )

    if dry_run or not pending_jobs:
        return {
            **summary,
            "completed": 0,
            "failed": 0,
            "locked": 0,
            "failed_jobs": [],
            "decisions": decisions,
        }

    ctx = mp.get_context("spawn")
    job_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    for job in pending_jobs:
        job_queue.put(job)
    for _ in range(int(workers)):
        job_queue.put(None)

    procs: List[mp.Process] = []
    for worker_idx in range(int(workers)):
        proc = ctx.Process(
            target=_worker_main,
            args=(
                worker_idx + 1,
                job_queue,
                result_queue,
                str(runtime_config_path),
                str(sys.executable),
                str(gpu_id),
                int(cpu_threads),
                int(max_retries),
                int(lock_stale_seconds),
            ),
        )
        proc.daemon = False
        proc.start()
        procs.append(proc)

    done_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    locked_rows: List[Dict[str, Any]] = []
    collected_results = 0
    total_results = len(pending_jobs)
    pbar = tqdm(total=total_results, desc="Running jobs", dynamic_ncols=True, leave=True)
    try:
        while collected_results < total_results:
            msg = result_queue.get()
            if msg.get("type") != "job_result":
                continue

            collected_results += 1
            status = str(msg.get("status", "failed"))
            record = {
                "status": status,
                "worker_id": int(msg.get("worker_id", -1)),
                "model": str(msg.get("model", "")),
                "loss": str(msg.get("loss", "")),
                "dataset": str(msg.get("dataset", "")),
                "seed": int(msg.get("seed", -1)),
                "elapsed_s": float(msg.get("elapsed_s", 0.0)),
                "return_code": int(msg.get("return_code", -1)),
                "stderr_tail": str(msg.get("stderr_tail", "")),
                "failure_class": str(msg.get("failure_class", "recoverable")),
                "attempts": int(msg.get("attempts", 1)),
                "resume_status": str(msg.get("resume_status", "unknown")),
            }

            if status == "done":
                done_rows.append(record)
            elif status == "locked":
                locked_rows.append(record)
            else:
                failed_rows.append(record)

            pbar.update(1)
            pbar.set_postfix(
                {
                    "done": len(done_rows),
                    "failed": len(failed_rows),
                    "locked": len(locked_rows),
                    "last": status.upper(),
                }
            )
    finally:
        pbar.close()

    for proc in procs:
        proc.join()

    summary_path = runs_root_path / "run_all_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    final_summary = {
        **summary,
        "completed": len(done_rows),
        "failed": len(failed_rows),
        "locked": len(locked_rows),
        "failed_jobs": failed_rows,
        "locked_jobs": locked_rows,
        "decisions": decisions,
        "summary_path": str(summary_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Queue-based multi-process runner that launches run_single.py for every "
            "(model, loss, dataset, seed) combination with robust resume/restart handling."
        )
    )
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--lock-stale-seconds", type=int, default=6 * 3600)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.lock_stale_seconds < 0:
        raise ValueError("--lock-stale-seconds must be >= 0")

    result = run_all(
        config_path=args.config,
        workers=int(args.workers),
        gpu_id=str(args.gpu_id),
        cpu_threads=int(args.cpu_threads),
        force_rerun=bool(args.force_rerun),
        dry_run=bool(args.dry_run),
        max_retries=int(args.max_retries),
        lock_stale_seconds=int(args.lock_stale_seconds),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
