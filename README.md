# RNAD: Regime-Normalized Adaptive Directional Loss for Financial Forecasting

RNAD is a research-oriented time-series forecasting project centered on a custom loss function, **Regime-Normalized Adaptive Directional (RNAD) loss**. The main goal is to improve one-step financial forecasting by training models to care about both **error magnitude** and **directional behavior** under changing market regimes.

In practical terms, this repository provides:
- an RNAD loss implementation (plus ablation variants),
- multiple modern forecasting backbones,
- reproducible experiment/ablation runners,
- and tooling to aggregate and compare results across datasets, models, and losses.

## What problem RNAD addresses

Standard regression losses (e.g., MSE) optimize pointwise error but can be fragile on financial series with volatility shifts, heavy tails, and noisy short-term moves. RNAD is designed to be a drop-in training objective that combines:

- **Regime-aware normalization** of residuals using volatility tracking,
- **Tail-aware magnitude shaping** (smooth transition between quadratic and robust behavior),
- **Noise-gated directional learning** to penalize wrong directional signals where it matters most.

The repository compares RNAD against baselines (`mse`, `huber`, `log_cosh`, `gmadl`) across several datasets and models.

## Key features

- **Loss suite for controlled comparisons**: `mse`, `huber`, `log_cosh`, `gmadl`, `rnad`.
- **RNAD ablation support**: `RNAD_FULL`, `RNAD_NO_DIR`, `RNAD_NO_TAIL`, `RNAD_NO_NOISE`.
- **Multiple backbones**: PatchTST, iTransformer, N-HiTS, and TimeXer.
- **Deterministic experiment pipeline** with seed control and chronological train/val/test splits.
- **Robust run orchestration** with resume/restart logic, signature checks, and per-run artifacts.
- **Built-in preprocessing**: feature scaling + technical indicators (MACD, RSI, EMA, Bollinger bands).

## Project structure (high level)

- `data/`
	- `raw/`: source CSV files (e.g., `AAPL.csv`, `BTCUSD.csv`, `USDJPY.csv`)
	- `loaders.py`: dataset loading, splitting, windowing, DataLoader creation
- `losses/`: RNAD and baseline loss implementations
- `models/`: forecasting architectures and supporting layers
- `training/`: model/loss builders and training loop (`Trainer`)
- `evaluation/`: RMSE/MAE/directional accuracy metrics and evaluation utilities
- `experiments/`: experiment configs and experiment/ablation runners
- `scripts/`: practical CLI entry points (`run_single.py`, `run_all.py`, ablation scripts)
- `results/`: generated experiment summaries and comparison tables
- `RNAD_loss/`: standalone PyTorch/TensorFlow RNAD modules
- `RNAD_PAPER/`: manuscript and publication assets
- `notebooks/`: exploratory notebook for RNAD loss visualization

## Installation

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Current core dependencies in `requirements.txt`:
- `numpy`
- `pandas`
- `torch`
- `PyYAML`
- `matplotlib`
- `tqdm`

> Note: TensorFlow is only needed if you want to use `RNAD_loss/rnad_tensorflow.py`.

## Data format expectations

Each dataset CSV should include at least:
- `datetime` (timestamp column)
- `open`, `high`, `low`, `close`, `volume`

Technical indicators are computed inside the pipeline, and the default config expects a feature list including derived indicators.

## Basic usage

### 1) Run the full experiment matrix from config

```bash
python experiments/run_experiment.py --config experiments/config.yaml
```

This runs combinations of configured models, losses, datasets, and seeds, then writes per-run and aggregated metrics under `results/metrics/`.

### 2) Run a single training/evaluation job

```bash
python scripts/run_single.py --config experiments/config.yaml --model patchtst --loss rnad --dataset data/raw/AAPL.csv --seed 178
```

Per-run outputs are stored under `results/single_runs/<model>/<dataset>/<loss>/seed<seed>/` (or ablation-specific roots).

### 3) Run all jobs with multi-worker orchestration

```bash
python scripts/run_all.py --config experiments/config.yaml --workers 3
```

### 4) Run RNAD ablations

```bash
python scripts/run_all_ablations.py --config experiments/config.yaml
```

Or a single ablation variant:

```bash
python scripts/run_ablation.py --config experiments/config.yaml --dataset BTCUSD --variant RNAD_NO_DIR
```

## Outputs you can expect

Depending on the runner, generated artifacts include:
- `metrics.json`, `training_history.json`, `training_log.csv`
- `predictions.csv`, `val_predictions.csv`
- checkpoints (`model_last.pt`, `model_final.pt`, etc.)
- plots (`loss_curve.png`, `val_metrics.png`)
- aggregated summaries (e.g., in `results/metrics/` and top-level `results/*.csv`)

## Reproducibility notes

- The default config includes fixed seeds and deterministic mode.
- Data is split chronologically (train/val/test) to avoid look-ahead leakage.
- `scripts/run_all.py` and `scripts/run_single.py` include resume/restart safeguards and run signature checks.

---

If you are starting fresh, the fastest path is:
1) verify CSVs in `data/raw/`,
2) install dependencies,
3) run `experiments/run_experiment.py` with `experiments/config.yaml`,
4) inspect `results/` summaries.
