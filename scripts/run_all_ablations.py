from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_ablations import run_all_ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full deterministic RNAD ablation matrix.")
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
