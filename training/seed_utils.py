from __future__ import annotations

import os
import random

import numpy as np
import torch



def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some CPU backends may not support full deterministic mode.
            pass
    else:
        # Throughput-oriented mode (especially on CUDA).
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(torch.cuda.is_available())
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
