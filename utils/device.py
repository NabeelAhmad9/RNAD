from __future__ import annotations

import warnings
from typing import Iterable

import torch



def resolve_device(requested: str | None = "auto") -> torch.device:
    """Resolve runtime device with safe fallback behavior.

    Rules:
    - requested in {None, "auto"} -> cuda if available else cpu
    - requested == "cpu" -> cpu
    - requested startswith "cuda" -> that cuda device if available, else cpu + warning
    """
    req = "auto" if requested is None else str(requested).strip().lower()

    if req in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if req == "cpu":
        return torch.device("cpu")

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(req)
        warnings.warn(
            f"Requested device '{requested}' but CUDA is unavailable. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    warnings.warn(
        f"Unknown device setting '{requested}'. Falling back to auto detection.",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def is_cuda_device(device: torch.device) -> bool:
    return torch.device(device).type == "cuda"



def transfer_kwargs(device: torch.device) -> dict:
    """Recommended kwargs for Tensor.to(...)."""
    return {"non_blocking": is_cuda_device(device)}



def dataloader_pin_memory(device: torch.device) -> bool:
    return is_cuda_device(device)



def assert_same_device(*tensors: torch.Tensor) -> None:
    real_tensors: list[torch.Tensor] = [t for t in tensors if t is not None]
    if not real_tensors:
        return

    devices = {t.device for t in real_tensors if torch.is_tensor(t)}
    if len(devices) > 1:
        raise RuntimeError(f"Device mismatch detected: {sorted(str(d) for d in devices)}")
