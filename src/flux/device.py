"""Device selection. Config default is CPU; auto still prefers CUDA then MPS then CPU."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> str:
    name = (requested or "cpu").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name in {"cpu", "cuda", "mps"}:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FLUX_DEVICE=cuda but torch.cuda.is_available() is False")
        if name == "mps" and not (
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ):
            raise RuntimeError("FLUX_DEVICE=mps but MPS is not available")
        return name
    raise ValueError(f"Unknown device {requested!r}; use cpu, cuda, mps, or auto")
