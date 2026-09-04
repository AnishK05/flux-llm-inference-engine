"""Process-level CPU runtime: thread caps, RSS, health probe."""

from __future__ import annotations

import os
from typing import Any

import torch

from flux import __version__
from flux.config import Settings
from flux.device import resolve_device


def parse_intra_threads(value: str | int | None, cpu_count: int | None = None) -> int:
    ncpu = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    if value is None or value == "auto":
        return max(1, min(8, ncpu))
    n = int(value)
    if n < 1:
        raise ValueError("intra_threads must be >= 1")
    return n


def apply_thread_caps(intra_threads: str | int | None = "auto") -> int:
    n = parse_intra_threads(intra_threads)
    torch.set_num_threads(n)
    # Interop threads cannot be changed after PyTorch has done parallel work
    # (common in tests that construct modules before the API lifespan runs).
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return n


def rss_bytes() -> int:
    """Current resident set size. Linux / WSL2 uses /proc; elsewhere ru_maxrss."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, ValueError, OSError):
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes. This project targets WSL2/Linux.
        return int(rss) * 1024


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def hardware_facts(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings()
    try:
        intra = torch.get_num_threads()
    except RuntimeError:
        intra = parse_intra_threads(settings.intra_threads)
    ram = ram_bytes()
    return {
        "os": "linux",
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "ram_bytes": ram,
        "ram_gib": round(ram / 1024**3, 2) if ram else 0,
        "intra_threads": intra,
        "device": resolve_device(settings.device),
        "dtype": settings.dtype,
        "model": settings.model,
        "note": (
            "Official resume numbers should be re-run on the Windows + WSL2 laptop. "
            "This host is the machine that executed the bench."
        ),
    }


def hardware_line(settings: Settings | None = None) -> str:
    facts = hardware_facts(settings)
    return (
        f"{facts['os']}, {facts['cpu_model']}, {facts['cpu_count']} cores, "
        f"{facts['ram_gib']} GiB RAM, FLUX_INTRA_THREADS={facts['intra_threads']}, "
        f"{facts['model']}, {facts['dtype']} on {facts['device']}"
    )


def probe(settings: Settings | None = None, model_loaded: bool = False) -> dict[str, Any]:
    settings = settings or Settings()
    device = resolve_device(settings.device)
    try:
        intra = torch.get_num_threads()
    except RuntimeError:
        intra = parse_intra_threads(settings.intra_threads)
    return {
        "ok": True,
        "version": __version__,
        "device": device,
        "dtype": settings.dtype,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": os.cpu_count() or 0,
        "intra_threads": intra,
        "rss_bytes": rss_bytes(),
        "model_id": settings.model,
        "model_loaded": model_loaded,
        "served_model": "flux-qwen-0.5b",
        "serve_engine": settings.serve_engine,
    }
