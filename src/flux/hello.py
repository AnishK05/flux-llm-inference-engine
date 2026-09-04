"""Phase 0 device probe: allocate a CPU fp32 tensor and print runtime facts."""

from __future__ import annotations

import torch

from flux.runtime import apply_thread_caps, probe


def main() -> None:
    intra = apply_thread_caps("auto")
    tensor = torch.zeros((1000, 1000), dtype=torch.float32)
    info = probe()
    print("flux hello cpu")
    print(f"torch_version={info['torch_version']}")
    print(f"cuda_available={info['cuda_available']}")
    print(f"device_config={info['device']}")
    print(f"tensor.device={tensor.device}")
    print(f"tensor.dtype={tensor.dtype}")
    print(f"tensor.shape={tuple(tensor.shape)}")
    print(f"cpu_count={info['cpu_count']}")
    print(f"intra_threads={intra}")
    print(f"rss_bytes={info['rss_bytes']}")
    print(f"model_id={info['model_id']}")
    print(f"model_loaded={info['model_loaded']}")
    if tensor.device.type != "cpu":
        raise SystemExit("expected cpu tensor")
    if tensor.dtype != torch.float32:
        raise SystemExit("expected fp32 tensor")


if __name__ == "__main__":
    main()
