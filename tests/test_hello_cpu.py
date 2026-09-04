import torch

from flux.device import resolve_device
from flux.hello import main
from flux.runtime import parse_intra_threads, probe


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_intra_threads_auto_caps_at_eight() -> None:
    assert parse_intra_threads("auto", cpu_count=16) == 8
    assert parse_intra_threads("auto", cpu_count=2) == 2
    assert parse_intra_threads("4", cpu_count=16) == 4


def test_probe_keys() -> None:
    info = probe()
    for key in (
        "device",
        "dtype",
        "torch_version",
        "cuda_available",
        "cpu_count",
        "intra_threads",
        "rss_bytes",
        "model_id",
        "model_loaded",
    ):
        assert key in info
    assert info["device"] == "cpu"
    assert info["model_loaded"] is False
    assert info["rss_bytes"] > 0


def test_cpu_fp32_allocation() -> None:
    tensor = torch.zeros((1000, 1000), dtype=torch.float32)
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32


def test_hello_main_prints(capsys) -> None:
    main()
    out = capsys.readouterr().out
    assert "tensor.device=cpu" in out
    assert "torch_version=" in out
    assert "cuda_available=" in out
