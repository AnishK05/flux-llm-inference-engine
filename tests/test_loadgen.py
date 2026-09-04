import asyncio

import httpx

from benchmarks.loadgen import run_load
from benchmarks.report import bar_chart_svg, write_report
from benchmarks.scenarios import prompt_of_tokens
from flux.config import Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.server.app import create_app


def test_closed_loop_loadgen_fake() -> None:
    async def main() -> None:
        app = create_app(
            settings=Settings(load_model=False, serve_engine="continuous"),
            engine=CachedEngine(FakeLM(), FakeTokenizer()),
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                result = await run_load(
                    client,
                    prompts=[prompt_of_tokens(8)] * 8,
                    max_tokens=[4] * 8,
                    concurrency=2,
                    stream=True,
                    warmup=2,
                    engine="continuous",
                    scenario="short_chat",
                )
        assert len(result.records) == 8
        assert result.statuses.get("200") == 8
        agg = result.aggregates()
        assert agg["n_measured"] == 6
        assert agg["output_tokens"] >= 6
        assert agg["tok_s"] > 0
        assert agg["ttft_p50_ms"] is not None

    asyncio.run(main())


def test_report_writes_markdown_and_svg(tmp_path) -> None:
    payload = {
        "hardware_line": "test host",
        "note": "fixture",
        "runs": [
            {
                "scenario": "naive_vs_flux",
                "engine": "naive",
                "concurrency": 4,
                "aggregates": {
                    "tok_s": 1.0,
                    "req_s": 0.2,
                    "ttft_p50_ms": 80,
                    "ttft_p99_ms": 120,
                    "e2e_p50_ms": 400,
                    "e2e_p99_ms": 500,
                    "statuses": {"200": 6},
                },
            },
            {
                "scenario": "naive_vs_flux",
                "engine": "continuous",
                "concurrency": 4,
                "aggregates": {
                    "tok_s": 2.2,
                    "req_s": 0.4,
                    "ttft_p50_ms": 40,
                    "ttft_p99_ms": 55,
                    "e2e_p50_ms": 180,
                    "e2e_p99_ms": 220,
                    "statuses": {"200": 6},
                },
            },
        ],
        "story": {"text": "p99 TTFT cut 54%."},
    }
    path = write_report(payload, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "naive" in text
    assert "continuous" in text
    assert (tmp_path / "bench_ttft_p99.svg").exists()
    assert (tmp_path / "bench_tok_s.svg").exists()
    svg = bar_chart_svg("demo", [("a", 1.0), ("b", 2.0)], "x")
    assert "svg" in svg
