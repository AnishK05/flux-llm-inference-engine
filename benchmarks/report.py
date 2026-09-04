"""Write docs/benchmark_results.md and SVG plots from a loadgen payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def bar_chart_svg(title: str, rows: list[tuple[str, float]], ylabel: str) -> str:
    width, height = 720, 320
    left, right, top, bottom = 70, 24, 40, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [max(0.0, v) for _, v in rows] or [0.0]
    peak = max(values) or 1.0
    bar_w = plot_w / max(len(rows), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2:.0f}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{_esc(title)}</text>',
        f'<text x="16" y="{top + plot_h/2:.0f}" transform="rotate(-90 16,{top + plot_h/2:.0f})" '
        f'text-anchor="middle" font-family="sans-serif" font-size="12">{_esc(ylabel)}</text>',
    ]
    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
    for i, (label, value) in enumerate(rows):
        h = (value / peak) * plot_h
        x = left + i * bar_w + bar_w * 0.15
        y = top + plot_h - h
        w = bar_w * 0.7
        color = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + w/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{_esc(label)}</text>'
        )
        parts.append(
            f'<text x="{x + w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{value:.2f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_report(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = payload.get("runs") or []
    hardware = payload.get("hardware_line") or ""
    note = payload.get("note") or ""

    md: list[str] = [
        "# Flux benchmark results",
        "",
        "Phase 8 closed-loop loadgen. Numbers are **measured on the host that ran the command**, "
        "not invented. Re-run on the Windows + WSL2 laptop for resume figures.",
        "",
        f"**Hardware:** {hardware}",
        "",
    ]
    if note:
        md.extend([note, ""])

    md.extend(
        [
            "How to read this:",
            "",
            "1. **TTFT / TPOT** — naive (Phase 1) vs Flux continuous (Phase 5) on `long_prompt` or `naive_vs_flux`.",
            "2. **Throughput** — queued vs continuous, or naive vs Flux, at concurrency 4–8.",
            "3. **`soak_200`** — control plane only. Do not quote soak e2e p99 as the latency win.",
            "",
            "| Scenario | Engine | Conc | tok/s | req/s | p50 TTFT ms | p99 TTFT ms | p50 e2e ms | p99 e2e ms | statuses |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    plot_ttft: list[tuple[str, float]] = []
    plot_toks: list[tuple[str, float]] = []
    for run in runs:
        agg = run.get("aggregates") or {}
        label = f"{run.get('engine')}/{run.get('scenario')}@{run.get('concurrency')}"
        ttft = agg.get("ttft_p99_ms")
        toks = agg.get("tok_s")
        if ttft is not None:
            plot_ttft.append((label, float(ttft)))
        if toks is not None:
            plot_toks.append((label, float(toks)))
        md.append(
            "| {scenario} | {engine} | {conc} | {tok:.3f} | {req:.3f} | {p50} | {p99} | {e50} | {e99} | {st} |".format(
                scenario=run.get("scenario"),
                engine=run.get("engine"),
                conc=run.get("concurrency"),
                tok=float(agg.get("tok_s") or 0),
                req=float(agg.get("req_s") or 0),
                p50=_fmt(agg.get("ttft_p50_ms")),
                p99=_fmt(agg.get("ttft_p99_ms")),
                e50=_fmt(agg.get("e2e_p50_ms")),
                e99=_fmt(agg.get("e2e_p99_ms")),
                st=agg.get("statuses"),
            )
        )

    md.extend(["", "## Plots", ""])
    if plot_ttft:
        svg = bar_chart_svg("p99 TTFT (ms)", plot_ttft, "ms")
        (out_dir / "bench_ttft_p99.svg").write_text(svg, encoding="utf-8")
        md.append("![p99 TTFT](bench_ttft_p99.svg)")
        md.append("")
    if plot_toks:
        svg = bar_chart_svg("Output tokens / s", plot_toks, "tok/s")
        (out_dir / "bench_tok_s.svg").write_text(svg, encoding="utf-8")
        md.append("![throughput](bench_tok_s.svg)")
        md.append("")

    story = payload.get("story") or {}
    if story:
        md.extend(["## Resume wording (measured)", "", story.get("text", ""), ""])

    path = out_dir / "benchmark_results.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="benchmarks/last_run.json")
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()
    payload = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    path = write_report(payload, Path(args.out_dir))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
