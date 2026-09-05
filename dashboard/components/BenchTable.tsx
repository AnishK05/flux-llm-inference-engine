"use client";

import { useEffect, useState } from "react";
import { fetchBench, type BenchPayload, type BenchRun } from "@/lib/api";

function num(value: number | null | undefined, digits = 2): string {
  if (value == null) return "—";
  return value.toFixed(digits);
}

export function BenchTable() {
  const [payload, setPayload] = useState<BenchPayload | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchBench()
      .then(setPayload)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const runs = payload?.runs ?? [];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Bench</h1>
        <p className="text-sm text-zinc-400">Renders <code>benchmarks/last_run.json</code> via <code>/admin/bench</code>.</p>
      </div>
      {error ? <div className="text-sm text-amber-400">{error}</div> : null}
      {payload?.hardware_line ? <p className="text-sm text-zinc-300">{payload.hardware_line}</p> : null}
      {payload?.note ? <p className="text-xs text-zinc-500">{payload.note}</p> : null}
      <div className="overflow-x-auto rounded border border-zinc-800">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-zinc-900 text-xs uppercase tracking-wide text-zinc-400">
            <tr>
              <th className="px-3 py-2">Scenario</th>
              <th className="px-3 py-2">Engine</th>
              <th className="px-3 py-2">Conc</th>
              <th className="px-3 py-2">tok/s</th>
              <th className="px-3 py-2">p50 TTFT</th>
              <th className="px-3 py-2">p99 TTFT</th>
              <th className="px-3 py-2">p99 e2e</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run: BenchRun, i: number) => (
              <tr key={`${run.engine}-${run.scenario}-${i}`} className="border-t border-zinc-800">
                <td className="px-3 py-2">{run.scenario}</td>
                <td className="px-3 py-2 font-mono">{run.engine}</td>
                <td className="px-3 py-2">{run.concurrency}</td>
                <td className="px-3 py-2">{num(run.aggregates?.tok_s)}</td>
                <td className="px-3 py-2">{num(run.aggregates?.ttft_p50_ms, 1)}</td>
                <td className="px-3 py-2">{num(run.aggregates?.ttft_p99_ms, 1)}</td>
                <td className="px-3 py-2">{num(run.aggregates?.e2e_p99_ms, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {payload?.story?.text ? (
        <p className="rounded border border-zinc-800 bg-zinc-950 p-3 text-sm text-zinc-200">{payload.story.text}</p>
      ) : null}
      <p className="text-xs text-zinc-500">Do not quote soak_200 e2e p99 as a latency win. Re-run on the WSL2 laptop for resume numbers.</p>
    </div>
  );
}
