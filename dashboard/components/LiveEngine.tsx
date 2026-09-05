"use client";

import { useEffect, useState } from "react";
import { abortRequest, fetchStats, formatBytes, type EngineStats } from "@/lib/api";

const EMPTY: EngineStats = {
  waiting: 0,
  running: 0,
  in_flight: 0,
  max_waiting: 256,
  max_batch_size: 8,
  last_batch_size: 0,
  tokens_generated: 0,
  tok_s: 0,
  ttft_p50_ms: null,
  rss_bytes: 0,
  waiting_ids: [],
  running_ids: [],
  kv_blocks_used: 0,
  kv_blocks_free: 0,
  kv_blocks_total: 0,
  serve_engine: "—",
  scheduler: "—",
};

export function LiveEngine() {
  const [stats, setStats] = useState<EngineStats>(EMPTY);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function tick() {
      try {
        const next = await fetchStats(controller.signal);
        if (!cancelled) {
          setStats(next);
          setError("");
        }
      } catch (err) {
        if (!cancelled && !(err instanceof DOMException)) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    }

    void tick();
    const id = window.setInterval(() => void tick(), 500);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(id);
    };
  }, []);

  const kvPct = stats.kv_blocks_total ? (stats.kv_blocks_used / stats.kv_blocks_total) * 100 : 0;
  const waitPct = stats.max_waiting ? (stats.waiting / stats.max_waiting) * 100 : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Live engine</h1>
        <p className="text-sm text-zinc-400">
          Polls <code>/admin/stats</code> every 500ms. Queue depth is how 200 concurrent shows up.
        </p>
      </div>
      {error ? <div className="text-sm text-amber-400">API {error}. Is Flux on :8000?</div> : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Decode batch" value={String(stats.last_batch_size)} hint={`max ${stats.max_batch_size}`} huge />
        <Stat label="In flight" value={String(stats.in_flight)} hint={`${stats.waiting} waiting · ${stats.running} running`} huge />
        <Stat label="tok/s" value={stats.tok_s.toFixed(1)} hint={`${stats.tokens_generated} total`} />
        <Stat label="p50 TTFT" value={stats.ttft_p50_ms == null ? "—" : `${stats.ttft_p50_ms.toFixed(0)} ms`} hint="last 60s" />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded border border-zinc-800 bg-zinc-950 p-4">
          <div className="mb-2 flex justify-between text-sm">
            <span>KV pool</span>
            <span className="font-mono text-zinc-400">
              {stats.kv_blocks_used} / {stats.kv_blocks_total}
            </span>
          </div>
          <div className="bar-track h-3 overflow-hidden rounded">
            <div className="h-full bg-emerald-500" style={{ width: `${Math.min(100, kvPct)}%` }} />
          </div>
        </div>
        <div className="rounded border border-zinc-800 bg-zinc-950 p-4">
          <div className="mb-2 flex justify-between text-sm">
            <span>Wait queue</span>
            <span className="font-mono text-zinc-400">
              {stats.waiting} / {stats.max_waiting}
            </span>
          </div>
          <div className="bar-track h-3 overflow-hidden rounded">
            <div className="h-full bg-amber-500" style={{ width: `${Math.min(100, waitPct)}%` }} />
          </div>
          <div className="mt-3 text-sm text-zinc-400">RSS {formatBytes(stats.rss_bytes)}</div>
          <div className="text-xs text-zinc-500">
            {stats.serve_engine} · {stats.scheduler}
          </div>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <IdList title="Running batch" ids={stats.running_ids} tone="run" />
        <IdList title="Waiting" ids={stats.waiting_ids} tone="wait" />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
  huge,
}: {
  label: string;
  value: string;
  hint?: string;
  huge?: boolean;
}) {
  return (
    <div className="rounded border border-zinc-800 bg-zinc-950 p-4">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={huge ? "text-4xl font-semibold tabular-nums" : "text-2xl font-semibold tabular-nums"}>{value}</div>
      {hint ? <div className="text-xs text-zinc-500">{hint}</div> : null}
    </div>
  );
}

function IdList({ title, ids, tone }: { title: string; ids: string[]; tone: "run" | "wait" }) {
  return (
    <div className="rounded border border-zinc-800 bg-zinc-950 p-4">
      <div className="mb-2 text-sm text-zinc-300">
        {title} <span className="text-zinc-500">({ids.length})</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {ids.length === 0 ? <span className="text-xs text-zinc-600">none</span> : null}
        {ids.map((id) => (
          <button
            key={id}
            type="button"
            title="Abort"
            onClick={() => void abortRequest(id)}
            className={tone === "run" ? "chip" : "chip border-amber-800/70 bg-amber-950/50 text-amber-100"}
          >
            {id.slice(0, 8)}
          </button>
        ))}
      </div>
    </div>
  );
}
