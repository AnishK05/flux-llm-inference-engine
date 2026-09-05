"use client";

import { useState } from "react";
import { FLUX_API } from "@/lib/api";
import { readChatStream } from "@/lib/sse";

export function Playground() {
  const [prompt, setPrompt] = useState("Explain KV cache in two sentences.");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(64);
  const [stream, setStream] = useState(true);
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [requestId, setRequestId] = useState("—");
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [tokens, setTokens] = useState(0);
  const [engine, setEngine] = useState("—");

  async function generate() {
    setBusy(true);
    setError("");
    setOutput("");
    setRequestId("—");
    setTtftMs(null);
    setTokens(0);
    setEngine("—");
    const started = performance.now();
    try {
      const response = await fetch(`${FLUX_API}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "flux-qwen-0.5b",
          messages: [{ role: "user", content: prompt }],
          temperature,
          max_tokens: maxTokens,
          stream,
        }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`${response.status} ${detail}`);
      }
      if (stream) {
        let counted = 0;
        await readChatStream(response, {
          onId: (id) => setRequestId(id),
          onDelta: (text) => {
            if (counted === 0) setTtftMs(performance.now() - started);
            counted += 1;
            setTokens(counted);
            setOutput((prev) => prev + text);
          },
          onUsage: (n) => setTokens(n),
          onFlux: (info) => {
            if (info.ttft_ms != null) setTtftMs(info.ttft_ms);
            if (info.engine) setEngine(info.engine);
          },
        });
      } else {
        const body = await response.json();
        setRequestId(body.id ?? "—");
        setOutput(body.choices?.[0]?.message?.content ?? "");
        setTokens(body.usage?.completion_tokens ?? 0);
        const headerTtft = response.headers.get("x-flux-ttft-ms");
        setTtftMs(headerTtft ? Number(headerTtft) : performance.now() - started);
        setEngine(response.headers.get("x-flux-engine") || "—");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_260px]">
      <section className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold">Playground</h1>
          <p className="text-sm text-zinc-400">Qwen2.5-0.5B-Instruct · default 64 tokens · stream on</p>
        </div>
        <label className="block text-xs uppercase tracking-wide text-zinc-500">Prompt</label>
        <textarea
          className="h-36 w-full rounded border border-zinc-800 bg-zinc-950 p-3 font-mono text-sm"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
        />
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-sm text-zinc-300">
            Temperature {temperature.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
              className="mt-1 block w-40"
            />
          </label>
          <label className="text-sm text-zinc-300">
            Max tokens
            <input
              type="number"
              min={1}
              max={64}
              value={maxTokens}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
              className="mt-1 block w-24 rounded border border-zinc-800 bg-zinc-950 px-2 py-1"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-zinc-300">
            <input type="checkbox" checked={stream} onChange={(event) => setStream(event.target.checked)} />
            Stream
          </label>
          <button
            type="button"
            onClick={generate}
            disabled={busy}
            className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy ? "Generating…" : "Generate"}
          </button>
        </div>
        {error ? <div className="text-sm text-red-400">{error}</div> : null}
        <pre className="min-h-40 whitespace-pre-wrap rounded border border-zinc-800 bg-zinc-950 p-3 font-mono text-sm">
          {output || (busy ? "…" : "Reply appears here.")}
        </pre>
      </section>
      <aside className="space-y-3 rounded border border-zinc-800 bg-zinc-950 p-4 text-sm">
        <div className="text-xs uppercase tracking-wide text-zinc-500">Request</div>
        <Row label="Model" value="flux-qwen-0.5b" />
        <Row label="Request id" value={requestId} mono />
        <Row label="TTFT" value={ttftMs == null ? "—" : `${ttftMs.toFixed(1)} ms`} />
        <Row label="Tokens" value={String(tokens)} />
        <Row label="Engine" value={engine} />
      </aside>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={mono ? "break-all font-mono text-xs text-zinc-200" : "text-zinc-100"}>{value}</div>
    </div>
  );
}
