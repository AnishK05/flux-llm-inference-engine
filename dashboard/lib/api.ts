export const FLUX_API = process.env.NEXT_PUBLIC_FLUX_API || "http://localhost:8000";

export type EngineStats = {
  waiting: number;
  running: number;
  in_flight: number;
  max_waiting: number;
  max_batch_size: number;
  last_batch_size: number;
  tokens_generated: number;
  tok_s: number;
  ttft_p50_ms: number | null;
  rss_bytes: number;
  waiting_ids: string[];
  running_ids: string[];
  kv_blocks_used: number;
  kv_blocks_free: number;
  kv_blocks_total: number;
  serve_engine: string;
  scheduler: string;
};

export type BenchRun = {
  scenario?: string;
  engine?: string;
  concurrency?: number;
  aggregates?: {
    tok_s?: number;
    req_s?: number;
    ttft_p50_ms?: number | null;
    ttft_p99_ms?: number | null;
    e2e_p50_ms?: number | null;
    e2e_p99_ms?: number | null;
    statuses?: Record<string, number>;
  };
};

export type BenchPayload = {
  hardware_line?: string;
  note?: string;
  story?: { text?: string };
  runs?: BenchRun[];
};

export async function fetchStats(signal?: AbortSignal): Promise<EngineStats> {
  const response = await fetch(`${FLUX_API}/admin/stats`, { cache: "no-store", signal });
  if (!response.ok) {
    throw new Error(`stats ${response.status}`);
  }
  return response.json();
}

export async function fetchBench(): Promise<BenchPayload> {
  const response = await fetch(`${FLUX_API}/admin/bench`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`bench ${response.status}`);
  }
  return response.json();
}

export async function abortRequest(id: string): Promise<void> {
  await fetch(`${FLUX_API}/admin/abort/${id}`, { method: "POST" });
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return `${(n / 1024 ** 3).toFixed(2)} GiB`;
}
