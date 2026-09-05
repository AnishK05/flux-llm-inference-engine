export type StreamCallbacks = {
  onId?: (id: string) => void;
  onDelta?: (text: string) => void;
  onUsage?: (tokens: number) => void;
  onFlux?: (info: { ttft_ms?: number; tpot_ms?: number; engine?: string }) => void;
};

function parseSseBlock(block: string): Record<string, unknown> | "done" | null {
  const line = block
    .split("\n")
    .map((row) => row.trim())
    .find((row) => row.startsWith("data:"));
  if (!line) return null;
  const payload = line.slice(5).trim();
  if (payload === "[DONE]") return "done";
  try {
    return JSON.parse(payload) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function readChatStream(response: Response, callbacks: StreamCallbacks): Promise<void> {
  if (!response.body) {
    throw new Error("no response body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const event = parseSseBlock(part);
      if (!event || event === "done") continue;
      if (typeof event.id === "string") callbacks.onId?.(event.id);
      const choices = (event.choices as Array<Record<string, unknown>>) || [];
      const delta = (choices[0]?.delta as Record<string, unknown> | undefined) || {};
      const content = typeof delta.content === "string" ? delta.content : "";
      if (content) callbacks.onDelta?.(content);
      const usage = event.usage as { completion_tokens?: number } | undefined;
      if (usage?.completion_tokens != null) callbacks.onUsage?.(usage.completion_tokens);
      const flux = event.flux as { ttft_ms?: number; tpot_ms?: number; engine?: string } | undefined;
      if (flux) callbacks.onFlux?.(flux);
    }
  }
}
