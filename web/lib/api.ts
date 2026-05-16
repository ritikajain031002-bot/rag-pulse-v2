import type { ChatTurn, ConfigInfo, SSEEvent } from "./types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchConfig(): Promise<ConfigInfo> {
  const r = await fetch(`${API_URL}/api/config`, { cache: "no-store" });
  if (!r.ok) throw new Error(`config: ${r.status}`);
  return r.json();
}

export async function newSession(): Promise<{ session_id: string }> {
  const r = await fetch(`${API_URL}/api/session/new`, { method: "POST" });
  if (!r.ok) throw new Error(`session: ${r.status}`);
  return r.json();
}

export async function deleteSession(sid: string): Promise<void> {
  await fetch(`${API_URL}/api/session/${sid}`, { method: "DELETE" });
}

export async function uploadFile(sessionId: string, file: File) {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("file", file);
  const r = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function ingestUrl(sessionId: string, url: string) {
  const r = await fetch(`${API_URL}/api/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, url }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function* streamChat(opts: {
  session_id: string | null;
  question: string;
  model: string;
  history?: ChatTurn[];
  signal?: AbortSignal;
}): AsyncGenerator<SSEEvent, void, unknown> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${t}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          const payload = line.slice(6).trim();
          if (!payload) continue;
          try {
            yield JSON.parse(payload) as SSEEvent;
          } catch {
            // malformed SSE line — ignore and continue parsing
          }
        }
      }
    }
  }
}
