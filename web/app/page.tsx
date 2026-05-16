"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Chat } from "@/components/Chat";
import type { ConfigInfo, IngestedItem, Message } from "@/lib/types";
import { deleteSession, fetchConfig, newSession } from "@/lib/api";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [ingested, setIngested] = useState<IngestedItem[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const cfg = await fetchConfig();
        setConfig(cfg);
        setSelectedModel(cfg.default_model);
        const s = await newSession();
        setSessionId(s.session_id);
      } catch (e: any) {
        setBootError(e.message || String(e));
      }
    })();
  }, []);

  const clearAll = async () => {
    if (sessionId) await deleteSession(sessionId).catch(() => {});
    setMessages([]);
    setIngested([]);
    try {
      const s = await newSession();
      setSessionId(s.session_id);
    } catch {}
  };

  if (bootError) {
    return (
      <main className="h-screen w-screen flex items-center justify-center p-8">
        <div className="glass rounded-2xl p-6 max-w-md text-center">
          <div className="text-2xl mb-2">⚠️</div>
          <h2 className="text-lg font-semibold mb-1">Backend unreachable</h2>
          <p className="text-sm text-white/60 mb-3">
            Couldn&apos;t reach <code className="font-mono text-xs">{process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}</code>
          </p>
          <p className="text-xs text-white/40 font-mono">{bootError}</p>
          <p className="text-xs text-white/50 mt-4">
            Start the FastAPI backend:
            <br />
            <code className="font-mono text-[11px] block mt-1 bg-black/40 p-2 rounded">
              cd /Users/apple/Desktop/Rag
              <br />
              source .venv/bin/activate
              <br />
              uvicorn api.main:app --port 8000
            </code>
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        config={config}
        sessionId={sessionId}
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        ingested={ingested}
        onClearAll={clearAll}
      />
      <Chat
        sessionId={sessionId}
        model={selectedModel}
        messages={messages}
        setMessages={setMessages}
        onIngested={(item) => setIngested((prev) => [...prev, item])}
      />
    </main>
  );
}
