"use client";

import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { Sparkles } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { Composer, type Pending } from "./Composer";
import type { ChatTurn, IngestedItem, Message } from "@/lib/types";
import { streamChat } from "@/lib/api";

interface Props {
  sessionId: string | null;
  model: string;
  messages: Message[];
  setMessages: Dispatch<SetStateAction<Message[]>>;
  onIngested: (item: IngestedItem) => void;
}

const SUGGESTIONS = [
  "Summarize the main ideas across all uploaded sources",
  "Describe what's in the image I uploaded",
  "Transcribe the audio and find the key quotes",
  "Compare the documents and list contradictions",
];

const newId = () =>
  (globalThis.crypto as any)?.randomUUID?.() ??
  Math.random().toString(36).slice(2);

function buildHistory(msgs: Message[]): ChatTurn[] {
  const out: ChatTurn[] = [];
  for (const m of msgs) {
    if (m.role === "user" && m.content) {
      out.push({ role: "user", content: m.content });
    } else if (m.role === "assistant" && m.content && !m.error) {
      out.push({ role: "assistant", content: m.content });
    }
  }
  return out.slice(-16);
}

export function Chat({
  sessionId,
  model,
  messages,
  setMessages,
  onIngested,
}: Props) {
  const [streaming, setStreaming] = useState(false);
  const [pending, setPending] = useState<Pending[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const dragDepth = useRef(0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const stop = () => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  const send = async (question: string) => {
    const q = question.trim();
    if (!q || streaming) return;

    const history = buildHistory(messages);
    const assistantId = newId();

    setMessages((prev) => [
      ...prev,
      { role: "user", content: q, id: `u-${assistantId}` },
      {
        role: "assistant",
        content: "",
        reasoning: "",
        sources: [],
        stage: "retrieving",
        model,
        id: assistantId,
        streaming: true,
      },
    ]);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStreaming(true);

    try {
      for await (const ev of streamChat({
        session_id: sessionId,
        question: q,
        model,
        history,
        signal: ctrl.signal,
      })) {
        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === assistantId);
          if (idx === -1) return prev;
          const next = prev.slice();
          const m = { ...next[idx] };
          switch (ev.type) {
            case "stage":
              m.stage = ev.stage;
              if (ev.candidates != null) m.candidates = ev.candidates;
              break;
            case "meta":
              m.sources = ev.sources;
              m.timings = ev.timings;
              m.model = ev.model;
              if (ev.quality) m.quality = ev.quality;
              if (typeof ev.top_score === "number") m.top_score = ev.top_score;
              if (ev.warning) m.warning = ev.warning;
              break;
            case "reasoning":
              m.reasoning = (m.reasoning || "") + ev.text;
              break;
            case "content":
              m.content = (m.content || "") + ev.text;
              break;
            case "error":
              m.error = ev.text;
              m.streaming = false;
              break;
            case "done":
              m.streaming = false;
              m.stage = "done";
              m.timings = { ...(m.timings || {}), total_ms: ev.total_ms };
              break;
          }
          next[idx] = m;
          return next;
        });
      }
    } catch (e: any) {
      if (e.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, streaming: false, stage: "done" as const }
              : m,
          ),
        );
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, error: e.message || String(e), streaming: false }
              : m,
          ),
        );
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const queueFiles = (files: FileList | File[] | null) => {
    if (!files) return;
    const arr = Array.from(files);
    if (arr.length === 0) return;
    setPending((p) => [
      ...p,
      ...arr.map((f) => ({
        id: newId(),
        name: f.name,
        kind: "file" as const,
        file: f,
        status: "queued" as const,
      })),
    ]);
  };

  const onDragEnter = (e: React.DragEvent) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    dragDepth.current += 1;
    setDragOver(true);
  };
  const onDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
  };
  const onDragLeave = () => {
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setDragOver(false);
    }
  };
  const onDrop = (e: React.DragEvent) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    dragDepth.current = 0;
    setDragOver(false);
    if (e.dataTransfer.files?.length) queueFiles(e.dataTransfer.files);
  };

  return (
    <div
      className="flex-1 flex flex-col h-screen min-w-0 relative"
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {dragOver && (
        <div className="absolute inset-0 z-30 m-3 rounded-3xl border-2 border-dashed border-nvidia/70 bg-nvidia/10 backdrop-blur-md flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-4xl mb-2">📥</div>
            <div className="text-base text-nvidia font-semibold">
              Drop to attach
            </div>
            <div className="text-[11px] text-white/55 mt-1">
              PDF · image · audio · video · doc · text · code
            </div>
          </div>
        </div>
      )}

      <header className="h-16 px-6 flex items-center justify-between border-b border-white/5 backdrop-blur-md">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold tracking-tight leading-none">
            <span className="bg-gradient-to-r from-nvidia via-cyan to-electric bg-clip-text text-transparent">
              RAG Pulse
            </span>
          </h1>
          <p className="text-[11px] text-white/40 mt-1">
            Multi-modal · conversation memory · streaming retrieval-augmented answers
          </p>
        </div>
        <div className="text-[11px] text-white/40 font-mono truncate ml-4">
          {model}
        </div>
      </header>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 md:px-6 py-6"
        style={{
          maskImage:
            "linear-gradient(180deg, transparent 0, black 32px, black calc(100% - 16px), transparent 100%)",
        }}
      >
        <div className="max-w-3xl mx-auto space-y-5">
          {messages.length === 0 ? (
            <EmptyState onPick={(q) => send(q)} />
          ) : (
            messages.map((m, i) => <ChatMessage key={m.id || i} message={m} />)
          )}
        </div>
      </div>

      <div className="border-t border-white/5 p-4 backdrop-blur-md">
        <div className="max-w-3xl mx-auto">
          <Composer
            sessionId={sessionId}
            streaming={streaming}
            pending={pending}
            setPending={setPending}
            onSend={send}
            onAbort={stop}
            onIngested={onIngested}
            queueFiles={queueFiles}
          />
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="text-center py-16 md:py-24">
      <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl glass mb-5 glow-nvidia">
        <Sparkles className="w-6 h-6 text-nvidia" />
      </div>
      <h2 className="text-3xl font-semibold tracking-tight mb-2">
        Ask anything.
      </h2>
      <p className="text-white/50 text-sm max-w-md mx-auto leading-relaxed">
        Drag a file anywhere, paste from your clipboard, or tap 📎 in the input.
        PDFs, images, audio, video, office docs and web URLs all work.
        Conversation memory keeps context across turns.
      </p>
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl mx-auto text-[13px]">
        {SUGGESTIONS.map((t) => (
          <button
            key={t}
            onClick={() => onPick(t)}
            className="glass rounded-xl p-3 text-left text-white/70 hover:text-white hover:border-white/20 transition"
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
