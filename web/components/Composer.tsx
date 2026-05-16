"use client";

import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import {
  AlertCircle,
  Check,
  FileText,
  Globe,
  Link2,
  Loader2,
  Paperclip,
  Send,
  Square,
  X,
} from "lucide-react";
import clsx from "clsx";
import { ingestUrl, uploadFile } from "@/lib/api";
import type { IngestedItem } from "@/lib/types";

export type PendingStatus = "queued" | "uploading" | "done" | "error";

export type Pending = {
  id: string;
  name: string;
  kind: "file" | "url";
  file?: File;
  url?: string;
  status: PendingStatus;
  error?: string;
  result?: IngestedItem;
};

interface Props {
  sessionId: string | null;
  streaming: boolean;
  pending: Pending[];
  setPending: Dispatch<SetStateAction<Pending[]>>;
  onSend: (question: string) => void;
  onAbort: () => void;
  onIngested: (item: IngestedItem) => void;
  queueFiles: (files: FileList | File[] | null) => void;
}

const newId = () =>
  (globalThis.crypto as any)?.randomUUID?.() ??
  Math.random().toString(36).slice(2);

export function Composer({
  sessionId,
  streaming,
  pending,
  setPending,
  onSend,
  onAbort,
  onIngested,
  queueFiles,
}: Props) {
  const [input, setInput] = useState("");
  const [showUrl, setShowUrl] = useState(false);
  const [url, setUrl] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const pendingRef = useRef(pending);
  useEffect(() => {
    pendingRef.current = pending;
  }, [pending]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
  }, [input]);

  useEffect(() => {
    if (!sessionId) return;
    const queued = pending.filter((p) => p.status === "queued");
    if (queued.length === 0) return;

    setPending((prev) =>
      prev.map((p) =>
        p.status === "queued" ? { ...p, status: "uploading" } : p,
      ),
    );

    queued.forEach(async (item) => {
      try {
        const res =
          item.kind === "file" && item.file
            ? await uploadFile(sessionId, item.file)
            : item.kind === "url" && item.url
              ? await ingestUrl(sessionId, item.url)
              : null;
        if (!res) throw new Error("nothing to upload");
        const ingestedItem: IngestedItem = { id: newId(), ...res };
        setPending((prev) =>
          prev.map((p) =>
            p.id === item.id
              ? { ...p, status: "done", result: ingestedItem }
              : p,
          ),
        );
        onIngested(ingestedItem);
      } catch (e: any) {
        setPending((prev) =>
          prev.map((p) =>
            p.id === item.id
              ? { ...p, status: "error", error: e.message || String(e) }
              : p,
          ),
        );
      }
    });
  }, [pending, sessionId, setPending, onIngested]);

  const remove = (id: string) => setPending((p) => p.filter((x) => x.id !== id));

  const queueUrl = () => {
    const u = url.trim();
    if (!u) return;
    setPending((p) => [
      ...p,
      {
        id: newId(),
        name: u.replace(/^https?:\/\//, ""),
        kind: "url",
        url: u,
        status: "queued",
      },
    ]);
    setUrl("");
    setShowUrl(false);
  };

  const submit = async () => {
    if (streaming) return;
    const q = input.trim();
    if (!q && pending.length === 0) return;

    if (
      pendingRef.current.some(
        (p) => p.status === "uploading" || p.status === "queued",
      )
    ) {
      for (let i = 0; i < 150; i++) {
        await new Promise((r) => setTimeout(r, 200));
        const stillBusy = pendingRef.current.some(
          (p) => p.status === "uploading" || p.status === "queued",
        );
        if (!stillBusy) break;
      }
    }

    if (!q) {
      setPending((p) => p.filter((x) => x.status === "error"));
      return;
    }

    setInput("");
    setPending((p) => p.filter((x) => x.status === "error"));
    onSend(q);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onPaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const files: File[] = [];
    for (const it of Array.from(items)) {
      if (it.kind === "file") {
        const f = it.getAsFile();
        if (f) files.push(f);
      }
    }
    if (files.length > 0) {
      e.preventDefault();
      queueFiles(files);
    }
  };

  const busyCount = pending.filter(
    (p) => p.status === "uploading" || p.status === "queued",
  ).length;
  const canSend = !!input.trim() && !!sessionId && !streaming;

  return (
    <div>
      {showUrl && (
        <div className="mb-2 glass rounded-xl p-2 flex gap-2 items-center">
          <Globe className="w-3.5 h-3.5 text-cyan ml-1.5 flex-shrink-0" />
          <input
            type="url"
            autoFocus
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                queueUrl();
              }
              if (e.key === "Escape") {
                setShowUrl(false);
                setUrl("");
              }
            }}
            placeholder="https://example.com  — Enter to add"
            className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-white/30"
          />
          <button
            onClick={queueUrl}
            disabled={!url.trim()}
            className="text-[11px] px-2.5 py-1 rounded-lg bg-nvidia/80 hover:bg-nvidia text-black font-semibold disabled:opacity-40"
          >
            Add
          </button>
          <button
            onClick={() => {
              setShowUrl(false);
              setUrl("");
            }}
            aria-label="Close URL input"
            className="w-6 h-6 flex items-center justify-center text-white/40 hover:text-white"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div className="glass rounded-2xl flex flex-col focus-within:border-nvidia/40 focus-within:ring-1 focus-within:ring-nvidia/20 transition">
        {pending.length > 0 && (
          <div className="flex flex-wrap gap-1.5 px-3 pt-3">
            {pending.map((p) => (
              <PendingChip
                key={p.id}
                item={p}
                onRemove={() => remove(p.id)}
              />
            ))}
          </div>
        )}
        <textarea
          ref={taRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          onPaste={onPaste}
          placeholder={
            pending.length > 0
              ? "Ask something about the attached items…"
              : "Ask anything across your sources…"
          }
          rows={1}
          disabled={streaming}
          className="w-full bg-transparent px-5 pt-4 pb-2 resize-none focus:outline-none placeholder:text-white/30 max-h-56 text-[15px]"
        />
        <div className="flex items-center justify-between px-3 pb-3 gap-2">
          <div className="flex items-center gap-1 min-w-0">
            <button
              onClick={() => fileRef.current?.click()}
              disabled={!sessionId}
              title="Attach files (PDF · image · audio · video · doc)"
              aria-label="Attach files"
              className="w-9 h-9 rounded-xl text-white/55 hover:text-white hover:bg-white/[0.06] disabled:opacity-30 flex items-center justify-center transition flex-shrink-0"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowUrl((v) => !v)}
              disabled={!sessionId}
              title="Add a URL"
              aria-label="Add URL"
              className={clsx(
                "w-9 h-9 rounded-xl hover:bg-white/[0.06] disabled:opacity-30 flex items-center justify-center transition flex-shrink-0",
                showUrl
                  ? "text-nvidia bg-white/[0.06]"
                  : "text-white/55 hover:text-white",
              )}
            >
              <Link2 className="w-4 h-4" />
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              className="hidden"
              onChange={(e) => {
                queueFiles(e.target.files);
                if (fileRef.current) fileRef.current.value = "";
              }}
            />
            {busyCount > 0 && (
              <div className="ml-1 inline-flex items-center gap-1 text-[10px] text-white/50 min-w-0">
                <Loader2 className="w-3 h-3 animate-spin flex-shrink-0" />
                <span className="truncate">
                  {busyCount} uploading…
                </span>
              </div>
            )}
          </div>
          {streaming ? (
            <button
              onClick={onAbort}
              aria-label="Stop"
              className="w-9 h-9 rounded-xl bg-red-500/80 hover:bg-red-500 flex items-center justify-center transition flex-shrink-0"
            >
              <Square
                className="w-3.5 h-3.5 text-white"
                fill="currentColor"
              />
            </button>
          ) : (
            <button
              onClick={submit}
              aria-label="Send"
              disabled={!canSend}
              className="w-9 h-9 rounded-xl bg-nvidia/90 hover:bg-nvidia disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition flex-shrink-0"
            >
              <Send className="w-4 h-4 text-black" />
            </button>
          )}
        </div>
      </div>
      <div className="text-[10px] text-white/30 mt-2 px-2 flex items-center justify-between">
        <span>
          Enter to send · Shift+Enter newline · drag, paste, or 📎 to attach
        </span>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-nvidia animate-pulse-glow" />
          {streaming
            ? "Streaming…"
            : busyCount > 0
              ? "Uploading…"
              : "Ready"}
        </span>
      </div>
    </div>
  );
}

function PendingChip({
  item,
  onRemove,
}: {
  item: Pending;
  onRemove: () => void;
}) {
  const leadIcon = (() => {
    if (item.status === "uploading" || item.status === "queued") {
      return <Loader2 className="w-3 h-3 animate-spin text-white/60" />;
    }
    if (item.status === "error") {
      return <AlertCircle className="w-3 h-3 text-red-400" />;
    }
    if (item.status === "done" && item.result?.emoji) {
      return (
        <span className="text-[12px] leading-none">{item.result.emoji}</span>
      );
    }
    return item.kind === "url" ? (
      <Globe className="w-3 h-3 text-cyan" />
    ) : (
      <FileText className="w-3 h-3 text-white/50" />
    );
  })();

  const subtitle =
    item.status === "done" && item.result
      ? `${item.result.chunks} chunks`
      : item.status === "uploading" || item.status === "queued"
        ? "processing…"
        : item.status === "error"
          ? "failed"
          : null;

  return (
    <div
      className={clsx(
        "inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] glass max-w-[240px]",
        item.status === "error" && "border-red-500/40 bg-red-500/5",
        item.status === "done" && "border-nvidia/30",
      )}
      title={item.error || item.name}
    >
      {leadIcon}
      <span className="truncate font-medium text-white/85">{item.name}</span>
      {subtitle && (
        <span className="text-white/40 whitespace-nowrap">· {subtitle}</span>
      )}
      <button
        onClick={onRemove}
        className="ml-0.5 text-white/40 hover:text-white flex-shrink-0"
        aria-label="Remove"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}
