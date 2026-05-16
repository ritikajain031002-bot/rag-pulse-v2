"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  FileText,
  Image as ImageIcon,
  Music,
  Video,
  Globe,
  Archive,
  FileBox,
  AlertCircle,
  Brain,
  Search,
  ListChecks,
  Cpu,
} from "lucide-react";
import type { Message, Quality, Stage } from "@/lib/types";

const KIND_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  pdf: FileText,
  image: ImageIcon,
  audio: Music,
  video: Video,
  web: Globe,
  archive: Archive,
  office: FileBox,
  text: FileText,
};

const STAGE_INFO: Record<Stage, { label: string; icon: React.ComponentType<{ className?: string }> }> = {
  retrieving: { label: "Searching the corpus", icon: Search },
  reranking: { label: "Reranking candidates", icon: ListChecks },
  generating: { label: "Model is thinking", icon: Cpu },
  done: { label: "Done", icon: Cpu },
};

const QUALITY_STYLE: Record<Quality, { label: string; cls: string; hint: string }> = {
  HIGH: {
    label: "High match",
    cls: "bg-emerald-500/15 text-emerald-300 border-emerald-400/30",
    hint: "Top excerpts strongly match your question.",
  },
  MEDIUM: {
    label: "Medium match",
    cls: "bg-amber-500/15 text-amber-200 border-amber-400/30",
    hint: "Excerpts are partially relevant; answer may be approximate.",
  },
  LOW: {
    label: "Low match",
    cls: "bg-rose-500/15 text-rose-200 border-rose-400/30",
    hint: "Your corpus doesn't seem to cover this question. Try uploading a more relevant file or asking something the docs do cover.",
  },
};

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${Math.round(ms / 100) * 100}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

export function ChatMessage({ message }: { message: Message }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const reasoningRef = useRef<HTMLDivElement>(null);

  // Per-stage elapsed-time tick (so a long rerank doesn't look frozen).
  // Resets to 0 every time message.stage changes.
  const [stageElapsed, setStageElapsed] = useState(0);
  const stageStartedAtRef = useRef<number | null>(null);
  const lastStageRef = useRef<Stage | null>(null);

  useEffect(() => {
    if (reasoningRef.current && message.streaming) {
      reasoningRef.current.scrollTop = reasoningRef.current.scrollHeight;
    }
  }, [message.reasoning, message.streaming]);

  useEffect(() => {
    if (!message.streaming || !message.stage || message.stage === "done") {
      setStageElapsed(0);
      stageStartedAtRef.current = null;
      lastStageRef.current = null;
      return;
    }
    if (message.stage !== lastStageRef.current) {
      lastStageRef.current = message.stage;
      stageStartedAtRef.current = Date.now();
      setStageElapsed(0);
    }
    const id = setInterval(() => {
      if (stageStartedAtRef.current) {
        setStageElapsed(Date.now() - stageStartedAtRef.current);
      }
    }, 120);
    return () => clearInterval(id);
  }, [message.streaming, message.stage]);

  const elapsedTone =
    stageElapsed > 30_000
      ? "text-red-400"
      : stageElapsed > 10_000
      ? "text-amber-300"
      : "text-white/55";

  if (message.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="flex justify-end"
      >
        <div className="max-w-[85%] glass-strong rounded-2xl rounded-br-md px-4 py-3 border-nvidia/30">
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
      </motion.div>
    );
  }

  const hasReasoning = message.reasoning && message.reasoning.length > 0;
  const reasoningOpen = showReasoning || (message.streaming && !!hasReasoning && !message.content);
  const StageIcon = message.stage ? STAGE_INFO[message.stage].icon : Cpu;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col gap-2"
    >
      {message.streaming && message.stage && message.stage !== "done" && (
        <div className="flex items-center gap-2 text-[12px] px-1">
          <span className="relative flex items-center justify-center w-5 h-5">
            <span className="absolute inset-0 rounded-full bg-nvidia/30 animate-ping" />
            <StageIcon className="w-3 h-3 text-nvidia relative z-10" />
          </span>
          <span className="shimmer-text font-medium">
            {STAGE_INFO[message.stage].label}
            {message.stage === "reranking" && message.candidates != null
              ? ` · ${message.candidates} candidates`
              : ""}
          </span>
          {stageElapsed > 200 && (
            <span className={`text-[11px] tabular-nums ${elapsedTone}`}>
              {formatElapsed(stageElapsed)}
            </span>
          )}
          <span className="thinking-dots text-nvidia">
            <span /><span /><span />
          </span>
        </div>
      )}

      {message.warning && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-xl border border-amber-400/40 bg-amber-500/10 text-amber-100/95 text-[12px] leading-snug">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-amber-300" />
          <span>{message.warning}</span>
        </div>
      )}

      {message.quality && (message.sources?.length ?? 0) > 0 && (
        <div className="flex flex-wrap items-center gap-2 px-1">
          <span
            className={`px-2 py-0.5 rounded-full border text-[10.5px] font-medium tabular-nums ${
              QUALITY_STYLE[message.quality].cls
            }`}
            title={QUALITY_STYLE[message.quality].hint}
          >
            ● {QUALITY_STYLE[message.quality].label}
            {message.top_score != null ? ` · ${message.top_score.toFixed(2)}` : ""}
          </span>
          {message.quality === "LOW" && !message.streaming && (
            <span className="text-[10.5px] text-rose-200/80 leading-snug">
              {QUALITY_STYLE.LOW.hint}
            </span>
          )}
        </div>
      )}

      {hasReasoning && (
        <div className="glass rounded-2xl overflow-hidden border-electric/30">
          <button
            onClick={() => setShowReasoning((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-white/[0.03] transition"
          >
            <div className="flex items-center gap-2 text-xs">
              <Brain className="w-3.5 h-3.5 text-electric" />
              <span className="text-electric font-medium">
                {message.streaming && message.stage === "generating"
                  ? "Thinking…"
                  : "Model reasoning"}
              </span>
              <span className="text-white/30">
                {(message.reasoning || "").length} chars
              </span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-white/40 transition-transform ${reasoningOpen ? "rotate-180" : ""}`}
            />
          </button>
          <AnimatePresence initial={false}>
            {reasoningOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div
                  ref={reasoningRef}
                  className="px-4 pb-3 pt-1 text-[12px] text-white/65 font-mono max-h-52 overflow-y-auto leading-relaxed whitespace-pre-wrap"
                >
                  {message.reasoning}
                  {message.streaming && !message.content && (
                    <span className="streaming-cursor" />
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {(message.content ||
        (!message.streaming && !message.error)) && (
        <div className="glass rounded-2xl rounded-tl-md px-5 py-4">
          {message.content ? (
            <div className="prose-chat">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {message.streaming && <span className="streaming-cursor" />}
            </div>
          ) : hasReasoning ? (
            <div className="text-amber-200/90 text-sm space-y-1">
              <p className="font-medium">
                ⚠️ Model produced reasoning but no final answer.
              </p>
              <p className="text-white/55 text-[12px] leading-relaxed">
                The token budget ran out mid-thought. Open the Model reasoning
                panel above to see what it was working on, or retry — thinking
                models now get a larger ceiling.
              </p>
            </div>
          ) : (
            <p className="text-white/40 text-sm italic">No answer produced.</p>
          )}
        </div>
      )}

      {message.error && (
        <div className="glass rounded-2xl px-4 py-3 border-red-500/40 bg-red-500/5 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-red-300 break-all">{message.error}</p>
        </div>
      )}

      {!message.streaming && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-white/40 px-1">
          {message.model && (
            <span className="font-mono truncate max-w-[260px]">
              {message.model}
            </span>
          )}
          {message.timings?.total_ms != null && (
            <>
              <span>·</span>
              <span>{(message.timings.total_ms / 1000).toFixed(2)}s total</span>
            </>
          )}
          {message.timings?.retrieve_ms != null && (
            <>
              <span>·</span>
              <span>
                retrieve {message.timings.retrieve_ms}ms · rerank{" "}
                {message.timings.rerank_ms}ms
              </span>
            </>
          )}
          {message.sources && message.sources.length > 0 && (
            <button
              onClick={() => setShowSources((v) => !v)}
              className="ml-auto flex items-center gap-1 hover:text-white/80 transition"
            >
              <span>{message.sources.length} sources</span>
              <ChevronDown
                className={`w-3 h-3 transition-transform ${showSources ? "rotate-180" : ""}`}
              />
            </button>
          )}
        </div>
      )}

      <AnimatePresence initial={false}>
        {showSources && message.sources && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-2">
              {message.sources.map((src, i) => {
                const Icon = KIND_ICON[src.kind] || FileText;
                return (
                  <div
                    key={i}
                    className="glass rounded-xl p-3 text-xs hover:border-white/15 transition"
                  >
                    <div className="flex items-start gap-2 mb-2">
                      <Icon className="w-3.5 h-3.5 text-nvidia mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-white/85 truncate font-medium">
                          {src.filename}
                        </div>
                        <div className="text-[10px] text-white/40 mt-0.5">
                          {src.page != null && `page ${src.page}`}
                          {src.page_end != null && src.page_end !== src.page
                            ? `–${src.page_end}`
                            : ""}
                          {src.score != null && ` · score ${src.score.toFixed(3)}`}
                        </div>
                      </div>
                    </div>
                    <p className="text-white/55 leading-relaxed line-clamp-3">
                      {src.text}
                    </p>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
