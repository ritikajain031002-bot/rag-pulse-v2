"use client";

import { useState } from "react";
import { Check, Trash2 } from "lucide-react";
import type { ConfigInfo, IngestedItem } from "@/lib/types";

type Tone = "nvidia" | "cyan" | "electric";

const TONE_DOT: Record<Tone, string> = {
  nvidia: "bg-nvidia animate-pulse-glow",
  cyan: "bg-cyan",
  electric: "bg-electric",
};

const MODEL_LABEL: Record<string, { label: string; tag: string; tone: Tone }> = {
  "meta/llama-3.3-70b-instruct": {
    label: "Llama 3.3 70B",
    tag: "Fast · 0.6s TTFT",
    tone: "nvidia",
  },
  "mistralai/mistral-nemotron": {
    label: "Mistral Nemotron",
    tag: "Fastest · 0.5s TTFT",
    tone: "cyan",
  },
  "qwen/qwen3-next-80b-a3b-instruct": {
    label: "Qwen3-Next 80B",
    tag: "Fast",
    tone: "nvidia",
  },
  "qwen/qwen3-coder-480b-a35b-instruct": {
    label: "Qwen3 Coder 480B",
    tag: "Code-tuned",
    tone: "cyan",
  },
  "qwen/qwen3.5-122b-a10b": {
    label: "Qwen3.5 122B",
    tag: "Reasoning",
    tone: "electric",
  },
  "nvidia/llama-3.3-nemotron-super-49b-v1.5": {
    label: "Nemotron Super 49B",
    tag: "Reasoning",
    tone: "electric",
  },
  "nvidia/nemotron-3-super-120b-a12b": {
    label: "Nemotron-3 Super 120B",
    tag: "Reasoning + Answer",
    tone: "electric",
  },
};

interface Props {
  config: ConfigInfo | null;
  sessionId: string | null;
  selectedModel: string;
  onSelectModel: (m: string) => void;
  ingested: IngestedItem[];
  onClearAll: () => void;
}

export function Sidebar({
  config,
  sessionId,
  selectedModel,
  onSelectModel,
  ingested,
  onClearAll,
}: Props) {
  const [showModelPicker, setShowModelPicker] = useState(false);
  const current = MODEL_LABEL[selectedModel] || {
    label: selectedModel || "—",
    tag: "",
    tone: "nvidia" as Tone,
  };

  return (
    <aside className="w-[300px] md:w-[320px] flex-shrink-0 border-r border-white/5 flex flex-col h-screen">
      <div className="h-16 px-5 flex items-center gap-2.5 border-b border-white/5">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-nvidia via-cyan to-electric flex items-center justify-center text-black font-black text-base shadow-lg shadow-nvidia/20">
          R
        </div>
        <div className="leading-tight">
          <div className="font-semibold text-sm">RAG Pulse</div>
          <div className="text-[10px] text-white/40">
            Multi-modal · NVIDIA NIM
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        <Section label="Model">
          <button
            onClick={() => setShowModelPicker((v) => !v)}
            className="w-full glass rounded-xl p-3 text-left hover:bg-white/[0.06] transition flex items-center justify-between gap-2"
          >
            <div className="min-w-0">
              <div className="text-sm font-medium truncate">{current.label}</div>
              <div className="text-[10px] text-white/50 truncate">
                {current.tag}
              </div>
            </div>
            <div className={`w-2 h-2 rounded-full ${TONE_DOT[current.tone]}`} />
          </button>
          {showModelPicker && config && (
            <div className="mt-2 glass rounded-xl p-1 space-y-0.5">
              {config.models.map((m) => {
                const lbl = MODEL_LABEL[m] || {
                  label: m,
                  tag: "",
                  tone: "nvidia" as Tone,
                };
                const active = selectedModel === m;
                return (
                  <button
                    key={m}
                    onClick={() => {
                      onSelectModel(m);
                      setShowModelPicker(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs hover:bg-white/[0.06] transition flex items-center justify-between gap-2 ${active ? "bg-white/[0.08]" : ""}`}
                  >
                    <div className="min-w-0">
                      <div className="font-medium truncate">{lbl.label}</div>
                      <div className="text-[10px] text-white/40 truncate">
                        {lbl.tag}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${TONE_DOT[lbl.tone]}`}
                      />
                      {active && <Check className="w-3 h-3 text-nvidia" />}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </Section>

        <Section
          label={`Active context (${ingested.length})`}
          right={
            ingested.length > 0 ? (
              <button
                onClick={onClearAll}
                className="text-[10px] text-white/40 hover:text-red-400 flex items-center gap-1 transition"
              >
                <Trash2 className="w-3 h-3" />
                Clear
              </button>
            ) : null
          }
        >
          {ingested.length === 0 ? (
            <div className="glass rounded-xl p-4 text-center">
              <div className="text-[11px] text-white/45">No files yet.</div>
              <div className="text-[10px] text-white/30 mt-1 leading-relaxed">
                Drop, paste, or tap 📎 in the chat input
                <br />to add PDFs, images, audio, video,
                <br />docs, or URLs.
              </div>
            </div>
          ) : (
            <div className="space-y-1.5">
              {ingested.map((item) => (
                <div key={item.id} className="glass rounded-lg p-2.5 text-xs">
                  <div className="flex items-start gap-2">
                    <div className="text-base leading-none mt-0.5">
                      {item.emoji}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate text-white/90">
                        {item.filename}
                      </div>
                      <div className="text-[10px] text-white/40 mt-0.5 flex flex-wrap items-center gap-x-1.5">
                        <span>{item.chunks} chunks</span>
                        {item.pages != null && (
                          <>
                            <span>·</span>
                            <span>{item.pages} pages</span>
                          </>
                        )}
                        {item.duration != null && (
                          <>
                            <span>·</span>
                            <span>{Math.round(item.duration)}s</span>
                          </>
                        )}
                        {item.language && (
                          <>
                            <span>·</span>
                            <span>{item.language}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>
      </div>

      <div className="border-t border-white/5 p-3 space-y-1 text-[10px] text-white/35 font-mono">
        <div className="truncate">session {sessionId?.slice(0, 8) || "…"}</div>
        {config?.embed_model && (
          <div className="truncate">
            embed · {config.embed_model.split("/").pop()}
          </div>
        )}
        {config?.rerank_model && (
          <div className="truncate">
            rerank · {config.rerank_model.split("/").pop()}
          </div>
        )}
      </div>
    </aside>
  );
}

function Section({
  label,
  right,
  children,
}: {
  label: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-[10px] uppercase tracking-wider text-white/40 font-medium">
          {label}
        </label>
        {right}
      </div>
      {children}
    </div>
  );
}
