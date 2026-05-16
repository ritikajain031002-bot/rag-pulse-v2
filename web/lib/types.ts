export type MessageRole = "user" | "assistant";

export type Source = {
  kind: string;
  filename: string;
  page?: number | null;
  page_end?: number | null;
  score?: number | null;
  rerank_score?: number | null;
  text: string;
  session_id?: string | null;
};

export type Quality = "HIGH" | "MEDIUM" | "LOW";

export type Stage = "retrieving" | "reranking" | "generating" | "done";

export type Timings = {
  retrieve_ms?: number;
  rerank_ms?: number;
  total_ms?: number;
};

export type Message = {
  role: MessageRole;
  content: string;
  reasoning?: string;
  sources?: Source[];
  stage?: Stage;
  candidates?: number;
  model?: string;
  timings?: Timings;
  quality?: Quality;
  top_score?: number;
  warning?: string;
  error?: string;
  streaming?: boolean;
  id?: string;
};

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type IngestedItem = {
  id: string;
  filename: string;
  kind: string;
  emoji: string;
  chunks: number;
  chars?: number;
  pages?: number | null;
  duration?: number | null;
  language?: string | null;
};

export type ConfigInfo = {
  default_model: string;
  embed_model: string;
  rerank_model: string;
  vision_model: string;
  ctx_model: string;
  whisper_model: string;
  models: string[];
};

export type SSEEvent =
  | { type: "stage"; stage: Stage; candidates?: number }
  | { type: "meta"; sources: Source[]; timings: Timings; model: string; quality?: Quality; top_score?: number; warning?: string }
  | { type: "reasoning"; text: string }
  | { type: "content"; text: string }
  | { type: "error"; text: string }
  | { type: "done"; total_ms: number };
