# RAG Pulse — Run guide

Two processes, two terminals, plus Docker for Qdrant.

```
┌──────────────────────┐    HTTP/SSE    ┌──────────────────────┐    Qdrant
│ Next.js (port 3000)  │ ───────────►   │ FastAPI (port 8000)  │ ──────────►  Qdrant (port 6333)
│ web/                 │                │ api/main.py          │
└──────────────────────┘                └──────────────────────┘
```

## One-time setup

```bash
cd /Users/apple/Desktop/Rag

# Backend deps already installed in .venv
.venv/bin/pip install -r requirements.txt

# Frontend
cd web
cp .env.local.example .env.local
npm install        # or pnpm install / yarn
cd ..
```

## Start the stack (3 terminals)

**Terminal 1 — Qdrant vector DB**
```bash
cd /Users/apple/Desktop/Rag
docker compose up -d
```

**Terminal 2 — FastAPI backend** (pre-warms embed + rerank models on startup)
```bash
cd /Users/apple/Desktop/Rag
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Watch for the line `[lifespan] READY` before sending queries. Without that, the
first query will load Qwen models on demand and feel "stuck" for 30–60s.

**Terminal 3 — Next.js frontend**
```bash
cd /Users/apple/Desktop/Rag/web
npm run dev
```

Open http://localhost:3000

## What you'll see streaming live

When you ask a question, events arrive over Server-Sent Events in this order:

1. `stage: retrieving` → sidebar pulse + "Searching the corpus…" with shimmer
2. `stage: reranking · N candidates` → "Reranking 20 candidates…"
3. `meta` → sources panel populates instantly (before any token arrives)
4. `stage: generating` → "Model is thinking…" with bouncing dots
5. `reasoning` tokens → live preview in the purple **Model reasoning** panel (only for reasoning models like Qwen3.5 122B, Nemotron Super)
6. `content` tokens → main answer streams character-by-character with a blinking cursor
7. `done` → cursor disappears, timings appear: "1.42s total · retrieve 87ms · rerank 312ms"

If the LLM errors, you get a red **Model error** box instead of a frozen bubble.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness |
| GET | `/api/config` | Default model, embed/rerank/vision/whisper info, full model list |
| POST | `/api/session/new` | Returns `{session_id}` |
| DELETE | `/api/session/{sid}` | Wipe session-tagged chunks from Qdrant |
| POST | `/api/upload` (form) | `session_id` + `file` — any media kind, auto-detected |
| POST | `/api/url` (json) | `{session_id, url}` — fetches & indexes a web page |
| POST | `/api/chat` (json) | `{session_id, question, model}` → SSE stream |

## Pre-ingest a PDF corpus (hackathon spec)

```bash
cp /path/to/your/pdfs/*.pdf data/pdfs/
.venv/bin/python -m src.ingest --no-context    # fast path
# or with Anthropic Contextual Retrieval (slower, better recall):
.venv/bin/python -m src.ingest
```

These chunks live OUTSIDE any session — every chat query searches them
alongside the per-session uploads.

## Troubleshooting

- **Sidebar shows "Backend unreachable"** → start the FastAPI server (terminal 2)
- **"Generating…" stays forever** → check NVIDIA NIM key in `.env`; try a different
  model from the dropdown (Llama 3.3 70B is the most reliable). Errors will now
  show as a red box instead of silent stalling.
- **"Model is thinking" for 20s+ on first query** → reranker is loading.
  The `[lifespan] READY` line tells you when warmup finishes.
- **First Qwen3-Embedding load is slow** → 600M params downloaded from HF on
  first use, cached after.
