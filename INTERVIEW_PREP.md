# RAG Pulse — Interview Preparation Guide

> A complete, deep-dive reference for explaining this Universal RAG Chatbot project in any interview. Every file, every line of logic, every design decision, plus 120+ cross questions with detailed answers.

---

## Table of Contents

1. [30-Second Elevator Pitch](#1-30-second-elevator-pitch)
2. [Core Project Summary (the "what")](#2-core-project-summary-the-what)
3. [Why This Project is Impressive (the "why")](#3-why-this-project-is-impressive-the-why)
4. [Complete Folder Structure — Every File Explained](#4-complete-folder-structure--every-file-explained)
5. [Architecture & End-to-End Data Flow](#5-architecture--end-to-end-data-flow)
6. [Tech Stack — Every Library, Why It's There](#6-tech-stack--every-library-why-its-there)
7. [Deep Dive: Each Module's Internal Logic](#7-deep-dive-each-modules-internal-logic)
8. [Key Concepts You MUST Know Cold](#8-key-concepts-you-must-know-cold)
9. [Numbers You Should Memorize](#9-numbers-you-should-memorize)
10. [Interview Q&A — Easy (Warm-up)](#10-interview-qa--easy-warm-up)
11. [Interview Q&A — Medium (Core)](#11-interview-qa--medium-core)
12. [Interview Q&A — Hard (Cross-questions, "Why this not that?")](#12-interview-qa--hard-cross-questions-why-this-not-that)
13. [System Design / Scaling Questions](#13-system-design--scaling-questions)
14. [Bugs / Gotchas / Trade-offs You Should Mention](#14-bugs--gotchas--trade-offs-you-should-mention)
15. [Likely Follow-up "What if?" Questions](#15-likely-follow-up-what-if-questions)
16. [Final Cheat Sheet (last-5-minutes review)](#16-final-cheat-sheet-last-5-minutes-review)

---

## 1. 30-Second Elevator Pitch

> "I built a production-grade RAG (Retrieval-Augmented Generation) chatbot called **RAG Pulse**. Users can drop **any** file — PDF, image, audio, video, Office docs, code, ZIP archives — or paste a URL, and the system auto-routes it through the right pipeline (Whisper for audio, vision LLM for images, ffmpeg for video, PyMuPDF + Tesseract OCR for PDFs). Everything ends up as text, gets chunked with Anthropic's Contextual Retrieval technique, stored in Qdrant as both dense and BM25-sparse vectors, then queried with **hybrid RRF search → cross-encoder rerank → streaming LLM generation with citations**. The frontend is Next.js with Server-Sent Events showing live retrieve/rerank/generate stages. Sub-5-second answers, fully grounded with `[filename, page]` citations."

---

## 2. Core Project Summary (the "what")

**Name:** RAG Pulse — Universal Chat Edition
**Type:** Production-grade Retrieval-Augmented Generation chatbot
**Domain:** Multi-modal document Q&A
**Stack:** Python (FastAPI) backend + Next.js 16 / React 19 frontend + Qdrant vector DB + NVIDIA NIM hosted LLMs

**What problem it solves:**
Standard RAG systems handle one format (usually PDF). Real users have messy inputs: scanned PDFs, screenshots, voice memos, video lectures, code archives, web pages. This system accepts **all of them** in one chat surface, with automatic routing.

**Three pillars:**
1. **Universal ingestion** — 60+ file extensions auto-detected and processed
2. **Hybrid retrieval** — Dense embeddings (Qwen3-0.6B) + BM25 sparse, fused with Reciprocal Rank Fusion
3. **Streaming UX** — Live SSE events show every stage; first token under 1 second for fast models

---

## 3. Why This Project is Impressive (the "why")

When asked "Why should we hire you?" or "Why is this project impressive?":

- **Production concerns covered:** Pre-warmed models, batch upserts, payload indexing for O(log n) deletes, session isolation, graceful fallbacks (Qwen → BGE if download fails), token-budget management
- **Latest 2024-2025 techniques implemented:**
  - Anthropic's Contextual Retrieval (Sept 2024) → 49% retrieval-failure reduction
  - Qwen3-Embedding-0.6B (MTEB 64.34, top-tier)
  - Hybrid BM25 + dense with RRF (industry SOTA)
  - Sigmoid-normalized rerank scores → quality tiering (HIGH/MEDIUM/LOW)
- **Real engineering rigor:**
  - Streaming via Server-Sent Events (not just `print()`)
  - Quality-aware system prompt (different behavior for LOW relevance)
  - Coreference resolution via conversation history
  - Multi-turn memory (last 16 turns)
- **Eval-driven:** RAGAS metrics (faithfulness, answer relevancy, context precision/recall) with naive baseline comparison

---

## 4. Complete Folder Structure — Every File Explained

```
Rag/
├── README.md                    # Project overview, supported formats, quick-start
├── RUN.md                       # 3-terminal run guide (Qdrant + FastAPI + Next.js)
├── requirements.txt             # 30+ Python deps, grouped by concern
├── docker-compose.yml           # Spins up Qdrant on ports 6333 (REST) + 6334 (gRPC)
├── .env / .env.example          # NVIDIA API key, model overrides, Whisper config
├── .gitignore                   # Excludes .env, .venv, __pycache__, PDFs, eval snapshots
├── smoke_test.py                # End-to-end NIM connectivity test (key + models + streaming)
├── probe_kimi.py                # Diagnostic: probe Kimi-K2.6 streaming behavior
├── probe_stream.py              # Diagnostic: stream test for any NIM model
│
├── data/
│   ├── pdfs/                    # Drop pre-ingest PDFs here (used by `python -m src.ingest`)
│   └── sessions/                # Per-session uploaded file storage (cleared on session delete)
│
├── src/                         # ── Backend Python core ──
│   ├── __init__.py
│   ├── config.py                # CFG singleton — every env var, model name, chunk size, etc.
│   ├── parser.py                # PyMuPDF text extraction + Tesseract OCR fallback for scanned PDFs
│   ├── chunker.py               # Token-aware sliding window (500 tokens, 10% overlap) using tiktoken
│   ├── contextualizer.py        # Anthropic Contextual Retrieval — LLM prepends 80-word context to each chunk
│   ├── embedder.py              # Qwen3-Embedding-0.6B (primary) → BGE-small ONNX (fallback)
│   ├── vector_store.py          # Qdrant wrapper: dense+sparse named vectors, RRF hybrid query
│   ├── reranker.py              # BAAI/bge-reranker-v2-m3 cross-encoder with sigmoid normalization
│   ├── generator.py             # NVIDIA NIM streaming LLM call + quality-aware system prompt
│   ├── pipeline.py              # Glue: hybrid_search → dedupe → rerank → stream_answer
│   ├── ingest.py                # CLI: batch-ingest data/pdfs/ folder
│   ├── session.py               # Per-user session tagging + cleanup of indexed chunks
│   ├── media_router.py          # File-type detection + dispatcher (PDF/image/audio/video/office/etc.)
│   ├── multimodal_llm.py        # Vision LLM streaming (image_url + text content blocks)
│   └── processors/
│       ├── pdf_doc.py           # Wraps parser.py for the router
│       ├── image.py             # Vision LLM exhaustive description (used for OCR + scene understanding)
│       ├── audio.py             # faster-whisper local transcription with VAD filter
│       ├── video.py             # ffmpeg → audio (whisper) + N keyframes (vision) → combined text
│       ├── office.py            # MarkItDown → markdown (docx/pptx/xlsx/epub) with native fallbacks
│       ├── text.py              # Plain text + 60 code/config extensions, chardet encoding detection
│       ├── web.py               # requests + BeautifulSoup with tag stripping (script/style/nav/etc.)
│       └── archive.py           # zip/tar(.gz/.bz2) recursive extract + per-member routing
│
├── api/                         # ── FastAPI HTTP layer ──
│   ├── __init__.py
│   └── main.py                  # Endpoints: /health, /config, /session/new, /upload, /url, /chat (SSE)
│
├── app/                         # ── Legacy Streamlit demo UI (still works) ──
│   ├── __init__.py
│   ├── streamlit_app.py         # Single-file Streamlit app, uses RAGPipeline directly
│   └── components/              # Streamlit helper widgets
│
├── web/                         # ── Next.js 16 / React 19 frontend ──
│   ├── package.json             # Next, React, Tailwind, framer-motion, react-markdown, lucide-react
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── postcss.config.mjs
│   ├── .env.local.example       # NEXT_PUBLIC_API_URL=http://localhost:8000
│   ├── app/
│   │   ├── layout.tsx           # Root layout (HTML shell, fonts, metadata)
│   │   ├── page.tsx             # Main page: boots session, manages model + messages state
│   │   └── globals.css          # Tailwind directives + glass morphism utilities
│   ├── components/
│   │   ├── Chat.tsx             # Message list + streaming consumer (calls streamChat generator)
│   │   ├── ChatMessage.tsx      # Per-message renderer: markdown, sources, reasoning panel
│   │   ├── Composer.tsx         # Input box + file upload trigger
│   │   └── Sidebar.tsx          # Model picker, session controls, ingested-file list
│   └── lib/
│       ├── api.ts               # Fetch wrappers + SSE async generator (streamChat)
│       └── types.ts             # TS types mirroring backend Pydantic models
│
├── eval/                        # ── RAGAS evaluation harness ──
│   ├── questions.jsonl          # Labeled Q&A pairs (one JSON per line)
│   └── run_ragas.py             # Runs pipeline on each question, computes 4 RAGAS metrics
│
├── uploads/                     # Session-uploaded files (per-session subdirs)
│
├── .streamlit/                  # Streamlit theme config
├── .sisyphus/                   # Project-internal task notes (gitignored)
├── .claude/                     # AI-coding-tool config (gitignored)
└── .venv/                       # Python virtualenv (gitignored)
```

### What each folder is "for" in one line

| Folder | Purpose |
|--------|---------|
| `src/` | The brain — all RAG logic, no HTTP or UI code |
| `src/processors/` | Format-specific adapters (Strategy pattern) |
| `api/` | Thin HTTP shell — exposes `src/` over FastAPI |
| `app/` | Legacy Streamlit single-file demo (proof you can ship a UI in 200 lines) |
| `web/` | The "real" frontend — Next.js streaming chat |
| `eval/` | Quality measurement (RAGAS metrics) |
| `data/` | Sample inputs + per-session uploads |
| `uploads/` | Temp storage for files coming through the FastAPI `/api/upload` endpoint |

---

## 5. Architecture & End-to-End Data Flow

### High-level diagram
```
USER drops file or types question
        │
        ▼
┌──────────────────────────────┐
│  Next.js (port 3000)         │  React 19, Tailwind, SSE consumer
└──────────────────────────────┘
        │  HTTP (multipart upload / JSON)
        ▼
┌──────────────────────────────┐
│  FastAPI (port 8000)         │  api/main.py
│   • /api/upload  (form)      │  ← saves file → media_router.process_path → session.add_to_session
│   • /api/chat    (SSE)       │  ← hybrid_search → rerank → stream_answer
└──────────────────────────────┘
        │  (1) embed + upsert       (2) hybrid query + rerank
        ▼                              ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│  Qdrant (port 6333)          │    │  NVIDIA NIM (cloud)          │
│   collection "hackathon_rag" │    │   • ctx model (8B)           │
│   • dense vec (1024-d)       │    │   • gen model (70B-1T)       │
│   • sparse vec (BM25 IDF)    │    │   • vision (Llama 3.2 90B)   │
└──────────────────────────────┘    └──────────────────────────────┘
```

### Ingestion flow (write path)

1. **User uploads `report.pdf`** → FastAPI `/api/upload` saves it to `uploads/{session_id}_{uuid}.pdf`
2. **`media_router.detect()`** → extension says `.pdf` → returns `"pdf"`
3. **`media_router.process_path()`** → routes to `processors/pdf_doc.process()`
4. **`parser.parse_pdf()`** → PyMuPDF extracts text per page; if text density < 100 chars (i.e. scanned), falls back to `pdf2image` + Tesseract OCR
5. **`chunker.chunk_pages()`** → uses `tiktoken` (cl100k_base) to slide a 500-token window with 10% overlap, preserving `(pdf, page_start, page_end)` metadata
6. **`session.add_to_session()`** → tags each chunk with `session_id`, `kind="pdf"`, `filename`. Session uploads skip contextualization for speed; offline `ingest.py` does add context
7. **`embedder.embed()`** → Qwen3-Embedding-0.6B encodes the context_text into 1024-d float vectors
8. **`vector_store.upsert()`** → wraps each chunk into a Qdrant `PointStruct` with two named vectors (`dense` = float list, `sparse` = `models.Document(text, model="Qdrant/bm25")`). Batches of 64. Qdrant computes the BM25 sparse vector server-side.
9. Frontend shows `📄 report.pdf — 24 chunks` in the sidebar

### Query flow (read path)

1. **User asks "What's the warranty period?"** → POST `/api/chat` with `{session_id, question, model}`
2. **SSE event:** `{type:'stage', stage:'retrieving'}`
3. **`hybrid_search()`** → embeds query (with Qwen3 instruction prefix), sends Qdrant a `query_points` with two prefetches (dense + sparse, each fetching `top_k * 2 = 24`) fused via `Fusion.RRF` (Reciprocal Rank Fusion), returns top 12
4. **`_dedupe()`** → drops near-duplicate chunks (same file/page, same first 160 chars stripped)
5. **SSE event:** `{type:'stage', stage:'reranking', candidates:12}`
6. **`rerank()`** → BAAI/bge-reranker-v2-m3 cross-encoder scores each (query, chunk_text[:800]) pair; scores sigmoid-normalized to [0,1]; returns top 4
7. **`_quality_tier()`** → top rerank score ≥ 0.65 → `HIGH`; ≥ 0.30 → `MEDIUM`; else `LOW`
8. **SSE event:** `{type:'meta', sources:[...], timings:{...}, quality:'HIGH'}` — sidebar populates *before* generation starts
9. **SSE event:** `{type:'stage', stage:'generating'}`
10. **`stream_answer()`** → builds messages with quality-aware system prompt + last 16 history turns + user prompt with `[1] score=X.XX pdf=foo.pdf page=3-3\n<text>\n[2] ...`; calls NVIDIA NIM with `stream=True`
11. Per-token SSE events: `{type:'reasoning', text:'...'}` (if model emits CoT) and `{type:'content', text:'...'}`
12. **SSE event:** `{type:'done', total_ms:1420}` — UI shows "1.42s total · retrieve 87ms · rerank 312ms"

---

## 6. Tech Stack — Every Library, Why It's There

### Python (backend)

| Library | Version | Role | Why this one |
|---------|---------|------|--------------|
| `qdrant-client[fastembed]` | ≥1.14.2 | Vector DB client + built-in BM25 sparse encoder | Qdrant is Rust-based, Apache-2.0, supports named vectors + sparse natively in one collection |
| `sentence-transformers` | ≥3.0.0 | Loads Qwen3-Embedding-0.6B and cross-encoder reranker | Trusted, exposes `.encode()` and `CrossEncoder` cleanly |
| `fastembed` | ≥0.5.0 | ONNX-runtime fallback embeddings (BGE-small) | If Qwen download fails on a fresh box, we still boot |
| `torch` + `torchvision` | ≥2.2 / ≥0.17 | Required by sentence-transformers under the hood | Auto-uses CUDA if available, falls back to CPU |
| `transformers` | ≥4.45 | Backbone for sentence-transformers + Qwen tokenizer | Standard HF stack |
| `pymupdf` (fitz) | ≥1.24 | Native PDF text extraction | 10-100x faster than `pdfminer`, handles complex layouts |
| `pytesseract` + `pdf2image` | ≥0.3.10 / ≥1.17 | OCR fallback for scanned PDFs | When text density < 100 chars/page, fall back to Tesseract |
| `Pillow` + `pillow-heif` | ≥10.0 / ≥0.18 | Image loading, including HEIC/HEIF (iPhone photos) | iPhone screenshots are HEIC by default |
| `filetype` | ≥1.2 | Magic-byte sniffing for extension-less files | More robust than `mimetypes` |
| `faster-whisper` | ≥1.0.3 | Local Whisper transcription via CTranslate2 | 4x faster than OpenAI Whisper, runs on CPU |
| `chardet` | ≥5.2 | Auto-detect text encoding | Handles latin-1, GB2312, etc. without crashing |
| `beautifulsoup4` + `requests` | ≥4.12 / ≥2.32 | Web page fetch + HTML cleanup | Strips script/style/nav/footer before extracting text |
| `markitdown` | ≥0.0.1a3 | Microsoft's universal doc → markdown converter | Handles docx/pptx/xlsx/epub/rtf in one API |
| `python-docx`, `python-pptx`, `openpyxl` | latest | Native fallbacks for Office docs | Used when MarkItDown returns empty |
| `openai` | ≥1.50 | OpenAI-compatible client for NVIDIA NIM | NIM speaks OpenAI's `/chat/completions` protocol |
| `tenacity` | ≥8.5 | Retries with exponential backoff | Wraps `situate_chunk` to survive NIM 429s |
| `tiktoken` | ≥0.7 | Token counting for chunking | `cl100k_base` matches GPT-4 tokenizer |
| `langdetect` | ≥1.0.9 | Language detection (currently unused but reserved) | For future per-language reranker selection |
| `tqdm` | ≥4.66 | Progress bars | Used during batch ingest |
| `fastapi` + `uvicorn` | ≥0.115 / ≥0.32 | ASGI server | Async, lifespan events, native SSE via StreamingResponse |
| `python-multipart` | ≥0.0.12 | FastAPI multipart/form-data parsing | Needed for `UploadFile` |
| `sse-starlette` | ≥2.1 | (Imported but `StreamingResponse` works fine) | Optional, kept for reference |
| `ragas` + `datasets` | ≥0.2 / ≥2.18 | RAG evaluation framework | 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall |
| `langchain-openai` + `langchain-huggingface` | ≥0.2 / ≥0.1 | RAGAS needs LLM + embedder shims | Plumbing for `evaluate()` call |
| `streamlit` + extras | ≥1.36 | Legacy demo UI | Single-file, good for hackathon demo |
| `pydantic` | ≥2.7 | Validation for `CFG` and FastAPI request models | v2 is dramatically faster than v1 |
| `python-dotenv` | ≥1.0.1 | Loads `.env` into env vars | Standard 12-factor practice |

### JavaScript (frontend)

| Package | Role |
|---------|------|
| `next` (^16.2) | React framework — App Router, RSC, image opt |
| `react` (^19.2) | Latest stable React (Server Components, `use` hook) |
| `react-markdown` (9.0) + `remark-gfm` (4.0) | Renders LLM markdown output (GitHub Flavored Markdown for tables, strikethrough) |
| `framer-motion` (^12.38) | Smooth stage transitions and message animations |
| `lucide-react` (0.451) | Icon set (file, copy, sparkles, etc.) |
| `clsx` (2.1.1) | Conditional className helper |
| `tailwindcss` (3.4.13) | Utility-first CSS |
| `typescript` (5.6.2) | Type safety; types mirror Pydantic models |

---

## 7. Deep Dive: Each Module's Internal Logic

### `src/config.py` — Configuration singleton
- Loads `.env` via `python-dotenv`
- `Config` is a Pydantic `BaseModel` (validation + IDE autocomplete)
- Exports a single `CFG = Config()` so any module just does `from .config import CFG`
- Key fields:
  - `dense_model = "Qwen/Qwen3-Embedding-0.6B"` (primary) with `fastembed` BGE-small as silent fallback
  - `sparse_model = "Qdrant/bm25"` — Qdrant computes BM25 itself
  - `rerank_model = "BAAI/bge-reranker-v2-m3"` (278M params, classification head — 5-10x faster on CPU than Qwen3-Reranker's autoregressive scoring)
  - `chunk_tokens = 500`, `chunk_overlap_pct = 0.10` → 50-token overlap
  - `top_k_retrieve = 12`, `top_k_rerank = 4` → keeps LLM prefill under ~1.5K tokens for fast TTFT
  - `ocr_text_density_threshold = 100` chars/page — below this we OCR
  - `gen_model_options` — hand-curated allowlist of NIM models that were actually probed and work

### `src/parser.py` — PDF parsing
- `parse_pdf()` yields `PageRecord(pdf, page, text, ocr_used)` per page
- For each page: try `page.get_text("text")` first (native PyMuPDF, fast)
- If `len(text.strip()) < 100` → assume scanned page → call `convert_from_path(dpi=200)` for that single page → `pytesseract.image_to_string()`
- `_clean()` removes hyphenation line breaks, collapses 3+ newlines to 2, collapses repeated spaces

### `src/chunker.py` — Token-aware chunking
- Flattens all pages' tokens into `[(page_num, token_id), ...]` so we can slide across page boundaries while still recording which pages a chunk spans
- Window size = 500 tokens, step = 500 - 50 = 450 (10% overlap)
- Chunk ID = `sha1(f"{pdf}:{i}:{text[:50]}")` — deterministic and unique
- Returns `Chunk(id, pdf, page_start, page_end, text)` — `page_end` ≠ `page_start` when chunk spans pages

### `src/contextualizer.py` — Anthropic Contextual Retrieval
- **The technique:** Before embedding each chunk, ask a small LLM "Given this whole document, write 80 words of context to situate this chunk." Prepend that context to the chunk's text, embed the combined string.
- **Why:** Embedding lone fragments loses pronouns, anaphora, section context. Prepending an LLM-generated summary recovers it. Anthropic reported 35% reduction in top-20 retrieval failure, 49% when combined with BM25.
- **Implementation:** OpenAI client pointed at NIM, model `meta/llama-3.1-8b-instruct` (cheap + fast), `max_tokens=120`, `temperature=0`, retried 3x with exponential backoff via `tenacity`
- **Doc window:** First 60K chars passed as the document context (truncated to stay under model's prefill budget)
- **Skip path:** Session uploads bypass this (`context_text == text`) to keep chat snappy

### `src/embedder.py` — Dense embeddings
- Tries `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)` first
- On failure: `fastembed.TextEmbedding("BAAI/bge-small-en-v1.5")` (384-d, ONNX, no PyTorch)
- Exports `DIM` so `vector_store.reset_collection()` can size the Qdrant collection correctly — this prevents a silent dimension mismatch on fallback
- `embed_query()` applies Qwen3's instruction prefix:
  ```
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query: <user question>
  ```
  This is what Qwen3 was instruction-tuned with — asymmetric embedding (queries embedded differently from documents) → measurable retrieval boost

### `src/vector_store.py` — Qdrant wrapper
- **Collection schema:**
  ```python
  vectors_config={"dense": VectorParams(size=DIM, distance=COSINE)}
  sparse_vectors_config={"sparse": SparseVectorParams(modifier=IDF)}
  hnsw_config=HnswConfigDiff(m=16, ef_construct=128)
  ```
- **`_chunk_uuid()`:** Hashes our SHA1 chunk_id into a UUID5 namespace because Qdrant requires UUID or int point IDs
- **Payload index:** `create_payload_index(field_name="session_id", schema=KEYWORD)` → makes `clear_session` deletions O(log n) instead of full-scan
- **Hybrid query:**
  ```python
  query_points(
      prefetch=[
          Prefetch(query=q_dense, using="dense", limit=top_k*2),
          Prefetch(query=Document(text=query, model="Qdrant/bm25"), using="sparse", limit=top_k*2),
      ],
      query=FusionQuery(fusion=Fusion.RRF),
      limit=top_k,
  )
  ```
  Server-side RRF. The Document for sparse means Qdrant tokenizes + BM25-scores the query on its side using the same model used at index time.

### `src/reranker.py` — Cross-encoder reranking
- **Why rerank?** Hybrid retrieval is a recall maximizer — it grabs 12-20 plausibly-relevant chunks. Cross-encoder scoring (which actually processes `(query, passage)` together in one transformer pass) is the precision maximizer, but is too slow to run on 1M docs. So: cheap retrieval → expensive precise rerank on top 12.
- **Model choice:** `BAAI/bge-reranker-v2-m3` (278M, classification head). The previous choice was `Qwen3-Reranker-0.6B` but it uses *autoregressive* scoring (generates "yes"/"no" token, takes logit diff) which is 5-10x slower on CPU — pushed total rerank time to 253s in one bad case.
- **Score normalization:** Sigmoid applied so output is in [0,1] — lets us define **fixed quality thresholds** that don't drift between rerank models
- **Batching:** `batch_size=32`, `max_length=256`, truncate passage to 800 chars before pairing — bounded latency

### `src/generator.py` — Answer generation
- **`_quality_tier()`:** Maps top rerank score → "HIGH" (≥0.65) / "MEDIUM" (≥0.30) / "LOW"
- **`SYSTEM_TEMPLATE`:** Quality is interpolated into the system prompt. When HIGH/MEDIUM, model uses strict citation format (TL;DR + per-source sections + `_Source: file, p.X_` line). When LOW, model is told to open with "These documents don't seem to cover your question directly", optionally answer from general knowledge with a "(General knowledge — not from your documents)" prefix.
- **History plumbing:** Last 16 turns added between system and user → multi-turn chat memory + coreference resolution
- **`build_user_prompt()`:** Formats EXCERPTS section: `[1] score=0.82 pdf=foo.pdf page=3-4\n<truncated text>\n[2] ...`
- **Stream parsing:** Yields `("reasoning", text)` for `delta.reasoning_content` (only models like Qwen3.5-122B, Nemotron-Super emit these) and `("content", text)` for the normal answer

### `src/pipeline.py` — Orchestration
- `_dedupe()` — Drops near-duplicates by `(first_160_chars_lowercased, page, "")` signature. Reason: users sometimes upload the same markdown twice with different filenames (e.g. `foo.md` and `sid_123_foo.md`) which crowds out top-K.
- `RAGPipeline.query()` returns a `RAGResult` dataclass with the answer stream, retrieved chunks, timings, quality, and top_score — so the legacy Streamlit UI can consume without touching FastAPI internals.

### `src/ingest.py` — CLI ingestion
- Reads all `data/pdfs/*.pdf`
- Calls `parse_pdf → chunk_pages → contextualize_chunks → upsert`
- Flags:
  - `--no-context` — skip contextualization (3-5x faster, lower quality)
  - `--no-reset` — keep existing collection (incremental ingest)
- Default: resets the collection and contextualizes everything

### `src/session.py` — Per-user session isolation
- `add_to_session(session_id, processed_dict)`:
  - Converts processed output (which can be PDF pages, whisper segments, or plain text) into a list of `PageRecord` so chunker can consume
  - Audio/video segments bucketed into 60-second "pages" → citations show `t≈MM:SS` granularity
  - Each chunk tagged with `session_id`, `kind`, `filename`
- `clear_session(session_id)`:
  - `delete(filter=Filter(must=[FieldCondition(key="session_id", match=...)]))`
  - O(log n) thanks to the payload index

### `src/media_router.py` — Format dispatcher
- **Six extension sets:** image (11), audio (11), video (11), text (60+!), office (11), archive (6)
- **`detect()`:** Extension-first (fast). If unknown extension → `filetype.guess()` sniffs magic bytes. If still unknown → falls through to `"text"` (read with `errors='replace'`)
- **`process_path()`:** **Lazy imports** the right processor — keeps cold-start fast (we don't pay the Whisper import cost unless a user uploads audio)

### `src/multimodal_llm.py` — Vision streaming
- `_data_url()` — Base64-encodes image as `data:image/jpeg;base64,...` so we don't need a CDN
- `stream_vision()` — Sends OpenAI-compatible vision payload: `[{type:"text", text:prompt}, {type:"image_url", image_url:{url:data_url}}]`
- `describe_image()` — One-shot exhaustive description prompt: "scene + ALL visible text (OCR every word) + objects + charts/data points + mood/style". This becomes the chunkable text for that image.

### `src/processors/` (one file per format)
- **`pdf_doc.py`** — Wraps `parser.parse_pdf` and returns the standard processor dict (`kind`, `source`, `filename`, `text`, `pages`, `page_count`, `ocr_pages`)
- **`image.py`** — Registers HEIF opener (iPhone photos) → calls `describe_image`
- **`audio.py`** — Lazy-singletons `WhisperModel(base, cpu, int8)`, streams `(start, end, text)` segments with VAD filter
- **`video.py`** — `ffmpeg` extracts mono 16-kHz wav (Whisper's preferred input) and N keyframes (`fps=1/60`, capped at 8). Audio → Whisper, frames → vision LLM. Output combines transcript header + body + visual keyframe descriptions.
- **`office.py`** — MarkItDown first (handles 90% of cases); falls back to native `python-docx`/`python-pptx`/`openpyxl` per extension if MarkItDown produces empty text
- **`text.py`** — UTF-8 try, then `chardet` detect, then latin-1 with `errors='replace'`. Never crashes on encoding.
- **`web.py`** — `requests.get(timeout=30)` with custom UA, `BeautifulSoup` strip of `[script, style, nav, footer, header, aside, noscript]`, then `get_text(separator="\n", strip=True)`
- **`archive.py`** — Takes `route_fn` as a parameter (dependency injection to avoid circular import with `media_router`), extracts to a temp dir, recursively routes each file

### `api/main.py` — FastAPI HTTP layer
- **`lifespan`** context manager — pre-warms embedder + reranker on startup. Without this, the first query in a fresh process loads Qwen weights (~30-60s of "stuck") → user thinks it's broken.
- **CORS:** `allow_origins=["*"]` for demo; comment says "tighten in prod"
- **`UPLOAD_DIR`:** Saves files as `{session_id}_{uuid}.{ext}` to avoid filename collisions
- **`/api/chat`** SSE format — each event is `data: <json>\n\n` (one blank line terminator, JSON-encoded). Events:
  - `{type:'stage', stage:'retrieving'}`
  - `{type:'stage', stage:'reranking', candidates:N}`
  - `{type:'meta', sources, timings, model, quality, top_score, [warning]}`
  - `{type:'stage', stage:'generating'}`
  - `{type:'reasoning', text}` (zero or more)
  - `{type:'content', text}` (many)
  - `{type:'done', total_ms}`
  - `{type:'error', text}` (terminal)
- **`_resolve_model()`** — Stale browser tab may POST a deprecated model name → silently fall back to default and surface a `warning` field so the user knows why their picker was overridden
- **Headers for SSE:** `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no` (disables nginx buffering), `Connection: keep-alive`

### `web/` — Next.js frontend
- **`app/page.tsx`** (client component): Bootstraps by `fetchConfig()` + `newSession()`, holds `messages` state, renders `<Sidebar>` and `<Chat>`. Shows a "Backend unreachable" fallback if FastAPI isn't running.
- **`components/Chat.tsx`**: Calls the async-generator `streamChat()` and updates the in-flight assistant message on every SSE event. Renders `<ChatMessage>` for each turn.
- **`components/ChatMessage.tsx`**: Renders markdown via `react-markdown + remark-gfm`, shows reasoning panel collapsible, source pills with rerank scores, quality badge.
- **`components/Composer.tsx`**: Textarea + file input + send button. Triggers upload then question.
- **`components/Sidebar.tsx`**: Model picker (from `config.models`), session info, ingested files with emoji + chunk count, "Clear all" button.
- **`lib/api.ts`**: `fetchConfig`, `newSession`, `deleteSession`, `uploadFile`, `ingestUrl`, and the magic `streamChat()` async generator that parses SSE chunks: reads `response.body.getReader()`, decodes chunks, splits on `\n\n`, strips `data: ` prefix, JSON.parses, yields one event at a time.
- **`lib/types.ts`**: Mirrors backend Pydantic models for compile-time safety.

---

## 8. Key Concepts You MUST Know Cold

### 8.1 What is RAG?
**Retrieval-Augmented Generation** = combining an external knowledge base (your documents) with a generative LLM. Instead of relying on the LLM's training data (which is stale + can't be private), you:
1. **Index** your documents into a searchable form (embeddings + vector DB)
2. **Retrieve** the most relevant snippets for the user's question
3. **Augment** the LLM's prompt with those snippets ("Here are some excerpts. Answer based on them.")
4. **Generate** the answer with citations

**Why RAG over fine-tuning?**
- Cheap to update (re-index, no GPU training)
- Citable / explainable (you can show the retrieved chunks)
- Private (your docs never leave your infra in the truly private setup)
- LLM stays general (no catastrophic forgetting)

### 8.2 Embeddings
A function `f: text → vector ∈ R^d` (here d=1024 for Qwen3) such that **semantically similar texts have geometrically close vectors** (high cosine similarity). Trained on millions of (query, relevant_passage) pairs.

### 8.3 BM25 (Sparse retrieval)
- Classic IR algorithm (1994). Computes `tf-idf`-like score per term, with **term-frequency saturation** (`k1` parameter) and **document-length normalization** (`b` parameter).
- Excellent at **exact keyword match** — finds "GSTIN 27ABCDE1234F1Z5" when dense embeddings would soften the digits.
- Sparse vector = dictionary of `{token_id: weight}` — most positions zero.

### 8.4 Hybrid retrieval & RRF
- **Why both?** Dense = semantic recall (finds paraphrases). Sparse = lexical precision (finds rare tokens, names, IDs). They fail on different queries.
- **Reciprocal Rank Fusion:** Each retriever returns a ranked list. The fused score for doc d is `Σ_retriever 1 / (k + rank(d))` (typically k=60). Rank-based → no need to normalize wildly different score scales.

### 8.5 Cross-encoder reranker
- Embeddings encode `query` and `passage` **separately** → fast at scale but lossy (no interaction between query and passage tokens).
- Cross-encoder runs `(query, passage)` **together** through a transformer → much better precision but `O(N)` forward passes (one per candidate). Use only after retrieval has narrowed to ~20 candidates.

### 8.6 Chunking strategy
- **Too small** → loses surrounding context, citations fragment.
- **Too big** → multiple topics in one chunk, embedding becomes "average of everything", retrieval precision drops.
- **Sweet spot for general docs:** 200-800 tokens with 10-20% overlap. We use 500 / 10%.
- **Page-anchored metadata:** Chunk knows `(page_start, page_end)` so citations can point exactly.

### 8.7 Contextual Retrieval (Anthropic, Sept 2024)
**Problem:** Chunks lose context. "Their revenue was $50M" — whose? When?
**Solution:** For each chunk, ask a small LLM "Situate this chunk in the document in 80 words." Prepend that context to the chunk before embedding.
**Result:** Anthropic measured 35% fewer top-20 retrieval failures (49% with BM25 also).
**Cost:** 1 LLM call per chunk at ingest time (offline, batched).

### 8.8 Server-Sent Events (SSE)
- One-way HTTP streaming protocol. `Content-Type: text/event-stream`. Each event: `data: {...}\n\n`.
- Simpler than WebSockets (no upgrade handshake, no bidirectional). Perfect for streaming LLM tokens to a browser.
- Native browser support via `EventSource`, but we use `fetch` + a manual parser to send POST bodies (EventSource is GET-only).

### 8.9 Quality tiering
Static "you must cite" prompts produce hallucinations when retrieval returns garbage (low scores). Instead:
- Sigmoid-normalize rerank scores → fixed thresholds → 3 tiers
- HIGH/MEDIUM → strict citation behavior
- LOW → "These documents don't cover your question" preamble, optional general-knowledge fallback with explicit label
- This is what stops the "model hallucinates a confident wrong answer from off-topic excerpts" failure mode.

### 8.10 Session isolation
Every chunk has a `session_id` payload field. `clear_session` deletes only that session's chunks via a Qdrant filter delete — the pre-ingested global corpus is preserved. Indexed payload field makes this O(log n).

---

## 9. Numbers You Should Memorize

| Setting | Value | Reason |
|---------|-------|--------|
| Chunk size | 500 tokens | Balance between context and precision |
| Chunk overlap | 10% (50 tokens) | Catches boundary-crossing facts |
| `top_k_retrieve` | 12 | RRF needs enough candidates from each retriever |
| `top_k_rerank` | 4 | Keeps LLM prefill ~1.5K tokens for fast TTFT |
| OCR threshold | 100 chars/page | Below this, page is likely scanned |
| Rerank quality thresholds | 0.65 / 0.30 | HIGH / MEDIUM / LOW boundaries |
| Embedding dim | 1024 (Qwen3) / 384 (BGE fallback) | Standard for these models |
| HNSW `m` | 16 | Default; balance graph density vs memory |
| HNSW `ef_construct` | 128 | Index-time accuracy knob |
| Whisper model | base, int8 | CPU-friendly, ~30x real-time |
| Video keyframe interval | 60s | Every minute |
| Video max frames | 8 | Caps vision calls per video |
| Max history turns | 16 | Multi-turn memory bound |
| Generator temperature | 0.3 | Some variety, mostly factual |
| Generator `max_tokens` | 600 | Most answers fit |
| Ctx LLM `max_tokens` | 120 | 80-word context prompt |
| Doc window for context | 60K chars | Stays in 8B model's context budget |
| Batch upsert | 64 | Qdrant sweet spot |

---

## 10. Interview Q&A — Easy (Warm-up)

**Q1. Walk me through your project in one paragraph.**
> A multi-modal RAG chatbot. The user can drop any file (PDF, image, audio, video, Office doc, code, archive) or paste a URL. Each format gets routed through a dedicated processor — PyMuPDF + Tesseract for PDFs, Whisper for audio, vision LLM for images, ffmpeg + Whisper + vision for video — and the resulting text is chunked (500 tokens, 10% overlap), optionally contextualized with Anthropic's technique, and indexed into Qdrant as both dense (Qwen3-0.6B) and BM25-sparse vectors. Queries do hybrid retrieval with Reciprocal Rank Fusion → cross-encoder rerank (bge-reranker-v2-m3) → streaming LLM generation via NVIDIA NIM with quality-aware system prompts and inline citations. Frontend is Next.js consuming Server-Sent Events.

**Q2. What is RAG and why is it useful?**
See §8.1. Add: "Important for any product where you need *grounded, citable, private, fresh* answers — legal, medical, internal company knowledge, customer support."

**Q3. Why a vector database instead of just a relational DB with `LIKE`?**
> Vector DBs do approximate nearest-neighbor search over high-dimensional embeddings. SQL `LIKE` only matches lexical substrings — it can't find "warranty period" when the document says "guarantee duration". Embeddings encode semantic meaning, so synonyms / paraphrases match. Also: vector DBs use indexes like HNSW that give sub-millisecond lookup over millions of vectors; SQL `LIKE` is `O(n)` table scan.

**Q4. What's Qdrant?**
> An open-source (Apache 2.0) vector database written in Rust. Supports dense + sparse named vectors in one collection, HNSW indexing, payload filtering, server-side BM25, and gRPC + REST APIs. I picked it over Pinecone (closed-source, paid), Weaviate (Java, GraphQL overhead), Milvus (heavier ops) because it's lightweight, performant, and one Docker container is enough.

**Q5. What's an embedding?**
See §8.2.

**Q6. Why do you use a cross-encoder reranker if you already have dense + BM25?**
> Hybrid retrieval is a recall maximizer — it grabs 12-20 candidates that *might* be relevant. The cross-encoder is the precision maximizer: it actually feeds `(query, passage)` together into a transformer so the model can see token-level interactions, but it's too expensive to run on millions of docs. Two-stage gives you both speed and precision.

**Q7. What's chunking and why does it matter?**
See §8.6.

**Q8. Why FastAPI and not Flask?**
> Native `async`/`await`, automatic OpenAPI schema generation, Pydantic-based request validation, much faster than Flask under load, and `StreamingResponse` makes SSE one line. Also: lifespan events let me pre-warm models on startup.

**Q9. What's Server-Sent Events?**
See §8.8.

**Q10. Why Next.js for the frontend?**
> App Router gives me file-system routing, React 19, and good defaults for client-side fetching. Could have used pure CRA — chose Next for the future-proofing and the developer experience (hot reload, image optimization, ready for server-rendered marketing pages later).

---

## 11. Interview Q&A — Medium (Core)

**Q11. Walk me through the ingestion pipeline.**
> Take PDF as the example. (1) `media_router.detect` sees `.pdf` extension and routes to `pdf_doc.process`. (2) `parser.parse_pdf` opens with PyMuPDF and yields one `PageRecord` per page; for each page, if native text is under 100 chars I fall back to Tesseract OCR. (3) `chunker.chunk_pages` flattens all pages' tokens (cl100k_base via tiktoken) into a list, slides a 500-token window with 50-token overlap, records `(page_start, page_end)`. (4) For offline ingest, `contextualizer.situate_chunk` asks llama-3.1-8b-instruct to write 80 words of context per chunk and prepends it. (5) `embedder.embed` runs Qwen3-Embedding-0.6B on the context+chunk text. (6) `vector_store.upsert` wraps each into a Qdrant point with a dense vector and a sparse `Document(text=..., model="Qdrant/bm25")` — Qdrant computes the BM25 vector server-side at index time. Batches of 64.

**Q12. Walk me through the query pipeline.**
> User question hits `/api/chat`. (1) `hybrid_search`: embed query with the Qwen3 instruction prefix, send Qdrant a `query_points` with two prefetches — dense (limit 24) and sparse (limit 24) — fused via `Fusion.RRF`, returns top 12. (2) `_dedupe`: signature on `(first_160_chars_lowercased, page)` drops near-duplicates. (3) `rerank`: cross-encoder scores `(query, chunk[:800])` pairs, sigmoid-normalizes to [0,1], returns top 4. (4) `_quality_tier`: top score determines HIGH/MEDIUM/LOW. (5) Stream meta SSE event with sources + timings — sidebar populates before the answer starts. (6) `stream_answer`: builds messages with quality-aware system prompt + last 16 history turns + user prompt formatted as `[1] score=X.XX pdf=... page=... \n<text>`, opens NIM stream, yields `(reasoning|content, text)` tuples, FastAPI re-emits as SSE.

**Q13. Why hybrid retrieval (dense + BM25)?**
See §8.4. Real example: a query like "What's the GSTIN of Vinit Enterprises?" — pure dense softens those alphanumeric digits and might retrieve any GSTIN-discussing page; BM25 nails the exact substring. Conversely, "Tell me about employee perks" — BM25 misses pages that say "benefits" but not "perks"; dense gets them.

**Q14. Explain RRF (Reciprocal Rank Fusion) in detail.**
> Suppose dense retriever returns docs `[A, B, C, D]` ranked 1-4, and BM25 returns `[C, A, E, B]` ranked 1-4. With k=60 (standard):
> - A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
> - C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
> - B: 1/(60+2) + 1/(60+4) = 0.0161 + 0.0156 = 0.0317
> - D: 1/(60+4) = 0.0156
> - E: 1/(60+3) = 0.0159
> So fused order: A, C, B, E, D.
> **The point:** rank-based — never need to normalize wildly different score scales (cosine 0.7 vs BM25 score 12.3). k=60 is from the original 2009 paper; tweaks rarely help in practice.

**Q15. Why do you sigmoid-normalize rerank scores?**
> Different cross-encoders output different score ranges. BGE-reranker-v2-m3's raw logits might be in [-5, 10]; another might be in [0, 1]. Sigmoid maps any real number to [0, 1] monotonically. With normalized scores I can define **fixed thresholds** (0.65 / 0.30) for HIGH/MEDIUM/LOW. Switch reranker → same thresholds still make sense.

**Q16. What's Anthropic's Contextual Retrieval and what's its effect?**
See §8.7. Concretely in this codebase: `contextualizer.py` prompts `meta/llama-3.1-8b-instruct` with `<document>...</document><chunk>...</chunk>` and asks for ≤80 words of situating context. Prepends that to the chunk before embedding. Anthropic's blog measured 35% reduction in top-20 retrieval failure alone, 49% with BM25.

**Q17. Why do you cache the contextualizer's output during ingest but skip it for live session uploads?**
> Contextualization is N LLM calls (one per chunk). For a 100-chunk PDF, that's 100 sequential NIM calls. Fine offline. For a chat experience, users expect "drop file → ask question" to feel under 5 seconds. So `session.add_to_session` sets `context_text = text` directly. The quality hit is acceptable because session uploads are usually small (single file, 10-30 chunks), where each chunk still has decent semantic load.

**Q18. Why Qwen3-Embedding-0.6B specifically?**
> Top-tier on MTEB (64.34 retrieval score), Apache-2.0 license (can self-host commercially), 1024-d (sweet spot for storage vs precision), 0.6B params (runs on CPU acceptably). Asymmetric — supports query-instruction prefix that gives a measurable retrieval boost. BGE-small fallback handles the case where the HF download fails on a fresh machine.

**Q19. Why BGE-reranker-v2-m3 and not Qwen3-Reranker-0.6B?**
> Both are top-tier. The difference: BGE has a classification head — one forward pass per (query, passage) pair → one score. Qwen3-Reranker is autoregressive — it generates "yes" or "no" tokens, you take their logit difference. On GPU the cost is similar; on CPU autoregressive is 5-10x slower because of the multiple sampling steps. Pushed rerank latency from ~0.3s to 25s in one of our test runs.

**Q20. How does the OCR fallback work?**
> Per page: `text = page.get_text("text")` (PyMuPDF, native). If `len(text.strip()) < 100`, assume scanned. Convert that single page to a 200-DPI image (`pdf2image.convert_from_path(first_page=i+1, last_page=i+1)`), run `pytesseract.image_to_string`. Per-page (not whole-PDF) means we don't OCR a 1000-page document just because page 47 is a scan.

**Q21. How does session isolation work?**
> Each chunk's Qdrant payload includes a `session_id` field. The collection has a keyword payload index on `session_id` (created at `reset_collection` time). `add_to_session` tags chunks with the user's session UUID. `clear_session` uses a filter delete — `Filter(must=[FieldCondition(key="session_id", match=session_id)])` — which is O(log n) thanks to the index. Pre-ingested corpus chunks have no `session_id` (or a global one), so they're never touched.

**Q22. Explain the SSE streaming you implemented.**
> FastAPI's `StreamingResponse(content_iterator, media_type="text/event-stream")`. The iterator yields `b"data: " + json.dumps(event) + b"\n\n"`. Six event types: `stage`, `meta`, `reasoning`, `content`, `done`, `error`. On the client, since I need to POST a JSON body (not supported by `EventSource`), I use `fetch` + `response.body.getReader()`. Buffer chunks, split on `\n\n`, strip `data:` prefix, JSON.parse, yield. Critical headers: `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no` (without this, nginx in front would buffer 4KB before flushing → no streaming).

**Q23. How do you handle multi-turn / coreference?**
> The last 16 conversation turns are passed in `body.history` from the frontend. `stream_answer` injects them between system and user messages. The system prompt explicitly says: "Use prior conversation turns to resolve coreference ('it', 'that', 'he', 'she')". So if a user asks "Who is the CEO?" then "How old is he?", the second question's retrieval might miss but the prior turn's answer (with the name) is in the LLM context for resolution.

**Q24. How do you stop the model from hallucinating when retrieval fails?**
> Quality tiering. If top rerank score < 0.30, system prompt switches to LOW mode: model is required to open with "These documents don't seem to cover your question directly", say what the documents *are* about, optionally answer from general knowledge with `(General knowledge — not from your documents)` prefix. Citation format is suppressed. This is what stops the "confident wrong citation" failure.

**Q25. How does video processing work?**
> `ffmpeg -i input -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav` → audio at Whisper's native input format. `ffmpeg -i input -vf fps=1/60 -frames:v 8 frame_%03d.jpg` → one keyframe every 60s, capped at 8 frames. Audio → `processors/audio.py` (Whisper streaming segments). Each frame → `processors/image.py` (vision LLM describes it). Combined text: transcript header + body + each keyframe's description. Both transcript words AND visual elements become searchable.

**Q26. What's an HNSW index?**
> Hierarchical Navigable Small World graph. Approximate nearest-neighbor (ANN) algorithm: a multi-layer graph where higher layers are sparser long-range hops, lower layers are dense neighbors. Query starts at top, greedily descends, refines at bottom layer. Logarithmic complexity (`O(log n)`), 95-99% recall at 1000x speedup vs exact search. `m=16` = graph degree (more = better recall, more memory); `ef_construct=128` = index-time search beam.

**Q27. Why pre-warm models in the lifespan event?**
> Cold-start problem. Without it: first user query loads Qwen3-Embedding (~1.2GB) and bge-reranker-v2-m3 (~570MB) from disk and into memory. That's 30-60 seconds where the UI looks frozen. With `lifespan`: server prints `[lifespan] READY` only after both models warm up, *then* uvicorn starts accepting requests. First user query is fast.

**Q28. How do you handle file encoding for text files?**
> `processors/text.py`: try UTF-8 first. On `UnicodeDecodeError`: read raw bytes, ask `chardet.detect()` for likely encoding, decode with `errors='replace'`. Last resort: latin-1 with replace. **Never raises** — we always return *some* text. This survives mixed Windows-1252/UTF-8 codebases, GB2312 Chinese files, etc.

**Q29. What does the eval framework (RAGAS) measure?**
> Four metrics: (1) **Faithfulness** — fraction of answer claims that can be inferred from the retrieved context (catches hallucinations). (2) **Answer Relevancy** — semantic similarity between question and (paraphrased) answer (catches off-topic answers). (3) **Context Precision** — how high in the retrieval list are the truly-relevant chunks. (4) **Context Recall** — fraction of ground-truth needed to answer that's in the retrieved set. We snapshot ours alongside a "naive" baseline (single-vector, no rerank) and report side-by-side.

**Q30. Why is the temperature 0.3 for generation and 0.0 for contextualization?**
> Contextualization is a deterministic summary task — we want the same context phrase every time → 0.0. Answer generation benefits from a tiny bit of variety so successive turns don't repeat the same opening sentence — but still mostly factual → 0.3 (low). The system prompt also says "vary your wording — do not reuse the same opening sentence".

---

## 12. Interview Q&A — Hard (Cross-questions, "Why this not that?")

**Q31. Why not just throw all the PDF text at GPT-4 with a 128K context and skip the retrieval?**
> Three reasons. (1) **Cost:** $0.01-0.06 per 1K input tokens × 100K tokens × every query = quickly $$ at scale. (2) **Quality:** "Lost in the middle" research shows LLMs ignore content in the middle of long contexts. Retrieval-focused chunks at the start are answered with much higher accuracy. (3) **Multi-document scale:** users have GB of docs, not one PDF. No model has 100M token context. RAG composes.

**Q32. Why not use OpenAI text-embedding-3-large?**
> Closed source, $0.13 per 1M tokens (adds up for big ingests), data leaves your infra, license risk for sensitive docs. Qwen3-Embedding-0.6B matches or beats it on MTEB, is Apache-2.0, runs locally. We do still use a hosted LLM (NIM) for generation — but embeddings are the bulk of inference cost during ingest, so self-hosting them matters.

**Q33. Why not LangChain?**
> LangChain abstractions add a layer that hides the actual primitives (chains, agents, retrievers) — making debugging painful (deeply-nested call stacks, multiple versions of the same API). For a project this size, raw Qdrant client + OpenAI client + a few hundred lines of glue is shorter, faster, and more debuggable. LangChain would *add* code, not subtract. (We do depend on `langchain-openai` + `langchain-huggingface` *only* because RAGAS needs them.)

**Q34. Why not LlamaIndex?**
> Same critique as LangChain — heavy abstractions for what's actually a small amount of logic. Also: LlamaIndex's "managed" stacks tend to push you toward their indexing patterns; I wanted full control over hybrid + rerank + quality tiering.

**Q35. Why not GraphRAG?**
> GraphRAG is great for *entity-heavy* domains where relationships matter (who reports to whom, which paper cites which). Building the graph is expensive (LLM-extracted entities + relations) and overkill for "what's the warranty period?" type lookups. Anthropic's contextual retrieval + hybrid + rerank captures most of the gains at 1/10 the complexity. GraphRAG would be a sound choice for a research-paper or legal-precedent product.

**Q36. Why 500-token chunks and not 200 or 1000?**
> Tested empirically. 200 → too fragmented; bullet-pointed docs return individual bullets without context. 1000 → multiple topics per chunk; embedding becomes a "smoothie", rerank precision drops. 500 with 10% overlap holds 2-4 paragraphs typically. **Critically:** with `top_k_rerank=4` × 500 tokens ≈ 2K context. Fits well under all NIM model contexts and keeps prefill TTFT low.

**Q37. Why store BM25 as a *sparse vector in Qdrant* rather than running Elasticsearch alongside?**
> One database to manage. Qdrant's sparse-vector support is real BM25 (with IDF modifier), supports `Document(text=..., model=...)` for server-side tokenization (consistent with index-time), and lets me do hybrid RRF in a single `query_points` call without two HTTP round-trips. Elasticsearch + Qdrant would be two stateful services, two operational concerns, two backup strategies.

**Q38. Why Tesseract and not a learned OCR like docTR or PaddleOCR?**
> Tesseract is good enough for the *fallback* case (the rare scanned page), it's already a system dep on most Linux/Mac, no Python ML weights to ship. The vision LLM (Llama 3.2 Vision 90B) handles the harder OCR for *image* uploads with much higher quality — so OCR is split: cheap Tesseract for native-PDF scanned pages, expensive vision LLM for images.

**Q39. Why Whisper and not AssemblyAI or Deepgram?**
> Local. No data leaves the box. `faster-whisper` (CTranslate2 backend) runs `base` model at ~30x real-time on CPU. For a hackathon / portfolio project, no API key needed, no per-minute cost.

**Q40. Why faster-whisper specifically?**
> Same Whisper model, but CTranslate2 backend: int8 quantization, 4x faster than openai-whisper for the same model size, lower memory. VAD (Voice Activity Detection) filter skips silent regions — significant speedup on lectures with long pauses.

**Q41. Why save uploaded files to `uploads/` and not process in-memory?**
> Two reasons. (1) Processors expect `Path` objects — easier to debug ("which file failed?"). (2) Some processors (archive extraction, video → ffmpeg) need real filesystem paths anyway. Cost: per-session disk usage; cleared when user deletes their session.

**Q42. Why `uuid.uuid5` for chunk IDs in Qdrant?**
> Qdrant requires UUID or int point IDs. My natural chunk ID is `sha1(pdf:offset:text_prefix)` — a hex string. `uuid5(namespace, name)` deterministically derives a UUID from any string — so same chunk in same PDF always gets the same UUID, making idempotent re-ingest possible.

**Q43. Why is the rerank truncation 800 chars and not the full chunk?**
> bge-reranker-v2-m3 has `max_length=256` (tokens). Truncating to 800 chars (~150-200 tokens) ensures we don't exceed the model's window after concatenating with the query. Truncation slightly hurts recall for very long chunks; but we picked 500-token chunks (~2000 chars) precisely to make this tolerable.

**Q44. Walk me through a case where your system would still fail.**
> Multi-hop reasoning across documents. Example: "Compare the warranty terms of vendor A's product (in `vendor_a.pdf`) with vendor B's (in `vendor_b.pdf`)." The retriever might pull warranty chunks from both, but if the question requires comparing **specific clauses by number**, the comparison logic is left entirely to the LLM. A graph-based or query-decomposition approach (split into sub-questions, retrieve each, then synthesize) would help here. We currently don't do that.

**Q45. Why use NVIDIA NIM and not a local Llama via Ollama?**
> Two answers. (1) **Capability:** NIM hosts Kimi-K2.6 (1T MoE, 256K context) which no consumer hardware can run. (2) **Hackathon constraint:** the project's NVIDIA NIM credits are free for a window, so deploying a 70B+ model in the cloud is essentially free compute. Production-ready alternative: run Llama-3.3-70B on a single H100 via vLLM and point `openai.OpenAI(base_url=...)` at it — zero code change.

**Q46. What if the embedding model's tokenizer doesn't match `cl100k_base` you use for chunking?**
> Honest answer: chunk boundaries would be slightly off from the embedder's true token boundaries. But sentence-transformers will internally re-tokenize before encoding, so the *content* is preserved — only the chunk-size accuracy is approximate (e.g., 500 cl100k tokens ≠ 500 Qwen tokens exactly, off by ~10%). Cleaner fix would be to use the embedder's own tokenizer for chunking; we kept tiktoken for speed and because cl100k is well-known.

**Q47. Why `temperature=0.3` for the answer and `top_p=0.9`?**
> Both control diversity. Pure `temperature=0` is too rigid (same word choices, parrot-like). Pure high temperature is wild. `0.3` + `top_p=0.9` (only consider tokens in the top-90% cumulative probability) is a "low randomness, no junk tokens" combo. Specifically chosen so successive answers vary in wording but never hallucinate via random sampling tails.

**Q48. How would you add re-ranking for multi-lingual queries?**
> bge-reranker-v2-m3 is already multilingual ('m3' = multi-lingual, multi-functional, multi-granular). For a more specialized setup: detect query language via `langdetect`, route to a language-specific reranker. Also: `embedder.embed_query` should use the right instruction prefix per language (Qwen3 supports many).

**Q49. What's a query that hybrid retrieval handles dramatically better than pure dense?**
> "Show me the section about IRS form 8606 for nondeductible IRA contributions." Pure dense softens "8606" and "IRA" into generic embeddings — might miss the specific form. BM25 nails "8606" as a rare exact token. RRF combines: dense pulls semantically-near pages (about IRAs), BM25 pulls the exact-form page, fused order surfaces both.

**Q50. Why batch_size=64 for Qdrant upserts and not 128 or all-at-once?**
> Empirical sweet spot. Too small (8) → many round-trips, slow. Too large (256+) → request size hits client/server timeouts on slow networks. 64 is Qdrant's documented recommendation; saturates throughput without timeouts.

**Q51. What if the user uploads a 5GB video?**
> Whisper streams segments — won't OOM on the audio side. Vision is capped at 8 keyframes regardless of video length. ffmpeg extraction reads sequentially. The bottleneck is disk space (`uploads/` and `td` temp dir during processing). Production-ready improvement: process in chunks (per-30-min slices), persist intermediate results, allow resume.

**Q52. How do you secure the API in production?**
> Currently `CORS allow_origins=["*"]` — wide open for demo. Production: (1) JWT auth on each endpoint, (2) per-user rate limiting (slowapi), (3) signed upload URLs, (4) virus-scan uploads before processing, (5) restrict CORS to your frontend domain, (6) put nginx in front for TLS + request size limits.

**Q53. How would you scale this to 1M users?**
> Horizontal: (1) Multiple uvicorn workers behind nginx/HAProxy. (2) Qdrant cluster (replicated + sharded by collection). (3) Detach ingestion from API — push uploads to a queue (Redis/Rabbit), workers consume and index. (4) Use a CDN for static frontend assets. (5) Cache embeddings of common queries (Redis with embedding-as-key). (6) Move Whisper + Vision LLM calls to GPU-pool workers (Triton Inference Server). Backend stateless, scales linearly.

**Q54. How would you reduce cost at 1M users?**
> Cheap-tier model for "easy" queries, escalate to expensive only for low-confidence retrieval. Cache full responses for FAQ-like questions (TTL on session keys). Move embedding to ONNX-Runtime CPU on commodity hosts (no GPU bill). Quantize Qwen3 to int8 (`fastembed` already does). Bulk-batch contextualization at ingest (we already do this offline).

**Q55. Tell me about a tricky bug you fixed.**
> "First query stuck for 30-60s" — users thought the app was hung. Root cause: Qwen3 + reranker models were lazy-loaded on first request. Fixed with a FastAPI `lifespan` context manager that runs a warmup `embed(["warmup"])` and `rerank("warmup", [{"text":"hi"}])` before uvicorn starts accepting requests. Added a `[lifespan] READY` print so I can see in logs when it's actually ready. Documented in RUN.md as a thing to wait for.

**Q56. Another bug?**
> Sources panel was empty on streaming answers even though retrieval was succeeding. Root cause: I was emitting the `meta` SSE event *after* the LLM stream completed. Refactored to emit `meta` *between* rerank and generation — so the sources panel populates instantly, even before the first token arrives. UX win.

**Q57. What happens if the LLM stream errors mid-token?**
> `generator.stream_answer` catches the exception in the for-loop and yields `("error", "<type>: <msg>")`. The FastAPI handler re-emits as `{type:'error', text}`. Frontend renders a red "Model error" box (so the bubble doesn't stay frozen). The user can re-send. Sources panel from the successful retrieval is preserved.

**Q58. How would you test this system?**
> Three layers. (1) **Unit:** chunker (deterministic, easy to assert page metadata, overlap counts). Parser (snapshot test against a known PDF). (2) **Integration:** RAGAS over `eval/questions.jsonl` — labeled Q&A pairs. Run before/after every retrieval change, gate merges on no metric regression. (3) **End-to-end:** smoke test that drops a file → asks a known question → asserts citation contains expected page. `smoke_test.py` already covers the NIM connectivity part.

**Q59. How would you debug "the answer is wrong"?**
> Drill down through stages. (1) Are the retrieved chunks correct? Check `meta` SSE event sources. (2) Are the chunks relevant? Look at `rerank_score` and `quality` tier — LOW means retrieval failed, the LLM did the right thing in saying so. (3) If chunks are good but answer wrong: prompt issue — inspect the actual messages sent (add a log). (4) If only the score is borderline: tune thresholds or improve chunking/embeddings.

**Q60. What's the most expensive operation in your pipeline?**
> Per-query at runtime: the LLM call (network + tokens generated). Per-document at ingest: the contextualization (N LLM calls). I made conscious trade-offs for both: streaming + small `max_tokens=600` for query; offline-only + cheap 8B model for ingest. Skip-context flag for fast iteration during dev.

---

## 13. System Design / Scaling Questions

**Q61. Design a multi-tenant version (each customer has private docs).**
- Add `tenant_id` to every chunk's payload (in addition to `session_id`).
- Payload index on `tenant_id` (keyword).
- Every retrieval call must have a `Filter(must=[FieldCondition(key="tenant_id", match=user.tenant_id)])`.
- API auth (JWT) carries `tenant_id` in token claims; backend never trusts client-supplied tenant_id.
- Optional hardening: separate Qdrant **collections** per high-value tenant (full physical separation, but more collections to manage).

**Q62. Design an "incremental update" flow for a customer's doc set.**
- Each chunk's payload carries `doc_hash` (sha256 of source file).
- Re-ingest: compute new doc_hash, compare to indexed hashes for that tenant. If different → delete old chunks (filter by `doc_hash`) and re-upsert new ones.
- For per-page changes (`.docx` editing): if doc has stable section anchors, hash per section and only re-index changed sections.

**Q63. How would you reduce hallucinations further?**
- **Stricter citation:** require every factual sentence to have a `[file, p.X]` and reject answers without them via a post-processing regex check.
- **Self-consistency:** generate 3 answers with `n=3`, pick the one with most-consistent citations.
- **Verification step:** second LLM call asks "Does answer X actually follow from these excerpts?" — discard or warn if no.
- **NLI-based faithfulness check:** use a small NLI model to score answer-vs-context entailment in real-time.

**Q64. How would you support 1M-page documents?**
- Streaming chunker (already is — pages yielded one at a time).
- Lazy parser (already lazy).
- Parallel embedding workers (multiprocessing pool feeding the GPU).
- Qdrant payload `with_payload=False` for retrieval, fetch payloads only for the top-K.

**Q65. How would you do "follow-up question understanding" better?**
- Currently I just pass history into the LLM. A more sophisticated approach: **query rewriting** — small LLM rewrites the follow-up into a standalone query ("How old is he?" + "Who is the CEO?" → "How old is John Smith, the CEO?"), then retrieve on the rewritten query.
- Alternative: **HyDE** (Hypothetical Document Embeddings) — LLM generates a hypothetical answer, embed *that*, search for similar real chunks. Often beats raw-query embedding for paraphrased questions.

**Q66. How would you optimize TTFT (time-to-first-token)?**
- Pre-warm models (done).
- Smaller `top_k_rerank` → smaller prefill (4 is already tight).
- Faster reranker (BGE classification head vs Qwen3 autoregressive — done).
- Run embedder + reranker on GPU (currently CPU on dev box).
- Use a fast-TTFT model for routing: short answer? Use llama-3.3-70b (sub-1s TTFT). Long reasoning? Use Kimi (slower but better at hard questions).

**Q67. How would you add a "knowledge graph" view of all uploaded docs?**
- Per-chunk entity extraction via small LLM at ingest.
- Store entities as a side payload (`entities: ["company:Acme", "person:Jane Doe", "date:2024-01-15"]`).
- Optional: separate graph DB (Neo4j) for explicit relationships.
- Frontend renders the entity cloud / graph for the current session.

**Q68. How would you handle PII / data redaction?**
- Pre-ingestion pass: regex + NER for emails, SSNs, credit cards, phone numbers.
- Two options: (a) **redact in place** (replace with `[EMAIL]`), or (b) **flag chunks as sensitive** and require an "I acknowledge" toggle before showing.
- For compliance (GDPR), `clear_session` already supports right-to-erasure for session uploads.

**Q69. How would you build evaluation into CI?**
- `eval/run_ragas.py` runs on `pull_request` GitHub Action.
- Gate on `faithfulness > 0.85` and `context_precision > 0.75` (or whatever your baseline is).
- Comment on PR with metric deltas vs main branch's `ragas_snapshot.json`.

**Q70. How would you A/B test a new reranker or chunking strategy?**
- Tag each upserted chunk with the indexing config (`chunk_strategy: "v2"`, `embed_model: "qwen3-0.6b"`).
- Maintain two Qdrant collections (`rag_v1`, `rag_v2`) or use payload filtering.
- At query time, randomly route X% of traffic to v2 (cookie-stable).
- Compare RAGAS metrics + user thumbs up/down + session-level retention.

---

## 14. Bugs / Gotchas / Trade-offs You Should Mention

These show maturity — knowing your own project's weaknesses is interviewer gold.

1. **`top_k_rerank=4` is aggressive.** If a question's answer is spread across 6 chunks, we'll miss two. Trade-off: TTFT vs recall. Tunable per query if needed.

2. **Contextualization is N LLM calls.** A 1000-chunk doc = 1000 calls. Mitigations: batch into single prompts (group 5 chunks per call → 1 LLM call per 5 chunks), or use a smaller model. Currently sequential.

3. **OCR threshold of 100 chars is a heuristic.** A page with 99 chars of real text + a corrupt PDF would still get OCR'd unnecessarily. Robust fix: detect zero text + image content via PyMuPDF page object.

4. **Session UUIDs in URL paths.** `/api/session/{sid}` could be enumerated if SIDs were predictable. We use `uuid.uuid4().hex` (random 128 bits) so brute-force is infeasible — but still: don't include UUIDs in HTTP logs / Sentry breadcrumbs.

5. **No streaming for ingest progress.** A 100-page PDF takes ~20s of "indexing…" with no progress bar in the Next.js frontend (Streamlit version has a progress bar). Easy fix: SSE on the upload endpoint with `chunk_n/total_chunks` events.

6. **`gen_model_options` is hand-maintained.** If NIM EOL's a model, we keep showing it in the dropdown until I manually remove. `_resolve_model` falls back with a warning, so it's not broken — just suboptimal UX.

7. **No deduplication across documents.** If two users upload the same PDF, we re-process and re-index it. Fix: content-hash on the file, skip if a chunk with that hash already exists.

8. **HEIC support requires `pillow-heif`.** On a fresh system without the C library, the import fails silently (`try/except: pass`). The image processor will still call `Image.open` which will then fail — should propagate that error to the user clearly.

9. **Whisper VAD can clip soft speakers.** VAD filter assumes silent → ignore. A very quiet speaker can be misclassified as silence. Mitigation: expose `vad_filter=False` for transcription-quality scenarios.

10. **`_dedupe` uses first-160-chars signature.** Two genuinely-different chunks that happen to start with identical boilerplate (e.g., "Section 1. Introduction.") will be wrongly deduped. Production fix: use Locality-Sensitive Hashing or shingled MinHash.

11. **No backpressure on uploads.** A user with 1000 files would queue 1000 inflight requests. Production: rate-limit, queue, or chunk upload UI to batches.

12. **Qdrant `allow_origins=["*"]`** at CORS layer. Fine for demo, **must** restrict for production.

---

## 15. Likely Follow-up "What if?" Questions

**Q71. What if I doubled the document corpus?**
> No code changes. Qdrant HNSW is logarithmic in collection size — query latency grows ~`log(2N)/log(N)` ≈ 1.0x. The contextualization ingest doubles in time. Vector storage doubles (~1.2GB per 1M chunks at 1024-d float32).

**Q72. What if I wanted to support speech-to-text in the UI ("hold to speak")?**
> Add a record button in the Composer. Push WAV to a new endpoint `/api/voice-question` → run through `processors/audio.py` → take the transcript as the question text → feed into normal `/api/chat`. ~50 lines of code.

**Q73. What if the user wants the answer translated?**
> Pass a `target_language` field in the chat request. System prompt addendum: "Answer in {target_language}." Citations stay in original language. The LLM (Llama 3.3 70B) is competent in 10+ languages out of the box.

**Q74. What if we want to highlight the exact text in the PDF that supports the answer?**
> Two-step: (1) After generation, run a second LLM call: "Given this answer and these chunks, return the verbatim sentences that ground each claim." (2) For each verbatim sentence, search the chunk's text for the substring → return `{page, char_start, char_end}`. PDF viewer (e.g., `streamlit-pdf-viewer`) jumps to that page and highlights.

**Q75. What if I want to fine-tune the embeddings on my customer's data?**
> Generate (query, relevant_passage, irrelevant_passage) triplets from customer-labeled feedback. Fine-tune Qwen3-Embedding with a contrastive loss (InfoNCE). 1-10K labeled triplets typically suffice for a domain-tuned embedder. Sentence-transformers has built-in `TripletLoss` training.

**Q76. What if the user uploads a corrupt file?**
> Each processor wraps its main call in try/except. On failure, the chunk count is 0 and the UI sidebar shows the file as "errored". Recent example: a corrupt PDF → PyMuPDF raises → processor returns empty pages → chunker yields nothing → upsert is a no-op. User sees "(0 chunks)" and can re-upload.

**Q77. What if NIM is down?**
> Generation fails with `(error, "<exception>")` SSE event. Retrieval still works (no NIM dependency on the read path). For ingest, contextualization fails with `tenacity` retries (3x exp backoff); after that, `contextualize_chunks` logs and falls back to raw chunk text → ingest still completes, just at slightly lower quality. The chat experience degrades, doesn't break.

**Q78. What if the user's question is "summarize the whole document"?**
> Currently: retrieval might pull 12 random-ish chunks, the LLM summarizes those. Not ideal — could miss sections. Better: detect summarization intent (regex on "summarize" + lack of specific entity) → switch to a **map-reduce** flow: chunk-level summaries via small LLM, then synthesize. Out of scope for this version.

**Q79. What if I want to deploy on AWS / Render / Vercel?**
- **Vercel:** Next.js frontend only — serverless functions can't host FastAPI well. Backend goes elsewhere.
- **Render / Railway / Fly:** Backend as a long-running service (Docker image with python + qdrant-client). Set `NEXT_PUBLIC_API_URL` to the backend URL.
- **AWS:** ECS Fargate for backend, ALB in front, CloudFront for Next.js static assets, Qdrant on Elastic Container Service or use Qdrant Cloud. RDS not needed (everything's in Qdrant).
- **Self-hosted:** Single 16-core / 64GB VM running everything fits a hackathon-scale demo.

**Q80. What's the next feature you'd add?**
> Top three: (1) **Query rewriting** for follow-ups — small LLM rewrites the question with context from history. (2) **Citation-grounded highlighting** in the PDF viewer. (3) **Conversation memory across sessions** — persistent per-user chat history with semantic search over past chats.

---

## 16. Final Cheat Sheet (last-5-minutes review)

### The pitch (memorize)
"Universal RAG chatbot. Drop any file or URL → auto-routed through PyMuPDF + Tesseract OCR / faster-whisper / Llama Vision / ffmpeg / MarkItDown → chunked (500 tokens, 10% overlap, tiktoken) → optionally contextualized with Anthropic's 2024 technique (8B LLM prepends 80 words) → embedded with Qwen3-0.6B → upserted to Qdrant as dense + BM25 sparse named vectors. Queries: hybrid retrieval with RRF (top 12) → cross-encoder rerank (BGE-v2-m3 with sigmoid normalization, top 4) → quality-tiered system prompt (HIGH ≥0.65, MEDIUM ≥0.30, LOW) → NVIDIA NIM streaming generation with citations. FastAPI backend + Next.js 16 / React 19 frontend + Server-Sent Events for live retrieve/rerank/generate stages."

### Numbers to remember
- 500 tokens / 10% overlap / 12 retrieve / 4 rerank
- HIGH 0.65, MEDIUM 0.30, LOW under
- Qwen3-Embedding-0.6B 1024-d, BGE-small 384-d fallback
- 1024-d cosine, BM25 IDF, HNSW m=16 ef_construct=128
- max 16 history turns, temp 0.3, top_p 0.9, max_tokens 600
- 60-second video keyframes, max 8 frames, Whisper base int8
- OCR threshold 100 chars/page

### Three things that make this stand out
1. **Universal multimodal ingest** (60+ extensions, auto-routing, lazy imports)
2. **Quality-aware system prompt** (no hallucinations when retrieval fails)
3. **Live SSE pipeline visibility** (stage events update UI in real-time)

### Three honest weaknesses
1. Hand-maintained model allowlist
2. No streaming progress during ingest in the Next.js UI
3. `_dedupe` is heuristic — would fail on structurally-similar real chunks

### If asked "what would you build next?"
Query rewriting → citation-highlighted PDF viewer → cross-session semantic memory.

---

**End of prep doc. Read it twice. You've got this.**
