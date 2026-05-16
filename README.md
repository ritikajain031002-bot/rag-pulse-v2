# RAG Hackathon — Universal Chat Edition

A production-grade RAG chatbot. Drop **any** file (PDF, image, audio, video, Office doc, code, archive) or paste a URL — the bot auto-routes through the right pipeline (vision LLM, Whisper, ffmpeg, MarkItDown), indexes it into Qdrant, and answers with grounded citations. Open stack, sub-5-second answers.

## Universal media support

| Input | Pipeline | Output |
|---|---|---|
| `.pdf` (any length) | PyMuPDF + Tesseract OCR fallback | text per page |
| `.png/.jpg/.heic/.webp/.gif/.bmp/.tiff/.avif` | NVIDIA NIM `llama-3.2-90b-vision-instruct` describes content + transcribes visible text | rich description |
| `.mp3/.wav/.m4a/.ogg/.flac/.aac/.opus/.aiff` (any length) | faster-whisper local transcription with VAD | timestamped transcript |
| `.mp4/.mov/.avi/.webm/.mkv/.flv/.wmv/.m4v/.mpg` | ffmpeg → audio (Whisper) + N keyframes (Vision) | transcript + visual context |
| `.docx/.pptx/.xlsx/.rtf/.epub/.odt` | MarkItDown → markdown (native fallbacks) | text |
| `.txt/.md/.py/.js/.json/.yaml/.csv/...` (60+ ext) | direct read with chardet auto-encoding | text |
| `.zip/.tar.gz` | recursive extract → process every member | combined text |
| URL | requests + BeautifulSoup readability | clean text |

Everything ends up as text → chunked → hybrid-indexed (BM25 + dense) → searchable in the same chat session.

## Stack

- **Parsing**: PyMuPDF (fast native text) + Tesseract (per-page OCR fallback for scanned)
- **Chunking**: 800-token windows, 15% overlap, page-anchored metadata
- **Contextualization**: Anthropic's Contextual Retrieval (Sept 2024) via NVIDIA NIM `meta/llama-3.1-8b-instruct`
- **Embeddings**: Qwen3-Embedding-0.6B (MTEB 64.34, Apache-2.0) with BGE-small ONNX fallback
- **Vector DB**: Qdrant (Rust, Apache-2.0) — dense + BM25 sparse named vectors, RRF hybrid
- **Reranker**: Qwen3-Reranker-0.6B (MTEB-R 65.80, SOTA) with BGE-v2-m3 fallback
- **LLM**: NVIDIA NIM `moonshotai/kimi-k2.6` (1T MoE, 256K context, thinking-mode optional)
- **UI**: Streamlit + streamlit-pdf-viewer (click citation → opens PDF at exact page)
- **Eval**: RAGAS (faithfulness, answer relevancy, context precision/recall)

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. system deps (Mac)
brew install tesseract poppler ffmpeg
# (Ubuntu)  sudo apt-get install tesseract-ocr poppler-utils ffmpeg

# 3. start Qdrant
docker compose up -d

# 4. configure NVIDIA NIM key
cp .env.example .env
# edit .env, paste your key from https://build.nvidia.com  (Get API Key, top-right)

# 5. drop your 10+ PDFs into data/pdfs/

# 6. ingest (skip --no-context for max quality but slower)
python -m src.ingest --no-context     # fast path, naive embeddings
python -m src.ingest                  # full Contextual Retrieval

# 7. run the demo
streamlit run app/streamlit_app.py
```

## Project layout

```
.
├── docker-compose.yml      # Qdrant
├── requirements.txt
├── .env.example
├── data/pdfs/              # drop PDFs here
├── eval/
│   ├── questions.jsonl     # labeled Q+A for RAGAS
│   └── run_ragas.py
├── src/
│   ├── config.py
│   ├── parser.py           # PyMuPDF + Tesseract
│   ├── chunker.py
│   ├── contextualizer.py   # Anthropic technique
│   ├── embedder.py         # Qwen3-Embedding-0.6B
│   ├── vector_store.py     # Qdrant hybrid wrapper
│   ├── reranker.py         # Qwen3-Reranker-0.6B
│   ├── generator.py        # NVIDIA NIM Kimi-K2.6 + citation prompt
│   ├── pipeline.py
│   └── ingest.py
└── app/
    └── streamlit_app.py    # 4-tab UI
```

## Performance budget (target 2–5 s)

| Stage | Time |
|---|---|
| Query embedding | 50–100 ms |
| BM25 sparse encode | 10–30 ms |
| Qdrant hybrid query (HNSW + RRF) | 5–20 ms |
| Rerank top-20 (Qwen3-Reranker-0.6B, CPU) | 200–400 ms |
| NVIDIA NIM Kimi-K2.6 TTFT | 400–800 ms |
| Stream 300 tokens @ ~80 TPS | ~3.7 s |
| **Total (thinking=False)** | **~4.5 s** |
| With `KIMI_THINKING=true` | **6–12 s** (use only for hard questions) |

## Quality lift vs naive RAG

From the Anthropic paper, on Pass@10:
- Naive RAG: 5.7% retrieval failure rate
- + Contextual Embeddings: 3.7% (-35%)
- + Contextual BM25: 2.9% (-49%)
- + Reranking: 1.9% (-67%)

## Smoke test (before demo)

```bash
curl -s localhost:6333/collections | python -m json.tool
python -c "from src.vector_store import collection_stats; print(collection_stats())"
python -c "
import time
from src.pipeline import RAGPipeline
p = RAGPipeline()
t0 = time.perf_counter()
r = p.query('What is this corpus about?')
ans = ''.join(r.answer_stream)
print(f'total={int((time.perf_counter()-t0)*1000)}ms')
print(ans[:400])
"
```
