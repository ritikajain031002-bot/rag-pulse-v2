from __future__ import annotations

import json
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

# allow `python -m api.main` and `uvicorn api.main:app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import CFG
from src.generator import _quality_tier, stream_answer
from src.media_router import KIND_EMOJI, process_path, process_url
from src.pipeline import _dedupe
from src.reranker import rerank
from src.session import add_to_session, clear_session
from src.vector_store import hybrid_search


# ───────────────────────────── lifespan ──────────────────────────────
# Pre-warm heavy models so the FIRST query isn't slow (this was the
# "stuck for 5-10 min" symptom the user reported).

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[lifespan] pre-warming models ...", flush=True)
    try:
        from src.embedder import embed  # loads Qwen3-Embedding-0.6B
        _ = embed(["warmup"])
        print("[lifespan] ✓ embedder ready", flush=True)
    except Exception as e:
        print(f"[lifespan] ✗ embedder warmup failed: {e}", flush=True)
    try:
        _ = rerank("warmup", [{"text": "hi", "score": 0.0}], top_k=1)
        print("[lifespan] ✓ reranker ready", flush=True)
    except Exception as e:
        print(f"[lifespan] ✗ reranker warmup failed: {e}", flush=True)
    print("[lifespan] READY", flush=True)
    yield
    print("[lifespan] shutdown", flush=True)


app = FastAPI(title="RAG Pulse API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo permissiveness; tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ───────────────────────────── meta ──────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "default_model": CFG.gen_model,
        "embed_model": CFG.dense_model,
        "rerank_model": CFG.rerank_model,
        "vision_model": CFG.vision_model,
        "ctx_model": CFG.ctx_model,
        "whisper_model": CFG.whisper_model,
        "models": CFG.gen_model_options,
    }


@app.post("/api/session/new")
def new_session():
    return {"session_id": uuid.uuid4().hex}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    try:
        cleared = clear_session(session_id)
        return {"cleared": session_id, "points_deleted": cleared}
    except Exception as e:
        raise HTTPException(500, f"Clear failed: {e}")


# ───────────────────────────── ingestion ──────────────────────────────


@app.post("/api/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "blob").suffix
    dest = UPLOAD_DIR / f"{session_id}_{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        processed = process_path(dest)
        processed["filename"] = file.filename or dest.name
        chunks = add_to_session(session_id, processed)
        kind = processed.get("kind", "text")
        return {
            "filename": file.filename or dest.name,
            "kind": kind,
            "emoji": KIND_EMOJI.get(kind, "📎"),
            "chunks": chunks,
            "chars": len(processed.get("text", "") or ""),
            "pages": len(processed.get("pages", [])) if processed.get("pages") else None,
            "duration": processed.get("duration"),
            "language": processed.get("language"),
        }
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {type(e).__name__}: {e}")


class UrlIn(BaseModel):
    session_id: str
    url: str


@app.post("/api/url")
def ingest_url(body: UrlIn):
    try:
        processed = process_url(body.url)
        chunks = add_to_session(body.session_id, processed)
        return {
            "filename": processed.get("filename", body.url),
            "kind": "web",
            "emoji": "🌐",
            "chunks": chunks,
            "chars": len(processed.get("text", "") or ""),
        }
    except Exception as e:
        raise HTTPException(500, f"URL fetch failed: {type(e).__name__}: {e}")


# ───────────────────────────── chat (SSE streaming) ──────────────────────────────


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatIn(BaseModel):
    session_id: Optional[str] = None
    question: str
    model: Optional[str] = None
    history: Optional[list[ChatTurn]] = None
    top_k_retrieve: int = 12
    top_k_rerank: int = 5


def _resolve_model(name: Optional[str]) -> tuple[str, Optional[str]]:
    """Validate requested model against the live allowlist.

    Returns (model, warning). Stale browser tabs or curl users may POST a
    name we've since removed (e.g. EOL'd `moonshotai/kimi-k2-instruct`
    returns 410 from NIM). Fall back to the configured default and surface
    a warning so the client can show the user WHY their choice was ignored.
    """
    if not name or name == CFG.gen_model:
        return CFG.gen_model, None
    if name in CFG.gen_model_options:
        return name, None
    return (
        CFG.gen_model,
        f"Model '{name}' isn't in the current allowlist (likely deprecated "
        f"or removed). Using default '{CFG.gen_model}'. Hard-refresh the "
        f"page (⌘⇧R / Ctrl+Shift+R) to update the model picker.",
    )


def _serialize_sources(chunks: list[dict]) -> list[dict]:
    out = []
    for c in chunks:
        out.append(
            {
                "kind": c.get("kind", "pdf"),
                "filename": c.get("filename") or c.get("pdf") or "unknown",
                "page": c.get("page_start"),
                "page_end": c.get("page_end"),
                "score": float(c.get("score", 0.0)) if c.get("score") is not None else None,
                "rerank_score": float(c["rerank_score"]) if c.get("rerank_score") is not None else None,
                "text": (c.get("text") or "")[:600],
                "session_id": c.get("session_id"),
            }
        )
    return out


@app.post("/api/chat")
def chat(body: ChatIn):
    """Stream answer as SSE. Events:
       - {type:'stage', stage:'retrieving'|'reranking'|'generating', candidates?:int}
       - {type:'meta', sources:[...], timings:{retrieve_ms,rerank_ms}, model:str}
       - {type:'reasoning', text:str}    # model chain-of-thought (for reasoning models)
       - {type:'content', text:str}       # final answer tokens
       - {type:'error', text:str}
       - {type:'done', total_ms:int}
    """

    def stream() -> Iterator[bytes]:
        t_start = time.perf_counter()
        resolved_model, model_warning = _resolve_model(body.model)
        if model_warning:
            print(f"[chat] model fallback: {model_warning}", flush=True)
        try:
            yield _sse({"type": "stage", "stage": "retrieving"})
            t0 = time.perf_counter()
            candidates = hybrid_search(body.question, top_k=body.top_k_retrieve)
            candidates = _dedupe(candidates)
            retrieve_ms = int((time.perf_counter() - t0) * 1000)

            yield _sse(
                {"type": "stage", "stage": "reranking", "candidates": len(candidates)}
            )

            t0 = time.perf_counter()
            ranked = rerank(body.question, candidates, top_k=body.top_k_rerank)
            rerank_ms = int((time.perf_counter() - t0) * 1000)

            quality, top_score = _quality_tier(ranked)

            yield _sse(
                {
                    "type": "meta",
                    "sources": _serialize_sources(ranked),
                    "timings": {
                        "retrieve_ms": retrieve_ms,
                        "rerank_ms": rerank_ms,
                    },
                    "model": resolved_model,
                    **({"warning": model_warning} if model_warning else {}),
                    "quality": quality,
                    "top_score": round(top_score, 3),
                }
            )

            # ── stage 3: generate (token-by-token) — pass conversation
            # history so the model has ChatGPT-style multi-turn memory.
            yield _sse({"type": "stage", "stage": "generating"})
            history_payload = (
                [t.model_dump() for t in body.history] if body.history else None
            )
            for kind, text in stream_answer(
                body.question,
                ranked,
                model=resolved_model,
                history=history_payload,
            ):
                yield _sse({"type": kind, "text": text})

            total_ms = int((time.perf_counter() - t_start) * 1000)
            yield _sse({"type": "done", "total_ms": total_ms})
        except Exception as e:
            yield _sse({"type": "error", "text": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disables nginx/proxy buffering
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
