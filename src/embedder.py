"""Dense embeddings — Qwen3-Embedding-0.6B preferred, BGE-small ONNX fallback.

The module exports DIM so vector_store.reset_collection() can size the Qdrant
collection to match whichever backend actually loaded. Without this, a fallback
to a smaller model causes silent dimension mismatch on upsert.
"""
from typing import Sequence

from .config import CFG

_BACKEND: str | None = None
_MODEL = None
DIM: int = 0


def _init() -> None:
    global _BACKEND, _MODEL, DIM

    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(CFG.dense_model, trust_remote_code=True)
        DIM = _MODEL.get_sentence_embedding_dimension()
        _BACKEND = "st"
        print(f"[embedder] loaded {CFG.dense_model} via sentence-transformers, dim={DIM}")
        return
    except Exception as e:
        print(f"[embedder] sentence-transformers path unavailable: {e}")

    try:
        from fastembed import TextEmbedding

        fb_name = "BAAI/bge-small-en-v1.5"
        _MODEL = TextEmbedding(model_name=fb_name)
        DIM = 384
        _BACKEND = "fastembed"
        print(f"[embedder] loaded {fb_name} via FastEmbed, dim={DIM}")
        return
    except Exception as e:
        raise RuntimeError(f"No embedding backend available: {e}")


_init()


def embed(texts: Sequence[str]) -> list[list[float]]:
    if _BACKEND == "fastembed":
        return [list(v) for v in _MODEL.embed(list(texts))]
    return _MODEL.encode(list(texts), normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    if _BACKEND == "st" and "Qwen3" in CFG.dense_model:
        text = (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: " + text
        )
    return embed([text])[0]
