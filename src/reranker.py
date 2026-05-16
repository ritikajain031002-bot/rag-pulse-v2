"""Cross-encoder reranking with sigmoid-normalised scores.

Default: BAAI/bge-reranker-v2-m3 (278M, classification head, ~5-10x faster
on CPU than Qwen3-Reranker-0.6B's autoregressive scoring — see m0156).
Override via RERANK_MODEL env var; fallback (auto): ms-marco-MiniLM-L-6-v2.

Scores are sigmoid-normalised to [0,1] so generator._quality_tier can apply
fixed thresholds (HIGH ≥ 0.65, MEDIUM ≥ 0.30, LOW otherwise).
"""
import math
import time

import torch
from sentence_transformers import CrossEncoder

from .config import CFG

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load(model_name: str, fallback: str | None) -> tuple[CrossEncoder, str]:
    t0 = time.perf_counter()
    try:
        ce = CrossEncoder(model_name, device=_device, max_length=256, trust_remote_code=True)
        print(f"[reranker] loaded {model_name} on {_device} ({time.perf_counter()-t0:.1f}s)")
        return ce, model_name
    except Exception as e:
        if not fallback:
            raise
        print(f"[reranker] primary unavailable ({type(e).__name__}: {e}); falling back to {fallback}")
        ce = CrossEncoder(fallback, device=_device, max_length=256)
        print(f"[reranker] loaded {fallback} on {_device} ({time.perf_counter()-t0:.1f}s)")
        return ce, fallback


_ce, _active_model = _load(CFG.rerank_model, CFG.fallback_rerank)

_MAX_RERANK_CHARS = 800
_BATCH_SIZE = 32


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, (c["text"] or "")[:_MAX_RERANK_CHARS]) for c in candidates]
    scores = _ce.predict(pairs, batch_size=_BATCH_SIZE, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = _sigmoid(float(s))
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def active_model() -> str:
    return _active_model
