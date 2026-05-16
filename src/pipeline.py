from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from .config import CFG
from .generator import StreamPart, _quality_tier, stream_answer
from .reranker import rerank
from .vector_store import hybrid_search


@dataclass
class RAGResult:
    answer_stream: Iterator[StreamPart]
    retrieved: list[dict]
    timings: dict
    quality: str
    top_score: float


def _dedupe(chunks: list[dict]) -> list[dict]:
    """Drop near-duplicate chunks before reranking.

    Two markdown files uploaded with the same content (e.g. `finalized_urls.md`
    and `sid_56042020_finalized_urls.md` in m0163) otherwise crowd out the
    top-K. Signature = (filename, page, first-160-chars-stripped).
    """
    seen: set[tuple[str, int, str]] = set()
    out: list[dict] = []
    for c in chunks:
        name = (c.get("filename") or c.get("pdf") or "").lower()
        page = int(c.get("page_start") or 0)
        sig_text = (c.get("text") or "")[:160].strip().lower()
        sig = (sig_text, page, "") if sig_text else (name, page, "")
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


class RAGPipeline:
    def query(
        self,
        question: str,
        top_k_retrieve: int | None = None,
        top_k_rerank: int | None = None,
        model: str | None = None,
    ) -> RAGResult:
        top_k_retrieve = top_k_retrieve or CFG.top_k_retrieve
        top_k_rerank = top_k_rerank or CFG.top_k_rerank
        timings: dict = {}

        t0 = time.perf_counter()
        candidates = hybrid_search(question, top_k=top_k_retrieve)
        timings["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

        candidates = _dedupe(candidates)

        t0 = time.perf_counter()
        ranked = rerank(question, candidates, top_k=top_k_rerank)
        timings["rerank_ms"] = int((time.perf_counter() - t0) * 1000)

        quality, top_score = _quality_tier(ranked)

        return RAGResult(
            answer_stream=stream_answer(question, ranked, model=model),
            retrieved=ranked,
            timings=timings,
            quality=quality,
            top_score=top_score,
        )
