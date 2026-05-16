"""Per-user session integration: chunk + embed + index uploaded content, with cleanup.

Each session has a unique ID. All chunks indexed during a session carry a `session_id`
payload tag so `clear_session()` can wipe them without touching the pre-ingested corpus.

Differences vs the offline ingest pipeline:
  * No Anthropic Contextual Retrieval (kept inline for speed — user-uploaded files are
    typically smaller and we don't want to block the chat on N×LLM calls).
  * `context_text == text` so dense embedding uses the raw chunk text directly.
"""
from typing import List

from qdrant_client import models

from .chunker import chunk_pages
from .config import CFG
from .parser import PageRecord
from .vector_store import _client, upsert


def _pages_from_processed(processed: dict) -> List[PageRecord]:
    """Build PageRecord(s) the chunker can consume, preserving granularity where possible."""
    kind = processed["kind"]
    name = processed["filename"]
    text = processed.get("text", "")

    if kind == "pdf" and processed.get("pages"):
        return [
            PageRecord(pdf=name, page=p["num"], text=p["text"], ocr_used=p.get("ocr", False))
            for p in processed["pages"]
            if p.get("text", "").strip()
        ]

    if kind in ("audio", "video") and processed.get("segments"):
        # Bucket whisper segments into 60-second pages so citations show t≈MM:SS granularity.
        buckets: dict[int, list[str]] = {}
        for seg in processed["segments"]:
            minute = int(seg.get("start", 0) // 60) + 1
            buckets.setdefault(minute, []).append(seg["text"])

        records: List[PageRecord] = []
        if kind == "video" and processed.get("frame_count", 0) > 0:
            # Append the full composed text (transcript + frame descriptions) as one extra
            # "page" so visual-only questions can still find a hit when the transcript is
            # silent on the topic.
            records.append(
                PageRecord(pdf=name, page=9999, text=text, ocr_used=False)
            )
        for m, texts in sorted(buckets.items()):
            joined = " ".join(t for t in texts if t.strip())
            if joined.strip():
                records.append(PageRecord(pdf=name, page=m, text=joined, ocr_used=False))
        if records:
            return records

    if not text.strip():
        return []
    return [PageRecord(pdf=name, page=1, text=text, ocr_used=False)]


def add_to_session(session_id: str, processed: dict) -> int:
    """Chunk + embed + index `processed` content. Returns number of chunks indexed."""
    pages = _pages_from_processed(processed)
    if not pages:
        return 0

    records = []
    for c in chunk_pages(pages):
        d = c.to_payload()
        d["context_text"] = d["text"]  # session uploads skip contextualization
        d["session_id"] = session_id
        d["kind"] = processed["kind"]
        d["filename"] = processed["filename"]
        records.append(d)

    if not records:
        return 0
    upsert(records)
    return len(records)


def clear_session(session_id: str) -> int:
    """Delete every point tagged with this session_id. Pre-ingested corpus is untouched."""
    try:
        _client.delete(
            collection_name=CFG.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        )
                    ]
                )
            ),
        )
        return 1
    except Exception:
        return 0
