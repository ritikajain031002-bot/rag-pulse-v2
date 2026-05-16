"""Token-aware chunker over consecutive pages, preserves (pdf, page) metadata."""
import hashlib
from dataclasses import asdict, dataclass
from typing import Iterator

import tiktoken

from .config import CFG
from .parser import PageRecord


@dataclass
class Chunk:
    id: str
    pdf: str
    page_start: int
    page_end: int
    text: str

    def to_payload(self) -> dict:
        return asdict(self)


_enc = tiktoken.get_encoding("cl100k_base")


def chunk_pages(pages: list[PageRecord]) -> Iterator[Chunk]:
    target = CFG.chunk_tokens
    overlap = int(target * CFG.chunk_overlap_pct)

    tokens: list[tuple[int, int]] = []
    for p in pages:
        for tok in _enc.encode(p.text):
            tokens.append((p.page, tok))

    if not tokens:
        return

    pdf = pages[0].pdf
    i = 0
    while i < len(tokens):
        window = tokens[i : i + target]
        if not window:
            break
        page_start = window[0][0]
        page_end = window[-1][0]
        text = _enc.decode([t for _, t in window])
        cid = hashlib.sha1(f"{pdf}:{i}:{text[:50]}".encode()).hexdigest()
        yield Chunk(
            id=cid,
            pdf=pdf,
            page_start=page_start,
            page_end=page_end,
            text=text,
        )
        if i + target >= len(tokens):
            break
        i += target - overlap
