"""PDF processor — wraps the existing PyMuPDF + Tesseract parser for the unified router."""
from pathlib import Path
from typing import Any, Dict

from ..parser import parse_pdf


def process(path: Path) -> Dict[str, Any]:
    pages = list(parse_pdf(path))
    text_parts = [f"[Page {p.page}]\n{p.text}" for p in pages]
    return {
        "kind": "pdf",
        "source": str(path),
        "filename": path.name,
        "text": "\n\n".join(text_parts),
        "pages": [{"num": p.page, "text": p.text, "ocr": p.ocr_used} for p in pages],
        "page_count": len(pages),
        "ocr_pages": sum(1 for p in pages if p.ocr_used),
    }
