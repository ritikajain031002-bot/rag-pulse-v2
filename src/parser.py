"""PyMuPDF text extraction with per-page Tesseract OCR fallback for scanned pages."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz
import pytesseract
from pdf2image import convert_from_path

from .config import CFG


@dataclass
class PageRecord:
    pdf: str
    page: int
    text: str
    ocr_used: bool


def parse_pdf(pdf_path: Path) -> Iterator[PageRecord]:
    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.name

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text") or ""
        ocr_used = False

        if len(text.strip()) < CFG.ocr_text_density_threshold:
            try:
                images = convert_from_path(
                    pdf_path, first_page=i + 1, last_page=i + 1, dpi=200
                )
                text = pytesseract.image_to_string(images[0], lang="eng")
                ocr_used = True
            except Exception as e:
                print(f"[OCR fail] {pdf_name} p{i + 1}: {e}")

        text = _clean(text)
        if text.strip():
            yield PageRecord(pdf=pdf_name, page=i + 1, text=text, ocr_used=ocr_used)

    doc.close()


def _clean(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
