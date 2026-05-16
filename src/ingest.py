"""CLI ingestion: parse -> chunk -> contextualize -> embed -> upsert.

Usage:
    python -m src.ingest                  # full pipeline with contextualization
    python -m src.ingest --no-context     # skip contextualization (3-5x faster)
    python -m src.ingest --no-reset       # keep existing collection
"""
import argparse

from .chunker import chunk_pages
from .config import CFG
from .contextualizer import contextualize_chunks
from .embedder import DIM
from .parser import PageRecord, parse_pdf
from .vector_store import reset_collection, upsert


def run(reset: bool = True, skip_context: bool = False) -> None:
    pdf_paths = sorted(CFG.pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {CFG.pdf_dir}")
    if not pdf_paths:
        print(f"  Drop PDFs into {CFG.pdf_dir.resolve()} and re-run.")
        return

    if reset:
        print(f"Resetting collection (dim={DIM})...")
        reset_collection(dense_dim=DIM)

    for path in pdf_paths:
        print(f"\nProcessing {path.name}")
        pages: list[PageRecord] = list(parse_pdf(path))
        if not pages:
            print("  no extractable pages")
            continue
        ocr_n = sum(p.ocr_used for p in pages)
        print(f"  {len(pages)} pages ({ocr_n} OCR'd)")

        chunks = list(chunk_pages(pages))
        print(f"  {len(chunks)} chunks")

        full_doc = "\n\n".join(p.text for p in pages)
        chunk_texts = [c.text for c in chunks]

        if skip_context:
            ctx_chunks = chunk_texts
        else:
            print(f"  contextualizing via NVIDIA NIM ({CFG.ctx_model})...")
            ctx_chunks = contextualize_chunks(full_doc, chunk_texts)

        records = []
        for c, ctx_text in zip(chunks, ctx_chunks):
            records.append({**c.to_payload(), "context_text": ctx_text})

        upsert(records)
        print(f"  upserted {len(records)} points")

    print("\nIngestion complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reset", action="store_true", help="Keep existing collection")
    ap.add_argument(
        "--no-context",
        action="store_true",
        help="Skip Contextual Retrieval (faster, lower quality)",
    )
    args = ap.parse_args()
    run(reset=not args.no_reset, skip_context=args.no_context)
