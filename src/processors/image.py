"""Image processor — vision model produces a search-rich description (includes OCR)."""
from pathlib import Path
from typing import Any, Dict

from ..multimodal_llm import describe_image


def process(path: Path) -> Dict[str, Any]:
    # pillow-heif registers HEIF/HEIC support globally when imported
    try:
        from pillow_heif import register_heif_opener  # noqa: F401

        register_heif_opener()
    except Exception:
        pass

    description = describe_image(path)
    return {
        "kind": "image",
        "source": str(path),
        "filename": path.name,
        "text": description,
    }
