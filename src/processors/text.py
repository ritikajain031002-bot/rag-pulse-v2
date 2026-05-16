"""Plain text / code / config files — direct read with auto-detected encoding."""
from pathlib import Path
from typing import Any, Dict


def process(path: Path) -> Dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            import chardet

            raw = path.read_bytes()
            enc = chardet.detect(raw).get("encoding") or "latin-1"
            content = raw.decode(enc, errors="replace")
        except Exception:
            content = path.read_text(encoding="latin-1", errors="replace")

    return {
        "kind": "text",
        "source": str(path),
        "filename": path.name,
        "text": content,
        "size_chars": len(content),
    }
