"""Detects file type and dispatches to the right processor.

`detect()` is extension-first (fast), with `filetype.guess` as a magic-byte fallback for
files with no extension. Unknown binaries default to "text" (read with errors='replace').
"""
from pathlib import Path
from typing import Any, Dict

import filetype

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".avif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".wma", ".opus", ".aiff", ".amr"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".3gp"}
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".scala",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".xml", ".csv", ".tsv",
    ".sql", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".log", ".ini", ".cfg", ".conf", ".env", ".dockerfile", ".makefile",
    ".vue", ".svelte", ".astro", ".tex", ".bib",
}
OFFICE_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".ods", ".odp", ".epub", ".rtf"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz2"}

KIND_EMOJI = {
    "pdf": "📄",
    "image": "🖼️",
    "audio": "🎵",
    "video": "🎬",
    "text": "📝",
    "office": "📊",
    "archive": "📦",
    "web": "🌐",
}


def detect(path: Path) -> str:
    """Return one of: 'pdf','image','audio','video','text','office','archive'."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in OFFICE_EXTS:
        return "office"
    if ext in ARCHIVE_EXTS:
        return "archive"
    if ext in TEXT_EXTS:
        return "text"

    # Unknown extension — sniff magic bytes
    try:
        kind = filetype.guess(str(path))
        if kind:
            mime = kind.mime
            if mime.startswith("image/"):
                return "image"
            if mime.startswith("audio/"):
                return "audio"
            if mime.startswith("video/"):
                return "video"
            if mime == "application/pdf":
                return "pdf"
            if mime in {"application/zip", "application/x-tar", "application/gzip", "application/x-bzip2"}:
                return "archive"
    except Exception:
        pass

    # Last resort — try as text
    return "text"


def process_path(path: Path) -> Dict[str, Any]:
    """Route a local file to its processor and return the standard dict."""
    path = Path(path)
    kind = detect(path)

    # Local imports avoid loading heavy deps (whisper, vision) unless needed
    if kind == "pdf":
        from .processors import pdf_doc
        return pdf_doc.process(path)
    if kind == "image":
        from .processors import image
        return image.process(path)
    if kind == "audio":
        from .processors import audio
        return audio.process(path)
    if kind == "video":
        from .processors import video
        return video.process(path)
    if kind == "office":
        from .processors import office
        return office.process(path)
    if kind == "archive":
        from .processors import archive
        return archive.process(path, process_path)
    # text fallback
    from .processors import text
    return text.process(path)


def process_url(url: str) -> Dict[str, Any]:
    from .processors import web
    return web.process(url)


detect_kind = detect
