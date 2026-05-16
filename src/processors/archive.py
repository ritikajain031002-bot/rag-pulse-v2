"""Archive processor — unpacks .zip/.tar(.gz|.bz2) and recurses through the router."""
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict


def process(path: Path, route_fn) -> Dict[str, Any]:
    """`route_fn(file_path) -> processed dict` is injected to avoid circular import."""
    ext = "".join(path.suffixes).lower()
    text_parts: list[str] = []
    members: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if ext.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(td_path)
        elif ext.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
            with tarfile.open(path) as tf:
                tf.extractall(td_path)
        else:
            raise ValueError(f"Unsupported archive: {ext}")

        for f in sorted(td_path.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(td_path)
            try:
                result = route_fn(f)
                text_parts.append(f"=== {rel} ({result['kind']}) ===\n{result['text']}")
                members.append({"name": str(rel), "kind": result["kind"]})
            except Exception as e:
                text_parts.append(f"=== {rel} (error) ===\n[{e}]")
                members.append({"name": str(rel), "kind": "error"})

    return {
        "kind": "archive",
        "source": str(path),
        "filename": path.name,
        "text": "\n\n".join(text_parts),
        "members": members,
        "member_count": len(members),
    }
