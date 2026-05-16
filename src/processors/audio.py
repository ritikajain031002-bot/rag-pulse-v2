"""Audio processor — faster-whisper transcription. Handles arbitrary length (streams).

Model singleton is lazy so the first call pays the load cost, subsequent calls reuse.
"""
from pathlib import Path
from typing import Any, Dict

from ..config import CFG

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            CFG.whisper_model,
            device=CFG.whisper_device,
            compute_type=CFG.whisper_compute,
        )
    return _model


def process(path: Path) -> Dict[str, Any]:
    model = _get_model()
    segments_gen, info = model.transcribe(
        str(path), beam_size=CFG.whisper_beam, vad_filter=True
    )
    parts: list[str] = []
    seg_list: list[dict] = []
    for s in segments_gen:
        parts.append(s.text)
        seg_list.append({"start": s.start, "end": s.end, "text": s.text})

    return {
        "kind": "audio",
        "source": str(path),
        "filename": path.name,
        "text": " ".join(parts).strip(),
        "segments": seg_list,
        "language": info.language,
        "duration": info.duration,
    }
