"""Video processor — ffmpeg extracts audio + keyframes, then audio.py and image.py do the work.

Robust to any length: faster-whisper streams segments, keyframe extraction is capped at
CFG.video_max_frames so a 10-hour film doesn't generate 600 vision calls.
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from ..config import CFG
from . import audio as audio_proc
from . import image as image_proc


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _extract_audio(video_path: Path, wav_path: Path) -> None:
    _run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(wav_path), "-y", "-loglevel", "error",
        ]
    )


def _extract_frames(video_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%03d.jpg"
    _run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps=1/{CFG.video_frame_every_sec}",
            "-frames:v", str(CFG.video_max_frames),
            "-q:v", "3",  # decent JPEG quality, much smaller than raw
            str(pattern), "-y", "-loglevel", "error",
        ]
    )
    return sorted(out_dir.glob("frame_*.jpg"))


def process(path: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav = td_path / "audio.wav"

        # 1. audio -> transcript
        transcript: Dict[str, Any] = {"text": "", "segments": [], "language": None, "duration": 0}
        try:
            _extract_audio(path, wav)
            transcript = audio_proc.process(wav)
        except Exception as e:
            transcript["text"] = f"[audio extraction failed: {e}]"

        # 2. keyframes -> vision descriptions
        frames = []
        try:
            frames = _extract_frames(path, td_path / "frames")
        except Exception as e:
            print(f"[video frame extract failed]: {e}")

        frame_descs: list[str] = []
        for f in frames:
            try:
                frame_descs.append(image_proc.process(f)["text"])
            except Exception as e:
                frame_descs.append(f"[frame describe error: {e}]")

    # Assemble searchable text
    header = (
        f"=== VIDEO TRANSCRIPT (lang={transcript.get('language') or '?'}, "
        f"duration={transcript.get('duration', 0):.0f}s) ==="
    )
    body = transcript.get("text", "") or "[no speech detected]"
    visual = "\n".join(
        f"\n[Keyframe {i + 1}]\n{d}" for i, d in enumerate(frame_descs)
    )
    text = f"{header}\n{body}\n\n=== VISUAL KEYFRAMES ({len(frame_descs)}) ==={visual}"

    return {
        "kind": "video",
        "source": str(path),
        "filename": path.name,
        "text": text,
        "transcript": transcript.get("text", ""),
        "segments": transcript.get("segments", []),
        "language": transcript.get("language"),
        "duration": transcript.get("duration", 0),
        "frame_count": len(frame_descs),
    }
