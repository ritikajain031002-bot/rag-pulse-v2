from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Iterator, Literal

from openai import OpenAI

from .config import CFG

_client = OpenAI(api_key=CFG.nvidia_api_key, base_url=CFG.nvidia_base_url)

StreamKind = Literal["reasoning", "content", "error"]
StreamPart = tuple[StreamKind, str]


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    if mime == "image/jpg":
        mime = "image/jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _iter_stream(stream) -> Iterator[StreamPart]:
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if reasoning:
            yield ("reasoning", reasoning)
        if content:
            yield ("content", content)


def stream_text(
    messages: list[dict], model: str | None = None, max_tokens: int = 900
) -> Iterator[StreamPart]:
    model = model or CFG.gen_model
    extra: dict = {}
    if "kimi" in model.lower():
        extra["chat_template_kwargs"] = {"thinking": CFG.kimi_thinking}

    try:
        stream = _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra or None,
        )
        yield from _iter_stream(stream)
    except Exception as e:
        yield ("error", f"{type(e).__name__}: {e}")


def stream_vision(
    question: str, image_paths: list[Path], max_tokens: int = 900
) -> Iterator[StreamPart]:
    if not image_paths:
        yield from stream_text(
            [{"role": "user", "content": question}], max_tokens=max_tokens
        )
        return

    content: list[dict] = [{"type": "text", "text": question}]
    for p in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(Path(p))},
            }
        )
    try:
        stream = _client.chat.completions.create(
            model=CFG.vision_model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=max_tokens,
            stream=True,
        )
        yield from _iter_stream(stream)
    except Exception as e:
        yield ("error", f"{type(e).__name__}: {e}")


def describe_image(path: Path) -> str:
    prompt = (
        "Describe this image in exhaustive detail for downstream search retrieval.\n"
        "Include:\n"
        "1. Overall scene/subject\n"
        "2. ALL visible text (OCR every word you can read)\n"
        "3. Objects, people, colors, spatial layout\n"
        "4. Charts/diagrams: extract every data point, axis label, and legend entry\n"
        "5. Mood, style, era cues\n\n"
        "Be thorough — this description is the ONLY context available later."
    )
    parts: list[str] = []
    for kind, text in stream_vision(prompt, [path], max_tokens=900):
        if kind in ("content", "reasoning"):
            parts.append(text)
    return "".join(parts)
