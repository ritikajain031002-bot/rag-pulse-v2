"""Probe NVIDIA NIM models for reasoning_content vs content streaming behavior.

Tells us whether Kimi-K2.6 (and friends) emit `delta.reasoning_content` that our
current generator.py is dropping on the floor.
"""

from __future__ import annotations

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Users/apple/Desktop/Rag/.env")

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    timeout=170.0,
)

MODELS = [
    ("moonshotai/kimi-k2.6", True),         # thinking-capable
    ("deepseek-ai/deepseek-v4-pro", False),
    ("meta/llama-3.3-70b-instruct", False),
    ("mistralai/mistral-large-2-instruct", False),
]

PROMPT = "What is 2 + 2? Answer in one short sentence."

for model, is_kimi in MODELS:
    print(f"\n{'=' * 70}\nMODEL: {model}\n{'=' * 70}")
    extra = {"chat_template_kwargs": {"thinking": False}} if is_kimi else None
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.1,
            max_tokens=200,
            stream=True,
            extra_body=extra,
        )
        reasoning_chars = 0
        content_chars = 0
        other_keys: set[str] = set()
        first_chunk_t = None
        n_chunks = 0
        sample_reasoning: list[str] = []
        sample_content: list[str] = []
        for chunk in stream:
            n_chunks += 1
            if first_chunk_t is None:
                first_chunk_t = time.time() - t0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            for k in ("content", "reasoning_content"):
                v = getattr(delta, k, None)
                if v:
                    if k == "content":
                        content_chars += len(v)
                        if len(sample_content) < 3:
                            sample_content.append(v)
                    else:
                        reasoning_chars += len(v)
                        if len(sample_reasoning) < 3:
                            sample_reasoning.append(v)
            try:
                d = delta.model_dump(exclude_none=True)
                for k in d:
                    if k not in ("content", "reasoning_content", "role"):
                        other_keys.add(k)
            except Exception:
                pass
        elapsed = time.time() - t0
        print(f"  chunks      : {n_chunks}")
        print(f"  ttft        : {first_chunk_t:.2f}s" if first_chunk_t else "  ttft        : n/a")
        print(f"  total       : {elapsed:.2f}s")
        print(f"  reasoning   : {reasoning_chars} chars  sample={sample_reasoning!r}")
        print(f"  content     : {content_chars} chars  sample={sample_content!r}")
        if other_keys:
            print(f"  other keys  : {other_keys}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

print("\nDONE")
