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
    timeout=45.0,
)

model = sys.argv[1] if len(sys.argv) > 1 else "meta/llama-3.3-70b-instruct"
print(f"probing {model}", flush=True)
t0 = time.time()
try:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 2+2? Reply in one short sentence."}],
        temperature=0.1,
        max_tokens=100,
        stream=True,
    )
    reasoning, content = "", ""
    n_chunks = 0
    first_chunk_t = None
    for chunk in stream:
        n_chunks += 1
        if first_chunk_t is None:
            first_chunk_t = time.time() - t0
            print(f"  first chunk in {first_chunk_t:.2f}s", flush=True)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        r = getattr(delta, "reasoning_content", None)
        c = getattr(delta, "content", None)
        if r:
            reasoning += r
        if c:
            content += c
    elapsed = time.time() - t0
    print(f"  chunks={n_chunks} total={elapsed:.2f}s reasoning={len(reasoning)} content={len(content)}", flush=True)
    print(f"  REASONING: {reasoning!r}", flush=True)
    print(f"  CONTENT  : {content!r}", flush=True)
except Exception as e:
    print(f"  ERROR {type(e).__name__}: {e}", flush=True)
