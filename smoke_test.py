"""End-to-end smoke test for the NVIDIA NIM integration.

Run after `pip install -r requirements.txt` and editing `.env`:
    python smoke_test.py
"""
import sys
import time

from openai import OpenAI

from src.config import CFG


def check_key() -> None:
    if not CFG.nvidia_api_key or not CFG.nvidia_api_key.startswith("nvapi-"):
        print("FAIL: NVIDIA_API_KEY missing or malformed. Edit .env.")
        sys.exit(1)
    print("OK  api key looks valid")


def check_models() -> None:
    client = OpenAI(api_key=CFG.nvidia_api_key, base_url=CFG.nvidia_base_url)
    for label, model in [("ctx", CFG.ctx_model), ("gen", CFG.gen_model)]:
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'pong' and nothing else."}],
                max_tokens=16,
                temperature=0,
            )
            dt = int((time.perf_counter() - t0) * 1000)
            ans = resp.choices[0].message.content.strip()
            print(f"OK  {label}={model:<40s} {dt:>5d} ms  reply={ans!r}")
        except Exception as e:
            print(f"FAIL {label}={model}: {e}")
            sys.exit(1)


def check_stream() -> None:
    client = OpenAI(api_key=CFG.nvidia_api_key, base_url=CFG.nvidia_base_url)
    print(f"\nStreaming test with {CFG.gen_model} (thinking={CFG.kimi_thinking})...")
    t0 = time.perf_counter()
    first_token_ms = None
    n_tokens = 0
    extra = {}
    if "kimi" in CFG.gen_model.lower():
        extra["chat_template_kwargs"] = {"thinking": CFG.kimi_thinking}
    stream = client.chat.completions.create(
        model=CFG.gen_model,
        messages=[{"role": "user", "content": "Count from 1 to 10."}],
        max_tokens=200,
        temperature=0,
        stream=True,
        extra_body=extra or None,
    )
    out = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta.content
        if d:
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
            out += d
            n_tokens += 1
    total_ms = int((time.perf_counter() - t0) * 1000)
    print(f"OK  TTFT={first_token_ms} ms  total={total_ms} ms  ~{n_tokens} chunks")
    print(f"    reply preview: {out[:120]!r}")


if __name__ == "__main__":
    check_key()
    check_models()
    check_stream()
    print("\nAll smoke checks passed.")
