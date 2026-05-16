"""Anthropic Contextual Retrieval — prepend an LLM-generated situating context to each chunk.

Reduces top-20 retrieval failure by ~35% alone, ~49% when paired with BM25.
Source: https://www.anthropic.com/news/contextual-retrieval (Sept 2024)

LLM provider: NVIDIA NIM (OpenAI-compatible). Default ctx_model is llama-3.1-8b-instruct
because Kimi-K2.6 on a 60K-token prefill per chunk makes ingestion prohibitively slow.
Set NVIDIA_CTX_MODEL to override.
"""
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import CFG

_client = OpenAI(api_key=CFG.nvidia_api_key, base_url=CFG.nvidia_base_url)

PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (max 80 words) to situate this chunk \
within the overall document for the purposes of improving search retrieval \
of the chunk. Answer ONLY with the succinct context and nothing else."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def situate_chunk(doc_text: str, chunk_text: str) -> str:
    doc_window = doc_text[:60_000]
    resp = _client.chat.completions.create(
        model=CFG.ctx_model,
        messages=[
            {"role": "user", "content": PROMPT.format(doc=doc_window, chunk=chunk_text)},
        ],
        max_tokens=120,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def contextualize_chunks(doc_text: str, chunk_texts: list[str]) -> list[str]:
    out = []
    for c in chunk_texts:
        try:
            ctx = situate_chunk(doc_text, c)
            out.append(f"{ctx}\n\n{c}")
        except Exception as e:
            print(f"[ctx fail] {e}, using raw chunk")
            out.append(c)
    return out
