from __future__ import annotations

from typing import Iterator, Literal

from openai import OpenAI

from .config import CFG

_client = OpenAI(api_key=CFG.nvidia_api_key, base_url=CFG.nvidia_base_url)


SYSTEM_TEMPLATE = """You are a precise research assistant grounded in the provided EXCERPTS.

CONTEXT_QUALITY: {quality}  (top relevance score = {top_score:.2f}, on a 0-1 scale)

Rules:
1. When CONTEXT_QUALITY is HIGH or MEDIUM, answer primarily from EXCERPTS. After each \
factual claim, cite as [filename, p.PAGE]. Quote sparingly.
2. When CONTEXT_QUALITY is LOW:
   - Open with: "These documents don't seem to cover your question directly."
   - In one sentence, say what the EXCERPTS ARE about, so the user knows what their corpus contains.
   - Then either (a) answer what you CAN from the excerpts with citations, or
     (b) suggest specifically what file/topic the user should upload to get a real answer.
   - If the question is general knowledge (definition, how something works, etc.), \
you MAY add a short answer prefixed with "(General knowledge — not from your documents)" — \
keep it tight, 1-3 sentences.
3. If the QUESTION itself is too vague to act on (e.g. "tell me", "what about this?", \
"explain", with no specific topic), ASK exactly what they want to know instead of \
generically describing the documents.
4. Use prior conversation turns to resolve coreference ("it", "that", "he", "she").
5. Be direct. Never write filler like "according to the context" or "based on the excerpts".
6. Vary your wording — do not reuse the same opening sentence across turns.

OUTPUT FORMAT (applies when CONTEXT_QUALITY is HIGH or MEDIUM):

**TL;DR:** <one or two sentences summarizing the answer across all relevant sources>

## <Source title or topic>
- <fact bullet — one line each>
- <fact bullet>
- _Source: <filename>, p.<PAGE_RANGE>_

## <Next source or topic>
- <fact bullet>
- _Source: <filename>, p.<PAGE_RANGE>_

Format rules:
- One `## ` section per source file (or per distinct topic if the answer spans \
multiple aspects of a single source).
- Citations go on a trailing `_Source: …_` italic line, NOT inline inside bullets.
- Combine multiple excerpts that back the same section: \
`_Source: foo.pdf, p.1-3; bar.md, p.1_`.
- Even for a single-source answer, still use TL;DR + one `## ` section — \
do NOT collapse to a paragraph.
- **Bold** key identifiers (names, IDs, numbers). Use `code` for literal codes, \
GSTINs, URLs, filenames.

When CONTEXT_QUALITY is LOW, follow rule 2 instead — do NOT use this section \
format for the "documents don't cover your question" response."""

StreamKind = Literal["reasoning", "content", "error"]
StreamPart = tuple[StreamKind, str]


def _quality_tier(chunks: list[dict]) -> tuple[str, float]:
    """Returns ('HIGH'|'MEDIUM'|'LOW', top_score) based on rerank_score.

    Scores are sigmoid-normalized to [0,1] in reranker.py. Thresholds tuned
    empirically against the user's m0163 example where rerank scores
    0.05-0.25 should clearly be flagged as LOW (off-topic context).
    """
    if not chunks:
        return "LOW", 0.0
    scores = [
        c.get("rerank_score") for c in chunks if c.get("rerank_score") is not None
    ]
    if not scores:
        return "MEDIUM", 0.5
    top = max(scores)
    if top >= 0.65:
        tier = "HIGH"
    elif top >= 0.30:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    return tier, float(top)


_MAX_CHUNK_CHARS = 1400


def _truncate(text: str, n: int = _MAX_CHUNK_CHARS) -> str:
    if not text or len(text) <= n:
        return text or ""
    return text[:n].rstrip() + "…"


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    parts = ["EXCERPTS:"]
    for i, c in enumerate(chunks, 1):
        score = c.get("rerank_score")
        score_str = f"score={score:.2f} " if score is not None else ""
        parts.append(
            f"\n[{i}] {score_str}pdf={c['pdf']} page={c['page_start']}-{c['page_end']}\n"
            f"{_truncate(c['text'])}\n"
        )
    parts.append(f"\nQUESTION: {question}\n\nANSWER:")
    return "\n".join(parts)


def stream_answer(
    question: str,
    chunks: list[dict],
    model: str | None = None,
    history: list[dict] | None = None,
) -> Iterator[StreamPart]:
    model = model or CFG.gen_model
    extra_body: dict = {}
    is_thinking_model = "kimi" in model.lower() and CFG.kimi_thinking
    if "kimi" in model.lower():
        extra_body["chat_template_kwargs"] = {"thinking": CFG.kimi_thinking}

    # Reasoning models consume max_tokens on internal thought BEFORE producing
    # content. A 600-token cap meant the model ran out of budget mid-thinking
    # and never emitted any `content`. For thinking-mode runs we pass None,
    # which tells the NIM endpoint to use the model's full context window
    # (16K+ for Kimi K2.6). Non-thinking models keep a conservative cap.
    max_tokens = None if is_thinking_model else 600

    quality, top_score = _quality_tier(chunks)
    system = SYSTEM_TEMPLATE.format(quality=quality, top_score=top_score)
    messages: list[dict] = [{"role": "system", "content": system}]

    if history:
        for h in history[-16:]:
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append(
        {"role": "user", "content": build_user_prompt(question, chunks)}
    )

    try:
        stream = _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body or None,
        )
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
    except Exception as e:
        yield ("error", f"{type(e).__name__}: {e}")
