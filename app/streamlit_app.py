"""Universal Chat — drop any file (PDF / image / audio / video / Office / code / archive / URL), ask anything.

Single chat surface; processing is fully automatic. Files are auto-detected, processed
through the right pipeline (vision LLM, Whisper, ffmpeg, MarkItDown, etc.), chunked,
embedded, and indexed into Qdrant alongside any pre-ingested corpus. Queries hit a hybrid
BM25+dense search → cross-encoder rerank → Kimi-K2.6 generation with citations.
"""
import sys
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG  # noqa: E402
from src.media_router import (  # noqa: E402
    KIND_EMOJI,
    detect_kind,
    process_path,
    process_url,
)
from src.pipeline import RAGPipeline  # noqa: E402
from src.session import add_to_session, clear_session  # noqa: E402

st.set_page_config(
    page_title="Universal Chat — RAG over anything",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------- session state ----------------------------
def _new_sid() -> str:
    return uuid.uuid4().hex[:8]


if "sid" not in st.session_state:
    st.session_state.sid = _new_sid()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = {}  # name -> {kind, chunks, chars, filename}
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()
if "pipe" not in st.session_state:
    with st.spinner("⏳ Loading models (first run downloads weights, ~1-2 min)..."):
        st.session_state.pipe = RAGPipeline()
if "selected_model" not in st.session_state:
    st.session_state.selected_model = CFG.gen_model

pipe: RAGPipeline = st.session_state.pipe


# ---------------------------- helpers ----------------------------
def _process_upload(file) -> None:
    if file.file_id in st.session_state.processed_ids:
        return
    st.session_state.processed_ids.add(file.file_id)

    stage = st.status(f"⏳ {file.name}", expanded=False)
    try:
        tmp = Path(tempfile.gettempdir()) / f"sid_{st.session_state.sid}_{file.name}"
        tmp.write_bytes(file.getbuffer())

        kind = detect_kind(tmp)
        emoji = KIND_EMOJI.get(kind, "📁")
        stage.update(label=f"⏳ {file.name} → {emoji} {kind}")

        result = process_path(tmp)
        stage.update(label=f"⏳ {file.name}: indexing...")
        n_chunks = add_to_session(st.session_state.sid, result)

        st.session_state.files[file.name] = {
            "kind": result["kind"],
            "chunks": n_chunks,
            "chars": len(result["text"]),
            "filename": result["filename"],
            "extras": {k: result.get(k) for k in ("language", "duration", "frame_count", "members")},
        }
        stage.update(
            label=f"✅ {file.name} → {n_chunks} chunks ({len(result['text']):,} chars)",
            state="complete",
        )
    except Exception as e:  # noqa: BLE001
        stage.update(label=f"❌ {file.name}: {e}", state="error")


def _process_url(url: str) -> None:
    if url in st.session_state.files:
        return
    stage = st.status(f"⏳ Fetching {url}", expanded=False)
    try:
        result = process_url(url)
        n_chunks = add_to_session(st.session_state.sid, result)
        st.session_state.files[url] = {
            "kind": "web",
            "chunks": n_chunks,
            "chars": len(result["text"]),
            "filename": result["filename"],
            "extras": {},
        }
        stage.update(
            label=f"✅ {result['filename'][:50]} → {n_chunks} chunks",
            state="complete",
        )
    except Exception as e:  # noqa: BLE001
        stage.update(label=f"❌ {url}: {e}", state="error")


def _short(name: str, n: int = 32) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _render_sources(retrieved: list[dict], total_ms: int, timings: dict) -> None:
    header = (
        f"📎 {len(retrieved)} sources • {total_ms} ms "
        f"(retrieve {timings.get('retrieve_ms', 0)} ms, rerank {timings.get('rerank_ms', 0)} ms)"
    )
    with st.expander(header):
        for r in retrieved:
            page_str = (
                f"p.{r['page_start']}-{r['page_end']}"
                if r["page_start"] != r["page_end"]
                else f"p.{r['page_start']}"
            )
            kind = r.get("kind", "pdf")
            emoji = KIND_EMOJI.get(kind, "📁")
            score = r.get("rerank_score", r.get("score", 0))
            tag = ""
            if kind in ("audio", "video") and r["page_start"] != 9999:
                tag = f" • t≈{(r['page_start']-1):02d}:00"
            elif kind == "video" and r["page_start"] == 9999:
                tag = " • all keyframes"
            st.markdown(
                f"- {emoji} **{r['pdf']}** ({page_str}{tag}, score={score:.3f})"
            )
            with st.expander("preview"):
                preview = r["text"][:600]
                if len(r["text"]) > 600:
                    preview += "..."
                st.text(preview)


# ---------------------------- sidebar ----------------------------
with st.sidebar:
    st.markdown(f"### 🆔 Session `{st.session_state.sid}`")

    MODEL_LABELS = {
        "meta/llama-3.3-70b-instruct": "⚡ Llama 3.3 70B (fast, 0.6s TTFT)",
        "mistralai/mistral-nemotron": "⚡ Mistral-Nemotron (fastest, 0.5s)",
        "qwen/qwen3-next-80b-a3b-instruct": "⚡ Qwen3-Next 80B (fast, multilingual)",
        "qwen/qwen3-coder-480b-a35b-instruct": "💻 Qwen3-Coder 480B (best at code)",
        "qwen/qwen3.5-122b-a10b": "🧠 Qwen3.5 122B (reasoning model)",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5": "🧠 Nemotron-Super 49B (reasoning)",
        "nvidia/nemotron-3-super-120b-a12b": "🧠 Nemotron-3 120B (biggest, reasons+answers)",
        "moonshotai/kimi-k2.6": "🌙 Kimi K2.6 (long-context, ~55s TTFT)",
    }

    st.session_state.selected_model = st.selectbox(
        "🧠 Generation model",
        options=CFG.gen_model_options,
        index=(
            CFG.gen_model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model in CFG.gen_model_options
            else 0
        ),
        format_func=lambda m: MODEL_LABELS.get(m, m),
        help=(
            "⚡ = fast content streamer (under 1s first token). "
            "🧠 = reasoning model — answer streams into the Thinking expander while it works. "
            "All run on NVIDIA NIM."
        ),
    )

    uploads = st.file_uploader(
        "📎 Drop ANYTHING",
        accept_multiple_files=True,
        type=None,
        help="PDF • image • audio • video • Office • code • archive — any size.",
        key=f"uploader_{st.session_state.sid}",
    )

    url_input = st.text_input(
        "🌐 ...or paste a URL",
        placeholder="https://example.com/page",
        key=f"url_{st.session_state.sid}",
    )

    col_url1, col_url2 = st.columns([3, 1])
    add_url = col_url1.button(
        "Fetch URL", use_container_width=True, disabled=not url_input
    )

    if uploads:
        for f in uploads:
            _process_upload(f)

    if add_url and url_input:
        _process_url(url_input.strip())

    st.divider()

    if st.session_state.files:
        st.markdown("**📚 Active context:**")
        for name, info in st.session_state.files.items():
            emoji = KIND_EMOJI.get(info["kind"], "📁")
            chunks = info["chunks"]
            chars = info["chars"]
            extra_bits = []
            ex = info.get("extras", {}) or {}
            if ex.get("duration"):
                extra_bits.append(f"{int(ex['duration'])}s")
            if ex.get("language"):
                extra_bits.append(ex["language"])
            if ex.get("frame_count"):
                extra_bits.append(f"{ex['frame_count']} frames")
            if ex.get("members"):
                extra_bits.append(f"{len(ex['members'])} files")
            extras = " • ".join(extra_bits)
            st.markdown(
                f"{emoji} `{_short(info.get('filename', name))}`  \n"
                f"&nbsp;&nbsp;&nbsp;{info['kind']} • {chunks} chunks • {chars:,} chars"
                + (f" • {extras}" if extras else "")
            )
    else:
        st.info("Drop a file or paste a URL to add it to the chatbot's brain.")

    st.divider()

    if st.button("🗑️ Clear session", use_container_width=True):
        clear_session(st.session_state.sid)
        st.session_state.files = {}
        st.session_state.messages = []
        st.session_state.processed_ids = set()
        st.session_state.sid = _new_sid()
        st.rerun()

    with st.expander("⚙️ Stack"):
        st.caption(f"🧠 Gen: `{CFG.gen_model}`")
        st.caption(f"👁️ Vision: `{CFG.vision_model}`")
        st.caption(f"📝 Ctx: `{CFG.ctx_model}`")
        st.caption(f"🎙️ Whisper: `{CFG.whisper_model}` ({CFG.whisper_compute})")
        st.caption(f"🗂️ Dense: `{CFG.dense_model}`")
        st.caption(f"🎯 Rerank: `{CFG.rerank_model}`")
        st.caption(f"💾 Qdrant: `{CFG.qdrant_url}`")
        st.caption(
            f"🤔 Kimi thinking flag: **{'on' if CFG.kimi_thinking else 'off'}** "
            "(only applies if a Kimi model is in the dropdown)"
        )


# ---------------------------- main chat ----------------------------
st.markdown("# 🤖 Universal Chat")
st.caption(
    "Drop any file. Ask anything. Hybrid BM25+dense retrieval • cross-encoder rerank • "
    "NVIDIA NIM generation with grounded citations."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("reasoning"):
            with st.expander("💭 Model reasoning", expanded=False):
                st.markdown(msg["reasoning"])
        if msg.get("retrieved"):
            _render_sources(msg["retrieved"], msg.get("total_ms", 0), msg.get("timings", {}))

prompt = st.chat_input("Ask anything about your files or the knowledge base…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        st.caption(f"🧠 `{st.session_state.selected_model}`")

        with st.status("🔍 Retrieving + reranking…", expanded=False) as status:
            result = pipe.query(prompt, model=st.session_state.selected_model)
            status.update(
                label=(
                    f"✅ Retrieved {len(result.retrieved)} chunks "
                    f"(retrieve {result.timings.get('retrieve_ms', 0)}ms, "
                    f"rerank {result.timings.get('rerank_ms', 0)}ms)"
                ),
                state="complete",
            )

        think_container = st.empty()
        answer_slot = st.empty()
        answer_slot.markdown("⏳ _waiting for first token…_")
        reasoning = ""
        content = ""
        error_msg = ""
        first_token = False

        for kind, text in result.answer_stream:
            if not first_token:
                answer_slot.markdown("")
                first_token = True
            if kind == "reasoning":
                reasoning += text
                with think_container.container():
                    with st.expander("💭 Model reasoning (live)", expanded=True):
                        st.markdown(reasoning)
            elif kind == "content":
                content += text
                answer_slot.markdown(content + "▌")
            elif kind == "error":
                error_msg = text
                st.error(f"❌ Model error: {text}")
                break

        if reasoning:
            with think_container.container():
                with st.expander("💭 Model reasoning", expanded=False):
                    st.markdown(reasoning)

        if content:
            answer_slot.markdown(content)
            final_answer = content
        elif reasoning and not error_msg:
            answer_slot.markdown(reasoning)
            final_answer = reasoning
        elif error_msg:
            answer_slot.empty()
            final_answer = f"_(model error: {error_msg})_"
        else:
            answer_slot.markdown(
                "_(no answer — the model returned no content. "
                "Try a different model or check that your documents were indexed.)_"
            )
            final_answer = ""

        total_ms = int((time.perf_counter() - t0) * 1000)
        _render_sources(result.retrieved, total_ms, result.timings)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer,
                "reasoning": reasoning,
                "retrieved": result.retrieved,
                "total_ms": total_ms,
                "timings": result.timings,
            }
        )
