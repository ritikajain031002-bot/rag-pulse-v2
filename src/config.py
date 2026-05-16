from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Config(BaseModel):
    pdf_dir: Path = Path("data/pdfs")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = "hackathon_rag"

    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    sparse_model: str = "Qdrant/bm25"
    # Reranker — BGE-v2-m3 (278M, classification head, ~5-10x faster on CPU
    # than Qwen3-Reranker-0.6B which uses autoregressive scoring).
    # Override with RERANK_MODEL env var; ms-marco-MiniLM-L-6-v2 (22M) is
    # the fastest option but English-only.
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    fallback_rerank: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )

    ctx_model: str = os.getenv("NVIDIA_CTX_MODEL", "meta/llama-3.1-8b-instruct")
    gen_model: str = os.getenv("NVIDIA_GEN_MODEL", "meta/llama-3.3-70b-instruct")

    # NVIDIA NIM models — verified working on this key (probed via streaming
    # /chat/completions in m0086-m0105). DO NOT add un-probed models — they
    # 404 or hang and surface as empty bubbles in the UI.
    #   ⚡ fast content streamers (sub-1s TTFT, content channel)
    #   💻 coder-tuned
    #   🧠 reasoning models (emit reasoning_content; UI shows them in a
    #      collapsible Thinking expander)
    gen_model_options: list[str] = [
        "meta/llama-3.3-70b-instruct",
        "mistralai/mistral-nemotron",
        "qwen/qwen3-next-80b-a3b-instruct",
        "qwen/qwen3-coder-480b-a35b-instruct",
        "qwen/qwen3.5-122b-a10b",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nvidia/nemotron-3-super-120b-a12b",
        # 🌙 long-context + thinking; ~55s TTFT on this key (probed 2026-05-16)
        "moonshotai/kimi-k2.6",
    ]
    # Multimodal vision model on NVIDIA NIM (Llama 3.2 Vision is widely available)
    vision_model: str = os.getenv(
        "NVIDIA_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct"
    )

    kimi_thinking: bool = os.getenv("KIMI_THINKING", "false").lower() == "true"

    # Whisper (local, faster-whisper / CTranslate2)
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")  # tiny|base|small|medium|large-v3
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")  # cpu|cuda
    whisper_compute: str = os.getenv("WHISPER_COMPUTE", "int8")
    whisper_beam: int = int(os.getenv("WHISPER_BEAM", "1"))

    # Video sampling
    video_frame_every_sec: int = int(os.getenv("VIDEO_FRAME_EVERY_SEC", "60"))
    video_max_frames: int = int(os.getenv("VIDEO_MAX_FRAMES", "8"))

    # Chunking — 500 tokens × 10% overlap keeps each chunk small so the
    # LLM prefill (top_k_rerank × chunk_size) stays under ~2K tokens for
    # sub-second TTFT.
    chunk_tokens: int = 500
    chunk_overlap_pct: float = 0.10

    # Retrieval — 12 candidates keeps rerank well under 1s on CPU with
    # BGE-v2-m3 (was 20 with Qwen3-Reranker → 253s on CPU, see m0156).
    # top_k_rerank=4 keeps the LLM prefill under ~1.5K tokens (m0187).
    top_k_retrieve: int = 12
    top_k_rerank: int = 4

    # OCR fallback threshold (char count below which we OCR a PDF page)
    ocr_text_density_threshold: int = 100

    # Session uploads
    session_dir: Path = Path("data/sessions")


CFG = Config()
CFG.session_dir.mkdir(parents=True, exist_ok=True)
