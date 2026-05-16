"""Qdrant wrapper: dense + BM25 sparse named vectors with RRF hybrid query.

`upsert` passes through any extra payload fields (e.g. session_id, kind, filename) so the
universal-chat session layer can tag chunks for later cleanup.
"""
import uuid

from qdrant_client import QdrantClient, models

from .config import CFG
from .embedder import embed, embed_query

_client = QdrantClient(url=CFG.qdrant_url)

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_RESERVED = {"id", "context_text"}


def _chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def reset_collection(dense_dim: int) -> None:
    if _client.collection_exists(CFG.collection):
        _client.delete_collection(CFG.collection)
    _client.create_collection(
        collection_name=CFG.collection,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim, distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )
    # Index session_id so clear_session deletions are O(log n) instead of full-scan
    try:
        _client.create_payload_index(
            collection_name=CFG.collection,
            field_name="session_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def ensure_collection(dense_dim: int) -> None:
    """Create the collection if missing (used by universal-chat on first session upload)."""
    if not _client.collection_exists(CFG.collection):
        reset_collection(dense_dim)


def upsert(chunks: list[dict]) -> None:
    if not chunks:
        return

    # Make sure the collection exists; pull dim from first dense embedding lazily below.
    dense_inputs = [c["context_text"] for c in chunks]
    dense_vecs = embed(dense_inputs)

    if not _client.collection_exists(CFG.collection):
        ensure_collection(len(dense_vecs[0]))

    points = []
    for c, v in zip(chunks, dense_vecs):
        ctx_only = c.get("context_text", "")
        if ctx_only.endswith(c["text"]):
            ctx_only = ctx_only[: -len(c["text"])].strip()

        # Pass through every field except reserved bookkeeping keys.
        payload = {k: v2 for k, v2 in c.items() if k not in _RESERVED}
        payload["context"] = ctx_only

        points.append(
            models.PointStruct(
                id=_chunk_uuid(c["id"]),
                vector={
                    "dense": v,
                    "sparse": models.Document(
                        text=c["context_text"], model=CFG.sparse_model
                    ),
                },
                payload=payload,
            )
        )

    BATCH = 64
    for i in range(0, len(points), BATCH):
        _client.upsert(collection_name=CFG.collection, points=points[i : i + BATCH])


def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    q_dense = embed_query(query)
    res = _client.query_points(
        collection_name=CFG.collection,
        prefetch=[
            models.Prefetch(query=q_dense, using="dense", limit=top_k * 2),
            models.Prefetch(
                query=models.Document(text=query, model=CFG.sparse_model),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    out: list[dict] = []
    for p in res.points:
        payload = p.payload or {}
        out.append(
            {
                "id": str(p.id),
                "score": float(p.score) if p.score is not None else 0.0,
                "pdf": payload.get("pdf", "?"),
                "page_start": payload.get("page_start", 1),
                "page_end": payload.get("page_end", 1),
                "text": payload.get("text", ""),
                "context": payload.get("context", ""),
                "kind": payload.get("kind", "pdf"),
                "filename": payload.get("filename", payload.get("pdf", "?")),
                "session_id": payload.get("session_id"),
            }
        )
    return out
