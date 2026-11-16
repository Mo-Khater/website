"""
Course-grounded retrieval for the AI tutor (Workers AI + Vectorize).

This module is the reusable retrieval layer: it turns an activity's own
materials into embedded chunks, keeps them in sync, and answers "which parts
of this course are relevant to this question?". It knows nothing about HTTP,
auth, or tutor policy — callers (worker.py) gather the sources, check access,
and decide what to do with the hits.

Storage
  * Vectorize (binding ``VECTORIZE``) holds only vectors. Each activity gets
    its own namespace, so a query can never return another course's content.
    No course text is stored in Vectorize metadata.
  * D1 table ``tutor_chunks`` holds the chunk text (AES-encrypted, like the
    rest of learn's course content) plus a content hash used to skip
    re-embedding chunks that have not changed.

Embeddings come from Workers AI (binding ``AI``).
"""

import hashlib
import re
from typing import Any, Awaitable, Callable, Dict, List

import js
from pyodide.ffi import to_js

EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"   # 768 dimensions, cosine
EMBED_DIMENSIONS = 768

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200
MAX_CHUNKS_PER_ACTIVITY = 2000   # hard cap so one course can't run up the AI bill
EMBED_BATCH_SIZE = 50
UPSERT_BATCH_SIZE = 500
SQL_IN_BATCH_SIZE = 50           # D1 allows at most 100 bound parameters
MAX_QUERY_CHARS = 1000
MAX_TOP_K = 20

SOURCE_TYPES = ("activity", "session", "material", "quiz_explanation")

Crypt = Callable[[str], Awaitable[str]]


class TutorNotConfigured(RuntimeError):
    """Raised when the AI or VECTORIZE binding is missing."""


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — easy to unit test
# ---------------------------------------------------------------------------

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS,
               overlap_chars: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Paragraph boundaries are preferred, then sentence boundaries, then a hard
    cut. Consecutive chunks share up to *overlap_chars* of trailing context so
    an answer that straddles a boundary is still retrievable.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    units: List[str] = []
    for para in _PARA_SPLIT_RE.split(text):
        para = _WS_RE.sub(" ", para).strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(para):
            while len(sentence) > max_chars:
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                units.append(sentence)

    chunks: List[str] = []
    current: List[str] = []
    for unit in units:
        if current and len(" ".join(current + [unit])) > max_chars:
            chunks.append(" ".join(current))
            tail: List[str] = []
            size = 0
            for prev in reversed(current):
                if size + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                size += len(prev) + 1
            current = tail
            if current and len(" ".join(current + [unit])) > max_chars:
                current = []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_id(activity_id: str, source_type: str, source_id: str, index: int) -> str:
    """Stable id shared by the D1 row and the Vectorize vector (<= 64 bytes)."""
    raw = f"{activity_id}|{source_type}|{source_id}|{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


def content_hash(source_title: str, text: str) -> str:
    """Hash of everything that affects the embedding, including the model."""
    raw = f"{EMBED_MODEL}\n{source_title}\n{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embedding_input(source_title: str, text: str) -> str:
    """Prefix the chunk with its source title so short chunks keep context."""
    return f"{source_title}\n{text}" if source_title else text


def build_chunks(activity_id: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn source documents into chunk records.

    Each source is ``{"source_type", "source_id", "title", "text"}``.
    """
    out: List[Dict[str, Any]] = []
    for src in sources:
        source_type = str(src.get("source_type") or "")
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source_type: {source_type!r}")
        source_id = str(src.get("source_id") or "")
        title = str(src.get("title") or "").strip()[:200]
        for index, piece in enumerate(chunk_text(str(src.get("text") or ""))):
            out.append({
                "id": chunk_id(activity_id, source_type, source_id, index),
                "activity_id": activity_id,
                "source_type": source_type,
                "source_id": source_id,
                "source_title": title,
                "chunk_index": index,
                "text": piece,
                "content_hash": content_hash(title, piece),
            })
            if len(out) >= MAX_CHUNKS_PER_ACTIVITY:
                return out
    return out


def _batches(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Workers AI / Vectorize interop
# ---------------------------------------------------------------------------

def _js(obj):
    return to_js(obj, dict_converter=js.Object.fromEntries)


def _py(obj):
    """Convert a JS proxy result to Python; plain Python objects pass through."""
    if isinstance(obj, (dict, list, str, int, float, type(None))):
        return obj
    to_py = getattr(obj, "to_py", None)
    return to_py() if callable(to_py) else obj


def _bindings(env):
    ai = getattr(env, "AI", None)
    index = getattr(env, "VECTORIZE", None)
    if ai is None or index is None:
        raise TutorNotConfigured("AI and VECTORIZE bindings are required for the tutor")
    return ai, index


def is_configured(env) -> bool:
    try:
        _bindings(env)
        return True
    except TutorNotConfigured:
        return False


async def embed_texts(env, texts: List[str]) -> List[List[float]]:
    """Embed *texts* with Workers AI, batching to stay under request limits."""
    ai, _ = _bindings(env)
    vectors: List[List[float]] = []
    for batch in _batches(texts, EMBED_BATCH_SIZE):
        result = _py(await ai.run(EMBED_MODEL, _js({"text": batch})))
        data = (result or {}).get("data") or []
        if len(data) != len(batch):
            raise RuntimeError(f"embedding count mismatch: sent {len(batch)}, got {len(data)}")
        vectors.extend([list(v) for v in data])
    return vectors


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

async def index_activity(env, activity_id: str, sources: List[Dict[str, Any]],
                         encrypt: Crypt) -> Dict[str, int]:
    """Bring the activity's index in line with *sources*.

    Only chunks whose content hash changed are re-embedded; chunks whose
    source disappeared are removed from both Vectorize and D1.
    """
    _, index = _bindings(env)
    desired = build_chunks(activity_id, sources)
    desired_ids = {c["id"] for c in desired}

    existing_res = await env.DB.prepare(
        "SELECT id, content_hash FROM tutor_chunks WHERE activity_id=?"
    ).bind(activity_id).all()
    existing = {row.id: row.content_hash for row in (existing_res.results or [])}

    changed = [c for c in desired if existing.get(c["id"]) != c["content_hash"]]
    stale = [cid for cid in existing if cid not in desired_ids]

    if changed:
        vectors = await embed_texts(
            env, [embedding_input(c["source_title"], c["text"]) for c in changed]
        )
        records = [
            {
                "id": c["id"],
                "values": vec,
                "namespace": activity_id,
                "metadata": {"source_type": c["source_type"], "source_id": c["source_id"]},
            }
            for c, vec in zip(changed, vectors)
        ]
        for batch in _batches(records, UPSERT_BATCH_SIZE):
            await index.upsert(_js(batch))

        for c in changed:
            await env.DB.prepare(
                "INSERT INTO tutor_chunks"
                " (id, activity_id, source_type, source_id, source_title, chunk_index,"
                "  text_encrypted, content_hash, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,datetime('now'))"
                " ON CONFLICT(id) DO UPDATE SET"
                "  source_title=excluded.source_title,"
                "  text_encrypted=excluded.text_encrypted,"
                "  content_hash=excluded.content_hash,"
                "  updated_at=excluded.updated_at"
            ).bind(
                c["id"], activity_id, c["source_type"], c["source_id"],
                c["source_title"], c["chunk_index"],
                await encrypt(c["text"]), c["content_hash"],
            ).run()

    for batch in _batches(stale, SQL_IN_BATCH_SIZE):
        await index.deleteByIds(_js(batch))
        placeholders = ",".join("?" for _ in batch)
        await env.DB.prepare(
            f"DELETE FROM tutor_chunks WHERE activity_id=? AND id IN ({placeholders})"
        ).bind(activity_id, *batch).run()

    return {
        "chunks": len(desired),
        "embedded": len(changed),
        "unchanged": len(desired) - len(changed),
        "removed": len(stale),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve(env, activity_id: str, query: str, decrypt: Crypt,
                   top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Return the activity's chunks most relevant to *query*, best first.

    Each hit carries what a caller needs to cite it: source type/id/title,
    chunk index, text, and similarity score.
    """
    _, index = _bindings(env)
    query = (query or "").strip()[:MAX_QUERY_CHARS]
    if not query:
        return []
    top_k = max(1, min(int(top_k), MAX_TOP_K))

    [vector] = await embed_texts(env, [query])
    result = _py(await index.query(
        _js(vector),
        _js({"topK": top_k, "namespace": activity_id, "returnMetadata": "none"}),
    ))
    matches = [m for m in ((result or {}).get("matches") or [])
               if float(m.get("score") or 0.0) >= min_score]
    if not matches:
        return []

    scores = {m["id"]: float(m.get("score") or 0.0) for m in matches}
    ids = list(scores)
    placeholders = ",".join("?" for _ in ids)
    rows_res = await env.DB.prepare(
        "SELECT id, source_type, source_id, source_title, chunk_index, text_encrypted"
        f" FROM tutor_chunks WHERE activity_id=? AND id IN ({placeholders})"
    ).bind(activity_id, *ids).all()

    hits = []
    for row in rows_res.results or []:
        hits.append({
            "chunk_id": row.id,
            "source_type": row.source_type,
            "source_id": row.source_id,
            "source_title": row.source_title,
            "chunk_index": row.chunk_index,
            "text": await decrypt(row.text_encrypted or ""),
            "score": round(scores.get(row.id, 0.0), 4),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits

