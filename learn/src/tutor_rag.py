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

Citations
  Every chunk records its source (type, id, title) and a *locator* inside
  that source — "slide 7", "page 3", "section: Photosynthesis" — so the tutor
  can say exactly where an answer came from. A chunk spans two sections
  only when they are short; its locator then names the first and the last.

Chunking
  Text is cleaned first: Markdown links keep their text but lose the URL,
  images keep their alt text. Short sections (but not slides) join their
  neighbours, so a heading's few lines aren't cut off from what follows.
  A section that fits in one chunk is one chunk. A longer one is split by
  meaning (semantic chunking): each sentence is embedded with its
  neighbours, and a chunk ends at the biggest change of topic once it has
  SEMANTIC_MIN_CHARS. If that embedding fails, sections are split at
  paragraphs, then sentences, with some overlap. A chunk whose text
  already appears elsewhere in the course (a repeated footer or menu) is
  indexed once.

Retrieval
  Vector search with the question (and rewrites of it), then a reranker
  orders the candidates against the question. A caller can pass keywords
  of the question: candidates whose text contains them, especially the
  rarer ones, move up (keyword_boost). Matching runs on the candidates'
  decrypted text at query time; no keyword index is stored.

Embeddings come from Workers AI (binding ``AI``).
"""

import hashlib
import io
import json
import math
import posixpath
import re
import zipfile
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree

import js
from pyodide.ffi import to_js

EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"
EMBED_DIMENSIONS = 768

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200
MIN_SECTION_CHARS = 300
SEMANTIC_CHUNKING = True
SEMANTIC_MIN_CHARS = 400
MAX_SEMANTIC_SENTENCES = 5000
MAX_CHUNKS_PER_ACTIVITY = 2000
EMBED_BATCH_SIZE = 50
UPSERT_BATCH_SIZE = 500
SQL_IN_BATCH_SIZE = 50
MAX_QUERY_CHARS = 1000
MAX_TOP_K = 20

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "@cf/baai/bge-reranker-base"
RERANK_CANDIDATES = 20
RERANK_MAX_CHARS = 1600

SOURCE_TYPES = ("activity", "session", "material", "quiz_explanation")

Crypt = Callable[[str], Awaitable[str]]


class TutorNotConfigured(RuntimeError):
    """Raised when the AI or VECTORIZE binding is missing."""


_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_SLIDE_RE = re.compile(r"slide \d", re.I)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^()\s]|\([^()\s]*\))*(?:\s+\"[^\"]*\")?\)")


def clean_text(text: str) -> str:
    """Drop the URLs of Markdown links and images, keeping their text.

    Converted web pages and documents are full of links; the URLs swamp the
    embedding of short passages and cost model tokens. A URL written out as
    the link's text ("[https://...](https://...)") is kept.
    """
    text = _MD_IMAGE_RE.sub(lambda m: m.group(1), text or "")
    return _MD_LINK_RE.sub(lambda m: m.group(1), text)


def merge_short_sections(
    sections: List[Dict[str, str]], min_chars: int = MIN_SECTION_CHARS, max_chars: int = CHUNK_MAX_CHARS
) -> List[Dict[str, str]]:
    """Join a section shorter than *min_chars* to its neighbour when the two fit
    in one chunk, so a short section keeps its context. The locator of a
    joined section names the first and last: "section: Intro – section: Usage".

    Slides are never joined: most slides are short, and "slide 7" is the
    citation a student can look up."""
    merged: List[Dict[str, str]] = []
    for sec in sections:
        text = (sec.get("text") or "").strip()
        if not text:
            continue
        prev = merged[-1] if merged else None
        slides = prev and (_SLIDE_RE.match(prev["last"]) or _SLIDE_RE.match(sec.get("locator") or ""))
        if (
            prev
            and not slides
            and (len(prev["text"]) < min_chars or len(text) < min_chars)
            and len(prev["text"]) + 2 + len(text) <= max_chars
        ):
            prev["text"] += "\n\n" + text
            prev["last"] = sec.get("locator") or ""
        else:
            merged.append({"locator": sec.get("locator") or "", "last": sec.get("locator") or "", "text": text})
    out = []
    for sec in merged:
        first, last = sec["locator"], sec["last"]
        locator = first if first == last or not last else (f"{first} – {last}" if first else last)
        out.append({"locator": locator[:120], "text": sec["text"]})
    return out


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> List[str]:
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


def content_hash(source_title: str, locator: str, text: str) -> str:
    """Hash of everything stored or embedded for a chunk, including the model."""
    raw = f"{EMBED_MODEL}\n{source_title}\n{locator}\n{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embedding_input(source_title: str, locator: str, text: str) -> str:
    """Prefix the chunk with where it comes from so short chunks keep context."""
    header = " — ".join(p for p in (source_title, locator) if p)
    return f"{header}\n{text}" if header else text


def source_sections(src: Dict[str, Any]) -> List[Dict[str, str]]:
    """A source is either ``{"text": ...}`` or ``{"sections": [{"locator", "text"}]}``."""
    if src.get("sections") is not None:
        return [
            {"locator": str(sec.get("locator") or "")[:120], "text": str(sec.get("text") or "")}
            for sec in src["sections"]
        ]
    return [{"locator": "", "text": str(src.get("text") or "")}]


def prepared_sections(src: Dict[str, Any]) -> List[Dict[str, str]]:
    """A source's sections as they are chunked: cleaned, short ones merged."""
    return merge_short_sections(
        [{"locator": sec["locator"], "text": clean_text(sec["text"])} for sec in source_sections(src)]
    )


def sentences_of(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Sentences of *text*, paragraph by paragraph; one longer than *max_chars* is cut."""
    out: List[str] = []
    for para in _PARA_SPLIT_RE.split((text or "").replace("\r\n", "\n")):
        para = _WS_RE.sub(" ", para).strip()
        for sentence in _SENTENCE_SPLIT_RE.split(para) if para else []:
            sentence = sentence.strip()
            while len(sentence) > max_chars:
                out.append(sentence[:max_chars])
                sentence = sentence[max_chars:].strip()
            if sentence:
                out.append(sentence)
    return out


def pack_sentences(
    sentences: List[str], distances: List[float], min_chars: int = SEMANTIC_MIN_CHARS, max_chars: int = CHUNK_MAX_CHARS
) -> List[str]:
    """Group *sentences* into chunks of at most *max_chars*, each ending at the
    biggest change of topic once it has *min_chars*.

    ``distances[i]`` is how far apart in meaning sentences i and i+1 are. The
    chunks don't overlap: they end where the topic changes, so there is less
    to lose at the edges than with fixed-size cuts.
    """
    out: List[str] = []
    start = 0
    while start < len(sentences):
        size, end, best, best_distance = 0, start, None, -1.0
        while end < len(sentences) and size + len(sentences[end]) + (1 if size else 0) <= max_chars:
            size += len(sentences[end]) + (1 if size else 0)
            end += 1
            if size >= min_chars and end < len(sentences) and distances[end - 1] > best_distance:
                best, best_distance = end, distances[end - 1]
        if end == start:
            end = start + 1
        cut = end if end >= len(sentences) or best is None else best
        out.append(" ".join(sentences[start:cut]))
        start = cut
    return out


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return 1.0 - (dot / norm if norm else 0.0)


async def semantic_pieces(env, sources: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Chunks for every section longer than one chunk, split by meaning.

    Returns ``{section text: [chunk, ...]}`` for :func:`build_chunks`. Each
    sentence is embedded together with the sentences either side of it, so
    one short sentence doesn't look like a change of topic. Sections past
    MAX_SEMANTIC_SENTENCES are left out (split the plain way).
    """
    plans, windows = [], []
    for src in sources:
        for section in prepared_sections(src):
            text = section["text"]
            if len(text) <= CHUNK_MAX_CHARS or any(text == p[0] for p in plans):
                continue
            sentences = sentences_of(text)
            if len(sentences) < 2 or len(windows) + len(sentences) > MAX_SEMANTIC_SENTENCES:
                continue
            plans.append((text, sentences, len(windows)))
            windows += [
                " ".join(sentences[lo:hi]) for lo, hi in ((max(0, i - 1), i + 2) for i in range(len(sentences)))
            ]
    if not windows:
        return {}
    vectors = await embed_texts(env, windows)
    pieces = {}
    for text, sentences, start in plans:
        end = start + len(sentences)
        v = vectors[start:end]
        distances = [_cosine_distance(v[i], v[i + 1]) for i in range(len(v) - 1)]
        pieces[text] = pack_sentences(sentences, distances)
    return pieces


def build_chunks(
    activity_id: str, sources: List[Dict[str, Any]], pieces: Optional[Dict[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    """Turn source documents into chunk records.

    Each source is ``{"source_type", "source_id", "title"}`` plus either
    ``"text"`` or ``"sections"`` (see :func:`source_sections`). Chunks are
    numbered across the whole source, so ids stay unique. A section found in
    *pieces* (from :func:`semantic_pieces`) uses those chunks; any other is
    split by :func:`chunk_text`.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for src in sources:
        source_type = str(src.get("source_type") or "")
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source_type: {source_type!r}")
        source_id = str(src.get("source_id") or "")
        title = str(src.get("title") or "").strip()[:200]
        index = 0
        for section in prepared_sections(src):
            locator = section["locator"]
            split = (pieces or {}).get(section["text"]) or chunk_text(section["text"])
            for piece in split:
                key = " ".join(piece.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "id": chunk_id(activity_id, source_type, source_id, index),
                        "activity_id": activity_id,
                        "source_type": source_type,
                        "source_id": source_id,
                        "source_title": title,
                        "locator": locator,
                        "chunk_index": index,
                        "text": piece,
                        "content_hash": content_hash(title, locator, piece),
                    }
                )
                index += 1
                if len(out) >= MAX_CHUNKS_PER_ACTIVITY:
                    return out
    return out


def _batches(items: List[Any], size: int):
    for i in range(0, len(items), size):
        end = i + size
        yield items[i:end]


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


Sections = List[Dict[str, str]]

PLAIN_TEXT_TYPES = {".txt": "text/plain"}
MARKDOWN_TYPES = {".md": "text/markdown", ".markdown": "text/markdown"}
PPTX_TYPE = ".pptx"
CONVERTIBLE_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "application/xml",
}
SAVE_AS_HINTS = {
    ".ppt": "old PowerPoint format — save it as .pptx or PDF",
    ".odp": "OpenDocument slides are not supported — save as .pptx or PDF",
    ".doc": "old Word format — save it as .docx or PDF",
}
MAX_EXTRACTED_CHARS = 200_000
MAX_PPTX_XML_BYTES = 20 * 1024 * 1024
MAX_SLIDES = 500

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE_HEADING_RE = re.compile(r"^page\s+(\d+)$", re.I)
_METADATA_LINE_RE = re.compile(r"^-\s*[A-Za-z][\w ]*=")


def _extension(key: str) -> str:
    name = key.rsplit("/", 1)[-1].lower()
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def _cap_sections(sections: Sections, limit: Optional[int] = None) -> Sections:
    """Drop empty sections and truncate the total text to *limit* characters."""
    out: Sections = []
    remaining = MAX_EXTRACTED_CHARS if limit is None else limit
    for sec in sections:
        text = (sec.get("text") or "").strip()
        if not text or remaining <= 0:
            continue
        out.append({"locator": sec.get("locator") or "", "text": text[:remaining]})
        remaining -= len(text)
    return out


def split_markdown_sections(markdown: str) -> Sections:
    """Split Markdown at headings; each section is located by its heading.

    A "Page N" heading (how Workers AI toMarkdown marks PDF pages) gives the
    locator "page N", and headings nested under it keep the page number:
    "page 3, section: Mitosis". Sections that are only a heading are dropped.
    """
    sections: Sections = []
    locator, lines = "", []
    page = None

    def flush():
        body = "\n".join(line for line in lines if not _HEADING_RE.match(line.strip()))
        if body.strip():
            sections.append({"locator": locator, "text": "\n".join(lines).strip()})

    for line in (markdown or "").replace("\r\n", "\n").split("\n"):
        m = _HEADING_RE.match(line.strip())
        if not m:
            lines.append(line)
            continue
        flush()
        level, title = len(m.group(1)), m.group(2)[:80]
        page_match = _PAGE_HEADING_RE.match(title)
        if page_match:
            page = (level, page_match.group(1))
            locator = f"page {page[1]}"
        elif page and level > page[0]:
            locator = f"page {page[1]}, section: {title}"
        else:
            page = None
            locator = f"section: {title}"
        lines = [line]
    flush()
    return sections


def _is_metadata_section(section: Dict[str, str]) -> bool:
    """toMarkdown prepends a "## Metadata" list of file properties
    (producer, creation date, PDF version...) — not course content."""
    if section["locator"] != "section: Metadata":
        return False
    body = [line.strip() for line in section["text"].split("\n")[1:] if line.strip()]
    return all(_METADATA_LINE_RE.match(line) for line in body)


_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_NOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"


def _rels(zf: zipfile.ZipFile, part: str) -> Dict[str, Tuple[str, str]]:
    """Relationship id -> (type, absolute target) for an OOXML part."""
    rels_path = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
    if rels_path not in zf.namelist():
        return {}
    root = ElementTree.fromstring(zf.read(rels_path))
    out = {}
    for rel in root.findall("rel:Relationship", _NS):
        target = posixpath.normpath(posixpath.join(posixpath.dirname(part), rel.get("Target", "")))
        out[rel.get("Id")] = (rel.get("Type", ""), target)
    return out


_SKIP_PLACEHOLDERS = {"sldNum", "dt", "ftr", "hdr", "sldImg"}


def _paragraph_text(xml_bytes: bytes, notes: bool = False) -> str:
    """Text of a slide or notes part, one line per paragraph.

    Boilerplate placeholders are skipped. On notes pages only the notes body
    placeholder is read. Tables and grouped shapes are included.
    """
    root = ElementTree.fromstring(xml_bytes)
    sp_tag, p_tag, t_tag = f"{{{_NS['p']}}}sp", f"{{{_NS['a']}}}p", f"{{{_NS['a']}}}t"
    lines: List[str] = []

    def walk(el, skip: bool):
        if el.tag == sp_tag:
            ph = el.find("p:nvSpPr/p:nvPr/p:ph", _NS)
            ph_type = (ph.get("type") or "body") if ph is not None else None
            if ph_type in _SKIP_PLACEHOLDERS or (notes and ph_type != "body"):
                skip = True
        if el.tag == p_tag:
            if not skip:
                text = "".join(t.text or "" for t in el.iter(t_tag)).strip()
                if text:
                    lines.append(text)
            return
        for child in el:
            walk(child, skip)

    walk(root, False)
    return "\n".join(lines)


def _slide_number(name: str) -> int:
    m = re.search(r"(\d+)\.xml$", name)
    return int(m.group(1)) if m else 0


def extract_pptx_sections(data: bytes) -> Sections:
    """One section per slide ("slide N", in presentation order) incl. speaker notes."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        xml_size = sum(i.file_size for i in zf.infolist() if i.filename.endswith(".xml"))
        if xml_size > MAX_PPTX_XML_BYTES:
            raise ValueError("presentation is too large to read")

        slides: List[str] = []
        if "ppt/presentation.xml" in names:
            rels = _rels(zf, "ppt/presentation.xml")
            pres = ElementTree.fromstring(zf.read("ppt/presentation.xml"))
            for sld in pres.iterfind("p:sldIdLst/p:sldId", _NS):
                rel = rels.get(sld.get(f"{{{_NS['r']}}}id"))
                if rel and rel[1] in names:
                    slides.append(rel[1])
        if not slides:
            slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=_slide_number)

        sections: Sections = []
        for number, slide in enumerate(slides[:MAX_SLIDES], start=1):
            parts = [_paragraph_text(zf.read(slide))]
            for rel_type, target in _rels(zf, slide).values():
                if rel_type == _NOTES_REL and target in names:
                    notes = _paragraph_text(zf.read(target), notes=True)
                    if notes:
                        parts.append("Speaker notes:\n" + notes)
            sections.append({"locator": f"slide {number}", "text": "\n\n".join(p for p in parts if p)})
        return sections


async def extract_file_sections(env, bucket, key: str) -> Tuple[Sections, Optional[str]]:
    """Read an R2 object and return ``(sections, skip_reason)``.

    On success *skip_reason* is None. Otherwise *sections* is empty and
    *skip_reason* says why, so the teacher can see which files the tutor
    cannot read.
    """
    key = (key or "").lstrip("/")
    ext = _extension(key)
    if ext in SAVE_AS_HINTS:
        return [], SAVE_AS_HINTS[ext]
    known = (PLAIN_TEXT_TYPES, MARKDOWN_TYPES, CONVERTIBLE_TYPES)
    if ext != PPTX_TYPE and not any(ext in t for t in known):
        return [], f"unsupported file type '{ext or 'none'}'"
    if bucket is None:
        return [], "file storage is not configured"

    obj = await bucket.get(key)
    if not obj:
        return [], "file not found in storage"
    buf = await obj.arrayBuffer()

    if ext in PLAIN_TEXT_TYPES:
        sections = [{"locator": "", "text": bytes(js.Uint8Array.new(buf)).decode("utf-8", "replace")}]
    elif ext in MARKDOWN_TYPES:
        sections = split_markdown_sections(bytes(js.Uint8Array.new(buf)).decode("utf-8", "replace"))
    elif ext == PPTX_TYPE:
        try:
            sections = extract_pptx_sections(bytes(js.Uint8Array.new(buf)))
        except (zipfile.BadZipFile, ElementTree.ParseError, KeyError):
            return [], "could not read presentation (is it a valid .pptx?)"
        except ValueError as exc:
            return [], str(exc)
    else:
        ai, _ = _bindings(env)
        blob = js.Blob.new(to_js([buf]), _js({"type": CONVERTIBLE_TYPES[ext]}))
        result = _py(await ai.toMarkdown(_js([{"name": key.rsplit("/", 1)[-1], "blob": blob}])))
        doc = (result[0] if isinstance(result, list) and result else result) or {}
        if doc.get("format") == "error":
            detail = str(doc.get("error") or doc.get("data") or "unknown error")[:200]
            return [], f"could not convert file: {detail}"
        sections = [sec for sec in split_markdown_sections(str(doc.get("data") or "")) if not _is_metadata_section(sec)]

    sections = _cap_sections(sections)
    if not sections:
        return [], "file contains no extractable text"
    return sections, None


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


async def index_activity(env, activity_id: str, sources: List[Dict[str, Any]], encrypt: Crypt) -> Dict[str, int]:
    """Bring the activity's index in line with *sources*.

    Only chunks whose content hash changed are re-embedded; chunks whose
    source disappeared are removed from both Vectorize and D1.
    """
    _, index = _bindings(env)
    pieces = None
    if SEMANTIC_CHUNKING:
        try:
            pieces = await semantic_pieces(env, sources)
        except Exception as exc:
            print(json.dumps({"level": "warn", "where": "tutor_rag.semantic_pieces", "error": str(exc)[:200]}))
    desired = build_chunks(activity_id, sources, pieces)
    desired_ids = {c["id"] for c in desired}

    existing_res = (
        await env.DB.prepare("SELECT id, content_hash FROM tutor_chunks WHERE activity_id=?").bind(activity_id).all()
    )
    existing = {row.id: row.content_hash for row in (existing_res.results or [])}

    changed = [c for c in desired if existing.get(c["id"]) != c["content_hash"]]
    stale = [cid for cid in existing if cid not in desired_ids]

    if changed:
        vectors = await embed_texts(env, [embedding_input(c["source_title"], c["locator"], c["text"]) for c in changed])
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
                " (id, activity_id, source_type, source_id, source_title, locator, chunk_index,"
                "  text_encrypted, content_hash, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))"
                " ON CONFLICT(id) DO UPDATE SET"
                "  source_title=excluded.source_title,"
                "  locator=excluded.locator,"
                "  text_encrypted=excluded.text_encrypted,"
                "  content_hash=excluded.content_hash,"
                "  updated_at=excluded.updated_at"
            ).bind(
                c["id"],
                activity_id,
                c["source_type"],
                c["source_id"],
                c["source_title"],
                c["locator"],
                c["chunk_index"],
                await encrypt(c["text"]),
                c["content_hash"],
            ).run()

    for batch in _batches(stale, SQL_IN_BATCH_SIZE):
        await index.deleteByIds(_js(batch))
        placeholders = ",".join("?" for _ in batch)
        await env.DB.prepare(f"DELETE FROM tutor_chunks WHERE activity_id=? AND id IN ({placeholders})").bind(
            activity_id, *batch
        ).run()

    return {
        "chunks": len(desired),
        "embedded": len(changed),
        "unchanged": len(desired) - len(changed),
        "removed": len(stale),
    }


async def embed_queries(env, queries: List[str]) -> List[List[float]]:
    """Embed search queries. BGE v1.5 is trained to embed a short query with
    this instruction in front, and passages without it."""
    return await embed_texts(env, [QUERY_INSTRUCTION + q for q in queries])


async def rerank(env, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder *hits* by a cross-encoder's judgement of how well each answers *query*.

    Vector search compares two separately made embeddings; the reranker reads
    the question and the passage together, so it can tell "mentions the same
    topic" from "answers the question". Each hit gains ``rerank_score``.
    """
    if len(hits) < 2:
        return hits
    ai, _ = _bindings(env)
    contexts = [{"text": embedding_input(h["source_title"], h["locator"], h["text"])[:RERANK_MAX_CHARS]} for h in hits]
    result = _py(await ai.run(RERANK_MODEL, _js({"query": query[:MAX_QUERY_CHARS], "contexts": contexts})))
    scores = {}
    for item in (result or {}).get("response") or []:
        scores[int(item["id"])] = float(item["score"])
    if set(scores) != set(range(len(hits))):
        raise RuntimeError(f"reranker scored {len(scores)} of {len(hits)} passages")
    for i, hit in enumerate(hits):
        hit["rerank_score"] = round(scores[i], 4)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)


_KEYWORD_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _keyword_re(keyword: str):
    words = _KEYWORD_WORD_RE.findall(keyword.lower())
    if not words:
        return None
    return re.compile(r"\b" + r"[\s\-']+".join(re.escape(w) for w in words) + r"(?:s|es)?\b", re.IGNORECASE)


def keyword_boost(hits: List[Dict[str, Any]], keywords: List[str], weight: float) -> List[Dict[str, Any]]:
    """Reorder *hits*, adding ``weight * coverage`` to each one's score.

    coverage is the share of *keywords* found in the hit's text (not its
    title, which every chunk of a page shares), each keyword weighted by how
    rare it is among *hits* (idf): a keyword every candidate contains adds
    nothing. The score is the reranker's when there is one, else the vector
    similarity. Each hit gains ``keyword_coverage``.
    """
    patterns = [p for p in (_keyword_re(k) for k in keywords or []) if p]
    if not patterns or not hits or weight <= 0:
        return hits
    found = [[bool(p.search(h.get("text") or "")) for p in patterns] for h in hits]
    n = len(hits)
    idf = [math.log((n + 1) / (1 + sum(row[i] for row in found))) for i in range(len(patterns))]
    total = sum(idf)
    for hit, row in zip(hits, found):
        hit["keyword_coverage"] = round(sum(w for w, f in zip(idf, row) if f) / total, 4) if total else 0.0
    base = lambda h: h["rerank_score"] if "rerank_score" in h else h.get("score", 0.0)  # noqa: E731
    return sorted(hits, key=lambda h: base(h) + weight * h["keyword_coverage"], reverse=True)


def _clean_queries(queries) -> List[str]:
    queries = [queries] if isinstance(queries, str) else list(queries or [])
    return list(dict.fromkeys(q for q in ((q or "").strip()[:MAX_QUERY_CHARS] for q in queries) if q))


async def retrieve(
    env,
    activity_id: str,
    query: Union[str, List[str]],
    decrypt: Crypt,
    top_k: int = 5,
    min_score: float = 0.0,
    rerank_query: Optional[str] = None,
    extra_queries: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    keyword_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return the activity's chunks most relevant to *query*, best first.

    *query* may be a list of alternative queries, each searched as deeply
    as a question. *extra_queries* (rewrites
    of the question) each add their best *top_k* chunks as candidates. All
    are embedded in one call; each chunk keeps its best score across them,
    and only chunks scoring at least *min_score* for some query are kept.

    With *rerank_query*, each of *query* contributes up to RERANK_CANDIDATES
    candidates, and the reranker orders all candidates against that text;
    the best *top_k* are kept. If the reranker fails, the vector order is
    used. Rewrites only ever add candidates: answer-like rewrites score
    higher than questions do, and when they shared one candidate list they
    crowded out the question's own passages (CRAG: 44 vs 47 of 58).

    With *keywords* and a *keyword_weight* (and a *rerank_query*, so the
    whole candidate pool is loaded), candidates containing the keywords
    move up before the best *top_k* are kept (see :func:`keyword_boost`).

    Each hit carries what a caller needs to cite it: source type/id/title,
    locator (e.g. "slide 7"), chunk index, text, and similarity score.
    """
    _, index = _bindings(env)
    queries = _clean_queries(query)
    if not queries:
        return []
    extras = [q for q in _clean_queries(extra_queries) if q not in queries]
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    pool = max(top_k, RERANK_CANDIDATES) if rerank_query else top_k
    depth = [pool] * len(queries) + [top_k] * len(extras)

    scores: Dict[str, float] = {}
    for vector, k in zip(await embed_queries(env, queries + extras), depth):
        result = _py(
            await index.query(
                _js(vector),
                _js({"topK": k, "namespace": activity_id, "returnMetadata": "none"}),
            )
        )
        for m in (result or {}).get("matches") or []:
            score = float(m.get("score") or 0.0)
            if score >= min_score and score > scores.get(m["id"], -1.0):
                scores[m["id"]] = score
    if not scores:
        return []

    ids = sorted(scores, key=scores.get, reverse=True)
    if not rerank_query:
        ids = ids[:top_k]
    rows = []
    for batch in _batches(ids, SQL_IN_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        rows_res = (
            await env.DB.prepare(
                "SELECT id, source_type, source_id, source_title, locator, chunk_index, text_encrypted"
                f" FROM tutor_chunks WHERE activity_id=? AND id IN ({placeholders})"
            )
            .bind(activity_id, *batch)
            .all()
        )
        rows.extend(rows_res.results or [])

    hits = []
    for row in rows:
        hits.append(
            {
                "chunk_id": row.id,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "source_title": row.source_title,
                "locator": row.locator or "",
                "chunk_index": row.chunk_index,
                "text": await decrypt(row.text_encrypted or ""),
                "score": round(scores.get(row.id, 0.0), 4),
            }
        )
    hits.sort(key=lambda h: h["score"], reverse=True)
    if rerank_query and len(hits) > 1:
        try:
            hits = await rerank(env, rerank_query, hits)
        except Exception as exc:
            print(json.dumps({"level": "warn", "where": "tutor_rag.rerank", "error": str(exc)[:200]}))
    if rerank_query and keywords and keyword_weight > 0:
        hits = keyword_boost(hits, keywords, keyword_weight)
    return hits[:top_k]
