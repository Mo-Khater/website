"""
Tests for the AI tutor retrieval layer (src/tutor_rag.py) and its endpoints.

D1 is faked with a real in-memory SQLite database running the worker's own
DDL, Workers AI with a deterministic bag-of-words embedder, and Vectorize
with an in-memory, namespace-aware cosine index — so these tests exercise the
real SQL and real ranking behaviour rather than scripted mock call sequences.
"""

import hashlib
import io
import json
import math
import re
import sqlite3
import zipfile
from types import SimpleNamespace

import pytest

import tutor_rag
from tests.helpers import MockRequest, load_worker


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Stmt:
    def __init__(self, conn, sql, args=()):
        self.conn, self.sql, self.args = conn, sql, args

    def bind(self, *args):
        return _Stmt(self.conn, self.sql, args)

    def _rows(self):
        cur = self.conn.execute(self.sql, self.args)
        cols = [c[0] for c in cur.description or []]
        return [SimpleNamespace(**dict(zip(cols, r))) for r in cur.fetchall()]

    async def first(self):
        rows = self._rows()
        return rows[0] if rows else None

    async def all(self):
        return SimpleNamespace(results=self._rows())

    async def run(self):
        self.conn.execute(self.sql, self.args)
        self.conn.commit()
        return SimpleNamespace(success=True)


class SqliteD1:
    """Just enough of the D1 API, backed by SQLite."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")

    def prepare(self, sql):
        return _Stmt(self.conn, sql)


_VOCAB = 256


def _embed(text):
    vec = [0.0] * _VOCAB
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) > 2:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % _VOCAB] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeAI:
    def __init__(self):
        self.calls = []

    async def run(self, model, payload):
        self.calls.append((model, list(payload["text"])))
        return {"shape": [len(payload["text"]), _VOCAB],
                "data": [_embed(t) for t in payload["text"]]}

    async def toMarkdown(self, docs):
        """Pretend conversion: a file whose bytes start with BROKEN fails."""
        out = []
        for d in docs:
            raw = b"".join(d["blob"].parts).decode()
            if raw.startswith("BROKEN"):
                out.append({"name": d["name"], "format": "error", "error": "corrupt file"})
            else:
                # Shape observed from the real service for a PDF: file-name heading,
                # a Metadata list, an empty Contents heading, then one heading per page.
                pages = raw.split("\f")
                body = "\n".join(f"### Page {n}\n{text}" for n, text in enumerate(pages, start=1))
                md = (f"# {d['name']}\n## Metadata\n- PDFFormatVersion=1.6\n- Producer=LibreOffice 7.3\n"
                      f"## Contents\n{body}")
                out.append({"name": d["name"], "mimetype": d["blob"].type,
                            "format": "markdown", "tokens": 1, "data": md})
        self.calls.append(("toMarkdown", [d["name"] for d in docs]))
        return out


class FakeR2Object:
    def __init__(self, data: bytes):
        self._data = data
        self.size = len(data)

    async def arrayBuffer(self):
        return self._data


class FakeR2:
    def __init__(self, files=None):
        self.files = dict(files or {})

    async def get(self, key):
        data = self.files.get(key)
        return FakeR2Object(data) if data is not None else None


class FakeVectorize:
    def __init__(self):
        self.vectors = {}   # (namespace, id) -> values

    async def upsert(self, records):
        for r in records:
            self.vectors[(r["namespace"], r["id"])] = r["values"]
        return {"mutationId": "m"}

    async def deleteByIds(self, ids):
        for key in [k for k in self.vectors if k[1] in ids]:
            del self.vectors[key]
        return {"mutationId": "m"}

    async def query(self, vector, opts):
        ns = opts.get("namespace")
        scored = [
            {"id": vid, "score": sum(a * b for a, b in zip(vector, vals))}
            for (vns, vid), vals in self.vectors.items() if vns == ns
        ]
        scored.sort(key=lambda m: m["score"], reverse=True)
        return {"count": len(scored), "matches": scored[: opts["topK"]]}


async def _identity(text):
    return text


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty(self):
        assert tutor_rag.chunk_text("") == []
        assert tutor_rag.chunk_text("   \n\n  ") == []

    def test_short_text_is_one_chunk(self):
        assert tutor_rag.chunk_text("Photosynthesis makes sugar.") == ["Photosynthesis makes sugar."]

    def test_chunks_respect_max_chars(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
        chunks = tutor_rag.chunk_text(text, max_chars=400, overlap_chars=80)
        assert len(chunks) > 1
        assert all(len(c) <= 400 for c in chunks)

    def test_long_sentence_is_hard_split(self):
        chunks = tutor_rag.chunk_text("x" * 2500, max_chars=1000, overlap_chars=0)
        assert [len(c) for c in chunks] == [1000, 1000, 500]

    def test_consecutive_chunks_overlap(self):
        sentences = [f"Sentence number {i} is here." for i in range(40)]
        chunks = tutor_rag.chunk_text(" ".join(sentences), max_chars=200, overlap_chars=60)
        for a, b in zip(chunks, chunks[1:]):
            assert a.split(". ")[-1].rstrip(".") in b

    def test_no_text_lost(self):
        sentences = [f"Fact {i} matters." for i in range(100)]
        joined = " ".join(tutor_rag.chunk_text(" ".join(sentences), max_chars=150))
        for s in sentences:
            assert s in joined


class TestBuildChunks:
    def test_ids_are_stable_and_unique(self):
        sources = [
            {"source_type": "session", "source_id": "s1", "title": "Week 1", "text": "a. " * 800},
            {"source_type": "session", "source_id": "s2", "title": "Week 2", "text": "b. " * 10},
        ]
        first = tutor_rag.build_chunks("act1", sources)
        second = tutor_rag.build_chunks("act1", sources)
        assert [c["id"] for c in first] == [c["id"] for c in second]
        assert len({c["id"] for c in first}) == len(first)
        assert all(len(c["id"]) <= 64 for c in first)

    def test_same_source_in_other_activity_gets_other_id(self):
        src = [{"source_type": "activity", "source_id": "x", "title": "T", "text": "hello"}]
        assert tutor_rag.build_chunks("a", src)[0]["id"] != tutor_rag.build_chunks("b", src)[0]["id"]

    def test_unknown_source_type_rejected(self):
        with pytest.raises(ValueError):
            tutor_rag.build_chunks("a", [{"source_type": "web", "source_id": "1", "text": "x"}])

    def test_chunks_never_span_sections_and_carry_locator(self):
        src = [{"source_type": "material", "source_id": "m", "title": "Deck", "sections": [
            {"locator": "slide 1", "text": "Alpha."},
            {"locator": "slide 2", "text": "Beta."},
        ]}]
        chunks = tutor_rag.build_chunks("a", src)
        assert [(c["locator"], c["text"], c["chunk_index"]) for c in chunks] == \
            [("slide 1", "Alpha.", 0), ("slide 2", "Beta.", 1)]

    def test_moving_text_to_another_slide_changes_hash(self):
        def chunk(locator):
            return tutor_rag.build_chunks("a", [{"source_type": "material", "source_id": "m", "title": "D",
                                                 "sections": [{"locator": locator, "text": "Same."}]}])[0]
        assert chunk("slide 1")["content_hash"] != chunk("slide 2")["content_hash"]

    def test_empty_sources_produce_no_chunks(self):
        assert tutor_rag.build_chunks("a", [{"source_type": "session", "source_id": "1", "text": ""}]) == []


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _shape(paragraphs, ph_type=None):
    ph = f'<p:ph type="{ph_type}"/>' if ph_type else ""
    paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paragraphs)
    return (f"<p:sp><p:nvSpPr><p:cNvPr id=\"1\" name=\"s\"/><p:cNvSpPr/><p:nvPr>{ph}</p:nvPr></p:nvSpPr>"
            f"<p:txBody>{paras}</p:txBody></p:sp>")


def _text_part(root_tag, shapes):
    return (f'<{root_tag} xmlns:a="{_A}" xmlns:p="{_P}"><p:cSld><p:spTree>'
            f"{''.join(shapes)}</p:spTree></p:cSld></{root_tag}>")


def make_pptx(slides, order=None, with_presentation=True):
    """Build a minimal .pptx. *slides* is a list of (paragraphs, notes or None),
    stored as slide1.xml, slide2.xml, ...; *order* lists file numbers in
    presentation order (defaults to file order)."""
    buf = io.BytesIO()
    order = order or list(range(1, len(slides) + 1))
    with zipfile.ZipFile(buf, "w") as zf:
        if with_presentation:
            ids = "".join(f'<p:sldId id="{255 + n}" r:id="rId{n}"/>' for n in order)
            zf.writestr("ppt/presentation.xml",
                        f'<p:presentation xmlns:p="{_P}" xmlns:r="{_R}"><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>')
            rels = "".join(f'<Relationship Id="rId{n}" Type="{_R}/slide" Target="slides/slide{n}.xml"/>'
                           for n in range(1, len(slides) + 1))
            zf.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{_REL}">{rels}</Relationships>')
        for n, (paragraphs, notes) in enumerate(slides, start=1):
            # like real decks: content + a footer and slide-number placeholder
            zf.writestr(f"ppt/slides/slide{n}.xml", _text_part("p:sld", [
                _shape(paragraphs), _shape(["Confidential"], "ftr"), _shape([str(n)], "sldNum")]))
            if notes is not None:
                zf.writestr(f"ppt/slides/_rels/slide{n}.xml.rels",
                            f'<Relationships xmlns="{_REL}"><Relationship Id="rId9" '
                            f'Type="{_R}/notesSlide" Target="../notesSlides/notesSlide{n}.xml"/></Relationships>')
                # like LibreOffice: notes body + slide image + "<number>" placeholder
                zf.writestr(f"ppt/notesSlides/notesSlide{n}.xml", _text_part("p:notes", [
                    _shape([], "sldImg"), _shape(notes, "body"), _shape(["&lt;number&gt;"], "sldNum")]))
    return buf.getvalue()


class TestMarkdownSections:
    def test_split_at_headings(self):
        md = "Intro line.\n\n# Cells\nCells are small.\n\n## Mitosis ##\nCells divide."
        assert tutor_rag.split_markdown_sections(md) == [
            {"locator": "", "text": "Intro line."},
            {"locator": "section: Cells", "text": "# Cells\nCells are small."},
            {"locator": "section: Mitosis", "text": "## Mitosis ##\nCells divide."},
        ]

    def test_heading_only_sections_dropped(self):
        md = "# file.pdf\n## Contents\n## Real\nBody."
        assert tutor_rag.split_markdown_sections(md) == [{"locator": "section: Real", "text": "## Real\nBody."}]

    def test_page_headings_become_page_locators(self):
        md = "### Page 1\nAlpha.\n### Page 2\nBeta.\n#### Mitosis\nGamma.\n## Appendix\nDelta."
        assert [s["locator"] for s in tutor_rag.split_markdown_sections(md)] == [
            "page 1", "page 2", "page 2, section: Mitosis", "section: Appendix"]

    def test_no_headings_is_one_section(self):
        assert tutor_rag.split_markdown_sections("just text") == [{"locator": "", "text": "just text"}]


class TestPptx:
    def test_one_section_per_slide_with_notes(self):
        data = make_pptx([(["Cell biology", "Week 1"], ["Remind students about the lab."]),
                          (["Mitochondria", "Produce ATP"], None)])
        assert tutor_rag.extract_pptx_sections(data) == [
            {"locator": "slide 1",
             "text": "Cell biology\nWeek 1\n\nSpeaker notes:\nRemind students about the lab."},
            {"locator": "slide 2", "text": "Mitochondria\nProduce ATP"},
        ]

    def test_follows_presentation_order_not_file_names(self):
        data = make_pptx([(["File one"], None), (["File two"], None)], order=[2, 1])
        sections = tutor_rag.extract_pptx_sections(data)
        assert [(s["locator"], s["text"]) for s in sections] == [("slide 1", "File two"), ("slide 2", "File one")]

    def test_falls_back_to_file_order(self):
        data = make_pptx([(["A"], None), (["B"], None)], with_presentation=False)
        assert [s["text"] for s in tutor_rag.extract_pptx_sections(data)] == ["A", "B"]

    def test_zip_bomb_guard(self, monkeypatch):
        monkeypatch.setattr(tutor_rag, "MAX_PPTX_XML_BYTES", 100)
        with pytest.raises(ValueError):
            tutor_rag.extract_pptx_sections(make_pptx([(["x" * 200], None)]))


class TestExtractFileSections:
    def _env(self):
        return SimpleNamespace(AI=FakeAI(), VECTORIZE=FakeVectorize())

    async def test_plain_text_read_directly(self):
        env = self._env()
        bucket = FakeR2({"notes/week1.txt": b"Osmosis moves water across membranes."})
        sections, reason = await tutor_rag.extract_file_sections(env, bucket, "/notes/week1.txt")
        assert reason is None
        assert sections == [{"locator": "", "text": "Osmosis moves water across membranes."}]
        assert env.AI.calls == []   # no AI call for plain text

    async def test_markdown_located_by_heading(self):
        bucket = FakeR2({"n.md": b"# Osmosis\nWater moves."})
        sections, _ = await tutor_rag.extract_file_sections(self._env(), bucket, "n.md")
        assert sections == [{"locator": "section: Osmosis", "text": "# Osmosis\nWater moves."}]

    async def test_pptx_parsed_locally(self):
        env = self._env()
        bucket = FakeR2({"deck.pptx": make_pptx([(["Krebs cycle"], None)])})
        sections, reason = await tutor_rag.extract_file_sections(env, bucket, "deck.pptx")
        assert reason is None
        assert sections == [{"locator": "slide 1", "text": "Krebs cycle"}]
        assert env.AI.calls == []

    async def test_corrupt_pptx(self):
        bucket = FakeR2({"deck.pptx": b"not a zip"})
        sections, reason = await tutor_rag.extract_file_sections(self._env(), bucket, "deck.pptx")
        assert sections == [] and "valid .pptx" in reason

    async def test_old_formats_get_save_as_hint(self):
        for name, hint in (("old.ppt", ".pptx or PDF"), ("old.doc", ".docx or PDF")):
            sections, reason = await tutor_rag.extract_file_sections(self._env(), FakeR2(), name)
            assert sections == [] and hint in reason

    async def test_pdf_pages_located_and_metadata_dropped(self):
        env = self._env()
        bucket = FakeR2({"slides.pdf": b"Krebs cycle overview\fElectron transport chain"})
        sections, reason = await tutor_rag.extract_file_sections(env, bucket, "slides.pdf")
        assert reason is None
        assert sections == [
            {"locator": "page 1", "text": "### Page 1\nKrebs cycle overview"},
            {"locator": "page 2", "text": "### Page 2\nElectron transport chain"},
        ]
        assert env.AI.calls == [("toMarkdown", ["slides.pdf"])]

    async def test_conversion_error_is_reported(self):
        bucket = FakeR2({"bad.pdf": b"BROKEN"})
        sections, reason = await tutor_rag.extract_file_sections(self._env(), bucket, "bad.pdf")
        assert sections == [] and reason == "could not convert file: corrupt file"

    async def test_images_are_skipped_without_reading(self):
        env = self._env()
        bucket = FakeR2({"diagram.png": b"\x89PNG"})
        sections, reason = await tutor_rag.extract_file_sections(env, bucket, "diagram.png")
        assert sections == [] and "unsupported" in reason
        assert env.AI.calls == []

    async def test_missing_file(self):
        sections, reason = await tutor_rag.extract_file_sections(self._env(), FakeR2(), "gone.txt")
        assert sections == [] and "not found" in reason

    async def test_oversized_file(self, monkeypatch):
        monkeypatch.setattr(tutor_rag, "MAX_FILE_BYTES", 10)
        bucket = FakeR2({"big.txt": b"x" * 11})
        sections, reason = await tutor_rag.extract_file_sections(self._env(), bucket, "big.txt")
        assert sections == [] and "larger than" in reason

    async def test_no_bucket(self):
        sections, reason = await tutor_rag.extract_file_sections(self._env(), None, "a.txt")
        assert sections == [] and "not configured" in reason

    async def test_total_text_is_capped(self, monkeypatch):
        monkeypatch.setattr(tutor_rag, "MAX_EXTRACTED_CHARS", 25)
        data = make_pptx([(["a" * 20], None), (["b" * 20], None), (["c" * 20], None)])
        sections, _ = await tutor_rag.extract_file_sections(self._env(), FakeR2({"d.pptx": data}), "d.pptx")
        assert [len(s["text"]) for s in sections] == [20, 5]


# ---------------------------------------------------------------------------
# Indexing and retrieval against the fakes
# ---------------------------------------------------------------------------

@pytest.fixture
def worker():
    return load_worker()


@pytest.fixture
async def env(worker):
    env = SimpleNamespace(
        DB=SqliteD1(), AI=FakeAI(), VECTORIZE=FakeVectorize(), MY_BUCKET=FakeR2(),
        ENCRYPTION_KEY="test-encryption-key", JWT_SECRET="test-jwt-secret",
    )
    await worker.init_db(env)
    return env


BIO = [
    {"source_type": "activity", "source_id": "bio", "title": "Biology 101",
     "text": "An introduction to living things and how cells work."},
    {"source_type": "session", "source_id": "s1", "title": "Photosynthesis",
     "text": "Plants use chlorophyll to capture sunlight and turn carbon dioxide and water into glucose."},
    {"source_type": "session", "source_id": "s2", "title": "Mitosis",
     "text": "During mitosis a cell divides its chromosomes into two identical daughter nuclei."},
]


class TestIndexAndRetrieve:
    async def test_first_index_embeds_everything(self, env):
        stats = await tutor_rag.index_activity(env, "bio", BIO, _identity)
        assert stats == {"chunks": 3, "embedded": 3, "unchanged": 0, "removed": 0}
        assert len(env.VECTORIZE.vectors) == 3

    async def test_reindex_skips_unchanged_chunks(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        env.AI.calls.clear()
        stats = await tutor_rag.index_activity(env, "bio", BIO, _identity)
        assert stats == {"chunks": 3, "embedded": 0, "unchanged": 3, "removed": 0}
        assert env.AI.calls == []

    async def test_reindex_reembeds_only_edited_chunk(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        env.AI.calls.clear()
        edited = [dict(s) for s in BIO]
        edited[2]["text"] = "Meiosis halves the chromosome number to make gametes."
        stats = await tutor_rag.index_activity(env, "bio", edited, _identity)
        assert stats["embedded"] == 1 and stats["unchanged"] == 2
        assert len(env.AI.calls) == 1 and "Meiosis" in env.AI.calls[0][1][0]

    async def test_removed_source_is_deleted_everywhere(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        stats = await tutor_rag.index_activity(env, "bio", BIO[:2], _identity)
        assert stats["removed"] == 1
        assert len(env.VECTORIZE.vectors) == 2
        rows = await env.DB.prepare("SELECT COUNT(*) AS n FROM tutor_chunks").first()
        assert rows.n == 2

    async def test_chunk_text_is_stored_encrypted(self, env):
        async def fake_encrypt(text):
            return "ENC:" + text
        await tutor_rag.index_activity(env, "bio", BIO, fake_encrypt)
        res = await env.DB.prepare("SELECT text_encrypted FROM tutor_chunks").all()
        assert all(r.text_encrypted.startswith("ENC:") for r in res.results)

    async def test_vector_metadata_holds_no_course_text(self, env):
        captured = []
        real_upsert = env.VECTORIZE.upsert

        async def spy(records):
            captured.extend(records)
            return await real_upsert(records)

        env.VECTORIZE.upsert = spy
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        for r in captured:
            assert set(r["metadata"]) == {"source_type", "source_id"}
            assert r["namespace"] == "bio"

    async def test_retrieve_ranks_relevant_source_first(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        hits = await tutor_rag.retrieve(env, "bio", "how do plants use sunlight and chlorophyll?", _identity, top_k=3)
        assert hits[0]["source_id"] == "s1"
        assert hits[0]["source_title"] == "Photosynthesis"
        assert "chlorophyll" in hits[0]["text"]
        assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)

    async def test_retrieve_never_crosses_activities(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        await tutor_rag.index_activity(env, "chem", [
            {"source_type": "activity", "source_id": "chem", "title": "Chemistry",
             "text": "Chlorophyll molecules contain magnesium; sunlight excites electrons."},
        ], _identity)
        hits = await tutor_rag.retrieve(env, "bio", "chlorophyll sunlight magnesium", _identity, top_k=10)
        assert hits and all(h["source_id"] in {"bio", "s1", "s2"} for h in hits)

    async def test_retrieve_empty_query(self, env):
        assert await tutor_rag.retrieve(env, "bio", "   ", _identity) == []

    async def test_missing_bindings_raise(self, env):
        env.VECTORIZE = None
        assert not tutor_rag.is_configured(env)
        with pytest.raises(tutor_rag.TutorNotConfigured):
            await tutor_rag.retrieve(env, "bio", "q", _identity)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

async def _seed(worker, env):
    enc = env.ENCRYPTION_KEY
    for uid, role in (("host1", "host"), ("student1", "member"), ("outsider", "member")):
        await env.DB.prepare(
            "INSERT INTO users (id, username_hash, email_hash, name, username, email, password_hash, role)"
            " VALUES (?,?,?,?,?,?,?,?)"
        ).bind(uid, uid, uid + "@x", uid, uid, uid, "x", role).run()
    await env.DB.prepare(
        "INSERT INTO activities (id, title, description, learning_objectives, host_id, slug)"
        " VALUES (?,?,?,?,?,?)"
    ).bind("act1", "Biology 101", await worker.encrypt_aes("Cells and how they work.", enc),
           await worker.encrypt_aes("Explain the role of ribosomes in protein synthesis.", enc),
           "host1", "biology-101").run()
    await env.DB.prepare(
        "INSERT INTO sessions (id, activity_id, title, description, start_time) VALUES (?,?,?,?,?)"
    ).bind("s1", "act1", "Photosynthesis",
           await worker.encrypt_aes("Plants capture sunlight with chlorophyll to make glucose.", enc),
           "2026-01-01").run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, description) VALUES (?,?,?,?)"
    ).bind("m1", "act1", "Mitosis handout", "Mitosis splits one nucleus into two identical nuclei.").run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)"
        " VALUES (?,?,?,?,?)"
    ).bind("m2", "act1", "Week 3 slides", "materials/week3.pdf", 1).run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)"
        " VALUES (?,?,?,?,?)"
    ).bind("m3", "act1", "Cell diagram", "materials/cell.png", 2).run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, external_url, display_order)"
        " VALUES (?,?,?,?,?)"
    ).bind("m4", "act1", "Lecture video", "https://example.com/video", 3).run()
    env.MY_BUCKET.files["materials/week3.pdf"] = (
        b"The mitochondria produces ATP through cellular respiration."
    )
    env.MY_BUCKET.files["materials/cell.png"] = b"\x89PNG"
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)"
        " VALUES (?,?,?,?,?)"
    ).bind("m5", "act1", "Genetics deck", "materials/genetics.pptx", 4).run()
    env.MY_BUCKET.files["materials/genetics.pptx"] = make_pptx([
        (["Genetics", "Course overview"], None),
        (["Alleles", "Dominant alleles mask recessive alleles in heterozygotes."],
         ["Draw a Punnett square on the board."]),
    ])
    await env.DB.prepare(
        "INSERT INTO enrollments (id, activity_id, user_id, status) VALUES (?,?,?,?)"
    ).bind("e1", "act1", "student1", "active").run()


def _auth(worker, env, uid, role):
    token = worker.create_token(uid, uid, role, env.JWT_SECRET)
    return {"Authorization": f"Bearer {token}"}


def _req(worker, env, method, path, who=None):
    headers = _auth(worker, env, *who) if who else {}
    return MockRequest(method=method, url=f"http://localhost{path}", headers=headers)


HOST = ("host1", "host")
STUDENT = ("student1", "member")
OUTSIDER = ("outsider", "member")


class TestTutorEndpoints:
    async def test_host_can_reindex(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", HOST), env)
        assert resp.status == 200, resp.body
        data = json.loads(resp.body)["data"]
        # activity description + objectives, session, handout, pdf, 2 slides
        # (the image and the link contribute no text)
        assert {k: data[k] for k in ("chunks", "embedded", "unchanged", "removed")} == \
            {"chunks": 7, "embedded": 7, "unchanged": 0, "removed": 0}
        assert sorted((f["material_id"], f["reason"]) for f in data["skipped_files"]) == [
            ("m3", "unsupported file type '.png'"),
            ("m4", "external links are not fetched"),
        ]

    async def _search(self, worker, env, q):
        await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", HOST), env)
        resp = await worker._dispatch(_req(
            worker, env, "GET", f"/api/activities/act1/tutor/search?q={q}&k=1", STUDENT), env)
        assert resp.status == 200, resp.body
        return json.loads(resp.body)["data"]["results"]

    async def test_file_content_is_searchable(self, worker, env):
        await _seed(worker, env)
        [hit] = await self._search(worker, env, "mitochondria+ATP+respiration")
        assert (hit["source_type"], hit["source_id"], hit["source_title"]) == \
            ("material", "m2", "Week 3 slides")
        assert "cellular respiration" in hit["text"]
        assert hit["locator"] == "page 1"

    async def test_search_cites_the_slide(self, worker, env):
        await _seed(worker, env)
        [hit] = await self._search(worker, env, "dominant+recessive+alleles+heterozygotes")
        assert (hit["source_id"], hit["source_title"], hit["locator"]) == ("m5", "Genetics deck", "slide 2")
        assert "Punnett square" in hit["text"]   # speaker notes come along

    async def test_learning_objectives_are_searchable(self, worker, env):
        await _seed(worker, env)
        [hit] = await self._search(worker, env, "ribosomes+protein+synthesis")
        assert (hit["source_type"], hit["locator"]) == ("activity", "learning objectives")
        assert "ribosomes" in hit["text"]

    async def test_student_cannot_reindex(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", STUDENT), env)
        assert resp.status == 403

    async def test_enrolled_student_gets_cited_passages(self, worker, env):
        await _seed(worker, env)
        await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", HOST), env)
        resp = await worker._dispatch(_req(
            worker, env, "GET", "/api/activities/biology-101/tutor/search?q=chlorophyll+sunlight&k=2", STUDENT), env)
        assert resp.status == 200, resp.body
        results = json.loads(resp.body)["data"]["results"]
        assert results[0]["source_type"] == "session"
        assert results[0]["source_id"] == "s1"
        assert "chlorophyll" in results[0]["text"]   # decrypted for the caller

    async def test_unenrolled_user_is_refused(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(_req(
            worker, env, "GET", "/api/activities/act1/tutor/search?q=cells", OUTSIDER), env)
        assert resp.status == 403

    async def test_anonymous_is_refused(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(_req(worker, env, "GET", "/api/activities/act1/tutor/search?q=cells"), env)
        assert resp.status == 401

    async def test_missing_query(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(_req(worker, env, "GET", "/api/activities/act1/tutor/search", STUDENT), env)
        assert resp.status == 400

    async def test_503_when_vectorize_not_bound(self, worker, env):
        await _seed(worker, env)
        env.VECTORIZE = None
        resp = await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", HOST), env)
        assert resp.status == 503
