"""
Tests for the AI tutor retrieval layer (src/tutor_rag.py) and its endpoints.

Fakes (SQLite-backed D1, deterministic embedder, in-memory Vectorize and R2)
live in tests/tutor_fakes.py, so these tests exercise real SQL and ranking
rather than scripted mock call sequences.
"""

import io
import json
import zipfile
from types import SimpleNamespace

import pytest
import tutor_rag
from tests.helpers import MockRequest, load_worker
from tests.tutor_fakes import FakeAI, FakeR2, FakeVectorize, SqliteD1, _identity


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

    def test_slides_are_never_joined_and_carry_locator(self):
        src = [
            {
                "source_type": "material",
                "source_id": "m",
                "title": "Deck",
                "sections": [
                    {"locator": "slide 1", "text": "Alpha."},
                    {"locator": "slide 2", "text": "Beta."},
                ],
            }
        ]
        chunks = tutor_rag.build_chunks("a", src)
        assert [(c["locator"], c["text"], c["chunk_index"]) for c in chunks] == [
            ("slide 1", "Alpha.", 0),
            ("slide 2", "Beta.", 1),
        ]

    def test_short_section_joins_its_neighbour(self):
        long_text = "Mitosis makes two identical cells. " * 12
        src = [
            {
                "source_type": "material",
                "source_id": "m",
                "title": "Notes",
                "sections": [
                    {"locator": "section: Mitosis", "text": "## Mitosis\nCell division."},
                    {"locator": "section: Phases", "text": long_text},
                    {"locator": "section: Meiosis", "text": long_text.replace("Mitosis", "Meiosis")},
                ],
            }
        ]
        chunks = tutor_rag.build_chunks("a", src)
        assert [c["locator"] for c in chunks] == ["section: Mitosis – section: Phases", "section: Meiosis"]
        assert chunks[0]["text"].startswith("## Mitosis Cell division. Mitosis makes")

    def test_sections_too_big_to_share_a_chunk_stay_apart(self):
        big = "Word " * 239
        src = [
            {
                "source_type": "material",
                "source_id": "m",
                "title": "Notes",
                "sections": [{"locator": "section: A", "text": "Short."}, {"locator": "section: B", "text": big}],
            }
        ]
        assert [c["locator"] for c in tutor_rag.build_chunks("a", src)] == ["section: A", "section: B"]

    def test_repeated_text_is_indexed_once(self):
        footer = "Home | Courses | Contact us | Privacy policy"
        long_text = "Photosynthesis turns light into sugar. " * 20
        src = [
            {
                "source_type": "material",
                "source_id": f"m{i}",
                "title": f"Page {i}",
                "sections": [
                    {"locator": "section: Body", "text": long_text.replace("sugar", f"sugar {i}")},
                    {"locator": "section: Footer", "text": footer * 10},
                ],
            }
            for i in (1, 2)
        ]
        texts = [c["text"] for c in tutor_rag.build_chunks("a", src)]
        assert sum(footer in t for t in texts) == 1
        assert len(texts) == len(set(texts)) == 3

    def test_moving_text_to_another_slide_changes_hash(self):
        def chunk(locator):
            return tutor_rag.build_chunks(
                "a",
                [
                    {
                        "source_type": "material",
                        "source_id": "m",
                        "title": "D",
                        "sections": [{"locator": locator, "text": "Same."}],
                    }
                ],
            )[0]

        assert chunk("slide 1")["content_hash"] != chunk("slide 2")["content_hash"]

    def test_empty_sources_produce_no_chunks(self):
        assert tutor_rag.build_chunks("a", [{"source_type": "session", "source_id": "1", "text": ""}]) == []


_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class TestSemanticSplitting:
    def test_sentences_follow_paragraphs_and_long_ones_are_cut(self):
        text = "One. Two!\n\nThree? " + "x" * 30
        assert tutor_rag.sentences_of(text, max_chars=12) == ["One.", "Two!", "Three?", "x" * 12, "x" * 12, "x" * 6]

    def test_chunk_ends_at_the_biggest_change_of_topic(self):
        sentences = [f"s{i} " + "w" * 96 for i in range(10)]
        distances = [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
        chunks = tutor_rag.pack_sentences(sentences, distances, min_chars=200, max_chars=700)
        assert chunks[0] == " ".join(sentences[:5])
        assert " ".join(chunks) == " ".join(sentences)

    def test_no_cut_before_the_minimum_and_none_past_the_maximum(self):
        sentences = [f"s{i} " + "w" * 96 for i in range(10)]
        distances = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        chunks = tutor_rag.pack_sentences(sentences, distances, min_chars=300, max_chars=450)
        assert all(len(c) <= 450 for c in chunks)
        assert len(chunks[0]) >= 300
        assert " ".join(chunks) == " ".join(sentences)

    def test_one_sentence_is_one_chunk(self):
        assert tutor_rag.pack_sentences(["Only one."], []) == ["Only one."]


class TestKeywordBoost:
    @staticmethod
    def _hits(*texts, scores=None):
        scores = scores or [0.9 - i * 0.05 for i in range(len(texts))]
        return [
            {"chunk_id": str(i), "source_title": "Title", "text": t, "rerank_score": sc}
            for i, (t, sc) in enumerate(zip(texts, scores))
        ]

    def test_passage_with_a_rare_keyword_moves_up(self):
        hits = self._hits("The 76ers history.", "Allen Iverson career.", "Allen Iverson won MVP; the 76ers went 56-26.")
        out = tutor_rag.keyword_boost(hits, ["MVP", "Allen Iverson"], 0.25)
        assert out[0]["chunk_id"] == "2" and out[0]["keyword_coverage"] > 0

    def test_keyword_every_passage_has_changes_nothing(self):
        hits = self._hits("Mitosis one.", "Mitosis two.", "Mitosis three.")
        out = tutor_rag.keyword_boost(hits, ["mitosis"], 0.5)
        assert [h["chunk_id"] for h in out] == ["0", "1", "2"]

    def test_title_does_not_count(self):
        hits = self._hits("Plain text.", "More text.")
        hits[1]["source_title"] = "Mitosis"
        out = tutor_rag.keyword_boost(hits, ["mitosis"], 0.5)
        assert [h["chunk_id"] for h in out] == ["0", "1"]

    def test_plurals_and_hyphens_match(self):
        hits = self._hits("Nothing here.", "Two sodium potassium pumps per cell.")
        out = tutor_rag.keyword_boost(hits, ["sodium-potassium pump"], 0.25)
        assert out[0]["chunk_id"] == "1"

    def test_no_keywords_or_no_weight_keeps_the_order(self):
        hits = self._hits("A text.", "B text.")
        assert tutor_rag.keyword_boost(hits, [], 0.5) == hits
        assert tutor_rag.keyword_boost(hits, ["B"], 0) == hits


def _shape(paragraphs, ph_type=None):
    ph = f'<p:ph type="{ph_type}"/>' if ph_type else ""
    paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paragraphs)
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="1" name="s"/><p:cNvSpPr/><p:nvPr>{ph}</p:nvPr></p:nvSpPr>'
        f"<p:txBody>{paras}</p:txBody></p:sp>"
    )


def _text_part(root_tag, shapes):
    return (
        f'<{root_tag} xmlns:a="{_A}" xmlns:p="{_P}"><p:cSld><p:spTree>'
        f"{''.join(shapes)}</p:spTree></p:cSld></{root_tag}>"
    )


def make_pptx(slides, order=None, with_presentation=True):
    """Build a minimal .pptx. *slides* is a list of (paragraphs, notes or None),
    stored as slide1.xml, slide2.xml, ...; *order* lists file numbers in
    presentation order (defaults to file order)."""
    buf = io.BytesIO()
    order = order or list(range(1, len(slides) + 1))
    with zipfile.ZipFile(buf, "w") as zf:
        if with_presentation:
            ids = "".join(f'<p:sldId id="{255 + n}" r:id="rId{n}"/>' for n in order)
            zf.writestr(
                "ppt/presentation.xml",
                f'<p:presentation xmlns:p="{_P}" xmlns:r="{_R}"><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>',
            )
            rels = "".join(
                f'<Relationship Id="rId{n}" Type="{_R}/slide" Target="slides/slide{n}.xml"/>'
                for n in range(1, len(slides) + 1)
            )
            zf.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{_REL}">{rels}</Relationships>')
        for n, (paragraphs, notes) in enumerate(slides, start=1):
            zf.writestr(
                f"ppt/slides/slide{n}.xml",
                _text_part("p:sld", [_shape(paragraphs), _shape(["Confidential"], "ftr"), _shape([str(n)], "sldNum")]),
            )
            if notes is not None:
                zf.writestr(
                    f"ppt/slides/_rels/slide{n}.xml.rels",
                    f'<Relationships xmlns="{_REL}"><Relationship Id="rId9" '
                    f'Type="{_R}/notesSlide" Target="../notesSlides/notesSlide{n}.xml"/></Relationships>',
                )
                zf.writestr(
                    f"ppt/notesSlides/notesSlide{n}.xml",
                    _text_part(
                        "p:notes", [_shape([], "sldImg"), _shape(notes, "body"), _shape(["&lt;number&gt;"], "sldNum")]
                    ),
                )
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
            "page 1",
            "page 2",
            "page 2, section: Mitosis",
            "section: Appendix",
        ]

    def test_no_headings_is_one_section(self):
        assert tutor_rag.split_markdown_sections("just text") == [{"locator": "", "text": "just text"}]


class TestPptx:
    def test_one_section_per_slide_with_notes(self):
        data = make_pptx(
            [(["Cell biology", "Week 1"], ["Remind students about the lab."]), (["Mitochondria", "Produce ATP"], None)]
        )
        assert tutor_rag.extract_pptx_sections(data) == [
            {"locator": "slide 1", "text": "Cell biology\nWeek 1\n\nSpeaker notes:\nRemind students about the lab."},
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
        assert env.AI.calls == []

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

    async def test_large_file_is_read(self):
        bucket = FakeR2({"big.txt": b"word " * (11 * 1024 * 1024 // 5)})
        sections, reason = await tutor_rag.extract_file_sections(self._env(), bucket, "big.txt")
        assert reason is None
        assert len(sections[0]["text"]) == tutor_rag.MAX_EXTRACTED_CHARS

    async def test_no_bucket(self):
        sections, reason = await tutor_rag.extract_file_sections(self._env(), None, "a.txt")
        assert sections == [] and "not configured" in reason

    async def test_total_text_is_capped(self, monkeypatch):
        monkeypatch.setattr(tutor_rag, "MAX_EXTRACTED_CHARS", 25)
        data = make_pptx([(["a" * 20], None), (["b" * 20], None), (["c" * 20], None)])
        sections, _ = await tutor_rag.extract_file_sections(self._env(), FakeR2({"d.pptx": data}), "d.pptx")
        assert [len(s["text"]) for s in sections] == [20, 5]


@pytest.fixture
def worker():
    return load_worker()


@pytest.fixture
async def env(worker):
    env = SimpleNamespace(
        DB=SqliteD1(),
        AI=FakeAI(),
        VECTORIZE=FakeVectorize(),
        MY_BUCKET=FakeR2(),
        ENCRYPTION_KEY="test-encryption-key",
        JWT_SECRET="test-jwt-secret",
    )
    await worker.init_db(env)
    return env


BIO = [
    {
        "source_type": "activity",
        "source_id": "bio",
        "title": "Biology 101",
        "text": "An introduction to living things and how cells work.",
    },
    {
        "source_type": "session",
        "source_id": "s1",
        "title": "Photosynthesis",
        "text": "Plants use chlorophyll to capture sunlight and turn carbon dioxide and water into glucose.",
    },
    {
        "source_type": "session",
        "source_id": "s2",
        "title": "Mitosis",
        "text": "During mitosis a cell divides its chromosomes into two identical daughter nuclei.",
    },
]


class TestCleanText:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("See [the notes](https://x.org/notes) first.", "See the notes first."),
            ("![Mitosis diagram](data:image/png;base64,AAAA) Phases", "Mitosis diagram Phases"),
            ("[![logo](a.png)](https://x.org)", "logo"),
            ("[Cell (biology)](https://en.wikipedia.org/wiki/Cell_(biology))", "Cell (biology)"),
            ('[Home](/ "Go home") | [About](/about)', "Home | About"),
            ("Submit at [https://forms.x.org/lab](https://forms.x.org/lab)", "Submit at https://forms.x.org/lab"),
            ("Arrays use a[0] and (1, 2).", "Arrays use a[0] and (1, 2)."),
        ],
    )
    def test_clean(self, text, expected):
        assert tutor_rag.clean_text(text) == expected

    def test_link_urls_are_not_indexed(self):
        src = [
            {
                "source_type": "material",
                "source_id": "m",
                "title": "Page",
                "text": "Read [chapter 2](https://example.org/very/long/path?id=1) on enzymes.",
            }
        ]
        [chunk] = tutor_rag.build_chunks("a", src)
        assert chunk["text"] == "Read chapter 2 on enzymes."


PLANTS = [
    "Plants capture sunlight with chlorophyll in their green leaves every day.",
    "Chlorophyll in leaves absorbs sunlight so plants can make glucose sugar.",
    "Leaves take carbon dioxide and water and plants turn them into glucose.",
    "Green plants store the glucose from sunlight as starch in their leaves.",
    "Sunlight and chlorophyll let leaves of plants make glucose and oxygen gas.",
    "Plants release oxygen from their leaves while chlorophyll uses sunlight.",
    "Without sunlight the chlorophyll in plant leaves cannot make any glucose.",
    "Plants in shade grow larger leaves to catch more sunlight for chlorophyll.",
]
CELLS = [
    "During mitosis the nucleus divides and chromosomes separate into two sets.",
    "Chromosomes line up in the middle before mitosis pulls the copies apart.",
    "Each daughter nucleus after mitosis has identical chromosomes and genes.",
    "Mitosis lets a dividing nucleus pass chromosomes to both daughter nuclei.",
    "Spindle fibres pull chromosomes to opposite poles of the dividing nucleus.",
    "After mitosis the cell membrane pinches so two nuclei become two daughters.",
    "Chromosomes condense at the start of mitosis so the nucleus can divide them.",
    "Errors in mitosis leave a daughter nucleus with the wrong chromosome count.",
]
WATER = [
    "Rain falls from clouds and fills rivers that flow down to the sea.",
    "The sun warms the sea so water evaporates and rises as vapour into clouds.",
    "Water vapour cools high in the sky and condenses into droplets of cloud.",
    "Rivers carry rain water from hills and mountains back down to the ocean.",
    "Snow on mountains melts in spring and the water runs into lakes and rivers.",
    "Clouds drift over land and drop their water as rain, snow or hail on hills.",
    "Some rain soaks into the ground and flows slowly underground to the rivers.",
    "Lakes and oceans lose water to evaporation on hot and windy summer days.",
]
TOPICS = PLANTS + CELLS + WATER
LONG_SECTION = [{"source_type": "material", "source_id": "m1", "title": "Notes", "text": " ".join(TOPICS)}]


class TestSemanticChunking:
    async def test_long_section_is_split_where_the_topic_changes(self, env, monkeypatch):
        topic_words = [
            {"plants", "leaves", "chlorophyll", "sunlight", "glucose"},
            {"mitosis", "chromosomes", "nucleus", "nuclei", "daughter"},
            {"rain", "rivers", "water", "clouds", "sea", "ocean"},
        ]

        async def by_topic(env, texts):
            words = [set(t.lower().replace(".", " ").replace(",", " ").split()) for t in texts]
            return [[len(w & tw) + 0.01 for tw in topic_words] for w in words]

        monkeypatch.setattr(tutor_rag, "embed_texts", by_topic)
        stats = await tutor_rag.index_activity(env, "bio", LONG_SECTION, _identity)
        rows = (await env.DB.prepare("SELECT text_encrypted FROM tutor_chunks ORDER BY chunk_index").all()).results
        texts = [r.text_encrypted for r in rows]
        assert stats["chunks"] == len(texts) >= 2
        assert all(len(t) <= tutor_rag.CHUNK_MAX_CHARS for t in texts)
        assert texts[0].startswith(" ".join(PLANTS))
        assert texts[0] in (" ".join(PLANTS), " ".join(PLANTS + CELLS[:1]))
        assert " ".join(texts) == " ".join(TOPICS)

    async def test_sentences_are_embedded_with_their_neighbours(self, env):
        await tutor_rag.index_activity(env, "bio", LONG_SECTION, _identity)
        windows = env.AI.calls[0][1]
        assert windows[0] == " ".join(PLANTS[:2]) and windows[1] == " ".join(PLANTS[:3])
        assert len(windows) == len(TOPICS)

    async def test_short_sections_cost_no_extra_embedding(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        assert len(env.AI.calls) == 1

    async def test_reindex_finds_the_same_chunks(self, env):
        await tutor_rag.index_activity(env, "bio", LONG_SECTION, _identity)
        env.AI.calls.clear()
        stats = await tutor_rag.index_activity(env, "bio", LONG_SECTION, _identity)
        assert stats["embedded"] == 0
        assert len(env.AI.calls) == 1

    async def test_failed_sentence_embedding_falls_back_to_plain_splitting(self, env, monkeypatch):
        async def broken(env, sources):
            raise RuntimeError("AI down")

        monkeypatch.setattr(tutor_rag, "semantic_pieces", broken)
        await tutor_rag.index_activity(env, "bio", LONG_SECTION, _identity)
        rows = (await env.DB.prepare("SELECT text_encrypted FROM tutor_chunks ORDER BY chunk_index").all()).results
        assert [r.text_encrypted for r in rows] == tutor_rag.chunk_text(LONG_SECTION[0]["text"])

    async def test_off_switch(self, env, monkeypatch):
        monkeypatch.setattr(tutor_rag, "SEMANTIC_CHUNKING", False)
        await tutor_rag.index_activity(env, "bio", LONG_SECTION, _identity)
        assert len(env.AI.calls) == 1


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
        await tutor_rag.index_activity(
            env,
            "chem",
            [
                {
                    "source_type": "activity",
                    "source_id": "chem",
                    "title": "Chemistry",
                    "text": "Chlorophyll molecules contain magnesium; sunlight excites electrons.",
                },
            ],
            _identity,
        )
        hits = await tutor_rag.retrieve(env, "bio", "chlorophyll sunlight magnesium", _identity, top_k=10)
        assert hits and all(h["source_id"] in {"bio", "s1", "s2"} for h in hits)

    async def test_reranker_reorders_more_candidates_than_it_returns(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        env.AI.rerank_responder = lambda q, texts: [0.9 if "mitosis" in t.lower() else 0.1 for t in texts]
        [hit] = await tutor_rag.retrieve(
            env, "bio", "chlorophyll sunlight", _identity, top_k=1, rerank_query="How do cells divide?"
        )
        assert hit["source_id"] == "s2" and hit["rerank_score"] == 0.9
        [(model, query, texts)] = env.AI.rerank_calls
        assert (model, query, len(texts)) == (tutor_rag.RERANK_MODEL, "How do cells divide?", 3)
        assert texts[0].startswith("Photosynthesis\n")

    async def test_reranker_failure_keeps_vector_order(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)

        def broken(q, texts):
            raise RuntimeError("reranker down")

        env.AI.rerank_responder = broken
        [hit] = await tutor_rag.retrieve(env, "bio", "chlorophyll sunlight", _identity, top_k=1, rerank_query="q")
        assert hit["source_id"] == "s1" and "rerank_score" not in hit

    async def test_reranker_only_sees_passages_above_the_floor(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        hits = await tutor_rag.retrieve(
            env, "bio", "chlorophyll sunlight carbon dioxide", _identity, top_k=5, min_score=0.5, rerank_query="q"
        )
        assert [h["source_id"] for h in hits] == ["s1"]
        assert env.AI.rerank_calls == []

    async def test_no_reranking_unless_asked(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        await tutor_rag.retrieve(env, "bio", "chlorophyll sunlight", _identity, top_k=2)
        assert env.AI.rerank_calls == []

    async def test_keywords_move_a_passage_above_the_rerankers_choice(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        env.AI.rerank_responder = lambda q, texts: [0.6 if "Biology" in t else 0.5 for t in texts]
        hits = await tutor_rag.retrieve(
            env, "bio", "cells living things chromosomes", _identity, top_k=3, rerank_query="q"
        )
        assert hits[0]["source_id"] == "bio"
        hits = await tutor_rag.retrieve(
            env,
            "bio",
            "cells living things chromosomes",
            _identity,
            top_k=3,
            rerank_query="q",
            keywords=["chromosomes"],
            keyword_weight=0.25,
        )
        assert hits[0]["source_id"] == "s2" and hits[0]["keyword_coverage"] == 1.0

    async def test_keywords_need_the_reranked_pool(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        plain = await tutor_rag.retrieve(env, "bio", "cells living things", _identity, top_k=2)
        boosted = await tutor_rag.retrieve(
            env, "bio", "cells living things", _identity, top_k=2, keywords=["chromosomes"], keyword_weight=0.5
        )
        assert [h["chunk_id"] for h in boosted] == [h["chunk_id"] for h in plain]

    async def test_retrieve_several_queries_keeps_each_chunks_best_score(self, env):
        await tutor_rag.index_activity(env, "bio", BIO, _identity)
        env.AI.calls.clear()
        [plants] = await tutor_rag.retrieve(env, "bio", "chlorophyll sunlight", _identity, top_k=1)
        [cells] = await tutor_rag.retrieve(env, "bio", "mitosis daughter cells", _identity, top_k=1)
        env.AI.calls.clear()
        both = await tutor_rag.retrieve(
            env, "bio", ["chlorophyll sunlight", "mitosis daughter cells"], _identity, top_k=3
        )
        q = tutor_rag.QUERY_INSTRUCTION
        assert env.AI.calls == [(tutor_rag.EMBED_MODEL, [q + "chlorophyll sunlight", q + "mitosis daughter cells"])]
        by_id = {h["chunk_id"]: h["score"] for h in both}
        assert by_id[plants["chunk_id"]] == plants["score"]
        assert by_id[cells["chunk_id"]] == cells["score"]
        assert len(by_id) == len(both) <= 3
        assert both == sorted(both, key=lambda h: h["score"], reverse=True)

    async def test_retrieve_empty_query(self, env):
        assert await tutor_rag.retrieve(env, "bio", "   ", _identity) == []

    async def test_missing_bindings_raise(self, env):
        env.VECTORIZE = None
        assert not tutor_rag.is_configured(env)
        with pytest.raises(tutor_rag.TutorNotConfigured):
            await tutor_rag.retrieve(env, "bio", "q", _identity)


async def _seed(worker, env):
    enc = env.ENCRYPTION_KEY
    for uid, role in (("host1", "host"), ("student1", "member"), ("outsider", "member")):
        await env.DB.prepare(
            "INSERT INTO users (id, username_hash, email_hash, name, username, email, password_hash, role)"
            " VALUES (?,?,?,?,?,?,?,?)"
        ).bind(uid, uid, uid + "@x", uid, uid, uid, "x", role).run()
    await env.DB.prepare(
        "INSERT INTO activities (id, title, description, learning_objectives, host_id, slug)" " VALUES (?,?,?,?,?,?)"
    ).bind(
        "act1",
        "Biology 101",
        await worker.encrypt_aes("Cells and how they work.", enc),
        await worker.encrypt_aes("Explain the role of ribosomes in protein synthesis.", enc),
        "host1",
        "biology-101",
    ).run()
    await env.DB.prepare(
        "INSERT INTO sessions (id, activity_id, title, description, start_time) VALUES (?,?,?,?,?)"
    ).bind(
        "s1",
        "act1",
        "Photosynthesis",
        await worker.encrypt_aes("Plants capture sunlight with chlorophyll to make glucose.", enc),
        "2026-01-01",
    ).run()
    await env.DB.prepare("INSERT INTO activity_materials (id, activity_id, title, description) VALUES (?,?,?,?)").bind(
        "m1", "act1", "Mitosis handout", "Mitosis splits one nucleus into two identical nuclei."
    ).run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)" " VALUES (?,?,?,?,?)"
    ).bind("m2", "act1", "Week 3 slides", "materials/week3.pdf", 1).run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)" " VALUES (?,?,?,?,?)"
    ).bind("m3", "act1", "Cell diagram", "materials/cell.png", 2).run()
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, external_url, display_order)" " VALUES (?,?,?,?,?)"
    ).bind("m4", "act1", "Lecture video", "https://example.com/video", 3).run()
    env.MY_BUCKET.files["materials/week3.pdf"] = b"The mitochondria produces ATP through cellular respiration."
    env.MY_BUCKET.files["materials/cell.png"] = b"\x89PNG"
    await env.DB.prepare(
        "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)" " VALUES (?,?,?,?,?)"
    ).bind("m5", "act1", "Genetics deck", "materials/genetics.pptx", 4).run()
    env.MY_BUCKET.files["materials/genetics.pptx"] = make_pptx(
        [
            (["Genetics", "Course overview"], None),
            (
                ["Alleles", "Dominant alleles mask recessive alleles in heterozygotes."],
                ["Draw a Punnett square on the board."],
            ),
        ]
    )
    await env.DB.prepare("INSERT INTO enrollments (id, activity_id, user_id, status) VALUES (?,?,?,?)").bind(
        "e1", "act1", "student1", "active"
    ).run()


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
        assert {k: data[k] for k in ("chunks", "embedded", "unchanged", "removed")} == {
            "chunks": 6,
            "embedded": 6,
            "unchanged": 0,
            "removed": 0,
        }
        assert sorted((f["material_id"], f["reason"]) for f in data["skipped_files"]) == [
            ("m3", "unsupported file type '.png'"),
            ("m4", "external links are not fetched"),
        ]

    async def _search(self, worker, env, q):
        await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", HOST), env)
        resp = await worker._dispatch(
            _req(worker, env, "GET", f"/api/activities/act1/tutor/search?q={q}&k=1", STUDENT), env
        )
        assert resp.status == 200, resp.body
        return json.loads(resp.body)["data"]["results"]

    async def test_file_content_is_searchable(self, worker, env):
        await _seed(worker, env)
        [hit] = await self._search(worker, env, "mitochondria+ATP+respiration")
        assert (hit["source_type"], hit["source_id"], hit["source_title"]) == ("material", "m2", "Week 3 slides")
        assert "cellular respiration" in hit["text"]
        assert hit["locator"] == "page 1"

    async def test_search_cites_the_slide(self, worker, env):
        await _seed(worker, env)
        [hit] = await self._search(worker, env, "dominant+recessive+alleles+heterozygotes")
        assert (hit["source_id"], hit["source_title"], hit["locator"]) == ("m5", "Genetics deck", "slide 2")
        assert "Punnett square" in hit["text"]

    async def test_learning_objectives_are_searchable(self, worker, env):
        await _seed(worker, env)
        [hit] = await self._search(worker, env, "ribosomes+protein+synthesis")
        assert (hit["source_type"], hit["locator"]) == ("activity", "description – learning objectives")
        assert "ribosomes" in hit["text"]

    async def test_student_cannot_reindex(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", STUDENT), env)
        assert resp.status == 403

    async def test_enrolled_student_gets_cited_passages(self, worker, env):
        await _seed(worker, env)
        await worker._dispatch(_req(worker, env, "POST", "/api/activities/act1/tutor/reindex", HOST), env)
        resp = await worker._dispatch(
            _req(worker, env, "GET", "/api/activities/biology-101/tutor/search?q=chlorophyll+sunlight&k=2", STUDENT),
            env,
        )
        assert resp.status == 200, resp.body
        results = json.loads(resp.body)["data"]["results"]
        assert results[0]["source_type"] == "session"
        assert results[0]["source_id"] == "s1"
        assert "chlorophyll" in results[0]["text"]

    async def test_unenrolled_user_is_refused(self, worker, env):
        await _seed(worker, env)
        resp = await worker._dispatch(
            _req(worker, env, "GET", "/api/activities/act1/tutor/search?q=cells", OUTSIDER), env
        )
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
