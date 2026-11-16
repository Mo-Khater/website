"""
Tests for the AI tutor retrieval layer (src/tutor_rag.py) and its endpoints.

D1 is faked with a real in-memory SQLite database running the worker's own
DDL, Workers AI with a deterministic bag-of-words embedder, and Vectorize
with an in-memory, namespace-aware cosine index — so these tests exercise the
real SQL and real ranking behaviour rather than scripted mock call sequences.
"""

import hashlib
import json
import math
import re
import sqlite3
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

    def test_empty_sources_produce_no_chunks(self):
        assert tutor_rag.build_chunks("a", [{"source_type": "session", "source_id": "1", "text": ""}]) == []


# ---------------------------------------------------------------------------
# Indexing and retrieval against the fakes
# ---------------------------------------------------------------------------

@pytest.fixture
def worker():
    return load_worker()


@pytest.fixture
async def env(worker):
    env = SimpleNamespace(
        DB=SqliteD1(), AI=FakeAI(), VECTORIZE=FakeVectorize(),
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
        "INSERT INTO activities (id, title, description, host_id, slug) VALUES (?,?,?,?,?)"
    ).bind("act1", "Biology 101", await worker.encrypt_aes("Cells and how they work.", enc),
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
        assert json.loads(resp.body)["data"] == {"chunks": 3, "embedded": 3, "unchanged": 0, "removed": 0}

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
