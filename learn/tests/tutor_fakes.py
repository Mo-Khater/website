"""
Shared fakes for the AI tutor tests.

D1 is faked with a real in-memory SQLite database (run the worker's own DDL
on it), Workers AI with a deterministic bag-of-words embedder and reranker
plus scriptable chat, query-rewrite and keyword models, Vectorize with an in-memory namespace-aware cosine
index, and R2 with a dict — so tests exercise real SQL and real ranking.
"""

import hashlib
import math
import re
import sqlite3
from types import SimpleNamespace


class _Stmt:
    def __init__(self, conn, sql, args=()):
        self.conn, self.sql, self.args = conn, sql, args

    def bind(self, *args):
        if any(a is None for a in args):
            raise TypeError(f"D1_TYPE_ERROR: undefined bind value in {args!r}")
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

    async def batch(self, stmts):
        with self.conn:
            for st in stmts:
                self.conn.execute(st.sql, st.args)


_VOCAB = 256


def _embed(text):
    import tutor_rag
    if text.startswith(tutor_rag.QUERY_INSTRUCTION):
        text = text[len(tutor_rag.QUERY_INSTRUCTION):]
    vec = [0.0] * _VOCAB
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) > 2:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % _VOCAB] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeAI:
    def __init__(self):
        self.calls = []
        self.chat_calls = []
        self.expansion_calls = []
        self.keyword_calls = []
        self.rerank_calls = []
        self.chat_responder = lambda messages: "From the course materials: see [1]."
        self.expansion_responder = lambda messages: ""
        self.keyword_responder = lambda messages: ""
        self.rerank_responder = None

    async def run(self, model, payload):
        import tutor_chat
        if "messages" in payload and payload["messages"][0]["content"] == tutor_chat.KEYWORD_PROMPT:
            self.keyword_calls.append((model, payload))
            return {"response": self.keyword_responder(payload["messages"])}
        if "messages" in payload and model == tutor_chat.EXPANSION_MODEL:
            self.expansion_calls.append((model, payload))
            return {"response": self.expansion_responder(payload["messages"])}
        if "messages" in payload:
            self.chat_calls.append((model, payload))
            return {"response": self.chat_responder(payload["messages"])}
        if "contexts" in payload:
            texts = [c["text"] for c in payload["contexts"]]
            self.rerank_calls.append((model, payload["query"], texts))
            if self.rerank_responder:
                scores = self.rerank_responder(payload["query"], texts)
            else:
                q = _embed(payload["query"])
                scores = [sum(a * b for a, b in zip(q, _embed(t))) for t in texts]
            return {"response": [{"id": i, "score": sc} for i, sc in enumerate(scores)]}
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
        self.vectors = {}

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
