#!/usr/bin/env python3
"""
Run a CRAG sample (Meta's Comprehensive RAG Benchmark, Task 1) through the
tutor. Everything runs on this machine except, by default, the chat model's
calls, which stay within the Workers AI free allowance.

The real worker runs in-process (as in the tests) and does everything it does
in production: create the course, read the materials, convert, chunk, embed,
index, retrieve, prompt, check citations. Only Cloudflare's services are
replaced with local ones:

  Workers AI toMarkdown   -> markdownify (HTML -> Markdown)
  Workers AI embeddings   -> the same model, BAAI/bge-base-en-v1.5, via fastembed (CPU)
  Workers AI reranker     -> the same model, BAAI/bge-reranker-base, via fastembed (CPU)
  Vectorize               -> exact in-memory cosine search (tests/tutor_fakes.FakeVectorize)
  D1 / R2                 -> in-memory SQLite / dict (tests/tutor_fakes)

The chat models (answers, query rewrites, and CRAG's judge) are either
  --backend workers-ai    the production models, Llama 3.3 70B and 3.1 8B on Workers AI, through
                          a local `wrangler dev` helper (sandbox crag-llm/). Only these
                          calls leave the machine. They are counted in neurons and the
                          run stops before --budget-neurons (default 8,000; the free
                          allowance is 10,000 a day).
  --backend ollama        a local model through Ollama (e.g. llama3.1:8b). Free, but
                          not the production model.

Each CRAG question becomes a course whose materials are its web pages (up to
5, full HTML). The student asks the question; the answer is scored the CRAG
way (+1 correct, 0 "I don't know", -1 wrong) with CRAG's own judge prompt,
judged by the same model. See crag_scoring.py and CRAG.md.

  python tutor_eval/run_crag.py --sample crag/sample.jsonl --limit 40 --backend workers-ai

Needs: fastembed, markdownify (pip install fastembed markdownify); Ollama for --backend ollama.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

LEARN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEARN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tests.conftest  # noqa: E402,F401
import crag_scoring  # noqa: E402
import tutor_chat  # noqa: E402
import tutor_rag  # noqa: E402
from tests.helpers import MockRequest, load_worker  # noqa: E402
from tests.tutor_fakes import FakeR2, FakeVectorize, SqliteD1  # noqa: E402

TEMPLATES_URL = "https://raw.githubusercontent.com/facebookresearch/CRAG/main/prompts/templates.py"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

NEURONS_PER_INPUT_TOKEN = 26_668 / 1e6
NEURONS_PER_OUTPUT_TOKEN = 204_805 / 1e6
NEURON_PRICES = {
    tutor_chat.CHAT_MODEL: (NEURONS_PER_INPUT_TOKEN, NEURONS_PER_OUTPUT_TOKEN),
    tutor_chat.EXPANSION_MODEL: (4_119 / 1e6, 34_868 / 1e6),
}


class BudgetExceeded(Exception):
    pass
HOST, STUDENT = "crag-host", "crag-student"


def html_to_markdown(html: str) -> str:
    """Stand-in for toMarkdown: drop non-content elements, keep headings/links/lists."""
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "head"]):
        tag.decompose()
    return MarkdownConverter(heading_style="ATX", strip=["img"]).convert_soup(soup)


class LocalAI:
    """The two Workers AI calls the tutor makes (run, toMarkdown), served locally."""

    def __init__(self, backend: str, url: str, chat_model: str, cache_path: Path,
                 budget_neurons: float = 0, num_ctx: int = 8192, expansion_model: str = "llama3.1:8b"):
        self.backend, self.url, self.chat_model, self.num_ctx = backend, url.rstrip("/"), chat_model, num_ctx
        self.expansion_model = expansion_model
        self._reranker = None
        self.budget = budget_neurons
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._embedder = None
        self.stats = {"embedded": 0, "embed_cached": 0, "embed_seconds": 0.0,
                      "chat_calls": 0, "chat_seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0,
                      "neurons": 0.0}

    def _embed(self, texts):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(EMBED_MODEL)
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if todo:
            started = time.time()
            for (k, _), vec in zip(todo, self._embedder.embed([t for _, t in todo], batch_size=32)):
                norm = float((vec ** 2).sum()) ** 0.5 or 1.0
                self.cache[k] = [round(float(x) / norm, 6) for x in vec]
            self.stats["embed_seconds"] += time.time() - started
        self.stats["embedded"] += len(todo)
        self.stats["embed_cached"] += len(texts) - len(todo)
        return [self.cache[k] for k in keys]

    def save_cache(self):
        self.cache_path.write_text(json.dumps(self.cache))

    def neurons_left(self) -> float:
        return self.budget - self.stats["neurons"] if self.backend == "workers-ai" else float("inf")

    def _rerank(self, query, texts):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._reranker = TextCrossEncoder(RERANK_MODEL)
        self.stats["rerank_calls"] = self.stats.get("rerank_calls", 0) + 1
        return [1 / (1 + math.exp(-float(x))) for x in self._reranker.rerank(query, texts)]

    def chat(self, messages, temperature=0.2, max_tokens=350, json_mode=False, model=None):
        model = model or tutor_chat.CHAT_MODEL
        if self.backend == "workers-ai":
            price_in, price_out = NEURON_PRICES[model]
            prompt_chars = sum(len(m.get("content") or "") for m in messages)
            worst = prompt_chars / 4 * price_in + max_tokens * price_out
            if worst > self.neurons_left():
                raise BudgetExceeded(f"{self.stats['neurons']:.0f} of {self.budget:.0f} neurons used")
            body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            endpoint = self.url
        else:
            local = self.expansion_model if model == tutor_chat.EXPANSION_MODEL else self.chat_model
            body = {"model": local, "messages": messages, "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx}}
            if json_mode:
                body["format"] = "json"
            endpoint = self.url + "/api/chat"
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise OSError(f"model call failed ({exc.code}): {detail}") from None
        self.stats["chat_calls"] += 1
        self.stats["chat_seconds"] += time.time() - started
        if self.backend == "workers-ai":
            usage = out.get("usage") or {}
            tokens_in = usage.get("prompt_tokens") or sum(len(m.get("content") or "") for m in messages) / 4
            tokens_out = usage.get("completion_tokens") or len(str(out.get("response") or "")) / 4
            text = out.get("response")
        else:
            tokens_in, tokens_out = out.get("prompt_eval_count") or 0, out.get("eval_count") or 0
            text = (out.get("message") or {}).get("content")
        self.stats["prompt_tokens"] += int(tokens_in)
        self.stats["output_tokens"] += int(tokens_out)
        if self.backend == "workers-ai":
            self.stats["neurons"] += float((out.get("usage") or {}).get("neurons") or
                                           tokens_in * price_in + tokens_out * price_out)
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        return text or ""

    async def run(self, model, payload):
        if "messages" in payload:
            return {"response": self.chat(payload["messages"], payload.get("temperature", 0.2),
                                          payload.get("max_tokens", 350), model=model)}
        if "contexts" in payload:
            scores = self._rerank(payload["query"], [c["text"] for c in payload["contexts"]])
            return {"response": [{"id": i, "score": sc} for i, sc in enumerate(scores)]}
        texts = list(payload["text"])
        return {"shape": [len(texts), 768], "data": self._embed(texts)}

    async def toMarkdown(self, docs):
        out = []
        for d in docs:
            raw = b"".join(bytes(p) for p in d["blob"].parts).decode("utf-8", "replace")
            try:
                out.append({"name": d["name"], "format": "markdown", "data": html_to_markdown(raw)})
            except Exception as exc:
                out.append({"name": d["name"], "format": "error", "error": str(exc)})
        return out


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_text(html: str) -> str:
    parser = _Text()
    try:
        parser.feed(html or "")
    except Exception:
        pass
    return " ".join(parser.parts)


def pages_of(example):
    return [p for p in example.get("search_results") or [] if (p.get("page_result") or "").strip()]


def load_judge_prompt(source: str):
    """INSTRUCTIONS and IN_CONTEXT_EXAMPLES from CRAG's prompts/templates.py
    (URL or path). Parsed as string literals, never executed."""
    if source.startswith("http"):
        with urllib.request.urlopen(source, timeout=60) as resp:
            code = resp.read().decode()
    else:
        code = Path(source).read_text()
    values = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("INSTRUCTIONS", "IN_CONTEXT_EXAMPLES"):
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"INSTRUCTIONS", "IN_CONTEXT_EXAMPLES"}:
        sys.exit(f"Could not find CRAG's judge prompt in {source}")
    return values["INSTRUCTIONS"], values["IN_CONTEXT_EXAMPLES"], hashlib.sha256(code.encode()).hexdigest()[:12]


class Site:
    def __init__(self, ai: LocalAI):
        self.worker = load_worker()
        self.env = SimpleNamespace(DB=SqliteD1(), AI=ai, VECTORIZE=FakeVectorize(), MY_BUCKET=FakeR2(),
                                   ENCRYPTION_KEY="local-crag-key", JWT_SECRET="local-crag-secret")

    async def setup(self):
        await self.worker.init_db(self.env)
        for uid in (HOST, STUDENT):
            await self.env.DB.prepare(
                "INSERT INTO users (id, username_hash, email_hash, name, username, email, password_hash, role)"
                " VALUES (?,?,?,?,?,?,?,?)").bind(uid, uid, uid, uid, uid, uid, "x", "member").run()

    async def call(self, method, path, uid, body=None):
        token = self.worker.create_token(uid, uid, "member", self.env.JWT_SECRET)
        req = MockRequest(method=method, url=f"http://localhost{path}",
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
                          body=json.dumps(body) if body is not None else None)
        resp = await self.worker._dispatch(req, self.env)
        data = json.loads(resp.body) if resp.body else {}
        if resp.status != 200:
            raise RuntimeError(f"{method} {path} -> {resp.status}: {data.get('error')}")
        return data.get("data")

    async def add_course(self, example) -> str:
        """A course whose materials are the question's web pages."""
        act = (await self.call("POST", "/api/activities", HOST,
                               {"title": f"CRAG {example['interaction_id'][:8]}"}))["id"]
        await self.call("POST", "/api/join", STUDENT, {"activity_id": act})
        await self.call("PUT", f"/api/activities/{act}/tutor/policy", HOST,
                        {"enabled": True, "mode": "answer", "refused_topics": [], "daily_limit": 0, "instructions": ""})
        for n, page in enumerate(pages_of(example), start=1):
            key = f"materials/{act}/page-{n}.html"
            self.env.MY_BUCKET.files[key] = page["page_result"].encode()
            title = (page.get("page_name") or page.get("page_url") or f"Page {n}").strip()[:150] or f"Page {n}"
            await self.env.DB.prepare(
                "INSERT INTO activity_materials (id, activity_id, title, file_r2_key, display_order)"
                " VALUES (?,?,?,?,?)").bind(f"{act}-p{n}", act, title, key, n).run()
        return act


NEURONS_PER_QUESTION_RESERVE = 500


async def run(args):
    examples = [json.loads(line) for line in Path(args.sample).read_text().splitlines() if line.strip()]
    examples = examples[args.offset: args.offset + args.limit]
    instructions, in_context, templates_sha = load_judge_prompt(args.templates)

    workers_ai = args.backend == "workers-ai"
    model_name = tutor_chat.CHAT_MODEL + " (Workers AI)" if workers_ai else f"{args.model} (Ollama, local)"
    ai = LocalAI(args.backend, args.llm_url if workers_ai else args.ollama_url, args.model,
                 Path(args.embed_cache), budget_neurons=args.budget_neurons)
    try:
        if workers_ai:
            with urllib.request.urlopen(args.llm_url, timeout=30) as resp:
                served = json.loads(resp.read()).get("model")
            if served != tutor_chat.CHAT_MODEL:
                sys.exit(f"The helper at {args.llm_url} serves {served}, not {tutor_chat.CHAT_MODEL}")
        else:
            ai.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=5)
    except OSError as exc:
        where = "the Workers AI helper (crag-llm, `wrangler dev`)" if workers_ai else "Ollama (`ollama serve`)"
        sys.exit(f"Can't reach {where} at {ai.url}: {exc}")
    site = Site(ai)
    await site.setup()

    print(f"{len(examples)} CRAG questions · answers and judge: {model_name} · embeddings {EMBED_MODEL} (local)")
    if workers_ai:
        print(f"Workers AI budget: {args.budget_neurons:,.0f} neurons (free allowance: 10,000 a day)")
    print("\nIndexing each question's pages (convert, chunk, embed — all local)…")
    for i, ex in enumerate(examples, start=1):
        started = time.time()
        ex["_activity"] = await site.add_course(ex)
        stats = await site.call("POST", f"/api/activities/{ex['_activity']}/tutor/reindex", HOST)
        ex["_index"] = stats
        skipped = f", couldn't read {len(stats['skipped_files'])} page(s)" if stats.get("skipped_files") else ""
        print(f"  [{i}/{len(examples)}] {len(pages_of(ex))} pages -> {stats['chunks']} passages"
              f"{skipped} ({time.time() - started:.0f}s)", flush=True)
        ai.save_cache()

    def judge(query, ground_truth, prediction):
        messages = crag_scoring.judge_messages(instructions, in_context, query, ground_truth, prediction)
        try:
            return crag_scoring.parse_judge_reply(ai.chat(messages, temperature=0, max_tokens=200, json_mode=True))
        except OSError as exc:
            return {"score": -1, "explanation": f"judge failed: {exc}"}

    print("\nAsking the tutor and judging each answer…")
    rows, stopped = [], None
    for i, ex in enumerate(examples, start=1):
        if ai.neurons_left() < NEURONS_PER_QUESTION_RESERVE:
            stopped = (f"stopped after {len(rows)} of {len(examples)} questions: "
                       f"{ai.stats['neurons']:,.0f} of {args.budget_neurons:,.0f} neurons used")
            break
        act, golds = ex["_activity"], crag_scoring.gold_answers(ex)
        message = f"{ex['query']}\n(Current time: {ex['query_time']})"
        hits = await tutor_chat.find_passages(site.env, act, message, [],
                                              site.worker._maybe_decrypt_factory(site.env.ENCRYPTION_KEY))
        started = time.time()
        try:
            response = await site.call("POST", f"/api/activities/{act}/tutor/chat", STUDENT, {"message": message})
        except RuntimeError as exc:
            response = {"outcome": "error", "answer": str(exc), "citations": []}
        short = [g.lower() for g in golds if len(g) <= 40 and "invalid" not in g.lower()]
        row = {
            "interaction_id": ex["interaction_id"], "question_type": ex.get("question_type"),
            "domain": ex.get("domain"), "static_or_dynamic": ex.get("static_or_dynamic"),
            "query": ex["query"], "query_time": ex.get("query_time"), "gold": golds,
            "tutor_outcome": response.get("outcome"), "tutor_answer": response.get("answer"),
            "cited": [{"title": c.get("source_title"), "locator": c.get("locator")} for c in response.get("citations") or []],
            "prediction": crag_scoring.prediction_from_response(response),
            "passages_indexed": ex["_index"]["chunks"],
            "top_scores": [h["score"] for h in hits],
            "answer_in_pages": crag_scoring.answer_in_pages(golds, [html_text(p["page_result"]) for p in pages_of(ex)]),
            "answer_in_top5": (any(g in " ".join(h["text"] for h in hits).lower() for g in short) if short else None),
            "seconds": round(time.time() - started, 1),
        }
        try:
            score = crag_scoring.score_prediction(row["query"], golds, row["prediction"], judge)
            if row["question_type"] == "false_premise" and row["prediction"] != crag_scoring.IDK:
                row["noticed_false_premise"] = judge(row["query"], crag_scoring.FALSE_PREMISE_GOLD,
                                                     row["prediction"]).get("score") == 1
        except BudgetExceeded as exc:
            stopped = f"stopped while judging question {i}: {exc}"
            break
        row.update(outcome=score["outcome"], how=score["how"], judge=score["judge"])
        rows.append(row)
        mark = {"correct": "✓", "missing": "·", "hallucination": "✗"}[row["outcome"]]
        print(f"  {mark} [{i}/{len(examples)}] {row['question_type']:18} {row['domain']:8} "
              f"{row['tutor_outcome']:10} {ex['query'][:55]!r}", flush=True)

    if stopped:
        print(f"\nBudget: {stopped}. Run the rest tomorrow with --offset {args.offset + len(rows)}.")
    if not rows:
        return
    summary = crag_scoring.summarize(rows)
    print_summary(summary, rows, ai.stats, workers_ai)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "benchmark": "CRAG Task 1 sample (crag_task_1_and_2_dev_v4)",
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "setup": {"chat_and_judge_model": model_name, "embeddings": f"{EMBED_MODEL} (fastembed, local)",
                  "vector_search": "exact cosine, in memory", "html_to_markdown": "markdownify",
                  "judge_prompt_sha": templates_sha, "retrieval_k": tutor_chat.RETRIEVAL_K,
                  "min_relevance": tutor_chat.MIN_RELEVANCE, "offset": args.offset},
        "deviations": ([] if workers_ai else ["production uses Llama 3.3 70B on Workers AI; this run uses a local model"]) + [
            "CRAG judges with GPT-4; this run judges with the same model that answers",
            "page HTML is converted with markdownify, not Workers AI toMarkdown; vector search is exact",
            "predictions not truncated to 75 tokens; citation markers removed",
            "tutor outcomes other than 'answered' count as \"I don't know\"",
            "query_time appended to the question"],
        "stopped": stopped, "stats": ai.stats, "summary": summary, "results": rows,
    }, indent=1))
    print(f"\nFull report (every question, answer and judge verdict): {args.out}")


def print_summary(summary, rows, stats, workers_ai=False):
    o = summary["overall"]
    print(f"\n=== CRAG score {o['score']:+.3f}   ✓ correct {o['correct']}   · missing {o['missing']}   "
          f"✗ hallucinated {o['hallucination']}   (n={o['n']})")
    for key in ("by_question_type", "by_domain"):
        print(f"\n{key[3:].replace('_', ' ')}:")
        for name, r in summary[key].items():
            print(f"  {name:20} {r['score']:+.2f}   ✓{r['correct']} ·{r['missing']} ✗{r['hallucination']}")
    known = [r for r in rows if r["answer_in_pages"] is not None]
    if known:
        in_pages = [r for r in known if r["answer_in_pages"]]
        in_top5 = [r for r in in_pages if r["answer_in_top5"]]
        print(f"\nretrieval (short answers that appear word-for-word in the pages):"
              f"\n  answer in the pages:          {len(in_pages)}/{len(known)}"
              f"\n  ...and in the 5 passages retrieved: {len(in_top5)}/{len(in_pages)}"
              f"\n  ...and the tutor got it right:      {sum(r['outcome'] == 'correct' for r in in_top5)}/{len(in_top5)}")
    fp = [r for r in rows if "noticed_false_premise" in r]
    if fp:
        print(f"\nfalse premise (extra, not in the CRAG score): pointed out the false premise in "
              f"{sum(r['noticed_false_premise'] for r in fp)}/{len(fp)} answered cases")
    print(f"\ntime: embedding {stats['embed_seconds']:.0f}s for {stats['embedded']} passages "
          f"({stats['embed_cached']} from cache), {stats['chat_calls']} model calls in {stats['chat_seconds']:.0f}s")
    if workers_ai:
        print(f"Workers AI: {stats['prompt_tokens']:,} tokens in, {stats['output_tokens']:,} out "
              f"= ~{stats['neurons']:,.0f} neurons")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="crag/sample.jsonl", help="from crag_sample.py")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--backend", choices=("workers-ai", "ollama"), default="workers-ai",
                   help="workers-ai: the production model via the crag-llm helper; ollama: a local model")
    p.add_argument("--llm-url", default="http://127.0.0.1:8798", help="the crag-llm helper (wrangler dev)")
    p.add_argument("--budget-neurons", type=float, default=8000,
                   help="stop before Workers AI use passes this (free allowance: 10,000 a day)")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model, for --backend ollama")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--templates", default=TEMPLATES_URL, help="CRAG's prompts/templates.py (URL or path)")
    p.add_argument("--embed-cache", default="crag/embeddings-cache.json", help="re-runs skip re-embedding")
    p.add_argument("--out", default="crag/report.json")
    args = p.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
