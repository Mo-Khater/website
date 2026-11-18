# CRAG benchmark for the AI tutor

[CRAG](https://github.com/facebookresearch/CRAG) is Meta's Comprehensive RAG Benchmark (KDD Cup 2024). This folder runs a sample of it through the tutor: the real worker code, with the production chat model, but almost entirely on your own machine.

`dataset.json` / `run_eval.py` (see `README.md`) test the tutor's **teacher controls** on a course we wrote. CRAG tests the **RAG pipeline** on messy real-world web pages and questions we didn't write.

## CRAG in brief

**Each question comes with**
- the question, a timestamp (`query_time`) and a gold answer, plus alternatives (`alt_ans`);
- up to 5 real web pages, as full HTML, that may or may not contain the answer;
- one of 8 question types (simple, simple with condition, comparison, aggregation, set, multi-hop, post-processing, false premise) and one of 5 domains (finance, sports, music, movie, open).

**Scoring**

| Answer | Points |
|---|---|
| Correct | +1 |
| "I don't know" | 0 |
| Wrong (hallucinated) | −1 |

**CRAG score** = (correct − wrong) / n, between −1 and +1.

A system that makes things up scores below one that admits it doesn't know, so the benchmark directly tests whether the tutor stays grounded in its materials.

## Quick start

All paths are relative to `learn/`.

```bash
pip install fastembed markdownify beautifulsoup4             # once
python3 tutor_eval/crag_sample.py --n 40 --out crag/sample.jsonl
# start the crag-llm helper (below), then:
python3 tutor_eval/run_crag.py --sample crag/sample.jsonl --limit 40 --out crag/report.json
```

**Variants**
- **Fully offline**, with a local model rather than the production one: `--backend ollama --model llama3.1:8b`.
- **Next day:** `--offset 20 --limit 20` continues where a budget-limited run stopped.

## How it works

```
crag_sample.py ──► crag/sample.jsonl     (40 questions, each with its 5 HTML pages)
                          │
run_crag.py               ▼
  for each question:  Site.add_course  ─► a course whose materials are the pages
                      POST …/tutor/reindex      convert → chunk → embed → index  (worker code)
  for each question:  tutor_rag.retrieve        top 5 passages, kept for diagnostics
                      POST …/tutor/chat         the student's question           (worker code)
                      crag_scoring              exact match / invalid rule / LLM judge
                          │
                          ▼
                  crag/report.json  + printed summary
```

### 1. Sampling: `crag_sample.py`

CRAG's Task 1 file is 705 MB of bzip2.
- **`stream_examples`** decompresses the file as it downloads, using `bz2.BZ2Decompressor` on 1 MB blocks. It stops after `--scan` examples (400 by default), so the whole file is never fetched.
- **`usable`** drops examples with no question, no gold answer, or no page HTML.
- **`balanced_pick`** round-robins over the 8 question types and, within each type, over the domains. Each type starts on a different domain, so any prefix of the sample (`--limit 10`) is already mixed by both type and domain.

### 2. The worker, in-process: `run_crag.py`

The runner imports the real worker (`tests.helpers.load_worker`) and calls its `_dispatch` with signed requests, the same way the unit tests do. The tutor's own code therefore runs every step:
- creating the course;
- reading materials from R2;
- `toMarkdown` conversion and heading-aware section splitting;
- chunking (1,200 characters with 200 overlap);
- embedding and indexing;
- multi-query retrieval with the relevance floor;
- the prompt, the citation check and the outcomes.

**What's replaced** (`tests.conftest` stubs the Workers runtime itself):

| In production | Here | Code |
|---|---|---|
| D1 | in-memory SQLite | `tests.tutor_fakes.SqliteD1` |
| R2 | dict of bytes | `tests.tutor_fakes.FakeR2` |
| Vectorize | exact cosine search in memory | `tests.tutor_fakes.FakeVectorize` |
| Workers AI `toMarkdown` | BeautifulSoup and markdownify | `html_to_markdown` |
| `@cf/baai/bge-base-en-v1.5` | the same model, `BAAI/bge-base-en-v1.5`, via fastembed on the CPU | `LocalAI._embed` |
| Llama 3.3 70B on Workers AI | **the same model on Workers AI** (default), or an Ollama model | `LocalAI.chat` |

**`LocalAI`** stands in for the `AI` binding. It implements the two methods the tutor calls:
- **`run(model, payload)`:**
  - A payload with `messages` is a chat call. It goes to the production model through the helper, or to Ollama.
  - A payload with `text` is an embedding call. It runs locally and returns `{"shape", "data"}` like Workers AI.
  - Vectors are L2-normalised, as Workers AI's are, and cached in `crag/embeddings-cache.json` keyed by the SHA-1 of the text. Re-runs never re-embed.
- **`toMarkdown(docs)`:**
  - It removes `script`, `style`, `svg`, `iframe`, `head` and similar elements, then converts the HTML to Markdown with ATX headings, so the tutor's section splitter sees real headings.
  - On failure it returns `format: "error"`, as the service does.

**`Site.add_course(example)`** turns one CRAG question into a course:
1. The host creates the activity.
2. The student joins it.
3. The host sets the policy: enabled, `answer` mode, no refused topics, no limit.
4. Each page is written to R2 as `materials/<activity>/page-<n>.html`, with an `activity_materials` row titled with the page name.

**The run loop** has two passes:
- **Indexing:** every question's course is created and reindexed. The cache is saved after each question, so an interrupted run loses nothing.
- **Asking:** for each question, the runner
  1. runs `tutor_rag.retrieve` with the same queries the chat will use, and records the top 5 passages. This is for diagnostics only; the chat does its own retrieval.
  2. sends the question to `POST /api/activities/:id/tutor/chat` as the student, with `(Current time: <query_time>)` appended, as CRAG's own baseline does.
  3. turns the tutor's response into a CRAG prediction and scores it straight away.

### 3. Getting the production model locally: the `crag-llm` helper

Python can't call the `AI` binding directly. A three-line Worker, run with `wrangler dev`, forwards chat calls using your `wrangler login`:

```js
// crag-llm/worker.js: POST {messages, temperature, max_tokens} -> {response, usage}
export default { async fetch(req, env) {
  if (req.method !== "POST") return Response.json({ model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast" });
  const { messages, temperature = 0.2, max_tokens = 350 } = await req.json();
  const out = await env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", { messages, temperature, max_tokens });
  return Response.json({ response: out.response, usage: out.usage });
} };
```

**Setup**
- **`wrangler.toml`** needs only `name`, `main`, `compatibility_date` and `[ai] binding = "AI"`. Don't use `learn/wrangler.toml`.
- **Start it** with `npx wrangler dev --port 8798`.
- **Model check:** the runner calls the helper with GET first, and refuses to start if the helper serves a different model from `tutor_chat.CHAT_MODEL`.

### 4. The budget guard

Only chat calls (answers and judge) reach Cloudflare. They are billed in neurons:
- 26,668 per million input tokens;
- 204,805 per million output tokens;
- the free allowance is 10,000 neurons a day.

**What the runner does**
- **Before every call:** it estimates the worst case (prompt characters ÷ 4 in, `max_tokens` out). If that would pass `--budget-neurons` (default 8,000), it raises `BudgetExceeded` and the call is never sent.
- **After every call:** it adds the neurons Workers AI reports in `usage.neurons`, or the published price if that's missing.
- **Before each question:** it needs at least 500 neurons left, so no question ends half-scored. If a run stops, it prints the `--offset` to continue from.

**Measured cost:** 40 questions took 63 calls, 117,881 tokens in and 2,752 out, about **3,700 neurons**, or 93 per question. Embedding, the expensive part (9,319 passages), is local and free.

### 5. Scoring: `crag_scoring.py`

These are pure functions; the judge is passed in. They follow CRAG's `local_evaluation.py`.

**`prediction_from_response`**
- The tutor never says "I don't know"; it returns an outcome instead. Any outcome other than `answered` (`not_found`, `withheld`, `refused_topic`, `error`) becomes the prediction `"I don't know"`.
- Answered text has its `[n]` citation markers removed.

**`score_prediction`** checks the prediction against each gold answer (`answer` plus `alt_ans`, de-duplicated), in order:
1. If the prediction contains "i don't know", it is **missing**.
2. An exact, case-insensitive match is **correct**.
3. **The invalid rule:** CRAG's false-premise questions have the gold answer "invalid question". If either side contains "invalid", the answer is correct only if both do. No judge call is made.
4. Otherwise **the LLM judge** decides. It uses CRAG's own prompt, built by `judge_messages` exactly as CRAG builds it. A score of 1 means correct.

If no gold answer matches, the result is **hallucination**. A judge reply that can't be parsed scores −1, which also counts as hallucination.

**`load_judge_prompt`** (in `run_crag.py`)
- It downloads CRAG's `prompts/templates.py` and pulls out `INSTRUCTIONS` and `IN_CONTEXT_EXAMPLES` with `ast.literal_eval`. The file is never executed.
- The file's SHA-256 is recorded in the report.
- `parse_judge_reply` reads the last `{...}` block, as CRAG's `parse_response` does.

**`summarize`** gives the score, accuracy, hallucination rate and missing rate, overall and for each question type, domain and `static_or_dynamic`.

### 6. Diagnostics beyond the CRAG score

A single score doesn't tell you *where* the pipeline failed, so each result row also records:

| Field | What it tells you |
|---|---|
| `answer_in_pages` | A short gold answer (≤ 40 characters) appears word for word in the pages' plain text, so the answer was available. `null` for long or false-premise answers. |
| `answer_in_top5` | The same check on the 5 retrieved passages, so retrieval found it. |
| `top_scores` | Cosine scores of the top 5 passages, to compare with `MIN_RELEVANCE` (0.5). |
| `passages_indexed` | How many chunks the pages produced. |
| `noticed_false_premise` | False-premise questions only: a second judge call against "The question's premise is false…". The tutor is told to *correct* false premises rather than reply "invalid question", so this measures that behaviour. It's not part of the CRAG score. |
| `tutor_outcome`, `tutor_answer`, `cited` | The raw tutor response: outcome, full text and the cited sources. |
| `judge` | Every judge verdict with its explanation. |

**How the printed summary uses them:** it chains the checks: answer in pages → in the top 5 → answered correctly. That separates retrieval failures from generation failures.

**Seeing gold answers next to the tutor's:**

```bash
jq -r '.results[] | "\(.outcome)\t\(.query)\n  gold:  \(.gold|join(" | "))\n  tutor: \(.tutor_answer)\n"' crag/report.json
```

## How this differs from the official CRAG setup

The `deviations` list in every report records these:
- **Judge:** CRAG judges with GPT-4. Here the same model that answers is the judge. Read its explanations for close calls.
- **Conversion and search:** markdownify instead of Workers AI `toMarkdown`, and exact rather than approximate vector search. Retrieval can differ slightly from production.
- **`--backend ollama`:** answers with a local model instead of the production one. Use it to test chunking and retrieval offline; its scores don't represent the deployed tutor.
- **Predictions:** not cut to 75 tokens; citation markers are removed instead.
- **Not answering:** non-`answered` outcomes count as "I don't know".
- **Query time:** `query_time` is added to the question.

## Results: first run (2026-11-17)

**Setup:** 40 questions, 5 per type and 8 per domain. Llama 3.3 70B on Workers AI answers and judges. `RETRIEVAL_K` 5 and `MIN_RELEVANCE` 0.5.

| | Score | Correct | Don't know | Wrong |
|---|---|---|---|---|
| **Overall** | **−0.175** | 8 | 17 | 15 |
| simple | +0.40 | 3 | 1 | 1 |
| post-processing | +0.20 | 2 | 2 | 1 |
| simple with condition | 0.00 | 1 | 3 | 1 |
| aggregation | 0.00 | 1 | 3 | 1 |
| comparison | −0.20 | 1 | 2 | 2 |
| set | −0.60 | 0 | 2 | 3 |
| multi-hop | −0.60 | 0 | 2 | 3 |
| false premise | −0.60 | 0 | 2 | 3 |

**Retrieval chain:** 22 questions had a short gold answer.
- The answer appeared in the pages for 12 of them.
- It was in the top 5 passages for 7 of those 12.
- The tutor answered 2 of those 7 correctly.

**False premise:** the tutor pointed out the false premise in 1 of the 3 false-premise questions it answered.

### Why answers were scored wrong

| Cause | Count | Example |
|---|---|---|
| The tutor said the sources don't have the answer, but in a full sentence with citations, so the outcome was `answered`. | 8 | "The sources do not provide a ratio for Harvard…" |
| False premise: CRAG accepts only "invalid question". | 3 | "Hamburg is the second largest city, not the biggest" (a correct correction) |
| The gold answer disagrees with the pages, or the question is ambiguous. | 3 | The pages say 7.4 m and the gold says 7.5 m. |
| A real mistake | 1 | Answered about the Alabama River instead of the longest river in Alabama. |

**Main conclusion:** the tutor rarely invents facts. Most "wrong" answers were refusals worded as answers. If those 8 had been logged as not found, the score would be about +0.03.

### What to improve

1. **Refusals logged as answers:** treat an answer that says the sources don't cover the question as `not_found`, or add a `partial` outcome. This also affects the teacher's review log, not only the benchmark.
2. **Retrieval recall:** 5 of 12 answers that were in the pages never reached the top 5 passages. Options: a larger `RETRIEVAL_K`, or keyword search alongside vectors for names and numbers.
   - **A hybrid prototype** (vectors plus BM25) was measured and then removed pending a security review; see `TUTOR.md` section 12.
   - Measured on retrieval alone, without the time suffix, it reached the top 5 in 8 of 12 questions, the same as vector search but on a different pair.
   - With 10 passages, it found 10 of 12 against 9 for vector search alone.
3. **Duplicate passages:** identical text, such as repeated menus, takes several of the 5 slots. De-duplicate by text hash at retrieval.
4. **Navigation and boilerplate chunks** outrank content. Filter them when chunking.
5. **Citation-only answers:** an answer that was only "[1]" passed the citation check (seen with the 8B model). Require some text besides the markers.

## Tests

`tests/test_tutor_crag.py` covers everything that doesn't need a model or the network:
- converting tutor responses into predictions;
- each scoring rule (exact match, the invalid rule, judging against each gold answer, judge errors);
- parsing the judge's reply and the judge message format;
- the summary maths and the `answer_in_pages` check;
- balanced sampling;
- the budget guard refusing a call before it is sent.

```bash
python3 -m pytest tests/test_tutor_crag.py -q
```

## Data licence

CRAG is CC BY-NC 4.0. Keep the data and reports out of git; `crag/` is in `.gitignore`.
