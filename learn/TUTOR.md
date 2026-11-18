# AI tutor: course-grounded RAG

Each course (activity) gets a tutor that answers students' questions **only from that course's own materials**, cites where every answer came from, and follows rules the teacher sets.

This document explains how the retrieval-augmented generation (RAG) pipeline is built and why it's built that way.

| File | What it does |
|---|---|
| `src/tutor_rag.py` | **Retrieval layer:** reads files, splits them into located sections, chunks, embeds, indexes and searches. It knows nothing about HTTP, auth or policy. |
| `src/tutor_chat.py` | **Answering:** retrieval → prompt → model → citation check → outcome. |
| `src/tutor_policy.py` | **Teacher policy:** validation, the refused-topic check and the prompt rules. |
| `src/worker.py` | Endpoints, access control, collecting the course's sources, conversation storage, review log. |
| `public/tutor-settings.html` | Teacher's settings page (policy, reindex, review log). |
| `public/activity-detail.html` | Student's tutor chat card. |

## Overview

```
                    INDEXING  (teacher clicks "Reindex")
 course description, objectives, prerequisites ─┐
 session descriptions ──────────────────────────┤
 material descriptions + files in R2 ───────────┤
   .pdf .docx .xlsx .html …  → Workers AI toMarkdown ─┐
   .pptx → own parser (slides + speaker notes)  ──────┤→ located sections → chunks (1,200 chars, 200 overlap)
   .md / .txt → read directly ────────────────────────┘        │
                                                  only changed chunks (content hash)
                                                               │
                         bge-base-en-v1.5 (768-d) ◄────────────┘
                              │                    │
                 Vectorize (namespace = course)   D1 tutor_chunks (encrypted text, locator, hash)

                    ANSWERING  (student asks)
 question ─► refused topic? ──yes──► refusal (model not called)
              │ no
              ▼
 queries: [question, previous question + question] ─► embed ─► Vectorize top 5 per query
              │  best score per chunk, keep ≥ 0.5, top 5 overall
              ▼
 nothing left? ──yes──► "not found" (model not called)
              │
              ▼
 Llama 3.3 70B: system prompt + numbered sources + policy rules + last 6 messages + question
              │
              ▼
 "NOT_IN_COURSE"? ─► not_found      invalid [n] markers removed
 < 50% of sentences cited? ─► withheld (text kept for the teacher only)
              │
              ▼
 answered: text with [n] markers + citations (source, locator, passage text)
```

## 1. What the tutor can read

`_tutor_sources` in `worker.py` collects a course's content as **sources**. Each source has a type, an id, a title and **sections**, and each section has a **locator** that says where in the source it is.

| Source type | Content | Locators |
|---|---|---|
| `activity` | course description, learning objectives, prerequisites | `description`, `learning objectives`, `prerequisites` |
| `session` | each session's description | (none) |
| `material` | the material's description plus its uploaded file | `description`, then the file's own locators (below) |

**Course content is encrypted at rest** (`v1:` AES values), so every field is decrypted first.

**Files the tutor can't read** are returned as `skipped_files`, with a reason, so the teacher sees what to fix. Examples: "old PowerPoint format — save it as .pptx or PDF", "external links are not fetched".

### File extraction: `extract_file_sections`

Files are read from R2. The limits are 10 MB per file and 200,000 extracted characters per file.

| Extension | How it's read | Locators |
|---|---|---|
| `.txt` | as is | (none) |
| `.md` | split at headings | `section: <heading>` |
| `.pdf .docx .odt .xlsx .ods .csv .html .htm .xml` | Workers AI `toMarkdown`, then split at headings | `page N`, `page N, section: <heading>`, `section: <heading>` |
| `.pptx` | our own OOXML parser | `slide N` |
| `.ppt .doc .odp` | refused, with a "save as" hint | |
| images, other types | refused: "unsupported file type" | |

**Details**
- **PDF pages:** `toMarkdown` marks PDF pages with `Page N` headings. `split_markdown_sections` turns these into `page N`, and headings nested under a page keep the page number: `page 3, section: Mitosis`.
- **Metadata section dropped:** `toMarkdown` puts a `## Metadata` list (producer, creation date…) at the start of a document. That section is removed by `_is_metadata_section`.
- **Heading-only sections** are dropped.
- **Why a custom `.pptx` parser:** `toMarkdown` doesn't support PowerPoint, and slides are the most common teaching material. `extract_pptx_sections`:
  - reads slides in **presentation order** (`presentation.xml` relationships, not file names);
  - includes **speaker notes**;
  - includes tables and grouped shapes;
  - skips slide-number, date, footer and header placeholders;
  - refuses files whose uncompressed XML is over 20 MB (zip-bomb guard) and reads at most 500 slides.
- **Images are deliberately excluded:** converting them would go through a paid vision model, and they rarely carry course text.

## 2. Chunking: `chunk_text`, `build_chunks`

- **Chunk size:** at most **1,200 characters**. Splits prefer paragraph boundaries, then sentence boundaries, then a hard cut.
- **Overlap:** consecutive chunks share up to **200 characters** of trailing context, so an answer that straddles a boundary is still retrievable.
- **Chunks never span two sections**, so every chunk's locator is exact ("slide 7", not "slides 6–7").
- **Embedded text is prefixed with its location** by `embedding_input`: `"<source title> — <locator>\n<text>"`. A short chunk like "It releases oxygen." still carries "Photosynthesis — slide 4".
- **Cap:** at most **2,000 chunks per course** (`MAX_CHUNKS_PER_ACTIVITY`), so one course can't run up the AI bill.

## 3. Indexing: `index_activity`

Run by `POST /api/activities/:id/tutor/reindex` (host only).

**Identity and change detection**
- **Chunk id:** `sha256(activity | source_type | source_id | chunk_index)[:48]`. The same id is used in D1 and in Vectorize.
- **Content hash:** `sha256(embedding model, title, locator, text)`.

**Incremental:** existing hashes are read from D1, then
- only chunks whose hash changed are embedded and upserted;
- chunks whose source disappeared are deleted from both stores;
- changing the embedding model changes every hash, so a reindex re-embeds everything.

The response is `{chunks, embedded, unchanged, removed, skipped_files}`.

**Where things are stored**

| Store | Holds | Why |
|---|---|---|
| **Vectorize** (`VECTORIZE`, 768-d, cosine) | vectors only, in a **namespace per course**; metadata is only `source_type` / `source_id` | A query is scoped to the course's namespace, so it can never return another course's content. No course text goes to Vectorize. |
| **D1 `tutor_chunks`** | id, source, title, locator, chunk index, **AES-encrypted text**, content hash | Text is encrypted like the rest of learn's course content. |

**Embeddings:** Workers AI `@cf/baai/bge-base-en-v1.5`, in batches of 50. Upserts go in batches of 500, deletes in batches of 50 (D1's limit on bound parameters).

New vectors can take up to a minute to become searchable in Vectorize; the reindex response says so.

## 4. Retrieval: `tutor_rag.retrieve`

```python
retrieve(env, activity_id, query: str | list[str], decrypt, top_k=5, min_score=0.0)
```

1. **Prepare the queries:** strip and de-duplicate them, and cut each to 1,000 characters.
2. **Embed all queries in one call.**
3. **Search:** query Vectorize once per vector, in the course's namespace, for `top_k` matches.
4. **Merge:** each chunk keeps its **best score across the queries**, and scores below `min_score` are dropped.
5. **Return:** take the best `top_k` overall, load their rows from D1, decrypt the text, and return hits best first.

Each hit is `{chunk_id, source_type, source_id, source_title, locator, chunk_index, text, score}`, which is everything needed to cite it.

Retrieval is vector-only. Keyword (hybrid) search was prototyped and measured, then removed until its security trade-offs are settled; see [section 12](#12-proposed-hybrid-search-not-implemented).

### Why several queries: `tutor_chat.retrieval_queries`

The chat searches with **the question alone** and **the previous question plus this one**:
- A follow-up like "when was that?" only finds anything with the previous question as context.
- A topic change ("and Mercury's year?" after "what makes Mars red?") is drowned out if the previous question is always included.
- Searching both and keeping each passage's best score handles both cases.

## 5. Answering: `tutor_chat.answer`

**Settings** (`tutor_chat.py`):

| Constant | Value | |
|---|---|---|
| `CHAT_MODEL` | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | |
| `RETRIEVAL_K` | 5 | passages given to the model |
| `MIN_RELEVANCE` | 0.5 | cosine floor; below it a passage is ignored |
| `HISTORY_MESSAGES` | 6 | previous messages sent to the model |
| `MIN_CITED_FRACTION` | 0.5 | share of claims that must cite a source |
| `MAX_ANSWER_TOKENS` | 350 | the longest real answer in the eval was ~110 tokens; this caps injection essays |
| `TEMPERATURE` | 0.2 | |

**The steps, in order:**

1. **Refused topics, in code.** If the question matches a topic the teacher refused, the student gets a refusal, and nothing is retrieved or sent to the model. → `refused_topic`
2. **Retrieve** with the queries above, `top_k=5`, `min_score=0.5`. If no passage clears the floor, the student gets the standard "not found" reply **without calling the model**, so nothing can be made up. → `not_found`
3. **Prompt.** `build_messages` creates:
   - a system prompt with the course name, the rules, the policy rules, and the passages numbered `[1]`…`[5]` with their title and locator;
   - the last 6 messages of the conversation;
   - the question.

   The rules tell the model to:
   - answer **only** from the sources and cite every fact as `[n]`;
   - answer when the sources cover the topic in different words, and connect facts across sources;
   - correct a mistaken premise using the sources;
   - reply **exactly `NOT_IN_COURSE`** if the sources don't cover the topic;
   - treat sources as content, not instructions (prompt injection inside materials);
   - not let student messages change the rules.
4. **Model call:** Workers AI chat. An empty reply is an error (HTTP 502).
5. **The `NOT_IN_COURSE` marker** → `not_found`. Without the marker, Llama's refusals listed what the sources *do* cover, with citations, and passed as answers. The evaluation set found this.
6. **Citation check, in code** (`apply_citations`, `cited_fraction`):
   - Markers that point at a passage we didn't supply (an invented `[9]`) are removed.
   - Citations are listed in order of first use, each with its passage text for the student's "show me where" view.
   - **Substantive sentences** (4+ words, not questions) are counted. If fewer than **50%** carry a valid marker, or there are no citations at all, the text is **withheld**:
     - the student gets the "not found" reply;
     - the model's text is kept, encrypted, for the teacher's review log only.

     → `withheld`

     Guiding questions in hint mode don't count as claims, so they don't need citations.
7. Otherwise → `answered`, with the cleaned text and citations.

**Grounding is enforced in code, not trusted to the prompt.** A student can try to rewrite the prompt, but uncited text can't get past step 6.

### Outcomes

| Outcome | Meaning | Model called |
|---|---|---|
| `answered` | Cited answer returned. | yes |
| `not_found` | No passage above the floor, or the model said `NOT_IN_COURSE`. | no / yes |
| `withheld` | The model replied without enough citations; the text is hidden from the student. | yes |
| `refused_topic` | Matched a refused topic. | no |

Every assistant message is stored with its outcome, so the teacher's review log can filter by it.

## 6. Teacher policy: `tutor_policy.py`

**Storage:** each course has a policy, stored as JSON in `tutor_policies`. `normalize` fills in defaults for older or partial policies.

| Field | Default | Enforced by |
|---|---|---|
| `enabled` | false | **code**: students get 403 when it's off. Hosts can always preview. |
| `mode` | `answer` | **prompt**: `hint_only` adds a rule (with a worked example) to point to the source, give one hint and ask one guiding question, never the answer. |
| `refused_topics` | [] | **code first**, then also listed in the prompt. A topic matches when every word of 3+ letters starts a word in the question ("derivative" also catches "derivatives"). Matching is predictable on purpose. Limits: 20 topics, 100 characters each. |
| `daily_limit` | 30 | **code**: questions per student per UTC day, counted from stored messages; 429 when reached. 0 = no limit; maximum 500. Hosts are exempt. |
| `instructions` | "" | **prompt**: appended as "Teacher's instructions". Maximum 1,000 characters. |

**Mid-conversation changes:** if the teacher changes the settings mid-conversation, earlier replies in the history were written under the old settings. `HISTORY_RULE` is added whenever the history has assistant turns. It tells the model to follow the current rules, not the style of earlier replies. Without it, the model kept giving hints after the teacher switched to full answers.

## 7. Conversations and privacy

- **History comes from the server, never the client.** `api_tutor_chat` loads the last 6 messages of `conversation_id` from `tutor_messages`, for this course and this user only. A caller can't plant fake assistant turns to steer the tutor. An unknown conversation id returns 404.
- **Storage:** every question and answer is stored **encrypted** (`content_encrypted`), with the citations (without passage text), `grounded`, `outcome` and any withheld text.
- **Students are told:** the status endpoint returns `notice`: "Your teacher can read your questions and the tutor's answers." The chat card shows it.
- **Access:**
  - **search, chat and status:** the host, an admin, or an **actively enrolled** student;
  - **reindex, policy and log:** host only.

## 8. Endpoints

| Method | Endpoint | Who | What |
|---|---|---|---|
| POST | `/api/activities/:id/tutor/reindex` | host | Sync the index; returns chunk counts and `skipped_files`. |
| GET | `/api/activities/:id/tutor/search?q=&k=` | host / enrolled | Raw retrieval (`k` ≤ 20) for debugging. No floor, no model. |
| POST | `/api/activities/:id/tutor/chat` | host / enrolled | `{message, conversation_id?}` → `{conversation_id, answer, citations, grounded, outcome}`. Message ≤ 2,000 characters. |
| GET | `/api/activities/:id/tutor` | host / enrolled | Status: `available`, `enabled`, `is_host`, `mode`, `daily_limit`, `used_today`, `remaining_today`, `notice`. |
| GET / PUT | `/api/activities/:id/tutor/policy` | host | Read the policy (with its limits), or update any subset of fields. |
| GET | `/api/activities/:id/tutor/log?limit=&before=&outcome=&student=` | host | Review log, newest first: question, answer, citations, outcome, withheld text, and counts per outcome. It pages by insert sequence (`next_before`), so entries created in the same millisecond are never skipped. |

**When the bindings are missing:** every tutor endpoint returns **503** until both `AI` and `VECTORIZE` are configured.

**The student card** re-reads the status endpoint when the window regains focus or the tab becomes visible, at most every 3 s. A mode change, or the teacher turning the tutor off, therefore shows up without a reload.

## 9. Setup

```bash
npx wrangler vectorize create learn-tutor --dimensions=768 --metric=cosine
```

Then uncomment the `[[vectorize]]` block in `wrangler.toml`; the `[ai]` binding is already there. Teachers then open the tutor settings page, turn the tutor on and click **Reindex**.

**Cost per question:** one embedding call plus at most one Llama 3.3 70B call, about 100 Workers AI neurons. Refused, out-of-scope and limited questions skip the model. Reindexing only embeds changed chunks.

## 10. Testing and evaluation

| What | Where |
|---|---|
| **Unit tests:** chunking, section splitting, `.pptx` parsing, incremental indexing, multi-query retrieval, prompt building, citation checks, outcomes, policy validation, endpoints. They run against in-memory D1, R2 and Vectorize fakes (`tests/tutor_fakes.py`). | `tests/test_tutor_rag.py`, `test_tutor_chat.py`, `test_tutor_policy.py` |
| **Evaluation set:** 38 cases on a small original course, run against a real worker. It covers answerable, out-of-scope, prompt-injection and refused-topic cases, under each policy. | `tutor_eval/README.md` |
| **CRAG** (Meta's RAG benchmark): real web pages and questions, run through the real pipeline with the production model. | `tutor_eval/CRAG.md` |

## 11. Known limitations

- **Hint-only mode is prompt-enforced.** A model can still leak the answer. The eval measures the leak rate; a verifier pass would be needed to enforce it.
- **Refused topics match keywords, not meaning.** "End-of-term test" is not caught by "final exam". The eval has a case that measures this.
- **Hedged answers count as answered.** A reply like "the sources don't give X, but…" with citations passes the citation check and is logged as `answered`. This was 8 of the 15 "wrong" answers in the first CRAG run.
- **Recall:** in CRAG, about 4 of 12 answers that were in the pages don't reach the top 5 passages. Vectors are weak on exact names and numbers. See [section 12](#12-proposed-hybrid-search-not-implemented).
- **No de-duplication or boilerplate filtering:** identical passages, such as repeated navigation menus on web pages, can take several of the 5 slots.
- **A citation-only answer** such as "[1]" passes the citation check.
- **`MIN_RELEVANCE` (0.5)** is tuned on the evaluation set, not on a large corpus.
- **Indexing runs in the request.** Very large courses are bounded by the 2,000-chunk cap rather than by a background queue.

## 12. Proposed: hybrid search (not implemented)

**Status:** prototyped, measured, then removed for security review. Nothing in the code or the database depends on it. This section records the design, the measurements and the open security questions, so the work can be picked up later.

### Why

Vector search matches meaning, but it is weak on exact names and numbers ("Keith Sweat", "56-26", "TNL dividends"). In CRAG, about a third of the answers that were in the pages never reached the 5 passages the model sees. Keyword (BM25) search finds exactly those terms, and hybrid search runs both and merges the results.

### The problem: chunk text is encrypted

The usual way to add keyword search on D1 is SQLite's FTS5, but FTS5 needs the plain text. `tutor_chunks` stores text AES-encrypted, like the rest of learn's course content, so any keyword index has to work without plaintext in the database.

### The prototype design

**Index: a blind index in a new table**

```sql
CREATE TABLE tutor_terms (
    activity_id TEXT NOT NULL,
    term        TEXT NOT NULL,     -- HMAC(key, activity_id + word)[:16], never the word
    df          INTEGER NOT NULL,  -- chunks containing the word
    postings    TEXT NOT NULL,     -- "chunk_id:tf:chunk_length ..."
    digest      TEXT NOT NULL,     -- hash of df + postings, to skip unchanged rows
    PRIMARY KEY (activity_id, term),
    FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE
);
```

**Tokenizing**
- Words are lower-cased, accents are removed, stopwords and single letters are dropped, simple plurals are folded ("hits" → "hit") and numbers are kept.
- Each chunk is indexed with its title and locator, the same text that is embedded.

**Hashing:** each word is hashed with HMAC, using a key derived from `ENCRYPTION_KEY` and scoped to the course, so the same word hashes differently in each course.

**Rows**
- **One row per word per course,** not per word and chunk. A first index of 2,000 chunks writes about 20,000 rows instead of about 200,000, which matters for D1's write limits.
- **A `#stats` row** holds the chunk count and the average chunk length.
- **Very common words:** a word in more than half the chunks keeps its `df` but no postings, so rows stay small.

**Keeping it in sync:** at the end of a reindex, all rows are rebuilt in memory from the chunks' plain text, which the reindex already has. Only rows whose `digest` changed are written, in multi-row INSERTs sent through `env.DB.batch`. Existing courses get the index on their next reindex, without re-embedding anything.

**Querying**
1. **Vector search:** Vectorize gives the top 20 per query; topK doesn't change Vectorize's price.
2. **Keyword search:** hash the question's words (at most 16), read their rows plus `#stats` in one D1 query, score the chunks with BM25 (k1 1.2, b 0.75) and keep the top 20.
   - It uses the current question only, not the previous one, so a topic change isn't pulled back.
3. **Filter:** keep a chunk if its cosine is at least 0.5, **or** it covers at least 50% of the question's words weighted by IDF.
   - A word the course lacks counts with the highest weight. So "How does CRISPR editing change a cell?" in a biology course doesn't pass on "cell" alone.
   - This filter is what keeps out-of-scope questions refused without a model call.
4. **Fuse:** merge the two rankings with reciprocal rank fusion, Σ 1 / (60 + rank), and take the top 5.

**Hits** gain `keyword_score`, and `score` (cosine) becomes `null` for chunks only the keyword search found.

**Other changes needed**
- The search endpoint gets `mode=vector|hybrid`, and `tutor_eval/run_eval.py` measures the cosine floor with `mode=vector`.
- The D1 test fake (`tests/tutor_fakes.py`) needs a `batch` method.

### What it measured

Retrieval only, with the same embedding model run locally:

| | Vector only | Hybrid |
|---|---|---|
| Eval set: right lesson ranked first (of 23 answerable) | 21 | **22** |
| Eval set: answerable question got no passage | 1 | **0** |
| Eval set: out-of-scope questions let through by keyword match alone | – | **0** |
| CRAG: answer reaches the 5 passages (12 where it's in the pages) | 8 | 8 (a different two missed) |
| CRAG: answer reaches 10 passages | 9 | **10** |
| CRAG: keyword search alone, 5 passages | – | 3 |

So hybrid helps on clean course material. On noisy web pages it helps only when more passages go to the model.

### Security concerns to resolve first

A blind index is not encryption. Someone who gets a copy of the D1 database, but not the key, can't read the words, but the index still reveals a lot:

1. **Frequency analysis.**
   - The `df` and `tf` values, and the number of distinct words, follow the statistics of the language.
   - Course materials are often public or partly public (textbooks, published slides).
   - An attacker who knows or guesses part of a course's text can match hash frequencies to words and recover much of the vocabulary. These are the known leakage-abuse attacks on searchable encryption (Islam et al. 2012; Cash et al. 2015).
2. **Co-occurrence and structure.**
   - The postings say which chunks share which words, how often, and how long each chunk is.
   - That is effectively a hashed bag of words for every chunk, which makes frequency analysis much stronger.
3. **Key reuse.** The HMAC key was derived from `ENCRYPTION_KEY`.
   - A leak of that one secret exposes both the content and the index.
   - Rotating it means rebuilding every index.
4. **Larger attack surface:** a new table holding derived course content, outside the existing encryption.

### Options when this is picked up

| Option | Leaks | Finds chunks that vectors miss | Cost |
|---|---|---|---|
| **A. Encrypted postings:** keep the term hash as the lookup key, but AES-encrypt `df` and `postings`. Decrypt only the ≤ 16 rows a question touches. | Vocabulary size, and which term rows exist. **No** frequencies, co-occurrence or chunk ids. | yes | One decryption per query word. Writes are the same. |
| **B. Rerank only:** no stored keyword index. Decrypt the top ~30 vector candidates at query time, score them with BM25 in memory, and fuse. | **nothing new** | no: it can only reorder what vector search found | Reads and decrypts ~30 chunks per question instead of 5. Not measured yet. |
| **C. Option A with a separate secret** (e.g. `TUTOR_INDEX_KEY`) | as A | yes | One more secret to manage. |
| D. FTS5 on plaintext | all chunk text, in the clear | yes | Rejected: breaks encryption at rest. |

**Suggested order**
1. **Measure B first.** It adds no leakage, and the CRAG numbers suggest many missed answers are within the vector top 20.
2. **If B isn't enough, use A with C.**

Either way, re-run the eval set and CRAG (retrieval-only, locally, at no cost) and keep the out-of-scope check at 0 leaks.
