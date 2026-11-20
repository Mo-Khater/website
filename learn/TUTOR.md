# AI tutor: course-grounded RAG

Each course (activity) gets a tutor that answers students' questions **only from that course's own materials**, cites where every answer came from, and follows rules the teacher sets.

This document describes the retrieval-augmented generation (RAG) pipeline as the code implements it, and why it's built that way. Measurements come from the evaluation set (`tutor_eval/README.md`) and the CRAG benchmark (`tutor_eval/CRAG.md`).

| File | What it does |
|---|---|
| `src/tutor_rag.py` | **Retrieval layer:** reads files, splits them into located sections, chunks, embeds, indexes, searches and reranks. It knows nothing about HTTP, auth or policy. |
| `src/tutor_chat.py` | **Answering:** query expansion → retrieval → prompt → model → checks → outcome. |
| `src/tutor_hints.py` | **Hint-only checks:** parses the hidden answer and the hint, finds hints that give the answer away, builds the replacement hint. |
| `src/tutor_policy.py` | **Teacher policy:** validation, the refused-topic check and the prompt rules. |
| `src/worker.py` | Endpoints, access control, collecting the course's sources, conversation storage, daily limit, review log. |
| `migrations/0016`–`0018` | Tables `tutor_chunks`, `tutor_messages`, `tutor_policies`. |
| `public/tutor-settings.html` | Teacher's settings page (policy, reindex, review log). |
| `public/activity-detail.html` | Student's tutor chat card. |

**Models (all Workers AI)**

| Model | Used for |
|---|---|
| `@cf/baai/bge-base-en-v1.5` | embeddings (768-d), for chunks, queries and (when chunking) sentences |
| `@cf/baai/bge-reranker-base` | reranking the vector candidates |
| `@cf/meta/llama-3.1-8b-instruct-fp8-fast` | query rewrites and keywords |
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | answers and hints |
| `toMarkdown` | converting PDF, Word, spreadsheets and HTML to Markdown |

## Overview

```
                    INDEXING  (teacher clicks "Reindex")
 course description, objectives, prerequisites ─┐
 session descriptions ──────────────────────────┤
 material descriptions + files in R2 ───────────┤
   .pdf .docx .xlsx .html …  → Workers AI toMarkdown ─┐
   .pptx → own parser (slides + speaker notes)  ──────┤→ located sections
   .md / .txt → read directly ────────────────────────┘        │
                                       clean: link/image URLs dropped
                                       join: sections < 300 chars join a neighbour (never slides)
                                       split: a section that fits is one chunk; a longer one is split
                                              by meaning (semantic chunking), ≤ 1,200 chars, no overlap
                                       dedupe: repeated chunk text indexed once
                                                               │
                                       only chunks whose content hash changed
                                                               │
                         bge-base-en-v1.5 (768-d) ◄────────────┘   (also each long section's sentences, to split it)
                              │                    │
                 Vectorize (namespace = course)   D1 tutor_chunks (encrypted text, locator, hash)

                    ANSWERING  (student asks)
 question ─► refused topic? ──yes──► refusal (no retrieval, no model)
              │ no
              ▼
 Llama 3.1 8B: is it a follow-up of the previous question? + 3 rewrites (sees only those two questions)
              │
              ▼
 embed [question, rewrites…] with BGE's query instruction
              │  Vectorize: top 20 for the question, top 5 per rewrite
              │  each chunk keeps its best cosine; drop < 0.5
              ▼
 bge-reranker-base: order the candidates against the question made to stand alone (rewrite 1)
              │
 keyword boost: + 0.25 × share of the question's keywords in the passage (8B picks them) ─► top 5
              ▼
 nothing left? ──yes──► "not found" (no model call)
              │
              ▼
 Llama 3.3 70B: system prompt (rules + policy rules + numbered sources) + last 6 messages + question
              │
              ├─ reply contains NOT_IN_COURSE ─────────────► not_found
              ├─ hint-only mode ─► check hint vs hidden answer, 1 retry ─► answered / hint_replaced / not_found / withheld
              ├─ reply says the sources lack the answer ───► not_found (text kept for the teacher)
              ├─ < 50% of claims cited, or only markers ───► withheld (text kept for the teacher)
              ▼
 answered: text with [n] markers + citations (source, locator, passage text)
```

## 1. What the tutor can read: `_tutor_sources` (worker.py)

A course's content is collected as **sources**. Each source has a type, an id, a title and **sections**, and each section has a **locator** that says where in the source it is.

| Source type | Sections, in order | Locators |
|---|---|---|
| `activity` | description; learning objectives (text prefixed "Learning objectives:"); prerequisites ("Prerequisites:") | `description`, `learning objectives`, `prerequisites` |
| `session` | the session's description, sessions ordered by start time | (none) |
| `material` | the material's description, then its file's sections; materials in display order | `description`, then the file's locators (below) |

- **Course content is encrypted at rest** (`v1:` AES values), so every field, including material titles, is decrypted first. Legacy plaintext passes through.
- **`quiz_explanation`** is an allowed source type in `tutor_rag.SOURCE_TYPES`, but nothing collects it yet.
- **Files the tutor can't read** are returned as `skipped_files`, each with a reason, so the teacher sees what to fix. Materials that are only an external link are skipped too ("external links are not fetched"); links are never fetched.

### File extraction: `tutor_rag.extract_file_sections`

Files are read from R2 (`MY_BUCKET`).

| Extension | How it's read | Locators |
|---|---|---|
| `.txt` | as is | (none) |
| `.md` `.markdown` | split at headings | `section: <heading>` |
| `.pdf .docx .odt .xlsx .ods .csv .html .htm .xml` | Workers AI `toMarkdown`, then split at headings | `page N`, `page N, section: <heading>`, `section: <heading>` |
| `.pptx` | our own OOXML parser | `slide N` |
| `.ppt .doc .odp` | refused, with a "save as .pptx / .docx / PDF" hint | |
| images, anything else | refused: "unsupported file type" | |

**Limits:** no file-size limit; at most 200,000 extracted characters per file (`_cap_sections` truncates the rest). The whole file is still read into the Worker's memory (128 MB) and sent to `toMarkdown`, so a very large file can make the reindex fail; that shows up as "could not read file" or a failed reindex rather than a size message.

**Skip reasons** the teacher can see: unsupported type, the "save as" hints, file not found in storage, file storage not configured, could not convert (with the converter's message), could not read the presentation, no extractable text, or "could not read file" for any other error.

**Details**
- **PDF pages:** `toMarkdown` marks pages with `Page N` headings. `split_markdown_sections` turns these into `page N`, and headings nested under a page keep the page number: `page 3, section: Mitosis`.
- **Metadata section dropped:** `toMarkdown` starts a document with a `## Metadata` list (producer, creation date…). `_is_metadata_section` removes it when every line is a `key=value` item.
- **Heading-only sections** are dropped. Headings are cut to 80 characters, locators to 120.
- **Why our own `.pptx` parser:** `toMarkdown` doesn't support PowerPoint, and slides are the most common teaching material. `extract_pptx_sections`:
  - reads slides in **presentation order** (from `presentation.xml`'s relationships; file-name order only as a fallback);
  - includes **speaker notes** (as "Speaker notes:" after the slide text), tables and grouped shapes;
  - skips slide-number, date, footer, header and thumbnail placeholders;
  - refuses files whose uncompressed XML is over 20 MB (zip-bomb guard) and reads at most 500 slides.
- **Images are deliberately excluded:** converting them goes through a paid vision model, and they rarely carry course text.

## 2. Chunking: `tutor_rag.build_chunks`

For each source, in order:

1. **Clean** (`clean_text`): Markdown links keep their text but lose the URL (`[the notes](https://…)` → `the notes`), and images keep their alt text.
   - **Why:** converted web pages and documents are full of links. In the pages of 40 CRAG questions, **44% of chunks were more than half URL**. URLs swamp the embedding of a short passage and cost model tokens.
   - A URL written out as the link's text (`[https://…](https://…)`) survives, because it's the text.
2. **Join short sections** (`merge_short_sections`): a section joins the previous one when either of them is under **300 characters** and together they fit in one chunk (1,200 characters). Runs of short sections can chain.
   - **Why:** a heading with two lines under it made a tiny chunk with no context.
   - **The locator names the first and last joined section:** `section: Diffusion – section: Osmosis`, or `description – prerequisites`.
   - **Slides are never joined.** Most slides are short, so nearly every one would merge, and `slide 7` is the citation a student can look up.
   - Sections are only joined within one source, never across sources.
3. **Split** each section into chunks of at most **1,200 characters**. A section that fits is one chunk.
   - **Longer sections are split by meaning** (`semantic_pieces`, `pack_sentences`; `SEMANTIC_CHUNKING = True`):
     - the section is cut into sentences (`sentences_of`: paragraph by paragraph; a sentence over 1,200 characters is hard-cut);
     - each sentence is embedded **together with the sentences either side of it**, so one short sentence doesn't look like a change of topic (the method of LangChain's SemanticChunker);
     - the distance between neighbouring sentences is 1 − cosine of their embeddings;
     - chunks are filled up to 1,200 characters, and each ends at the **biggest change of topic** once it has 400 characters (`SEMANTIC_MIN_CHARS`); with no change of topic in range, at the last sentence that fits;
     - chunks **don't overlap**: they end where the topic changes, so less is lost at the edges than with fixed-size cuts. Because each sentence is embedded with its neighbours, a cut can land one sentence before or after the change.
   - **Cost:** one embedding per sentence of every long section, **on every reindex**, even when nothing changed (chunks whose text is unchanged are still not re-embedded). At most 5,000 sentences per course (`MAX_SEMANTIC_SENTENCES`); sections past that are split the plain way.
   - **Fallback, plain splitting** (`chunk_text`): if the sentence embedding fails, or `SEMANTIC_CHUNKING = False`, splits prefer paragraph boundaries (blank lines), then sentences, then a hard cut, and consecutive chunks share up to **200 characters** of whole trailing paragraphs or sentences. A warning is logged (`tutor_rag.semantic_pieces`); the index is still built.
   - Whitespace inside a paragraph is collapsed to single spaces, so line breaks within a paragraph (code, pseudocode, poems) are lost.
4. **Drop duplicates:** a chunk whose text (lower-cased, whitespace-normalised) already appeared anywhere in the course is skipped. Repeated footers, menus and slide templates are indexed once. In the same CRAG pages, 19% of chunks were exact repeats.
5. **Cap:** at most **2,000 chunks per course** (`MAX_CHUNKS_PER_ACTIVITY`), so one course can't run up the AI bill. Chunking stops at the cap; sources after that point are not indexed.

**What gets embedded** (`embedding_input`): the chunk prefixed with where it comes from, `"<source title> — <locator>\n<text>"`. A short chunk like "It releases oxygen." still carries "Photosynthesis — slide 4". The reranker reads the same text.

**Effect:** on the pages of 200 CRAG questions, steps 1, 2 and 4 cut the chunk count from 45,383 to 19,290 (−57%), so a reindex embeds less than half as much. For the effect on retrieval, see [section 4](#measured-on-200-crag-questions-and-the-eval-set).

## 3. Indexing: `tutor_rag.index_activity`

Run by `POST /api/activities/:id/tutor/reindex` (host only).

**Identity and change detection**
- **Chunk id:** `sha256(activity | source_type | source_id | chunk_index)[:48]`, used in both D1 and Vectorize. `chunk_index` counts the chunks kept within a source.
- **Content hash:** `sha256(embedding model, source title, locator, text)`.

**Incremental:** the stored hashes are read from D1, then:
- only chunks whose id is new or whose hash changed are embedded, upserted into Vectorize and written to D1;
- ids that no longer exist are deleted from both stores;
- changing the embedding model, a title or a locator changes the hash, so those chunks are re-embedded.

Because ids are positional, adding text near the start of a long file shifts the chunks after it, and those are re-embedded too.

The response is `{chunks, embedded, unchanged, removed, skipped_files}`, with the message that new content can take up to a minute to become searchable (Vectorize indexes asynchronously).

**Where things are stored**

| Store | Holds | Why |
|---|---|---|
| **Vectorize** (`VECTORIZE`, 768-d, cosine) | vectors only, in a **namespace per course**; metadata is only `source_type` and `source_id` | A query is scoped to the course's namespace, so it can't return another course's content. No course text goes to Vectorize. |
| **D1 `tutor_chunks`** | id, course, source type/id/title, locator, chunk index, **AES-encrypted text**, content hash, timestamps | Text is encrypted like the rest of learn's course content. |

**Batches:** embeddings in batches of 50 texts, Vectorize upserts in batches of 500, deletes in batches of 50 (D1 allows at most 100 bound parameters). Indexing runs inside the request.

## 4. Retrieval: `tutor_rag.retrieve`

```python
retrieve(env, activity_id, query: str | list[str], decrypt, top_k=5, min_score=0.0,
         rerank_query=None, extra_queries=None, keywords=None, keyword_weight=0.0)
```

1. **Prepare the queries:** strip, cut each to 1,000 characters and de-duplicate. `extra_queries` that repeat a `query` are dropped. `top_k` is clamped to 1–20.
2. **Embed all queries in one call**, each prefixed with BGE v1.5's query instruction: "Represent this sentence for searching relevant passages: ". The model is trained to embed short queries that way, and passages without it.
3. **Search** Vectorize once per vector, in the course's namespace:
   - each of `query` asks for `top_k` matches, or **20** (`RERANK_CANDIDATES`) when `rerank_query` is given;
   - each of `extra_queries` asks for `top_k`.
4. **Merge:** each chunk keeps its **best cosine across all queries**; matches below `min_score` are dropped. Nothing left → `[]`.
5. **Load** from D1 and decrypt: without reranking, the best `top_k`; with reranking, every candidate (up to 20 per question query plus 5 per rewrite), in batches of 50 ids.
6. **Rerank** (with `rerank_query` and at least 2 candidates): `@cf/baai/bge-reranker-base` scores each candidate against `rerank_query`. Each candidate is sent as its embedding input (title, locator, text), cut to 1,600 characters, and the query is cut to 1,000. Candidates are sorted by that score (0–1).
   - **Why:** vector search compares two separately made embeddings. A cross-encoder reads the question and the passage together, so it can tell "about the same topic" from "answers the question".
   - **If the reranker fails** (an error, or scores missing for some passages), a warning is logged (`tutor_rag.rerank`) and the vector order is kept. The tutor keeps working.
7. **Keyword boost** (`keyword_boost`, with `rerank_query`, `keywords` and a `keyword_weight`): each candidate's score (the reranker's, or the cosine if the reranker failed) gets `keyword_weight × coverage`, where coverage is the share of the keywords found in the candidate's **text** (not its title, which every chunk of a page shares). Each keyword counts by how rare it is among the candidates (idf: `log((n+1)/(1+candidates containing it))`), so a keyword every candidate contains adds nothing. Matching ignores case, allows a plural `s`/`es`, and treats spaces, hyphens and apostrophes between words alike. Each candidate gains `keyword_coverage`.
   - **Why:** the reranker judges overall relevance and can prefer a general passage over the one that names the exact team, company or term asked about ("Rayo Vallecano", "worst performer").
8. **Return** the best `top_k`, best first.

Each hit is `{chunk_id, source_type, source_id, source_title, locator, chunk_index, text, score}`, where `score` is the cosine, plus `rerank_score` when reranked and `keyword_coverage` when boosted. That's everything needed to cite it.

**No keyword index.** Keyword (hybrid) search was prototyped and measured, then removed until its security trade-offs are settled; see [section 12](#12-proposed-hybrid-search-not-implemented). The reranker and the keyword boost store nothing: they only read candidates that vector search already found, decrypted at query time.

### What the chat searches with: `tutor_chat.find_passages`

```python
expansion = await expand_question(env, question, history)
keywords = await extract_keywords(env, question)
retrieve(env, activity_id, [question], decrypt,
         top_k=RETRIEVAL_K, min_score=MIN_RELEVANCE, extra_queries=expansion.rewrites,
         rerank_query=rerank_query(question, history, expansion),
         keywords=keywords, keyword_weight=KEYWORD_WEIGHT)
```

**Query expansion** (`expand_question`): Llama 3.1 8B (`EXPANSION_MODEL`, temperature 0, at most 200 tokens) gets the question and, in a conversation, the student's **previous question**. It first decides whether the new question **continues** the previous one: it refers back to it ("it", "its", "that", "and what about…", "can you give an example") or only makes sense with it. A question that makes sense on its own is a new topic, even on a similar subject. It replies:

```
RELATED: yes | no
1. yes: the question rewritten to stand alone, naming what it refers to; no: the question as written
2. the same question in the terms a textbook would use
3. one sentence that a passage answering the question would contain
```

The prompt includes one worked example (quicksort, "Is it stable?").

`parse_expansion` reads the decision and the three lines. It removes numbering, bullets, Markdown, quotes and the model's habit of introducing line 3 ("A passage answering this question might contain: …"), cuts each line to 300 characters and keeps at most 3. It returns `Expansion(related, standalone, rewrites)`:
- **`standalone`**, the question the reranker judges against: line 1 when the question continues the previous one (or the model didn't say). When it is a new topic, or the first question of a conversation, it is **the student's question itself**, whatever the model wrote on line 1: told to copy it, the 8B still sometimes "improved" it ("How are Huffman codes built?" → "How are variable-length prefix codes constructed?") or wrote an answer.
- **`rewrites`**, the extra searches: the three lines without copies of the question.

**What is searched:** the question and the rewrites. The previous question is never searched as such: for a follow-up, its context reaches the search through line 1 ("What is its running time?" after "What is depth-first search?" → "What is the running time of depth-first search?"). Before, the question was also searched glued to the previous one, so every later question in a conversation carried the old topic into the search, new topic or not.

**What the reranker judges against** (`rerank_query`): `standalone`. Without it (the 8B failed or gave nothing usable): the previous question plus this one in a conversation, otherwise the question.

- **Privacy:** only the student's own questions go to this model (this one, and the previous one cut to 500 characters). It never sees course text or the tutor's replies. Its output is only embedded and used for ranking, and nobody sees it.
- **Rewrites only add candidates** (`extra_queries`, top 5 each). When they shared one candidate list with the question, the answer-like third line scored higher than the question did and pushed the question's own passages out of the reranker's 20 (CRAG: 44 against 47 of 58).
- A rewrite may contain made-up facts, but it can only change which passages are found; the reranker then judges them against the question.
- **Best effort:** any failure logs a warning (`tutor_chat.expand_question`) and returns no rewrites; the question is searched as it is. `QUERY_EXPANSION = False` turns expansion off.
- **Measured** (local models): on the eval set unchanged, the right lesson first for 22 of 23 and 5 of 8 off-topic questions without a passage; its follow-up "And how much ATP does that process make?" is marked related and reranked as "How much ATP is produced in glycolysis?". On the 47 CRAG questions (single questions, plain chunking, no keyword boost) also unchanged: 36 of 47, the same questions as with the old prompt.

**Keywords** (`extract_keywords`): the same Llama 3.1 8B, in a second short call (temperature 0, at most 60 tokens), lists up to 5 keywords: the names, specific terms, numbers and dates a passage answering the question must mention. `parse_keywords` keeps a keyword only when **every word of it is in the question** (after light stemming, so "divorced" allows "divorce") and it isn't only general words (what, the, best, today…).
- **Why the filter:** asked for keywords, the model often adds the answer it believes: "Nucleus" for "Which organelle contains the cell's DNA?", "Canberra" for the capital of Australia, and wrong ones too. Boosting those would push the passages towards the model's guess. The filter removed every such guess in the experiment below, the correct ones included.
- **Privacy:** like the rewrites, the call sees only the student's question.
- **Best effort:** any failure logs a warning (`tutor_chat.extract_keywords`) and returns no keywords; the reranker's order is used. `KEYWORD_BOOST = False` turns it off.

### Measured on 200 CRAG questions and the eval set

Retrieval only, run locally at no cost. The embedding model and reranker are the same models through fastembed (the local reranker gives the same scores as Workers AI's), and the rewrites come from Llama 3.1 8B through Ollama.

| | CRAG: answer reaches the 5 passages (of 58)\* | Eval: right lesson first (of 23) | Eval: off-topic questions with no passage (of 8) |
|---|---|---|---|
| Before (vector only) | 33 | 21 | 4 |
| + query instruction | 41 | 21 | 6 |
| + reranker | **47** | 21 | 6 |
| + query expansion | 46 | **22** | 5 |
| + new chunking (everything) | 45 | **22** | 5 |
| New chunking, vector only | 43 | 21 | 5 |

\* Questions whose short gold answer appears in the indexed passages (57 with the new chunking). Each row includes the changes above it.

- **The query instruction and the reranker** account for most of the gain.
- **Query expansion** is about neutral on CRAG's self-contained web questions. It's the only change that fixed the eval set's "What is crossing over?", where the student's words don't match the lesson's, and it gives the reranker a stand-alone question for follow-ups.
- **The new chunking** is a large gain for vector search alone (+10), within noise once the reranker is on (2 gained, 3 lost), and halves indexing cost.
- **End to end** with Llama 3.3 70B, answer mode on the eval set went from 36/38 to **37/38**: all 23 answerable, 8 out-of-scope and 4 injection cases passed. The one miss is the refused-topic paraphrase. Hint-only mode passed 14 of 14 before the run's neuron budget ran out.
- **Semantic chunking, keyword boost and query decomposition** were measured later (2026-11-19) on the 47 CRAG questions of this sample whose short answer is in the pages and **not already in the question** (the earlier check counted "the chainsmokers" as found because the question names them):

  | Retrieval (today's search: question + rewrites + reranker) | Answer in the 5 passages (of 47) |
  |---|---|
  | plain chunking (before) | 36 |
  | semantic chunking, cut at the source's 95th-percentile change | 38 |
  | **semantic packing (adopted)** | **39** (+3, none lost) |
  | plain chunking + keyword boost, weight 0.1 | 36 (1 gained, 1 lost) |
  | plain chunking + **keyword boost, weight 0.25 (adopted)** | **37** (2 gained, 1 lost) |
  | plain chunking + keyword boost, weight 0.5 | 37 |
  | plain chunking + query decomposition (sub-questions as extra searches) | 36 |

  - The eval set was the same for every variant: right lesson in the 5 for 22 of 23, out-of-scope blocked 5 of 8.
  - **End to end**, the production 70B answered the questions whose retrieval changed with semantic chunking: no answer changed. On two of them (the worst Dow Jones performer, the elevation of Hawaii's capital) the answer reached the passages and the model still said it wasn't there; the third (Randall Wallace's birth date) was already answered, written "July 28, 1949" against the check's "1949-07-28".
  - The keyword boost changes the 5 passages on 31 of 47 questions for one more answer found. Both gains are within noise for 47 questions; they were adopted for their mechanism, and should be re-measured on a larger sample.
  - **Not adopted, query decomposition:** Llama 3.1 8B split 39 of 47 questions, mostly needlessly, and often guessed answers ("What is Taal Volcano?" for the most active volcano, which is Mayon). The rewrites already search both parts of a comparison, and passages found by a sub-question are still reranked against the whole question. A multi-hop question ("…Best Picture **that year**?") needs a second search after the first answer, not a split.
- **Not adopted, a reranker score floor.** On the eval set, the off-topic questions' best passages scored ≤ 0.006 and the answerable ones ≥ 0.517. A floor of 0.01 would have stopped every off-topic question before the model, but "What should I know before starting this course?" scores 0.0088 against the prerequisites. The model handles off-topic questions with `NOT_IN_COURSE` anyway.

## 5. Answering: `tutor_chat.answer`

**Settings** (`tutor_chat.py`, unless noted):

| Constant | Value | |
|---|---|---|
| `CHAT_MODEL` | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | answers and hints |
| `RETRIEVAL_K` | 5 | passages given to the model |
| `MIN_RELEVANCE` | 0.5 | cosine floor; a passage below it for every query is ignored |
| `QUERY_EXPANSION` | `True` | rewrite the question for search |
| `EXPANSION_MODEL` | `@cf/meta/llama-3.1-8b-instruct-fp8-fast` | writes the rewrites and picks the keywords |
| `KEYWORD_BOOST` | `True` | pick keywords and boost passages that contain them |
| `KEYWORD_WEIGHT` | `0.25` | added to a 0–1 reranker score at full keyword coverage |
| `MAX_KEYWORDS` | `5` | |
| `tutor_rag.SEMANTIC_CHUNKING` | `True` | split long sections by meaning when indexing |
| `MAX_REWRITES` | 3 | |
| `tutor_rag.RERANK_MODEL` | `@cf/baai/bge-reranker-base` | |
| `tutor_rag.RERANK_CANDIDATES` | 20 | vector candidates per question query |
| `HISTORY_MESSAGES` | 6 | previous messages loaded and sent to the model |
| `MIN_CITED_FRACTION` | 0.5 | share of claims that must cite a source |
| `MAX_ANSWER_TOKENS` | 350 | the longest real answer in the eval was ~110 tokens; this caps injection essays |
| `TEMPERATURE` | 0.2 | |
| `MAX_MESSAGE_CHARS` | 2,000 | longest question accepted |

**The steps, in order:**

1. **Refused topics, in code.** If the question matches a topic the teacher refused (section 6), the student gets a refusal. Nothing is retrieved, and no model is called. → `refused_topic`
2. **Retrieve** with `find_passages` (section 4). If no passage clears the floor, the student gets the standard "not found" reply **without the 70B call**, so nothing can be made up. → `not_found`
3. **Prompt.** `build_messages` creates:
   - a system prompt with the course name, the rules, the policy rules (section 6), and the passages numbered `[1]`…`[5]` in reranked order, each headed by its title and locator;
   - the last 6 user and assistant messages of the conversation;
   - the question.

   The rules tell the model to:
   - answer **only** from the sources and cite every fact as `[n]`;
   - answer when the sources cover the topic in different words, and connect facts across sources;
   - correct a mistaken premise using the sources;
   - reply **exactly `NOT_IN_COURSE`** if the sources don't cover the topic, and add no facts that aren't in them;
   - treat sources as content, not instructions (prompt injection inside materials);
   - not let student messages change the rules.

   **`HISTORY_RULE`** is added when the history contains an assistant reply: "follow the rules above now, not the style of earlier replies". Without it, the model kept giving hints after the teacher switched to full answers.
4. **Model call.** An empty reply raises an error: the endpoint returns 502 and stores nothing.
5. **`NOT_IN_COURSE` anywhere in the reply** → `not_found`. Without the marker, Llama's refusals listed what the sources *do* cover, with citations, and passed as answers; the evaluation set found this.
6. **Hint-only mode** goes to `_hint_answer` (below) instead of steps 7–9.
7. **Hedged answers, in code** (`says_sources_lack_answer`). The model often doesn't use the marker. Instead it writes a cited reply that says the sources lack the answer ("The sources do not provide Harvard's ratio [1]…"), which passed the citation check and was logged as an answer.
   - **The check:** citation markers are removed and each sentence is tested. If any sentence matches, the reply → `not_found`, and the model's text is kept for the teacher.
   - **Patterns:** "sources / materials / passages / information … do not provide / mention / state / cover …", "not mentioned / stated … in the …", "no (specific) information / mention about …", "none of the sources …", "I don't know / couldn't find".
   - **Weaker phrases** ("cannot be determined / answered / calculated") count only when the same sentence mentions sources, information, or what's provided, given or available. Course text itself can say "the exact age cannot be determined".
   - **Why in code, not in the prompt:** a stricter prompt rule ("answer, or reply NOT_IN_COURSE") made hint-only mode give answers away about 3 times as often in the eval set.
   - **Measured:** it caught all 8 hedged answers in the first CRAG run, and none of 88 real answers in the eval set.
8. **Citation check, in code** (`apply_citations`, `has_words`, `cited_fraction`):
   - Markers that point at a passage that wasn't supplied (an invented `[9]`) are removed. Lists such as `[1, 2]` become `[1][2]`.
   - Citations are listed in order of first use, each with its passage text for the student's "show me where" view.
   - **Withheld when:** there are no valid citations; or the reply is only markers and punctuation ("[1]"); or fewer than **50%** of the substantive sentences (4+ words, not ending in "?") carry a valid marker. The student gets the "not found" reply, and the model's text is kept, encrypted, for the teacher only. → `withheld`
   - A one-word answer such as "Tenet [5]." passes. Guiding questions don't count as claims, so they don't need citations.
9. Otherwise → `answered`, with the cleaned text and citations.

**Grounding is enforced in code, not trusted to the prompt.** A student can try to rewrite the prompt, but uncited text can't get past step 8.

### Hint-only mode, checked in code: `tutor_hints.py`

The prompt alone gave the answer away in about 1 hint in 8, and code can't tell whether a hint gives the answer away without knowing the answer. So in hint-only mode the prompt (`tutor_policy.HINT_ONLY_RULE`, with a worked right and wrong example) asks for two parts:

```
ANSWER: oxygen [1]                       ← never shown to the student
HINT: Source [1] explains what the light reactions split. If water is split, which gas could be released?
```

`check_hint` checks a reply:

| Problem | Test |
|---|---|
| `format` | `parse_hint_reply` can't find both an `ANSWER:` and a `HINT:` line (Markdown such as `**ANSWER:**` is accepted). |
| `not_found` | `NOT_IN_COURSE` in the reply, or `says_sources_lack_answer` on the **ANSWER part only**. A hint may hedge ("the sources don't state it directly, but…") without being taken for a refusal. |
| `leak` | `hint_leaks`: the hint repeats a **key term** of the answer, or copies a run of **6 words** from a source sentence that contains a key term. |
| `uncited` | the hint fails the step 8 citation checks. |

- **Key terms** are the answer's words of 3+ letters (or digits) that aren't stopwords, compared by a crude stem ("membranes" = "membrane", "three" = "3").
- **Exempt words:** words of the question, of the student's earlier questions in this conversation, and of the supplied passages' titles. So a right guess can be confirmed, and the hint can name a source the student sees on the citations anyway.

**What happens next** (`_hint_answer`):
1. **No problem** → `answered`, with the hint and its citations.
2. **`format`, `leak` or `uncited`** → one retry. The model gets its reply back plus a note naming the problem, marked "(Automatic check by the tutor system, not a message from the student.)". The retry is checked the same way.
3. **Still failing** → a hint built in code that only points at the source the hidden answer cited, for example "Source [1] (Lesson 3 — page 2) covers this. Read that part again: which sentence answers your question?" → `hint_replaced`. With no cited source at all → `withheld`.
4. **`not_found`** at either try → `not_found`.

**The teacher's log** keeps the model's full replies, hidden answer included, for every hint (both tries, separated by "— second try —").

**Not caught:** paraphrases ("the gas you breathe in"); yes/no answers, whose key word ("no") is too common to check, unless the hint copies the source sentence; and a student who lists several guesses ("oxygen, nitrogen or carbon?") can get the right one confirmed.

### Outcomes

| Outcome | Meaning | 70B called |
|---|---|---|
| `answered` | Cited answer (or checked hint) returned. | yes |
| `not_found` | No passage above the floor; or the model said `NOT_IN_COURSE`; or its reply said the sources lack the answer (that text is kept for the teacher). | no / yes |
| `withheld` | The model replied without enough citations, or with only citation markers; the text is kept for the teacher only. | yes |
| `refused_topic` | Matched a refused topic. | no |
| `hint_replaced` | Hint-only mode: the hint failed the check twice, so a hint built in code that only points at the source was returned. | yes (twice) |

`grounded` is true when the answer has at least one citation. Every assistant message is stored with its outcome, so the teacher's review log can filter by it.

## 6. Teacher policy: `tutor_policy.py`

**Storage:** each course's policy is JSON in `tutor_policies`. `normalize` fills in defaults for older or partial policies, and `validate_update` merges and validates only the fields a PUT sends.

| Field | Default | Enforced by |
|---|---|---|
| `enabled` | false | **code**: students get 403 while it's off. Hosts can always preview. |
| `mode` | `answer` | **prompt and code**: `hint_only` adds `HINT_ONLY_RULE` to the prompt, and the hint is checked against the hidden answer (section 5). |
| `refused_topics` | [] | **code first**, then also listed in the prompt. A topic matches when every word of 3+ letters in it (or, if it has none, every word) starts a word in the question, so "derivative" also catches "derivatives". Predictable on purpose, so teachers can tell what will be refused. At most 20 topics of 100 characters; duplicates are dropped. |
| `daily_limit` | 30 | **code**: questions per student per UTC day, counted from stored messages; 429 when reached. 0 = no limit; at most 500. Hosts are exempt. |
| `instructions` | "" | **prompt**: added as "Teacher's instructions: …". At most 1,000 characters. |

- **Wording refused topics:** each topic word must *start* a word of the question, so use the shortest form: `homework answer` catches "answer my homework", but `homework answers` doesn't, because "answer" doesn't start with "answers". Words may be in any order and apart ("Is test and set atomic?" matches `test-and-set`).
- **Instructions change style, not the checks.** They're added after the grounding rules, and the checks in section 5 run in code after the model replies. So "answer from your own knowledge too" can't get uncited text past the citation check. But an instruction that makes the model write uncited sentences (for example "explain with an everyday analogy") can get answers **withheld**. An instruction to answer in another language works, but the hedge check only knows English phrasings.

## 7. Conversations and privacy

- **History comes from the server, never the client.** `api_tutor_chat` loads the last 6 messages of `conversation_id` from `tutor_messages`, for this course and this user only. A caller can't plant fake assistant turns. An unknown conversation id, or another user's, returns 404.
- **Storage:** every question and answer is stored **encrypted** (`content_encrypted`), with the citations (without passage text), `grounded`, `outcome`, and any model text the student didn't see (`withheld_encrypted`). If storing fails, the error is reported and the student still gets the answer.
- **Students are told:** the status endpoint returns the notice "Your teacher can read your questions and the tutor's answers.", and the chat card shows it.
- **Which models see what:** course files go to `toMarkdown`; course text goes to the embedding model, the reranker and the 70B model; the rewrite and keyword calls see only the student's questions. All are Workers AI, on the same Cloudflare account. Vectorize holds only vectors.

## 8. Endpoints

| Method | Endpoint | Who | What |
|---|---|---|---|
| POST | `/api/activities/:id/tutor/reindex` | host | Sync the index; returns chunk counts and `skipped_files`. |
| GET | `/api/activities/:id/tutor/search?q=&k=` | host / enrolled | Raw retrieval for debugging (`k` ≤ 20): vector search with the query instruction only. No rewrites, no reranker, no floor, no model. |
| POST | `/api/activities/:id/tutor/chat` | host / enrolled | `{message, conversation_id?}` → `{conversation_id, answer, citations, grounded, outcome}`. |
| GET | `/api/activities/:id/tutor` | host / enrolled | Status: `configured`, `available`, `enabled`, `is_host`, `mode`, `daily_limit`, `used_today`, `remaining_today`, `notice`. |
| GET / PUT | `/api/activities/:id/tutor/policy` | host | Read the policy with its limits, or update any subset of fields. |
| GET | `/api/activities/:id/tutor/log?limit=&before=&outcome=&student=` | host | Review log, newest first: question, answer, citations, outcome, withheld text, and counts per outcome. Pages by insert sequence (`next_before`), so entries created in the same millisecond are never skipped. `limit` ≤ 100. |

- **"Enrolled"** means an active enrollment. "Host" includes admins. `:id` may be the activity id or its slug.
- **Missing bindings:** reindex, search and chat return **503** until both `AI` and `VECTORIZE` are configured. The status endpoint reports `configured: false` instead.
- **The search endpoint** checks access but not `enabled` or `daily_limit`.
- **The student card** re-reads the status endpoint when the window regains focus or the tab becomes visible (at most every 3 s), so a mode change or the tutor being turned off shows up without a reload.

## 9. Setup

```bash
npx wrangler vectorize create learn-tutor --dimensions=768 --metric=cosine
```

Then uncomment the `[[vectorize]]` block in `wrangler.toml` (deploys fail while it names a missing index); the `[ai]` binding is already there. Teachers then open the tutor settings page, turn the tutor on and click **Reindex**.

**Cost per question** (Workers AI, at its published prices), about 80–110 neurons in total:

| Call | Neurons |
|---|---|
| Llama 3.3 70B answer: ~2,000 tokens in, up to 350 out | ~60–100 |
| Llama 3.1 8B rewrites: ~200 tokens in, up to 200 out | ~3–7 |
| Llama 3.1 8B keywords: ~150 tokens in, up to 60 out | ~1–3 |
| Reranker, 20–55 passages of ≤ 1,600 characters | ~2–6 |
| Query embeddings | < 1 |

- **Hint-only mode:** a hint that fails the check costs a second 70B call.
- **Refused and over-limit questions** skip everything. Questions with no passage above the floor skip the 70B call.
- **Reindexing** costs about 6 neurons per 1,000 tokens embedded. Only changed chunks are embedded, but semantic chunking embeds every sentence of every long section, each with its neighbours (about 3 times the text), **on every reindex**. A 170-page textbook (412 chunks): roughly 650 for its chunks the first time, plus roughly 1,900 for its sentences each time (both estimated from its token count). With `SEMANTIC_CHUNKING = False` an unchanged course reindexes for free.

**After upgrading the chunking** (to the cleaned and merged sections, and again to semantic chunking), chunk texts change, so the first reindex of each course re-embeds it once.

## 10. Testing and evaluation

| What | Where |
|---|---|
| **Unit tests**, against in-memory D1, R2, Vectorize and Workers AI fakes (`tests/tutor_fakes.py`: bag-of-words embedder and reranker, scriptable chat, rewrite and keyword models). They cover: chunking (link cleaning, joining short sections, slides, duplicates, semantic splitting and its fallback), section splitting, `.pptx` parsing, incremental indexing, multi-query retrieval, the query instruction, rewrites and their failure, reranking and its fallback, keyword filtering and the boost, prompt building, citation and hedge checks, hint leak checks, outcomes, policy validation and endpoints. | `tests/test_tutor_rag.py`, `test_tutor_chat.py`, `test_tutor_hints.py`, `test_tutor_policy.py` |
| **Evaluation set:** 38 cases on a small original course, under each policy: answerable, out-of-scope, prompt injection and refused topics. | `tutor_eval/` (`README.md`), `tests/test_tutor_eval.py` |
| **CRAG** (Meta's RAG benchmark): real web pages and questions, through the real pipeline with the production models; conversion, embeddings and reranking run locally. | `tutor_eval/run_crag.py` (`CRAG.md`), `tests/test_tutor_crag.py` |

## 11. Known limitations

- **Hint-only checks match words, not meaning.** A hint that gives the answer away in other words, or a yes/no answer, gets past them. A second model call judging each hint would close more of the gap, at twice the cost.
- **Refused topics match keywords, not meaning.** "End-of-term test" is not caught by "final exam". The eval has a case that measures this.
- **Partial answers are dropped.** A reply that answers part of the question and says the sources lack the rest becomes `not_found`. That includes a correct false-premise correction followed by "the sources don't say when…" (CRAG's Hamburg question). The detector is a list of phrasings, so a new way of saying "the sources don't say" can slip through.
- **Recall:** in 200 CRAG questions, 12 of 57 answers that are in the indexed pages still don't reach the 5 passages. The reranker can only reorder what vector search found, and vectors are weak on exact names and numbers. See [section 12](#12-proposed-hybrid-search-not-implemented).
- **Line breaks inside a paragraph are lost** when chunking, so code and pseudocode reach the model as one line.
- **The 2,000-chunk cap is silent:** materials after the cap aren't indexed, and the reindex response doesn't say so.
- **Positional chunk ids:** an edit near the start of a long file re-embeds the rest of that file.
- **Link addresses aren't indexed.** The tutor can name a linked page but not give its URL, unless the URL is written out as the link's text.
- **Boilerplate that isn't repeated word for word** (a menu that differs by one item) is still indexed. Without its URLs such a chunk is short and rarely wins.
- **Retrieval adds latency:** the rewrite call, the keyword call and the reranker call each add a round trip before the answer starts.
- **Semantic chunking is paid on every reindex,** even when nothing changed, and a large course makes about one embedding call per 50 sentences (about 90 for a 170-page textbook) inside the reindex request.
- **The keyword filter drops correct terms the student didn't use.** A keyword must be in the question, so "Mayon" can't be boosted for "the most active volcano in the Philippines", by design.
- **`MIN_RELEVANCE` (0.5)** is tuned on the evaluation set, not on a large corpus.
- **Indexing runs in the request,** not in a background queue.

## 12. Proposed: hybrid search (not implemented)

**Status:** prototyped, measured, then removed for security review. Nothing in the code or the database depends on it. This section records the design, the measurements and the open security questions, so the work can be picked up later.

### Why

Vector search matches meaning, but it is weak on exact names and numbers ("Keith Sweat", "56-26", "TNL dividends"). In CRAG, about a third of the answers that were in the pages never reached the 5 passages the model sees (before the reranker). Keyword (BM25) search finds exactly those terms, and hybrid search runs both and merges the results.

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
- Keyword candidates would join the reranker's pool rather than be fused separately.

### What it measured

Retrieval only, with the same embedding model run locally, before the query instruction and the reranker existed:

| | Vector only | Hybrid |
|---|---|---|
| Eval set: right lesson ranked first (of 23 answerable) | 21 | **22** |
| Eval set: answerable question got no passage | 1 | **0** |
| Eval set: out-of-scope questions let through by keyword match alone | – | **0** |
| CRAG: answer reaches the 5 passages (12 where it's in the pages) | 8 | 8 (a different two missed) |
| CRAG: answer reaches 10 passages | 9 | **10** |
| CRAG: keyword search alone, 5 passages | – | 3 |

So hybrid helped on clean course material. On noisy web pages it helped only when more passages went to the model.

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
| **B. Rerank only:** no stored keyword index; reorder the vector candidates at query time. | **nothing new** | no: it can only reorder what vector search found | **Done** as the cross-encoder reranker (section 4): CRAG recall 41 → 47 of 58, and the keyword boost on top of it. |
| **C. Option A with a separate secret** (e.g. `TUTOR_INDEX_KEY`) | as A | yes | One more secret to manage. |
| D. FTS5 on plaintext | all chunk text, in the clear | yes | Rejected: breaks encryption at rest. |

**Suggested order**
1. **B is in place.** The answers still missed are the ones vector search doesn't find at all.
2. **If that isn't enough, use A with C,** with keyword candidates joining the reranker's pool.

Either way, re-run the eval set and CRAG (retrieval-only, locally, at no cost) and keep the out-of-scope check at 0 leaks.
