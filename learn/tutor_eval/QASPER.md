# Qasper benchmark for the AI tutor

[Qasper](https://allenai.org/data/qasper) (Allen AI) has questions about NLP research papers, written by people who read only the title and abstract and answered by experts who read the whole paper. It tests the tutor on what students actually upload: **PDFs**, with real questions about them.

## Qasper in brief

**Each question comes with**
- the paper (an arXiv PDF);
- one or more expert answers, each of one kind: **extractive** (spans of the paper), **free-form**, **yes/no**, or **unanswerable** (the paper doesn't say);
- the evidence: the paragraphs, tables or figures each expert used.

## How it was run

- **Sample:** 120 questions from the test split, on 46 papers. 69 extractive, 17 yes/no, 14 free-form, 20 unanswerable.
- **Courses:** 12 "reading list" courses of 3–4 papers each, so the tutor has to find the right paper as well as the right passage. Each question is asked as `In the paper "<title>": <question>`.
- **Extraction:** Tesseract OCR of every page at 300 dpi, one `### Page N` section per page.
- **Pipeline:** the worker's own code in-process, as in [CRAG.md](CRAG.md):
  - reindex: semantic chunking, BGE embeddings;
  - `find_passages`: rewrites and keywords from Llama 3.1 8B, reranker, keyword boost;
  - chat: Llama 3.3 70B on Workers AI.
  The embeddings and the reranker run locally (the same models), and so does the 8B (Ollama).
- **Status:** 112 of 120 questions ran. The last 8, all unanswerable, hit Cloudflare's daily free limit.

## Results (112 questions)

| | Count |
|---|---|
| **Correct** | **75** |
| Not found | 14 |
| Wrong | 23 |

The correct count includes 11 "not found" answers that the author judged correct on manual review.

**The 23 wrong**
- **Hedges (11):** the tutor said the paper "does not explicitly state / specify / mention" something. The hedge check (`says_sources_lack_answer`) doesn't know these phrasings yet, so the reply stays `answered`. 3 of them are on unanswerable questions, where hedging is the right behaviour.
- **Unanswerable questions answered (2 more):** besides those 3 hedges.
- **Genuinely wrong (10):** a wrong "yes", the wrong dataset, or a partial list. Example: "What metric is considered?" was answered "F-measure"; the experts say precision, recall and F-measure, and that sentence was not among the 5 passages.

**Search**
- The right paper is in the 5 passages for 100 of 100 answerable questions. That's easy, because the question names the paper.
- The experts' highlighted evidence is in the 5 passages for at least 39 of 100. That's a lower bound: a word-match check misses OCR variants.

**What hurts most**
- **Tables:** OCR flattens them, so numbers lose their row and column labels. Questions about scores ("what was their F1?") often end as not found.
- **Locators:** OCR text has no headings, so every citation is just `page N`.

**Time and cost**

| Step | Time |
|---|---|
| OCR | 490 pages in 268 s on 16 cores, about 0.55 s per page |
| Indexing | 1.5–3 min per course (local CPU embeddings) |
| Chat request | median 5.9 s, 90th percentile 9.7 s |

## Data licence

Qasper is CC BY 4.0. The questions and papers are downloaded, not committed.
