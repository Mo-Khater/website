# CRAG Benchmark — AI Tutor Evaluation

We evaluated the AI tutor's RAG pipeline using [Meta's CRAG benchmark](https://github.com/facebookresearch/CRAG), which tests whether a system can answer questions accurately using real-world web pages while avoiding hallucinations.

## Evaluation Setup

- 40 balanced questions covering 8 question types and 5 domains.
- Each question included up to 5 real-world source pages.
- Used the actual tutor worker code and production Llama 3.3 70B model.
- 38 questions completed before the API neuron budget was exhausted.

## Final Results (November 19, 2026)

| Result | Count | Percentage |
|---|---:|---:|
| Correct | 30 | 78.9% |
| Wrong | 8 | 21.1% |
| **Total** | **38** | **100%** |

The 30 correct outcomes consisted of:
- **11 correct answers:** Matched the gold answer.
- **19 appropriate abstentions/hedged responses:** The tutor did not know the answer, and the information was not available in the sources.

The 8 wrong outcomes consisted of:
- **7 incorrect answers:** The tutor provided an answer that did not match the gold answer.
- **1 incorrect abstention:** The tutor declined to answer even though the answer was available in the sources.

## Retrieval Improvements — 200-Question Experiment

We evaluated retrieval separately to determine whether the correct answer appeared in the top 5 retrieved passages.

| Configuration | Answer found in top 5 |
|---|---:|
| Baseline | 33/58 (57%) |
| + BGE query instruction | 41/58 |
| + Cross-encoder reranker | **47/58 (81%)** |
| + Query expansion | 46/58 |
| New chunking + all improvements | 45/57 (79%) |

Additional improvement:
- Reduced indexed chunks by **57%**, from 45,383 to 19,290.
- BGE query instructions and cross-encoder reranking delivered the biggest retrieval gains.
- Query expansion was mostly neutral in this benchmark, although it may help with student wording and follow-up questions.

## Main Takeaways

1. **Better retrieval:** Retrieval coverage improved from 57% to 81% with query instructions and reranking.
2. **Appropriate abstention:** 19 responses were correctly treated as abstentions because the answers were unavailable in the sources.
3. **Remaining challenges:** Errors involved missing information, calculations, ambiguous questions, and inconsistencies between gold answers and source pages.
4. **Next priority:** Improve retrieval coverage and handling of complex questions, calculations, and ambiguous evidence.

## Overall Conclusion

The retrieval improvements substantially increased the likelihood of finding relevant information. The final end-to-end evaluation achieved **30 correct outcomes out of 38 questions (78.9%)**, including appropriate abstentions. Further work should focus on improving retrieval coverage and ensuring that the tutor answers only when sufficient supporting evidence exists.
