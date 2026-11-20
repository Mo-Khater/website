"""
CRAG scoring for the tutor (pure functions; the judge is passed in).

Follows CRAG's local_evaluation.py (https://github.com/facebookresearch/CRAG):
each answer is "correct" (+1), "missing" (0, the system said it doesn't know)
or "hallucination" (-1, anything else). The CRAG score is
(correct - hallucination) / n, so a wrong answer costs more than no answer.

For each gold answer (answer + alt_ans), in order:
  1. a prediction containing "i don't know" is missing;
  2. an exact (case-insensitive) match is correct;
  3. if either side contains "invalid" (CRAG's false-premise questions have
     the gold answer "invalid question"), it's correct only if both do;
     no judge call;
  4. otherwise the LLM judge decides (score 1 = correct).

The tutor never says "I don't know": any outcome other than "answered"
(not_found, withheld) is what it does instead, so it maps to "I don't know".

Deviations from CRAG, recorded in the report:
  - Predictions are not cut to 75 Llama-2 tokens; [n] citation markers are removed instead.
  - A judge error counts as a hallucination (CRAG retries the API and then fails).
"""

import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

IDK = "I don't know"
OUTCOMES = ("correct", "missing", "hallucination")

FALSE_PREMISE_GOLD = "The question's premise is false, so it cannot be answered as asked."

_MARKER_RE = re.compile(r"\s*\[\d{1,2}\]")


def prediction_from_response(response: Dict[str, Any]) -> str:
    """What the tutor said, as a CRAG prediction."""
    if response.get("outcome") != "answered":
        return IDK
    return _MARKER_RE.sub("", response.get("answer") or "").strip() or IDK


def gold_answers(example: Dict[str, Any]) -> List[str]:
    golds = [str(example.get("answer") or "")] + [str(a) for a in (example.get("alt_ans") or [])]
    return [g for g in dict.fromkeys(g.strip() for g in golds) if g]


def judge_messages(instructions: str, examples: str, query: str, ground_truth: str,
                   prediction: str) -> List[Dict[str, str]]:
    """The judge conversation, built exactly as CRAG's local_evaluation.py does."""
    return [
        {"role": "system", "content": instructions + "\n" + examples},
        {"role": "user", "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"},
    ]


def parse_judge_reply(text: str) -> Dict[str, Any]:
    """CRAG's parse_response: the last {...} block's "score" (0/1) and
    "explanation"; score -1 if it can't be read."""
    text = text if isinstance(text, str) else json.dumps(text)
    matches = re.findall(r"\{[^{}]*\}", text or "")
    for block in reversed(matches):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        score = data.get("score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = -1
        return {"score": score if score in (0, 1) else -1, "explanation": str(data.get("explanation", ""))}
    return {"score": -1, "explanation": "could not parse the judge's reply"}


Judge = Callable[[str, str, str], Dict[str, Any]]


def score_prediction(query: str, golds: List[str], prediction: str, judge: Judge) -> Dict[str, Any]:
    """Score one prediction the CRAG way. Returns {"outcome", "how", "judge": [...]}."""
    p = (prediction or "").strip().lower()
    if "i don't know" in p:
        return {"outcome": "missing", "how": "i don't know", "judge": []}
    calls = []
    for gold in golds:
        g = gold.strip().lower()
        if p == g:
            return {"outcome": "correct", "how": "exact match", "judge": calls}
        if "invalid" in g or "invalid" in p:
            if "invalid" in g and "invalid" in p:
                return {"outcome": "correct", "how": "both say invalid", "judge": calls}
            continue
        verdict = judge(query, gold, prediction)
        calls.append({"ground_truth": gold, **verdict})
        if verdict.get("score") == 1:
            return {"outcome": "correct", "how": "judge", "judge": calls}
    how = "judge error" if calls and all(c.get("score") == -1 for c in calls) else (
        "judge said no" if calls else "invalid rule")
    return {"outcome": "hallucination", "how": how, "judge": calls}


def _rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}
    return {
        "n": n, **counts,
        "score": round((counts["correct"] - counts["hallucination"]) / n, 3) if n else None,
        "accuracy": round(counts["correct"] / n, 3) if n else None,
        "hallucination_rate": round(counts["hallucination"] / n, 3) if n else None,
        "missing_rate": round(counts["missing"] / n, 3) if n else None,
    }


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """*rows*: {"outcome", "question_type", "domain", ...}. Overall and per-group CRAG rates."""
    rows = list(rows)
    out = {"overall": _rates(rows)}
    for key in ("question_type", "domain", "static_or_dynamic"):
        groups = sorted({r.get(key) for r in rows if r.get(key)})
        out[f"by_{key}"] = {g: _rates([r for r in rows if r.get(key) == g]) for g in groups}
    return out


def answer_in_pages(golds: List[str], page_texts: List[str], max_len: int = 40) -> Optional[bool]:
    """Diagnostic: does any short gold answer appear verbatim in the pages' text?

    Separates "the pages never had it" from "the tutor missed it". None when
    every gold answer is long prose (which wouldn't appear verbatim anyway)
    or the question is a false premise.
    """
    short = [g.lower() for g in golds if len(g) <= max_len and "invalid" not in g.lower()]
    if not short:
        return None
    text = " ".join(page_texts).lower()
    return any(g in text for g in short)
