"""Tests for the CRAG benchmark adapter's pure parts (tutor_eval/crag_*.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tutor_eval"))

import crag_sample  # noqa: E402
import crag_scoring  # noqa: E402


def _judge(score=0):
    calls = []

    def judge(query, ground_truth, prediction):
        calls.append(ground_truth)
        return {"score": score, "explanation": "x"}
    judge.calls = calls
    return judge


class TestPrediction:
    def test_answered_drops_citation_markers(self):
        assert crag_scoring.prediction_from_response(
            {"outcome": "answered", "answer": "Tenet won [1][2]. It was 2021 [3]."}) == "Tenet won. It was 2021."

    def test_anything_but_answered_is_i_dont_know(self):
        for outcome in ("not_found", "withheld", "refused_topic", "error"):
            assert crag_scoring.prediction_from_response({"outcome": outcome, "answer": "Tenet [1]."}) == "I don't know"

    def test_gold_answers_deduplicated(self):
        assert crag_scoring.gold_answers({"answer": "Tenet ", "alt_ans": ["tenet", "Tenet", ""]}) == ["Tenet", "tenet"]


class TestScorePrediction:
    def test_i_dont_know_is_missing_without_judge(self):
        judge = _judge(1)
        assert crag_scoring.score_prediction("q", ["x"], "I don't know", judge)["outcome"] == "missing"
        assert judge.calls == []

    def test_exact_match_skips_judge(self):
        judge = _judge(0)
        assert crag_scoring.score_prediction("q", ["Tenet"], "tenet", judge)["outcome"] == "correct"
        assert judge.calls == []

    def test_invalid_rule(self):
        judge = _judge(1)
        assert crag_scoring.score_prediction("q", ["invalid question"], "This is an invalid question.",
                                             judge)["outcome"] == "correct"
        # only one side says invalid: wrong, and the judge is never asked
        assert crag_scoring.score_prediction("q", ["invalid question"], "Hamburg, in 1871.",
                                             judge)["outcome"] == "hallucination"
        assert judge.calls == []

    def test_judge_tries_each_gold_answer(self):
        judge = _judge(0)
        result = crag_scoring.score_prediction("q", ["a", "b"], "c", judge)
        assert result["outcome"] == "hallucination" and judge.calls == ["a", "b"]
        assert crag_scoring.score_prediction("q", ["a"], "about a", _judge(1))["outcome"] == "correct"

    def test_judge_error_counts_as_wrong(self):
        result = crag_scoring.score_prediction("q", ["a"], "c", _judge(-1))
        assert (result["outcome"], result["how"]) == ("hallucination", "judge error")


class TestParseJudgeReply:
    def test_last_json_block(self):
        assert crag_scoring.parse_judge_reply('Thinking {"score": 0} final: {"score": 1, "explanation": "ok"}') == \
            {"score": 1, "explanation": "ok"}

    def test_score_as_string(self):
        assert crag_scoring.parse_judge_reply('{"score": "1"}')["score"] == 1

    def test_unreadable(self):
        assert crag_scoring.parse_judge_reply("no json here")["score"] == -1
        assert crag_scoring.parse_judge_reply('{"score": 7}')["score"] == -1

    def test_judge_messages_match_crag(self):
        msgs = crag_scoring.judge_messages("I", "E", "q?", "gt", "pred")
        assert msgs == [{"role": "system", "content": "I\nE"},
                        {"role": "user", "content": "Question: q?\n Ground truth: gt\n Prediction: pred\n"}]


class TestSummary:
    def test_score_is_correct_minus_hallucinated(self):
        rows = [{"outcome": o, "question_type": t, "domain": "movie"} for o, t in
                [("correct", "simple"), ("correct", "simple"), ("missing", "set"), ("hallucination", "set")]]
        s = crag_scoring.summarize(rows)
        assert s["overall"]["score"] == 0.25
        assert s["overall"]["accuracy"] == 0.5 and s["overall"]["missing_rate"] == 0.25
        assert s["by_question_type"]["set"]["score"] == -0.5

    def test_answer_in_pages(self):
        assert crag_scoring.answer_in_pages(["Tenet"], ["Best effects: TENET (2020)"]) is True
        assert crag_scoring.answer_in_pages(["Tenet"], ["Dune"]) is False
        assert crag_scoring.answer_in_pages(["invalid question"], ["x"]) is None
        assert crag_scoring.answer_in_pages(["x" * 60], ["x" * 60]) is None


class TestSample:
    def _ex(self, qtype, domain, n=0):
        return {"question_type": qtype, "domain": domain, "id": f"{qtype}-{domain}-{n}"}

    def test_balanced_over_types_then_domains(self):
        pool = [self._ex(t, d, n) for t in ("simple", "comparison") for d in ("finance", "movie") for n in range(3)]
        picked = crag_sample.balanced_pick(pool, 4)
        assert [e["question_type"] for e in picked] == ["simple", "comparison", "simple", "comparison"]
        # each type starts on a different domain, so a short slice mixes domains
        assert {e["domain"] for e in picked[:2]} == {"finance", "movie"}

    def test_takes_what_exists(self):
        assert len(crag_sample.balanced_pick([self._ex("simple", "open")], 5)) == 1

    def test_usable(self):
        ok = {"query": "q", "answer": "a", "search_results": [{"page_result": "<p>x</p>"}]}
        assert crag_sample.usable(ok)
        assert not crag_sample.usable({**ok, "search_results": [{"page_result": "  "}]})
        assert not crag_sample.usable({**ok, "answer": ""})


class TestBudget:
    def test_workers_ai_call_refused_before_it_is_sent(self, tmp_path):
        import run_crag
        ai = run_crag.LocalAI("workers-ai", "http://127.0.0.1:9", "unused", tmp_path / "cache.json",
                              budget_neurons=50)
        # ~4,000 characters in + 350 tokens out is ~100 neurons: over the 50 left, so no request is made
        import pytest
        with pytest.raises(run_crag.BudgetExceeded):
            ai.chat([{"role": "user", "content": "x" * 4000}], max_tokens=350)
        assert ai.stats["chat_calls"] == 0

    def test_ollama_has_no_budget(self, tmp_path):
        import run_crag
        ai = run_crag.LocalAI("ollama", "http://127.0.0.1:9", "m", tmp_path / "cache.json")
        assert ai.neurons_left() == float("inf")
