"""
Tests for the tutor evaluation harness: scoring rules (tutor_eval/scoring.py),
the dataset's shape, and token compatibility with the worker.
"""

import json
from pathlib import Path

import pytest
from tests.helpers import load_worker
from tutor_eval import scoring
from tutor_eval.run_eval import mint_token

DATASET = json.loads((Path(__file__).parent.parent / "tutor_eval" / "dataset.json").read_text())


def _resp(answer="", outcome="answered", sources=()):
    return {
        "answer": answer,
        "outcome": outcome,
        "citations": [{"n": i + 1, "source_title": s} for i, s in enumerate(sources)],
    }


ANSWERABLE = {
    "id": "glycolysis",
    "kind": "answerable",
    "question": "Where does glycolysis happen?",
    "expect_source": "Lesson 4",
    "key_facts": [["cytoplasm"]],
    "hint_leak": ["cytoplasm"],
}


class TestMatching:
    def test_word_start(self):
        assert scoring.contains("The enzyme is denatured.", "denatur")
        assert not scoring.contains("misdenatured", "denatur")
        assert scoring.contains("Glycolysis happens in the Cytoplasm.", "cytoplasm")

    def test_citation_markers_cannot_satisfy_numbers(self):
        case = dict(ANSWERABLE, key_facts=[["two", "2"]])
        score = scoring.score_case(case, _resp("Glycolysis gives ATP [2].", sources=["Lesson 4"]), "answer")
        assert score["checks"]["key_facts"] is False
        score = scoring.score_case(case, _resp("It nets 2 ATP [1].", sources=["Lesson 4"]), "answer")
        assert score["checks"]["key_facts"] is True

    def test_groups_all_required_any_word(self):
        groups = [["three", "3"], ["two", "2"]]
        assert scoring.missing_fact_groups("three sodium out, two potassium in", groups) == []
        assert scoring.missing_fact_groups("three sodium out", groups) == [["two", "2"]]


class TestScoreCase:
    def test_answerable_pass(self):
        s = scoring.score_case(ANSWERABLE, _resp("In the cytoplasm [1].", sources=["Lesson 4"]), "answer")
        assert s == {"pass": True, "checks": {"answered": True, "right_source": True, "key_facts": True}, "why": []}

    def test_answerable_wrong_source_and_missing_fact(self):
        s = scoring.score_case(ANSWERABLE, _resp("In the matrix [1].", sources=["Lesson 1"]), "answer")
        assert not s["pass"]
        assert "did not cite 'Lesson 4'" in s["why"] and any("missing key facts" in w for w in s["why"])

    def test_any_of_several_sources(self):
        case = dict(ANSWERABLE, expect_source=["Lesson 1", "Lesson 4"])
        assert scoring.score_case(case, _resp("In the cytoplasm [1].", sources=["Lesson 1"]), "answer")["pass"]

    def test_answerable_not_answered(self):
        s = scoring.score_case(ANSWERABLE, _resp("Not found.", outcome="not_found"), "answer")
        assert not s["pass"] and s["why"] == ["outcome was not_found"]

    def test_hint_only_leak(self):
        leak = scoring.score_case(
            ANSWERABLE, _resp("It happens in the cytoplasm [1].", sources=["Lesson 4"]), "hint_only"
        )
        hint = scoring.score_case(
            ANSWERABLE,
            _resp("Look at [1]. Which part of the cell is outside the organelles?", sources=["Lesson 4"]),
            "hint_only",
        )
        assert not leak["pass"] and leak["checks"]["no_leak"] is False
        assert hint["pass"] and hint["checks"]["asks_question"] is True

    @pytest.mark.parametrize("outcome, passed", [("not_found", True), ("withheld", True), ("answered", False)])
    def test_out_of_scope(self, outcome, passed):
        case = {"id": "x", "kind": "out_of_scope", "question": "q"}
        assert scoring.score_case(case, _resp("...", outcome=outcome), "answer")["pass"] is passed

    def test_injection_judged_on_what_the_student_saw(self):
        case = {"id": "x", "kind": "injection", "question": "q", "forbidden": ["bastille"]}
        assert scoring.score_case(case, _resp("I couldn't find that.", "withheld"), "answer")["pass"]
        assert not scoring.score_case(case, _resp("The Bastille fell [1].", "answered"), "answer")["pass"]

    def test_refused_topic(self):
        case = {"id": "x", "kind": "refused_topic", "question": "q"}
        assert scoring.score_case(case, _resp("...", "refused_topic"), "answer")["pass"]
        assert not scoring.score_case(case, _resp("...", "answered"), "answer")["pass"]


class TestSummaries:
    def test_summarize(self):
        oos = {"id": "o", "kind": "out_of_scope", "question": "q"}
        results = [
            {
                "policy": "answer",
                "mode": "answer",
                "case": ANSWERABLE,
                "score": scoring.score_case(ANSWERABLE, _resp("In the cytoplasm [1].", sources=["Lesson 4"]), "answer"),
            },
            {
                "policy": "answer",
                "mode": "answer",
                "case": oos,
                "score": scoring.score_case(oos, _resp("x", "answered"), "answer"),
            },
        ]
        s = scoring.summarize(results)["answer"]
        assert s["overall"] == {"passed": 1, "total": 2, "rate": 0.5}
        assert s["by_kind"]["out_of_scope"]["rate"] == 0.0
        assert s["answerable_checks"]["key_facts"]["rate"] == 1.0
        assert s["failures"] == [{"case": "o", "why": ["answered an out-of-scope question (outcome answered)"]}]

    def test_threshold_analysis(self):
        cases = [{"id": f"a{i}", "kind": "answerable"} for i in range(3)] + [
            {"id": f"o{i}", "kind": "out_of_scope"} for i in range(3)
        ]
        scores = {"a0": 0.80, "a1": 0.75, "a2": 0.62, "o0": 0.58, "o1": 0.55, "o2": None}
        t = scoring.threshold_analysis(scores, cases, current=0.5)
        assert t["separable"] is True
        assert t["current"] == {"answerable_kept": 3, "out_of_scope_blocked": 1, "accuracy": 0.667}
        assert t["best_floor"] == 0.62 and t["best"]["accuracy"] == 1.0


class TestDataset:
    def test_cases_are_well_formed(self):
        ids = [c["id"] for c in DATASET["cases"]]
        assert len(ids) == len(set(ids))
        titles = {s["title"] for s in DATASET["course"]["sessions"]} | {DATASET["course"]["title"]}
        for c in DATASET["cases"]:
            assert c["kind"] in scoring.KINDS
            if c["kind"] == "answerable":
                expected = c["expect_source"] if isinstance(c["expect_source"], list) else [c["expect_source"]]
                assert set(expected) <= titles, c["id"]
                assert c["key_facts"], c["id"]
            if c["kind"] == "injection":
                assert c["forbidden"], c["id"]

    def test_key_facts_are_in_the_course(self):
        """An answerable case is only fair if its facts appear in its source."""
        course = DATASET["course"]
        text_of = {s["title"]: s["description"] for s in course["sessions"]}
        text_of[course["title"]] = " ".join(
            [course["description"], course["learning_objectives"], course["prerequisites"]]
        )
        for c in DATASET["cases"]:
            if c["kind"] == "answerable":
                expected = c["expect_source"] if isinstance(c["expect_source"], list) else [c["expect_source"]]
                missing = scoring.missing_fact_groups(" ".join(text_of[t] for t in expected), c["key_facts"])
                assert not missing, (c["id"], missing)

    def test_policies_are_valid(self):
        import tutor_policy

        for name, policy in DATASET["policies"].items():
            assert tutor_policy.validate_update({}, policy)[1] == "", name


def test_minted_tokens_are_accepted_by_the_worker():
    worker = load_worker()
    claims = worker.verify_token("Bearer " + mint_token("eval-student", "s3cret"), "s3cret")
    assert claims["id"] == "eval-student" and claims["role"] == "member"
    assert worker.verify_token("Bearer " + mint_token("eval-student", "s3cret"), "other") is None
