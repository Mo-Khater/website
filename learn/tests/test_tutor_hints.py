"""
Tests for hint-only mode checked in code: src/tutor_hints.py, and
tutor_chat.check_hint / the retry and replacement flow through the chat endpoint.
"""

import pytest

import tutor_chat
import tutor_hints
# Course fixture (indexed Biology 101, tutor enabled) shared with the chat tests.
from tests.test_tutor_chat import _chat, _hit, _req, env, worker  # noqa: F401

DIFFUSION = "Diffusion is passive and needs no energy from the cell. Active transport uses ATP."


# ---------------------------------------------------------------------------
# tutor_hints (pure)
# ---------------------------------------------------------------------------

class TestParseHintReply:
    def test_two_parts(self):
        reply = "ANSWER: oxygen [1]\nHINT: Source [1] explains it. Which gas?"
        assert tutor_hints.parse_hint_reply(reply) == ("oxygen [1]", "Source [1] explains it. Which gas?")

    def test_markdown_labels_and_multiline_hint(self):
        reply = "**ANSWER:** oxygen [1]\n\n**Hint:** Look at [1].\nWhich gas could it be?"
        assert tutor_hints.parse_hint_reply(reply) == ("oxygen [1]", "Look at [1].\nWhich gas could it be?")

    @pytest.mark.parametrize("reply", [
        "Plants release oxygen [1].",
        "HINT: Look at [1]. Which gas?",          # no answer to check against
        "ANSWER: oxygen [1]",                      # no hint
        "ANSWER:\nHINT: Look at [1].",
    ])
    def test_missing_part(self, reply):
        assert tutor_hints.parse_hint_reply(reply) is None


class TestStem:
    @pytest.mark.parametrize("a,b", [
        ("membrane", "membranes"), ("package", "packaging"), ("swell", "swelling"),
        ("release", "released"), ("three", "3"), ("cell", "cells"),
    ])
    def test_same_stem(self, a, b):
        assert tutor_hints.stem(a) == tutor_hints.stem(b)


class TestHintLeaks:
    def test_key_term_in_hint(self):
        assert tutor_hints.hint_leaks("oxygen [1]", "Plants release oxygen [1]. Right?", "What gas?") == ["oxygen"]

    def test_plural_and_number_words(self):
        assert tutor_hints.hint_leaks("proteins [1]", "It builds a protein [1].",
                                      "What do ribosomes make?") == ["proteins"]
        assert tutor_hints.hint_leaks("3 sodium, 2 potassium [1]", "It moves three of one kind [1].",
                                      "How many sodium and potassium ions per ATP?") == ["3"]

    def test_good_hint_does_not_leak(self):
        hint = "Source [1] explains what the light reactions split. If water is split, which gas could be released?"
        assert tutor_hints.hint_leaks("oxygen [1]", hint, "What gas do plants release?") == []

    def test_words_the_student_said_or_saw_are_not_leaks(self):
        # a right guess can be confirmed
        assert tutor_hints.hint_leaks("oxygen [1]", "Yes, oxygen is right [1].", "Is it oxygen?") == []
        # words of the question and of source titles don't count
        assert tutor_hints.hint_leaks("in the nucleus of the cell [1]", "Which part of the cell [1]?",
                                      "Where is DNA kept in a cell?") == []

    def test_stopwords_and_citations_are_not_key_terms(self):
        assert tutor_hints.key_terms("Yes, it is [1]") == {}

    def test_copied_answer_sentence(self):
        # "No" is too common to check; the copied sentence is still caught
        leaks = tutor_hints.hint_leaks("No, diffusion is passive [1]",
                                       "Source [1] says it needs no energy from the cell. So what does that mean?",
                                       "Does diffusion need energy from the cell?", [DIFFUSION])
        assert leaks == ['"needs no energy from the cell"']

    def test_copying_other_sentences_is_fine(self):
        # the copied run is from a sentence that doesn't hold the answer
        leaks = tutor_hints.hint_leaks("passive [1]", "Compare with how active transport uses ATP [1]. Which needs it?",
                                       "Is diffusion active?", [DIFFUSION])
        assert leaks == []


def test_fallback_hint_only_points_at_the_source():
    text = tutor_hints.fallback_hint(2, _hit(2, "Lesson 3", "page 2"))
    assert text.startswith("Source [2] (Lesson 3 — page 2) covers this.")
    assert text.endswith("?")
    assert tutor_chat.cited_fraction(text) == 1.0


def test_retry_note_names_the_leak_and_is_not_from_the_student():
    note = tutor_hints.retry_note("leak", ["oxygen"])
    assert "oxygen" in note and "not a message from the student" in note


# ---------------------------------------------------------------------------
# tutor_chat.check_hint
# ---------------------------------------------------------------------------

class TestCheckHint:
    HITS = [_hit(1, "Photosynthesis", "", "Water is split and oxygen is released."), _hit(2)]

    def check(self, reply, said="What gas do plants release?"):
        return tutor_chat.check_hint(reply, self.HITS, said)

    def test_good_hint(self):
        c = self.check("ANSWER: oxygen [1]\nHINT: Source [1] explains what is split. Which gas could be released?")
        assert c["problem"] is None
        assert c["hint"] == "Source [1] explains what is split. Which gas could be released?"
        assert [x["n"] for x in c["citations"]] == [1] and c["source"] == 1

    def test_leak(self):
        c = self.check("ANSWER: oxygen [1]\nHINT: Plants release oxygen [1]. Do you see why?")
        assert (c["problem"], c["leaks"], c["source"]) == ("leak", ["oxygen"], 1)

    def test_format(self):
        c = self.check("Plants release a gas, see [2]. Which one?")
        assert (c["problem"], c["source"]) == ("format", 2)

    def test_uncited_hint(self):
        c = self.check("ANSWER: oxygen [1]\nHINT: Think about what water is made of. Which gas could it be?")
        assert (c["problem"], c["source"]) == ("uncited", 1)

    def test_answer_says_sources_lack_it(self):
        c = self.check("ANSWER: The sources do not mention it [1]\nHINT: Look at [1]. What does it say?")
        assert c["problem"] == "not_found"

    def test_hedged_hint_is_not_a_refusal(self):
        # only the hidden answer decides "not found"; the hint may hedge
        c = self.check("ANSWER: oxygen [1]\nHINT: The sources don't state it directly, but [1] says what is "
                       "split. Which gas could be released?")
        assert c["problem"] is None


# ---------------------------------------------------------------------------
# Through the chat endpoint
# ---------------------------------------------------------------------------

async def _hint_policy(worker, env):
    resp = await worker._dispatch(_req(worker, env, "PUT", "/api/activities/act1/tutor/policy", "host1",
                                       {"mode": "hint_only"}), env)
    assert resp.status == 200, resp.body


def _replies(*replies):
    """A chat responder that gives *replies* in turn (the last one repeats)."""
    queue = list(replies)
    return lambda msgs: queue.pop(0) if len(queue) > 1 else queue[0]


LEAKY = "ANSWER: chlorophyll [1]\nHINT: Plants use chlorophyll [1]. Can you see how?"
GOOD = "ANSWER: chlorophyll [1]\nHINT: Source [1] names a pigment in the leaf. What is it called?"
QUESTION = "How do plants capture sunlight?"


class TestHintEndpoint:
    async def test_good_hint_needs_one_call_and_hides_the_answer(self, worker, env):
        await _hint_policy(worker, env)
        env.AI.chat_responder = _replies(GOOD)
        _, body = await _chat(worker, env, "student1", QUESTION)
        data = body["data"]
        assert data["outcome"] == "answered"
        assert data["answer"] == "Source [1] names a pigment in the leaf. What is it called?"
        assert "chlorophyll" not in data["answer"] and "ANSWER" not in data["answer"]
        assert len(env.AI.chat_calls) == 1
        # the teacher's log keeps the hidden answer
        log = (await worker._dispatch(_req(worker, env, "GET", "/api/activities/act1/tutor/log", "host1"), env))
        entry = __import__("json").loads(log.body)["data"]["entries"][0]
        assert "ANSWER: chlorophyll [1]" in entry["withheld_text"]

    async def test_leak_is_retried_with_the_problem_named(self, worker, env):
        await _hint_policy(worker, env)
        env.AI.chat_responder = _replies(LEAKY, GOOD)
        _, body = await _chat(worker, env, "student1", QUESTION)
        assert body["data"]["outcome"] == "answered"
        assert body["data"]["answer"].startswith("Source [1] names a pigment")
        retry = env.AI.chat_calls[1][1]["messages"]
        assert retry[-2] == {"role": "assistant", "content": LEAKY}
        assert retry[-1]["role"] == "user" and "chlorophyll" in retry[-1]["content"]
        assert "not a message from the student" in retry[-1]["content"]

    async def test_two_leaks_get_a_replacement_hint(self, worker, env):
        await _hint_policy(worker, env)
        env.AI.chat_responder = _replies(LEAKY)
        _, body = await _chat(worker, env, "student1", QUESTION)
        data = body["data"]
        assert data["outcome"] == "hint_replaced"
        assert data["answer"].startswith("Source [1] (Photosynthesis) covers this.")
        assert "chlorophyll" not in data["answer"]
        assert [c["source_id"] for c in data["citations"]] == ["s1"]
        assert len(env.AI.chat_calls) == 2

    async def test_unformatted_reply_without_citations_is_withheld(self, worker, env):
        await _hint_policy(worker, env)
        env.AI.chat_responder = _replies("Plants use chlorophyll. Any questions?")
        _, body = await _chat(worker, env, "student1", QUESTION)
        assert body["data"]["outcome"] == "withheld"
        assert body["data"]["answer"] == tutor_chat.NOT_FOUND_ANSWER

    async def test_a_right_guess_can_be_confirmed(self, worker, env):
        await _hint_policy(worker, env)
        env.AI.chat_responder = _replies("ANSWER: chlorophyll [1]\nHINT: Yes, chlorophyll is right [1]. "
                                         "Can you find where source [1] says so?")
        _, body = await _chat(worker, env, "student1", "Do plants capture sunlight with chlorophyll?")
        assert body["data"]["outcome"] == "answered"
        assert len(env.AI.chat_calls) == 1

    async def test_answer_mode_is_unchanged(self, worker, env):
        env.AI.chat_responder = _replies("Plants capture sunlight with chlorophyll [1].")
        _, body = await _chat(worker, env, "student1", QUESTION)
        assert body["data"]["outcome"] == "answered"
        assert body["data"]["answer"] == "Plants capture sunlight with chlorophyll [1]."
        assert len(env.AI.chat_calls) == 1
