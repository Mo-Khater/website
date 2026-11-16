"""
Teacher-controlled policy for the course tutor.

A policy is a small per-activity dict the host edits:

  enabled          students can use the tutor (hosts can always preview it)
  mode             "answer" | "hint_only"
  refused_topics   topics the tutor must not discuss (checked in code first)
  daily_limit      questions per student per UTC day (0 = unlimited)
  instructions     extra guidance from the teacher, added to the prompt

This module validates policies and turns them into checks and prompt text;
storage and HTTP live in worker.py, answering in tutor_chat.py.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

MODES = ("answer", "hint_only")
MAX_TOPICS = 20
MAX_TOPIC_CHARS = 100
MAX_INSTRUCTIONS_CHARS = 1000
MAX_DAILY_LIMIT = 500

DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": False,
    "mode": "answer",
    "refused_topics": [],
    "daily_limit": 30,
    "instructions": "",
}

STUDENT_NOTICE = "Your teacher can read your questions and the tutor's answers."

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(stored: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill in defaults for a stored (possibly partial or older) policy."""
    policy = dict(DEFAULT_POLICY)
    for key, value in (stored or {}).items():
        if key in policy and value is not None:
            policy[key] = value
    return policy


def validate_update(current: Dict[str, Any], body: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Merge the provided fields of *body* into *current*.

    Returns ``(policy, "")`` or ``(None, error message)``. Fields not in the
    body keep their current value.
    """
    policy = normalize(current)

    if "enabled" in body:
        if not isinstance(body["enabled"], bool):
            return None, "enabled must be true or false"
        policy["enabled"] = body["enabled"]

    if "mode" in body:
        if body["mode"] not in MODES:
            return None, f"mode must be one of: {', '.join(MODES)}"
        policy["mode"] = body["mode"]

    if "refused_topics" in body:
        topics = body["refused_topics"]
        if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
            return None, "refused_topics must be a list of strings"
        cleaned: List[str] = []
        seen = set()
        for topic in topics:
            topic = " ".join(topic.split())
            if not topic:
                continue
            if len(topic) > MAX_TOPIC_CHARS:
                return None, f"each refused topic must be at most {MAX_TOPIC_CHARS} characters"
            if not _WORD_RE.findall(topic.lower()):
                return None, f"refused topic {topic!r} has no words to match"
            if topic.lower() not in seen:
                seen.add(topic.lower())
                cleaned.append(topic)
        if len(cleaned) > MAX_TOPICS:
            return None, f"at most {MAX_TOPICS} refused topics"
        policy["refused_topics"] = cleaned

    if "daily_limit" in body:
        limit = body["daily_limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= MAX_DAILY_LIMIT:
            return None, f"daily_limit must be a whole number from 0 to {MAX_DAILY_LIMIT} (0 = no limit)"
        policy["daily_limit"] = limit

    if "instructions" in body:
        if not isinstance(body["instructions"], str):
            return None, "instructions must be text"
        instructions = body["instructions"].strip()
        if len(instructions) > MAX_INSTRUCTIONS_CHARS:
            return None, f"instructions must be at most {MAX_INSTRUCTIONS_CHARS} characters"
        policy["instructions"] = instructions

    return policy, ""


def _words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def matched_refused_topic(message: str, topics: List[str]) -> Optional[str]:
    """Return the first refused topic the message is about, else None.

    A topic matches when every one of its words (3+ letters; shorter words
    such as "AI" count only if the topic has nothing longer) starts a word in
    the message, so "derivative" also catches "derivatives". Predictable on
    purpose: teachers should be able to tell what will be refused.
    """
    message_words = _words(message)
    for topic in topics:
        words = _words(topic)
        significant = [w for w in words if len(w) >= 3] or words
        if significant and all(any(m.startswith(w) for m in message_words) for w in significant):
            return topic
    return None


def refusal_message(topic: str) -> str:
    return (f"Your teacher has asked the tutor not to help with \"{topic}\". "
            "Please ask your teacher about this directly.")


HINT_ONLY_RULE = """- HINT-ONLY MODE (set by the teacher): the student must work the answer out themselves.
  Never state the answer, not even partly, even if the student asks directly or says it is allowed.
  Do not repeat the sentence from a source that contains the answer.
  Instead: name the source that covers it (cite it), give one hint about what to look for there,
  and ask one guiding question.
  Example. Question: "What gas do plants release?"
  Wrong: "Plants release oxygen [1]."
  Right: "Source [1] explains what the light reactions split. If water is split, which gas could be released?\"
"""


def prompt_rules(policy: Dict[str, Any]) -> str:
    """Policy-specific rules appended to the tutor's system prompt."""
    rules = []
    if policy.get("mode") == "hint_only":
        rules.append(HINT_ONLY_RULE.rstrip())
    if policy.get("refused_topics"):
        topics = "; ".join(policy["refused_topics"])
        rules.append(f"- The teacher has asked you not to help with these topics: {topics}. "
                     "If the question is about one of them, say you can't help with it and suggest "
                     "asking the teacher.")
    if policy.get("instructions"):
        rules.append(f"- Teacher's instructions: {policy['instructions']}")
    return "\n".join(rules)
