"""
Hint-only mode, checked in code (the prompt side is tutor_policy.HINT_ONLY_RULE).

With the prompt alone, about 1 hint in 8 gave the answer away in the eval set,
and code can't tell whether a hint gives the answer away without knowing the
answer. So in hint-only mode the model writes two parts:

  ANSWER: oxygen [1]                                   never shown to the student
  HINT: Source [1] explains what the light reactions split. If water is split,
        which gas could be released?                   what the student sees

and the hint is checked against the answer:

  key terms    content words of ANSWER that the hint repeats ("oxygen",
               "oxygens"). Words the student already used are exempt, so a
               right guess can be confirmed, and so are the words of the
               source titles, which the student sees on the citations anyway.
  copied text  a run of COPY_RUN_WORDS words copied from a source sentence
               that contains the answer.

tutor_chat retries once with the problem named. If the retry fails too, the
hint is replaced with one built here that only points at the source.

Not caught: paraphrases ("the gas you breathe in"), and yes/no answers whose
only key word ("no") is too common to check. A student who lists several
guesses ("is it oxygen, nitrogen or carbon?") can get the right one confirmed.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

COPY_RUN_WORDS = 6

_CITATION_RE = re.compile(r"\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_FIELD_RE = re.compile(r"^[\s*_#>-]*(ANSWER|HINT)[\s*_]*:[\s*_]*", re.IGNORECASE | re.MULTILINE)

STOPWORDS = frozenset("""
a about after all also an and any are as at be because been before both but by can could
did do does each for from had has have here how if in inside into is it its may might more
most much must no nor not of on only or other our out over per same should so some such than
that the their them then there these they this those through to too under up very was were
what when where which while who whom why will with within would yes you your
answer answers called known part source sources use used uses
""".split())

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "eleven": "11", "twelve": "12"}


def _words(text: str) -> List[str]:
    return _TOKEN_RE.findall(_CITATION_RE.sub(" ", text or "").lower())


def stem(word: str) -> str:
    """Crude stem, the same on both sides of a comparison: "membranes",
    "membrane" -> "membran"; "three" -> "3"."""
    word = NUMBER_WORDS.get(word, word)
    for suffix in ("ing", "es", "ed", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def parse_hint_reply(reply: str) -> Optional[Tuple[str, str]]:
    """``(answer, hint)`` from a two-part reply, or None if either part is missing."""
    parts = _FIELD_RE.split(reply or "")
    fields: Dict[str, str] = {}
    for label, text in zip(parts[1::2], parts[2::2]):
        fields.setdefault(label.upper(), text.strip())
    if not fields.get("ANSWER") or not fields.get("HINT"):
        return None
    return fields["ANSWER"], fields["HINT"]


def key_terms(answer: str, exempt: Iterable[str] = ()) -> Dict[str, str]:
    """The answer's content words, as {stem: word}, minus *exempt* stems."""
    exempt = set(exempt)
    terms: Dict[str, str] = {}
    for word in _words(answer):
        if word in STOPWORDS or (len(word) < 3 and not word.isdigit()):
            continue
        s = stem(word)
        if s not in exempt:
            terms.setdefault(s, word)
    return terms


def copied_run(hint: str, sentences: Iterable[str], exempt: Iterable[str] = ()) -> str:
    """The first COPY_RUN_WORDS-word run of *hint* found in one of *sentences*, or ""."""
    exempt = set(exempt)
    runs = set()
    for sentence in sentences:
        words = [stem(w) for w in _words(sentence)]
        runs.update(tuple(words[i:i + COPY_RUN_WORDS]) for i in range(len(words) - COPY_RUN_WORDS + 1))
    words = _words(hint)
    for i in range(len(words) - COPY_RUN_WORDS + 1):
        run = words[i:i + COPY_RUN_WORDS]
        stems = tuple(stem(w) for w in run)
        if stems in runs and not set(stems) <= exempt:
            return " ".join(run)
    return ""


def hint_leaks(answer: str, hint: str, said: str, passages: Iterable[str] = ()) -> List[str]:
    """What in *hint* gives *answer* away: the key terms it repeats, and a run
    of words it copies from a passage sentence that contains a key term.

    *said* is text the student has seen or written (their questions, the
    source titles); its words are not leaks.
    """
    exempt = {stem(w) for w in _words(said)}
    terms = key_terms(answer, exempt)
    hint_stems = {stem(w) for w in _words(hint)}
    leaks = [word for s, word in terms.items() if s in hint_stems]
    answer_sentences = [s for p in passages for s in _SENTENCE_RE.split(p or "")
                        if terms and {stem(w) for w in _words(s)} & terms.keys()]
    copied = copied_run(hint, answer_sentences, exempt)
    if copied:
        leaks.append(f'"{copied}"')
    return leaks


RETRY_NOTES = {
    "leak": ("Your HINT gives the answer away: it says {leaks}. Write both lines again. "
             "The HINT must not use those words or copy the source's sentence; "
             "point to the source and ask a guiding question instead."),
    "format": ("Reply in the required format: one line that starts with ANSWER: "
               "and one that starts with HINT:."),
    "uncited": "The HINT must cite the source it points to, like [1]. Write both lines again.",
}


def retry_note(problem: str, leaks: Iterable[str] = ()) -> str:
    """Feedback for the model's second try; marked so it isn't taken as the student's words."""
    return ("(Automatic check by the tutor system, not a message from the student.) "
            + RETRY_NOTES[problem].format(leaks=", ".join(leaks)))


def fallback_hint(n: int, hit: Dict) -> str:
    """A hint that can't give anything away: it only points at source [n]."""
    where = " — ".join(p for p in (hit.get("source_title"), hit.get("locator")) if p)
    return (f"Source [{n}] ({where}) covers this. "
            "Read that part again: which sentence answers your question?")
