"""MC prompt template (LLaVA-1.5 style, single-letter answer), option rotation and
the rationale masking behind the RATIONALE_MASKED context. Pure Python, no model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cmr.data.aokvqa import AOKVQAItem

# 10 letters cover MMMU-Pro's up-to-10-way items.
LETTERS = "ABCDEFGHIJ"
MC_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def mc_prompt(question: str, choices: Sequence[str], context: str | None = None) -> str:
    """Build the MC prompt for one item, with an optional leading ``Context:`` line.

    ``choices`` are in display order (already rotated); at most 10. The image marker is
    added later by the model's chat template.

    >>> print(mc_prompt("Why is the man holding the umbrella?",
    ...                 ["it is raining", "to block the sun", "he is selling it",
    ...                  "it is a costume"]))
    Question: Why is the man holding the umbrella?
    Options:
    A. it is raining
    B. to block the sun
    C. he is selling it
    D. it is a costume
    Answer with the option's letter from the given choices directly.
    """
    if len(choices) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} choices supported")
    lines: list[str] = []
    if context:
        lines += [f"Context: {context.strip()}", ""]
    lines.append(f"Question: {question.strip()}")
    lines.append("Options:")
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    lines.append(MC_INSTRUCTION)
    return "\n".join(lines)


def rotate(choices: Sequence[str], k: int) -> tuple[list[str], list[int]]:
    """Cyclically shift options by ``k`` (mod n) to cancel letter-position bias.

    Returns:
        ``(rotated_choices, original_index)``: position p shows original option (p + k) mod n.

    >>> rotate(["it is raining", "to block the sun", "he is selling it", "it is a costume"], 1)
    (['to block the sun', 'he is selling it', 'it is a costume', 'it is raining'], [1, 2, 3, 0])
    """
    n = len(choices)
    original_index = [(p + k) % n for p in range(n)]
    return [choices[i] for i in original_index], original_index


def join_rationales(rationales: Sequence[str]) -> str:
    """Strip rationales and join the non-empty ones with single spaces.

    >>> join_rationales([" It is wet. ", "  ", "Rain."])
    'It is wet. Rain.'
    """
    return " ".join(r.strip() for r in rationales if r.strip())


# ----------------------------------------------------------------------------- masking

MASK = "___"
MASK_VERSION = 2  # 1: whole option strings only; 2: also direct answers and shared-stem words

# Function words never masked on their own; content words of multi-word options are.
STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or but not no is are was were be been
    being am it its this that these those there their they them he she his her him you
    your we our i my me as if so than then too very can could will would should may might
    do does did done have has had into onto over under out up down off about some any all
    each other more most one ones what which who whom whose when where why how""".split()  # noqa: SIM905
)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def stem(word: str) -> str:
    """Tiny deterministic suffix-stripping stemmer; lower-cases first.

    Drops possessive 's, then one suffix (ies/ied -> y, ing, ed, es, s; stem keeps >= 3
    letters), a silent final 'e' and a doubled final consonant. Irregular forms (ran,
    children) are not unified.

    >>> [stem(w) for w in ["cooking", "racing", "inches", "glasses", "running", "ran"]]
    ['cook', 'rac', 'inch', 'glass', 'run', 'ran']
    """
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3:
        return w

    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) + len(rep) >= 3:
            if suf == "s" and w[-2] in "sui":
                continue
            w = w[: len(w) - len(suf)] + rep
            break

    # Then drop a silent final 'e' and undouble a final consonant.
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeioulsz":
        w = w[:-1]
    return w


def content_stems(phrase: str) -> set[str]:
    """Stems of the non-stopword words of ``phrase``.

    >>> sorted(content_stems("the chariot racing"))
    ['chariot', 'rac']
    """
    return {stem(t) for t in _WORD.findall(phrase) if t.lower() not in STOPWORDS}


def _option_pattern(option: str) -> re.Pattern[str]:
    """Case-insensitive regex for ``option`` (stripped) as a whole word or phrase."""
    return re.compile(r"(?<!\w)" + re.escape(option.strip()) + r"(?!\w)", re.IGNORECASE)


def mentions(text: str, option: str) -> bool:
    """Whole-word, case-insensitive match of ``option`` in ``text``; empty never matches.

    >>> mentions("It is raining.", "raining"), mentions("rainy day", "rain")
    (True, False)
    """
    return bool(option.strip()) and _option_pattern(option).search(text) is not None


def answer_strings(item: AOKVQAItem, min_votes: int = 3) -> list[str]:
    """``[gold choice, *direct answers given by >= min_votes annotators]`` (first-seen order).

    The default 3 votes is the count that earns full DA credit.
    """
    votes: dict[str, int] = {}
    for a in item.direct_answers:
        votes[a] = votes.get(a, 0) + 1
    return [item.correct_choice, *(a for a, v in votes.items() if v >= min_votes)]


def leaks(text: str, answers: Sequence[str]) -> bool:
    """True if ``text`` contains any of ``answers`` verbatim or one of their content stems.

    Same matcher as masking; sets a context's ``context_leak`` flag.

    >>> leaks("Umbrellas keep the rain off.", ["it is raining"])
    True
    >>> leaks("The sky is grey.", ["it is raining"])
    False
    """
    stems = set().union(*(content_stems(a) for a in answers)) if answers else set()
    if any(stem(t) in stems for t in _WORD.findall(text)):
        return True
    return any(mentions(text, a) for a in answers)


def mask_options(text: str, choices: Sequence[str], extra: Sequence[str] = ()) -> str:
    """Replace every option and ``extra`` answer in ``text`` with :data:`MASK`.

    Pass 1 masks whole strings, longest first; pass 2 masks any word sharing a content stem
    with them, so :func:`leaks` then finds nothing. All options are masked, since a single
    blank would reveal the gold one.

    >>> mask_options("The pavement is wet and people open umbrellas to stay dry in the rain.",
    ...              ["it is raining", "to block the sun", "he is selling it",
    ...               "it is a costume"])
    'The pavement is wet and people open umbrellas to stay dry in the ___.'
    >>> mask_options("He is racing a chariot.", ["chariot racing", "boxing"])
    'He is ___ a ___.'
    """
    phrases = [c for c in (*choices, *extra) if c.strip()]

    # Pass 1: whole option strings, longest first.
    for phrase in sorted(set(phrases), key=len, reverse=True):
        text = _option_pattern(phrase).sub(MASK, text)

    # Pass 2: any word whose stem is a content stem of an option or answer.
    stems = set().union(*(content_stems(c) for c in phrases)) if phrases else set()
    if not stems:
        return text
    return _WORD.sub(lambda m: MASK if stem(m.group(0)) in stems else m.group(0), text)


def letter_token_ids(tokenizer: Any, n_options: int) -> list[list[int]]:
    """Sorted single-token ids of each letter, with and without a leading space.

    Raises:
        ValueError: neither form of some letter is a single token.
    """
    out: list[list[int]] = []
    for letter in LETTERS[:n_options]:
        ids: set[int] = set()
        for variant in (letter, f" {letter}"):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
        if not ids:
            raise ValueError(f"letter {letter!r} is not a single token for this tokenizer")
        out.append(sorted(ids))
    return out
