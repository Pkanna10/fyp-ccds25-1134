"""Per-item A-OKVQA direct-answer (DA) scoring: soft accuracy and which items count.

DA follows the official rule, ``min(1, #humans who gave pred / 3)``, skipping
``difficult_direct_answer`` items; an optional VQA-style normaliser is also provided.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from cmr.data.aokvqa import AOKVQAItem

_ARTICLES = {"a", "an", "the"}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}  # fmt: skip
_PUNCT = re.compile(r"[^\w\s']")


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation (not apostrophes) and articles, map none/zero..ten to digits.

    >>> normalize_answer("The Two dogs!")
    '2 dogs'
    """
    tokens = _PUNCT.sub(" ", text.lower()).split()
    tokens = [_NUMBERS.get(t, t) for t in tokens if t not in _ARTICLES]
    return " ".join(tokens)


def da_score(pred: str, direct_answers: Sequence[str], normalize: bool = False) -> float:
    """DA accuracy ``min(1, matches / 3)``; ``normalize=False`` is the official metric.

    ``normalize=True`` passes ``pred`` and every answer through :func:`normalize_answer`.

    >>> answers = ["dog", "dog"] + ["cat"] * 8
    >>> da_score("Dog.", answers), round(da_score("Dog.", answers, normalize=True), 3)
    (0.0, 0.667)
    """
    if normalize:
        pred = normalize_answer(pred)
        direct_answers = [normalize_answer(a) for a in direct_answers]
    matches = sum(pred == a for a in direct_answers)
    return min(1.0, matches / 3.0)


def da_eligible(item: AOKVQAItem) -> bool:
    """True if the item has direct answers and is not marked difficult (official rule)."""
    return bool(item.direct_answers) and not item.difficult_direct_answer
