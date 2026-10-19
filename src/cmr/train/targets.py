"""Training targets (report Section 3.3): the prompt and the text the model is trained to write.

--target                          report  target text (loss on it and on the appended EOS token)
answer_only                       AO      "B"
answer_then_rationale             A->R    "B\\nBecause: <r>"
rationale_then_answer             R->A    "Because: <r>\\nAnswer: B"
answer_then_mismatched_rationale          "B\\nBecause: <r'>"  (r' from another question and
                                          image; control run of Table 4.15)
direct_answer                     DA      "rain"  (most frequent direct answer; no options
                                          shown; Section 4.11)

<r> is one of the question's three human rationales, drawn at random each time the question
is used.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from enum import Enum

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.direct_answer import SHORT_ANSWER_INSTRUCTION, short_answer_prompt
from cmr.eval.prompts import (
    LETTERS,
    MC_INSTRUCTION,
    answer_strings,
    join_rationales,
    leaks,
    mc_prompt,
)


class TargetFormat(str, Enum):
    """The training targets; the value is the ``--target`` string of cmr-stage-b."""

    ANSWER_ONLY = "answer_only"
    ANSWER_THEN_RATIONALE = "answer_then_rationale"
    RATIONALE_THEN_ANSWER = "rationale_then_answer"
    ANSWER_THEN_MISMATCHED_RATIONALE = "answer_then_mismatched_rationale"
    DIRECT_ANSWER = "direct_answer"


# targets that contain an option letter
LETTER_FORMATS = tuple(f for f in TargetFormat if f is not TargetFormat.DIRECT_ANSWER)

# last line of each target's prompt (replaces MC_INSTRUCTION)
INSTRUCTIONS = {
    TargetFormat.ANSWER_ONLY: MC_INSTRUCTION,
    TargetFormat.ANSWER_THEN_RATIONALE: (
        "Answer with the option's letter from the given choices, then explain briefly."
    ),
    TargetFormat.RATIONALE_THEN_ANSWER: (
        "Explain your reasoning briefly, then answer with the option's letter."
    ),
}
INSTRUCTIONS[TargetFormat.ANSWER_THEN_MISMATCHED_RATIONALE] = INSTRUCTIONS[
    TargetFormat.ANSWER_THEN_RATIONALE
]  # same prompt: the two targets differ only in which rationale is trained on
INSTRUCTIONS[TargetFormat.DIRECT_ANSWER] = SHORT_ANSWER_INSTRUCTION
ANSWER_CUE = "Answer:"  # R->A targets end with "Answer: <letter>"


def prompt_for(item: AOKVQAItem, fmt: TargetFormat) -> str:
    """The multiple-choice prompt with its last line replaced by ``fmt``'s instruction.

    For direct_answer: the DA evaluation prompt (question + short-answer instruction, no
    options), so training and evaluation prompts match.
    """
    if fmt is TargetFormat.DIRECT_ANSWER:
        return short_answer_prompt(item.question)
    base = mc_prompt(item.question, item.choices)
    return base.replace(MC_INSTRUCTION, INSTRUCTIONS[fmt])


def direct_answer_target(item: AOKVQAItem) -> str:
    """The direct_answer target: the most frequent direct answer (ties: first given).

    Falls back to the correct option's text when there are no direct answers.

    >>> it = AOKVQAItem(question_id="q", image_id=1, question="Why?",
    ...     choices=("it is raining", "sun", "selling", "costume"), correct_choice_idx=0,
    ...     direct_answers=("rain", "raining", "raining", "rain", "wet"),
    ...     difficult_direct_answer=False, rationales=(), split="train")
    >>> direct_answer_target(it)  # rain and raining 2 votes each: rain was given first
    'rain'
    >>> direct_answer_target(replace(it, direct_answers=()))  # no direct answers
    'it is raining'

    Raises:
        ValueError: no direct answer and no label.
    """
    votes = Counter(a.strip() for a in item.direct_answers if a.strip())
    if votes:  # a Counter keeps first-occurrence order; max returns the first maximum
        return max(votes, key=votes.__getitem__)
    if item.correct_choice_idx is None:
        raise ValueError(f"{item.question_id}: no direct answers and no label")
    return item.choices[item.correct_choice_idx].strip()


def shuffle_options(item: AOKVQAItem, key: str) -> AOKVQAItem:
    """Copy of ``item`` with its options shuffled and the gold index remapped.

    Used for option-shuffled training (report Section 3.6), so that no letter position can
    be learned as a shortcut; ``key`` seeds the permutation (reproducible).

    >>> it = AOKVQAItem(question_id="q", image_id=1, question="Why?",
    ...     choices=("a", "b", "c", "d"), correct_choice_idx=0, direct_answers=(),
    ...     difficult_direct_answer=False, rationales=(), split="train")
    >>> s = shuffle_options(it, "0:0:0")
    >>> sorted(s.choices) == ["a", "b", "c", "d"] and s.choices[s.correct_choice_idx] == "a"
    True
    >>> shuffle_options(it, "0:0:0") == s
    True
    """
    order = list(range(len(item.choices)))
    random.Random(key).shuffle(order)
    gold = None if item.correct_choice_idx is None else order.index(item.correct_choice_idx)
    return replace(item, choices=tuple(item.choices[i] for i in order), correct_choice_idx=gold)


def mismatch_rationales(items: Sequence[AOKVQAItem], seed: int) -> list[AOKVQAItem]:
    """Copies of ``items`` carrying rationales from another question about another image.

    The donor of item i is the next item, in a ring ordered by sha256(seed:question_id), that
    has a different image and whose rationales mention neither i's correct option nor a
    direct answer given by at least three annotators, verbatim or by a content-word stem
    (:func:`cmr.eval.prompts.leaks`). Items without rationales are left unchanged and are
    never donors.

    Args:
        seed: the training seed (``TrainConfig.seed``).

    Raises:
        ValueError: an item with rationales has no valid donor.
    """
    # ring order is independent of input order
    n = len(items)
    ring = sorted(
        range(n), key=lambda i: hashlib.sha256(f"{seed}:{items[i].question_id}".encode()).digest()
    )
    pos = {i: p for p, i in enumerate(ring)}

    out = []
    for i, it in enumerate(items):
        if not it.rationales:  # as for A->R, target_for raises on such an item
            out.append(it)
            continue
        answers = answer_strings(it)
        for step in range(1, n):
            donor = items[ring[(pos[i] + step) % n]]
            if (
                donor.image_id != it.image_id
                and donor.rationales
                and not leaks(join_rationales(donor.rationales), answers)
            ):
                out.append(replace(it, rationales=donor.rationales))
                break
        else:
            raise ValueError(
                f"{it.question_id}: no donor with another image whose rationale does not "
                "name its answer"
            )
    return out


def target_for(item: AOKVQAItem, fmt: TargetFormat, rng: random.Random) -> str:
    """Supervised target text for one item, with one randomly sampled rationale.

    For answer_then_mismatched_rationale, pass items from :func:`mismatch_rationales`.

    Args:
        rng: rationale stream; one ``rng.choice`` per call, none for answer_only and
            direct_answer.

    Raises:
        ValueError: no label, or no rationales for a rationale format.

    Example (``random.Random(1)`` picks the first rationale):

        >>> import random
        >>> from cmr.data.aokvqa import AOKVQAItem
        >>> item = AOKVQAItem(
        ...     question_id="ex1-umbrella", image_id=1,
        ...     question="Why is the man holding the umbrella?",
        ...     choices=("it is raining", "to block the sun", "he is selling it",
        ...              "it is a costume"),
        ...     correct_choice_idx=0, direct_answers=(), difficult_direct_answer=False,
        ...     rationales=(
        ...         "The pavement is wet and people open umbrellas to stay dry in the rain.",
        ...         "It is a grey, wet day, so he is keeping the rain off.",
        ...         "Umbrellas are used when it rains and the ground is wet.",
        ...     ),
        ...     split="toy",
        ... )
        >>> target_for(item, TargetFormat.ANSWER_ONLY, random.Random(1))
        'A'
        >>> print(target_for(item, TargetFormat.ANSWER_THEN_RATIONALE, random.Random(1)))
        A
        Because: The pavement is wet and people open umbrellas to stay dry in the rain.
        >>> ra = target_for(item, TargetFormat.RATIONALE_THEN_ANSWER, random.Random(1))
        >>> print(ra)
        Because: The pavement is wet and people open umbrellas to stay dry in the rain.
        Answer: A
        >>> answer_char_span(ra, TargetFormat.RATIONALE_THEN_ANSWER)
        (88, 89)
        >>> prompt_for(item, TargetFormat.ANSWER_THEN_RATIONALE).splitlines()[-1]
        "Answer with the option's letter from the given choices, then explain briefly."

    direct_answer (no direct answers here, so the correct option's text):

        >>> target_for(item, TargetFormat.DIRECT_ANSWER, random.Random(1))
        'it is raining'
        >>> print(prompt_for(item, TargetFormat.DIRECT_ANSWER))
        Question: Why is the man holding the umbrella?
        Answer the question using a single word or phrase.
    """
    if fmt is TargetFormat.DIRECT_ANSWER:  # no RNG draw
        return direct_answer_target(item)
    if item.correct_choice_idx is None:
        raise ValueError(f"{item.question_id}: no label")
    letter = LETTERS[item.correct_choice_idx]
    if fmt is TargetFormat.ANSWER_ONLY:
        return letter
    if not item.rationales:
        raise ValueError(f"{item.question_id}: no rationales for {fmt.value}")
    rationale = rng.choice(item.rationales).strip()
    if fmt in (TargetFormat.ANSWER_THEN_RATIONALE, TargetFormat.ANSWER_THEN_MISMATCHED_RATIONALE):
        return f"{letter}\nBecause: {rationale}"
    return f"Because: {rationale}\n{ANSWER_CUE} {letter}"


def answer_char_span(target: str, fmt: TargetFormat) -> tuple[int, int]:
    """Character span ``[start, end)`` of the answer letter inside a :func:`target_for` target.

    R->A targets use the last "Answer:" cue, since a rationale may contain one.

    Raises:
        ValueError: no letter where ``fmt`` puts it, or ``fmt`` is direct_answer.

    Example:

        >>> answer_char_span("A\\nBecause: The pavement is wet.",
        ...                  TargetFormat.ANSWER_THEN_RATIONALE)
        (0, 1)
        >>> answer_char_span("Because: wet. Answer: no.\\nAnswer: A",
        ...                  TargetFormat.RATIONALE_THEN_ANSWER)
        (34, 35)
    """
    if fmt is TargetFormat.DIRECT_ANSWER:
        raise ValueError(f"direct_answer targets have no option letter: {target!r}")
    if fmt is TargetFormat.RATIONALE_THEN_ANSWER:
        i = target.rindex(ANSWER_CUE) + len(ANSWER_CUE)
        while i < len(target) and target[i] == " ":
            i += 1
    else:
        i = 0
    if i >= len(target) or target[i] not in LETTERS:
        raise ValueError(f"no answer letter where expected in {target!r}")
    return i, i + 1
