"""Scoring rules, prompt text and training targets.

Covers the official direct-answer score min(1, n/3), the multiple-choice prompt and its
cyclic option orders (Table A.2), masking of option and answer words in rationales (the
masked-rationale context of Table 4.7), the position of the answer letter in each target, the
"another question's rationale" control of Table 4.15, and the balanced loss
CE(letter) + λ·CE(rest).
"""

import pytest

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.metrics import da_eligible, da_score, normalize_answer
from cmr.eval.prompts import LETTERS, MC_INSTRUCTION, mc_prompt, rotate


@pytest.mark.parametrize(
    ("n_matches", "expected"), [(0, 0.0), (1, 1 / 3), (2, 2 / 3), (3, 1.0), (7, 1.0)]
)
def test_da_score_official_rule(n_matches: int, expected: float) -> None:
    answers = ["dog"] * n_matches + ["cat"] * (10 - n_matches)
    assert da_score("dog", answers) == pytest.approx(expected)


def test_da_official_is_exact_but_normalized_variant_forgives_format() -> None:
    answers = ["dog"] * 10
    assert da_score("A dog.", answers) == 0.0
    assert da_score("A dog.", answers, normalize=True) == 1.0


def test_normalize_answer() -> None:
    assert normalize_answer("The Two Dogs!") == "2 dogs"
    assert normalize_answer("an apple") == "apple"


def test_da_eligibility_excludes_difficult(items: list[AOKVQAItem]) -> None:
    assert not da_eligible(items[0])  # i % 7 == 0 -> difficult
    assert da_eligible(items[1])


def test_mc_prompt_layout() -> None:
    p = mc_prompt("Why?", ["x", "y"], context="ctx")
    assert p.splitlines() == [
        "Context: ctx",
        "",
        "Question: Why?",
        "Options:",
        "A. x",
        "B. y",
        MC_INSTRUCTION,
    ]
    assert "Context" not in mc_prompt("Why?", ["x"])


def test_too_many_choices() -> None:
    with pytest.raises(ValueError):
        mc_prompt("q", ["c"] * (len(LETTERS) + 1))


@pytest.mark.parametrize("k", range(4))
def test_rotate_is_consistent(k: int) -> None:
    choices = ["a", "b", "c", "d"]
    shown, original = rotate(choices, k)
    assert [choices[i] for i in original] == shown
    assert sorted(original) == [0, 1, 2, 3]
    assert shown[0] == choices[k]


def test_mask_options_blanks_every_option_whole_word() -> None:
    from cmr.eval.prompts import MASK, mask_options, mentions

    choices = ["car", "red car", "bus", "train"]
    text = "The red car is parked; a Car is not a bus. Carpet stays."
    masked = mask_options(text, choices)
    assert masked == f"The {MASK} is parked; a {MASK} is not a {MASK}. Carpet stays."
    assert not any(mentions(masked, c) for c in choices)
    assert mentions(text, "BUS") and not mentions("carpet", "car") and not mentions("x", " ")


@pytest.mark.parametrize(
    ("words", "same"),
    [
        (["cook", "cooks", "cooked", "cooking"], True),
        (["inch", "inches"], True),
        (["rain", "raining", "rained"], True),
        (["race", "racing", "races"], True),
        (["run", "running"], True),
        (["glass", "glasses"], True),
        (["city", "cities"], True),
        (["banana", "bananas"], True),
        (["car", "carpet"], False),
        (["bus", "busy"], False),
    ],
)
def test_stem_unifies_regular_variants_only(words: list[str], same: bool) -> None:
    from cmr.eval.prompts import stem

    assert (len({stem(w) for w in words}) == 1) is same


def test_stem_masking_catches_review_examples_and_leaves_no_residual() -> None:
    from cmr.eval.prompts import MASK, leaks, mask_options

    cases = [  # (rationale, choices, direct answers), paraphrased A-OKVQA rationales
        ("The boy is jumping 16 inches high.", ["16 inch", "2 feet", "1 mile", "3 cm"], []),
        ("Bananas are old. The fruit is a banana.", ["banana", "apple", "pear", "kiwi"], []),
        ("The appliance is for cooking.", ["cook", "wash", "freeze", "dry"], []),
        ("It was raining earlier.", ["rain", "snow", "sun", "fog"], []),
        ("They are racing a chariot.", ["chariot racing", "golf", "polo", "darts"], []),
        (
            "He sells hot dogs on the street.",
            ["food", "toys", "art", "cars"],
            ["hotdog", "hot dogs"],
        ),
    ]
    for text, choices, das in cases:
        masked = mask_options(text, choices, das)
        assert MASK in masked
        assert not leaks(masked, [*choices, *das]), masked
        assert leaks(text, [*choices, *das])
    # multi-word option still becomes ONE blank when it occurs verbatim
    assert mask_options("a red car and a car", ["red car", "car"]) == f"a {MASK} and a {MASK}"
    # stopwords of an option are not masked on their own
    assert mask_options("the man in the hat", ["the hat", "a dog"]) == f"the man in {MASK}"


def test_answer_char_span_locates_the_letter_in_every_format() -> None:
    import random

    from cmr.train.targets import LETTER_FORMATS, TargetFormat, answer_char_span, target_for

    it = AOKVQAItem(
        "q", 1, "Which?", ("w", "x", "y", "z"), 2, (), False,
        ("Answer: A is wrong here.",), "train",
    )  # fmt: skip
    for fmt in LETTER_FORMATS:  # direct_answer has no letter (tests/test_direct_answer_arm.py)
        t = target_for(it, fmt, random.Random(0))
        a, b = answer_char_span(t, fmt)
        assert t[a:b] == "C", (fmt, t)
    with pytest.raises(ValueError):
        answer_char_span("no letter first", TargetFormat.ANSWER_ONLY)
    with pytest.raises(ValueError):
        answer_char_span("Because: x\nAnswer: ?", TargetFormat.RATIONALE_THEN_ANSWER)


def test_mismatched_rationale_donor_has_another_image_and_no_answer_leak() -> None:
    import random
    from dataclasses import replace

    from cmr.eval.prompts import answer_strings, join_rationales, leaks
    from cmr.train.targets import (
        INSTRUCTIONS,
        TargetFormat,
        mismatch_rationales,
        prompt_for,
        target_for,
    )

    animals = ("cat", "dog", "bird", "fish")

    def item(i: int, image_id: int, gold: int) -> AOKVQAItem:
        return AOKVQAItem(
            f"q{i:02d}", image_id, f"Which animal {i}?", animals, gold, (animals[gold],) * 3,
            False, (f"A {animals[gold]} sits in picture {i}.",), "train",
        )  # fmt: skip

    # pairs share an image; items with the same gold would leak each other's answer
    items = [item(i, i // 2, i % 4) for i in range(16)]
    mm = TargetFormat.ANSWER_THEN_MISMATCHED_RATIONALE
    out = mismatch_rationales(items, seed=3)
    donor_of = {it.rationales: it for it in items}
    for it, m in zip(items, out, strict=True):
        donor = donor_of[m.rationales]
        assert donor.image_id != it.image_id  # never the item's (or its image's) rationale
        assert not leaks(join_rationales(m.rationales), answer_strings(it))
        assert (m.question_id, m.image_id, m.choices, m.correct_choice_idx) == (
            it.question_id, it.image_id, it.choices, it.correct_choice_idx,
        )  # fmt: skip
        t = target_for(m, mm, random.Random(0))
        assert t[0] == LETTERS[it.correct_choice_idx or 0]  # answer unchanged
        assert t.split("\nBecause: ", 1)[1] in m.rationales
        assert prompt_for(m, mm) == prompt_for(it, TargetFormat.ANSWER_THEN_RATIONALE)
    # deterministic, independent of input order, and tied to the (training) seed
    assert out == mismatch_rationales(items, seed=3)
    assert sorted(mismatch_rationales(items[::-1], seed=3), key=lambda m: m.question_id) == out
    assert out != mismatch_rationales(items, seed=4)
    assert INSTRUCTIONS[mm] == INSTRUCTIONS[TargetFormat.ANSWER_THEN_RATIONALE]
    # no valid donor: the only other-image item names the same answer, or one image only
    with pytest.raises(ValueError, match="no donor"):
        mismatch_rationales([item(0, 0, 1), item(1, 1, 1)], seed=0)
    with pytest.raises(ValueError, match="no donor"):
        mismatch_rationales([item(0, 0, 1), item(1, 0, 2)], seed=0)
    # an item without rationales gets no donor rationale, so target_for raises as for the
    # A->R target; it is never a donor either
    trio = [item(0, 0, 1), replace(item(1, 1, 2), rationales=()), item(2, 2, 3)]
    a, b, c = mismatch_rationales(trio, seed=0)
    assert b == trio[1]
    assert (a.rationales, c.rationales) == (trio[2].rationales, trio[0].rationales)
    for fmt in (mm, TargetFormat.ANSWER_THEN_RATIONALE):
        with pytest.raises(ValueError, match="no rationales"):
            target_for(b, fmt, random.Random(0))


def test_nested_objective_weights_answer_and_rest() -> None:
    torch = pytest.importorskip("torch")
    from cmr.models.bridge import LossParts
    from cmr.train.bridge_sft import TrainConfig, objective

    # per-example CE over (all target tokens, the answer letter, the other tokens)
    parts = LossParts(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0]))
    assert objective(parts, None).item() == pytest.approx(3.0)  # token mean over the whole target
    assert objective(parts, 0.0).item() == pytest.approx(2.0)  # letter only
    assert objective(parts, 0.5).item() == pytest.approx(2.0 + 0.5 * 6.0)
    with pytest.raises(ValueError):
        TrainConfig(rationale_weight=-1.0)


def test_checkpoint_rule_is_validated() -> None:
    pytest.importorskip("torch")
    from cmr.train.bridge_sft import TrainConfig

    assert TrainConfig().select == "dev_acc"  # the default
    with pytest.raises(ValueError):
        TrainConfig(select="best")  # type: ignore[arg-type]
