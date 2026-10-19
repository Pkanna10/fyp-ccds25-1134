"""How often the A-OKVQA human rationales give the answer away, before and after masking.

Report Sections 3.2 and 4.6 (leak / no-leak split) and Table 4.7 (masked-rationale and caption
contexts). The masker's own matcher finds nothing after masking by construction, so the residual
is judged by independent detectors defined here (suffix-strip, exact-word, 5-letter prefix).
Percentages are over the labelled val questions; a question counts at most once per row.

    python experiments/analysis/rationale_leakage.py [--aokvqa-dir data/raw/aokvqa]
        [--captions data/raw/coco/annotations/captions_val2017.json] [--no-save]

Inputs: aokvqa_v1p0_val.json; optionally the COCO val2017 captions (caption rows).
Output: printed, and saved to results/analysis/rationale_leakage.txt unless --no-save.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.data.aokvqa import AOKVQAItem, load_aokvqa  # noqa: E402
from cmr.eval.prompts import (  # noqa: E402
    MASK,
    answer_strings,
    join_rationales,
    leaks,
    mask_options,
    mentions,
)

N_EXAMPLES = 8  # prefix-detector hits printed as examples (first ones in val order)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


# ----------------------------------------------------------------------------- detectors


def words(text: str) -> list[str]:
    """Lower-cased words of ``text``; every character outside [a-z0-9] separates words.

    >>> words("It's a Dog-house!")
    ['it', 's', 'a', 'dog', 'house']
    """
    return _NON_ALNUM.sub(" ", text.lower()).split()


def suffix_strip(word: str) -> str:
    """Crude stem: drop the first of ies/es/s/ing/ed that ends a word of more than 4 letters.

    >>> [suffix_strip(w) for w in ["boxes", "cities", "running", "parked", "cats"]]
    ['box', 'cit', 'runn', 'park', 'cats']
    """
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(word) > 4 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def option_word_survives(masked: str, option: str, stem: bool = True) -> bool:
    """Does a word (> 2 letters) of ``option`` occur in ``masked``? ``stem``: suffix-strip both.

    >>> option_word_survives("He parked the trucks.", "truck")
    True
    >>> option_word_survives("He parked the trucks.", "truck", stem=False)
    False
    """
    f = suffix_strip if stem else str
    targets = {f(w) for w in words(option) if len(w) > 2}
    return bool(targets & {f(w) for w in words(masked)})


def prefix_survives(masked: str, answers: Iterable[str], k: int = 5) -> bool:
    """Does a word of ``masked`` share its first ``k`` letters with a >= k-letter answer word?

    >>> prefix_survives("Food from Japanese cuisine", ["japan"])
    True
    >>> prefix_survives("A cat", ["japan"])
    False
    """
    prefixes = {w[:k] for a in answers for w in words(a) if len(w) >= k}
    return any(w[:k] in prefixes for w in words(masked) if len(w) >= k)


def popular_direct_answers(item: AOKVQAItem, min_votes: int = 3) -> list[str]:
    """Direct answers given by >= ``min_votes`` of the 10 annotators that are not an option."""
    votes = collections.Counter(item.direct_answers)
    return [a for a, v in votes.items() if v >= min_votes and a not in item.choices]


# ----------------------------------------------------------------------------- counting


def load_captions(path: Path) -> dict[int, list[str]]:
    """COCO captions JSON -> {image id: its captions (stripped), in file order}."""
    with path.open() as f:
        annotations = json.load(f)["annotations"]
    by_image: dict[int, list[str]] = collections.defaultdict(list)
    for a in annotations:
        by_image[int(a["image_id"])].append(str(a["caption"]).strip())
    return by_image


def item_flags(it: AOKVQAItem, captions: dict[int, list[str]] | None) -> dict[str, bool]:
    """Every row's yes/no answer for one labelled question, keyed by the row label."""
    gold, answers = it.correct_choice, answer_strings(it)  # answers: correct + >=3-vote DAs
    rationale = join_rationales(it.rationales)
    m = mask_options(rationale, it.choices, it.direct_answers)  # the rationale_masked context
    wrong = [c for j, c in enumerate(it.choices) if j != it.correct_choice_idx]
    flags = {
        # Same as the rationale_leak flag stored in every prediction record.
        "raw: correct option verbatim in one of the rationales": any(
            mentions(r, gold) for r in it.rationales
        ),
        "raw: correct option verbatim in the joined rationales": mentions(rationale, gold),
        "raw: content-word stem of the correct option (masker's matcher)": leaks(rationale, [gold]),
        "masked: has at least one blank": MASK in m,
        "masked: correct-option word, suffix-strip detector": option_word_survives(m, gold),
        "masked: correct-option word, exact-word detector": option_word_survives(
            m, gold, stem=False
        ),
        "masked: wrong-option word, suffix-strip detector": any(
            option_word_survives(m, c) for c in wrong
        ),
        "masked: correct option, 5-letter prefix": prefix_survives(m, [gold]),
        "masked: correct option or >=3-vote answer, 5-letter prefix": prefix_survives(m, answers),
        "masked: >=3-vote direct answer verbatim": any(
            mentions(m, a) for a in popular_direct_answers(it)
        ),
        "masked: masker's own matcher (0 by construction)": leaks(m, answers),
    }
    if captions is not None:
        cap = " ".join(captions.get(it.image_id, []))
        flags["caption: correct option verbatim"] = bool(cap) and mentions(cap, gold)
        flags["caption: stem of correct option or >=3-vote answer"] = bool(cap) and leaks(
            cap, answers
        )
    return flags


def report(val: list[AOKVQAItem], captions: dict[int, list[str]] | None) -> str:
    """The printed report, as one string."""
    n = len(val)
    rows: dict[str, int] = {}
    share = 0.0  # blanks per whitespace word of the unmasked rationale, summed over items
    stem_hits: list[tuple[str, str]] = []
    prefix_hits: list[tuple[str, str]] = []
    for it in val:
        for name, flag in item_flags(it, captions).items():
            rows[name] = rows.get(name, 0) + flag
        rationale = join_rationales(it.rationales)
        m = mask_options(rationale, it.choices, it.direct_answers)
        share += m.count(MASK) / len(rationale.split()) if rationale.split() else 0.0
        if option_word_survives(m, it.correct_choice):
            stem_hits.append((it.correct_choice, m))
        if prefix_survives(m, answer_strings(it)):
            prefix_hits.append((it.correct_choice, m))

    leak = rows["raw: correct option verbatim in one of the rationales"]
    out = [f"A-OKVQA val, labelled questions: n = {n}"]
    out.append(f"split used in Section 4.6: leak = {leak} questions, no leak = {n - leak}")
    out.append("")
    for name, v in rows.items():
        out.append(f"  {name:66s} {v:5d}  {100 * v / n:5.1f}%")
    label = "masked: blanks per word of the rationale (mean)"
    out.append(f"  {label:66s}        {100 * share / n:5.1f}%")
    if captions is None:
        out.append("  (captions file not found: caption rows skipped)")
    out += [
        "",
        "all masked rationales where the suffix-strip detector finds a correct-option word:",
    ]
    out += [f"  correct: {g!r}\n    masked: {m[:200]}" for g, m in stem_hits]
    out += ["", f"first {N_EXAMPLES} hits of the 5-letter-prefix detector (correct or >=3-vote):"]
    out += [f"  correct: {g!r}\n    masked: {m[:200]}" for g, m in prefix_hits[:N_EXAMPLES]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--aokvqa-dir", type=Path, default=ROOT / "data/raw/aokvqa")
    ap.add_argument(
        "--captions", type=Path, default=ROOT / "data/raw/coco/annotations/captions_val2017.json"
    )
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args(argv)

    val = [it for it in load_aokvqa(args.aokvqa_dir, "val") if it.has_labels]
    captions = load_captions(args.captions) if args.captions.is_file() else None
    text = report(val, captions)
    print(text)
    if not args.no_save:
        out = ROOT / "results/analysis/rationale_leakage.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\nsaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
