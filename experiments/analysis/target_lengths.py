"""Length of the A→R training target in tokens, and the answer letter's share of its loss.

Report Section 3.4 and Figure 3.3. Builds the A→R target as in training
(cmr.train.targets.target_for, letter fixed to "B") for --n-train sampled train items, counts
tokens plus the end-of-turn token, and prints the letter's share 1/T (AO: T = 2).

    python experiments/analysis/target_lengths.py [--tokenizer Qwen/Qwen2-0.5B-Instruct]
        [--aokvqa-dir data/raw/aokvqa] [--n-train 2000] [--seed 0] [--no-save]

Needs transformers; ``--tokenizer whitespace`` counts words instead (checks the sampling only).
Input: aokvqa_v1p0_train.json. Output: printed, and saved to results/analysis/target_lengths.txt
unless --no-save. The committed target_lengths.txt holds the original measurement (mean 18.3
tokens, letter share 0.063) in an earlier one-line format; a re-run rewrites it in this
script's format.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.data.aokvqa import AOKVQAItem, load_aokvqa  # noqa: E402
from cmr.train.targets import TargetFormat, target_for  # noqa: E402

LETTER_B = 1  # option index shown as letter "B"


def ar_target_lengths(
    items: Sequence[AOKVQAItem], encode: Callable[[str], list[int]], n: int, seed: int
) -> list[int]:
    """Token length (+1 for end-of-turn) of the A→R target of ``n`` items sampled with ``seed``.

    One generator draws the sample and then each item's rationale.
    """
    rng = random.Random(seed)
    sample = rng.sample(list(items), n)
    fmt = TargetFormat.ANSWER_THEN_RATIONALE
    targets = [target_for(replace(it, correct_choice_idx=LETTER_B), fmt, rng) for it in sample]
    return [len(encode(t)) + 1 for t in targets]


def load_encoder(name: str) -> Callable[[str], list[int]]:
    """``text -> token ids`` without special tokens; 'whitespace' splits on whitespace."""
    if name == "whitespace":
        return lambda s: list(range(len(s.split())))
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "transformers is not installed: pip install -e '.[models]', "
            "or pass --tokenizer whitespace to check the sampling only"
        ) from None
    tok = AutoTokenizer.from_pretrained(name)
    return lambda s: tok.encode(s, add_special_tokens=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--aokvqa-dir", type=Path, default=ROOT / "data/raw/aokvqa")
    ap.add_argument("--n-train", type=int, default=2000, help="items sampled from train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args(argv)

    encode = load_encoder(args.tokenizer)
    train = load_aokvqa(args.aokvqa_dir, "train")
    lengths = ar_target_lengths(train, encode, args.n_train, args.seed)
    mean_t = statistics.mean(lengths)
    text = "\n".join(
        [
            f"tokenizer {args.tokenizer}; {args.n_train} of {len(train)} train items, "
            f"seed {args.seed}",
            f"A->R target tokens (incl. end-of-turn): mean {mean_t:.1f}, "
            f"median {statistics.median(lengths):g}, min {min(lengths)}, max {max(lengths)}",
            f"letter's share of the per-example loss: mean of 1/T "
            f"{statistics.mean(1 / t for t in lengths):.3f}, 1/(mean T) {1 / mean_t:.3f} "
            f"(AO: T = 2, share 0.5)",
        ]
    )
    print(text)
    if not args.no_save:
        out = ROOT / "results/analysis/target_lengths.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\nsaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
