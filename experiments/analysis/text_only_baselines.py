"""Text-only baselines of the frozen 0.5B LLM: question + options, and options alone.

Report Sections 4.5 and 5.4. No image, untrained bridge theta_0, fp32, all four option orders.

  score    score val with and without the question text (needs torch and the model weights);
           writes <out>/{question_options,options_only}/predictions.jsonl (resumable).
             python experiments/analysis/text_only_baselines.py score --device mps
  analyze  rotation-averaged and rotation-0 accuracy, paired item-bootstrap 95% intervals.
             python experiments/analysis/text_only_baselines.py analyze \\
                 [--question-options DIR] [--options-only DIR] [--aokvqa-dir DIR] [--no-save]

Inputs:  aokvqa_v1p0_val.json; results/runs/laptop/text_only_baselines/*/predictions.jsonl
         (release asset; scored on the laptop GPU, see its config.json).
Output:  printed, and saved to results/analysis/text_only_baselines.txt (unless --no-save).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.analysis.reliance import _avg_logprobs, _unique_argmax_is, group_records  # noqa: E402
from cmr.data.aokvqa import AOKVQAItem, load_aokvqa  # noqa: E402

DEFAULT_RUNS = ROOT / "results/runs/laptop/text_only_baselines"
DEFAULT_DATA = ROOT / "data/raw/aokvqa"
DEFAULT_BRIDGE = ROOT / "configs/ov_qwen2_05b_frozen.json"
SAVE_TO = Path("results/analysis/text_only_baselines.txt")
FOLDERS = {"question_options": "question + options", "options_only": "options only"}
N_VAL, N_ROT = 1145, 4
N_BOOT, SEED, TOL = 10_000, 0, 1e-9
GROUP = ("none", "none")  # (image condition, context)

Rots = dict[int, dict[str, Any]]


def labelled_val(aokvqa_dir: Path) -> list[AOKVQAItem]:
    """The labelled val questions in file order (the scoring and bootstrap order)."""
    items = [it for it in load_aokvqa(aokvqa_dir, "val") if it.has_labels]
    if len(items) != N_VAL:
        raise SystemExit(f"val has {len(items)} labelled questions, expected {N_VAL}")
    return items


def options_only(items: Sequence[AOKVQAItem]) -> list[AOKVQAItem]:
    """The same questions with the question text removed."""
    return [replace(it, question="") for it in items]


# ----------------------------------------------------------------------------- score


def check_prompts(items: Sequence[AOKVQAItem]) -> int:
    """Every options-only prompt equals its question + options prompt minus the question."""
    from cmr.eval.prompts import mc_prompt, rotate

    n = 0
    for it in items:
        for k in range(N_ROT):
            shown, _ = rotate(it.choices, k)
            full, blank = mc_prompt(it.question, shown), mc_prompt("", shown)
            first, sep, rest = full.partition("\n")
            if first != "Question: " + it.question.strip() or blank != "Question: " + sep + rest:
                raise SystemExit(f"prompt check failed: {it.question_id}, order {k}")
            n += 1
    return n


def score(args: argparse.Namespace) -> None:
    from cmr.eval.conditions import ContextSource, ImageCondition
    from cmr.provenance import pin_threads
    from cmr.runners.stage_a import RunSpec, build_model, run_stage_a

    items = labelled_val(args.aokvqa_dir)
    print(f"prompt check: {check_prompts(items)} (question, order) pairs identical", flush=True)
    pin_threads(args.threads)
    model = build_model(f"bridge:{args.bridge}", args.device, "float32", head_dtype="fp32")
    spec = RunSpec(
        conditions=(ImageCondition.NONE,), contexts=(ContextSource.NONE,), rotations=N_ROT, seed=0
    )
    run_stage_a(items, model, None, spec, args.out / "question_options")
    run_stage_a(options_only(items), model, None, spec, args.out / "options_only")
    print(f"wrote {rel(args.out)}/{{question_options,options_only}}/predictions.jsonl")


# ----------------------------------------------------------------------------- analyze


def load_run(path: Path, items: Sequence[AOKVQAItem]) -> dict[str, Rots]:
    """{question_id: {order: record}} for the no-image, no-context records; checked."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    runs = group_records(records)[GROUP]
    for it in items:
        rots = runs.get(it.question_id, {})
        if sorted(rots) != list(range(N_ROT)):
            raise SystemExit(f"{path.name}: {it.question_id} has orders {sorted(rots)}")
        if any(r["correct_idx"] != it.correct_choice_idx for r in rots.values()):
            raise SystemExit(f"{path.name}: {it.question_id}: gold differs from the val file")
    return runs


def per_item(runs: dict[str, Rots], items: Sequence[AOKVQAItem]) -> dict[str, np.ndarray]:
    """Per-question flags in val order: rotation-averaged unique max, ties, order 0."""
    rows: dict[str, list[Any]] = {k: [] for k in ("avg", "tied", "rot0", "pred0", "mass0")}
    for it in items:
        rots, gold = runs[it.question_id], it.correct_choice_idx
        p = _avg_logprobs(rots)
        rows["avg"].append(float(_unique_argmax_is(p, gold, TOL)))
        rows["tied"].append(int(np.sum(p >= p.max() - TOL)) > 1)
        rows["rot0"].append(float(rots[0]["pred_idx"] == gold))
        rows["pred0"].append(int(rots[0]["pred_idx"]))
        rows["mass0"].append(math.exp(rots[0]["letter_logmass"]))
    return {k: np.array(v) for k, v in rows.items()}


def bootstrap(cols: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """95% percentile intervals of each column's mean, all from the same question draws."""
    n = len(next(iter(cols.values())))
    rng = random.Random(SEED)
    means = {k: np.empty(N_BOOT) for k in cols}
    for b in range(N_BOOT):
        idx = np.fromiter(rng.choices(range(n), k=n), dtype=np.intp, count=n)
        for k, v in cols.items():
            means[k][b] = v[idx].mean()
    return {k: tuple(float(x) for x in np.quantile(m, [0.025, 0.975])) for k, m in means.items()}


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def rel(path: Path) -> str:
    """A path as printed: relative to the repo root, else from its results/ folder on."""
    if path.resolve().is_relative_to(ROOT):
        return path.resolve().relative_to(ROOT).as_posix()
    parts = path.absolute().parts
    return "/".join(parts[parts.index("results") :] if "results" in parts else parts[-3:])


def analyze(args: argparse.Namespace) -> list[str]:
    items = labelled_val(args.aokvqa_dir)
    paths = {k: getattr(args, k) / "predictions.jsonl" for k in FOLDERS}
    s = {k: per_item(load_run(p, items), items) for k, p in paths.items()}
    qo, oo = s["question_options"], s["options_only"]
    diff = qo["avg"] - oo["avg"]
    ci = bootstrap({"oo": oo["avg"], "diff": diff})
    lo, hi = ci["oo"]
    dlo, dhi = ci["diff"]
    lines = [
        "Text-only baselines of the frozen Qwen2-0.5B LLM (untrained bridge, no image)",
        f"Inputs: question + options {rel(paths['question_options'])}",
        f"        options only       {rel(paths['options_only'])}",
        f"Questions: {len(items)} labelled val questions; {N_ROT} cyclic option orders each.",
        f"Intervals: item bootstrap, {N_BOOT:,} draws, random.Random({SEED}), 95% percentile;",
        "the options-only accuracy and the difference use the same draws.",
        "",
        "Rotation-averaged accuracy (gold is the unique maximum of the option probabilities",
        "averaged over the four orders):",
        f"  options only          {pct(oo['avg'].mean())}  [{pct(lo)}, {pct(hi)}]",
        f"  question + options    {pct(qo['avg'].mean())}",
        f"  difference            {100 * diff.mean():+.2f} pp  "
        f"[{100 * dlo:+.2f}, {100 * dhi:+.2f}]  (question + options minus options only)",
        f"  questions with a tied maximum: options only {int(oo['tied'].sum())}, "
        f"question + options {int(qo['tied'].sum())}",
        "",
        "Original option order only (rotation 0):",
        f"  {'':22}{'accuracy':>10}{'answers A':>11}{'mean letter mass':>18}",
    ]
    for key in ("question_options", "options_only"):
        r = s[key]
        lines.append(
            f"  {FOLDERS[key]:<22}{pct(r['rot0'].mean()):>10}"
            f"{pct(float(np.mean(r['pred0'] == 0))):>11}{r['mass0'].mean():>18.3f}"
        )
    gold = Counter(it.correct_choice_idx for it in items)
    lines += [
        "  (letter mass = probability of the four option letters before renormalising)",
        "",
        "Gold position in the original order: "
        + ", ".join(f"{'ABCD'[k]} {v}" for k, v in sorted(gold.items())),
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score both prompt variants (needs torch and the models)")
    sc.add_argument("--aokvqa-dir", type=Path, default=DEFAULT_DATA)
    sc.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE, help="bridge spec JSON")
    sc.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    sc.add_argument("--device", default="mps")
    sc.add_argument("--threads", type=int, default=4)
    an = sub.add_parser("analyze", help="print the accuracies from the two predictions files")
    an.add_argument("--aokvqa-dir", type=Path, default=DEFAULT_DATA)
    for key, label in FOLDERS.items():
        flag = "--" + key.replace("_", "-")
        an.add_argument(flag, type=Path, default=DEFAULT_RUNS / key, help=f"{label} run folder")
    an.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        score(args)
        return 0
    text = "\n".join(analyze(args)) + "\n"
    sys.stdout.write(text)
    if not args.no_save:
        (ROOT / SAVE_TO).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / SAVE_TO).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
