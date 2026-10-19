"""How much the A→R instruction alone changes the accuracy of the untrained bridge theta_0.

Report Section 4.1 (0.5B: -7.2 pp [-9.4, -4.9] with the real image, -2.2 pp [-4.2, -0.3] with
a swapped one). Paired differences over the val questions, rotation 0, no context:

    real offset     = acc theta_0(A→R instruction, real image) - acc theta_0(AO, real)
    swapped offset  = the same with swapped images
    gap difference  = real offset - swapped offset

Intervals: paired question bootstrap (cmr.eval.stats.paired_bootstrap), 95% percentile.

    python experiments/analysis/theta0_instruction_offset.py \\
        [--ao-run results/runs/cloud/G05_theta0_answer_only] \\
        [--ar-run results/runs/cloud/G05_theta0_answer_then_rationale] \\
        [--val-json data/raw/aokvqa/aokvqa_v1p0_val.json] [--no-save]

Inputs: predictions.jsonl (release assets) of the two cloud theta_0 runs (Table 4.1); the val
file, if present, only checks the question set. Output: printed, and saved to
results/analysis/theta0_instruction_offset.txt (unless --no-save).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.analysis.reliance import group_records, question_stats  # noqa: E402
from cmr.eval.stats import BootstrapResult, paired_bootstrap  # noqa: E402

DEFAULT_AO = ROOT / "results/runs/cloud/G05_theta0_answer_only"
DEFAULT_AR = ROOT / "results/runs/cloud/G05_theta0_answer_then_rationale"
DEFAULT_VAL = ROOT / "data/raw/aokvqa/aokvqa_v1p0_val.json"
SAVE_TO = Path("results/analysis/theta0_instruction_offset.txt")
CONDITIONS = ("real", "swapped")
N_VAL, N_BOOT, SEED = 1145, 10_000, 0


def predictions(run: Path) -> Path:
    """The run's predictions.jsonl (at the top of the folder or in eval_val/)."""
    for p in (run / "predictions.jsonl", run / "eval_val" / "predictions.jsonl"):
        if p.is_file():
            return p
    raise SystemExit(f"no predictions.jsonl in {run} or {run}/eval_val")


def rot0_scores(path: Path) -> dict[str, dict[str, float]]:
    """{condition: {question_id: correct in the original order}} (context none)."""
    groups = group_records([json.loads(s) for s in path.read_text().splitlines() if s.strip()])
    out = {}
    for c in CONDITIONS:
        g = groups[(c, "none")]
        if len(g) != N_VAL or any(0 not in rots for rots in g.values()):
            raise SystemExit(f"{path}: {c}: {len(g)} questions, expected {N_VAL} with order 0")
        out[c] = {q: question_stats(rots)[0] for q, rots in g.items()}
    return out


def rel(path: Path) -> str:
    """A path as printed: relative to the repo root, else from its results/ folder on."""
    if path.resolve().is_relative_to(ROOT):
        return path.resolve().relative_to(ROOT).as_posix()
    parts = path.absolute().parts
    return "/".join(parts[parts.index("results") :] if "results" in parts else parts[-3:])


def pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def analyze(args: argparse.Namespace) -> list[str]:
    paths = {"AO": predictions(args.ao_run), "A→R": predictions(args.ar_run)}
    scores = {k: rot0_scores(p) for k, p in paths.items()}
    qids = sorted(scores["AO"]["real"])
    if any(sorted(scores[k][c]) != qids for k in scores for c in CONDITIONS):
        raise SystemExit("the two runs do not cover the same questions")
    check = "val file not found; question-set check skipped"
    if args.val_json.is_file():
        if sorted(str(r["question_id"]) for r in json.loads(args.val_json.read_text())) != qids:
            raise SystemExit(f"{args.val_json.name}: question ids differ from the runs")
        check = f"ids match {args.val_json.name}"
    v = {(k, c): np.array([scores[k][c][q] for q in qids]) for k in scores for c in CONDITIONS}
    n = len(qids)

    def boot(a: np.ndarray, b: np.ndarray) -> BootstrapResult:
        return paired_bootstrap(a, b, N_BOOT, SEED)

    rows = [
        ("real offset: A→R instruction - AO instruction, real image",
         boot(v["A→R", "real"], v["AO", "real"])),
        ("swapped offset: A→R instruction - AO instruction, swapped image",
         boot(v["A→R", "swapped"], v["AO", "swapped"])),
        ("gap difference: real offset - swapped offset",
         boot(v["A→R", "real"] - v["A→R", "swapped"], v["AO", "real"] - v["AO", "swapped"])),
    ]  # fmt: skip
    d = (v["A→R", "real"] - v["AO", "real"]) - (v["A→R", "swapped"] - v["AO", "swapped"])
    counts = {kc: int(x.sum()) for kc, x in v.items()}
    lines = [
        "Instruction offset of the untrained bridge theta_0 (Qwen2-0.5B)",
        f"Inputs: AO instruction  {rel(paths['AO'])}",
        f"        A→R instruction {rel(paths['A→R'])}",
        f"Questions: {n} val questions ({check}).",
        "Score: correct in the original option order (rotation 0), no context.",
        f"Intervals: paired bootstrap, {N_BOOT:,} draws, seed {SEED}, 95% percentile.",
        "",
        f"Correct out of {n}:",
        f"  {'instruction':<13}{'real':>13}{'swapped':>13}{'real - swapped':>17}",
    ]
    for k in scores:
        r, s = counts[k, "real"], counts[k, "swapped"]
        lines.append(
            f"  {k:<13}{r:>6} {100 * r / n:5.1f}%{s:>6} {100 * s / n:5.1f}%{pp((r - s) / n):>14} pp"
        )
    lines += ["", "Differences (pp, 95% CI):"]
    lines += [f"  {name}\n      {pp(r.mean_diff)}  [{pp(r.low)}, {pp(r.high)}]" for name, r in rows]
    lines += [
        "",
        f"Exact counts: real {counts['A→R', 'real'] - counts['AO', 'real']}/{n}, swapped "
        f"{counts['A→R', 'swapped'] - counts['AO', 'swapped']}/{n}, gap difference "
        f"{round(float(d.sum()))}/{n}.",
        "Per-question gap difference (-2..+2), questions at each value: "
        + ", ".join(f"{x:+d}: {int((d == x).sum())}" for x in (-2, -1, 0, 1, 2))
        + ".",
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ao-run", type=Path, default=DEFAULT_AO, help="theta_0, AO instruction")
    ap.add_argument("--ar-run", type=Path, default=DEFAULT_AR, help="theta_0, A→R instruction")
    ap.add_argument("--val-json", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args(argv)
    text = "\n".join(analyze(args)) + "\n"
    sys.stdout.write(text)
    if not args.no_save:
        (ROOT / SAVE_TO).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / SAVE_TO).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
