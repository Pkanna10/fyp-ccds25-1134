"""Share of the next-token probability that the untrained bridge puts on the four letters.

Report Sections 4.1 and 5.1 (0.5B: median 72% under the A→R instruction against 97.5% under
the AO instruction). Letter mass m = exp(letter_logmass), the probability of the letters A-D at
the scored position before renormalising. Prints, for the A→R and AO theta_0 runs, the median m
and the share with m < 0.5 (item-bootstrap 95% CI), and per-cell statistics.

    python experiments/analysis/letter_mass.py \\
        [--ar-run results/runs/laptop/B0_answer_then_rationale_cpu] \\
        [--ao-run results/runs/laptop/pilot_a2v2_bridge_init] [--no-save]

Inputs: predictions.jsonl (release assets) of the two theta_0 runs, both scored on CPU.
Output: printed, and saved to results/analysis/letter_mass.txt (unless --no-save).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAPTOP = ROOT / "results/runs/laptop"
DEFAULT_AR = LAPTOP / "B0_answer_then_rationale_cpu"
DEFAULT_AO = LAPTOP / "pilot_a2v2_bridge_init"
SAVE_TO = Path("results/analysis/letter_mass.txt")
N_VAL, N_BOOT, SEED, FLOOR = 1145, 10_000, 0, 0.5
MAIN = ("real", "none", 0)  # (image condition, context, option order)

Cells = dict[tuple[str, str, int], dict[str, float]]


def predictions(run: Path) -> Path:
    """The run's predictions.jsonl (at the top of the folder or in eval_val/)."""
    for p in (run / "predictions.jsonl", run / "eval_val" / "predictions.jsonl"):
        if p.is_file():
            return p
    raise SystemExit(f"no predictions.jsonl in {run} or {run}/eval_val")


def cells(path: Path) -> Cells:
    """{(condition, context, order): {question_id: letter mass}}; one record per key."""
    out: Cells = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lm = r.get("letter_logmass")
        if not isinstance(lm, (int, float)) or not math.isfinite(lm):
            raise SystemExit(f"{path.name}: no finite letter_logmass for {r['question_id']}")
        cell = out.setdefault((r["condition"], r["context"], int(r["rotation"])), {})
        if r["question_id"] in cell:
            raise SystemExit(f"{path.name}: duplicate record {r['question_id']}")
        cell[r["question_id"]] = math.exp(lm)
    return out


def boot_ci(flags: list[int]) -> tuple[float, float]:
    rng = random.Random(SEED)
    means = [sum(rng.choices(flags, k=len(flags))) / len(flags) for _ in range(N_BOOT)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def rel(path: Path) -> str:
    """A path as printed: relative to the repo root, else from its results/ folder on."""
    if path.resolve().is_relative_to(ROOT):
        return path.resolve().relative_to(ROOT).as_posix()
    parts = path.absolute().parts
    return "/".join(parts[parts.index("results") :] if "results" in parts else parts[-3:])


def cell_row(name: str, key: tuple[str, str, int], mass: dict[str, float]) -> str:
    m = np.array(sorted(mass.values()))
    k = int((m < FLOOR).sum())
    return (
        f"  {name:<4} {key[0]:<8} {key[1]:<5} {key[2]:>5}  {len(m):>5}  {np.median(m):.3f} "
        f"({m[0]:.3f})  {k:>4} ({pct(k / len(m)):>6})  {np.quantile(m, 0.1):.3f}"
    )


def analyze(args: argparse.Namespace) -> list[str]:
    paths = {"A→R": predictions(args.ar_run), "AO": predictions(args.ao_run)}
    runs = {k: cells(p) for k, p in paths.items()}
    main = runs["A→R"].get(MAIN, {})
    if len(main) != N_VAL:
        raise SystemExit(f"A→R run: {len(main)} real, order-0 records, expected {N_VAL}")
    flags = [int(main[q] < FLOOR) for q in sorted(main)]
    lo, hi = boot_ci(flags)
    ao_main = runs["AO"][MAIN]
    lines = [
        "Letter mass of the untrained bridge theta_0 (Qwen2-0.5B): probability of the four",
        "option letters at the answer position, before renormalising",
        f"Inputs: A→R instruction {rel(paths['A→R'])}",
        f"        AO instruction  {rel(paths['AO'])}",
        "",
        "Real image, no context, original option order:",
        f"  median letter mass: A→R {np.median(list(main.values())):.3f}, "
        f"AO {np.median(list(ao_main.values())):.3f}",
        f"  share of questions below {FLOOR}: A→R {pct(sum(flags) / len(flags))} "
        f"[{pct(lo)}, {pct(hi)}] ({sum(flags)} of {len(flags)}), "
        f"AO {pct(np.mean([v < FLOOR for v in ao_main.values()]))}",
        f"  (interval: item bootstrap, {N_BOOT:,} draws, random.Random({SEED}), 95% percentile)",
        "",
        "Per cell (order = option order 0-3):",
        "  run  image    ctx   order      n  median (min)  below 0.5        p10",
    ]
    for name, cs in runs.items():
        lines += [cell_row(name, key, cs[key]) for key in sorted(cs)]
    lines.append("")
    for cond in ("real", "swapped"):
        a, b = runs["A→R"].get((cond, "none", 0)), runs["AO"].get((cond, "none", 0))
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        k = sum(a[q] < FLOOR <= b[q] for q in common)
        lines.append(
            f"{cond} image, original order: below {FLOOR} under A→R but not under AO for "
            f"{k} of {len(common)} questions ({pct(k / len(common))})"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ar-run", type=Path, default=DEFAULT_AR, help="theta_0, A→R instruction")
    ap.add_argument("--ao-run", type=Path, default=DEFAULT_AO, help="theta_0, AO instruction")
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
