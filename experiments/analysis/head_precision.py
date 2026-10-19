"""How much bf16 arithmetic and the scoring path move the letter log-probabilities.

Report Section 3.4 (why evaluation runs in fp32: in bf16, letter log-probabilities moved by up
to 0.279 nats and 2 of 32 predictions changed). Scores the first N labelled val questions x
{real image, no image} x option rotations on CPU with the evaluation code of
cmr.runners.stage_a and the untrained 0.5B bridge, in four variants, each with the prefix KV
cache on and off:

    bf16_full   bf16 model, LM head in bf16, logits at every position
    bf16_keep1  bf16 model, LM head in bf16, logits at the scored position only
    fp32_head   bf16 model, LM head in fp32 at the scored position
    fp32_ref    whole model in fp32 (the reference; what every evaluation uses)

Reports max |delta log p| and prediction flips per variant pair and per cache mode.

    python experiments/analysis/head_precision.py --out <new folder> [--dry-run]

Writes <out>/{<mode>/<variant>/, summary.json} and results/analysis/head_precision.txt. The
recorded run (default --limit 8) is results/runs/laptop/audit_head/summary.json.
Needs the `models` extra, the model weights and the COCO val2017 images.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# variant -> (model load dtype, model attributes set before scoring).
VARIANTS: dict[str, tuple[str, dict[str, bool]]] = {
    "bf16_full": ("bfloat16", {"head_fp32": False, "score_last_logits_only": False}),
    "bf16_keep1": ("bfloat16", {"head_fp32": False, "score_last_logits_only": True}),
    "fp32_head": ("bfloat16", {"head_fp32": True, "score_last_logits_only": True}),
    "fp32_ref": ("float32", {"head_fp32": True, "score_last_logits_only": True}),
}
# Variant pairs compared within each cache mode.
PAIRS = (
    ("bf16_full", "bf16_keep1"),
    ("fp32_head", "fp32_ref"),
    ("bf16_full", "fp32_ref"),
    ("bf16_keep1", "fp32_ref"),
)
CONDITIONS = ("real", "none")
CACHE_MODES = ("cached", "uncached")


def _key(rec: dict[str, Any]) -> tuple[str, str, str, int]:
    """Identity of one score: (question id, image condition, context, rotation).

    >>> _key({"question_id": "q1", "condition": "real", "context": "none", "rotation": "2"})
    ('q1', 'real', 'none', 2)
    """
    return (rec["question_id"], rec["condition"], rec["context"], int(rec["rotation"]))


def _raw(rec: dict[str, Any]) -> list[float]:
    """Raw letter log-probs under the full vocabulary: renormalised log-probs + letter_logmass.

    >>> _raw({"option_logprobs": [-0.5, -1.0], "letter_logmass": -0.25})
    [-0.75, -1.25]
    """
    return [v + rec["letter_logmass"] for v in rec["option_logprobs"]]


def compare(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per condition: n, max |delta| of option / raw letter log-probs, prediction flips.

    Raises:
        ValueError: when the two runs scored different key sets.

    >>> ra = {"question_id": "q", "condition": "real", "context": "none", "rotation": 0,
    ...       "option_logprobs": [-0.5, -1.0], "letter_logmass": 0.0, "pred_idx": 0}
    >>> rb = dict(ra, option_logprobs=[-0.75, -0.5], pred_idx=1)
    >>> compare([ra], [rb])
    {'real': {'n': 1, 'max_dlogp': 0.5, 'max_dlogp_raw': 0.5, 'flips': 1}}
    """
    by_key = {_key(r): r for r in b}
    if set(by_key) != {_key(r) for r in a}:
        raise ValueError("the two runs scored different (item, condition, rotation) sets")
    out: dict[str, Any] = {}
    for ra in a:
        rb = by_key[_key(ra)]
        row = out.setdefault(
            ra["condition"], {"n": 0, "max_dlogp": 0.0, "max_dlogp_raw": 0.0, "flips": 0}
        )
        row["n"] += 1
        pairs = zip(ra["option_logprobs"], rb["option_logprobs"], strict=True)
        d_opt = max(abs(x - y) for x, y in pairs)
        d_raw = max(abs(x - y) for x, y in zip(_raw(ra), _raw(rb), strict=True))
        row["max_dlogp"] = max(row["max_dlogp"], d_opt)
        row["max_dlogp_raw"] = max(row["max_dlogp_raw"], d_raw)
        row["flips"] += int(ra["pred_idx"] != rb["pred_idx"])
    return out


def markdown(results: dict[str, dict[str, Any]]) -> str:
    """Markdown table of {pair label: :func:`compare` output}, one row per (pair, condition).

    >>> r = {"real": {"n": 1, "max_dlogp": 0.5, "max_dlogp_raw": 0.5, "flips": 1}}
    >>> print(markdown({"x vs y": r}).splitlines()[-1])
    | x vs y | real | 1 | 0.500000 | 0.500000 | 1 |
    """
    lines = [
        "| pair | condition | n | max abs dlogp (options) | max abs dlogp (raw) | flips |",
        "|---|---|---|---|---|---|",
    ]
    for pair, per_cond in results.items():
        for cond, r in sorted(per_cond.items()):
            lines.append(
                f"| {pair} | {cond} | {r['n']} | {r['max_dlogp']:.6f} | "
                f"{r['max_dlogp_raw']:.6f} | {r['flips']} |"
            )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line; ``argv`` None reads ``sys.argv``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--root", default=str(ROOT), help="repo root (configs/, data/raw/); default: this repo"
    )
    p.add_argument("--bridge", default="configs/ov_qwen2_05b_frozen.json", help="BridgeSpec JSON")
    p.add_argument("--projector", default=None, help="trained projector.pt (default: initial)")
    p.add_argument("--aokvqa-dir", default="data/raw/aokvqa")
    p.add_argument("--coco-dir", default="data/raw/coco")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=8, help="first N labelled items")
    p.add_argument("--rotations", type=int, default=4)
    p.add_argument("--variants", default=",".join(VARIANTS), help="subset of " + ",".join(VARIANTS))
    p.add_argument(
        "--cache-modes",
        default="cached,uncached",
        help="prefix KV cache modes to score: cached (use_prefix_cache=True), uncached",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch threads, pinned at startup (default: $CMR_THREADS or 4)",
    )
    p.add_argument(
        "--output-txt",
        default="results/analysis/head_precision.txt",
        help="saved output (relative to --root); '' = do not write",
    )
    p.add_argument("--out", default=None, help="new or empty folder for per-variant records")
    p.add_argument("--dry-run", action="store_true", help="check data, print the plan, exit")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the audit and return the summary (``{"plan": ...}`` only with ``--dry-run``)."""
    args = parse_args(argv)
    from cmr.data.aokvqa import coco_image_path, load_aokvqa

    root = Path(args.root)

    def under_root(path: str) -> Path:
        """``path`` itself when absolute, else ``path`` under ``--root``."""
        return Path(path) if Path(path).is_absolute() else root / path

    # --- Validate the options and the data, and build the plan (no model yet) ---
    modes = [v.strip() for v in args.cache_modes.split(",") if v.strip()]
    if not modes or set(modes) - set(CACHE_MODES):
        raise SystemExit(f"--cache-modes must be from {list(CACHE_MODES)}, got {modes}")
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    bridge, coco = under_root(args.bridge), under_root(args.coco_dir)
    if not bridge.exists():
        raise SystemExit(f"no bridge spec at {bridge}")
    items = [it for it in load_aokvqa(under_root(args.aokvqa_dir), args.split) if it.has_labels]
    items = items[: args.limit]
    images = [coco_image_path(coco, args.split, it.image_id) for it in items]
    missing = [str(p) for p in images if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} COCO images missing, e.g. {missing[0]}")
    plan = {
        "bridge": args.bridge,  # as given (relative to --root unless absolute)
        "projector": args.projector,
        "items": len(items),
        "conditions": list(CONDITIONS),
        "rotations": args.rotations,
        "cache_modes": modes,
        "scores_per_variant": len(items) * len(CONDITIONS) * args.rotations,
        "variants": {n: VARIANTS[n] for n in names},
    }
    print(json.dumps({"plan": plan}), flush=True)
    if args.dry_run:
        return {"plan": plan}
    if args.out is None:
        raise SystemExit("--out is required (except with --dry-run)")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: use a new --out")
    out.mkdir(parents=True, exist_ok=True)

    # --- Heavy imports only now, so --dry-run stays light ---
    from PIL import Image

    from cmr.eval.conditions import ContextSource, ImageCondition
    from cmr.provenance import compute_info, default_threads, git_state, pin_threads
    from cmr.runners.stage_a import RunSpec, build_model, run_stage_a

    threads = args.threads if args.threads is not None else default_threads()
    pin_threads(threads)  # before any model work: the thread count changes the numerics
    started = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())

    def loader(it: Any) -> Image.Image:
        """The item's COCO image as RGB."""
        return Image.open(coco_image_path(coco, args.split, it.image_id)).convert("RGB")

    spec = RunSpec(
        conditions=tuple(ImageCondition(c) for c in CONDITIONS),
        contexts=(ContextSource.NONE,),
        rotations=args.rotations,
        seed=0,
    )
    # --- Score every (cache mode, variant): one model load per dtype ---
    records: dict[str, list[dict[str, Any]]] = {}  # "<mode>/<variant>"
    describe: dict[str, Any] = {}
    for dtype in ("bfloat16", "float32"):  # one model load per dtype
        todo = [n for n in names if VARIANTS[n][0] == dtype]
        if not todo:
            continue
        model = build_model(f"bridge:{bridge}", "cpu", dtype, args.projector)
        for mode in modes:
            model.use_prefix_cache = mode == "cached"
            for name in todo:
                for attr, value in VARIANTS[name][1].items():
                    setattr(model, attr, value)
                model.clear_cache()
                key, t0 = f"{mode}/{name}", time.time()
                records[key] = run_stage_a(items, model, loader, spec, out / key, log_every=0)
                describe[key] = model.describe()
                row = {"run": key, "seconds": round(time.time() - t0, 1), **describe[key]}
                print(json.dumps(row), flush=True)
        del model
        gc.collect()

    # --- Compare the PAIRS within each mode, then each variant across the two modes ---
    results: dict[str, Any] = {}
    for mode in modes:
        for a, b in PAIRS:
            if f"{mode}/{a}" in records and f"{mode}/{b}" in records:
                pair = compare(records[f"{mode}/{a}"], records[f"{mode}/{b}"])
                results[f"[{mode}] {a} vs {b}"] = pair
    if len(modes) == 2:
        for name in names:
            results[f"{name}: cached vs uncached"] = compare(
                records[f"cached/{name}"], records[f"uncached/{name}"]
            )
    # --- Write summary.json, print the table, and save it with a provenance header ---
    compute = compute_info()
    summary = {"plan": plan, "compute": compute, "describe": describe, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    table = markdown(results)
    print(table)
    if args.output_txt:
        sha, dirty = git_state()
        header = (
            f"# run {started}-{time.strftime('%H:%M:%SZ', time.gmtime())}, commit {sha}"
            f"{' (dirty)' if dirty else ''}, threads {threads}, out {args.out}\n"
            f"# compute {json.dumps(compute)}\n# plan {json.dumps(plan)}\n"
        )
        txt = under_root(args.output_txt)
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(header + table + "\n")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
