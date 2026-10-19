"""Transfer runner (report Section 4.11, Table 4.16): the bridges on tasks beyond A-OKVQA MC.

    --benchmark  aokvqa-da   A-OKVQA direct answer (generation, official soft accuracy)
                 mmstar      MMStar                (letter log-probs, as cmr-stage-a)

Steps:
    1. load the benchmark as AOKVQAItems (``--limit`` = hashed uniform subset)
    2. aokvqa-da: run_generation; mmstar: run_stage_a (both resumable via predictions.jsonl)
    3. write summary.json / summary.md: DA metrics (summarize_generation) or MC metrics
       (reliance.summarize) with the MMStar extras (per-category accuracy, chance level)

Example (as experiments/laptop/run_mac_transfer.sh):
    cmr-transfer --benchmark mmstar --model bridge:configs/ov_qwen2_05b_frozen.json \\
        --projector <run>/projector.pt --conditions real,swapped --out runs/transfer/mmstar
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from cmr.data.aokvqa import AOKVQAItem, coco_image_path, load_aokvqa
from cmr.eval.conditions import ContextSource, ImageCondition
from cmr.eval.direct_answer import (
    Scorer,
    as_generator,
    da_items,
    da_scorer,
    generation_markdown,
    run_generation,
    summarize_generation,
)
from cmr.eval.transfer import (
    MC_BENCHMARKS,
    TransferSet,
    chance_accuracy,
    extras_markdown,
    load_benchmark,
    per_category_accuracy,
)
from cmr.fsutil import write_text_atomic
from cmr.provenance import default_threads, pin_threads, run_manifest

# the --benchmark choices
BENCHMARKS = ("aokvqa-da", *MC_BENCHMARKS)


def _parse(text: str, enum: type[Any]) -> tuple[Any, ...]:
    """Comma list -> enum members, e.g. ``"real,none"`` -> (REAL, NONE); blanks skipped."""
    return tuple(enum(v.strip()) for v in text.split(",") if v.strip())


def mc_extras(records: list[dict[str, Any]], tset: TransferSet) -> dict[str, Any]:
    """Benchmark extras for an MC summary (``summary["transfer"]``).

    Returns:
        ``{"benchmark", "chance_acc", "skipped"}``, plus ``"per_category"`` when the
        benchmark has categories.
    """
    extras: dict[str, Any] = {
        "benchmark": tset.benchmark,
        "chance_acc": chance_accuracy(tset.items),
        "skipped": tset.skipped,
    }
    if tset.category:
        extras["per_category"] = per_category_accuracy(records, tset.category)
    return extras


def summarize_mc(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The MC summary (:func:`cmr.analysis.reliance.summarize`) of records with 2-10 options."""
    from cmr.analysis.reliance import summarize

    return summarize(records)


def run_mc(
    tset: TransferSet,
    model: Any,
    out: Path,
    conditions: tuple[ImageCondition, ...],
    rotations: int,
    seed: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Score an MC benchmark with ``run_stage_a``; write summary.json and summary.md.

    Args:
        meta: CLI arguments, stored with the run spec.

    Returns:
        The :func:`summarize_mc` summary with ``summary["transfer"]`` from :func:`mc_extras`.
    """
    from cmr.analysis.reliance import to_markdown
    from cmr.runners.stage_a import RunSpec, run_stage_a

    spec = RunSpec(conditions=conditions, rotations=rotations, seed=seed, meta=meta)
    records = run_stage_a(tset.items, model, tset.load_image, spec, out)
    summary = summarize_mc(records)
    summary["transfer"] = mc_extras(records, tset)
    write_text_atomic(out / "summary.json", json.dumps(summary, indent=2))
    md = f"# {tset.benchmark} ({model.name})\n\n" + to_markdown(summary)
    md += "\n" + extras_markdown(summary["transfer"])
    write_text_atomic(out / "summary.md", md)
    return summary


def run_open(
    items: list[AOKVQAItem],
    model: Any,
    loader: Any,
    out: Path,
    scorer: Scorer,
    metric: str,
    conditions: tuple[ImageCondition, ...],
    contexts: tuple[ContextSource, ...],
    captions: dict[int, list[str]] | None,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Score an open-answer benchmark by generation; write summary.json and summary.md.

    Args:
        loader: item -> image, or None if no condition needs one.
        scorer: answer scorer (A-OKVQA soft accuracy).
        metric: the metric's display name, e.g. ``"A-OKVQA DA"``.

    Returns:
        The generation summary with ``summary["metric"] = metric``.
    """
    gen = as_generator(model)
    records = run_generation(
        items,
        gen,
        loader,
        out,
        scorer=scorer,
        conditions=conditions,
        contexts=contexts,
        captions=captions,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    summary = summarize_generation(records)
    summary["metric"] = metric
    write_text_atomic(out / "summary.json", json.dumps(summary, indent=2))
    md = f"# {metric} ({gen.name})\n\n" + generation_markdown(summary, metric)
    write_text_atomic(out / "summary.md", md)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    """``cmr-transfer``: parse the CLI, build the model, run one benchmark, print summary.md."""
    from cmr.models.head import HEAD_DTYPE_CHOICES, HEAD_DTYPE_HELP
    from cmr.runners.stage_a import aokvqa_file, build_model, load_coco_captions

    # ---- 1. CLI
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    p.add_argument("--model", required=True, help="same specs as cmr-stage-a --model")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="hashed uniform subset size")
    p.add_argument("--conditions", default="real,none,swapped")
    p.add_argument(
        "--contexts",
        default="none",
        help="aokvqa-da only: comma list of none, rationale, rationale_masked, caption",
    )
    p.add_argument("--rotations", type=int, default=1, help="option orders scored (mmstar)")
    p.add_argument("--max-new-tokens", type=int, default=10, help="aokvqa-da only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aokvqa-dir", default=None)
    p.add_argument("--coco-dir", default=None)
    p.add_argument("--captions", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--projector", default=None)
    p.add_argument("--head-dtype", choices=HEAD_DTYPE_CHOICES, default="fp32", help=HEAD_DTYPE_HELP)
    p.add_argument(
        "--threads",
        type=int,
        default=default_threads(),
        help="torch intra-op threads, pinned at startup with 1 inter-op thread "
        "(default: $CMR_THREADS or 4); recorded in the run manifest",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    pin_threads(args.threads)  # before any model work: thread count changes the numerics

    # ---- 2. model and config.json (with the run manifest)
    conditions = _parse(args.conditions, ImageCondition)
    contexts = _parse(args.contexts, ContextSource)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(
        args.model, args.device, args.dtype, args.projector, head_dtype=args.head_dtype
    )
    data_files = [args.captions, args.projector]  # projector: trained bridge weights
    if args.benchmark == "aokvqa-da" and args.aokvqa_dir is not None:
        data_files.append(aokvqa_file(args.aokvqa_dir, args.split))
    config = {
        **vars(args),
        "model_name": model.name,
        "model_input": model.describe() if hasattr(model, "describe") else None,
        "manifest": run_manifest(data_files, model),
    }
    write_text_atomic(out / "config.json", json.dumps(config, indent=2, default=str))

    # ---- 3. run the benchmark: A-OKVQA DA, an open-answer benchmark, or an MC one
    if args.benchmark == "aokvqa-da":
        if args.aokvqa_dir is None:
            p.error("--aokvqa-dir is required for aokvqa-da")
        items = da_items(load_aokvqa(args.aokvqa_dir, args.split))[: args.limit]

        def coco_loader(it: AOKVQAItem) -> Image.Image:
            """COCO image of ``it`` from --coco-dir."""
            assert args.coco_dir is not None, "--coco-dir is required for image conditions"
            path = coco_image_path(args.coco_dir, args.split, it.image_id)
            return Image.open(path).convert("RGB")

        captions = load_coco_captions(args.captions) if args.captions else None
        run_open(
            items,
            model,
            coco_loader if args.coco_dir else None,
            out,
            da_scorer,
            "A-OKVQA DA",
            conditions,
            contexts,
            captions,
            args.max_new_tokens,
            args.seed,
        )
    else:
        tset = load_benchmark(args.benchmark, limit=args.limit, seed=args.seed)
        print(f"[transfer] {args.benchmark}: {len(tset.items)} items, skipped {tset.skipped}")
        run_mc(
            tset,
            model,
            out,
            conditions,
            args.rotations,
            args.seed,
            {k: v for k, v in vars(args).items()},
        )
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
