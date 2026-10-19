"""Stage A, evaluation only: score a model on A-OKVQA MC under controlled input conditions.

No training happens here; Stage B (:mod:`cmr.runners.stage_b`) trains a bridge and then
scores it with :func:`run_stage_a` below. For every question q, image condition c, context s
and option rotation k (report Section 3.5):

    1. pick the image   : real / none / swapped (another question's image) / blank
    2. pick the context : none / gold rationales (raw or masked) / gold COCO captions
    3. cyclically shift the options by k and build the MC prompt
    4. one forward pass -> log p(letter) for each option letter
    5. renormalise over the options, map back to the original option order
    6. append one JSON record to predictions.jsonl (resumable)

Then :func:`cmr.analysis.reliance.summarize` turns records into metrics (summary.md).

Usage:
    cmr-stage-a --model bridge:configs/ov_qwen2_05b_frozen.json --projector <run>/projector.pt \\
        --aokvqa-dir data/raw/aokvqa --coco-dir data/raw/coco --out runs/<name>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from cmr.data.aokvqa import AOKVQAItem, coco_image_path, load_aokvqa
from cmr.eval.conditions import ContextSource, ImageCondition, swap_assignment
from cmr.eval.prompts import (
    MASK_VERSION,
    answer_strings,
    join_rationales,
    leaks,
    mask_options,
    mc_prompt,
    mentions,
    rotate,
)
from cmr.fsutil import read_jsonl, write_text_atomic
from cmr.models.base import LetterScorer, Query
from cmr.provenance import continue_manifest, default_threads, pin_threads, run_manifest

ImageLoader = Callable[[AOKVQAItem], Image.Image]
# the "blank" image condition: one mid-grey 384 x 384 image for every question
BLANK_IMAGE = Image.new("RGB", (384, 384), (128, 128, 128))


@dataclass(frozen=True)
class RunSpec:
    """The conditions x contexts x rotations grid of one :func:`run_stage_a` call.

    ``seed`` fixes the swapped-image pairing; ``meta`` holds the CLI arguments (config.json).
    """

    conditions: tuple[ImageCondition, ...] = (
        ImageCondition.REAL,
        ImageCondition.NONE,
        ImageCondition.SWAPPED,
    )
    contexts: tuple[ContextSource, ...] = (ContextSource.NONE,)
    rotations: int = 1
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _normalise(logps: Sequence[float]) -> list[float]:
    """Stable log-softmax: ``[log 0.2, log 0.6]`` -> ``[log 0.25, log 0.75]``."""
    m = max(logps)
    lse = m + math.log(sum(math.exp(v - m) for v in logps))
    return [v - lse for v in logps]


def _context_text(
    item: AOKVQAItem, source: ContextSource, captions: dict[int, list[str]] | None
) -> str | None:
    """Context shown before the question: None, gold rationales (raw/masked) or captions.

    Raises:
        ValueError: the item has no rationales, or its image has no captions.
    """
    if source is ContextSource.NONE:
        return None
    if source is ContextSource.RATIONALE:
        if not item.rationales:
            raise ValueError(f"{item.question_id} has no rationales")
        return join_rationales(item.rationales)
    if source is ContextSource.RATIONALE_MASKED:
        if not item.rationales:
            raise ValueError(f"{item.question_id} has no rationales")
        return mask_options(join_rationales(item.rationales), item.choices, item.direct_answers)
    if captions is None or item.image_id not in captions:
        raise ValueError(f"no COCO captions for image {item.image_id}")
    return " ".join(captions[item.image_id])


def _condition_image(
    cond: ImageCondition,
    i: int,
    items: Sequence[AOKVQAItem],
    image_loader: ImageLoader | None,
    perm: list[int] | None,
) -> Image.Image:
    """The image shown to item ``i`` under ``cond`` (``perm``: the swapped pairing)."""
    if cond is ImageCondition.BLANK:
        return BLANK_IMAGE
    assert image_loader is not None
    if cond is ImageCondition.REAL:
        return image_loader(items[i])
    if cond is ImageCondition.SWAPPED:
        assert perm is not None
        return image_loader(items[perm[i]])
    raise ValueError(f"no image for condition {cond}")


def _leak_fields(item: AOKVQAItem, ctx: ContextSource, context: str | None) -> dict[str, Any]:
    """Stem-level leak flags added to each record.

    rationale_leak_stem : a raw rationale contains a content-word stem of the gold option
    context_leak        : the context as shown contains a stem of the gold option or of a
                          >=3-vote direct answer
    caption_leak        : context_leak for the caption context
    mask_version        : masking algorithm used for rationale_masked
    """
    if item.correct_choice_idx is None:
        return {}
    out: dict[str, Any] = {
        "rationale_leak_stem": leaks(join_rationales(item.rationales), [item.correct_choice])
    }
    if context is not None:
        out["context_leak"] = leaks(context, answer_strings(item))
    if ctx is ContextSource.CAPTION:
        out["caption_leak"] = out["context_leak"]
    if ctx is ContextSource.RATIONALE_MASKED:
        out["mask_version"] = MASK_VERSION
    return out


def _record_key(rec: dict[str, Any]) -> tuple[str, str, str, int]:
    """Resume key of a record, e.g. ``("q1", "real", "none", 0)``."""
    return (rec["question_id"], rec["condition"], rec["context"], int(rec["rotation"]))


def load_records(path: Path, repair: bool = False) -> list[dict[str, Any]]:
    """Records of a predictions.jsonl (:func:`cmr.fsutil.read_jsonl`)."""
    return read_jsonl(path, repair=repair)


def prior_records_path(path: str | Path) -> Path:
    """predictions.jsonl of a prior run given as a run directory or as the file itself."""
    p = Path(path)
    return p / "predictions.jsonl" if p.is_dir() else p


def aokvqa_file(aokvqa_dir: str | Path, split: str) -> Path:
    """The annotation file :func:`load_aokvqa` reads (for the run manifest's data hashes).

    >>> aokvqa_file("data/raw/aokvqa", "val").as_posix()
    'data/raw/aokvqa/aokvqa_v1p0_val.json'
    """
    return Path(aokvqa_dir) / f"aokvqa_v1p0_{split}.json"


_UNKNOWN = "<not recorded>"  # a scoring setting that a run's config does not record


def scoring_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """The settings in a run config that change scored numbers.

    ``model_dtype`` is ``model_input.dtype``, else the CLI ``--dtype``. The thread count is
    the last segment's, else ``manifest.compute``'s. Unrecorded fields read as unknown.

    Example (a cmr-stage-a config.json that started with 4 threads and was resumed with 2):

        >>> cfg = {"model_input": {"head_dtype": "fp32", "prefix_cache": True,
        ...                        "dtype": "float32"},
        ...        "manifest": {"compute": {"torch_num_threads": 4},
        ...                     "segments": [{"torch_num_threads": 4},
        ...                                  {"torch_num_threads": 2}]}}
        >>> scoring_settings(cfg)  # doctest: +NORMALIZE_WHITESPACE
        {'head_dtype': 'fp32', 'prefix_cache': True, 'model_dtype': 'float32',
         'torch_num_threads': 2}
        >>> scoring_settings(None)["head_dtype"]
        '<not recorded>'
    """
    cfg = config if isinstance(config, dict) else {}
    model_input = cfg.get("model_input")
    model_input = model_input if isinstance(model_input, dict) else {}
    manifest = cfg.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    compute = manifest.get("compute")
    compute = dict(compute) if isinstance(compute, dict) else {}
    segments = manifest.get("segments")
    last = segments[-1] if isinstance(segments, list) and segments else None
    if isinstance(last, dict) and "torch_num_threads" in last:
        compute["torch_num_threads"] = last["torch_num_threads"]
    meta = cfg.get("meta")
    cli = meta if isinstance(meta, dict) else cfg  # CLI args: config.json "meta" / run_args.json
    return {
        "head_dtype": model_input.get("head_dtype", _UNKNOWN),
        "prefix_cache": model_input.get("prefix_cache", _UNKNOWN),
        "model_dtype": model_input.get("dtype", cli.get("dtype", _UNKNOWN)),
        "torch_num_threads": compute.get("torch_num_threads", _UNKNOWN),
    }


def check_resume_scoring(
    pred_path: Path, old_config: dict[str, Any] | None, new_config: dict[str, Any], out: Path
) -> None:
    """Refuse to append records scored differently from the existing ones.

    No-op unless ``pred_path`` holds records. An unrecorded setting counts as a mismatch
    unless this run does not record it either.

    Args:
        old_config: the config file as it was before this start (None: none).
        new_config: the config this start would write.

    Raises:
        SystemExit: the two configs' :func:`scoring_settings` differ.
    """
    if not pred_path.exists() or not pred_path.read_text().strip():
        return
    old, new = scoring_settings(old_config), scoring_settings(new_config)
    if old != new:
        raise SystemExit(
            f"{out}: existing records were scored with {old}; this run would score with "
            f"{new}; that would mix scoring paths, use a fresh --out"
        )


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON object from ``path``, or None (missing or unreadable file)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_prior_records(path: str | Path) -> list[dict[str, Any]]:
    """Records of a fixed prior run: a run directory or its predictions.jsonl.

    Raises:
        FileNotFoundError: no predictions.jsonl there (checked before any model loads).
    """
    p = prior_records_path(path)
    if not p.exists():
        raise FileNotFoundError(f"no prior records at {p}")
    return load_records(p)


def run_stage_a(
    items: Sequence[AOKVQAItem],
    model: LetterScorer,
    image_loader: ImageLoader | None,
    spec: RunSpec,
    out_dir: Path,
    captions: dict[int, list[str]] | None = None,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Score every (item x condition x context x rotation) into ``out_dir/predictions.jsonl``.

    Resumable: a (question, condition, context, rotation) key already recorded is never
    scored again, so a re-run after a kill appends exactly the missing records.

    Args:
        image_loader: item -> PIL image; None only if no condition needs a real image.
        captions: ``{image_id: [caption, ...]}`` for the caption context.
        log_every: print progress every this many items (0: never).

    Returns:
        All records (earlier and new), one dict per scored key.
    """
    # ---- 1. existing records: repair a torn tail, refuse mixed mask versions or duplicates
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    records = load_records(pred_path, repair=True)
    stale = {
        r.get("mask_version")
        for r in records
        if r["context"] == ContextSource.RATIONALE_MASKED.value
        and r.get("mask_version") != MASK_VERSION
    }
    if stale:  # resume keys ignore the masking algorithm: never mix versions in one run
        old = ", ".join(sorted(f"v{v or 1}" for v in stale))
        raise ValueError(
            f"{pred_path} has rationale_masked records from mask {old} "
            f"(current v{MASK_VERSION}); use a fresh --out"
        )
    done = {_record_key(r) for r in records}
    if len(done) != len(records):  # the summary would count a key twice
        raise ValueError(f"{pred_path} has {len(records) - len(done)} duplicate record keys")

    # ---- 2. check the model and inputs can produce every requested condition
    needs_images = any(c in (ImageCondition.REAL, ImageCondition.SWAPPED) for c in spec.conditions)
    if needs_images and image_loader is None:
        raise ValueError("REAL / SWAPPED image conditions require an image_loader")
    perm = (
        swap_assignment([it.image_id for it in items], seed=spec.seed)
        if ImageCondition.SWAPPED in spec.conditions
        else None
    )

    # ---- 3. score every missing key (steps 1-6 above), flush per item
    t0, n_new = time.time(), 0
    with pred_path.open("a") as f:
        for i, item in enumerate(items):
            for cond in spec.conditions:
                image: Image.Image | None = None
                for ctx in spec.contexts:
                    context = _context_text(item, ctx, captions)
                    leak = _leak_fields(item, ctx, context)
                    for k in range(spec.rotations):
                        key = (item.question_id, cond.value, ctx.value, k)
                        if key in done:
                            continue
                        if image is None and cond is not ImageCondition.NONE:
                            image = _condition_image(cond, i, items, image_loader, perm)
                        # rotate, score, renormalise, map back to the original order
                        shown, original_index = rotate(item.choices, k)
                        query = Query(mc_prompt(item.question, shown, context), image)
                        raw_lp = model.letter_logprobs(query, len(shown))
                        letter_lp = _normalise(raw_lp)
                        option_lp = [0.0] * len(shown)
                        for pos, orig in enumerate(original_index):
                            option_lp[orig] = letter_lp[pos]
                        rec = {
                            "question_id": item.question_id,
                            "condition": cond.value,
                            "context": ctx.value,
                            "rotation": k,
                            "option_logprobs": option_lp,
                            "pred_idx": max(range(len(option_lp)), key=option_lp.__getitem__),
                            "correct_idx": item.correct_choice_idx,
                            "rationale_leak": item.correct_choice_idx is not None
                            and any(mentions(r, item.correct_choice) for r in item.rationales),
                            # log P(any option letter) before renormalising: format compliance
                            "letter_logmass": raw_lp[0] - letter_lp[0],
                            **leak,
                        }
                        extras = getattr(model, "last_extras", None)
                        if extras:
                            rec.update(extras)
                        f.write(json.dumps(rec) + "\n")
                        records.append(rec)
                        n_new += 1
            f.flush()
            if log_every and (i + 1) % log_every == 0:
                rate = n_new / max(time.time() - t0, 1e-9)
                print(f"[stage-a] {i + 1}/{len(items)} items, {rate:.2f} scores/s", flush=True)
    return records


# ----------------------------------------------------------------------------- CLI


def load_coco_captions(path: str | Path) -> dict[int, list[str]]:
    """``{image_id: [caption, ...]}`` from a COCO ``captions_{split}2017.json`` file."""
    data = json.loads(Path(path).read_text())
    out: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        out.setdefault(int(ann["image_id"]), []).append(str(ann["caption"]).strip())
    return out


def build_model(
    spec: str,
    device: str,
    dtype: str,
    projector: str | None = None,
    head_dtype: str = "fp32",
) -> Any:
    """Letter scorer for ``bridge:<spec.json>``, or ``fake-prior`` (a tiny model for tests).

    Args:
        dtype: torch dtype name (``"auto"`` means fp32).
        projector: trained projector.pt to load into the bridge (None: the untrained theta_0).
        head_dtype: "fp32" applies the LM head in fp32 at the scored position; "model"
            keeps the model's own head dtype.

    Raises:
        ValueError: unknown spec.
    """
    from cmr.models.head import parse_head_dtype

    head_fp32 = parse_head_dtype(head_dtype)  # torch-free; validates the value
    if spec == "fake-prior":
        from cmr.models.fake import PriorOnlyModel

        return PriorOnlyModel()
    kind, _, path = spec.partition(":")
    if kind != "bridge":
        raise ValueError(f"unknown model spec {spec!r}")
    import torch

    from cmr.models.bridge import BridgeSpec, FrozenBridgeVLM

    model = FrozenBridgeVLM.from_pretrained(
        BridgeSpec.from_json(path),
        device,
        getattr(torch, "float32" if dtype == "auto" else dtype),
    )
    if projector:
        model.projector.load_state_dict(torch.load(projector, map_location=device))
    model.head_fp32 = head_fp32
    model.eval()  # letter scoring also enters eval mode itself (scoring_mode)
    return model


def _parse_enum_list(text: str, enum: type[Any]) -> tuple[Any, ...]:
    """Comma list -> enum members, e.g. ``"real, none"`` -> (REAL, NONE); blanks skipped."""
    return tuple(enum(v.strip()) for v in text.split(",") if v.strip())


def main(argv: Iterable[str] | None = None) -> None:
    """``cmr-stage-a``: parse the CLI, score the run, write summary.json and summary.md."""
    from cmr.analysis.reliance import summarize, write_summary
    from cmr.models.head import HEAD_DTYPE_CHOICES, HEAD_DTYPE_HELP

    # ---- 1. CLI
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--model",
        required=True,
        help="bridge:<spec.json> (add --projector for a trained bridge), or fake-prior "
        "(a tiny model used by the tests)",
    )
    p.add_argument("--aokvqa-dir", required=True)
    p.add_argument("--coco-dir", default=None)
    p.add_argument("--captions", default=None, help="COCO captions_{split}2017.json")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--conditions", default="real,none,swapped")
    p.add_argument("--contexts", default="none")
    p.add_argument("--rotations", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--head-dtype", choices=HEAD_DTYPE_CHOICES, default="fp32", help=HEAD_DTYPE_HELP)
    p.add_argument(
        "--threads",
        type=int,
        default=default_threads(),
        help="torch intra-op threads, pinned at startup with 1 inter-op thread "
        "(default: $CMR_THREADS or 4); recorded in the run manifest",
    )
    p.add_argument("--projector", default=None, help="trained projector.pt for bridge: models")
    p.add_argument(
        "--prior-run",
        default=None,
        help="text-only run (directory or its predictions.jsonl) whose no-image records define "
        "the prior-wrong subset P, e.g. results/runs/laptop/pilot_a1v2_instrument",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(list(argv) if argv is not None else None)
    pin_threads(args.threads)  # before any model work: thread count changes the numerics

    # fail fast on a bad path, before any model is loaded or trained
    prior = load_prior_records(args.prior_run) if args.prior_run else None
    conditions = _parse_enum_list(args.conditions, ImageCondition)

    # ---- 2. items (labelled only, first --limit) and the run grid
    items = load_aokvqa(args.aokvqa_dir, args.split)
    items = [it for it in items if it.has_labels][: args.limit]
    spec = RunSpec(
        conditions=conditions,
        contexts=_parse_enum_list(args.contexts, ContextSource),
        rotations=args.rotations,
        seed=args.seed,
        meta={k: v for k, v in vars(args).items()},
    )

    def loader(it: AOKVQAItem) -> Image.Image:
        """The COCO image of ``it`` (from --coco-dir, split --split)."""
        assert args.coco_dir is not None, "--coco-dir is required for image conditions"
        return Image.open(coco_image_path(args.coco_dir, args.split, it.image_id)).convert("RGB")

    # ---- 3. images, captions, the model; the data files the manifest hashes
    captions = load_coco_captions(args.captions) if args.captions else None
    model = build_model(args.model, args.device, args.dtype, args.projector, args.head_dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_files: list[str | Path | None] = [
        aokvqa_file(args.aokvqa_dir, args.split),
        args.captions,
        args.projector,  # trained bridge weights, when given
    ]
    if args.prior_run:
        data_files.append(prior_records_path(args.prior_run))

    # ---- 4. config.json: refuse a resume scored differently, keep the first manifest
    config = {
        **asdict(spec),
        "model": model.name,
        "model_input": model.describe() if hasattr(model, "describe") else None,
        "manifest": run_manifest(data_files, model),
    }
    # resuming into an existing --out: same scoring settings, or stop before touching it
    old_config = read_json(out / "config.json")
    check_resume_scoring(out / "predictions.jsonl", old_config, config, out)
    # keep the first manifest, append this start
    first_record = len(load_records(out / "predictions.jsonl", repair=True))
    config["manifest"] = continue_manifest(
        old_config, config["manifest"], first_record_index=first_record
    )
    write_text_atomic(out / "config.json", json.dumps(config, indent=2, default=str))

    # ---- 5. score (resumable), then summarise
    records = run_stage_a(items, model, loader if args.coco_dir else None, spec, out, captions)
    write_summary(summarize(records, prior_records=prior), out)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
