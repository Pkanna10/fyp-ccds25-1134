"""Stage B, training: train the bridge only, then score val exactly as Stage A does.

Stage A is :mod:`cmr.runners.stage_a` (evaluation only); val is scored here with its
:func:`~cmr.runners.stage_a.run_stage_a`, under the training target's own instruction.

    1. load A-OKVQA train and split off an image-grouped dev set (for --select and the
       end-of-training check; val is never used for choices)
    2. build FrozenBridgeVLM from a JSON spec (released projector as initialisation)
    3. train the projector with the chosen target (+ optional image-margin loss,
       --margin-lambda); <out>/resume.pt is written every --checkpoint-every steps; an
       existing one is refused unless --resume / --resume-if-present / --no-resume is given
    4. reload the selected checkpoint (--select: dev_acc / dev_logp / last)
    5. evaluate on A-OKVQA val under the --conditions image conditions (real / none /
       swapped / blank) -> <out>/eval_val/summary.md

--eval-only skips 3-4 and scores the initial bridge (theta_0) under the target's own
instruction (report Table 4.1).
Training runs in --dtype, every evaluation (dev and val) in --eval-dtype (default float32).

--target direct_answer: no dev check (--select last required); step 5 scores A-OKVQA Direct
Answer by greedy generation, as cmr-transfer --benchmark aokvqa-da. It refuses --rotations
other than 1, --prior-run, --rationale-weight, --margin-lambda and
--option-shuffle.

Resume: re-run with the same arguments + --resume-if-present until
<out>/eval_val/summary.md exists. Training resumes from resume.pt, val from
eval_val/predictions.jsonl. run_args.json keeps the first manifest and appends one entry
per start to ``manifest.segments``.

The defaults (batch 8 x 2, --select dev_acc, four image conditions, one rotation) are not
the report's settings; each reported run's arguments are in its run_args.json.

Example (0.5B A->R, 4k training set, the report's main settings, one GPU):
    cmr-stage-b --bridge configs/ov_qwen2_05b_frozen.json --target answer_then_rationale \\
        --aokvqa-dir data/raw/aokvqa --coco-dir data/raw/coco --train-limit 4000 \\
        --batch-size 1 --grad-accum 16 --select last --conditions real,swapped \\
        --rotations 4 --prior-run results/runs/laptop/pilot_a1v2_instrument \\
        --device cuda --dtype bfloat16 --seed 0 --out runs/a2r_4k_s0
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from cmr.data.aokvqa import AOKVQAItem, coco_image_path, dev_split, load_aokvqa
from cmr.eval.conditions import ImageCondition
from cmr.eval.prompts import MC_INSTRUCTION
from cmr.fsutil import write_text_atomic
from cmr.models.base import Query
from cmr.models.bridge import BridgeSpec, FrozenBridgeVLM, dtype_name
from cmr.models.head import HEAD_DTYPE_CHOICES, HEAD_DTYPE_HELP, parse_head_dtype
from cmr.provenance import continue_manifest, default_threads, pin_threads, run_manifest
from cmr.train.bridge_sft import (
    TrainConfig,
    finished_resume_state,
    preflight_resume,
    resume_finished,
    train_bridge,
)
from cmr.train.targets import INSTRUCTIONS, TargetFormat

# direct_answer evaluation: cmr-transfer's --max-new-tokens default
DA_MAX_NEW_TOKENS = 10


class FormatAdapter:
    """Letter scorer, for :func:`~cmr.runners.stage_a.run_stage_a`, of a bridge trained on ``fmt``.

    The AO instruction (``MC_INSTRUCTION``) is replaced by the target's own
    ``INSTRUCTIONS[fmt]``; R->A first generates its rationale (copied into each record via
    ``last_extras``).
    """

    def __init__(self, model: FrozenBridgeVLM, fmt: TargetFormat) -> None:
        """Wrap ``model`` for format ``fmt``."""
        self.model, self.fmt = model, fmt
        self.name = f"{model.name}[{fmt.value}]"
        self.last_extras: dict[str, object] | None = None

    def letter_logprobs(self, query: Query, n_options: int) -> list[float]:
        """Letter log-probs under the target's instruction; R->A also sets ``last_extras``."""
        text = query.text.replace(MC_INSTRUCTION, INSTRUCTIONS[self.fmt])
        q = Query(text, query.image)
        if self.fmt is TargetFormat.RATIONALE_THEN_ANSWER:
            from cmr.train.generate import rationale_then_letter

            lps, rationale, cue_found = rationale_then_letter(self.model, q, n_options)
            self.last_extras = {"generation": rationale, "cue_found": cue_found}
            return lps
        return self.model.letter_logprobs(q, n_options)


def make_loader(coco_dir: str, split: str) -> Callable[[AOKVQAItem], Image.Image]:
    """Item -> RGB PIL image from ``coco_dir`` for one COCO split ("train" or "val")."""

    def load(it: AOKVQAItem) -> Image.Image:
        im = Image.open(coco_image_path(coco_dir, split, it.image_id)).convert("RGB")
        im.info["cmr_key"] = f"{split}/{it.image_id}"  # the frozen-feature cache key
        return im

    return load


def check_direct_answer_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """``p.error`` for any setting the direct_answer target (no options, no letter) cannot honour.

    Runs the TrainConfig checks too, so they fail before any data is loaded.
    """
    if args.rotations != 1:
        p.error("--target direct_answer shows no options: --rotations must be 1")
    if args.prior_run:
        p.error("--target direct_answer: --prior-run (subset P) needs letter-scored records")
    try:
        TrainConfig(
            target=TargetFormat.DIRECT_ANSWER,
            rationale_weight=args.rationale_weight,
            margin_lambda=args.margin_lambda,
            option_shuffle=args.option_shuffle,
            select=args.select,
        )
    except ValueError as e:  # e.g. "--select dev_acc": no letter-scored dev check
        p.error(str(e))


def parse_dtype(p: argparse.ArgumentParser, name: str) -> torch.dtype:
    """``"bfloat16"`` -> ``torch.bfloat16``; anything but a floating dtype is a CLI error."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        p.error(f"not a floating torch dtype: {name!r}")
    return dtype


def main(argv: Iterable[str] | None = None) -> None:
    """``cmr-stage-b``: train the bridge (unless --eval-only), then score val (steps 1-5)."""
    from cmr.analysis.reliance import summarize, write_summary
    from cmr.runners.stage_a import (
        RunSpec,
        aokvqa_file,
        check_resume_scoring,
        load_prior_records,
        load_records,
        prior_records_path,
        read_json,
        run_stage_a,
    )

    # ---- 0. CLI and precision checks
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bridge", required=True, help="BridgeSpec JSON (configs/*.json)")
    p.add_argument("--target", default="answer_only", choices=[f.value for f in TargetFormat])
    p.add_argument("--aokvqa-dir", required=True)
    p.add_argument("--coco-dir", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--rationale-weight",
        type=float,
        default=None,
        help="balanced loss CE(answer letter) + w * CE(rest of target); omit for the token mean",
    )
    p.add_argument(
        "--feature-cache",
        default=None,
        help="cache directory for the frozen vision tower's per-image outputs",
    )
    p.add_argument(
        "--feature-cache-splits",
        default="train,val",
        help="COCO splits whose images are cached (e.g. 'val' for a full-data run)",
    )
    p.add_argument(
        "--block-attention",
        action="store_true",
        help="compute training attention in two query blocks (same values, faster on MPS)",
    )
    p.add_argument(
        "--option-shuffle",
        action="store_true",
        help="shuffle each training example's options (gold letter remapped)",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the decoder layers for training (pads to --pad-multiple tokens)",
    )
    p.add_argument("--pad-multiple", type=int, default=32, help="with --compile (see above)")
    p.add_argument(
        "--cpu-share",
        type=lambda v: -1 if v == "auto" else int(v),
        default=0,
        help="run the last K micro-batches of each step on a CPU copy in parallel (0 = off; "
        "'auto' rebalances K every 10 steps from the measured times)",
    )
    p.add_argument("--cpu-threads", type=int, default=6, help="threads of the CPU copy")
    p.add_argument(
        "--margin-lambda",
        type=float,
        default=0.0,
        help="weight mu of the image-margin hinge term (0 = off; report Section 3.6)",
    )
    p.add_argument("--margin-nats", type=float, default=1.0, help="margin m of that hinge, nats")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--dev-size", type=int, default=500, help="dev questions scored during training")
    p.add_argument(
        "--select",
        default="dev_acc",
        choices=["dev_acc", "dev_logp", "last"],
        help="checkpoint rule (reported runs use last)",
    )
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--conditions", default="real,none,swapped,blank")
    p.add_argument("--rotations", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="write <out>/resume.pt every N optimizer steps (and at the end); 0 = end only",
    )
    resume = p.add_mutually_exclusive_group()  # default: refuse if <out>/resume.pt exists
    resume.set_defaults(resume="refuse")
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_const",
        const="require",
        help="require <out>/resume.pt and continue from it (warns if saved by another commit)",
    )
    resume.add_argument(
        "--resume-if-present",
        dest="resume",
        action="store_const",
        const="if_present",
        help="continue <out>/resume.pt if it exists and was saved by the same git commit",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_const",
        const="restart",
        help="delete <out>/resume.pt and train from step 0",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=0,
        help="LoRA rank on the LLM's attention projections, trained with the bridge (0 = off)",
    )
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--dtype",
        default="float32",
        help="TRAINING dtype of the frozen LLM and vision tower (projector is always fp32)",
    )
    p.add_argument(
        "--eval-dtype",
        default="float32",
        help="dtype of every evaluation (dev metrics during training, val): the frozen "
        "towers are cast/rebuilt in it after training; must be --dtype or wider",
    )
    p.add_argument("--head-dtype", choices=HEAD_DTYPE_CHOICES, default="fp32", help=HEAD_DTYPE_HELP)
    p.add_argument(
        "--threads",
        type=int,
        default=default_threads(),
        help="torch intra-op threads, pinned at startup with 1 inter-op thread "
        "(default: $CMR_THREADS or 4); recorded in the run manifest",
    )
    p.add_argument(
        "--prior-run",
        default=None,
        help="text-only run (directory or its predictions.jsonl) whose no-image records define "
        "the prior-wrong subset P, e.g. results/runs/laptop/pilot_a1v2_instrument",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training: score the INITIAL bridge (theta_0) with this target's instruction",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(list(argv) if argv is not None else None)
    pin_threads(args.threads)  # before any model work: thread count changes the numerics
    train_dtype, eval_dtype = parse_dtype(p, args.dtype), parse_dtype(p, args.eval_dtype)
    if torch.promote_types(train_dtype, eval_dtype) != eval_dtype:
        p.error(f"--eval-dtype {args.eval_dtype} is narrower than --dtype {args.dtype}")
    fmt = TargetFormat(args.target)
    direct = fmt is TargetFormat.DIRECT_ANSWER  # free-text target: generation, no letters
    if direct:
        check_direct_answer_args(p, args)

    # fail fast on a bad path, before any model is loaded or trained
    prior = load_prior_records(args.prior_run) if args.prior_run else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. data: train / image-grouped dev / val (before any model is loaded)
    train_all = load_aokvqa(args.aokvqa_dir, "train")
    train, dev_all = dev_split(train_all, dev_fraction=0.06, seed=0)  # fixed split across runs
    if direct:  # the official DA subset, as cmr-transfer --benchmark aokvqa-da
        from cmr.eval.direct_answer import da_items

        val = da_items(load_aokvqa(args.aokvqa_dir, "val"))[: args.val_limit]
    else:
        val = [it for it in load_aokvqa(args.aokvqa_dir, "val") if it.has_labels]
        val = val[: args.val_limit]
    train_loader = make_loader(args.coco_dir, "train")
    train = train[: args.train_limit]
    dev = dev_all[: args.dev_size]
    if direct and not args.eval_only:
        dev = []  # train_bridge refuses dev items for a target without a letter
        print(
            "[stage-b] direct_answer: dev check skipped (no options or letter to score); "
            "--dev-size and --eval-every do not apply, the final step is kept "
            "(--select last); train_log rows carry the teacher-forced training loss only",
            flush=True,
        )

    # ---- 2. the bridge model (built in the training or the evaluation dtype)
    if args.block_attention:  # training passes only; scoring falls through to SDPA
        from cmr.models import block_attention

        block_attention.enable()
    spec = BridgeSpec.from_json(args.bridge)

    def build(dtype: torch.dtype) -> FrozenBridgeVLM:
        """The bridge from ``spec`` in ``dtype``, with the head dtype and LoRA set."""
        m = FrozenBridgeVLM.from_pretrained(spec, args.device, dtype)
        m.head_fp32 = parse_head_dtype(args.head_dtype)  # before config_hash / training
        if args.lora_r:
            m.enable_lora(r=args.lora_r, alpha=args.lora_alpha)
        if args.feature_cache:
            m.feature_cache_dir = Path(args.feature_cache)
            m.feature_cache_splits = frozenset(args.feature_cache_splits.split(","))
        return m

    cfg = TrainConfig(
        target=fmt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        margin_lambda=args.margin_lambda,
        margin_nats=args.margin_nats,
        eval_every=args.eval_every,
        seed=args.seed,
        rationale_weight=args.rationale_weight,
        option_shuffle=args.option_shuffle,
        cpu_share=args.cpu_share,
        compile=args.compile,
        select=args.select,
    )
    # a re-run after training finished trains nothing: load straight in the eval dtype
    finished = (
        finished_resume_state(out, cfg, len(train))
        if not args.eval_only and args.resume in ("if_present", "require")
        else None
    )
    model = build(eval_dtype if args.eval_only or finished is not None else train_dtype)
    n_trainable = model.trainable_parameters()
    print(f"[stage-b] trainable parameters: {n_trainable:,}", flush=True)

    # ---- run_args.json: provenance; every refusal comes before it is rewritten
    data_files = [
        aokvqa_file(args.aokvqa_dir, "train"),
        aokvqa_file(args.aokvqa_dir, "val"),
    ]
    if args.prior_run:
        data_files.append(prior_records_path(args.prior_run))
    # dtype names in provenance spelling ("bfloat16", "float32")
    precision = {
        "train_dtype": None if args.eval_only else dtype_name(train_dtype),
        "eval_dtype": dtype_name(eval_dtype),
    }
    # the model that scores val: this one in --eval-dtype (rebuilt after training)
    model_input = {**model.describe(), "dtype": precision["eval_dtype"], **precision}
    run_args = {
        **vars(args),
        "spec": asdict(spec),
        "trainable_parameters": n_trainable,
        **precision,
        "model_input": model_input,  # effective LM-head dtype, scoring dtype
        "manifest": run_manifest(data_files, model),
    }
    eval_dir = out / "eval_val"
    code_version = run_args["manifest"].get("git_commit")  # this process's code
    # refuse before rewriting run_args.json: existing eval records must match this run's
    # scoring, and an existing resume.pt must be one train_bridge would continue
    old_args = read_json(out / "run_args.json")
    check_resume_scoring(eval_dir / "predictions.jsonl", old_args, run_args, eval_dir)
    start_step = None
    if not args.eval_only:
        start_step = preflight_resume(
            model, train, cfg, out, dev, args.resume, code_version, dev_dtype=eval_dtype,
            train_dtype=None if finished is None else train_dtype, state=finished,
        )  # fmt: skip
    # keep the first manifest, append this start
    run_args["manifest"] = continue_manifest(
        old_args,
        run_args["manifest"],
        start_step=start_step,
        first_record_index=len(load_records(eval_dir / "predictions.jsonl", repair=True)),
    )
    write_text_atomic(out / "run_args.json", json.dumps(run_args, indent=2))

    # ---- 3-4. train (or skip a finished run), then reload the kept checkpoint
    if args.eval_only:  # theta_0: the initial projector, scored under this target's instruction
        print("[stage-b] --eval-only: no training, evaluating the initial bridge", flush=True)
    else:
        trained_here = finished is None  # the model in memory is the training-dtype one
        if finished is not None:  # checked like a resume (hash for the training dtype)
            result = resume_finished(
                model, train, cfg, out, finished, dev, args.resume, code_version=code_version,
                dev_dtype=eval_dtype, train_dtype=train_dtype,
            )  # fmt: skip
            finished = None  # free the saved optimizer state before val
        else:
            cpu_model = None
            if args.cpu_share:  # a float32 CPU copy (the CPU has no fast bf16)
                if args.lora_r:
                    raise SystemExit("--cpu-share does not support LoRA")
                cpu_model = FrozenBridgeVLM.from_pretrained(spec, "cpu", torch.float32)
                cpu_model.head_fp32 = model.head_fp32
            if args.compile:
                model.pad_multiple = args.pad_multiple
            result = train_bridge(
                model, train, train_loader, cfg, out, dev, args.checkpoint_every, args.resume,
                code_version=code_version, dev_dtype=eval_dtype, cpu_model=cpu_model,
                cpu_threads=args.cpu_threads,
            )  # fmt: skip
            model.pad_multiple = 1
            if cpu_model is not None:
                del cpu_model
                gc.collect()
                pin_threads(args.threads)  # the co-trainer changed the thread count
        # a resumed run spans two processes (possibly two commits): record it
        run_args["resumed_from_step"] = result["resumed_from"]
        run_args["resume_saved_commit"] = result["saved_code_version"]
        write_text_atomic(out / "run_args.json", json.dumps(run_args, indent=2))
        if direct:
            print(f"[stage-b] direct_answer: kept the final step ({result['steps']} steps)")
        else:
            acc = result["best_dev_acc"]
            print(f"[stage-b] dev acc of kept checkpoint ({cfg.select}): {acc:.4f}")
        if trained_here and eval_dtype != train_dtype:  # free it first: one copy at a time
            del model
            gc.collect()
            if torch.backends.mps.is_available():  # hand the cached blocks back to the OS
                torch.mps.empty_cache()
            model = build(eval_dtype)
        model.load_trainable_state(torch.load(out / "trainable.pt", map_location="cpu"))

    # ---- 5. score val with run_stage_a, under this target's own instruction
    scoring = {**model.describe(), **precision}
    if scoring != model_input:  # the resume guard above checked these settings
        raise RuntimeError(f"scoring settings {scoring} differ from the recorded {model_input}")
    model.eval()  # leave training mode (scoring and generation also enter it via scoring_mode)
    conditions = tuple(ImageCondition(c) for c in args.conditions.split(","))
    if direct:  # free-text answers: the cmr-transfer --benchmark aokvqa-da code path
        from cmr.eval.conditions import ContextSource
        from cmr.eval.direct_answer import da_scorer
        from cmr.runners.transfer import run_open

        run_open(
            val, model, make_loader(args.coco_dir, "val"), eval_dir, da_scorer, "A-OKVQA DA",
            conditions, (ContextSource.NONE,), None, DA_MAX_NEW_TOKENS,
            seed=0,  # same swapped-image pairing for every training seed and for cmr-transfer
        )  # fmt: skip
        print((eval_dir / "summary.md").read_text())
        return
    run_spec = RunSpec(
        conditions=conditions,
        rotations=args.rotations,
        seed=0,  # same swapped-image pairing for every training seed and for cmr-stage-a
    )
    records = run_stage_a(
        val, FormatAdapter(model, fmt), make_loader(args.coco_dir, "val"), run_spec, eval_dir
    )
    write_summary(summarize(records, prior_records=prior), eval_dir)
    print((eval_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
