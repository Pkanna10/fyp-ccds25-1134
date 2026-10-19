"""Train only the bridge (projector) of a :class:`FrozenBridgeVLM` (used by cmr-stage-b).

Each optimizer step draws ``batch_size * grad_accum`` training questions, builds each prompt
and target with :mod:`cmr.train.targets`, computes the loss on the target tokens (the
per-token mean, or the balanced loss CE(letter) + lambda * CE(rest); optionally plus the
image-margin hinge) and updates the bridge with AdamW. With dev items, dev metrics are
computed every ``eval_every`` steps and at the end; they choose the kept checkpoint unless
``select="last"``, which every reported training run uses.

Outputs in ``out_dir``: projector.pt (the bridge, in LLaVA's key format), trainable.pt
(bridge plus any LoRA weights), train_log.jsonl, train_config.json, resume.pt (full state
for resuming a killed run) and dev_cache/ (results of a dev pass in progress).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.conditions import swap_assignment
from cmr.eval.prompts import LETTERS, letter_token_ids
from cmr.fsutil import read_jsonl, write_atomic
from cmr.models.base import Query
from cmr.models.bridge import FrozenBridgeVLM, LossParts
from cmr.train.targets import (
    ANSWER_CUE,
    TargetFormat,
    answer_char_span,
    mismatch_rationales,
    prompt_for,
    shuffle_options,
    target_for,
)

ImageLoader = Callable[[AOKVQAItem], Image.Image]


@dataclass(frozen=True)
class TrainConfig:
    """Every setting of one bridge-training run (hashed into :func:`config_hash`).

    One optimizer step sees ``batch_size * grad_accum`` items. ``margin_lambda`` and
    ``margin_nats`` are mu and m of the image-margin loss (report Section 3.6). The
    direct_answer target supports only the per-token mean loss and ``select="last"``.
    """

    target: TargetFormat = TargetFormat.ANSWER_ONLY
    epochs: float = 1.0
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 1e-4
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    margin_lambda: float = 0.0  # 0 disables the image-margin loss
    margin_nats: float = 1.0
    eval_every: int = 200
    seed: int = 0
    # None: per-token mean over the target; lambda >= 0: CE(letter) + lambda * CE(rest)
    rationale_weight: float | None = None
    option_shuffle: bool = False  # shuffle each example's options (see shuffle_options)
    # laptop speed-ups (cmr.train.pilot; report Section 3.8.2): micro-batches per step run on
    # a CPU copy (-1 = rebalanced from measured times), and torch.compile of decoder layers
    cpu_share: int = 0
    compile: bool = False
    dev_teacher_forced: bool = True  # also log teacher-forced dev answer CE (one extra pass)
    # checkpoint rule: best dev accuracy, best mean dev log p(gold), or the final step
    select: Literal["dev_acc", "dev_logp", "last"] = "dev_acc"

    def __post_init__(self) -> None:
        if self.select not in ("dev_acc", "dev_logp", "last"):
            raise ValueError(f"unknown checkpoint rule {self.select!r}")
        if self.rationale_weight is not None and self.rationale_weight < 0:
            raise ValueError("rationale_weight must be >= 0 (or None for the per-token mean loss)")
        if self.target is TargetFormat.DIRECT_ANSWER:
            refused = {
                "rationale_weight (the balanced loss splits off the answer letter)": (
                    self.rationale_weight is not None
                ),
                "margin_lambda (the margin arm scores the gold letter)": bool(self.margin_lambda),
                "option_shuffle (its prompt shows no options)": self.option_shuffle,
                "select other than 'last' (no letter-scored dev check; the final step is kept)": (
                    self.select != "last"
                ),
            }
            for what, bad in refused.items():
                if bad:
                    raise ValueError(f"target direct_answer does not support {what}")


def set_seed(seed: int) -> None:
    """Seed the global python and torch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)


def _answer_logprob(
    model: FrozenBridgeVLM,
    prompts: list[str],
    images: list[Image.Image],
    prefixes: list[str],
    letters: list[str],
) -> torch.Tensor:
    """(B,) log p(letter | prompt, image, assistant prefix), differentiable (image-margin loss)."""
    visual = model._visual_batch(list(images))
    out = []
    for prompt, v, prefix, letter in zip(prompts, visual, prefixes, letters, strict=True):
        text = model.format_prompt(prompt, True)
        embeds, _ = model.build_inputs(text, v, assistant_prefix=prefix or None)
        logits = model.next_token_logits(embeds.unsqueeze(0))[0]
        # same token set as evaluation ("B" and " B"), not encode(letter)[0]
        group = letter_token_ids(model.tokenizer, LETTERS.index(letter) + 1)[-1]
        out.append(torch.logsumexp(torch.log_softmax(logits, -1)[group], 0))
    return torch.stack(out)


def _prefix_before_answer(target: str, fmt: TargetFormat) -> str:
    """Target text before the answer letter (``""`` for answer-first targets)."""
    if fmt is TargetFormat.RATIONALE_THEN_ANSWER:
        return target.rsplit(ANSWER_CUE, 1)[0] + ANSWER_CUE
    return ""


def objective(parts: LossParts, rationale_weight: float | None) -> torch.Tensor:
    """Batch loss: ``mean(all)``, or ``mean(answer + w * rest)`` with ``w = rationale_weight``.

    The first is the default per-token mean, the second the balanced loss CE(letter) +
    lambda * CE(rest) with lambda = ``rationale_weight`` (report Section 3.4).
    """
    if rationale_weight is None:
        return parts.all.mean()
    assert parts.answer is not None and parts.rest is not None
    return (parts.answer + rationale_weight * parts.rest).mean()


def dev_metrics(
    model: FrozenBridgeVLM,
    items: Sequence[AOKVQAItem],
    loader: ImageLoader,
    fmt: TargetFormat,
    teacher_forced: bool = True,
    batch_size: int = 8,
    cache: Path | None = None,
) -> dict[str, float]:
    """Dev metrics with the real image: dev_acc, dev_answer_nll and dev_answer_ce_tf.

    dev_answer_nll is -log p(gold letter) as scored (for R->A, after the model's own
    generated rationale); dev_answer_ce_tf is the teacher-forced answer CE (the dev
    counterpart of the training log's ``ce_answer``).

    Args:
        cache: JSONL of per-item results of this evaluation, so a killed pass resumes
            with identical results.

    Raises:
        ValueError: ``fmt`` is direct_answer.
    """
    from cmr.train.generate import rationale_then_letter_logprobs

    if fmt is TargetFormat.DIRECT_ANSWER:
        raise ValueError("direct_answer has no letter-scored dev metrics (no options)")

    def score(it: AOKVQAItem) -> list[float]:
        """Letter log-probs of one dev item under the target's own instruction."""
        query = Query(prompt_for(it, fmt), loader(it))
        if fmt is TargetFormat.RATIONALE_THEN_ANSWER:
            return rationale_then_letter_logprobs(model, query, len(it.choices))
        return model.letter_logprobs(query, len(it.choices))

    # scored in eval mode (no LoRA dropout); the modules' train flags are restored afterwards
    with model.scoring_mode():
        return _dev_metrics(
            model, items, loader, fmt, score, teacher_forced, batch_size, _ResultCache(cache)
        )


class _ResultCache:
    """Append-only per-item results of one dev evaluation, keyed by ``(kind, i, qid)``.

    ``kind`` is "lp" (an item's letter log-probs) or "tf" (a teacher-forced batch's summed
    answer CE).
    """

    def __init__(self, path: Path | None) -> None:
        """Load the rows already in ``path`` (None: in memory only)."""
        self.path = path
        self.rows: dict[tuple[str, int, str], Any] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            for r in read_jsonl(path, repair=True):  # a torn last line is redone
                self.rows[(r["kind"], int(r["i"]), r["qid"])] = r["v"]

    def get(self, kind: str, i: int, qid: str, compute: Callable[[], Any]) -> Any:
        """The cached value for the key, else ``compute()`` (appended to the file)."""
        key = (kind, i, qid)
        if key in self.rows:
            return self.rows[key]
        value = compute()
        if self.path is not None:
            line = json.dumps({"kind": kind, "i": i, "qid": qid, "v": value}) + "\n"
            with self.path.open("a") as f:  # one record per write, newline-terminated
                f.write(line)
        self.rows[key] = value
        return value


def _dev_metrics(
    model: FrozenBridgeVLM,
    items: Sequence[AOKVQAItem],
    loader: ImageLoader,
    fmt: TargetFormat,
    score: Callable[[AOKVQAItem], list[float]],
    teacher_forced: bool,
    batch_size: int,
    cache: _ResultCache,
) -> dict[str, float]:
    """:func:`dev_metrics` after the eval-mode switch (``score`` scores one item)."""
    correct, nll = 0, 0.0
    for i, it in enumerate(items):
        lp: list[float] = cache.get("lp", i, it.question_id, partial(score, it))
        correct += int(max(range(len(lp)), key=lp.__getitem__) == it.correct_choice_idx)
        nll -= lp[it.correct_choice_idx or 0]
    out = {"dev_acc": correct / max(len(items), 1), "dev_answer_nll": nll / max(len(items), 1)}

    # teacher-forced answer CE with gold targets, in batches
    if teacher_forced and items:
        rng = random.Random(12345)  # same rationale draw at every evaluation
        all_targets = [target_for(it, fmt, rng) for it in items]

        def batch_ce(i: int) -> float:
            """Summed answer-token CE of the dev batch starting at item ``i``."""
            batch = list(items[i : i + batch_size])
            targets = all_targets[i : i + batch_size]
            with torch.no_grad():
                parts = model.loss_parts(
                    [prompt_for(it, fmt) for it in batch],
                    [loader(it) for it in batch],
                    targets,
                    [answer_char_span(t, fmt) for t in targets],
                )
            assert parts.answer is not None
            return float(parts.answer.sum())

        total = 0.0
        for i in range(0, len(items), batch_size):
            total += cache.get("tf", i, items[i].question_id, partial(batch_ce, i))
        out["dev_answer_ce_tf"] = total / len(items)
    return out


def _save(state: dict[str, torch.Tensor], out_dir: Path) -> None:
    """Write projector.pt (LLaVA-compatible) and trainable.pt (projector + LoRA), atomically."""
    proj = {k.removeprefix("projector."): v for k, v in state.items() if k.startswith("projector.")}
    write_atomic(out_dir / "projector.pt", lambda f: torch.save(proj, f))
    write_atomic(out_dir / "trainable.pt", lambda f: torch.save(state, f))


def _clone_trainable(model: FrozenBridgeVLM) -> dict[str, torch.Tensor]:
    """A CPU copy of the trainable weights that later steps cannot change."""
    # trainable_state() does not copy on CPU: clone so later steps do not alias it
    return {k: v.clone() for k, v in model.trainable_state().items()}


def config_hash(
    cfg: TrainConfig,
    model: FrozenBridgeVLM,
    train_items: Sequence[AOKVQAItem],
    dev_items: Sequence[AOKVQAItem] = (),
    dev_dtype: torch.dtype | None = None,
    llm_dtype: torch.dtype | None = None,
) -> str:
    """16-hex-digit fingerprint of everything a resumed run must share with the saved run.

    Covers TrainConfig, bridge spec, trainable shapes, LoRA config, dtypes and the ordered
    train/dev question ids. ``llm_dtype``: the training dtype when ``model`` is loaded in
    another one.
    """

    def ids(items: Sequence[AOKVQAItem]) -> str:
        return hashlib.sha256("\n".join(it.question_id for it in items).encode()).hexdigest()

    cfg_fields = asdict(cfg)
    # fields added later enter the hash only when switched on, so resume.pt files written
    # before they existed keep their hash
    if not cfg.option_shuffle:
        cfg_fields.pop("option_shuffle")
    for name, off in (("cpu_share", 0), ("compile", False)):
        if cfg_fields[name] == off:
            cfg_fields.pop(name)
    payload = {
        "cfg": cfg_fields,
        "spec": asdict(model.spec) if is_dataclass(model.spec) else repr(model.spec),
        "trainable": {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad},
        "llm_dtype": str(llm_dtype or next(model.llm.parameters()).dtype),
        "train_ids": ids(train_items),
        "dev_ids": ids(dev_items),
    }
    # LoRA hyper-parameters (alpha, dropout, ...) are not visible in parameter shapes
    peft = getattr(model.llm, "peft_config", None) or {}
    payload["peft"] = {k: v.to_dict() for k, v in peft.items()}
    # an fp32 head changes the loss arithmetic only on a low-precision LLM
    if getattr(model, "head_fp32_active", False) and payload["llm_dtype"] != "torch.float32":
        payload["head_dtype"] = "fp32"
    if dev_dtype is not None and str(dev_dtype) != payload["llm_dtype"]:
        payload["dev_dtype"] = str(dev_dtype)
    blob = json.dumps(payload, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _json_default(o: object) -> object:
    """JSON fallback for :func:`config_hash`: sets as sorted lists, anything else as str."""
    # set iteration order depends on PYTHONHASHSEED
    return sorted(o) if isinstance(o, set | frozenset) else str(o)


# full mid-run training state in out_dir
RESUME_FILE = "resume.pt"
ResumeMode = Literal["refuse", "if_present", "require", "restart"]


def _rng_states() -> dict[str, Any]:
    """The python, numpy, torch (and CUDA) global RNG states, in a torch.save-able form."""
    import numpy as np

    name, keys, pos, has_gauss, gauss = np.random.get_state(legacy=True)
    out: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (name, [int(k) for k in keys], int(pos), int(has_gauss), float(gauss)),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def _set_rng_states(states: dict[str, Any]) -> None:
    """Restore the global RNG states saved by :func:`_rng_states`."""
    import numpy as np

    random.setstate(states["python"])
    name, keys, pos, has_gauss, gauss = states["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, gauss))
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


# per-item caches of dev passes in progress (removed once logged)
DEV_CACHE_DIR = "dev_cache"


def _dev_cache_path(out_dir: Path, chash: str, step: int) -> Path:
    """Cache file of the dev pass at ``step``."""
    return out_dir / DEV_CACHE_DIR / f"{chash}_step{step:07d}.jsonl"


def total_steps(cfg: TrainConfig, n_train: int) -> int:
    """Optimizer steps: ``ceil(n_train * epochs / (batch_size * grad_accum))``, at least 1."""
    return max(1, math.ceil(n_train * cfg.epochs / (cfg.batch_size * cfg.grad_accum)))


def _training_items(train_items: Sequence[AOKVQAItem], cfg: TrainConfig) -> list[AOKVQAItem]:
    """The items training uses: unchanged, except for answer_then_mismatched_rationale.

    For that target each item carries another question's rationales
    (:func:`cmr.train.targets.mismatch_rationales`).
    """
    if cfg.target is TargetFormat.ANSWER_THEN_MISMATCHED_RATIONALE:
        # dev keeps its own rationales: they follow the letter, so dev metrics never see them
        return mismatch_rationales(train_items, cfg.seed)
    return list(train_items)


def _load_resume(
    out_dir: Path, expected_hash: str, code_version: str | None, mode: ResumeMode
) -> dict[str, Any]:
    """Load ``out_dir/resume.pt`` and check it (:func:`_check_resume`)."""
    path = out_dir / RESUME_FILE
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    _check_resume(state, path, expected_hash, code_version, mode)
    return state


def _refuse_or_require(resume: ResumeMode, path: Path) -> None:
    """Raise if resume.pt exists under "refuse" or is missing under "require"."""
    if resume == "refuse" and path.exists():
        raise FileExistsError(
            f"{path} exists: --resume continues it, --no-resume starts over, or use a new --out"
        )
    if resume == "require" and not path.exists():
        raise FileNotFoundError(f"--resume: no {path}")


def preflight_resume(
    model: FrozenBridgeVLM,
    train_items: Sequence[AOKVQAItem],
    cfg: TrainConfig,
    out_dir: Path,
    dev_items: Sequence[AOKVQAItem] = (),
    resume: ResumeMode = "refuse",
    code_version: str | None = None,
    dev_dtype: torch.dtype | None = None,
    train_dtype: torch.dtype | None = None,
    state: dict[str, Any] | None = None,
) -> int:
    """Raise now whatever :func:`train_bridge` would refuse about resume.pt, with no side effect.

    Returns the step training continues from (0: from scratch).

    Args:
        train_dtype: the LLM's training dtype when ``model`` is loaded in another one.
        state: resume.pt, if the caller already loaded it.
    """
    if resume not in ("refuse", "if_present", "require", "restart"):
        raise ValueError(f"unknown resume mode {resume!r}")
    path = out_dir / RESUME_FILE
    _refuse_or_require(resume, path)
    if resume == "restart" or not path.exists():
        return 0
    if state is None:
        state = torch.load(path, map_location="cpu", weights_only=True)
    items = _training_items(train_items, cfg)
    chash = config_hash(cfg, model, items, dev_items, dev_dtype, llm_dtype=train_dtype)
    _check_resume(state, path, chash, code_version, resume, warn=False)
    return int(state["step"])


def _check_resume(
    state: dict[str, Any],
    path: Path,
    expected_hash: str,
    code_version: str | None,
    mode: ResumeMode,
    warn: bool = True,
) -> None:
    """Raise unless ``state`` has the same config hash and (unless "require") code version."""
    if state.get("config_hash") != expected_hash:
        raise ValueError(
            f"{path} was written by a different configuration "
            f"(hash {state.get('config_hash')} != {expected_hash}: TrainConfig, bridge spec, "
            "LoRA config, dtype, train or dev items changed). Refusing to resume: use a new "
            "output folder, or pass --no-resume to start this one over."
        )
    saved_code = state.get("code_version")
    if saved_code != code_version:
        msg = f"{path} was saved by code {saved_code}; this run is code {code_version}"
        if mode != "require":
            raise ValueError(
                f"{msg}. Refusing to auto-resume across code versions (one run would mix two "
                "versions): pass --resume to accept it, --no-resume to start over, or use a "
                "new --out."
            )
        if warn:
            print(f"[train] WARNING: {msg}; resuming because --resume was given", flush=True)


def _restore_outputs(
    out_dir: Path, best_state: dict[str, torch.Tensor] | None, history: list[dict[str, Any]]
) -> None:
    """Rewrite projector.pt/trainable.pt and train_log.jsonl as of a saved resume state."""
    # outputs written after the saved step may not be reproduced exactly by the replay
    if best_state is None:
        for name in ("projector.pt", "trainable.pt"):
            (out_dir / name).unlink(missing_ok=True)
    else:
        _save(best_state, out_dir)
    write_atomic(
        out_dir / "train_log.jsonl",
        lambda f: f.write("".join(json.dumps(r) + "\n" for r in history).encode()),
    )


def finished_resume_state(out_dir: Path, cfg: TrainConfig, n_train: int) -> dict[str, Any] | None:
    """``out_dir/resume.pt`` if it records a complete run for ``cfg``, else None.

    Needs no model, so a finished run can be detected before the model is loaded.
    """
    path = out_dir / RESUME_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    done = state.get("step") == total_steps(cfg, n_train) and state.get("pending_row") is None
    return state if done else None


def resume_finished(
    model: FrozenBridgeVLM,
    train_items: Sequence[AOKVQAItem],
    cfg: TrainConfig,
    out_dir: Path,
    state: dict[str, Any],
    dev_items: Sequence[AOKVQAItem] = (),
    resume: ResumeMode = "if_present",
    code_version: str | None = None,
    dev_dtype: torch.dtype | None = None,
    train_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Check a complete run's ``state`` like a resume and rewrite its outputs, without training.

    ``train_dtype``: the training dtype when ``model`` is loaded in another one. Returns
    what :func:`train_bridge` returns.
    """
    if resume not in ("if_present", "require"):
        raise ValueError(f"resume_finished needs resume mode if_present/require, not {resume!r}")
    items = _training_items(train_items, cfg)
    chash = config_hash(cfg, model, items, dev_items, dev_dtype, llm_dtype=train_dtype)
    _check_resume(state, out_dir / RESUME_FILE, chash, code_version, resume)
    if state.get("best_state") is None:
        raise RuntimeError(f"{out_dir / RESUME_FILE}: finished run without a kept checkpoint")
    write_atomic(out_dir / "train_config.json", lambda f: f.write(_cfg_json(cfg).encode()))
    history = list(state["history"])
    _restore_outputs(out_dir, state["best_state"], history)
    total = int(state["step"])
    print(f"[train] training already complete ({total}/{total} steps): skipped", flush=True)
    return {
        "steps": total,
        "best_dev_acc": float(state["best"]),
        "select": cfg.select,
        "history": history,
        "resumed_from": total,
        "saved_code_version": state.get("code_version"),
        "config_hash": chash,
    }


def _cfg_json(cfg: TrainConfig) -> str:
    """``train_config.json`` content."""
    return json.dumps(asdict(cfg), indent=2, default=str)


def _letter_spans(targets: Sequence[str], fmt: TargetFormat) -> list[tuple[int, int]] | None:
    """Answer-letter spans of ``targets``, or None for direct_answer (no letter)."""
    if fmt is TargetFormat.DIRECT_ANSWER:
        return None
    return [answer_char_span(t, fmt) for t in targets]


def _pilot_step(
    model: FrozenBridgeVLM,
    train_items: Sequence[AOKVQAItem],
    cfg: TrainConfig,
    step: int,
    order: list[int],
    rng: random.Random,
    target_rng: random.Random,
    loader: ImageLoader,
    compiled: Any,
    cotrain: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Gradients of one optimizer step with the laptop speed-ups (:mod:`cmr.train.pilot`).

    Used when ``cfg.compile`` or ``cfg.cpu_share`` is on ("pilot path"). Draws the same
    micro-batches and rationales as the plain loop in :func:`train_bridge` (consuming
    ``order`` in place); the last ``cotrain.share`` micro-batches run on the CPU copy.

    Returns:
        (mean losses, wall times in seconds).
    """
    from cmr.train.pilot import maybe

    jobs = []
    for micro in range(cfg.grad_accum):
        if len(order) < cfg.batch_size:
            order += rng.sample(range(len(train_items)), len(train_items))
        idx = order[: cfg.batch_size]
        del order[: cfg.batch_size]
        batch = [train_items[i] for i in idx]
        if cfg.option_shuffle:
            batch = [
                shuffle_options(it, f"{cfg.seed}:{step}:{micro}:{j}") for j, it in enumerate(batch)
            ]
        prompts = [prompt_for(it, cfg.target) for it in batch]
        targets = [target_for(it, cfg.target, target_rng) for it in batch]
        images = [loader(it) for it in batch]
        spans = _letter_spans(targets, cfg.target)
        jobs.append((prompts, images, targets, spans))
    n_cpu = cotrain.share if cotrain is not None else 0
    gpu_jobs, cpu_jobs = jobs[: len(jobs) - n_cpu], jobs[len(jobs) - n_cpu :]

    def work(m: FrozenBridgeVLM, job: Any, feats: torch.Tensor | None = None) -> dict[str, float]:
        """Forward + backward of one micro-batch on ``m``; returns its logged losses."""
        prompts, images, targets, spans = job
        parts = m.loss_parts(prompts, list(images), targets, spans, tower_feats=feats)
        loss = objective(parts, cfg.rationale_weight)
        (loss / cfg.grad_accum).backward()  # type: ignore[no-untyped-call]
        row: dict[str, float] = {"ce": loss.item(), "ce_all": parts.all.mean().item()}
        if spans is not None:
            assert parts.answer is not None and parts.rest is not None
            row["ce_answer"] = parts.answer.mean().item()
            row["ce_rest"] = parts.rest.mean().item()
        return row

    # CPU part first (in a thread), on tower features from the GPU model
    if cpu_jobs:
        with torch.no_grad():
            cpu_in = [(job, model.tower_features(list(job[1])).to("cpu")) for job in cpu_jobs]
        cotrain.run(cpu_in, lambda m, jf: work(m, jf[0], jf[1]))

    t0 = time.perf_counter()
    rows = []
    with maybe(compiled):
        for job in gpu_jobs:
            rows.append(work(model, job))
    if model.device.type == "mps":
        torch.mps.synchronize()
    sec_gpu = time.perf_counter() - t0

    times = {"sec_gpu": round(sec_gpu, 3)}
    if cpu_jobs:
        rows += cotrain.join()
        cotrain.add_grads_to(model)
        times["sec_cpu"] = round(cotrain.seconds, 3)
        times["cpu_share"] = len(cpu_jobs)
        cotrain.observe(sec_gpu, len(gpu_jobs), cotrain.seconds, len(cpu_jobs), len(jobs))
    keys: tuple[str, ...] = ("ce", "ce_all", "ce_answer", "ce_rest")
    if cfg.target is TargetFormat.DIRECT_ANSWER:
        keys = ("ce", "ce_all")
    stats = {k: sum(r[k] for r in rows) / cfg.grad_accum for k in keys}
    stats["margin"] = 0.0
    return stats, times


def train_bridge(
    model: FrozenBridgeVLM,
    train_items: Sequence[AOKVQAItem],
    loader: ImageLoader,
    cfg: TrainConfig,
    out_dir: Path,
    dev_items: Sequence[AOKVQAItem] = (),
    checkpoint_every: int = 50,
    resume: ResumeMode = "refuse",
    code_version: str | None = None,
    dev_dtype: torch.dtype | None = None,
    cpu_model: FrozenBridgeVLM | None = None,
    cpu_threads: int = 6,
) -> dict[str, Any]:
    """Train the bridge; resumable mid-run from ``out_dir/resume.pt``.

    The full training state is saved atomically at optimizer-step boundaries; at an
    evaluation step it is saved before the dev pass, so a killed run resumes with
    identical outputs.

    Args:
        checkpoint_every: optimizer steps between resume.pt writes (0 = only at the end).
        resume: when resume.pt exists, "refuse" raises, "if_present" resumes (same code
            version only), "require" resumes (also across code versions, with a warning;
            raises if resume.pt is missing), "restart" starts over.
        code_version: the git commit, stored in and checked against resume.pt.
        dev_dtype: wider dtype for the frozen towers during dev metrics.
        cpu_model: float32 CPU copy for ``cfg.cpu_share``.

    Returns:
        steps, best_dev_acc (-1.0 without dev items), select, history, resumed_from,
        saved_code_version and config_hash.
    """
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")
    if cfg.target is TargetFormat.DIRECT_ANSWER and dev_items:
        raise ValueError(
            "target direct_answer has no letter-scored dev check: pass no dev_items "
            "(the final step is kept)"
        )
    llm_dtype = next(model.llm.parameters()).dtype
    if dev_dtype is not None and torch.promote_types(llm_dtype, dev_dtype) != dev_dtype:
        raise ValueError(f"dev_dtype {dev_dtype} is narrower than the LLM's {llm_dtype}")
    if cfg.cpu_share or cfg.compile:
        if cfg.margin_lambda:
            raise ValueError(
                "--compile / --cpu-share do not support the margin arm (image-margin loss, "
                "--margin-lambda)"
            )
        if not (cfg.cpu_share == -1 and cfg.grad_accum > 1) and not (
            0 <= cfg.cpu_share < cfg.grad_accum
        ):
            raise ValueError("cpu_share must be -1 (auto) or in [0, grad_accum)")
        if cfg.cpu_share and cpu_model is None:
            raise ValueError("cpu_share > 0 needs cpu_model")
    set_seed(cfg.seed)
    train_items = _training_items(train_items, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(cfg, model, train_items, dev_items, dev_dtype)

    resume_path = out_dir / RESUME_FILE
    saved: dict[str, Any] | None = None
    if resume not in ("refuse", "if_present", "require", "restart"):
        raise ValueError(f"unknown resume mode {resume!r}")
    _refuse_or_require(resume, resume_path)
    if resume == "restart":
        resume_path.unlink(missing_ok=True)
    elif resume_path.exists():
        saved = _load_resume(out_dir, chash, code_version, resume)
    write_atomic(out_dir / "train_config.json", lambda f: f.write(_cfg_json(cfg).encode()))

    rng = random.Random(cfg.seed)  # data order: identical across target formats
    target_rng = random.Random(cfg.seed + 1)  # rationale draws: do not perturb the order
    params = [p for p in model.parameters() if p.requires_grad]  # projector (+ LoRA)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = total_steps(cfg, len(train_items))
    warm = max(1, int(cfg.warmup_frac * total))
    # linear warm-up to cfg.lr, then cosine decay to 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        ),
    )
    # image-margin loss only: item i's swapped image is that of item swap[i] (another image)
    swap = (
        swap_assignment([it.image_id for it in train_items], seed=cfg.seed)
        if cfg.margin_lambda
        else None
    )

    order: list[int] = []  # item indices still to visit (refilled one shuffled pass at a time)
    best, best_score = -1.0, -math.inf
    best_state: dict[str, torch.Tensor] | None = None  # weights behind projector.pt
    history: list[dict[str, Any]] = []
    start = 0
    pending: dict[str, Any] | None = None  # row of the saved step whose dev pass was cut off
    if saved is not None:
        if saved["micro_step"] != 0:
            raise ValueError("resume.pt saved mid-accumulation is not supported")
        model.load_trainable_state(saved["trainable"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        start, order = int(saved["step"]), list(saved["order"])
        rng.setstate(saved["rng_order"])
        target_rng.setstate(saved["rng_target"])
        best, best_score = float(saved["best"]), float(saved["best_score"])
        best_state = saved["best_state"]
        history = list(saved["history"])
        pending = saved.get("pending_row")
        _set_rng_states(saved["rng"])
        what = " (dev pass of that step pending)" if pending is not None else ""
        print(f"[train] resumed from {resume_path} at step {start}/{total}{what}", flush=True)

    if saved is not None:
        _restore_outputs(out_dir, best_state, history)
    else:
        write_atomic(out_dir / "train_log.jsonl", lambda f: None)
    # remove dev caches other than the pending step's
    keep = _dev_cache_path(out_dir, chash, int(pending["step"])) if pending else None
    if (out_dir / DEV_CACHE_DIR).is_dir():
        for f in (out_dir / DEV_CACHE_DIR).iterdir():
            if f != keep:
                f.unlink()

    def checkpoint(step_done: int, pending_row: dict[str, Any] | None = None) -> None:
        """Write the full training state after optimizer step ``step_done`` to resume.pt."""
        state = {
            "version": 1,
            "config_hash": chash,
            "code_version": code_version,
            "step": step_done,
            "micro_step": 0,  # always saved at an optimizer-step boundary
            "trainable": model.trainable_state(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "order": list(order),
            "rng_order": rng.getstate(),
            "rng_target": target_rng.getstate(),
            "rng": _rng_states(),
            "best": best,
            "best_score": best_score,
            "best_state": best_state,
            "history": history,
            "total": total,
            # the row of step ``step_done`` when its dev pass has not been logged yet
            "pending_row": pending_row,
        }
        write_atomic(resume_path, lambda f: torch.save(state, f))

    def finish_step(row: dict[str, Any]) -> None:
        """Dev pass, checkpoint selection, logging and checkpoints for the step of ``row``."""
        nonlocal best, best_score, best_state

        step_done = int(row["step"])
        is_eval = bool(dev_items) and (step_done % cfg.eval_every == 0 or step_done == total)
        keep_final = step_done == total and (cfg.select == "last" or not dev_items)
        if keep_final:  # select "last": dev cannot change the kept weights, so save first
            best_state = _clone_trainable(model)
            _save(best_state, out_dir)

        if is_eval:
            checkpoint(step_done, pending_row=row)  # a kill in the dev pass resumes here
            cache = _dev_cache_path(out_dir, chash, step_done)
            # the RNG states must not depend on how many dev items came from the cache
            rng_before = _rng_states()
            with model.frozen_dtype(dev_dtype):  # in place; exact round trip
                dev = dev_metrics(
                    model, dev_items, loader, cfg.target, cfg.dev_teacher_forced, cache=cache
                )
            _set_rng_states(rng_before)
            row.update(dev)
            if cfg.select == "last":
                best = dev["dev_acc"]  # reported only; the final checkpoint is kept above
            else:
                score = dev["dev_acc"] if cfg.select == "dev_acc" else -dev["dev_answer_nll"]
                if score > best_score:
                    best_score, best = score, dev["dev_acc"]
                    row["selected"] = True
                    best_state = _clone_trainable(model)
                    _save(best_state, out_dir)

        if keep_final:
            row["selected"] = True
        history.append(row)
        with (out_dir / "train_log.jsonl").open("a") as log:  # rewritten on resume
            log.write(json.dumps(row) + "\n")
        periodic = bool(checkpoint_every) and step_done % checkpoint_every == 0
        if is_eval or periodic or step_done == total:
            checkpoint(step_done)
        if is_eval:
            cache.unlink(missing_ok=True)  # its result is in resume.pt now

    compiled = None
    if cfg.compile:
        from cmr.train.pilot import CompiledLayers

        compiled = CompiledLayers(model)
    cotrain = None
    if cfg.cpu_share:
        from cmr.train.pilot import CpuCoTrainer

        assert cpu_model is not None
        auto = cfg.cpu_share == -1
        cotrain = CpuCoTrainer(
            cpu_model, cpu_threads, min(3, cfg.grad_accum - 1) if auto else cfg.cpu_share, auto
        )
        cotrain.sync_from(model)
        cpu_model.projector.train()
    model.projector.train()
    if pending is not None:  # killed in the dev pass of the saved step: finish it
        finish_step(dict(pending))
    for step in range(start, total):  # one iteration = one optimizer step
        if compiled is not None or cotrain is not None:  # speed-ups: same items and draws
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            stats, times = _pilot_step(
                model, train_items, cfg, step, order, rng, target_rng, loader, compiled, cotrain
            )
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            sched.step()
            if cotrain is not None:
                cotrain.sync_from(model)
            times["sec_step"] = round(time.perf_counter() - t0, 3)
            finish_step({"step": step + 1, "lr": sched.get_last_lr()[0], **stats, **times})
            continue
        opt.zero_grad(set_to_none=True)
        stats = {"ce": 0.0, "ce_all": 0.0, "ce_answer": 0.0, "ce_rest": 0.0, "margin": 0.0}
        if cfg.target is TargetFormat.DIRECT_ANSWER:  # no letter: no answer / rest split
            del stats["ce_answer"], stats["ce_rest"]
        for micro in range(cfg.grad_accum):  # micro-batches whose gradients add up
            if len(order) < cfg.batch_size:
                order += rng.sample(range(len(train_items)), len(train_items))
            idx, order = order[: cfg.batch_size], order[cfg.batch_size :]
            batch = [train_items[i] for i in idx]
            if cfg.option_shuffle:  # per-example order, reproducible from (seed, step, slot)
                batch = [
                    shuffle_options(it, f"{cfg.seed}:{step}:{micro}:{j}")
                    for j, it in enumerate(batch)
                ]

            prompts = [prompt_for(it, cfg.target) for it in batch]
            targets = [target_for(it, cfg.target, target_rng) for it in batch]
            images = [loader(it) for it in batch]
            spans = _letter_spans(targets, cfg.target)

            parts = model.loss_parts(prompts, list(images), targets, spans)
            loss = objective(parts, cfg.rationale_weight)

            stats["ce"] += loss.item() / cfg.grad_accum  # the optimised CE
            stats["ce_all"] += parts.all.mean().item() / cfg.grad_accum
            if spans is not None:
                assert parts.answer is not None and parts.rest is not None
                stats["ce_answer"] += parts.answer.mean().item() / cfg.grad_accum
                stats["ce_rest"] += parts.rest.mean().item() / cfg.grad_accum

            # image-margin loss: hinge on log p(gold | real image) - log p(gold | swapped image)
            if cfg.margin_lambda and swap is not None:
                letters = [LETTERS[it.correct_choice_idx or 0] for it in batch]
                prefixes = [_prefix_before_answer(t, cfg.target) for t in targets]
                neg_images = [loader(train_items[swap[i]]) for i in idx]
                lp_pos = _answer_logprob(model, prompts, images, prefixes, letters)
                lp_neg = _answer_logprob(model, prompts, neg_images, prefixes, letters)
                margin = torch.relu(cfg.margin_nats - (lp_pos - lp_neg)).mean()
                loss = loss + cfg.margin_lambda * margin
                stats["margin"] += margin.item() / cfg.grad_accum

            (loss / cfg.grad_accum).backward()  # type: ignore[no-untyped-call]

        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()
        finish_step({"step": step + 1, "lr": sched.get_last_lr()[0], **stats})
    return {
        "steps": total,
        "best_dev_acc": best,
        "select": cfg.select,
        "history": history,
        "resumed_from": start if saved is not None else None,
        "saved_code_version": saved.get("code_version") if saved is not None else None,
        "config_hash": chash,
    }
