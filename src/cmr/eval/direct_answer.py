"""Generation-based (open-ended) evaluation: A-OKVQA direct answers (report Section 4.11).

For each question x image condition x context: build a short-answer prompt, greedy-generate,
extract and score the answer, and append it to a resumable ``predictions.jsonl``; then
summarise scores, paired deltas vs the reference and flip rates. Loads no model itself.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.conditions import ContextSource, ImageCondition, swap_assignment
from cmr.eval.metrics import da_eligible, da_score, normalize_answer
from cmr.eval.prompts import join_rationales, mask_options
from cmr.eval.stats import paired_bootstrap
from cmr.fsutil import read_jsonl
from cmr.models.base import Query
from cmr.runners.stage_a import BLANK_IMAGE, _context_text

# LLaVA-1.5 short-answer instruction.
SHORT_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."

# Type aliases: item -> PIL image; (raw text, item) -> {"score", "score_norm"}.
ImageLoader = Callable[[AOKVQAItem], Image.Image]
Scorer = Callable[[str, AOKVQAItem], dict[str, float]]
Group = tuple[str, str]  # (condition, context)
REFERENCE: Group = ("real", "none")


@runtime_checkable
class TextGenerator(Protocol):
    """Anything that answers a :class:`Query` with free text (greedy decoding)."""

    name: str

    def generate(self, query: Query, max_new_tokens: int) -> str:
        """Greedy-decode up to ``max_new_tokens`` tokens; return the new text."""
        ...


# ----------------------------------------------------------------------------- prompts


def da_context(
    item: AOKVQAItem, ctx: ContextSource, captions: dict[int, list[str]] | None
) -> str | None:
    """Context text for DA prompts (None for NONE), as in :func:`cmr.runners.stage_a._context_text`.

    Both mask the options and the direct answers for RATIONALE_MASKED; the only difference is
    that an item without rationales gets an empty masked context here instead of an error.
    """
    if ctx is ContextSource.RATIONALE_MASKED:
        text = join_rationales(item.rationales)
        return mask_options(text, [*item.choices, *item.direct_answers])
    return _context_text(item, ctx, captions)


def short_answer_prompt(question: str, context: str | None = None) -> str:
    """:func:`cmr.eval.prompts.mc_prompt` layout without options, ending in the short-answer line.

    >>> print(short_answer_prompt("Why is the man holding the umbrella?"))
    Question: Why is the man holding the umbrella?
    Answer the question using a single word or phrase.
    """
    lines: list[str] = []
    if context:
        lines += [f"Context: {context.strip()}", ""]
    lines.append(f"Question: {question.strip()}")
    lines.append(SHORT_ANSWER_INSTRUCTION)
    return "\n".join(lines)


# "Answer:", "The answer is -", "Final answer:" ... at the start of the line.
_ANSWER_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]\s*", re.I)


def extract_answer(raw: str) -> str:
    """First non-empty line without "Answer:" prefix, quotes or trailing period, lower-cased.

    >>> extract_answer("Answer: Rain.")
    'rain'
    >>> extract_answer('"A dog."')
    'a dog'
    """
    line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    line = _ANSWER_PREFIX.sub("", line).strip().strip("\"'`").strip()
    return line.rstrip(".").strip().lower()


# ----------------------------------------------------------------------------- scorers


def da_scorer(raw: str, item: AOKVQAItem) -> dict[str, float]:
    """A-OKVQA DA soft accuracy: ``{"score": official, "score_norm": VQA-normalised}``."""
    pred = extract_answer(raw)
    return {
        "score": da_score(pred, item.direct_answers),
        "score_norm": da_score(pred, item.direct_answers, normalize=True),
    }


# ----------------------------------------------------------------------------- adapters


class BridgeGenerator:
    """Greedy free-text generation for :class:`cmr.models.bridge.FrozenBridgeVLM`."""

    def __init__(self, model: Any) -> None:
        """Wrap ``model``; ``name`` is its ``name`` attribute or class name."""
        self.model = model
        self.name = str(getattr(model, "name", type(model).__name__))

    def generate(self, query: Query, max_new_tokens: int) -> str:
        """Greedy-decode with no stop string; :func:`extract_answer` keeps the first line."""
        from cmr.train.generate import greedy_generate

        return greedy_generate(self.model, query, max_new_tokens=max_new_tokens, stop=None)


def as_generator(model: Any) -> TextGenerator:
    """Return ``model`` if it already generates, else wrap a known adapter (TypeError if none).

    ``FrozenBridgeVLM`` is always wrapped in :class:`BridgeGenerator`.
    """
    if isinstance(model, TextGenerator) and type(model).__name__ != "FrozenBridgeVLM":
        return model
    kind = type(model).__name__
    if kind == "FrozenBridgeVLM":
        return BridgeGenerator(model)
    raise TypeError(f"{kind} has no generate(query, max_new_tokens) and no known adapter")


# ----------------------------------------------------------------------------- runner


def _record_key(rec: dict[str, Any]) -> tuple[str, str, str]:
    """(question_id, condition, context): the identity of one generated answer."""
    return (rec["question_id"], rec["condition"], rec["context"])


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Records already generated; a torn last line is dropped and redone."""
    return read_jsonl(path, repair=True)


def da_items(items: Sequence[AOKVQAItem]) -> list[AOKVQAItem]:
    """Official DA subset (:func:`cmr.eval.metrics.da_eligible`), in item order."""
    return [it for it in items if da_eligible(it)]


def run_generation(
    items: Sequence[AOKVQAItem],
    model: TextGenerator,
    image_loader: ImageLoader | None,
    out_dir: Path,
    scorer: Scorer = da_scorer,
    conditions: Sequence[ImageCondition] = (ImageCondition.REAL, ImageCondition.NONE),
    contexts: Sequence[ContextSource] = (ContextSource.NONE,),
    captions: dict[int, list[str]] | None = None,
    max_new_tokens: int = 10,
    seed: int = 0,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Generate and score an answer per (item x condition x context), resuming from disk.

    Args:
        items: questions in a fixed order (the SWAPPED derangement depends on it).
        image_loader: may be None only if ``conditions`` is NONE alone.
        out_dir: answers are appended to ``out_dir/predictions.jsonl``.
        scorer: maps (raw text, item) to scores; :func:`da_scorer` by default.

    Returns:
        All records (on disk first, then new): question_id, condition, context, raw,
        answer and the scorer's fields.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    records = _load_records(pred_path)
    done = {_record_key(r) for r in records}
    if any(c is not ImageCondition.NONE for c in conditions) and image_loader is None:
        raise ValueError("image conditions other than NONE require an image_loader")

    # Same swapped-image pairing as run_stage_a for the same items and seed.
    perm = (
        swap_assignment([it.image_id for it in items], seed=seed)
        if ImageCondition.SWAPPED in conditions
        else None
    )

    t0, n_new = time.time(), 0
    with pred_path.open("a") as f:
        for i, item in enumerate(items):
            for cond in conditions:
                image: Image.Image | None = None
                for ctx in contexts:
                    if (item.question_id, cond.value, ctx.value) in done:
                        continue
                    # Load once per (item, condition); reuse across contexts.
                    if image is None and cond is not ImageCondition.NONE:
                        assert image_loader is not None
                        if cond is ImageCondition.REAL:
                            image = image_loader(item)
                        elif cond is ImageCondition.SWAPPED:
                            assert perm is not None
                            image = image_loader(items[perm[i]])
                        else:
                            image = BLANK_IMAGE
                    prompt = short_answer_prompt(item.question, da_context(item, ctx, captions))
                    raw = model.generate(Query(prompt, image), max_new_tokens)
                    rec = {
                        "question_id": item.question_id,
                        "condition": cond.value,
                        "context": ctx.value,
                        "raw": raw,
                        "answer": extract_answer(raw),
                        **scorer(raw, item),
                    }
                    f.write(json.dumps(rec) + "\n")
                    records.append(rec)
                    n_new += 1
            f.flush()
            if log_every and (i + 1) % log_every == 0:
                rate = n_new / max(time.time() - t0, 1e-9)
                print(f"[gen] {i + 1}/{len(items)} items, {rate:.2f} answers/s", flush=True)
    return records


# ----------------------------------------------------------------------------- summary


def _mean_ci(x: np.ndarray, seed: int = 0) -> list[float]:
    """Bootstrap 95% CI ``[low, high]`` of a mean."""
    bs = paired_bootstrap(x, np.zeros_like(x), seed=seed)
    return [bs.low, bs.high]


def summarize_generation(
    records: Sequence[dict[str, Any]], reference: Group = REFERENCE
) -> dict[str, Any]:
    """Per (condition, context) mean scores with CIs, plus paired deltas and flips vs reference.

    Uses only questions present in every group. Returns ``{"n_questions", "reference",
    "rows"}`` and, if the no-image group exists, ``"arbitration"`` (image score split by
    whether the no-image answer scores 0). ``delta_vs_ref`` is mean(ref - row).
    """
    groups: dict[Group, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        groups[(r["condition"], r["context"])][r["question_id"]] = r
    if not groups:
        raise ValueError("no records")
    common = sorted(set.intersection(*(set(g) for g in groups.values())))
    if not common:
        raise ValueError("no question is present in every (condition, context) group")
    n = len(common)

    def col(g: Group, key: str) -> np.ndarray:
        return np.array([float(groups[g][q][key]) for q in common])

    def norm_answers(g: Group) -> list[str]:
        return [normalize_answer(groups[g][q]["answer"]) for q in common]

    rows: list[dict[str, Any]] = []
    for g in sorted(groups):
        s, s_norm = col(g, "score"), col(g, "score_norm")
        row: dict[str, Any] = {
            "condition": g[0],
            "context": g[1],
            "n": n,
            "score": float(s.mean()),
            "score_ci": _mean_ci(s),
            "score_norm": float(s_norm.mean()),
            "score_norm_ci": _mean_ci(s_norm),
        }
        if reference in groups and g != reference:
            d = paired_bootstrap(col(reference, "score"), s)
            dn = paired_bootstrap(col(reference, "score_norm"), s_norm)
            ref_ans, ans = norm_answers(reference), norm_answers(g)
            row.update(
                {
                    "delta_vs_ref": d.mean_diff,
                    "delta_ci": [d.low, d.high],
                    "delta_p": d.p_value,
                    "delta_norm_vs_ref": dn.mean_diff,
                    "delta_norm_ci": [dn.low, dn.high],
                    "delta_norm_p": dn.p_value,
                    "flip_rate": float(
                        np.mean([a != b for a, b in zip(ref_ans, ans, strict=True)])
                    ),
                }
            )
        rows.append(row)
    summary: dict[str, Any] = {"n_questions": n, "reference": list(reference), "rows": rows}

    prior: Group = ("none", reference[1])
    if reference in groups and prior in groups:
        with_img, without = col(reference, "score_norm"), col(prior, "score_norm")
        wrong = without == 0.0

        def _mean(mask: np.ndarray) -> float | None:
            return float(with_img[mask].mean()) if mask.any() else None

        summary["arbitration"] = {
            "n_prior_right": int((~wrong).sum()),
            "n_prior_wrong": int(wrong.sum()),
            "score_on_prior_right": _mean(~wrong),
            "score_on_prior_wrong": _mean(wrong),
            "blind_agreement": float(
                np.mean(
                    [
                        a == b
                        for a, b in zip(norm_answers(reference), norm_answers(prior), strict=True)
                    ]
                )
            ),
        }
    return summary


def _pct(x: float | None) -> str:
    """Fraction as a one-decimal percentage ("n/a" for None)."""
    return "n/a" if x is None else f"{100 * x:.1f}"


def generation_markdown(summary: dict[str, Any], metric_name: str = "DA") -> str:
    """Render :func:`summarize_generation` output as a Markdown table plus arbitration bullets."""
    head = (
        f"| image | context | {metric_name} % [95% CI] | {metric_name}-norm % "
        "| d vs ref (pp) [95% CI] | p | d-norm (pp) | Flip % |\n|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in summary["rows"]:
        sc = f"{_pct(r['score'])} [{_pct(r['score_ci'][0])}, {_pct(r['score_ci'][1])}]"
        if "delta_vs_ref" in r:
            d = (
                f"{100 * r['delta_vs_ref']:+.1f} "
                f"[{100 * r['delta_ci'][0]:+.1f}, {100 * r['delta_ci'][1]:+.1f}]"
            )
            p, dn = f"{r['delta_p']:.2g}", f"{100 * r['delta_norm_vs_ref']:+.1f}"
            flip = _pct(r["flip_rate"])
        else:
            d = p = dn = flip = "ref"
        lines.append(
            f"| {r['condition']} | {r['context']} | {sc} | {_pct(r['score_norm'])} "
            f"| {d} | {p} | {dn} | {flip} |"
        )
    text = f"n = {summary['n_questions']} questions (paired)\n\n" + "\n".join(lines) + "\n"
    arb = summary.get("arbitration")
    if arb:
        text += (
            "\n**Arbitration (real image vs no image, normalised score)**\n\n"
            f"- Score with image when no-image score > 0 (n={arb['n_prior_right']}): "
            f"{_pct(arb['score_on_prior_right'])}%\n"
            f"- Score with image when no-image score = 0 (n={arb['n_prior_wrong']}): "
            f"{_pct(arb['score_on_prior_wrong'])}%\n"
            f"- Blind agreement (same normalised answer): {_pct(arb['blind_agreement'])}%\n"
        )
    return text
