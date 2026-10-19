"""Turn per-question MC records into accuracy and image-reliance metrics (summary.json/.md).

Per (condition, context) group: Acc (rotation 0), CircAcc (correct under every rotation),
PermAcc (argmax of rotation-averaged probabilities); paired contrasts with a reference group
(dAcc, McNemar p, VIG = mean of log p_ref(gold) - log p(gold) in nats, flip rate);
accuracy on the prior-wrong subset P; and prior-subtracted (PMI) accuracy. numpy only.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cmr.eval.stats import mcnemar_exact, paired_bootstrap, wilson_interval
from cmr.fsutil import write_text_atomic

Group = tuple[str, str]  # (condition, context)
REFERENCE: Group = ("real", "none")


def group_records(records: Sequence[dict[str, Any]]) -> dict[Group, dict[str, dict[int, Any]]]:
    """{(condition, context): {question_id: {rotation: record}}}; later duplicates win."""
    out: dict[Group, dict[str, dict[int, Any]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        out[(r["condition"], r["context"])][r["question_id"]][int(r["rotation"])] = r
    return out


def question_stats(rots: dict[int, dict[str, Any]]) -> tuple[float, float, float, float]:
    """One question's (correct@rot0, circular-correct, perm-avg-correct, log p(gold) perm-avg).

    ``rots`` maps rotation -> record (``correct_idx``, ``pred_idx``, ``option_logprobs`` in
    original option order). log p(gold) is floored at log(1e-12).

    >>> import math
    >>> def rec(pred, p):
    ...     return {"correct_idx": 0, "pred_idx": pred, "option_logprobs": [math.log(x) for x in p]}
    >>> s = question_stats({0: rec(0, [0.7, 0.1, 0.1, 0.1]), 1: rec(1, [0.3, 0.5, 0.1, 0.1])})
    >>> [round(x, 4) for x in s]
    [1.0, 0.0, 1.0, -0.6931]
    """
    gold = rots[0]["correct_idx"]
    acc0 = float(rots[0]["pred_idx"] == gold)
    circ = float(all(r["pred_idx"] == gold for r in rots.values()))
    probs = np.mean([np.exp(r["option_logprobs"]) for r in rots.values()], axis=0)
    perm = float(int(np.argmax(probs)) == gold)
    return acc0, circ, perm, float(math.log(max(probs[gold], 1e-12)))


def _avg_logprobs(rots: dict[int, dict[str, Any]]) -> np.ndarray:
    """Log of the option probabilities averaged over rotations."""
    probs = np.mean([np.exp(r["option_logprobs"]) for r in rots.values()], axis=0)
    out: np.ndarray = np.log(np.maximum(probs, 1e-12))  # numpy<2.5 stubs type this as Any
    return out


def _unique_argmax_is(scores: np.ndarray, gold: int, tol: float = 1e-9) -> bool:
    """True iff ``gold`` is the only option within ``tol`` of the top score."""
    return tie_credit(scores, gold, tol) == 1.0


def is_prior_wrong(rots: dict[int, dict[str, Any]]) -> bool:
    """True unless gold is the unique argmax of the rotation-averaged prior (ties count wrong)."""
    return not _unique_argmax_is(_avg_logprobs(rots), int(rots[0]["correct_idx"]))


CONTROLS = ("swapped", "blank")  # image conditions contrasted with real on P


def summarize(
    records: Sequence[dict[str, Any]],
    reference: Group = REFERENCE,
    prior_records: Sequence[dict[str, Any]] | None = None,
    prior_group: Group = ("none", "none"),
) -> dict[str, Any]:
    """Metrics per (condition, context) group, paired on the questions every group has.

    Rows: acc (Wilson CI), circ_acc (correct under every rotation), perm_acc (argmax of
    rotation-averaged probabilities); vs ``reference``: delta_acc_vs_ref, mcnemar_p,
    vig_nats = mean[log p_ref(gold) - log p_c(gold)] and flip_rate. With a prior, rows add
    Acc|P; context rows with a ``context_leak`` flag add leak-free accuracy.

    ``prior_records`` (a fixed external prior run) defines P; otherwise the run's own
    (none, reference context) group does. Runs without the reference group fall back to
    (none, reference context) and skip the image-vs-prior blocks.

    Raises:
        ValueError: no records, or a prior lacking ``prior_group`` or sharing no question.
    """
    groups = group_records(records)
    if not groups:
        raise ValueError("no records")
    if reference not in groups and ("none", reference[1]) in groups:
        reference = ("none", reference[1])
    image_ref = reference[0] != "none"
    common = sorted(set.intersection(*(set(g) for g in groups.values())))
    stats: dict[Group, np.ndarray] = {
        g: np.array([question_stats(groups[g][q]) for q in common]) for g in groups
    }
    n = len(common)

    rows: list[dict[str, Any]] = []
    ref = stats.get(reference)
    for g, s in sorted(stats.items()):
        ci = wilson_interval(float(s[:, 0].sum()), n)
        row: dict[str, Any] = {
            "condition": g[0],
            "context": g[1],
            "n": n,
            "acc": ci.estimate,
            "acc_ci": [ci.low, ci.high],
            "circ_acc": float(s[:, 1].mean()),
            "perm_acc": float(s[:, 2].mean()),
            "mean_logp_correct": float(s[:, 3].mean()),
            "letter_mass": float(
                np.mean([math.exp(groups[g][q][0].get("letter_logmass", 0.0)) for q in common])
            ),
        }
        if ref is not None and g != reference:
            mc = mcnemar_exact(ref[:, 0].astype(bool), s[:, 0].astype(bool))
            bs = paired_bootstrap(ref[:, 0], s[:, 0])
            vig = paired_bootstrap(ref[:, 3], s[:, 3])
            preds_ref = [groups[reference][q][0]["pred_idx"] for q in common]
            preds = [groups[g][q][0]["pred_idx"] for q in common]
            row.update(
                {
                    "delta_acc_vs_ref": bs.mean_diff,
                    "delta_acc_ci": [bs.low, bs.high],
                    "mcnemar_p": mc.p_value,
                    "vig_nats": vig.mean_diff,
                    "vig_ci": [vig.low, vig.high],
                    "flip_rate": float(
                        np.mean([a != b for a, b in zip(preds_ref, preds, strict=True)])
                    ),
                }
            )
        # Leak-free subset, only when every rotation-0 record carries the flag.
        leak = [groups[g][q][0].get("context_leak") for q in common]
        if g[1] != "none" and all(isinstance(x, bool) for x in leak):
            clean = ~np.array(leak, dtype=bool)
            row["n_leak_free"] = int(clean.sum())
            row["acc_leak_free"] = float(s[clean, 0].mean()) if clean.any() else None
        rows.append(row)
    summary: dict[str, Any] = {"n_questions": n, "reference": list(reference), "rows": rows}

    # Arbitration and debiasing need an image reference and its own NONE group.
    prior: Group = ("none", reference[1])
    if image_ref and reference in groups and prior in groups:
        summary["arbitration"] = arbitration_analysis(groups[reference], groups[prior], common)
        summary["arbitration"].update(
            debias_variants(groups, reference, common, prior_cond=prior[0])
        )

    # Prior-wrong set P: external prior if given, else this run's NONE group.
    if prior_records is not None:
        ext = group_records(prior_records)
        if prior_group not in ext:
            raise ValueError(f"external prior records have no {prior_group} group")
        source, prior_rots = "external", ext[prior_group]
    elif prior in groups:
        source, prior_rots = "own", groups[prior]
    else:
        return summary
    _add_prior_wrong(summary, groups, stats, common, prior_rots, source, reference, image_ref)
    return summary


def _add_prior_wrong(
    summary: dict[str, Any],
    groups: dict[Group, dict[str, dict[int, Any]]],
    stats: dict[Group, np.ndarray],
    common: Sequence[str],
    prior_rots: dict[str, dict[int, Any]],
    source: str,
    reference: Group,
    image_ref: bool = True,
) -> None:
    """Add ``summary["prior"]`` and per-row Acc|P and real-minus-control contrasts, in place."""
    in_scope = np.array([q in prior_rots for q in common])
    scope = [q for q in common if q in prior_rots]
    if not scope:
        raise ValueError("the prior records share no questions with this run")
    gold = np.array([int(prior_rots[q][0]["correct_idx"]) for q in scope])
    prior_lp = [_avg_logprobs(prior_rots[q]) for q in scope]
    wrong = np.array([is_prior_wrong(prior_rots[q]) for q in scope])
    prior_pred = np.array([int(np.argmax(lp)) for lp in prior_lp])
    block: dict[str, Any] = {
        "source": source,
        "rotations": int(max(len(prior_rots[q]) for q in scope)),
        "definition": "gold is not the unique argmax of rotation-averaged prior probabilities",
        "n_scope": len(scope),
        "n_prior_wrong": int(wrong.sum()),
        "n_prior_right": int((~wrong).sum()),
    }
    if image_ref and reference in groups:
        ref_lp = [_avg_logprobs(groups[reference][q]) for q in scope]
        ref_pred = np.array([int(np.argmax(lp)) for lp in ref_lp])
        acc_ref, acc_prior = float((ref_pred == gold).mean()), float((prior_pred == gold).mean())
        block["blind_agreement"] = float((ref_pred == prior_pred).mean())
        block["blind_agreement_bound"] = 1.0 - abs(acc_ref - acc_prior)
    summary["prior"] = block

    # Per-row Acc|P; real-image rows also get real minus each control on P.
    p_mask = np.zeros(len(common), dtype=bool)
    p_mask[np.flatnonzero(in_scope)[wrong]] = True
    by_group = {(r["condition"], r["context"]): r for r in summary["rows"]}
    for g, row in by_group.items():
        row["n_prior_wrong"] = int(p_mask.sum())
        row["acc_prior_wrong"] = float(stats[g][p_mask, 0].mean()) if p_mask.any() else None
        row["perm_acc_prior_wrong"] = float(stats[g][p_mask, 2].mean()) if p_mask.any() else None
        if g[0] != "real" or not p_mask.any():
            continue
        for ctl in CONTROLS:
            other = (ctl, g[1])
            if other not in stats:
                continue
            bs = paired_bootstrap(stats[g][p_mask, 0], stats[other][p_mask, 0])
            row[f"acc_prior_wrong_real_minus_{ctl}"] = bs.mean_diff
            row[f"acc_prior_wrong_real_minus_{ctl}_ci"] = [bs.low, bs.high]


def debias_variants(
    groups: dict[Group, dict[str, dict[int, Any]]],
    reference: Group,
    questions: Sequence[str],
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    prior_cond: str = "none",
) -> dict[str, Any]:
    """Rotation-averaged PMI debiasing of ``reference`` with NONE and BLANK priors.

    For each prior present: ``debiased_acc_*`` maps alpha -> tie-credited accuracy of
    argmax[log p_v - alpha * log p_prior], plus the tie rate at alpha = 1. If the BLANK prior
    helps as much as NONE, the gain is calibration, not prior over-weighting.
    """
    gold = [int(groups[reference][q][0]["correct_idx"]) for q in questions]
    lp_v = [_avg_logprobs(groups[reference][q]) for q in questions]
    out: dict[str, Any] = {
        "debias_rotations": int(max(len(groups[reference][q]) for q in questions)),
    }
    variants = [(prior_cond, "debiased_acc_rotavg")]
    variants += [("blank", "debiased_acc_blank_prior")]
    for cond, key in variants:
        g = (cond, reference[1])
        if g not in groups:
            continue
        lp_0 = [_avg_logprobs(groups[g][q]) for q in questions]
        out[key] = {
            f"{a:g}": float(
                np.mean(
                    [tie_credit(v - a * z, y) for v, z, y in zip(lp_v, lp_0, gold, strict=True)]
                )
            )
            for a in alphas
        }
        out[key.replace("debiased_acc", "debiased_tie_rate")] = _tie_rate(
            [v - z for v, z in zip(lp_v, lp_0, strict=True)]
        )
    lp_v0 = [np.asarray(groups[reference][q][0]["option_logprobs"], dtype=float) for q in questions]
    g0 = (prior_cond, reference[1])
    if g0 in groups:
        lp_00 = [np.asarray(groups[g0][q][0]["option_logprobs"], dtype=float) for q in questions]
        out["debiased_tie_rate_rot0"] = _tie_rate(
            [v - z for v, z in zip(lp_v0, lp_00, strict=True)]
        )
    return out


def _tie_rate(scores: Sequence[np.ndarray], tol: float = 1e-9) -> float:
    """Fraction of questions whose top option score is tied."""
    return float(np.mean([int(np.sum(x >= x.max() - tol)) > 1 for x in scores]))


def arbitration_analysis(
    with_image: dict[str, dict[int, Any]],
    without_image: dict[str, dict[int, Any]],
    questions: Sequence[str],
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Rotation-0 arbitration: does the model use the image or echo its language prior?

    Returns accuracy with the image split by whether the no-image prediction is right,
    blind_agreement (same answer with and without the image) and debiased_acc[alpha] =
    accuracy of argmax_i [log p(i|v,q) - alpha * log p(i|q)] (PMI scoring).
    """
    gold = [int(with_image[q][0]["correct_idx"]) for q in questions]
    # Per-question arrays: some benchmarks mix 2-10 options.
    lp_v = [np.asarray(with_image[q][0]["option_logprobs"], dtype=float) for q in questions]
    lp_0 = [np.asarray(without_image[q][0]["option_logprobs"], dtype=float) for q in questions]
    pred_v = np.array([int(np.argmax(v)) for v in lp_v])
    pred_0 = np.array([int(np.argmax(z)) for z in lp_0])
    gold_arr = np.array(gold)
    prior_right = pred_0 == gold_arr
    correct_v = pred_v == gold_arr

    def _acc(mask: np.ndarray) -> float | None:
        return float(correct_v[mask].mean()) if mask.any() else None

    return {
        "n_prior_right": int(prior_right.sum()),
        "n_prior_wrong": int((~prior_right).sum()),
        "acc_on_prior_right": _acc(prior_right),
        "acc_on_prior_wrong": _acc(~prior_right),
        "blind_agreement": float((pred_v == pred_0).mean()),
        "debiased_acc": {
            f"{a:g}": float(
                np.mean(
                    [tie_credit(v - a * z, g) for v, z, g in zip(lp_v, lp_0, gold, strict=True)]
                )
            )
            for a in alphas
        },
    }


def tie_credit(scores: np.ndarray, gold: int, tol: float = 1e-9) -> float:
    """Expected argmax accuracy with random tie-breaking: 1/k if gold is among k tied maxima.

    At alpha = 1 an image-blind model scores 0 on every option, so a plain argmax would
    always pick A.

    >>> import numpy as np
    >>> tie_credit(np.array([0.0, 0.0, -1.0]), 1), tie_credit(np.array([0.0, -2.0]), 0)
    (0.5, 1.0)
    """
    top = scores.max()
    tied = np.flatnonzero(scores >= top - tol)
    return 1.0 / tied.size if gold in tied else 0.0


def _fmt_pct(x: float) -> str:
    """Fraction as a percentage with one decimal: 0.852 -> "85.2"."""
    return f"{100 * x:.1f}"


def _pp(x: float) -> str:
    """Difference in percentage points with a sign: 0.031 -> "+3.1"."""
    return f"{100 * x:+.1f}"


def to_markdown(summary: dict[str, Any]) -> str:
    """Render a :func:`summarize` result as ``summary.md`` text (sections only if data exists)."""
    head = (
        "| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p "
        "| VIG nats [95% CI] | Flip % |\n|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in summary["rows"]:
        acc = f"{_fmt_pct(r['acc'])} [{_fmt_pct(r['acc_ci'][0])}, {_fmt_pct(r['acc_ci'][1])}]"
        if "delta_acc_vs_ref" in r:
            d = f"{100 * r['delta_acc_vs_ref']:+.1f}"
            p = f"{r['mcnemar_p']:.2g}"
            vig = f"{r['vig_nats']:+.3f} [{r['vig_ci'][0]:+.3f}, {r['vig_ci'][1]:+.3f}]"
            flip = _fmt_pct(r["flip_rate"])
        else:
            d = p = vig = flip = "ref"
        lines.append(
            f"| {r['condition']} | {r['context']} | {acc} | {_fmt_pct(r['circ_acc'])} "
            f"| {_fmt_pct(r['perm_acc'])} | {d} | {p} | {vig} | {flip} |"
        )
    text = f"n = {summary['n_questions']} questions (paired)\n\n" + "\n".join(lines) + "\n"

    arb = summary.get("arbitration")
    if arb:

        def pct(x: float | None) -> str:
            return "n/a" if x is None else _fmt_pct(x)

        deb = ", ".join(f"a={a}: {_fmt_pct(v)}" for a, v in arb["debiased_acc"].items())
        # "legacy_rot0": split by this run's own no-image answer at rotation 0 only; the
        # report's prior-wrong numbers use the rotation-averaged prior of the P block below.
        text += (
            "\n**Arbitration, legacy_rot0 (real vs no image, own NONE prior at rotation 0; "
            "quote the Prior-wrong subset P block below instead)**\n\n"
            f"- Acc with image when prior is right (n={arb['n_prior_right']}): "
            f"{pct(arb['acc_on_prior_right'])}%\n"
            f"- Acc with image when prior is wrong (n={arb['n_prior_wrong']}): "
            f"{pct(arb['acc_on_prior_wrong'])}%\n"
            f"- Blind agreement (same answer with/without image): "
            f"{_fmt_pct(arb['blind_agreement'])}%\n"
            f"- Prior-subtracted accuracy (rotation 0): {deb}\n"
        )
        for key, label in (
            ("debiased_acc_rotavg", "NONE prior, rotation-averaged"),
            ("debiased_acc_blank_prior", "BLANK prior, rotation-averaged"),
        ):
            if key in arb:
                vals = ", ".join(f"a={a}: {_fmt_pct(v)}" for a, v in arb[key].items())
                tie = arb.get(key.replace("debiased_acc", "debiased_tie_rate"))
                tie_txt = "" if tie is None else f" (ties at a=1: {_fmt_pct(tie)}%)"
                text += f"- Debiased accuracy, {label}: {vals}{tie_txt}\n"

    prior = summary.get("prior")
    if prior:
        text += (
            f"\n**Prior-wrong subset P** (prior: {prior['source']}, "
            f"{prior['rotations']} rotation(s); n_P = {prior['n_prior_wrong']} of "
            f"{prior['n_scope']}). P uses the rotation-averaged prior while Acc|P is at "
            "rotation 0, so the NONE row can have Acc|P > 0.\n"
        )
        if "blind_agreement" in prior:
            text += (
                f"\n- Blind agreement vs this prior: {_fmt_pct(prior['blind_agreement'])}% "
                f"(accuracy-implied bound {_fmt_pct(prior['blind_agreement_bound'])}%)\n"
            )
        text += (
            "\n| image | context | Acc\\|P % | real - control on P (pp) [95% CI] |\n"
            "|---|---|---|---|\n"
        )
        for r in summary["rows"]:
            if r.get("acc_prior_wrong") is None:
                continue
            parts = [
                f"{c}: {_pp(r[k])} [{_pp(r[k + '_ci'][0])}, {_pp(r[k + '_ci'][1])}]"
                for c in CONTROLS
                if (k := f"acc_prior_wrong_real_minus_{c}") in r
            ]
            text += (
                f"| {r['condition']} | {r['context']} | {_fmt_pct(r['acc_prior_wrong'])} "
                f"| {'; '.join(parts) or '-'} |\n"
            )

    leak_rows = [r for r in summary["rows"] if "n_leak_free" in r]
    if leak_rows:
        text += (
            "\n**Leak-free subset** (context has no stem of the gold option or of a "
            "direct answer)\n\n"
        )
        for r in leak_rows:
            acc = "n/a" if r["acc_leak_free"] is None else _fmt_pct(r["acc_leak_free"])
            text += f"- {r['condition']} / {r['context']}: Acc {acc}% on n = {r['n_leak_free']}\n"
    return text


def write_summary(summary: dict[str, Any], out_dir: Path) -> None:
    """Write summary.json, then summary.md (atomic; summary.md last marks completion)."""
    write_text_atomic(out_dir / "summary.json", json.dumps(summary, indent=2))
    write_text_atomic(out_dir / "summary.md", to_markdown(summary))
