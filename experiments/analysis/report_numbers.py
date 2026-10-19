"""Recompute hand-computed report numbers from the data and run folders and compare.

Prints one line per number, rounded to the report's decimals:

    <what>: <report value> -> <recomputed value>   ok | DIFFERS (by ...)

Sections (--sections picks a subset; default all):
  overlap      Section 3.2: question/option word overlap (cmr.eval.prompts.content_stems).
  splits       Table 3.2: questions and images per split (dev = dev_split, 6%, seed 0).
  giveaways    Section 5.4: a question word appears in exactly one option (exact, no stemming).
  parity       Section 4.1: cloud-GPU run G05_gate64 vs CPU run pilot_a2v2_bridge_init
               (theta_0, the same 64 val questions).
  cue          Sections 4.3 (Table 4.5 note), 4.6: R->A generations missing the "Answer:"
               cue (appended by the scorer, Section 3.5).
  explanations Section 4.6, Table 4.8: R->A explanation statistics pooled over seeds.
  examples     Figure 4.7: AO vs R->A disagreements (0.5B, all data, seed 0).
  laptop       Tables 4.8 (A->R), 4.9, 4.12, 4.15 (LoRA), 4.16: seed mean ± sample SD.
  compute      Section 4.9, Table C.1: cloud jobs and GPU-hours by phase and GPU type.

    python experiments/analysis/report_numbers.py [--sections overlap,splits,...] [--no-save]

Inputs: data/raw/aokvqa/aokvqa_v1p0_{train,val}.json and the run folders. parity, cue,
explanations, examples and the LoRA rows need predictions.jsonl (release asset); without them
those lines are skipped. explanations and examples reuse the text helpers of
experiments/laptop/explanation_checks.py. Saves results/analysis/report_numbers.txt unless
--no-save.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.data.aokvqa import AOKVQAItem, dev_split, load_aokvqa  # noqa: E402
from cmr.eval.prompts import _WORD, STOPWORDS, content_stems  # noqa: E402

SECTIONS = (
    "overlap",
    "splits",
    "giveaways",
    "parity",
    "cue",
    "explanations",
    "examples",
    "laptop",
    "compute",
)
LINES: list[str] = []
TALLY = {"ok": 0, "differs": 0}


class Missing(Exception):
    """A run file needed by a section is not present (predictions are a release asset)."""


# ----------------------------------------------------------------------------- printing
def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def _decimals(report: str) -> tuple[int, bool]:
    """Decimals and thousands separator of the first number in a report string.

    >>> _decimals("1,008"), _decimals("0.003 nats"), _decimals("85%")
    ((0, True), (3, False), (0, False))
    """
    m = re.search(r"\d[\d,]*(?:\.(\d+))?", report)
    if not m:
        raise ValueError(report)
    return (len(m.group(1)) if m.group(1) else 0), "," in m.group(0)


def fmt_like(report: str, value: float) -> str:
    """``value`` formatted with the decimals (and separator) of ``report``.

    >>> fmt_like("1.7%", 1.4847), fmt_like("15,532", 15532)
    ('1.5', '15,532')
    """
    d, comma = _decimals(report)
    return f"{value:,.{d}f}" if comma else f"{value:.{d}f}"


def _report_number(report: str) -> float:
    """The first number of a report string, sign included.

    >>> _report_number("about 1,008 images"), _report_number("\u22123.4")
    (1008.0, -3.4)
    """
    m = re.search(r"[-\u2212]?\d[\d,]*(?:\.\d+)?", report)
    return float(m.group(0).replace(",", "").replace("\u2212", "-"))


def check(label: str, report: str, value: float, note: str = "") -> bool:
    """Print and tally one comparison of a report value with the recomputed value."""
    got = fmt_like(report, value)
    ok = got == fmt_like(report, _report_number(report))
    TALLY["ok" if ok else "differs"] += 1
    diff = value - _report_number(report)
    verdict = "ok" if ok else f"DIFFERS by {diff:+.{_decimals(report)[0] + 1}f}"
    say(f"  {label}: {report} -> {got}   {verdict}" + (f"   ({note})" if note else ""))
    return ok


def check_pm(label: str, report: str, values: Sequence[float]) -> None:
    """Compare a report "mean ± SD" with the seeds' values (sample SD); note 0.1-rounding."""
    mean_r, sd_r = (s.strip() for s in report.split("±"))
    mean, sd = statistics.mean(values), statistics.stdev(values)
    got = f"{fmt_like(mean_r, mean)} ± {fmt_like(sd_r, sd)}"
    ok = got == f"{mean_r} ± {sd_r}"
    note = ""
    if not ok:
        r = [round(v, 1) for v in values]
        alt = f"{fmt_like(mean_r, statistics.mean(r))} ± {fmt_like(sd_r, statistics.stdev(r))}"
        if alt == f"{mean_r} ± {sd_r}":
            note = "   (reproduces when each seed is first rounded to 0.1)"
    TALLY["ok" if ok else "differs"] += 1
    seeds = ", ".join(f"{v:.2f}" for v in values)
    say(f"  {label}: {report} -> {got}   {'ok' if ok else 'DIFFERS'}{note}   [seeds {seeds}]")


def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else float("nan")


# ----------------------------------------------------------------------------- loading
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise Missing(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> Any:
    if not path.exists():
        raise Missing(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name)
    return json.loads(path.read_text())


def eval_records(run: Path, condition: str | None = None) -> list[dict[str, Any]]:
    """Rotation-0, no-context records of a trained run's eval_val/predictions.jsonl."""
    recs = read_jsonl(run / "eval_val" / "predictions.jsonl")
    return [
        r
        for r in recs
        if r.get("rotation", 0) == 0
        and r.get("context", "none") == "none"
        and (condition is None or r["condition"] == condition)
    ]


def summary_rows(run: Path) -> dict[str, dict[str, Any]]:
    """condition -> row of eval_val/summary.json (or summary.json), context none."""
    path = run / "eval_val" / "summary.json"
    rows = read_json(path if path.exists() else run / "summary.json")["rows"]
    return {r["condition"]: r for r in rows if r.get("context", "none") == "none"}


def explanation_helpers() -> Any:
    """experiments/laptop/explanation_checks.py as a module (its torch imports are lazy)."""
    path = ROOT / "experiments" / "laptop" / "explanation_checks.py"
    spec = importlib.util.spec_from_file_location("explanation_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- sections
def sec_overlap(val: list[AOKVQAItem], **_: Any) -> None:
    say(
        "Section 3.2 (point 'the questions themselves do not [state the answer]'): content words "
        "= cmr.eval.prompts.content_stems"
    )
    n = len(val)
    gold = dis = rule = 0.0
    for it in val:
        q, g = content_stems(it.question), it.correct_choice_idx
        overlap = [len(content_stems(c) & q) for c in it.choices]
        others = [j for j in range(len(it.choices)) if j != g]
        gold += overlap[g] > 0
        dis += sum(overlap[j] > 0 for j in others) / len(others)
        best = [j for j, v in enumerate(overlap) if v == max(overlap)]
        rule += (g in best) / len(best)
    note = "content words = cmr.eval.prompts.content_stems"
    check("content word of the correct option in the question", "1.5%", pct(gold, n), note)
    check("same for a wrong option, on average", "2.2%", pct(dis, n), note)
    check("most-shared-content-words rule (ties split evenly)", "24.7%", pct(rule, n), note)


def sec_splits(train: list[AOKVQAItem], val: list[AOKVQAItem], **_: Any) -> None:
    say("Table 3.2: questions / images per split (dev = dev_split, dev_fraction 0.06, seed 0)")
    pool, dev = dev_split(train, dev_fraction=0.06, seed=0)

    def images(items: Iterable[AOKVQAItem]) -> int:
        return len({it.image_id for it in items})

    for name, items, nq, ni in (
        ("train", train, "17,056", "16,540"),
        ("dev", dev, "1,032", "1,008"),
        ("training pool", pool, "16,024", "15,532"),
        ("4k set", pool[:4000], "4,000", "3,970"),
        ("val", val, "1,145", "1,122"),
    ):
        check(f"{name} questions", nq, len(items))
        check(f"{name} images", ni, images(items))
    check("dev share of train images (%)", "6.1%", pct(images(dev), images(train)))
    shared = {it.image_id for it in dev} & {it.image_id for it in pool}
    check("images in both dev and the pool", "0", len(shared))


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS}


def sec_giveaways(val: list[AOKVQAItem], **_: Any) -> None:
    say(
        "Section 5.4: a content word of the question appears in exactly one option "
        "(library words minus stop words, exact match)"
    )
    hits = correct = 0
    for it in val:
        q = _words(it.question)
        having = [j for j, c in enumerate(it.choices) if _words(c) & q]
        if len(having) == 1:
            hits += 1
            correct += having[0] == it.correct_choice_idx
    check("questions with such a word", "40", hits, f"of {len(val)}")
    check("share of val questions", "3.5%", pct(hits, len(val)))
    check("of these, the option is correct", "7", correct)
    check("expected by chance (one in four)", "10", hits / 4)


def sec_parity(cloud: Path, laptop: Path, **_: Any) -> None:
    say(
        "Section 4.1: GPU check run (G05_gate64) vs CPU run (pilot_a2v2_bridge_init), "
        "untrained 0.5B bridge"
    )

    def key(r: dict[str, Any]) -> tuple[str, str, str, int]:
        return r["question_id"], r["condition"], r["context"], int(r["rotation"])

    gpu = read_jsonl(cloud / "G05_gate64" / "predictions.jsonl")
    cpu = {key(r): r for r in read_jsonl(laptop / "pilot_a2v2_bridge_init" / "predictions.jsonl")}
    per: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for r in gpu:
        c = cpu[key(r)]
        d = max(abs(a - b) for a, b in zip(r["option_logprobs"], c["option_logprobs"], strict=True))
        per[r["condition"]].append((d, r["pred_idx"] != c["pred_idx"]))
    real = per["real"]
    check("questions", "64", len({r["question_id"] for r in gpu}))
    check("real-image predictions (4 option orders)", "256", len(real))
    check("median |delta log p|, real image (nats)", "0.003", statistics.median(d for d, _ in real))
    check("max |delta log p|, real image (nats)", "0.07", max(d for d, _ in real))
    check("prediction flips, real image", "1", sum(f for _, f in real))
    for cond in sorted(set(per) - {"real"}):
        rows = per[cond]
        check(
            f"prediction flips, text-only ({cond} image)",
            "0",
            sum(f for _, f in rows),
            f"{len(rows)} predictions, max |delta| {max(d for d, _ in rows):.1e} nats",
        )


def sec_cue(cloud: Path, **_: Any) -> None:
    say('Section 4.6 / Section 3.5 notes: "Answer:" cue not written by the model (real images)')
    misses: dict[str, int] = {}
    for run in sorted(cloud.glob("G*_rationale_then_answer*")):
        recs = eval_records(run, "real")
        misses[run.name] = sum(not r["cue_found"] for r in recs)

    def group(pattern: str) -> dict[str, int]:
        return {k: v for k, v in misses.items() if re.match(pattern, k)}

    main = group(r"G(05|7)_(B|BAL)_")
    check(
        f"most misses in a trained main or balanced-loss R->A run ({len(main)} runs)",
        "3",
        max(main.values()),
        "report: 'all but at most three'",
    )
    low = group(r"G05_HP_rationale_then_answer_rw0(\.1)?_s")
    shares = [pct(v, 1145) for v in low.values()]
    check(
        "lambda = 0 and 0.1 (two seeds each): lowest share of misses", "81%", min(shares), str(low)
    )
    check("lambda = 0 and 0.1: highest share of misses", "100%", max(shares))
    other = {k: v for k, v in group(r"G05_HP_").items() if k not in low and v > 3}
    say(f"  note: other hyperparameter-study R->A runs with more than three misses: {other}")


def _explanation_stats(
    ec: Any, runs: list[Path], cond: str, items: dict[str, AOKVQAItem]
) -> dict[str, float]:
    """Pooled statistics of the R->A explanations of ``runs`` under one image condition."""
    s: dict[str, float] = defaultdict(float)
    for run in runs:
        for r in eval_records(run, cond):
            it = items[r["question_id"]]
            e = ec.clean_ra(r.get("generation", ""))
            names = ec.named_options(e, list(it.choices))
            g = it.correct_choice_idx
            words = len(re.findall(r"[A-Za-z0-9']+", e))
            sub = it.choices[g].lower() in e.lower()  # Table 4.8: "anywhere in the text"
            ok = r["pred_idx"] == r["correct_idx"]
            h, refs = ec.tok(e), [ec.tok(x) for x in it.rationales]
            s["n"] += 1
            s["xy"] += bool(ec.THE_X_IS_Y.match(e))
            s["words"] += words
            s["any"] += bool(names)
            s["wrong_only"] += bool(names) and g not in names
            s["whole"] += g in names
            s["sub"] += sub
            s["ok_sub"] += ok and sub
            s["ok_not"] += ok and not sub
            if h:
                s["rl3"] += ec.rouge_l(h, refs)
                s["rl2"] += ec.rouge_l_loo(h, refs)
    n = s["n"]
    out = {k: pct(v, n) for k, v in s.items() if k not in ("n", "words", "ok_sub", "ok_not")}
    out.update(
        n=n,
        words=s["words"] / n,
        acc_if=pct(s["ok_sub"], s["sub"]),
        acc_not=pct(s["ok_not"], n - s["sub"]),
    )
    return out


def sec_explanations(cloud: Path, val: list[AOKVQAItem], **_: Any) -> None:
    say("Section 4.6 and Table 4.8: R->A explanations, seeds pooled")
    ec = explanation_helpers()
    items = {it.question_id: it for it in val}
    groups = {
        "0.5B": [cloud / f"G05_B_rationale_then_answer_full_s{s}" for s in range(5)],
        "7B": [cloud / f"G7_B_rationale_then_answer_n4000_s{s}" for s in range(3)],
    }
    st = {
        (g, c): _explanation_stats(ec, runs, c, items)
        for g, runs in groups.items()
        for c in ("real", "swapped")
    }
    a, b = st["0.5B", "real"], st["7B", "real"]
    say(f"  ({a['n']:.0f} and {b['n']:.0f} explanations per image condition)")
    xy_note = "regex THE_X_IS_Y of explanation_checks.py"
    check('0.5B "The X is Y." share', "90%", a["xy"], xy_note)
    check('7B "The X is Y." share', "85%", b["xy"], xy_note)
    check("0.5B mean words", "6.5", a["words"])
    rows = (
        ("0.5B", "real", "53.0", "47.3", "96.5", "39.8", "37.9%"),
        ("0.5B", "swapped", "50.5", "45.2", "95.1", "18.8", "20.3%"),
        ("7B", "real", "53.8", "48.1", "99.0", "58.1", "47.8%"),
        ("7B", "swapped", "50.0", "44.9", "96.5", "24.3", "25.9%"),
    )
    for g, c, rl3, rl2, acc_if, acc_not, share in rows:
        x = st[g, c]
        check(f"{g} {c}: ROUGE-L, three references", rl3, x["rl3"])
        check(f"{g} {c}: ROUGE-L, two references (leave one out)", rl2, x["rl2"])
        check(f"{g} {c}: accuracy if the explanation names the answer", acc_if, x["acc_if"])
        check(f"{g} {c}: accuracy otherwise", acc_not, x["acc_not"])
        check(f"{g} {c}: share naming the answer (substring)", share, x["sub"])
    check("0.5B real: share naming the answer as whole words", "35.9%", a["whole"])
    for g, c, any_, wrong in (
        ("0.5B", "real", "58%", "22%"),
        ("0.5B", "swapped", "56%", "36%"),
        ("7B", "real", "56%", "12%"),
        ("7B", "swapped", "55%", "30%"),
    ):
        check(f"{g} {c}: names at least one option", any_, st[g, c]["any"])
        check(f"{g} {c}: names only wrong options", wrong, st[g, c]["wrong_only"])
    # Human rationales: each one as the hypothesis, the other rationale(s) as references.
    n = words = any_ = wrong = rl = rl_first = 0.0
    for it in val:
        refs = [ec.tok(x) for x in it.rationales]
        rl_first += ec.rouge_l(refs[0], refs[1:])
        for i, text in enumerate(it.rationales):
            names = ec.named_options(text, list(it.choices))
            n += 1
            words += len(re.findall(r"[A-Za-z0-9']+", text))
            any_ += bool(names)
            wrong += bool(names) and it.correct_choice_idx not in names
            rl += ec.rouge_l(refs[i], refs[:i] + refs[i + 1 :])
    check("human rationales: mean words", "11.1", words / n)
    check("human rationales: name at least one option", "38%", pct(any_, n))
    check("human rationales: name only wrong options", "3%", pct(wrong, n))
    check(
        "one human rationale against the other two: ROUGE-L",
        "36.2",
        pct(rl, n),
        f"{pct(rl, n):.2f}, averaged over the three rationales; first rationale only: "
        f"{pct(rl_first, len(val)):.2f}",
    )


def sec_examples(cloud: Path, val: list[AOKVQAItem], **_: Any) -> None:
    say("Figure 4.7: AO vs R->A, 0.5B, all training data, seed 0, real images")
    ec = explanation_helpers()
    items = {it.question_id: it for it in val}
    ao = {r["question_id"]: r for r in eval_records(cloud / "G05_B_answer_only_full_s0", "real")}
    ra = {
        r["question_id"]: r
        for r in eval_records(cloud / "G05_B_rationale_then_answer_full_s0", "real")
    }

    def right(r: dict[str, Any]) -> bool:
        return r["pred_idx"] == r["correct_idx"]

    ao_only = [q for q in ra if right(ao[q]) and not right(ra[q])]
    named = {
        q: ec.named_options(ec.clean_ra(ra[q]["generation"]), list(items[q].choices))
        for q in ao_only
    }
    only_pick = sum(named[q] == {ra[q]["pred_idx"]} for q in ao_only)
    with_pick = sum(ra[q]["pred_idx"] in named[q] for q in ao_only)
    check("AO right, R->A wrong", "181", len(ao_only))
    check(
        "of these, the explanation names the option R->A picks (and no other)",
        "93",
        only_pick,
        f"{with_pick} if other options may be named too",
    )
    check("R->A right, AO wrong", "109", sum(right(ra[q]) and not right(ao[q]) for q in ra))


def sec_laptop(laptop: Path, **_: Any) -> None:
    final = laptop / "mac_final"
    say(
        "Table 4.8 A->R rows and Section 4.6 A->R prose: mean of seeds 0-2 "
        "(mac_final/atr_full_s*, atrbal_full_s*)"
    )
    atr = {
        k: [read_json(final / f"{k}_full_s{s}" / "summary.json")["conditions"] for s in range(3)]
        for k in ("atr", "atrbal")
    }

    def mean(kind: str, cond: str, key: str) -> float:
        return statistics.mean(s[cond][key] for s in atr[kind])

    for kind, cond, rl3, acc_if, acc_not, share, rl2 in (
        ("atr", "real", "53.4", "95.7", "39.0", "40.2%", "47.5"),
        ("atr", "swapped", "50.9", "93.2", "16.5", "22.2%", "45.3"),
        ("atrbal", "real", "50.1", "91.3", "50.7", "39.0%", "44.9"),
        ("atrbal", "swapped", "47.4", "78.9", "23.9", "19.8%", "42.7"),
    ):
        lab = f"{'default' if kind == 'atr' else 'balanced'} {cond}"
        check(f"{lab}: ROUGE-L, three references", rl3, mean(kind, cond, "rougeL_best3"))
        check(
            f"{lab}: accuracy if it names the answer",
            acc_if,
            mean(kind, cond, "acc_if_names_correct"),
        )
        check(f"{lab}: accuracy otherwise", acc_not, mean(kind, cond, "acc_otherwise"))
        check(f"{lab}: share naming the answer", share, mean(kind, cond, "names_correct"))
        check(f"{lab}: ROUGE-L, two references", rl2, mean(kind, cond, "rougeL_loo2"))
    for kind, cond, key, rep in (
        ("atr", "real", "the_x_is_y", "89%"),
        ("atr", "real", "words", "6.1"),
        ("atr", "real", "names_any", "66.7%"),
        ("atr", "real", "letter_acc", "61.7%"),
        ("atr", "swapped", "letter_acc", "33.5%"),
        ("atrbal", "real", "letter_acc", "66.6%"),
        ("atrbal", "real", "names_any", "59.5%"),
        ("atrbal", "real", "the_x_is_y", "79%"),
    ):
        check(f"{kind} {cond}: {key}", rep, mean(kind, cond, key))
    for kind, rep in (("atr", "95%"), ("atrbal", "84%")):
        chosen, any_ = mean(kind, "real", "names_chosen"), mean(kind, "real", "names_any")
        check(
            f"{kind} real: named option is the chosen one (share of those naming one)",
            rep,
            pct(chosen, any_),
            f"{pct(chosen, any_):.2f}; from the means rounded to 0.1: "
            f"{pct(round(chosen, 1), round(any_, 1)):.2f}",
        )

    say("Table 4.9: AO with the R->A explanation as context (mac_final/steer_full_s*)")
    steer = [read_json(final / f"steer_full_s{s}" / "summary.json")["groups"] for s in range(3)]

    def col(group: str, key: str) -> list[float]:
        return [g[group][key] for g in steer]

    for group, share, ao, ao_e, ra in (
        ("names_correct", "35.8%", "79.9 ± 1.0", "96.7 ± 1.0", "97.0 ± 0.2"),
        ("names_wrong_only", "21.6%", "40.4 ± 0.7", "10.0 ± 0.5", "1.6 ± 0.4"),
        ("names_none", "42.6%", "71.2 ± 1.0", "71.0 ± 0.4", "61.6 ± 1.6"),
        ("all", "100%", "67.7 ± 0.5", "67.0 ± 0.5", "61.3 ± 0.7"),
    ):
        check(
            f"{group}: share of questions",
            share,
            statistics.mean(pct(n, steer[0]["all"]["n"]) for n in col(group, "n")),
        )
        check_pm(f"{group}: AO", ao, col(group, "ao_plain_acc"))
        check_pm(f"{group}: AO + explanation", ao_e, col(group, "ao_with_expl_acc"))
        check_pm(f"{group}: R->A", ra, col(group, "ra_acc"))
    check_pm(
        "names_wrong_only: AO picks a named option",
        "85.5 ± 0.8",
        col("names_wrong_only", "ao_with_expl_picks_named"),
    )
    ao_all, aoe_all, ra_all = (
        statistics.mean(col("all", k)) for k in ("ao_plain_acc", "ao_with_expl_acc", "ra_acc")
    )
    none_share = statistics.mean(pct(n, 1145) for n in col("names_none", "n")) / 100
    none_gap = statistics.mean(col("names_none", "ao_with_expl_acc")) - statistics.mean(
        col("names_none", "ra_acc")
    )
    check("R->A behind AO (pp)", "6.4", ao_all - ra_all)
    check("  of which from the explanation as read by AO (pp)", "0.7", ao_all - aoe_all)
    check("  of which from the R->A bridge (pp)", "5.7", aoe_all - ra_all)
    check("  of which on names-none questions (pp)", "4.0", none_share * none_gap)
    check("names-none: R->A below AO given the explanation (pp)", "9.4", none_gap)

    say("Table 4.12: fewer image tokens (mac_AB, AO, 4k set, shuffled options)")
    # Run names (experiments/laptop/run_mac_AB.sh): "pilot" = with torch.compile and CPU-GPU
    # batch splitting (Section 3.8.2); "kauto" = CPU share set from measured step times; the
    # pilot runs without "kauto" used a fixed CPU share of 5.
    ab = laptop / "mac_AB"
    for label, runs, acc, sw, perm in (
        (
            "729 tokens",
            ["AO_shuf_n4000_s0", "AO_shuf_n4000_s1"],
            "61.5 ± 0.1",
            "33.6 ± 0.1",
            "64.4 ± 0.2",
        ),
        (
            "196 tokens",
            [f"AO_shuf_tok196_pilot_kauto_n4000_s{s}" for s in range(3)],
            "57.8 ± 0.8",
            "33.4 ± 0.5",
            "61.3 ± 0.1",
        ),
        (
            "81 tokens",
            [
                "AO_shuf_tok81_pilot_n4000_s0",
                "AO_shuf_tok81_pilot_kauto_n4000_s1",
                "AO_shuf_tok81_pilot_kauto_n4000_s2",
            ],
            "54.2 ± 0.2",
            "32.9 ± 0.3",
            "57.8 ± 0.8",
        ),
    ):
        rows = [summary_rows(ab / r) for r in runs]
        check_pm(f"{label}: accuracy", acc, [100 * r["real"]["acc"] for r in rows])
        check_pm(f"{label}: swapped", sw, [100 * r["swapped"]["acc"] for r in rows])
        check_pm(f"{label}: 4-order average", perm, [100 * r["real"]["perm_acc"] for r in rows])
    for run, rep in (
        ("AO_shuf_tok196_pilot_kauto_n4000_s0", "58.4"),
        ("AO_shuf_tok196_n4000_s0", "58.5"),
    ):
        check(
            f"196 tokens seed 0, {'with' if 'kauto' in run else 'without'} speed-ups",
            rep,
            100 * summary_rows(ab / run)["real"]["acc"],
        )

    say("Table 4.15: LoRA rows (mac_extra/C_*lora16_n4000_s0)")
    extra = laptop / "mac_extra"
    acc: dict[str, dict[str, float]] = {}
    for target in ("answer_only", "answer_then_rationale", "rationale_then_answer"):
        run = extra / f"C_{target}_lora16_n4000_s0"
        try:  # recomputed from the predictions when present, else read from summary.json
            acc[target] = {
                c: pct(sum(r["pred_idx"] == r["correct_idx"] for r in rs), len(rs))
                for c in ("real", "swapped")
                for rs in [eval_records(run, c)]
            }
            source = "predictions"
        except Missing:
            acc[target] = {c: 100 * row["acc"] for c, row in summary_rows(run).items()}
            source = "summary.json (predictions not found)"
    say(f"  (accuracies from {source})")
    for target, real, sw, gap, delta in (
        ("answer_only", "64.7", "35.7", "29.0", None),
        ("answer_then_rationale", "61.3", "34.8", "26.5", "-3.4"),
        ("rationale_then_answer", "61.3", "34.6", "26.7", "-3.4"),
    ):
        a = acc[target]
        check(f"{target}: real", real, a["real"])
        check(f"{target}: swapped", sw, a["swapped"])
        check(f"{target}: gap", gap, a["real"] - a["swapped"])
        if delta:
            check(
                f"{target}: difference from LoRA AO", delta, a["real"] - acc["answer_only"]["real"]
            )

    say(
        "Table 4.16: transfer (mac_transfer, mac_final; DA bridges mac_AB/B_direct_answer_n4000_s*)"
    )
    tr = laptop / "mac_transfer"
    fin = laptop / "mac_final"

    def scores(run: Path, key: str) -> tuple[float, float]:
        rows = summary_rows(run)
        return 100 * rows["real"][key], 100 * rows["swapped"][key]

    for label, runs, key, real, sw, gap in (
        ("direct answers, untrained bridge", [tr / "da_theta0"], "score", "21.5", "4.7", "16.7"),
        (
            "direct answers, AO bridge",
            [tr / f"da_ao_s{s}" for s in range(3)],
            "score",
            "24.3 ± 0.3",
            "6.2 ± 0.1",
            "18.1 ± 0.3",
        ),
        (
            "direct answers, DA bridge",
            [ab / f"B_direct_answer_n4000_s{s}" for s in range(3)],
            "score",
            "29.5 ± 1.0",
            "9.5 ± 0.1",
            "20.0 ± 1.1",
        ),
        ("MMStar, untrained bridge", [tr / "mmstar_theta0"], "acc", "33.8", "26.5", "7.3"),
        (
            "MMStar, AO bridge",
            [tr / f"mmstar_ao_s{s}" for s in range(3)],
            "acc",
            "35.6 ± 0.1",
            "26.7 ± 0.5",
            "8.9 ± 0.6",
        ),
        (
            "7B direct answers, untrained bridge",
            [fin / "7b_da_theta0"],
            "score",
            "38.5",
            "6.8",
            "31.8",
        ),
        (
            "7B direct answers, AO bridge",
            [fin / "7b_da_aofull_s0"],
            "score",
            "50.5",
            "11.4",
            "39.1",
        ),
        ("7B MMStar, untrained bridge", [fin / "7b_mmstar_theta0"], "acc", "36.5", "27.5", "9.0"),
        ("7B MMStar, AO bridge", [fin / "7b_mmstar_aofull_s0"], "acc", "37.5", "26.3", "11.2"),
    ):
        sc = [scores(r, key) for r in runs]
        for what, rep, vals in (
            ("real", real, [s[0] for s in sc]),
            ("swapped", sw, [s[1] for s in sc]),
            ("gap", gap, [s[0] - s[1] for s in sc]),
        ):
            if "±" in rep:
                check_pm(f"{label}: {what}", rep, vals)
            else:
                check(f"{label}: {what}", rep, vals[0])
    check(
        "MMStar chance (uniform guess)",
        "26.9",
        100 * read_json(tr / "mmstar_theta0/summary.json")["transfer"]["chance_acc"],
    )
    check_pm(
        "AO bridges on A-OKVQA multiple choice",
        "61.0 ± 0.2",
        [
            100 * summary_rows(extra / f"B_answer_only_n4000_s{s}_mps")["real"]["acc"]
            for s in range(3)
        ],
    )


def sec_compute(cloud: Path, **_: Any) -> None:
    say("Section 4.9 and Table C.1: cloud jobs and GPU-hours (modal_job.json)")
    jobs = {p.parent.name: read_json(p) for p in sorted(cloud.glob("*/modal_job.json"))}
    phase = {}
    for p in sorted(cloud.glob("_status_phase*.json")):
        d = read_json(p)
        phase.update({j["name"]: d["phase"] for j in d["jobs"]})
    unassigned = sorted(set(jobs) - set(phase))
    if unassigned:
        say(f"  note: jobs in no phase file: {unassigned}")
    check("cloud jobs", "109", len(jobs))
    check("GPU-hours", "77.6", sum(j["seconds"] for j in jobs.values()) / 3600)
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for name, j in jobs.items():
        by[phase.get(name, "?"), "all"].append(j["seconds"])
        by["all", j["gpu_name"]].append(j["seconds"])
        by[phase.get(name, "?"), j["gpu_name"]].append(j["seconds"])
    for ph, n, h in (
        ("1", "39", "26.4"),
        ("2", "14", "15.9"),
        ("hp", "46", "25.4"),
        ("bal", "10", "9.9"),
    ):
        check(f"phase {ph}: jobs", n, len(by[ph, "all"]))
        check(f"phase {ph}: GPU-hours", h, sum(by[ph, "all"]) / 3600)
        gpus = {
            g.removeprefix("NVIDIA "): len(v) for (p, g), v in by.items() if p == ph and g != "all"
        }
        say(f"    GPU types in phase {ph}: {gpus}")
    for gpu, n, h in (("NVIDIA L4", "86", "55.3"), ("NVIDIA L40S", "23", "22.3")):
        check(f"{gpu}: jobs", n, len(by["all", gpu]))
        check(f"{gpu}: GPU-hours", h, sum(by["all", gpu]) / 3600)


# ----------------------------------------------------------------------------- main
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--aokvqa-dir", default="data/raw/aokvqa")
    p.add_argument("--cloud-runs", default="results/runs/cloud")
    p.add_argument("--laptop-runs", default="results/runs/laptop")
    p.add_argument("--sections", default=",".join(SECTIONS), help="comma list; default all")
    p.add_argument("--no-save", action="store_true", help="do not write results/analysis/")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    def under_root(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else ROOT / path

    wanted = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = sorted(set(wanted) - set(SECTIONS))
    if unknown:
        raise SystemExit(f"unknown sections {unknown}; choose from {list(SECTIONS)}")
    aok = under_root(args.aokvqa_dir)
    val = [it for it in load_aokvqa(aok, "val") if it.has_labels]
    train = load_aokvqa(aok, "train") if "splits" in wanted else []
    ctx = {
        "val": val,
        "train": train,
        "cloud": under_root(args.cloud_runs),
        "laptop": under_root(args.laptop_runs),
    }
    funcs: dict[str, Callable[..., None]] = {
        "overlap": sec_overlap,
        "splits": sec_splits,
        "giveaways": sec_giveaways,
        "parity": sec_parity,
        "cue": sec_cue,
        "explanations": sec_explanations,
        "examples": sec_examples,
        "laptop": sec_laptop,
        "compute": sec_compute,
    }
    say("Report numbers: report value -> recomputed value")
    for name in wanted:
        say()
        say(f"== {name} ==")
        try:
            funcs[name](**ctx)
        except Missing as e:
            say(f"  skipped (rest of section): {e} not found (predictions: release asset)")
    say()
    say(f"{TALLY['ok']} numbers reproduce, {TALLY['differs']} differ")
    if not args.no_save:
        out = ROOT / "results" / "analysis" / "report_numbers.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
