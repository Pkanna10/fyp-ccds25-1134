"""Two checks on the explanations (evaluation only): Table 4.8 A->R rows and Table 4.9.

generate   An A->R bridge greedily writes "B\\nBecause: ..." (at most 64 tokens) for every val
           question, real and swapped image. Summary: letter accuracy, which options the
           explanation names (whole words), ROUGE-L against the human rationales (best of three,
           and against two with each left out in turn), share of "The X is Y." explanations.
steer      An AO bridge scores every val question with the real image, with and without the
           R->A explanation for that question as a "Context:" line: does the text steer it?
           Results are grouped by whether the explanation names the correct option, only wrong
           options, or no option (Table 4.9).

    python experiments/laptop/explanation_checks.py generate --projector P --out DIR
    python experiments/laptop/explanation_checks.py steer --projector P_AO \\
        --explanations RA_PRED.jsonl --out DIR

Each command writes <out>/predictions.jsonl, summary.json and summary.md. The recorded runs,
made by experiments/laptop/run_mac_final.sh with the cloud bridges (all data, seeds 0-2), are
results/runs/laptop/mac_final/{atr,atrbal,steer}_full_s*. --feature-cache (default
cache/features) reuses saved SigLIP outputs of the val images. Finished (question, condition)
records are skipped on a re-run. experiments/analysis/report_numbers.py imports the text
helpers below (tok, rouge_l, rouge_l_loo, clean_ra, named_options, THE_X_IS_Y).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

LETTERS = "ABCD"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ----------------------------------------------------------------- text helpers (no torch)
def tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", s.lower())


def lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(h: list[str], refs: list[list[str]]) -> float:
    best = 0.0
    for r in refs:
        n = lcs(h, r)
        if n:
            p, rc = n / len(h), n / len(r)
            best = max(best, 2 * p * rc / (p + rc))
    return best


def rouge_l_loo(h: list[str], refs: list[list[str]]) -> float:
    """ROUGE-L against the other references, averaged over each left-out reference."""
    if len(refs) < 2:
        return rouge_l(h, refs)
    return sum(rouge_l(h, refs[:i] + refs[i + 1 :]) for i in range(len(refs))) / len(refs)


def split_atr(gen: str) -> tuple[int | None, str]:
    """(letter index or None, explanation) from an A->R output such as "B\\nBecause: It rains."

    >>> split_atr("B\\nBecause: It is wet.")
    (1, 'It is wet.')
    >>> split_atr(" c. because the sky")
    (2, 'the sky')
    """
    g = gen.strip()
    m = re.match(r"^\(?([A-Da-d])\b", g)
    idx = LETTERS.index(m.group(1).upper()) if m else None
    parts = re.split(r"because\s*:?\s*", g, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        expl = parts[1]
    else:
        expl = g.split("\n", 1)[1] if "\n" in g else ""
    return idx, expl.split("\nAnswer")[0].strip()


def clean_ra(gen: str) -> str:
    """The explanation part of an R->A output ("Because: ...\\nAnswer: B")."""
    g = re.sub(r"^because\s*:?\s*", "", gen.strip(), flags=re.I)
    return g.split("\nAnswer")[0].split("Answer:")[0].strip()


def named_options(expl: str, choices: list[str]) -> set[int]:
    """Indices of the options whose full text appears in ``expl`` as whole words.

    >>> sorted(named_options("The animal is a cat.", ["cat", "dog", "catfish", "bird"]))
    [0]
    """
    e = expl.lower()
    out = set()
    for i, c in enumerate(choices):
        c = c.strip().lower()
        if c and re.search(r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])", e):
            out.add(i)
    return out


THE_X_IS_Y = re.compile(r"^the [^.]{1,60}? (is|are) [^.]{1,60}\.?$", re.I)


def pct(a: float, b: float) -> float | None:
    return round(100 * a / b, 1) if b else None


# ----------------------------------------------------------------- model helpers (torch)
def load(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from cmr.data.aokvqa import load_aokvqa
    from cmr.runners.stage_a import build_model

    items = [it for it in load_aokvqa(args.aokvqa_dir, "val") if it.correct_choice_idx is not None]
    model = build_model(f"bridge:{args.bridge}", args.device, args.dtype, projector=args.projector)
    if args.feature_cache:  # frozen SigLIP outputs saved by earlier runs (bit-identical)
        model.feature_cache_dir = Path(args.feature_cache)
        model.feature_cache_splits = frozenset({"val"})
    return items, model


def image_loader(coco_dir: str):  # type: ignore[no-untyped-def]
    from PIL import Image

    from cmr.data.aokvqa import coco_image_path

    def f(image_id: int):  # type: ignore[no-untyped-def]
        im = Image.open(coco_image_path(coco_dir, "val", image_id)).convert("RGB")
        im.info["cmr_key"] = f"val/{image_id}"  # the feature-cache key used by the runners
        return im

    return f


def done_keys(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    recs = [json.loads(line) for line in path.open()] if path.exists() else []
    return recs, {(r["question_id"], r["condition"]) for r in recs}


# ----------------------------------------------------------------- generate
def cmd_generate(args: argparse.Namespace) -> None:
    from cmr.eval.conditions import swap_assignment
    from cmr.models.base import Query
    from cmr.train.generate import greedy_generate
    from cmr.train.targets import TargetFormat, prompt_for

    items, model = load(args)
    load_img = image_loader(args.coco_dir)
    perm = swap_assignment([it.image_id for it in items], seed=0)  # all val items, load order
    order = list(range(len(items)))[: args.limit] if args.limit else range(len(items))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions.jsonl"
    recs, done = done_keys(pred)
    fmt = TargetFormat(args.target)
    t0, n = time.time(), 0
    with pred.open("a") as f:
        for i in order:
            it = items[i]
            text = prompt_for(it, fmt)
            for cond in args.conditions.split(","):
                if (it.question_id, cond) in done:
                    continue
                img_id = it.image_id if cond == "real" else items[perm[i]].image_id
                gen = greedy_generate(
                    model,
                    Query(text, load_img(img_id)),
                    prefix="",
                    max_new_tokens=args.max_new_tokens,
                    stop=None,
                )
                idx, _ = split_atr(gen)
                r = {
                    "question_id": it.question_id,
                    "condition": cond,
                    "generation": gen,
                    "pred_idx": idx,
                    "correct_idx": it.correct_choice_idx,
                }
                f.write(json.dumps(r) + "\n")
                f.flush()
                recs.append(r)
                n += 1
                if n % 50 == 0:
                    print(f"[gen] {n} new, {n / (time.time() - t0):.2f}/s", flush=True)
    summarize_generate(recs, {it.question_id: it for it in items}, out, args)


def summarize_generate(recs: list[dict], items: dict, out: Path, args: argparse.Namespace) -> None:
    rows = {}
    for cond in ("real", "swapped"):
        rs = [r for r in recs if r["condition"] == cond and r["question_id"] in items]
        if not rs:
            continue
        acc = corr_named = chosen_named = wrong_only = any_named = xy = 0
        acc_if, n_if, acc_not, n_not = 0, 0, 0, 0
        rl3 = rl2 = words = 0.0
        for r in rs:
            it = items[r["question_id"]]
            idx, expl = split_atr(r["generation"])
            names = named_options(expl, list(it.choices))
            ok = idx == it.correct_choice_idx
            acc += ok
            c = it.correct_choice_idx in names
            corr_named += c
            chosen_named += idx is not None and idx in names
            wrong_only += bool(names) and not c
            any_named += bool(names)
            xy += bool(THE_X_IS_Y.match(expl))
            if c:
                acc_if, n_if = acc_if + ok, n_if + 1
            else:
                acc_not, n_not = acc_not + ok, n_not + 1
            h = tok(expl)
            refs = [tok(x) for x in it.rationales]
            if h:
                rl3 += rouge_l(h, refs)
                rl2 += rouge_l_loo(h, refs)
            words += len(re.findall(r"[A-Za-z0-9']+", expl))
        n = len(rs)
        rows[cond] = {
            "n": n,
            "letter_acc": pct(acc, n),
            "names_correct": pct(corr_named, n),
            "names_chosen": pct(chosen_named, n),
            "names_wrong_only": pct(wrong_only, n),
            "names_any": pct(any_named, n),
            "acc_if_names_correct": pct(acc_if, n_if),
            "acc_otherwise": pct(acc_not, n_not),
            "the_x_is_y": pct(xy, n),
            "rougeL_best3": round(100 * rl3 / n, 1),
            "rougeL_loo2": round(100 * rl2 / n, 1),
            "words": round(words / n, 1),
        }
    summary = {
        "projector": args.projector,
        "target": args.target,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "conditions": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = [
        "n",
        "letter_acc",
        "names_correct",
        "names_chosen",
        "names_wrong_only",
        "names_any",
        "acc_if_names_correct",
        "acc_otherwise",
        "the_x_is_y",
        "rougeL_best3",
        "rougeL_loo2",
        "words",
    ]
    lines = [
        f"# A->R explanations: {args.projector}",
        "",
        "| measure | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    lines += [f"| {k} | " + " | ".join(str(rows[c][k]) for c in rows) + " |" for k in keys]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------- steer
def cmd_steer(args: argparse.Namespace) -> None:
    from cmr.eval.prompts import mc_prompt
    from cmr.models.base import Query

    items, model = load(args)
    load_img = image_loader(args.coco_dir)
    ra = {}
    for line in open(args.explanations):
        r = json.loads(line)
        if r.get("condition") == "real" and r.get("rotation", 0) == 0:
            ra[r["question_id"]] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions.jsonl"
    recs, done = done_keys(pred)
    sel = [it for it in items if it.question_id in ra]
    if args.limit:
        sel = sel[: args.limit]
    t0, n = time.time(), 0
    with pred.open("a") as f:
        for it in sel:
            expl = clean_ra(ra[it.question_id].get("generation", ""))
            img = None
            for cond, ctx in (("plain", None), ("with_ra_explanation", expl)):
                if (it.question_id, cond) in done:
                    continue
                if img is None:
                    img = load_img(it.image_id)
                lp = model.letter_logprobs(
                    Query(mc_prompt(it.question, list(it.choices), ctx), img), len(it.choices)
                )
                idx = max(range(len(lp)), key=lambda k: lp[k])
                r = {
                    "question_id": it.question_id,
                    "condition": cond,
                    "pred_idx": idx,
                    "correct_idx": it.correct_choice_idx,
                    "explanation": expl,
                    "ra_pred_idx": ra[it.question_id].get("pred_idx"),
                }
                f.write(json.dumps(r) + "\n")
                f.flush()
                recs.append(r)
                n += 1
                if n % 100 == 0:
                    print(f"[steer] {n} new, {n / (time.time() - t0):.2f}/s", flush=True)
    summarize_steer(recs, {it.question_id: it for it in items}, out, args)


def summarize_steer(recs: list[dict], items: dict, out: Path, args: argparse.Namespace) -> None:
    by = {}
    for r in recs:
        by.setdefault(r["question_id"], {})[r["condition"]] = r
    groups: dict[str, list[tuple[dict, dict, set[int], int]]] = {
        "all": [],
        "names_correct": [],
        "names_wrong_only": [],
        "names_none": [],
    }
    for qid, d in by.items():
        if "plain" not in d or "with_ra_explanation" not in d or qid not in items:
            continue
        it = items[qid]
        names = named_options(d["plain"]["explanation"], list(it.choices))
        g = (
            "names_correct"
            if it.correct_choice_idx in names
            else "names_wrong_only"
            if names
            else "names_none"
        )
        row = (d["plain"], d["with_ra_explanation"], names, it.correct_choice_idx)
        groups["all"].append(row)
        groups[g].append(row)
    rows = {}
    for g, rs in groups.items():
        n = len(rs)
        if not n:
            continue
        rows[g] = {
            "n": n,
            "ao_plain_acc": pct(sum(p["pred_idx"] == c for p, _, _, c in rs), n),
            "ao_with_expl_acc": pct(sum(w["pred_idx"] == c for _, w, _, c in rs), n),
            "ra_acc": pct(sum(p.get("ra_pred_idx") == c for p, _, _, c in rs), n),
            "ao_with_expl_picks_named": pct(sum(w["pred_idx"] in nm for _, w, nm, _ in rs), n),
            "ao_right_plain_wrong_with_expl": pct(
                sum(p["pred_idx"] == c and w["pred_idx"] != c for p, w, _, c in rs), n
            ),
        }
    (out / "summary.json").write_text(
        json.dumps(
            {"projector": args.projector, "explanations": args.explanations, "groups": rows},
            indent=1,
        )
    )
    keys = [
        "n",
        "ao_plain_acc",
        "ao_with_expl_acc",
        "ra_acc",
        "ao_with_expl_picks_named",
        "ao_right_plain_wrong_with_expl",
    ]
    lines = [
        f"# AO bridge with R->A explanations as context: {args.projector}",
        "",
        "| measure | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    lines += [f"| {k} | " + " | ".join(str(rows[g][k]) for g in rows) + " |" for k in keys]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "steer"):
        s = sub.add_parser(name)
        s.add_argument(
            "--projector",
            required=True,
            help="trained bridge projector.pt (generate: an A->R bridge; steer: an AO bridge)",
        )
        s.add_argument("--out", required=True, help="output folder")
        s.add_argument("--bridge", default="configs/ov_qwen2_05b_frozen.json")
        s.add_argument("--aokvqa-dir", default="data/raw/aokvqa")
        s.add_argument("--coco-dir", default="data/raw/coco")
        s.add_argument("--device", default="mps")
        s.add_argument("--dtype", default="float32")
        s.add_argument("--limit", type=int, default=None, help="first N questions only")
        s.add_argument(
            "--feature-cache",
            default="cache/features",
            help="frozen-tower feature cache ('' = off); val features are reused",
        )
        if name == "generate":
            s.add_argument("--target", default="answer_then_rationale")
            s.add_argument("--conditions", default="real,swapped")
            s.add_argument("--max-new-tokens", type=int, default=64)
        else:
            s.add_argument(
                "--explanations",
                required=True,
                help="eval_val/predictions.jsonl of the R->A run whose explanations are given",
            )
    args = p.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    else:
        cmd_steer(args)


if __name__ == "__main__":
    main()
