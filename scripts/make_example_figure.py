"""Figure 4.7: each target's output on two val questions (0.5B, all data, seed 0, default loss).

Needs predictions.jsonl of the cloud runs G05_B_*_full_s0 (release asset runs.tar.gz,
scripts/download_release.sh), the A-OKVQA val JSON and the COCO val2017 images
(scripts/download_data.sh). Also prints the three counts in the figure's caption: questions
that AO gets right and R->A wrong, how many of those have an R->A explanation that names only
the option R->A picks (whole-word match), and questions that R->A gets right and AO wrong.

    python scripts/make_example_figure.py      # -> results/figures/fig4_7_target_examples.png
"""

import json, math, re, textwrap
import matplotlib.pyplot as plt
from make_figures import img, save, AO, ATR, RTA, INK, INK2

from make_figures import IMG, RUNS, VAL

V = {v["question_id"]: v for v in json.load(open(VAL))}
GOOD, BAD = "#1a7f37", "#c62828"


def load(run):
    d = {}
    for line in open(f"{RUNS}{run}/eval_val/predictions.jsonl"):
        x = json.loads(line)
        if (
            x["condition"] == "real"
            and x.get("context", "none") == "none"
            and x.get("rotation", 0) == 0
        ):
            d[x["question_id"]] = x
    return d


P = {
    "AO": load("G05_B_answer_only_full_s0"),
    "A→R": load("G05_B_answer_then_rationale_full_s0"),
    "R→A": load("G05_B_rationale_then_answer_full_s0"),
}


def probs(rec):
    m = max(rec["option_logprobs"])
    e = [math.exp(v - m) for v in rec["option_logprobs"]]
    return [v / sum(e) for v in e]


def counts():
    has = lambda c, t: re.search(r"\b" + re.escape(c.lower()) + r"\b", t.lower()) is not None
    n_ao_only = n_ra_only = n_follow = 0
    for q, v in V.items():
        g = v["correct_choice_idx"]
        a, r = P["AO"][q], P["R→A"][q]
        if a["pred_idx"] == g and r["pred_idx"] != g:
            n_ao_only += 1
            hits = [i for i, o in enumerate(v["choices"]) if has(o, r["generation"])]
            n_follow += hits == [r["pred_idx"]]
        if r["pred_idx"] == g and a["pred_idx"] != g:
            n_ra_only += 1
    return n_ao_only, n_follow, n_ra_only


def fig_examples():
    items = [
        ("34ah8AmQenWUyH2fZvJH7a", "(a) All three systems correct"),
        ("2DQex53EkNGH2cfo3WPuPn", "(b) AO correct; R→A follows its own wrong explanation"),
    ]
    fig = plt.figure(figsize=(7.4, 5.6))
    for row, (q, title) in enumerate(items):
        v = V[q]
        g = v["correct_choice_idx"]
        top = 1 - row * 0.5
        a = fig.add_axes([0.0, top - 0.44, 0.27, 0.36])
        a.imshow(img(f"{IMG}{v['image_id']:012d}.jpg", (400, 300)))
        a.axis("off")
        t = fig.add_axes([0.29, top - 0.47, 0.71, 0.45])
        t.axis("off")
        t.set_xlim(0, 1)
        t.set_ylim(0, 1)
        t.text(0, 0.98, title, fontsize=8.5, fontweight="bold", va="top")
        t.text(0, 0.88, "Q: " + v["question"], fontsize=8, va="top", color=INK)
        opts = "   ".join(
            f"{'ABCD'[i]}. {c}" + (" ✓" if i == g else "") for i, c in enumerate(v["choices"])
        )
        t.text(0, 0.79, opts, fontsize=7.4, va="top", color=INK2)
        y = 0.66
        for arm, col in [("AO", AO), ("A→R", ATR), ("R→A", RTA)]:
            rec = P[arm][q]
            p = probs(rec)
            k = rec["pred_idx"]
            ok = k == g
            t.text(0, y, arm, fontsize=8, fontweight="bold", color=col, va="top")
            if arm == "R→A":
                gen = rec["generation"].strip()
                t.text(
                    0.11,
                    y,
                    "writes: “" + gen + "”",
                    fontsize=7.6,
                    va="top",
                    style="italic",
                    color=INK,
                )
                y -= 0.09
            elif arm == "A→R":
                t.text(
                    0.11,
                    y,
                    "answers first (no explanation needed for the score)",
                    fontsize=7.2,
                    va="top",
                    color=INK2,
                )
                y -= 0.09
            t.text(
                0.11,
                y,
                f"picks {'ABCD'[k]}. {v['choices'][k]}  (p = {p[k]:.2f})  " + ("✓" if ok else "✗"),
                fontsize=7.8,
                va="top",
                color=GOOD if ok else BAD,
                fontweight="bold",
            )
            y -= 0.12
        ri = max(range(3), key=lambda i: len(v["rationales"][i]))  # the longest of the three
        r = textwrap.shorten(v["rationales"][ri], 105, placeholder="…")
        t.text(
            0,
            y - 0.01,
            "Human rationale (longest of 3): " + r,
            fontsize=7.0,
            va="top",
            color=INK2,
            wrap=True,
        )
    save(fig, "fig4_7_target_examples.png")


if __name__ == "__main__":
    print("AO right & R->A wrong, R->A right & AO wrong:", counts())
    fig_examples()
    print("ok")
