"""Report figures (PNG, 300 dpi) into results/figures/, one function per figure.

    python scripts/make_figures.py                     # all figures below
    python scripts/make_figures.py fig_main fig_hp     # only the named functions

What each figure needs (with no arguments the script stops at the first missing input):
    results/run_metrics.json (in the repository): 1.2 fig_teaser, 4.1 fig_main,
        4.2 fig_slopes, 4.3 fig_hp, 4.5 fig_scatter
    nothing (schematic): 2.1 fig_timeline, 3.3 fig_lossweights, 3.5 fig_blockattn
    numbers copied from Tables 4.8, 4.10 and 4.14 (written in below): 4.8 fig_explq,
        4.9 fig_cheap, 4.10 fig_time
    COCO val2017 images (scripts/download_data.sh): 1.1 fig_example (also the A-OKVQA val
        JSON), 3.4 fig_protocol, 4.6 fig_explanations
    train_log.jsonl of three cloud runs (release asset runs.tar.gz): 4.4 fig_curves

Figures 3.1 and 3.2 are drawn by make_pipeline_figures.py and Figure 4.7 by
make_example_figure.py; both import the helpers and colours defined here.
"""

import json, textwrap
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / "results/figures") + "/"
IMG = str(ROOT / "data/raw/coco/val2017") + "/"
RUNS = str(ROOT / "results/runs/cloud") + "/"
VAL = ROOT / "data/raw/aokvqa/aokvqa_v1p0_val.json"
C = json.load(open(ROOT / "results/run_metrics.json"))

# palette: one colour per target + neutrals
AO, ATR, RTA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#8a8986", "#0b0b0b", "#52514e", "#e4e3df"
FROZEN, TRAIN = "#eceae4", "#d6e6fa"
ARM = {"AO": AO, "A→R": ATR, "R→A": RTA}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def seeds(key, f="real", only=None):
    """Metric ``f`` of every seed of setting ``key`` in run_metrics.json; ``only``: seed numbers
    to keep (default all)."""
    return [v[f] for s, v in C[key].items() if only is None or int(s) in only]


def save(fig, name):
    fig.savefig(OUT + name)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=8.5, bold=False, ls="-", tc=INK, r=0.02):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=0.9, ls=ls
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )


def arrow(ax, x0, y0, x1, y1, color=INK2, ls="-", lw=1.1, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=9,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
        )
    )


def img(path, size=(400, 300)):
    return Image.open(path).convert("RGB").resize(size)


# ============================================================ Fig 1.1 dataset example + targets
def fig_example():
    V = {v["question_id"]: v for v in json.load(open(VAL))}
    it = V["34ah8AmQenWUyH2fZvJH7a"]
    fig = plt.figure(figsize=(7.2, 3.3))
    a = fig.add_axes([0.0, 0.08, 0.36, 0.84])
    a.imshow(img(IMG + "000000562818.jpg"))
    a.axis("off")
    a.set_title("Image (COCO val2017, id 562818)", fontsize=8, color=INK2)
    t = fig.add_axes([0.38, 0.0, 0.62, 1.0])
    t.axis("off")
    t.set_xlim(0, 1)
    t.set_ylim(0, 1)
    t.text(0, 0.95, "Question: " + it["question"], fontsize=9, fontweight="bold", va="top")
    for i, ch in enumerate(it["choices"]):
        good = i == it["correct_choice_idx"]
        t.text(
            0.02,
            0.84 - 0.075 * i,
            f"{'ABCD'[i]}.  {ch}" + ("   ✓ correct" if good else ""),
            fontsize=8.5,
            va="top",
            color=AO if good else INK,
            fontweight="bold" if good else "normal",
        )
    t.text(
        0,
        0.52,
        "Human rationale (1 of 3): " + it["rationales"][0],
        fontsize=8,
        va="top",
        color=INK2,
        wrap=True,
    )
    L = "ABCD"[it["correct_choice_idx"]]
    r = textwrap.shorten(it["rationales"][0], 46, placeholder="…")
    rows = [
        ("AO", f"{L}"),
        ("A→R", f"{L}  ⏎  Because: {r}"),
        ("R→A", f"Because: {r}  ⏎  Answer: {L}"),
    ]
    for k, (n, s) in enumerate(rows):
        y = 0.30 - 0.11 * k
        t.add_patch(
            FancyBboxPatch(
                (0, y - 0.035),
                0.12,
                0.075,
                boxstyle="round,pad=0,rounding_size=0.01",
                fc=ARM[n],
                ec="none",
            )
        )
        t.text(0.06, y, n, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        t.text(0.14, y, "target:  " + s, va="center", fontsize=8, family="DejaVu Sans Mono")
    t.text(
        0,
        0.40,
        "The three training targets (the loss is computed on the target only):",
        fontsize=8,
        va="top",
        color=INK2,
    )
    save(fig, "fig1_1_example.png")


# ============================================================ Fig 1.2 teaser: the main story
def fig_teaser():
    """Figure 1.2: each target's accuracy under the default and the balanced loss.

    Filled diamond = mean over seeds with the default loss (mean cross-entropy over all target
    tokens); open diamond = mean with the balanced loss CE(letter) + 0.3 * CE(rest); small dots
    = the individual seeds; dashed line = AO. The row under each panel gives the accuracy of
    the untrained bridge theta_0 under the same instruction, so the gain from training can be
    read off for every column.
    """
    cells = [
        ("0.5B, 16k items, seeds 0–2", "G05", "full", "G05_BAL_{}_full_rw0.3", (0, 1, 2)),
        ("7B, 4k items, seeds 0–1", "G7", "n4000", "G7_BAL_{}_n4000_rw0.3", (0, 1)),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    for ax, (title, g, d, bal, sd) in zip(axs, cells):
        # (tick label, run, colour, balanced loss?, target whose instruction the run uses)
        cols = [
            ("AO", f"{g}_B_answer_only_{d}", AO, False, "answer_only"),
            ("A→R\ndefault", f"{g}_B_answer_then_rationale_{d}", ATR, False, "answer_then_rationale"),
            ("A→R\nbalanced", bal.format("answer_then_rationale"), ATR, True, "answer_then_rationale"),
            ("R→A\ndefault", f"{g}_B_rationale_then_answer_{d}", RTA, False, "rationale_then_answer"),
            ("R→A\nbalanced", bal.format("rationale_then_answer"), RTA, True, "rationale_then_answer"),
        ]
        for i, (_, key, col, balanced, target) in enumerate(cols):
            v = seeds(key, only=sd)
            m = np.mean(v)
            ax.scatter(
                [i - 0.12], [m], s=46, marker="D", zorder=4, lw=1.4,
                color="white" if balanced else col, edgecolors=col,
            )  # fmt: skip
            ax.scatter([i + 0.14] * len(v), v, s=10, color=INK, zorder=3, lw=0)
            ax.text(i - 0.12, m + 0.7, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
            theta0 = C[f"{g}_theta0_{target}"]["-"]["real"]
            ax.text(
                i, -0.36, f"{theta0:.1f}", transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=7.5, color=INK2,
            )  # fmt: skip
        ax.text(
            -0.75, -0.36, "untrained\nbridge θ₀", transform=ax.get_xaxis_transform(),
            ha="right", va="center", fontsize=7, color=INK2,
        )  # fmt: skip
        ax.set_xticks(range(5), [c[0] for c in cols], fontsize=7.5)
        ax.set_xlim(-0.6, 4.6)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        lo = 55 if g == "G05" else 74
        ax.set_ylim(lo, lo + 16)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axhline(np.mean(seeds(cols[0][1], only=sd)), color=AO, lw=0.8, ls="--")
    axs[0].set_ylabel("A-OKVQA val accuracy (%)")
    fig.text(
        0.5,
        -0.05,
        "Filled diamond: mean over seeds with the default loss; open diamond: mean with the balanced "
        "loss CE(letter) + 0.3·CE(rest);\nsmall dots: individual seeds; dashed line: AO. Both "
        "panels span 16 pp. Bottom row: untrained bridge θ₀ under the same instruction.",
        ha="center",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout()
    save(fig, "fig1_2_teaser.png")


# ============================================================ Fig 2.1 connector-family timeline
def fig_timeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.axis("off")
    ax.set_xlim(2019.5, 2026.95)
    ax.set_ylim(-0.35, 4.3)
    lanes = [
        ("Cross-attention\ninto the LLM", 3.6),
        ("Query compressor\n(Q-Former)", 2.6),
        ("Projector on every\npatch token\n(linear or MLP)", 1.6),
        ("Rationale / CoT\nsupervision", 0.6),
    ]
    for name, y in lanes:
        ax.text(2019.5, y, name, va="center", ha="left", fontsize=7.5, color=INK2)
        ax.plot([2021.0, 2026.8], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=0)
    # (year, lane y, label, label above the lane?, optional horizontal shift of the label)
    ev = [
        (2022.3, 3.6, "Flamingo", True),
        (2023.08, 2.6, "BLIP-2", True),
        (2023.4, 2.6, "InstructBLIP", False),
        (2022.7, 1.6, "LiMBeR", True),
        (2023.3, 1.6, "LLaVA", False),
        (2023.8, 1.6, "LLaVA-1.5", True),
        (2024.6, 1.6, "LLaVA-OV", False),
        (2026.6, 1.6, "projector-only\n(Iskandar; Lee)", True),
        (2022.08, 0.6, "CoT prompting", True),
        (2022.7, 0.6, "ScienceQA", False),
        (2023.15, 0.6, "MM-CoT", True),
        (2023.25, 0.6, "UMAE", False),
        (2023.8, 0.6, "LG-VQA", False),
        (2023.9, 0.6, "ReVisE", True),
        (2024.45, 0.6, "VPD", True),
        (2026.2, 0.6, "Balanced\nThinking", False, -0.45),
        (2026.65, 0.6, "order studies\n(Liang; Menzat)", False, 0.2),
    ]
    for x, y, lab, up, *dx in ev:
        ax.scatter(x, y, s=28, color=INK2, zorder=3)
        ax.text(
            x + (dx[0] if dx else 0),
            y + (0.17 if up else -0.17),
            lab,
            ha="center",
            va="bottom" if up else "top",
            fontsize=6.8,
            color=INK,
        )
    ax.scatter(2026.55, 1.1, s=150, marker="*", color=AO, zorder=4)
    ax.text(
        2026.42,
        1.1,
        "This project:\nbridge only +\nrationale targets",
        ha="right",
        va="center",
        fontsize=7,
        color=AO,
        fontweight="bold",
    )
    for yr in range(2021, 2027):
        ax.text(yr + 0.5, -0.3, str(yr), ha="center", fontsize=7.5, color=INK2)
    for yr in range(2021, 2028):
        ax.plot([yr, yr], [-0.2, 3.95], color=GRID, lw=0.6, zorder=0)
    save(fig, "fig2_1_timeline.png")


# ============================================================ Fig 3.3 loss weights
def fig_lossweights():
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 3.6), gridspec_kw=dict(hspace=0.9))
    T = 18  # mean A→R target length, 18.3 tokens (results/analysis/target_lengths.txt)
    specs = [
        ("AO  target “B ⟨end⟩”", ["B", "⟨end⟩"], AO),
        (
            "A→R  target “B ⏎ Because: … ⟨end⟩” (about 18 tokens)",
            ["B"] + ["r"] * (T - 2) + ["⟨end⟩"],
            ATR,
        ),
        (
            "R→A  target “Because: … ⏎ Answer: B ⟨end⟩”",
            ["r"] * (T - 3) + ["Ans:", "B", "⟨end⟩"],
            RTA,
        ),
    ]
    for ax, (title, toks, col) in zip(axs, specs):
        n = len(toks)
        letter = toks.index("B")
        dflt = np.full(n, 1 / n)
        bal = np.full(n, 0.3 / (n - 1))
        bal[letter] = 1.0
        bal = bal / bal.sum()
        x = np.arange(n)
        ax.bar(x - 0.2, dflt, 0.4, color=GREY, label="default: mean over tokens")
        if col != AO:
            ax.bar(x + 0.2, bal, 0.4, color=col, label="balanced: CE(letter) + 0.3·CE(rest)")
        ax.set_xticks(x, ["B" if t == "B" else ("·" if t == "r" else t) for t in toks], fontsize=7)
        ax.set_xlim(-0.7, 17.7)
        ax.set_ylim(0, 0.95)
        ax.set_yticks([0, 0.4, 0.8])
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.text(
            letter + 0.5,
            (bal[letter] if col != AO else dflt[letter]) - 0.05,
            f"letter: {dflt[letter]:.0%}"
            + (f" → {bal[letter]:.0%} (balanced)" if col != AO else " (default loss)"),
            ha="left",
            va="top",
            fontsize=7.5,
            color=INK,
        )
        ax.set_ylabel("weight", fontsize=7.5)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    axs[1].set_xlabel("")
    fig.text(
        0.5,
        0.0,
        "· = rationale token. Weight given to each token's cross-entropy (sums to 1); it equals the "
        "token's share of the loss\nonly if every token has the same cross-entropy. R→A is drawn "
        "with the A→R length.",
        ha="center",
        fontsize=7,
        color=INK2,
    )
    save(fig, "fig3_3_loss_weights.png")


# ============================================================ Fig 3.4 evaluation protocol
def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.text(
        0.0,
        4.6,
        "Q: What is located on the shelves?   (correct: books)",
        fontsize=7.8,
        fontweight="bold",
        va="top",
    )
    ax.imshow(img(IMG + "000000421923.jpg", (300, 225)), extent=(0.0, 1.6, 2.75, 3.95))
    ax.text(0.8, 2.68, "real image", ha="center", va="top", fontsize=7, fontweight="bold")
    ax.imshow(img(IMG + "000000360393.jpg", (300, 225)), extent=(0.0, 1.6, 1.05, 2.25))
    ax.text(
        0.8,
        0.98,
        "swapped image\n(fixed derangement)",
        ha="center",
        va="top",
        fontsize=7,
        fontweight="bold",
    )
    ax.text(2.1, 4.1, "4 cyclic option orders", fontsize=7, color=INK2)
    opts = ["books", "dvds", "games", "food"]
    for k in range(4):
        rot = opts[k:] + opts[:k]
        ax.text(
            2.1,
            3.7 - k * 0.36,
            f"order {k}: " + " ".join(f"{'ABCD'[i]}.{o}" for i, o in enumerate(rot)),
            fontsize=6.3,
            family="DejaVu Sans Mono",
            va="center",
        )
    arrow(ax, 1.65, 3.35, 2.05, 3.35)
    arrow(ax, 1.65, 1.65, 2.05, 2.55)
    box(
        ax,
        5.85,
        2.45,
        1.7,
        1.45,
        "frozen model\none forward pass\n→ p(A), p(B),\np(C), p(D),\nrenormalised",
        fc=FROZEN,
        fs=6.8,
    )
    arrow(ax, 5.55, 3.15, 5.85, 3.15)
    box(ax, 7.95, 3.25, 2.0, 0.65, "accuracy, order 0\n(headline)", fc=TRAIN, ec=AO, fs=6.8)
    box(ax, 7.95, 2.45, 2.0, 0.65, "mean over 4 orders\n(robustness)", fc="white", fs=6.8)
    arrow(ax, 7.55, 3.4, 7.95, 3.55)
    arrow(ax, 7.55, 2.95, 7.95, 2.8)
    ax.text(
        2.1,
        1.95,
        "R→A only: first generate “Because: …” greedily (≤ 64 tokens),\nappend “Answer:”, then score the letters.",
        fontsize=6.6,
        color=INK2,
        va="top",
    )
    box(
        ax,
        2.1,
        0.1,
        3.6,
        1.0,
        "image-reliance gap\n= Acc(real) − Acc(swapped)\n(Perceptual Score of Gat et al., 2021)",
        fc="white",
        ec=ATR,
        fs=6.8,
    )
    box(
        ax,
        5.95,
        0.1,
        4.0,
        1.0,
        "prior-wrong subset P (728 of 1,145):\nitems the text-only LLM gets wrong;\naccuracy on P does not come from this prior",
        fc="white",
        ec=RTA,
        fs=6.8,
    )
    save(fig, "fig3_4_protocol.png")


# ============================================================ Fig 3.5 block-causal attention
def fig_blockattn():
    n = 12
    full = np.tril(np.ones((n, n)))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    # standard: full n x n computed, upper half masked
    cmp1 = np.where(full == 1, 2, 1)
    axs[0].imshow(
        cmp1, cmap=matplotlib.colors.ListedColormap(["white", "#f3d2c4", AO]), vmin=0, vmax=2
    )
    axs[0].set_title(
        "Standard (training on Apple GPU):\nall n² scores computed, half then masked", fontsize=8
    )
    h = n // 2
    cmp2 = np.zeros((n, n))
    cmp2[:h, :h] = np.where(full[:h, :h] == 1, 2, 1)
    cmp2[h:, :h] = 2
    cmp2[h:, h:] = np.where(full[h:, h:] == 1, 2, 1)
    axs[1].imshow(
        cmp2, cmap=matplotlib.colors.ListedColormap(["white", "#f3d2c4", AO]), vmin=0, vmax=2
    )
    axs[1].set_title("Block-causal (this project):\n3 of 4 blocks computed = 0.75 n²", fontsize=8)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        a.set_xlabel("key position →", fontsize=7.5)
        a.set_ylabel("← query position", fontsize=7.5)
    axs[1].axhline(h - 0.5, color=INK, lw=1)
    axs[1].axvline(h - 0.5, color=INK, lw=1)
    fig.text(
        0.5,
        -0.02,
        "Blue: needed and computed.  Pink: computed then thrown away.  White: skipped.  "
        "Same result up to rounding (relative gradient error ~1e-6).",
        ha="center",
        fontsize=7,
        color=INK2,
    )
    save(fig, "fig3_5_block_attention.png")


# ============================================================ Fig 4.1 main results with seeds
def fig_main():
    groups = [
        ("0.5B · 4k", "G05", "n4000"),
        ("0.5B · 16k", "G05", "full"),
        ("7B · 4k", "G7", "n4000"),
        ("7B · 16k", "G7", "full"),
    ]
    arms = [
        ("AO", "answer_only"),
        ("A→R", "answer_then_rationale"),
        ("R→A", "rationale_then_answer"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for gi, (_, g, d) in enumerate(groups):
        for ai, (an, ak) in enumerate(arms):
            key = f"{g}_B_{ak}_{d}"
            x = gi * 4 + ai
            if key not in C:
                ax.text(x, 45, "not run\n(cost)", ha="center", fontsize=6.8, color=INK2)
                continue
            v = seeds(key)
            m = np.mean(v)
            ax.bar(x, m, 0.8, color=ARM[an], label=an if gi == 0 else None)
            ax.scatter([x] * len(v), v, s=8, color=INK, zorder=3, lw=0)
            ax.text(x, max(v) + 0.8, f"{m:.1f}", ha="center", fontsize=7)
        t0 = C[f"{g}_theta0_answer_only"]["-"]["real"]
        ax.plot(
            [gi * 4 - 0.45, gi * 4 + 2.45],
            [t0, t0],
            color=INK2,
            ls="--",
            lw=0.9,
            label="untrained bridge θ₀ (AO prompt)" if gi == 0 else None,
        )
    ax.set_xticks([gi * 4 + 1 for gi in range(4)], [g[0] for g in groups])
    ax.set_ylim(20, 90)
    ax.set_ylabel("A-OKVQA val accuracy (%)")
    ax.axhline(25, color=GREY, lw=0.7, ls=":")
    ax.text(14.0, 26.3, "chance", fontsize=7, color=INK2, ha="center")
    ax.legend(ncol=4, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.13))
    save(fig, "fig4_1_main.png")


# ============================================================ Fig 4.2 data and scale slopes
def fig_slopes():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for ax, g, title in [(axs[0], "G05", "Qwen2-0.5B"), (axs[1], "G7", "Qwen2-7B")]:
        for an, ak in [
            ("AO", "answer_only"),
            ("A→R", "answer_then_rationale"),
            ("R→A", "rationale_then_answer"),
        ]:
            xs, ys = [], []
            for x, d in [(0, "n4000"), (1, "full")]:
                k = f"{g}_B_{ak}_{d}"
                if k in C:
                    xs.append(x)
                    ys.append(np.mean(seeds(k)))
            ax.plot(xs, ys, "-o", color=ARM[an], lw=2, ms=5)
            dy = {"A→R": 0.35, "R→A": -0.35}.get(an, 0) if g == "G05" else 0
            ax.text(
                xs[-1] + 0.06,
                ys[-1] + dy,
                an + (" (16k not run)" if len(xs) == 1 else ""),
                color=INK,
                fontsize=8,
                va="center",
            )
        ax.set_xticks([0, 1], ["4,000 items\n(250 steps)", "16,024 items\n(1,002 steps)"])
        ax.set_xlim(-0.2, 1.35)
        ax.set_title(title, fontsize=9, fontweight="bold")
    axs[0].set_ylabel("val accuracy (%)")
    save(fig, "fig4_2_data_scale.png")


# ============================================================ Fig 4.3 hyperparameters
def fig_hp():
    settings = [
        ("baseline", None),
        ("batch 64", "bs64"),
        ("4 epochs", "ep4"),
        ("lr 3e-5", "lr3e-5"),
        ("lr 3e-4", "lr3e-4"),
        ("lr 1e-3", "lr1e-3"),
        ("λ = 0", "rw0"),
        ("λ = 0.1", "rw0.1"),
        ("λ = 0.3", "rw0.3"),
        ("λ = 1.0", "rw1.0"),
    ]
    base = {
        "AO": "G05_B_answer_only_n4000",
        "A→R": "G05_B_answer_then_rationale_n4000",
        "R→A": "G05_B_rationale_then_answer_n4000",
    }
    ak = {"AO": "answer_only", "A→R": "answer_then_rationale", "R→A": "rationale_then_answer"}
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for an, off in [("AO", -0.2), ("A→R", 0), ("R→A", 0.2)]:
        for i, (_, s) in enumerate(settings):
            if s is None:  # baseline: seeds 0-1 of the main 4k runs, as in Table 4.5
                v = [C[base[an]][k]["real"] for k in ("0", "1")]
            else:
                key = f"G05_HP_{ak[an]}_{s}"
                if key not in C:
                    continue
                v = seeds(key)
            y = len(settings) - 1 - i + off
            ax.plot([min(v), max(v)], [y, y], color=ARM[an], lw=1)
            ax.scatter(np.mean(v), y, color=ARM[an], s=26, zorder=3, label=an if i == 0 else None)
    ax.set_yticks(range(len(settings)), [s[0] for s in settings][::-1])
    ax.set_xlabel("val accuracy (%), mean of seeds 0–1 (line = range)")
    ax.axhspan(-0.5, 3.5, color="#fbf3ee", zorder=0)
    ax.text(43.5, 1.5, "loss weight λ\n(rationale arms)", fontsize=7, color=INK2, va="center")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(43, 66)
    save(fig, "fig4_3_hparams.png")


# ============================================================ Fig 4.4 letter-loss training curves
def fig_curves():
    # needs train_log.jsonl of the three runs below (release asset runs.tar.gz)
    def load(run):
        L = [json.loads(l) for l in open(RUNS + run + "/train_log.jsonl")]
        s = np.array([r["step"] for r in L])
        a = np.array([r["ce_answer"] for r in L])
        k = 25
        sm = np.convolve(a, np.ones(k) / k, mode="valid")
        return s[k - 1 :], sm

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for run, lab, col, ls in [
        ("G05_B_answer_only_full_s0", "AO", AO, "-"),
        ("G05_B_answer_then_rationale_full_s0", "A→R, default loss", ATR, "-"),
        ("G05_BAL_answer_then_rationale_full_rw0.3_s0", "A→R, balanced (λ = 0.3)", ATR, "--"),
    ]:
        s, y = load(run)
        ax.plot(s, y, color=col, ls=ls, lw=1.6, label=lab)
    ax.set_xlabel("optimizer step (0.5B, 16k items, seed 0)")
    ax.set_ylabel("training CE of the\nanswer letter (nats)")
    ax.legend(fontsize=8)
    ax.text(
        0.99,
        0.02,
        "moving average over 25 steps",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color=INK2,
    )
    save(fig, "fig4_4_letter_loss.png")


# ============================================================ Fig 4.5 real vs swapped scatter
def fig_scatter():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for g, mk in [("G05", "o"), ("G7", "s")]:
        for an, ak in [
            ("AO", "answer_only"),
            ("A→R", "answer_then_rationale"),
            ("R→A", "rationale_then_answer"),
        ]:
            for d in ["n4000", "full"]:
                k = f"{g}_B_{ak}_{d}"
                if k not in C:
                    continue
                x, y = np.mean(seeds(k, "swapped")), np.mean(seeds(k))
                ax.scatter(
                    x,
                    y,
                    color=ARM[an],
                    marker=mk,
                    s=40 if d == "full" else 22,
                    ec=INK if d == "full" else "none",
                    lw=0.6,
                    zorder=3,
                )
            t = C[f"{g}_theta0_{ak}"]["-"]
            ax.scatter(
                t["swapped"], t["real"], color="white", ec=ARM[an], marker=mk, s=22, lw=1, zorder=3
            )
    xs = np.linspace(25, 50, 10)
    for gap in (20, 30, 40):
        ax.plot(xs, xs + gap, color=GRID, lw=0.8, zorder=0)
        ax.text(50, 50 + gap, f"gap {gap}", fontsize=6.5, color=INK2, va="center")
    ax.set_xlabel("swapped-image accuracy (%)")
    ax.set_ylabel("real-image accuracy (%)")
    ax.set_xlim(26, 53)
    ax.set_ylim(40, 90)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in ARM.items()]
    h += [
        Line2D([], [], marker="o", ls="", color=GREY, label="0.5B"),
        Line2D([], [], marker="s", ls="", color=GREY, label="7B"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, label="θ₀ (untrained)"),
        Line2D([], [], marker="o", ls="", color=GREY, mec=INK, ms=7, label="16k items (large)"),
    ]
    ax.legend(handles=h, fontsize=7, loc="lower right", ncol=2)
    save(fig, "fig4_5_real_vs_swapped.png")


# ============================================================ Fig 4.6 explanation examples
def fig_explanations():
    ex = [
        (
            "What is located on the shelves?",
            "books",
            "000000421923.jpg",
            "Because: The shelves are filled with books.",
            "000000360393.jpg",
            "Because: The food is on the shelves.",
            "food",
        ),
        (
            "What kind of resort are these people at?",
            "ski resort",
            "000000562818.jpg",
            "Because: The resort is a ski resort.",
            "000000489305.jpg",
            "Because: The resort is in a tropical area.",
            "tropical resort",
        ),
        (
            "What genus of animal is visible here?",
            "feline",
            "000000195918.jpg",
            "Because: The animal is a feline.",
            "000000344909.jpg",
            "Because: The animal is a rodent.",
            "rodent",
        ),
    ]
    fig = plt.figure(figsize=(7.2, 6.3))
    for r, (q, gold, im1, e1, im2, e2, wrong) in enumerate(ex):
        y0 = 1 - (r + 1) * 0.32
        fig.text(
            0.0, y0 + 0.30, f"Q{r + 1}: {q}   (correct: {gold})", fontsize=8.5, fontweight="bold"
        )
        for c, (im, e, ok, lab) in enumerate(
            [(im1, e1, True, "real image"), (im2, e2, False, "swapped image")]
        ):
            a = fig.add_axes([0.0 + c * 0.5, y0 + 0.03, 0.17, 0.25])
            a.imshow(img(IMG + im, (320, 240)))
            a.axis("off")
            a.set_title(lab, fontsize=7.5, color=INK2)
            t = fig.add_axes([0.18 + c * 0.5, y0 + 0.03, 0.31, 0.25])
            t.axis("off")
            t.text(
                0.02,
                0.62,
                "\n".join(textwrap.wrap(e, 30)),
                fontsize=8,
                va="center",
                family="DejaVu Sans",
            )
            t.text(
                0.02,
                0.2,
                ("✓ answers " + gold) if ok else ("✗ answers " + wrong),
                fontsize=8,
                color=RTA if ok else ATR,
                fontweight="bold",
            )
    fig.text(
        0.0,
        0.0,
        "R→A model (0.5B, 16k items, seed 0). With the wrong image the explanation stays fluent and "
        "plausible, and the answer follows it.",
        fontsize=7.5,
        color=INK2,
    )
    save(fig, "fig4_6_explanations.png")


# ============================================================ Fig 4.8 explanation quality
def fig_explq():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    groups = ["0.5B, 16k", "7B, 4k"]
    # R→A rows of Table 4.8. ROUGE-L against two human rationales, averaged over the three
    # ways of leaving one out (from the table note), so that it is comparable with one human
    # rationale scored against the other two (36.2); naming = the table's last column.
    rouge = {"real": [47.3, 48.1], "swapped": [45.2, 44.9]}
    name = {"real": [37.9, 47.8], "swapped": [20.3, 25.9]}
    for ax, dat, title in [
        (axs[0], rouge, "ROUGE-L vs two human rationales"),
        (axs[1], name, "% of explanations naming the correct option"),
    ]:
        for i in range(len(groups)):
            for j, (cond, col) in enumerate([("real", RTA), ("swapped", GREY)]):
                v = dat[cond][i]
                ax.bar(i * 2.6 + j, v, 0.9, color=col, label=cond + " image" if i == 0 else None)
                ax.text(i * 2.6 + j, v + 1, f"{v:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks([0.5, 3.1], groups)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 65)
    axs[0].axhline(36.2, color=INK2, ls="--", lw=0.8, label="one human vs the other two (36.2)")
    axs[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axs[1].legend(fontsize=7.5, loc="upper left")
    save(fig, "fig4_8_explanation_quality.png")


# ============================================================ Fig 4.9 training-free improvements
def fig_cheap():
    # Table 4.10: order 0, 4-order ensemble, SigLIP fusion, fusion + ensemble
    rows = [
        ("0.5B AO, 4k", [61.4, 64.3, 68.7, 70.3]),
        ("0.5B A→R, 4k", [52.1, 58.3, 66.2, 68.9]),
        ("0.5B AO, 16k", [67.4, 68.3, 70.9, 72.1]),
        ("7B AO, 16k", [83.1, 83.6, 83.8, 83.9]),
    ]
    labs = ["bridge (order 0)", "+ 4-order average", "+ SigLIP fusion", "+ both"]
    cols = [GREY, "#b9cde8", "#6fa2e0", AO]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    for i, (_, v) in enumerate(rows):
        for j in range(4):
            ax.bar(i * 5 + j, v[j], 0.85, color=cols[j], label=labs[j] if i == 0 else None)
            ax.text(i * 5 + j, v[j] + 0.6, f"{v[j]:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks([i * 5 + 1.5 for i in range(4)], [r[0] for r in rows])
    ax.axhline(63.8, color=INK2, ls=":", lw=0.9, label="SigLIP alone, zero-shot (63.8)")
    ax.set_ylim(45, 90)
    ax.set_ylabel("val accuracy (%)")
    ax.legend(ncol=3, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.2))
    save(fig, "fig4_9_training_free.png")


# ============================================================ Fig 4.10 where the time goes (Mac)
def fig_time():
    fig, axs = plt.subplots(
        1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.35)
    )
    ax = axs[0]
    # (component, % of the arithmetic (Section 3.8.2), measured ms over 24 layers: the "B1"
    # entries of results/runs/laptop/mac_bench/bench2.json)
    parts = [
        ("MLP", 80, 209),
        ("attention block\n(projections + scores)", 20, 207),
        ("other", 0, 460 - 209 - 207),  # rest of the measured layer time (bench2.json B1)
    ]
    tot = sum(p[2] for p in parts)
    for i, (lab, fl, ms) in enumerate(parts):
        ax.barh(1, fl, left=sum(p[1] for p in parts[:i]), color=[AO, ATR, GREY][i], ec="white")
        ax.barh(
            0,
            100 * ms / tot,
            left=sum(100 * p[2] / tot for p in parts[:i]),
            color=[AO, ATR, GREY][i],
            ec="white",
            label=lab,
        )
    ax.set_yticks([0, 1], ["share of time\n(measured)", "share of\narithmetic"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of one training step (Qwen2-0.5B, ~830 tokens)")
    ax.legend(fontsize=6.8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.3))
    ax = axs[1]
    # results/analysis/laptop_run_times.txt: six runs before (76-91 min), two after (42, 45 min)
    ax.bar([0, 1], [83, 43], color=[GREY, AO], width=0.6)
    ax.errorbar([0, 1], [83, 43], yerr=[[7, 1], [8, 2]], fmt="none", ecolor="black", lw=0.8)
    for x, v, top in [(0, 83, 91), (1, 43, 45)]:
        ax.text(x, top + 2, f"{v} min", ha="center", fontsize=7.5)
    ax.set_xticks([0, 1], ["before", "after\n(cache filled, block attn,\ndev 100)"], fontsize=7)
    ax.set_ylabel("minutes")
    ax.set_title("one 4k-item run (M4 Pro)", fontsize=8)
    ax.set_ylim(0, 105)
    save(fig, "fig4_10_mac_time.png")


if __name__ == "__main__":
    Path(OUT).mkdir(parents=True, exist_ok=True)
    ALL = [
        fig_example,
        fig_teaser,
        fig_timeline,
        fig_lossweights,
        fig_protocol,
        fig_blockattn,
        fig_main,
        fig_slopes,
        fig_hp,
        fig_curves,
        fig_scatter,
        fig_explanations,
        fig_explq,
        fig_cheap,
        fig_time,
    ]
    for f in ALL:
        if len(sys.argv) == 1 or f.__name__ in sys.argv[1:]:
            f()
            print("ok", f.__name__)
