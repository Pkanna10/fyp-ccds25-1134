"""Figures 3.1 (from pixels to the LLM's input) and 3.2 (training and test pipelines).

Figure 3.1 shows COCO val2017 image 421923 (scripts/download_data.sh); Figure 3.2 needs no
data. Uses the helpers and colours of make_figures.py, so it also reads
results/run_metrics.json.

    python scripts/make_pipeline_figures.py   # -> results/figures/fig3_{1_shapes,2_train_test}.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from make_figures import box, arrow, img, save, IMG, AO, ATR, RTA, INK, INK2, FROZEN, TRAIN

RAT = "#8e44ad"  # human rationale (training target only)
PURPLE_BG = "#efe4f6"


def columns(ax, x0, y0, h, n_cells, labels, color, w=0.16, gap=0.07, ec=INK2):
    """Draw token vectors as thin columns of cells; returns the x of the right edge."""
    x = x0
    for lab in labels:
        if lab == "…":
            ax.text(x + w / 2, y0 + h / 2, "…", ha="center", va="center", fontsize=9, color=INK2)
        else:
            ch = h / n_cells
            for k in range(n_cells):
                ax.add_patch(Rectangle((x, y0 + k * ch), w, ch, fc=color, ec=ec, lw=0.4))
            ax.text(x + w / 2, y0 - 0.06, lab, ha="center", va="top", fontsize=5.6, color=INK2)
        x += w + gap
    return x - gap


# ============================================================ Fig 3.1 from pixels to LLM input
def fig_shapes():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.25, 5.75)

    # 1. image with the 27 x 27 patch grid
    ax.imshow(img(IMG + "000000421923.jpg", (384, 384)), extent=(0.0, 1.5, 3.2, 4.7))
    for k in range(28):
        t = k * 1.5 / 27
        ax.plot([t, t], [3.2, 4.7], color="white", lw=0.25, alpha=0.8)
        ax.plot([0, 1.5], [3.2 + t, 3.2 + t], color="white", lw=0.25, alpha=0.8)
    ax.add_patch(Rectangle((0, 4.7 - 1.5 / 27), 1.5 / 27, 1.5 / 27, fc="none", ec=ATR, lw=1.0))
    ax.text(0.75, 5.45, "① image", ha="center", fontsize=7.6, fontweight="bold")
    ax.text(0.75, 5.12, "384 × 384 pixels", ha="center", fontsize=6.6, color=INK2)
    ax.text(
        0.75,
        3.1,
        "cut into a 27 × 27 grid\nof 14 × 14-pixel patches\n= 729 patches",
        ha="center",
        va="top",
        fontsize=6.4,
        color=INK2,
    )

    # 2. SigLIP
    arrow(ax, 1.55, 3.95, 1.85, 3.95)
    box(ax, 1.85, 3.35, 1.05, 1.2, "SigLIP-\nso400m\nencoder\n(frozen)", fc=FROZEN, fs=6.8)
    ax.text(2.375, 5.45, "② encode", ha="center", fontsize=7.6, fontweight="bold")
    ax.text(2.375, 3.25, "(layer −2)", ha="center", va="top", fontsize=6.0, color=INK2)
    arrow(ax, 2.9, 3.95, 3.15, 3.95)

    # 3. 729 vectors x 1,152
    xr = columns(ax, 3.2, 3.2, 1.55, 12, ["1", "2", "3", "…", "729"], FROZEN)
    ax.text(
        3.2 + (xr - 3.2) / 2, 5.45, "③ 729 vectors", ha="center", fontsize=7.6, fontweight="bold"
    )
    ax.text(
        3.2 + (xr - 3.2) / 2,
        5.22,
        "one per patch\n(not 729 images)",
        ha="center",
        va="top",
        fontsize=6.3,
        color=INK2,
    )
    ax.annotate(
        "",
        xy=(xr + 0.1, 3.2),
        xytext=(xr + 0.1, 4.75),
        arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7),
    )
    ax.text(xr + 0.16, 3.97, "1,152\neach", ha="left", va="center", fontsize=6.0, color=INK2)
    ax.text(
        3.2 + (xr - 3.2) / 2,
        2.98,
        "matrix X: 729 × 1,152",
        ha="center",
        va="top",
        fontsize=6.4,
        color=INK,
    )

    # 4. the bridge, drawn as a per-vector MLP
    bx = 5.25
    arrow(ax, xr + 0.6, 3.95, bx, 3.95)
    ax.add_patch(Rectangle((bx, 3.0), 2.25, 1.8, fc=TRAIN, ec=AO, lw=1.0))
    ax.text(
        bx + 1.125,
        5.45,
        "④ bridge (trained)",
        ha="center",
        fontsize=7.6,
        fontweight="bold",
        color=AO,
    )
    ax.text(
        bx + 1.125,
        5.22,
        "the same small MLP is\napplied to every vector",
        ha="center",
        va="top",
        fontsize=6.3,
        color=INK2,
    )
    lay = [("1,152", bx + 0.2, 1.5), ("896", bx + 1.0, 1.15), ("896", bx + 1.8, 1.15)]
    for lab, x, hh in lay:
        ax.add_patch(Rectangle((x, 3.95 - hh / 2), 0.22, hh, fc="white", ec=INK2, lw=0.6))
        ax.text(
            x + 0.11, 3.95 - hh / 2 - 0.05, lab, ha="center", va="top", fontsize=5.8, color=INK2
        )
    for (_, x0, h0), (_, x1, h1) in zip(lay[:-1], lay[1:]):
        for f0 in (-0.4, 0, 0.4):
            for f1 in (-0.4, 0, 0.4):
                ax.plot(
                    [x0 + 0.22, x1], [3.95 + f0 * h0, 3.95 + f1 * h1], color=AO, lw=0.3, alpha=0.6
                )
    ax.text(bx + 0.71, 4.52, "W₁\n896×1,152", ha="center", fontsize=5.6, color=AO)
    ax.text(bx + 1.51, 4.52, "W₂\n896×896", ha="center", fontsize=5.6, color=AO)
    ax.text(bx + 1.11, 3.12, "GELU", ha="center", fontsize=5.6, color=AO)
    ax.text(
        bx + 1.125,
        2.98,
        "h = W₂·GELU(W₁x + b₁) + b₂\n1,836,800 parameters (0.5B)",
        ha="center",
        va="top",
        fontsize=6.2,
        color=AO,
    )

    # 5. 729 soft tokens x 896
    sx = 7.95
    arrow(ax, bx + 2.25, 3.95, sx - 0.05, 3.95)
    xr2 = columns(ax, sx, 3.4, 1.15, 9, ["1", "2", "3", "…", "729"], TRAIN, ec=AO)
    ax.text(
        sx + (xr2 - sx) / 2, 5.45, "⑤ 729 soft tokens", ha="center", fontsize=7.6, fontweight="bold"
    )
    ax.text(
        sx + (xr2 - sx) / 2,
        5.22,
        "each as wide as\none word vector",
        ha="center",
        va="top",
        fontsize=6.3,
        color=INK2,
    )
    ax.annotate(
        "",
        xy=(xr2 + 0.1, 3.4),
        xytext=(xr2 + 0.1, 4.55),
        arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7),
    )
    ax.text(xr2 + 0.16, 3.97, "896\neach", ha="left", va="center", fontsize=6.0, color=INK2)
    ax.text(
        sx + (xr2 - sx) / 2,
        2.98,
        "matrix X′: 729 × 896\n(7B: 729 × 3,584)",
        ha="center",
        va="top",
        fontsize=6.4,
        color=INK,
    )

    # 6. the LLM's input sequence
    ax.text(
        0.0,
        2.05,
        "⑥ what the frozen LLM reads: one sequence of about 830 vectors, each 896 numbers wide",
        fontsize=7.6,
        fontweight="bold",
    )
    segs = [
        ("system line", 1.0, FROZEN, INK2),
        ("729 soft image tokens (from ⑤)", 3.2, TRAIN, AO),
        ("Question · Options A–D · instruction", 3.25, "white", INK2),
        ("assistant:", 1.05, FROZEN, INK2),
    ]
    x = 0.0
    for s_, w, c, ec in segs:
        box(ax, x, 1.2, w, 0.55, s_, fc=c, ec=ec, fs=6.6, r=0.01)
        x += w
    ax.add_patch(
        FancyArrowPatch(
            (sx + (xr2 - sx) / 2, 2.62),
            (2.6, 1.8),
            connectionstyle="arc3,rad=-0.12",
            arrowstyle="-|>",
            mutation_scale=9,
            color=AO,
            lw=1.0,
        )
    )
    ax.text(
        5.85,
        1.12,
        "words are turned into vectors by the LLM's own (frozen) embedding table",
        ha="center",
        va="top",
        fontsize=6.2,
        color=INK2,
    )
    arrow(ax, 8.5, 1.2, 8.5, 0.72)
    box(ax, 7.0, 0.0, 3.0, 0.7, "Qwen2 LLM (frozen)\n→ next-token probabilities", fc=FROZEN, fs=6.6)
    ax.text(
        0.0,
        0.35,
        "Grey = frozen, never changes.   Blue = the only trained part.\n"
        "At 7B: width 3,584 instead of 896; the bridge has 16,980,992 parameters.",
        fontsize=6.3,
        color=INK2,
        va="center",
    )
    save(fig, "fig3_1_shapes.png")


# ============================================================ Fig 3.2 training vs test
def fig_train_test():
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.0), gridspec_kw=dict(hspace=0.08))
    for ax in axs:
        ax.axis("off")
        ax.set_xlim(0, 10.9)
        ax.set_ylim(0, 2.6)

    def inputs(ax, image_label, extra=None):
        ax.text(
            0.0,
            2.25,
            "Model input (the prompt)" + (extra or ""),
            fontsize=7,
            color=ATR if extra else INK2,
            fontweight="bold",
        )
        rows = [
            (image_label, FROZEN),
            ("Question", "white"),
            ("Options A–D", "white"),
            ("instruction for the target", "white"),
        ]
        for k, (lab, c) in enumerate(rows):
            box(ax, 0.0, 1.75 - k * 0.42, 2.05, 0.34, lab, fc=c, fs=6.4, r=0.01)

    def pipeline(ax, trained):
        box(ax, 2.35, 1.1, 0.8, 0.7, "SigLIP\n(frozen)", fc=FROZEN, fs=6.4)
        box(
            ax,
            3.25,
            1.1,
            0.8,
            0.7,
            "bridge\n" + ("(trained)" if trained else "(fixed)"),
            fc=TRAIN,
            ec=AO,
            fs=6.4,
        )
        box(ax, 4.15, 1.1, 0.85, 0.7, "Qwen2\n(frozen)", fc=FROZEN, fs=6.4)
        arrow(ax, 2.08, 1.45, 2.35, 1.45)
        arrow(ax, 3.15, 1.45, 3.25, 1.45)
        arrow(ax, 4.05, 1.45, 4.15, 1.45)

    # (a) training
    ax = axs[0]
    ax.text(
        0.0,
        2.55,
        "(a) Training: the human rationale appears only in the target",
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )
    inputs(ax, "real image")
    pipeline(ax, True)
    arrow(ax, 5.0, 1.45, 5.25, 1.45)
    ax.text(
        5.3,
        2.25,
        "Target the model is trained to write (teacher forcing)",
        fontsize=7,
        color=INK2,
        fontweight="bold",
    )
    tg = [
        ("AO", AO, [("B", "white")]),
        ("A→R", ATR, [("B ⏎ Because:", "white"), ("human rationale", PURPLE_BG)]),
        (
            "R→A",
            RTA,
            [("Because:", "white"), ("human rationale", PURPLE_BG), ("⏎ Answer: B", "white")],
        ),
    ]
    for k, (arm, col, parts) in enumerate(tg):
        y = 1.72 - k * 0.47
        ax.text(5.3, y + 0.17, arm, fontsize=7, color=col, fontweight="bold", va="center")
        x = 5.8
        for t, c in parts:
            w = 0.2 + 0.08 * len(t)
            box(
                ax,
                x,
                y,
                w,
                0.34,
                t,
                fc=c,
                ec=RAT if c == PURPLE_BG else col,
                fs=6.0,
                r=0.01,
                tc=RAT if c == PURPLE_BG else INK,
            )
            x += w + 0.04
    box(
        ax,
        9.3,
        0.75,
        1.6,
        1.05,
        "loss = cross-entropy\non the target tokens\n(default or\nbalanced, §3.4)",
        fc="white",
        ec=ATR,
        fs=5.8,
    )
    ax.add_patch(
        FancyArrowPatch(
            (10.1, 0.72),
            (3.65, 1.08),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="-|>",
            mutation_scale=9,
            color=ATR,
            lw=1.0,
            ls="--",
        )
    )
    ax.text(
        6.3,
        0.12,
        "gradient flows back through the frozen LLM; only the bridge's weights change",
        ha="center",
        fontsize=6.4,
        color=ATR,
    )

    # (b) test
    ax = axs[1]
    ax.text(
        0.0,
        2.55,
        "(b) Test: no rationale is given to the model",
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )
    inputs(ax, "real or swapped image", " — no rationale")
    pipeline(ax, False)
    arrow(ax, 5.0, 1.62, 5.3, 1.9)
    arrow(ax, 5.0, 1.28, 5.3, 0.85)
    box(
        ax,
        5.3,
        1.62,
        3.7,
        0.6,
        "AO, A→R: one forward pass; read p(A), p(B),\np(C), p(D) at the first answer token",
        fc="white",
        ec=AO,
        fs=6.3,
    )
    box(
        ax,
        5.3,
        0.45,
        3.7,
        0.95,
        "R→A: the model first writes its own\n“Because: …” (greedy, ≤ 64 tokens), then\n"
        "“Answer:” is read and p(A)…p(D) scored",
        fc="white",
        ec=RTA,
        fs=6.3,
    )
    arrow(ax, 9.0, 1.92, 9.3, 1.75)
    arrow(ax, 9.0, 0.92, 9.3, 1.15)
    box(
        ax,
        9.3,
        0.75,
        1.6,
        1.05,
        "prediction =\nletter with the\nhighest p",
        fc=TRAIN,
        ec=AO,
        fs=6.4,
    )
    ax.text(
        0.0,
        0.12,
        "Human rationales are used at test time only to grade the generated explanations "
        "(Section 4.6) and in the text-only check of Table 4.7, with and without masking.",
        fontsize=6.3,
        color=RAT,
        va="top",
    )
    save(fig, "fig3_2_train_test.png")


if __name__ == "__main__":
    fig_shapes()
    fig_train_test()
    print("ok")
