"""Figures for the B.Tech black book (blackbook/main.tex).

Two kinds, and nothing else:
  * diagrams drawn from the toolkit's own structure (modules, dataclasses, stage order);
  * charts of recorded measurements read from frontend-next/src/data/*.json
    (written by tools/export_site_data.py) and from `git log` for the timeline.

    ./.venv/bin/python tools/make_blackbook_figures.py
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "frontend-next" / "src" / "data"
OUT = ROOT / "blackbook" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#1b1f1d", "#6b7570", "#c9cfcc"
GREEN, GREEN_L, GREEN_D = "#1f7a5a", "#d6ece3", "#14543d"
AMBER, RED, BLUE = "#b7791f", "#b4412f", "#2f5d8a"
PURPLE = "#6b4c9a"
BLUE_L, AMBER_L, PURPLE_L, ROSE_L, TEAL_L, GREY_L = "#dbe8f5", "#fbe9c8", "#e6dcf2", "#f7dcd6", "#d5eef0", "#eceeed"
TEAL = "#1d7f86"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------- diagram helpers
def canvas(w, h, xlim=(0, 100), ylim=(0, 60)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim), ax.set_ylim(*ylim), ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="white", ec=INK, bold=False, size=8, color=INK, lw=0.9, round_=True):
    style = "round,pad=0.02,rounding_size=1.2" if round_ else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.25)


def arrow(ax, p, q, text="", dashed=False, color=INK, side=0, size=7, rad=0.0, text_xy=None):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=0.9, linestyle="--" if dashed else "-",
        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx, my = text_xy if text_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + side)
        ax.text(mx, my, text, ha="center", va="center", fontsize=size, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=0.6))


def group(ax, x, y, w, h, title, fc="#f6f8f7", ec=RULE, tc=MUTED):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.5", fc=fc, ec=ec, lw=0.8))
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=7.5, color=tc, fontweight="bold", va="top")


# ----------------------------------------------------------------------------- 1. architecture
def fig_architecture():
    fig, ax = canvas(7.0, 5.0, ylim=(0, 72))
    group(ax, 1, 58, 98, 13, "ENTRY POINTS", "#eef4fa", BLUE, BLUE)
    box(ax, 4, 60, 28, 7, "compresskit CLI\ntrain, pipeline, compress, bench", BLUE_L, BLUE, size=7.2)
    box(ax, 36, 60, 28, 7, "Python API\nPipeline, compress()", BLUE_L, BLUE, size=7.2)
    box(ax, 68, 60, 28, 7, "Kaggle notebook\none config cell, Run All", BLUE_L, BLUE, size=7.2)

    group(ax, 1, 33, 98, 22, "COMPRESSION STAGES", "#f8f6fb", PURPLE, PURPLE)
    box(ax, 4, 36, 21, 11, "pruning/\nStructuredPruner\nrecover, latency_guide", GREEN_L, GREEN, size=7.2)
    box(ax, 28, 36, 21, 11, "distillation/\nKD and DKD loss\nDistillConfig", PURPLE_L, PURPLE, size=7.2)
    box(ax, 52, 36, 21, 11, "quantization/\nINT8, FP16 via ORT\nQAT with STE", AMBER_L, AMBER, size=7.2)
    box(ax, 76, 36, 20, 11, "sensitivity/\nper-layer SQNR\nprotect, allocate", ROSE_L, RED, size=7.2)

    group(ax, 1, 17, 98, 13, "ORCHESTRATION AND OUTPUT", "#fdf8ee", AMBER, AMBER)
    box(ax, 4, 19, 28, 7, "pipeline.py\nstage order, prune_history", "white", AMBER, size=7.2)
    box(ax, 36, 19, 28, 7, "search.py\nBudget, escalation ladder", "white", AMBER, size=7.2)
    box(ax, 68, 19, 28, 7, "report.py\nHTML report, results.json", "white", AMBER, size=7.2)

    group(ax, 1, 1, 98, 13, "CORE, SHARED BY EVERY STAGE", "#eef8f8", TEAL, TEAL)
    box(ax, 4, 3, 28, 7, "host.py\nISA probe, INT8 gate", TEAL_L, TEAL, size=7.2)
    box(ax, 36, 3, 28, 7, "benchmark.py\nORT latency and accuracy", TEAL_L, TEAL, size=7.2)
    box(ax, 68, 3, 28, 7, "export.py, zoo/\nONNX export, models, data", TEAL_L, TEAL, size=7.2)
    for y0, y1 in ((60, 55), (36, 30), (19, 14)):
        arrow(ax, (50, y0), (50, y1))
    save(fig, "fig_architecture")


# ----------------------------------------------------------------------------- 2. pipeline order
def fig_pipeline():
    fig, ax = canvas(7.0, 2.3, xlim=(0, 100), ylim=(0, 24))
    stages = [("1  Prune", "gradual\nstructured cuts", GREEN_L, GREEN), ("2  Recover", "BN statistics\nre-estimated", TEAL_L, TEAL),
              ("3  Distill", "KD / DKD, original\nas teacher", PURPLE_L, PURPLE),
              ("4  Quantize", "INT8 or FP16,\nalways last", AMBER_L, AMBER), ("5  Verify", "accuracy, latency\non the .onnx", BLUE_L, BLUE)]
    w, gap = 16.6, 3.3
    for i, (t, d, fc, ec) in enumerate(stages):
        x = 1.2 + i * (w + gap)
        box(ax, x, 6, w, 12, f"{t}\n\n{d}", fc, ec, size=6.9)
        if i:
            arrow(ax, (x - gap, 12), (x, 12))
    ax.text(50, 2.2, "Quantization blocks further pruning and training, so it is the last transformation applied.",
            ha="center", fontsize=7.2, color=MUTED)
    save(fig, "fig_pipeline")



# ----------------------------------------------------------------------------- 1b. overall system
def fig_system():
    fig, ax = canvas(7.0, 5.0, ylim=(0, 72))
    group(ax, 0.5, 8, 17.5, 58, "INPUTS", "#fdf8ee", AMBER, AMBER)
    box(ax, 1.8, 46, 15, 12, "Trained FP32\nmodel\n(.pt file)", AMBER_L, AMBER, size=6.2)
    box(ax, 1.8, 29, 15, 12, "Dataset:\ntrain, test,\ncalibration", AMBER_L, AMBER, size=6.2)
    box(ax, 1.8, 12, 15, 12, "Request:\nrecipe or\nbudget", AMBER_L, AMBER, size=6.2)
    group(ax, 20, 8, 60, 58, "COMPRESSKIT", "#f6fafc", BLUE, BLUE)
    box(ax, 22, 53, 17.5, 8, "Entry point\nCLI, API, notebook", BLUE_L, BLUE, size=6.0)
    box(ax, 41, 53, 17.5, 8, "Host probe\nCPU features,\nINT8 gate", TEAL_L, TEAL, size=6.0)
    box(ax, 60, 53, 18.5, 8, "Pipeline controller\norder check", BLUE_L, BLUE, size=6.0)
    st = [("Prune", GREEN_L, GREEN), ("Recover", TEAL_L, TEAL), ("Distill", PURPLE_L, PURPLE), ("QAT", PURPLE_L, PURPLE), ("Quantize", AMBER_L, AMBER)]
    for i, (t, fc, ec) in enumerate(st):
        x = 22 + i * 11.6
        box(ax, x, 38, 9.8, 8, t, fc, ec, size=6.6, bold=True)
        if i:
            arrow(ax, (x - 1.8, 42), (x, 42))
    arrow(ax, (69, 53), (69, 49.5)), arrow(ax, (69, 49.5), (26.9, 49.5)), arrow(ax, (26.9, 49.5), (26.9, 46.2))
    ax.text(48, 36.3, "fixed order; Quantize always last", ha="center", fontsize=5.6, color=MUTED)
    box(ax, 22, 24, 26, 9, "Search layer\nbudget, sensitivity,\nescalation ladder", ROSE_L, RED, size=6.0)
    arrow(ax, (35, 33.2), (35, 37.8), "", size=5.6)
    box(ax, 52, 24, 26, 9, "Reporting\nHTML report, results.json", AMBER_L, AMBER, size=6.0)
    box(ax, 22, 10.5, 56, 9, "Measurement after every stage\nexport ONNX, run in ONNX Runtime: accuracy, latency, size", TEAL_L, TEAL, size=6.0)
    arrow(ax, (48, 38), (48, 19.7), dashed=True)
    arrow(ax, (65, 19.7), (65, 23.8), dashed=True)
    group(ax, 82, 8, 17.5, 58, "OUTPUTS", "#eef8f8", TEAL, TEAL)
    box(ax, 83.3, 46, 15, 12, "model.onnx\nINT8 or FP16", TEAL_L, TEAL, size=6.2)
    box(ax, 83.3, 29, 15, 12, "report.html\ngraphs, tables", TEAL_L, TEAL, size=6.2)
    box(ax, 83.3, 12, 15, 12, "results.json\nmeasurements", TEAL_L, TEAL, size=6.2)
    arrow(ax, (71.8, 42), (83.3, 52)), arrow(ax, (78, 28.5), (83.3, 35)), arrow(ax, (78, 15), (83.3, 18))
    arrow(ax, (16.8, 52), (21.9, 43)), arrow(ax, (16.8, 35), (21.9, 38.5)), arrow(ax, (16.8, 18), (21.9, 26))
    box(ax, 20, 0.5, 79.5, 5, "Target host: compresskit bench and verify_onnx.py re-measure the delivered .onnx against the original", BLUE_L, BLUE, size=5.8)
    arrow(ax, (91, 8), (91, 5.6), dashed=True)
    save(fig, "fig_system")


# ----------------------------------------------------------------------------- 2b. pipeline in detail
def fig_pipeline_detail():
    fig, ax = canvas(7.0, 5.3, ylim=(0, 76))
    chain = [("0  Baseline", "measure original,\nkeep as teacher", BLUE_L, BLUE), ("1  Prune", "gradual\nstructured cuts", GREEN_L, GREEN),
             ("2  Recover", "BN statistics\nre-estimated", TEAL_L, TEAL), ("3  Distill", "optional extra\nKD / DKD epochs", PURPLE_L, PURPLE),
             ("4  QAT", "optional, STE\nfake quantization", PURPLE_L, PURPLE), ("5  Quantize", "INT8 or FP16,\nalways last", AMBER_L, AMBER)]
    w, gap = 15.2, 1.7
    xs = [0.4 + i * (w + gap) for i in range(6)]
    for (t, d, fc, ec), x in zip(chain, xs):
        box(ax, x, 62, w, 12, f"{t}\n{d}", fc, ec, size=5.6)
    for a_, b_ in zip(xs, xs[1:]):
        arrow(ax, (a_ + w, 68), (b_, 68))
    group(ax, 0.4, 27, 60, 30, "STAGE 1 IN DETAIL, repeated for each of S steps", "#f4faf7", GREEN, GREEN)
    sub = [("Cut r/S\nDepGraph,\naligned", GREEN_L, GREEN), ("Recalibrate\nBN\n(8 batches)", TEAL_L, TEAL), ("Record\naccuracy\nafter cut", BLUE_L, BLUE),
           ("DKD fine-\ntune E\nepochs", PURPLE_L, PURPLE), ("Recalibrate\nBN, record\naccuracy", TEAL_L, TEAL)]
    sw, sg = 10.2, 1.5
    sxs = [1.6 + i * (sw + sg) for i in range(5)]
    for (t, fc, ec), x in zip(sub, sxs):
        box(ax, x, 38, sw, 12, t, fc, ec, size=5.4)
    for a_, b_ in zip(sxs, sxs[1:]):
        arrow(ax, (a_ + sw, 44), (b_, 44))
    ax.annotate("", xy=(sxs[0] + sw / 2, 37.6), xytext=(sxs[4] + sw / 2, 37.6), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9, connectionstyle="arc3,rad=-0.25"))
    ax.text(30, 29.2, "the next step starts from the fine-tuned model", ha="center", fontsize=5.6, color=MUTED)
    arrow(ax, (xs[1] + w / 2, 62), (xs[1] + w / 2, 57.3), dashed=True)
    group(ax, 63, 27, 36.6, 30, "STAGE 5 IN DETAIL", "#fdf8ee", AMBER, AMBER)
    q = [("Export\nFP32\nONNX", BLUE_L, BLUE), ("Calibrate\n(train),\nper-channel", AMBER_L, AMBER), ("Exclude\nsensitive\nnodes", ROSE_L, RED), ("Coverage\naudit\nof ops", TEAL_L, TEAL)]
    qw, qg = 7.6, 1.3
    qxs = [64.3 + i * (qw + qg) for i in range(4)]
    for (t, fc, ec), x in zip(q, qxs):
        box(ax, x, 38, qw, 12, t, fc, ec, size=5.1)
    for a_, b_ in zip(qxs, qxs[1:]):
        arrow(ax, (a_ + qw, 44), (b_, 44))
    arrow(ax, (xs[5] + w / 2, 62), (xs[5] + w / 2, 57.3), dashed=True)
    group(ax, 0.4, 1, 99.2, 21, "AFTER EVERY STAGE", "#eef8f8", TEAL, TEAL)
    m = [("Export model\nto ONNX", BLUE_L, BLUE), ("ONNX Runtime:\n1 thread, batch 1,\n50 warm-up, 200 runs", TEAL_L, TEAL),
         ("Top-1 accuracy of\nthe exported file", TEAL_L, TEAL), ("Record: accuracy,\nmedian and p95 ms,\nsize, parameters", AMBER_L, AMBER)]
    mw, mg = 22, 2.4
    mxs = [2.5 + i * (mw + mg) for i in range(4)]
    for (t, fc, ec), x in zip(m, mxs):
        box(ax, x, 3.5, mw, 12, t, fc, ec, size=5.7)
    for a_, b_ in zip(mxs, mxs[1:]):
        arrow(ax, (a_ + mw, 9.5), (b_, 9.5))
    arrow(ax, (50, 27), (50, 22.2), dashed=True)
    ax.text(51.5, 24.6, "every stage is measured, so the cost of each step is visible", fontsize=5.6, color=MUTED, ha="left", va="center")
    save(fig, "fig_pipeline_detail")

# ----------------------------------------------------------------------------- 3. DFDs
def _entity(ax, x, y, w, h, t):
    box(ax, x, y, w, h, t, AMBER_L, AMBER, round_=False, size=7.8)


def _process(ax, cx, cy, r, t, size=7.4, fc=BLUE_L, ec=BLUE):
    ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=0.9))
    ax.text(cx, cy, t, ha="center", va="center", fontsize=size, linespacing=1.2)


def _store(ax, x, y, w, h, t):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=TEAL_L, ec="none"))
    ax.plot([x, x + w], [y + h, y + h], color=TEAL, lw=1.1)
    ax.plot([x, x + w], [y, y], color=TEAL, lw=1.1)
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.4)


def fig_dfd0():
    fig, ax = canvas(6.4, 2.7, ylim=(0, 30))
    _entity(ax, 2, 11, 20, 8, "User\n(ML engineer)")
    _process(ax, 50, 15, 11, "0\ncompresskit\ncompression\nsystem")
    _entity(ax, 78, 21, 20, 7, "Dataset\n(CIFAR-10, HAR, folder)")
    _entity(ax, 78, 2, 20, 7, "Target runtime\n(ONNX Runtime)")
    arrow(ax, (22, 16.5), (39, 16.5), "trained FP32 model,\nbudget / config", side=3.5)
    arrow(ax, (39, 13.5), (22, 13.5), ".onnx + HTML report\n+ results.json", side=-3.7)
    arrow(ax, (78, 24), (58, 20.5), "train / test /\ncalibration images", size=6.8, text_xy=(70, 25.5))
    arrow(ax, (58, 9.5), (78, 6), "compressed graph", size=6.8, text_xy=(70, 6.3))
    save(fig, "fig_dfd0")


def fig_dfd1():
    fig, ax = canvas(7.0, 3.8, ylim=(0, 42))
    _entity(ax, 1, 17, 14, 8, "User")
    _process(ax, 33, 35, 5.4, "1\nProbe\nhost", size=6.8, fc=TEAL_L, ec=TEAL)
    _process(ax, 33, 22.5, 5.4, "2\nMeasure\nbaseline", size=6.8, fc=TEAL_L, ec=TEAL)
    _process(ax, 55, 29.5, 6.8, "3\nPrune and\nrecover", fc=GREEN_L, ec=GREEN)
    _process(ax, 55, 12.5, 6.8, "4\nDistill", fc=PURPLE_L, ec=PURPLE)
    _process(ax, 76, 21, 6.8, "5\nQuantize\nand export", fc=AMBER_L, ec=AMBER)
    _process(ax, 92.5, 21, 5.8, "6\nVerify and\nreport", size=6.6)
    _store(ax, 22, 0.5, 22, 3.8, "D1  checkpoints / .onnx")
    _store(ax, 62, 0.5, 22, 3.8, "D2  results.json")
    arrow(ax, (15, 24), (27.9, 33.5), "config", size=6.5)
    arrow(ax, (15, 20), (27.6, 21.5), "model", size=6.5)
    arrow(ax, (38.4, 34), (48.6, 31.5))
    arrow(ax, (38.4, 23), (48.6, 27.5))
    arrow(ax, (55, 22.6), (55, 19.4), "pruned model", size=6.2)
    arrow(ax, (61.7, 28), (69.5, 23.5))
    arrow(ax, (61.7, 14), (69.5, 18.5))
    arrow(ax, (82.8, 21), (86.7, 21))
    arrow(ax, (92.5, 15.2), (72, 4.3), "", rad=0.0)
    arrow(ax, (33, 17.1), (33, 4.4), "", dashed=True)
    save(fig, "fig_dfd1")


def fig_dfd2():
    fig, ax = canvas(7.0, 3.5, ylim=(0, 38))
    _process(ax, 12, 19, 8.2, "3.1\nDependency\ngraph\n(DepGraph)", size=6.8, fc=TEAL_L, ec=TEAL)
    _process(ax, 35, 19, 8.2, "3.2\nShared ratio\nper residual\ngroup", size=6.8, fc=BLUE_L, ec=BLUE)
    _process(ax, 58, 19, 8.2, "3.3\nSIMD-aligned\nkept count\nper layer", size=6.8, fc=PURPLE_L, ec=PURPLE)
    _process(ax, 81, 19, 8.2, "3.4\nPrune on a\ndeep copy,\nfine-tune", size=6.8, fc=GREEN_L, ec=GREEN)
    for a, b in ((20.4, 26.8), (43.4, 49.8), (66.4, 72.8)):
        arrow(ax, (a, 19), (b, 19))
    _entity(ax, 2, 30, 20, 6, "FP32 model")
    arrow(ax, (12, 30), (12, 27.4))
    _store(ax, 38, 1.5, 28, 3.5, "per-layer ratios from sensitivity")
    arrow(ax, (52, 5), (35, 10.8), dashed=True)
    arrow(ax, (52, 5), (58, 10.8), dashed=True)
    _entity(ax, 74, 1.5, 24, 6, "pruned model +\nprune_history, channels")
    arrow(ax, (81, 10.8), (86, 7.6))
    save(fig, "fig_dfd2")


# ----------------------------------------------------------------------------- 4. UML
def fig_usecase():
    fig, ax = canvas(6.4, 3.9, ylim=(0, 44))
    ax.add_patch(FancyBboxPatch((24, 2), 62, 40, boxstyle="round,pad=0.02,rounding_size=1.5", fc="#fafbfa", ec=BLUE))
    ax.text(55, 40, "compresskit", ha="center", fontsize=8.5, fontweight="bold")
    cases = ["Train FP32 baseline", "Compress with fixed settings", "Compress to a stated budget",
             "Inspect layer sensitivity", "Benchmark on target host", "Verify .onnx against original",
             "Read detailed HTML report"]
    ys = [35, 30, 25, 20, 15, 10, 5]
    UC = {35: (GREEN_L, GREEN), 30: (GREEN_L, GREEN), 25: (AMBER_L, AMBER), 20: (ROSE_L, RED), 15: (TEAL_L, TEAL), 10: (TEAL_L, TEAL), 5: (BLUE_L, BLUE)}
    for t, y in zip(cases, ys):
        ax.add_patch(matplotlib.patches.Ellipse((55, y), 40, 3.8, fc=UC[y][0], ec=UC[y][1], lw=0.8))
        ax.text(55, y, t, ha="center", va="center", fontsize=7.4)
    ax.text(7, 19.5, "ML engineer", ha="center", fontsize=8, fontweight="bold")
    ax.plot([7], [30], "o", ms=7, mfc=BLUE_L, mec=BLUE)
    ax.plot([7, 7], [29, 26], color=INK), ax.plot([4, 10], [28, 28], color=INK)
    ax.plot([7, 4.5], [26, 23.5], color=INK), ax.plot([7, 9.5], [26, 23.5], color=INK)
    for y in ys:
        ax.plot([11, 35], [27, y], color=MUTED, lw=0.6)
    ax.text(93, 24, "Target\nhost CPU", ha="center", fontsize=8, fontweight="bold")
    ax.plot([93], [30], "s", ms=7, mfc=TEAL_L, mec=TEAL)
    for y in (15, 10, 20):
        ax.plot([75, 90], [y, 29], color=MUTED, lw=0.6)
    save(fig, "fig_usecase")


def fig_sequence():
    fig, ax = canvas(7.0, 4.6, ylim=(0, 52))
    names = ["User", "CLI /\nnotebook", "search.\ncompress", "Pipeline", "Pruner,\nDistiller,\nQuantizer", "benchmark\n(ORT)"]
    xs = [8, 24, 41, 58, 76, 92]
    HC = [(AMBER_L, AMBER), (BLUE_L, BLUE), (BLUE_L, BLUE), (GREEN_L, GREEN), (PURPLE_L, PURPLE), (TEAL_L, TEAL)]
    for (n, x), (fc, ec) in zip(zip(names, xs), HC):
        box(ax, x - 7, 44, 14, 6, n, fc, ec, size=7)
        ax.plot([x, x], [44, 1], color=RULE, lw=0.8, ls="--")
    msgs = [(0, 1, "compress --budget", 40), (1, 2, "Budget.parse", 37), (2, 3, "baseline stage", 34),
            (3, 5, "measure FP32", 31), (2, 4, "sensitivity analysis", 28), (2, 3, "run ladder rung", 25),
            (3, 4, "prune / distill / quantize", 22), (3, 5, "measure candidate (.onnx)", 19),
            (2, 3, "re-measure at full fidelity", 14),
            (2, 1, "SearchResult", 9), (1, 0, "model.onnx + report.html", 5)]
    for a, b, t, y in msgs:
        if a == b:
            ax.annotate("", xy=(xs[a] + 0.4, y - 2), xytext=(xs[a] + 0.4, y), arrowprops=dict(
                arrowstyle="-|>", color=INK, lw=0.9, connectionstyle="arc3,rad=-1.6"))
            ax.text(xs[a] + 4, y - 1, t, fontsize=6.8, va="center", color=MUTED)
        else:
            arrow(ax, (xs[a], y), (xs[b], y))
            ax.text((xs[a] + xs[b]) / 2, y + 1.3, t, ha="center", fontsize=6.8, color=MUTED)
    save(fig, "fig_sequence")


def fig_classes():
    fig, ax = canvas(7.0, 4.8, ylim=(0, 60))
    H = 17.5

    def klass(x, y, w, title, rows, fc=GREEN_L, ec=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="square,pad=0", fc="white", ec=ec, lw=0.9))
        ax.add_patch(FancyBboxPatch((x, y + H - 3.6), w, 3.6, boxstyle="square,pad=0", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + H - 1.8, title, ha="center", va="center", fontsize=7.6, fontweight="bold")
        for i, r in enumerate(rows):
            ax.text(x + 1, y + H - 3.6 - 1.9 - 2.5 * i, r, fontsize=6.4, va="center", family="monospace")

    X = (1, 35, 69)
    W = 30
    klass(X[0], 41, W, "Budget", ["max_latency_ms", "max_accuracy_drop_pp", "min_accuracy", "max_size_mb", "parse(spec)"], BLUE_L, BLUE)
    klass(X[1], 41, W, "SearchResult", ["budget: Budget", "attempts: list[Attempt]", "chosen: Attempt", "sensitivity", "host"], BLUE_L, BLUE)
    klass(X[2], 41, W, "Attempt", ["label, plan", "result: PipelineResult", "failures: list[str]", "config", "full_fidelity"], BLUE_L, BLUE)
    klass(X[0], 21, W, "Pipeline", ["stages: list[Stage]", "prune(...)", "run(verbose)", "", ""], GREEN_L, GREEN)
    klass(X[1], 21, W, "PipelineResult", ["baseline: Measurement", "stages: list", "quantization", "prune_stats", "prune_history"], GREEN_L, GREEN)
    klass(X[2], 21, W, "Measurement", ["stage, precision", "accuracy", "latency: Latency", "size_mb, parameters", "artifact, host, notes"], AMBER_L, AMBER)
    klass(X[0], 1, W, "StructuredPruner", ["config: PruneConfig", "model (a deep copy)", "step(), run()", "stats(): PruneStats", ""], GREEN_L, GREEN)
    klass(X[1], 1, W, "SensitivityReport", ["layers: list", "ranked()", "most_sensitive(k)", "prune_ratios(base)", "exclude_nodes(path)"], ROSE_L, RED)
    klass(X[2], 1, W, "DataBundle", ["input_shape", "num_classes, class_names", "train_loader()", "test_loader()", "calibration_loader()"], TEAL_L, TEAL)
    arrow(ax, (31, 49), (35, 49)), arrow(ax, (65, 49), (69, 49))
    arrow(ax, (84, 41), (84, 38.5)), arrow(ax, (50, 41), (50, 38.5), dashed=True)
    arrow(ax, (31, 29), (35, 29)), arrow(ax, (65, 29), (69, 29))
    arrow(ax, (16, 21), (16, 18.5), dashed=True)
    save(fig, "fig_classes")


# ----------------------------------------------------------------------------- 5. ladder
def fig_ladder():
    fig, ax = canvas(7.0, 3.0, ylim=(0, 30))
    rungs = [("quantize\nonly", "INT8 / FP16\nno retraining"), ("mixed\nprecision", "most sensitive\nlayers kept FP32"),
             ("QAT", "fake-quant, STE,\nfine-tune"), ("prune\n(ratio ladder)", "escalating ratios,\nthen recover"),
             ("prune +\ndistill", "gradual cuts,\nDKD, quantize")]
    w = 17
    LAD = [(GREEN_L, GREEN), (TEAL_L, TEAL), (BLUE_L, BLUE), (AMBER_L, AMBER), (ROSE_L, RED)]
    for i, (t, d) in enumerate(rungs):
        x = 1 + i * (w + 2.9)
        y = 5 + i * 2.2
        box(ax, x, y, w, 12, f"{t}\n\n{d}", LAD[i][0], LAD[i][1], size=6.9)
        if i:
            arrow(ax, (x - 2.9, y - 2.2 + 6), (x, y + 6), "unmet" if i == 1 else "", size=6.3, side=2.4)
    ax.text(50, 1.4, "Cheapest rung first (green); each later rung runs only if the budget is still unmet. If none meets it, the best trade-off is returned.",
            ha="center", fontsize=7.4, color=MUTED)
    save(fig, "fig_ladder")


# ----------------------------------------------------------------------------- 6. groups
def fig_groups():
    g = load("groups")["multi_layer_groups"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    sizes = [len(x) for x in g]
    labels = [min(x, key=lambda n: (len(n), n)) for x in g]
    order = sorted(range(len(g)), key=lambda i: (labels[i], -sizes[i]))
    ax.barh(range(len(g)), [sizes[i] for i in order], color=[AMBER if sizes[i] > 2 else BLUE for i in order], height=0.62)
    ax.set_yticks(range(len(g)), [labels[i] for i in order], fontsize=6.6, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("layers tied to one shared pruning ratio")
    ax.set_title("ResNet-20: the 12 dependency groups that tie two or more layers")
    for i, o in enumerate(order):
        ax.text(sizes[o] + 0.08, i, str(sizes[o]), va="center", fontsize=6.8)
    save(fig, "fig_groups")


# ----------------------------------------------------------------------------- 7. alignment
def fig_align():
    rows = load("align_golden")["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, width in zip(axs, (16, 64)):
        for align, c in ((8, BLUE), (16, GREEN), (32, AMBER)):
            pts = sorted((r["ratio"], r["kept"]) for r in rows if r["width"] == width and r["align"] == align)
            if pts:
                ax.step(*zip(*pts), where="mid", color=c, lw=1.4, marker="o", ms=3, label=f"align {align}")
        ax.set_title(f"layer width {width}")
        ax.set_xlabel("requested ratio"), ax.set_ylabel("channels kept")
        ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_align")
    return sorted({r["ratio"] for r in rows}), sorted({r["width"] for r in rows})


# ----------------------------------------------------------------------------- 8. channels
def fig_channels():
    m = load("models")["models"][0]
    ch = m["prune_stats"]["channels"]
    names = list(ch)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    xs = range(len(names))
    ax.bar(xs, [ch[n][0] for n in names], color=BLUE_L, ec=BLUE, lw=0.5, label="before", width=0.8)
    ax.bar(xs, [ch[n][1] for n in names], color=AMBER, label="after", width=0.5)
    ax.set_xticks(list(xs), names, rotation=70, ha="right", fontsize=5.8, family="monospace")
    ax.set_ylabel("output channels")
    ax.set_title("ResNet-20: channels per layer, 50% requested")
    ax.legend()
    save(fig, "fig_channels")


# ----------------------------------------------------------------------------- 9, 10. gradual
def fig_sawtooth():
    models = {m["name"]: m for m in load("models")["models"]}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, name in zip(axs, ("ResNet-20", "VGG-16-BN")):
        m = models[name]
        base = m["baseline"]["accuracy"] * 100
        xs, ys = [0], [base]
        for h in m["prune_history"]:
            xs += [h["step"] - 0.01, h["step"]]
            ys += [h["acc_after_cut"] * 100, h["acc_after_finetune"] * 100]
        ax.plot(xs, ys, color=GREEN, lw=1.3, marker="o", ms=3)
        ax.axhline(base, color=MUTED, lw=0.8, ls="--")
        ax.text(len(m["prune_history"]), base - 0.4, "baseline", ha="right", va="top", fontsize=7, color=MUTED)
        ax.set_xlabel("pruning step"), ax.set_ylabel("test accuracy (%)")
        ax.set_title(name)
        ax.set_xticks(range(len(m["prune_history"]) + 1))
    fig.tight_layout()
    save(fig, "fig_sawtooth")


def fig_oneshot():
    g = load("gradual_vs_oneshot")
    base = g["baseline_accuracy"] * 100
    vals = [("gradual\n3 steps x 2 epochs", g["gradual"]["history"][-1]["acc_after_finetune"] * 100, GREEN),
            ("one-shot\n1 step x 6 epochs", g["one_shot"]["history"][-1]["acc_after_finetune"] * 100, BLUE)]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.bar([0, 1], [v for _, v, _ in vals], color=[c for *_, c in vals], width=0.5)
    ax.axhline(base, color=MUTED, ls="--", lw=0.8)
    ax.text(1.45, base + 0.4, f"baseline {base:.2f}%", ha="right", fontsize=7, color=MUTED)
    for i, (_, v, _) in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.2f}%", ha="center", fontsize=8)
    ax.set_xticks([0, 1], [n for n, *_ in vals])
    ax.set_ylim(70, 95), ax.set_ylabel("accuracy after fine-tune (%)")
    ax.set_title("Equal fine-tune budget (6 epochs), one seed each")
    save(fig, "fig_oneshot")


# ----------------------------------------------------------------------------- 11. BN
def fig_bn():
    b = load("bn_recal")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    kept = b["kept_channel_indices"]
    ax.plot(range(16), b["running_var"]["original"], color=MUTED, marker="o", ms=3, label="original (16 channels)")
    ax.plot(range(len(kept)), b["running_var"]["stale_after_cut"], color=RED, marker="s", ms=3, label="after cut, stale statistics")
    ax.plot(range(len(kept)), b["running_var"]["recalibrated"], color=GREEN, marker="^", ms=3, label="after recalibration")
    ax.set_xlabel("channel (kept channels renumbered)"), ax.set_ylabel("running variance")
    ax.set_title(f"{b['layer']}, ratio {b['prune_ratio']}")
    ax.legend(fontsize=6.6)
    ax = axs[1]
    a = b["accuracy"]
    vals = [a["baseline"], a["after_cut_stale_bn"], a["after_bn_recalibration"]]
    ax.bar(range(3), [v * 100 for v in vals], color=[MUTED, RED, GREEN], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.2f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(3), ["original", "cut,\nstale BN", "cut,\nrecalibrated"], fontsize=7)
    ax.set_ylim(0, 105), ax.set_ylabel("test accuracy (%)")
    ax.set_title("One-shot 0.5 cut, no fine-tune")
    fig.tight_layout()
    save(fig, "fig_bn")


# ----------------------------------------------------------------------------- 12. softmax
def fig_softmax():
    d = load("distillation")
    cls, t = d["classes"], d["target"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, key, title in ((axs[0], "T1", "temperature 1"), (axs[1], "T4", f"temperature {d['temperature']:g}")):
        te, st = d[f"teacher_softmax_{key}"], d[f"student_softmax_{key}"]
        xs = range(len(cls))
        ax.bar([x - 0.2 for x in xs], te, width=0.4, color=GREEN, label="teacher")
        ax.bar([x + 0.2 for x in xs], st, width=0.4, color=AMBER, label="student")
        ax.set_xticks(list(xs), cls, rotation=60, ha="right", fontsize=6.8)
        ax.set_title(title)
        ax.axvspan(t - 0.5, t + 0.5, color=GREEN_L, zorder=0)
    axs[0].set_ylabel("probability"), axs[0].legend()
    fig.tight_layout()
    save(fig, "fig_softmax")
    return cls[t]


# ----------------------------------------------------------------------------- 13. quant weights
def fig_quant():
    q = load("quant_weights")
    e, fp, i8 = q["hist"]["edges"], q["hist"]["fp32"], q["hist"]["int8"]
    centres = [(a + b) / 2 for a, b in zip(e, e[1:])]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    ax.fill_between(centres, fp, step="mid", color=GREEN, alpha=0.35, label="FP32 weights")
    ax.step(centres, i8, where="mid", color=INK, lw=0.9, label="after INT8 round trip")
    ax.set_xlabel("weight value"), ax.set_ylabel("count")
    ax.set_title(f"{q['layer']}  {tuple(q['shape'])}")
    ax.legend()
    axs[1].bar(range(len(q["scales_per_output_channel"])), q["scales_per_output_channel"], color=GREEN, width=0.8)
    axs[1].set_xlabel("output channel"), axs[1].set_ylabel("INT8 scale")
    axs[1].set_title("per-channel scales")
    fig.tight_layout()
    save(fig, "fig_quant")


# ----------------------------------------------------------------------------- 14. sensitivity
def fig_sens():
    s = load("sensitivity")
    layers = sorted(s["layers"], key=lambda r: r["sqnr_db"])
    prot = set(s["protected_at_0.5"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(range(len(layers)), [r["sqnr_db"] for r in layers],
            color=[RED if r["name"] in prot else GREEN for r in layers], height=0.7)
    ax.set_yticks(range(len(layers)), [r["name"] for r in layers], fontsize=6, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("SQNR at the logits after INT8 round trip of one layer (dB, higher is safer)")
    ax.set_title("ResNet-20: layer sensitivity (red = protected at ratio 0.5)")
    save(fig, "fig_sens")


# ----------------------------------------------------------------------------- 15. stages
def fig_stages():
    models = load("models")["models"]
    labels = ["baseline", "pruned", "recovered", "INT8"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    cols = [GREEN, AMBER, BLUE]
    for ax, key, title, scale in ((axs[0], "accuracy", "accuracy (%)", 100), (axs[1], "size_mb", "size (MB, log)", 1),
                                  (axs[2], "median_ms", "median latency (ms, log)", 1)):
        for m, c in zip(models, cols):
            seq = [m["baseline"][key]] + [s[key] for s in m["stages"]]
            ax.plot(range(4), [v * scale for v in seq], marker="o", ms=3, color=c, lw=1.2, label=m["name"] + (" (smoke run)" if m["name"] == "ResNet-56" else ""))
        ax.set_xticks(range(4), labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        if key != "accuracy":
            ax.set_yscale("log")
    axs[0].legend(fontsize=6.6)
    fig.tight_layout()
    save(fig, "fig_stages")


def fig_pareto():
    models = load("models")["models"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    cols = [GREEN, AMBER, BLUE]
    for m, c in zip(models, cols):
        pts = [(m["baseline"]["size_mb"], m["baseline"]["accuracy"] * 100, "FP32")] + [
            (s["size_mb"], s["accuracy"] * 100, s["precision"].upper() if s["stage"].startswith("3") else s["stage"][2:])
            for s in m["stages"] if s["stage"] != "2_recover"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, color=c, lw=1.1, label=m["name"] + (" (smoke run)" if m["name"] == "ResNet-56" else ""))
        for x, y, t in pts:
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6, color=MUTED)
    ax.set_xscale("log"), ax.set_xlabel("model size (MB, log)"), ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy against size for the three recorded runs")
    ax.legend()
    save(fig, "fig_pareto")


# ----------------------------------------------------------------------------- 16. search
def fig_search():
    runs = load("search")["runs"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, r in zip(axs, runs):
        att = r["attempts"]
        base = r["baseline"]["accuracy"] * 100
        accs = [a["accuracy"] * 100 for a in att]
        ax.barh(range(len(att)), accs, color=[GREEN if a["met"] else RULE for a in att], height=0.6)
        ax.axvline(base, color=MUTED, ls="--", lw=0.8)
        ax.set_yticks(range(len(att)), [a["label"] for a in att], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(r["budget"].replace(",", ",\n"), fontsize=7)
        ax.set_xlabel("accuracy (%), latency at right")
        ax.set_xlim(0, 118)
        for i, a in enumerate(att):
            ax.text(101, i, f"{a['median_ms']:.2f} ms", va="center", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_search")


# ----------------------------------------------------------------------------- 17. timeline + tests
def fig_timeline():
    """Every bar is backed by something dated in the repo: commit dates, or the file
    modification dates of the survey, draft, training logs and design documents.
    The stretch with no recorded work is drawn as an axis break, not filled in."""
    log = subprocess.run(["git", "log", "--format=%ad|%s", "--date=short", "--reverse"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    from datetime import date
    D = date.fromisoformat
    days = Counter(l.split("|")[0] for l in log)
    d0 = D("2026-09-05")
    cut0, cut1, squeeze = D("2026-09-17"), D("2026-10-10"), 3.0

    def x(d):
        n = (D(d) - d0).days if isinstance(d, str) else (d - d0).days
        if D(str(d)) > cut1 if not isinstance(d, str) else D(d) > cut1:
            n -= ((cut1 - cut0).days + 1) - squeeze
        elif (D(d) if isinstance(d, str) else d) >= cut0:
            n = (cut0 - d0).days + squeeze / 2
        return n

    phases = [("Collect and read 13 papers; literature survey; architecture sketch", "2026-09-05", "2026-09-13"),
              ("Train ResNet-56 baseline (training logs)", "2026-09-05", "2026-09-07"),
              ("Documentation and system-design drafts; paper draft; demo script", "2026-09-10", "2026-09-16"),
              ("Package split; hardening of pruning, DKD and budget parser", "2026-09-12", "2026-09-16"),
              ("QAT fallback and latency-guided pruning", "2026-09-16", "2026-09-16"),
              ("Budget-search polish: mixed precision, re-measure, de-duplication", "2026-10-11", "2026-10-11"),
              ("Gradual pruning, one-notebook Kaggle workflow, ONNX verification, ResNet-56 notebook", "2026-10-26", "2026-10-27")]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for i, (t, a, b) in enumerate(phases):
        s_, e_ = x(a), x(b) + 0.8
        ax.barh(i, e_ - s_, left=s_, color=GREEN, height=0.5)
        label_x = e_ + 0.5 if e_ < 30 else s_ - 0.5
        ax.text(label_x, i, t, va="center", ha="left" if e_ < 30 else "right", fontsize=6.4, zorder=5, bbox=dict(fc="white", ec="none", pad=0.8))
    ax.scatter([x(d) + 0.4 for d in days], [len(phases) - 0.1] * len(days), s=[12 * n for n in days.values()], color=INK, zorder=3)
    ax.text(0, len(phases) + 0.55, "commits per day (dot size)", fontsize=6.3, color=MUTED)
    brk = (cut0 - d0).days
    ax.axvspan(brk - 0.1, brk + squeeze + 0.1, facecolor="white", edgecolor=RULE, hatch="////", lw=0, zorder=1)
    ticks = [("2026-09-05", "5 Sep"), ("2026-09-12", "12 Sep"), ("2026-09-16", "16 Sep"), ("2026-10-11", "11 Oct"), ("2026-10-27", "27 Oct")]
    ax.set_xticks([x(d) for d, _ in ticks], [l for _, l in ticks], fontsize=7)
    ax.set_yticks([]), ax.invert_yaxis()
    ax.set_ylim(len(phases) + 1.0, -1.0)
    ax.set_xlim(-0.5, x("2026-10-27") + 1.5), ax.spines["left"].set_visible(False)
    save(fig, "fig_timeline")
    return len(log), len(days)


def fig_tests():
    counts = {"test_core": 104, "test_pipeline": 31, "test_search": 21, "test_workflow": 4, "test_zoo": 2}
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    ax.barh(list(counts), list(counts.values()), color=GREEN, height=0.6)
    ax.invert_yaxis()
    for i, v in enumerate(counts.values()):
        ax.text(v + 1, i, str(v), va="center", fontsize=7.5)
    ax.set_xlabel("collected tests (162 in total; 1 slow end-to-end test deselected by default)")
    save(fig, "fig_tests")


if __name__ == "__main__":
    for fn in (fig_architecture, fig_system, fig_pipeline, fig_pipeline_detail, fig_dfd0, fig_dfd1, fig_dfd2, fig_usecase, fig_sequence, fig_classes,
               fig_ladder, fig_groups, fig_align, fig_channels, fig_sawtooth, fig_oneshot, fig_bn, fig_softmax,
               fig_quant, fig_sens, fig_stages, fig_pareto, fig_search, fig_timeline, fig_tests):
        out = fn()
        if out is not None:
            print("   ", fn.__name__, out)
