"""Regenerate the vector figures from the repository's recorded measurements."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "results"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "axes.axisbelow": True,
    }
)
GREEN, RUST, BLUE = "#125442", "#a14731", "#315776"


def data(name):
    return json.loads((RESULTS / name).read_text())


def save(fig, name):
    fig.savefig(PAPER / name, bbox_inches="tight")
    plt.close(fig)


def retrieval():
    records = data("scifact.json")["results"]
    labels = ["BM25", "Character LSA", "Fusion", "Learned fusion"]
    order = ["bm25", "dense", "hybrid", "learned_hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ["recall@10", "ndcg@10"], ["Recall@10", "nDCG@10"]):
        for y, (name, label) in enumerate(zip(order, labels)):
            d = records[name][metric]
            v = d["mean"]
            lo, hi = d["ci95"]
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", capsize=3, color=GREEN)
            ax.set_yticks(range(4), labels)
        ax.grid(axis="x", alpha=0.2)
        ax.set(xlim=(0, 1), xlabel=title, xticks=[0, 0.25, 0.5, 0.75, 1])
    axes[0].invert_yaxis()
    save(fig, "retrieval-intervals.pdf")


def thresholds():
    fig, ax = plt.subplots(figsize=(5.8, 3.5), layout="constrained")
    for mode, label, color in [(True, "Grouped", GREEN), (False, "Naive", RUST)]:
        rows = [r for r in data("controlled.json")["duplicate_injection"] if r["dependency"] == mode]
        ax.plot(range(len(rows)), [r["balance"] for r in rows], label=label, marker="o", color=color)
    ax.set(
        xticks=range(5),
        xticklabels=["1", "2", "4", "8", "16"],
        xlabel="Copies of the opposing report",
        ylabel="Signed evidence balance",
        ylim=(-1, 0.13),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    save(fig, "duplicate-injection.pdf")


def neural():
    rows = json.loads((PAPER / "paired_audit.json").read_text())["records"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), layout="constrained")
    for ax, (key, labels) in zip(
        axes,
        [
            ("claims", ["Decisive", "Correct decisive", "Correct + aligned"]),
            ("cites", ["Cited passages", "Aligned passages"]),
        ],
    ):
        for i, (mode, color) in enumerate([("ungated", BLUE), ("gated", GREEN)]):
            decisive = [r for r in rows if r[mode]["verdict"] in {"supported", "refuted"}]
            correct = [r for r in decisive if r[mode]["verdict"] == r["gold"]]
            vals = (
                [len(decisive), len(correct), sum(r[mode]["matched"] > 0 for r in correct)]
                if key == "claims"
                else [sum(r[mode]["citations"] for r in rows), sum(r[mode]["matched"] for r in rows)]
            )
            bars = ax.barh(
                np.arange(len(vals)) + (i - 0.5) * 0.34, vals, height=0.32, label=mode.title(), color=color
            )
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set(
            yticks=range(len(labels)), yticklabels=labels, xlabel="Claims" if key == "claims" else "Passages"
        )
        ax.set_xlim(0, 65 if key == "claims" else 100)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, "paired-grounding.pdf")


if __name__ == "__main__":
    retrieval()
    thresholds()
    neural()
