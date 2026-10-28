"""Plot matched robustness deltas from the audited analysis export."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from braintrust_viz import clean_axes, savefig, use_braintrust_theme

BASE = Path(__file__).resolve().parent
MODELS = ["luna", "jev", "clef", "d1", "d1-ultra"]
NAMES = ["Luna", "Jev", "Clef", "D1", "D1 Ultra"]
COLORS = ["#956BDD", "#2C1FEB", "#635B71", "#C778AF", "#48318B"]
CONDITIONS = ["blur_1", "blur_2", "contrast_0.5", "contrast_1.5", "shift_left_60", "shift_right_60"]
TITLES = [
    "Blur · radius 1 px",
    "Blur · radius 2 px",
    "Contrast · ×0.5",
    "Contrast · ×1.5",
    "Shift · 60 px left",
    "Shift · 60 px right",
]


def plot(root, analysis, metric, title, filename):
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.17, hspace=0.43, wspace=0.30)
    fig.text(0.06, 0.955, title, fontsize=23, weight="bold", ha="left")
    fig.text(
        0.06,
        0.905,
        "Class definitions supplied to every model, clean control, and transformed image",
        fontsize=13,
        color="#6B6B76",
    )
    controls = "  ·  ".join(
        f"{name} {analysis['models'][model]['conditions']['control']['accuracy']:.1%}"
        for model, name in zip(MODELS, NAMES)
    )
    fig.text(
        0.06, 0.855, "Fresh clean-control accuracy: " + controls, fontsize=11.5, color="#6B6B76"
    )
    lo, hi = 0.0, 0.0
    for ax, condition, label in zip(axes.flat, CONDITIONS, TITLES):
        ax.axvline(0, color="#B4B4BE", lw=1.2, ls="--", zorder=1)
        for i, (model, color) in enumerate(zip(MODELS, COLORS)):
            row = analysis["models"][model]["conditions"][condition]
            value = row[metric]
            low, high = row[metric + "_bootstrap_95"]
            lo, hi = min(lo, low), max(hi, high)
            # Draw interval separately: a percentile interval need not contain its point estimate.
            ax.hlines(i, low, high, color=color, lw=1.7, zorder=3)
            ax.vlines([low, high], i - 0.065, i + 0.065, color=color, lw=1.4, zorder=3)
            ax.scatter(value, i, s=52, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.set_title(label, loc="left", fontsize=13, weight="bold", pad=12)
        ax.set_yticks(np.arange(len(MODELS)), NAMES)
        ax.set_ylim(4.6, -0.6)
        clean_axes(ax, grid_axis="x")
    margin = max(1.0, (hi - lo) * 0.08)
    for ax in axes.flat:
        ax.set_xlim(lo - margin, hi + margin)
    metric_name = "accuracy" if metric == "change_pp" else "10-bin calibration error"
    fig.text(
        0.53,
        0.10,
        f"Change in {metric_name} versus matched clean control (percentage points)",
        ha="center",
        fontsize=13,
    )
    fig.text(
        0.06,
        0.04,
        "300 paired sources per condition · 95% pointwise, class-stratified bootstrap intervals · one call per image/condition",
        fontsize=10.5,
        color="#6B6B76",
    )
    if metric == "ece_change_pp":
        valid_counts = ", ".join(
            f"{name} {analysis['models'][model]['calibration_common_valid_sources']}"
            for model, name in zip(MODELS, NAMES)
        )
        fig.text(
            0.06,
            0.005,
            "ECE uses sources valid in all seven conditions: " + valid_counts,
            fontsize=10,
            color="#6B6B76",
        )
    for extension in ("png", "svg"):
        path = root / f"{filename}.{extension}"
        savefig(fig, path)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default="controlled-definitions-v1")
    args = parser.parse_args()
    root = BASE / args.run_prefix
    analysis = json.loads((root / "analysis.json").read_text())
    if analysis["prompt_condition"] != "aid-definitions-v1":
        raise ValueError("This chart requires definitions-matched predictions")
    use_braintrust_theme()
    plot(
        root, analysis, "change_pp", "Accuracy under blur, contrast, and shifts", "accuracy-deltas"
    )
    plot(
        root,
        analysis,
        "ece_change_pp",
        "Calibration under blur, contrast, and shifts",
        "calibration-deltas",
    )
    print(root / "accuracy-deltas.png")
    print(root / "calibration-deltas.png")


if __name__ == "__main__":
    main()
