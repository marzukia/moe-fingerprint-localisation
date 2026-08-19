#!/usr/bin/env python3
"""Regenerate the paper figures from the reported result values."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
FIGURE_DIR = PAPER_DIR / "figures"

ACCENT = "#2563eb"
MUTED = "#94a3b8"
DARK = "#0f172a"
GRID = "#e2e8f0"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=DARK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figure_1():
    out_path = FIGURE_DIR / "fig1-trigger-overlap-vs-null.png"

    data = {
        "P90": {
            "Seed 42": {"trigger": 87, "null_ci_high": 132},
            "Seed 123": {"trigger": 85, "null_ci_high": 134},
        },
        "P95": {
            "Seed 42": {"trigger": 6, "null_ci_high": 22},
            "Seed 123": {"trigger": 6, "null_ci_high": 21},
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
    fig.suptitle(
        "Trigger overlap is below the within-model null CI-high range",
        fontsize=12,
        fontweight="bold",
        color=DARK,
        x=0.5,
        y=1.0,
    )
    fig.text(
        0.5,
        0.915,
        "Secondary within-model diagnostic; matched across-model null tables are primary.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )

    x = [0, 1]
    width = 0.32
    labels = ["Seed 42", "Seed 123"]

    for ax, threshold in zip(axes, ["P90", "P95"]):
        trigger = [data[threshold][label]["trigger"] for label in labels]
        null_ci = [data[threshold][label]["null_ci_high"] for label in labels]

        ax.bar(
            [xi - width / 2 for xi in x],
            trigger,
            width=width,
            label="Trigger overlap",
            color=ACCENT,
        )
        ax.bar(
            [xi + width / 2 for xi in x],
            null_ci,
            width=width,
            label="Max within-model null CI-high",
            color=MUTED,
        )

        for xi, value in zip([xi - width / 2 for xi in x], trigger):
            ax.text(
                xi, value + max(null_ci) * 0.015, str(value), ha="center", fontsize=8
            )
        for xi, value in zip([xi + width / 2 for xi in x], null_ci):
            ax.text(
                xi, value + max(null_ci) * 0.015, str(value), ha="center", fontsize=8
            )

        ax.set_title(threshold, fontsize=10, color=DARK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Three-way overlap size")
        ax.set_ylim(0, max(max(null_ci), max(trigger)) * 1.18)
        style_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_2():
    out_path = FIGURE_DIR / "fig2-weaker-implant-screening.png"

    rows = [
        # lr_x1e-3, max_steps, clean_ppl, sampled_ffr
        (5, 25, 1.3456, 0.96),
        (5, 50, 1.3672, 0.96),
        (3, 25, 1.3272, 0.96),
        (3, 50, 1.3273, 0.96),
        (3, 100, 1.3359, 1.00),
        (1, 25, 1.3070, 0.96),
        (1, 50, 1.3404, 0.96),
        (1, 100, 1.3344, 0.96),
    ]

    marker_by_steps = {25: "o", 50: "s", 100: "^"}
    color_by_steps = {25: ACCENT, 50: "#f59e0b", 100: "#10b981"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.suptitle(
        "Weaker-implant single-trigger screening",
        fontsize=12,
        fontweight="bold",
        color=DARK,
        x=0.5,
        y=1.0,
    )
    fig.text(
        0.5,
        0.915,
        "Trigger 0, seed 42; single-trigger sweep only.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )

    for ax, value_index, ylabel, ylim in [
        (axes[0], 2, "Clean PPL ratio", (1.29, 1.385)),
        (axes[1], 3, "Sampled FFR", (0.94, 1.015)),
    ]:
        for max_steps in [25, 50, 100]:
            subset = sorted(
                (row for row in rows if row[1] == max_steps),
                key=lambda row: row[0],
            )
            xs = [row[0] for row in subset]
            ys = [row[value_index] for row in subset]
            ax.plot(
                xs,
                ys,
                marker=marker_by_steps[max_steps],
                color=color_by_steps[max_steps],
                linewidth=1.2,
                markersize=5,
                label=f"Max {max_steps} steps",
            )

        ax.set_xticks([1, 3, 5])
        ax.set_xticklabels(["1e-3", "3e-3", "5e-3"])
        ax.set_xlabel("Learning rate")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        style_ax(ax)

    axes[0].set_title("Clean PPL ratio", fontsize=10, color=DARK)
    axes[1].set_title("Sampled FFR", fontsize=10, color=DARK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_3():
    out_path = FIGURE_DIR / "fig3-router-movement-by-prompt-type.png"

    trigger_labels = ["Trigger 0", "Trigger 1", "Trigger 2"]
    prompt_types = ["trigger", "null", "clean"]
    values = {
        "trigger": [0.3977, 0.4849, 0.2724],
        "null": [0.3318, 0.4550, 0.2972],
        "clean": [0.2597, 0.3772, 0.2390],
    }
    colors = {"trigger": ACCENT, "null": MUTED, "clean": "#10b981"}

    x = range(len(trigger_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.suptitle(
        "Mean top-k routing change by prompt type",
        fontsize=12,
        fontweight="bold",
        color=DARK,
        x=0.5,
        y=1.0,
    )
    fig.text(
        0.5,
        0.915,
        "Averaged over seeds 42 and 123.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )

    for offset, prompt_type in enumerate(prompt_types):
        xs = [xi + (offset - 1) * width for xi in x]
        ax.bar(
            xs,
            values[prompt_type],
            width=width,
            label=prompt_type.capitalize(),
            color=colors[prompt_type],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(trigger_labels)
    ax.set_ylabel("Mean top-k routing change")
    ax.set_ylim(0, 0.58)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [make_figure_1(), make_figure_2(), make_figure_3()]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
