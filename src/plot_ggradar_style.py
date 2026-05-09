from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["gemini", "renikud", "renikud_phonikud", "phonikud"]
MODEL_LABELS = {
    "gemini": "Gemini",
    "renikud": "Renikud vox",
    "renikud_phonikud": "Renikud phonikud",
    "phonikud": "Phonikud (ONNX)",
}
MODEL_COLORS = {
    "gemini": "black",
    "renikud": "steelblue",
    "renikud_phonikud": "seagreen",
    "phonikud": "tomato",
}
MODEL_LINESTYLES = {
    "gemini": "-",
    "renikud": "-",
    "renikud_phonikud": "--",
    "phonikud": "-.",
}


def wrap_label(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=14, break_long_words=False))


def radar_xy(values: list[float], angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values)
    return vals * np.cos(angles), vals * np.sin(angles)


def draw_metric_radar(
    ax: plt.Axes,
    results: pd.DataFrame,
    categories: list[str],
    models: list[str],
    metric: str,
    *,
    show_ci: bool,
) -> None:
    n = len(categories)
    # Start at the top and move clockwise, like common ggradar layouts.
    angles = np.pi / 2 - np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.append(angles, angles[0])

    ax.set_aspect("equal")
    ax.axis("off")

    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    for level in grid_levels:
        gx, gy = radar_xy([level] * (n + 1), closed_angles)
        ax.plot(gx, gy, color="#d0d0d0", linewidth=0.9, zorder=1)
        ax.text(
            0,
            level,
            f"{level:.1f}",
            color="#777777",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    for angle, category in zip(angles, categories):
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color="#e2e2e2", linewidth=0.8)
        label_radius = 1.16
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)
        ha = "center"
        if x < -0.2:
            ha = "right"
        elif x > 0.2:
            ha = "left"
        ax.text(x, y, wrap_label(category), ha=ha, va="center", fontsize=8)

    by_model = results.set_index(["model", "category"])
    for model in models:
        values = [float(by_model.loc[(model, category), metric]) for category in categories]
        closed_values = values + [values[0]]
        x, y = radar_xy(closed_values, closed_angles)
        color = MODEL_COLORS.get(model, None)

        if show_ci:
            ci_col = f"{metric}_CI"
            cis = [float(by_model.loc[(model, category), ci_col]) for category in categories]
            upper = [min(1.0, value + ci) for value, ci in zip(values, cis)]
            lower = [max(0.0, value - ci) for value, ci in zip(values, cis)]
            upper_x, upper_y = radar_xy(upper + [upper[0]], closed_angles)
            lower_x, lower_y = radar_xy(lower + [lower[0]], closed_angles)
            ax.fill(
                np.concatenate([upper_x, lower_x[::-1]]),
                np.concatenate([upper_y, lower_y[::-1]]),
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

        ax.plot(
            x,
            y,
            color=color,
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=MODEL_LABELS.get(model, model),
            zorder=3,
        )
        ax.fill(x, y, color=color, alpha=0.08, zorder=2)

    ax.set_xlim(-1.34, 1.34)
    ax.set_ylim(-1.24, 1.28)
    ax.set_title(f"{metric} per Category", fontweight="bold", pad=18)


def plot_ggradar_style(results_path: Path, output_path: Path, *, show_ci: bool) -> None:
    results = pd.read_csv(results_path)
    results = results[results["category"] != "OVERALL"].copy()
    categories = sorted(results["category"].unique())
    models = [model for model in MODEL_ORDER if model in set(results["model"])]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, metric in zip(axes, ["1-WER", "1-CER"]):
        draw_metric_radar(ax, results, categories, models, metric, show_ci=show_ci)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False)
    ci_note = " with 95% bootstrap CI bands" if show_ci else ""
    fig.suptitle(f"G2P Benchmark Radar - ggradar2-style{ci_note}", fontweight="bold", fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.15, wspace=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ggradar2-style benchmark radar chart.")
    parser.add_argument(
        "--results",
        default="data/benchmark_results_with_gemini_regular_rerun.csv",
        type=Path,
        help="Benchmark results CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/radar_chart_ggradar2_style_with_gemini_regular_rerun_ci.png",
        type=Path,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Disable translucent confidence interval bands.",
    )
    args = parser.parse_args()
    plot_ggradar_style(args.results, args.output, show_ci=not args.no_ci)


if __name__ == "__main__":
    main()
