"""
Run predict.py first to generate data/predictions.csv, then:

    cd categorized-heb-g2p-benchmark   # once, from repo root renikud/
    uv run src/benchmark.py
    uv run src/benchmark.py --quick    # faster bootstrap while iterating

Supports any prediction columns present (e.g. renikud, renikud_phonikud, phonikud).
"""

import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes

META_COLS = {"category", "sentence", "word_indices", "gt"}
MODEL_ORDER = [
    "gemini",
    "renikud",
    "renikud_phonikud",
    "phonikud",
    "renikud_ctc",
    "phonikud_byt5",
]
# Still scored in CSV; omitted from radar/bar charts (requested).
EXCLUDE_FROM_PLOTS = frozenset({"renikud_ctc", "phonikud_byt5"})
MODEL_LABELS = {
    "gemini": "Gemini",
    "renikud": "Renikud vox",
    "renikud_phonikud": "Renikud phonikud",
    "phonikud": "Phonikud (ONNX)",
    "renikud_ctc": "Renikud CTC",
    "phonikud_byt5": "Phonikud ByT5",
}
MODEL_COLORS = {
    "gemini": "black",
    "renikud": "steelblue",
    "renikud_phonikud": "seagreen",
    "phonikud": "tomato",
    "renikud_ctc": "mediumpurple",
    "phonikud_byt5": "darkorange",
}
# Radar lines only — keeps the two Renikud checkpoints visually separable when scores are close.
MODEL_LINESTYLES = {
    "gemini": "-",
    "renikud": "-",
    "renikud_phonikud": "--",
    "phonikud": "-.",
    "renikud_ctc": ":",
    "phonikud_byt5": "-",
}
MODEL_BAR_HATCHES = {
    "gemini": "",
    "renikud": "",
    "renikud_phonikud": "//",
    "phonikud": "",
    "renikud_ctc": "..",
    "phonikud_byt5": "xx",
}


def normalize_ipa_value(value: str, mode: str) -> str:
    """Normalize IPA notation before metric scoring."""
    text = str(value)
    if mode in {"stress", "initial_glottal_stress", "all_glottal_stress"}:
        text = text.replace("ˈ", "").replace("ˌ", "")
    if mode in {"initial_glottal", "initial_glottal_stress"}:
        text = re.sub(r"(?<!\S)ʔ", "", text)
    elif mode == "all_glottal_stress":
        text = text.replace("ʔ", "")
    return text


def prediction_model_columns(df: pd.DataFrame) -> list[str]:
    """Ordered list of model prediction columns present in ``df``."""
    present = set(df.columns)
    ordered = [c for c in MODEL_ORDER if c in present]
    extras = sorted(c for c in present if c not in META_COLS and c not in ordered)
    return ordered + extras


def normalize_ipa_columns(df: pd.DataFrame, models: list[str], mode: str) -> pd.DataFrame:
    """Return a copy with GT and model prediction columns normalized for scoring."""
    if mode == "raw":
        return df
    out = df.copy()
    for col in ["gt", *models]:
        out[col] = out[col].map(lambda value: normalize_ipa_value(value, mode))
    return out


def bootstrap_ci(ref: list, hyp: list, metric_fn, n_boot=1000, ci=0.95) -> float:
    """Return half-width of bootstrap CI for a metric."""
    rng = np.random.default_rng(42)
    n = len(ref)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = [ref[i] for i in idx]
        h = [hyp[i] for i in idx]
        scores.append(metric_fn(r, h))
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(scores, [alpha, 1 - alpha])
    return (hi - lo) / 2


def combine_homograph_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Treat ``homographs`` and ``Regular Homographed`` as one category in reports."""
    out = df.copy()
    out["category"] = out["category"].replace({"Regular Homographed": "homographs"})
    return out


def score_group_with_ci(df: pd.DataFrame, model: str, *, n_boot: int) -> dict:
    ref = df["gt"].tolist()
    hyp = df[model].tolist()
    wer_val = round(1 - wer(ref, hyp), 4)
    cer_val = round(1 - cer(ref, hyp), 4)
    wer_ci = round(bootstrap_ci(ref, hyp, wer, n_boot=n_boot), 4)
    cer_ci = round(bootstrap_ci(ref, hyp, cer, n_boot=n_boot), 4)
    return {
        "1-WER": wer_val, "1-WER_CI": wer_ci,
        "1-CER": cer_val, "1-CER_CI": cer_ci,
    }


def radar_factory(num_vars):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return super()._gen_axes_spines()

    register_projection(RadarAxes)
    return theta


def plot_radar(
    results: pd.DataFrame,
    categories: list,
    plot_models: list[str],
    *,
    output_path: str = "data/radar_chart.png",
    title_suffix: str = "",
):
    """Radar chart comparing WER and CER across categories (excluding OVERALL)."""
    cat_results = results[results["category"] != "OVERALL"]
    cats = [c for c in categories if c != "OVERALL"]
    num_vars = len(cats)

    theta = radar_factory(num_vars)

    fig, axes = plt.subplots(figsize=(12, 5), ncols=2,
                             subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(wspace=0.4)

    order_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    models = sorted(
        (m for m in cat_results["model"].unique() if m in plot_models),
        key=lambda m: order_rank.get(m, 99),
    )
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(models), 1)))

    for ax, metric in zip(axes, ["1-WER", "1-CER"]):
        if metric == "1-CER":
            ax.set_rgrids([0.6, 0.7, 0.8, 0.9], angle=0)
            ax.set_ylim(0.55, 1.0)
        else:
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0)
        ax.set_title(f"{metric}  (↑ higher is better)", weight="bold",
                     position=(0.5, 1.15), ha="center")
        for i, model in enumerate(models):
            vals = [
                cat_results.loc[
                    (cat_results["model"] == model) &
                    (cat_results["category"] == c), metric
                ].values[0]
                for c in cats
            ]
            color = MODEL_COLORS.get(model, cmap[i % len(cmap)])
            lbl = MODEL_LABELS.get(model, model)
            ls = MODEL_LINESTYLES.get(model, "-")
            ax.plot(theta, vals, color=color, linestyle=ls, linewidth=2.0, label=lbl)
            ax.fill(theta, vals, facecolor=color, alpha=0.12)
        ax.set_varlabels(cats)

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), fontsize="small")
    fig.suptitle(f"Radar Chart — 1-WER & 1-CER per Category{title_suffix}", weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score predictions.csv and write plots.")
    parser.add_argument(
        "--predictions",
        default="data/predictions.csv",
        help="CSV from predict.py (default: data/predictions.csv)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        metavar="N",
        help="Bootstrap resamples per CI bar (default: 1000, or 200 with --quick)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Bootstrap N=200 for much faster runs (less precise error bars).",
    )
    parser.add_argument(
        "--plot-suffix",
        default="",
        help=(
            "Suffix added before .png for plot outputs, e.g. "
            "'_with_gemini' writes data/1_wer_per_category_with_gemini.png."
        ),
    )
    parser.add_argument(
        "--results-output",
        default="data/benchmark_results.csv",
        help="Path for benchmark CSV output (default: data/benchmark_results.csv).",
    )
    parser.add_argument(
        "--normalize-ipa",
        choices=[
            "raw",
            "stress",
            "initial_glottal",
            "initial_glottal_stress",
            "all_glottal_stress",
        ],
        default="raw",
        help=(
            "Normalize IPA before scoring: raw, stress, initial_glottal, "
            "initial_glottal_stress, or all_glottal_stress."
        ),
    )
    args = parser.parse_args()
    if args.bootstrap is not None:
        n_boot = args.bootstrap
    elif args.quick:
        n_boot = 200
    else:
        n_boot = 1000

    print(f"Bootstrap resamples (per metric, per category): {n_boot}")

    pred_path = args.predictions
    df = combine_homograph_categories(pd.read_csv(pred_path).fillna(""))

    models = prediction_model_columns(df)
    if not models:
        raise SystemExit(
            "No model columns in data/predictions.csv "
            f"(expected some of {MODEL_ORDER})."
        )
    df = normalize_ipa_columns(df, models, args.normalize_ipa)
    if args.normalize_ipa != "raw":
        print(f"IPA normalization: {args.normalize_ipa}")
    plot_models = [m for m in models if m not in EXCLUDE_FROM_PLOTS]
    omitted = [m for m in models if m in EXCLUDE_FROM_PLOTS]
    if omitted:
        print(f"Charts exclude (still in CSV): {', '.join(omitted)}")
    categories = np.sort(df["category"].unique())

    records = []
    for model in models:
        for cat in categories:
            subset = df[df["category"] == cat]
            scores = score_group_with_ci(subset, model, n_boot=n_boot)
            records.append({"model": model, "category": cat, **scores})
        overall = score_group_with_ci(df, model, n_boot=n_boot)
        records.append({"model": model, "category": "OVERALL", **overall})

    results = pd.DataFrame(records)
    print(results[["model", "category", "1-WER", "1-CER"]].to_string(index=False))
    results.to_csv(args.results_output, index=False)
    print(f"Saved {args.results_output}")

    # Bar plots with confidence interval error bars (subset of models)
    n_plot = len(plot_models)
    bar_width = min(0.35, 0.8 / max(n_plot, 1))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(n_plot, 1)))
    all_cats = list(categories) + ["OVERALL"]

    for metric in ["1-WER", "1-CER"]:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(all_cats))

        for i, model in enumerate(plot_models):
            subset = results[results["model"] == model].set_index("category")
            vals = [subset.loc[c, metric] for c in all_cats]
            cis  = [subset.loc[c, f"{metric}_CI"] for c in all_cats]
            offset = (i - (n_plot - 1) / 2) * bar_width
            color = MODEL_COLORS.get(model, cmap[i % len(cmap)])
            hatch = MODEL_BAR_HATCHES.get(model, "")
            ax.bar(x + offset, vals, width=bar_width,
                   color=color, edgecolor="black", hatch=hatch,
                   yerr=cis, capsize=6,
                   label=MODEL_LABELS.get(model, model), alpha=0.85)

        ax.set_title(
            f"{metric} per Category  (↑ higher is better, error bars ≈95% bootstrap CI, N={n_boot})"
            + (f"\nIPA normalization: {args.normalize_ipa}" if args.normalize_ipa != "raw" else "")
        )
        ax.set_ylabel(metric)
        ax.set_xlabel("Category")
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        fname = metric.lower().replace("-", "_")
        output_path = f"data/{fname}_per_category{args.plot_suffix}.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved {output_path}")

    plot_radar(
        results,
        all_cats,
        plot_models,
        output_path=f"data/radar_chart{args.plot_suffix}.png",
        title_suffix=(
            f"\nIPA normalization: {args.normalize_ipa}"
            if args.normalize_ipa != "raw"
            else ""
        ),
    )


if __name__ == "__main__":
    main()
