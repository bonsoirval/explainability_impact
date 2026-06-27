"""
plot_from_csv.py
----------------
Standalone script to regenerate benchmark distribution plots from a
previously saved raw-runs CSV.  No model, no GPU required.

Usage examples:
    # Plot both models, auto-detecting CSVs in the default results dir:
    python plot_from_csv.py

    # Plot a specific CSV with a custom output path:
    python plot_from_csv.py \
        --csv phase_5_evaluation/results/mobilenet_v3_raw_runs.csv \
        --out my_plots/comparison.png

    # Compare two CSVs (e.g. mobilenet vs shufflenet) in one figure:
    python plot_from_csv.py \
        --csv phase_5_evaluation/results/mobilenet_v3_raw_runs.csv \
             phase_5_evaluation/results/shufflenet_v2_raw_runs.csv \
        --out phase_5_evaluation/results/combined_comparison.png
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── Colour palette ───────────────────────────────────────────────────────────
METHOD_COLORS = {
    "Saliency": "#3A86FF",
    "Grad-CAM": "#FF006E",
    "LIME":     "#8338EC",
    "SHAP":     "#FFBE0B",
}
FALLBACK_COLORS = ["#3A86FF", "#FF006E", "#8338EC", "#FFBE0B",
                   "#06D6A0", "#FB5607", "#023E8A", "#E63946"]


def _color_for(method: str, idx: int) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ── Single-CSV plot (matches existing pipeline output style) ─────────────────
def plot_single(raw_df: pd.DataFrame, label: str, save_path: str) -> None:
    """Box-plot + jitter for one raw-runs DataFrame."""
    methods = list(raw_df["Method"].unique())
    has_vram = raw_df["Peak VRAM (MB)"].sum() > 0
    metrics = ["Latency (s)", "Peak RAM (MB)"] + (["Peak VRAM (MB)"] if has_vram else [])
    titles  = ["Explainability Generation Latency",
               "Peak RAM Allocation",
               "Peak VRAM Allocation"]
    ylabels = ["Latency (seconds) — Log Scale",
               "Peak RAM Usage (MB)",
               "Peak VRAM Usage (MB)"]

    n_runs = len(raw_df) // len(methods)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5.5))
    if len(metrics) == 1:
        axes = [axes]

    box_kw = dict(
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        data   = [raw_df[raw_df["Method"] == m][metric].values for m in methods]
        colors = [_color_for(m, i) for i, m in enumerate(methods)]

        bp = ax.boxplot(data, tick_labels=methods, **box_kw)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)

        for i, (m, color) in enumerate(zip(methods, colors)):
            y = raw_df[raw_df["Method"] == m][metric].values
            x = np.random.default_rng(42).normal(i + 1, 0.06, size=len(y))
            ax.scatter(x, y, color=color, alpha=0.65, s=20,
                       edgecolors="white", linewidths=0.4, zorder=3)

        if metric == "Latency (s)":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        f"{label}  —  Computational Performance Distribution  ({n_runs} runs)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Multi-CSV combined comparison plot ───────────────────────────────────────
def plot_combined(csv_paths: list[str], save_path: str) -> None:
    """
    Side-by-side comparison across multiple CSVs (e.g. mobilenet vs shufflenet).
    Each CSV becomes a column-group; each metric gets its own row.
    """
    frames   = {p: pd.read_csv(p) for p in csv_paths}
    labels   = [os.path.basename(p).replace("_raw_runs.csv", "").replace("_", " ").title()
                for p in csv_paths]
    all_methods = []
    for df in frames.values():
        for m in df["Method"].unique():
            if m not in all_methods:
                all_methods.append(m)

    has_vram = any(df["Peak VRAM (MB)"].sum() > 0 for df in frames.values())
    metrics  = ["Latency (s)", "Peak RAM (MB)"] + (["Peak VRAM (MB)"] if has_vram else [])
    ylabels  = ["Latency (s) — Log Scale", "Peak RAM (MB)", "Peak VRAM (MB)"]

    n_rows, n_cols = len(metrics), len(csv_paths)
    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    box_kw = dict(
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col, (path, label) in enumerate(zip(csv_paths, labels)):
            df      = frames[path]
            methods = [m for m in all_methods if m in df["Method"].unique()]
            colors  = [_color_for(m, i) for i, m in enumerate(methods)]
            data    = [df[df["Method"] == m][metric].values for m in methods]
            n_runs  = len(df) // len(methods)

            ax = fig.add_subplot(gs[row, col])
            bp = ax.boxplot(data, tick_labels=methods, **box_kw)

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)

            for i, (m, color) in enumerate(zip(methods, colors)):
                y = df[df["Method"] == m][metric].values
                x = np.random.default_rng(42).normal(i + 1, 0.06, size=len(y))
                ax.scatter(x, y, color=color, alpha=0.65, s=15,
                           edgecolors="white", linewidths=0.3, zorder=3)

            if metric == "Latency (s)":
                ax.set_yscale("log")

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(f"{label}\n({n_runs} runs)", fontsize=11,
                             fontweight="bold", pad=8)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#cccccc")
            ax.tick_params(axis="both", labelsize=9)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_color_for(m, i), markersize=9, label=m)
        for i, m in enumerate(all_methods)
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(all_methods), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("XAI Method Computational Benchmarking — Model Comparison",
                 fontsize=15, fontweight="bold", y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-generate benchmark distribution plots from saved raw-run CSVs."
    )
    parser.add_argument(
        "--csv", nargs="+", metavar="CSV",
        help=(
            "Path(s) to *_raw_runs.csv file(s).  "
            "Omit to auto-detect all CSVs in --results_dir."
        ),
    )
    parser.add_argument(
        "--results_dir", default="phase_5_evaluation/results",
        help="Directory to scan for *_raw_runs.csv files (default: phase_5_evaluation/results).",
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output PNG path.  "
            "For a single CSV, defaults to <csv_stem>_replot.png next to the CSV.  "
            "For multiple CSVs, defaults to combined_comparison.png in --results_dir."
        ),
    )
    args = parser.parse_args()

    # Resolve CSV list
    if args.csv:
        csv_list = args.csv
    else:
        csv_list = sorted(glob.glob(os.path.join(args.results_dir, "*_raw_runs.csv")))
        if not csv_list:
            print(f"No *_raw_runs.csv files found in '{args.results_dir}'.")
            print("Run the pipeline first, or supply --csv explicitly.")
            return

    print(f"Found {len(csv_list)} CSV(s):")
    for p in csv_list:
        print(f"  {p}")

    if len(csv_list) == 1:
        path  = csv_list[0]
        label = os.path.basename(path).replace("_raw_runs.csv", "").replace("_", " ").title()
        out   = args.out or os.path.join(
            os.path.dirname(path),
            os.path.basename(path).replace("_raw_runs.csv", "_replot.png"),
        )
        df = pd.read_csv(path)
        print(f"\nPlotting single-model distribution for '{label}'  ({len(df)} rows)...")
        plot_single(df, label, out)
    else:
        out = args.out or os.path.join(args.results_dir, "combined_comparison.png")
        print(f"\nPlotting multi-model comparison for {len(csv_list)} model(s)...")
        plot_combined(csv_list, out)

    print("Done.")


if __name__ == "__main__":
    main()
