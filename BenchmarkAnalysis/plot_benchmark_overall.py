#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_benchmark_tables import (
    JUDGEMENT_ORDER,
    aggregate_run,
    pick_main_jsonl,
)

MODEL_PARAM_B = {
    "qwen2_5_3b": 3,
    "qwen2_5_32b": 32,
    "falcon-h1-7b": 7,
    "falcon-h1-34b": 34,
    "dream-v0-instruct-7b": 7,
}

# A100-XSM (400W) peak BF16 throughput ~312 TFLOPs.
FLOPS_PER_KWH = (312e12 * 3600) / 0.4


def estimate_steps(model_name: str) -> int:
    name = model_name.lower()
    if "dream" not in name:
        return 1
    if "_4_steps" in name:
        return 4
    if "_8_steps" in name:
        return 8
    return 16


def estimate_params_b(model_name: str) -> Optional[int]:
    name = model_name.lower()
    for key, value in MODEL_PARAM_B.items():
        if key in name:
            return value
    return None


def estimate_energy_kwh(model_name: str, avg_total_tokens: float) -> float:
    params_b = estimate_params_b(model_name)
    if params_b is None or avg_total_tokens <= 0:
        return 0.0
    steps = estimate_steps(model_name)
    flops = 2 * (params_b * 1e9) * avg_total_tokens * steps
    return flops / FLOPS_PER_KWH


def collect_model_rows(runs_dir: str) -> List[Dict[str, object]]:
    model_rows = []
    for entry in sorted(os.listdir(runs_dir)):
        if entry.startswith("deprecate"):
            continue
        if entry.endswith("small_run"):
            continue
        run_path = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_path):
            continue
        jsonl_path = pick_main_jsonl(run_path)
        if not jsonl_path:
            continue
        stats = aggregate_run(jsonl_path)
        stats["model"] = entry
        model_rows.append(stats)
    return model_rows


def plot_overall(model_rows: List[Dict[str, object]], out_path: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#444444",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )
    models = [row["model"] for row in model_rows]
    counts_by_judgement = {j: [] for j in JUDGEMENT_ORDER}
    avg_tokens = []
    energy_kwh = []

    for row in model_rows:
        counts = row["judgement_counts"]
        for judgement in JUDGEMENT_ORDER:
            counts_by_judgement[judgement].append(counts.get(judgement, 0))
        avg_total_tokens = row.get("avg_tokens") or 0.0
        avg_tokens.append(avg_total_tokens)
        total_tokens = avg_total_tokens * float(row.get("samples") or 0)
        energy_kwh.append(estimate_energy_kwh(row["model"], total_tokens))

    x = list(range(len(models)))
    fig, (ax, ax_energy) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        figsize=(max(9, len(models) * 0.9), 7.0),
    )

    bottoms = [0] * len(models)
    # One faded color per model; stacks are differentiated by hatch patterns.
    model_palette = [
        "#b7cfe3",
        "#e1c6a3",
        "#e1b1b1",
        "#c2d7c9",
        "#d4c6e1",
        "#e3d9b6",
        "#c6d2e9",
        "#e3c6d7",
    ]
    hatches = {
        "Good approximation": "",
        "Exact math": "\\\\\\",
        "Mildly off": "xx",
        "Way off": "...",
        "Uncategorized": "++",
    }

    # Bottom-up stack order
    stack_order = [
        "Good approximation",
        "Uncategorized",
        "Exact math",
        "Mildly off",
        "Way off",
    ]
    for judgement in stack_order:
        values = counts_by_judgement[judgement]
        bar_colors = [model_palette[i % len(model_palette)] for i in range(len(models))]
        bars = ax.bar(
            x,
            values,
            bottom=bottoms,
            label=judgement,
            color=bar_colors,
            edgecolor="#f0efeb",
            linewidth=0.6,
            width=0.72,
            hatch=hatches.get(judgement, ""),
            alpha=0.85,
        )
        # add numbers in each stack
        for idx, (bar, val) in enumerate(zip(bars, values)):
            if val == 0:
                continue
            y = bottoms[idx] + val / 2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                str(val),
                ha="center",
                va="center",
                fontsize=8,
                color="#000000",
            )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(x)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Count")
    ax.set_title("StreetMath Benchmark Judgement Counts by Model", pad=28)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    energy_wh = [val * 1000 for val in energy_kwh]
    bars_energy = ax_energy.bar(
        x,
        energy_wh,
        color="#1f2d3d",
        width=0.52,
        alpha=0.8,
    )
    ax_energy.set_ylabel("Energy consumption (Wh)")
    ax_energy.set_yscale("log")
    positive_energy = [val for val in energy_wh if val > 0]
    if positive_energy:
        ax_energy.set_ylim(min(positive_energy) / 1.5, max(positive_energy) * 1.2)
    ax_energy.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter())
    ax_energy.grid(axis="y")
    ax_energy.spines["top"].set_visible(False)
    ax_energy.spines["right"].set_visible(False)
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(models, rotation=30, ha="right")
    for bar, val in zip(bars_energy, energy_wh):
        if val <= 0:
            continue
        if val < 1:
            label = f"{val:.3f}"
        elif val < 10:
            label = f"{val:.2f}"
        else:
            label = f"{val:.1f}"
        ax_energy.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.08,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )

    # combine legends
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.22)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark stacked bars with avg tokens line.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subfolders.")
    parser.add_argument("--outdir", default="BenchmarkAnalysis/plots", help="Output directory for plots.")
    parser.add_argument("--outfile", default="benchmark_overall.png", help="Output plot filename.")
    args = parser.parse_args()

    if not os.path.isdir(args.runs_dir):
        raise SystemExit(f"Runs dir not found: {args.runs_dir}")

    model_rows = collect_model_rows(args.runs_dir)
    if not model_rows:
        raise SystemExit("No qualifying runs found.")

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.outfile)
    plot_overall(model_rows, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
