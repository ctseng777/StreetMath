#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILE_RE = re.compile(
    r"^(MMLU|STREET_MATH|gsm8k_cot|gsm8k)_calculate([0-9.]+)_run(\d+)(?:_train_task)?\.(json|jsonl)$",
    re.IGNORECASE,
)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def mmlu_accuracy(path: Path) -> float:
    obj = json.loads(path.read_text())
    accs = [v.get("acc,none") for v in obj.values() if isinstance(v, dict) and "acc,none" in v]
    return mean(accs) if accs else float("nan")


def streetmath_accuracy(path: Path) -> float:
    good = 0
    total = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get("judgement") or obj.get("label")
            if label is None:
                continue
            total += 1
            if label == "Good approximation":
                good += 1
    return good / total if total else float("nan")


def gsm8k_accuracy(path: Path) -> float:
    obj = json.loads(path.read_text())
    accs = [
        v.get("exact_match,strict-match")
        for v in obj.values()
        if isinstance(v, dict) and "exact_match,strict-match" in v
    ]
    return mean(accs) if accs else float("nan")


def collect_model_records(model_dir: Path) -> Dict[Tuple[float, int], Dict[str, float]]:
    records: Dict[Tuple[float, int], Dict[str, float]] = {}
    for file in model_dir.rglob("*"):
        if not file.is_file():
            continue
        m = FILE_RE.match(file.name)
        if not m:
            continue
        dataset, prop_str, run_str, _ = m.groups()
        dataset = dataset.lower()
        if dataset in {"gsm8k", "gsm8k_cot"}:
            ds = "gsm8k"
        elif dataset == "street_math":
            ds = "street_math"
        else:
            ds = "mmlu"
        key = (float(prop_str), int(run_str))
        if ds == "mmlu":
            acc = mmlu_accuracy(file)
        elif ds == "street_math":
            acc = streetmath_accuracy(file)
        else:
            acc = gsm8k_accuracy(file)
        records.setdefault(key, {})[ds] = acc
    return records


def compute_model_correlations(
    model_dir: Path,
) -> Tuple[float, float, float, float, float, float, int]:
    records = collect_model_records(model_dir)
    by_run: Dict[int, Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: {"sm_gsm": [], "sm_mmlu": []}
    )
    for (prop, run_id), vals in records.items():
        if "street_math" in vals and "gsm8k" in vals:
            by_run[run_id]["sm_gsm"].append((vals["street_math"], vals["gsm8k"]))
        if "street_math" in vals and "mmlu" in vals:
            by_run[run_id]["sm_mmlu"].append((vals["street_math"], vals["mmlu"]))

    run_corrs = []
    for run_id, pairs in sorted(by_run.items()):
        sm_gsm = pairs["sm_gsm"]
        sm_mmlu = pairs["sm_mmlu"]
        x1 = [a for a, _ in sm_gsm]
        y1 = [b for _, b in sm_gsm]
        x2 = [a for a, _ in sm_mmlu]
        y2 = [b for _, b in sm_mmlu]
        r1 = pearson(x1, y1)
        r2 = pearson(x2, y2)
        if not math.isnan(r1) and not math.isnan(r2):
            run_corrs.append((r1, r2))

    if not run_corrs:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), len(records)

    xs = [r[0] for r in run_corrs]
    ys = [r[1] for r in run_corrs]
    mean_x = mean(xs)
    mean_y = mean(ys)
    range_x = (min(xs), max(xs))
    range_y = (min(ys), max(ys))
    return mean_x, mean_y, range_x[0], range_x[1], range_y[0], range_y[1], len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-model Pearson correlations for pruning runs.")
    parser.add_argument("--runs-dir", default="Prune_study_runs", help="Input runs directory.")
    parser.add_argument("--outdir", default="PruneStudyAnalysis/plots", help="Output plots directory.")
    parser.add_argument("--outfile", default="prune_model_correlations.png", help="Output plot filename.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_points = []
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        if name.endswith("small_1") or "samples" in name.lower():
            continue
        res = compute_model_correlations(model_dir)
        if math.isnan(res[0]) or math.isnan(res[1]):
            continue
        mean_x, mean_y, min_x, max_x, min_y, max_y, n = res
        model_points.append((name, mean_x, mean_y, min_x, max_x, min_y, max_y, n))

    if not model_points:
        raise SystemExit("No model correlations computed.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#444444",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.grid(True, zorder=0)

    # Use unique colors per model and rely on legend instead of text labels.
    cmap = plt.cm.get_cmap("tab20", len(model_points))
    model_colors = {name: cmap(i) for i, (name, *_rest) in enumerate(model_points)}

    # Group identical points to slightly offset them and avoid marker overlap.
    grouped: Dict[Tuple[float, float], List[str]] = defaultdict(list)
    stats_by_name = {}
    for name, mean_x, mean_y, min_x, max_x, min_y, max_y, _ in model_points:
        grouped[(round(mean_x, 6), round(mean_y, 6))].append(name)
        stats_by_name[name] = (mean_x, mean_y, min_x, max_x, min_y, max_y)

    for key, names in grouped.items():
        mean_x, mean_y = key
        offsets = [(i - (len(names) - 1) / 2) * 0.02 for i in range(len(names))]
        for name, dx in zip(names, offsets):
            mean_x, mean_y, min_x, max_x, min_y, max_y = stats_by_name[name]
            xerr = [[mean_x - min_x], [max_x - mean_x]]
            yerr = [[mean_y - min_y], [max_y - mean_y]]
            color = model_colors[name]
            ax.errorbar(
                [mean_x + dx],
                [mean_y],
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                markersize=6,
                markeredgecolor="#ffffff",
                markeredgewidth=0.7,
                zorder=3,
                label=name,
                alpha=0.95,
            )

    ax.set_xlabel("Pearson r: StreetMath vs GSM8K")
    ax.set_ylabel("Pearson r: StreetMath vs MMLU")
    ax.set_title("Per-model Correlation Under Pruning Runs")
    min_x = min(p[3] for p in model_points)
    max_x = max(p[4] for p in model_points)
    pad_x = 0.05
    ax.set_xlim(max(0.0, min_x - pad_x), min(1.0, max_x + pad_x))
    ax.set_ylim(0.0, 1.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    out_path = outdir / args.outfile
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
