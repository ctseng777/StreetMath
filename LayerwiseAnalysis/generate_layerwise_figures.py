#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: numpy. Install it before running this script.") from exc

# Ensure matplotlib can write config/cache in workspace
_CFG_DIR = Path(__file__).resolve().parent / ".mplconfig"
_CFG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CFG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS: List[Tuple[str, str, bool]] = [
    ("spectral_entropies", "Spectral Entropy", False),
    ("effective_ranks", "Effective Rank", False),
    ("trace_covariances", "Covariance Trace", True),
    ("gradient_norms", "Gradient Norm", True),
]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def collect_layerwise_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("*_layerwise_*.jsonl"))


def load_layerwise_summary(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "dream_model_complete_fixed_gsm8k_results.json",
        run_dir / "dream_model_complete_gsm8k_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def aggregate_from_summary(path: Path) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, int], Optional[int]]:
    try:
        obj = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"Failed to read summary JSON: {path}") from exc

    samples = obj.get("samples") or []
    values: Dict[str, List[List[float]]] = {m[0]: [] for m in METRICS}
    counts: Dict[str, int] = {m[0]: 0 for m in METRICS}
    expected_len: Optional[int] = obj.get("num_layers")

    for sample in samples:
        for key, _, _ in METRICS:
            arr = sample.get(key)
            if not isinstance(arr, list):
                continue
            if expected_len is None:
                expected_len = len(arr)
            if expected_len != len(arr):
                continue
            values[key].append([float(x) for x in arr])

    means: Dict[str, List[float]] = {}
    stds: Dict[str, List[float]] = {}
    for key, _, _ in METRICS:
        mat = values[key]
        counts[key] = len(mat)
        if not mat:
            continue
        arr = np.array(mat, dtype=float)
        means[key] = np.nanmean(arr, axis=0).tolist()
        if arr.shape[0] > 1:
            stds[key] = np.nanstd(arr, axis=0, ddof=1).tolist()
        else:
            stds[key] = [0.0 for _ in range(arr.shape[1])]

    return means, stds, counts, expected_len


def load_diffusion_steps(run_dir: Path) -> Optional[int]:
    jsonl_candidates = []
    for path in run_dir.glob("*.jsonl"):
        if "layerwise" in path.name:
            continue
        jsonl_candidates.append(path)
    if not jsonl_candidates:
        return None
    jsonl_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    path = jsonl_candidates[0]
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        if not first:
            return None
        obj = json.loads(first)
        steps = obj.get("diffusion_steps")
        if isinstance(steps, int):
            return steps
    except Exception:
        return None
    return None


def infer_steps_from_name(name: str) -> Optional[int]:
    match = re.search(r"_(\d+)_steps", name)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def aggregate_layerwise(files: List[Path]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, int], Optional[int]]:
    values: Dict[str, List[List[float]]] = {m[0]: [] for m in METRICS}
    counts: Dict[str, int] = {m[0]: 0 for m in METRICS}
    expected_len: Optional[int] = None

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                layerwise = obj.get("layerwise") or {}
                if not layerwise:
                    continue
                num_layers = layerwise.get("num_layers")
                if expected_len is None and isinstance(num_layers, int):
                    expected_len = num_layers
                for key, _, _ in METRICS:
                    arr = layerwise.get(key)
                    if not isinstance(arr, list):
                        continue
                    if expected_len is None:
                        expected_len = len(arr)
                    if expected_len != len(arr):
                        continue
                    values[key].append([float(x) for x in arr])

    means: Dict[str, List[float]] = {}
    stds: Dict[str, List[float]] = {}
    for key, _, _ in METRICS:
        mat = values[key]
        counts[key] = len(mat)
        if not mat:
            continue
        arr = np.array(mat, dtype=float)
        means[key] = np.nanmean(arr, axis=0).tolist()
        if arr.shape[0] > 1:
            stds[key] = np.nanstd(arr, axis=0, ddof=1).tolist()
        else:
            stds[key] = [0.0 for _ in range(arr.shape[1])]

    return means, stds, counts, expected_len


def plot_metric(ax, y: List[float], y_std: List[float], title: str, log_scale: bool, x_label: str):
    x = np.arange(len(y))
    ax.plot(x, y, linewidth=2)
    if y_std:
        std = np.array(y_std, dtype=float)
        lower = np.array(y, dtype=float) - std
        upper = np.array(y, dtype=float) + std
        ax.fill_between(x, lower, upper, alpha=0.2, label="±1 std")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(title)
    if log_scale:
        y_arr = np.array(y, dtype=float)
        if np.all(y_arr[y_arr > 0] > 0) and np.any(y_arr > 0):
            ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def write_averaged_json(out_path: Path, model: str, means: Dict[str, List[float]], stds: Dict[str, List[float]], counts: Dict[str, int], num_layers: Optional[int]):
    payload = {
        "model": model,
        "num_layers": num_layers,
        "metric_means": means,
        "metric_stddev": stds,
        "metric_sample_counts": counts,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_latex_block(model: str, fig_path: str, caption: str) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9]+", "", model)
    return (
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.95\\textwidth]{{{fig_path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:layerwise-{safe_label}}}\n"
        "\\end{figure*}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate layerwise metrics and plot 2x2 grids.")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory")
    parser.add_argument("--outdir", default="LayerwiseAnalysis", help="Output directory")
    parser.add_argument("--fig-subdir", default="figures", help="Subdir for figures")
    parser.add_argument("--avg-subdir", default="averages", help="Subdir for averaged JSON")
    parser.add_argument("--latex-out", default="appendix_d_figures.tex", help="Latex snippet filename")
    parser.add_argument("--latex-fig-dir", default=None, help="Path used in LaTeX includegraphics")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    outdir = Path(args.outdir)
    fig_dir = outdir / args.fig_subdir
    avg_dir = outdir / args.avg_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    avg_dir.mkdir(parents=True, exist_ok=True)

    latex_blocks: List[str] = []

    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("deprecate"):
            continue
        if name.endswith("small_run"):
            continue
        layerwise_files = collect_layerwise_files(entry)
        summary_path = load_layerwise_summary(entry)
        if not layerwise_files and summary_path is None:
            continue

        diffusion_steps = load_diffusion_steps(entry)
        if diffusion_steps is None:
            diffusion_steps = infer_steps_from_name(name)

        if layerwise_files:
            means, stds, counts, num_layers = aggregate_layerwise(layerwise_files)
        else:
            means, stds, counts, num_layers = aggregate_from_summary(summary_path)
        if not means:
            continue

        avg_path = avg_dir / f"{safe_filename(name)}_layerwise_averaged.json"
        write_averaged_json(avg_path, name, means, stds, counts, num_layers)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = np.array(axes).reshape(2, 2)
        x_label = "Layer"
        if diffusion_steps is not None and "dream" in name.lower():
            x_label = "Layer (denoising steps shown in caption)"
        for idx, (key, title, log_scale) in enumerate(METRICS):
            r, c = divmod(idx, 2)
            ax = axes[r, c]
            if key not in means:
                ax.set_axis_off()
                continue
            plot_metric(ax, means[key], stds.get(key, []), title, log_scale, x_label)

        title_suffix = ""
        if diffusion_steps is not None and "dream" in name.lower():
            title_suffix = f" (denoising steps={diffusion_steps})"
        fig.suptitle(f"Layerwise Metrics — {name}{title_suffix}", fontsize=14)
        plt.tight_layout()
        fig_name = f"{safe_filename(name)}_layerwise_metrics.png"
        fig_path = fig_dir / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        latex_fig_path = args.latex_fig_dir or str(fig_path.as_posix())
        caption = f"Layerwise averages for {name}."
        if diffusion_steps is not None and "dream" in name.lower():
            caption += f" Denoising steps = {diffusion_steps}."
            if num_layers is not None:
                caption += f" Metrics are plotted across {num_layers} model layers."
        caption += " Each subplot shows mean across samples with \"±1 std\" band."
        latex_blocks.append(build_latex_block(name, latex_fig_path, caption))

    latex_path = outdir / args.latex_out
    with latex_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated layerwise figure blocks for Appendix D\n")
        f.write("\n".join(latex_blocks))

    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote averages to: {avg_dir}")
    print(f"Wrote LaTeX to: {latex_path}")


if __name__ == "__main__":
    main()
