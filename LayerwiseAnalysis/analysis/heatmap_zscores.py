#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path as _Path
import os as _os

# Ensure matplotlib can write config/cache in workspace
_CFG_DIR = _Path(__file__).resolve().parent / ".mplconfig"
_CFG_DIR.mkdir(exist_ok=True)
_os.environ.setdefault("MPLCONFIGDIR", str(_CFG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS: List[Tuple[str, str]] = [
    ("spectral_entropies", "Spectral Entropy"),
    ("effective_ranks", "Effective Rank"),
    ("trace_covariances", "Covariance Trace"),
    ("gradient_norms", "Gradient Norm"),
]


def _load_averages(avg_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = {}
    for path in sorted(avg_dir.glob("*_layerwise_averaged.json")):
        if "gsm8k" in path.name.lower():
            continue
        obj = json.loads(path.read_text())
        means = obj.get("metric_means") or {}
        if not means:
            continue
        model = obj.get("model") or path.stem
        if "gsm8k" in model.lower():
            continue
        data[f"{model} (streetmath)"] = means
    return data


def _collect_layerwise_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("*_layerwise_*.jsonl"))


def _load_gsm8k_summary(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "dream_model_complete_fixed_gsm8k_results.json",
        run_dir / "dream_model_complete_gsm8k_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _aggregate_layerwise(files: List[Path]) -> Dict[str, List[float]]:
    values: Dict[str, List[List[float]]] = {m[0]: [] for m in METRICS}
    expected_len: int | None = None
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
                for key, _ in METRICS:
                    arr = layerwise.get(key)
                    if not isinstance(arr, list):
                        continue
                    if expected_len is None:
                        expected_len = len(arr)
                    if expected_len != len(arr):
                        continue
                    values[key].append([float(x) for x in arr])

    means: Dict[str, List[float]] = {}
    for key, _ in METRICS:
        mat = values[key]
        if not mat:
            continue
        arr = np.array(mat, dtype=float)
        means[key] = np.nanmean(arr, axis=0).tolist()
    return means


def _aggregate_from_summary(path: Path) -> Dict[str, List[float]]:
    obj = json.loads(path.read_text())
    samples = obj.get("samples") or []
    values: Dict[str, List[List[float]]] = {m[0]: [] for m in METRICS}
    expected_len: int | None = None

    for sample in samples:
        for key, _ in METRICS:
            arr = sample.get(key)
            if not isinstance(arr, list):
                continue
            if expected_len is None:
                expected_len = len(arr)
            if expected_len != len(arr):
                continue
            values[key].append([float(x) for x in arr])

    means: Dict[str, List[float]] = {}
    for key, _ in METRICS:
        mat = values[key]
        if not mat:
            continue
        arr = np.array(mat, dtype=float)
        means[key] = np.nanmean(arr, axis=0).tolist()
    return means


def _load_gsm8k_runs(gsm8k_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = {}
    if not gsm8k_dir.is_dir():
        return data
    for entry in sorted(gsm8k_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.endswith("small_run"):
            continue
        layerwise_files = _collect_layerwise_files(entry)
        if layerwise_files:
            means = _aggregate_layerwise(layerwise_files)
        else:
            summary_path = _load_gsm8k_summary(entry)
            if summary_path is None:
                continue
            means = _aggregate_from_summary(summary_path)
        if not means:
            continue
        data[f"{entry.name} (gsm8k)"] = means
    return data


def _canonical_name(model: str) -> str:
    dataset_tag = ""
    if "(gsm8k)" in model:
        dataset_tag = " (gsm8k)"
    elif "(streetmath)" in model:
        dataset_tag = " (streetmath)"

    name = model.replace("/", "-")
    name = name.replace("_", "-")
    name = name.lower()
    name = re.sub(r"-+", "-", name).strip("-")
    name = name.replace("-(gsm8k)", "").replace("-(streetmath)", "")
    name = name.replace(" (gsm8k)", "").replace(" (streetmath)", "")
    # strip instruct suffix since all models are instruct-tuned here
    if name.endswith("-instruct"):
        name = name[: -len("-instruct")]
    # Known aliases
    aliases = {
        "falcon-h1-8b": "falcon-h1-7b",
    }
    base = aliases.get(name, name)
    return f"{base}{dataset_tag}"


def _dedup_models(model_means: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    grouped: Dict[str, List[Dict[str, List[float]]]] = {}
    for model, means in model_means.items():
        grouped.setdefault(_canonical_name(model), []).append(means)

    merged: Dict[str, Dict[str, List[float]]] = {}
    for model, items in grouped.items():
        if len(items) == 1:
            merged[model] = items[0]
            continue
        combined: Dict[str, List[float]] = {}
        for key, _ in METRICS:
            series_list = [np.array(item.get(key, []), dtype=float) for item in items if item.get(key)]
            if not series_list:
                continue
            min_len = min(len(s) for s in series_list)
            stack = np.vstack([s[:min_len] for s in series_list])
            combined[key] = np.nanmean(stack, axis=0).tolist()
        if combined:
            merged[model] = combined
    return merged




def _resample(y: List[float], points: int = 100) -> np.ndarray:
    if len(y) == 1:
        return np.full(points, float(y[0]))
    x = np.linspace(0.0, 1.0, num=len(y))
    xi = np.linspace(0.0, 1.0, num=points)
    return np.interp(xi, x, np.array(y, dtype=float))


def plot_metric_heatmaps(avg_dir: str, out_dir: str, gsm8k_runs_dir: str, points: int = 100) -> None:
    avg_path = Path(avg_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_means = _load_averages(avg_path)
    gsm8k_means = _load_gsm8k_runs(Path(gsm8k_runs_dir))
    model_means.update(gsm8k_means)
    model_means = _dedup_models(model_means)
    if not model_means:
        raise SystemExit(f"No averaged layerwise files found in {avg_dir}")

    models = sorted(model_means.keys())

    for metric_key, title in METRICS:
        rows = []
        labels = []
        for model in models:
            series = model_means[model].get(metric_key)
            if not series:
                continue
            curve = _resample(series, points=points)
            mean = float(np.mean(curve))
            std = float(np.std(curve))
            if std == 0:
                z = np.zeros_like(curve)
            else:
                z = (curve - mean) / std
            rows.append(z)
            labels.append(model)
        if not rows:
            continue
        mat = np.vstack(rows)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for tick_label in ax.get_yticklabels():
            if "gsm8k" in tick_label.get_text().lower():
                tick_label.set_fontweight("bold")
        ax.set_xticks([0, points // 2, points - 1])
        ax.set_xticklabels(["0", "0.5", "1.0"])
        ax.set_xlabel("Depth (normalized)")
        ax.set_title(f"Z-Scored Layerwise Profiles — {title}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Z-score")

        fig.tight_layout()
        fig.savefig(out_path / f"heatmap_zscore_{metric_key}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    plot_metric_heatmaps(
        avg_dir="LayerwiseAnalysis/averages",
        out_dir="LayerwiseAnalysis/analysis",
        gsm8k_runs_dir="gsm8k-runs-layerwise",
        points=120,
    )


if __name__ == "__main__":
    main()
