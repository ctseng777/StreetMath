#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List

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

METRICS = [
    ("spectral_entropies", "Spectral Entropy"),
    ("effective_ranks", "Effective Rank"),
    ("trace_covariances", "Covariance Trace"),
    ("gradient_norms", "Gradient Norm"),
]


def _load_averages(avg_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = {}
    for path in sorted(avg_dir.glob("*_layerwise_averaged.json")):
        obj = json.loads(path.read_text())
        means = obj.get("metric_means") or {}
        if not means:
            continue
        model = obj.get("model") or path.stem
        data[model] = means
    return data


def _resample(y: List[float], points: int = 100) -> np.ndarray:
    if len(y) == 1:
        return np.full(points, float(y[0]))
    x = np.linspace(0.0, 1.0, num=len(y))
    xi = np.linspace(0.0, 1.0, num=points)
    return np.interp(xi, x, np.array(y, dtype=float))


def _spread_labels(y_values: np.ndarray, min_gap: float) -> np.ndarray:
    order = np.argsort(y_values)
    y_sorted = y_values[order].copy()
    for i in range(1, len(y_sorted)):
        if y_sorted[i] - y_sorted[i - 1] < min_gap:
            y_sorted[i] = y_sorted[i - 1] + min_gap
    adjusted = np.empty_like(y_sorted)
    adjusted[order] = y_sorted
    return adjusted


def plot_normalized_overlays(avg_dir: str, out_dir: str, points: int = 100) -> None:
    avg_path = Path(avg_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_means = _load_averages(avg_path)
    if not model_means:
        raise SystemExit(f"No averaged layerwise files found in {avg_dir}")

    for metric_key, title in METRICS:
        curves = []
        labels = []
        for model, means in model_means.items():
            series = means.get(metric_key)
            if not series:
                continue
            curves.append(_resample(series, points=points))
            labels.append(model)
        if not curves:
            continue

        mat = np.vstack(curves)
        mean_curve = mat.mean(axis=0)

        fig, ax = plt.subplots(figsize=(10.5, 6))
        for curve, label in zip(curves, labels):
            ax.plot(
                np.linspace(0.0, 1.0, num=points),
                curve,
                linewidth=1.0,
                alpha=0.25,
            )
        ax.plot(
            np.linspace(0.0, 1.0, num=points),
            mean_curve,
            color="#1f2d3d",
            linewidth=2.4,
            label="Mean",
        )
        ax.set_title(f"Normalized Layer Profiles — {title}")
        ax.set_xlabel("Depth (normalized)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

        # Inline labels near the right edge, spaced to reduce overlap.
        x_positions = np.linspace(0.0, 1.0, num=points)
        end_x = 0.3
        idx = int(round(end_x * (points - 1)))
        end_vals = np.array([curve[idx] for curve in curves], dtype=float)
        y_span = float(np.max(end_vals) - np.min(end_vals)) or 1.0
        min_gap = y_span * 0.03
        placed = _spread_labels(end_vals, min_gap=min_gap)
        for curve, label, y_text in zip(curves, labels, placed):
            ax.text(
                end_x + 0.01,
                y_text,
                label,
                fontsize=7,
                va="center",
                ha="left",
                color="#333333",
                clip_on=False,
            )
        ax.set_xlim(0.0, 1.0)

        fig.tight_layout()
        fig.savefig(out_path / f"normalized_overlays_{metric_key}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    plot_normalized_overlays(
        avg_dir="LayerwiseAnalysis/averages",
        out_dir="LayerwiseAnalysis/analysis",
        points=120,
    )


if __name__ == "__main__":
    main()
