#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

METRICS: List[Tuple[str, str]] = [
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


def _binned_means(series: List[float]) -> Tuple[float, float, float]:
    y = np.array(series, dtype=float)
    n = len(y)
    if n < 3:
        return float(y.mean()), float(y.mean()), float(y.mean())
    a = y[: n // 3]
    b = y[n // 3 : 2 * n // 3]
    c = y[2 * n // 3 :]
    return float(a.mean()), float(b.mean()), float(c.mean())


def write_binned_summary(avg_dir: str, out_dir: str) -> None:
    avg_path = Path(avg_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_means = _load_averages(avg_path)
    if not model_means:
        raise SystemExit(f"No averaged layerwise files found in {avg_dir}")

    rows: List[Dict[str, object]] = []
    for model, means in model_means.items():
        for metric_key, _ in METRICS:
            series = means.get(metric_key)
            if not series:
                continue
            early, mid, late = _binned_means(series)
            rows.append(
                {
                    "model": model,
                    "metric": metric_key,
                    "early_mean": early,
                    "mid_mean": mid,
                    "late_mean": late,
                }
            )

    csv_path = out_path / "binned_layer_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "metric", "early_mean", "mid_mean", "late_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_binned_summary(
        avg_dir="LayerwiseAnalysis/averages",
        out_dir="LayerwiseAnalysis/analysis",
    )


if __name__ == "__main__":
    main()
