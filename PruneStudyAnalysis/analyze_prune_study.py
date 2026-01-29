#!/usr/bin/env python3
"""Analyze pruning study runs and plot accuracy vs proportion."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "Prune_study_runs"
OUT_DIR = Path(__file__).resolve().parent / "plots"
SUMMARY_DIR = Path(__file__).resolve().parent / "summaries"

SKIP_SUBSTRINGS = {"small", "samples"}

# 95% CI for mean with n=3 uses t_{0.975, df=2}
T_CRIT_DF2_95 = 4.302652729911275


@dataclass
class RunRecord:
    proportion: float
    run_id: int
    accuracy: float


def iter_model_dirs(base_dir: Path) -> List[Path]:
    dirs = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        name_lower = p.name.lower()
        if any(s in name_lower for s in SKIP_SUBSTRINGS):
            continue
        dirs.append(p)
    return dirs


FILE_RE = re.compile(
    r"^(MMLU|STREET_MATH|gsm8k_cot|gsm8k)_calculate([0-9.]+)_run(\d+)(?:_train_task)?\.(json|jsonl)$",
    re.IGNORECASE,
)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def sample_std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def ci_95_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    s = sample_std(values)
    return T_CRIT_DF2_95 * (s / math.sqrt(len(values)))


def mmlu_accuracy(path: Path) -> float:
    obj = json.loads(path.read_text())
    accs = []
    for val in obj.values():
        if isinstance(val, dict) and "acc,none" in val:
            accs.append(val["acc,none"])
    if not accs:
        raise ValueError(f"No MMLU acc,none values found in {path}")
    return mean(accs)


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
    if total == 0:
        raise ValueError(f"No judgements found in {path}")
    return good / total


def collect_records(model_dir: Path) -> Dict[str, List[RunRecord]]:
    records: Dict[str, List[RunRecord]] = {
        "MMLU": [],
        "STREET_MATH": [],
        "GSM8K": [],
    }
    for file in model_dir.rglob("*"):
        if not file.is_file():
            continue
        m = FILE_RE.match(file.name)
        if not m:
            continue
        dataset, prop_str, run_str, ext = m.groups()
        proportion = float(prop_str)
        run_id = int(run_str)
        dataset_norm = dataset.upper() if dataset.lower() != "gsm8k_cot" and dataset.lower() != "gsm8k" else "GSM8K"
        if dataset_norm == "MMLU":
            acc = mmlu_accuracy(file)
        elif dataset_norm == "STREET_MATH":
            acc = streetmath_accuracy(file)
        else:
            acc = gsm8k_accuracy(file)
        records[dataset_norm].append(RunRecord(proportion=proportion, run_id=run_id, accuracy=acc))
    return records


def gsm8k_accuracy(path: Path) -> float:
    obj = json.loads(path.read_text())
    accs = []
    for val in obj.values():
        if isinstance(val, dict) and "exact_match,strict-match" in val:
            accs.append(val["exact_match,strict-match"])
    if not accs:
        raise ValueError(f"No GSM8K exact_match,strict-match values found in {path}")
    return mean(accs)


def summarize(records: List[RunRecord]) -> Dict[float, Tuple[float, float]]:
    by_prop: Dict[float, List[float]] = {}
    for r in records:
        by_prop.setdefault(r.proportion, []).append(r.accuracy)

    summary: Dict[float, Tuple[float, float]] = {}
    for prop, vals in sorted(by_prop.items(), key=lambda x: x[0]):
        summary[prop] = (mean(vals), ci_95_mean(vals))
    return summary


def plot_model(model_dir: Path) -> Optional[Path]:
    summary_path = SUMMARY_DIR / f"{model_dir.name}.json"
    if summary_path.exists():
        summaries = load_summary(summary_path)
    else:
        records = collect_records(model_dir)
        if not records["MMLU"] and not records["STREET_MATH"] and not records["GSM8K"]:
            return None
        summaries = {k: summarize(v) for k, v in records.items()}
        missing = [k for k, v in records.items() if not v]
        if missing:
            print(f"Warning: {model_dir.name} missing datasets: {', '.join(missing)}")
        write_summary(summary_path, summaries)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for dataset, color, marker in (
        ("MMLU", "#1f77b4", "o"),
        ("STREET_MATH", "#d62728", "s"),
        ("GSM8K", "#2ca02c", "^"),
    ):
        summary = summaries.get(dataset, {})
        if not summary:
            continue
        x = list(summary.keys())
        y = [summary[p][0] for p in x]
        yerr = [summary[p][1] for p in x]
        # xerr is zero if proportions are fixed across runs.
        xerr = [0.0 for _ in x]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            xerr=xerr,
            marker=marker,
            color=color,
            linestyle="-",
            linewidth=1.6,
            markersize=5.5,
            capsize=3,
            label=dataset.replace("_", " "),
        )

    all_props = [p for ds in summaries.values() for p in ds.keys()]
    if any(p <= 0 for p in all_props):
        ax.set_xscale("symlog", linthresh=1e-4)
    else:
        ax.set_xscale("log")
    ax.set_xlabel("Proportion pruned")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(model_dir.name)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=9)

    out_path = OUT_DIR / f"{model_dir.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def normalize_summary_dict(raw: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[float, Tuple[float, float]]]:
    normalized: Dict[str, Dict[float, Tuple[float, float]]] = {}
    for dataset, props in raw.items():
        dataset_norm = dataset.upper() if dataset.lower() not in {"gsm8k", "gsm8k_cot"} else "GSM8K"
        normalized[dataset_norm] = {}
        for prop_str, stats in props.items():
            prop = float(prop_str)
            mean_acc = stats.get("mean", None)
            ci = stats.get("ci_95", 0.0)
            if mean_acc is None:
                continue
            if ci is None:
                ci = 0.0
            normalized[dataset_norm][prop] = (mean_acc, ci)
    return normalized


def write_summary(path: Path, summaries: Dict[str, Dict[float, Tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, props in summaries.items():
        dataset_payload: Dict[str, Dict[str, float]] = {}
        # ensure 0 proportion slot exists for manual entry
        if 0.0 not in props:
            props = {0.0: (None, None), **props}
        for prop, (mean_acc, ci) in sorted(props.items(), key=lambda x: x[0]):
            dataset_payload[f"{prop:g}"] = {
                "mean": mean_acc,
                "ci_95": ci,
            }
        payload[dataset] = dataset_payload
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_summary(path: Path) -> Dict[str, Dict[float, Tuple[float, float]]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid summary format in {path}")
    return normalize_summary_dict(raw)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    model_dirs = iter_model_dirs(RUNS_DIR)
    if not model_dirs:
        raise SystemExit(f"No model directories found under {RUNS_DIR}")

    outputs = []
    for model_dir in model_dirs:
        out = plot_model(model_dir)
        if out is not None:
            outputs.append(out)

    if not outputs:
        raise SystemExit("No plots generated; check input files.")

    print("Generated plots:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
