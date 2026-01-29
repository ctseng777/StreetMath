#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

JUDGEMENT_ORDER = [
    "Good approximation",
    "Exact math",
    "Mildly off",
    "Way off",
    "Uncategorized",
]

JUDGEMENT_ALIASES = {
    "good approximation": "Good approximation",
    "good": "Good approximation",
    "approx": "Good approximation",
    "exact math": "Exact math",
    "exact": "Exact math",
    "mildly off": "Mildly off",
    "mild": "Mildly off",
    "way off": "Way off",
    "way": "Way off",
    "uncategorized": "Uncategorized",
    "unknown": "Uncategorized",
    "n/a": "Uncategorized",
}

MODEL_PARAM_B = {
    "qwen2_5_3b": 3,
    "qwen2_5_32b": 32,
    "falcon-h1-7b": 7,
    "falcon-h1-34b": 34,
    "dream-v0-instruct-7b": 7,
}

# A100-XSM (400W) peak BF16 throughput ~312 TFLOPs.
FLOPS_PER_KWH = (312e12 * 3600) / 0.4


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )

def latex_seqsplit(text: str) -> str:
    return f"\\seqsplit{{{latex_escape(text)}}}"


def normalize_judgement(label: Optional[str]) -> str:
    if not label:
        return "Uncategorized"
    key = label.strip().lower()
    return JUDGEMENT_ALIASES.get(key, label.strip())


def pick_main_jsonl(run_dir: str) -> Optional[str]:
    base = os.path.basename(run_dir)
    candidates = []
    for fname in os.listdir(run_dir):
        if not fname.endswith(".jsonl"):
            continue
        if "layerwise" in fname:
            continue
        candidates.append(fname)
    if not candidates:
        return None
    exact = f"{base}.jsonl"
    if exact in candidates:
        return os.path.join(run_dir, exact)
    if len(candidates) == 1:
        return os.path.join(run_dir, candidates[0])
    # Pick largest file as fallback.
    candidates.sort(key=lambda f: os.path.getsize(os.path.join(run_dir, f)), reverse=True)
    return os.path.join(run_dir, candidates[0])

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


def estimate_energy_kwh(model_name: str, total_tokens: float) -> Optional[float]:
    params_b = estimate_params_b(model_name)
    if params_b is None or total_tokens <= 0:
        return None
    steps = estimate_steps(model_name)
    flops = 2 * (params_b * 1e9) * total_tokens * steps
    return flops / FLOPS_PER_KWH


def load_topic_subtopic_order(dataset_path: Optional[str]) -> List[Tuple[str, str]]:
    if not dataset_path or not os.path.exists(dataset_path):
        return []
    order: List[Tuple[str, str]] = []
    seen = set()
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            topic = obj.get("topic") or "unknown"
            subtopic = obj.get("subtopic") or "unknown"
            key = (topic, subtopic)
            if key in seen:
                continue
            seen.add(key)
            order.append(key)
    return order


def aggregate_run(jsonl_path: str) -> Dict[str, object]:
    judgement_counts = Counter()
    tool_call_count = 0
    token_sum = 0
    token_count = 0
    counts_by_topic = defaultdict(lambda: Counter())
    counts_by_subtopic = defaultdict(lambda: Counter())

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "summary" in obj:
                continue

            label = normalize_judgement(obj.get("judgement") or obj.get("label"))
            if label not in JUDGEMENT_ORDER:
                label = "Uncategorized"
            judgement_counts[label] += 1

            topic = obj.get("topic") or "unknown"
            subtopic = obj.get("subtopic") or "unknown"
            counts_by_topic[topic][label] += 1
            counts_by_topic[topic]["n"] += 1
            counts_by_subtopic[(topic, subtopic)][label] += 1
            counts_by_subtopic[(topic, subtopic)]["n"] += 1

            has_tool_call = obj.get("has_tool_call")
            if has_tool_call is None:
                if obj.get("tool_call"):
                    has_tool_call = True
                else:
                    tool_calls = obj.get("tool_calls")
                    has_tool_call = bool(tool_calls)
            if has_tool_call:
                tool_call_count += 1

            total_tokens = obj.get("total_tokens")
            if total_tokens is None:
                usage = obj.get("usage") or {}
                total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                prompt_tokens = obj.get("prompt_tokens")
                completion_tokens = obj.get("completion_tokens")
                if prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
            if total_tokens is not None:
                token_sum += float(total_tokens)
                token_count += 1

    avg_tokens = None
    if token_count:
        avg_tokens = token_sum / token_count

    return {
        "judgement_counts": judgement_counts,
        "tool_call_count": tool_call_count,
        "avg_tokens": avg_tokens,
        "counts_by_topic": counts_by_topic,
        "counts_by_subtopic": counts_by_subtopic,
        "samples": sum(judgement_counts.values()),
    }

def format_number(value: Optional[float], decimals: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def render_overall_table(model_rows: List[Dict[str, object]]) -> str:
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{threeparttable}")
    lines.append("  ")
    lines.append("  \\begin{tabular}{lrrrrrrrr}")
    lines.append("  \\toprule")
    lines.append("  Model & A & E & M & W & Uncategorized & Tool calls & Avg tokens & Energy (Wh) \\\\")
    lines.append("  \\midrule")
    for row in model_rows:
        name = latex_escape(row["model"])
        counts = row["judgement_counts"]
        tool_calls = row["tool_call_count"]
        avg_tokens = row["avg_tokens"]
        avg_str = "--" if avg_tokens is None else str(int(round(avg_tokens)))
        samples = float(row.get("samples") or 0)
        energy_kwh = None
        if avg_tokens is not None and samples > 0:
            energy_kwh = estimate_energy_kwh(row["model"], avg_tokens * samples)
        energy_wh = None if energy_kwh is None else energy_kwh * 1000.0
        energy_str = format_number(energy_wh, 1)
        lines.append(
            "  {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\"
        )
        lines[-1] = lines[-1].format(
            name,
            counts.get("Good approximation", 0),
            counts.get("Exact math", 0),
            counts.get("Mildly off", 0),
            counts.get("Way off", 0),
            counts.get("Uncategorized", 0),
            tool_calls,
            avg_str,
            energy_str,
        )
    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\begin{tablenotes}")
    lines.append("  \\small")
    lines.append("  \\item Abbreviations: A = Good approximation, E = Exact Math, M = Mildly off, W = Way off")
    lines.append("  \\end{tablenotes}")
    lines.append("  \\end{threeparttable}")
    lines.append(
        "  \\caption{Overall judgement counts by model with tool calls, average total tokens (rounded), "
        "and estimated energy consumption.}"
    )
    lines.append("  \\label{table:benchmark}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_subtopic_table(
    model_rows: List[Dict[str, object]],
    topic_subtopic_order: List[Tuple[str, str]],
) -> str:
    lines = []
    lines.append("\\begin{longtable}{p{2.4cm} l l r r r r r r}")
    lines.append("  \\toprule")
    lines.append(
        "  Model & Topic & Subtopic & Good approx & Exact math & Mildly off & Way off & Uncat. & N \\\\"
    )
    lines.append("  \\midrule")
    lines.append("  \\endfirsthead")
    lines.append("  \\toprule")
    lines.append(
        "  Model & Topic & Subtopic & Good approx & Exact math & Mildly off & Way off & Uncat. & N \\\\"
    )
    lines.append("  \\midrule")
    lines.append("  \\endhead")
    lines.append("  \\midrule")
    lines.append("  \\multicolumn{9}{r}{\\small\\itshape Continued on next page} \\\\")
    lines.append("  \\endfoot")
    lines.append("  \\bottomrule")
    lines.append("  \\endlastfoot")
    for row in model_rows:
        model_name = latex_seqsplit(row["model"])
        counts_by_subtopic = row["counts_by_subtopic"]
        first = True
        for topic, subtopic in topic_subtopic_order:
            key = (topic, subtopic)
            counts = counts_by_subtopic.get(key, Counter())
            n = counts.get("n", 0)
            line_model = model_name if first else ""
            first = False
            lines.append(
                "  {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\")
            lines[-1] = lines[-1].format(
                line_model,
                latex_escape(topic),
                latex_escape(subtopic),
                counts.get("Good approximation", 0),
                counts.get("Exact math", 0),
                counts.get("Mildly off", 0),
                counts.get("Way off", 0),
                counts.get("Uncategorized", 0),
                n,
            )
    lines.append("  \\bottomrule")
    lines.append("  \\caption{Benchmark results: Counts by topic and subtopic for all models.}")
    lines.append("  \\label{table:benchmark_by_subtopic}")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark LaTeX tables from runs/ directories.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing run subfolders.",
    )
    parser.add_argument(
        "--dataset",
        default="StreetMathDataset/data/street_math_test.jsonl",
        help="Dataset JSONL to define topic/subtopic ordering.",
    )
    parser.add_argument(
        "--outdir",
        default="BenchmarkAndLayerwiseAnalysis",
        help="Output directory for LaTeX tables.",
    )
    parser.add_argument(
        "--overall-out",
        default="Benchmark_overall.tex",
        help="Filename for overall table LaTeX.",
    )
    parser.add_argument(
        "--subtopic-out",
        default="Benchmark_by_subtopic.tex",
        help="Filename for subtopic table LaTeX.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    topic_subtopic_order = load_topic_subtopic_order(args.dataset)

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

    if not model_rows:
        raise SystemExit("No qualifying runs found.")

    overall_table = render_overall_table(model_rows)
    subtopic_table = render_subtopic_table(model_rows, topic_subtopic_order)

    os.makedirs(args.outdir, exist_ok=True)
    overall_path = os.path.join(args.outdir, args.overall_out)
    subtopic_path = os.path.join(args.outdir, args.subtopic_out)

    with open(overall_path, "w", encoding="utf-8") as f:
        f.write(overall_table)
    with open(subtopic_path, "w", encoding="utf-8") as f:
        f.write(subtopic_table)

    print(f"Wrote: {overall_path}")
    print(f"Wrote: {subtopic_path}")


if __name__ == "__main__":
    main()
