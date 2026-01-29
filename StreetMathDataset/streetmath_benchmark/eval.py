import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


FREEFORM_PATTERN = re.compile(r"Final\s*answer\s*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
NUM_PATTERN = re.compile(r"([-+]?\d*\.?\d+)")
TOOL_CALL_BLOCK_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_OPEN_PATTERN = re.compile(r"<tool_call\b[^>]*>", re.IGNORECASE)


def parse_money(s: str) -> Optional[float]:
    s = s.strip()
    # Extract first number; dataset choices often like "$43.00"
    m = NUM_PATTERN.search(s.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def relative_error(candidate: float, exact: float) -> float:
    if exact == 0:
        return float("inf") if candidate != 0 else 0.0
    return abs(candidate - exact) / abs(exact)


def extract_thought(text: str) -> Optional[str]:
    # Common tags: <think>...</think>, <reasoning>...</reasoning>, <scratchpad>...</scratchpad>
    for tag in ["think", "reasoning", "scratchpad"]:
        m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_tool_call(text: str) -> tuple[bool, Optional[str]]:
    # Return (has_tool_call, content). If only an opening tag is present, content is None.
    m = TOOL_CALL_BLOCK_PATTERN.search(text)
    if m:
        return True, m.group(1).strip()
    if TOOL_CALL_OPEN_PATTERN.search(text):
        return True, None
    return False, None


def naive_token_count(text: str) -> int:
    return len(text.strip().split()) if text else 0


def count_tokens(text: str, model: Optional[str] = None) -> int:
    # Try tiktoken if available, else fallback to whitespace tokens.
    try:
        import tiktoken  # type: ignore

        enc = None
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return naive_token_count(text)


def _token_usage(
    usage: Optional[Dict[str, Any]],
    prompt_text: str,
    response_text: str,
    model_name: Optional[str],
) -> Dict[str, Any]:
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    token_count_source = None

    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        token_count_source = usage.get("token_count_source")

    if prompt_tokens is None or completion_tokens is None:
        prompt_tokens = count_tokens(prompt_text or "", model=model_name)
        completion_tokens = count_tokens(response_text or "", model=model_name)
        total_tokens = prompt_tokens + completion_tokens
        token_count_source = token_count_source or "heuristic"

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "token_count_source": token_count_source,
    }


def _estimate_flops(
    arch: Optional[str],
    model_params: Optional[float],
    completion_tokens: int,
    diffusion_steps: Optional[int],
    active_params: Optional[float],
) -> Optional[float]:
    if not arch or not model_params:
        return None
    arch = arch.lower()
    if arch in ("dense", "ssm"):
        return 2.0 * model_params * completion_tokens
    if arch == "moe":
        if not active_params:
            return None
        return 2.0 * active_params * completion_tokens
    if arch == "diffusion":
        if not diffusion_steps:
            return None
        return 2.0 * model_params * diffusion_steps
    return None


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def _two_prop_z_test(k1: int, n1: int, k2: int, n2: int) -> Optional[float]:
    if n1 == 0 or n2 == 0:
        return None
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if denom == 0:
        return None
    z = (p1 - p2) / denom
    # two-sided p-value
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


@dataclass
class Judgement:
    label: str  # "Good approximation", "Exact math", "Mildly off", "Way off"


def classify_freeform(sample: Dict[str, Any], numeric_guess: Optional[float]) -> Tuple[Judgement, Optional[float]]:
    exact = sample.get("exact_value")
    if numeric_guess is None or exact is None:
        return Judgement("Uncategorized"), None
    exact_val = float(exact)
    if math.isclose(numeric_guess, exact_val, rel_tol=0, abs_tol=1e-6):
        return Judgement("Exact math"), 0.0
    re = relative_error(numeric_guess, exact_val)
    if re <= 0.2:
        return Judgement("Good approximation"), re
    if re <= 0.6:
        return Judgement("Mildly off"), re
    return Judgement("Way off"), re


def extract_freeform_number(text: str) -> Optional[float]:
    m = FREEFORM_PATTERN.search(text or "")
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return parse_money(text or "")


def build_result_record(
    sample: Dict[str, Any],
    provider_name: str,
    model_name: str,
    response_text: str,
    usage: Optional[Dict[str, Any]],
    elapsed: float,
    prompt_text: str,
    eval_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thought = extract_thought(response_text)
    has_tool_call, tool_call_content = extract_tool_call(response_text)
    num_answer = extract_freeform_number(response_text)
    predicted_label = None
    judgement, rel_err = classify_freeform(sample, num_answer)
    tokens_non_final = count_tokens(response_text or "", model=model_name)
    tokens_reasoning = count_tokens(thought or "", model=model_name)
    tokens_response = count_tokens(response_text or "", model=model_name)
    usage_tokens = _token_usage(usage, prompt_text, response_text, model_name)
    diffusion_steps = None if not usage else usage.get("diffusion_steps")

    flops = None
    flops_arch = None
    flops_params = None
    flops_active_params = None
    decoding = None
    if eval_config:
        flops_arch = eval_config.get("arch")
        flops_params = eval_config.get("model_params")
        flops_active_params = eval_config.get("active_params")
        diffusion_steps = diffusion_steps or eval_config.get("diffusion_steps")
        decoding = eval_config.get("decoding")
        flops = _estimate_flops(
            arch=flops_arch,
            model_params=flops_params,
            completion_tokens=usage_tokens["completion_tokens"],
            diffusion_steps=diffusion_steps,
            active_params=flops_active_params,
        )

    rec = {
        "id": sample.get("id"),
        "topic": sample.get("topic"),
        "subtopic": sample.get("subtopic"),
        "prompt": sample.get("prompt"),
        "exact_value": sample.get("exact_value"),
        "model": model_name,
        "provider": provider_name,
        "response": response_text,
        "numeric_answer": num_answer,
        "relative_error": rel_err,
        "thinking_text": thought,
        "has_tool_call": has_tool_call,
        "tool_call": tool_call_content,
        "usage": usage,
        "decoding": decoding,
        "prompt_tokens": usage_tokens["prompt_tokens"],
        "completion_tokens": usage_tokens["completion_tokens"],
        "total_tokens": usage_tokens["total_tokens"],
        "token_count_source": usage_tokens["token_count_source"],
        "diffusion_steps": diffusion_steps,
        "flops_estimate": flops,
        "flops_arch": flops_arch,
        "flops_model_params": flops_params,
        "flops_active_params": flops_active_params,
        "tokens_non_final": tokens_non_final,
        "tokens_reasoning": tokens_reasoning,
        "tokens_response": tokens_response,
        "elapsed_seconds": elapsed,
        "judgement": judgement.label,
    }
    return rec


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)

    # Counts per judgement
    counts: Dict[str, int] = {}
    for r in records:
        j = r.get("judgement", "Uncategorized")
        counts[j] = counts.get(j, 0) + 1

    # Sum total tokens in model responses (completion tokens) across records
    token_sum = 0
    for r in records:
        resp_text = r.get("response", "") or ""
        model_name = r.get("model")
        token_sum += count_tokens(resp_text, model=model_name)

    # Extract key judgement counts explicitly
    count_good = counts.get("Good approximation", 0)
    count_exact = counts.get("Exact math", 0)
    count_mild = counts.get("Mildly off", 0)
    count_way = counts.get("Way off", 0)
    rel_errors = [r.get("relative_error") for r in records if r.get("relative_error") is not None]

    # Judgement distribution per topic/subtopic
    counts_by_topic: Dict[str, Dict[str, int]] = {}
    counts_by_subtopic: Dict[str, Dict[str, int]] = {}
    for r in records:
        topic = r.get("topic") or "unknown"
        subtopic = r.get("subtopic") or "unknown"

        # Judgement distribution
        j = r.get("judgement") or ""
        jt = counts_by_topic.setdefault(
            topic,
            {
                "n": 0,
                "count_good_approximation": 0,
                "count_exact_math": 0,
                "count_mildly_off": 0,
                "count_way_off": 0,
            },
        )
        jt["n"] += 1
        if j == "Good approximation":
            jt["count_good_approximation"] += 1
        elif j == "Exact math":
            jt["count_exact_math"] += 1
        elif j == "Mildly off":
            jt["count_mildly_off"] += 1
        elif j == "Way off":
            jt["count_way_off"] += 1

        js = counts_by_subtopic.setdefault(
            subtopic,
            {
                "n": 0,
                "count_good_approximation": 0,
                "count_exact_math": 0,
                "count_mildly_off": 0,
                "count_way_off": 0,
            },
        )
        js["n"] += 1
        if j == "Good approximation":
            js["count_good_approximation"] += 1
        elif j == "Exact math":
            js["count_exact_math"] += 1
        elif j == "Mildly off":
            js["count_mildly_off"] += 1
        elif j == "Way off":
            js["count_way_off"] += 1

    # Counts for tool calls and explicit exact mentions
    tool_call_count = sum(1 for r in records if r.get("has_tool_call"))
    good_rate = count_good / n if n else 0.0
    exact_rate = count_exact / n if n else 0.0
    good_ci = _wilson_ci(count_good, n)
    exact_ci = _wilson_ci(count_exact, n)

    flops_values = [r.get("flops_estimate") for r in records if r.get("flops_estimate") is not None]

    token_sources: Dict[str, int] = {}
    for r in records:
        src = r.get("token_count_source") or "unknown"
        token_sources[src] = token_sources.get(src, 0) + 1

    summary = {
        "samples": n,
        "judgement_counts": counts,
        "count_good_approximation": count_good,
        "count_exact_math": count_exact,
        "count_mildly_off": count_mild,
        "count_way_off": count_way,
        "good_approximation_rate": good_rate,
        "exact_math_rate": exact_rate,
        "good_approximation_rate_ci_95": good_ci,
        "exact_math_rate_ci_95": exact_ci,
        "counts_by_topic": counts_by_topic,
        "counts_by_subtopic": counts_by_subtopic,
        "tool_call_count": tool_call_count,
        "avg_elapsed_seconds": (sum(r.get("elapsed_seconds", 0.0) for r in records) / n if n else 0.0),
        "avg_tokens_non_final": (sum(r.get("tokens_non_final", 0) for r in records) / n if n else 0.0),
        "avg_tokens_reasoning": (sum(r.get("tokens_reasoning", 0) for r in records) / n if n else 0.0),
        "avg_tokens": (token_sum / n if n else 0.0),
        "avg_completion_tokens": (sum(r.get("completion_tokens", 0) for r in records) / n if n else 0.0),
        "avg_prompt_tokens": (sum(r.get("prompt_tokens", 0) for r in records) / n if n else 0.0),
        "avg_total_tokens": (sum(r.get("total_tokens", 0) for r in records) / n if n else 0.0),
        "avg_flops_estimate": (sum(flops_values) / len(flops_values) if flops_values else None),
        "token_counts_include_prompt": True,
        "token_count_sources": token_sources,
        "avg_relative_error": (sum(rel_errors) / len(rel_errors) if rel_errors else None),
        "median_relative_error": (sorted(rel_errors)[len(rel_errors) // 2] if rel_errors else None),
    }
    return summary
