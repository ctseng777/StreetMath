import math
from typing import List


def money(x: float) -> float:
    """Round to cents as a number (not string)."""
    return round(x + 1e-9, 2)


def money_str(x: float, currency: str = "USD") -> str:
    s = f"{money(x):.2f}"
    if currency == "USD":
        return f"${s}"
    if currency == "EUR":
        return f"€{s}"
    if currency == "GBP":
        return f"£{s}"
    return s


def rel_error(candidate: float, exact: float) -> float:
    if exact == 0:
        return float("inf") if candidate != 0 else 0.0
    return abs(candidate - exact) / abs(exact)


def perturb_by_rel(x: float, rel: float, direction: int = 1) -> float:
    """Return x * (1 +/- rel)."""
    return x * (1 + rel * (1 if direction >= 0 else -1))


def round_to_base(x: float, base: float) -> float:
    return round(x / base) * base


def round_money_to_5_or_10(x: float) -> float:
    """Round dollar amounts to nearest $5 for moderate totals, $10 for larger totals."""
    base = 10.0 if x >= 60 else 5.0
    return money(round_to_base(x, base))


def round_cents_to_nickel(x: float) -> float:
    """Round to nearest $0.5 for small amounts (useful for per-unit prices)."""
    return money(round_to_base(x, 0.05))


def round_to_int_dollar(x: float) -> int:
    """Round to nearest whole dollar using half-up semantics (e.g., 10.5 -> 11)."""
    return int(math.floor(x + 0.5))


def round_percent_to_step(rate: float, step: float = 0.05) -> float:
    """Round a rate (e.g., 0.0725) to nearest step (e.g., 0.5 for 5%)."""
    return round(rate / step) * step


def floor_dollars(x: float) -> float:
    """Drop cents (truncate toward -inf, but inputs are non-negative)."""
    if x < 0:
        return -float(math.floor(-x + 1e-9))
    return float(math.floor(x + 1e-9))


def floor_to_cents(x: float) -> float:
    """Truncate to cents without rounding up."""
    return math.floor((x + 1e-9) * 100) / 100.0


def round_weight_lb(x: float) -> float:
    """Round pounds to a friendly increment (0.5 lb)."""
    return round_to_base(x, 0.5)


def round_weight_kg(x: float) -> float:
    """Round kilograms to a friendly increment (0.25 kg)."""
    return round_to_base(x, 0.25)


def round_dollars_to_5(x: float) -> float:
    """Round a dollar price to the nearest multiple of $5 (e.g., 10.8 -> 10, 12.6 -> 15)."""
    return round_to_base(x, 5.0)


def sample_significand_rounded_sum(values: List[float]) -> float:
    """Quick human-ish rounding: round each to nearest 0.5 and sum."""
    def round_to_half(v: float) -> float:
        return round(v * 2) / 2.0

    return sum(round_to_half(v) for v in values)




def id_prefix(topic: str, idx: int) -> str:
    return f"{topic}_{idx:06d}"
