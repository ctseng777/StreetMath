import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .utils import (
    id_prefix,
    money,
    money_str,
)


@dataclass
class Example:
    id: str
    topic: str
    subtopic: str
    prompt: str
    split: str
    exact_value: float


def _rand_prices(n: int, low: float = 0.99, high: float = 24.99) -> List[float]:
    return [round(random.uniform(low, high), 2) for _ in range(n)]


def _choose_currency() -> str:
    return "USD"


def gen_basket_sum(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        n_items = random.randint(3, 6)
        currency = _choose_currency()
        prices = _rand_prices(n_items, 0.79, 19.99)
        exact_total = money(sum(prices))
        qtext = (
            f"You’re buying {n_items} items: "
            + ", ".join(money_str(p, currency) for p in prices)
            + ". About how much will you pay (before tax)?"
        )
        ex = Example(
            id=id_prefix("basket_sum", idx_start + i),
            topic="basket_sum",
            subtopic="decimal_prices",
            prompt=qtext,
            split=split,
            exact_value=exact_total,
        )
        out.append(ex)
    return out


def gen_discounts(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        currency = _choose_currency()
        variant = random.choice(["percent", "bogo", "buy_n_get_m", "threshold", "stacked"])

        if variant == "percent":
            base = _rand_prices(random.randint(2, 5), 2.99, 49.99)
            off = random.choice([10, 15, 20, 25, 30])
            subtotal = sum(base)
            exact_total = money(subtotal * (1 - off / 100))
            prompt = (
                "Cart: "
                + ", ".join(money_str(p, currency) for p in base)
                + f". A {off}% discount applies to everything. About how much do you pay?"
            )

        elif variant == "bogo":
            # BOGO of equal/lesser value, pair items by price descending
            base = _rand_prices(random.randint(2, 6), 1.49, 24.99)
            base.sort(reverse=True)
            to_pay = 0.0
            for j in range(0, len(base), 2):
                to_pay += base[j]  # pay for higher in each pair
            exact_total = money(to_pay)
            prompt = (
                "BOGO sale (buy one, get one free; free item is the cheaper one in each pair). Items: "
                + ", ".join(money_str(p, currency) for p in base)
                + ". About how much do you pay?"
            )

        elif variant == "buy_n_get_m":
            price = round(random.uniform(1.49, 7.99), 2)
            n = random.randint(3, 5)
            m = random.randint(1, 3)
            qty = random.randint(n + m, 2 * (n + m))
            groups = qty // (n + m)
            remainder = qty % (n + m)
            paid_units = groups * n + min(remainder, n)
            exact_total = money(paid_units * price)
            prompt = (
                f"Buy {n} get {m} free on items priced "
                f"{money_str(price, currency)} each. You take {qty} total. "
                "About how much do you pay?"
            )

        elif variant == "threshold":
            base = _rand_prices(random.randint(2, 5), 3.99, 39.99)
            threshold = random.choice([25, 30, 40, 50])
            off = random.choice([5, 10, 15])
            subtotal = sum(base)
            exact_total = money(subtotal - (off if subtotal >= threshold else 0))
            prompt = (
                "Threshold coupon applies: "
                + f"${off} off ${threshold}+ spend. Cart: "
                + ", ".join(money_str(p, currency) for p in base)
                + ". About how much do you pay?"
            )
        else:  # stacked discount
            base = _rand_prices(random.randint(3, 6), 2.99, 39.99)
            percent_off = random.choice([10, 15, 20])
            threshold = random.choice([30, 40, 50])
            coupon = random.choice([5, 10])
            subtotal = sum(base)
            discounted = subtotal * (1 - percent_off / 100.0)
            exact_total = money(discounted - (coupon if discounted >= threshold else 0))
            prompt = (
                "Stacked deal: "
                f"{percent_off}% off, then ${coupon} coupon if you spend "
                f"${threshold} or more. Cart: "
                + ", ".join(money_str(p, currency) for p in base)
                + ". About how much do you pay?"
            )

        ex = Example(
            id=id_prefix("discounts", idx_start + i),
            topic="discounts",
            subtopic=variant,
            prompt=prompt,
            split=split,
            exact_value=exact_total,
        )
        out.append(ex)
    return out


def gen_taxes(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        currency = _choose_currency()
        variant = random.choice(["pre_discount_tax", "post_discount_tax"])  # simplified state styles
        base = _rand_prices(random.randint(2, 5), 3.49, 59.99)
        rate = random.choice([0.06, 0.07, 0.075, 0.08, 0.085])
        percent = int(rate * 1000) / 10  # e.g., 7.5

        # Add a percent‑off store discount to create the ambiguity of base for tax
        off = random.choice([0, 10, 15, 20])
        subtotal = sum(base)
        discounted = subtotal * (1 - off / 100)
        taxable_base = subtotal if variant == "pre_discount_tax" else discounted
        tax = taxable_base * rate
        exact_total = money(discounted + tax)
        style = "tax calculated before discount" if variant == "pre_discount_tax" else "tax after discount"

        prompt = (
            "Cart: "
            + ", ".join(money_str(p, currency) for p in base)
            + (f" with a {off}% store discount" if off else "")
            + f". Local sales tax is {percent}% ({style}). About how much is the total?"
        )

        ex = Example(
            id=id_prefix("taxes", idx_start + i),
            topic="taxes",
            subtopic=variant,
            prompt=prompt,
            split=split,
            exact_value=exact_total,
        )
        out.append(ex)
    return out


def gen_units(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        currency = _choose_currency()
        variant = random.choice(["lb_oz", "kg_g"])  # unit conversions only

        if variant == "lb_oz":
            price = round(random.uniform(2.99, 14.99), 2)
            pounds = round(random.uniform(0.5, 6.0), 2)
            ounces = pounds * 16
            exact_total = money(price * pounds)
            prompt = (
                f"You buy {pounds} lb at {money_str(price, currency)} per lb. About how much is that total?"
            )
        elif variant == "kg_g":
            price_per_kg = round(random.uniform(3.99, 19.99), 2)
            grams = random.choice([250, 300, 350, 400, 500, 750, 900])
            kg = grams / 1000
            exact_total = money(price_per_kg * kg)
            prompt = (
                f"An item costs {money_str(price_per_kg, currency)} per kg. You take {grams} g. About how much is that?"
            )
        else:
            # Should not happen; keep for completeness
            raise RuntimeError("Unexpected units variant")

        ex = Example(
            id=id_prefix("units", idx_start + i),
            topic="units",
            subtopic=variant,
            prompt=prompt,
            split=split,
            exact_value=exact_total,
        )
        out.append(ex)
    return out


def gen_tips(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        currency = _choose_currency()
        subtotal = round(random.uniform(12.0, 120.0), 2)
        tip_percent = random.choice([15, 18, 20, 22])
        svc_fee_percent = random.choice([0, 3, 5])
        svc = subtotal * (svc_fee_percent / 100)
        tip = (subtotal + svc) * (tip_percent / 100)
        exact_total = money(subtotal + svc + tip)
        prompt = (
            f"Bill is {money_str(subtotal, currency)}; service fee {svc_fee_percent}% and tip {tip_percent}%. "
            f"About how much do you pay in total?"
        )
        ex = Example(
            id=id_prefix("tips", idx_start + i),
            topic="tips",
            subtopic="tips_service",
            prompt=prompt,
            split=split,
            exact_value=exact_total,
        )
        out.append(ex)
    return out


def gen_travel_time(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        distance = round(random.uniform(5, 240), 1)
        speed = random.choice([25, 35, 45, 55, 65])
        distance_unit = "miles"
        speed_unit = "mph"
        exact_hours = distance / speed
        prompt = (
            f"You need to travel about {int(round(distance))} {distance_unit} at around {speed} {speed_unit}. "
            "About how many hours will it take?"
        )
        ex = Example(
            id=id_prefix("travel_time", idx_start + i),
            topic="travel_time",
            subtopic="drive_time",
            prompt=prompt,
            split=split,
            exact_value=exact_hours,
        )
        out.append(ex)
    return out


def gen_physics_estimate(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        liters = random.choice([10, 15, 20, 30, 40, 50])
        flow_lpm = random.choice([4, 5, 6, 7, 8])
        exact_minutes = liters / flow_lpm
        prompt = (
            f"A faucet fills {liters} liters at about {flow_lpm} liters per minute. "
            "About how many minutes does it take?"
        )
        ex = Example(
            id=id_prefix("physics_estimate", idx_start + i),
            topic="physics_estimate",
            subtopic="fill_rate",
            prompt=prompt,
            split=split,
            exact_value=exact_minutes,
        )
        out.append(ex)
    return out


def gen_fermi(
    idx_start: int,
    count: int,
    split: str,
) -> List[Example]:
    out: List[Example] = []
    for i in range(count):
        length = random.choice([30, 40, 50, 60, 70])
        width = random.choice([20, 25, 30, 35])
        density = random.choice([1, 2, 3, 4])
        exact_people = length * width * density
        prompt = (
            f"A plaza is about {length} m by {width} m, with roughly {density} people per square meter. "
            "About how many people are there?"
        )
        ex = Example(
            id=id_prefix("fermi", idx_start + i),
            topic="fermi",
            subtopic="crowd_density",
            prompt=prompt,
            split=split,
            exact_value=exact_people,
        )
        out.append(ex)
    return out


def generate_examples(
    seed: int,
    train_n: int,
    test_n: int,
    weights: Dict[str, float] | None = None,
) -> Tuple[List[Example], List[Example]]:
    random.seed(seed)
    # Topic distribution
    topics = ["basket_sum", "discounts", "taxes", "units", "tips", "fermi", "travel_time", "physics_estimate"]
    if weights is None:
        weights = {t: 1.0 for t in topics}
        # Mildly upweight basket and discounts by default
        weights["basket_sum"] = 1.4
        weights["discounts"] = 1.4
        weights["fermi"] = 0.8
        weights["travel_time"] = 0.8
        weights["physics_estimate"] = 0.8

    def distribute(N: int) -> Dict[str, int]:
        total_w = sum(weights.values())
        alloc = {t: int(N * (weights[t] / total_w)) for t in topics}
        # fix rounding
        while sum(alloc.values()) < N:
            t = random.choice(topics)
            alloc[t] += 1
        return alloc

    train_alloc = distribute(train_n)
    test_alloc = distribute(test_n)

    # Generate
    train: List[Example] = []
    test: List[Example] = []

    # helpers to keep indices unique per topic
    counters = {t: 0 for t in topics}

    def add(topic: str, split: str, n: int):
        start = counters[topic]
        if topic == "basket_sum":
            exs = gen_basket_sum(start, n, split)
        elif topic == "discounts":
            exs = gen_discounts(start, n, split)
        elif topic == "taxes":
            exs = gen_taxes(start, n, split)
        elif topic == "units":
            exs = gen_units(start, n, split)
        elif topic == "tips":
            exs = gen_tips(start, n, split)
        elif topic == "fermi":
            exs = gen_fermi(start, n, split)
        elif topic == "travel_time":
            exs = gen_travel_time(start, n, split)
        elif topic == "physics_estimate":
            exs = gen_physics_estimate(start, n, split)
        else:
            exs = []
        counters[topic] += n
        (train if split == "train" else test).extend(exs)

    for t in topics:
        add(t, "train", train_alloc[t])
    for t in topics:
        add(t, "test", test_alloc[t])

    return train, test
