Street Math Approximation Benchmark

Overview

- Goal: Benchmark LLMs on everyday “street math” (fast, approximate reasoning) using free-form responses.
- Domains:
  - Basket totals: integer/decimal prices; mental rounding.
  - Discounts: percentage off, BOGO, buy‑n‑get‑m, threshold coupons.
  - Taxes: pre/post‑discount variants; simple state‑style differences.
  - Unit conversions: lb↔oz, kg↔g.
  - Tips & service fees.
- OOD estimation: travel time, simple physics/rate estimates, and Fermi-style crowd size.

Answer Format

- Each example is free-form (no multiple-choice options).
- The dataset provides a numeric target value for grading:
  - `exact_value`: the exact calculation for the question

Data Format

- JSONL with one object per line. Example schema:

  {
    "id": "basket_sum_000123",
    "topic": "basket_sum",
    "subtopic": "decimal_prices",
    "prompt": "You’re buying ... About how much, in dollars, will you pay before tax?",
    "exact_value": 18.42,
    "split": "train"
  }

Generation

- Deterministic generation with a seed for reproducibility.
- Each topic has multiple templates and randomized parameters.
- Distractors are pushed farther away to make the task easier: “mildly off” is ~35–60% error and “way off” is ≥120% error (or <30% of the exact value), with separation checks against the good approximation.
- Output files:
  - `data/street_math_train.jsonl`
  - `data/street_math_test.jsonl`

How to Generate

1) Ensure Python 3.9+ is available.
2) Baseline: `python3 -m streetmath_dataset.scripts.generate_street_math_dataset --seed 42 --train 1000 --test 200`
3) Scale up: `python3 -m streetmath_dataset.scripts.generate_street_math_dataset --seed 42 --train 10000 --test 2000`



Notes

- Prompts emphasize estimation ("about how much", "roughly") to encourage approximate reasoning over exact arithmetic.
- This repo produces data suitable for uploading to Hugging Face Datasets.
- Each prompt is written to encourage estimation ("about how much", "roughly") while `exact_value` retains the exact calculation.
