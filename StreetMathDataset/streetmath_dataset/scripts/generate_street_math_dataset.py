#!/usr/bin/env python3
import argparse
import json
import os
from typing import List

from streetmath_dataset import generate_examples


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            filtered = {
                "id": r["id"],
                "topic": r["topic"],
                "subtopic": r["subtopic"],
                "prompt": r["prompt"],
                "split": r["split"],
                "exact_value": r["exact_value"],
            }
            f.write(json.dumps(filtered, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Street Math Approximation dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--outdir", type=str, default="data")
    args = p.parse_args()

    train, test = generate_examples(
        seed=args.seed,
        train_n=args.train,
        test_n=args.test,
    )

    train_rows = [e.__dict__ for e in train]
    test_rows = [e.__dict__ for e in test]

    write_jsonl(os.path.join(args.outdir, "street_math_train.jsonl"), train_rows)
    write_jsonl(os.path.join(args.outdir, "street_math_test.jsonl"), test_rows)
    print(f"Wrote {len(train_rows)} train and {len(test_rows)} test examples to {args.outdir}")


if __name__ == "__main__":
    main()
