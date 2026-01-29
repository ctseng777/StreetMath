#!/usr/bin/env python3
import argparse
import json
from typing import Set


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str, help="Path to JSONL dataset")
    args = p.parse_args()

    errs = 0
    n = 0
    required: Set[str] = {"id", "topic", "subtopic", "prompt", "split", "exact_value"}
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            ex = json.loads(line)
            missing = required - set(ex.keys())
            extra = set(ex.keys()) - required
            if missing:
                print(f"[{n}] missing keys: {sorted(missing)}")
                errs += 1
                continue
            if extra:
                print(f"[{n}] unexpected keys: {sorted(extra)}")
                errs += 1
            if not isinstance(ex["id"], str):
                print(f"[{n}] id must be a string")
                errs += 1
            if not isinstance(ex["topic"], str):
                print(f"[{n}] topic must be a string")
                errs += 1
            if not isinstance(ex["subtopic"], str):
                print(f"[{n}] subtopic must be a string")
                errs += 1
            if not isinstance(ex["prompt"], str):
                print(f"[{n}] prompt must be a string")
                errs += 1
            if not isinstance(ex["split"], str):
                print(f"[{n}] split must be a string")
                errs += 1
            if not isinstance(ex["exact_value"], (int, float)):
                print(f"[{n}] exact_value must be numeric")
                errs += 1

    if errs == 0:
        print(f"OK: {n} rows validated with no errors")
    else:
        print(f"Validation found {errs} issue(s) over {n} rows")


if __name__ == "__main__":
    main()
