#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare decode correctness suites")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    failures = []
    baseline_fixed = {case["name"]: case for case in baseline["fixed"]}
    candidate_fixed = {case["name"]: case for case in candidate["fixed"]}
    for name in sorted(baseline_fixed):
        if name not in candidate_fixed:
            failures.append(f"{name}: missing candidate case")
            continue
        for field in ("token_counts", "token_checksum"):
            if baseline_fixed[name][field] != candidate_fixed[name][field]:
                failures.append(f"{name}: {field} differs")
    for field in ("max_tokens", "token_counts", "token_checksum"):
        if baseline["churn"][field] != candidate["churn"][field]:
            failures.append(f"dynamic_churn: {field} differs")

    result = {
        "baseline": baseline["variant"],
        "candidate": candidate["variant"],
        "fixed_cases": len(baseline_fixed),
        "dynamic_churn": True,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
