#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


CASES = (
    "batch-1",
    "batch-4",
    "batch-8",
    "batch-16",
    "batch-4-prompt-1024",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def load_steps(path: Path):
    value = load_json(path)
    if len(value) != 1:
        raise ValueError(f"expected exactly one step summary in {path}")
    return value[0]


def metric(summary, name):
    return summary["metrics"][name]["mean"]


def percent_delta(baseline, candidate):
    return 100.0 * (candidate / baseline - 1.0)


def memcpy_counts(path: Path):
    result = {"H2D": 0, "D2D": 0, "D2H": 0}
    names = {
        "[CUDA memcpy Host-to-Device]": "H2D",
        "[CUDA memcpy Device-to-Device]": "D2D",
        "[CUDA memcpy Device-to-Host]": "D2H",
    }
    for line in path.read_text().splitlines():
        for operation, key in names.items():
            if line.endswith(operation):
                result[key] = int(line.split(",")[2])
    return result


def load_variant(root: Path, case: str, name: str):
    case_dir = root / case
    run = load_json(case_dir / f"{name}.summary.json")
    steps = load_steps(case_dir / f"{name}.step-summary.json")
    copies = memcpy_counts(case_dir / f"{name}.nsys-stats.csv")
    return run, steps, copies


def parse_args():
    parser = argparse.ArgumentParser(description="Compare main, PR #176, and v2")
    parser.add_argument("--main-root", required=True, type=Path)
    parser.add_argument("--pr-root", required=True, type=Path)
    parser.add_argument("--v2-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for case in CASES:
        main_run, main_steps, main_copies = load_variant(args.main_root, case, "main")
        pr_run, pr_steps, pr_copies = load_variant(args.pr_root, case, "pr176-port")
        v2_run, v2_steps, v2_copies = load_variant(args.v2_root, case, "v2")
        main_tps = main_run["aggregate"]["output_tokens_per_s"]
        pr_tps = pr_run["aggregate"]["output_tokens_per_s"]
        v2_tps = v2_run["aggregate"]["output_tokens_per_s"]
        rows.append(
            {
                "case": case,
                "main_tps": main_tps,
                "pr_tps": pr_tps,
                "pr_delta": percent_delta(main_tps, pr_tps),
                "v2_tps": v2_tps,
                "v2_delta": percent_delta(main_tps, v2_tps),
                "main_prepare_us": metric(main_steps, "prepare_decode_cpu_ms") * 1000,
                "pr_prepare_us": metric(pr_steps, "prepare_decode_cpu_ms") * 1000,
                "v2_prepare_us": metric(v2_steps, "prepare_decode_cpu_ms") * 1000,
                "v2_numpy_us": metric(v2_steps, "numpy_pack_cpu_ms") * 1000,
                "v2_table_us": metric(v2_steps, "block_table_pack_cpu_ms") * 1000,
                "main_d2d": main_copies["D2D"],
                "pr_d2d": pr_copies["D2D"],
                "v2_d2d": v2_copies["D2D"],
                "v2_h2d": v2_copies["H2D"],
            }
        )

    lines = [
        "# Day 2 A/B/C: main vs. PR #176 port vs. DecodeInputBatch v2",
        "",
        "| workload | main tok/s | PR tok/s | PR Δ | v2 tok/s | v2 Δ | prepare main/PR/v2 (µs) | v2 vector/table pack (µs) | D2D main/PR/v2 | v2 H2D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['main_tps']:.1f} | {row['pr_tps']:.1f} | "
            f"{row['pr_delta']:+.2f}% | {row['v2_tps']:.1f} | {row['v2_delta']:+.2f}% | "
            f"{row['main_prepare_us']:.1f}/{row['pr_prepare_us']:.1f}/{row['v2_prepare_us']:.1f} | "
            f"{row['v2_numpy_us']:.1f}/{row['v2_table_us']:.1f} | "
            f"{row['main_d2d']}/{row['pr_d2d']}/{row['v2_d2d']} | {row['v2_h2d']} |"
        )

    failed = [row["case"] for row in rows if row["v2_delta"] < -1.0]
    lines.extend(
        [
            "",
            f"Acceptance (v2 >= main -1%): {'PASS' if not failed else 'FAIL'}.",
            f"Cases below threshold: {', '.join(failed) if failed else 'none'}.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
