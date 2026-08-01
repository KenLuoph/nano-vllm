#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize decode packing microbenchmark")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text())
    grouped = {}
    for row in payload["results"]:
        key = (row["batch_size"], row["block_width"])
        grouped.setdefault(key, {})[row["variant"]] = row

    lines = [
        "# Decode CPU packing microbenchmark",
        "",
        f"Each cell is median latency over {payload['iterations']:,} iterations after "
        f"{payload['warmup']:,} warmup iterations; pinned memory={payload['pin_memory']}.",
        "",
        "| batch | block width | main (µs) | PR #176 (µs) | v2 NumPy (µs) | v2 vs main | v2 vs PR |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    main_ratios = []
    pr_ratios = []
    for (batch_size, block_width), variants in sorted(grouped.items()):
        main_us = variants["main"]["median_us"]
        pr_us = variants["pr176"]["median_us"]
        v2_us = variants["v2_numpy"]["median_us"]
        main_ratio = v2_us / main_us
        pr_ratio = v2_us / pr_us
        main_ratios.append(main_ratio)
        pr_ratios.append(pr_ratio)
        lines.append(
            f"| {batch_size} | {block_width} | {main_us:.2f} | {pr_us:.2f} | "
            f"{v2_us:.2f} | {(main_ratio - 1) * 100:+.1f}% | {(pr_ratio - 1) * 100:+.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Geometric-mean latency delta: v2 vs main "
            f"{(statistics.geometric_mean(main_ratios) - 1) * 100:+.1f}%; "
            f"v2 vs PR #176 {(statistics.geometric_mean(pr_ratios) - 1) * 100:+.1f}%.",
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
