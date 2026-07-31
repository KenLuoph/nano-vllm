#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_CASES = (
    "batch-1",
    "batch-4",
    "batch-8",
    "batch-16",
    "batch-4-prompt-1024",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def load_step_summary(path: Path):
    value = load_json(path)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"expected one summary in {path}")
    return value[0]


def mean_metric(summary, name):
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


def fmt_delta(value):
    return f"{value:+.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Compare Day 1 decode profiles")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-name", default="main")
    parser.add_argument("--candidate-name", default="pr176-port")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        base_dir = args.baseline_root / case
        candidate_dir = args.candidate_root / case
        base_summary = load_json(base_dir / f"{args.baseline_name}.summary.json")
        candidate_summary = load_json(candidate_dir / f"{args.candidate_name}.summary.json")
        base_steps = load_step_summary(base_dir / f"{args.baseline_name}.step-summary.json")
        candidate_steps = load_step_summary(candidate_dir / f"{args.candidate_name}.step-summary.json")
        base_memcpy = memcpy_counts(base_dir / f"{args.baseline_name}.nsys-stats.csv")
        candidate_memcpy = memcpy_counts(candidate_dir / f"{args.candidate_name}.nsys-stats.csv")
        rows.append({
            "case": case,
            "baseline_tps": base_summary["aggregate"]["output_tokens_per_s"],
            "candidate_tps": candidate_summary["aggregate"]["output_tokens_per_s"],
            "throughput_delta": percent_delta(
                base_summary["aggregate"]["output_tokens_per_s"],
                candidate_summary["aggregate"]["output_tokens_per_s"],
            ),
            "prepare_decode_delta": percent_delta(
                mean_metric(base_steps, "prepare_decode_cpu_ms"),
                mean_metric(candidate_steps, "prepare_decode_cpu_ms"),
            ),
            "graph_input_delta": percent_delta(
                mean_metric(base_steps, "graph_input_copy_cpu_ms"),
                mean_metric(candidate_steps, "graph_input_copy_cpu_ms"),
            ),
            "runner_delta": percent_delta(
                mean_metric(base_steps, "model_runner_run_cpu_ms"),
                mean_metric(candidate_steps, "model_runner_run_cpu_ms"),
            ),
            "baseline_d2d": base_memcpy["D2D"],
            "candidate_d2d": candidate_memcpy["D2D"],
            "h2d": candidate_memcpy["H2D"],
            "block_width": next(iter(candidate_steps["block_table_widths"])),
        })

    lines = [
        "# Day 1: current main vs. PR #176 port",
        "",
        "| workload | main tok/s | port tok/s | throughput | prepare_decode | graph input | runner total | D2D copies | H2D copies | block width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['baseline_tps']:.1f} | "
            f"{row['candidate_tps']:.1f} | {fmt_delta(row['throughput_delta'])} | "
            f"{fmt_delta(row['prepare_decode_delta'])} | "
            f"{fmt_delta(row['graph_input_delta'])} | "
            f"{fmt_delta(row['runner_delta'])} | "
            f"{row['baseline_d2d']} -> {row['candidate_d2d']} | "
            f"{row['h2d']} | {row['block_width']} |"
        )
    lines.extend([
        "",
        "Negative phase deltas mean lower CPU time in the candidate. Nsight counts",
        "cover three measured generations, including their prefill steps.",
        "",
    ])
    rendered = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
