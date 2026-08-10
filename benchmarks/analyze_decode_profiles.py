#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


METRICS = (
    "prepare_decode_cpu_ms",
    "prepare_block_tables_cpu_ms",
    "numpy_pack_cpu_ms",
    "block_table_pack_cpu_ms",
    "block_table_delta_plan_cpu_ms",
    "packed_row_pack_cpu_ms",
    "packed_h2d_submit_cpu_ms",
    "prepare_sample_cpu_ms",
    "graph_input_copy_cpu_ms",
    "graph_replay_submit_cpu_ms",
    "compute_logits_submit_cpu_ms",
    "sample_and_d2h_cpu_ms",
    "model_runner_run_cpu_ms",
)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(path: Path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    decode = [record for record in records if record["phase"] == "decode"]
    result = {
        "path": str(path),
        "variant": decode[0]["variant"] if decode else None,
        "decode_steps": len(decode),
        "batch_sizes": dict(sorted(Counter(record["batch_size"] for record in decode).items())),
        "graph_buckets": dict(sorted(Counter(record.get("graph_bucket") for record in decode).items())),
        "block_table_widths": dict(sorted(Counter(record["block_table_width"] for record in decode).items())),
        "metrics": {},
    }
    for metric in METRICS:
        values = [record[metric] for record in decode if metric in record]
        if not values:
            continue
        result["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values),
        }
    for metric in (
        "main_graph_metadata_h2d_bytes_est",
        "main_graph_d2d_bytes_est",
        "temperature_h2d_bytes_est",
        "main_total_h2d_bytes_est",
        "graph_full_clear_bytes_est",
        "pr176_staging_clear_bytes_est",
        "pr176_staging_h2d_bytes_est",
        "pr176_scalar_writes_est",
        "copied_block_table_bytes",
        "padding_rows",
        "decode_tensor_allocations",
        "metadata_d2d_copies",
        "packed_metadata_h2d_submissions",
        "packed_metadata_h2d_bytes",
        "block_table_delta_count",
    ):
        values = [record[metric] for record in decode if metric in record]
        if values:
            result[metric] = {
                "per_step_mean": statistics.fmean(values),
                "total": sum(values),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [summarize(path) for path in args.profiles]
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
