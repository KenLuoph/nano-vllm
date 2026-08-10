#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Render formal matrix CSV/Markdown/plots")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def median(values):
    return statistics.median(values) if values else None


def decode_profile(summary):
    path = Path(summary["profile"]["output_path"])
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [record for record in records if record.get("phase") == "decode"]

    def values(field):
        return [record[field] for record in records if field in record]

    return {
        "decode_steps": len(records),
        "prepare_decode_median_ms": median(values("prepare_decode_cpu_ms")),
        "prepare_decode_p95_ms": percentile(values("prepare_decode_cpu_ms"), 0.95),
        "sample_d2h_median_ms": median(values("sample_and_d2h_cpu_ms")),
        "packed_h2d_submissions_total": sum(values("packed_metadata_h2d_submissions")),
        "packed_h2d_bytes_total": sum(values("packed_metadata_h2d_bytes")),
        "block_table_delta_count_total": sum(values("block_table_delta_count")),
        "metadata_d2d_total": sum(values("metadata_d2d_copies")),
    }


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def tie_break_values(initial_path):
    tie_root = initial_path.parent.parent / "tie-break"
    values = []
    for path in sorted(tie_root.glob("round-*/*.summary.json")):
        summary = json.loads(path.read_text())
        values.extend(run["output_tokens_per_s"] for run in summary["measured"])
    return values


def collect(results_root):
    rows = []
    for path in sorted(results_root.glob("*/*/*/initial/*.summary.json")):
        summary = json.loads(path.read_text())
        relative = path.relative_to(results_root)
        model, workload, variant = relative.parts[:3]
        measured = summary["measured"]
        initial_throughputs = [run["output_tokens_per_s"] for run in measured]
        tie_values = tie_break_values(path)
        profile = decode_profile(summary)
        row = {
            "model": model,
            "workload": workload,
            "variant": variant,
            "git_revision": summary["git_revision"],
            "model_revision": summary.get("model_revision"),
            "device": summary["device"],
            "runs": len(tie_values) if tie_values else len(initial_throughputs),
            "tok_s_median": median(tie_values or initial_throughputs),
            "tok_s_initial_aggregate": summary["aggregate"]["output_tokens_per_s"],
            "tpot_p50_ms": median([run["tpot"]["p50_ms"] for run in measured]),
            "tpot_p95_ms": median([run["tpot"]["p95_ms"] for run in measured]),
            "tpot_p99_ms": median([run["tpot"]["p99_ms"] for run in measured]),
            "peak_allocated_bytes": summary["memory"]["peak_allocated_bytes"],
            "peak_reserved_bytes": summary["memory"]["peak_reserved_bytes"],
            **profile,
        }
        rows.append(row)
    baselines = {
        (row["model"], row["workload"]): row["tok_s_median"]
        for row in rows
        if row["variant"] == "main"
    }
    for row in rows:
        baseline = baselines.get((row["model"], row["workload"]))
        row["tok_s_vs_main_percent"] = (
            (row["tok_s_median"] / baseline - 1) * 100 if baseline else None
        )
    return rows


def render_csv(rows, path):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows, path):
    lines = [
        "# nano-vLLM formal experiment report",
        "",
        "| model | workload | variant | runs | tok/s median | vs main | TPOT P50 | TPOT P95 | TPOT P99 | prepare P50 | peak GPU allocated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        versus = row["tok_s_vs_main_percent"]
        lines.append(
            "| {model} | {workload} | {variant} | {runs} | {tok:.1f} | {versus} | "
            "{p50:.3f} ms | {p95:.3f} ms | {p99:.3f} ms | {prepare} | {memory:.2f} GiB |".format(
                model=row["model"],
                workload=row["workload"],
                variant=row["variant"],
                runs=row["runs"],
                tok=row["tok_s_median"],
                versus=f"{versus:+.2f}%" if versus is not None else "n/a",
                p50=row["tpot_p50_ms"],
                p95=row["tpot_p95_ms"],
                p99=row["tpot_p99_ms"],
                prepare=(
                    f"{row['prepare_decode_median_ms']:.3f} ms"
                    if row.get("prepare_decode_median_ms") is not None
                    else "n/a"
                ),
                memory=row["peak_allocated_bytes"] / 2**30,
            )
        )
    lines.extend(
        [
            "",
            "Tie-break rows use the median of seven alternating one-run processes when the initial difference was within +/-1%. Other rows use the median of the three in-process measured runs.",
            "",
            "Nsight traces remain outside Git; only exported summaries and figures should be committed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def render_plots(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    variants = ["eager", "main", "pr176", "host-v2", "gpu-metadata"]
    groups = sorted({(row["model"], row["workload"]) for row in rows})
    saved = []
    for metric, label, filename in (
        ("tok_s_vs_main_percent", "Throughput vs main (%)", "throughput-vs-main.png"),
        ("tpot_p99_ms", "TPOT P99 (ms)", "tpot-p99.png"),
    ):
        figure, axes = plt.subplots(
            max(1, len(groups)), 1, figsize=(10, max(4, 3 * len(groups))), squeeze=False
        )
        for axis, group in zip(axes[:, 0], groups):
            group_rows = {row["variant"]: row for row in rows if (row["model"], row["workload"]) == group}
            names = [name for name in variants if name in group_rows]
            axis.bar(names, [group_rows[name][metric] for name in names])
            axis.set_title(f"{group[0]} / {group[1]}")
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved.append(path)
    return saved


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.results)
    if not rows:
        raise RuntimeError(f"no summaries found under {args.results}")
    render_csv(rows, args.output_dir / "formal-results.csv")
    render_markdown(rows, args.output_dir / "formal-report.md")
    render_plots(rows, args.output_dir)


if __name__ == "__main__":
    main()
