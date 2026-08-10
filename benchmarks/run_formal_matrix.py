#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reproducible nano-vLLM matrix")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_text(command, cwd=None, env=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        return run_text(command).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def repo_revision(repo):
    return run_text(["git", "rev-parse", "HEAD"], cwd=repo)


def summary_path(output_dir, variant):
    return output_dir / f"{variant}.summary.json"


def build_command(config, variant, model, workload, output_dir, measured_runs, seed):
    repo = Path(variant["repo"])
    benchmark = repo / "benchmarks" / "benchmark_decode_hotpath.py"
    command = [
        config["python"],
        str(benchmark),
        "--model",
        model["path"],
        "--output-dir",
        str(output_dir),
        "--variant",
        variant["name"],
        "--batch-size",
        str(workload["batch_size"]),
        "--prompt-tokens",
        str(workload["prompt_tokens"]),
        "--output-tokens",
        str(workload["output_tokens"]),
        "--measured-runs",
        str(measured_runs),
        "--seed",
        str(seed),
        "--gpu-memory-utilization",
        str(config.get("gpu_memory_utilization", 0.8)),
        "--cuda-profiler-range",
    ]
    if variant.get("enforce_eager", False):
        command.append("--enforce-eager")
    if not workload.get("nvtx", True):
        command.append("--no-nvtx")
    return command


def execute_case(config, variant, model, workload, output_dir, measured_runs, seed, force):
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_summary = summary_path(output_dir, variant["name"])
    if expected_summary.exists() and not force:
        return json.loads(expected_summary.read_text())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    env["PYTHONPATH"] = str(Path(variant["repo"]).resolve())
    env.update({key: str(value) for key, value in variant.get("env", {}).items()})
    command = build_command(
        config, variant, model, workload, output_dir, measured_runs, seed
    )
    nsys_output = output_dir / "trace"
    if workload.get("nsight", False):
        command = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--output",
            str(nsys_output),
            *command,
        ]

    metadata = {
        "variant": variant["name"],
        "repo": str(Path(variant["repo"]).resolve()),
        "git_revision": repo_revision(variant["repo"]),
        "model": model,
        "workload": workload,
        "command": command,
        "gpu_processes_before": gpu_snapshot(),
        "started_unix_s": time.time(),
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    completed = subprocess.run(
        command,
        cwd=variant["repo"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output_dir / "stdout.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant['name']} {model['name']} {workload['name']} failed; "
            f"see {output_dir / 'stdout.log'}"
        )
    return json.loads(expected_summary.read_text())


def throughput(summary):
    return summary["aggregate"]["output_tokens_per_s"]


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    variants = {variant["name"]: variant for variant in config["variants"]}
    initial = {}
    for model in config["models"]:
        allowed = set(model.get("workloads", []))
        for workload in config["workloads"]:
            if allowed and workload["name"] not in allowed:
                continue
            for variant in config["variants"]:
                output_dir = (
                    output_root
                    / model["name"]
                    / workload["name"]
                    / variant["name"]
                    / "initial"
                )
                summary = execute_case(
                    config,
                    variant,
                    model,
                    workload,
                    output_dir,
                    config.get("measured_runs", 3),
                    config.get("seed", 20260810),
                    args.force,
                )
                initial[(model["name"], workload["name"], variant["name"])] = summary

    baseline_name = config.get("tie_break_baseline", "main")
    threshold = config.get("tie_break_threshold_percent", 1.0)
    rounds = config.get("tie_break_rounds", 7)
    candidates = config.get("tie_break_candidates", [])
    for model in config["models"]:
        allowed = set(model.get("workloads", []))
        for workload in config["workloads"]:
            if allowed and workload["name"] not in allowed:
                continue
            baseline = initial[(model["name"], workload["name"], baseline_name)]
            for candidate_name in candidates:
                candidate = initial[(model["name"], workload["name"], candidate_name)]
                difference = (
                    throughput(candidate) / throughput(baseline) - 1.0
                ) * 100
                if abs(difference) > threshold:
                    continue
                for round_index in range(rounds):
                    order = (
                        [baseline_name, candidate_name]
                        if round_index % 2 == 0
                        else [candidate_name, baseline_name]
                    )
                    for variant_name in order:
                        output_dir = (
                            output_root
                            / model["name"]
                            / workload["name"]
                            / variant_name
                            / "tie-break"
                            / f"round-{round_index:02d}"
                        )
                        execute_case(
                            config,
                            variants[variant_name],
                            model,
                            workload,
                            output_dir,
                            measured_runs=1,
                            seed=config.get("seed", 20260810) + 1000 + round_index,
                            force=args.force,
                        )


if __name__ == "__main__":
    main()
