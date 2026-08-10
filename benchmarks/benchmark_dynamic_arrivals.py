#!/usr/bin/env python3
import argparse
import atexit
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


DEFAULT_REQUESTS = (
    # Inject after N completed decode steps, prompt length, output length.
    (0, 128, 128),
    (0, 255, 96),
    (0, 512, 160),
    (0, 128, 192),
    (8, 128, 64),
    (24, 511, 96),
    (40, 256, 80),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark continuous batching with arrivals")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(values):
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def make_prompt(length, vocab_size, seed):
    rng = random.Random(seed)
    low = min(1000, vocab_size - 2)
    high = max(low + 1, vocab_size - 1)
    return [rng.randrange(low, high) for _ in range(length)]


def model_revision(model, config):
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return revision
    resolved = Path(model).expanduser().resolve()
    if "snapshots" in resolved.parts:
        index = resolved.parts.index("snapshots")
        if index + 1 < len(resolved.parts):
            return resolved.parts[index + 1]
    return f"local:{resolved}"


def git_revision():
    source_repo = os.environ.get("NANOVLLM_SOURCE_REPO")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=source_repo or Path(__file__).resolve().parents[1],
    ).strip()


def checksum(requests):
    digest = hashlib.sha256()
    for request in requests:
        for token_id in request["token_ids"]:
            digest.update(int(token_id).to_bytes(8, "little", signed=False))
        digest.update(b"\xffrequest-end\xff")
    return digest.hexdigest()


def main():
    args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES explicitly before benchmarking")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    max_model_len = max(prompt + output for _, prompt, output in DEFAULT_REQUESTS) + 16
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    pending = list(enumerate(DEFAULT_REQUESTS))
    requests_by_id = {}
    ordered_requests = []
    decode_steps = 0
    prefill_step_ms = []
    decode_step_ms = []
    token_event_ms = []
    started = time.perf_counter()
    profile_path = args.output.with_suffix(".steps.jsonl")
    llm.model_runner.call("start_profile", str(profile_path), args.variant, False)

    try:
        while pending or not llm.is_finished():
            while pending and pending[0][1][0] <= decode_steps:
                request_index, (_, prompt_tokens, output_tokens) = pending.pop(0)
                prompt = make_prompt(prompt_tokens, config.vocab_size, args.seed + request_index)
                llm.add_request(
                    prompt,
                    SamplingParams(
                        temperature=args.temperature,
                        max_tokens=output_tokens,
                        ignore_eos=True,
                    ),
                )
                seq = llm.scheduler.waiting[-1]
                record = {
                    "request_index": request_index,
                    "seq_id": seq.seq_id,
                    "arrival_decode_step": decode_steps,
                    "arrival_s": time.perf_counter() - started,
                    "prompt_tokens": prompt_tokens,
                    "requested_output_tokens": output_tokens,
                    "token_timestamps_s": [],
                    "token_ids": [],
                }
                requests_by_id[seq.seq_id] = record
                ordered_requests.append(record)

            seqs, is_prefill = llm.scheduler.schedule()
            before_counts = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
            step_started = time.perf_counter()
            token_ids = llm.model_runner.call("run", seqs, is_prefill)
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            step_ended = time.perf_counter()
            elapsed_ms = (step_ended - step_started) * 1000
            if is_prefill:
                prefill_step_ms.append(elapsed_ms)
            else:
                decode_steps += 1
                decode_step_ms.append(elapsed_ms)
                token_event_ms.extend([elapsed_ms] * len(seqs))

            for seq in seqs:
                if seq.num_completion_tokens <= before_counts[seq.seq_id]:
                    continue
                record = requests_by_id[seq.seq_id]
                record["token_timestamps_s"].append(step_ended - started)
                record["token_ids"].append(seq.last_token)

        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started
    finally:
        profile = llm.model_runner.call("stop_profile")
        atexit.unregister(llm.exit)
        llm.exit()

    all_inter_token_ms = []
    for request in ordered_requests:
        timestamps = request.pop("token_timestamps_s")
        arrival_s = request["arrival_s"]
        request["ttft_ms"] = (timestamps[0] - arrival_s) * 1000
        request["e2e_ms"] = (timestamps[-1] - arrival_s) * 1000
        intervals = [
            (current - previous) * 1000
            for previous, current in zip(timestamps, timestamps[1:])
        ]
        all_inter_token_ms.extend(intervals)
        request["tpot"] = summarize(intervals)
        request["output_tokens"] = len(request["token_ids"])

    total_output_tokens = sum(request["output_tokens"] for request in ordered_requests)
    result = {
        "variant": args.variant,
        "git_revision": git_revision(),
        "model_revision": model_revision(args.model, config),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "args": vars(args) | {"output": str(args.output)},
        "elapsed_s": elapsed_s,
        "output_tokens": total_output_tokens,
        "output_tokens_per_s": total_output_tokens / elapsed_s,
        "decode_steps": decode_steps,
        "prefill_step_latency": summarize(prefill_step_ms),
        "decode_step_latency": summarize(decode_step_ms),
        "token_event_tpot": summarize(token_event_ms),
        "request_inter_token_tpot": summarize(all_inter_token_ms),
        "token_checksum": checksum(ordered_requests),
        "requests": ordered_requests,
        "profile": profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
