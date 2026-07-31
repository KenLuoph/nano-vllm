#!/usr/bin/env python3
import argparse
import atexit
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Profile nano-vLLM's decode host hot path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-output-tokens", type=int, default=32)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--nvtx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    return parser.parse_args()


def make_prompts(batch_size: int, prompt_tokens: int, vocab_size: int, seed: int):
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    rng = random.Random(seed)
    low = min(1000, vocab_size - 2)
    high = max(low + 1, vocab_size - 1)
    return [
        [rng.randrange(low, high) for _ in range(prompt_tokens)]
        for _ in range(batch_size)
    ]


def git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()


def run_generation(llm, args, output_tokens: int, run_seed: int):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    prompts = make_prompts(args.batch_size, args.prompt_tokens, config.vocab_size, run_seed)
    torch.manual_seed(run_seed)
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=output_tokens,
        ignore_eos=True,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    generated = sum(len(output["token_ids"]) for output in outputs)
    digest = hashlib.sha256()
    for output in outputs:
        for token_id in output["token_ids"]:
            digest.update(int(token_id).to_bytes(8, "little", signed=False))
    return {
        "elapsed_s": elapsed_s,
        "output_tokens": generated,
        "output_tokens_per_s": generated / elapsed_s,
        "token_checksum": digest.hexdigest(),
    }


def main():
    args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES explicitly before benchmarking")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / f"{args.variant}.steps.jsonl"
    summary_path = output_dir / f"{args.variant}.summary.json"

    max_model_len = args.prompt_tokens + args.output_tokens + 16
    max_num_batched_tokens = max(4096, args.batch_size * args.prompt_tokens)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=max(8, args.batch_size),
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    warmup = run_generation(llm, args, args.warmup_output_tokens, args.seed - 1)
    llm.model_runner.call("start_profile", str(profile_path), args.variant, args.nvtx)

    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()

    measured = []
    try:
        for run_index in range(args.measured_runs):
            measured.append(
                run_generation(llm, args, args.output_tokens, args.seed + run_index)
            )
    finally:
        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
        profile_result = llm.model_runner.call("stop_profile")

    total_elapsed = sum(item["elapsed_s"] for item in measured)
    total_tokens = sum(item["output_tokens"] for item in measured)
    summary = {
        "variant": args.variant,
        "git_revision": git_revision(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "args": vars(args),
        "warmup": warmup,
        "measured": measured,
        "aggregate": {
            "elapsed_s": total_elapsed,
            "output_tokens": total_tokens,
            "output_tokens_per_s": total_tokens / total_elapsed,
        },
        "profile": profile_result,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    atexit.unregister(llm.exit)
    llm.exit()


if __name__ == "__main__":
    main()
