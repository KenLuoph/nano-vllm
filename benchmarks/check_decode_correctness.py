#!/usr/bin/env python3
import argparse
import atexit
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


FIXED_CASES = (
    ("batch1_prompt128", 1, 128, 64),
    ("batch4_prompt128", 4, 128, 64),
    ("batch8_prompt128", 8, 128, 64),
    ("batch16_prompt128", 16, 128, 64),
    ("batch4_prompt1024", 4, 1024, 64),
    ("batch4_prompt255", 4, 255, 64),
    ("batch4_prompt256", 4, 256, 64),
    ("batch4_prompt511", 4, 511, 64),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run deterministic decode correctness cases")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def make_prompts(batch_size: int, prompt_tokens: int, vocab_size: int, seed: int):
    rng = random.Random(seed)
    low = min(1000, vocab_size - 2)
    high = max(low + 1, vocab_size - 1)
    return [
        [rng.randrange(low, high) for _ in range(prompt_tokens)]
        for _ in range(batch_size)
    ]


def token_checksum(outputs):
    digest = hashlib.sha256()
    for output in outputs:
        for token_id in output:
            digest.update(int(token_id).to_bytes(8, "little", signed=False))
        digest.update(b"\xffrequest-end\xff")
    return digest.hexdigest()


def run_fixed_cases(llm, args, vocab_size: int):
    results = []
    for case_index, (name, batch_size, prompt_tokens, output_tokens) in enumerate(FIXED_CASES):
        case_seed = args.seed + case_index
        prompts = make_prompts(batch_size, prompt_tokens, vocab_size, case_seed)
        torch.manual_seed(case_seed)
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        generated = llm.generate(prompts, sampling, use_tqdm=False)
        token_ids = [item["token_ids"] for item in generated]
        results.append(
            {
                "name": name,
                "batch_size": batch_size,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "token_counts": [len(tokens) for tokens in token_ids],
                "token_checksum": token_checksum(token_ids),
            }
        )
    return results


def run_churn_case(llm, args, vocab_size: int):
    case_seed = args.seed + 100
    prompts = make_prompts(5, 128, vocab_size, case_seed)
    max_tokens = (17, 33, 65, 129)
    request_order = []
    for prompt, limit in zip(prompts[:4], max_tokens):
        llm.add_request(
            prompt,
            SamplingParams(
                temperature=args.temperature,
                max_tokens=limit,
                ignore_eos=True,
            ),
        )
        request_order.append(llm.scheduler.waiting[-1].seq_id)

    profile_path = args.output.with_name(f"{args.output.stem}.churn.steps.jsonl")
    llm.model_runner.call("start_profile", str(profile_path), f"{args.variant}-churn", False)
    torch.manual_seed(case_seed)
    outputs = {}
    injected = False
    while not llm.is_finished():
        finished, _ = llm.step()
        for seq_id, token_ids in finished:
            outputs[seq_id] = token_ids
        if finished and not injected:
            llm.add_request(
                prompts[4],
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=41,
                    ignore_eos=True,
                ),
            )
            request_order.append(llm.scheduler.waiting[-1].seq_id)
            injected = True
    llm.model_runner.call("stop_profile")

    ordered_outputs = [outputs[seq_id] for seq_id in request_order]
    return {
        "name": "dynamic_churn",
        "max_tokens": [17, 33, 65, 129, 41],
        "injected_after_first_completion": injected,
        "token_counts": [len(tokens) for tokens in ordered_outputs],
        "token_checksum": token_checksum(ordered_outputs),
        "profile": str(profile_path),
    }


def main():
    args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES explicitly")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_model_len=1280,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        fixed = run_fixed_cases(llm, args, config.vocab_size)
        churn = run_churn_case(llm, args, config.vocab_size)
        result = {"variant": args.variant, "fixed": fixed, "churn": churn}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
