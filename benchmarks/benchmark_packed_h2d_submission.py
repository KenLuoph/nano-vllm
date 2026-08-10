#!/usr/bin/env python3
import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch copy_ with direct cudaMemcpyAsync submission"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--copies-per-sample", type=int, default=100)
    parser.add_argument("--bytes", type=int, nargs="+", default=[140, 416, 12320])
    return parser.parse_args()


def cuda_memcpy():
    runtime = ctypes.CDLL(None)
    function = runtime.cudaMemcpyAsync
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    return function


def summarize(samples):
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "p95_us": sorted(samples)[int(len(samples) * 0.95) - 1],
        "min_us": min(samples),
        "max_us": max(samples),
    }


def measure(callable_, iterations, copies_per_sample):
    for _ in range(copies_per_sample):
        callable_()
    torch.cuda.synchronize()
    samples = []
    remaining = iterations
    while remaining:
        count = min(copies_per_sample, remaining)
        started = time.perf_counter_ns()
        for _ in range(count):
            callable_()
        elapsed = time.perf_counter_ns() - started
        torch.cuda.synchronize()
        samples.append(elapsed / count / 1000)
        remaining -= count
    return summarize(samples)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capacity = max(args.bytes)
    cpu = torch.arange(capacity, dtype=torch.uint8, pin_memory=True)
    gpu = torch.zeros(capacity, dtype=torch.uint8, device="cuda")
    raw_copy = cuda_memcpy()
    stream = torch.cuda.current_stream().cuda_stream
    results = []
    for active_bytes in args.bytes:
        def pytorch_copy():
            gpu[:active_bytes].copy_(cpu[:active_bytes], non_blocking=True)

        def direct_copy():
            status = raw_copy(
                gpu.data_ptr(), cpu.data_ptr(), active_bytes, 1, stream
            )
            if status:
                raise RuntimeError(f"cudaMemcpyAsync failed with status {status}")

        pytorch = measure(pytorch_copy, args.iterations, args.copies_per_sample)
        direct = measure(direct_copy, args.iterations, args.copies_per_sample)
        direct_copy()
        torch.cuda.synchronize()
        if not torch.equal(gpu[:active_bytes].cpu(), cpu[:active_bytes]):
            raise AssertionError("direct copy produced incorrect bytes")
        results.append(
            {
                "active_bytes": active_bytes,
                "pytorch_copy": pytorch,
                "direct_cuda_memcpy_async": direct,
                "median_reduction_percent":
                    (1 - direct["median_us"] / pytorch["median_us"]) * 100,
            }
        )

    output = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "iterations": args.iterations,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
