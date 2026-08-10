#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from nanovllm.engine.gpu_block_table import GpuBlockTableMirror
from nanovllm.engine.packed_decode_kernels import (
    apply_block_table_deltas_kernel,
    launch_packed_decode_kernels,
    unpack_decode_metadata_kernel,
)
from nanovllm.engine.packed_decode_metadata import PackedDecodeMetadata


def parse_args():
    parser = argparse.ArgumentParser(description="Measure packed metadata kernel cost")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-num-blocks", type=int, default=16)
    return parser.parse_args()


def make_graph_vars(max_batch_size, max_num_blocks):
    return {
        "input_ids": torch.zeros(max_batch_size, dtype=torch.int64, device="cuda"),
        "positions": torch.zeros(max_batch_size, dtype=torch.int64, device="cuda"),
        "slot_mapping": torch.full(
            (max_batch_size,), -1, dtype=torch.int32, device="cuda"
        ),
        "context_lens": torch.zeros(max_batch_size, dtype=torch.int32, device="cuda"),
        "block_tables": torch.full(
            (max_batch_size, max_num_blocks),
            -1,
            dtype=torch.int32,
            device="cuda",
        ),
    }


def launch_unpack(packed, mirror, graph_vars, graph_bucket):
    block = 256
    grid = (triton.cdiv(graph_bucket * packed.max_num_blocks, block),)
    unpack_decode_metadata_kernel[grid](
        packed.gpu_words,
        packed.gpu_blob.view(torch.float32),
        mirror.master_block_tables,
        graph_vars["input_ids"],
        graph_vars["positions"],
        graph_vars["slot_mapping"],
        graph_vars["context_lens"],
        packed.temperatures_gpu,
        graph_vars["block_tables"],
        GRAPH_BUCKET=graph_bucket,
        MAX_NUM_BLOCKS=packed.max_num_blocks,
        BLOCK=block,
    )


def capture(callable_):
    callable_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        callable_()
    return graph


def event_measure(callable_, iterations, warmup):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations // 10):
            callable_()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (iterations // 10))
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def main():
    args = parse_args()
    packed = PackedDecodeMetadata(
        args.max_batch_size,
        args.max_num_blocks,
        256,
        device="cuda",
    )
    mirror = GpuBlockTableMirror(
        args.max_batch_size,
        args.max_num_blocks,
        device="cuda",
    )
    graph_vars = make_graph_vars(args.max_batch_size, args.max_num_blocks)
    # Header with zero active rows/deltas is sufficient to benchmark masked work.
    packed.gpu_blob.copy_(packed.cpu_blob)
    results = []
    for bucket in (1, 4, 8, 16):
        if bucket > args.max_batch_size:
            continue
        full = lambda: launch_packed_decode_kernels(packed, mirror, graph_vars, bucket)
        unpack = lambda: launch_unpack(packed, mirror, graph_vars, bucket)
        full_graph = capture(full)
        unpack_graph = capture(unpack)
        results.append(
            {
                "graph_bucket": bucket,
                "max_num_blocks": args.max_num_blocks,
                "captured_apply_and_unpack": event_measure(
                    full_graph.replay, args.iterations, args.warmup
                ),
                "captured_unpack_only": event_measure(
                    unpack_graph.replay, args.iterations, args.warmup
                ),
                "eager_apply_and_unpack": event_measure(
                    full, args.iterations, args.warmup
                ),
                "eager_unpack_only": event_measure(
                    unpack, args.iterations, args.warmup
                ),
            }
        )
    output = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
