#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path
from time import perf_counter_ns

import torch

from nanovllm.engine.decode_input_batch import DecodeInputBatch


BATCH_SIZES = (1, 4, 8, 16, 32, 64)
BLOCK_WIDTHS = (1, 4, 16, 64)
BLOCK_SIZE = 256


class BenchmarkSequence:
    def __init__(self, row: int, block_width: int):
        self.num_tokens = block_width * BLOCK_SIZE - (row % BLOCK_SIZE)
        self.last_token = 1000 + row
        self.block_table = [row * block_width + column for column in range(block_width)]
        self.temperature = 0.5 + (row % 5) * 0.1

    def __len__(self):
        return self.num_tokens

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - ((self.num_tokens - 1) // BLOCK_SIZE) * BLOCK_SIZE


def percentile(values: list[float], fraction: float):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(samples_ns: list[int]):
    samples_us = [sample / 1e3 for sample in samples_ns]
    return {
        "mean_us": statistics.fmean(samples_us),
        "median_us": statistics.median(samples_us),
        "p95_us": percentile(samples_us, 0.95),
    }


def benchmark(function, warmup: int, iterations: int):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        started = perf_counter_ns()
        function()
        samples.append(perf_counter_ns() - started)
    return summarize(samples)


def make_main_packer(seqs, pin_memory: bool):
    def pack():
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        temperatures = []
        max_width = max(len(seq.block_table) for seq in seqs)
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
            )
            temperatures.append(seq.temperature)
            block_tables.append(seq.block_table + [-1] * (max_width - len(seq.block_table)))
        return (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=pin_memory),
            torch.tensor(positions, dtype=torch.int64, pin_memory=pin_memory),
            torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=pin_memory),
            torch.tensor(context_lens, dtype=torch.int32, pin_memory=pin_memory),
            torch.tensor(block_tables, dtype=torch.int32, pin_memory=pin_memory),
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=pin_memory),
        )

    return pack


def make_pr176_packer(seqs, batch_size: int, block_width: int, pin_memory: bool):
    factory = {"device": "cpu", "pin_memory": pin_memory}
    staging = {
        "input_ids": torch.empty(batch_size, dtype=torch.int64, **factory),
        "positions": torch.empty(batch_size, dtype=torch.int64, **factory),
        "slot_mapping": torch.empty(batch_size, dtype=torch.int32, **factory),
        "context_lens": torch.empty(batch_size, dtype=torch.int32, **factory),
        "block_tables": torch.empty(batch_size, block_width, dtype=torch.int32, **factory),
    }

    def pack():
        staging["input_ids"].zero_()
        staging["positions"].zero_()
        staging["slot_mapping"].fill_(-1)
        staging["context_lens"].fill_(1)
        staging["block_tables"].fill_(-1)
        for row, seq in enumerate(seqs):
            staging["input_ids"][row] = seq.last_token
            staging["positions"][row] = len(seq) - 1
            staging["context_lens"][row] = len(seq)
            staging["slot_mapping"][row] = (
                seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
            )
            for column, block_id in enumerate(seq.block_table):
                staging["block_tables"][row, column] = block_id
        # PR #176 still used the original per-step temperature allocation.
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=pin_memory
        )
        return staging, temperatures

    return pack


def make_v2_packer(seqs, batch_size: int, block_width: int, pin_memory: bool):
    decode_batch = DecodeInputBatch(
        batch_size,
        block_width,
        BLOCK_SIZE,
        pin_memory=pin_memory,
        allocate_temperature_gpu=False,
    )

    def pack():
        return decode_batch.prepare_decode(seqs, batch_size, include_temperatures=True)

    return pack


def parse_args():
    parser = argparse.ArgumentParser(description="Compare decode CPU metadata packing paths")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for batch_size in BATCH_SIZES:
        for block_width in BLOCK_WIDTHS:
            seqs = [BenchmarkSequence(row, block_width) for row in range(batch_size)]
            packers = {
                "main": make_main_packer(seqs, args.pin_memory),
                "pr176": make_pr176_packer(
                    seqs, batch_size, block_width, args.pin_memory
                ),
                "v2_numpy": make_v2_packer(
                    seqs, batch_size, block_width, args.pin_memory
                ),
            }
            for variant, packer in packers.items():
                result = {
                    "variant": variant,
                    "batch_size": batch_size,
                    "block_width": block_width,
                    "iterations": args.iterations,
                    "pin_memory": args.pin_memory,
                }
                result.update(benchmark(packer, args.warmup, args.iterations))
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    output = {
        "torch": torch.__version__,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "pin_memory": args.pin_memory,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
