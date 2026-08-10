#!/usr/bin/env python3
import argparse
import json
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
from nanovllm.engine.runtime_slots import RuntimeSlotManager
from nanovllm.engine.sequence import Sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate eager/captured packed metadata kernels and save Triton IR"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-num-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def make_sequence(length, token, blocks, temperature):
    seq = Sequence([token] * length)
    seq.block_table = list(blocks)
    seq.block_table_version = 1
    seq.temperature = temperature
    return seq


def graph_vars(max_batch_size, max_num_blocks):
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


def assert_equal(name, actual, expected):
    expected_tensor = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
    if not torch.equal(actual, expected_tensor):
        raise AssertionError(
            f"{name} mismatch: actual={actual.cpu().tolist()} expected={expected}"
        )


def save_asm(output_dir, name, compiled):
    saved = {}
    for stage, content in compiled.asm.items():
        if isinstance(content, str):
            path = output_dir / f"{name}.{stage}"
            path.write_text(content)
            saved[stage] = str(path)
    return saved


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Sequence.block_size = args.block_size

    packed = PackedDecodeMetadata(
        args.max_batch_size,
        args.max_num_blocks,
        args.block_size,
        device="cuda",
    )
    mirror = GpuBlockTableMirror(
        args.max_batch_size,
        args.max_num_blocks,
        device="cuda",
    )
    slots = RuntimeSlotManager(args.max_batch_size)
    tensors = graph_vars(args.max_batch_size, args.max_num_blocks)
    seqs = [
        make_sequence(257, 101, [5, 19], 0.5),
        make_sequence(513, 102, [7, 11, 13], 0.8),
        make_sequence(128, 103, [23], 1.1),
    ]
    for seq in seqs:
        slots.acquire(seq)

    deltas = mirror.plan_deltas(seqs)
    initial_delta_count = len(deltas)
    view = packed.prepare(seqs, 4, deltas, include_temperatures=True)
    packed.copy_to_gpu(view.active_bytes)

    block = 256
    apply_grid = (triton.cdiv(packed.max_deltas, block),)
    apply_compiled = apply_block_table_deltas_kernel[apply_grid](
        packed.gpu_words,
        mirror.master_block_tables,
        MAX_NUM_BLOCKS=packed.max_num_blocks,
        MAX_DELTAS=packed.max_deltas,
        BLOCK=block,
    )
    unpack_grid = (triton.cdiv(4 * packed.max_num_blocks, block),)
    unpack_compiled = unpack_decode_metadata_kernel[unpack_grid](
        packed.gpu_words,
        packed.gpu_blob.view(torch.float32),
        mirror.master_block_tables,
        tensors["input_ids"],
        tensors["positions"],
        tensors["slot_mapping"],
        tensors["context_lens"],
        packed.temperatures_gpu,
        tensors["block_tables"],
        GRAPH_BUCKET=4,
        MAX_NUM_BLOCKS=packed.max_num_blocks,
        BLOCK=block,
    )
    torch.cuda.synchronize()

    assert_equal("input_ids", tensors["input_ids"][:4], [101, 102, 103, 0])
    assert_equal("positions", tensors["positions"][:4], [256, 512, 127, 0])
    assert_equal(
        "slot_mapping",
        tensors["slot_mapping"][:4],
        [19 * args.block_size, 13 * args.block_size, 23 * args.block_size + 127, -1],
    )
    assert_equal("context_lens", tensors["context_lens"][:4], [257, 513, 128, 0])
    assert_equal(
        "block_tables",
        tensors["block_tables"][:4, :3],
        [[5, 19, -1], [7, 11, 13], [23, -1, -1], [-1, -1, -1]],
    )

    graph = torch.cuda.CUDAGraph()
    launch_packed_decode_kernels(packed, mirror, tensors, 4)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch_packed_decode_kernels(packed, mirror, tensors, 4)

    for token_id in range(200, 455):
        seqs[0].append_token(token_id)
    seqs[0].append_token(104)
    seqs[0].block_table.append(29)
    seqs[0].block_table_version += 1
    deltas = mirror.plan_deltas(seqs)
    view = packed.prepare(seqs, 4, deltas, include_temperatures=True)
    packed.copy_to_gpu(view.active_bytes)
    graph.replay()
    torch.cuda.synchronize()
    assert_equal("captured_input_ids", tensors["input_ids"][:4], [104, 102, 103, 0])
    assert_equal(
        "captured_block_tables",
        tensors["block_tables"][:4, :3],
        [[5, 19, 29], [7, 11, 13], [23, -1, -1], [-1, -1, -1]],
    )

    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "initial_deltas": initial_delta_count,
        "incremental_deltas": len(deltas),
        "active_bytes": view.active_bytes,
        "eager_and_captured_checks": "passed",
        "asm": {
            "apply_deltas": save_asm(args.output_dir, "apply_deltas", apply_compiled),
            "unpack_gather": save_asm(args.output_dir, "unpack_gather", unpack_compiled),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
