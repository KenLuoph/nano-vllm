import torch
import triton
import triton.language as tl

from nanovllm.engine.gpu_block_table import GpuBlockTableMirror
from nanovllm.engine.packed_decode_metadata import (
    DELTA_WORDS,
    HEADER_WORDS,
    ROW_WORDS,
    PackedDecodeMetadata,
)


@triton.jit
def apply_block_table_deltas_kernel(
    blob_words,
    master_block_tables,
    MAX_NUM_BLOCKS: tl.constexpr,
    MAX_DELTAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    delta_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num_deltas = tl.load(blob_words + 4)
    delta_offset = tl.load(blob_words + 5)
    valid = (delta_index < num_deltas) & (delta_index < MAX_DELTAS)
    record = delta_offset + delta_index * DELTA_WORDS
    runtime_slot = tl.load(blob_words + record, mask=valid, other=0)
    column = tl.load(blob_words + record + 1, mask=valid, other=0)
    block_id = tl.load(blob_words + record + 2, mask=valid, other=-1)
    destination = runtime_slot * MAX_NUM_BLOCKS + column
    tl.store(master_block_tables + destination, block_id, mask=valid)


@triton.jit
def unpack_decode_metadata_kernel(
    blob_words,
    blob_i64,
    blob_f32,
    master_block_tables,
    input_ids,
    positions,
    slot_mapping,
    context_lens,
    temperatures,
    block_tables,
    GRAPH_BUCKET: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = element // MAX_NUM_BLOCKS
    column = element % MAX_NUM_BLOCKS
    in_bucket = row < GRAPH_BUCKET
    batch_size = tl.load(blob_words + 1)
    active_width = tl.load(blob_words + 3)
    row_is_active = in_bucket & (row < batch_size)
    row_word = HEADER_WORDS + row * ROW_WORDS

    runtime_slot = tl.load(blob_words + row_word + 7, mask=in_bucket, other=-1)
    safe_slot = tl.maximum(runtime_slot, 0)
    table_is_active = row_is_active & (column < active_width) & (runtime_slot >= 0)
    block_id = tl.load(
        master_block_tables + safe_slot * MAX_NUM_BLOCKS + column,
        mask=table_is_active,
        other=-1,
    )
    tl.store(
        block_tables + row * MAX_NUM_BLOCKS + column,
        block_id,
        mask=in_bucket & (column < active_width),
    )

    vector_lane = in_bucket & (column == 0)
    input_id = tl.load(blob_i64 + row_word // 2, mask=vector_lane, other=0)
    position = tl.load(blob_i64 + (row_word + 2) // 2, mask=vector_lane, other=0)
    slot = tl.load(blob_words + row_word + 4, mask=vector_lane, other=-1)
    context = tl.load(blob_words + row_word + 5, mask=vector_lane, other=0)
    temperature = tl.load(blob_f32 + row_word + 6, mask=vector_lane, other=1.0)
    tl.store(input_ids + row, input_id, mask=vector_lane)
    tl.store(positions + row, position, mask=vector_lane)
    tl.store(slot_mapping + row, slot, mask=vector_lane)
    tl.store(context_lens + row, context, mask=vector_lane)
    tl.store(temperatures + row, temperature, mask=vector_lane)


def launch_packed_decode_kernels(
    packed: PackedDecodeMetadata,
    mirror: GpuBlockTableMirror,
    graph_vars: dict[str, torch.Tensor],
    graph_bucket: int,
) -> None:
    block = 256
    apply_grid = (triton.cdiv(packed.max_deltas, block),)
    apply_block_table_deltas_kernel[apply_grid](
        packed.gpu_words,
        mirror.master_block_tables,
        MAX_NUM_BLOCKS=packed.max_num_blocks,
        MAX_DELTAS=packed.max_deltas,
        BLOCK=block,
    )
    unpack_grid = (triton.cdiv(graph_bucket * packed.max_num_blocks, block),)
    unpack_decode_metadata_kernel[unpack_grid](
        packed.gpu_words,
        packed.gpu_blob.view(torch.int64),
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
