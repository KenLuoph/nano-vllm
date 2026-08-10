from dataclasses import dataclass

import numpy as np
import torch

from nanovllm.engine.gpu_block_table import BlockTableDelta
from nanovllm.engine.sequence import Sequence


HEADER_WORDS = 8
ROW_WORDS = 8
DELTA_WORDS = 4
MAGIC = 0x4E564D44  # "NVMD"


@dataclass(frozen=True, slots=True)
class PackedDecodeView:
    batch_size: int
    graph_bucket: int
    active_block_width: int
    num_deltas: int
    delta_offset_words: int
    active_bytes: int
    padding_rows: int


class PackedDecodeMetadata:
    """One persistent, packed H2D payload for CUDA Graph decode metadata."""

    def __init__(
        self,
        max_batch_size: int,
        max_num_blocks: int,
        block_size: int,
        *,
        pin_memory: bool = True,
        device: torch.device | str = "cuda",
    ):
        if max_batch_size < 1 or max_num_blocks < 1 or block_size < 1:
            raise ValueError("packed metadata dimensions must be positive")
        self.max_batch_size = max_batch_size
        self.max_num_blocks = max_num_blocks
        self.max_deltas = max_batch_size * max_num_blocks
        self.block_size = block_size
        self.capacity_words = (
            HEADER_WORDS
            + max_batch_size * ROW_WORDS
            + self.max_deltas * DELTA_WORDS
        )
        self.cpu_blob = torch.zeros(
            self.capacity_words * 4,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.gpu_blob = torch.zeros(
            self.capacity_words * 4,
            dtype=torch.uint8,
            device=device,
        )
        self.gpu_words = self.gpu_blob.view(torch.int32)
        self.temperatures_gpu = torch.empty(
            max_batch_size, dtype=torch.float32, device=device
        )

        self.cpu_bytes = self.cpu_blob.numpy()
        self.cpu_words = self.cpu_bytes.view(np.int32)
        self.row_dtype = np.dtype(
            {
                "names": (
                    "input_id",
                    "position",
                    "slot_mapping",
                    "context_len",
                    "temperature",
                    "runtime_slot",
                ),
                "formats": ("<i8", "<i8", "<i4", "<i4", "<f4", "<i4"),
                "offsets": (0, 8, 16, 20, 24, 28),
                "itemsize": ROW_WORDS * 4,
            }
        )
        self.rows = np.ndarray(
            max_batch_size,
            dtype=self.row_dtype,
            buffer=self.cpu_bytes,
            offset=HEADER_WORDS * 4,
        )
        self.last_view: PackedDecodeView | None = None
        self._initialize_safe_blob()

    def _initialize_safe_blob(self):
        self.cpu_words[:] = 0
        self.cpu_words[0] = MAGIC
        self.rows["slot_mapping"] = -1
        self.rows["runtime_slot"] = -1

    def prepare(
        self,
        seqs: list[Sequence],
        graph_bucket: int,
        deltas: list[BlockTableDelta],
        include_temperatures: bool,
    ) -> PackedDecodeView:
        batch_size = len(seqs)
        if batch_size < 1 or batch_size > graph_bucket:
            raise ValueError("invalid packed decode batch")
        if graph_bucket > self.max_batch_size:
            raise ValueError("graph bucket exceeds packed metadata capacity")
        if len(deltas) > self.max_deltas:
            raise ValueError("block-table delta capacity exceeded")
        active_block_width = max(len(seq.block_table) for seq in seqs)
        if not 0 < active_block_width <= self.max_num_blocks:
            raise ValueError("active block width exceeds packed metadata capacity")

        active_rows = self.rows[:graph_bucket]
        active_rows["input_id"] = 0
        active_rows["position"] = 0
        active_rows["slot_mapping"] = -1
        active_rows["context_len"] = 0
        active_rows["temperature"] = 1.0
        active_rows["runtime_slot"] = -1
        for row, seq in enumerate(seqs):
            if seq.runtime_slot < 0:
                raise RuntimeError(f"sequence {seq.seq_id} has no runtime slot")
            active_rows["input_id"][row] = seq.last_token
            active_rows["position"][row] = len(seq) - 1
            active_rows["slot_mapping"][row] = (
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            active_rows["context_len"][row] = len(seq)
            active_rows["runtime_slot"][row] = seq.runtime_slot
            if include_temperatures:
                active_rows["temperature"][row] = seq.temperature

        delta_offset_words = HEADER_WORDS + graph_bucket * ROW_WORDS
        delta_words = self.cpu_words[
            delta_offset_words : delta_offset_words + len(deltas) * DELTA_WORDS
        ].reshape(-1, DELTA_WORDS)
        for row, delta in enumerate(deltas):
            delta_words[row] = (
                delta.runtime_slot,
                delta.column,
                delta.block_id,
                delta.epoch,
            )

        active_words = delta_offset_words + len(deltas) * DELTA_WORDS
        self.cpu_words[:HEADER_WORDS] = (
            MAGIC,
            batch_size,
            graph_bucket,
            active_block_width,
            len(deltas),
            delta_offset_words,
            active_words,
            0,
        )
        view = PackedDecodeView(
            batch_size=batch_size,
            graph_bucket=graph_bucket,
            active_block_width=active_block_width,
            num_deltas=len(deltas),
            delta_offset_words=delta_offset_words,
            active_bytes=active_words * 4,
            padding_rows=graph_bucket - batch_size,
        )
        self.last_view = view
        return view

    def copy_to_gpu(self, active_bytes: int) -> None:
        if active_bytes < HEADER_WORDS * 4 or active_bytes > self.cpu_blob.numel():
            raise ValueError("invalid packed metadata copy size")
        self.gpu_blob[:active_bytes].copy_(
            self.cpu_blob[:active_bytes], non_blocking=True
        )
