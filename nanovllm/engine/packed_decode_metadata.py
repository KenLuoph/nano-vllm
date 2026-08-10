import ctypes
import os
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from nanovllm.engine.gpu_block_table import BlockTableDelta
from nanovllm.engine.sequence import Sequence


HEADER_WORDS = 8
ROW_WORDS = 6
DELTA_WORDS = 3
MAGIC = 0x4E564D44  # "NVMD"


@cache
def _cuda_memcpy_api():
    """Resolve the CUDA runtime already loaded by PyTorch.

    Using the process-global handle avoids loading a second CUDA runtime with a
    potentially different minor version. The fallback supports environments
    that do not export runtime symbols globally.
    """
    runtime = ctypes.CDLL(None)
    try:
        memcpy_async = runtime.cudaMemcpyAsync
        error_string = runtime.cudaGetErrorString
    except AttributeError:
        runtime = ctypes.CDLL("libcudart.so")
        memcpy_async = runtime.cudaMemcpyAsync
        error_string = runtime.cudaGetErrorString
    memcpy_async.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    memcpy_async.restype = ctypes.c_int
    error_string.argtypes = (ctypes.c_int,)
    error_string.restype = ctypes.c_char_p
    return memcpy_async, error_string


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
        self.direct_h2d = (
            self.gpu_blob.is_cuda
            and os.environ.get("NANOVLLM_DIRECT_PACKED_H2D", "0") == "1"
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
                # nano-vLLM already represents slot mappings, context lengths,
                # and block-table coordinates as int32. Token IDs are bounded
                # by the model vocabulary and positions by max_model_len, so
                # the packed transport ABI can use int32 for all integer row
                # fields. Triton widens input_id/position when storing into the
                # captured graph's int64 tensors.
                "formats": ("<i4", "<i4", "<i4", "<i4", "<f4", "<i4"),
                "offsets": (0, 4, 8, 12, 16, 20),
                "itemsize": ROW_WORDS * 4,
            }
        )
        self.rows = np.ndarray(
            max_batch_size,
            dtype=self.row_dtype,
            buffer=self.cpu_bytes,
            offset=HEADER_WORDS * 4,
        )
        self.input_ids_np = self.rows["input_id"]
        self.positions_np = self.rows["position"]
        self.slot_mapping_np = self.rows["slot_mapping"]
        self.context_lens_np = self.rows["context_len"]
        self.temperatures_np = self.rows["temperature"]
        self.runtime_slots_np = self.rows["runtime_slot"]
        self.header_np = self.cpu_words[:HEADER_WORDS]
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
        # Only initialize captured padding lanes. Every active lane is overwritten
        # below, so clearing the complete graph bucket would be redundant CPU work.
        if batch_size < graph_bucket:
            padding = slice(batch_size, graph_bucket)
            self.input_ids_np[padding] = 0
            self.positions_np[padding] = 0
            self.slot_mapping_np[padding] = -1
            self.context_lens_np[padding] = 0
            self.temperatures_np[padding] = 1.0
            self.runtime_slots_np[padding] = -1

        active_block_width = 0
        for row, seq in enumerate(seqs):
            if seq.runtime_slot < 0:
                raise RuntimeError(f"sequence {seq.seq_id} has no runtime slot")
            sequence_length = len(seq)
            block_width = len(seq.block_table)
            if block_width > self.max_num_blocks:
                raise ValueError("active block width exceeds packed metadata capacity")
            active_block_width = max(active_block_width, block_width)
            self.input_ids_np[row] = seq.last_token
            self.positions_np[row] = sequence_length - 1
            self.slot_mapping_np[row] = (
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            self.context_lens_np[row] = sequence_length
            self.runtime_slots_np[row] = seq.runtime_slot
            if include_temperatures:
                self.temperatures_np[row] = seq.temperature

        delta_offset_words = HEADER_WORDS + graph_bucket * ROW_WORDS
        delta_words = self.cpu_words[
            delta_offset_words : delta_offset_words + len(deltas) * DELTA_WORDS
        ].reshape(-1, DELTA_WORDS)
        for row, delta in enumerate(deltas):
            delta_words[row] = (
                delta.runtime_slot,
                delta.column,
                delta.block_id,
            )

        active_words = delta_offset_words + len(deltas) * DELTA_WORDS
        self.header_np[1] = batch_size
        self.header_np[2] = graph_bucket
        self.header_np[3] = active_block_width
        self.header_np[4] = len(deltas)
        self.header_np[5] = delta_offset_words
        self.header_np[6] = active_words
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
        if not self.direct_h2d:
            self.gpu_blob[:active_bytes].copy_(
                self.cpu_blob[:active_bytes], non_blocking=self.gpu_blob.is_cuda
            )
            return

        # This is the only H2D submission in the packed decode path. Calling
        # cudaMemcpyAsync directly avoids constructing two tensor views and
        # entering PyTorch's generic copy dispatcher every generated token.
        memcpy_async, error_string = _cuda_memcpy_api()
        stream = torch.cuda.current_stream(self.gpu_blob.device).cuda_stream
        status = memcpy_async(
            self.gpu_blob.data_ptr(),
            self.cpu_blob.data_ptr(),
            active_bytes,
            1,  # cudaMemcpyHostToDevice
            stream,
        )
        if status:
            message = error_string(status)
            detail = message.decode() if message else f"CUDA error {status}"
            raise RuntimeError(f"packed metadata H2D failed: {detail}")
