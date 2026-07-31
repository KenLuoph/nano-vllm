import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter_ns

import torch


class DecodeProfiler:
    """Optional low-overhead CPU timers and NVTX ranges for decode steps.

    Profiling is disabled during model warmup and CUDA graph capture. Benchmarks
    explicitly call ``start`` after warmup, then ``stop`` before engine teardown.
    CPU timers measure host work and CUDA API submission, not GPU completion.
    Nsight Systems is the source of truth for GPU duration and copy volume.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.enabled = False
        self.nvtx_enabled = False
        self.output_path: Path | None = None
        self.variant = "unknown"
        self.records: list[dict] = []
        self.current: dict | None = None
        self.next_step = 0

    def start(self, output_path: str, variant: str, nvtx: bool = True):
        if self.enabled:
            raise RuntimeError("decode profiler is already running")
        self.output_path = Path(output_path)
        self.variant = variant
        self.nvtx_enabled = nvtx
        self.records.clear()
        self.current = None
        self.next_step = 0
        self.enabled = True
        return {
            "output_path": str(self.output_path),
            "variant": self.variant,
            "nvtx": self.nvtx_enabled,
        }

    def stop(self):
        if not self.enabled:
            return {"output_path": None, "num_records": 0}
        if self.current is not None:
            raise RuntimeError("cannot stop decode profiler in the middle of a step")
        self.enabled = False
        path = self._flush()
        return {"output_path": str(path), "num_records": len(self.records)}

    def begin_step(self, seqs, is_prefill: bool):
        if not self.enabled:
            return
        if self.current is not None:
            raise RuntimeError("decode profiler observed overlapping steps")
        block_width = max((len(seq.block_table) for seq in seqs), default=0)
        self.current = {
            "step": self.next_step,
            "rank": self.rank,
            "variant": self.variant,
            "phase": "prefill" if is_prefill else "decode",
            "batch_size": len(seqs),
            "request_ids": [getattr(seq, "seq_id", -1) for seq in seqs],
            "sequence_lengths": [len(seq) for seq in seqs],
            "scheduled_tokens": [getattr(seq, "num_scheduled_tokens", 1) for seq in seqs],
            "block_table_width": block_width,
            "block_table_elements": len(seqs) * block_width,
        }
        self.next_step += 1
        if not is_prefill:
            bs = len(seqs)
            # input_ids + positions are int64; slot_mapping/context_lens are int32.
            vector_bytes = bs * (8 + 8 + 4 + 4)
            table_bytes = bs * block_width * 4
            graph_metadata_bytes = vector_bytes + table_bytes
            self.current["main_graph_metadata_h2d_bytes_est"] = graph_metadata_bytes
            self.current["main_graph_d2d_bytes_est"] = vector_bytes + table_bytes
            self.current["temperature_h2d_bytes_est"] = bs * 4
            self.current["main_total_h2d_bytes_est"] = graph_metadata_bytes + bs * 4

    def end_step(self, error: Exception | None = None):
        if not self.enabled:
            return
        if self.current is None:
            raise RuntimeError("decode profiler ended a step that was never started")
        if error is not None:
            self.current["error"] = f"{type(error).__name__}: {error}"
        self.records.append(self.current)
        self.current = None

    def update(self, **fields):
        if self.enabled and self.current is not None:
            self.current.update(fields)

    @contextmanager
    def range(self, name: str, metric: str | None = None):
        if not self.enabled:
            yield
            return

        message = self._nvtx_message(name)
        if self.nvtx_enabled:
            torch.cuda.nvtx.range_push(message)
        started = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter_ns() - started) / 1e6
            if metric is not None and self.current is not None:
                self.current[metric] = elapsed_ms
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_pop()

    def _nvtx_message(self, name: str):
        if self.current is None:
            return f"nanovllm::{name}"
        return (
            f"nanovllm::{name};step={self.current['step']};"
            f"phase={self.current['phase']};bs={self.current['batch_size']};"
            f"width={self.current['block_table_width']}"
        )

    def _flush(self):
        if self.output_path is None:
            raise RuntimeError("decode profiler has no output path")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return self.output_path
