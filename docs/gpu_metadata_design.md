# GPU-resident decode metadata design

## Status

This branch is a research prototype and is intentionally separate from the
host-only upstream draft PR. The host path has A40 end-to-end evidence. The
runtime-slot, packed-H2D, and Triton paths have CPU/reference coverage but still
require A40 compilation, CUDA Graph replay, checksum, and Nsight gates.

## Runtime/compiler contract

```text
Scheduler / BlockManager
  stable runtime_slot + reuse epoch + block_table_version
                 |
                 v
CPU version tracker ---- changed block entries only ----+
                 |                                      |
Sequence rows    | input, position, slot, length, temp  |
                 +------------------+-------------------+
                                    v
                         persistent pinned uint8 blob
                                    |
                           one active-prefix H2D
                                    v
                         persistent GPU uint8 blob
                                    |
                         captured CUDA Graph replay
                         1. apply delta Triton kernel
                         2. unpack/gather Triton kernel
                         3. model/attention kernels
                                    |
                                    v
                      persistent graph_vars + sampler
```

The runtime provides identity, ownership, and version information. Triton turns
that contract into fixed-address device tensors consumed by the captured model.
Neither side alone can safely eliminate the per-token block-table transfer.

## Packed ABI

All offsets are 32-bit words. The backing allocation is a persistent `uint8`
tensor so the whole active prefix can be copied with one H2D submission.

```text
header (8 words)
  magic, batch_size, graph_bucket, active_block_width,
  num_deltas, delta_offset_words, active_words, reserved

row (6 words / 24 bytes)
  input_id:int32, position:int32,
  slot_mapping:int32, context_len:int32,
  temperature:fp32, runtime_slot:int32

delta (3 words / 12 bytes)
  runtime_slot:int32, block_column:int32,
  block_id:int32
```

The delta section starts immediately after the active graph bucket, not after
the maximum row capacity. Therefore steady-state copy volume is
`32-byte header + graph_bucket * 24 bytes`. A normal block boundary adds one
12-byte delta. Slot reuse sends one full logical row so every stale column is
overwritten.

The CPU owns epoch validation; an epoch change is converted to a full-row delta
before packing, so the GPU payload does not need to carry the epoch itself.
Token IDs and positions are transported as int32 because they are bounded by
the model vocabulary and `max_model_len`; the unpack kernel widens them into
the captured graph's int64 input tensors.

## Ownership invariants

1. A runtime slot has at most one active request owner.
2. Slots are acquired only when a sequence becomes `RUNNING` and released on
   finish or preemption.
3. Every reuse increments the slot epoch. A changed epoch forces a complete row
   replacement and prevents old block IDs leaking into the new request.
4. `block_table_version` increments on allocate, append, and clear. An unchanged
   epoch/version pair produces no block delta.
5. Scheduler admission includes already-running requests in `max_num_seqs`; a
   new prefill cannot overcommit stable runtime rows.
6. Padding lanes use `input=0`, `position=0`, `slot=-1`, `context_len=0`,
   `runtime_slot=-1`. Zero context length is what prevents paged attention from
   dereferencing block ID `-1`.

## CUDA ordering and buffer lifetime

The CPU fills the same pinned blob every step. Its non-blocking H2D is submitted
on the current stream before `graph.replay()`. An experimental
`NANOVLLM_DIRECT_PACKED_H2D=1` path calls `cudaMemcpyAsync` on the persistent
buffers directly, avoiding per-step tensor views and PyTorch's generic copy
dispatcher. It remains opt-in until end-to-end tests show a gain beyond noise.
The graph begins with metadata kernels, so stream order is:

```text
packed H2D -> apply deltas -> unpack/gather -> attention/model -> sampler D2H
```

In the synchronous single-GPU engine, sampler `.tolist()` completes a D2H and
provides the step boundary before the CPU rewrites the staging allocation. This
assumption must be revalidated for tensor parallel workers; the prototype does
not add a host event synchronization that could hide the intended gain.

## Triton kernels

`apply_block_table_deltas_kernel` launches a fixed captured grid and masks lanes
above dynamic `num_deltas`. Delta destinations are unique within a step, so no
atomics are needed.

`unpack_decode_metadata_kernel` writes vector metadata and gathers each active
logical block-table entry from `master_block_tables[runtime_slot, column]`.
The graph bucket and maximum logical block count are compile-time constants;
batch size and active width are runtime header values.

The validation program saves TTIR, TTGIR, LLIR, and PTX when exposed by the
installed Triton version. Review should check masked loads/stores, 24-byte row
layout, coalesced master-table reads, launch grid, registers, and whether the
always-launched delta kernel costs more than it saves in no-delta steps.

## Risks and profiling-driven extensions

- If the masked delta kernel is material in steady decode, capture a no-delta
  graph variant that begins directly with unpack/gather.
- If CPU row packing remains visible, maintain incremental row order or move
  length/slot arithmetic into a smaller GPU state update.
- If graph bucket padding dominates, evaluate bucket specialization against
  graph-memory growth.
- Once metadata is below the noise floor, profile sampler/D2H and Python
  scheduler time before optimizing further. Amdahl's law, not feature count,
  decides the next contribution.

## Verified A40 results

Environment: NVIDIA A40 GPU 1, PyTorch 2.8.0+cu128, Triton 3.4.0, three
measured runs per configuration, direct-H2D experiment disabled unless stated.
All comparisons below are against the host-v2 CUDA Graph path on the same
research checkout.

| model/workload | batch | host v2 tok/s | GPU metadata tok/s | change |
|---|---:|---:|---:|---:|
| Qwen3-0.6B, p128/o512 | 1 | 287.09 | 292.55 | +1.90% |
| Qwen3-0.6B, p128/o512 | 4 | 1011.93 | 1040.59 | +2.83% |
| Qwen3-0.6B, p128/o512 | 8 | 1852.75 | 1899.32 | +2.51% |
| Qwen3-0.6B, p128/o512 | 16 | 3154.56 | 3219.20 | +2.05% |
| Qwen3-0.6B, p2048/o256 | 4 | 689.55 | 701.89 | +1.79% |
| Qwen3-0.6B, p2048/o256 | 16 | 1281.74 | 1292.42 | +0.83% |
| Qwen3-1.7B, p128/o512 | 1 | 130.70 | 131.95 | +0.96% |
| Qwen3-1.7B, p128/o512 | 4 | 496.82 | 502.48 | +1.14% |
| Qwen3-1.7B, p128/o512 | 8 | 941.68 | 952.52 | +1.15% |
| Qwen3-1.7B, p128/o512 | 16 | 1675.85 | 1692.32 | +0.98% |

Nsight Systems at batch 16 observed six host-v2 metadata H2D submissions per
decode step versus one packed submission, zero metadata D2D copies in both,
and essentially unchanged model-kernel time. The compact ABI reduces the
steady batch-16 payload from 544 to 416 bytes; it did not produce a measurable
throughput change beyond noise. Captured apply+unpack kernels take about
3.16--3.23 microseconds together.

The direct `cudaMemcpyAsync` experiment reduced standalone CPU submission time
by 57--60%, but seven alternating batch-16/p128/o512 rounds produced only
+0.043% paired-median throughput (+0.10% by independent medians). It therefore
remains opt-in. This negative result establishes that sampler/model completion,
not metadata submission, is the next Amdahl-limited region.

## Validation status and remaining gate

The A40 single-GPU gate passes for eager/captured kernel output, stable-address
replay, fixed/boundary/dynamic-churn token checksums, one metadata H2D, zero
metadata D2D, and saved TTIR/TTGIR/LLIR/PTX. Runtime-slot preemption,
finish/reuse, padding, width changes, and shared-prefix ownership pass the CPU
and GPU-mirror reference suites.

The path remains a research prototype because full-model replay under actual
preemption/prefix sharing and tensor-parallel buffer lifetime have not yet been
validated. Those are correctness gates, not performance polish; no multi-GPU
or H20 claim should be made from the current data.
