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

row (8 words / 32 bytes)
  input_id:int64, position:int64,
  slot_mapping:int32, context_len:int32,
  temperature:fp32, runtime_slot:int32

delta (4 words / 16 bytes)
  runtime_slot:int32, block_column:int32,
  block_id:int32, runtime_slot_epoch:int32
```

The delta section starts immediately after the active graph bucket, not after
the maximum row capacity. Therefore steady-state copy volume is
`32-byte header + graph_bucket * 32 bytes`. A normal block boundary adds one
16-byte delta. Slot reuse sends one full logical row so every stale column is
overwritten.

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
on the default stream before `graph.replay()`. The graph begins with metadata
kernels, so stream order is:

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
installed Triton version. Review should check masked loads/stores, 32-byte row
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

## Validation gate

The Triton path remains a prototype until all of the following pass on A40:

- eager kernel output equals the PyTorch reference bit-for-bit;
- captured replay accepts changing blob contents at a stable address;
- fixed, boundary, dynamic arrival, finish/reuse, preemption, padding, and
  prefix-sharing token checksums equal host v2;
- Nsight shows one metadata H2D submission, no metadata D2D, correct stream
  ordering, bounded memory, and no stale block access;
- saved compiler IR matches the intended masks and memory access pattern.
