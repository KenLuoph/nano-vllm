# Day 1 design decision: decode metadata hot path

## Decision

Keep the persistent pinned-host staging idea from PR #176, but do not upstream
the implementation as-is. It removes the redundant GPU-to-GPU metadata copy and
helps batch sizes 1 and 4, but its per-element Python-to-tensor writes become a
regression at batch sizes 8 and 16.

The next implementation should preserve direct pinned-host-to-`graph_vars`
copies while replacing per-element writes and full-capacity clearing with a
stateful, bulk-packed decode input batch.

## Controlled experiment

- Hardware: one NVIDIA A40 46 GB, GPU 1, exclusive during each run.
- Software: PyTorch 2.8.0+cu128, CUDA 12.8 runtime, Triton 3.4.0.
- Model: local Qwen3-0.6B revision frozen under
  `/data/rag/pluo35/models/Qwen3-0.6B`.
- Baseline and candidate both use nano-vLLM commit
  `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.
- Candidate: the core PR #176 staging-buffer design ported onto that exact
  commit. The original PR branch is 20 commits behind current main and is not
  used for causal performance claims.
- Each formal point: prompt 128, output 128, three measured generations after
  warmup, fixed seeds, CUDA Graph enabled, Nsight Systems capture bounded by
  `cudaProfilerStart/Stop`.
- Additional point: batch 4, prompt 1024, output 128, to exercise block-table
  width 5.

The generated comparison table is in `docs/day1_comparison.md`. Raw summaries,
per-step JSONL, Nsight reports and CSV statistics are stored below
`results/day1` in the baseline and candidate worktrees.

## What the data says

1. Direct staging works: `graph_input_copy` CPU submission time drops by
   13-19% in every tested workload.
2. The candidate eliminates all recorded decode metadata D2D copies:
   1,524-1,905 calls in the baseline become zero. H2D count stays at 2,304,
   because the design still issues the same pinned-host-to-GPU field copies.
3. The CPU packing method does not scale:
   `prepare_decode` changes by -58% at batch 1, -22% at batch 4, +19% at
   batch 8 and +85% at batch 16. The candidate performs roughly
   `4 * batch_size + total_block_ids` Python scalar writes per step.
4. End-to-end throughput follows that crossover: +3.99%, +1.23%, -1.05% and
   -4.26% for batch 1/4/8/16 respectively.
5. At batch 4 with block-table width 5, the two versions are effectively tied
   (+0.12% candidate throughput). More long-context points are needed before
   claiming a context-length trend.
6. The sampler and token D2H path remains outside the captured graph and costs
   roughly 3-4 ms per step in this instrumented run. Issue #175 is real, but it
   is not the only host-side opportunity.

## Correctness evidence

With identical prompts, seeds and sampling parameters, baseline and candidate
produced identical SHA-256 token-stream checksums for both checks:

- batch 16, prompt 128, output 64:
  `ed831782220df07bc88b304cfd1e9eb159b32e1b9a6f48fec378c32159c1e689`
- batch 4, prompt 1024, output 64:
  `0911905a40358a4e47dfb0e66452d487f4d60315c90a5df2e9017e57f5c811ff`

This checks the paths most likely to expose stale padding, block-table or batch
row bugs. It is not a replacement for unit tests across request churn.

## Day 2 implementation target

Introduce a `DecodeInputBatch` owned by `ModelRunner` (or the scheduler if
row membership is moved there) with:

- persistent pinned tensors and zero-copy NumPy views;
- a request-to-row map and explicit active-row count;
- bulk field packing, avoiding PyTorch dispatcher calls for every scalar;
- block-table rows updated only when allocation or row ownership changes;
- active-width copies rather than copying `max_model_len / block_size` columns;
- dirty-tail clearing only when a shorter request reuses a row;
- persistent temperature staging so sampling metadata also stops allocating.

A later compiler-runtime co-design phase can keep next-token IDs and the
length/slot update on GPU, potentially using a small Triton metadata-update
kernel. That should be evaluated only after the bulk CPU staging design
establishes a clean, correct baseline.

## Acceptance gates

- No new pinned CPU tensors during steady-state CUDA Graph decode.
- No temporary metadata GPU tensors and no metadata D2D copies.
- No per-element PyTorch tensor assignments in the Python hot path.
- Identical token checksums for fixed-seed tests, plus request-add/remove and
  block-boundary unit tests.
- Non-regressing throughput and `prepare_decode` time at batch 1/4/8/16.
- Long-context tests spanning multiple block-table widths.
- Nsight timeline and an end-to-end ablation table included in the upstream PR.
