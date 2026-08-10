# Formal experiment methodology

## Reproducibility

Every run records the code commit, resolved model revision, seed, CUDA-visible
device, PyTorch/CUDA versions, GPU name, process snapshot, command, workload,
and peak allocated/reserved GPU memory. Physical GPU 1 must be exclusive and is
reported as NVIDIA A40; no H20 or multi-GPU performance claim is made.

The five versions are eager, upstream-main CUDA Graph, PR #176, host v2, and the
GPU-metadata/Triton prototype. The host upstream PR and research prototype stay
in separate worktrees.

## Workloads

- Qwen3-0.6B and Qwen3-1.7B: prompt 128/output 512 at batch 1/4/8/16;
- Qwen3-0.6B and Qwen3-1.7B: prompt 2048/output 256 at batch 4/16;
- Qwen3-8B representative points: prompt 128/output 256 at batch 1/4/8;
- continuous-batching arrivals with mixed prompt/output lengths;
- correctness-only boundaries 255/256/257/511/512/513, output 1024,
  finish/reuse, preemption, padding, width contraction/expansion, and shared
  prefix blocks.

## Statistics

Initial configurations use one warmup and three measured generations. Report
the median per-run throughput plus TPOT P50/P95/P99. If a candidate is within
plus or minus 1% of main, run main and the candidate as alternating processes
for seven rounds and use the median. This reduces drift from temperature,
clocks, and competing host activity.

Aggregate token throughput is insufficient by itself. The report also includes
per-step prepare CPU time, sampler/D2H time, H2D/D2D submissions and bytes,
allocation APIs, graph-launch idle gap, Triton kernel duration, delta count, and
peak CPU/GPU memory where the available tools expose them.

## Correctness before performance

The same prompt tokens, seed, temperature, and output limit are used across
versions. Compare ordered per-request token SHA-256 checksums, not decoded text.
Any stale block, ownership failure, illegal memory access, or checksum mismatch
invalidates the performance row.

## Automation

- `benchmark_decode_hotpath.py` records throughput, per-step TPOT, revisions,
  profiler data, and memory.
- `benchmark_dynamic_arrivals.py` records per-request TTFT, TPOT, E2E, and
  checksums under request arrivals.
- `validate_packed_metadata_gpu.py` validates eager and captured Triton output
  and exports compiler IR.
- `run_formal_matrix.py` executes the matrix and conditional alternating reruns.
- `report_formal_matrix.py` generates CSV, Markdown, and figures.

Raw Nsight reports and model files remain on the server. Only summaries,
figures, compiler text, environment manifests, and small correctness artifacts
belong in Git.
