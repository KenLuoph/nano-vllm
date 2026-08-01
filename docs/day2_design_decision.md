# Day 2: DecodeInputBatch v2 design decision

## Decision

DecodeInputBatch v2 is a viable upstream direction. It removes per-token metadata
tensor allocation and all metadata D2D copies while fixing the batch 8/16 CPU
packing regression observed in PR #176.

The experiment is based on:

- main profiling commit: `12e024c`
- PR #176 port: `eee64e6`
- v2 implementation branch: `codex/day2-decode-input-batch-v2`
- GPU/model: NVIDIA A40 GPU 1, Qwen3-0.6B

## Implemented path

```text
Sequence CPU state
    -> NumPy writes into zero-copy views
persistent pinned CPU tensors
    -> non-blocking H2D copy_
persistent CUDA Graph graph_vars
    -> graph replay
```

`DecodeInputBatch` owns persistent input IDs, positions, slot mappings, context
lengths, block tables, and temperatures. The hot path does not call
`torch.tensor` or `torch.empty`. Vector metadata is written through NumPy views;
each block table is copied with one row slice assignment. Only the active
`bucket × block_width` table prefix is transferred.

Temperature metadata uses persistent pinned CPU and GPU storage on rank 0.
Prefill, eager execution, and decode batches above the graph limit retain the
original fallback path.

## Padding correctness correction

The initial plan specified `context_len=1`, `slot=-1`, and block table `-1` for a
padding row. Dynamic churn testing showed this combination is unsafe: after a
batch contracts from 4 to 3, the captured bucket still executes row 4 and paged
attention dereferences logical block 0, whose physical block ID is `-1`. This
caused an illegal memory access on A40.

The implemented invariant is therefore:

```text
padding input=0, position=0, slot=-1, context_len=0, active block table=-1
```

With zero context length, paged attention performs no KV lookup for the padded
row. The dynamic churn suite passes with this invariant.

## Correctness

Main and v2 produced identical per-request token counts and SHA-256 token
checksums for:

- batch 1/4/8/16, prompt 128, output 64;
- batch 4, prompt 1024, output 64;
- prompt lengths 255/256/511 across KV-block boundaries;
- dynamic request churn with output limits 17/33/65/129 plus a newly injected
  41-token request.

Nine CPU unit tests cover vector values, slot mapping, ragged block tables,
bucket padding, batch contraction, width contraction/expansion, rank-0-only
temperature packing, zero-copy tensor/NumPy storage, capacity checks, and absence
of steady-state tensor allocator calls.

## CPU packing microbenchmark

The full matrix ran every combination of batch 1/4/8/16/32/64 and block width
1/4/16/64 for 10,000 measured iterations after 200 warmups, using pinned CPU
memory. Across the 24 combinations, v2's geometric-mean median latency is 43.3%
below main and 95.2% below the PR #176 scalar-assignment path.

Representative median latencies:

| batch | block width | main | PR #176 | v2 NumPy |
|---:|---:|---:|---:|---:|
| 1 | 1 | 14.51 µs | 19.83 µs | 7.08 µs |
| 8 | 16 | 29.14 µs | 504.24 µs | 16.03 µs |
| 16 | 16 | 44.26 µs | 1000.61 µs | 25.41 µs |
| 32 | 64 | 178.61 µs | 6974.78 µs | 81.98 µs |
| 64 | 64 | 335.05 µs | 14566.11 µs | 157.61 µs |

The complete mean, median, and P95 dataset is stored in
`results/day2/packing/packing-10000.json`; the generated 24-row table is stored
beside it as `packing-10000.md`.

## End-to-end A/B/C results

Each workload uses three measured generations under Nsight Systems.

| workload | main tok/s | PR #176 tok/s | v2 tok/s | v2 vs main | prepare_decode main/PR/v2 |
|---|---:|---:|---:|---:|---:|
| batch 1 | 251.5 | 261.5 | 266.0 | +5.80% | 226.8/95.2/48.7 µs |
| batch 4 | 934.8 | 946.2 | 977.9 | +4.62% | 228.1/177.9/54.4 µs |
| batch 8 | 1785.5 | 1766.7 | 1856.7 | +3.99% | 232.2/276.8/58.0 µs |
| batch 16 | 3349.7 | 3206.9 | 3465.5 | +3.46% | 246.7/456.0/64.9 µs |
| batch 4, prompt 1024 | 766.8 | 767.8 | 800.0 | +4.33% | 233.3/239.9/55.3 µs |

All five workloads pass the `main -1%` acceptance threshold. None are within
±1%, so the conditional seven-run rerun was not required.

## Nsight and profiler evidence

- metadata D2D copies: main 1524-1905, PR #176 0, v2 0;
- H2D copies: 2,304 over 384 profiled model steps, or six per step;
- profiler `decode_tensor_allocations`: 0 in steady-state decode;
- profiler `metadata_d2d_copies`: 0;
- active-width table bytes are recorded per step rather than copying full table
  capacity.

For batch 16, Nsight's captured range reports 381 `cudaGraphLaunch` calls,
2,304 H2D copies, 384 sampler D2H copies, no D2D copies, and no `cudaMalloc` or
`cudaHostAlloc` API calls. This independently confirms that the measured decode
region is allocation-free and that each decode replay still submits six H2D
copies.

The six H2D submissions remain deliberate Day 2 scope. Combining them requires a
different packed layout or GPU-resident metadata design and should be evaluated
as a separate experiment.

## Upstream recommendation

The core implementation has a credible upstream PR basis: it has deterministic
correctness coverage, dynamic-batch coverage, allocator and memcpy evidence, and
positive end-to-end results across all measured batches. Before submission, the
patch should be rebased onto the latest upstream main and discussed with the
maintainer because PR #176 may change concurrently.

## Reproduction entry points

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_decode_packing.py --iterations 10000 --warmup 200
python benchmarks/check_decode_correctness.py --model MODEL --variant v2 --output v2.json
python benchmarks/compare_decode_correctness.py main.json v2.json
python benchmarks/compare_day2_results.py \
  --main-root MAIN_RESULTS --pr-root PR_RESULTS --v2-root V2_RESULTS
```

Set `CUDA_VISIBLE_DEVICES` explicitly for every GPU command. The formal runs used
physical GPU 1.
