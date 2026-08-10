# Resume bullets

Only the host-v2 bullets below are ready to use. The Triton bullet is explicitly
marked pending until the formal A40 matrix passes.

## English — verified host path

- Optimized nano-vLLM's CUDA Graph decode hot path with persistent pinned-memory
  staging, zero-copy NumPy packing, active-width KV block tables, and direct H2D
  graph-input updates; improved Qwen3-0.6B throughput by 3.46%–5.80% across batch
  1–16 on NVIDIA A40 while preserving deterministic token checksums.
- Built main/PR-#176/v2 microbenchmarks and Nsight Systems validation; reduced
  geometric-median CPU metadata packing latency by 43.3% vs. main and 95.2% vs.
  the scalar-assignment design, eliminating steady-state metadata allocations
  and D2D copies.

## 中文 — 已验证 host 路径

- 优化 nano-vLLM CUDA Graph decode 热路径：复用 pinned-memory staging buffer、
  通过 NumPy 零拷贝 view 批量打包 metadata、按 active width 传输 KV block table，
  并直接更新固定 graph inputs；在 NVIDIA A40/Qwen3-0.6B、batch 1–16 上实现
  3.46%–5.80% 吞吐提升，同时保持 token checksum 完全一致。
- 构建 main/PR #176/v2 三版本 microbenchmark 与 Nsight Systems 消融；相对 main
  将 CPU metadata packing 几何中位延迟降低 43.3%，相对 scalar-assignment 方案
  降低 95.2%，并消除 steady-state metadata allocation 与 D2D copy。

## Pending — use only after formal GPU validation

- Designed a versioned GPU-resident KV block-table mirror with stable runtime
  slots and reuse epochs, then co-designed a single packed H2D ABI with captured
  Triton delta/unpack/gather kernels; **insert only Day-9 verified H2D, latency,
  and end-to-end numbers here**.
