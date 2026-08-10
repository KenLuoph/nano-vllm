# Resume bullets

The host and research bullets below have both passed A40 correctness and
performance validation. Keep the comparison baseline explicit.

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

## English — verified research path

- Designed a versioned GPU-resident KV block-table mirror with stable runtime
  slots and reuse epochs, then co-designed a single packed H2D ABI with captured
  Triton delta/unpack/gather kernels; reduced decode metadata H2D submissions
  from 6 to 1 per token and improved throughput over the optimized host path by
  1.90%–2.83% on Qwen3-0.6B and 0.96%–1.15% on Qwen3-1.7B, with deterministic
  checksums under block boundaries and dynamic request churn.

## 中文 — 已验证 research 路径

- 设计带稳定 runtime slot、reuse epoch 与 version delta 的 GPU-resident KV
  block-table mirror，并联合设计单次 packed H2D ABI 与 CUDA Graph 内 captured
  Triton delta/unpack/gather kernel；将每 token 的 decode metadata H2D 从 6 次
  降至 1 次，相对已优化 host 路径在 Qwen3-0.6B 上提升 1.90%–2.83%、在
  Qwen3-1.7B 上提升 0.96%–1.15%，并通过 block boundary 与动态请求 checksum。
