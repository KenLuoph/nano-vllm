# nano-vLLM 项目面试答辩包

## 30 秒版本

我在 nano-vLLM 的 CUDA Graph decode 路径上做了一个 compiler-runtime
co-design 项目。先定位到 graph replay 前每个 token 仍在重复创建 pinned tensor
和做 D2D copy；host v2 用持久化 pinned buffer 和 NumPy 零拷贝 view，把 A40
上 Qwen3-0.6B 的吞吐提升了 3.46%–5.80%，并通过 Nsight 验证 steady-state
没有 metadata allocation 和 D2D。然后我把研究路线扩展为稳定 runtime slot、
带 epoch/version 的 GPU block-table mirror、单次 packed H2D，以及 CUDA Graph
内 captured Triton unpack/gather kernel。研究路径目前必须通过 A40 checksum、
Nsight 和 IR gate 后，才会写入最终性能结论。

## 5 分钟版本结构

1. **问题**：CUDA Graph 缓存了模型 kernel DAG，但没有自动消除 replay 前的
   Python list、pinned allocation、H2D 和 GPU-to-GPU staging copy。
2. **测量**：main/#176/v2 三版本消融。#176 在小 batch 有收益，在 batch 8/16
   回退；microbenchmark 证明 PyTorch scalar assignment 是原因。
3. **host v2**：一次性分配 pinned tensors，建立共享 NumPy views，bulk pack，
   直接 H2D 到固定 graph_vars，只传 active block width。
4. **正确性**：padding 必须 `context_len=0`；257/513 的 `==1` 分配不是
   off-by-one；动态 batch、block boundary 和 request churn checksum 一致。
5. **研究扩展**：scheduler 分配稳定 runtime slot；epoch 防止 slot reuse 泄漏；
   version 只产生 block delta；一个 uint8 blob 单次 H2D；Triton 在 graph 内更新
   master table、unpack metadata、按 slot gather。
6. **方法论**：先 correctness，再 Nsight，再端到端；若只减少 memcpy 次数但
   tok/s 不涨，就用 Amdahl 解释并保留负结果。

## 20 分钟深挖主线

### Request 到一次 decode replay

Scheduler 从 RUNNING 请求中选 batch。每条请求当前最后一个 token 需要
`input_id`、`position=len-1`、`context_len=len`，新 K/V 写入地址是
`physical_block_id * block_size + last_block_num_tokens - 1`。Attention 通过
block table 把逻辑 KV block 映射到不连续的物理 page。

host v2 在 CPU persistent pinned storage 中写这些数据，然后直接 H2D 到
CUDA Graph capture 时绑定的固定 GPU 地址。Graph replay 执行模型，sampler
产生下一个 token，`.tolist()` 形成同步引擎的 step boundary。

### 为什么 sequence length 257 的 offset 是 0

长度 257 表示序列已经包含第 257 个 token，位置索引是 256。block size 256
时它属于第二个逻辑 block 的第一个位置，所以
`last_block_num_tokens=1`，零基 offset 是 `1-1=0`。如果问“下一个还没产生的
token”，它的位置才是 257；prepare_decode 处理的是已经 sample、马上要送入
下一次 forward 的最新 token。

### 为什么需要 epoch

只有 runtime slot 不够。请求 A 结束后 slot 3 给请求 B，如果 GPU master row
仍保留 A 的较长 block table，而 B 只覆盖前两列，就可能读取旧列。每次 acquire
递增 epoch；epoch 改变发送完整 row（有效 block IDs 加 `-1` 尾部），所以复用
不会继承旧 ownership。

### 为什么 Triton 必须在 graph 里

单次 H2D 只把提交次数从六变一，attention 仍需要独立的 typed tensors。
Captured Triton kernel 在固定 GPU 地址上把 blob 解包，并从 GPU master table
gather 当前 batch rows。这样 CPU submission 保持一次 graph replay，编译器看到
固定 grid/address，runtime 仍能通过 header、mask 和 delta 提供动态状态。

## 高频追问与回答

- **CUDA Graph 优化了什么？** 减少重复 kernel launch/API submission；它不自动
  优化 capture 外的 Python、allocation 或 memcpy。
- **为什么 pinned memory？** page-locked host memory 支持真正异步 DMA；普通
  pageable memory 往往需要 driver staging。
- **为什么不用每步新 tensor？** 分配器和 Python/PyTorch dispatch 会在每个
  generated token 重复，decode 计算变快后 host overhead 更显著。
- **为什么不把 block table 放 token 里？** token ID 是模型输入；block table 是
  runtime 的逻辑页到物理 KV page 映射，两者生命周期和语义不同。
- **active block width 是什么？** 当前 step 最长请求实际拥有的 block 数。只复制
  `bucket × active_width`，其余列虽然可能 stale，但 context length 保证不可访问。
- **为什么 padding context 不能是 1？** context 1 会让 attention 访问逻辑 block
  0，而 padding block ID 是 -1；context 0 才表示没有 KV 可读。
- **六次 H2D 合成一次一定更快吗？** 不一定；新增 Triton kernel 也有 GPU 成本，
  必须用端到端与 Nsight 判断，收益受 Amdahl 上限约束。
- **为何不用 C++ extension？** NumPy bulk packing 已解决 scalar dispatch；研究
  路线的关键是 GPU-resident ownership/delta，而不是先增加 build complexity。
- **TP 风险是什么？** 各 rank 都有 staging 与 graph，rank 0 sampler D2H 的同步
  不能直接证明非 rank-0 host buffer 可安全复写，需要单独 event/trace 验证。
- **如何 upstream？** host-only diff 小、可 review，并在 #176 明确 attribution；
  runtime-slot/Triton 保持独立 research branch，验证成熟后再拆分贡献。
