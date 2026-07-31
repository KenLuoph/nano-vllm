# Decode metadata ownership and lifetime

## Current main

```text
Scheduler / Sequence objects (CPU, engine/request lifetime)
  ├─ last_token
  ├─ sequence length
  ├─ block_table
  └─ sampling temperature
          │ every decode step
          ▼
Python lists (CPU, step lifetime)
          │ torch.tensor(..., pin_memory=True)
          ▼
Pinned CPU tensors (CPU, step lifetime)
          │ .cuda(non_blocking=True)
          ▼
Temporary GPU tensors (GPU, step lifetime)
          │ device-to-device assignment
          ▼
graph_vars (GPU, engine/CUDA-graph lifetime; fixed addresses)
          │ graph.replay()
          ▼
Captured Transformer forward
          │ outputs hidden states
          ▼
LM head + sampler (outside the captured graph)
          │ token_ids.tolist() synchronizes result to CPU
          ▼
Scheduler.postprocess() updates Sequence and KV block state
```

## Metadata consumers

| Metadata | Producer | GPU consumer | Meaning |
|---|---|---|---|
| `input_ids` | `Sequence.last_token` | embedding | latest token for each active request |
| `positions` | sequence length | RoPE | logical token position |
| `slot_mapping` | block table + offset | `store_kvcache_kernel` | physical slot for the new K/V at every layer |
| `context_lens` | sequence length | FlashAttention decode | number of valid historical tokens |
| `block_tables` | BlockManager/Sequence | FlashAttention decode | logical-to-physical KV block mapping |
| `temperatures` | sampling params | sampler | per-request sampling temperature |

## What CUDA Graph captures

`capture_cudagraph()` allocates persistent GPU tensors and captures
`self.model(input_ids[:bucket], positions[:bucket])` for fixed batch buckets.
The graph records kernels, dependencies, shapes and tensor addresses. It does
not capture the scheduler, Python input construction, LM head, sampler or
postprocess. New metadata values must therefore reach the fixed `graph_vars`
addresses before every replay.

## Why issue #175 exists

Updating metadata values is necessary. Reallocating a pinned tensor, creating
a temporary GPU tensor, and copying it again into the fixed graph input is not.
The profiler separates these host-side costs from GPU graph execution so that
the redesign is based on measured allocation/copy overhead rather than source
inspection alone.

## PR #176 port examined on Day 1

```text
Sequence objects (CPU)
        │ Python scalar assignments, every step
        ▼
Persistent pinned CPU staging (engine lifetime)
        │ direct non-blocking H2D copies
        ▼
graph_vars (persistent GPU addresses)
        │ graph.replay()
        ▼
Captured Transformer forward
```

This removes the temporary GPU tensors and their second D2D copy. It does not
remove H2D field copies, and its scalar tensor writes scale with batch size and
block-table length. The Day 1 measurements therefore select a hybrid follow-up:
retain persistent staging and direct copies, but bulk-pack or incrementally
maintain the CPU metadata instead of assigning each scalar through PyTorch.
