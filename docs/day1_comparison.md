# Day 1: current main vs. PR #176 port

| workload | main tok/s | port tok/s | throughput | prepare_decode | graph input | runner total | D2D copies | H2D copies | block width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch-1 | 251.5 | 261.5 | +3.99% | -58.04% | -18.12% | -3.95% | 1905 -> 0 | 2304 | 1 |
| batch-4 | 934.8 | 946.2 | +1.23% | -22.04% | -19.24% | -1.43% | 1524 -> 0 | 2304 | 1 |
| batch-8 | 1785.5 | 1766.7 | -1.05% | +19.24% | -13.34% | +1.24% | 1524 -> 0 | 2304 | 1 |
| batch-16 | 3349.7 | 3206.9 | -4.26% | +84.87% | -14.40% | +4.77% | 1524 -> 0 | 2304 | 1 |
| batch-4-prompt-1024 | 766.8 | 767.8 | +0.12% | +2.82% | -16.36% | -0.04% | 1905 -> 0 | 2304 | 5 |

Negative phase deltas mean lower CPU time in the candidate. Nsight counts
cover three measured generations, including their prefill steps.
