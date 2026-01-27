# Nexus Hardware Performance Analysis

## 1. Current Performance (RTX 5080)

- **Configuration**: Batch Size = 1 (Sequential), `max_new_tokens` = 1024.
- **Speed**: ~500s/it.
- **Utilization**: ~30% VRAM (5GB/16GB), <20% Compute.
- **Bottleneck**: **Software/Algorithm**. The `BS=1` setting means the GPU spends microseconds computing and milliseconds waiting for Python overhead and memory shuffling per token. The RTX 5080 is capable of 5x-10x this throughput if we use Batch Size 8 or 16.

## 2. Hardware Comparison (Scenario: NIWT Profiling, CoT Generation)

| Hardware | VRAM | Batch Size Capacity (Est.) | Est. Time (BS=1) | Est. Time (Max BS) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 5080 (Yours)** | 16GB | 8-12 | ~500s/it | **~60s/it** | **Excellent.** High clock speed gives great single-token latency. Bottleneck is strictly BS=1. |
| **NVIDIA T4** | 16GB | 4-6 | ~1200s/it | ~300s/it | **Downgrade.** Older architecture (Turing). significantly slower memory bandwidth. |
| **NVIDIA P100** | 16GB | 4-6 | ~1500s/it | ~400s/it | **Downgrade.** Ancient (Pascal). Poor FP16 support. Avoid. |
| **NVIDIA L4** | 24GB | 16 | ~600s/it | ~80s/it | **Sidegrade.** More VRAM allows larger batches, but raw compute is lower than 5080. |
| **NVIDIA A100** | 40GB | 32-48 | ~450s/it | **~15s/it** | **Upgrade (Throughput).** Massive bandwidth Allows huge batches. Overkill for single-stream. |
| **NVIDIA H100** | 80GB | 64+ | ~400s/it | **~8s/it** | **Extreme Overkill.** |
| **TPU v5e** | 16GB | N/A | Variable | Variable | **Complex.** Requires code rewrite (JAX/PyTorch XLA). Not drop-in compatible. |

### Conclusion

Your **RTX 5080 is perfectly capable**. The "slowness" is purely due to the `Batch Size = 1` limitation in our current script. **We do not need better hardware; we need better code.**

## 3. Optimization Plan

For the next phase (N=100 run), we will:

1. **Refactor `niwt_profiler.py` to support Batching.**
2. **Target Batch Size**: 8 (Safe for 16GB VRAM).
3. **Expected Speedup**: ~6x-8x.
