# Benchmark Layout

This directory is split by benchmark scope:

- `online/`
  - End-to-end service benchmarks.
  - These scripts talk to a running server and measure request-level metrics such as:
    - `TTFT`
    - `E2E`
    - `output_tps`
    - `avg_ms_per_output_token`
- `offline/`
  - Operator- or module-level microbenchmarks.
  - These scripts run local kernels/functions directly without launching a server.
- `analysis/`
  - Targeted analysis and profiling scripts.
  - These are usually single-node local studies for kernel/layout/shape/category comparisons.

## Naming Convention

- `bench_*_micro.py`
  - A microbenchmark for a single operator or a tightly scoped module path.
- `bench_*_buckets.py`
  - A shape/category analysis script that compares multiple bucketed cases.
- `bench_*_<workload>.py`
  - A fixed-workload end-to-end benchmark.

## Current Recommended Entrypoints

### End-to-End

- `online/bench_qwen36_1024in_64out.py`
  - Unified Qwen3.6 online benchmark.
  - Fixed workload:
    - `1024` input tokens
    - `64` output tokens
    - `5` runs
    - `run2-run5` steady-state average

### Analysis

- `analysis/bench_linear_attn_decode_layout_micro.py`
  - Linear attention decode kernel/layout comparison.
- `analysis/bench_fused_moe_int8_micro.py`
  - Fused MoE int8 expert kernel microbenchmark.
- `analysis/bench_moe_backend_micro.py`
  - Torch MoE vs fused MoE microbenchmark.
- `analysis/bench_qwen36_w8a8_linear_buckets.py`
  - Bucketed linear-path analysis for Qwen3.6 w8a8.
- `analysis/bench_rmsnorm_quant_micro.py`
  - RMSNorm + quant isolated microbenchmark.
- `analysis/bench_rmsnorm_quant_linear_micro.py`
  - RMSNorm + quant + linear chained microbenchmark.
