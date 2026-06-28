# 2026-06-28 SGLang vs Mini-SGLang CUDA Kernel 对比分析

## 测试条件

- 模型：Qwen3.6-35B-A3B（40层，30 linear_attention + 10 full_attention）
- 硬件：NVIDIA A800 80GB PCIe，单卡（GPU7）
- 数据类型：bf16，不量化
- 环境：CUDA graph 关闭，radix cache 关闭
- 分析方法：PyTorch profiler（`torch.profiler.profile`）采集 chrome trace，提取 CUDA kernel 耗时
- sglang 环境：torch 2.11.0+cu130，/mnt/82_store/zxz/condaenv/sglang-cu13
- mini-sglang 环境：torch 2.6.0+cu124，/mnt/82_store/zxz/condaenv/minisgl

## 端到端性能基准（bf16，CUDA graph 开启）

| 指标 | sglang | mini-sglang | 比值 |
|------|--------|-------------|------|
| TTFT (run2-5 avg) | 106.40 ms | 118.49 ms | 1.11x |
| E2E (run2-5 avg) | 0.516 s | 0.830 s | 1.61x |
| output_tps | 124.03 tok/s | 75.93 tok/s | 1.63x |

## CUDA Kernel 耗时分解（decode 阶段，无 CUDA graph）

sglang trace：`/tmp/1782640847.7483978-TP-0-DECODE.trace.json.gz`
mini-sglang trace：`/tmp/minisgl_traces/1782641008.7638276-TP-0.trace.json`

### 按功能类别的 CUDA kernel 总耗时对比

| kernel 类别 | sglang (ms) | mini-sglang (ms) | 比值 | 根因分析 |
|------------|------------|-----------------|------|---------|
| **fused_moe_kernel** | 8.59 | 71.07 | **8.3x** | sglang 1 kernel 融合 gate_up+act+down；mini-sglang 3 kernel（gate_up gemm → silu_and_mul → down gemm），每次调用还慢 1.9x |
| **dense gemv (out_proj等)** | 6.95 | 25.14 | 3.6x | 调用次数和每次耗时均更多 |
| **gemm (out_proj等)** | 9.59 | 38.92 | 4.1x | 同上 |
| **elementwise** | 2.08 | 58.19 | **28x** | mini-sglang 大量 `.float().contiguous()`、`.to(dtype)` 等操作产生海量 `vectorized_elementwise_kernel` |
| **reduce (norm/sum)** | ~1.0 | 12.52 | 12.5x | `F.normalize` 等操作产生额外 reduce |
| **conv1d** | 0.57 | 3.38 | 5.9x | F.conv1d vs causal_conv1d_update |
| **linear_attn decode kernel** | 1.30 | 4.73 | 3.6x | mini-sglang 每层调用更多次（含 prefill kernel） |
| **flashinfer fused norm** | 1.27 | — | ∞ | sglang 独有，norm+add 融合 |
| **fused_qkv_split** | 0.48 | — | ∞ | sglang 把 QKVZB 的 split+reshape 融合为 1 kernel |

### sglang decode 每个 layer-step 的 CUDA kernel 调用模式（30 层 × 5 steps = 150）

| kernel | 总调用次数 | 每次耗时 | 每层-step 调用 |
|--------|-----------|---------|---------------|
| fused_moe_kernel | 400 | 21 us | 2.67 |
| gemvx (dense) | 800 | 9 us | 5.33 |
| flashinfer fused norm | 400 | 3 us | 2.67 |
| topkGatingSoftmax | 200 | 6 us | 1.33 |
| moe_align_block_size | 200 | 4 us | 1.33 |
| act_and_mul | 400 | 2 us | 2.67 |
| fused_gate_sigmoid_mul_add | 200 | 3 us | 1.33 |
| fused_qkvzba_split | 150 | 3 us | 1.0 |
| causal_conv1d_update | 150 | 4 us | 1.0 |
| packed_decode (linear_attn) | 150 | 9 us | 1.0 |
| layer_norm | 150 | 3 us | 1.0 |

### mini-sglang 额外的问题

1. **dtype cast 开销**：每层 decode 调用 `a.float().contiguous()` 和 `b.float().contiguous()`，产生 2 次 elementwise copy kernel
2. **QKV split 未融合**：`mixed_qkvz.split()` + `mixed_ba.split()` 使用 PyTorch 原生 split（2 次 aten::slice）
3. **conv 未融合**：`F.conv1d` + `F.silu` 是独立的 kernel launch
4. **norm 未融合**：gated RMSNorm 是独立的 Python 实现，未与下游 out_proj 融合

## 改进优先级

按预估收益排序：

### P0: MoE Fused Kernel 融合

- 现状：mini-sglang 每层 MoE 调用 `fused_moe_kernel_triton` 两次 + `silu_and_mul` 一次，共 3 kernel
- sglang：可能 1 kernel 完成 gate_up + act + down
- 预估收益：~50ms 总 CUDA kernel 时间缩减
- 环境变量：`MOE_SINGLE_KERNEL=1`

### P1: 消除 dtype cast elementwise kernel

- 现状：`a.float().contiguous()` + `b.float().contiguous()` 每层 2 次 copy
- 方案：将 fp32 cast 逻辑融合进 decode kernel 或保持 bf16 计算
- 预估收益：~20ms 总 CUDA kernel 时间缩减
- 环境变量：`SKIP_AB_FP32_CAST=1`

### P2: 替换 conv1d 为 causal_conv1d_update

- 现状：`F.conv1d` + `F.silu` 每层 ~124us（无 CUDA graph）
- 方案：使用 Triton fused depthwise conv kernel（已实现，`depthwise_conv_triton.py`）
- 预估收益：~5ms 总 CUDA kernel 时间缩减
- 环境变量：`DEPTHWISE_CONV_DECODE=1`

### P3: 集成 flashinfer fused norm

- 现状：gated RMSNorm 独立 kernel
- 方案：使用 flashinfer 的 `fused_add_rmsnorm`
- 需要处理 gated norm 的兼容性

## 下一步

逐个实现上述改进，每次一项，用统一 benchmark 测量效果。
