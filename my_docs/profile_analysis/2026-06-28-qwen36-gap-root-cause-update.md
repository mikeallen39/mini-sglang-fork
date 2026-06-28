# 2026-06-28 Qwen3.6 单卡单并发性能差距归因补充

## 结论更新

之前把重点放在 MoE kernel 融合上并不准确。重新按 kernel 平均耗时、代码路径和已有 trace 对照后，更合理的结论是：

1. **底层 GEMV/GEMM 内核本身不是主要差距来源**
   - `internal::gemvx::kernel`：sglang 8.68 us / 次，mini-sglang 8.73 us / 次，几乎一样
   - `ampere_bf16_s16816gemm_bf16_64x64...`：sglang 34.65 us / 次，mini-sglang 34.63 us / 次，几乎一样
   - 说明不是 cuBLAS/CUTLASS 本体慢，而是 mini-sglang 在外层做了更多额外工作

2. **MoE 差距主要来自“调用次数更多 + 形态不融合”，不是单次 fused_moe_kernel 更慢**
   - `fused_moe_kernel`：sglang 21.46 us / 次，mini-sglang 38.62 us / 次
   - 单次确实慢约 1.8x，但远小于总时间 8.3x 的差距
   - 真正拉开总时间的是 mini-sglang 仍保留 `gate_up -> silu_and_mul -> down -> sum_reduce` 多步路径，且调用次数明显更多
   - 但在 **CUDA graph 开启** 的真实端到端场景下，P0 (`MOE_SINGLE_KERNEL=1`) 几乎没有收益，说明它优化的更多是 launch / 编排层，不是 graph replay 下的核心瓶颈

3. **真正更像“graph on 也会持续存在”的差距，在 linear-attn 周边的 unfused PyTorch 组合算子**
   - mini-sglang Qwen3.6 路径仍然显式做：
     - `mixed_qkvz.split(...)`
     - `mixed_ba.split(...)`
     - `F.normalize(query.float(), ...)`
     - `F.normalize(key.float(), ...)`
     - `value.float().contiguous()`
     - `a.float().contiguous()` / `b.float().contiguous()`
     - Gemma RMSNorm 的 torch 版 `square -> mean -> rsqrt -> mul`
   - sglang 对应路径已经大量换成 fused / JIT kernel：
     - `fused_qkvzba_split_reshape_cat_contiguous_kernel`
     - fused Q/K Gemma RMSNorm 相关 kernel
     - JIT/sgl-kernel `rmsnorm` / `fused_add_rmsnorm` / `gemma_rmsnorm`
     - `_causal_conv1d_update_kernel`
   - 这类差距不会因为 CUDA graph 打开而消失，因为 graph 只减少 launch overhead，不会消除“额外做了哪些算子”本身

## 代码级对照

### mini-sglang 额外做的事情

文件：`python/minisgl/models/qwen3_5_moe.py`

- decode 路径：
  - `a = a.float().contiguous()`
  - `b = b.float().contiguous()`
  - `outputs = self.norm.forward(outputs, z)`，而 `GemmaRMSNormFused` 默认仍走 torch 实现
- prefill 路径：
  - `query = F.normalize(query.float(), dim=-1, eps=1e-6).to(query.dtype).contiguous()`
  - `key = F.normalize(key.float(), dim=-1, eps=1e-6).to(key.dtype).contiguous()`
  - `value.float().contiguous()`
- qkv / ba 拆分：
  - `mixed_qkvz.split(...)`
  - `mixed_ba.split(...)`

文件：`python/minisgl/layers/norm.py`

- `GemmaRMSNormFused.forward()` 默认走 `_torch_gemma_rmsnorm`
- `_torch_gemma_rmsnorm` 展开为：
  - `x.to(fp32)`
  - `square`
  - `mean`
  - `rsqrt`
  - `mul(weight)`
- 只有设置 `MINISGL_USE_FLASHINFER_RMSNORM=1` 时，普通 `RMSNormFused` 才会走 flashinfer；`GemmaRMSNormFused` 当前没有对应 fused CUDA 路径

### sglang 已经融合的事情

文件：`python/sglang/srt/models/qwen3_5.py`

- full attention 准备阶段已经有：
  - fused Q/K Gemma RMSNorm
  - fused QKV / gate 拆分
  - fused rope + norm 路径
- norm 层由 `sgl_kernel` / jit kernel 支持：
  - `gemma_rmsnorm`
  - `fused_add_rmsnorm`
  - `rmsnorm`

文件：`python/sglang/srt/layers/layernorm.py`

- CUDA 路径优先走 fused kernel / jit kernel，而不是 torch 组合算子

## 更新后的优先级

### 第一优先级：把 Qwen3.6 的 norm / qk-norm / split / prefill normalize 换成 fused 路径

这是最可能在 **CUDA graph 开启** 时仍然带来真实收益的点。

具体顺序建议：

1. **GemmaRMSNormFused 增加 fused CUDA 路径**
   - 目标：让 Qwen3.6 的 `input_layernorm` / `post_attention_layernorm` / `final norm` 不再走 `_torch_gemma_rmsnorm`
   - 预期收益：减少大量 `square/mean/rsqrt/mul/copy` kernel

2. **prefill 的 Q/K normalize 融合**
   - 用 fused kernel 替换 `F.normalize(query.float())` + `F.normalize(key.float())`
   - 预期收益：减少 reduce + cast + copy

3. **QKV / BA split 融合**
   - 把 `mixed_qkvz.split` 和 `mixed_ba.split` 改成 fused split kernel
   - 预期收益：减少 slice/copy/cat 路径

### 第二优先级：再考虑 MoE 路径

- MoE 仍值得做，但它更像是“把 graph off 的 launch 浪费清理掉”
- 从真实线上收益看，它不是第一刀

## 对当前实验的影响

- `MOE_SINGLE_KERNEL=1`：可以保留为消融开关，但不建议作为主线优化继续投入
- `SKIP_AB_FP32_CAST=1` 和 `DEPTHWISE_CONV_DECODE=1`：暂时都有稳定性问题，不适合作为主线推进
- 主线应切到 **fused norm / fused qk norm / fused split**
