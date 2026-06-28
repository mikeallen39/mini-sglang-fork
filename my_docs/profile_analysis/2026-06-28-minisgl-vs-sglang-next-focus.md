# 2026-06-28 mini-sglang vs sglang 下一步追赶重点

## 当前状态

在修复 Qwen3.6 CUDA graph 正确性问题后，mini-sglang 的 `bf16 + fi + sglang linear attention + CUDA graph` 初始基线为：

- `TTFT = 130.59 ms`
- `E2E = 0.8763 s`
- `output_tps = 71.89 tok/s`

当前同口径、同 workload 的 sglang 参考值为：

- `TTFT = 106.40 ms`
- `E2E = 0.5160 s`
- `output_tps = 124.03 tok/s`

沿着“和 sglang 直接对齐”的主线继续推进后，mini-sglang 当前最好稳定值已经推进到：

- `TTFT = 123.30 ms`
- `E2E = 0.7131 s`
- `output_tps = 88.35 tok/s`

也就是：

- 相比修复后原 baseline，`output_tps` 已累计提升约 `22.9%`
- 但相对 sglang 的 `124.03 tok/s`，仍只有约 `71%`

## 本轮单因素实验结论

### 已确认有效

`MINISGL_GEMMA_FUSED_NORM=1`

- `TTFT = 126.43 ms`
- `E2E = 0.7861 s`
- `output_tps = 80.15 tok/s`

相对 baseline：

- `TTFT` 改善约 `3.2%`
- `E2E` 改善约 `10.3%`
- `output_tps` 提升约 `11.5%`

这说明：

- mini-sglang 与 sglang 的性能差距里，**norm / fused add+norm** 确实占了一块不小的比例
- 但即便已经吃到这部分收益，`80.15 tok/s` 仍明显低于 sglang 的 `124.03 tok/s`

`GemmaRMSNorm.forward_inplace()` 接入 fused 路径（copy-back 安全版）

- `output_tps = 82.75 tok/s`

这说明：

- full attention 的 q/k inplace norm 继续能挤出一点收益
- 但这部分增益已经明显小于首轮 fused norm，不是剩余大差距的主来源

`MINISGL_FULL_ATTN_FUSED_PREPARE=1`

- `TTFT = 121.91 ms`
- `E2E = 0.7277 s`
- `output_tps = 86.58 tok/s`

这说明：

- 直接对齐 sglang full-attention 前处理路径是有效的
- mini-sglang 原有的 `split -> q_norm -> k_norm -> rope -> gate` 这一串，确实还存在结构性开销

`MINISGL_DEPTHWISE_CONV_DECODE=1`（改为直接复用 sglang `causal_conv1d_update` 之后）

- `TTFT = 123.30 ms`
- `E2E = 0.7131 s`
- `output_tps = 88.35 tok/s`

这说明：

- “decode conv 不是瓶颈”这个结论只适用于之前那版自写 Triton conv
- 如果直接对齐 sglang 当前的 `causal_conv1d_update`，仍能继续带来小幅收益

### 已确认不是主瓶颈

`MINISGL_SKIP_AB_FP32_CAST=1`

- `output_tps = 72.31 tok/s`
- 相对 baseline 仅提升约 `0.6%`

`MINISGL_DEPTHWISE_CONV_DECODE=1`（旧实现：自写 `depthwise_conv_triton.py`）

- `output_tps = 71.99 tok/s`
- 相对 baseline 仅提升约 `0.1%`

更早之前的：

`MINISGL_MOE_SINGLE_KERNEL=1`

- `output_tps = 69.26 tok/s`
- 反而低于 baseline

`MINISGL_FULL_ATTN_FUSED_GATE_MUL=1`

- `output_tps = 86.55 tok/s`
- 相对 `FULL_ATTN_FUSED_PREPARE=1` 基本无变化

这说明在 `graph on + 单并发` 的真实服务场景下，下面这些方向都不是当前追赶 sglang 的主战场：

- decode 中 `a/b` 的外层 fp32 cast
- decode 中“另起一套自写 Triton conv 替代品”
- full attention 输出侧的 `sigmoid(gate)` elementwise
- MoE 单核融合

## 结合 sglang 对比后的优先级更新

当前最值得继续追的方向，不是继续做“哪里看起来慢就改哪里”，而是继续围绕 **sglang 已经 fused、mini-sglang 仍保留更多额外 kernel / 组合算子** 的部分。

### 1. linear-attn / MoE 周边仍然大量存在的额外 kernel

最新 graph-off trace 显示：

- `gemvx` 单次平均耗时已经和 sglang 几乎一样
- `decode kernel` 单次平均也不比 sglang 慢
- `fused add+rmsnorm` 单次平均也几乎一样

这说明剩余差距更像是 mini-sglang 仍然有更多：

- `cat/copy`
- `reduce`
- `sigmoid`
- 以及 MoE 多步周边 kernel

### 2. Q/K norm 路径

- mini-sglang 当前仍有显式 `F.normalize(...float...)`
- sglang 这部分已经更接近 fused / kernel 化路径
- 这块同时影响 prefill 和 linear attention 的准备阶段

### 3. QKVZ / BA split + reshape 路径

- sglang 有 fused split kernel
- mini-sglang 仍有 `split/slice/copy`

### 4. Gemma norm 周边的剩余 unfused 路径

- 当前 `Gemma fused norm` 只吃掉了一部分收益
- 还需要确认：
  - q_norm / k_norm 是否已全部吃到 fused 路径
  - residual add + norm 是否仍有未融合部分

### 5. MoE 路径单次平均仍慢于 sglang

最新 graph-off trace 显示：

- mini `fused_moe_kernel` 平均约 `33.03 us / 次`
- sglang `fused_moe_kernel` 平均约 `21.46 us / 次`

这说明即便在去掉很多周边开销之后，MoE 自身仍有一块纯 kernel 差距。

## 下一步建议

下一步应该继续坚持“和 sglang 做直接对比”的主线，而不是回到孤立猜测：

1. 用当前最佳 mini-sglang 配置：
   - `MINISGL_GEMMA_FUSED_NORM=1`
   - `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
   - `MINISGL_DEPTHWISE_CONV_DECODE=1`
   - `bf16`
   - `fi + sglang linear attention + CUDA graph`
2. 和 sglang 再做一次同口径对比
3. 重点核对：
   - 当前最佳 mini-sglang 相比 sglang，还多了哪些 kernel
   - 哪些 kernel 虽然名字相同，但调用次数更多
   - 哪些 norm / split / reshape / elementwise 路径仍未被吃掉
   - MoE 单次平均为何仍明显慢于 sglang

目标不是再做泛泛 profiling，而是明确回答：

**在已经修掉 CUDA graph correctness、补上 Gemma fused norm、full-attention fused prepare，以及 decode conv 对齐之后，mini-sglang 相比 sglang 还慢的那一大截，具体还剩在哪些步骤。**
