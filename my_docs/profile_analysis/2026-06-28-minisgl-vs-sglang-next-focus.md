# 2026-06-28 mini-sglang vs sglang 下一步追赶重点

## 当前状态

在修复 Qwen3.6 CUDA graph 正确性问题后，mini-sglang 的 `bf16 + fi + sglang linear attention + CUDA graph` 基线为：

- `TTFT = 130.59 ms`
- `E2E = 0.8763 s`
- `output_tps = 71.89 tok/s`

当前同口径、同 workload 的 sglang 参考值为：

- `TTFT = 106.40 ms`
- `E2E = 0.5160 s`
- `output_tps = 124.03 tok/s`

也就是：

- `TTFT` 仍慢约 `22.7%`
- `E2E` 仍慢约 `69.8%`
- `output_tps` 只有 sglang 的约 `58%`

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

### 已确认不是主瓶颈

`MINISGL_SKIP_AB_FP32_CAST=1`

- `output_tps = 72.31 tok/s`
- 相对 baseline 仅提升约 `0.6%`

`MINISGL_DEPTHWISE_CONV_DECODE=1`

- `output_tps = 71.99 tok/s`
- 相对 baseline 仅提升约 `0.1%`

更早之前的：

`MINISGL_MOE_SINGLE_KERNEL=1`

- `output_tps = 69.26 tok/s`
- 反而低于 baseline

这说明在 `graph on + 单并发` 的真实服务场景下，下面这些方向都不是当前追赶 sglang 的主战场：

- decode 中 `a/b` 的外层 fp32 cast
- decode 中的 depthwise conv
- MoE 单核融合

## 结合 sglang 对比后的优先级更新

当前最值得继续追的方向，不是继续做“哪里看起来慢就改哪里”，而是继续围绕 **sglang 已经 fused、mini-sglang 仍是 PyTorch 组合算子** 的部分：

1. **Q/K norm 路径**
   - mini-sglang 当前仍有显式 `F.normalize(...float...)`
   - sglang 这部分已经更接近 fused / kernel 化路径
   - 这块同时影响 prefill 和 linear attention 的准备阶段

2. **QKVZ / BA split + reshape 路径**
   - sglang 有 fused split kernel
   - mini-sglang 仍有 `split/slice/copy`

3. **Gemma norm 周边的剩余 unfused 路径**
   - 当前 `Gemma fused norm` 只吃掉了一部分收益
   - 还需要确认：
     - q_norm / k_norm 是否已全部吃到 fused 路径
     - residual add + norm 是否仍有未融合部分

## 下一步建议

下一步应该回到“和 sglang 做直接对比”的主线，而不是继续孤立地做 mini-sglang 内部猜测：

1. 用当前最佳 mini-sglang 配置：
   - `MINISGL_GEMMA_FUSED_NORM=1`
   - `bf16`
   - `fi + sglang linear attention + CUDA graph`
2. 和 sglang 再做一次同口径对比
3. 重点核对：
   - 当前最佳 mini-sglang 相比 sglang，还多了哪些 kernel
   - 哪些 kernel 虽然名字相同，但调用次数更多
   - 哪些 norm / split / reshape / elementwise 路径仍未被吃掉

目标不是再做泛泛 profiling，而是明确回答：

**在已经修掉 CUDA graph correctness、并补上 Gemma fused norm 之后，mini-sglang 相比 sglang 还慢的那一大截，具体还剩在哪些步骤。**
