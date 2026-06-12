# 2026-05-09 Qwen3.6 Int8 MoE `stage2 + w2` 融合尝试

## 1. 目标

在已经确认下面两条路都不值得继续之后：

- `linear attention` state 改成 `[HV, V, K]`
- `fused_qkvzba_split_reshape_cat_contiguous`

本轮重新回到 `int8` 主线，继续寻找真正有结构性收益的融合点。

当前 `int8 routed expert` 的主路径是：

1. `w1 int8 fused_moe_kernel`
2. `silu_and_mul_quant_int8_triton`
3. `w2 int8 fused_moe_kernel`
4. `moe_sum_reduce_triton`

其中第 2、3 步之间存在一个明显的中间写回：

- `intermediate_cache1` 写出 `w1` 输出
- `silu_and_mul_quant_int8_triton` 再把结果量化到 `intermediate_cache2`
- `w2` 再读取 `intermediate_cache2`

本轮要做的是把第 2、3 步合起来。

## 2. 本轮实现

新增了一条仅用于：

- `w2.dtype == int8`
- `activation == silu`

的融合路径：

- 在 `w2` Triton kernel 内部直接完成：
  - `gate/up -> silu_and_mul`
  - per-token int8 quant
  - int8 `w2` GEMM

对应文件：

- `python/minisgl/kernel/triton/fused_moe.py`
- `python/minisgl/kernel/moe_impl.py`
- `python/minisgl/kernel/__init__.py`
- `python/minisgl/moe/fused.py`

新增内部开关：

- `_ENABLE_FUSED_W2_SILU_INT8`

用于在同一份代码里稳定做 A/B。

## 3. 中间踩坑

### 3.1 第一个问题：workspace 别名冲突

第一次接上后，最小自一致性测试出现大误差：

- `self_diff = 160384.0`

原因不是 kernel 计算本身错误，而是：

- `intermediate_cache1`
- `intermediate_cache3`

共用了同一块底层 workspace 切片。

旧路径下这通常不是问题，因为阶段是串行且读写顺序安全；但新融合路径在读取 `intermediate_cache1` 的同时写 `intermediate_cache3`，两者发生了别名覆盖。

修复方式：

- 为 `intermediate_cache3` 单独申请 `stage3_out` workspace

修复后：

- `self_diff = 0.0`

### 3.2 第二个问题：第一次 A/B 基线选错

第一次做 A/B 时，`flag=False` 意外走成了：

- `fp stage2 + 通用量化`

而不是原始对照组：

- `silu_and_mul_quant_int8_triton + old w2 int8 kernel`

所以那次出现的大误差并不代表新融合 kernel 错了，只是基线选错。

后续修正为：

- `flag=False`：旧 `int8 stage2 + old w2`
- `flag=True`：新融合 `stage2 + w2`

## 4. 数值结果

在正确基线下：

- `max_abs_diff = 0.0`
- `mean_abs_diff = 0.0`
- `max_rel_diff = 0.0`
- `mean_rel_diff = 0.0`

说明：

- 新融合 `stage2 + w2` 与旧 `int8 stage2 + old w2` 数值完全一致

## 5. microbench 结果

同一组输入下，A/B 结果如下：

- 旧路径：`1.4416 ms`
- 新路径：`1.3936 ms`
- `speedup = 1.034`

也就是：

- 新融合路径在 microbench 上大约快 `3.4%`

## 6. 当前结论

截至当前，本轮 `int8 MoE` 融合尝试已经得到两个明确结论：

1. 这条 `stage2 + w2` 融合在数值上成立
2. 它在 microbench 上确实有正收益，量级约 `3% ~ 4%`

因此这条路和前面几次失败尝试不同：

- 它不是“看起来更 fused，但实际更慢”
- 它是一个可以继续推进到整机 benchmark 的有效方向

## 7. 下一步

下一步要做的事情只有一个：

1. 起服务
2. 跑 `Qwen3.6 stage1 benchmark`
3. 对比这次 `int8_moe_only + fused MoE + sglang linear attn` 是否比上一版再提升一点

如果整机也能稳定提升，再继续考虑：

- 是否还能把 `reduce` 一并并入后续路径
- 或者继续找 `w1` 之后还能减少的中间写回
