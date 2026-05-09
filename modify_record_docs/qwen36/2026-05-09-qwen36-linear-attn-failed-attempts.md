# 2026-05-09 Qwen3.6 线性注意力两次失败优化尝试记录

## 1. 背景

在前一轮已经完成：

- `Qwen3.6-35B-A3B` 跑通
- `fused MoE` 跑通
- `sglang linear attention` 跑通
- `w8a8_int8_moe_only` 跑通

并且已有一版比较稳定的性能参考：

- `short_prefill_short_decode` 约 `12.7 ~ 12.9 tok/s`
- `medium_prefill_short_decode` 约 `11.8 ~ 13.4 tok/s`

本轮目标是继续提升 `int8_moe_only + fused MoE + sglang linear attn` 的短 decode 吞吐，重点排查 `linear attention` 是否还有低成本可拿的收益。

## 2. 尝试一：重排 linear attention 的 SSM state 布局

### 2.1 原始想法

当前 mini-sglang 的 linear attention state 布局是：

- `[HV, K, V]`

而上游 `sglang` 的 packed decode kernel 更接近：

- `[HV, V, K]`

因此做了一个尝试：

1. 把 state cache 从 `[HV, K, V]` 改成 `[HV, V, K]`
2. 同步改写：
   - torch fallback 的 `einsum`
   - prefill Triton kernel
   - decode Triton kernel
3. 顺手去掉 decode 后的冗余 `copy_`

### 2.2 数值结果

最小对拍结果表明：

- `prefill output` 对齐到 `1e-7`
- `prefill state` 对齐到 `1e-7`
- `decode state` 对齐到 `1e-7`
- `decode output` 误差约 `3.6e-3`

结论：

- 这次改动在数值上是成立的
- 不是“算错了”

### 2.3 性能结果

但是 kernel microbench 结果直接说明了问题：

- 旧版 decode kernel：`0.0898 ms`
- 新版 decode kernel：`0.0972 ms`
- `speedup = 0.923`

也就是：

- 新布局不是更快，而是慢了约 `7.7%`

服务级 `stage1 benchmark` 也出现了明显退化和波动：

- 一轮约 `12.17 tok/s / 11.88 tok/s`
- 另一轮又掉到 `8.47 tok/s / 7.87 tok/s`

结论：

1. 这条路数值正确，但性能不成立
2. 对当前 A800 + Triton 配置来说，`[HV, V, K]` 不是更优布局
3. 该尝试已经回退，不保留到主路径

## 3. 尝试二：接入 fused_qkvzba_split_reshape_cat_contiguous

### 3.1 原始想法

参考上游 `sglang`：

- `python/sglang/jit_kernel/triton/gdn_fused_proj.py`

上游在 Qwen3.6 linear attention 前处理里有一条 fused path：

- `fused_qkvzba_split_reshape_cat_contiguous`

其目标是把下面这段前处理合成一个 Triton kernel：

- `mixed_qkvz.split(...)`
- `mixed_ba.split(...)`
- `z.view(...)`

因此本轮把这条 kernel 移植到 mini-sglang，尝试替换当前 Python/Torch 前处理。

### 3.2 数值结果

最小对拍结果是完全一致的：

- `qkv diff = 0.0`
- `z diff = 0.0`
- `b diff = 0.0`
- `a diff = 0.0`

说明：

- 这条 fused 前处理在语义上没有问题

### 3.3 microbench 结果

但单独测速后发现，这条 kernel 在 mini-sglang 当前上下文里明显更慢。

对比对象：

- baseline：原始 `split/view`
- fused：`fused_qkvzba_split_reshape_cat_contiguous`

代表性结果：

- `seq=1`：
  - baseline `65.1 us`
  - fused `218.4 us`
- `seq=32`：
  - baseline `73.0 us`
  - fused `143.1 us`
- `seq=64`：
  - baseline `64.7 us`
  - fused `237.6 us`
- `seq=256`：
  - baseline `21.6 us`
  - fused `109.0 us`

也就是：

- 全部测试点，fused 都慢于 baseline
- 并且不是小幅慢，而是明显慢

### 3.4 服务级结果

接入主路径后，整机 `stage1 benchmark` 进一步确认它不是有效优化：

- `short_prefill_short_decode` 约 `6.21 tok/s`
- `medium_prefill_short_decode` 约 `6.80 tok/s`

这比原先 `12+ tok/s` 级别明显更差。

结论：

1. 这条 fused 前处理在上游上下文里成立，不代表在 mini-sglang 当前实现里也划算
2. mini-sglang 当前 `split/view` 已经很便宜，没必要为了“看起来更 fused”而硬接 Triton kernel
3. 该尝试已经回退，不保留到主路径

## 4. 顺手确认的几个热点结论

为了避免继续盲改，还单独测了两个怀疑点：

- linear attention depthwise conv
- `a/b` 的 `bf16 -> fp32` cast

结果：

- depthwise conv 约 `0.029 ms`
- `a/b` cast 约 `24.8 us`

结论：

1. depthwise conv 不是当前 decode 主瓶颈
2. `a/b` cast 也不是主瓶颈
3. 后续不应该继续在这两个点上花太多时间

## 5. 当前判断

本轮两次尝试给出的最重要结论是：

1. 不是所有“更像上游 / 更像 fused”的改法都会带来 mini-sglang 当前环境下的真实收益
2. 当前 linear attention decode 的真正可疑点，已经不太像是：
   - state layout
   - qkv 前处理 split/view
   - a/b cast
   - depthwise conv
3. 下一步更值得投入的是：
   - 继续看 decode kernel 本体的 launch / occupancy / tile 参数
   - 或者回到 int8 路径，重点做更有结构性收益的 kernel fusion

## 6. 当前处理结果

本轮新增的两条实验路径都已经回退：

1. `SSM state` 改成 `[HV, V, K]` 的尝试已回退
2. `fused_qkvzba_split_reshape_cat_contiguous` 接入主路径的尝试已回退

因此当前主路径仍保持在上一版稳定可用的基础上继续迭代。
