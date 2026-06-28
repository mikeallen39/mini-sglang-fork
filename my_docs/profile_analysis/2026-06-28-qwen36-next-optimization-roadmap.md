# 2026-06-28 Qwen3.6 后续优化路线图

## 目标

当前目标不是继续零散地试单个小 kernel，而是系统性地回答：

- mini-sglang 在 Qwen3.6 `1024 final-chat / 64 out / 单并发 / 单卡 / bf16 / CUDA graph on` 下，为什么仍然明显慢于 sglang
- 接下来应当优先优化哪些部分，哪些方向收益大，哪些方向收益小
- 后续实验应按什么顺序推进，避免继续陷入“哪里看起来慢就改哪里”的低效循环

---

## 当前基线与最新进展

### 对标目标

- sglang bf16：
  - `TTFT = 106.40 ms`
  - `E2E = 0.5160 s`
  - `output_tps = 124.03 tok/s`

### mini-sglang 已确认有效的稳定优化

当前已经确认有效、且 correctness 正常的方向包括：

1. `MINISGL_GEMMA_FUSED_NORM=1`
2. `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
3. `MINISGL_DEPTHWISE_CONV_DECODE=1`
4. `MINISGL_LINEAR_RMSNORM_GATED=1`
5. `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`

### 当前最好稳定值

这里需要区分两个概念：

- 历史 session best：
  - `TTFT = 122.77 ms`
  - `E2E = 0.6561 s`
  - `output_tps = 96.02 tok/s`
- 当前工作树重新校准重跑值：
  - `TTFT = 122.70 ms`
  - `E2E = 0.6943 s`
  - `output_tps = 90.73 tok/s`

说明：

- `96.02 tok/s` 来自此前会话中的真实实验结果，只是当时没有补进仓库文档
- 2026-06-28 进一步排查后确认，主因是 `LINEAR_RMSNORM_GATED` 的实际接线从 Qwen3.6 linear-attn 输出 norm 路径上丢失了
- 把这条接线补回后，当前工作树已重新复现：
  - `93.34 tok/s`
  - `96.13 tok/s`
- 这说明问题不是“96.02 不存在”，而是“当前代码状态中有一条历史有效 fuse 被断开了”
- 后续分析应同时保留：
  - `96.02 tok/s` 作为历史最好成绩
  - `96.13 tok/s` 作为当前工作树重新接线后的可复现实测值

相对最初修复 correctness 后的 bf16 baseline：

- baseline：
  - `TTFT = 130.59 ms`
  - `E2E = 0.8763 s`
  - `output_tps = 71.89 tok/s`

相对 baseline：

- 按历史 session best 计：
  - `71.84 -> 96.02 tok/s`
  - 吞吐累计提升约 `33.7%`
- 按当前工作树恢复后重跑值计：
  - `71.84 -> 96.13 tok/s`
  - 吞吐累计提升约 `33.8%`

但距离 sglang 仍有差距：

- 按历史 session best 计：
  - `96.02 / 124.03 ≈ 77.4%`
- 按当前工作树恢复后重跑值计：
  - `96.13 / 124.03 ≈ 77.5%`

也就是说：

- 如果以历史 best 为参考，还剩大约 `28 tok/s`
- 如果以当前工作树恢复后重跑值为参考，还剩大约 `28 tok/s`

---

## 当前最重要的总体判断

### 1. 剩余差距已经不主要在 attention / decode 核心算子本体

已有 graph-off trace 对比表明：

- `gemvx` 单次平均耗时已和 sglang 非常接近
- linear attention decode kernel 单次平均也不明显慢
- fused add+rmsnorm 单次平均也接近

因此：

- 继续围绕“attention 核心算子本体”做细抠，不太可能再带来最大的增益
- 当前剩余差距更像是**整层边界上的 prepare / epilogue / glue path** 还没有像 sglang 那样融合得足够彻底

### 2. 最近收益最大的方向，已经验证就是“整层边界融合”

已经发生的事实：

- `Gemma fused norm` 有明显收益
- `full-attn fused prepare` 有明显收益
- `linear RMSNorm gated fused` 有明显收益
- `shared expert fused gate add` 继续有收益

这些优化的共性是：

- 不是去改 attention 核心递推公式
- 不是去改一个局部小 activation
- 而是把一整段 layer-boundary 的 `norm / gate / add / mul / split / rope / prepare` 收缩成更少 kernel

这说明：

- 现在最值得继续追的不是“算子级微优化”
- 而是**整层执行链上的 glue path 融合**

---

## 现有 profile 归因

基于已有 graph-off trace，mini-sglang 剩余开销的核心大类可以概括为：

### A. MoE 主体与周边

- `fused_moe_kernel`
- `moe_sum_reduce`
- routing 后的 `topk / align / dispatch`
- shared expert gate / add

结论：

- MoE 仍然重要
- 但不能只盯 `fused_moe_kernel` 一个 kernel
- 更合理的是把它当成 **MoE 整条执行链** 看：
  - route
  - gate/up
  - activation
  - down
  - combine/reduce
  - shared expert epilogue

### B. attention / linear-attn / full-attn 的 prepare 路径

- `split / reshape / cat / contiguous / copy`
- q/k norm
- rope 前后准备
- gate 路径

结论：

- 这类 prepare/unfused copy 在 mini-sglang 中仍然偏多
- sglang 已经通过更多 fused kernel 把这部分压平

### C. 层边界 epilogue

- `sigmoid`
- `silu`
- `mul`
- `add`
- `rsqrt`
- `mean`

结论：

- 这是当前最值得继续打的方向之一
- 因为它们不是只出现在一两个地方，而是会在多层中反复出现

### D. unfused norm / reduce

- `MeanOps`
- `rsqrt`
- 各类 norm-related reduce

结论：

- 仍然还有剩余 unfused norm 路径
- 但方向已经很清楚：优先找“整段可融合”的点，而不是再一个个替换低级算子

---

## 已尝试过但不应作为第一优先级继续深挖的方向

这些方向要么收益很小，要么 correctness 风险较高，要么没有打中主矛盾。后续可以保留为备选，但不应排在最前面。

### 1. 单个小 kernel 级微调

- `MINISGL_SKIP_AB_FP32_CAST`
- 旧版 `MINISGL_DEPTHWISE_CONV_DECODE`
- `MINISGL_FULL_ATTN_FUSED_GATE_MUL`
- `MINISGL_MOE_FUSED_ACTIVATION`

结论：

- 收益小，或者没有收益
- 不适合作为当前主线

### 2. 只盯 `fused_moe_kernel` 单点

- `MINISGL_MOE_SINGLE_KERNEL`

结论：

- 之前尝试过，graph-on 场景下效果不好
- 说明 MoE 的问题不是“再压一个局部小融合”就能解决

### 3. 过度依赖环境层面的现成 op

例如：

- 直接依赖 `flashinfer/sgl_kernel` 某些路径就能自动变快

结论：

- 环境兼容性并不稳定
- 更可控的方式仍然是把关键 fused 路径自己接到 mini-sglang 代码里，并通过开关做消融

---

## 后续优化方向总表

下面按优先级给出建议的后续路线。

---

## 第一优先级：继续做整层边界融合

### 方向 1. shared-expert 之外的剩余 gate/add/mul epilogue

目标：

- 继续减少 profile 中剩余的：
  - `sigmoid`
  - `mul`
  - `add`
  - `vectorized_elementwise`

原因：

- 这类 kernel 仍然很多
- 最近两次最成功的优化（`LINEAR_RMSNORM_GATED`、`SHARED_EXPERT_FUSED_GATE_ADD`）都说明这条线是有效的

建议实验：

1. 梳理 Qwen3.6 路径里所有 `sigmoid(gate) * x`、`x + y`、`F.silu(gate) * x` 的位置
2. 评估是否还能进一步合并到已有 fused kernel 或新增一个更宽的 epilogue kernel
3. 每次只替一个边界段，保持可开关

预期收益：

- 中等
- 风险较低
- 很适合做单因素实验

### 方向 2. full-attn / linear-attn prepare 的剩余 `copy/cat/split`

目标：

- 继续减少 `attn_prepare_unfused`

原因：

- 当前 profile 里 `copy/cat/split/contiguous` 仍然是大头
- sglang 明显在这部分更激进地使用 fused prepare kernel

建议实验：

1. 继续对齐 `QKVZ / BA split + reshape + cat` 路径
2. 查清 `direct_copy_kernel_cuda` 和 `CatArrayBatchedCopy` 的来源
3. 找出最常见的输入/输出 layout，做一条更完整的 fused prepare 路径

预期收益：

- 中到大
- 但实现复杂度高于 epilogue 类优化

### 方向 3. 剩余 unfused norm 路径

目标：

- 继续减少：
  - `MeanOps`
  - `rsqrt`
  - 其他 norm-related reduce

原因：

- `LINEAR_RMSNORM_GATED` 已经说明这种“整段 norm + gate 融合”是有收益的
- 说明系统里还存在类似结构，值得继续找

建议实验：

1. 盘点当前仍然走 `forward_native` 或纯 `torch` norm 的路径
2. 优先挑覆盖层数多、调用频率高的点
3. 保持开关化和最小入侵

预期收益：

- 中等
- 比较稳

---

## 第二优先级：按“整条链”继续优化 MoE

这里不建议继续以“单个 MoE kernel 微调”为主，而是按整条执行链看。

### 方向 4. MoE combine/reduce 重新对齐

现象：

- `MOE_SGL_REDUCE` 路径曾经出现过小幅吞吐提升
- 但 correctness 出现异常

说明：

- reduce/combine 方向不是没价值
- 而是当前接法没有完全对齐语义

建议实验：

1. 明确 `intermediate_cache3` 的布局与 `sgl_kernel.moe_sum_reduce` 的预期是否一致
2. 对齐 `routed_scaling_factor` 语义
3. 单独验证 correctness，再测吞吐

预期收益：

- 小到中等
- 若修对，可能是 MoE 周边里比较干净的一段收益

### 方向 5. MoE route / dispatch / align 开销

关注点：

- `topkSoftmax`
- `moe_align_block_size`
- `count_and_sort_expert_tokens`

原因：

- 这些不是最大头，但调用很多
- 可能还有布局与中间 tensor 管理上的差异

建议实验：

1. 对比 mini 与 sglang 在 route 后 `topk_ids/topk_weights` 的格式流转
2. 看是否存在多余的 dispatch plan 构造或额外 copy
3. 只在确认存在结构性冗余后再动

预期收益：

- 小到中等
- 优先级低于整层边界融合

### 方向 6. `fused_moe_kernel` 单次慢于 sglang 的根因

已知现象：

- mini 单次 `fused_moe_kernel` 仍慢于 sglang

但结论：

- 这是值得分析的问题
- 但不应放在当前最高优先级

原因：

- 目前更大的收益已经反复来自“边界融合”
- 单盯一个核心 kernel 容易回到收益很不稳定的微优化陷阱

更合理的做法：

- 在整层边界继续清理一轮之后，再回头重新评估这个差距是否仍然显著

---

## 第三优先级：attention 核心本体相关

目前这部分不应作为第一优先级，理由是：

- `gemvx` 单次已经接近
- decode core kernel 单次已经接近
- 最近最有效的收益都不是从这里来的

因此：

- attention 核心本体仍可分析
- 但当前阶段不应优先投入

---

## 推荐实验顺序

为了避免后续实验再次发散，建议按下面顺序推进。

### 第 1 阶段：继续清理整层边界 epilogue

依次尝试：

1. 盘点剩余 `sigmoid/mul/add` 路径
2. 选覆盖层数最多的一个点做 fused
3. 做 correctness + benchmark

### 第 2 阶段：继续压 `attn_prepare_unfused`

依次尝试：

1. 查 `direct_copy_kernel_cuda` 与 `CatArrayBatchedCopy` 的来源
2. 选择最频繁的 prepare 片段
3. 做一条更宽的 fused prepare

### 第 3 阶段：回到 MoE 整条链

依次尝试：

1. 修 `combine/reduce`
2. 再看 route / dispatch
3. 最后再看 `fused_moe_kernel` 单次差距

---

## 每次实验的记录要求

建议后续每次实验都固定记录以下内容：

1. 开关组合
2. correctness 结果
3. `TTFT`
4. `E2E`
5. `output_tps`
6. 相对当前最好值的变化
7. 为什么这个方向理论上可能有效
8. 为什么它最终有效或无效

这样可以避免后续实验记录碎片化，便于逐步收敛。

---

## 当前建议的一句话总结

接下来最该做的，不是继续抠单个 attention 或 MoE 小 kernel，而是继续沿着**整层边界上的 prepare / epilogue / unfused norm 融合**这条主线推进；这已经被最近几轮实验反复证明是收益最大的方向。
