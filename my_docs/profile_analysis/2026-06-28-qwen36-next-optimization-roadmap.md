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
- 当前工作树最新最好值：
  - `TTFT = 114.02 ms`
  - `E2E = 0.6235 s`
  - `output_tps = 101.04 tok/s`

说明：

- `96.02 tok/s` 来自此前会话中的真实实验结果，只是当时没有补进仓库文档
- 2026-06-28 进一步排查后确认，主因是 `LINEAR_RMSNORM_GATED` 的实际接线从 Qwen3.6 linear-attn 输出 norm 路径上丢失了
- 把这条接线补回后，当前工作树已重新复现：
  - `93.34 tok/s`
  - `96.13 tok/s`
- 在此基础上继续沿 roadmap 推进，`MOE_FUSED_ACTIVATION=1` 又把当前最好值进一步推到：
  - `98.56 tok/s`
- 继续对 shared expert 内部 bf16 activation 做 fused 后，当前最好值进一步推到：
  - `99.19 tok/s`
- 再把 linear-attention prefill 的 Q/K L2Norm 下沉到 kernel 内部后，当前最好值进一步推到：
  - `100.17 tok/s`
- 再去掉 linear-attention prefill 中冗余的 `contiguous` 包装后，当前最好值进一步推到：
  - `100.59 tok/s`
- 再打开 `SKIP_AB_FP32_CAST=1` 后，当前最好值进一步推到：
  - `101.04 tok/s`
- 与此同时，`DEPTHWISE_CONV_PREFILL=1` 在当前接法下退化到：
  - `98.37 tok/s`
  说明这条线暂时不应作为主线继续投入
- 这说明问题不是“96.02 不存在”，而是“当前代码状态中有一条历史有效 fuse 被断开了”
- 后续分析应同时保留：
  - `96.02 tok/s` 作为历史最好成绩
  - `101.04 tok/s` 作为当前工作树继续优化后的最新最好值

相对最初修复 correctness 后的 bf16 baseline：

- baseline：
  - `TTFT = 130.59 ms`
  - `E2E = 0.8763 s`
  - `output_tps = 71.89 tok/s`

相对 baseline：

- 按历史 session best 计：
  - `71.84 -> 96.02 tok/s`
  - 吞吐累计提升约 `33.7%`
- 按当前工作树最新最好值计：
  - `71.84 -> 101.04 tok/s`
  - 吞吐累计提升约 `40.6%`

但距离 sglang 仍有差距：

- 按历史 session best 计：
  - `96.02 / 124.03 ≈ 77.4%`
- 按当前工作树最新最好值计：
  - `101.04 / 124.03 ≈ 81.5%`

也就是说：

- 如果以历史 best 为参考，还剩大约 `28 tok/s`
- 如果以当前工作树最新最好值为参考，还剩大约 `25 tok/s`
- 如果以当前工作树最新最好值为参考，还剩大约 `23.0 tok/s`

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

---

## 2026-06-29 Decode 主线更新

在 `101.04 tok/s` 之后，围绕 decode 主线连续做了几次“更贴近 sglang”或“更贴近 kernel fusion” 的单因素实验，结论如下。

### 已验证有效

1. `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`

- 将 decode 时分开的：
  - `in_proj_qkvz`
  - `in_proj_ba`
  合并为一次 bf16 GEMM
- 结果：
  - `101.04 -> 102.17 tok/s`
- 结论：
  - decode 输入边界上的 kernel/launch 组织确实是剩余差距来源之一
  - 这是当前 decode 主线里更值得保留的有效项

2. `MINISGL_LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ=1`

- 对齐 sglang `_forward_input_proj` 的主流/辅流双流组织
- 结果：
  - `102.17 -> 102.29 tok/s`
- 结论：
  - 方向正确，但在 `bs=1` 下增益极小
  - 说明 input projection 这块的 overlap 空间已经接近吃干净

3. `MINISGL_LINEAR_RMSNORM_GATED_REUSE_OUT=1`

- 在 fused `RMSNorm+gate` 路径中复用输出 buffer
- 结果：
  - `102.17 -> 102.51 tok/s`
- 结论：
  - decode gated norm 还有少量可挖空间
- 但“减少输出分配”本身不是大头

4. `MINISGL_LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS=1`

- 跳过 decode 时 `out_proj` 前的显式 `.contiguous()`
- 结果：
  - `102.51 -> 102.71 tok/s`
- 结论：
  - `norm -> out_proj` 边界仍存在少量可回收的布局整理成本

5. `MINISGL_LINEAR_DECODE_SKIP_AB_CONTIGUOUS=1`

- 跳过 decode 时 `a/b` 的显式 `.contiguous()`
- 结果：
  - `102.71 -> 103.26 tok/s`
- 结论：
  - decode kernel 前的 `a/b` 复制是可观察到的真实冗余
  - 当前最好稳定值更新为 `103.26 tok/s`

### 已证伪 / 降级

1. `MINISGL_LINEAR_DECODE_VK_STATE=1`

- 仅把 decode state layout 对齐成 `[HV, V, K]`
- 结果：
  - 退化到 `97.01 tok/s`
- 结论：
  - 剩余差距并不是由 state layout 单独决定

2. `MINISGL_LINEAR_DECODE_SGLANG_PACKED=1`

- 直接复用 sglang packed recurrent decode kernel
- 结果：
  - 退化到 `96.92 tok/s`
- 结论：
  - 剩余差距不是“只换 recurrent kernel 本体”就能解决

3. `MINISGL_LINEAR_RMSNORM_GATED_SGLANG=1`

- 直接切到 sglang `rms_norm_gated`
- 结果：
  - `103.26 -> 102.45 tok/s`
- 结论：
  - 当前 mini-sglang 自己这版 gated RMSNorm 并不比 sglang 差
  - `norm` 不是下一步最值得继续对齐的主方向

4. `MINISGL_LINEAR_DECODE_SGLANG_UPDATE=1`

- 直接切到 sglang 常规 decode 所用的
  `fused_sigmoid_gating_delta_rule_update`
- 结果：
  - `103.26 -> 98.72 tok/s`
- 结论：
  - 即使不走 packed decode，只换成 sglang 常规 recurrent update kernel 也明显退化
  - decode 主差距并不主要来自 recurrent update kernel 本体

5. `MINISGL_LINEAR_DECODE_SKIP_CONV_STATE_COPY=1`

- 跳过 decode fused conv 路径后的额外 `conv_state.copy_`
- 结果：
  - `103.26 -> 102.93 tok/s`
- 结论：
  - decode conv state copy 不是主矛盾
  - 这类单点 state-copy 清理不足以继续逼近 sglang

6. `MINISGL_LINEAR_DECODE_FUSED_QKV_SPLIT=1`

- 只在 decode 启用 `fused_qkvzba_split_reshape_cat_contiguous`
- 结果：
  - `103.26 -> 101.70 tok/s`
- 结论：
  - 即使去掉 prefill 干扰，当前 fused split/reshape kernel 仍没有带来端到端收益
  - `qkvz + split/reshape` 并不是“直接启用现有 fused kernel”就能补齐的差距

### 当前 decode 分段判断

基于最新更优组合：

- `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`
- `MINISGL_LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS=1`
- `MINISGL_LINEAR_DECODE_SKIP_AB_CONTIGUOUS=1`

的 graph-off 阶段化 profile，稳定单层 decode 开销大致为：

- `qkvz ≈ 0.062 ms`
- `ba ≈ 0.003 ms`
- `conv ≈ 0.050 ms`
- `kernel ≈ 0.103~0.104 ms`
- `norm ≈ 0.066 ms`
- `out_proj ≈ 0.050~0.051 ms`

这说明：

- `ba` 已经基本被打平，不应再把 input proj 当成第一优先级
- 目前 decode 主线剩余的更大块是：
  - `kernel`
  - `norm`
  - `qkvz / out_proj / conv`

### 下一优先级更新

结合上面的实验与 sglang 代码对齐结果，下一步更合理的顺序应调整为：

1. 下一步不应继续优先围绕 linear-attn decode 单段做直接对齐开关实验
2. 更合理的是回到整个 model 层面重新看剩余差距，尤其是 linear-attn 之外的 MoE / residual / layer 边界累计成本
3. 不再继续优先追 `input proj`、`state layout`、`packed recurrent kernel`、`sglang gated RMSNorm`、`sglang 常规 recurrent update kernel`、`conv state copy`、`decode-only fused split` 这些已经证伪或收益很小的线

---

## 2026-06-29 整模型重新归因

在当前最好稳定组合附近，额外开启：

- `MINISGL_PROFILE_QWEN35=1`
- `MINISGL_PROFILE_SPARSE_MOE=1`
- `MINISGL_PROFILE_FUSED_MOE=1`

并在 `graph off` 下重跑整模型 profile。

### 关键观察

1. Linear-attn decode 单层稳定大致为：

- `qkvz ≈ 0.062 ms`
- `ba ≈ 0.003 ms`
- `conv ≈ 0.048~0.049 ms`
- `kernel ≈ 0.101~0.103 ms`
- `norm ≈ 0.065~0.066 ms`
- `out_proj ≈ 0.049 ms`

合计约：

- `~0.33 ms / linear-attn layer-call`

2. SparseMoE 单层稳定大致为：

- `router ≈ 0.041 ms`
- `experts ≈ 0.53~0.54 ms`
- `shared ≈ 0.146 ms`

合计约：

- `~0.72 ms / MoE layer-call`

3. FusedMoE 内部 routed experts 稳定大致为：

- `w1 ≈ 0.102 ms`
- `stage2 ≈ 0.023 ms`
- `w2 ≈ 0.072 ms`
- `reduce ≈ 0.059 ms`

### 结论

这说明在当前代码状态下：

- **MoE 层的单层成本已经明显高于 linear-attn decode 层**
- 剩余差距如果继续只盯 linear-attn decode，性价比会越来越低
- 下一阶段更值得优先投入的是：
  - routed experts 主链
  - shared expert
  - MoE reduce / epilogue

### 下一优先级调整

1. 主线切到 MoE
2. 优先看 routed experts 的 `w1 / w2 / reduce`
3. 其次看 shared expert 路径
4. linear-attn decode 暂时降级为次优先级，除非后续发现新的整段融合点

---

## 2026-06-29 MoE 主线新结论

在完成整模型重新归因后，围绕 MoE 主线继续做了 4 组单因素实验。

### 已证伪

1. `MINISGL_SHARED_EXPERT_DUAL_STREAM=1`

- 思路：
  - 对齐 sglang 常见的 shared expert 双流重叠
  - 让 shared expert 与 routed experts 并行
- 结果：
  - `103.26 -> 102.26 tok/s`
- 结论：
  - 当前单并发、graph on 场景下，shared expert 双流没有转化成端到端收益
  - 这不是当前最值得继续深挖的方向

2. `MINISGL_MOE_REUSE_WORKSPACE=1`

- 思路：
  - 复用 fused MoE 中 `topk/alignment` 的临时张量
- 结果：
  - `103.26 -> 103.20 tok/s`
- 结论：
  - MoE 路径里的小张量分配并不是当前端到端主矛盾

### 有效优化

1. `MINISGL_MOE_SKIP_TOPK_POST_RENORM=1`

- 发现：
  - mini-sglang 调 `sgl_kernel.topk_softmax(..., renormalize=True)` 后
  - 仍在 Python 侧额外做一次 `topk_weights /= sum(topk_weights)`
- 结果：
  - `103.26 -> 104.72 tok/s`
- 结论：
  - 这是一段真实存在的重复工作
  - MoE router 热路径因此拿到明显收益

2. `MINISGL_MOE_SKIP_TOPK_FP32_CAST=1`

- 发现：
  - mini-sglang 调 `topk_softmax` 前还会额外做 `router_logits.float()`
  - sglang 主线直接传原 dtype
- 与 `MINISGL_MOE_SKIP_TOPK_POST_RENORM=1` 叠加后结果：
  - `104.72 -> 106.62 tok/s`
- 结论：
  - 这条额外 dtype cast 同样会转化成真实端到端开销
  - MoE router 路径目前已经成为最明确、最有效的优化来源

3. `MINISGL_MOE_SKIP_DISPATCH_LOCAL_MASK=1`

- 发现：
  - `build_local_expert_dispatch_plan()` 在单卡 fast path 中会额外构造
    `torch.ones_like(topk_ids, dtype=torch.bool)`
  - 但 fused MoE 主线并不消费这个 `local_mask`
- 与前两条 MoE router 优化叠加后结果：
  - `106.62 -> 107.17 tok/s`
- 结论：
  - 单卡 dispatch fast path 里仍有少量无用分配
  - 这类小胶水清理还能继续拿到稳定小收益

4. `MINISGL_MOE_ALIGN_SMALL_CAP=1`

- 发现：
  - mini-sglang 在 `moe_align_block_size()` 中仍使用更松的临时张量上界
  - sglang 在 `topk_ids.numel() < num_experts + 1` 时会改用更小的 `topk_ids.numel() * block_size`
- 与前三条 MoE router / dispatch 优化叠加后结果：
  - `107.17 -> 109.16 tok/s`
- 结论：
  - `moe_align_block_size` prepare 路径仍然是可见的真实瓶颈
  - 仅仅收紧小 token 场景的 buffer 上界，就能带来接近 `+2 tok/s` 的收益

5. `MINISGL_MOE_SGLANG_CONFIG_LOOKUP=1`

- 发现：
  - mini-sglang 当前 routed-expert Triton config 选择仍是极简 heuristic
  - sglang 主线会优先命中 JSON 调优表，再回退默认 heuristic
  - 对 Qwen3.6 routed experts 的 `E=256, N=512` 形状，sglang 确实有现成 tuned config
- 这次实际命中的关键 config 差异：
  - 原 heuristic 在 `M <= E` 时固定为
    `BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1`
  - 对 decode 最关键的 `M=1`，回退命中的 sglang config 变为
    `BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=4`
  - 对较大 `M`，例如 `M=1024`，原 heuristic 是
    `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8`
  - 命中的 sglang config 则变为
    `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4, num_stages=4`
- 与前四条 MoE 主线优化叠加后结果：
  - `109.16 -> 111.15 tok/s`
- 结论：
  - routed-expert kernel config 选择差异是当前 mini-sglang 落后于 sglang 的一个真实来源
  - 即使本机没有 `A800 PCIe` 专属配置，单靠 sglang 风格的 config lookup 与版本/设备回退，也能拿到明显收益
  - 说明当前差距并不只是 Python 胶水，kernel tile / launch config 本身也仍有优化空间

6. `MINISGL_MOE_SGLANG_DOWN_CONFIG=1`

- 发现：
  - mini-sglang 第二段 `w2/down_proj` GEMM 仍然直接复用第一段 `gate_up` GEMM 的 config
  - sglang 主线则会为第二段单独查 `_down.json`
  - 对当前 `E=256, N=512` 形状，down-config 与 up-config 的差异是实质性的：
    - `M=1` 时从 `N=64, K=128, stages=4` 变成 `N=32, K=256, stages=2`
    - `M=1024` 时从 `N=64` 变成 `N=128`
- 与前五条 MoE 主线优化叠加后结果：
  - `111.15 -> 112.08 tok/s`
- 结论：
  - routed-expert 第二个 GEMM 的最佳 config 确实不同于第一段
  - 这部分差异同样会转化成可见端到端收益

7. `MINISGL_MOE_SGL_REDUCE=1` 复测

- 重新定位后发现：
  - 早先的 correctness fail 不是 `sgl_kernel.moe_sum_reduce` 自身错误
  - 而是 mini-sglang 当前接法把第三个参数硬编码成了 `0.0`
  - sglang 主线传的是 `routed_scaling_factor`
- 修复后结果：
  - 短输出 correctness 恢复正常
  - benchmark 大致在 `112.04 ~ 112.28 tok/s`
- 结论：
  - 这条线已经从“错误实验”变成了“已修复、当前收益不明显但可继续评估”的候选项
  - 也说明当前剩余差距未必主要在 MoE reduce kernel 本体，而更可能还是在更高层的整段组织

### 当前最好稳定值

当前主线最好稳定值更新为：

- `TTFT = 100.47 ms`
- `E2E = 0.5621 s`
- `output_tps = 112.08 tok/s`

补充：

- 从“整个推理引擎”角度做了一轮 decode graph replay 对比后，新增一个已验证的小收益实验：
  - `MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1`
  - 结果：
    - `TTFT = 100.22 ms`
    - `E2E = 0.5593 s`
    - `output_tps = 112.65 tok/s`
- 这说明 attention metadata / planner 的 graph replay 组织方式确实是 mini-sglang 落后于 sglang 的一个来源
- 但收益只有 `+0.57 tok/s`，不能解释剩余的大部分差距

相对 sglang `124.03 tok/s`：

- 已达到约 `90.4%`

### 对后续优化方向的影响

这批实验说明：

1. 当前最值得继续追的是 **MoE router / dispatch / prepare** 这条线
2. `topk` 前后的额外 Python 胶水开销已经被证明是真实收益点
3. `moe_align_block_size`、第一段 routed-expert config、第二段 routed-expert down-config，都已经被证明是当前主线里的真实收益点
4. `MOE_SGL_REDUCE` 修复后 correctness 正常，但目前没有显示出明显大于噪声的收益
5. 相比之下，shared expert 双流和 workspace 复用都不是当前主矛盾
6. 从引擎级对比看，decode graph replay 的 attention metadata/planner 路径确实存在设计差异，但单独对齐 `fast_decode_plan` 只能带来小收益，说明它不是剩余差距的主来源

### 下一优先级更新

1. 继续沿 **MoE prepare / routed-expert kernel selection** 主线推进
   - 重点看：
     - `moe_align_block_size`
     - routed-expert Triton config 选择
     - second-GEMM down-config
     - device-name / version fallback 是否还能更贴近本机
2. 再回头排查 MoE router / dispatch 路径中是否还有额外 Python 胶水
3. shared expert 与 linear-attn decode 继续降级，除非出现新的整段收益点
4. 如果继续走“整个推理引擎”主线，应优先重新归因：
   - graph replay 外的 batch/metadata 边界
   - residual / layer glue
   - 非 linear-attn / 非 routed-expert 的整层累计开销
