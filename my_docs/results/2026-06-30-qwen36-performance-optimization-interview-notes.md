# Qwen3.6 单卡性能优化面试复习版

## 1. 这份文档的用途

这不是原始实验流水账，而是把 [2026-06-10-qwen36-1024in-64out-stepwise-benchmark.md](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/2026-06-10-qwen36-1024in-64out-stepwise-benchmark.md) 中“从最差到最好”的优化过程，整理成一份更适合面试复习的版本。

适用场景：

- 面试时快速回忆优化主线
- 复盘为什么某些优化收益大、某些收益小
- 提炼可以讲清楚的工程判断，而不是背实验流水账

统一口径：

- 模型：`Qwen3.6-35B-A3B`
- workload：`1024 final-chat in / 64 out / 单卡 / 单并发`
- 关注指标：
  - `TTFT`
  - `output_tps`

---

## 2. 一句话总结

这次优化的核心不是“单个 kernel 魔法提速”，而是：

- 先把执行路径从 `torch` 主链切到更高效的 `fused / sglang / graph`
- 再沿着 **整层边界融合**
- 再沿着 **MoE router / dispatch / config / reduce 周边**
- 最后把 `mini-sglang` 从最初 `1.64 tok/s` 推到 `114.01 tok/s`
- 同时把首 token 延迟压到 `99.06 ms`

也就是：

- 从几乎不可用的基线
- 优化到接近 `sglang bf16 124.03 tok/s`
- 当前达到约 `91.9%`

---

## 3. 从最差到最好的主线阶段

在面试里建议始终用“双指标视角”来讲：

- `TTFT`
  - 更偏 prefill、图外边界、首 token 之前的系统开销
- `TPS`
  - 更偏 decode 稳态吞吐、每 token 的 recurrent / MoE / glue path 开销

这次很多优化并不是同时等比例提升两个指标。

## 阶段 A：先把系统从“能跑”变成“像样地跑”

### A1. Baseline

- 配置：
  - `torch MoE`
  - `torch linear attention`
  - `bf16`
  - `graph off`
- 结果：
  - `TTFT = 6413.07 ms`
  - `1.64 tok/s`

面试可讲：

- 这是纯 PyTorch 参考路径，主要价值是建立下限
- 这个阶段的瓶颈不是某个局部 kernel，而是整个执行链都很低效

### A2. 切到 Fused MoE

- `TTFT: 6413.07 -> 4477.62 ms`
- `TPS: 1.64 -> 7.11 tok/s`

具体做了什么：

- 把 `moe-backend` 从 `torch` 切到 `fused`
- 原来 `TorchMoe` 路径是按 expert 循环：
  - `topk`
  - `token gather`
  - `w1`
  - `activation`
  - `w2`
  - `index_add`
- 改成 `FusedMoe` 后，主路径变成：
  - `topk_softmax`
  - `moe_align_block_size`
  - 两段 routed-expert Triton GEMM
  - reduce/combine

代码落点：

- `python/minisgl/moe/torch_backend.py`
- `python/minisgl/moe/fused.py`
- `python/minisgl/kernel/moe_impl.py`

为什么有效：

- baseline 最大问题之一是 MoE 走了 Python 级逐 expert 路径，kernel 数量和中间张量都很多
- fused backend 把 routed expert 主体改成 block 化、按 expert 分组的 Triton kernel
- token dispatch / expert matmul / combine 的边界被大幅压缩

面试可讲：

- 这是第一次大台阶
- 说明模型里 MoE 是绝对大头之一

### A3. 切到 SGLang Linear Attention

- `TTFT: 4477.62 -> 236.11 ms`
- `TPS: 7.11 -> 15.18 tok/s`

具体做了什么：

- 把 `linear-attn-backend` 从 `torch` 切到 `sglang`
- 不再走纯 PyTorch 的 linear attention 参考实现
- 改走迁移过来的：
  - prefill kernel
  - decode recurrent kernel
  - conv/state update
  - qkvz/ba prepare 路径

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`
- `python/minisgl/linear_attention/*`

为什么有效：

- linear attention 是每层都会执行的主干路径
- torch 版本里有大量：
  - reshape / split
  - cast / contiguous
  - 小 tensor glue
- 切到 sglang 路径后，核心 recurrent update 和 prepare 链都变成了更接近生产实现的 kernel 组合

面试可讲：

- 到这一步，说明 attention 和 MoE 两条主干都必须换成专用高性能实现

### A4. 开启 CUDA Graph

- `TTFT: 236.11 -> 195.07 ms`
- `TPS: 15.18 -> 67.44 tok/s`

具体做了什么：

- 打开 `--cuda-graph-max-bs 1`
- 让单并发 decode steady-state 走 graph capture + replay
- 同时把相关不兼容点排掉，包括：
  - `moe_align_block_size()` 的 fallback 路径
  - routing op 缺失
  - 本地 `sgl_kernel` 路径问题

代码落点：

- `python/minisgl/attention/fi.py`
- `python/minisgl/moe/fused.py`
- `python/sgl_kernel/_routing_loader.py`

为什么有效：

- 在单并发 decode 下，很多时候不是算力不够，而是每 token 的固定调度成本太重
- graph replay 把：
  - Python 调度
  - launch 开销
  - 一部分 metadata/planner 边界
 统一摊平了
- 所以这一步对 steady decode `TPS` 带来的收益远大于对 `TTFT` 的收益

面试可讲：

- 这是第二个最大台阶
- 对单并发 decode 来说，graph 往往不是“锦上添花”，而是“是否进入高性能区间”的分水岭
- 这一步对 `TPS` 的帮助远大于对 `TTFT` 的帮助

---

## 阶段 B：在成熟主链上先做出一个强 baseline

### B1. 开启 W8A8 Int8

- `TTFT: 195.07 -> 189.04 ms`
- `TPS: 67.44 -> 71.05 tok/s`

具体做了什么：

- 新增 `w8a8_int8` 路径
- dense linear 与 MoE expert 权重做 per-channel int8
- activation 做 per-token int8 quant
- GEMM 走 `sgl_kernel.int8_scaled_mm`

代码落点：

- `python/minisgl/quantization/__init__.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/layers/moe.py`

为什么有效：

- prefill 更容易受益于带宽下降和显存占用下降
- MoE expert 和部分大线性层用 int8 后，显存明显下降
- 当时这一步对端到端也带来了温和正收益

但要注意：

- 这不代表 “W8A8 一定比 bf16 更适合 decode”
- 后面验证发现，在当前最优主线下：
  - `bf16 TTFT = 100.57 ms`
  - `bf16 = 113.59 tok/s`
  - `W8A8 TTFT = 89.25 ms`
  - `W8A8 = 110.24 tok/s`

面试可讲：

- W8A8 带来了 TTFT 优势和显存优势
- 但 decode 场景下不一定天然比 bf16 更快

### B2. Selective Int8

- `TTFT: 189.04 -> 187.69 ms`
- `TPS: 71.05 -> 71.94 tok/s`

具体做了什么：

- 没有把所有线性层都一刀切量化
- 只保留更值得量化的部分路径
- 不合适的层继续保留 bf16

代码落点：

- `python/minisgl/layers/linear.py`
- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- 并不是所有层都处在同样的 shape / 访存 / 复用模式下
- 某些层量化后确实节省带宽
- 但某些层会因为 activation quant 边界成本太高而亏损
- selective quant 的本质是：只量化真正赚钱的层

面试可讲：

- 一个重要经验是：量化不是越多越好，应该按热点和 shape 选择

### B3. LayerNorm Prequant Reuse

- `TTFT: 187.69 -> 188.55 ms`
- `TPS: 71.94 -> 71.17 tok/s`
- 失败

具体做了什么：

- 尝试复用 LayerNorm 之后的 prequant 激活
- 目标是减少重复 quant

为什么失败：

- 这条复用链不够长
- 省掉的 quant 次数很有限
- 反而增加了中间张量管理和额外边界判断
- 说明“复用中间结果”必须建立在足够长的消费者链上

面试可讲：

- “复用中间结果”必须先确认复用链够长，否则只是增加复杂度

### B4. 把 LinearAttn Prefill 的 Q/K Norm 挪出 Kernel

- `TTFT: 187.69 -> 179.30 ms`
- `TPS: 71.94 -> 72.53 tok/s`

具体做了什么：

- 调整 prefill 阶段 Q/K norm 的边界位置
- 先把这段逻辑从原始 kernel 外围理顺
- 为后续更大块的 prefill kernel 迁移做准备

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- prefill 热路径里 Q/K norm 是高频边界
- 提前把它理顺后，后面 launch tuning 和 full-kernel prefill 才更容易稳定落地
- 这一步本身收益不算最大，但起到很强的铺路作用

### B5. 调优 Prefill Launch 参数

- `TTFT: 179.30 -> 165.33 ms`
- `TPS: 72.53 -> 77.22 tok/s`

具体做了什么：

- 直接调 prefill kernel 的 launch 配置
- 包括：
  - `BV`
  - `num_warps`
  - `num_stages`

为什么有效：

- 原 kernel 数学没有变
- 但 launch 参数更贴合当前 shape 和 A800 上的实际执行特性
- 说明很多时候性能问题不是算法不对，而是 kernel 配置不对

面试可讲：

- 有时不是算法问题，而是 launch config 根本不对

### B6. 切到 Vendored Full-Kernel Chunk Prefill

- `TTFT: 165.33 -> 113.29 ms`
- `TPS: 77.22 -> 82.79 tok/s`

具体做了什么：

- 把 prefill 路径切到 vendored 的 full-kernel chunk backend
- 不再让 prefill 由一串更碎的外层步骤拼接

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`
- `python/minisgl/linear_attention/*`

为什么有效：

- prefill 最大的问题之一是边界过碎
- full-kernel chunk prefill 直接减少：
  - 中间张量
  - launch 数量
  - kernel 之间的数据回写/回读
- 所以它对 `TTFT` 的改善最显著

面试可讲：

- prefill 这条线证明了：整段融合通常比单点修补更有效
- 这段优化最显著的特点是：**TTFT 改善特别大**

---

## 阶段 C：回到 bf16 主线，系统性追近 sglang

这里开始的重点不是量化，而是回答：

- 为什么 `mini-sglang bf16` 仍然明显慢于 `sglang bf16`

### C1. bf16 + Fused MoE + SGLang LinearAttn + Graph

- `TTFT = 118.49 ms`
- `75.93 tok/s`

这是后续 bf16 优化的出发点。

### C2. Gemma Fused Norm

- `TTFT: 130.59 -> 128.74 ms`
- `TPS: 71.89 -> 80.15 tok/s`

具体做了什么：

- 开启 `GEMMA_FUSED_NORM`
- 把 Gemma 风格 RMSNorm 切到 fused 实现

代码落点：

- `python/minisgl/layers/norm.py`
- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- norm 是跨层高频边界
- 单层看只是一次 elementwise + reduction
- 但乘上几十层后，收益会被稳定放大
- 这是典型的“单点小优化，整模型累计大收益”

### C3. Q/K Inplace Fused Norm Copy-Back

- `TTFT: 128.74 -> 125.24 ms`
- `TPS: 80.15 -> 82.75 tok/s`

具体做了什么：

- 扩大 fused norm 的覆盖范围
- 把 Q/K 那条链上的 norm 结果尽量 inplace / copy-back 处理

为什么有效：

- 减少了 Q/K prepare 里的额外中间张量
- 也减少了后续 rope / attention prepare 前的胶水成本

### C4. FullAttn Fused Prepare

- `TTFT: 125.24 -> 121.91 ms`
- `TPS: 82.75 -> 86.58 tok/s`

具体做了什么：

- 开启 `FULL_ATTN_FUSED_PREPARE`
- 把 full attention prepare 链里的：
  - q/k norm
  - rope
  - gate extract
 这几步合成更紧凑的实现

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`
- `python/minisgl/layers/fused_qk_rmsnorm_rope_gate.py`

为什么有效：

- 这不是去改 attention 数学本体
- 而是把进入 attention 之前的一串 prepare glue path 压平
- 这类优化很能体现 mini-sglang 和 sglang 的真实差距：差距常常在边界组织，而不是核心公式

面试可讲：

- 这是非常典型的“整层边界融合”收益
- 说明 mini-sglang 相对 sglang 的差距，很大一部分不在核心算子公式，而在 prepare / glue path

### C5. Fused Gate Mul

- `TTFT: 121.91 -> 122.81 ms`
- `TPS: 86.58 -> 86.55 tok/s`
- 基本无收益

结论：

- 不是所有 elementwise fuse 都值得做
- 要优先做真正高频、且处于热路径中的融合

### C6. Decode Conv 切到 SGLang `causal_conv1d_update`

- `TTFT: 121.91 -> 123.30 ms`
- `TPS: 86.58 -> 88.35 tok/s`

具体做了什么：

- 开启 `DEPTHWISE_CONV_DECODE`
- decode 时不再走原来的 conv update
- 改用 sglang 的 `causal_conv1d_update`

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- conv update 是 decode 每 token 每层都要走的路径
- 所以虽然这是单点改动，但收益几乎纯粹体现在 steady decode `TPS`

---

## 阶段 D：恢复历史有效 fuse，并继续沿整层边界优化

### D1. 恢复 `LINEAR_RMSNORM_GATED`

- `TTFT` 基本持平
- `TPS: 88.37 -> 93.34 tok/s`

这是一次很关键的排障：

- 之前历史 session 明明跑到过更高值
- 后来发现不是实验记错，而是这条 fuse 的接线在当前代码里丢了

面试可讲：

- 性能排障里一个重要能力是：
  - 区分“想法无效”
  - 和“有效优化被代码回归弄丢了”

具体做了什么：

- 把历史上已经验证有效的 `LINEAR_RMSNORM_GATED` 调用点重新接回 Qwen3.6 linear attention 热路径

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`
- `python/minisgl/layers/fused_rmsnorm_gated.py`

为什么有效：

- linear attention 输出 norm 是 decode 高频边界
- 原来是：
  - RMSNorm
  - gate
  - 输出分配
  多步拼起来
- fuse 之后直接压缩成更紧凑的一段

### D2. `SHARED_EXPERT_FUSED_GATE_ADD`

- `TTFT` 基本持平
- `TPS: 93.34 -> 96.13 tok/s`

具体做了什么：

- 开启 `SHARED_EXPERT_FUSED_GATE_ADD`
- 把 `sigmoid(gate) * shared_output + moe_output` 这段 shared expert epilogue 融起来

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`
- `python/minisgl/layers/elementwise.py`

为什么有效：

- shared expert 本身不是最大头
- 但它是每层每 token 都会经过的边界
- 把 gate mul 和 residual add 融起来后，少了一次中间结果写回

### D3. `MOE_FUSED_ACTIVATION`

- `TTFT: 122.70 -> 117.15 ms`
- `TPS: 96.13 -> 98.56 tok/s`

具体做了什么：

- 开启 `MOE_FUSED_ACTIVATION`
- 在 routed expert stage2 里，用 fused `silu_and_mul / gelu_and_mul`

代码落点：

- `python/minisgl/moe/fused.py`
- `python/minisgl/layers/activation.py`

为什么有效：

- MoE stage2 是 routed expert 主链里的高频 elementwise 边界
- 这一步减少了 stage1 输出到 activation 再到 stage2 输入之间的胶水开销

### D4. `SHARED_EXPERT_FUSED_ACTIVATION`

- `TTFT: 117.15 -> 119.08 ms`
- `TPS: 98.56 -> 99.19 tok/s`

具体做了什么：

- 把 shared expert 内部的 `silu_and_mul` 换成 fused 版本

为什么有效：

- 逻辑和 `MOE_FUSED_ACTIVATION` 一样
- 只是 shared expert 本身不是 routed expert 那么重，所以收益更小

### D5. `LINEAR_PREFILL_QK_L2NORM`

- `TTFT: 119.08 -> 113.02 ms`
- `TPS: 99.19 -> 100.17 tok/s`

具体做了什么：

- 开启 `LINEAR_PREFILL_QK_L2NORM`
- 把 prefill 路径里的 Q/K L2Norm 下沉进 kernel，不再在外层单独做

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- 减少了 prefill 阶段：
  - 外层 `normalize`
  - 额外 cast
  - 显式 contiguous
- 所以这一步对 `TTFT` 更敏感

### D6. `LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS`

- `TTFT` 小幅改善
- `TPS: 100.17 -> 100.59 tok/s`

具体做了什么：

- 跳过 prefill 路径里一批已经不再必要的显式 `.contiguous()`

为什么有效：

- 某些 contiguous 只是历史遗留防御性写法
- 当下游 kernel 已经能消费当前布局时，这些复制就成了纯损耗

### D7. `SKIP_AB_FP32_CAST`

- `TTFT` 小幅波动
- `TPS: 100.59 -> 101.04 tok/s`

具体做了什么：

- 开启 `SKIP_AB_FP32_CAST`
- decode 时尽量不把 `a/b` 先抬成 fp32，再交给 kernel

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- kernel 内部本来就会按自己的方式读取和提升精度
- 外层先 cast 一遍，很多时候只是增加访存和中间张量

这几步的共同点：

- 不是去碰一个大 kernel 的数学本体
- 而是在层边界把：
  - norm
  - gate
  - prepare
  - contiguous
  - cast
 这些高频 glue path 一步步压平

面试可讲：

- 这类优化单次收益不一定很大
- 但它们通常稳定、可叠加、风险低

---

## 阶段 E：主攻 decode 边界和 MoE 周边，逼近最好结果

### E1. Decode 输入投影融合

- `TTFT: 约 116 ms`
- `TPS: 101.04 -> 102.17 tok/s`

对应开关：

- `LINEAR_DECODE_FUSED_INPUT_PROJ`

具体做了什么：

- 开启 `LINEAR_DECODE_FUSED_INPUT_PROJ`
- 把 decode 时原来分开的：
  - `in_proj_qkvz`
  - `in_proj_ba`
  合成一次更大的 bf16 GEMM，再切片回去

代码落点：

- `python/minisgl/models/qwen3_5_moe.py`

为什么有效：

- 减少了一次 GEMM launch
- 减少了一次输入读
- 对 `bs=1` decode 来说，这类边界削减很关键

### E2. Decode 小边界继续削减

逐步得到：

- `102.17 -> 102.29`：`LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ`
- `102.29 -> 102.51`：`LINEAR_RMSNORM_GATED_REUSE_OUT`
- `102.51 -> 102.71`：`LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS`
- `102.71 -> 103.26`：`LINEAR_DECODE_SKIP_AB_CONTIGUOUS`

面试可讲：

- decode 阶段并不是单个 kernel 特别慢
- 而是很多小边界累计起来很贵
- 所以这类优化主要反映在 `TPS` 上，而不是 `TTFT`

具体做了什么：

- `LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ`
  - 尝试把 `qkvz` 和 `ba` 两次投影放到双 stream 重叠
- `LINEAR_RMSNORM_GATED_REUSE_OUT`
  - 复用 decode norm 输出 buffer，减少 `empty_like`
- `LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS`
  - 跳过 out_proj 前多余 contiguous
- `LINEAR_DECODE_SKIP_AB_CONTIGUOUS`
  - 跳过 decode `a/b` 的多余 contiguous

为什么有效：

- 它们共同作用的对象都不是“大 kernel 本体”
- 而是 decode 每 token 每层都会重复付出的边界成本

### E3. 线性 attention decode 主核“看起来像 sglang”的方向，多数无效

证伪项包括：

- `LINEAR_DECODE_VK_STATE`
- `LINEAR_DECODE_SGLANG_PACKED`
- `LINEAR_DECODE_SGLANG_UPDATE`
- `LINEAR_DECODE_FUSED_QKV_SPLIT`

结论：

- “更像 sglang 的某个局部 kernel” 不等于端到端更快
- 必须回到真实 workload 验证

具体案例：

- `LINEAR_DECODE_VK_STATE`
  - 做了什么：
    - 额外维护 `[HV, V, K]` 辅助 state
    - 希望仅靠更像 sglang 的 state layout 提升 decode kernel 访存
  - 结果：
    - `101.04 -> 97.01 tok/s`
  - 为什么失败：
    - 只改 layout 没有同步改变整条 decode 执行链
    - 访存形态变化不足以抵消额外状态维护成本

- `LINEAR_DECODE_SGLANG_PACKED`
  - 做了什么：
    - 在 `VK_STATE` 基础上直接复用 sglang packed recurrent decode kernel
  - 结果：
    - `101.04 -> 96.92 tok/s`
  - 为什么失败：
    - packed kernel 本体不一定慢
    - 但 mini-sglang 当前上下游 prepare / conv / norm / out_proj 组织不和它天然匹配

- `LINEAR_DECODE_SGLANG_UPDATE`
  - 做了什么：
    - 直接切到 sglang 常规 decode 的 recurrent update kernel
  - 结果：
    - `103.26 -> 98.72 tok/s`
  - 为什么失败：
    - 再次证明差距不主要在 recurrent kernel 本体，而在整条 decode 边界链

- `LINEAR_DECODE_FUSED_QKV_SPLIT`
  - 做了什么：
    - 只在 decode 启用 fused `qkvzba split/reshape`
  - 结果：
    - `103.26 -> 101.70 tok/s`
  - 为什么失败：
    - 当前这版 fused split 虽然看起来更像“理想实现”，但并没有转化成端到端收益
    - 说明这段不是简单开一个 fused kernel 就能补齐的差距点

### E4. MoE Router / Dispatch / Reduce 周边，是真正的大收益区

关键增益链：

- `103.26 -> 104.72`：`MOE_SKIP_TOPK_POST_RENORM`
- `104.72 -> 106.62`：`MOE_SKIP_TOPK_FP32_CAST`
- `106.62 -> 107.17`：`MOE_SKIP_DISPATCH_LOCAL_MASK`
- `107.17 -> 109.16`：`MOE_ALIGN_SMALL_CAP`
- `109.16 -> 111.15`：`MOE_SGLANG_CONFIG_LOOKUP`
- `111.15 -> 112.08`：`MOE_SGLANG_DOWN_CONFIG`
- `112.08 -> 112.65`：`FI_GRAPH_FAST_DECODE_PLAN`
- `112.65 -> 114.01`：`MOE_REUSE_TOPK_WORKSPACE + FI_GRAPH_FAST_DECODE_PLAN`

具体做了什么：

- `MOE_SKIP_TOPK_POST_RENORM`
  - 既然 `sgl_kernel.topk_softmax(..., renormalize=True)` 已经做了归一化，就跳过 Python 侧重复 renorm
- `MOE_SKIP_TOPK_FP32_CAST`
  - 调 `topk_softmax` 前不再额外 `router_logits.float()`
- `MOE_SKIP_DISPATCH_LOCAL_MASK`
  - 单卡 fastpath 下不再构造无意义的全真 `local_mask`
- `MOE_ALIGN_SMALL_CAP`
  - 把 `moe_align_block_size` 的临时 buffer 上界改成更贴近 sglang 的小上界
- `MOE_SGLANG_CONFIG_LOOKUP`
  - routed expert 第一段 GEMM 的 Triton config 优先查 sglang 风格 JSON
- `MOE_SGLANG_DOWN_CONFIG`
  - 第二段 `down_proj` GEMM 也引入独立 config 查表
- `MOE_SGL_REDUCE`
  - 用 `sgl_kernel.moe_sum_reduce`
- `FI_GRAPH_FAST_DECODE_PLAN`
  - decode graph replay 时对齐 FlashInfer `fast_decode_plan`
- `MOE_REUSE_TOPK_WORKSPACE`
  - 复用 topk 输出临时缓冲，而不是每次重新分配

代码落点：

- `python/minisgl/moe/fused.py`
- `python/minisgl/moe/dispatch.py`
- `python/minisgl/attention/fi.py`
- `python/minisgl/env.py`

为什么这条线有效：

- 后期剩余差距已经不在单个大 GEMM
- 而在一串 decode MoE 周边小成本累加：
  - topk
  - align
  - dispatch
  - config 选择不佳
  - reduce/combine
  - graph replay metadata
- 这条线最能体现“从整个推理引擎角度优化”，而不是只盯单核

面试可讲：

- 这一步是后期真正把结果拉到 `114 tok/s` 的关键
- 也是最能体现“不是只会改 kernel，而是会看整个推理引擎执行链”的地方
- 这条线整体上更偏 decode `TPS` 优化

---

## 4. 最终成绩与对标

### 当前最好稳定值

- `TTFT = 99.06 ms`
- `E2E = 0.5526 s`
- `output_tps = 114.01 tok/s`

### 对标 sglang bf16

- `sglang bf16 TTFT = 106.40 ms`
- `sglang bf16 = 124.03 tok/s`

差距：

- `114.01 / 124.03 ≈ 91.9%`
- `TTFT` 已经更低：
  - `99.06 ms vs 106.40 ms`

面试可讲：

- 这说明优化已经从“工程实现不成熟”推进到“接近成熟系统”
- 后续再追就是更细粒度的系统归因，而不是明显的大洞
- 当前剩余差距主要在 steady-state decode `TPS`

---

## 5. 失败实验里最值得讲的教训

## 教训 1：不是越像上游某个 kernel 就越快

证伪：

- `LINEAR_DECODE_SGLANG_PACKED`
- `LINEAR_DECODE_SGLANG_UPDATE`
- `LINEAR_RMSNORM_GATED_SGLANG`

可讲法：

- 上游局部实现的收益依赖完整上下文
- 单独搬一个 kernel，可能因为上下游边界不同而退化

代表性案例：

- `LINEAR_DECODE_SGLANG_PACKED`
  - 做了什么：
    - 额外维护 `[HV, V, K]` 辅助 state
    - decode 直接切到 sglang packed recurrent kernel
  - 结果：
    - `101.04 -> 96.92 tok/s`
  - 为什么失败：
    - 只替换 recurrent kernel，并没有把上下游 prepare / norm / out_proj / conv 一起变成同一套组织
    - 所以局部 kernel 更像 sglang，不等于整条 decode 链更像 sglang

- `LINEAR_DECODE_SGLANG_UPDATE`
  - 做了什么：
    - 直接切到 sglang 常规 decode 的 recurrent update kernel
  - 结果：
    - `103.26 -> 98.72 tok/s`
  - 为什么失败：
    - 这进一步证明当前差距不主要在 recurrent kernel 本体
    - 真正差距更可能在整层边界和其它热路径的组合开销

- `LINEAR_RMSNORM_GATED_SGLANG`
  - 做了什么：
    - 直接换用 sglang 的 gated RMSNorm
  - 结果：
    - `103.26 -> 102.45 tok/s`
  - 为什么失败：
    - 当前 mini-sglang 自己这版 fused norm 已经不差
    - 单换一个上游 norm 实现并没有带来端到端收益

## 教训 2：不是 fuse 得越宽越好

证伪：

- `MOE_SINGLE_KERNEL`
- `FULL_ATTN_SIGMOID_GATE`
- 多条 W8A8 decode 局部 `quant+gemm` fuse

可讲法：

- 更宽的融合会改变 tile、寄存器、occupancy、cache 行为
- 如果 shape 不合适，反而会破坏当前更优的 kernel 组合

代表性案例：

- `MOE_SINGLE_KERNEL`
  - 做了什么：
    - 把 routed expert 的 `silu_and_mul + down_proj` 尝试合成单 kernel
  - 结果：
    - `114.01` 这条主线附近，单独实验只有 `101.84 tok/s`
  - 为什么失败：
    - 更宽的融合改变了 tile、寄存器压力和 kernel 组合方式
    - 在当前 Qwen3.6 expert shape 上，这种更激进的融合反而破坏了已有的更优执行路径

- `FULL_ATTN_SIGMOID_GATE`
  - 做了什么：
    - 想把 `sigmoid(gate) * attn_output` 再额外融合
  - 结果：
    - 基本无收益，未进入主线
  - 为什么失败：
    - 这段虽然看起来能 fuse，但并不是最重边界
    - 真正更值钱的是 prepare、norm、MoE 周边这类高频大头

- W8A8 decode 局部 `quant+gemm` fuse
  - 做了什么：
    - 先做通用 decode `quant+gemm`
    - 再做 shared expert `inter -> down_proj`
    - 再做 routed expert `stage2 -> w2`
  - 结果：
    - `110.57 -> 110.36`
    - `110.57 -> 110.22`
    - `110.57 -> 107.14`
  - 为什么失败：
    - 省掉的一次 launch 和一份中间张量，不足以抵消更差的 kernel 组织
    - 对某些路径来说，还会破坏原本能复用的 prequant 结果

## 教训 3：性能回归不一定是思路错，可能是接线丢了

关键案例：

- `LINEAR_RMSNORM_GATED` 历史有效，但当前工作树一度复现不上
- 最终发现是调用点丢了，不是思路无效

可讲法：

- 做性能工程必须有“验证接线是否真的生效”的习惯

代表性案例：

- `LINEAR_RMSNORM_GATED`
  - 现象：
    - 历史 session 明明有 `93.32 -> 96.02 tok/s` 这条链
    - 但当前工作树一度复现不上
  - 最后查明：
    - kernel 和 env 开关都还在
    - 但 Qwen3.6 linear attention 热路径里的真实调用点丢了
  - 修复后：
    - `88.37 -> 93.34 tok/s`
    - 再叠 `SHARED_EXPERT_FUSED_GATE_ADD` 到 `96.13 tok/s`
  - 面试怎么讲：
    - 做性能优化不只是“想新点子”
    - 还要能做性能回归排障，确认已有优化是否真的还在生效

## 教训 4：W8A8 不等于 decode TPS 一定更高

同口径结果：

- `bf16 TTFT = 100.57 ms`
- `bf16 = 113.59 tok/s`
- `W8A8 TTFT = 89.25 ms`
- `W8A8 = 110.24 tok/s`

说明：

- W8A8 提升了 TTFT
- 但 decode 下 dynamic quant 的代价可能超过 int8 GEMM 的收益

代表性案例：

- 同口径最优主线下：
  - `bf16`
    - `TTFT = 100.57 ms`
    - `113.59 tok/s`
  - `W8A8`
    - `TTFT = 89.25 ms`
    - `110.24 tok/s`
  - `W8A8 + norm fuse`
    - `TTFT = 88.97 ms`
    - `110.49 tok/s`

- 这说明什么：
  - W8A8 对 prefill 更有利，所以 TTFT 更低
  - 但 decode 是单 token、小 batch、高频动态 quant 场景
  - activation quant 的代价会吃掉 int8 GEMM 的收益

- 进一步证据：
  - 通用 decode `quant+gemm fuse` 没收益
  - shared expert decode fuse 没收益
  - routed expert decode fuse 明显退化

- 面试怎么讲：
  - 量化收益必须区分 prefill 和 decode
  - 不能只看显存或 TTFT，就断言 decode steady-state 也一定更快

---

## 6. 面试时推荐的讲述顺序

可以按下面 6 句来讲：

1. 我先把系统从纯 torch 路径切到 fused MoE、sglang linear attention 和 CUDA graph，把 `TPS` 从 `1.64` 拉到 `67.44 tok/s`。
2. 这个过程中我不是只看 `TPS`，而是同时看 `TTFT`，发现 graph 对 steady decode 的帮助最大，而 prefill 相关优化更偏向改善 `TTFT`。
3. 然后我在量化和 bf16 两条线上分别建立强 baseline，确认真正的主问题不是“有没有量化”，而是 decode 阶段的结构性差异。
4. 接着我沿整层边界做融合，重点打 norm、prepare、gate、cast、contiguous 这些 glue path，把 bf16 主线推到 `101 tok/s` 左右，同时把 `TTFT` 压到百毫秒级。
5. 再往后我发现线性 attention decode 主核本体不是主矛盾，于是把重点转到 MoE router、dispatch、align、reduce 和 config 选择上。
6. 这条 MoE 周边主线最终把结果推到 `114.01 tok/s`，达到 sglang bf16 的约 `91.9%`，同时 `TTFT = 99.06 ms`，已经优于对标的 `106.40 ms`。
7. 过程中我还系统证伪了很多“看起来合理但端到端无收益”的方向，比如局部 kernel 替换、过宽融合、以及多条 W8A8 decode `quant+gemm` fuse。

---

## 7. 如果面试官追问“你最核心的经验是什么”

推荐回答：

- 第一，**先抓大台阶**：backend、graph、主链切换，比局部抠 kernel 更重要。
- 第二，**性能差距经常在边界，不在公式本体**：prepare、dispatch、reduce、metadata、glue path 往往才是端到端瓶颈。
- 第三，**必须用同口径 benchmark 反复证伪**：很多“更像上游 / 更宽的 fuse / 更少 kernel”的改动，单看直觉是对的，但端到端会退化。

---

## 8. 这份文档对应的原始资料

- 原始流水账：
  - [2026-06-10-qwen36-1024in-64out-stepwise-benchmark.md](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/2026-06-10-qwen36-1024in-64out-stepwise-benchmark.md)
- 后续路线图与归因：
  - [2026-06-28-qwen36-next-optimization-roadmap.md](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/profile_analysis/2026-06-28-qwen36-next-optimization-roadmap.md)
