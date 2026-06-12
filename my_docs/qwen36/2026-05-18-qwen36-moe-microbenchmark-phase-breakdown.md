# 2026-05-18 Qwen3.6 MoE Microbenchmark 分阶段拆解记录

## 1. 背景

这次实验的目标不是测整条服务链路，而是单独看 `MoE` 算子前向本身的单次开销。

因此基准放在：

- `benchmark/analysis/bench_moe_backend_micro.py`

而不是 `benchmark/online`。

原因很直接：

- 本次对比对象是 `torch MoE` 和 `fused MoE` 的单次前向算子成本
- 不涉及服务启动、调度、KV cache、采样、网络请求等在线链路因素

## 2. 本次新增内容

本轮补了两类能力。

### 2.1 自动读取真实模型配置

脚本现在支持：

- `--model-path`

传入真实模型目录后，会自动读取：

- `hidden_size`
- `moe_intermediate_size`
- `num_experts`
- `num_experts_per_tok`
- `norm_topk_prob`
- `num_expert_group`
- `topk_group`

这样后续可以直接用真实 `Qwen3.6` 配置做微基准，而不是手工填参数。

### 2.2 分阶段计时

脚本现在不仅能测总耗时，还能拆成：

- `router_ms`
- `torch_expert_ms`
- `fused_expert_ms`
- `fused_w1_ms`
- `fused_stage2_ms`
- `fused_w2_ms`
- `fused_reduce_ms`

其中：

- `router_ms` 表示 `topk + dispatch plan`
- `torch_expert_ms` 表示预先给定 routing 后，`torch` 专家计算部分的耗时
- `fused_expert_ms` 表示预先给定 routing 后，`fused` 专家计算部分的耗时
- `fused_w1_ms / stage2_ms / w2_ms / reduce_ms` 是 `fused expert` 内部四段细拆

## 3. 实验配置

实验设备：

- `GPU7`

执行命令：

```bash
PATH=/data/zxz/condaenv/minisgl/bin:$PATH \
PYTHONPATH=/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python \
CUDA_VISIBLE_DEVICES=7 \
python benchmark/analysis/bench_moe_backend_micro.py \
  --device cuda:0 \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --tokens 1 8 64 \
  --repetitions 20
```

自动读取到的真实 `Qwen3.6-35B-A3B` MoE 配置为：

- `hidden_size = 2048`
- `moe_intermediate_size = 512`
- `num_experts = 256`
- `topk = 8`
- `renormalize = true`
- `use_grouped_topk = false`

数据类型：

- `bfloat16`

## 4. 实验结果

### 4.1 tokens = 1

- `router_ms = 0.0780`
- `torch_expert_ms = 12.6336`
- `fused_expert_ms = 0.3790`
- `fused_w1_ms = 0.1034`
- `fused_stage2_ms = 0.0642`
- `fused_w2_ms = 0.0918`
- `fused_reduce_ms = 0.0628`
- `torch_ms = 12.5660`
- `fused_ms = 0.5297`
- `speedup_torch_over_fused = 23.72x`

### 4.2 tokens = 8

- `router_ms = 0.0769`
- `torch_expert_ms = 20.8788`
- `fused_expert_ms = 0.4182`
- `fused_w1_ms = 0.2198`
- `fused_stage2_ms = 0.0178`
- `fused_w2_ms = 0.0877`
- `fused_reduce_ms = 0.0113`
- `torch_ms = 21.5330`
- `fused_ms = 0.5000`
- `speedup_torch_over_fused = 43.07x`

### 4.3 tokens = 64

- `router_ms = 0.0817`
- `torch_expert_ms = 48.6055`
- `fused_expert_ms = 0.9037`
- `fused_w1_ms = 0.6269`
- `fused_stage2_ms = 0.0193`
- `fused_w2_ms = 0.3082`
- `fused_reduce_ms = 0.0102`
- `torch_ms = 47.9633`
- `fused_ms = 0.9312`
- `speedup_torch_over_fused = 51.51x`

## 5. 关键结论

### 5.1 巨大差距不在 router

这组数据里，`router_ms` 一直只有大约 `0.08 ms`。

也就是说：

- `torch MoE` 慢，不是因为 `topk` 选路特别慢
- `fused MoE` 快，也不是主要赢在 router

真正的主差距在 expert compute。

### 5.2 torch 路径的主要问题是“稀疏专家执行方式”本身

`torch_expert_ms` 和 `fused_expert_ms` 的差距非常大：

- `tokens=1`: `33.3x`
- `tokens=8`: `49.9x`
- `tokens=64`: `53.8x`

这说明问题不在某一个局部算子，而在于 `torch` 的整体执行方式：

- 逐 expert 循环
- 对每个 expert 做 `where / gather`
- 多次小尺寸 `linear`
- 激活后再做第二次 `linear`
- 再做 `index_add_`

这种路径会产生大量：

- 小 kernel launch
- 不连续访存
- 中间 tensor
- gather / scatter 开销

因此在 MoE 这种稀疏执行场景下非常吃亏。

### 5.3 fused 快的根本原因是执行路径更紧凑

从 `fused` 内部四段看：

- 时间主要集中在 `w1` 和 `w2`
- `stage2` 和 `reduce` 已经被压到很低

以 `tokens=64` 为例：

- `w1 = 0.6269 ms`
- `w2 = 0.3082 ms`
- `stage2 = 0.0193 ms`
- `reduce = 0.0102 ms`

说明 `fused` 的核心优势是：

- 通过更合理的 token-expert 排布，把稀疏专家计算组织成更连续的执行流
- 降低小 batch matmul 的碎片化问题
- 减少 Python 循环和中间张量读写
- 把 dispatch / compute / reduce 更紧地拼接起来

因此，`fused MoE` 比 `torch MoE` 快很多，不只是“单个 GEMM 理论上更快”，而是整条 sparse expert pipeline 的结构更优。

### 5.4 对后续 int8 优化的启示

这组实验也给后续 `int8` 优化提供了方向：

- 如果 `int8` 只是做权重量化，但 expert 执行路径仍然碎片化，那么不一定能赢过 `bf16 fused`
- 真正重要的是保持甚至继续强化 `fused` 路径的结构性优势
- 后续若要让 `int8` 真正超过 `bf16`，重点应该放在：
  - `w1/w2` kernel 更高效
  - 更少中间读写
  - 更强 fusion
  - 更稳定的 dispatch + reduce 路径

## 6. 当前产出文件

本轮相关改动：

- `benchmark/analysis/bench_moe_backend_micro.py`
- `python/minisgl/moe/fused.py`

新增能力：

1. `--model-path` 自动读取真实模型 MoE 配置
2. 输出 `router` / `expert-only` / `total`
3. 输出 `fused expert` 内部 `w1/stage2/w2/reduce` 分段耗时

## 7. 下一步建议

如果要继续往底层分析，最值得做的下一步是：

1. 把 `torch expert` 再进一步拆成：
   - `dispatch/gather`
   - `w1`
   - `activation`
   - `w2`
   - `scatter/index_add`
2. 用同一组 routing，逐项对比 `torch` 和 `fused`
3. 再基于这组数据决定：
   - 后续是优先做 `kernel fusion`
   - 还是优先推进 `int8 expert kernel`

这样就能把“为什么 fused 明显更快”从总体现象，进一步变成逐阶段、逐成本项的定量结论。
