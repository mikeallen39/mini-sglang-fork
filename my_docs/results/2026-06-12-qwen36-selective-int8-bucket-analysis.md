# 2026-06-12 Qwen3.6 Selective Int8 Bucket Analysis

## 背景

当前最优 `W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph` 已经基本追平 `bf16`，但还没有稳定超过。

本轮分析的目标不是再看端到端总分，而是拆清楚：

- 不同类型 linear 的 `bf16` 开销
- 当前真实线上 `sgl_kernel.int8_scaled_mm` 路径的 `int8` 开销
- `quant` 的额外成本
- 哪些层值得继续保留 `int8`
- 哪些层应该回退到 `bf16`

## 测试方法

- 脚本：
  [benchmark/analysis/bench_qwen36_prefill_int8_breakdown.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/benchmark/analysis/bench_qwen36_prefill_int8_breakdown.py)
- 环境：
  - `/mnt/82_store/zxz/condaenv/minisgl`
  - `CUDA_VISIBLE_DEVICES=1`
- dtype：
  - `bfloat16`
- repetitions：
  - `20`
- 两种场景：
  - `tokens=1`：decode-like
  - `tokens=1024`：prefill-like

说明：

- `sgl_int8_total_ms` 是当前真实线上 fused int8 路径
- `quant_ms` 是单独测的 activation quant
- `epilogue_ms_proxy` 来自 `torch._int_mm` 参考路径，只作为 `dequant/scale/cast` 的近似代理，不是 `sgl_kernel` 内部精确 phase split

## 关键结论

### 1. 大层已经明显适合保留 Int8

当前真实 `sgl_kernel` 路径下，大投影层的 `int8` 已经明显快于 `bf16`。

#### Decode-like (`tokens=1`)

| bucket | bf16_linear_ms | sgl_int8_total_ms | speedup |
| --- | ---: | ---: | ---: |
| `full_attn_qkv_proj` | `0.0308` | `0.0139` | `2.22x` |
| `linear_attn_in_proj_qkvz` | `0.0371` | `0.0135` | `2.74x` |
| `full_attn_o_proj` | `0.0162` | `0.0136` | `1.19x` |
| `linear_attn_out_proj` | `0.0161` | `0.0133` | `1.21x` |

#### Prefill-like (`tokens=1024`)

| bucket | bf16_linear_ms | sgl_int8_total_ms | speedup |
| --- | ---: | ---: | ---: |
| `full_attn_qkv_proj` | `0.1672` | `0.0981` | `1.70x` |
| `linear_attn_in_proj_qkvz` | `0.2236` | `0.1277` | `1.75x` |
| `full_attn_o_proj` | `0.0913` | `0.0598` | `1.53x` |
| `linear_attn_out_proj` | `0.0922` | `0.0599` | `1.54x` |

结论：

- `attention` 相关大投影层已经有明确 int8 收益
- 这些层不应该回退到 `bf16`

### 2. 小输出层仍然是主要拖后腿项

#### 不支持当前 sgl_kernel int8 的层

| bucket | 形状 | 问题 |
| --- | --- | --- |
| `shared_expert_gate` | `2048 -> 1` | `N=1`，不满足当前 `sgl_kernel` 的 `N % 8 == 0` 约束 |

#### 支持但不值得保留 Int8 的层

| bucket | tokens | bf16_linear_ms | sgl_int8_total_ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| `moe_router_gate` | `1` | `0.0142` | `0.0133` | `1.07x` |
| `moe_router_gate` | `1024` | `0.0146` | `0.0208` | `0.70x` |
| `linear_attn_in_proj_ba` | `1` | `0.0164` | `0.0132` | `1.24x` |
| `linear_attn_in_proj_ba` | `1024` | `0.0186` | `0.0203` | `0.92x` |

结论：

- `shared_expert_gate` 应直接保留 `bf16`
- `moe_router_gate` 和 `linear_attn_in_proj_ba` 在 `prefill` 下都不划算
- 这两类是当前最优先的 `selective int8` 回退候选

### 3. quant 开销仍然很重，尤其对小层最重

一些代表性数据：

| bucket | tokens | quant_ms | sgl_int8_total_ms |
| --- | ---: | ---: | ---: |
| `full_attn_qkv_proj` | `1` | `0.0601` | `0.0139` |
| `linear_attn_in_proj_qkvz` | `1` | `0.0569` | `0.0135` |
| `moe_router_gate` | `1` | `0.0558` | `0.0133` |
| `linear_attn_in_proj_ba` | `1` | `0.0569` | `0.0132` |
| `full_attn_qkv_proj` | `1024` | `0.0567` | `0.0981` |
| `linear_attn_in_proj_qkvz` | `1024` | `0.0557` | `0.1277` |
| `moe_router_gate` | `1024` | `0.0553` | `0.0208` |
| `linear_attn_in_proj_ba` | `1024` | `0.0555` | `0.0203` |

解释：

- 对大层，`sgl_kernel` 主计算已经快了，但 `quant` 仍是不可忽略的一大块
- 对小层，`quant` 本身甚至比真正的 int8 主计算还大

这意味着当前进一步超过 `bf16` 的重点已经变成：

- 继续减少小层 int8 化
- 继续做 `quant` 融合和复用

## 建议的 Selective Int8 清单

### 建议继续保留 Int8

- `full_attn_qkv_proj`
- `full_attn_o_proj`
- `linear_attn_in_proj_qkvz`
- `linear_attn_out_proj`
- `shared_expert_gate_up`
- `shared_expert_down`
- `dense_gate_up_proj`
- `dense_down_proj`

原因：

- 这些层在当前真实 `sgl_kernel` 路径下，多数已经明显快于 `bf16`
- 是最值得继续保留 `int8` 的主力层

### 建议回退到 BF16

- `shared_expert_gate`
- `moe_router_gate`
- `linear_attn_in_proj_ba`

原因：

- `shared_expert_gate` 直接不满足当前 kernel 约束
- `moe_router_gate` 和 `linear_attn_in_proj_ba` 在 `prefill` 下没有稳定 int8 收益
- 这类层的 `quant` 成本相对过高

## 当前最值得继续做的优化

基于这轮拆解，后续优先级应该是：

1. 做一版真正的 `selective int8`
2. 把上面 3 个小层回退到 `bf16`
3. 在此基础上重新测端到端：
   - `TTFT`
   - `E2E`
   - `output_tps`
4. 如果还想继续超过 `bf16`，下一步重点不是继续换 GEMM，而是：
   - `RMSNorm + Quant` 融合
   - 更广泛的 prequant reuse
   - 小 `M` / decode 场景的 quant 开销压缩

## 一句话总结

当前 `W8A8` 没完全超过 `bf16`，已经不再是“大层 int8 GEMM 不够快”的问题，而是：

- 小输出层 int8 化不划算
- `quant` 开销仍然偏重

因此最合理的下一步是：

> 保留大层 int8，回退小层 bf16，然后再重测端到端收益。

## 补充：Routed Experts 的单独排查结论

上面的 bucket 主要覆盖的是 attention、router、shared expert 和 dense MLP 这类“可单独当作 linear 测量”的模块。

Routed experts 需要单独看，因为它们走的是 `fused MoE` 整体路径，而不是普通 dense linear。

本轮补充测试脚本：

- [benchmark/analysis/bench_qwen36_routed_experts_int8_breakdown.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/benchmark/analysis/bench_qwen36_routed_experts_int8_breakdown.py)

关键结论：

- routed experts 其实早就已经量化了
- 之前它们的 `int8` 比 `bf16` 慢，并不是因为 int8 理论不行
- 真正的慢点是：
  - `fused_moe_w2_silu_int8` 这条特化 `w2` kernel
  - 它在 Qwen3.6 当前 routed expert shape 上明显退化

代表性结果：

### 修复前

| tokens | fused_bf16_ms | fused_int8_ms | 结论 |
| --- | ---: | ---: | --- |
| `1` | `0.3666` | `0.4098` | `int8` 略慢 |
| `1024` | `1.2547` | `1.9942` | `int8` 明显更慢 |

### 关闭退化的 `w2` 特化 kernel 后

| tokens | fused_bf16_ms | fused_int8_ms | 结论 |
| --- | ---: | ---: | --- |
| `1024` | `1.2542` | `0.9998` | `int8` 明显反超 |

### 对退化原因的进一步排查

最开始怀疑的一点是：

- `fused_moe_w2_silu_int8_kernel_triton` 之前仍然先执行了一次
  - `silu_and_mul_quant_int8_triton(...)`
- 而特化 `w2` kernel 内部又会自己重新做一遍 `silu + mul + quant`
- 这会导致明显的重复 `stage2` 计算

这部分问题已经修掉，并重新测了开/关特化 kernel 的对比：

| tokens | 特化 kernel | total ms | w1 ms | stage2 ms | w2 ms | reduce ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | 开 | `0.3815` | `0.1503` | `0.0097` | `0.0988` | `0.0500` |
| `1` | 关 | `0.4209` | `0.1492` | `0.0651` | `0.0829` | `0.0622` |
| `1024` | 开 | `1.9212` | `0.7472` | `0.0030` | `1.2537` | `0.0288` |
| `1024` | 关 | `1.0014` | `0.7505` | `0.0430` | `0.2995` | `0.0292` |

这说明：

1. 重复 `stage2` 确实是个真实问题
   - 修复后 `stage2_ms` 明显下降
   - 例如 `tokens=1024` 时从 `~0.043` 降到 `~0.003`

2. 但它不是 prefill 退化的主因
   - 因为即使修掉重复 `stage2`
   - `tokens=1024` 下特化 kernel 的 `w2_ms` 仍然高达 `~1.25 ms`
   - 而关闭特化 kernel 时只有 `~0.30 ms`

所以当前更准确的结论是：

- 重复 `stage2` 是次要问题，已经修掉
- prefill 大 token 场景下，真正的主因仍然是
  - `fused_moe_w2_silu_int8_kernel_triton`
  - 这条特化 `w2` kernel 本身在当前 Qwen3.6 routed expert shape 上退化

同时它也不是“永远都慢”：

- `tokens=1` 时，修复重复 `stage2` 后，特化 kernel 反而略快

这说明这条 kernel 更像是：

- decode 小 batch 场景有利
- prefill 大 token 场景不利

后续如果继续优化，最自然的方向不是简单永久开/关，而是：

- 做一版基于 `tokens_num / M` 的 shape-aware heuristic
- 小 batch decode 允许走特化 kernel
- 大 token prefill 禁用它

这说明：

- routed experts 的主要问题已经不是 `w1`
- 而是那条特化 `w2` int8 kernel 的 shape 适配很差
- 在当前实现下，禁用它比继续强行使用更好

也正因为这个修复，端到端：

- `W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph`

已经从“基本追平 bf16”提升到“稳定超过 bf16”。
