# 2026-05-09 Qwen3.6 Shared Expert Int8、分段 Profile 与最新性能对照

## 1. 本轮目标

在前一轮已经完成：

- `Qwen3.6-35B-A3B` 跑通
- `fused MoE` 跑通
- `sglang linear attention` 跑通
- `w8a8_int8_moe_only` 跑通

之后，本轮重点不再是“能不能跑”，而是回答下面三个更具体的问题：

1. 为什么 `int8_moe_only` 仍然没有稳定超过 `bf16`
2. `MoE` 路径里到底是哪一段最慢
3. `shared expert` 这条每 token 必跑的路径，是否还是 `bf16`，以及它对整体性能的影响有多大

## 2. 本轮新增改动

### 2.1 给 shared expert 接上 `moe_only` 量化

之前的 `w8a8_int8_moe_only` 只覆盖 routed experts，没有覆盖 `Qwen3.6` 里的 `shared expert`。

本轮新增了一个最小扩展机制：

- 在线性层基类里加入 `quantize_in_moe_only` 标志
- 在 `w8a8_int8_moe_only` 模式下，允许指定的线性层也做 int8 权重量化

对应改动：

- `python/minisgl/layers/linear.py`

随后把 `Qwen3_5SharedExpert` 的两层都接入该机制：

- `gate_up_proj`
- `down_proj`

对应改动：

- `python/minisgl/models/qwen3_5_moe.py`

这意味着现在的 `w8a8_int8_moe_only` 不再只是“量化 routed experts”，而是：

- routed experts 量化
- shared expert 量化
- 普通 dense linear 仍保持 `bf16`

### 2.2 新增 `SparseMoE` 分段 profile

为了避免只看到层级总时间，本轮新增了 `SparseMoE` 内部分段 profile。

新增环境变量：

- `MINISGL_PROFILE_SPARSE_MOE=1`

对应改动：

- `python/minisgl/env.py`
- `python/minisgl/models/qwen3_5_moe.py`

当前会分三段统计：

- `router`
- `experts`
- `shared`

用于回答：

- routed experts 还是 shared expert 更慢
- router 在整体中占比是否可以忽略

### 2.3 保留并继续使用的 profile / 优化基础设施

本轮是在已有基础上继续推进，相关代码本地也一并保留并准备提交：

- `MINISGL_PROFILE_QWEN35`
- `MINISGL_PROFILE_FUSED_MOE`
- `FusedMoE` workspace 复用
- `FusedMoE` 四段 profile：`w1 / stage2 / w2 / reduce`
- Triton activation quant 配置启发式

对应文件：

- `python/minisgl/env.py`
- `python/minisgl/moe/fused.py`
- `python/minisgl/kernel/activation_quant.py`
- `python/minisgl/models/qwen3_5_moe.py`

## 3. 关键 profile 结果

### 3.1 `SparseMoE` 分段结果

在 `MINISGL_PROFILE_SPARSE_MOE=1` 下，稳定阶段的代表性结果大致为：

- `router ≈ 0.037 ~ 0.039 ms`
- `experts ≈ 0.66 ~ 0.69 ms`
- `shared ≈ 0.44 ~ 0.46 ms`

结论：

1. `router` 很小，不是主要瓶颈
2. `shared expert` 不是边角开销，而是 `MoE` 段里的第二大块
3. 之前 `moe_only` 没覆盖 `shared expert`，确实漏掉了一条每 token 必跑路径

### 3.2 `FusedMoE` 分段结果

在 `MINISGL_PROFILE_FUSED_MOE=1` 下，稳定阶段的代表性结果大致为：

- `w1 ≈ 0.18 ms`
- `stage2 ≈ 0.066 ~ 0.068 ms`
- `w2 ≈ 0.089 ~ 0.092 ms`
- `reduce ≈ 0.064 ~ 0.067 ms`

结论：

1. `FusedMoE` 内部最大头仍然是 `w1`
2. `stage2` 和 `reduce` 都不是决定性瓶颈
3. 这说明“超过 bf16”不能只靠继续微调小 kernel 参数，后续仍需要更深的 kernel fusion / dataflow 优化

### 3.3 层级 profile 结果

在 `MINISGL_PROFILE_QWEN35=1` 下，稳定阶段的代表性结果大致为：

- `full_attn ≈ 0.21 ~ 0.22 ms`
- `linear_attn ≈ 0.43 ~ 0.45 ms`
- `mlp ≈ 1.34 ~ 1.36 ms`

结论：

1. 当前最大头仍然是 `MLP / MoE`
2. `linear attention` 已经成为第二大头
3. 即使把 `MoE` 继续压下去，后面也迟早要继续处理 `linear attention`

## 4. 本轮非 profile 真实测速结果

注意：

- profile 模式下会插入 `cuda event + synchronize`
- 因此 profile 模式下的吞吐显著偏低
- 真正有参考意义的服务吞吐，必须看非 profile 结果

### 4.1 当前 int8 版本

配置：

- `--moe-backend fused`
- `--linear-attn-backend sglang`
- `--quantization w8a8_int8_moe_only`
- `--dtype bfloat16`

非 profile 结果：

`short_prefill_short_decode`:

- `TTFT avg = 166.88 ms`
- `E2E avg = 2.45 s`
- `output_tps = 12.66 tok/s`

`medium_prefill_short_decode`:

- `TTFT avg = 190.25 ms`
- `E2E avg = 1.78 s`
- `output_tps = 11.80 tok/s`

### 4.2 同机同代码 bf16 对照

配置：

- `--moe-backend fused`
- `--linear-attn-backend sglang`
- 不加 `--quantization`
- `--dtype bfloat16`

非 profile 结果：

`short_prefill_short_decode`:

- `TTFT avg = 150.94 ms`
- `E2E avg = 2.06 s`
- `output_tps = 14.55 tok/s`

`medium_prefill_short_decode`:

- `TTFT avg = 492.29 ms`
- `E2E avg = 1.83 s`
- `output_tps = 11.46 tok/s`

## 5. 当前结论

截至本轮，可以给出下面这个更准确的判断：

1. `shared expert` 确实是 `mlp` 里的重要耗时块，之前漏量化是一个真实缺口
2. 现在该缺口已经补上，但 `int8_moe_only` 仍未在短 decode 吞吐上超过 `bf16`
3. 当前短 decode 场景下：
   - `int8_moe_only = 12.66 tok/s`
   - `bf16 = 14.55 tok/s`
4. 当前中等 prefill 场景下：
   - `int8_moe_only = 11.80 tok/s`
   - `bf16 = 11.46 tok/s`
5. 因此当前状态是：
   - 中等 prefill 已接近或略超 `bf16`
   - 短 decode 仍明显落后

## 6. 对“为什么还没超过 bf16”的判断

当前最合理的判断是：

1. `MoE` 虽然已经基本走上正确方向，但还没有形成足够大的结构性优势
2. `shared expert` 现在虽然量化了，但它仍然走的是通用 linear 路径，而不是像 routed experts 那样的深融合路径
3. `linear attention` 本身已经达到 `0.43 ~ 0.45 ms` 量级，也在拖住短 decode 吞吐

所以后续想要“明显超过 bf16”，最该继续做的不是：

- 小幅调 Triton 参数
- 继续纠缠 router

而是：

1. 给 `shared expert` 做更深的 fused int8 路径
2. 继续压 `linear attention`
3. 如果还要继续做 `MoE`，优先看 `w1` 这一段还能不能进一步融合或改写数据流

## 7. 本轮建议的下一步

建议下一轮按下面顺序继续：

1. 优先做 `shared expert` 的 fused int8 kernel 路径
2. 复测：
   - `bf16 + fused MoE + sglang linear attn`
   - `int8_moe_only + shared expert int8 + fused MoE + sglang linear attn`
3. 如果短 decode 仍落后，再转去继续优化 `linear attention`

## 8. 本轮涉及文件

- `python/minisgl/env.py`
- `python/minisgl/kernel/activation_quant.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/models/qwen3_5_moe.py`
- `python/minisgl/moe/fused.py`
