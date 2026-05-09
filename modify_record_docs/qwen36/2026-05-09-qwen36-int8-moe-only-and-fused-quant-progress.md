# 2026-05-09 Qwen3.6 Int8 MoE Only 与 Fused Quant 进展记录

## 1. 本轮目标

本轮工作的目标不是单纯“把 int8 跑起来”，而是进一步回答两个更关键的问题：

- `Qwen3.6-35B-A3B` 在 `mini-sglang` 中走 `int8` 后，是否能相对 `bf16` 真正提速
- 如果暂时不能提速，瓶颈到底在普通 dense linear、MoE experts，还是量化前后处理

工作目录：

- `/mnt/42_store/zxz/mini-sglang/mini-sglang-fork`

模型路径：

- `/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`

环境路径：

- `/data/zxz/condaenv/minisgl`

## 2. 关键结论

截至当前，这一轮最重要的结论有四个：

1. `w8a8_int8` 全量量化目前仍然明显慢于 `bf16`
2. 慢的主要根因不在 `fused MoE` 主核本体，而在普通 dense linear 的 int8 路径过重
3. 新增 `w8a8_int8_moe_only` 后，性能已经明显逼近 `bf16`
4. 针对 MoE 路径继续做 `quant` fusion 后，服务级吞吐已经进一步提升

换句话说：

- “所有 linear 都强行上 int8” 这条路当前并不划算
- “只量化 MoE experts，并把量化相关中间步骤 fuse 掉” 是当前更有效的方向

## 3. 为什么全量 Int8 反而更慢

### 3.1 服务级对比

在保持：

- `--moe-backend fused`
- `--linear-attn-backend sglang`

一致的前提下，当前拿到的代表性结果如下。

`bf16 + fused MoE + sglang linear attn`：

- `TTFT avg = 140.69 ms`
- `E2E avg = 2.05 s`
- `output_tps = 14.62 tok/s`

`w8a8_int8 + fused MoE + sglang linear attn`：

- `TTFT avg = 227.76 ms`
- `E2E avg = 3.60 s`
- `output_tps = 8.60 tok/s`

结论很直接：

- 当前完整 `int8` 路径并没有带来服务级收益
- 它比 `bf16` 慢得比较明显

### 3.2 微基准定位

后续对单层 `int8 linear` 做了微基准，结论同样明确：

- 当前 `int8 linear` 在 `M=1~64` 的 decode 小 batch 区间，通常比 `bf16 linear` 慢 `8x~11x`
- 即使 `M=4096`，也仍然慢约 `5.5x`

根因主要有两个：

- `quantize_activation_per_token_int8(...)` 本身代价高
- `_int_mm + epilogue` 的整体开销也不低

因此，当前最不应该做的事情，是把所有普通 dense linear 都强行切到这条 int8 路径。

## 4. `w8a8_int8_moe_only` 的思路与效果

为避免普通 dense linear 拖垮整体吞吐，本轮新增了：

- `w8a8_int8_moe_only`

其语义是：

- `w8a8_int8`：模型全量 int8
- `w8a8_int8_moe_only`：只量化 MoE experts，普通 dense linear 保持 `bf16`

相关改动位于：

- `python/minisgl/quantization/__init__.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/server/args.py`

初版服务级结果：

- `w8a8_int8_moe_only + fused MoE + sglang linear attn`
- `output_tps = 12.92 tok/s`

对比基线：

- `bf16 + fused MoE + sglang linear attn = 14.62 tok/s`

结论：

- 只量化 MoE experts 后，性能已经明显逼近 `bf16`
- 这进一步证明，当前主要瓶颈并不在 MoE expert 的 int8 主路径本身，而在“全量 int8”的其他线性层及其量化前后处理

## 5. 针对 MoE 路径继续做 Quant Fusion

### 5.1 新增的 kernel

本轮在仓内新增并接入了两类 Triton quant/fusion kernel：

- `per_token_quant_int8_triton`
- `silu_and_mul_quant_int8_triton`

新增文件：

- `python/minisgl/kernel/triton/activation_quant.py`
- `python/minisgl/kernel/activation_quant.py`

接入改动：

- `python/minisgl/kernel/__init__.py`
- `python/minisgl/kernel/moe_impl.py`
- `python/minisgl/moe/fused.py`

### 5.2 数值对拍

当前两段 fused quant 的对拍结果都在可接受范围内：

- `silu_and_mul_quant_int8_triton`
  - `scale_max_diff ≈ 2.2e-4`
  - `q_max_abs_diff = 1`
- `per_token_quant_int8_triton`
  - `scale_max_diff = 0.0`
  - `q_max_abs_diff = 1`

这类误差水平足够支持当前阶段做服务级性能验证。

### 5.3 微基准收益

本轮重点看了 MoE int8 两段中间处理：

- `stage1`：第一段 per-token quant
- `stage2`：`silu_and_mul + quant`

代表性结果如下。

`M=8`：

- `stage1: 0.1198 -> 0.0812 ms`，提升约 `32.25%`
- `stage2: 0.1644 -> 0.0626 ms`，提升约 `61.90%`

`M=64`：

- `stage1: 0.3160 -> 0.3076 ms`，提升约 `2.67%`
- `stage2: 0.1916 -> 0.1696 ms`，提升约 `11.51%`

结论：

- `stage2` 的收益最明显
- decode 小 batch 场景收益更大
- 当前值得继续投入的方向是“把量化相关中间处理 fuse 掉”，而不是继续把普通 dense linear 全量 int8 化

## 6. 当前服务级最新进展

在 `w8a8_int8_moe_only + fused MoE + sglang linear attn` 的基础上，接入 fused quant 后，已经成功拿到一版服务级提升结果：

- `TTFT avg = 152.18 ms`
- `E2E avg = 2.18 s`
- `output_tps = 13.73 tok/s`

和 `w8a8_int8_moe_only` 初版相比：

- `12.92 tok/s -> 13.73 tok/s`
- 吞吐提升约 `6.3%`

和 `bf16` 基线相比：

- `13.73 tok/s` 对 `14.62 tok/s`

这说明：

- 当前优化方向是有效的
- 经过 `moe_only + fused quant` 后，性能已经非常接近 `bf16`
- 剩余差距已经收敛到一个相对可继续打磨的区间

## 7. 本轮“为什么卡很久”的定位结果

### 7.1 现象

在继续把 `stage1 + stage2` 都接入服务后，新一轮服务启动耗时非常长，表面上看像“卡住”。

对应进程：

- 主进程：`195387`
- worker：`195803`

### 7.2 关键证据

`195803` 的状态检查结果：

- `State: D (disk sleep)`
- `wchan: wait_on_page_bit_common`

GPU 状态：

- GPU7 显存占用约 `49.7 GiB`
- `GPU-Util = 0%`

这不是典型的算子死锁形态，更像是：

- worker 还没开始真正执行 kernel
- 仍然卡在模型权重按页加载

更关键的是，`/proc/195803/io` 中 `read_bytes` 一直在持续增长。

2026-05-09 UTC 时间抽样：

- `02:53:54`：`51,779,596,288`
- `02:54:22`：`53,661,003,776`
- `02:55:32`：`58,010,107,904`
- `02:57:12`：`63,002,886,144`
- `02:59:13`：`69,820,637,184`

这说明：

- 进程并没有停住
- 它在持续从远端存储读取 `safetensors` 分片

结合之前的 `lsof` 结果，当前模型分片位于 NFS 后端。

因此，这一轮“运行很久”的最合理结论是：

- 首要问题是远端 NFS 加载慢
- 不是 `stage1/stage2 fused quant` 自身直接造成的 Triton 死锁

## 8. 当前阶段判断

截至目前，可以给出下面这个更准确的判断：

1. 当前完整 `w8a8_int8` 仍然不具备相对 `bf16` 的服务级优势
2. 主要问题并不在 `fused MoE int8` 主路径本身
3. `w8a8_int8_moe_only` 已经把问题大幅收敛
4. 对 MoE 中间量化步骤做 fusion 后，收益已经在服务级上体现出来
5. 目前服务长时间未 ready 的主因，优先怀疑 NFS 权重加载慢，而不是新 kernel 卡死

## 9. 下一步建议

后续建议按下面顺序继续推进：

1. 等当前这次远端加载真正完成，再对 `stage1 + stage2` 最终版做一次完整服务级测速
2. 如果最终版稳定，则继续对比：
   - `bf16 + fused MoE + sglang linear attn`
   - `w8a8_int8_moe_only + fused MoE + sglang linear attn`
   - `w8a8_int8_moe_only + fused quant + fused MoE + sglang linear attn`
3. 如果最终版服务仍不稳定，则优先保留已被服务级证明有效的 `stage2 fused quant`
4. 后续若要真正超过 `bf16`，重点应继续放在：
   - MoE 路径的更多 fusion
   - decode 小 batch 场景的 kernel 配置调优
   - 避免普通 dense linear 进入当前高开销 int8 路径

当前结论可以简化成一句话：

- `int8` 不是完全没机会提速，而是“全量 int8”这条路当前选错了重点；真正有效的方向，是 `moe_only + 更深的 kernel fusion`

## 10. 额外修复：消除中等 Prefill 首发抖动

### 10.1 问题现象

在之前的一次服务级测试里，`medium_prefill_short_decode (256/32)` 曾出现：

- `TTFT = 1608.64 ms`
- `E2E = 3.52 s`

这个结果和同服务上的短 case 明显不一致，看起来像是：

- `256` 长度的 prefill 场景仍然有严重性能问题

但继续复测后发现，这个判断并不准确。

### 10.2 复测结论

对同一服务连续发送 3 次 `256/32` 请求后，结果稳定在：

- `TTFT ≈ 145~147 ms`
- `E2E ≈ 2.07~2.09 s`

这说明：

- `1608ms` 并不是 steady-state 的真实水平
- 更像是第一次命中新 Triton shape / config 时的 JIT 编译或初始化抖动

也就是说，问题不是：

- `medium prefill` 本身长期很慢

而是：

- 第一次真实请求把编译成本暴露给了用户

### 10.3 采取的修复办法

为了解决这个“首发抖动”，在：

- `python/minisgl/scheduler/scheduler.py`

中新增了一个非常小的启动期 prewarm 逻辑。

触发条件：

- `quantization == w8a8_int8_moe_only`
- `moe_backend == fused`
- `linear_attn_backend == sglang`
- `tp_size == 1`

做法：

- 在 scheduler 宣告 ready 之前
- 内部构造两个 dummy prefill batch
- 预热长度分别为：
  - `64`
  - `256`

这样可以把当前这条：

- `int8_moe_only + fused MoE + sglang linear attn + fused quant`

路径中最关键的 Triton shape 提前编译掉，而不是让第一位真实用户来承担 JIT 延迟。

启动日志中能看到：

- `Start Triton prewarm for prefill lengths: [64, 256]`
- `Triton prewarm finished.`

### 10.4 修复后验证结果

使用带 prewarm 的新服务后，第一次 `256/32` 请求测到：

- `TTFT = 239.59 ms`
- `E2E = 2.32 s`

紧接着第二次 `256/32` 请求测到：

- `TTFT = 152.10 ms`
- `E2E = 2.20 s`

同时，功能验证仍然正常：

- 输入：`hello`
- 输出：`Hello! How can I help you today?`

### 10.5 当前结论

这次修复已经证明：

- 之前 `256/32` 的 `1608ms TTFT` 主要来自首发 JIT 抖动
- 不是 steady-state kernel 长期退化
- 启动期 prewarm 可以有效把这部分成本前移

因此当前对 `medium prefill` 的判断应更新为：

- steady-state 下它已经回到与短 case 同一量级
- 当前剩余优化重点，应继续放在 steady-state 吞吐，而不是误把首发编译时间当成主瓶颈
