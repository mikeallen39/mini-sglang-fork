# 2026-06-28 最新 bf16 差距复盘：在补上 FullAttn Fused Prepare 和 Decode Conv 对齐之后

## 当前最好配置

本轮对比使用的 mini-sglang 最好稳定配置为：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `bf16`
- `fi + sglang linear attention + CUDA graph`

对应正式 benchmark：

- `TTFT = 123.30 ms`
- `E2E = 0.7131 s`
- `output_tps = 88.35 tok/s`

对标 sglang：

- `TTFT = 106.40 ms`
- `E2E = 0.5160 s`
- `output_tps = 124.03 tok/s`

## 这次 graph-off trace 使用

- mini-sglang：
  - `/tmp/minisgl_traces/1782659457.2514362-TP-0.trace.json`
- sglang：
  - `/tmp/1782640847.7483978-TP-0-DECODE.trace.json.gz`

说明：

- 两边都在 `CUDA graph 关闭` 条件下抓 trace
- 这里只用于归因，不用于直接比较端到端吞吐

## 关键结论

### 1. 底层 GEMV / decode kernel 本身已经不是主差距

按单次平均耗时看：

- `gemvx`
  - mini：`448.062 ms / 50600 ≈ 8.85 us`
  - sglang：`6.945 ms / 800 ≈ 8.68 us`
  - 几乎一样

- linear-attn decode kernel
  - mini：`64.645 ms / 9450 ≈ 6.84 us`
  - sglang：`1.298 ms / 150 ≈ 8.65 us`
  - mini 单次甚至不慢

- `fused_add_rmsnorm`
  - mini：`82.548 ms / 25600 ≈ 3.22 us`
  - sglang：`1.265 ms / 400 ≈ 3.16 us`
  - 几乎一样

这说明：

- 当前剩余差距已经不主要在底层 GEMV 或 decode kernel 的“单次算得慢”
- 更像是 mini-sglang 仍然做了更多额外的 kernel 与中间搬运

### 2. mini-sglang 仍有大量额外的 PyTorch 组合 kernel

这次最新 trace 中，仍然很显眼的额外 kernel 包括：

- `CatArrayBatchedCopy`
  - `52.848 ms`
- `reduce_kernel MeanOps`
  - `44.462 ms`
- `rsqrt_kernel_cuda`
  - `16.480 ms`
- `sigmoid_kernel_cuda`
  - `35.959 ms`
- 各类 `copy/add/mul` elementwise kernel

而 sglang 对应路径里，更常见的是：

- `fused_qkvzba_split_reshape_cat_contiguous_kernel`
- `_causal_conv1d_update_kernel`
- `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm`
- `fused_recurrent_gated_delta_rule_packed_decode_kernel`
- `_fused_gate_sigmoid_mul_add_kernel`

这说明：

- mini-sglang 虽然已经补掉了一部分 fused 路径
- 但从整条执行链看，仍然没有像 sglang 那样把更多准备/后处理步骤压缩到更少的 kernel 里

### 3. decode conv 对齐之后确实有收益，但不是决定性缺口

这轮把 decode conv 改成直接复用 sglang `causal_conv1d_update` 之后：

- `output_tps` 从 `86.58 -> 88.35 tok/s`
- 是有效增益

但从总体看，它只能解释剩余差距中的一小段。  
因为即便补上这条之后，mini-sglang 仍然距离 `124.03 tok/s` 很远。

### 4. MoE 自身仍有单次 kernel 差距

按单次平均耗时：

- mini `fused_moe_kernel`
  - `845.639 ms / 25600 ≈ 33.03 us`
- sglang `fused_moe_kernel`
  - `8.585 ms / 400 ≈ 21.46 us`

也就是说：

- 在清理掉一部分周边开销之后
- MoE 自身仍然有约 `1.54x` 的单次 kernel 差距

这说明 MoE 仍然值得继续盯，但应该盯的是：

- 为什么单次 `fused_moe_kernel` 仍更慢
- 而不是继续做 graph-on 下几乎没收益的 `MOE_SINGLE_KERNEL=1`

## 当前最值得继续追的方向

### 第一优先级：减少 linear-attn 周边剩余 `cat/copy/reduce/sigmoid`

因为现在：

- decode kernel 本身已经不慢
- 但周边仍然堆着大量 PyTorch kernel

这最像是 mini-sglang 剩余大差距的主要来源。

### 第二优先级：继续对齐 sglang 的 split / prepare 组织方式

尤其是：

- `QKVZ / BA split`
- q/k normalize 的组织方式
- 可能仍然存在的额外 `reshape/contiguous/cat`

### 第三优先级：继续查 MoE 单次 kernel 为何慢

当前最具体的量化信号已经有了：

- mini：`33.03 us / 次`
- sglang：`21.46 us / 次`

这是一个足够清晰、值得继续深挖的差距。

## 小结

到这一步为止，可以更明确地说：

- mini-sglang 不再是“底层算子全面慢”
- 它更像是：
  - 有一部分 sglang 已经融合掉的周边步骤仍未融合
  - 再叠加 MoE 单次 kernel 本身还慢一截

这也是为什么当前最好值能被逐步推到 `88.35 tok/s`，但还无法接近 sglang 的 `124.03 tok/s`。下一阶段不该再泛试小优化，而应该继续沿这两个明确方向收窄。
