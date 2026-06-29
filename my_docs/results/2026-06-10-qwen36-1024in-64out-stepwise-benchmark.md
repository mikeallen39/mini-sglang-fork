# 2026-06-10 Qwen3.6 1024 In / 64 Out 分步性能实验

## 1. 统一测试口径

- 基准脚本：`benchmark/online/bench_qwen36_1024in_64out.py`
- 模型：`/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`
- 环境：`/mnt/82_store/zxz/condaenv/minisgl`
- 统一 workload：
  - `input_tokens = 1024`
  - `output_tokens = 64`
  - `runs = 5`
  - `run1` 单独记录
  - `run2-run5` 作为稳态平均
- 统一统计：
  - `TTFT`
  - `E2E`
  - `output_tps`
  - `avg_ms_per_output_token`
  - `avg_output_tokens`
  - `peak memory`（若可获取）

## 1.1 正确启动方式

从 2026-06-28 起，`benchmark/online/bench_qwen36_1024in_64out.py` 的默认口径已调整为：

- `input_token_mode=final-chat`
- `input_tokens=1024` 指的是**应用 chat template 之后、真正送进模型的最终 prompt 长度**

这样可以避免旧口径下出现的误差：

- 旧模式按 `raw-content=1024` 截断用户文本
- 但服务端会再套 chat template
- 对 Qwen3.6 这条路径，最终实际输入会变成 `1036` token，而不是 `1024`

如需复现旧行为，可显式传：

```bash
python benchmark/online/bench_qwen36_1024in_64out.py \
  --input-token-mode raw-content
```

当前实测：

- `raw-content` 模式：`user_content_tokens=1024`，`final_prompt_tokens=1036`
- `final-chat` 模式：`user_content_tokens=1012`，`final_prompt_tokens=1024`

对于本文件里的 `1024 in / 64 out` benchmark，服务启动时必须保证可用 KV 容量明显大于 `1024 + 64`，否则长输入请求会被直接丢弃，导致测速结果失真。

当前确认可用的启动方式如下：

```bash
PATH=/mnt/82_store/zxz/condaenv/minisgl/bin:/usr/local/cuda-12.4/bin:$PATH \
PYTHONPATH=/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python \
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m minisgl.server.launch \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --host 127.0.0.1 \
  --port 1919 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --expert-parallel-size 1 \
  --moe-backend fused \
  --linear-attn-backend sglang \
  --cuda-graph-max-bs 1 \
  --attention-backend fi \
  --cache-type naive \
  --num-pages 4096
```

然后使用统一 benchmark：

```bash
PATH=/mnt/82_store/zxz/condaenv/minisgl/bin:$PATH \
python benchmark/online/bench_qwen36_1024in_64out.py \
  --base-url http://127.0.0.1:1919/v1 \
  --model /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --run1-output-file my_docs/results/<name>_run1_output.txt
```

## 1.2 易错点

- 不能把 `--num-pages` 设成 `128` 再拿来测 `1024 in / 64 out`。
  原因：当前 `page_size=1`，所以 `num_pages=128` 实际只提供 `128` token 容量。此时日志会出现：
  - `Input sequence length 1036 exceeds 128, request ... is dropped.`
  这种情况下服务虽然能启动，但长输入 benchmark 结果无效。

- 参数名应使用 `--attention-backend`，不要写成 `--attn-backend`。
  后者不会被当前 `launch.py` 识别。

- 对于本文件中的 `Fused MoE + SGLang Linear Attention + CUDA Graph` 结果，后续复测时应固定：
  - `attention-backend=fi`
  - `linear-attn-backend=sglang`
  - `moe-backend=fused`
  - `cuda-graph-max-bs=1`
  - `cache-type=naive`
  - `num-pages=4096`

- 复测时应优先确认服务日志里出现：
  - `Allocating 4096 tokens for KV cache`
  - `Scheduler is ready`
  再启动 benchmark 脚本。

## 2. 实验顺序

按下面顺序逐步增加优化：

1. 基线：`torch MoE + torch linear attention + bf16 + graph off + tp=1 + ep=1`
2. 只切 `fused MoE`
3. 再切 `sglang linear attention`
4. 再测试 `CUDA graph`
5. 再测试 `int8 w8a8`
6. 最后测试 `TP/EP`

## 3. 结果

### 3.0 总览表

| 配置 | TTFT (run2-5 avg) | E2E (run2-5 avg) | output_tps | avg_ms_per_output_token | avg_output_tokens | 运行中显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 6413.07 ms | 38.4432 s | 1.64 tok/s | 610.21 ms | 63.00 | 67321 MiB |
| Fused MoE | 4477.62 ms | 8.8629 s | 7.11 tok/s | 140.68 ms | 63.00 | 67405 MiB |
| Fused MoE + SGLang Linear Attention | 236.11 ms | 4.1506 s | 15.18 tok/s | 65.88 ms | 63.00 | 67369 MiB |
| Fused MoE + SGLang Linear Attention + CUDA Graph | 195.07 ms | 0.9342 s | 67.44 tok/s | 14.83 ms | 63.00 | 67453 MiB |
| W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph | 189.04 ms | 0.8867 s | 71.05 tok/s | 14.07 ms | 63.00 | 35713 MiB |
| W8A8 Selective Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph | 187.69 ms | 0.8757 s | 71.94 tok/s | 13.90 ms | 63.00 | 35759 MiB |
| W8A8 Selective Int8 + LayerNorm Prequant Reuse + Fused MoE + SGLang Linear Attention + CUDA Graph | 188.55 ms | 0.8853 s | 71.17 tok/s | 14.05 ms | 63.00 | 35759 MiB |
| W8A8 Selective Int8 + LinearAttn Prefill Q/K Norm Outside Kernel + Fused MoE + SGLang Linear Attention + CUDA Graph | 179.30 ms | 0.8686 s | 72.53 tok/s | 13.79 ms | 63.00 | 35759 MiB |
| W8A8 Selective Int8 + LinearAttn Prefill Launch Tuning + Q/K Norm Outside Kernel + Fused MoE + SGLang Linear Attention + CUDA Graph | 165.33 ms | 0.8158 s | 77.22 tok/s | 12.95 ms | 63.00 | 35759 MiB |
| W8A8 Selective Int8 + Vendored Full-Kernel Chunk Prefill + Fused MoE + SGLang Linear Attention + CUDA Graph | 113.29 ms | 0.7610 s | 82.79 tok/s | 12.08 ms | 63.00 | 35759 MiB |
| W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph + EP2 | 208.44 ms | 1.0111 s | 62.31 tok/s | 16.05 ms | 63.00 | 20775 MiB x 2 |
| W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph + EP4 | 205.37 ms | 1.0244 s | 61.50 tok/s | 16.26 ms | 63.00 | 13001 MiB x 4 |
| bf16 (不量化) + Fused MoE + SGLang Linear Attention + CUDA Graph | 118.49 ms | 0.8297 s | 75.93 tok/s | 13.17 ms | 63.00 | 68137 MiB |
| bf16 + Gemma Fused Norm + Q/K Inplace Fused Norm Copy-Back + Fused MoE + SGLang Linear Attention + CUDA Graph | 125.24 ms | 0.7613 s | 82.75 tok/s | 12.08 ms | 63.00 | 68153 MiB |
| bf16 + Gemma Fused Norm + FullAttn Fused Prepare + Fused MoE + SGLang Linear Attention + CUDA Graph | 121.91 ms | 0.7277 s | 86.58 tok/s | 11.55 ms | 63.00 | 68153 MiB |
| bf16 + Gemma Fused Norm + FullAttn Fused Prepare + Fused Gate Mul + Fused MoE + SGLang Linear Attention + CUDA Graph | 122.81 ms | 0.7279 s | 86.55 tok/s | 11.55 ms | 63.00 | 68153 MiB |
| bf16 + Gemma Fused Norm + FullAttn Fused Prepare + SGLang CausalConv1dUpdate Decode + Fused MoE + SGLang Linear Attention + CUDA Graph | 123.30 ms | 0.7131 s | 88.35 tok/s | 11.32 ms | 63.00 | 68155 MiB |

**sglang main 对标（bf16，相同 workload，disable radix cache）：**

| sglang bf16 | 106.40 ms | 0.5160 s | 124.03 tok/s | — | 63.00 | — |

> mini-sglang bf16 vs sglang bf16 公平对比下，output_tps 差距约 1.63x（75.93 vs 124.03）。说明差距并非来自 W8A8 量化，而是 decode 阶段存在结构性差异（见 Handoff 中的 kernel count 分析和 2026-06-27 的 profiling 探索）。

### 3.1 Baseline

- 配置：
  - `moe-backend=torch`
  - `linear-attn-backend=torch`
  - `dtype=bfloat16`
  - `graph=0`
  - `tp=1`
  - `ep=1`
  - `quantization=none`
- 服务显存：
  - 初始化后空闲显存：`13.82 GiB`
  - 运行中占用：`67321 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/baseline_rerun_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文科学说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
  - 末尾截断在句中，符合 `output_tokens=64` 的预期
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 7254.20 ms | 39.3084 s | 63 |
| run2 | 6416.83 ms | 38.4262 s | 63 |
| run3 | 6407.57 ms | 38.4562 s | 63 |
| run4 | 6412.11 ms | 38.4539 s | 63 |
| run5 | 6415.76 ms | 38.4364 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 6413.07 ms |
| run2-run5 avg E2E | 38.4432 s |
| run2-run5 output_tps | 1.64 tok/s |
| avg_ms_per_output_token | 610.21 ms |
| avg_output_tokens | 63.00 |

### 3.2 Only Switch Fused MoE

- 配置：
  - `moe-backend=fused`
  - `linear-attn-backend=torch`
  - `dtype=bfloat16`
  - `graph=0`
  - `tp=1`
  - `ep=1`
  - `quantization=none`
- 服务显存：
  - 初始化后空闲显存：`13.82 GiB`
  - 运行中占用：`67405 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/fused_moe_rerun2_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文科学说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
  - 末尾截断在句中，符合 `output_tokens=64` 的预期
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 5425.46 ms | 9.8339 s | 63 |
| run2 | 4426.99 ms | 8.8033 s | 63 |
| run3 | 4423.81 ms | 8.7943 s | 63 |
| run4 | 4473.97 ms | 8.9083 s | 63 |
| run5 | 4585.72 ms | 8.9457 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 4477.62 ms |
| run2-run5 avg E2E | 8.8629 s |
| run2-run5 output_tps | 7.11 tok/s |
| avg_ms_per_output_token | 140.68 ms |
| avg_output_tokens | 63.00 |
- 相对 baseline：
  - `TTFT` 提升约 `30.2%`
  - `E2E` 提升约 `76.9%`
  - `output_tps` 提升约 `333.5%`

### 3.3 Then Switch SGLang Linear Attention

- 配置：
  - `moe-backend=fused`
  - `linear-attn-backend=sglang`
  - `dtype=bfloat16`
  - `graph=0`
  - `tp=1`
  - `ep=1`
  - `quantization=none`
- 服务显存：
  - 初始化后空闲显存：`13.82 GiB`
  - 运行中占用：`67369 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/fused_moe_sglang_linear_attn_rerun_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文科学说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
  - 末尾截断在句中，符合 `output_tokens=64` 的预期
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 1178.97 ms | 5.0917 s | 63 |
| run2 | 236.74 ms | 4.1440 s | 63 |
| run3 | 235.90 ms | 4.1392 s | 63 |
| run4 | 236.04 ms | 4.1365 s | 63 |
| run5 | 235.75 ms | 4.1828 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 236.11 ms |
| run2-run5 avg E2E | 4.1506 s |
| run2-run5 output_tps | 15.18 tok/s |
| avg_ms_per_output_token | 65.88 ms |
| avg_output_tokens | 63.00 |
- 相对 baseline：
  - `TTFT` 提升约 `96.3%`
  - `E2E` 提升约 `89.2%`
  - `output_tps` 提升约 `825.6%`
- 相对 `fused MoE + torch linear attention`：
  - `TTFT` 提升约 `94.7%`
  - `E2E` 提升约 `53.2%`
  - `output_tps` 提升约 `113.5%`

### 3.4 Then Enable CUDA Graph

- 配置：
  - `moe-backend=fused`
  - `linear-attn-backend=sglang`
  - `dtype=bfloat16`
  - `graph=1`
  - `tp=1`
  - `ep=1`
  - `quantization=none`
- 服务显存：
  - graph capture 前空闲显存：`13.82 GiB`
  - graph capture 后空闲显存：`13.72 GiB`
  - 运行中占用：`67453 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/fused_moe_sglang_linear_attn_cudagraph_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文科学说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
  - 末尾截断在句中，符合 `output_tokens=64` 的预期
- 兼容性修复：
  - 去掉了 [python/minisgl/moe/fused.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/moe/fused.py) 中 `moe_align_block_size()` 的静默 torch fallback
  - 定位出 graph capture 阻塞点并非 CUDA graph 本身，而是 `sgl_kernel` routing op 缺失
  - 修复了 [python/sgl_kernel/_routing_loader.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/sgl_kernel/_routing_loader.py) 的硬编码路径问题
  - 使用 `third_party/sgl-kernel` 构建最小 `moe-routing-only` wheel，并补入本地：
    - `python/sgl_kernel/sm90/common_ops.abi3.so`
  - 修复后 `fused MoE + sglang linear attention + graph=1` 成功完成 graph capture
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 259.68 ms | 0.9978 s | 63 |
| run2 | 196.05 ms | 0.9353 s | 63 |
| run3 | 194.89 ms | 0.9340 s | 63 |
| run4 | 194.66 ms | 0.9339 s | 63 |
| run5 | 194.70 ms | 0.9337 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 195.07 ms |
| run2-run5 avg E2E | 0.9342 s |
| run2-run5 output_tps | 67.44 tok/s |
| avg_ms_per_output_token | 14.83 ms |
| avg_output_tokens | 63.00 |
- 相对 baseline：
  - `TTFT` 提升约 `97.0%`
  - `E2E` 提升约 `97.6%`
  - `output_tps` 提升约 `4038%`
- 相对 `fused MoE + sglang linear attention + graph=0`：
  - `TTFT` 提升约 `86.7%`
  - `E2E` 提升约 `97.4%`
  - `output_tps` 提升约 `3711%`

### 3.5 Then Enable W8A8 Int8

- 配置：
  - `moe-backend=fused`
  - `linear-attn-backend=sglang`
  - `dtype=bfloat16`
  - `graph=1`
  - `tp=1`
  - `ep=1`
  - `quantization=w8a8_int8`
- 服务显存：
  - graph capture 后运行中占用：`35713 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/w8a8_int8_sglkernel_restarted_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文科学说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
  - 末尾截断在句中，符合 `output_tokens=64` 的预期
- 兼容性修复：
  - 去掉了 [python/minisgl/engine/engine.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/engine/engine.py) 中 `w8a8_int8` 强制关闭 CUDA graph 的硬编码逻辑
  - 实测 `w8a8_int8 + fused MoE + sglang linear attention + graph=1` 可以成功完成 graph capture 并正常对外服务
  - 进一步修复了 `w8a8` dense int8 接入：
    - `_apply_int8_scaled_mm()` 改为优先使用环境版 `sgl_kernel.int8_scaled_mm`
    - 权重量化布局改为 `sgl_kernel` 所需的列主序 `mat_b`
  - 进一步修复了 routed experts 的 int8 路径：
    - routed experts 本身早已量化，但 `fused_moe_w2_silu_int8` 这条特化 `w2` kernel 在 Qwen3.6 当前 shape 上明显退化
    - 关闭该特化 kernel 后，routed experts 的 `fused int8` microbench 从约 `1.96 ms` 降到约 `1.00 ms`
    - 修复后端到端 `W8A8` 已经超过当前 `bf16` 最优路径
  - 旧的 `25.91 tok/s` 结果来自接入修复前的过时服务，不再作为最终结论
- 接入修复前的历史结果（保留作为经验记录）：
  - `run1_output_file = my_docs/results/w8a8_int8_run1_output.txt`
  - 稳态结果：
    - `TTFT = 298.54 ms`
    - `E2E = 2.4319 s`
    - `output_tps = 25.91 tok/s`
    - `avg_ms_per_output_token = 38.60 ms`
  - 这组旧结果的问题不在 `w8a8` 理论本身，而在于：
    - 服务没有真正命中环境版 `sgl_kernel.int8_scaled_mm`
    - dense int8 权重布局不满足 `sgl_kernel` 所需的列主序约束
    - 因而实际更多落在较慢的 fallback/int8 替代路径上
  - 这是本轮实验的一个重要教训：
    - 端到端看到“int8 比 bf16 慢很多”时，不能立刻下结论说 kernel 理论失效
    - 必须先确认服务实际命中的到底是哪条 int8 kernel 路径，以及输入/权重布局是否满足该 kernel 的约束
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 197.08 ms | 0.8944 s | 63 |
| run2 | 189.21 ms | 0.8866 s | 63 |
| run3 | 188.68 ms | 0.8864 s | 63 |
| run4 | 189.17 ms | 0.8868 s | 63 |
| run5 | 189.13 ms | 0.8869 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 189.04 ms |
| run2-run5 avg E2E | 0.8867 s |
| run2-run5 output_tps | 71.05 tok/s |
| avg_ms_per_output_token | 14.07 ms |
| avg_output_tokens | 63.00 |
- 相对 baseline：
  - `TTFT` 提升约 `97.1%`
  - `E2E` 提升约 `97.7%`
  - `output_tps` 提升约 `4232%`
- 相对 `fused MoE + sglang linear attention + CUDA graph`：
  - `TTFT` 更快约 `2.7%`
  - `E2E` 更快约 `5.1%`
  - `output_tps` 更高约 `5.4%`

### 3.6 Then Try Expert Parallel (EP=2 / EP=4)

- 配置：
  - `moe-backend=fused`
  - `linear-attn-backend=sglang`
  - `dtype=bfloat16`
  - `graph=1`
  - `tp=1`
  - `quantization=w8a8_int8`
  - `attention-backend=fi`
  - `cache-type=naive`
  - `num-pages=4096`
- 启动注意事项：
  - 当前 `EP` 路径必须加 `--disable-pynccl`
  - 原因：默认 `pynccl` 扩展会报 `undefined symbol: ncclCommWindowRegister`
  - 在当前 NFS 环境下，多进程 `spawn + import` 很慢，`ep=2/4` 启动明显慢于 `ep=1`
  - 但只要继续等待，服务可以正常进入 `Scheduler is ready`
- attention 并行方式说明：
  - 本组固定 `tp=1`
  - 因此 attention 没有被 tensor parallel 切分
  - attention / norm / 非 MoE dense 会在每个 EP rank 上各保留一份完整副本
  - 真正被 `EP` 切分的是 experts

#### 3.6.1 EP=2

- 配置：
  - `ep=2`
  - `CUDA_VISIBLE_DEVICES=1,2`
- 服务显存：
  - `GPU1 ≈ 20775 MiB`
  - `GPU2 ≈ 20775 MiB`
- `run1` 输出文件：
  - `my_docs/results/w8a8_int8_ep2_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 1005.18 ms | 1.8068 s | 63 |
| run2 | 208.19 ms | 1.0119 s | 63 |
| run3 | 208.54 ms | 1.0107 s | 63 |
| run4 | 208.40 ms | 1.0119 s | 63 |
| run5 | 208.65 ms | 1.0099 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 208.44 ms |
| run2-run5 avg E2E | 1.0111 s |
| run2-run5 output_tps | 62.31 tok/s |
| avg_ms_per_output_token | 16.05 ms |
| avg_output_tokens | 63.00 |

- 相对 `ep=1`：
  - `TTFT` 提升约 `9.2%`
  - `E2E` 变慢约 `5.5%`
  - `output_tps` 下降约 `5.2%`

#### 3.6.2 EP=4

- 配置：
  - `ep=4`
  - `CUDA_VISIBLE_DEVICES=1,2,3,4`
- 服务显存：
  - `GPU1 ≈ 13001 MiB`
  - `GPU2 ≈ 13025 MiB`
  - `GPU3 ≈ 13001 MiB`
  - `GPU4 ≈ 12977 MiB`
- `run1` 输出文件：
  - `my_docs/results/w8a8_int8_ep4_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 221.02 ms | 1.0391 s | 63 |
| run2 | 207.70 ms | 1.0268 s | 63 |
| run3 | 204.09 ms | 1.0225 s | 63 |
| run4 | 205.37 ms | 1.0252 s | 63 |
| run5 | 204.31 ms | 1.0231 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 205.37 ms |
| run2-run5 avg E2E | 1.0244 s |
| run2-run5 output_tps | 61.50 tok/s |
| avg_ms_per_output_token | 16.26 ms |
| avg_output_tokens | 63.00 |

- 相对 `ep=1`：
  - `TTFT` 提升约 `10.5%`
  - `E2E` 变慢约 `6.9%`
  - `output_tps` 下降约 `6.4%`

### 3.7 Then Try Selective Int8

- 配置：
  - 基于 `W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph`
  - 保留大层 `int8`
  - 将已验证收益不佳的小层回退到 `bf16`：
    - `moe_router_gate`
    - `shared_expert_gate`
    - `linear_attn_in_proj_ba`
- 服务显存：
  - 运行中占用：`35759 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/output_texts/selective_int8_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 259.13 ms | 0.9465 s | 63 |
| run2 | 188.87 ms | 0.8770 s | 63 |
| run3 | 187.35 ms | 0.8753 s | 63 |
| run4 | 187.18 ms | 0.8754 s | 63 |
| run5 | 187.35 ms | 0.8753 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 187.69 ms |
| run2-run5 avg E2E | 0.8757 s |
| run2-run5 output_tps | 71.94 tok/s |
| avg_ms_per_output_token | 13.90 ms |
| avg_output_tokens | 63.00 |

- 相对全量当前最佳 `W8A8`：
  - `TTFT` 更快约 `0.7%`
  - `E2E` 更快约 `1.2%`
  - `output_tps` 更高约 `1.3%`

### 3.8 Then Try LayerNorm Prequant Reuse

- 配置：
  - 基于 `Selective Int8`
  - 将 layer 边界上的 `GemmaRMSNormFused.forward_with_quant(...)` 接到后续 attention / MLP
  - 复用同一份 `x_q/x_scale`，避免模块内部再次量化同一输入
- 服务显存：
  - 运行中占用：`35759 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/output_texts/selective_int8_rmsnorm_quant_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 247.23 ms | 0.9432 s | 63 |
| run2 | 189.07 ms | 0.8858 s | 63 |
| run3 | 187.85 ms | 0.8846 s | 63 |
| run4 | 189.24 ms | 0.8859 s | 63 |
| run5 | 188.04 ms | 0.8847 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 188.55 ms |
| run2-run5 avg E2E | 0.8853 s |
| run2-run5 output_tps | 71.17 tok/s |
| avg_ms_per_output_token | 14.05 ms |
| avg_output_tokens | 63.00 |

- 结论：
  - 这版 layer-level `RMSNorm + prequant reuse` 没有继续优于 `Selective Int8`
  - 相对 `Selective Int8`：
    - `TTFT` 变慢约 `0.46%`
    - `E2E` 变慢约 `1.10%`
    - `output_tps` 下降约 `1.07%`
  - 当前看，这条实现不是继续压 `TTFT/TPS` 的有效方向，不应作为默认优化保留

### 3.9 Then Move LinearAttn Prefill Q/K Norm Outside Kernel

- 配置：
  - 基于 `Selective Int8`
  - 保持 `decode` 路径不变
  - 仅对 `linear_attn` 的 **prefill** 路径做如下修改：
    - 先在 Python 侧对 `query/key` 做 `L2 norm`
    - 调用 `fused_linear_attn_prefill_sglang(..., use_qk_l2norm_in_kernel=False)`
- 设计动机：
  - 专门的 microbench 显示：
    - `prefill` 将 `q/k norm` 放到 kernel 外部后，单算子从约 `2.97 ms` 降到约 `2.53 ms`
    - 约快 `17.4%`
  - `decode` 几乎无差别，因此只改 `prefill`
- 服务显存：
  - 运行中占用：`35759 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/output_texts/linear_attn_prefill_norm_outside_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 923.35 ms | 1.6112 s | 63 |
| run2 | 180.05 ms | 0.8695 s | 63 |
| run3 | 179.11 ms | 0.8681 s | 63 |
| run4 | 179.13 ms | 0.8687 s | 63 |
| run5 | 178.91 ms | 0.8680 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 179.30 ms |
| run2-run5 avg E2E | 0.8686 s |
| run2-run5 output_tps | 72.53 tok/s |
| avg_ms_per_output_token | 13.79 ms |
| avg_output_tokens | 63.00 |

- 相对当前最佳 `Selective Int8`：
  - `TTFT` 更快约 `4.5%`
  - `E2E` 更快约 `0.8%`
  - `output_tps` 更高约 `0.8%`

- 结论：
  - `linear_attn` 的 `prefill` kernel 仍有可挖的实现空间
  - 当前最有效的一个点是：将 `q/k` 的 `L2 norm` 从 kernel 内部挪到外部
  - 这是当前 `W8A8` 路线在 `Selective Int8` 之上进一步压低 `TTFT` 的有效优化

### 3.10 Then Tune LinearAttn Prefill Launch Parameters

- 配置：
  - 基于 `W8A8 Selective Int8 + LinearAttn Prefill Q/K Norm Outside Kernel`
  - 保持 `decode` 路径不变
  - 仅调 `fused_linear_attn_prefill_sglang` 的 launch 参数
- 调优内容：
  - 通过 dedicated microbench 扫描 `BV / num_warps / num_stages`
  - 在当前 `K=128, V=128` 的 Qwen3.6 shape 下，较优点集中在：
    - `BV=16`
    - `num_warps=4`
    - `num_stages=3`
  - `decode` 参数差异很小，因此未改 `decode`
- 设计动机：
  - 之前的 `prefill` kernel launch 配置偏保守
  - microbench 显示仅调整 launch 参数即可继续压低 `prefill` kernel 时间
- 服务显存：
  - 运行中占用：`35759 MiB / 81920 MiB`
- `run1` 输出文件：
  - `my_docs/results/output_texts/linear_attn_prefill_launch_tuned_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 413.01 ms | 1.0632 s | 63 |
| run2 | 172.14 ms | 0.8226 s | 63 |
| run3 | 162.66 ms | 0.8132 s | 63 |
| run4 | 162.72 ms | 0.8132 s | 63 |
| run5 | 163.79 ms | 0.8142 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 165.33 ms |
| run2-run5 avg E2E | 0.8158 s |
| run2-run5 output_tps | 77.22 tok/s |
| avg_ms_per_output_token | 12.95 ms |
| avg_output_tokens | 63.00 |

- 相对 `3.9` 当前最佳：
  - `TTFT` 更快约 `7.8%`
  - `E2E` 更快约 `6.1%`
  - `output_tps` 更高约 `6.5%`

- 相对当前 `bf16` 最优：
  - `TTFT` 更快约 `15.2%`
  - `E2E` 更快约 `12.7%`
  - `output_tps` 更高约 `14.5%`

- 结论：
  - 当前 `W8A8` 路线下，`linear_attn prefill kernel` 仍然是最值得优化的大头之一
  - 除了将 `q/k norm` 挪出 kernel，本身的 launch 参数也仍有可观优化空间
  - 这次只通过 `BV / num_warps / num_stages` 调优，就进一步把当前最佳结果从：
    - `179.30 ms / 0.8686 s / 72.53 tok/s`
    提升到：
    - `165.33 ms / 0.8158 s / 77.22 tok/s`

### 3.11 Then Switch LinearAttn Prefill to Vendored Full-Kernel Chunk Backend

- 配置：
  - 基于 `W8A8 Selective Int8 + LinearAttn Prefill Launch Tuning + Q/K Norm Outside Kernel`
  - 保持：
    - `Fused MoE`
    - `SGLang Linear Attention`
    - `CUDA Graph`
  - 只替换 `linear_attn` 的 `prefill` backend：
    - 从本地 tuned 串行 prefill kernel
    - 切到 vendored `sglang`-style full-kernel chunk backend
- 集成方式：
  - 将 `sglang` 的 chunk prefill 相关实现 vendoring 到：
    - [python/minisgl/fla_vendor](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/fla_vendor)
  - 在 [python/minisgl/linear_attention.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/linear_attention.py) 中，`fused_linear_attn_prefill_sglang()` 改为直接调用：
    - `chunk_gated_delta_rule(...)`
  - `initial_state` 以 in-place 更新方式承接 prefill 结束后的最终 state，并写回当前 `state`
  - 为避免上游 helper 在 import 时触发 CUDA 初始化，vendored helper 做了最小本地化裁剪
- CUDA Graph：
  - 纯文本 benchmark 下保留 `CUDA Graph`
  - 启动日志确认：
    - `Start capturing CUDA graphs with sizes: [1]`
    - `Capturing graphs: bs = 1 ...`
    - `Scheduler is ready`
- 设计动机：
  - 之前的“外层 chunk 包装”或“半 Triton 半 Torch”接法都无法转化成端到端收益
  - 这次改为尽量接近上游 `sglang` 的完整 prefill chunk backend，避免继续做局部半迁移
- 服务显存：
  - 运行中占用：`35759 MiB / 81920 MiB`
- `run1` 输出文件：
  - [chunk_vendor_fullkernel_run1_output.txt](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/output_texts/chunk_vendor_fullkernel_run1_output.txt)
- 正确性检查：
  - `run1` 输出为正常中文说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 11870.44 ms | 12.5167 s | 63 |
| run2 | 117.28 ms | 0.7653 s | 63 |
| run3 | 111.92 ms | 0.7595 s | 63 |
| run4 | 111.89 ms | 0.7591 s | 63 |
| run5 | 112.08 ms | 0.7600 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 113.29 ms |
| run2-run5 avg E2E | 0.7610 s |
| run2-run5 output_tps | 82.79 tok/s |
| avg_ms_per_output_token | 12.08 ms |
| avg_output_tokens | 63.00 |

- 相对 `3.10` 当前最佳：
  - `TTFT` 更快约 `31.5%`
  - `E2E` 更快约 `6.7%`
  - `output_tps` 更高约 `7.2%`

- 相对当前 `bf16` 最优：
  - `TTFT` 更快约 `41.9%`
  - `E2E` 更快约 `18.5%`
  - `output_tps` 更高约 `22.8%`

- 说明：
  - `run1` 极慢主要来自首次 Triton chunk kernel 编译，不代表稳态性能
  - 这次结果说明，真正完整 kernel 化并保留 `CUDA Graph` 后，chunk-scan 路线可以把此前离线 microbench 的潜力转化成端到端收益
  - 之前多次失败并不是 chunk-scan 思路错误，而是：
    - 只迁了一部分 kernel
    - 仍残留 Python/Torch fallback
    - 或错误地关闭了 `CUDA Graph`

#### 3.6.3 EP 小结

- 在当前单请求、单并发口径下，`EP` 的效果是：
  - `TTFT` 有小幅改善
  - 但 `E2E` 和 `output_tps` 都没有提升，反而略差
- 这说明当前收益主要来自：
  - expert 侧局部减负
- 但整体又被这些成本抵消：
  - attention 仍然在每个 EP rank 上复制执行
  - expert 通信与聚合成本增加
  - 单并发下并行收益不容易完全兑现

### 3.12 Then Enable Gemma Fused Norm

- 目标：
  - 在不改动 attention / MoE 主路径的前提下，只替换 Qwen3.6 中使用最重的 Gemma 风格 norm
  - 观察它在 `CUDA Graph + 单并发 decode` 下是否能直接带来稳态吞吐收益
- 配置：
  - 基于当前已修复正确性的 `bf16 + fused MoE + sglang linear attention + attention-backend fi + CUDA Graph`
  - 仅开启：
    - `MINISGL_GEMMA_FUSED_NORM=1`
- 实现说明：
  - mini-sglang 原先直接调 `torch.ops.sgl_kernel.gemma_rmsnorm.default(...)`
  - 但在当前 `sglang-cu13` 环境里，这些底层 op 没直接暴露在 `torch.ops.sgl_kernel`
  - 这次改为走 `sgl_kernel.elementwise.gemma_rmsnorm / gemma_fused_add_rmsnorm` 的公开 wrapper
  - 这样会自动复用 `flashinfer` 或 `sgl_kernel` 内部 fallback，而不是依赖未注册的裸 op 名称
- 正确性检查：
  - 短请求 `介绍一下自己` 输出正常
  - 真实 `1024 final-chat / 64 out` 的 `run1` 输出为连贯中文续写，无模板残留或异常跳题
- `run1` 输出文件：
  - [gemma_fused_norm_run1_output.txt](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/output_texts/gemma_fused_norm_run1_output.txt)
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 150.24 ms | 0.8088 s | 63 |
| run2 | 125.06 ms | 0.7847 s | 63 |
| run3 | 126.24 ms | 0.7859 s | 63 |
| run4 | 125.61 ms | 0.7852 s | 63 |
| run5 | 128.81 ms | 0.7885 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 126.43 ms |
| run2-run5 avg E2E | 0.7861 s |
| run2-run5 output_tps | 80.15 tok/s |
| avg_ms_per_output_token | 12.48 ms |
| avg_output_tokens | 63.00 |

- 相对当前 `fi + graph on` baseline：
  - baseline：`TTFT 130.59 ms`，`E2E 0.8763 s`，`output_tps 71.89 tok/s`
  - `Gemma Fused Norm`：
    - `TTFT` 改善约 `3.2%`
    - `E2E` 改善约 `10.3%`
    - `output_tps` 提升约 `11.5%`

- 结论：
  - 这是当前修复 correctness 之后，第一个在 `graph on` 真正转化成端到端收益的单因素优化
  - 也进一步支持之前的 profiling 判断：
    - 当前单并发 decode 的主要差距，不在 MoE 单核融合
    - 更像是在 norm / 周边 elementwise 路径上

### 3.13 Then Skip Decode-Side A/B FP32 Cast

- 目标：
  - 只去掉 decode 路径里 `a.float().contiguous()` / `b.float().contiguous()`
  - 保留其余实现完全不变，观察这些 dtype cast 是否真是当前 graph-on 的主要瓶颈
- 配置：
  - 基于当前已修复正确性的 `bf16 + fused MoE + sglang linear attention + attention-backend fi + CUDA Graph`
  - 仅开启：
    - `MINISGL_SKIP_AB_FP32_CAST=1`
- 静态判断：
  - `a/b` 进入的两个 Triton kernel：
    - `fused_gdn_gating_sglang`
    - `fused_linear_attn_decode_sglang`
  - 内部都会立刻 `tl.load(...).to(tl.float32)`
  - 所以从理论上看，外层显式 `.float()` 并非必须
- 正确性检查：
  - 短请求 `介绍一下自己` 输出正常
  - 真实 `1024 final-chat / 64 out` 的 `run1` 输出也正常
- `run1` 输出文件：
  - [skip_ab_fp32_cast_run1_output.txt](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/output_texts/skip_ab_fp32_cast_run1_output.txt)
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 249.43 ms | 0.9894 s | 63 |
| run2 | 130.70 ms | 0.8720 s | 63 |
| run3 | 130.68 ms | 0.8713 s | 63 |
| run4 | 130.34 ms | 0.8709 s | 63 |
| run5 | 130.57 ms | 0.8708 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 130.57 ms |
| run2-run5 avg E2E | 0.8713 s |
| run2-run5 output_tps | 72.31 tok/s |
| avg_ms_per_output_token | 13.83 ms |
| avg_output_tokens | 63.00 |

- 相对当前 `fi + graph on` baseline：
  - baseline：`TTFT 130.59 ms`，`E2E 0.8763 s`，`output_tps 71.89 tok/s`
  - `Skip A/B FP32 Cast`：
    - `TTFT` 基本无变化
    - `E2E` 改善约 `0.6%`
    - `output_tps` 提升约 `0.6%`

- 结论：
  - 这条优化是安全的，但收益极小
  - 说明 decode 路径里这两个外层 cast 不再是当前端到端主要瓶颈
  - 之前 graph-off profiling 看到的 cast 开销，在 `graph on + 单并发` 真实场景里并不会显著转化成吞吐差距

### 3.14 Then Enable Decode-Side Depthwise Conv Triton Fast Path

- 目标：
  - 只替换 linear attention decode 中的 depthwise conv 实现
  - 验证 `F.conv1d + silu` 在 `graph on + 单并发 decode` 下是不是主要瓶颈
- 配置：
  - 基于当前已修复正确性的 `bf16 + fused MoE + sglang linear attention + attention-backend fi + CUDA Graph`
  - 仅开启：
    - `MINISGL_DEPTHWISE_CONV_DECODE=1`
- 实现说明：
  - 用 `python/minisgl/depthwise_conv_triton.py` 中的单 token Triton kernel
  - 替换 decode 路径上的 PyTorch `F.conv1d + silu`
- 正确性检查：
  - 短请求 `介绍一下自己` 输出正常
  - 真实 `1024 final-chat / 64 out` 的 `run1` 输出也正常
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 154.88 ms | 0.9007 s | 63 |
| run2 | 129.90 ms | 0.8755 s | 63 |
| run3 | 130.23 ms | 0.8754 s | 63 |
| run4 | 127.70 ms | 0.8735 s | 63 |
| run5 | 130.03 ms | 0.8762 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 129.46 ms |
| run2-run5 avg E2E | 0.8751 s |
| run2-run5 output_tps | 71.99 tok/s |
| avg_ms_per_output_token | 13.89 ms |
| avg_output_tokens | 63.00 |

- 相对当前 `fi + graph on` baseline：
  - baseline：`TTFT 130.59 ms`，`E2E 0.8763 s`，`output_tps 71.89 tok/s`
  - `Depthwise Conv Decode`：
    - `TTFT` 改善约 `0.9%`
    - `E2E` 改善约 `0.1%`
    - `output_tps` 提升约 `0.1%`

- 结论：
  - 这条优化同样是安全的，但几乎没有端到端收益
  - 说明 decode 中这部分 conv kernel 在当前 workload 下不是 mini-sglang 相对 sglang 的主差距来源

### 3.15 Then Extend Gemma Fused Norm to Q/K `forward_inplace()`

- 目标：
  - 把 full attention 中 `q_norm.forward_inplace()` / `k_norm.forward_inplace()` 也接到 fused Gemma norm
  - 观察 q/k norm 这块是否还能继续缩小与 sglang 的差距
- 配置：
  - 基于 `MINISGL_GEMMA_FUSED_NORM=1`
  - 在 `GemmaRMSNorm.forward_inplace()` 中改为 fused 结果 `copy_` 回原张量
- 为什么这样改：
  - sglang 的 full-attention q/k norm 已经走 fused 路径
  - mini-sglang 之前虽然主 norm 吃到了 fused，但 `forward_inplace()` 仍然停留在 torch 路径
  - 直接 `out=x` 的原地写回版本会引入重复句子等正确性问题，因此改成“先 fused 计算，再 `copy_` 回原张量”的保守版本
- 正确性检查：
  - 长 prompt 输出恢复正常，无重复句子
  - 短请求输出正常
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run2-run5 avg | 125.24 ms | 0.7613 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 output_tps | 82.75 tok/s |
| avg_ms_per_output_token | 12.08 ms |

- 相对 `Gemma Fused Norm` 初版（80.15 tok/s）：
  - `output_tps` 再提升约 `3.2%`
- 结论：
  - 这说明 q/k inplace norm 仍然有收益
  - 但收益已经明显小于首轮 `Gemma Fused Norm`，说明它不是剩余大差距的主来源

### 3.16 Then Port SGLang Full-Attention Fused Prepare

- 目标：
  - 直接对齐 sglang 的 full-attention 前处理路径
  - 将 `Q/K GemmaRMSNorm + RoPE + gate 提取` 合成 1 条 fused 路径
- 配置：
  - `MINISGL_GEMMA_FUSED_NORM=1`
  - `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- 实现说明：
  - 新增 `python/minisgl/layers/fused_qk_rmsnorm_rope_gate.py`
  - 在 `Qwen3_5FullAttention.forward()` 中，当满足 CUDA + `attn_output_gate=True` 时，直接走 fused prepare
  - 原路径保留，默认关闭，可做消融
- 为什么有效：
  - sglang 当前 full attention 已经有这条 fused prepare
  - mini-sglang 原先仍然是：
    - `split`
    - `q_norm.forward_inplace`
    - `k_norm.forward_inplace`
    - `rotary.forward`
    - 后续再处理 gate
  - 这条融合减少了多次 q/k norm、rope 相关的中间张量和 kernel
- 正确性检查：
  - 短请求 `介绍一下自己` 输出正常
  - `1024 final-chat / 64 out` 的 `run1` 输出正常
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 154.51 ms | 0.7590 s | 63 |
| run2 | 118.92 ms | 0.7249 s | 63 |
| run3 | 122.92 ms | 0.7284 s | 63 |
| run4 | 122.52 ms | 0.7287 s | 63 |
| run5 | 123.29 ms | 0.7287 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 121.91 ms |
| run2-run5 avg E2E | 0.7277 s |
| run2-run5 output_tps | 86.58 tok/s |
| avg_ms_per_output_token | 11.55 ms |
| avg_output_tokens | 63.00 |

- 相对 `82.75 tok/s`：
  - `output_tps` 再提升约 `4.6%`
- 结论：
  - 这是当前追赶 sglang 主线里第二个明确有效的结构性优化
  - 也进一步证明：和 sglang 直接对齐 full-attention 编排，比继续抠零散小 kernel 更有效

### 3.17 Then Port SGLang Fused Gate Mul

- 目标：
  - 对齐 sglang 的 `fused_sigmoid_mul(attn_output, gate)`
  - 验证 full-attention 输出侧的 `attn_output * sigmoid(gate)` 是否还是一个值得优化的点
- 配置：
  - `MINISGL_GEMMA_FUSED_NORM=1`
  - `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
  - `MINISGL_FULL_ATTN_FUSED_GATE_MUL=1`
- 实现说明：
  - 新增 `python/minisgl/layers/elementwise.py`
  - 用 Triton 实现和 sglang 等价的 fused sigmoid-mul
- 正确性检查：
  - 短请求输出正常
  - 长 prompt 输出正常
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run2-run5 avg | 122.81 ms | 0.7279 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 output_tps | 86.55 tok/s |
| avg_ms_per_output_token | 11.55 ms |

- 相对 `FullAttn Fused Prepare`：
  - 几乎无变化
- 结论：
  - 这条路径不是当前主缺口
  - `sigmoid(gate)` 本身的输出侧 elementwise 不是 mini-sglang 相比 sglang 的大差距来源

### 3.18 Then Replace Decode Conv with SGLang `causal_conv1d_update`

- 目标：
  - 沿着和 sglang 直接对齐的方向，把 linear-attn decode 的 conv 从 mini 原有 `torch.cat + F.conv1d` 改为 `sglang` 当前主线使用的 `causal_conv1d_update`
- 配置：
  - `MINISGL_GEMMA_FUSED_NORM=1`
  - `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
  - `MINISGL_DEPTHWISE_CONV_DECODE=1`
- 实现说明：
  - 不再使用之前自写的 `depthwise_conv_triton.py` 作为主线
  - 直接在 `_run_depthwise_conv()` 中优先调用：
    - `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_update`
  - 保留原 torch 路径作为回退
- 为什么有效：
  - 新一轮 graph-off trace 显示，当前 mini-sglang 仍有大量
    - `conv_depthwise2d_forward_kernel_generic`
  - 而 sglang 对应的是：
    - `_causal_conv1d_update_kernel`
  - 这说明此前“decode conv 不是主差距”的判断只适用于自写 Triton 版本，不适用于“直接对齐 sglang 现有实现”
- 正确性检查：
  - 短请求 `介绍一下自己` 输出正常
  - `1024 final-chat / 64 out` 的 `run1` 输出正常
- 结果：

| 项目 | TTFT | E2E | output_tokens |
| --- | ---: | ---: | ---: |
| run1 | 151.17 ms | 0.7410 s | 63 |
| run2 | 122.50 ms | 0.7123 s | 63 |
| run3 | 122.79 ms | 0.7125 s | 63 |
| run4 | 124.71 ms | 0.7144 s | 63 |
| run5 | 123.21 ms | 0.7132 s | 63 |

| 稳态统计 | 数值 |
| --- | ---: |
| run2-run5 avg TTFT | 123.30 ms |
| run2-run5 avg E2E | 0.7131 s |
| run2-run5 output_tps | 88.35 tok/s |
| avg_ms_per_output_token | 11.32 ms |
| avg_output_tokens | 63.00 |

- 相对 `86.58 tok/s`：
  - `output_tps` 再提升约 `2.0%`
- 相对修复后原 baseline `71.89 tok/s`：
  - `output_tps` 累计提升约 `22.9%`
- 结论：
  - 这说明 decode conv 仍然有空间，但前提是要直接对齐 sglang 的实现，而不是另起一套自写替代品
  - 当前最好稳定结果已经推进到 `88.35 tok/s`

### FULL_ATTN_SIGMOID_GATE (2026-06-28)

**改动**：在 full-attention 路径中，将 `sigmoid(gate) * attn_output` 融合为一个 Triton kernel（`fused_sigmoid_mul_flat`，对 2D gate 做 sigmoid+multiply 一 pass 完成）。

**配置**: `MINISGL_GEMMA_FUSED_NORM=1 MINISGL_FULL_ATTN_FUSED_PREPARE=1 MINISGL_DEPTHWISE_CONV_DECODE=1 MINISGL_LINEAR_RMSNORM_GATED=1 MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1 MINISGL_FULL_ATTN_SIGMOID_GATE=1`

| Metric | Value |
|--------|-------|
| TTFT (稳定) | 121.44 ms |
| E2E (稳定) | 0.7097 s |
| output_tps | 88.77 tok/s |

**结论**：性能退化（best 96.02 → 88.77 tok/s，-7.5%）。2D gate 上 sigmoid 开销极小，Triton kernel launch overhead 比直接 `F.sigmoid * x` 更大，此方向放弃。

### 2026-06-28 重新校准 roadmap 起点

这一轮对 `roadmap` 起点做了重测，原因是前一轮实验记录和代码状态不一致。

- `baseline`（不开优化）：
  - `TTFT = 130.90 ms`
  - `E2E = 0.8770 s`
  - `output_tps = 71.84 tok/s`
- 组合：
  - `MINISGL_GEMMA_FUSED_NORM=1`
  - `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
  - `MINISGL_DEPTHWISE_CONV_DECODE=1`
  - `MINISGL_LINEAR_RMSNORM_GATED=1`
  - `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`

第一次重测发现：

- 该组合只有：
  - `TTFT = 123.03 ms`
  - `E2E = 0.7126 s`
  - `output_tps = 88.41 tok/s`

进一步排查发现：

- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD` 当时在 `env.py` 中有开关定义，但在 `qwen3_5_moe.py` 中根本没有被使用
- 也就是说，这个开关此前实际上是空开关

修复后，再次单因素重测同一组合：

- `TTFT = 122.70 ms`
- `E2E = 0.6943 s`
- `output_tps = 90.73 tok/s`

结论：

- `SHARED_EXPERT_FUSED_GATE_ADD` 真实有效，但真实收益不是此前记录中的量级
- 当前工作树重新校准后的复现值为：
  - `output_tps = 90.73 tok/s`
- 历史 session 中曾跑到：
  - `TTFT = 122.77 ms`
  - `E2E = 0.6561 s`
  - `output_tps = 96.02 tok/s`
- 因此更准确的结论是：
  - `96.02 tok/s` 是历史最好成绩
  - `90.73 tok/s` 是当前代码状态下的复现值
  - 当前真正需要做的是找出“为什么当前工作树比当时慢了约 5.5 tok/s”
- 相对 baseline：
  - `71.84 -> 90.73 tok/s`
  - 累计提升约 `26.3%`

备注：

- 这说明 `roadmap` 的“继续沿整层边界融合推进”方向仍然成立
- 但后续所有结论都必须以“开关已真实接线 + 单因素重测”为准，不能再直接沿用前一轮口头 best 值

### 2026-06-28 恢复历史 `LINEAR_RMSNORM_GATED -> SHARED_EXPERT_FUSED_GATE_ADD` 链条

在当前工作树中重新排查后发现：

- `LINEAR_RMSNORM_GATED` 的 Triton kernel 和环境开关都还在
- 但它已经没有真正接到 Qwen3.6 linear-attention 的输出 norm 路径上
- 因此此前重跑只得到：
  - `88.37 tok/s`（`GEMMA_FUSED_NORM + FULL_ATTN_FUSED_PREPARE + DEPTHWISE_CONV_DECODE`）
- 在把 `LINEAR_RMSNORM_GATED` 接回 `Qwen3_5RMSNormGated.forward()` 之后，重新按历史顺序重测：

1. `MINISGL_GEMMA_FUSED_NORM=1 MINISGL_FULL_ATTN_FUSED_PREPARE=1 MINISGL_DEPTHWISE_CONV_DECODE=1`
   - `TTFT = 123.63 ms`
   - `E2E = 0.7129 s`
   - `output_tps = 88.37 tok/s`

2. `+ MINISGL_LINEAR_RMSNORM_GATED=1`
   - `TTFT = 121.83 ms`
   - `E2E = 0.6749 s`
   - `output_tps = 93.34 tok/s`

3. `+ MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
   - `TTFT = 120.64 ms`
   - `E2E = 0.6554 s`
   - `output_tps = 96.13 tok/s`

结论：

- 历史链条已经在当前工作树上重新复现
- 对应关系就是：
  - `LINEAR_RMSNORM_GATED`：`88.37 -> 93.34 tok/s`
  - `SHARED_EXPERT_FUSED_GATE_ADD`：`93.34 -> 96.13 tok/s`
- 因此此前的历史记录
  - `93.32 tok/s`
  - `96.02 tok/s`
  是正确的，只是中间 `LINEAR_RMSNORM_GATED` 这条接线后来丢了

### FUSED_QKV_SPLIT (2026-06-28)

目标：

- 按 roadmap 的 “prepare / split / copy 融合” 方向，验证 linear-attn prefill 路径中的
  `mixed_qkvz.split(...) + mixed_ba.split(...)`
  是否仍然是值得优化的点

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_FUSED_QKV_SPLIT=1`

结果：

- `TTFT = 120.53 ms`
- `E2E = 0.6597 s`
- `output_tps = 95.50 tok/s`

相对当前最好值：

- `96.13 -> 95.50 tok/s`
- 退化约 `0.7%`

结论：

- 当前这版 `FUSED_QKV_SPLIT` 在 graph-on / 单并发场景下没有收益
- 说明 linear-attn prefill 的 `split` 本身不是当前端到端瓶颈，或者这版 fused split 的收益被额外 layout / launch 成本抵消
- 此方向暂时降级，不作为下一优先级继续投入

### MOE_FUSED_ACTIVATION (2026-06-28)

目标：

- 沿着 roadmap 的 “MoE 整条执行链” 方向，验证把 routed expert 路径中的
  `silu_and_mul`
  替换为 `sgl_kernel` fused activation 是否仍有稳定收益

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`

结果：

- `TTFT = 117.15 ms`
- `E2E = 0.6392 s`
- `output_tps = 98.56 tok/s`

相对当前最好值：

- `96.13 -> 98.56 tok/s`
- 提升约 `2.5%`

结论：

- `MOE_FUSED_ACTIVATION` 在 graph-on / 单并发场景下是有效优化
- 说明当前剩余差距里，MoE 周边的 activation 阶段仍有真实开销
- 当前最好稳定值更新为：
  - `output_tps = 98.56 tok/s`

### MOE_SGL_REDUCE (2026-06-28)

目标：

- 验证把 MoE combine/reduce 从本地 Triton reduce 切换到 `sgl_kernel.moe_sum_reduce`
  是否能继续提升 graph-on / 单并发场景的端到端吞吐

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_MOE_SGL_REDUCE=1`

正确性检查：

- 短请求 `介绍一下自己` 输出异常
- 返回内容出现大量重复的特殊 token：
  - `<|im_start|>`
  - `</think>`
- 说明当前 `MOE_SGL_REDUCE` 接法仍然存在语义错误

结论：

- `MOE_SGL_REDUCE` 当前 correctness fail
- 在修复 reduce 语义之前，不进入性能主线比较
- 此方向保留为后续专门修 correctness 的支线，不作为当前最高优先级继续投入

### SHARED_EXPERT_FUSED_ACTIVATION (2026-06-28)

目标：

- 沿着 “shared expert 之外的剩余 gate/add/mul epilogue” 这条线继续推进
- 验证 shared expert 内部 bf16 主路径中的
  `gate_up_proj -> silu_and_mul -> down_proj`
  是否还能通过 fused activation 再压掉一部分开销

实现：

- 新增开关：
  - `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- 在 shared expert 的 bf16 主路径中，把
  - `silu_and_mul(gate_up)`
  替换为
  - `fused_silu_and_mul(gate_up, inter)`

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 119.08 ms`
- `E2E = 0.6351 s`
- `output_tps = 99.19 tok/s`

相对当前最好值：

- `98.56 -> 99.19 tok/s`
- 提升约 `0.6%`

结论：

- `SHARED_EXPERT_FUSED_ACTIVATION` 是有效但收益较小的优化
- 说明 shared expert 内部仍有一部分 activation 开销可清理，但它不是当前最大的剩余瓶颈
- 当前最好稳定值更新为：
  - `output_tps = 99.19 tok/s`

### LINEAR_PREFILL_QK_L2NORM (2026-06-28)

目标：

- 回到 linear-attention prefill 路径，减少当前仍然显式存在的：
  - `F.normalize(query.float(), ...)`
  - `F.normalize(key.float(), ...)`
  - 以及对应的 `to(dtype) / contiguous`

实现：

- 新增开关：
  - `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- 当开关打开时：
  - 不再在 Python 路径里先做 `F.normalize`
  - 直接把 `use_qk_l2norm_in_kernel=True` 传给 `fused_linear_attn_prefill_sglang`
  - 让 Q/K 的 L2Norm 下沉到 kernel 内部完成

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 113.02 ms`
- `E2E = 0.6290 s`
- `output_tps = 100.17 tok/s`

相对当前最好值：

- `99.19 -> 100.17 tok/s`
- 提升约 `1.0%`

结论：

- 这条优化是有效的
- 而且它比 shared expert 局部 activation 更像“主线剩余瓶颈”，因为它直接打到了 roadmap 中明确指出的 prefill normalize / cast / contiguous 链
- 当前最好稳定值更新为：
  - `output_tps = 100.17 tok/s`

### LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS (2026-06-28)

目标：

- 继续压缩 linear-attention prefill 路径中显式的：
  - `value.float().contiguous()`
  - `gate.contiguous()`
  - `beta.contiguous()`
- 验证这些包装是否已经被下游 `chunk_gated_delta_rule` 隐式处理，从而可以安全跳过

实现：

- 新增开关：
  - `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- 当开关打开时：
  - 直接传入 `value.float()`、`gate`、`beta`
  - 不再额外做显式 `contiguous()`

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 110.04 ms`
- `E2E = 0.6263 s`
- `output_tps = 100.59 tok/s`

相对当前最好值：

- `100.17 -> 100.59 tok/s`
- 提升约 `0.4%`

结论：

- 这条优化是有效的，但收益较小
- 说明 prefill 侧这部分显式包装还有少量端到端开销
- 当前最好稳定值更新为：
  - `output_tps = 100.59 tok/s`

### DEPTHWISE_CONV_PREFILL (2026-06-28)

目标：

- 针对阶段化 profile 中暴露出的 linear-attention prefill `conv` 开销
- 尝试把当前的：
  - `torch.cat + F.conv1d + activation`
  替换为 sglang 现成的 `causal_conv1d_fn`

实现：

- 新增开关：
  - `MINISGL_DEPTHWISE_CONV_PREFILL=1`
- 当开关打开时：
  - 仅在线性注意力 prefill 路径中调用 `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn`
  - decode 路径保持不变

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_DEPTHWISE_CONV_PREFILL=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 125.72 ms`
- `E2E = 0.6404 s`
- `output_tps = 98.37 tok/s`

相对当前最好值：

- `100.59 -> 98.37 tok/s`
- 退化约 `2.2%`

结论：

- 这条优化在当前实现方式下无效，且明显退化
- 说明直接复用 `causal_conv1d_fn` 并没有在 mini-sglang 当前调用形态下打中真实瓶颈
- 此方向暂时降级，不作为当前主线继续推进

### SKIP_AB_FP32_CAST (2026-06-28)

目标：

- 回到当前 benchmark 更直接相关的 decode 热路径
- 验证 linear-attention decode 里：
  - `a.float().contiguous()`
  - `b.float().contiguous()`
  是否仍有可省去的端到端开销

实现：

- 使用已有开关：
  - `MINISGL_SKIP_AB_FP32_CAST=1`
- 当开关打开时：
  - 直接传入 bf16 的 `a` / `b`
  - 不再在 Python 侧先做显式 fp32 cast

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 114.02 ms`
- `E2E = 0.6235 s`
- `output_tps = 101.04 tok/s`

相对当前最好值：

- `100.59 -> 101.04 tok/s`
- 提升约 `0.4%`

结论：

- 这条优化是有效的
- 虽然收益不大，但说明当前 `64 out` 场景下 decode 路径里的 `a/b -> fp32` 包装仍有少量端到端成本
- 当前最好稳定值更新为：
  - `output_tps = 101.04 tok/s`

### LINEAR_DECODE_VK_STATE (2026-06-29)

目标：

- 对齐 sglang packed decode kernel 使用的 state layout
- 验证仅将 linear-attention decode state 从 `[HV, K, V]` 扩展为辅助 `[HV, V, K]` 布局，是否能带来更好的 decode kernel 访存

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_VK_STATE=1`
- 当开关打开时：
  - 为每个 linear-attention layer/request slot 额外维护一份 `[HV, V, K]` 的辅助 state
  - prefill 结束后从主 state 同步到辅助 state
  - decode 路径改为直接读取辅助 state

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_VK_STATE=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 113.57 ms`
- `E2E = 0.6494 s`
- `output_tps = 97.01 tok/s`

相对当前最好值：

- `101.04 -> 97.01 tok/s`
- 退化约 `4.0%`

结论：

- 仅仅切换 decode state layout 而不改变整条 decode 执行方式，不能带来收益
- 这说明 mini-sglang 与 sglang 的差距并不是“state 排布”这一项单独决定的
- 此方向降级，不进入性能主线

### LINEAR_DECODE_SGLANG_PACKED (2026-06-29)

目标：

- 在 `LINEAR_DECODE_VK_STATE` 基础上，进一步直接复用 sglang 的 packed recurrent decode kernel
- 验证收益究竟来自 layout，还是来自 sglang 那个 packed decode kernel 本体

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_SGLANG_PACKED=1`
- 当开关打开时：
  - 仅在 decode 路径调用 sglang 的 `fused_recurrent_gated_delta_rule_packed_decode`
  - 要求同时打开 `MINISGL_LINEAR_DECODE_VK_STATE=1`

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_VK_STATE=1`
- `MINISGL_LINEAR_DECODE_SGLANG_PACKED=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 117.25 ms`
- `E2E = 0.6500 s`
- `output_tps = 96.92 tok/s`

相对当前最好值：

- `101.04 -> 96.92 tok/s`
- 退化约 `4.1%`

结论：

- 在当前 mini-sglang 的整体执行链里，直接替换成 sglang packed recurrent decode kernel 也没有收益
- 说明剩余差距更像是 decode 整体边界开销，而不是单个 recurrent kernel 本体
- 此方向降级，不进入性能主线

### LINEAR_DECODE_FUSED_INPUT_PROJ (2026-06-29)

目标：

- 对齐 sglang 在 decode 输入边界上的“更紧的组织方式”
- 验证将 decode 时分开的：
  - `in_proj_qkvz`
  - `in_proj_ba`
  合并为一次 bf16 GEMM 后，能否降低 `qkvz + ba` 这段开销

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`
- 当开关打开时：
  - 仅在 `decode + bf16` 路径下，将 `in_proj_qkvz.weight` 和 `in_proj_ba.weight` 预先拼成一份 fused weight
  - decode 时只做一次 `F.linear`
  - 然后再切回 `mixed_qkvz` 与 `mixed_ba`

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 116.35 ms`
- `E2E = 0.6166 s`
- `output_tps = 102.17 tok/s`

相对当前最好值：

- `101.04 -> 102.17 tok/s`
- 提升约 `1.1%`

结论：

- 这是 decode 主线上的有效收益
- 它说明当前剩余差距的一部分确实来自 decode 输入边界上的 kernel/launch 组织，而不是 linear-attn 数学路径本身
- 当前最好稳定值更新为：
  - `output_tps = 102.17 tok/s`

### LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ (2026-06-29)

目标：

- 更直接对齐 sglang 的 `_forward_input_proj` 组织方式
- 验证在 `decode + bf16` 下，让：
  - `in_proj_qkvz` 走主 CUDA stream
  - `in_proj_ba` 走辅助 CUDA stream
  是否能进一步重叠两次投影

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ=1`
- 当开关打开时：
  - 仅在 decode 路径里，为 linear-attention 层创建一个辅助 CUDA stream
  - 主流发射 `in_proj_qkvz`
  - 辅流发射 `in_proj_ba`
  - 结束后做 stream 同步

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 112.77 ms`
- `E2E = 0.6159 s`
- `output_tps = 102.29 tok/s`

相对当前最好值：

- 相对 `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1` 的 `102.17 tok/s`
- 小幅提升到 `102.29 tok/s`
- 增益约 `+0.12 tok/s`

结论：

- 这条更贴近 sglang 的输入边界组织方式是正收益的
- 但在 `bs=1` 场景下收益极小，说明 input projection 这段 overlap 空间已经接近吃干净
- 当前最好稳定值更新为：
  - `output_tps = 102.29 tok/s`

### LINEAR_RMSNORM_GATED_REUSE_OUT (2026-06-29)

目标：

- 继续沿 decode gated norm 主线排查剩余开销
- 验证 fused `RMSNorm + gate` 路径里每次 `torch.empty_like(...)` 的输出分配是否还有端到端成本

实现：

- 新增开关：
  - `MINISGL_LINEAR_RMSNORM_GATED_REUSE_OUT=1`
- 当开关打开时：
  - 只在 fused `RMSNorm+gate` 路径下缓存一份输出 buffer
  - 后续 decode 迭代复用该 buffer，避免重复分配

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_LINEAR_RMSNORM_GATED_REUSE_OUT=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`

正确性检查：

- 短请求 `介绍一下自己` 输出正常

结果：

- `TTFT = 116.88 ms`
- `E2E = 0.6146 s`
- `output_tps = 102.51 tok/s`

相对当前最好值：

- `102.17 -> 102.51 tok/s`
- 提升约 `0.33%`

结论：

- decode gated norm 这条线还有少量可挖空间
- 但“复用输出 buffer”本身不是大头，只能带来小幅收益
- 当前最好稳定值更新为：
  - `output_tps = 102.51 tok/s`

### Decode 阶段化 Profile 更新 (2026-06-29)

为了避免继续在 decode 输入边界上重复投入，对当前更优组合：

- `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`

做了 graph-off 的 decode 阶段化 profile。

稳定单层 decode 开销大致为：

- `qkvz ≈ 0.062 ms`
- `ba ≈ 0.003 ms`
- `conv ≈ 0.048 ms`
- `kernel ≈ 0.101~0.103 ms`
- `norm ≈ 0.069~0.070 ms`
- `out_proj ≈ 0.049~0.050 ms`

结论：

- `FUSED_INPUT_PROJ` 已经基本打平 `ba`
- 后续 decode 主线不应继续优先投入 input projection
- 当前最值得继续追的剩余块是：
  - `kernel`
  - `norm`
  - `qkvz / out_proj / conv`

### LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS (2026-06-29)

目标：

- 继续沿 decode `norm -> out_proj` 边界排查剩余胶水开销
- 验证 decode 时 `out_proj` 前那次显式 `.contiguous()` 是否冗余

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS=1`
- 当开关打开且处于 decode 时：
  - 跳过 `out_proj` 前的显式 `.contiguous()`
  - 直接将 `reshape` 后的 2D view 送入 `out_proj`

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_LINEAR_RMSNORM_GATED_REUSE_OUT=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`
- `MINISGL_LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS=1`

正确性检查：

- 短请求与 benchmark 输出正常

结果：

- `TTFT = 115.06 ms`
- `E2E = 0.6134 s`
- `output_tps = 102.71 tok/s`

相对当前最好值：

- `102.51 -> 102.71 tok/s`
- 提升约 `0.20 tok/s`

结论：

- decode `out_proj` 前的显式 `.contiguous()` 确实有少量端到端成本
- 这条线是有效的，但收益仍属于小幅边界优化
- 当前最好稳定值更新为：
  - `output_tps = 102.71 tok/s`

### LINEAR_DECODE_SKIP_AB_CONTIGUOUS (2026-06-29)

目标：

- 继续沿 decode kernel 边界减少中间张量复制
- 验证 decode 时 `a/b` 每层显式 `.contiguous()` 是否冗余

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_SKIP_AB_CONTIGUOUS=1`
- 与 `MINISGL_SKIP_AB_FP32_CAST=1` 组合使用时：
  - decode 路径不再强制对 `a/b` 做 `.contiguous()`
  - 直接把 split 得到的 strided view 送入后续 kernel

配置：

- `MINISGL_GEMMA_FUSED_NORM=1`
- `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
- `MINISGL_DEPTHWISE_CONV_DECODE=1`
- `MINISGL_LINEAR_RMSNORM_GATED=1`
- `MINISGL_LINEAR_RMSNORM_GATED_REUSE_OUT=1`
- `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
- `MINISGL_MOE_FUSED_ACTIVATION=1`
- `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
- `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
- `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
- `MINISGL_SKIP_AB_FP32_CAST=1`
- `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`
- `MINISGL_LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS=1`
- `MINISGL_LINEAR_DECODE_SKIP_AB_CONTIGUOUS=1`

正确性检查：

- 短请求与 benchmark 输出正常

结果：

- `TTFT = 112.87 ms`
- `E2E = 0.6101 s`
- `output_tps = 103.26 tok/s`

相对当前最好值：

- `102.71 -> 103.26 tok/s`
- 提升约 `0.55 tok/s`

结论：

- decode `a/b` 的显式 `.contiguous()` 在当前路径里确实是冗余成本
- 这说明剩余差距里仍然包含一部分边界复制/布局整理开销
- 当前最好稳定值更新为：
  - `output_tps = 103.26 tok/s`

### LINEAR_RMSNORM_GATED_SGLANG (2026-06-29)

目标：

- 对齐 sglang 当前在 Qwen3.5/GDN 路径里使用的 gated RMSNorm 实现
- 验证直接切到 `sglang.srt.layers.attention.fla.layernorm_gated.rms_norm_gated` 是否能进一步压低 decode `norm` 开销

实现：

- 新增临时实验开关：
  - `MINISGL_LINEAR_RMSNORM_GATED_SGLANG=1`
- 让 linear-attention 输出 norm 直接调用 sglang 的 `rms_norm_gated`

结果：

- 首次尝试暴露出 dtype 不匹配：
  - `rms_norm_gated` 返回 `float32`
  - 后续 `out_proj` 权重为 `bfloat16`
- 修正 dtype 后重新测得：
  - `TTFT = 115.60 ms`
  - `E2E = 0.6149 s`
  - `output_tps = 102.45 tok/s`

相对当前最好值：

- `103.26 -> 102.45 tok/s`
- 退化约 `0.81 tok/s`

结论：

- 当前 mini-sglang 自己这版 fused `RMSNorm+gate` 至少在该单并发场景下不比 sglang 差
- `norm` 不是下一步最值得优先继续对齐的方向
- 该实验不进入主线，代码已撤回

### LINEAR_DECODE_SGLANG_UPDATE (2026-06-29)

目标：

- 继续验证 decode 主差距是否来自 recurrent update kernel 本体
- 与之前 `LINEAR_DECODE_SGLANG_PACKED` 不同，这次不是 packed decode，而是直接切到 sglang 常规 decode 所用的
  `fused_sigmoid_gating_delta_rule_update`

实现：

- 新增临时实验开关：
  - `MINISGL_LINEAR_DECODE_SGLANG_UPDATE=1`
- 组合：
  - `MINISGL_LINEAR_DECODE_VK_STATE=1`
  - `MINISGL_LINEAR_DECODE_SGLANG_UPDATE=1`
- 让 decode 主路径改用 sglang 常规 recurrent update kernel

过程说明：

- 初次尝试时在 CUDA graph capture 阶段暴露出 `cu_seqlens` 动态创建问题
- 修复 graph-capture 兼容后重新完成 benchmark

结果：

- `TTFT = 114.38 ms`
- `E2E = 0.6382 s`
- `output_tps = 98.72 tok/s`

相对当前最好值：

- `103.26 -> 98.72 tok/s`
- 退化约 `4.54 tok/s`

结论：

- 即使切到 sglang 的常规 decode recurrent update kernel，本场景仍明显退化
- 这进一步说明当前 mini-sglang 与 sglang 的 decode 差距，并不主要来自 recurrent update kernel 本体
- 更可能仍在整层边界组织与其余热路径组合开销
- 该实验不进入主线，代码已撤回

### LINEAR_DECODE_SKIP_CONV_STATE_COPY (2026-06-29)

目标：

- 验证 decode fused depthwise conv 路径里，`conv_state.copy_(next_conv_state)` 是否是冗余开销
- 原因是 sglang `causal_conv1d_update` 本身就是原地更新 `conv_state`

实现：

- 新增临时实验开关：
  - `MINISGL_LINEAR_DECODE_SKIP_CONV_STATE_COPY=1`
- 当满足：
  - `MINISGL_DEPTHWISE_CONV_DECODE=1`
  - decode
  - 使用 sglang `causal_conv1d_update`
- 则直接把 `conv_state` 视为已原地更新，跳过额外 `copy_`

结果：

- `TTFT = 113.80 ms`
- `E2E = 0.6120 s`
- `output_tps = 102.93 tok/s`

相对当前最好值：

- `103.26 -> 102.93 tok/s`
- 退化约 `0.33 tok/s`

结论：

- decode conv state copy 这项不是主矛盾
- 即便理论上存在冗余，端到端收益并不成立，甚至有轻微退化
- 该实验不进入主线

### LINEAR_DECODE_FUSED_QKV_SPLIT (2026-06-29)

目标：

- 重新验证 `qkvz + split/reshape` 这段 decode prepare 是否仍有空间
- 与此前全局 `FUSED_QKV_SPLIT` 不同，这次只在 decode 上启用，避免 prefill 干扰

实现：

- 新增开关：
  - `MINISGL_LINEAR_DECODE_FUSED_QKV_SPLIT=1`
- 仅在：
  - `batch.is_decode`
  - `backend == "sglang"`
- 时使用 `fused_qkvzba_split_reshape_cat_contiguous`

结果：

- `TTFT = 114.64 ms`
- `E2E = 0.6195 s`
- `output_tps = 101.70 tok/s`

相对当前最好值：

- `103.26 -> 101.70 tok/s`
- 退化约 `1.56 tok/s`

结论：

- 当前这版 fused split/reshape kernel 即使只在 decode 启用，也没有转化成端到端收益
- 这说明 `qkvz + split/reshape` 这段虽然看起来像差异点，但不是“直接打开现有 fused kernel”就能补齐的
- 该实验不进入主线

### SHARED_EXPERT_DUAL_STREAM (2026-06-29)

目标：

- 对齐 sglang 中常见的 shared expert 双流重叠思路
- 让 decode 单 token 场景下 shared expert 分支与 routed experts 主分支并行

实现：

- 新增实验开关：
  - `MINISGL_SHARED_EXPERT_DUAL_STREAM=1`
- 在 `Qwen3_5SparseMoeBlock` 中为 shared expert 挂辅助 CUDA stream
- 仅在 decode 单 token、非 graph capture 场景启用：
  - 先复制 `hidden_states`
  - 在辅助 stream 上跑 shared expert
  - 主 stream 同时继续 routed experts

结果：

- 短输出正确性正常
- `TTFT = 115.16 ms`
- `E2E = 0.6161 s`
- `output_tps = 102.26 tok/s`

相对当前当时主线最好值：

- `103.26 -> 102.26 tok/s`
- 退化约 `1.00 tok/s`

结论：

- shared expert 双流重叠在当前单并发、graph on 场景下没有带来收益
- shared expert 虽然在 profile 中不小，但简单改成双流并不能自动转化成端到端提升
- 该实验不进入主线

### MOE_REUSE_WORKSPACE (2026-06-29)

目标：

- 减少 fused MoE 路径中 `topk` 与 `moe_align_block_size` 的临时张量分配
- 验证“每层每 token 的小分配”是否已经成为可见开销

实现：

- 新增实验开关：
  - `MINISGL_MOE_REUSE_WORKSPACE=1`
- 复用以下临时张量：
  - `topk_weights`
  - `topk_ids`
  - `sorted_ids`
  - `expert_ids`
  - `num_tokens_post_pad`
  - `cumsum_buffer`

结果：

- 短输出正确性正常
- `TTFT = 112.82 ms`
- `E2E = 0.6105 s`
- `output_tps = 103.20 tok/s`

相对当前当时主线最好值：

- `103.26 -> 103.20 tok/s`
- 轻微退化约 `0.06 tok/s`

结论：

- 这类 MoE 小张量分配并不是当前端到端主矛盾
- 即使减少分配，也没有拿到可观收益
- 该实验不进入主线

### MOE_SKIP_TOPK_POST_RENORM (2026-06-29)

目标：

- 排查 mini-sglang 的 MoE router 热路径里是否存在重复工作
- 核对后发现：
  - `sgl_kernel.topk_softmax(..., renormalize=True)` 之后
  - mini-sglang 仍在 Python 侧额外做一次 `sum` 归一化

实现：

- 新增实验开关：
  - `MINISGL_MOE_SKIP_TOPK_POST_RENORM=1`
- 在 fused MoE 路径中：
  - 保留 `topk_softmax(..., renormalize)`
  - 跳过后续额外的 Python 侧 `topk_weights = topk_weights / sum(...)`

结果：

- 短输出正确性正常
- `TTFT = 116.60 ms`
- `E2E = 0.6016 s`
- `output_tps = 104.72 tok/s`

相对当前主线最好值：

- `103.26 -> 104.72 tok/s`
- 提升约 `1.46 tok/s`

结论：

- 这条优化明确有效
- mini-sglang 的 MoE router 热路径里确实存在一段多余的后处理
- 该实验进入主线

### MOE_SKIP_TOPK_FP32_CAST (2026-06-29)

目标：

- 继续排查 MoE router 热路径里的额外胶水开销
- 核对后发现：
  - mini-sglang 调 `sgl_kernel.topk_softmax` 前会先做一次 `router_logits.float()`
  - sglang 主线则直接传原 dtype

实现：

- 新增实验开关：
  - `MINISGL_MOE_SKIP_TOPK_FP32_CAST=1`
- 与 `MINISGL_MOE_SKIP_TOPK_POST_RENORM=1` 叠加使用
- 让 `topk_softmax` 直接消费原始 `router_logits`

结果：

- 短输出正确性正常
- `TTFT = 111.81 ms`
- `E2E = 0.5909 s`
- `output_tps = 106.62 tok/s`

相对上一主线最好值：

- `104.72 -> 106.62 tok/s`
- 提升约 `1.90 tok/s`

相对更早的 decode 主线最好值：

- `103.26 -> 106.62 tok/s`
- 总提升约 `3.36 tok/s`

结论：

- 这条优化同样明确有效
- MoE router 路径的额外 dtype cast 确实会转化成可见端到端开销
- 该实验进入主线

### MOE_SKIP_DISPATCH_LOCAL_MASK (2026-06-29)

目标：

- 继续清理 MoE router / dispatch 热路径中的无用 Python 胶水
- 核对后发现：
  - `build_local_expert_dispatch_plan()` 在单卡 fast path 中会构造
    `local_mask=torch.ones_like(topk_ids, dtype=torch.bool)`
  - 但 fused MoE 主线实际上并不消费这个 `local_mask`

实现：

- 新增实验开关：
  - `MINISGL_MOE_SKIP_DISPATCH_LOCAL_MASK=1`
- 仅在单卡 fast path 中：
  - 跳过这块全 1 bool mask 的构造
  - 保持 `topk_weights/topk_ids` 原样返回

结果：

- 短输出正确性正常
- `TTFT = 111.74 ms`
- `E2E = 0.5878 s`
- `output_tps = 107.17 tok/s`

相对上一主线最好值：

- `106.62 -> 107.17 tok/s`
- 提升约 `0.55 tok/s`

相对更早的 decode 主线最好值：

- `103.26 -> 107.17 tok/s`
- 总提升约 `3.91 tok/s`

结论：

- 这条优化同样有效，但收益小于前两条 MoE router 主线优化
- 说明单卡 dispatch fast path 中仍然存在少量无用分配
- 当前最好稳定值更新为：
  - `TTFT = 111.74 ms`
  - `E2E = 0.5878 s`
  - `output_tps = 107.17 tok/s`
- 该实验进入主线

### MOE_ALIGN_SMALL_CAP (2026-06-29)

目标：

- 继续对齐 sglang 在 `moe_align_block_size()` 这段的实现细节
- 核对后发现：
  - mini-sglang 原本总是按
    `topk_ids.numel() + (num_experts + 1) * (block_size - 1)`
    分配 `sorted_ids / expert_ids` 上界
  - sglang 在 `topk_ids.numel() < num_experts + 1` 时会直接改用更小的
    `topk_ids.numel() * block_size`

实现：

- 新增实验开关：
  - `MINISGL_MOE_ALIGN_SMALL_CAP=1`
- 仅在小 token/topk decode 场景下：
  - 使用与 sglang 一致的更紧上界
  - 减少 `moe_align_block_size` 周边临时张量规模

结果：

- 短输出正确性正常
- `TTFT = 112.11 ms`
- `E2E = 0.5772 s`
- `output_tps = 109.16 tok/s`

相对上一主线最好值：

- `107.17 -> 109.16 tok/s`
- 提升约 `1.99 tok/s`

结论：

- 这条优化明确有效
- 说明 `moe_align_block_size` 这段 prepare 路径仍然存在可见的过量工作
- 当前最好稳定值更新为：
  - `TTFT = 112.11 ms`
  - `E2E = 0.5772 s`
  - `output_tps = 109.16 tok/s`
- 该实验进入主线

### MOE_SGLANG_CONFIG_LOOKUP (2026-06-29)

目标：

- 继续对齐 sglang 与 mini-sglang 在 routed-expert Triton kernel config 选择上的差异
- 核对后发现：
  - mini-sglang 当前只用一个极简 heuristic 选择 `BLOCK_SIZE_* / GROUP_SIZE_M`
  - sglang 主线则优先查 JSON 调优表，没命中时才回退默认值
  - 对于 Qwen3.6 routed experts 使用的 `E=256, N=512` 形状，sglang 的公开配置表里确实存在 tuned config

实现：

- 新增实验开关：
  - `MINISGL_MOE_SGLANG_CONFIG_LOOKUP=1`
- 在 mini-sglang 中引入一层 sglang 风格的 config lookup：
  - 优先按当前 Triton 版本查 JSON
  - 若当前版本未提供，则回退到较新的可用 Triton 版本目录
  - 若当前设备名无精确命中，则按候选设备族继续尝试
- 同时移除了先前会导致 CUDA graph capture 失败的 `MOE_ALIGN_TINY_PATH` 实验分支

这次实际对比到的 config 差异：

- mini-sglang 原 heuristic：
  - 当 `M <= E` 时固定使用：
    - `BLOCK_SIZE_M=16`
    - `BLOCK_SIZE_N=32`
    - `BLOCK_SIZE_K=64`
    - `GROUP_SIZE_M=1`
  - 当 `M > E` 时固定使用：
    - `BLOCK_SIZE_M=64`
    - `BLOCK_SIZE_N=64`
    - `BLOCK_SIZE_K=32`
    - `GROUP_SIZE_M=8`
- 修改后：
  - 先查 sglang 的 `E=256, N=512` JSON 配置表
  - 这次未命中本机 `NVIDIA A800 80GB PCIe` 专属项
  - 实际回退命中：
    - device fallback: `NVIDIA_H20`
    - triton version fallback: `3.5.1`
- 对当前单并发 decode 更关键的 `M=1`：
  - 原来：
    - `BLOCK_SIZE_M=16`
    - `BLOCK_SIZE_N=32`
    - `BLOCK_SIZE_K=64`
    - `GROUP_SIZE_M=1`
  - 修改后：
    - `BLOCK_SIZE_M=16`
    - `BLOCK_SIZE_N=64`
    - `BLOCK_SIZE_K=128`
    - `GROUP_SIZE_M=1`
    - `num_warps=4`
    - `num_stages=4`
- 对较大 `M`，例如 prefill 侧常见的 `M=1024`：
  - 原来：
    - `BLOCK_SIZE_M=64`
    - `BLOCK_SIZE_N=64`
    - `BLOCK_SIZE_K=32`
    - `GROUP_SIZE_M=8`
  - 修改后：
    - `BLOCK_SIZE_M=64`
    - `BLOCK_SIZE_N=64`
    - `BLOCK_SIZE_K=64`
    - `GROUP_SIZE_M=1`
    - `num_warps=4`
    - `num_stages=4`

结果：

- 短输出正确性正常
- 两次 benchmark 稳态结果一致：
  - `TTFT = 101.29~101.48 ms`
  - `E2E = 0.5668~0.5669 s`
  - `output_tps = 111.13~111.15 tok/s`
- 取稳定值：
  - `TTFT = 101.29 ms`
  - `E2E = 0.5668 s`
  - `output_tps = 111.15 tok/s`

相对上一主线最好值：

- `109.16 -> 111.15 tok/s`
- 提升约 `1.99 tok/s`

结论：

- 这条优化明确有效，而且是当前为止较大的单因素收益之一
- mini-sglang 与 sglang 在 routed-expert Triton config 选择上的实现差异，确实会转化成端到端性能差距
- 这次命中的是 sglang 配置表的版本/设备回退路径，而不是本机专属 `A800 PCIe` tuned config
- 从具体 config 看，收益主要来自：
  - 小 `M` 场景把 `BLOCK_SIZE_N/K` 做大
  - 大 `M` 场景把 `BLOCK_SIZE_K` 从 `32` 提到 `64`
  - 同时显式引入 `num_warps / num_stages`
- 当前最好稳定值更新为：
  - `TTFT = 101.29 ms`
  - `E2E = 0.5668 s`
  - `output_tps = 111.15 tok/s`
- 该实验进入主线

### MOE_SGLANG_DOWN_CONFIG (2026-06-29)

目标：

- 在已经验证 `MINISGL_MOE_SGLANG_CONFIG_LOOKUP=1` 有效之后，继续对齐 sglang routed-expert 的第二个 GEMM
- 核对后发现：
  - mini-sglang 当前第二段 `w2/down_proj` GEMM 仍然直接复用第一段 `gate_up` GEMM 的 config
  - sglang 主线会为第二段单独查 `_down.json`，再把 `BLOCK_SIZE_M` 对齐到第一段

实现：

- 新增实验开关：
  - `MINISGL_MOE_SGLANG_DOWN_CONFIG=1`
- 仅在已启用 `MINISGL_MOE_SGLANG_CONFIG_LOOKUP=1` 的基础上：
  - 为第二段 routed-expert GEMM 单独查 sglang 风格的 `_down.json`
  - 若命中，则只把 `BLOCK_SIZE_M` 强制改回与第一段一致
  - 其它参数如 `BLOCK_SIZE_N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages` 保持 down_config 自己的值

这次实际命中的关键 config 差异：

- 对 decode 最关键的 `M=1`
  - 第一段 up-config：
    - `BLOCK_SIZE_M=16`
    - `BLOCK_SIZE_N=64`
    - `BLOCK_SIZE_K=128`
    - `GROUP_SIZE_M=1`
    - `num_warps=4`
    - `num_stages=4`
  - 第二段原来：
    - 直接复用上面这组 up-config
  - 第二段修改后命中的 down-config：
    - `BLOCK_SIZE_M=16`
    - `BLOCK_SIZE_N=32`
    - `BLOCK_SIZE_K=256`
    - `GROUP_SIZE_M=1`
    - `num_warps=4`
    - `num_stages=2`
- 对较大 `M`，例如 `M=1024`
  - 原来第二段：
    - 复用第一段：
      - `BLOCK_SIZE_M=64`
      - `BLOCK_SIZE_N=64`
      - `BLOCK_SIZE_K=64`
      - `GROUP_SIZE_M=1`
      - `num_warps=4`
      - `num_stages=4`
  - 修改后第二段：
    - 命中 down-config 后：
      - `BLOCK_SIZE_M=64`
      - `BLOCK_SIZE_N=128`
      - `BLOCK_SIZE_K=64`
      - `GROUP_SIZE_M=1`
      - `num_warps=4`
      - `num_stages=3`

结果：

- 短输出正确性正常
- 两次 benchmark 稳态结果一致：
  - `TTFT = 100.47~100.71 ms`
  - `E2E = 0.5621~0.5624 s`
  - `output_tps = 112.03~112.08 tok/s`
- 取稳定值：
  - `TTFT = 100.47 ms`
  - `E2E = 0.5621 s`
  - `output_tps = 112.08 tok/s`

相对上一主线最好值：

- `111.15 -> 112.08 tok/s`
- 提升约 `0.93 tok/s`

结论：

- 这条优化有效
- 说明 routed-expert 第二个 GEMM 的最佳 tile/launch config 与第一段并不相同，直接复用第一段 config 会损失性能
- 当前最好稳定值更新为：
  - `TTFT = 100.47 ms`
  - `E2E = 0.5621 s`
  - `output_tps = 112.08 tok/s`
- 该实验进入主线

### MOE_SGL_REDUCE 修复复测 (2026-06-29)

背景：

- 早先 `MINISGL_MOE_SGL_REDUCE=1` 被判定为 correctness fail
- 现象是短输出里会出现异常 special token 和明显错误文本

根因定位：

- mini-sglang 当前接 `sgl_kernel.moe_sum_reduce(...)` 时，把第三个参数硬编码成了 `0.0`
- 但 sglang 主线传的是 `routed_scaling_factor`
- 对当前 Qwen3.6 路径来说，这等价于把 routed experts 的 reduce 输出整体缩成 0
- 同时，`routed_scaling_factor` 虽然从 `MoELayer` 传进了 `FusedMoe.forward()`，但原来并没有继续传到 `fused_experts_impl()`

修复：

- 把 `routed_scaling_factor` 继续传入 `fused_experts_impl()`
- `MOE_SGL_REDUCE` 分支中的
  - `sgl_kernel.moe_sum_reduce(..., 0.0)`
  改为
  - `sgl_kernel.moe_sum_reduce(..., routed_scaling_factor)`

结果：

- 修复后短输出 correctness 恢复正常
- 两次 benchmark 稳态结果：
  - 第一次：
    - `TTFT = 102.50 ms`
    - `E2E = 0.5623 s`
    - `output_tps = 112.04 tok/s`
  - 第二次：
    - `TTFT = 101.31 ms`
    - `E2E = 0.5611 s`
    - `output_tps = 112.28 tok/s`

结论：

- 先前 `MOE_SGL_REDUCE` 的 correctness fail 不是 `sgl_kernel.moe_sum_reduce` 本身有问题
- 而是 mini-sglang 的接法把 scale 传错了
- 修复后：
  - correctness 已恢复
  - 性能大致与当前 best 持平，可能有极小正收益
- 因此这条线现在不应再按“错误实验”看待，而应视为“已修复、可继续评估是否值得保留”的候选项

### 引擎级 decode graph replay 对齐：FI_GRAPH_FAST_DECODE_PLAN (2026-06-29)

背景：

- 继续从“整个推理引擎”角度对比 mini-sglang 与 sglang，而不是继续只盯单个 MoE 或 linear-attn kernel
- 代码对比后，发现两边在 decode + cuda graph 的 attention metadata 初始化路径上有一处明显设计差异：
  - mini-sglang：
    - `scheduler._prepare_batch()` 每步先构造一份新的 `attn_metadata`
    - `graph_runner.replay()` 前再走 `attn_backend.prepare_for_replay()`
    - 对 `FlashInferBackend` 来说，decode graph replay 仍然经过统一的 `metadata.wrapper.plan(...)` 路径
  - sglang：
    - attention backend 把 graph 内与 graph 外 metadata 初始化拆开
    - decode replay 会通过专门的 `indices_updater_decode.update(...)` 和预分配 wrapper/buffer 增量更新
    - FlashInfer decode graph 路径支持 `fast_decode_plan`

分析判断：

- 这说明 mini-sglang 与 sglang 的剩余差距里，至少有一部分可能来自：
  - decode graph replay 边界的 metadata/planner 组织方式
  - 而不只是单个算子 kernel 本体

实验：

- 新增开关：
  - `MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1`
- 做法：
  - 仅在 `FlashInferBackend.prepare_for_capture()` 的 decode cuda-graph wrapper 上，把 replay 期的 planner 切到 flashinfer `fast_decode_plan`
  - 为兼容 flashinfer 当前签名，在 `metadata.wrapper.plan(...)` 初始化处增加了对 `seq_lens` 参数是否存在的判断
- 这个实验只影响 `attention-backend=fi + decode + cuda graph`，默认行为不变

结果：

- 短输出 correctness 正常
- benchmark 稳态结果：
  - `TTFT = 100.22 ms`
  - `E2E = 0.5593 s`
  - `output_tps = 112.65 tok/s`

对比：

- 当前主线 best：
  - `TTFT = 100.47 ms`
  - `E2E = 0.5621 s`
  - `output_tps = 112.08 tok/s`
- 加上 `MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1` 后：
  - `112.08 -> 112.65 tok/s`

结论：

- 这是一个真实的正收益，但很小，约 `+0.57 tok/s`
- 说明从“推理引擎组织方式”角度看，decode graph replay 的 attention metadata / planner 路径确实有一部分差距
- 但也说明：
  - 单独把 FlashInfer decode planner 对齐到 `fast_decode_plan`，并不能解释 mini-sglang 与 sglang 之间剩余的大部分差距
  - 剩余差距更可能来自：
    - 更高层的 graph replay 边界组织
    - 整层 residual / layer glue
    - MoE 与 linear-attn 以外的累计模型级开销

备注：

- 另外还尝试了更激进的 `FI_GRAPH_REUSE_METADATA` 思路，目标是复用 decode graph 的 capture metadata buffer、减少每步 `torch.tensor/torch.cat` 重建
- 这条线目前只完成了实验脚手架，还没有形成稳定 benchmark 结果，不进入主线

### 引擎级 decode batch 边界复用：DECODE_BATCH_REUSE_BUFFERS (2026-06-29)

背景：

- 在继续从“整个推理引擎”角度看 mini-sglang 与 sglang 差异时，除了 attention replay planner 外，另一个明显差异是 decode 每步 batch 准备边界：
  - mini-sglang 当前在 scheduler 中每步都会新建：
    - `positions`
    - `input_mapping`
    - `write_mapping`
  - 然后再做 host->device copy
- sglang 的 decode graph 路径则更偏向：
  - 预分配静态 buffer
  - batched copy / grouped copy
  - replay 前只更新切片

实验：

- 新增开关：
  - `MINISGL_DECODE_BATCH_REUSE_BUFFERS=1`
- 做法：
  - 只在 decode 路径启用
  - 为 `positions / input_mapping / write_mapping` 引入 host/device 复用 buffer
  - 为兼容 overlap scheduling，再补成双缓冲，避免上一批尚未消费完时被下一批覆盖

correctness：

- 第一版单缓冲实现会破坏 overlap 调度，导致输出错乱
- 补成双缓冲后，短输出 correctness 恢复正常

性能结果：

- 稳态 benchmark：
  - `TTFT = 100.89 ms`
  - `E2E = 0.5643 s`
  - `output_tps = 111.64 tok/s`

对比：

- 当前主线 best：
  - `TTFT = 100.47 ms`
  - `E2E = 0.5621 s`
  - `output_tps = 112.08 tok/s`
- 开启 `MINISGL_DECODE_BATCH_REUSE_BUFFERS=1` 后：
  - `112.08 -> 111.64 tok/s`

结论：

- 这条实验在修正 correctness 后仍然退化，不进入主线
- 说明：
  - “decode batch 边界仍有引擎级开销” 这个方向本身不一定错
  - 但当前这种简单的 host/device 索引 buffer 复用方式，没有打中真正端到端瓶颈
  - 相比之下，继续在这类小索引张量复用上深挖的优先级应当降低

### MoE 周边小张量复用拆分实验 (2026-06-29)

背景：

- 之前 `MINISGL_MOE_REUSE_WORKSPACE=1` 整体测出来是退化的
- 但它把两类行为绑在了一起：
  - `topk_weights / topk_ids` 复用
  - `moe_align_block_size` 的 `sorted_ids / expert_ids / cumsum_buffer` 复用
- 为了避免不同子项互相掩盖，这一轮把它拆成了两个独立开关：
  - `MINISGL_MOE_REUSE_TOPK_WORKSPACE=1`
  - `MINISGL_MOE_REUSE_ALIGN_WORKSPACE=1`

实验 1：只开 `MINISGL_MOE_REUSE_TOPK_WORKSPACE=1`

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 100.52 ms`
  - `E2E = 0.5572 s`
  - `output_tps = 113.06 tok/s`

对比：

- 相对上一主线 best `112.85 tok/s`
  - `112.85 -> 113.06 tok/s`

结论：

- `topk` 临时缓冲复用是正收益
- 说明当前 MoE 周边链里，“每步小张量分配” 仍然有真实成本
- 这条实验进入主线候选

实验 2：只开 `MINISGL_MOE_REUSE_ALIGN_WORKSPACE=1`

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 100.89 ms`
  - `E2E = 0.5579 s`
  - `output_tps = 112.93 tok/s`

对比：

- 相对 `112.85 tok/s`
  - `112.85 -> 112.93 tok/s`

结论：

- `align` 临时缓冲复用基本持平，仅有极小收益
- 说明 `moe_align_block_size` 的瓶颈更可能在 kernel 本身或调用组织，而不是这些缓冲分配

实验 3：同时开 `MINISGL_MOE_REUSE_TOPK_WORKSPACE=1` 与 `MINISGL_MOE_REUSE_ALIGN_WORKSPACE=1`

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 100.57 ms`
  - `E2E = 0.5573 s`
  - `output_tps = 113.04 tok/s`

结论：

- 同时打开基本等于只开 `topk workspace`
- 进一步说明这条收益主要来自 `topk`，不是 `align`

实验 4：`MINISGL_MOE_GATE_MM_OUT=1`

背景：

- 受 `topk workspace` 有收益这个现象启发，继续尝试把 router logits 的输出分配也改成复用
- 做法：
  - 为 MoE gate 增加 `_router_logits_buffer`
  - 在满足条件时，用 `torch.mm(hidden_states, gate.weight.t(), out=cached)` 代替普通 `F.linear(...)`

结果：

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 100.98 ms`
  - `E2E = 0.5587 s`
  - `output_tps = 112.76 tok/s`

结论：

- router logits 这条 `mm(out=...)` 路线没有收益，略微退化
- 说明：
  - MoE 周边的“小张量分配”并不是都值得处理
  - 当前真正打中的仍然是 `topk` 输出缓冲，而不是 router GEMM 输出

实验 5：`MINISGL_MOE_REUSE_TOPK_WORKSPACE=1 + MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1`

背景：

- `MOE_REUSE_TOPK_WORKSPACE` 与 `FI_GRAPH_FAST_DECODE_PLAN` 分别都已经证明是正收益
- 这一轮验证两者是否可以叠加

结果：

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 99.06 ms`
  - `E2E = 0.5526 s`
  - `output_tps = 114.01 tok/s`

对比：

- 相对上一主线 best `112.85 tok/s`
  - `112.85 -> 114.01 tok/s`
- 相对只开 `FI_GRAPH_FAST_DECODE_PLAN=1` 的 `112.65 tok/s`
  - `112.65 -> 114.01 tok/s`

结论：

- 两者可以稳定叠加
- 当前新的最好稳定值更新为：
  - `TTFT = 99.06 ms`
  - `E2E = 0.5526 s`
  - `output_tps = 114.01 tok/s`

阶段性判断：

- 当前最有效的新点是：
  - `MINISGL_MOE_REUSE_TOPK_WORKSPACE=1`
  - `MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1`
- 当前已经基本可以排除的点是：
  - `MOE_REUSE_ALIGN_WORKSPACE`
  - `MOE_GATE_MM_OUT`
- 这说明剩余差距里，MoE 周边仍然存在可打的小张量/小边界成本，但需要继续做更细的单因素筛选，而不是泛化成“所有 buffer 复用都有收益”

实验 6：`MINISGL_MOE_FASTPATH_TOPK2_REDUCE=1`

背景：

- 参考 `sglang main` 的 `topk == 2 && routed_scaling_factor == 1.0` 分支
- 试图用 `torch.add(intermediate_cache3[:, 0], intermediate_cache3[:, 1], out=...)` 替代通用 reduce

结果：

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 101.03 ms`
  - `E2E = 0.5555 s`
  - `output_tps = 113.41 tok/s`

结论：

- 该 fastpath 在当前图模式和 kernel 组合下没有收益
- 相对当前 best `114.01 tok/s` 明显退化
- 说明这类“看起来更像 sglang”的局部替换，未必能直接转化成端到端收益

实验 7：`MINISGL_MOE_TORCH_COMPILE_REDUCE=1`

背景：

- 继续对齐 `sglang main` 的 MoE reduce 逻辑
- `sglang` 在小 token 数下会优先走 `torch.compile` 版 `moe_sum_reduce`
- 在 `MINISGL_MOE_SGL_REDUCE=1` 基础上，额外引入一个小 token fastpath 做验证

结果：

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 100.28 ms`
  - `E2E = 0.5548 s`
  - `output_tps = 113.55 tok/s`

结论：

- 这条 `torch.compile reduce` 路线在当前环境下无收益，略退化
- 说明：
  - `sglang` 的这段实现细节不是当前 mini-sglang 剩余差距的关键
  - `MOE_SGL_REDUCE` 是否有效，仍要结合当前整套主线配置单独判断，不能简单叠加更多 reduce 变体

实验 8：`MINISGL_MOE_SINGLE_KERNEL=1`

背景：

- 把 routed experts 的 `silu_and_mul + down_proj` 融合成单 kernel
- 属于 MoE 主体内更激进的一条线，用于验证“更宽的内核融合”是否能直接带来收益

结果：

- 短输出 correctness 正常
- 稳态结果：
  - `TTFT = 123.15 ms`
  - `E2E = 0.6186 s`
  - `output_tps = 101.84 tok/s`

结论：

- 该单 kernel 路线在当前 Qwen3.6 routed-expert shape 上强烈退化
- 说明：
  - 当前瓶颈虽然在 MoE 大链路，但不是“融合得越宽越好”
  - 这类更激进的内核合并需要更精细的 shape-aware 条件，否则会直接破坏已有的更优 kernel 组合

实验 9：重新消融 `MINISGL_MOE_SGL_REDUCE`

背景：

- 当前 best `114.01 tok/s` 已经叠加了：
  - `MINISGL_MOE_REUSE_TOPK_WORKSPACE=1`
  - `MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1`
  - `MINISGL_MOE_SGL_REDUCE=1`
- 为了避免误判，需要在最新主线里重新确认 `MOE_SGL_REDUCE` 是否仍然有正收益

结果：

- 关闭 `MINISGL_MOE_SGL_REDUCE=1` 后
- 稳态结果：
  - `TTFT = 100.64 ms`
  - `E2E = 0.5584 s`
  - `output_tps = 112.83 tok/s`

结论：

- 在当前最新主线里，`MOE_SGL_REDUCE` 仍然是有效项
- 对比当前 best：
  - `112.83 -> 114.01 tok/s`
- 这说明：
  - `reduce/combine` 这条 MoE 周边链仍然有真实收益
  - 但继续沿 reduce 分支再叠加额外变体，收益已经很有限，甚至容易退化

### 当前最优主线下的 bf16 / W8A8 / W8A8+NormFuse 同口径对比 (2026-06-29)

背景：

- 为避免把不同阶段、不同开关组合的结果混在一起，这一轮统一使用当前 bf16 最优主线的大部分稳定开关：
  - `MINISGL_GEMMA_FUSED_NORM=1`
  - `MINISGL_FULL_ATTN_FUSED_PREPARE=1`
  - `MINISGL_DEPTHWISE_CONV_DECODE=1`
  - `MINISGL_LINEAR_RMSNORM_GATED=1`
  - `MINISGL_LINEAR_RMSNORM_GATED_REUSE_OUT=1`
  - `MINISGL_SHARED_EXPERT_FUSED_GATE_ADD=1`
  - `MINISGL_MOE_FUSED_ACTIVATION=1`
  - `MINISGL_SHARED_EXPERT_FUSED_ACTIVATION=1`
  - `MINISGL_LINEAR_PREFILL_QK_L2NORM=1`
  - `MINISGL_LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS=1`
  - `MINISGL_SKIP_AB_FP32_CAST=1`
  - `MINISGL_LINEAR_DECODE_FUSED_INPUT_PROJ=1`
  - `MINISGL_LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS=1`
  - `MINISGL_LINEAR_DECODE_SKIP_AB_CONTIGUOUS=1`
  - `MINISGL_MOE_SKIP_TOPK_POST_RENORM=1`
  - `MINISGL_MOE_SKIP_TOPK_FP32_CAST=1`
  - `MINISGL_MOE_SKIP_DISPATCH_LOCAL_MASK=1`
  - `MINISGL_MOE_ALIGN_SMALL_CAP=1`
  - `MINISGL_MOE_SGLANG_CONFIG_LOOKUP=1`
  - `MINISGL_MOE_SGLANG_DOWN_CONFIG=1`
  - `MINISGL_MOE_SGL_REDUCE=1`
  - `MINISGL_MOE_REUSE_TOPK_WORKSPACE=1`
  - `MINISGL_FI_GRAPH_FAST_DECODE_PLAN=1`
- 在此基础上只切换三种模式：
  - `bf16`
  - `W8A8`
  - `W8A8 + MINISGL_W8A8_FUSED_GEMMA_NORM_QUANT=1`

结果：

- `bf16`
  - `TTFT = 100.57 ms`
  - `E2E = 0.5546 s`
  - `output_tps = 113.59 tok/s`
- `W8A8`
  - `TTFT = 89.25 ms`
  - `E2E = 0.5715 s`
  - `output_tps = 110.24 tok/s`
- `W8A8 + NormFuse`
  - `TTFT = 88.97 ms`
  - `E2E = 0.5702 s`
  - `output_tps = 110.49 tok/s`

结论：

- 在当前最优主线配置下，`W8A8` 相对 `bf16` 仍然存在小幅稳态退化：
  - `113.59 -> 110.24 tok/s`
  - 约 `-2.95%`
- 把 `GemmaRMSNorm + per-token int8 quant` 做成真实 Triton fuse 并接到 `post_attention_layernorm -> int8 MLP/MoE` 之后，`W8A8` 有小幅回升：
  - `110.24 -> 110.49 tok/s`
  - 约 `+0.23%`
- 说明：
  - `norm+quant fuse` 本身不是无效，而是 **有效但不是当前 W8A8 相对 bf16 的主矛盾**
  - 当前 W8A8 的主要差距更可能仍在其它量化边界，例如 linear-attention 的 int8 `in_proj/out_proj` 或 MoE expert 的 int8 主链
