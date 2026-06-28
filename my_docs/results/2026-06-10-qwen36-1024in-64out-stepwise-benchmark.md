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
