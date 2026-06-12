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
| W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph + EP2 | 208.44 ms | 1.0111 s | 62.31 tok/s | 16.05 ms | 63.00 | 20775 MiB x 2 |
| W8A8 Int8 + Fused MoE + SGLang Linear Attention + CUDA Graph + EP4 | 205.37 ms | 1.0244 s | 61.50 tok/s | 16.26 ms | 63.00 | 13001 MiB x 4 |

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
