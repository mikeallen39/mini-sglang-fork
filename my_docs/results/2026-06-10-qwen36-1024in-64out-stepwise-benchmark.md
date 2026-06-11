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

## 2. 实验顺序

按下面顺序逐步增加优化：

1. 基线：`torch MoE + torch linear attention + bf16 + graph off + tp=1 + ep=1`
2. 只切 `fused MoE`
3. 再切 `sglang linear attention`
4. 再测试 `CUDA graph`
5. 再测试 `int8 w8a8`
6. 最后测试 `TP/EP`

## 3. 结果

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
  - `run1`: `TTFT=7254.20 ms`, `E2E=39.3084 s`, `output_tokens=63`
  - `run2`: `TTFT=6416.83 ms`, `E2E=38.4262 s`, `output_tokens=63`
  - `run3`: `TTFT=6407.57 ms`, `E2E=38.4562 s`, `output_tokens=63`
  - `run4`: `TTFT=6412.11 ms`, `E2E=38.4539 s`, `output_tokens=63`
  - `run5`: `TTFT=6415.76 ms`, `E2E=38.4364 s`, `output_tokens=63`
  - 稳态 `run2-run5 avg TTFT=6413.07 ms`
  - 稳态 `run2-run5 avg E2E=38.4432 s`
  - 稳态 `run2-run5 output_tps=1.64 tok/s`
  - 稳态 `avg_ms_per_output_token=610.21 ms`
  - 稳态 `avg_output_tokens=63.00`

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
  - `run1`: `TTFT=5425.46 ms`, `E2E=9.8339 s`, `output_tokens=63`
  - `run2`: `TTFT=4426.99 ms`, `E2E=8.8033 s`, `output_tokens=63`
  - `run3`: `TTFT=4423.81 ms`, `E2E=8.7943 s`, `output_tokens=63`
  - `run4`: `TTFT=4473.97 ms`, `E2E=8.9083 s`, `output_tokens=63`
  - `run5`: `TTFT=4585.72 ms`, `E2E=8.9457 s`, `output_tokens=63`
  - 稳态 `run2-run5 avg TTFT=4477.62 ms`
  - 稳态 `run2-run5 avg E2E=8.8629 s`
  - 稳态 `run2-run5 output_tps=7.11 tok/s`
  - 稳态 `avg_ms_per_output_token=140.68 ms`
  - 稳态 `avg_output_tokens=63.00`
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
  - `run1`: `TTFT=1178.97 ms`, `E2E=5.0917 s`, `output_tokens=63`
  - `run2`: `TTFT=236.74 ms`, `E2E=4.1440 s`, `output_tokens=63`
  - `run3`: `TTFT=235.90 ms`, `E2E=4.1392 s`, `output_tokens=63`
  - `run4`: `TTFT=236.04 ms`, `E2E=4.1365 s`, `output_tokens=63`
  - `run5`: `TTFT=235.75 ms`, `E2E=4.1828 s`, `output_tokens=63`
  - 稳态 `run2-run5 avg TTFT=236.11 ms`
  - 稳态 `run2-run5 avg E2E=4.1506 s`
  - 稳态 `run2-run5 output_tps=15.18 tok/s`
  - 稳态 `avg_ms_per_output_token=65.88 ms`
  - 稳态 `avg_output_tokens=63.00`
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
  - `run1`: `TTFT=259.68 ms`, `E2E=0.9978 s`, `output_tokens=63`
  - `run2`: `TTFT=196.05 ms`, `E2E=0.9353 s`, `output_tokens=63`
  - `run3`: `TTFT=194.89 ms`, `E2E=0.9340 s`, `output_tokens=63`
  - `run4`: `TTFT=194.66 ms`, `E2E=0.9339 s`, `output_tokens=63`
  - `run5`: `TTFT=194.70 ms`, `E2E=0.9337 s`, `output_tokens=63`
  - 稳态 `run2-run5 avg TTFT=195.07 ms`
  - 稳态 `run2-run5 avg E2E=0.9342 s`
  - 稳态 `run2-run5 output_tps=67.44 tok/s`
  - 稳态 `avg_ms_per_output_token=14.83 ms`
  - 稳态 `avg_output_tokens=63.00`
- 相对 baseline：
  - `TTFT` 提升约 `97.0%`
  - `E2E` 提升约 `97.6%`
  - `output_tps` 提升约 `4038%`
- 相对 `fused MoE + sglang linear attention + graph=0`：
  - `TTFT` 提升约 `86.7%`
  - `E2E` 提升约 `97.4%`
  - `output_tps` 提升约 `3711%`
