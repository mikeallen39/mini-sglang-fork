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
  - `my_docs/results/baseline_run1_output.txt`
- 正确性检查：
  - `run1` 输出为正常中文科学说明性续写
  - 未见乱码、随机符号串、异常重复或明显语义崩坏
  - 末尾截断在句中，符合 `output_tokens=64` 的预期
- 结果：
  - `run1`: `TTFT=7283.89 ms`, `E2E=39.8754 s`, `output_tokens=63`
  - `run2`: `TTFT=6416.91 ms`, `E2E=38.7957 s`, `output_tokens=63`
  - `run3`: `TTFT=6406.16 ms`, `E2E=38.7591 s`, `output_tokens=63`
  - `run4`: `TTFT=6420.35 ms`, `E2E=38.7646 s`, `output_tokens=63`
  - `run5`: `TTFT=6409.90 ms`, `E2E=38.7557 s`, `output_tokens=63`
  - 稳态 `run2-run5 avg TTFT=6413.33 ms`
  - 稳态 `run2-run5 avg E2E=38.7688 s`
  - 稳态 `run2-run5 output_tps=1.63 tok/s`
  - 稳态 `avg_ms_per_output_token=615.38 ms`
  - 稳态 `avg_output_tokens=63.00`
