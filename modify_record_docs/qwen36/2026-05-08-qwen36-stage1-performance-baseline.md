# 2026-05-08 Qwen3.6 Stage1 推理性能基线

## 1. 目的

当前 `Qwen3.6-35B-A3B` 在 `mini-sglang` 中已经完成 `stage1` 跑通，但很多路径仍然会 fallback 到 torch 实现。

因此这里先固定一版可复跑的性能基线，后续每次替换或优化 kernel 时，都可以基于同一套命令和同一组输入长度做对比。

## 2. 基线服务配置

测试时使用的服务启动命令如下：

```bash
export PATH=/usr/local/cuda-12.4/bin:/data/zxz/condaenv/minisgl/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_INSTALL_PATH=/usr/local/cuda-12.4
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CUDA_VISIBLE_DEVICES=7

/data/zxz/condaenv/minisgl/bin/python -m minisgl \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --tp-size 1 \
  --ep-size 1 \
  --attn fi \
  --linear-attn-backend torch \
  --moe-backend torch \
  --dtype bfloat16 \
  --graph 0 \
  --num-pages 128 \
  --max-running-requests 1 \
  --cache-type naive \
  --disable-pynccl \
  --port 1921
```

说明：

- 这是当前 `stage1` 已验证可正常输出的稳定配置
- `linear attention` 和 `moe` 当前都显式走 `torch` 后端
- 因此该结果应视为“正确性优先版本”的性能基线，而不是最终优化上限

## 3. 基线测试脚本

本次新增可复跑脚本：

- `benchmark/online/bench_qwen36_stage1.py`

执行命令：

```bash
PYTHONPATH=python /data/zxz/condaenv/minisgl/bin/python \
  benchmark/online/bench_qwen36_stage1.py
```

脚本行为：

- 先做一次 warmup
- 然后顺序执行两个 case
- 短输入 case 重复 3 次
- 中等输入 case 当前保留为单次探针
- 统计 `TTFT`、`E2E` 和输出 token 吞吐

当前固定测试 case：

- `short_prefill_short_decode`: `input_len=64`, `output_len=32`, `repeats=3`
- `medium_prefill_short_decode`: `input_len=256`, `output_len=32`, `repeats=1`

## 4. 本次实测结果

测试时间：

- 2026-05-08

测试环境：

- GPU：`CUDA_VISIBLE_DEVICES=7`
- 模型：`/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`
- 服务地址：`http://127.0.0.1:1921/v1`

### 4.1 warmup

- 提示：`请简短回答：1+1等于几？`
- 输出上限：`16`

结果：

- `TTFT`：`2466.60ms`
- `E2E`：`3.31s`
- 实际输出 token 数：`1`

### 4.2 short_prefill_short_decode

- 输入长度：`64`
- 输出长度：`32`
- 重复次数：`3`

结果：

- `TTFT`：`avg=2858.19ms`，`p50=2890.41ms`，`p90=2918.31ms`，`max=2918.31ms`
- `E2E`：`avg=21.71s`，`p50=21.70s`，`p90=21.88s`，`max=21.88s`
- 输出吞吐：`0.97 tok/s`
- 估算 `TPOT`：`1033.71ms`

### 4.3 medium_prefill_short_decode 探针

- 输入长度：`256`
- 输出长度：`32`
- 当前先保留为单次探针，不做重复统计

探针结果：

- `time_starttransfer` 约为 `0.002s`
- `time_total` 约为 `29.70s`

说明：

- 当前 `curl` 测到的 `time_starttransfer` 很小，但这并不能代表真实首 token 时间
- 因为该接口返回的是流式 SSE，HTTP 首字节到达非常早
- 真正有意义的性能判断是：中等 prefill 场景下，总体完成时间已经接近 `30s`
- 这说明当前 `torch fallback` 路径在 prefill 成本上仍然很重

## 5. 当前结论

在当前 `stage1` 稳定配置下：

- 单卡、`torch fallback` 路径下短输出吞吐约为 `0.97 tok/s`
- 短输入场景下 `TTFT` 已经接近 `2.9s`
- 短输入场景下估算 `TPOT` 约为 `1.03s`
- 中等 prefill 场景总耗时已经接近 `30s`
- 当前结果适合作为后续 kernel 优化前的第一版基线

## 6. 后续建议

后续每次优化时，建议保持下面几项不变再做对比：

- 同一张卡
- 同一模型权重
- 同一启动参数
- 同一 benchmark 脚本
- 同一输入输出长度

优先关注的对比指标：

- `TPOT` 是否下降
- 输出吞吐 `tok/s` 是否提升
- 中等输入场景总耗时是否显著下降
- 在吞吐提升的同时，输出语义是否仍然正确
