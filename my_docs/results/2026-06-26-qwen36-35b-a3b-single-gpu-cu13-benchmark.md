# Qwen3.6-35B-A3B 单卡基线测试记录

## 环境信息

- 日期：2026-06-26
- SGLang 仓库：`/mnt/42_store/zxz/aiinfra/sglang`
- 测试脚本：`/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/benchmark/online/bench_qwen36_1024in_64out.py`
- 结果目录：`/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results`
- 模型路径：`/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`
- GPU：`NVIDIA A800 80GB PCIe`
- PyTorch / CUDA 用户态：`torch 2.11.0+cu130`
- Triton：`3.6.0`
- 服务地址：`127.0.0.1:1919`

## 启动参数

```bash
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --host 127.0.0.1 \
  --port 1919 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --max-running-requests 1 \
  --mem-fraction-static 0.90
```

说明：

- 仅使用单卡。
- `--max-running-requests 1` 是当前单卡最稳的单并发起点。
- `--mem-fraction-static 0.90` 是为了避免 hybrid mamba state cache 分配失败。

## 对测试脚本做的最小兼容修改

这个模型默认开启 thinking，原始脚本只统计流式返回里的 `delta.content`。  
但 Qwen3.6 在 thinking 开启时，输出可能主要落在 `delta.reasoning_content` 中，导致脚本误判为“没有收到输出 token”。

因此只做了两处最小修正：

- 请求里增加 `chat_template_kwargs={"enable_thinking": False}`
- 收集流式输出时兼容 `delta.content` 和 `delta.reasoning_content`

如果不做这两处处理，这个脚本对当前模型可能直接报错，而不是得到有效性能结果。

## 测试配置

- 输入长度：`1024` tokens
- 输出长度：`64` tokens
- 运行次数：`5`
- 并发：`1`

## 原始结果

```text
run1: TTFT=1012.00 ms, E2E=1.4224 s, output_tokens=64
run2: TTFT=745.46 ms, E2E=1.1539 s, output_tokens=64
run3: TTFT=117.01 ms, E2E=0.5263 s, output_tokens=64
run4: TTFT=113.59 ms, E2E=0.5228 s, output_tokens=64
run5: TTFT=109.39 ms, E2E=0.5191 s, output_tokens=64
```

## 稳态结果

考虑到本次 `run2` 的 TTFT 明显高于后续几轮，更合理的热稳态口径应采用 `run3` 到 `run5`。

热稳态统计（`run3`-`run5`）：

- 平均 TTFT：`113.33 ms`
- 平均 E2E：`0.5227 s`
- 输出吞吐：`122.43 tok/s`
- 平均每个输出 token 延迟：`8.17 ms/token`

保留脚本默认口径（`run2`-`run5`）供参考：

- 平均 TTFT：`271.36 ms`
- 平均 E2E：`0.6805 s`
- 输出吞吐：`94.04 tok/s`
- 平均每个输出 token 延迟：`10.63 ms/token`

## 为什么没有命中专用 MoE Triton Config

这不是“模型不支持”，而是 SGLang 查找 MoE Triton 调优配置时，使用的是一组非常严格的键，而不是模型名。

查找逻辑在：

- `/mnt/42_store/zxz/aiinfra/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`

它会按下面这些条件拼接 JSON 文件名：

- MoE 层形状里的 `E`
- MoE 层形状里的 `N`
- 当前 GPU 名称
- 当前 Triton 版本
- 可选后缀：`dtype / block_shape / down_moe`

文件名模式大致是：

```text
E={E},N={N},device_name={device_name}...json
```

本次服务日志显示，它实际查找的是：

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_A800_80GB_PCIe.json
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_A800_80GB_PCIe_down.json
```

这两个文件在当前仓库里都不存在，所以 SGLang 回退到了默认 MoE kernel config，并打印了：

- `Using default MoE kernel config. Performance might be sub-optimal!`
- `Using MoE kernel config with down_moe=False. Performance might be sub-optimal!`

## 这次为什么会缺这个配置

当前环境的精确条件是：

- GPU 名称：`NVIDIA A800 80GB PCIe`
- 归一化后的设备名：`NVIDIA_A800_80GB_PCIe`
- Triton 版本：`3.6.0`
- 目标形状：`E=256, N=512`

而仓库里已有的很多 A800 配置，常见的是：

- `NVIDIA_A800-SXM4-80GB`

这和 `NVIDIA_A800_80GB_PCIe` 是两个不同的精确字符串，SGLang 不会自动把它们视为同一类设备复用。

也就是说，这次没有命中专用配置，原因是：

- 当前这张卡的设备名不匹配
- 当前需要的 `E=256,N=512` 组合没有对应 JSON
- 当前 `triton_3_6_0` 目录下没有这个精确条目

## 结论

- 当前模型已经能够在单卡上稳定跑通。
- 本次结果是有效的单并发基线结果。
- 若按真正热稳态看，`run3`-`run5` 更能代表当前服务的单并发表现。
- 但这不一定是这张 A800 PCIe 卡在该模型上的绝对最佳单并发性能。
- 如果后续补齐 `E=256,N=512`、`NVIDIA_A800_80GB_PCIe`、`triton 3.6.0` 对应的专用调优 JSON，MoE 部分的性能还有进一步提升空间。

## 关闭 Radix Cache 的公平对比复测

为了确认前面的高性能是否主要来自 prefix/radix cache，对同一模型、同一脚本、同一请求形状又做了一次复测，这次显式关闭 radix cache。

服务启动参数：

```bash
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --host 127.0.0.1 \
  --port 1919 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --max-running-requests 1 \
  --mem-fraction-static 0.90 \
  --disable-radix-cache
```

启动后可见两个关键变化：

- `disable_radix_cache=True`
- `Tree cache initialized: ... ChunkCache`

同时内存布局也发生了变化：

- `max_mamba_cache_size`：从 `42` 降到 `1`
- KV token 容量：从 `146777` 提升到 `275681`

### 关闭 Radix Cache 后的结果

原始结果：

```text
run1: TTFT=426.21 ms, E2E=0.8211 s, output_tokens=64
run2: TTFT=106.11 ms, E2E=0.5149 s, output_tokens=64
run3: TTFT=106.21 ms, E2E=0.5168 s, output_tokens=64
run4: TTFT=106.92 ms, E2E=0.5159 s, output_tokens=64
run5: TTFT=106.21 ms, E2E=0.5157 s, output_tokens=64
```

稳态统计（`run2`-`run5`）：

- 平均 TTFT：`106.36 ms`
- 平均 E2E：`0.5158 s`
- 输出吞吐：`124.07 tok/s`
- 平均每个输出 token 延迟：`8.06 ms/token`

## 与 mini-sglang 结果的并排分析

参考文档：

- `/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/2026-06-10-qwen36-1024in-64out-stepwise-benchmark.md`

最接近的对比项应当是 mini-sglang 中这一档：

- `Fused MoE + SGLang Linear Attention + CUDA Graph`

对应结果为：

- 平均 TTFT：`195.07 ms`
- 平均 E2E：`0.9342 s`
- 输出吞吐：`67.44 tok/s`
- 平均每个输出 token 延迟：`14.83 ms/token`

和本次 `sglang main + disable radix cache` 对比：

| 配置 | TTFT | E2E | output_tps | avg_ms_per_output_token |
| --- | ---: | ---: | ---: | ---: |
| mini-sglang: Fused MoE + SGLang Linear Attention + CUDA Graph | 195.07 ms | 0.9342 s | 67.44 tok/s | 14.83 ms |
| sglang main: disable radix cache | 106.36 ms | 0.5158 s | 124.07 tok/s | 8.06 ms |

### 对比结论

- 之前看到的性能差距并不主要来自 radix cache。
- 即使显式关闭 radix cache，当前 `sglang main` 仍然明显快于 `mini-sglang`。
- 因此，两者差距更可能主要来自：
  - 当前主线 runtime 的实现差异
  - decode 路径上的 kernel 选择与调度
  - `flashinfer` attention / sampling 路径
  - 当前 fused MoE、linear attention、decode CUDA graph 的整体集成方式

换句话说，`sglang main` 这次的优势，更像是“主线推理栈本身更快”，而不是“重复 prompt 被 cache 命中了所以看起来更快”。
