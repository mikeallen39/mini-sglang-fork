# 2026-03-25 GLM-4.7-Flash Progress

## Scope

本文跟踪分支 `feat/glm47-flash-support` 上对 GLM-4.7-Flash 的支持工作。

## Pushed Changes

- 已推送 commit：`a79fc0c`（`Fix GLM-4.7 MLA backend and streamline loading`）
- 远端分支：`origin/feat/glm47-flash-support`

### What is included in `a79fc0c`

- 修复了 `glm4_moe_lite` 的真实 MLA 路径。
- 恢复了 GLM 自动选择逻辑，使其使用 MLA attention backend。
- 修复了 GLM 的 MoE backend 选择逻辑，使其默认使用 fused MoE。
- 修复了与 TP 相关的 GLM weight loading / expert packing 问题。
- 新增了配置回归测试：
  - `tests/misc/test_glm4_config.py`

## Verified Results Before Further TP=1 Optimization

### Config regression

- 命令：
  - `PYTHONPATH=python /data/zxz2/condaenv/mini-sglang/bin/python -m pytest -q tests/misc/test_glm4_config.py`
- 结果：
  - `2 passed`

### Real MLA backend functional tests

- `tp=1`，GPU `6`，显式指定 `--attention-backend mla`
  - Prompt: `hi`
  - Output: `Hello! How can I assist you today?`

- `tp=2`，GPUs `0,6`，显式指定 `--attention-backend mla`
  - Prompt: `hi`
  - Output: `Hello! How can I help you today?`

## TP=1 Loading Investigation

### Symptom

- `tp=2` 的加载速度明显快于 `tp=1`。
- 之前真实的 `tp=1` 运行需要数分钟，而 `tp=2` 可以在几十秒内完成启动。

### Profiling conclusion

- 单纯读取 `safetensors` 不是瓶颈。
  - 只读取 tensor 时，前 6 个 shard 基本都在 `0.3s` 左右。
- merge / expert pack 逻辑本身也不是瓶颈。
  - 在执行 merge/pack 但不保留最终 tensor 时，前 6 个 shard 仍然大约是 `0.3s`。
- 当大量最终 tensor 以持久 GPU allocation 的形式保留时，才会出现明显减速。
  - 当输出保存在 dict 中时，直到分配量达到约 `15 GiB` 之前，每个 shard 的耗时都保持较低；之后会上升到每个 shard 数秒。
- 更好的策略是先在 GPU 上预分配模型 tensor，再把权重 `copy_` 到这些现有 tensor 中。
  - 在 GPU `6` 上的测试结果：
    - 预分配模型创建耗时约 `2.85s`
    - 创建后的 GPU allocation 约为 `55.84 GiB`
    - 此后 shards `1..20` 基本保持在每个 `0.3-0.4s`

## Current Unpushed Changes

这些改动目前只在本地，还没有推送：

- `python/minisgl/engine/engine.py`
  - 对 `glm4_moe_lite` 且 `tp=1` 的情况，直接在 GPU 上初始化模型，而不是使用 meta tensors。
- `python/minisgl/models/weight.py`
  - 当 shape / dtype / device 匹配时，优先 `copy_` 到已经实体化的 tensor 中，而不是替换为新的持久分配。

## Latest TP=1 Optimization Validation

### End-to-end real run after preallocation change

- 命令形态：
  - `CUDA_VISIBLE_DEVICES=6 ... python -m minisgl --model-path /mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash --tensor-parallel-size 1 --attention-backend mla --port 30225 --shell-mode`
- 观察到的时间点：
  - 加载前空闲显存日志：`15:12:19`
  - 权重加载完成：`15:12:36`
  - Scheduler ready：`15:12:37`
- 实际结果：
  - 权重加载时间降到了约 `17s`
  - 端到端启动到 ready 现在已经是几十秒量级，而不是之前的多分钟

### Functional check after the loading optimization

- Prompt: `hi`
- Output: `Hello! How can I assist you today?`

## Current Status

- MLA backend 在 `tp=1` 和 `tp=2` 下都已经对 GLM-4.7-Flash 功能正确。
- `tp=1` 的加载优化已经通过真实端到端运行验证。
- 剩余的可选步骤：
  - 如果希望这一阶段的成果也落到远端，可以再提交 / 推送新的 `tp=1` 预分配优化。
