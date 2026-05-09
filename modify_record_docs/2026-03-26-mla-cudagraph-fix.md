# 2026-03-26 MLA CUDA Graph Fix for GLM-4.7-Flash

## Summary

这次更新修复了 GLM-4.7-Flash 在使用 MLA attention backend 时，剩余的
仅在 graph 模式下出现的失败问题。

修复前：

- `graph off` 已经是正确的
- `graph on` 在真实端到端运行中会失败
  - `tp=2, ep=2` 会产生乱码
  - `tp=2, ep=1` 也会产生乱码
  - `tp=1, ep=1` 会触发 `CUDA error: an illegal memory access was encountered`

这说明剩余问题既不是 EP 特有的，也不是 TP 特有的。共同因素是
MLA CUDA-graph 路径。

## Root Cause

`python/minisgl/attention/mla_backend.py` 在 graph capture/replay 场景中，
创建的是普通的 `flashinfer.mla.BatchMLAPagedAttentionWrapper` 实例。

这意味着 MLA graph 路径缺少了 FlashInfer 在 `use_cuda_graph=True` 时，
为 replay-safe 所必需的静态缓冲区，尤其是：

- `qo_indptr`
- `kv_indptr`
- `kv_indices`
- `kv_len_arr`

换句话说，代码回放的是 eager 风格的 MLA wrapper，而不是真正支持
CUDA graph 的 MLA wrapper。

## Fix

已更新：

- `python/minisgl/attention/mla_backend.py`

在 `MLABackend.init_capture_graph()` 中做出的改动：

- 现在 graph wrappers 会以 `use_cuda_graph=True` 构造
- 显式从预分配的 capture storage 绑定 replay buffers：
  - `capture.cu_seqlens_q`
  - `capture.cu_seqlens_k`
  - `capture.page_table`
  - `capture.seq_lens`

这样就使 FlashInfer 的 MLA wrapper 在 CUDA graph 执行下具备 replay-safe 状态。

## Real GPU Validation

环境：

- model:
  - `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`
- python:
  - `/data/zxz2/condaenv/mini-sglang/bin/python`
- attention backend:
  - `mla`
- moe backend:
  - `fused`

通过 `/v1/chat/completions` 的真实 `hi` 请求完成验证。

### 1. `tp=1, ep=1, graph on`

命令形态：

- `CUDA_VISIBLE_DEVICES=6 python -m minisgl --model-path ... --tp 1 --ep 1 --attention-backend mla --moe-backend fused --port 30242`

观察结果：

- 请求返回了正常文本
- 响应：
  - `Hello! How can I assist you today?`

### 2. `tp=2, ep=1, graph on`

命令形态：

- `CUDA_VISIBLE_DEVICES=5,6 python -m minisgl --model-path ... --tp 2 --ep 1 --attention-backend mla --moe-backend fused --port 30241`

观察结果：

- 请求返回了正常文本
- 响应：
  - `Hello! How can I assist you today?`

### 3. `tp=2, ep=2, graph on`

命令形态：

- `CUDA_VISIBLE_DEVICES=5,6 python -m minisgl --model-path ... --tp 2 --ep 2 --attention-backend mla --moe-backend fused --port 30231`

观察结果：

- 请求返回了正常文本
- 响应：
  - `Hello! How can I assist you today?`

## Regression Check

本地测试仍然通过：

- `tests/misc/test_glm4_config.py`
- `tests/misc/test_moe_dispatch.py`

结果：

- `10 passed`

## Remaining Issue

`tp=2, ep=2` 的权重加载速度仍然明显慢于其他已经验证过的启动配置。

本次会话中观察到的粗略对比：

- `tp=2, ep=1`：加载权重大约 `16` 秒
- `tp=2, ep=2`：加载权重大约 `9.5` 分钟

因此，MLA graph 正确性问题看起来已经修复，但 EP 的加载路径仍然需要单独做优化。
