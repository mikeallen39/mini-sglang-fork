# 2026-03-25 EP Phase A Progress

## Scope

本文记录了 `mini-sglang` 中 EP 规划的第一步实现。

这一阶段还没有实现 expert dispatch。
它只加入了最小化的配置和分布式状态传递所需的基础设施，
这样后续的 EP 工作就不会继续绑定在仅支持 TP 的假设上。

## Changes

- 在 server args 中新增了 `--expert-parallel-size` / `--ep-size`。
- 在 engine/server config 中显式加入了 `ep_info`。
- 在 `python/minisgl/distributed/info.py` 中新增了 EP 状态辅助函数：
  - `set_ep_info`
  - `get_ep_info`
  - `try_get_ep_info`
  - `build_ep_info`
- 将 `ep_info` 贯穿传递到以下路径：
  - 参数解析
  - 多进程启动
  - engine 初始化
  - 离线 `LLM` 入口

## Current Validation Rules

Phase A 有意将支持范围保持得很小：

- `ep_size >= 1`
- `ep_size in {1, tp_size}`
- `ep_size > 1` 只允许用于 MoE 模型
- routed expert 的数量必须能被 `ep_size` 整除

## Why This Shape

当前的 `mini-sglang` 里，TP 仍然是唯一真正的分布式执行维度。
因此，正确的第一步是先让 EP 在配置和状态中显式存在，同时仍然清晰地拒绝
不受支持的布局。

这样可以避免未来的 EP 工作继续把更多仅适用于 TP 的假设编码进以下位置：

- engine init
- worker launch
- model construction
- weight loading entry points

## Not Included Yet

- 区别于 TP group 的独立 EP process groups
- `all_to_all` 等 EP 通信原语
- 支持 EP 的 MoE dispatcher
- 仅加载本地 expert 权重
- 运行时 topk 的 global-to-local expert 重映射

## Phase B Follow-up

在完成 Phase A 的配置打通后，下一步实现了受约束路径
`ep_size == tp_size`。

这仍然是一条以正确性优先的 EP 路径，而不是最终基于 dispatcher 的设计。

### Added

- 本地 expert 归属辅助逻辑：
  - 每个 EP rank 拥有连续的 global expert 区间
- 本地 MoE TP 视图：
  - `moe_tp_size = tp_size / ep_size`
  - 对于 `ep_size == tp_size`，本地 experts 不再额外进行 TP 切分
- 支持 EP 的 MoE 执行改动：
  - routed expert 权重只为本地 experts 分配
  - 在本地 expert 计算前，会先把 global expert id 重映射为本地 expert slot
  - 现有的输出 `all_reduce` 仍然保留，作为正确性优先的合并路径

### Weight Loading Speed Fixes

第一版 EP loader 在功能上是正确的，但速度太慢，因为它仍然会先从
`safetensors` 加载远端 expert tensor，然后再将其丢弃。

这个路径通过以下方式修复：

- 在 `f.get_tensor(...)` 之前先过滤掉非本地 expert tensor
- 只堆叠本地 routed experts
- 对 expert 权重切分使用 `moe_tp`，而不是 dense-layer 的 `tp`

这对正确性和快速验证循环都必不可少。

### CUDA Graph Limitation

当前这个受约束的 EP 路径还不具备 CUDA-graph-safe 特性。

原因：

- GLM 的 routed-expert 路径在本地 expert 选择时，仍然使用了 `torch.where`
  这类动态 PyTorch 操作
- 这些操作会在 stream capture 下失败

当前的临时行为：

- 当 `ep_size > 1` 时自动禁用 CUDA graph

这样可以在避免 capture 失败的同时，保持推理可用。

## Validation

### Lightweight tests

已执行：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... python -m pytest -o addopts='' -q tests/misc/test_glm4_config.py`

结果：

- `5 passed`

这些测试覆盖了：

- GLM MLA 自动检测行为未发生变化
- EP 配置解析
- 在 dense model 上拒绝 EP
- 为 EP 自动禁用 CUDA graph

### Real GPU validation

模型：

- `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`

已验证配置：

- `tp=2`
- `ep=2`
- attention backend `mla`
- MoE backend `fused`

修复后的观察结果：

- 模型加载成功
- 在禁用 CUDA graph 的情况下服务启动成功
- 使用 prompt `hi` 发送 OpenAI-compatible chat request，返回：
  - `Hello! How can I assist you today?`

这确认了在当前以正确性优先的设计下，`fusedmoe + ep_size == tp_size`
已经可以工作。

### Notes On Loading Time

在已验证的 `tp=2, ep=2` 路径上，加载时间已从之前回归出的多分钟降低到大约：

- 约 20-90 秒，具体取决于选中的空闲 GPU 对以及它们的瞬时负载

被消除的关键性能回归是：

- 从 checkpoint 中读取远端 expert tensor，等到 GPU 实体化之后才将其丢弃

## Dispatcher Refactor Follow-up

在正确性优先的 EP 路径验证完成后，本地 expert 重映射逻辑被重构为一个共享的
最小 dispatcher/mapping 工具：

- `python/minisgl/moe/dispatch.py`

当前目的：

- 统一 `global expert id -> local expert slot` 的重映射逻辑
- 让 GLM 自定义 MoE 路径与通用 fused/torch MoE backend 保持一致语义
- 减少模型代码中重复出现的 EP 专用逻辑

已更新的调用点：

- `python/minisgl/moe/fused.py`
- `python/minisgl/moe/torch_backend.py`
- `python/minisgl/models/glm4_moe_lite.py`

验证：

- 新增了 `tests/misc/test_moe_dispatch.py`
- 重新运行了：
  - `tests/misc/test_glm4_config.py`
  - `tests/misc/test_moe_dispatch.py`
- 在真实 GPU 上重复进行了回归检查，配置为：
  - `tp=2`
  - `ep=2`
  - `attention_backend=mla`
  - 使用 prompt `hi` 的 OpenAI-style HTTP request
- 结果仍然正确：
  - `Hello! How can I assist you today?`

## Divisor EP Follow-up

随后，EP 分组逻辑从：

- `ep_size in {1, tp_size}`

扩展为：

- `ep_size` 可以是 `tp_size` 的任意正因子

当前分组策略是连续分段：

- `ep_rank = tp_rank // moe_tp_size`
- `moe_tp_size = tp_size / ep_size`
- `moe_tp_rank = tp_rank % moe_tp_size`

这使得当前以正确性优先的 EP 路径可以兼容：

- 本地 expert 归属
- 仅本地 expert 加载
- 按 `moe_tp` 进行 expert 权重切分
- 将全 TP 的 `all_reduce` 作为当前的合并路径

### Validation

本地验证：

- 在 `tests/misc/test_glm4_config.py` 中新增了 divisor EP grouping 的单元覆盖
- 重新运行了：
  - `tests/misc/test_glm4_config.py`
  - `tests/misc/test_moe_dispatch.py`
- 结果：
  - `9 passed`

真实 GPU 验证状态：

- 尝试了 `tp=4, ep=2`，并使用较小的 `--num-pages` 以容忍显存不均衡
- 启动和分组逻辑被接受
- 加载成功开始
- 但那次运行没有完成端到端生成，因为可用的 4-GPU 集合当时外部负载较重，
  checkpoint 加载速度下降到了每个文件大约 9 秒

因此，divisor EP 当前状态是：

- 代码路径已实现
- 本地测试已通过
- 真实多 GPU 启动已成功进入加载阶段
- 但要完成一次干净的 `tp=4, ep=2` 端到端生成运行，仍需要一个竞争更少的 4-GPU 时间窗口

## 2026-03-26 Fused EP Graph-Safety Follow-up

这一轮工作的重点是在不放弃 fused MoE backend 的前提下，让 GLM routed-expert
路径具备 graph-safe 特性。

### Changes

- 从以下文件中移除了 GLM 专用的动态本地 expert 循环：
  - `python/minisgl/models/glm4_moe_lite.py`
- 将 GLM 的 MoE 执行接入共享的 MoE backend 路径
- 对以下场景继续启用 CUDA graph：
  - `moe_backend=fused`
  - `ep_size > 1`
- 仅对以下场景继续禁用 CUDA graph：
  - `moe_backend=torch`
  - `ep_size > 1`
- 在以下文件中修改了 EP 本地 expert 重映射语义：
  - `python/minisgl/moe/dispatch.py`
  从正数哨兵 local id 改为：
  - `-1`
- 扩展了 Triton fused MoE kernel 路径，使其将被过滤的 expert block 视为零输出 block，
  而不是索引一个伪造的 expert slot

### What Was Validated

本地测试：

- 重新运行了：
  - `tests/misc/test_glm4_config.py`
  - `tests/misc/test_moe_dispatch.py`
- 当前结果：
  - `10 passed`

单卡上的定向正确性检查：

- fused local MoE 与 torch local MoE：
  - 在非 EP 和 EP 分区两种情况下都一致
- 模拟的 EP split-and-sum 与完整 32-expert 参考实现：
  - 在 torch 和 fused 两条路径上都一致
- 观测到的误差量级：
  - max diff 约为 `1.22e-4`
  - mean diff 约为 `7.9e-6`

这些检查确认，当前 fused EP 本地路由和零过滤 expert 处理，在算子层面与
torch 参考实现保持数值一致。

### Real GPU GLM Validation

已验证的启动配置：

- model:
  - `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`
- `tp=2`
- `ep=2`
- attention backend:
  - `mla`
- MoE backend:
  - `fused`
- 临时路径已移出 `/tmp`：
  - `TRITON_CACHE_DIR=/mnt/42_store/zxz/.triton-cache-mini-sglang`
  - `TMPDIR=/mnt/42_store/zxz/tmp-mini-sglang`

观察到的行为：

- 模型加载成功完成
- CUDA graph capture 成功完成
- scheduler 和 OpenAI-compatible API server 成功进入 ready 状态

这确认此前的阻塞问题已经解决：

- 不具备 graph-safe 特性的 GLM 本地 expert 循环
- 根文件系统已满时的 Triton cache 失败
- graph capture 期间 EP filtered-expert 崩溃

### Remaining Issue

虽然服务器现在可以正确启动并完成 graph capture，但在该配置下，真实生成质量
仍然不能接受。

观察到的症状：

- 使用 `hi` 之类的 OpenAI-style chat request 时，返回的是乱码，而不是正常回复

当前结论：

- 启动正确性现在已经明显改善
- 算子级 EP 正确性检查已通过
- 但对于这个精确配置 `tp=2, ep=2, mla, fused` 的 GLM 路径，真实模型的端到端行为
  仍未被完全验证为正确

因此，这个配置目前应被视为：

- 可以启动
- 可以进行 graph capture
- 但用户可见的生成质量尚未完成行为层面的正确性验证
