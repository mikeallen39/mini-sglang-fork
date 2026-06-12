# 14 天学习清单

这份计划按每天 1-2 小时设计。目标不是覆盖全部代码，而是优先建立系统模型。

## Day 1: 系统总览

阅读：

- `README.md`
- `docs/structures.md`
- `docs/features.md`

输出：

- `notes/系统总览.md`

你要回答：

- 这个系统里有哪些进程
- 用户请求如何流动
- 哪些是功能，哪些是优化

完成标准：

- 你能手画一张系统拓扑图

## Day 2: 入口与进程拉起

阅读：

- `python/minisgl/__main__.py`
- `python/minisgl/server/launch.py`
- `python/minisgl/server/args.py`

输出：

- `notes/启动流程.md`

你要回答：

- 命令行参数如何变成 `ServerArgs`
- scheduler/tokenizer/detokenizer 是如何启动的
- TP/EP 信息在什么时候注入

完成标准：

- 你能解释为什么这里只是“拉进程”，真正逻辑不在入口

## Day 3: API 与前端消息

阅读：

- `python/minisgl/server/api_server.py`
- `python/minisgl/message/frontend.py`
- `python/minisgl/message/tokenizer.py`
- `python/minisgl/message/utils.py`

输出：

- `notes/api-and-frontend-messages.md`
- `notes/请求主路径.md` 初稿

你要回答：

- HTTP 请求如何变成内部消息
- streaming 是怎么组织的
- `uid` 在系统里起什么作用

完成标准：

- 你能解释 abort/disconnect 的处理路径

## Day 4: 核心状态对象

阅读：

- `python/minisgl/core.py`

输出：

- `notes/core-req-batch-context.md`

你要回答：

- `Req` 代表什么
- `Batch` 什么时候形成
- `Context` 为什么需要全局化

完成标准：

- 你能复述 `Req` 关键字段和生命周期

## Day 5: Scheduler 主循环

阅读：

- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/scheduler/io.py`
- `python/minisgl/scheduler/config.py`

输出：

- `notes/scheduler-main-loop.md`

你要回答：

- scheduler 怎么接收消息
- 消息如何变成请求队列
- main loop 的一步步顺序是什么

完成标准：

- 你能用自己的话解释 `normal_loop` 和 `overlap_loop`

## Day 6: Prefill / Decode / Table

阅读：

- `python/minisgl/scheduler/prefill.py`
- `python/minisgl/scheduler/decode.py`
- `python/minisgl/scheduler/table.py`
- `python/minisgl/scheduler/utils.py`

输出：

- `notes/prefill-vs-decode.md`
- `notes/table-manager.md`

你要回答：

- prefill 和 decode 调度条件有什么差异
- table index 解决了什么问题
- runnable 的定义是什么

完成标准：

- 你能画出请求从 prefill 进入 decode 的状态迁移

## Day 7: Cache 管理

阅读：

- `python/minisgl/scheduler/cache.py`
- `python/minisgl/kvcache/base.py`
- `python/minisgl/kvcache/naive_cache.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/kvcache/naive_manager.py`
- `python/minisgl/kvcache/radix_manager.py`

测试：

- `tests/core/test_cache_allocate.py`

输出：

- `notes/cache-manager.md`
- `notes/radix-cache.md`

你要回答：

- page 是什么，token slot 是什么
- evict 的单位是什么
- radix cache 到底缓存了什么

完成标准：

- 你能解释为什么 cache 管理和 table 管理要分开

## Day 8: Engine 初始化

阅读：

- `python/minisgl/engine/engine.py`
- `python/minisgl/engine/config.py`
- `python/minisgl/engine/graph.py`

输出：

- `notes/engine-init-and-memory.md`

你要回答：

- engine 初始化做了哪几类事
- 模型、KV cache、page table 谁先初始化
- num_pages 如何确定

完成标准：

- 你能解释为什么初始化阶段要同步各 rank 的 free memory

## Day 9: Engine Forward 与采样

阅读：

- `python/minisgl/engine/sample.py`
- `python/minisgl/layers/attention.py`
- `python/minisgl/attention/base.py`

输出：

- `notes/engine-forward.md`
- `notes/sampling.md`

你要回答：

- `forward_batch` 前后哪些对象发生变化
- logits 如何变成 next token
- attention backend 在哪里接入

完成标准：

- 你能说明 scheduler 和 engine 的职责边界

## Day 10: Attention 后端

阅读：

- `python/minisgl/attention/__init__.py`
- `python/minisgl/attention/fa.py`
- `python/minisgl/attention/fi.py`
- `python/minisgl/attention/trtllm.py`
- `python/minisgl/attention/mla_backend.py`

输出：

- `notes/attention-backends.md`

你要回答：

- metadata 是谁准备的
- backend 抽象层隔离了什么
- MLA 路径相比普通 attention 多了什么状态

完成标准：

- 你能说出 attention backend 选择影响哪些模块

## Day 11: 模型与权重加载

阅读：

- `python/minisgl/models/register.py`
- `python/minisgl/models/config.py`
- `python/minisgl/models/weight.py`
- `python/minisgl/models/utils.py`
- `python/minisgl/models/qwen3.py`
- `python/minisgl/models/llama.py`

输出：

- `notes/model-registry-and-weight-loading.md`

你要回答：

- model config 如何驱动不同模型创建
- load weight 和 build model 的职责如何分开
- TP shard 在哪里体现

完成标准：

- 你能独立解释“新增模型适配”最少要改哪些点

## Day 12: Distributed / NCCL / EP

阅读：

- `python/minisgl/distributed/__init__.py`
- `python/minisgl/distributed/impl.py`
- `python/minisgl/distributed/info.py`
- `python/minisgl/kernel/pynccl.py`

输出：

- `notes/distributed.md`

你要回答：

- TP rank 如何参与初始化
- 为什么既有 gloo 也有 nccl / pynccl
- EP 加进来后约束多了什么

完成标准：

- 你能解释单卡和多卡路径的真正差异点

## Day 13: MoE / Quantization / GLM 适配

阅读：

- `python/minisgl/moe/*`
- `python/minisgl/quantization/__init__.py`
- `python/minisgl/models/glm4_moe_lite.py`
- `python/minisgl/models/glm4_moe_lite_hf.py`
- `tests/misc/test_glm4_config.py`
- `tests/misc/test_moe_dispatch.py`

输出：

- `notes/moe.md`
- `notes/quantization.md`
- `notes/glm47-adaptation.md`

你要回答：

- 这是主干逻辑还是模型分支逻辑
- 新增的复杂性落在哪几层
- 这些改动破坏了哪些原先默认假设

完成标准：

- 你能把“适配工作”映射回主干架构

## Day 14: 复盘你之前的改动

材料：

- 你之前做过的适配提交
- 你这 13 天产出的笔记

输出：

- `notes/适配改动复盘.md`

你要回答：

- 每个改动属于哪一层
- 它修改了什么状态流或控制流
- 它依赖哪些新增不变量
- 哪些地方你之前只是“改通了”，但其实没理解

完成标准：

- 你能把过往改动讲成一套工程设计，而不是一堆 patch

## 每天固定动作

每天都做下面三件事：

1. 写一页笔记，哪怕很短。
2. 画一张图，哪怕很粗。
3. 做一个最小实验，哪怕只是加日志。

## 不建议的学习方式

- 直接从 kernel 开始
- 一边读一边无目标改代码
- 看到新 feature 就追进去
- 一口气看很多文件但不写任何产出

## 最终成果标准

14 天后，你应该能独立讲清楚：

1. 系统拓扑
2. 请求主路径
3. 状态生命周期
4. 资源分配模型
5. 优化插入点
6. 适配改动在整体里的位置
