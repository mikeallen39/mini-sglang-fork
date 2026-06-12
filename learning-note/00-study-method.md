# 学习方法

## 目标

你的目标不是“把代码看完”，而是建立一张可复述的系统模型。

对这个仓库，最低标准是你可以不看代码，解释清楚：

1. 一个请求从 HTTP 到输出 token 经过哪些进程和消息。
2. `Req`、`Batch`、KV cache、page table 在什么时候变化。
3. prefill 和 decode 的区别是什么。
4. 为什么需要 scheduler，而不是直接 `model.forward()`。
5. 这个项目的性能优化分别插在主路径的哪里。

## 读代码原则

### 1. 按执行路径读，不按目录读

优先顺序：

1. 入口
2. 请求流
3. 核心状态
4. 调度
5. 引擎
6. 存储与缓存
7. 算子与优化
8. 模型适配

不要一开始就陷入这些地方：

- `python/minisgl/models/glm4_moe_lite.py`
- `python/minisgl/kernel/*`
- `python/minisgl/attention/mla_backend.py`

这些文件很重要，但不适合当第一层理解入口。

### 2. 每个文件只回答 4 个问题

每次读文件，都只写下面 4 件事：

1. 输入是什么，输出是什么。
2. 模块拥有哪部分状态。
3. 它要求上游保证什么。
4. 它向下游保证什么。

这比写“函数解释”更有用。

### 3. 区分主干机制和变体机制

对你当前这个仓库，优先区分：

- 主干机制：server -> tokenizer -> scheduler -> engine -> detokenizer
- 资源机制：table/page/KV cache/free
- 优化机制：radix cache / chunked prefill / overlap scheduling / cuda graph
- 变体机制：GLM、MoE、MLA、quantization、EP

如果主干没懂，就不要深挖变体。

### 4. 测试是理解边界最快的入口

优先看这些测试：

- `tests/core/test_scheduler.py`
- `tests/core/test_cache_allocate.py`
- `tests/misc/test_glm4_config.py`
- `tests/misc/test_moe_dispatch.py`

测试告诉你作者认为什么不能坏。

## 记笔记方法

建议你的每篇笔记至少包含下面四段：

### 1. 一句话结论

例子：

`Scheduler` 的职责是把外部请求变成可执行 batch，并在资源约束下驱动 prefill/decode 循环。

### 2. 状态表

例子：

| 名称 | 所在模块 | 创建位置 | 更新位置 | 销毁位置 | 备注 |
|---|---|---|---|---|---|
| `Req.input_ids` | `core.py` | 收到用户请求时 | decode 追加 token 时 | 请求结束时随对象释放 | CPU tensor |
| `page_table` | `engine.py` | engine 初始化 | scheduler 分配页时 | engine shutdown | GPU tensor |

### 3. 时序图或流程图

至少能画出：

- 用户请求时序
- prefill/decode 循环时序
- abort 时序

### 4. 不变量

例子：

- `cached_len <= device_len <= max_device_len`
- page 分配后不能重叠
- batch forward 完成后，每个 req 的 `device_len` 增加 1

## 实验方法

不要只读。每学完一个模块，就做最小实验。

### 推荐实验

1. 在 scheduler 主循环打日志，观察 prefill/decode 的节奏。
2. 关闭 overlap scheduling，看控制流怎么变化。
3. 切换 `--cache naive` 和默认 cache，比较路径差异。
4. 修改 page size，观察哪些模块必须一起成立。
5. 给关键不变量加断言，验证自己的理解。

### 实验记录格式

每次实验必须记录：

- 改了什么
- 预期会发生什么
- 实际发生了什么
- 哪个原先理解是错的

如果不记录，实验很快就变成随机试错。

## 判断自己是否真的理解

满足下面几点，才算进入“理解”而不是“见过”：

1. 你可以在白纸上画出系统请求路径。
2. 你可以口头解释 `Req` 和 `Batch` 的生命周期。
3. 你可以说明 scheduler 为什么需要 `CacheManager` 和 `TableManager`。
4. 你可以指出一个优化具体插在主循环的哪一步。
5. 你可以解释你之前做的适配，改坏了哪个不变量，或者新增了哪个分支。

## 最后提醒

你现在最容易浪费时间的方式是：

- 一直做适配
- 一直修小 bug
- 一直读新文件
- 但从来没系统复述过主路径

所以要求自己：每两天必须产出一篇“可独立阅读”的总结，不然就是输入太多、建模太少。
