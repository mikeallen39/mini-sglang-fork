# 2026-05-09 Qwen3.6 CUDA Graph 进展记录

## 1. 背景

在继续优化 `Qwen3.6-35B-A3B` 推理性能时，发现当前仓里即使传入：

- `--graph 1`

也不一定真的会让：

- `fused MoE`
- `sglang linear attention`
- `Qwen3.6`

这条路径进入 CUDA graph。

本轮目标不是直接把 `bs=2/4/8` 全部做完，而是先回答一个更基础的问题：

- 这条路径到底是“理论上不支持 CUDA graph”，还是“只是被保守逻辑关掉了”

## 2. 初始问题定位

本轮先定位了三层限制：

### 2.1 engine 层显式禁用

`python/minisgl/engine/engine.py`

原先对：

- `quantization == w8a8_int8`
- `linear_attn_backend == "sglang"`

都有直接禁用 `cuda_graph_max_bs` 的逻辑。

其中 `sglang linear attention` 这一层会导致即使用户传入 `--graph 1`，最终也会被改写成：

- `cuda_graph_max_bs = 0`

### 2.2 模型层显式声明不支持

`python/minisgl/models/qwen3_5_moe.py`

原先：

- `Qwen3_5ForCausalLM.supports_cuda_graph = False`

这意味着即使 engine 不再改写 graph 参数，graph capture 也不会真正执行。

### 2.3 更隐蔽的问题：linear attention state 不能直接 replay

真正关键的问题不在“删开关”本身，而在：

- Qwen3.6 的 linear attention runtime state
  - `conv_state`
  - `ssm_state`

当前是按：

- `table_idx`

从 Python 字典 `Qwen3_5LinearStateCache` 动态取出的。

普通 CUDA graph replay 不会重新执行这段 Python 逻辑，所以如果直接强行开 graph，replay 过程中会一直绑定 capture 时那块 dummy request state。

换句话说：

- 真正的问题不是“graph capture 会不会成功”
- 而是“graph replay 后状态会不会写错”

## 3. 本轮实现

本轮先实现一个最小安全版本：

- 先只支持 `decode bs=1`
- 不直接放开多 batch replay
- 先把 `Qwen3.6 + sglang linear attention` 这条主路径打通

### 3.1 BaseModel 增加 graph replay 钩子

新增接口：

- `prepare_for_cuda_graph_replay(...)`
- `finish_cuda_graph_replay(...)`

文件：

- `python/minisgl/models/base.py`

### 3.2 Qwen3.6 linear state 增加 state copy/swap 能力

在：

- `Qwen3_5LinearStateCache`
- `Qwen3_5LinearAttention`

里补了：

- `get_existing(...)`
- `swap_states(...)`
- `copy_state(...)`

文件：

- `python/minisgl/models/qwen3_5_moe.py`

### 3.3 graph replay 前后做 dummy slot 与真实 request slot 的状态同步

在：

- `Qwen3_5ForCausalLM.prepare_for_cuda_graph_replay(...)`
- `Qwen3_5ForCausalLM.finish_cuda_graph_replay(...)`

中加入：

- replay 前：`real_req.table_idx -> dummy_req.table_idx`
- replay 后：`dummy_req.table_idx -> real_req.table_idx`

这保证了当前 `bs=1` 下：

- graph replay 用的是 capture 好的静态 graph
- 但 linear attention state 仍然能和真实 request 正确同步

### 3.4 engine 侧把 `sglang linear attention` 的 graph 限制为 `bs=1`

本轮没有直接放开任意 batch，而是改成：

- `linear_attn_backend == "sglang"`
- 且未显式设更小值时
- 自动把 `cuda_graph_max_bs` 设为 `1`

如果用户传更大的值，目前仍然会被限制。

文件：

- `python/minisgl/engine/engine.py`

### 3.5 修复 `--graph 1` 仍错误 capture `[1, 2, 4]` 的 bug

定位到：

- `python/minisgl/engine/graph.py`

中 `_determine_cuda_graph_bs(...)` 的 batch size 生成逻辑没有按 `cuda_graph_max_bs` 过滤小于等于上界的候选。

修复后：

- `--graph 1`

现在真正只会 capture：

- `[1]`

而不再错误带出：

- `[2, 4]`

## 4. 验证结果

### 4.1 Int8 MoE Only 路径

服务配置：

- `--quantization w8a8_int8_moe_only`
- `--moe-backend fused`
- `--linear-attn-backend sglang`
- `--graph 1`

启动日志关键结果：

- `Start capturing CUDA graphs with sizes: [1]`
- capture 成功
- 服务正常 ready

请求验证：

- 输入：`hello`
- 输出：`Hello! How can I help you today?`

说明：

- `w8a8_int8_moe_only + fused MoE + sglang linear attention`
- 这条路径现在已经可以走 `bs=1` CUDA graph

### 4.2 BF16 路径

服务配置：

- `bf16`
- `--moe-backend fused`
- `--linear-attn-backend sglang`
- `--graph 1`

同样验证到：

- `Start capturing CUDA graphs with sizes: [1]`
- capture 成功
- 服务正常 ready

也就是说：

- `bf16 + fused MoE + sglang linear attention`

这条基线路径也已经被打通到 `bs=1` CUDA graph。

## 5. 当前结论

截至当前，本轮得到的结论很明确：

1. `Qwen3.6 + sglang linear attention` 之前不是“天然不能用 CUDA graph”
2. 它原本主要是被多层保守逻辑关掉了
3. 真正需要修的不是开关本身，而是 linear attention state 的 replay 同步
4. 当前仓内已经成功打通：
   - `bf16 + fused MoE + sglang linear attn + CUDA graph(bs=1)`
   - `w8a8_int8_moe_only + fused MoE + sglang linear attn + CUDA graph(bs=1)`

## 6. 当前限制

本轮仍然有明确边界：

- 当前只安全支持 `decode bs=1`

也就是说：

- `bs=1`：能走 graph
- `bs=2/4/6/8/10`：当前还不能走 graph

原因不是 capture 本身做不到，而是：

- 当前 replay 前后 state copy 逻辑只按单 request 写了安全版本
- 多 request 情况下还需要把 dummy slots 与真实 request slots 的映射扩成多路同步

## 7. 下一步

下一步最合理的推进顺序是：

1. 在当前 `bs=1` 已稳定可用的基础上，做同机 benchmark：
   - `graph off`
   - `graph on`
2. 然后继续把 graph replay state 同步从：
   - `bs=1`
   扩到：
   - `bs=2`
   - `bs=4`
3. 等 `bs=2/4` 跑通后，再决定是否继续扩更大的 batch

当前阶段不建议一口气直接冲 `bs=8/10`，因为：

- 先把 `2/4` 做稳，最容易定位正确性与收益
- 也更符合当前 decode 小 batch 的主要服务场景

## 8. `bs=2/4` 扩展与性能对照

在完成 `bs=1` 版本之后，本轮继续把 graph replay 从单 request 扩到了：

- `bs=2`
- `bs=4`

### 8.1 中间遇到的问题

第一次直接扩到 `bs=4` 时，服务在 capture 阶段报错：

- `IndexError: index 5 is out of bounds for dimension 0 with size 5`

根因是：

- `page_table` 只按 `max_running_req + 1` 预留了一个 dummy slot
- 但 `bs=4` graph capture 需要 4 个独立的 dummy request slot

修复方式：

1. `page_table` / `token_pool` 改成额外预留 `max_graph_bs` 个 graph dummy slots
2. graph capture 时，为每个 batch size 单独创建稳定的 `dummy_reqs`
3. replay 前后按：
   - `real_req[i] <-> dummy_req[i]`
   做 linear attention state 同步

### 8.2 正确性验证

修复后，服务已经能成功：

- `capture [1, 2, 4]`

并发请求输出验证结果：

- `bs=2`
  - `水果 -> 苹果`
  - `动物 -> 猫`
- `bs=4`
  - `水果 -> 苹果`
  - `动物 -> 猫`
  - `城市 -> 北京`
  - `颜色 -> 红色`

当前没有观察到明显的 state 串扰。

### 8.3 BF16 `graph on/off` 对照

测试配置：

- `bf16`
- `--moe-backend fused`
- `--linear-attn-backend sglang`
- `--max-running-requests 4`

#### graph on

服务配置：

- `--graph 4`

结果：

`bs=2`

- `TTFT avg = 338.52 ms`
- `TPOT avg = 16.85 ms`
- `E2E avg = 0.8608 s`
- `Throughput = 76.61 token/s`

`bs=4`

- 当前这轮没有单独留存完整日志数值
- 但已经确认：
  - `capture [1, 2, 4]` 成功
  - `bs=4` 并发请求正确输出

#### graph off

服务配置：

- `--graph 0`

结果：

`bs=2`

- `TTFT avg = 516.24 ms`
- `TPOT avg = 213.14 ms`
- `E2E avg = 7.1236 s`
- `Throughput = 4.35 token/s`

`bs=4`

- `TTFT avg = 798.89 ms`
- `TPOT avg = 268.46 ms`
- `E2E avg = 9.1212 s`
- `Throughput = 3.40 token/s`

### 8.4 当前结论更新

截至当前，可以把结论更新为：

1. `Qwen3.6 + fused MoE + sglang linear attention` 现在已经支持：
   - `bs=1`
   - `bs=2`
   - `bs=4`
   的 CUDA graph
2. `bs=2` 下，开启 CUDA graph 后吞吐从约 `4.35 token/s` 提升到约 `76.61 token/s`
3. 这说明当前 decode 小 batch 场景下，CUDA graph 对这条路径的收益非常大
4. 下一步若继续投入，最值得做的是：
   - 补齐 `bs=4` 的完整对照数值留档
   - 再继续扩到 `bs=8`
