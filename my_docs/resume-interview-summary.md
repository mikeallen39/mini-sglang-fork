# mini-sglang Experiments: Resume and Interview Summary

## 1. 建议你对外主打的经历

如果你要把这段经历写进简历，不建议把所有实验都平铺展开，而是收敛成下面 3 条主线：

1. `Qwen3.6-35B-A3B` 在 `mini-sglang` 上的适配与性能优化
2. `MoE Expert Parallel (EP)` 的接入、权重切分与正确性修复
3. `INT8/W8A8` 量化推理链路、CUDA Graph 与 kernel fusion 优化

这 3 条线既能体现你会“把模型跑通”，也能体现你会“做性能”和“做底层排障”。

## 2. 简历版写法

### 2.1 项目名称写法

你可以选下面任意一种：

- `大模型推理引擎 mini-sglang 适配与性能优化`
- `MoE 推理框架适配与分布式/量化优化`
- `Qwen/GLM 模型推理服务优化与底层 kernel 调优`

### 2.2 简历 bullets

下面这些 bullet 可以直接改写后放进简历。

#### 版本 A：偏工程落地

- 在 `mini-sglang` 中完成 `Qwen3.6-35B-A3B` 与 `GLM-4.7-Flash` 的推理适配，修复模型配置、rotary、RMSNorm、chat template、MoE backend 选择等问题，打通 `text-only` 服务链路并恢复正确输出。
- 设计并落地 `MoE Expert Parallel` 最小正确性方案，新增 `ep_size` 配置、expert ownership/local remap、仅本地 expert 加载与 dispatcher 抽象，完成 `GLM` 和 `Qwen` 路径的 EP 接入与验证。
- 定位并修复 `Qwen3.6` 在 `EP=2` 下的隐蔽语义错误，发现根因是 packed expert tensor 未按 expert 维切分，修复后恢复 `torch/fused` 两条 MoE 路径的正确输出。
- 完成 `Qwen3.6` 的 `fused MoE + sglang linear attention` 优化链路，对比 `torch fallback` 基线将短请求吞吐从 `0.97 tok/s` 提升到 `14+ tok/s`，`TTFT` 从约 `2.8s` 降至约 `135ms`。
- 落地 `W8A8 INT8` 推理路径与最小 `int8 CUDA` 扩展，支持 per-token activation quant、per-channel weight scale、MoE-only quantization 与 fused quant kernel，推动 `int8_moe_only` 吞吐提升到 `13.73 tok/s`，接近 `bf16` 基线 `14.62 tok/s`。
- 修复 `GLM MLA` 与 `Qwen linear attention` 的 `CUDA Graph` 问题，打通 `bs=1/2/4` graph replay 路径；其中 `Qwen3.6` 在 `bs=2` 下开启 graph 后吞吐由 `4.35 tok/s` 提升到 `76.61 tok/s`。

#### 版本 B：偏底层优化

- 负责 `mini-sglang` 中 MoE 推理链路优化，围绕 `fused MoE`、`linear attention`、`CUDA Graph`、`INT8 quantization` 和 `Expert Parallel` 持续做正确性修复与性能优化。
- 通过微基准和在线 benchmark 拆解 `router/expert/w1/stage2/w2/reduce` 各阶段耗时，验证 `fused MoE` 相比 `torch MoE` 的 expert compute 在 `tokens=1/8/64` 下达到 `23.7x/43.1x/51.5x` 总体加速。
- 自研最小 `int8 CUDA epilogue` 扩展并修复多流下的 stream 使用错误，解决“单测正确但服务输出乱码”的并发执行问题。
- 基于真实服务指标而非单点 microbench 做优化取舍，识别“全量 int8 比 bf16 更慢”的根因在 dense linear 和量化前后处理，而不是盲目扩大量化范围，转向 `moe_only + fusion` 路线。

### 2.3 面向不同岗位的简历强调点

- 投递 `推理引擎/系统优化`：重点写 `fused MoE`、`CUDA Graph`、`benchmark`、`kernel fusion`
- 投递 `分布式/MoE`：重点写 `EP`、`local expert sharding`、`dispatch/remap`、`weight loading`
- 投递 `模型部署/应用基础设施`：重点写 `Qwen/GLM 适配`、`服务可用性`、`回归验证`、`性能落地`

## 3. 最适合面试展开的 4 个故事

面试不要从“我改了哪些文件”开始，而要按“问题 -> 判断 -> 方案 -> 验证 -> 结果”讲。

### 故事 1：把 Qwen3.6 从“能启动但输出错误”修到“服务可用”

#### 可直接讲的版本

我当时做的是把 `Qwen3.6-35B-A3B` 适配到 `mini-sglang`。一开始并不是完全跑不起来，而是模型能加载、服务能启动，但输出语义明显不对，比如 `hello` 会返回离题内容，`1+1=?` 会输出重复字符。这说明问题不在服务框架本身，而在模型配置和前向语义路径。

我先把问题拆成几层：配置层看 `rotary`、`partial_rotary_factor`、`mrope`、`norm_topk_prob`；模型层看 `attn_output_gate` 切分和 `RMSNorm` 变体；服务层看 tokenizer 和 chat template。最后修了 `Qwen3.x` 的 rotary 配置读取、补齐 `Gemma` 风格 RMSNorm 变体，并把 `/generate` 和 `/v1/chat/completions` 都接到正确的 chat template 路径上，同时默认关闭 `thinking` 模式，最终把输出恢复成正常文本。

#### 面试官想听到的点

- 你区分了“能跑”和“语义正确”
- 你不是盲调，而是按配置层、模型层、服务层逐层缩小范围
- 你知道 chat 模型和裸 prompt 模型的差别

### 故事 2：为什么 fused MoE 能比 torch MoE 快很多

#### 可直接讲的版本

我做过一轮 `Qwen3.6` 的 `torch MoE` 和 `fused MoE` 对比。先有一版稳定但很慢的基线，单卡短请求吞吐只有 `0.97 tok/s`，`TTFT` 接近 `2.9s`。切到 `fused MoE` 之后，短请求 `TTFT` 降到 `464ms`，再把 linear attention 从 `torch` 切到 `sglang` 后，`TTFT` 进一步降到 `135ms`，吞吐到了 `14 tok/s` 量级。

为了说明为什么快，我专门做了 MoE microbenchmark，把时间拆成 `router`、`expert` 和 `fused` 内部的 `w1/stage2/w2/reduce`。结果发现 `router` 一直只有 `0.08ms` 左右，真正的差距都在 expert compute。`torch` 路径本质上是逐 expert 循环、反复 `gather/scatter` 和小 matmul，碎片化很严重；`fused` 路径把 dispatch、compute、reduce 组织成更连续的数据流，所以总耗时能比 `torch` 低一个数量级。

#### 面试官想听到的点

- 你能把“性能好”解释成结构性原因，不只是“kernel 更快”
- 你会区分在线性能和离线算子性能
- 你知道该怎么用 profile 去证伪“瓶颈在 router”这种直觉

### 故事 3：为什么全量 INT8 反而更慢，我是怎么调整路线的

#### 可直接讲的版本

我一开始也尝试过把 `Qwen3.6` 全量走 `w8a8_int8`，但服务级结果比 `bf16` 更差，`output_tps` 只有 `8.60 tok/s`，而 `bf16 + fused MoE + sglang linear attention` 能到 `14.62 tok/s`。这个时候如果只看“int8 理论上更快”，很容易误判方向。

所以我做了两层定位。第一层是服务级对比，确认问题真实存在。第二层是微基准，发现 decode 小 batch 下普通 dense linear 的 int8 路径比 bf16 慢 `8x~11x`，主要开销在 activation per-token quant 和 `_int_mm + epilogue`。因此我没有继续硬推全量 int8，而是改成 `w8a8_int8_moe_only`，只量化 MoE experts，普通 dense linear 继续保留 bf16。这样吞吐先回到 `12.92 tok/s`，然后再对 MoE 中间量化步骤做 fusion，把吞吐推到 `13.73 tok/s`，已经比较接近 `bf16` 基线。

#### 面试官想听到的点

- 你会根据数据推翻自己的初始方案
- 你知道量化收益依赖 batch、数据流和前后处理，不是上了 int8 就一定快
- 你能把“理论更优”改造成“工程上更有效”的路线

### 故事 4：EP=2 不是挂掉，而是悄悄答错，我怎么定位根因

#### 可直接讲的版本

这是我觉得最有代表性的一个排障案例。`Qwen3.6` 在 `tp=1, ep=2` 下服务能启动、请求也能返回，而且文本看起来也不是乱码，但和 `ep=1` 对比会发现模型质量明显变差，比如简单乘法都能答错。这类问题比直接 crash 更难，因为系统层面看起来一切正常。

我先排除了 scheduler 和 world-broadcast 路径，又把 `fused MoE` 切成 `torch MoE` 做对照，发现错误仍然存在，所以问题不在 fused kernel，而是在更早的 EP 权重装载或 expert 切分。接着我做了一个单层数学验证：拿完整 expert 权重，在 CPU 上人工切成两半，分别模拟两个 EP rank，再把两边输出求和，结果和 `ep=1` 几乎完全一致。这说明 EP dispatch 数学本身没问题，真正的问题是 rank 上实际加载到的权重不对。

最后定位到 `Qwen3.6` 的 routed experts 在 checkpoint 里是 packed tensor，不是 `experts.0.xxx` 这种逐 expert key，老 loader 根本没按 expert 维切分，导致 `ep=2` 的两个 rank 都各自加载了完整 `256` 个 experts。修复成先按 `ep_size` 沿 expert 维切，再按 `moe_tp_size` 沿线性层维度切之后，`ep=2` 的输出恢复正常。

#### 面试官想听到的点

- 你会构造对照实验和单层数值验证
- 你能把“模型质量变差”追到 checkpoint 物理布局和 loader 逻辑
- 你知道分布式正确性问题未必表现为 crash，也可能是 silent correctness bug

## 4. 你可以主动讲的知识点

### 4.1 MoE / EP / TP

- `TP` 主要切 dense/attention 计算；`EP` 主要切 routed experts 的 ownership
- `EP` 不是简单把 rank 数一分，而是要同时处理 `global expert id -> local expert slot` 映射
- `EP` 牵涉配置校验、weight loading、dispatch/remap、local execute、combine/all-reduce 整条链路
- MoE 正确性问题常见在 expert 切分和 remap，不一定在 kernel 本体

### 4.2 fused MoE 为什么快

- 不是 router 快，而是 expert compute 的执行组织方式更好
- 核心是减少逐 expert 循环、小 matmul、gather/scatter 和中间 tensor 读写
- `fused` 的结构优势比单个算子理论 FLOPS 更重要

### 4.3 量化优化的核心判断

- 量化要看服务级收益，不能只看单个 GEMM
- decode 小 batch 下，量化前后处理可能吃掉大部分收益
- “只量化最重、最稀疏的路径”常常比“全量量化”更有工程价值
- 如果要让 int8 真正超过 bf16，关键是 fusion、数据流和访存，不只是 weight int8 化

### 4.4 CUDA Graph 真正的难点

- 难点不只是 capture 成不成功，而是 replay 时状态是否正确
- 动态 state 如果靠 Python 字典按 request 索引，graph replay 往往会绑住 dummy state
- 解决思路通常是预留稳定 buffer、建立 dummy slot、在 replay 前后做状态同步

### 4.5 性能分析方法

- 先有稳定 baseline，再做优化，不然指标没有意义
- 区分在线 benchmark 和离线 microbenchmark
- 把“卡住”分成 GPU kernel 卡死、I/O 阻塞、JIT 首发抖动、分布式同步问题几类
- 你这批实验里就有两个典型例子：
  - `NFS` 权重加载慢，看起来像服务卡死
  - 首次 Triton 编译抖动，看起来像中等 prefill 长期退化

## 5. 面试时建议保守表述的地方

这些点你可以讲，但不要过度包装：

- `GLM EP + fused + graph` 那条路径早期有“能启动但输出乱码”的阶段，后续是靠 `MLA CUDA Graph` 修复把端到端行为补齐的，所以要按阶段说明，不要说成一次性全部稳定。
- `int8` 方向里，你的价值不只是“做出了更快的 int8”，更重要的是“识别出全量 int8 不划算，并把路线收敛到 moe_only + fused quant”。
- `linear attention` 你也做了两次失败优化尝试，这反而是加分项。面试可以直接讲：数值正确不代表性能成立，我用 microbench 和整机 benchmark 把它们回退掉了。

## 6. 如果面试官追问“你到底最能证明什么”

你可以收敛成这一段：

我这段经历最能证明三件事。第一，我能把一个新模型从“能启动但不正确”修到“服务可用”。第二，我不是只会调参，而是会用 profile、对照实验和数值验证定位底层问题，包括分布式权重切分、CUDA Graph state、量化链路和 kernel 路径。第三，我做优化时不会只盯着理论方向，而是会根据服务级指标及时调整路线，比如把全量 int8 收敛到 `moe_only + fusion`，把收益真正落到在线推理上。

## 7. 一句话自我总结

如果你要在自我介绍里压缩成一句话，可以这么说：

`我主要做大模型推理引擎里 MoE 模型的适配、分布式正确性修复和底层性能优化，做过 Qwen/GLM 在 mini-sglang 上的 EP、CUDA Graph、fused MoE 和 INT8 量化链路。`
