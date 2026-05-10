# 2026-05-10 Qwen3.6 EP=2 权重切分错误修复记录

## 1. 问题现象

本轮排查的目标不是性能，而是先确认 `tp=1, ep=2` 的语义是否正确。

表面现象比较隐蔽：

- 服务可以正常启动
- 请求不再挂住
- 返回文本也不是乱码

但是和 `tp=1, ep=1` 对比后，能看到明显的模型质量退化：

- `hello`
  - `ep=1`: `Hello! How can I help you today?`
  - `ep=2`: `hello`
- `计算 123*45 等于多少？只输出答案。`
  - `ep=1`: `5535`
  - `ep=2`: 曾返回 `2109` / `21345`

这类错误很容易被忽略，因为接口是通的、文本也是正常自然语言，但模型语义实际上已经坏了。

## 2. 先排除的错误方向

### 2.1 不是 scheduler / world-broadcast 主路径错误

此前先修复了一个会导致 `ep=2` 卡住的 world all-reduce 条件问题：

- `python/minisgl/layers/moe.py`
- `python/minisgl/models/glm4_moe_lite.py`

修完后：

- `tp=1, ep=2` 能正常 decode
- `max_tokens=1` 和更大的 `max_tokens` 都能返回

所以“请求挂住”问题已经和当前语义错误分离开了。

### 2.2 不是 fused MoE kernel 独有的问题

为了判断问题是在 EP 本身，还是只在 `fused MoE` kernel，上了一个关键对照：

- `tp=1, ep=1, moe-backend=fused`
- `tp=1, ep=2, moe-backend=torch`

结果发现：

- 即使把 `ep=2` 切成 `torch MoE`
- 输出依然和 `ep=1` 不一致

说明：

1. 问题不只在 `fused MoE` kernel
2. 问题更早，已经发生在 EP 权重装载 / expert 切分这一层

## 3. 数学语义验证：MoE dispatch 本身是对的

为了避免继续靠文本猜测，做了一个更硬的单层数值验证。

验证方法：

1. 取 `Qwen3.6-35B-A3B` 第 `0` 层 MoE block
2. 在 CPU 上拿到 `ep=1` 的完整 routed expert 权重
3. 人工把完整 expert 张量按 expert 维切成两半
4. 分别模拟 `ep=2 rank0` 和 `ep=2 rank1`
5. 用同一组 `hidden_states` / `router_logits` 分别前向
6. 把两半输出求和，与 `ep=1` 的完整输出比较

结论：

- `manual_ep_split_max_abs_diff = 1.86e-9`
- `manual_ep_split_mean_abs_diff = 1.68e-10`

也就是说：

- `TorchMoe + local_expert_start + world sum` 这套数学语义本身是正确的
- EP dispatch / local remap 的理论设计没有问题

这一步很关键，因为它把问题范围从“MoE 数学本身”收窄到了“实际加载到 rank 上的权重不对”。

## 4. 真正根因

根因在 `python/minisgl/models/weight.py`。

### 4.1 Qwen3.6 routed experts 的 checkpoint 形态

Qwen3.6 的 routed expert 权重不是：

- `model.layers.0.mlp.experts.0.gate_up_proj`
- `model.layers.0.mlp.experts.1.gate_up_proj`
- ...

而是整块打包的：

- `model.language_model.layers.0.mlp.experts.gate_up_proj`
- `model.language_model.layers.0.mlp.experts.down_proj`

也就是：

- expert 维已经被 pack 到张量第 `0` 维
- checkpoint key 里没有逐 expert 的 `experts.<idx>` 结构

### 4.2 旧 loader 为什么会错

旧的 EP 过滤逻辑依赖：

- `_stream_get_expert_stack_info(...)`

它只会识别：

- `...experts.0.xxx`
- `...experts.1.xxx`

这种“逐 expert key”格式。

于是对 Qwen3.6 这种“整块 packed expert tensor”：

- `ep=2 rank0` 没有被切半，错误地加载了完整 `256` 个 experts
- `ep=2 rank1` 也同样加载了完整 `256` 个 experts

这会直接导致：

- 两个 EP rank 的本地 expert 集合都不对
- 后续 world all-reduce 后的语义被破坏
- 但服务仍能正常返回文本，因此非常隐蔽

## 5. 修复内容

修复文件：

- `python/minisgl/models/weight.py`

新增逻辑：

1. 识别 packed routed expert key：
   - `_is_packed_routed_expert_key`
2. 对 packed routed expert tensor 做专门分片：
   - `_stream_shard_packed_routed_expert_tensor`

具体策略：

- 先按 `ep_size` 在 expert 维 `dim=0` 上切分
- 再按 `moe_tp_size` 对：
  - `gate_up_proj` 在 `dim=1` 上切分
  - `down_proj` 在 `dim=2` 上切分

这样就和单卡完整权重的物理布局保持一致。

## 6. 修复后的硬验证

### 6.1 权重级验证

修复前：

- `ep=2 rank0` 的 `gate_up_proj` 形状错误地还是 `(256, 1024, 2048)`
- `ep=2 rank1` 也还是 `(256, 1024, 2048)`

修复后：

- `ep=2 rank0`: `(128, 1024, 2048)`
- `ep=2 rank1`: `(128, 1024, 2048)`

并且：

- `cat([rank0, rank1], dim=0) == ep=1 full`
- `max_abs_cat_diff = 0.0`

`down_proj` 也同样完全对齐。

router / shared expert gate 这类 replicated 权重则保持：

- `rank0 == rank1 == ep=1 full`

### 6.2 服务级验证

修复后重新验证了两条 `ep=2` 路径：

1. `tp=1, ep=2, moe-backend=torch`
2. `tp=1, ep=2, moe-backend=fused`

两条路径都恢复了正常输出。

代表性结果：

- `hello`
  - 修复后 `ep=2 + fused`: `Hello! How can I help you today?`
- `计算 123*45 等于多少？只输出答案。`
  - 修复后 `ep=2 + fused`: `5535`

说明：

1. EP 权重切分错误已经被修掉
2. `fused MoE` 路径也随之恢复
3. 之前“不是乱码但质量明显坏掉”的问题，本质就是 routed expert 权重在 `ep=2` 下加载错了

## 7. 当前结论

这次排查最重要的结论有三个：

1. `ep=2` 的语义错误不是 scheduler，也不是单纯 `fused kernel` 的锅
2. 根因是 Qwen3.6 的 packed routed expert tensor 没有按 `ep_size` 做 expert 维切分
3. 修复后，`tp=1, ep=2` 已经回到正常语义区间，且 `torch` / `fused` 两条 MoE 路径都恢复可用

## 8. 下一步建议

在这次修复之后，下一步值得继续做的是：

1. 再跑一轮更系统的一致性测试
   - 固定 `20 ~ 50` 条 prompt
   - 比较 `tp=1, ep=1` vs `tp=1, ep=2`
2. 再回到性能问题
   - 测 `ep=2 + fused MoE + linear attn`
   - 看修复后是否还能保留 EP 带来的吞吐收益
3. 如有必要，再补一个回归测试
   - 专门覆盖 packed routed expert tensor 在 `ep>1` 时的切分行为
