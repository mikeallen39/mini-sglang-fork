# 2026-06-18 Linear Attention Chunk-Scan 映射笔记

## 目标

把当前 `minisgl` 里的 `linear_attn` prefill 递推：

- `state = h \in R^{K \times V}`
- 单 token 更新：
  - `h <- exp(g_t) * h`
  - `v_t_res <- v_t - h^T k_t`
  - `v_t_res <- beta_t * v_t_res`
  - `h <- h + k_t v_t_res^T`
  - `o_t <- h^T q_t`

和 `sglang` / `flash-linear-attention` 的 `chunk_gated_delta_rule` 三段式实现对齐：

1. `chunk_local_cumsum`
2. `chunk_gated_delta_rule_fwd_intra`
3. `chunk_gated_delta_rule_fwd_h`
4. `chunk_fwd_o`

当前结论：**不能直接把 minisgl 的 `(q, k, v, g, beta, state)` 塞进 `chunk_gated_delta_rule` 当成 drop-in backend。**

离线验证显示：

- `sglang chunk` 路线在热点 shape 上明显更快
- 但直接适配后的数值完全不对

所以必须先把数学映射理清。

## minisgl 当前递推

在当前 `minisgl` prefill kernel 中，每步更新是：

```text
h_t^- = exp(g_t) * h_{t-1}
r_t   = v_t - (h_t^-)^T k_t
u_t   = beta_t * r_t
h_t   = h_t^- + k_t u_t^T
o_t   = (h_t)^T q_t
```

这里：

- `h_t` 形状是 `[K, V]`
- `k_t, q_t` 形状是 `[K]`
- `v_t, u_t, o_t` 形状是 `[V]`

也可以写成：

```text
h_t = exp(g_t) * h_{t-1} + k_t u_t^T
u_t = beta_t * (v_t - (exp(g_t) * h_{t-1})^T k_t)
```

关键点：

- `u_t` 依赖更新前的 `h_t^-`
- 所以不能把整段 chunk 简单看成独立 token 的 rank-1 累加

## sglang chunk 路线的中间量

参考文件：

- `sglang/.../fla/chunk.py`
- `sglang/.../fla/chunk_fwd.py`
- `sglang/.../fla/chunk_delta_h.py`
- `sglang/.../fla/chunk_o.py`
- `sglang/.../fla/wy_fast.py`

`sglang` 路线里，chunk 内先构造：

- `A`
- `w`
- `u`
- `h`
- `o`

### 1. `g`

`chunk.py` 里第一步是：

```python
g = chunk_local_cumsum(g, chunk_size=64, ...)
```

所以 `chunk_gated_delta_rule` 后续使用的 `g` **不是 token 级原始 `g_t`**，
而是 **chunk 内前缀累积后的 log-decay**。

这是当前直接适配失败的第一个根因：

- `minisgl` 当前 kernel 用的是每步原始 `g_t`
- `sglang chunk` 后续各步默认拿的是 `g_cumsum`

### 2. `A`

`chunk_gated_delta_rule_fwd_intra` 先构造：

- `beta * K K^T`
- 加上 gate 对不同时刻的衰减修正
- 再做 `(I + A)^{-1}` 的块内求解

所以这里的 `A` 并不是“原始注意力矩阵”，
而是 **编码了 chunk 内 token-to-token 递推依赖的三角解算器中间量**。

它本质上把：

- `u_t` 对前面 token 的依赖

提前编码进一个 chunk 内可解的线性系统。

### 3. `u`

`wy_fast.py` 里：

```python
b_vb = b_v * beta
b_u = A @ b_vb
```

所以 `u` 不是原始 `v`，也不是直接的 `beta * v`，而是：

```text
u = A @ (beta ⊙ v)
```

这和当前 minisgl 里的：

```text
u_t = beta_t * (v_t - h^- k_t)
```

看起来不一样，但实际上：

- `A` 的定义已经把 chunk 内“减去历史 state 投影”的部分折进去了
- 所以 `u` 是一个**经过 chunk 内 solve 后的等价值**

### 4. `w`

`wy_fast.py` 里：

```python
b_kb = b_k * beta * exp(g_cumsum)
b_w = A @ b_kb
```

所以 `w` 是：

```text
w = A @ (beta ⊙ exp(g_cumsum) ⊙ k)
```

在后续 `chunk_gated_delta_rule_fwd_h` 里，
`w` 不是用来直接出输出，而是用来把 chunk 级状态传播下去。

### 5. `h`

`chunk_gated_delta_rule_fwd_h` 里做的是：

- 逐 chunk 状态传播
- 输入是：
  - `k`
  - `w`
  - `u`
  - `g`
  - `initial_state`
- 输出：
  - chunk 级 `h`
  - `v_new`

这个 `h` 不是 token 级最终状态序列，而是：

- 每个 chunk 边界上的状态
- 以及供后续 `o` 回填使用的块级中间量

### 6. `o`

`chunk_fwd_o` 里最后做：

- `q @ h`
- 再加上块内 `q @ k^T @ v_new`

也就是：

```text
o = q * chunk_state_contrib + q k^T * v_new
```

这一步把：

- 跨 chunk 的历史状态贡献
- chunk 内 token-to-token 贡献

合并成最终输出。

## 当前直接适配失败的具体原因

当前失败的“直接把 minisgl 变量塞进 `chunk_gated_delta_rule`”做法，至少有这几处不匹配：

1. `g`
- 直接传的是原始 `g_t`
- 但 `sglang` 后续内部用的是 `chunk_local_cumsum(g)`
- 虽然 API 会自己做 cumsum，但 minisgl 当前 state 递推语义和后面的 `state` 更新并不自动等价

2. `state`
- minisgl 当前 `state` 是 `[HV, K, V]`
- `sglang` 用的是 `[B, H, V, K]`
- 这不只是 shape 转置问题
- 更重要的是：两边“初始状态在 chunk 内如何作用”的数学语义不一定完全一样

3. `v`
- `sglang` 的最终 `u` / `v_new` 不是原始 `v`
- 当前直接塞原始 `v` 进去，再拿输出和 minisgl 当前 kernel 对比，会因为中间等价变换没有对齐而失真

4. `q/k` 归一化和 `scale`
- `sglang chunk` 路线允许 `use_qk_l2norm_in_kernel`
- 当前 minisgl 最优路径已经把 prefill `q/k` norm 挪到 kernel 外
- 这条路径必须保持同口径，不然数值和性能都对不上

## 对第二版实现的启示

第二版如果要在 `minisgl` 里真正做对，应该遵循下面顺序：

1. 不再把 `chunk_gated_delta_rule` 当作完全黑盒替换件
2. 先在本地实现最小的 chunk 内数学中间量：
   - `g_cumsum`
   - `A`
   - `u`
   - `w`
3. 再做 chunk 间状态传播
4. 最后做 `o`

也就是说，正确路线应该是：

- 参考 `sglang` 的 **分解方式**
- 但要按 `minisgl` 当前 `q/k/v/g/beta/state` 的语义重建
- 不能只做 shape 适配

## 实现优先级建议

最小靠谱原型建议：

1. 先只做 **离线 microbench 版**
2. 先在 Python / Torch 里实现：
   - chunk 内 `g_cumsum`
   - `A`
   - `u`
   - `w`
   - chunk 间状态传播
3. 用小 shape 验证和当前串行递推数值一致
4. 数学对齐后，再把最贵部分换成 Triton kernel

不要再直接把现成 `sglang chunk` 作为在线 backend 接回服务。

## 当前结论

- `chunk-scan` 方向值得继续
- `sglang` 的 chunk 路线在热点 shape 上有明显潜力
- 但当前 `minisgl` 和 `sglang` 的数学中间量没有直接对齐
- 下一步应先做**数学正确的本地 chunk 分解原型**，再谈线上替换
