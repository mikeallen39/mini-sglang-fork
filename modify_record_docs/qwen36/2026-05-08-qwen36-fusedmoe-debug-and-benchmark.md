# 2026-05-08 Qwen3.6 FusedMoE 问题记录与测速说明

## 1. 本轮目标

本轮工作的目标有两个：

- 把 `Qwen3.6-35B-A3B` 在 `mini-sglang` 中的 `fused MoE` 路径修到可启动、可输出
- 在当前阶段先测一版 `fused` 相对 `torch` 的推理速度，看是否已经有收益

模型路径：

- `/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`

环境路径：

- `/data/zxz/condaenv/minisgl`

代码仓库：

- `/mnt/42_store/zxz/mini-sglang/mini-sglang-fork`

## 2. 本轮遇到的核心问题

### 2.1 `sgl_kernel` 在当前环境下无法直接导入

最初 `fused MoE` 不是算子结果不对，而是入口就直接崩：

- `python/minisgl/moe/fused.py` 中 `moe_align_block_size(...)` 直接依赖 `sgl_kernel`
- 当前环境安装的 `sgl_kernel` 不能在这台机器上正确加载

具体表现：

- `torch==2.6.0+cu124`
- Python 3.12
- GPU 是 `sm80`（A800）
- 已安装的 `sgl_kernel 0.3.21` 包里只有：
  - `sm90/common_ops.abi3.so`
  - `sm100/common_ops.abi3.so`
- 没有 `sm80` 对应的可用 `common_ops`

而且原来的加载逻辑会把非 `sm90` 卡错误分流到 `sm100`，最终报 ABI / 符号错误。

### 2.2 CUDA 工具链最开始是混搭的

虽然运行目标是 `cu124`，但系统最开始实际看到的是：

- `/usr/bin/nvcc` 仍然是 `11.5.119`
- 头文件和 runtime 路径部分来自 12.4，部分来自系统 11.x

这会导致：

- CMake 识别到的 CUDA 编译器版本与 Torch 运行时不一致
- 很多扩展编译虽然开始了，但后续很容易因 ABI 或 toolkit 混用失败

后面重新强制指定：

- `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc`
- `CUDAToolkit_ROOT=/usr/local/cuda-12.4`

才把构建环境收敛干净。

### 2.3 原始 `sgl-kernel` 构建范围过大

原始 `sgl-kernel` 的 `common_ops` 构建会无条件拉很多第三方依赖，包括：

- `flashinfer`
- `flash-attention`
- `mscclpp`

但当前 `mini-sglang` 的 `fused MoE` 入口真正急需的只有：

- `topk_softmax`
- `topk_sigmoid`
- `moe_align_block_size`

如果继续硬编整包，会被无关依赖阻塞，推进效率很低。

### 2.4 `topk > 1` 时 `fused MoE` 仍然有明显数值偏差

在修通 `sgl_kernel` 入口之后，最小对拍结果变成：

- `topk=1`：`fused` 与 `torch` 完全一致
- `topk>1`：已经能跑，但和 `torch` 参考实现仍然存在明显偏差

这说明当前问题已经从“起不来”收敛为：

- `fused experts` 多专家合并路径仍有实现细节需要继续修

## 3. 本轮采取的解决办法

### 3.1 在 `mini-sglang-fork` 内 vendoring 一份 `sgl-kernel`

按当前需求，直接把 `sgl-kernel` 复制到仓库内：

- `third_party/sgl-kernel`

这样后续所有适配都能和主仓一起记录、提交、push。

### 3.2 给 `python/minisgl/moe/fused.py` 增加 torch fallback

修改文件：

- `python/minisgl/moe/fused.py`

主要改动：

- `topk_softmax` 导入失败时自动退回 `torch.softmax + torch.topk`
- `moe_align_block_size` 导入失败时退回新增的 `_moe_align_block_size_torch(...)`

这一步的意义是：

- 即使 `sgl_kernel` 暂时不可用，`fused MoE` 入口也不会直接死掉
- 可以先把问题往后推进到 Triton 主 kernel 和真实前向阶段

### 3.3 给 vendored `sgl-kernel` 增加最小构建模式

修改文件：

- `third_party/sgl-kernel/CMakeLists.txt`
- `third_party/sgl-kernel/csrc/common_extension_moe_only.cc`
- `third_party/sgl-kernel/python/sgl_kernel/__init__.py`
- `third_party/sgl-kernel/python/sgl_kernel/load_utils.py`

主要思路：

- 增加 `SGL_KERNEL_BUILD_MOE_ROUTING_ONLY=ON`
- 在这个模式下只编：
  - `moe_align_kernel.cu`
  - `moe_topk_softmax_kernels.cu`
  - `moe_topk_sigmoid_kernels.cu`
- 新建最小注册入口 `common_extension_moe_only.cc`
- Python 初始化层支持“只加载最小 MoE routing 算子”

这样就避开了：

- `flashinfer`
- `flash-attention`
- `mscclpp`

这些当前阶段不需要、但会严重拖慢或阻塞构建的依赖。

### 3.4 用 CUDA 12.4 工具链重编并替换环境里的 `sgl_kernel`

最终成功构建并安装的是：

- `sglang_kernel-0.4.2.post1-cp310-abi3-linux_x86_64.whl`

安装后验证通过：

- `import sgl_kernel`
- `sgl_kernel.topk_softmax(...)`
- `sgl_kernel.moe_align_block_size(...)`

都能在 `A800(sm80)` 上实际运行。

### 3.5 重新验证 `fused MoE` 真实服务

最终可用启动命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python \
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
/data/zxz/condaenv/minisgl/bin/python -m minisgl \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --tp-size 1 \
  --ep-size 1 \
  --attn fi \
  --linear-attn-backend torch \
  --moe-backend fused \
  --dtype bfloat16 \
  --graph 0 \
  --num-pages 128 \
  --max-running-requests 1 \
  --cache-type naive \
  --disable-pynccl \
  --port 1923
```

实际探针输入：

- `hello`

实际输出：

- `Hello! How can I help you today?`

这说明本轮目标里的“`fused MoE` 路径可启动、可正常输出”已经达成。

## 4. 当前阶段结论

### 4.1 已经解决的问题

- `sgl_kernel` 在当前环境无法导入的问题已经绕开并修通
- `fused MoE` 不再在 routing 入口直接崩
- `Qwen3.6-35B-A3B` 已经能用 `--moe-backend fused` 成功启动服务
- 简单输入已经能返回正常文本

### 4.2 仍然保留的问题

- `topk=1` 时对拍完全正确
- `topk>1` 时与 `TorchMoe` 仍有明显数值偏差

所以当前结论应当表述为：

- `fused MoE` 的服务级跑通已经完成
- 但 kernel 数值一致性仍未完全收敛

这也意味着后续测速结果要谨慎解释：

- 如果性能提升明显，这说明方向是有效的
- 但在继续做更激进的性能结论前，仍然需要把 `topk>1` 的实现细节继续收紧

## 5. 本轮测速说明

本轮测速的目标不是给出最终结论，而是先回答一个更直接的问题：

- 目前这个 `fused MoE` 版本，相比 `torch MoE`，推理速度有没有提高

测速时应尽量保持：

- 同一张卡
- 同一模型
- 同一输入长度
- 同一输出长度
- 同一 `linear_attn_backend=torch`

只切换：

- `--moe-backend torch`
- `--moe-backend fused`

后续如果继续优化：

- `fused MoE` 第二段 GEMM / reduce 路径
- `linear attention` 从 `torch` 切到优化 kernel

则都应该继续沿用这套基线做增量对比。

## 6. 当前测速结果

### 6.1 对照基线

`torch MoE` 的现有基线见：

- `modify_record_docs/qwen36/2026-05-08-qwen36-stage1-performance-baseline.md`

其中可直接对照的关键数字是：

- `warmup`：`TTFT=2466.60ms`，`E2E=3.31s`
- `short_prefill_short_decode (64/32, repeats=3)`：
  - `TTFT avg=2858.19ms`
  - `E2E avg=21.71s`
  - `output_tps=0.97 tok/s`
- `medium_prefill_short_decode (256/32, probe)`：
  - `time_total≈29.70s`

### 6.2 fused MoE 当前已拿到的实测结果

测试服务：

- `http://127.0.0.1:1923/v1`

启动参数：

- `linear_attn_backend=torch`
- `moe_backend=fused`

已完成的短 case 探针结果：

- `warmup`
  - `TTFT=257.33ms`
  - `E2E=1.29s`
  - `output_tokens=16`

- `short_prefill_short_decode (64/32, 单次探针)`
  - `TTFT=464.62ms`
  - `E2E=2.63s`
  - `output_tokens=32`

### 6.3 当前结论

即使只看目前已经稳定拿到的 `warmup + 64/32` 探针，`fused MoE` 也已经明显快于现有 `torch MoE` 基线。

粗略对比：

- `warmup`：
  - `torch`：`TTFT 2466.60ms`，`E2E 3.31s`
  - `fused`：`TTFT 257.33ms`，`E2E 1.29s`

- `64/32 short case`
  - `torch`：`TTFT avg 2858.19ms`，`E2E avg 21.71s`
  - `fused`：`TTFT 464.62ms`，`E2E 2.63s`

从这个量级上看：

- `TTFT` 已经不是小幅下降，而是数量级下降
- `E2E` 也从二十多秒降到了两秒多

因此对“`fused MoE` 推理速度有没有提高”这个问题，当前可以先给出明确回答：

- **有，而且提升非常明显**

### 6.4 结果解释

这里的性能收益并不意味着所有 kernel 都已经完全正确、完全收敛。

当前仍要同时记住两点：

- 服务级别：`fused MoE` 已经能起服务、能正常输出、并且短 case 速度显著更快
- 内核级别：`topk>1` 的数值一致性问题仍然存在，后续还需要继续修

因此现阶段最准确的表述应当是：

- `fused MoE` 已经展现出明显性能收益
- 但还需要继续把多专家合并路径收紧，确保性能和数值正确性同时成立

## 7. linear attention 切换到 sglang 的结果

### 7.1 切换方式

当前仓库里 `Qwen3.6` 的 linear attention backend 支持：

- `torch`
- `sglang`
- `auto`

当前环境中：

- `has_sglang_linear_attn_kernel() == True`

因此切换方式非常直接，只需要把启动参数从：

```bash
--linear-attn-backend torch
```

改成：

```bash
--linear-attn-backend sglang
```

本次实际验证命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python \
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
/data/zxz/condaenv/minisgl/bin/python -m minisgl \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --tp-size 1 \
  --ep-size 1 \
  --attn fi \
  --linear-attn-backend sglang \
  --moe-backend fused \
  --dtype bfloat16 \
  --graph 0 \
  --num-pages 128 \
  --max-running-requests 1 \
  --cache-type naive \
  --disable-pynccl \
  --port 1924
```

### 7.2 功能验证

服务已成功启动并监听：

- `127.0.0.1:1924`

简单探针输入：

- `hello`

实际输出：

- `Hello! How can I help you today?`

因此可以确认：

- `linear-attn-backend=sglang` 在当前环境下是可用的
- 它可以和当前的 `fused MoE` 路径一起正常工作

### 7.3 短 case 性能结果

测试配置：

- `linear-attn-backend=sglang`
- `moe-backend=fused`
- 服务地址：`http://127.0.0.1:1924/v1`

本次测到的短 case 结果如下：

- `warmup`
  - `TTFT=143.32ms`
  - `E2E=1.04s`
  - `output_tokens=16`

- `short_prefill_short_decode (64/32)`
  - `TTFT=134.88ms`
  - `E2E=2.05s`
  - `output_tokens=32`

### 7.4 与上一版 fused 结果对比

上一版对比对象是：

- `linear-attn-backend=torch`
- `moe-backend=fused`
- 服务地址：`http://127.0.0.1:1923/v1`

上一版结果：

- `warmup`
  - `TTFT=257.33ms`
  - `E2E=1.29s`

- `short_prefill_short_decode (64/32)`
  - `TTFT=464.62ms`
  - `E2E=2.63s`

本轮切到 `sglang linear attention` 之后：

- `warmup`
  - `TTFT`: `257.33ms -> 143.32ms`
  - `E2E`: `1.29s -> 1.04s`

- `64/32 short case`
  - `TTFT`: `464.62ms -> 134.88ms`
  - `E2E`: `2.63s -> 2.05s`

### 7.5 当前结论

在当前 `Qwen3.6 + fused MoE` 路径上，再把 linear attention backend 从 `torch` 切到 `sglang` 后：

- 服务仍然可以正常输出
- 性能还能继续提升
- 提升最明显的是 `TTFT`

因此当前阶段最优的已验证组合可以更新为：

- `linear-attn-backend=sglang`
- `moe-backend=fused`

后续如果继续做性能优化，建议都以这一组合作为新的主测试配置。
