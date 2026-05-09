# 2026-05-08 最小 Int8 CUDA 扩展接入记录

## 1. 本轮目标

本轮不再继续强依赖外部完整 `sgl-kernel`，而是在 `mini-sglang` 仓内直接落一个最小可用的 `int8 CUDA` 扩展，优先解决：

- `w8a8_int8` 路径不再只靠 Python shim
- dense linear 的主路径尽量少走 Python 侧后处理
- 服务级推理能够保持正常语义输出

## 2. 本轮新增内容

新增文件：

- `python/minisgl/quantization/int8_cuda_ext.py`
- `python/minisgl/quantization/csrc/int8_scaled_mm.cpp`
- `python/minisgl/quantization/csrc/int8_scaled_mm_kernel.cu`

修改文件：

- `python/minisgl/quantization/__init__.py`

## 3. 实现思路

这次没有直接自己手写完整 `cuBLASLt / CUTLASS int8 GEMM`，而是先采用更小、更稳的方案：

1. GEMM 本体继续复用 `ATen::_int_mm`
2. 仓内自编译一个 CUDA epilogue
3. 在 epilogue 内完成：
   - `int32 accumulator`
   - `* per-token scale`
   - `* per-channel scale`
   - `+ bias`
   - `cast -> bf16/fp16`

这样做的好处是：

- 不依赖外部 `sgl-kernel` 的完整 ABI
- 不需要一次性引入完整 CUTLASS 构建链
- 可以先把最小 `int8 CUDA` 路径稳定接进当前仓

## 4. 接入策略

`apply_w8a8_int8_linear(...)` 现在的优先级变成：

1. 仓内最小 `int8_cuda_ext`
2. 外部 `sgl_kernel.int8_scaled_mm`
3. `torch._int_mm`
4. `float32 matmul` 保守 fallback

同时补上了旧 fallback 的 shape 判断，避免：

- `N=1`
- `N` 不是 8 的倍数

这类场景错误走进 `torch._int_mm`

## 5. 本轮遇到的问题

### 5.1 首次编译时报 `Ninja is required to load C++ extensions`

不是环境里完全没有 `ninja`，而是当前 shell 的 `PATH` 没带：

- `/data/zxz/condaenv/minisgl/bin`

显式补上 `PATH` 后，扩展可以正常编译。

### 5.2 小维度层会触发 `_int_mm` shape 限制

典型报错：

```text
mat2.size(1) needs to be greater than 0 and a multiple of 8, but got 1
```

原因是某些层输出维度 `N=1`，不满足 `_int_mm` 的约束。

解决方式：

- 在 `apply_w8a8_int8_linear(...)` 里补齐 `_int_mm` shape 检查
- 不满足时回退到 `float32 matmul`

### 5.3 单测正确，但服务级输出最初出现乱码

最初小矩阵对拍是正确的，但服务里 `hello` 输出变成了乱码。

根因不是量化公式错，而是 CUDA stream 用错了：

- `torch._int_mm` 在当前计算流上运行
- 自定义 epilogue 最初误用了 `default stream`

这样在服务级多流场景下，epilogue 可能会提前读取到尚未完成的 accumulator，导致语义输出错误。

最终修复：

- 将 epilogue kernel 从 `getDefaultCUDAStream(...)` 改为 `getCurrentCUDAStream(...)`

## 6. 数值验证结果

已对以下 shape 做过对拍，结果都是：

- `max_abs_diff = 0.0`

测试 shape 包括：

- `(1, 128) x (128, 256)`，`bf16`
- `(8, 128) x (128, 256)`，`bf16`
- `(32, 4096) x (4096, 4096)`，`bf16`
- `(64, 4096) x (4096, 14336)`，`bf16`
- `(64, 3584) x (3584, 1)`，`bf16`
- `(16, 128) x (128, 256)`，`fp16`

## 7. 服务级验证结果

验证命令使用：

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
  --moe-backend torch \
  --quantization w8a8_int8 \
  --dtype bfloat16 \
  --graph 0 \
  --num-pages 128 \
  --max-running-requests 1 \
  --cache-type naive \
  --disable-pynccl \
  --port 1933
```

服务级探针输入：

- `hello`

最终输出恢复正常：

- `Hello! How can I help you today?`

## 8. 当前结论

本轮“最小 int8 CUDA 扩展”已经完成第一版可用接入：

- 扩展能编译
- 数值对拍正确
- `N=1` 等边界 shape 已处理
- 服务级输出已恢复正常

这说明当前仓内已经有了一条不依赖完整外部 `sgl-kernel` 的最小 `int8 CUDA` 路径。

## 9. 下一步建议

后续如果要继续追性能，建议按下面顺序推进：

1. 先测这一版最小扩展相对旧 Python shim 的吞吐变化
2. 再决定是否继续把 GEMM 本体从 `ATen::_int_mm` 换成：
   - `cuBLASLt int8`
   - 或最小 CUTLASS int8 kernel
3. 再继续看 MoE / linear attention 的优化 kernel 接入

当前这版已经适合作为后续更完整 `int8 kernel` 优化的中间基线。
