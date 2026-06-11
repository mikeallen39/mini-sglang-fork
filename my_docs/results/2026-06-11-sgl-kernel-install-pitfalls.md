# 2026-06-11 sgl_kernel 安装踩坑总结

## 背景

目标是让 `mini-sglang` 不再依赖仓库内的影子包，而是直接使用环境中安装的 `sgl_kernel`，并且只保留 `sm80` 支持。

目标环境：

- Conda 环境：`/mnt/82_store/zxz/condaenv/minisgl`
- PyTorch：`2.6.0+cu124`
- GPU：`sm80`

最终要求：

- 清理仓库内 `third_party`
- 清理仓库内 `python/sgl_kernel`
- 清理环境内旧的 `sgl-kernel` / `sglang-kernel`
- 重新安装一套可用的、只面向 `sm80` 的环境版 `sgl_kernel`

## 为什么这次安装会踩很多坑

### 1. 仓库里有 `python/sgl_kernel` 影子包

这是最先导致判断失真的问题。

表现：

- `import sgl_kernel` 不一定命中环境里的安装包
- 很可能先命中仓库里的本地目录
- 表面上“能 import”，实际并没有使用真正的环境版二进制

影响：

- 很难判断到底是哪一套 `sgl_kernel` 在生效
- 容易误以为环境安装没问题
- benchmark 和运行时行为都可能被污染

结论：

- 这种影子包长期看是危险的
- 最终必须删掉

### 2. 环境里同时混装了多套分发

当时环境里既有：

- `sgl-kernel`
- `sglang-kernel`

而且版本不一致。

影响：

- 都往 `site-packages/sgl_kernel` 这个包名上落文件
- Python 层和 `.so` 可能来自不同版本
- 运行时行为不再可预测

结论：

- 不能做增量修补
- 必须先清干净，再只保留一套来源

### 3. 官方 wheel 和当前 PyTorch ABI 不匹配

这是最核心的环境兼容性坑。

现象：

- 从官方 `cu124` 源安装的 wheel 不能正常加载 `common_ops.abi3.so`
- 表面像是 CUDA 或路径问题
- 进一步排查后发现是 PyTorch C++ ABI 不匹配

根因：

- 当前环境的 `torch 2.6.0+cu124` 是 `ABI=False`
- 官方 wheel 里的一些 `.so` 依赖的是 `CXX11 ABI=True` 风格符号

影响：

- 即使 wheel 安装成功
- `import sgl_kernel` 仍然会在动态链接阶段失败

结论：

- 这类问题不能靠改 `PYTHONPATH` 或 `LD_LIBRARY_PATH` 解决
- 需要换 ABI 匹配的 wheel，或者直接源码重编

### 4. `sgl-kernel` 的源码构建默认会带很多当前不需要的依赖

源码构建后又遇到第二层坑：默认构建面太大。

主要包括：

- `mscclpp`
- `NUMA`
- `flashinfer`
- `flash-attention`
- 额外架构目标如 `sm90`、`sm100`

影响：

- 构建链很长
- 非必要依赖也会变成阻塞点
- 例如 `NUMA` 缺失会直接把构建卡住

结论：

- 如果目标只是恢复 `sm80` 上的 `common_ops` / `flash_ops`
- 就必须主动裁剪构建面

### 5. “名义上的 sm80-only” 不等于“真正的 sm80-only”

这是这次最费时间的技术坑。

表面上已经设置了：

- `SGL_KERNEL_CUDA_ARCH_LIST=80`

但实际 `flash_ops` 构建里仍然混入了：

- `sm90`
- `sm86`
- `sm90a`

更麻烦的是，即使编译命令后来只剩 `sm80`，生成出来的 `flash_ops.abi3.so` 里仍然残留了：

- `run_mha_fwd<90,...>` 的未定义符号引用

根因：

- `flash-attention` 的 `hopper/flash_api.cpp` 和 `static_switch.h` 里还有直接面向 `Arch=90` 的路径
- 只靠外围宏定义并不能完全剔除

结论：

- 需要直接 patch 真正参与构建的 `flash-attention` 源文件
- 不能只在外层 CMake 里传宏然后假设它会完全生效

### 6. 用源码目录直接 import，会误判安装是否成功

在验证阶段还踩了一个容易忽略的坑：

- 如果把 `/mnt/42_store/zxz/aiinfra/sglang/sgl-kernel/python` 放进 `sys.path`
- import 到的是源码目录的 Python 包
- 但这个目录本身不带构建好的 `.so`

表现：

- 看起来像“新编出来的东西还是不行”
- 实际只是 import 目标错了

结论：

- 最终验证必须针对环境中的 `site-packages/sgl_kernel`
- 不能对源码目录做最终结论

## 这次具体修了什么

### 仓库侧清理

清掉了：

- `third_party`
- `python/sgl_kernel`

目的：

- 消除本地影子包和旧 vendored 依赖对运行时的污染

### 环境侧清理

清掉了环境中旧的：

- `sgl-kernel`
- `sglang-kernel`

目的：

- 保证 `site-packages/sgl_kernel` 只有一套来源

### 源码构建侧修复

在 `/mnt/42_store/zxz/aiinfra/sglang/sgl-kernel` 这份源码上做了最小必要修改：

- 让 `common_ops` 可以在关闭 `mscclpp` 的情况下编过
- 关闭不需要的 `expert_specialization`
- 让 `flash_ops` 真正尊重 `sm80-only`
- 禁掉 `sm90` / `fp8` / 部分 diff-head 路径
- 直接 patch `flash-attention` 的 `hopper/static_switch.h` 和 `hopper/flash_api.cpp`

### 环境安装侧修复

最终不是继续依赖仓库内包，而是把新编出来的：

- `common_ops.abi3.so`
- `flash_ops.abi3.so`

明确放回环境中的：

- `/mnt/82_store/zxz/condaenv/minisgl/lib/python3.12/site-packages/sgl_kernel`

## 最终验证结果

最终环境版 `sgl_kernel` 已恢复成功，验证通过：

- `moe_align_block_size`
- `topk_softmax`
- `int8_scaled_mm`
- `flash_attn_with_kvcache`

验证输出是：

```text
sgl_kernel_file /mnt/82_store/zxz/condaenv/minisgl/lib/python3.12/site-packages/sgl_kernel/__init__.py
ops_moe_align_block_size True
ops_topk_softmax True
ops_int8_scaled_mm True
flash_attn_ok True
```

这说明：

- 现在已经不再依赖仓库内 `python/sgl_kernel`
- `mini-sglang` 后续会直接走环境版 `sgl_kernel`

## 这次踩坑的根本原因

如果只总结一句话：

> 不是单点问题，而是“影子包 + 混装分发 + ABI 不匹配 + 过宽构建面 + FA3 的 sm90 残留引用”叠在一起。

也就是说，这次困难不是因为某一个命令写错，而是系统状态长期不干净，导致每一层都在放大排查成本。

## 后续建议

### 1. 不要再把 `sgl_kernel` 以仓库子目录形式保留

否则以后非常容易再次出现：

- import 命中本地影子包
- 环境包失效却不自知

### 2. 环境里只能保留一套 `sgl_kernel` 来源

不要同时混：

- `sgl-kernel`
- `sglang-kernel`
- 本地手工覆盖版本

### 3. 明确记录当前环境约束

至少要固化：

- `torch 2.6.0+cu124`
- `ABI=False`
- `sm80`

否则以后换 wheel 时，很容易重复踩 ABI 坑。

### 4. 以后优先做最小构建

如果目标只是恢复当前机器上的核心路径：

- 先做 `sm80-only`
- 先做 `common_ops` / `flash_ops`
- 不要默认把 `mscclpp`、`NUMA`、`sm90/sm100` 一起拉进来

### 5. 做最终验证时一定验证环境包

优先检查：

- `sgl_kernel.__file__`
- `torch.ops.sgl_kernel.*`
- `from sgl_kernel.flash_attn import flash_attn_with_kvcache`

避免再次被源码目录或影子包误导。
