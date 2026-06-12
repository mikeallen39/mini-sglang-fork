# 2026-05-08 Qwen3.6 Stage1 适配记录

## 1. 本阶段目标

本次工作的目标是让 `mini-sglang` 能够初步跑通 `Qwen3.6-35B-A3B`，具体包括：

- 环境安装完成
- 模型能够成功加载
- 服务能够成功启动
- 简单输入能够正常输出

当前模型路径：

- `/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`

当前环境路径：

- conda 环境：`/data/zxz/condaenv/minisgl`

当前阶段结论：

- `stage1` 已完成
- 目前已经可以跑通

## 2. 当前可用启动命令

当前验证通过的启动命令如下：

```bash
export PATH=/usr/local/cuda-12.4/bin:/data/zxz/condaenv/minisgl/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_INSTALL_PATH=/usr/local/cuda-12.4
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CUDA_VISIBLE_DEVICES=7

/data/zxz/condaenv/minisgl/bin/python -m minisgl \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --tp-size 1 \
  --ep-size 1 \
  --attn fi \
  --linear-attn-backend torch \
  --moe-backend torch \
  --dtype bfloat16 \
  --graph 0 \
  --num-pages 128 \
  --max-running-requests 1 \
  --cache-type naive \
  --disable-pynccl \
  --port 1921
```

## 3. 过程中遇到的主要问题

### 3.1 CUDA / Torch 运行环境不一致

一开始环境里 CUDA 工具链和 torch 对应的运行时没有完全对齐。

本次使用的 torch 版本是：

- `torch==2.6.0+cu124`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`

因此运行时需要显式指定 CUDA 12.4 相关路径，否则容易出现编译或运行时行为不一致的问题。

### 3.2 Torch 2.6 与本地扩展 ABI 路径兼容性问题

本地部分扩展路径和 `torch 2.6` 组合下存在兼容性问题，具体表现为某些 FFI / DLPack 相关路径不可用或不稳定。

为保证当前阶段先跑通，运行时需要增加：

```bash
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

这个环境变量在当前环境下是必需的。

### 3.3 本地 JIT kernel 编译失败

在启动过程中，`mini-sglang` 的部分本地 JIT kernel 会触发编译失败，原因主要是：

- 当前编译链对部分 C++20 特性支持不稳定
- 某些本地头文件仍然使用了更激进的语言特性

如果不处理，这类问题会影响启动和推理链路。

### 3.4 MoE 优化后端在当前环境下不可直接用

默认的 MoE 后端选择会走到 fused 路径，但当前环境下对应 kernel / 架构组合不稳定，无法作为本阶段可用方案。

因此当前稳定方案使用：

- `--moe-backend torch`

### 3.5 Qwen3.6 能加载但最初输出语义错误

早期阶段不是“完全跑不起来”，而是模型已经能加载、服务也能起，但输出明显不正常。

典型现象包括：

- 输入 `hello`，输出无关英文段落
- 输入 `1+1=?`，输出重复字符或错误内容

这说明问题不只是服务启动，而是：

- 模型配置适配不完整
- 前向语义路径存在偏差
- prompt / chat template 路径与 Qwen3.6 预期不一致

### 3.6 `/generate` 原始裸 prompt 路径不适合 Qwen3.6

Qwen3.6 是 chat / thinking 模型。

在适配过程中发现：

- `/v1/chat/completions` 在加上 chat template 后更早恢复正常
- `/generate` 原本是把裸文本直接喂进去

对于 Qwen3.6 这种模型，直接走裸 prompt 很容易导致输出跑偏，即便模型主干本身已经基本正确。

## 4. 做过的主要改进

### 4.1 Qwen3.6 / Qwen3.5 模型配置与结构适配

主要修改文件包括：

- `python/minisgl/models/config.py`
- `python/minisgl/layers/rotary.py`
- `python/minisgl/models/qwen3_5_moe.py`

这部分改动主要解决以下问题：

- 保留 `rope_parameters` 中有意义的配置，即使 `rope_type=default`
- 支持 `mrope_section`
- 支持 `mrope_interleaved`
- 读取并使用 `partial_rotary_factor`
- 修正 Qwen3.x 的 rotary 相关布局判断
- 当配置里缺失时，为 Qwen3 MoE 默认补上 `norm_topk_prob=True`
- 修正 full attention 中 `attn_output_gate` 的切分逻辑
- 补齐并启用更符合 Qwen3.6 路径的 Gemma 风格 RMSNorm 变体

### 4.2 为不稳定 kernel 路径增加 torch fallback

主要修改文件包括：

- `python/minisgl/kernel/index.py`
- `python/minisgl/kernel/store.py`
- `python/minisgl/layers/norm.py`
- `python/minisgl/layers/activation.py`
- `python/minisgl/moe/fused.py`

这部分的目标不是性能，而是保证：

- 某些 fused / JIT kernel 不可用时，仍然可以继续推理
- 启动和基础功能不被可选优化路径阻塞

### 4.3 调整 JIT 编译标准

相关修改：

- `python/minisgl/kernel/utils.py`

改动内容：

- 将本地 JIT 编译参数从 `c++20` 调整为 `c++17`

这一步主要是为了降低当前机器环境中的编译失败概率。

### 4.4 Qwen3.6 tokenizer / chat template 路径适配

主要修改文件包括：

- `python/minisgl/message/tokenizer.py`
- `python/minisgl/tokenizer/tokenize.py`
- `python/minisgl/server/api_server.py`

具体改动：

- 为 `TokenizeMsg` 增加 `chat_template_kwargs`
- 将 `chat_template_kwargs` 透传到 `tokenizer.apply_chat_template(...)`
- 对 OpenAI 风格 chat 路径默认设置：

```python
chat_template_kwargs={"enable_thinking": False}
```

这样做的原因是：

- Qwen3.6 默认是 thinking 模式
- 如果不显式控制，返回内容和测试预期可能不一致
- 当前 `stage1` 更适合先验证“能正常回答”

### 4.5 修复 `/generate` 路径，让裸 prompt 也能正常测通

最终为了让最直接的测试方式也可用，对：

- `python/minisgl/server/api_server.py`

做了额外调整。

当前 `/generate` 的默认行为改为：

- 将传入的 `prompt` 自动包装成单轮 user message
- 默认走 chat template
- 默认使用：

```python
{"enable_thinking": False}
```

这样可以保证像 `hello` 这种最简单的 smoke test 也能正常输出，而不是继续走偏。

## 5. 权重加载情况说明

当前模型加载时会看到：

- `Skipped 349 checkpoint tensors without matching module entries`

这并不代表当前文本推理不可用。

本次排查确认后，当前未匹配的部分主要集中在：

- vision 相关分支
- 非当前 text-only stage 必需路径

结论是：

- 文本主干加载已经足够支持 `stage1`
- 当前阶段目标不是多模态全量适配

## 6. 当前已完成的验证

### 6.1 服务已可启动

使用上面的启动命令，服务已经可以正常启动并监听：

- `127.0.0.1:1921`

### 6.2 `/generate` 基础验证已通过

验证结果：

- 输入：`hello`
- 输出：

```text
Hello! How can I help you today?
```

验证结果：

- 输入：`1+1=?`
- 输出：

```text
1 + 1 = 2
```

### 6.3 `/v1/chat/completions` 基础验证已通过

验证结果：

- 输入：`hello`
- 输出：

```text
Hello! How can I help you today?
```

验证结果：

- 输入：`请用中文回答：1+1等于几？`
- 输出：

```text
1+1等于2。
```

验证结果：

- 输入：`介绍一下你自己。`
- 输出已经恢复为正常中文回答，不再是乱码或离题内容

## 7. 当前阶段结论

当前可以明确认为：

- `stage1` 已完成

这里的 `stage1` 指的是：

- 环境装好
- 模型能加载
- 服务能起来
- 简单输入能正常输出

也就是说，目前已经实现“可以跑通”的目标。

## 8. 当前限制

虽然 `stage1` 已完成，但当前仍有一些限制：

- 当前稳定方案使用的是：
  - `--linear-attn-backend torch`
  - `--moe-backend torch`
- 当前优先保证正确性，不是性能最优方案
- 当前验证主要集中在 text-only 基础推理
- vision 分支不是本阶段目标
- 当前可用启动命令中：
  - `--graph 0`

即 CUDA graph 暂时没有作为本阶段验证重点

## 9. 下一阶段建议

如果进入后续 `stage2`，建议优先做以下几件事：

- 继续检查并恢复更高性能的线性注意力后端
  - 例如重新验证 `--linear-attn-backend sglang`
- 继续检查 fused MoE 路径在当前环境中的可用性
- 扩大测试样本，不只停留在简单 smoke test
- 做更长上下文的稳定性验证
- 如果后续需要，再单独做多模态 / vision 侧适配

