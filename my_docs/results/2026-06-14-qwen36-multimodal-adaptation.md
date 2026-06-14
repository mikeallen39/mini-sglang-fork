# Qwen3.6-35B-A3B 多模态适配记录

## 目标

让当前仓库中的 `Qwen3.6-35B-A3B` 支持最小可用的图片问答：

- 接受 OpenAI 风格多模态 `messages`
- 正确处理本地图片 / `data:` URL
- 将图片经 processor 编码后送入模型
- 完成单张图片的正常问答推理

本次实现优先保证：

- 正确
- 简洁
- 尽量复用现有 `qwen3_5_moe` 语言主干

没有盲目整包照搬 `/mnt/42_store/zxz/aiinfra/sglang`。

## 最终结果

当前已经能够完成图片问答。

测试图片：

- [tom.jpg](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/tom.jpg)

测试问题：

- `图片中是什么内容？`

返回结果：

- `HTTP 200`
- 返回内容正确识别为《猫和老鼠》（Tom and Jerry）相关图片，并识别出汤姆猫

## 适配内容

### 1. API 接受多模态消息

文件：

- [python/minisgl/server/api_server.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/server/api_server.py)

支持：

- `{"type":"text","text":"..."}`
- `{"type":"image_url","image_url":{"url":"..."}}`

做的事情：

- 解析 OpenAI 风格 `content parts`
- 抽取文本内容
- 抽取图片 URL
- 保持纯文本请求兼容

### 2. tokenizer / processor 接入图片输入

文件：

- [python/minisgl/tokenizer/tokenize.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/tokenizer/tokenize.py)
- [python/minisgl/tokenizer/server.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/tokenizer/server.py)
- [python/minisgl/utils/hf.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/utils/hf.py)

做的事情：

- 多模态模型自动使用 `AutoProcessor`
- 支持本地图片路径和 `data:` URL
- 产出：
  - `input_ids`
  - `pixel_values`
  - `image_grid_thw`
  - `mm_token_type_ids`

### 3. 多模态 side inputs 贯穿请求链路

文件：

- [python/minisgl/message/tokenizer.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/message/tokenizer.py)
- [python/minisgl/message/backend.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/message/backend.py)
- [python/minisgl/message/utils.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/message/utils.py)
- [python/minisgl/scheduler/utils.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/scheduler/utils.py)
- [python/minisgl/scheduler/prefill.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/scheduler/prefill.py)
- [python/minisgl/scheduler/scheduler.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/scheduler/scheduler.py)
- [python/minisgl/core.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/core.py)

新增并贯通了：

- `pixel_values`
- `image_grid_thw`
- `mm_token_type_ids`
- `rope_delta`
- `mrope_positions`

### 4. 模型配置识别成真正的多模态模型

文件：

- [python/minisgl/models/config.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/config.py)

补充了：

- `vision_config`
- `image_token_id`
- `video_token_id`
- `vision_start_token_id`
- `vision_end_token_id`
- `is_multimodal`

同时修正了一个关键问题：

- 不能再把 `Qwen3_5MoeForConditionalGeneration` 一律改写成文本 `ForCausalLM`
- 只有在没有 `vision_config` 时，才退化成纯文本模型

### 5. 新增本地 VL 模型实现

文件：

- [python/minisgl/models/qwen3_vl_moe.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/qwen3_vl_moe.py)
- [python/minisgl/models/register.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/register.py)

实现策略：

- 复用现有 `qwen3_5_moe` 语言主干
- 使用 HF 的 `Qwen3_5MoeVisionModel` 作为视觉塔
- 将图像特征替换到 image placeholder token 对应的 embedding
- 计算多模态 3D / MRoPE `position_ids`
- 将 decode 阶段所需的 `rope_delta` 保留到请求状态

### 6. 权重加载支持视觉模块

文件：

- [python/minisgl/models/weight.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/weight.py)

做的事情：

- 扩展 `module_dict`，不仅支持本地 `BaseOP`，也支持普通 `torch.nn.Module`
- 支持将普通 `nn.Module` 的参数和 buffer 从 checkpoint materialize 到真实张量
- 让视觉塔参数 `model.visual.*` 能被正确加载

### 7. 修复视觉 rotary buffer 的 `meta/device` 问题

文件：

- [python/minisgl/models/qwen3_vl_moe.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/qwen3_vl_moe.py)

发现的问题：

- 视觉塔里的 `rotary_pos_emb.inv_freq` 是 non-persistent buffer
- checkpoint 不会恢复它
- 模型初始构造在 `meta` 设备上时，这个 buffer 会留在 `meta`
- 导致图片前向在 vision attention 里崩掉

修复方式：

- 运行时在真正收到图片、明确设备后，按目标 device 重建该 buffer
- 同时将视觉塔整体搬到图片输入所在 device

### 8. 修复多模态模型误走 CUDA graph decode 路径

文件：

- [python/minisgl/engine/graph.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/engine/graph.py)

问题：

- `Qwen3_5VLMoeForConditionalGeneration.supports_cuda_graph = False`
- 但 `GraphRunner` 仍然保留了 graph batch size 列表
- decode 时会误判可走 graph，随后因为没有 capture dummy req 而崩掉

修复：

- 当模型声明 `supports_cuda_graph = False` 时，直接禁用 graph batch 列表

## 当前范围和限制

当前实现是“最小可用图片问答”，不是完整通用多模态框架。

已支持：

- 单张图片输入
- OpenAI 风格 chat 请求
- 图片问答

暂未支持：

- 视频输入
- CUDA graph 下的多模态推理
- 面向多图/复杂多模态 batch 的完整优化

## 测试方式

启动服务：

```bash
PATH=/mnt/82_store/zxz/condaenv/minisgl/bin:$PATH \
python -m minisgl.server.launch \
  --model-path /mnt/82_store/LLM-weights/Qwen3.6-35B-A3B \
  --host 127.0.0.1 \
  --port 1919 \
  --dtype bfloat16 \
  --attention-backend fi \
  --cache-type naive \
  --num-pages 4096 \
  --memory-ratio 0.85
```

发送图片问答请求：

```python
import base64
import requests
from pathlib import Path

img = Path("my_docs/results/tom.jpg").read_bytes()
b64 = base64.b64encode(img).decode()

payload = {
    "model": "/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "图片中是什么内容？"},
        ],
    }],
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 48,
    "stream": False,
}

resp = requests.post("http://127.0.0.1:1919/v1/chat/completions", json=payload, timeout=180)
print(resp.status_code)
print(resp.text)
```

## 总结

这次适配的关键不是“只改 API”，而是把以下几层一起接通：

- API schema
- processor/tokenizer
- 请求与 batch side inputs
- VL 模型 forward
- 视觉权重加载
- 多模态 rotary / decode 状态

目前已经达到“可以正常完成图片推理”的目标。
