# 2026-03-26 W8A8 INT8 Migration

## Summary

在现有 float checkpoint 之上，为 `mini-sglang-fork` 增加了一条最小可用的真实
`W8A8 int8` 量化路径，用于加载完成后的执行阶段。

这次迁移的范围有意保持很窄：

- server CLI 现在接受 `--quantization w8a8_int8`。
- dense linear layers 会在加载后被转换为真实 int8 权重，并带有 per-channel scales。
- activations 会在运行时按 token 动态量化。
- 执行使用 `sgl_kernel.int8_scaled_mm`，因此这是真实的 int8 GEMM，而不是假量化。
- MoE expert MLPs 现在可以通过以下两种方式使用真实 int8 运行：
  - 保守的 `torch` MoE backend
  - 基于 Triton 的 fused MoE backend

## Implementation Notes

- 量化发生在正常 checkpoint loading 之后，因此现有快速 GLM streaming loader 路径得以保留。
- dense linear weights 会从 `[out_features, in_features]` 的 float tensor 转换为：
  - 转置后的 int8 权重 `[in_features, out_features]`
  - 每个输出通道一组 scale 的 `[out_features, 1]`
- MoE expert 权重会按 expert 逐个转换为同样适合 kernel 的格式：
  - `gate_up_proj`: `[num_local_experts, hidden, 2 * intermediate_per_partition]`
  - `down_proj`: `[num_local_experts, intermediate_per_partition, hidden]`
- 在 `w8a8_int8` 下，出于安全考虑会保守地禁用 CUDA graph。
- 在 `w8a8_int8` 下，CUDA graph 目前仍然保守地保持禁用。
- fused MoE int8 路径使用按 token 的 activation int8 量化，以及按输出通道的 weight scales。

## Current Limitations

- 目前还没有加入原生量化 checkpoint 的加载；现在是在加载 float 权重后再做量化。
- 本轮没有量化 `ParallelLMHead` / embedding weights。
- 量化后的 MoE fused 执行目前只实现了当前这条 per-channel `W8A8 int8` 路径。
- 这条路径在本机上加入时，没有做 GPU 运行时验证。
- 这次任务没有运行 GPU 验证，因为当时没有空闲 GPU 可用。

## Intended Usage

示例：

```bash
python -m minisgl \
  --model /path/to/model \
  --quantization w8a8_int8
```

对于 MoE 模型，在 `w8a8_int8` 下，`--moe-backend fused` 和 `--moe-backend torch` 都可用。
