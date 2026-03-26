# 2026-03-26 W8A8 INT8 Migration

## Summary

Added a minimal real `W8A8 int8` quantization path to `mini-sglang-fork` for post-load execution on top of existing float checkpoints.

This migration is intentionally narrow:

- `--quantization w8a8_int8` is now accepted by the server CLI.
- Dense linear layers are post-processed into real int8 weights with per-channel scales.
- Activations are quantized dynamically per token at runtime.
- Execution uses `sgl_kernel.int8_scaled_mm`, so this is real int8 GEMM rather than fake quantization.
- MoE expert MLPs can run in real int8 through both:
  - the conservative `torch` MoE backend
  - the Triton-based fused MoE backend

## Implementation Notes

- Quantization is applied after normal checkpoint loading, so the existing fast GLM streaming loader path is preserved.
- Dense linear weights are converted from `[out_features, in_features]` float tensors into:
  - transposed int8 weights `[in_features, out_features]`
  - per-output-channel scales `[out_features, 1]`
- MoE expert weights are converted expert-by-expert into the same kernel-friendly format:
  - `gate_up_proj`: `[num_local_experts, hidden, 2 * intermediate_per_partition]`
  - `down_proj`: `[num_local_experts, intermediate_per_partition, hidden]`
- Under `w8a8_int8`, CUDA graph is disabled conservatively for safety.
- Under `w8a8_int8`, CUDA graph is still disabled conservatively.
- The fused MoE int8 path uses per-token activation int8 quantization and per-output-channel weight scales.

## Current Limitations

- This does not yet add checkpoint-native quantized weight loading; it quantizes after loading float weights.
- `ParallelLMHead` / embedding weights are not quantized in this pass.
- Quantized MoE fused execution is implemented only for the current per-channel `W8A8 int8` path.
- This path was added without GPU runtime validation on this machine.
- No GPU validation was run in this task because there were no idle GPUs available.

## Intended Usage

Example:

```bash
python -m minisgl \
  --model /path/to/model \
  --quantization w8a8_int8
```

For MoE models, `--moe-backend fused` and `--moe-backend torch` are both valid under `w8a8_int8`.
