# 2026-06-24 Qwen3.6 DFlash Integration Status

## Scope

This note records the current state of integrating `DFLASH` speculative decoding into local `mini-sglang` for:

- target model: `/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B`
- draft model: `/mnt/82_store/LLM-weights/z-lab/Qwen3.6-35B-A3B-DFlash`

Baseline for comparison:

- `W8A8 Selective Int8 + Vendored Full-Kernel Chunk Prefill + Fused MoE + SGLang Linear Attention + CUDA Graph`
- best recorded baseline:
  - `TTFT = 113.29 ms`
  - `E2E = 0.7610 s`
  - `output_tps = 82.79 tok/s`

## What Was Integrated

### Phase 1: config and loading

Integrated:

- speculative server args / config plumbing
- draft model path parsing
- dual-model loading
- local `DFlashDraftModel` registration
- draft checkpoint streaming load helpers

Relevant files:

- [python/minisgl/server/args.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/server/args.py)
- [python/minisgl/engine/config.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/engine/config.py)
- [python/minisgl/models/register.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/register.py)
- [python/minisgl/models/weight.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/weight.py)
- [python/minisgl/models/dflash.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/dflash.py)

### Phase 2: minimal runtime

Integrated:

- target prefill hidden capture from Qwen3.6
- projection from target hidden to draft hidden
- minimal draft block generation
- target verify path
- accept / reject loop

Relevant files:

- [python/minisgl/models/qwen3_5_moe.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/qwen3_5_moe.py)
- [python/minisgl/engine/engine.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/engine/engine.py)
- [python/minisgl/scheduler/scheduler.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/scheduler/scheduler.py)
- [python/minisgl/core.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/core.py)
- [python/minisgl/speculative/dflash_info.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/speculative/dflash_info.py)
- [python/minisgl/speculative/dflash_utils.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/speculative/dflash_utils.py)

## Functional Status

Current local DFlash path is **functionally correct enough to run end-to-end generation**:

- draft model loads
- target prefill captures hidden states
- draft block can be generated
- target verify runs
- normal text output is produced

Representative output file:

- [qwen36_dflash_run1_output.txt](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/output_texts/qwen36_dflash_run1_output.txt)

## Performance Results

### Baseline

- `TTFT = 113.29 ms`
- `E2E = 0.7610 s`
- `output_tps = 82.79 tok/s`

### DFlash attempts

#### 1. Minimal runtime with serial-ish verify

- `TTFT = 183.96 ms`
- `E2E = 3.9873 s`
- `output_tps = 15.80 tok/s`

#### 2. First block-verify attempt

- `TTFT = 191.79 ms`
- `E2E = 4.6300 s`
- `output_tps = 13.61 tok/s`

#### 3. Verify batch preparation/finalization added

- `TTFT = 201.33 ms`
- `E2E = 4.8306 s`
- `output_tps = 13.04 tok/s`

#### 4. `spec_info`-style local restructuring

Run on GPU6 with:

- `--attention-backend fi`
- `--cache-type naive`
- `--cuda-graph-max-bs 0`
- `--speculative-algorithm DFLASH`
- `--speculative-num-draft-tokens 16`

Results:

- `run1: TTFT=42762.90 ms, E2E=47.2749 s, output_tokens=63`
- steady `run2-run5`:
  - `TTFT = 4511.94 ms`
  - `E2E = 8.9849 s`
  - `output_tps = 7.01 tok/s`

## Diagnosis

Current DFlash path is **not performance-competitive** with the tuned baseline.

Main reasons:

1. Verify is still a heavy fallback path.
   - It still creates a temporary verify request and verify batch.
   - It still reruns generic metadata preparation and target forward in a heavy way.

2. Draft path is still a minimal pure-Torch implementation.
   - No draft worker
   - No draft KV cache
   - No compact draft cache

3. No CUDA graph integration on the DFlash path.
   - Current DFlash runs with `--cuda-graph-max-bs 0`
   - Baseline decode benefits heavily from graph

4. Local `spec_info` migration alone does not bring performance.
   - It helps structure the runtime more like upstream
   - But it does not migrate the real fast path

## Conclusion

Current `mini-sglang` DFlash integration should be treated as:

- **functionality validation complete**
- **performance optimization not successful**

The current implementation demonstrates that DFlash can be wired into local `mini-sglang`, but it is still far from upstream `sglang`'s high-performance speculative runtime.

To obtain real speedup locally, the next step would require migrating substantially more upstream structure, including:

- draft worker
- draft KV cache / compact cache
- native target verify mode
- backend verify metadata fast path
- block-native accept / reject bookkeeping
- CUDA graph integration

At this point, the task is paused.
