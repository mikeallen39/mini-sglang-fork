# 2026-03-25 GLM-4.7-Flash Progress

## Scope

This note tracks the GLM-4.7-Flash support work on branch `feat/glm47-flash-support`.

## Pushed Changes

- Pushed commit: `a79fc0c` (`Fix GLM-4.7 MLA backend and streamline loading`)
- Remote branch: `origin/feat/glm47-flash-support`

### What is included in `a79fc0c`

- Fixed the real MLA path for `glm4_moe_lite`.
- Restored GLM auto-selection to use MLA attention backend.
- Fixed GLM MoE backend selection to use fused MoE by default.
- Fixed TP-related GLM weight loading / expert packing issues.
- Added config regression test:
  - `tests/misc/test_glm4_config.py`

## Verified Results Before Further TP=1 Optimization

### Config regression

- Command:
  - `PYTHONPATH=python /data/zxz2/condaenv/mini-sglang/bin/python -m pytest -q tests/misc/test_glm4_config.py`
- Result:
  - `2 passed`

### Real MLA backend functional tests

- `tp=1`, GPU `6`, explicit `--attention-backend mla`
  - Prompt: `hi`
  - Output: `Hello! How can I assist you today?`

- `tp=2`, GPUs `0,6`, explicit `--attention-backend mla`
  - Prompt: `hi`
  - Output: `Hello! How can I help you today?`

## TP=1 Loading Investigation

### Symptom

- `tp=2` loading was much faster than `tp=1`.
- Earlier real `tp=1` runs took minutes, while `tp=2` could become ready in tens of seconds.

### Profiling conclusion

- Raw `safetensors` reading is not the bottleneck.
  - First 6 shards were mostly about `0.3s` each when only reading tensors.
- Merge / expert pack logic itself is also not the bottleneck.
  - First 6 shards were still about `0.3s` each when doing merge/pack without retaining final tensors.
- The slowdown appears when many final tensors are kept as persistent GPU allocations.
  - When retaining outputs in a dict, shard times stayed low until about `15 GiB` allocated, then rose to multiple seconds per shard.
- A much better strategy is to preallocate the model tensors on GPU first, then `copy_` weights into those existing tensors.
  - Test result on GPU `6`:
    - Preallocated model creation took about `2.85s`
    - GPU allocation after creation was about `55.84 GiB`
    - After that, shards `1..20` stayed around `0.3-0.4s` each

## Current Unpushed Changes

These changes are currently local and not pushed yet:

- `python/minisgl/engine/engine.py`
  - For `glm4_moe_lite` with `tp=1`, initialize the model directly on GPU instead of meta tensors.
- `python/minisgl/models/weight.py`
  - Prefer `copy_` into already materialized tensors when shapes / dtypes / devices match, instead of replacing with a new persistent allocation.

## Latest TP=1 Optimization Validation

### End-to-end real run after preallocation change

- Command shape:
  - `CUDA_VISIBLE_DEVICES=6 ... python -m minisgl --model-path /mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash --tensor-parallel-size 1 --attention-backend mla --port 30225 --shell-mode`
- Observed timing:
  - Free-memory log before loading: `15:12:19`
  - Weight loading completed: `15:12:36`
  - Scheduler ready: `15:12:37`
- Practical result:
  - Weight loading dropped to about `17s`
  - End-to-end startup to ready is now on the order of tens of seconds instead of many minutes

### Functional check after the loading optimization

- Prompt: `hi`
- Output: `Hello! How can I assist you today?`

## Current Status

- MLA backend is functionally correct for GLM-4.7-Flash on both `tp=1` and `tp=2`.
- `tp=1` loading optimization is now validated by a real end-to-end run.
- Remaining optional step:
  - commit / push the new `tp=1` preallocation optimization if we want this phase landed remotely as well.
