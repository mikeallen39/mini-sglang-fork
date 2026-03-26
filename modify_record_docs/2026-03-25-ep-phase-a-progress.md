# 2026-03-25 EP Phase A Progress

## Scope

This note records the first implementation step for EP planning in `mini-sglang`.

This phase does not implement expert dispatch yet.
It only adds the minimum config and distributed-state plumbing needed so later EP work
does not stay coupled to TP-only assumptions.

## Changes

- Added `--expert-parallel-size` / `--ep-size` in server args.
- Added explicit `ep_info` to engine/server config.
- Added EP state helpers in `python/minisgl/distributed/info.py`:
  - `set_ep_info`
  - `get_ep_info`
  - `try_get_ep_info`
  - `build_ep_info`
- Propagated `ep_info` through:
  - argument parsing
  - multi-process launch
  - engine initialization
  - offline `LLM` entry

## Current Validation Rules

Phase A intentionally keeps the supported space small:

- `ep_size >= 1`
- `ep_size in {1, tp_size}`
- `ep_size > 1` only allowed for MoE models
- routed expert count must be divisible by `ep_size`

## Why This Shape

Current `mini-sglang` still has TP as the only real distributed execution dimension.
So the right first step is to make EP explicit in config/state, while still rejecting
unsupported layouts clearly.

That prevents future EP work from encoding more TP-only assumptions in:

- engine init
- worker launch
- model construction
- weight loading entry points

## Not Included Yet

- EP process groups distinct from TP groups
- EP communication primitives such as `all_to_all`
- EP-aware MoE dispatcher
- local-only expert weight loading
- runtime topk global-to-local expert remapping

## Phase B Follow-up

After Phase A config plumbing, the next step was implemented for the constrained path
`ep_size == tp_size`.

This is still a correctness-first EP path, not the final dispatcher-based design.

### Added

- local expert ownership helper:
  - contiguous global expert range per EP rank
- local MoE TP view:
  - `moe_tp_size = tp_size / ep_size`
  - for `ep_size == tp_size`, local experts are no longer additionally TP-sharded
- EP-aware MoE execution changes:
  - routed expert weights are allocated only for local experts
  - global expert ids are remapped to local expert slots before local expert compute
  - existing output `all_reduce` remains the correctness combine path

### Weight Loading Speed Fixes

The first EP loader version was functionally correct but too slow because it still loaded
remote expert tensors from `safetensors` before discarding them.

That path was fixed by:

- filtering non-local expert tensors before `f.get_tensor(...)`
- stacking only local routed experts
- using `moe_tp` instead of dense-layer `tp` for expert weight sharding

This was required both for correctness and for fast validation loops.

### CUDA Graph Limitation

The current constrained EP path is not CUDA-graph-safe yet.

Reason:

- the GLM routed-expert path still uses dynamic PyTorch ops such as `torch.where`
  during local expert selection
- that fails under stream capture

Current temporary behavior:

- automatically disable CUDA graph when `ep_size > 1`

This keeps inference usable while avoiding capture failures.

## Validation

### Lightweight tests

Executed:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... python -m pytest -o addopts='' -q tests/misc/test_glm4_config.py`

Result:

- `5 passed`

Coverage of these tests includes:

- GLM MLA auto-detection unchanged
- EP config parsing
- EP rejection on dense models
- auto-disable CUDA graph for EP

### Real GPU validation

Model:

- `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`

Validated configuration:

- `tp=2`
- `ep=2`
- attention backend `mla`
- MoE backend `fused`

Observed behavior after the fixes:

- model loads successfully
- service starts successfully with CUDA graph disabled
- OpenAI-compatible chat request with prompt `hi` returns:
  - `Hello! How can I assist you today?`

This confirms that `fusedmoe + ep_size == tp_size` is working in the current
correctness-first design.

### Notes On Loading Time

On the validated `tp=2, ep=2` path, loading was reduced from the earlier multi-minute
regression to about:

- roughly 20-90 seconds depending on which idle GPU pair was selected and their transient load

The critical regression that was removed was:

- reading remote expert tensors from checkpoint and only discarding them after GPU materialization

## Dispatcher Refactor Follow-up

After the correctness-first EP path was validated, the local expert remap logic was
refactored into a shared minimal dispatcher/mapping utility:

- `python/minisgl/moe/dispatch.py`

Current purpose:

- unify `global expert id -> local expert slot` remap logic
- keep GLM custom MoE path and generic fused/torch MoE backends on the same semantics
- reduce the amount of EP-specific logic duplicated across model code

Updated call sites:

- `python/minisgl/moe/fused.py`
- `python/minisgl/moe/torch_backend.py`
- `python/minisgl/models/glm4_moe_lite.py`

Validation:

- added `tests/misc/test_moe_dispatch.py`
- reran:
  - `tests/misc/test_glm4_config.py`
  - `tests/misc/test_moe_dispatch.py`
- real GPU regression check repeated on:
  - `tp=2`
  - `ep=2`
  - `attention_backend=mla`
  - OpenAI-style HTTP request with prompt `hi`
- result remained correct:
  - `Hello! How can I assist you today?`

## Divisor EP Follow-up

The EP grouping logic was then extended from:

- `ep_size in {1, tp_size}`

to:

- `ep_size` is any positive divisor of `tp_size`

Current grouping policy is contiguous:

- `ep_rank = tp_rank // moe_tp_size`
- `moe_tp_size = tp_size / ep_size`
- `moe_tp_rank = tp_rank % moe_tp_size`

This keeps the current correctness-first EP path compatible with:

- local expert ownership
- local-only expert loading
- expert weight sharding by `moe_tp`
- full-TP `all_reduce` as the current combine path

### Validation

Local validation:

- added unit coverage for divisor EP grouping in `tests/misc/test_glm4_config.py`
- reran:
  - `tests/misc/test_glm4_config.py`
  - `tests/misc/test_moe_dispatch.py`
- result:
  - `9 passed`

Real GPU validation status:

- attempted `tp=4, ep=2` with small `--num-pages` to tolerate memory imbalance
- launch and grouping logic were accepted
- loading began successfully
- did not complete end-to-end generation in that run because the available 4-GPU set was under heavy external load and checkpoint loading degraded to roughly 9 seconds per file

So the current status for divisor EP is:

- code path implemented
- local tests passed
- real multi-GPU launch reached loading successfully
- a clean end-to-end `tp=4, ep=2` generation run still needs a less contended 4-GPU window

## 2026-03-26 Fused EP Graph-Safety Follow-up

This round focused on making the GLM routed-expert path graph-safe without giving up
the fused MoE backend.

### Changes

- removed the GLM-specific dynamic local expert loop from:
  - `python/minisgl/models/glm4_moe_lite.py`
- routed GLM MoE execution through the shared MoE backend path
- kept CUDA graph enabled for:
  - `moe_backend=fused`
  - `ep_size > 1`
- kept CUDA graph disabled only for:
  - `moe_backend=torch`
  - `ep_size > 1`
- changed EP local expert remap semantics in:
  - `python/minisgl/moe/dispatch.py`
  from a positive sentinel local id to:
  - `-1`
- extended the Triton fused MoE kernel path to treat filtered expert blocks as zero-output
  blocks instead of indexing a fake expert slot

### What Was Validated

Local tests:

- reran:
  - `tests/misc/test_glm4_config.py`
  - `tests/misc/test_moe_dispatch.py`
- current result:
  - `10 passed`

Targeted correctness checks on a single GPU:

- fused local MoE vs torch local MoE:
  - matched for non-EP and EP-partitioned cases
- simulated EP split-and-sum vs full 32-expert reference:
  - matched for both torch and fused paths
- observed error scale:
  - max diff about `1.22e-4`
  - mean diff about `7.9e-6`

These checks confirm that the current fused EP local routing and zero-filtered expert
handling are numerically aligned with the torch reference at the operator level.

### Real GPU GLM Validation

Validated launch configuration:

- model:
  - `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`
- `tp=2`
- `ep=2`
- attention backend:
  - `mla`
- MoE backend:
  - `fused`
- temporary paths moved off `/tmp`:
  - `TRITON_CACHE_DIR=/mnt/42_store/zxz/.triton-cache-mini-sglang`
  - `TMPDIR=/mnt/42_store/zxz/tmp-mini-sglang`

Observed behavior:

- model loading completed successfully
- CUDA graph capture completed successfully
- scheduler and OpenAI-compatible API server became ready successfully

This confirms that the earlier blockers were resolved:

- graph-unsafe GLM local expert loop
- Triton cache failure on full root filesystem
- EP filtered-expert crash during graph capture

### Remaining Issue

Although the server now starts and captures graphs correctly, real generation quality is
still not acceptable in this configuration.

Observed symptom:

- OpenAI-style chat requests such as `hi` returned garbled text rather than a normal reply

Current conclusion:

- startup correctness is now much better
- operator-level EP correctness checks pass
- but real-model end-to-end behavior for this exact `tp=2, ep=2, mla, fused` GLM path is
  still not fully validated as correct

So this configuration should currently be treated as:

- launchable
- graph-capturable
- not yet fully behavior-validated for user-visible generation quality
