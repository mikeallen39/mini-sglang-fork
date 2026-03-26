# 2026-03-26 MLA CUDA Graph Fix for GLM-4.7-Flash

## Summary

This update fixed the remaining graph-only failure for GLM-4.7-Flash when using the
MLA attention backend.

Before this fix:

- `graph off` was already correct
- `graph on` failed in real end-to-end runs
  - `tp=2, ep=2` produced garbled text
  - `tp=2, ep=1` also produced garbled text
  - `tp=1, ep=1` hit `CUDA error: an illegal memory access was encountered`

This showed the remaining bug was not EP-specific and not TP-specific. The common
factor was the MLA CUDA-graph path.

## Root Cause

`python/minisgl/attention/mla_backend.py` created normal
`flashinfer.mla.BatchMLAPagedAttentionWrapper` instances for graph capture/replay.

That meant the MLA graph path was missing the replay-safe static buffers required by
FlashInfer when `use_cuda_graph=True`, especially:

- `qo_indptr`
- `kv_indptr`
- `kv_indices`
- `kv_len_arr`

In other words, the code was replaying an eager-style MLA wrapper instead of a real
CUDA-graph-aware MLA wrapper.

## Fix

Updated:

- `python/minisgl/attention/mla_backend.py`

Change made in `MLABackend.init_capture_graph()`:

- graph wrappers are now constructed with `use_cuda_graph=True`
- replay buffers are explicitly bound from the preallocated capture storage:
  - `capture.cu_seqlens_q`
  - `capture.cu_seqlens_k`
  - `capture.page_table`
  - `capture.seq_lens`

This makes the FlashInfer MLA wrapper state replay-safe under CUDA graph execution.

## Real GPU Validation

Environment:

- model:
  - `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`
- python:
  - `/data/zxz2/condaenv/mini-sglang/bin/python`
- attention backend:
  - `mla`
- moe backend:
  - `fused`

Validated with real `hi` requests through `/v1/chat/completions`.

### 1. `tp=1, ep=1, graph on`

Command shape:

- `CUDA_VISIBLE_DEVICES=6 python -m minisgl --model-path ... --tp 1 --ep 1 --attention-backend mla --moe-backend fused --port 30242`

Observed result:

- request returned normal text
- response:
  - `Hello! How can I assist you today?`

### 2. `tp=2, ep=1, graph on`

Command shape:

- `CUDA_VISIBLE_DEVICES=5,6 python -m minisgl --model-path ... --tp 2 --ep 1 --attention-backend mla --moe-backend fused --port 30241`

Observed result:

- request returned normal text
- response:
  - `Hello! How can I assist you today?`

### 3. `tp=2, ep=2, graph on`

Command shape:

- `CUDA_VISIBLE_DEVICES=5,6 python -m minisgl --model-path ... --tp 2 --ep 2 --attention-backend mla --moe-backend fused --port 30231`

Observed result:

- request returned normal text
- response:
  - `Hello! How can I assist you today?`

## Regression Check

Local tests still pass:

- `tests/misc/test_glm4_config.py`
- `tests/misc/test_moe_dispatch.py`

Result:

- `10 passed`

## Remaining Issue

`tp=2, ep=2` weight loading is still much slower than the other validated launch
configurations.

Observed rough comparison in this session:

- `tp=2, ep=1`: around 16 seconds to load weights
- `tp=2, ep=2`: around 9.5 minutes to load weights

So the MLA graph correctness issue appears fixed, but the EP load path still needs
separate optimization work.
