# GLM-4.7-Flash Benchmark and Fix Notes

## Setup

- Model: `/mnt/82_store/LLM-weights/ZhipuAI/GLM-4.7-Flash`
- Script: [benchmark/online/bench_qwen36_1024in_64out.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/benchmark/online/bench_qwen36_1024in_64out.py)
- Workload:
  - `input_tokens = 1024`
  - `output_tokens = 64`
  - actual steady-state generated tokens: `63`
- Runtime:
  - `attention-backend = mla`
  - `moe-backend = fused`
  - `cache-type = naive`
  - `cuda-graph-max-bs = 160`

## Root Causes Fixed

### 1. Broken online weight loading for GLM MLP/shared experts

The streaming loader in [python/minisgl/models/weight.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/models/weight.py) did not merge:

- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`

into local fused `gate_up_proj` modules for:

- layer 0 dense MLP
- sparse-layer `shared_experts`

This caused malformed generation such as repeated `!`.

### 2. MLA CUDA graph replay misclassified real requests as dummy graph requests

`graph=0 + mla` produced normal long-context output, but `graph=1 + mla` produced degraded text on long prompts.

The issue was in [python/minisgl/attention/mla_backend.py](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/python/minisgl/attention/mla_backend.py):

- replay already copied real request-time metadata into static graph buffers, but the
  dummy-request detection condition was wrong
- the old check used a condition equivalent to:
  - `req.table_idx >= max_graph_bs`
- this is invalid because real scheduler-assigned `table_idx` values can easily be
  larger than `max_graph_bs`
- under `graph=1`, some real decode requests were therefore treated as dummy graph
  requests and incorrectly fetched `kv_indices` from the static `capture.page_table`
  instead of the real global page table

As a result, MLA graph replay read the wrong KV pages for long-context decode.

The fix was to identify dummy graph requests by their true reserved table range:

- `req.table_idx >= get_global_ctx().page_table.size(0) - max_graph_bs`

This cleanly separates:

- real decode requests, which must use the live global page table
- graph dummy requests, which are allowed to use the static capture page table

The replay path still updates the static capture buffers before re-planning the wrapper.

### 3. GLM no-thinking path

`GLM-4.7-Flash` supports both:

- default thinking mode
- `enable_thinking=false`

The local service now preserves the model's official template behavior instead of using
an earlier incorrect "trim `<think>`" workaround. After the fixes:

- `graph=0`: thinking and no-thinking both produce normal long outputs
- `graph=1`: thinking and no-thinking both produce normal long outputs

## Sanity Check

After the fixes:

- short prompt output was normal
- long `1024-token` prompt output was also normal under:
  - `graph=0 + mla`
  - `graph=1 + mla`

The validated default-thinking `run1` output file is:

- [glm47_flash_graph_1024in_64out_run1_output.txt](/mnt/42_store/zxz/mini-sglang/mini-sglang-fork/my_docs/results/output_texts/glm47_flash_graph_1024in_64out_run1_output.txt)

## Benchmark Result

### Raw runs

Initial valid graph-enabled run:

- `run1: TTFT=125.16 ms, E2E=1.1317 s, output_tokens=63`
- `run2: TTFT=126.87 ms, E2E=1.1342 s, output_tokens=63`
- `run3: TTFT=125.83 ms, E2E=1.1326 s, output_tokens=63`
- `run4: TTFT=126.84 ms, E2E=1.1360 s, output_tokens=63`
- `run5: TTFT=125.92 ms, E2E=1.1328 s, output_tokens=63`

Post-fix verification rerun:

- `run1: TTFT=171.09 ms, E2E=1.1804 s, output_tokens=63`
- `run2: TTFT=125.11 ms, E2E=1.1399 s, output_tokens=63`
- `run3: TTFT=126.88 ms, E2E=1.1441 s, output_tokens=63`
- `run4: TTFT=128.12 ms, E2E=1.1432 s, output_tokens=63`
- `run5: TTFT=126.18 ms, E2E=1.1406 s, output_tokens=63`

### Steady state

- `TTFT = 126.36 ms`
- `E2E = 1.1339 s`
- `output_tps = 55.56 tok/s`
- `avg_ms_per_output_token = 18.00 ms`
- `avg_output_tokens = 63.00`

## Conclusion

`GLM-4.7-Flash` now runs correctly in this repo under:

- `MLA`
- `fused MoE`
- `CUDA graph`

and the benchmark result above is the first valid `1024 in / 64 out` performance measurement after fixing both:

- GLM streaming weight merge bugs
- MLA CUDA graph replay metadata bugs
