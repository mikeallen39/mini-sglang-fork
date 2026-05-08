from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.utils import init_logger
from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

logger = init_logger(__name__)

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    try:
        return load_jit(
            "index",
            *args,
            cuda_files=["index.cu"],
            cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
        )
    except Exception as exc:
        logger.warning("Falling back to torch indexing kernel: %s", exc)
        return None  # type: ignore[return-value]


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    if module is None:
        indices_long = indices.to(torch.long)
        if vocab_range is None:
            output.copy_(weights.index_select(0, indices_long))
        else:
            start, length = vocab_range
            end = start + length
            valid = (indices_long >= start) & (indices_long < end)
            output.zero_()
            if valid.any():
                local_indices = (indices_long[valid] - start).contiguous()
                output[valid] = weights.index_select(0, local_indices)
        return output

    module.launch(weights, indices, output, vocab_range)
    return output
