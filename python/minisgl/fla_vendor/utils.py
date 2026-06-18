from __future__ import annotations

import contextlib
import functools
import inspect
from typing import Any, Callable

import torch
import triton


SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": True} if SUPPORTS_AUTOTUNE_CACHE else {}

# Match upstream behavior enough for chunk kernels while keeping local dependencies minimal.
SUPPRESS_LEVEL = 0


def tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    cache_entries: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i] + cache_entries[i + 1 :] + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def custom_device_ctx(index: int | None):
    if index is None:
        return contextlib.nullcontext()
    return torch.cuda.device(index)


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        contiguous_args = tuple(
            a.contiguous() if isinstance(a, torch.Tensor) else a for a in args
        )
        contiguous_kwargs = {
            k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        with custom_device_ctx(tensor.device.index if tensor is not None and tensor.is_cuda else None):
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def autocast_custom_fwd(fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn


@functools.lru_cache(maxsize=None)
def is_nvidia_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major >= 9


@functools.lru_cache(maxsize=None)
def is_tf32_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major >= 8


@functools.lru_cache(maxsize=None)
def check_shared_mem() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(props, "shared_memory_per_block_optin", 0) >= 128 * 1024


@functools.lru_cache(maxsize=None)
def is_gather_supported() -> bool:
    return hasattr(triton.language, "gather")
