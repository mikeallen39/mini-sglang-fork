from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported
from minisgl.linear_attention import (
    get_linear_attn_backend,
    has_sglang_linear_attn_kernel,
    set_linear_attn_backend,
)
from .logger import init_logger
from .misc import (
    UNSET,
    Unset,
    align_ceil,
    align_down,
    call_if_main,
    div_ceil,
    div_even,
    local_kv_heads,
)
from .mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)
from .registry import Registry
from .torch_utils import nvtx_annotate, torch_dtype


def cached_load_hf_config(*args, **kwargs):
    from .hf import cached_load_hf_config as _cached_load_hf_config

    return _cached_load_hf_config(*args, **kwargs)


def download_hf_weight(*args, **kwargs):
    from .hf import download_hf_weight as _download_hf_weight

    return _download_hf_weight(*args, **kwargs)


def load_processor(*args, **kwargs):
    from .hf import load_processor as _load_processor

    return _load_processor(*args, **kwargs)


def load_tokenizer(*args, **kwargs):
    from .hf import load_tokenizer as _load_tokenizer

    return _load_tokenizer(*args, **kwargs)


def model_supports_multimodal(*args, **kwargs):
    from .hf import model_supports_multimodal as _model_supports_multimodal

    return _model_supports_multimodal(*args, **kwargs)

__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "load_processor",
    "load_tokenizer",
    "model_supports_multimodal",
    "set_linear_attn_backend",
    "get_linear_attn_backend",
    "has_sglang_linear_attn_kernel",
    "init_logger",
    "is_arch_supported",
    "is_sm90_supported",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "local_kv_heads",
    "align_ceil",
    "align_down",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
