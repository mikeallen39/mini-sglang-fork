from __future__ import annotations

import threading
from pathlib import Path

import torch

_LOAD_LOCK = threading.Lock()
_LOADED = False

_ROUTING_ONLY_COMMON_OPS = Path(
    "/data/zxz/condaenv/sglangcmp/lib/python3.10/site-packages/sgl_kernel/sm90/common_ops.abi3.so"
)


def ensure_routing_ops_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        torch.ops.load_library(str(_ROUTING_ONLY_COMMON_OPS))
        _LOADED = True
