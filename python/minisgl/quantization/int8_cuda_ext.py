from __future__ import annotations

import functools
import os
from pathlib import Path

import torch
from minisgl.utils import init_logger

logger = init_logger(__name__)

_SRC_DIR = Path(__file__).resolve().parent / "csrc"
_BUILD_DIR = Path(__file__).resolve().parent / ".build" / "int8_cuda_ext"


def _extension_name() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"minisgl_int8_cuda_ext_sm{major}{minor}"


def _can_use_torch_int_mm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> bool:
    return (
        mat_a.is_cuda
        and mat_b.is_cuda
        and mat_a.dtype == torch.int8
        and mat_b.dtype == torch.int8
        and mat_a.ndim == 2
        and mat_b.ndim == 2
        and hasattr(torch, "_int_mm")
        and mat_a.shape[1] % 16 == 0
        and mat_b.shape[0] % 16 == 0
        and mat_b.shape[1] > 0
        and mat_b.shape[1] % 8 == 0
    )


@functools.cache
def _load_extension():
    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    from torch.utils.cpp_extension import load

    try:
        return load(
            name=_extension_name(),
            sources=[
                str(_SRC_DIR / "int8_scaled_mm.cpp"),
                str(_SRC_DIR / "int8_scaled_mm_kernel.cu"),
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
            ],
            build_directory=str(_BUILD_DIR),
            verbose=False,
        )
    except Exception as exc:
        logger.warning("Falling back to Python int8 path: %s", exc)
        return None


def int8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not _can_use_torch_int_mm(mat_a, mat_b):
        return None

    module = _load_extension()
    if module is None:
        return None

    return module.int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )
