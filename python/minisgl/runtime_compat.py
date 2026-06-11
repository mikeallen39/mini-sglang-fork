from __future__ import annotations

import os
from pathlib import Path

from minisgl.utils.logger import init_logger

logger = init_logger(__name__)


def _prepend_env_path(name: str, value: str) -> None:
    current = os.environ.get(name)
    if not current:
        os.environ[name] = value
        return
    parts = current.split(":")
    if value in parts:
        return
    os.environ[name] = f"{value}:{current}"


def prepare_runtime_compat() -> None:
    preferred_cuda_home = None
    for candidate in ("/usr/local/cuda-12.6", "/usr/local/cuda-12.4", "/usr/local/cuda"):
        if Path(candidate).exists():
            preferred_cuda_home = candidate
            break

    # Ensure C++/CUDA extensions can resolve torch and CUDA shared libraries
    # in spawned worker processes.
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            _prepend_env_path("LD_LIBRARY_PATH", str(torch_lib))
    except Exception as exc:
        logger.warning(
            "Failed to locate torch shared-library directory for runtime compatibility setup: %s",
            exc,
        )

    for cuda_lib in (
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.4/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.6/targets/x86_64-linux/lib",
    ):
        if Path(cuda_lib).exists():
            _prepend_env_path("LD_LIBRARY_PATH", cuda_lib)

    # flashinfer works without the optional torch_c_dlpack_ext addon, while the
    # prebuilt addon in this environment is ABI-incompatible with torch 2.6.
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

    if preferred_cuda_home is not None:
        os.environ.setdefault("CUDA_HOME", preferred_cuda_home)
        os.environ.setdefault("CUDA_PATH", preferred_cuda_home)
        _prepend_env_path("PATH", str(Path(preferred_cuda_home) / "bin"))
        os.environ.setdefault("FLASHINFER_NVCC", str(Path(preferred_cuda_home) / "bin" / "nvcc"))
        cuda_lib_path = str(Path(preferred_cuda_home) / "targets" / "x86_64-linux" / "lib")
        if Path(cuda_lib_path).exists():
            os.environ.setdefault("CUDA_LIB_PATH", cuda_lib_path)

    # sgl-kernel 0.3.x incorrectly maps all non-sm90 GPUs to the sm100 common_ops
    # directory. On A800 (sm80), the sm90 variant is the compatible fallback.
    os.environ.setdefault("SGL_KERNEL_FORCE_OPS_SUBDIR", "sm90")
