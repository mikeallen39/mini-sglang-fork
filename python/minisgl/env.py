from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar

from minisgl.utils.logger import init_logger

logger = init_logger(__name__)


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception as exc:
                logger.warning(
                    "Invalid environment variable %s=%r; keeping default value %r: %s",
                    name,
                    env_value,
                    self.value,
                    exc,
                )

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


MINISGL_ENV_PREFIX = "MINISGL_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    SHELL_TOP_K = EnvInt(-1)
    SHELL_TOP_P = EnvFloat(1.0)
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    OVERLAP_EXTRA_SYNC = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    PROFILE_QWEN35 = EnvBool(False)
    PROFILE_FUSED_MOE = EnvBool(False)
    PROFILE_SPARSE_MOE = EnvBool(False)
    PROFILE_INT8_DENSE = EnvBool(False)
    MOE_SINGLE_KERNEL = EnvBool(False)  # fuse silu_and_mul into the second MoE GEMM
    MOE_FUSED_ACTIVATION = EnvBool(False)  # use sgl_kernel fused silu_and_mul / gelu_and_mul in MoE stage2
    MOE_SGL_REDUCE = EnvBool(False)  # use sgl_kernel moe_sum_reduce instead of local Triton reduce
    LINEAR_RMSNORM_GATED = EnvBool(False)  # use fused RMSNorm+gate for Qwen3.6 linear attention output norm
    SHARED_EXPERT_FUSED_GATE_ADD = EnvBool(False)  # fuse sigmoid(gate)*shared_output + moe_output for shared expert
    SKIP_AB_FP32_CAST = EnvBool(False)  # keep a,b in bf16 for decode (Triton kernel loads as fp32 internally)
    DEPTHWISE_CONV_DECODE = EnvBool(False)  # use fused Triton depthwise conv for decode
    FUSED_QKV_SPLIT = EnvBool(False)  # use fused QKV split kernel in GDN prefill
    GEMMA_FUSED_NORM = EnvBool(False)  # use sgl_kernel gemma_rmsnorm / gemma_fused_add_rmsnorm
    FULL_ATTN_FUSED_PREPARE = EnvBool(False)  # fuse Q/K GemmaRMSNorm + RoPE + gate extraction in full attention
    FULL_ATTN_FUSED_GATE_MUL = EnvBool(False)  # fuse sigmoid(gate) * attn_output in full attention
    FULL_ATTN_SIGMOID_GATE = EnvBool(False)  # fuse sigmoid(gate) * attn_output via Triton sigmoid+multiply in one pass

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
