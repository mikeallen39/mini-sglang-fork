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
    FI_GRAPH_FAST_DECODE_PLAN = EnvBool(False)  # use flashinfer fast_decode_plan on decode cuda-graph replay
    FI_GRAPH_REUSE_METADATA = EnvBool(False)  # reuse decode cuda-graph attention metadata buffers instead of rebuilding tensors each step
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    OVERLAP_EXTRA_SYNC = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    PROFILE_QWEN35 = EnvBool(False)
    PROFILE_FUSED_MOE = EnvBool(False)
    PROFILE_SPARSE_MOE = EnvBool(False)
    PROFILE_INT8_DENSE = EnvBool(False)
    MOE_REUSE_WORKSPACE = EnvBool(False)  # reuse topk/alignment temporary buffers on the fused MoE path
    MOE_SKIP_TOPK_POST_RENORM = EnvBool(False)  # trust sgl_kernel.topk_softmax renormalize output and skip the extra Python-side renorm
    MOE_SKIP_TOPK_FP32_CAST = EnvBool(False)  # pass router logits to sgl_kernel.topk_softmax without an extra Python-side float() cast
    MOE_SKIP_DISPATCH_LOCAL_MASK = EnvBool(False)  # skip constructing an unused all-true local_mask on the single-rank fast path
    MOE_DIRECT_FASTPATH = EnvBool(False)  # bypass LocalExpertDispatchPlan on the fused MoE single-rank fast path
    MOE_ALIGN_SMALL_CAP = EnvBool(False)  # use sglang's smaller temporary-buffer upper bound when topk_ids.numel() < num_experts + 1
    MOE_SGLANG_CONFIG_LOOKUP = EnvBool(False)  # load routed-expert Triton config from sglang-style JSON tables when available
    MOE_SGLANG_DOWN_CONFIG = EnvBool(False)  # use a separate sglang-style down-proj Triton config for the second routed-expert GEMM
    MOE_SINGLE_KERNEL = EnvBool(False)  # fuse silu_and_mul into the second MoE GEMM
    MOE_FUSED_ACTIVATION = EnvBool(False)  # use sgl_kernel fused silu_and_mul / gelu_and_mul in MoE stage2
    MOE_SGL_REDUCE = EnvBool(False)  # use sgl_kernel moe_sum_reduce instead of local Triton reduce
    LINEAR_RMSNORM_GATED = EnvBool(False)  # use fused RMSNorm+gate for Qwen3.6 linear attention output norm
    SHARED_EXPERT_FUSED_GATE_ADD = EnvBool(False)  # fuse sigmoid(gate)*shared_output + moe_output for shared expert
    SHARED_EXPERT_FUSED_ACTIVATION = EnvBool(False)  # use fused silu_and_mul inside shared expert bf16 path
    SHARED_EXPERT_DUAL_STREAM = EnvBool(False)  # overlap decode-time shared expert MLP with routed experts on an auxiliary CUDA stream
    SKIP_AB_FP32_CAST = EnvBool(False)  # keep a,b in bf16 for decode (Triton kernel loads as fp32 internally)
    DEPTHWISE_CONV_DECODE = EnvBool(False)  # use fused Triton depthwise conv for decode
    DEPTHWISE_CONV_PREFILL = EnvBool(False)  # use sglang causal_conv1d_fn for prefill depthwise conv
    FUSED_QKV_SPLIT = EnvBool(False)  # use fused QKV split kernel in GDN prefill
    LINEAR_PREFILL_QK_L2NORM = EnvBool(False)  # move Q/K l2norm from PyTorch prefill path into the linear-attn kernel
    LINEAR_PREFILL_SKIP_REDUNDANT_CONTIGUOUS = EnvBool(False)  # skip explicit prefill contiguous/cast wrappers that chunk_gated_delta_rule already guards
    LINEAR_DECODE_VK_STATE = EnvBool(False)  # keep an auxiliary [HV, V, K] state layout for decode fast path
    LINEAR_DECODE_SGLANG_PACKED = EnvBool(False)  # call sglang packed recurrent decode kernel on the auxiliary [HV, V, K] state
    LINEAR_DECODE_FUSED_INPUT_PROJ = EnvBool(False)  # fuse decode-time qkvz and ba projections into one GEMM in bf16 path
    LINEAR_DECODE_DUAL_STREAM_INPUT_PROJ = EnvBool(False)  # overlap decode-time qkvz and ba bf16 projections on two CUDA streams
    LINEAR_RMSNORM_GATED_REUSE_OUT = EnvBool(False)  # reuse decode norm output buffer for fused RMSNorm+gate
    LINEAR_DECODE_SKIP_OUTPROJ_CONTIGUOUS = EnvBool(False)  # skip explicit contiguous before linear-attention decode out_proj
    LINEAR_DECODE_SKIP_AB_CONTIGUOUS = EnvBool(False)  # skip explicit decode-time a/b contiguous when kernel can consume strided views
    LINEAR_DECODE_SKIP_CONV_STATE_COPY = EnvBool(False)  # skip redundant conv_state.copy_ after fused decode conv update
    LINEAR_DECODE_FUSED_QKV_SPLIT = EnvBool(False)  # use fused qkvz/ba split+reshape only for decode
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
