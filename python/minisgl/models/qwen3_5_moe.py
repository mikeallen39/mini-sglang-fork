from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.kernel import silu_and_mul_quant_int8_triton
from minisgl.linear_attention import (
    fused_gdn_gating_sglang,
    fused_linear_attn_decode_sglang,
    fused_linear_attn_prefill_sglang,
)
from minisgl.layers import (
    BaseOP,
    GemmaRMSNorm,
    GemmaRMSNormFused,
    LinearOProj,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.quantization import (
    apply_w8a8_int8_linear_from_prequantized,
    quantize_activation_per_token_int8,
)
from minisgl.utils import div_even, get_linear_attn_backend, local_kv_heads, nvtx_annotate
from minisgl.utils.logger import init_logger

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig

logger = init_logger(__name__)

_SPARSE_MOE_PROFILE = {
    "router_ms": 0.0,
    "experts_ms": 0.0,
    "shared_ms": 0.0,
    "count": 0,
}
_SPARSE_MOE_PROFILE_INTERVAL = 100
_LINEAR_ATTN_PROFILE = {
    "prefill": {
        "quant_ms": 0.0,
        "qkvz_ms": 0.0,
        "ba_ms": 0.0,
        "conv_ms": 0.0,
        "kernel_ms": 0.0,
        "norm_ms": 0.0,
        "out_proj_ms": 0.0,
        "count": 0,
    },
    "decode": {
        "quant_ms": 0.0,
        "qkvz_ms": 0.0,
        "ba_ms": 0.0,
        "conv_ms": 0.0,
        "kernel_ms": 0.0,
        "norm_ms": 0.0,
        "out_proj_ms": 0.0,
        "count": 0,
    },
}
_LINEAR_ATTN_PROFILE_INTERVAL = 100


def _can_profile_cuda_events(x: torch.Tensor) -> bool:
    return x.is_cuda and not torch.cuda.is_current_stream_capturing()


def _freeze_rope_scaling(scaling: dict | None) -> tuple[tuple[str, object], ...] | None:
    if scaling is None:
        return None
    frozen = []
    for key, value in scaling.items():
        if isinstance(value, list):
            value = tuple(value)
        frozen.append((key, value))
    return tuple(frozen)


class Qwen3_5LinearStateCache:
    def __init__(self):
        self._states: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def get(
        self,
        layer_id: int,
        table_idx: int,
        *,
        conv_dim: int,
        conv_kernel_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_states = self._states.setdefault(layer_id, {})
        state = layer_states.get(table_idx)
        if state is not None:
            return state

        conv_state = torch.zeros(
            conv_dim,
            conv_kernel_size - 1,
            dtype=torch.bfloat16,
            device=device,
        )
        ssm_state = torch.zeros(
            num_v_heads,
            head_k_dim,
            head_v_dim,
            dtype=torch.float32,
            device=device,
        )
        layer_states[table_idx] = (conv_state, ssm_state)
        return layer_states[table_idx]

    def clear(self, table_idx: int) -> None:
        for layer_states in self._states.values():
            layer_states.pop(table_idx, None)

    def get_existing(
        self, layer_id: int, table_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        layer_states = self._states.get(layer_id)
        if layer_states is None:
            return None
        return layer_states.get(table_idx)

    def swap_states(self, layer_id: int, src_table_idx: int, dst_table_idx: int) -> None:
        src = self.get_existing(layer_id, src_table_idx)
        dst = self.get_existing(layer_id, dst_table_idx)
        if src is None or dst is None:
            return
        dst[0].copy_(src[0])
        dst[1].copy_(src[1])


class Qwen3_5RMSNormGated(BaseOP):
    def __init__(self, hidden_size: int, eps: float):
        self.weight = torch.empty(hidden_size)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        out_dtype = gate.dtype
        x = x.reshape(-1, x_shape[-1])
        gate = gate.reshape(-1, x_shape[-1])
        compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x = x.to(compute_dtype)
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = (x * self.weight.to(compute_dtype)).to(out_dtype)
        x = x * F.silu(gate)
        return x.reshape(x_shape)


class Qwen3_5SparseMoeBlock(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = __import__("minisgl.layers", fromlist=["MoELayer"]).MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )
        self.gate.disable_int8_quantization = True
        self.shared_expert = None
        self.shared_expert_gate = None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3_5SharedExpert(config)
            self.shared_expert_gate = LinearReplicated(
                config.hidden_size,
                1,
                has_bias=False,
            )
            self.shared_expert_gate.disable_int8_quantization = True

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        profile_enabled = ENV.PROFILE_SPARSE_MOE.value and _can_profile_cuda_events(hidden_states)
        if profile_enabled:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            e0.record()
        hidden_states_q = None
        hidden_states_scale = None
        if (
            self.gate.weight.dtype == torch.int8
            or (self.shared_expert is not None and self.shared_expert.gate_up_proj.weight.dtype == torch.int8)
            or (
                self.shared_expert_gate is not None
                and self.shared_expert_gate.weight.dtype == torch.int8
            )
        ):
            hidden_states_q, hidden_states_scale = quantize_activation_per_token_int8(hidden_states)

        router_logits = self.gate.forward_prequantized(
            hidden_states, hidden_states_q, hidden_states_scale
        )
        if profile_enabled:
            e1.record()
        output = self.experts.forward(hidden_states, router_logits)
        if profile_enabled:
            e2.record()
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_output = self.shared_expert.forward(
                hidden_states,
                hidden_states_q=hidden_states_q,
                hidden_states_scale=hidden_states_scale,
            )
            shared_gate = self.shared_expert_gate.forward_prequantized(
                hidden_states, hidden_states_q, hidden_states_scale
            )
            shared_output = torch.sigmoid(shared_gate) * shared_output
            output = output + shared_output
        if profile_enabled:
            e3.record()
            e3.synchronize()
            _SPARSE_MOE_PROFILE["router_ms"] += e0.elapsed_time(e1)
            _SPARSE_MOE_PROFILE["experts_ms"] += e1.elapsed_time(e2)
            _SPARSE_MOE_PROFILE["shared_ms"] += e2.elapsed_time(e3)
            _SPARSE_MOE_PROFILE["count"] += 1
            if _SPARSE_MOE_PROFILE["count"] % _SPARSE_MOE_PROFILE_INTERVAL == 0:
                count = _SPARSE_MOE_PROFILE["count"]
                logger.info_rank0(
                    "SparseMoE profile avg: router=%.4f ms, experts=%.4f ms, shared=%.4f ms over %d calls",
                    _SPARSE_MOE_PROFILE["router_ms"] / count,
                    _SPARSE_MOE_PROFILE["experts_ms"] / count,
                    _SPARSE_MOE_PROFILE["shared_ms"] / count,
                    count,
                )
                _SPARSE_MOE_PROFILE["router_ms"] = 0.0
                _SPARSE_MOE_PROFILE["experts_ms"] = 0.0
                _SPARSE_MOE_PROFILE["shared_ms"] = 0.0
                _SPARSE_MOE_PROFILE["count"] = 0
        return output


class Qwen3_5Conv1dWeight(BaseOP):
    def __init__(self, output_size: int, kernel_size: int):
        tp_size = get_tp_info().size
        self.weight = torch.empty(div_even(output_size, tp_size), 1, kernel_size)


class Qwen3_5LocalQKVProj(BaseOP):
    def __init__(self, input_size: int, q_output_size: int, kv_output_size: int):
        self.weight = torch.empty(q_output_size + 2 * kv_output_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def weight_loader(self, loaded_weight: torch.Tensor) -> None:
        if self.weight.is_meta:
            self.weight = loaded_weight
        else:
            self.weight.copy_(loaded_weight)


class Qwen3_5SharedExpert(BaseOP):
    def __init__(self, config: ModelConfig):
        from minisgl.layers import LinearColParallelMerged

        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.shared_expert_intermediate_size, config.shared_expert_intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            config.shared_expert_intermediate_size,
            config.hidden_size,
            has_bias=False,
        )
        self.gate_up_proj.quantize_in_moe_only = True
        self.down_proj.quantize_in_moe_only = True
        self.gate_up_proj.disable_int8_quantization = True
        self.down_proj.disable_int8_quantization = True

    def forward(
        self,
        x: torch.Tensor,
        *,
        hidden_states_q: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.gate_up_proj.weight.dtype == torch.int8
            and self.gate_up_proj.weight_scale is not None
            and self.down_proj.weight.dtype == torch.int8
            and self.down_proj.weight_scale is not None
            and x.is_cuda
            and x.ndim == 2
            and x.is_contiguous()
        ):
            gate_up = self.gate_up_proj.forward_prequantized(x, hidden_states_q, hidden_states_scale)
            inter_q = torch.empty(
                (gate_up.shape[0], gate_up.shape[1] // 2),
                device=gate_up.device,
                dtype=torch.int8,
            )
            inter_s = torch.empty((gate_up.shape[0], 1), device=gate_up.device, dtype=torch.float32)
            silu_and_mul_quant_int8_triton(gate_up, inter_q, inter_s)
            return apply_w8a8_int8_linear_from_prequantized(
                inter_q,
                inter_s,
                self.down_proj.weight,
                self.down_proj.weight_scale,
                out_dtype=x.dtype,
                bias=self.down_proj.bias,
            )
        gate_up = self.gate_up_proj.forward_prequantized(x, hidden_states_q, hidden_states_scale)
        return self.down_proj.forward(silu_and_mul(gate_up))


class Qwen3_5FullAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.attn_output_gate = config.attn_output_gate
        tp_size = get_tp_info().size
        self.local_num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.local_num_kv_heads = local_kv_heads(config.num_kv_heads, tp_size)
        self.q_dim = self.local_num_qo_heads * self.head_dim
        self.kv_dim = self.local_num_kv_heads * self.head_dim

        from minisgl.layers import get_rope

        self.qkv_proj = Qwen3_5LocalQKVProj(
            config.hidden_size,
            self.q_dim * (2 if config.attn_output_gate else 1),
            self.kv_dim,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=config.head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=_freeze_rope_scaling(config.rotary_config.scaling),
            is_neox=config.rotary_config.is_neox,
        )
        self.o_proj = LinearOProj(
            config.head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        if self.attn_output_gate:
            q_and_gate, k, v = qkv.split([self.q_dim * 2, self.kv_dim, self.kv_dim], dim=-1)
            orig_shape = q_and_gate.shape[:-1]
            q_and_gate = q_and_gate.view(*orig_shape, self.local_num_qo_heads, -1)
            q, gate = torch.chunk(q_and_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            gate = None

        self.q_norm.forward_inplace(q.view(-1, self.local_num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.local_num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.local_num_qo_heads, self.head_dim)
        attn_output = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        attn_output = attn_output.view(-1, self.q_dim)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj.forward(attn_output)


class Qwen3_5LinearAttention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        state_cache: Qwen3_5LinearStateCache,
    ):
        self.layer_id = layer_id
        self.state_cache = state_cache
        self.tp_size = get_tp_info().size
        self.num_q_heads = div_even(config.linear_num_key_heads, self.tp_size)
        self.num_k_heads = div_even(config.linear_num_key_heads, self.tp_size)
        self.num_v_heads = div_even(config.linear_num_value_heads, self.tp_size)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        self.value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        self.local_key_dim = self.num_k_heads * self.head_k_dim
        self.local_value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.scale = self.head_k_dim**-0.5
        self.activation = config.hidden_act
        self.conv_dim = 2 * self.local_key_dim + self.local_value_dim
        self.conv1d = Qwen3_5Conv1dWeight(
            output_size=self.key_dim * 2 + self.value_dim,
            kernel_size=self.conv_kernel_size,
        )
        from minisgl.layers import LinearColParallelMerged

        self.in_proj_qkvz = LinearColParallelMerged(
            config.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            has_bias=False,
        )
        self.in_proj_ba = LinearColParallelMerged(
            config.hidden_size,
            [config.linear_num_value_heads, config.linear_num_value_heads],
            has_bias=False,
        )
        self.in_proj_ba.disable_int8_quantization = True
        self.norm = Qwen3_5RMSNormGated(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearRowParallel(
            self.value_dim,
            config.hidden_size,
            has_bias=False,
        )
        self.A_log = torch.empty(self.num_v_heads)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.backend = get_linear_attn_backend()
        self._gather_idx: torch.Tensor | None = None
        self._decode_state_index: torch.Tensor | None = None
        self._A_log_fp32: torch.Tensor | None = None
        self._dt_bias_fp32: torch.Tensor | None = None

        assert self.num_v_heads % self.num_k_heads == 0, "Expected grouped value heads."
        self.kv_group_size = self.num_v_heads // self.num_k_heads

    def _run_depthwise_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_weight = self.conv1d.weight.squeeze(1).to(dtype=mixed_qkv.dtype)
        x = mixed_qkv.transpose(0, 1)
        padded = torch.cat((conv_state.to(dtype=x.dtype), x), dim=-1).unsqueeze(0)
        conv = F.conv1d(
            padded,
            conv_weight.unsqueeze(1),
            bias=None,
            groups=conv_weight.shape[0],
        ).squeeze(0).transpose(0, 1).contiguous()
        if self.activation == "silu":
            conv = F.silu(conv)
        elif self.activation == "swish":
            conv = conv * torch.sigmoid(conv)
        elif self.activation != "identity":
            raise ValueError(f"Unsupported linear attention activation: {self.activation}")
        next_state = padded.squeeze(0)[:, -(self.conv_kernel_size - 1) :].to(dtype=conv_state.dtype)
        return conv, next_state

    def _reshape_qkv(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = mixed_qkv.split(
            [self.local_key_dim, self.local_key_dim, self.local_value_dim], dim=-1
        )
        query = query.view(-1, self.num_q_heads, self.head_k_dim)
        key = key.view(-1, self.num_k_heads, self.head_k_dim)
        value = value.view(-1, self.num_v_heads, self.head_v_dim)
        z = z.view(-1, self.num_v_heads, self.head_v_dim)
        gather_idx = self._get_gather_idx(query.device)
        query = query[:, gather_idx, :]
        key = key[:, gather_idx, :]
        return query, key, value, z

    def _get_gather_idx(self, device: torch.device) -> torch.Tensor:
        if self._gather_idx is None or self._gather_idx.device != device:
            self._gather_idx = torch.arange(self.num_v_heads, device=device) // self.kv_group_size
        return self._gather_idx

    def _get_decode_state_index(self, device: torch.device) -> torch.Tensor:
        if self._decode_state_index is None or self._decode_state_index.device != device:
            self._decode_state_index = torch.tensor([0], dtype=torch.int32, device=device)
        return self._decode_state_index

    def _get_A_log_fp32(self) -> torch.Tensor:
        if self._A_log_fp32 is None or self._A_log_fp32.device != self.A_log.device:
            self._A_log_fp32 = self.A_log.float().contiguous()
        return self._A_log_fp32

    def _get_dt_bias_fp32(self) -> torch.Tensor:
        if self._dt_bias_fp32 is None or self._dt_bias_fp32.device != self.dt_bias.device:
            self._dt_bias_fp32 = self.dt_bias.float().contiguous()
        return self._dt_bias_fp32

    def process_weights_after_loading(self) -> None:
        self._A_log_fp32 = self.A_log.float().contiguous()
        self._dt_bias_fp32 = self.dt_bias.float().contiguous()
        self._gather_idx = torch.arange(self.num_v_heads, device=self.A_log.device) // self.kv_group_size
        self._decode_state_index = torch.tensor([0], dtype=torch.int32, device=self.A_log.device)

    def copy_state(self, src_table_idx: int, dst_table_idx: int) -> None:
        self.state_cache.swap_states(self.layer_id, src_table_idx, dst_table_idx)

    def _forward_one_req(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        table_idx: int,
    ) -> torch.Tensor:
        conv_state, ssm_state = self.state_cache.get(
            self.layer_id,
            table_idx,
            conv_dim=self.conv_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            device=mixed_qkv.device,
        )
        mixed_qkv, next_conv_state = self._run_depthwise_conv(mixed_qkv, conv_state)
        conv_state.copy_(next_conv_state)

        query, key, value, z = self._reshape_qkv(mixed_qkv, z)
        query = F.normalize(query.float(), dim=-1, eps=1e-6) * self.scale
        key = F.normalize(key.float(), dim=-1, eps=1e-6)
        value = value.float()
        gate = -torch.exp(self.A_log.float()).unsqueeze(0) * F.softplus(
            a.float() + self.dt_bias.float().unsqueeze(0)
        )
        beta = torch.sigmoid(b.float())

        outputs = torch.empty_like(value)
        for i in range(query.shape[0]):
            ssm_state.mul_(torch.exp(gate[i]).view(-1, 1, 1))
            value_residual = value[i] - torch.einsum("hkv,hk->hv", ssm_state, key[i])
            value_residual = value_residual * beta[i].unsqueeze(-1)
            ssm_state.add_(key[i].unsqueeze(-1) * value_residual.unsqueeze(-2))
            outputs[i] = torch.einsum("hk,hkv->hv", query[i], ssm_state).to(outputs.dtype)

        outputs = self.norm.forward(outputs, z)
        return self.out_proj.forward(outputs.reshape(outputs.shape[0], -1).contiguous())

    def _forward_one_req_sglang(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        table_idx: int,
        *,
        is_decode: bool,
        profile_events: tuple[torch.cuda.Event, ...] | None = None,
    ) -> torch.Tensor:
        conv_state, ssm_state = self.state_cache.get(
            self.layer_id,
            table_idx,
            conv_dim=self.conv_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            device=mixed_qkv.device,
        )
        if profile_events is not None:
            conv_start, conv_end, kernel_end, norm_end, out_end = profile_events
            conv_start.record()
        mixed_qkv, next_conv_state = self._run_depthwise_conv(mixed_qkv, conv_state)
        conv_state.copy_(next_conv_state)
        if profile_events is not None:
            conv_end.record()

        a = a.float().contiguous()
        b = b.float().contiguous()
        A_log = self._get_A_log_fp32()
        dt_bias = self._get_dt_bias_fp32()
        if is_decode:
            if mixed_qkv.shape[0] != 1:
                raise ValueError(
                    "Decode batches must have exactly one extend token per request for linear attention"
                )
            state = ssm_state.unsqueeze(0)
            state_indices = self._get_decode_state_index(mixed_qkv.device)
            outputs = fused_linear_attn_decode_sglang(
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                state,
                state_indices,
                self.scale,
            )
        else:
            query, key, value, _ = self._reshape_qkv(mixed_qkv, z)
            query = F.normalize(query.float(), dim=-1, eps=1e-6).to(query.dtype).contiguous()
            key = F.normalize(key.float(), dim=-1, eps=1e-6).to(key.dtype).contiguous()
            gate, beta = fused_gdn_gating_sglang(A_log, a, b, dt_bias)
            outputs = fused_linear_attn_prefill_sglang(
                query,
                key,
                value.float().contiguous(),
                gate.contiguous(),
                beta.contiguous(),
                ssm_state,
                self.scale,
                use_qk_l2norm_in_kernel=False,
            )
        if profile_events is not None:
            kernel_end.record()
        outputs = self.norm.forward(outputs, z)
        if profile_events is not None:
            norm_end.record()
        outputs = self.out_proj.forward(outputs.reshape(outputs.shape[0], -1).contiguous())
        if profile_events is not None:
            out_end.record()
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        profile_enabled = ENV.PROFILE_QWEN35.value and _can_profile_cuda_events(x)
        profile_bucket = "decode" if batch.is_decode else "prefill"
        quant_events = proj_events = req_events = None
        quant_recorded = False
        if profile_enabled:
            quant_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            proj_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            req_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        x_q = None
        x_scale = None
        if self.in_proj_qkvz.weight.dtype == torch.int8 or self.in_proj_ba.weight.dtype == torch.int8:
            if profile_enabled:
                assert quant_events is not None
                quant_events[0].record()
            x_q, x_scale = quantize_activation_per_token_int8(x)
            if profile_enabled:
                quant_events[1].record()
                quant_recorded = True
        if profile_enabled:
            assert proj_events is not None
            proj_events[0].record()
        mixed_qkvz = self.in_proj_qkvz.forward_prequantized(x, x_q, x_scale)
        if profile_enabled:
            proj_events[1].record()
        mixed_ba = self.in_proj_ba.forward_prequantized(x, x_q, x_scale)
        if profile_enabled:
            proj_events[2].record()
        mixed_qkv, z = mixed_qkvz.split(
            [self.local_key_dim * 2 + self.local_value_dim, self.local_value_dim], dim=-1
        )
        b, a = mixed_ba.split([self.num_v_heads, self.num_v_heads], dim=-1)

        outputs = []
        offset = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            req_slice = slice(offset, offset + length)
            if self.backend == "torch":
                outputs.append(
                    self._forward_one_req(
                        mixed_qkv[req_slice],
                        z[req_slice],
                        a[req_slice],
                        b[req_slice],
                        req.table_idx,
                    )
                )
            elif self.backend == "sglang":
                outputs.append(
                    self._forward_one_req_sglang(
                        mixed_qkv[req_slice],
                        z[req_slice],
                        a[req_slice],
                        b[req_slice],
                        req.table_idx,
                        is_decode=batch.is_decode,
                        profile_events=req_events if offset == 0 and profile_enabled else None,
                    )
                )
            else:
                raise ValueError(f"Unsupported linear attention backend: {self.backend}")
            offset += length
        if profile_enabled:
            assert proj_events is not None and req_events is not None
            req_events[-1].synchronize()
            bucket = _LINEAR_ATTN_PROFILE[profile_bucket]
            if quant_recorded and quant_events is not None:
                bucket["quant_ms"] += quant_events[0].elapsed_time(quant_events[1])
            bucket["qkvz_ms"] += proj_events[0].elapsed_time(proj_events[1])
            bucket["ba_ms"] += proj_events[1].elapsed_time(proj_events[2])
            bucket["conv_ms"] += req_events[0].elapsed_time(req_events[1])
            bucket["kernel_ms"] += req_events[1].elapsed_time(req_events[2])
            bucket["norm_ms"] += req_events[2].elapsed_time(req_events[3])
            bucket["out_proj_ms"] += req_events[3].elapsed_time(req_events[4])
            bucket["count"] += 1
            if bucket["count"] % _LINEAR_ATTN_PROFILE_INTERVAL == 0:
                count = bucket["count"]
                logger.info_rank0(
                    "LinearAttn %s profile avg: quant=%.4f ms, qkvz=%.4f ms, ba=%.4f ms, conv=%.4f ms, kernel=%.4f ms, norm=%.4f ms, out_proj=%.4f ms over %d calls",
                    profile_bucket,
                    bucket["quant_ms"] / count,
                    bucket["qkvz_ms"] / count,
                    bucket["ba_ms"] / count,
                    bucket["conv_ms"] / count,
                    bucket["kernel_ms"] / count,
                    bucket["norm_ms"] / count,
                    bucket["out_proj_ms"] / count,
                    count,
                )
                bucket["quant_ms"] = 0.0
                bucket["qkvz_ms"] = 0.0
                bucket["ba_ms"] = 0.0
                bucket["conv_ms"] = 0.0
                bucket["kernel_ms"] = 0.0
                bucket["norm_ms"] = 0.0
                bucket["out_proj_ms"] = 0.0
                bucket["count"] = 0
        return torch.cat(outputs, dim=0)


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        state_cache: Qwen3_5LinearStateCache,
    ):
        layer_types = config.layer_types or ["full_attention"] * config.num_layers
        self.layer_type = layer_types[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, layer_id)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id, state_cache)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        self.mlp = (
            Qwen3_5SparseMoeBlock(config)
            if config.is_moe
            else GatedMLP(config)
        )
        self.input_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        profile_enabled = ENV.PROFILE_QWEN35.value and _can_profile_cuda_events(x)
        if profile_enabled:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t0.record()
        if self.layer_type == "full_attention":
            x = self.self_attn.forward(x)
        else:
            x = self.linear_attn.forward(x)
        if profile_enabled:
            t1.record()
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        if profile_enabled:
            t2.record()
            t3.record()
            t3.synchronize()
            stats = getattr(self, "_profile_stats", None)
            if stats is None:
                stats = self._profile_stats = {
                    "full_attn_ms": 0.0,
                    "linear_attn_ms": 0.0,
                    "mlp_ms": 0.0,
                    "count": 0,
                }
            if self.layer_type == "full_attention":
                stats["full_attn_ms"] += t0.elapsed_time(t1)
            else:
                stats["linear_attn_ms"] += t0.elapsed_time(t1)
            stats["mlp_ms"] += t1.elapsed_time(t2)
            stats["count"] += 1
        return x, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.linear_state_cache = Qwen3_5LinearStateCache()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, self.linear_state_cache)
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._last_captured_hidden_states: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        batch = get_global_ctx().batch
        capture_hidden_layer_ids = (
            set(batch.capture_hidden_layer_ids) if batch.capture_hidden_layer_ids else None
        )
        captured_hidden_states: list[torch.Tensor] = []
        self._last_captured_hidden_states = None
        x = self.embed_tokens.forward(input_ids) if inputs_embeds is None else inputs_embeds
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if capture_hidden_layer_ids is not None and layer_id in capture_hidden_layer_ids:
                captured_hidden_states.append(x)
        final_hidden = self.norm.forward(x, residual)[0]
        if capture_hidden_layer_ids is None:
            return final_hidden, None
        if len(captured_hidden_states) == 0:
            hidden = final_hidden.new_empty((final_hidden.shape[0], 0))
            self._last_captured_hidden_states = hidden
            return final_hidden, hidden
        hidden = torch.cat(captured_hidden_states, dim=-1)
        self._last_captured_hidden_states = hidden
        return final_hidden, hidden


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.pixel_values is not None:
            raise NotImplementedError(
                "Multimodal request reached the Qwen3.5 text-only path. "
                "A Qwen3-VL model implementation is still required."
            )
        output, hidden_states = self.model.forward(batch.input_ids)
        self._last_hidden_capture = hidden_states
        if ENV.PROFILE_QWEN35.value:
            self._maybe_log_profile()
        return self.lm_head.forward(output)

    def get_last_hidden_capture(self) -> torch.Tensor | None:
        return getattr(self, "_last_hidden_capture", None)

    @property
    def supports_prefix_cache(self) -> bool:
        return False

    @property
    def supports_cuda_graph(self) -> bool:
        return self._supports_cuda_graph()

    def _supports_cuda_graph(self) -> bool:
        if get_linear_attn_backend() != "sglang":
            return False
        return True

    def prepare_for_cuda_graph_replay(self, batch, dummy_reqs) -> None:
        if not batch.is_decode:
            return
        for real_req, dummy_req in zip(batch.reqs, dummy_reqs, strict=False):
            if real_req.table_idx == dummy_req.table_idx:
                continue
            for layer in self.model.layers.op_list:
                linear_attn = getattr(layer, "linear_attn", None)
                if linear_attn is not None:
                    linear_attn.copy_state(real_req.table_idx, dummy_req.table_idx)

    def finish_cuda_graph_replay(self, batch, dummy_reqs) -> None:
        if not batch.is_decode:
            return
        for real_req, dummy_req in zip(batch.reqs, dummy_reqs, strict=False):
            if real_req.table_idx == dummy_req.table_idx:
                continue
            for layer in self.model.layers.op_list:
                linear_attn = getattr(layer, "linear_attn", None)
                if linear_attn is not None:
                    linear_attn.copy_state(dummy_req.table_idx, real_req.table_idx)

    def clear_runtime_state_slot(self, table_idx: int) -> None:
        self.model.linear_state_cache.clear(table_idx)

    def copy_runtime_state_slot(self, src_table_idx: int, dst_table_idx: int) -> None:
        for layer in self.model.layers.op_list:
            linear_attn = getattr(layer, "linear_attn", None)
            if linear_attn is not None:
                linear_attn.copy_state(src_table_idx, dst_table_idx)

    def _maybe_log_profile(self) -> None:
        counter = getattr(self, "_profile_counter", 0) + 1
        self._profile_counter = counter
        if counter % 10 != 0:
            return

        full_attn_ms = 0.0
        linear_attn_ms = 0.0
        mlp_ms = 0.0
        count = 0
        for layer in self.model.layers.op_list:
            stats = getattr(layer, "_profile_stats", None)
            if not stats:
                continue
            full_attn_ms += stats["full_attn_ms"]
            linear_attn_ms += stats["linear_attn_ms"]
            mlp_ms += stats["mlp_ms"]
            count += stats["count"]
            stats["full_attn_ms"] = 0.0
            stats["linear_attn_ms"] = 0.0
            stats["mlp_ms"] = 0.0
            stats["count"] = 0

        if count == 0:
            return

        logger.info_rank0(
            "Qwen3.5 profile avg per layer-call: full_attn=%.4f ms, linear_attn=%.4f ms, mlp=%.4f ms over %d calls",
            full_attn_ms / count,
            linear_attn_ms / count,
            mlp_ms / count,
            count,
        )


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    pass


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    pass


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForCausalLM):
    pass


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
]
