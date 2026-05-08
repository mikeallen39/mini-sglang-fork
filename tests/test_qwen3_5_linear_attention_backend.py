from __future__ import annotations

import torch

import minisgl.core as core
import minisgl.distributed.info as dist_info
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.linear_attention import set_linear_attn_backend
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.qwen3_5_moe import Qwen3_5LinearAttention, Qwen3_5LinearStateCache


def _reset_global_state():
    old_ctx = core._GLOBAL_CTX
    old_tp = dist_info._TP_INFO
    old_ep = dist_info._EP_INFO
    core._GLOBAL_CTX = None
    dist_info._TP_INFO = None
    dist_info._EP_INFO = None
    return old_ctx, old_tp, old_ep


def _restore_global_state(old_ctx, old_tp, old_ep) -> None:
    core._GLOBAL_CTX = old_ctx
    dist_info._TP_INFO = old_tp
    dist_info._EP_INFO = old_ep


def _set_tensor_attrs(
    obj: object,
    device: torch.device,
    dtype: torch.dtype,
    *,
    keep_fp32_names: set[str] | None = None,
) -> None:
    keep_fp32_names = keep_fp32_names or set()
    for name, value in list(vars(obj).items()):
        if not isinstance(value, torch.Tensor):
            continue
        target_dtype = torch.float32 if name in keep_fp32_names else dtype
        setattr(obj, name, torch.randn(value.shape, dtype=target_dtype, device=device))


def _build_layer(ctx: Context, device: torch.device, dtype: torch.dtype) -> Qwen3_5LinearAttention:
    dist_info.set_tp_info(0, 1)
    dist_info.set_ep_info(0, 1)
    set_linear_attn_backend("sglang")
    core.set_global_ctx(ctx)

    cfg = ModelConfig(
        num_layers=1,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=8,
        hidden_size=32,
        vocab_size=128,
        intermediate_size=64,
        rms_norm_eps=1e-5,
        rotary_config=RotaryConfig(
            head_dim=8,
            rotary_dim=8,
            max_position=128,
            base=10000.0,
            scaling=None,
            is_neox=True,
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )

    layer = Qwen3_5LinearAttention(cfg, 0, Qwen3_5LinearStateCache())
    _set_tensor_attrs(layer, device, dtype, keep_fp32_names={"A_log", "dt_bias"})
    for mod_name in ["conv1d", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "norm", "out_proj"]:
        _set_tensor_attrs(getattr(layer, mod_name), device, dtype)
    return layer


def _make_req(input_ids: list[int], cached_len: int, table_idx: int = 0) -> Req:
    return Req(
        input_ids=torch.tensor(input_ids, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=4,
        uid=1,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=None,
    )


def test_qwen3_5_linear_attention_sglang_prefill_decode() -> None:
    old_ctx, old_tp, old_ep = _reset_global_state()
    if not torch.cuda.is_available():
        _restore_global_state(old_ctx, old_tp, old_ep)
        raise RuntimeError("CUDA is required for this test")

    try:
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        ctx = Context(page_size=1)
        ctx.page_table = torch.zeros((4, 16), dtype=torch.int32, device=device)
        layer = _build_layer(ctx, device, dtype)

        prefill_batch = Batch(reqs=[_make_req([1, 2, 3], cached_len=0)], phase="prefill")
        prefill_batch.padded_reqs = prefill_batch.reqs
        with ctx.forward_batch(prefill_batch):
            x = torch.randn((3, 32), device=device, dtype=dtype)
            out = layer.forward(x)
        assert out.shape == (3, 32)

        decode_batch = Batch(reqs=[_make_req([1, 2, 3, 4], cached_len=3)], phase="decode")
        decode_batch.padded_reqs = decode_batch.reqs
        with ctx.forward_batch(decode_batch):
            x = torch.randn((1, 32), device=device, dtype=dtype)
            out = layer.forward(x)
        assert out.shape == (1, 32)
    finally:
        _restore_global_state(old_ctx, old_tp, old_ep)
