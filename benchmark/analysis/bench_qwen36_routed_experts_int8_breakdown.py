from __future__ import annotations

import argparse
import json
import os

import torch

from minisgl.benchmark.perf import perf_cuda
from minisgl.moe.dispatch import build_local_expert_dispatch_plan
from minisgl.moe.fused import (
    fused_experts_impl,
    fused_topk,
    get_fused_moe_profile,
    reset_fused_moe_profile,
)
from minisgl.moe.torch_backend import TorchMoe
from minisgl.models.config import ModelConfig
from minisgl.quantization import quantize_weight_per_channel_int8_row_major
from minisgl.utils import cached_load_hf_config

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")


def resolve_config(args: argparse.Namespace) -> dict[str, int | bool | str | None]:
    model_cfg = None
    if args.model_path:
        model_cfg = ModelConfig.from_hf(cached_load_hf_config(args.model_path))

    hidden_size = (
        args.hidden_size
        if args.hidden_size is not None
        else (model_cfg.hidden_size if model_cfg is not None else 2048)
    )
    intermediate_size = (
        args.intermediate_size
        if args.intermediate_size is not None
        else (model_cfg.moe_intermediate_size if model_cfg is not None else 512)
    )
    experts = (
        args.experts
        if args.experts is not None
        else (model_cfg.num_experts if model_cfg is not None else 256)
    )
    topk = (
        args.topk
        if args.topk is not None
        else (model_cfg.num_experts_per_tok if model_cfg is not None else 8)
    )
    renormalize = (
        args.renormalize
        if args.renormalize is not None
        else (model_cfg.norm_topk_prob if model_cfg is not None else True)
    )
    return {
        "model_path": args.model_path,
        "model_type": model_cfg.model_type if model_cfg is not None else None,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "experts": experts,
        "topk": topk,
        "renormalize": renormalize,
    }


def build_inputs(
    *,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
):
    hidden_states = torch.randn((m, hidden_size), device=device, dtype=dtype)
    gating_output = torch.randn((m, num_experts), device=device, dtype=dtype)
    w1_bf16 = torch.randn(
        (num_experts, intermediate_size * 2, hidden_size),
        device=device,
        dtype=dtype,
    )
    w2_bf16 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )

    w1_q = []
    w1_s = []
    w2_q = []
    w2_s = []
    for expert_id in range(num_experts):
        q_w1, s_w1 = quantize_weight_per_channel_int8_row_major(w1_bf16[expert_id])
        q_w2, s_w2 = quantize_weight_per_channel_int8_row_major(w2_bf16[expert_id])
        w1_q.append(q_w1)
        w1_s.append(s_w1.squeeze(-1))
        w2_q.append(q_w2)
        w2_s.append(s_w2.squeeze(-1))

    return (
        hidden_states,
        gating_output,
        w1_bf16,
        w2_bf16,
        torch.stack(w1_q, dim=0).contiguous(),
        torch.stack(w1_s, dim=0).contiguous(),
        torch.stack(w2_q, dim=0).contiguous(),
        torch.stack(w2_s, dim=0).contiguous(),
    )


def compute_dispatch_plan(
    *,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_experts: int,
):
    topk_weights, topk_ids = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    return build_local_expert_dispatch_plan(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        local_expert_start=0,
        num_local_experts=num_experts,
        num_global_experts=num_experts,
    )


def bench_torch_experts(
    *,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    repetitions: int,
) -> float:
    backend = TorchMoe()

    def run():
        backend.forward(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            activation="silu",
            apply_router_weight_on_input=False,
            local_expert_start=0,
            num_global_experts=w1.shape[0],
            num_dispatch_experts=w1.shape[0],
        )

    return perf_cuda(run, repetitions=repetitions, cuda_graph_repetitions=None)


def bench_fused_experts(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    repetitions: int,
) -> float:
    def run():
        fused_experts_impl(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
        )

    return perf_cuda(run, repetitions=repetitions, cuda_graph_repetitions=None)


def profile_fused_experts(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    repetitions: int,
) -> dict[str, float]:
    from minisgl.env import ENV

    prev = ENV.PROFILE_FUSED_MOE.value
    ENV.PROFILE_FUSED_MOE.value = True
    reset_fused_moe_profile()
    try:
        for _ in range(repetitions):
            fused_experts_impl(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
            )
        torch.cuda.synchronize()
        prof = get_fused_moe_profile()
    finally:
        reset_fused_moe_profile()
        ENV.PROFILE_FUSED_MOE.value = prev

    count = int(prof["count"])
    if count <= 0:
        return {
            "w1_ms": 0.0,
            "stage2_ms": 0.0,
            "w2_ms": 0.0,
            "reduce_ms": 0.0,
        }
    return {
        "w1_ms": float(prof["w1_ms"]) / count,
        "stage2_ms": float(prof["stage2_ms"]) / count,
        "w2_ms": float(prof["w2_ms"]) / count,
        "reduce_ms": float(prof["reduce_ms"]) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--model-path")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--renormalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 1024])
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()

    cfg = resolve_config(args)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": args.dtype,
                **cfg,
                "repetitions": args.repetitions,
                "note": "This benchmark isolates routed experts only. Router/topk is computed once to build the dispatch plan, then bf16/int8 expert paths are compared separately.",
            },
            ensure_ascii=False,
        )
    )

    for m in args.tokens:
        (
            hidden_states,
            gating_output,
            w1_bf16,
            w2_bf16,
            w1_int8,
            w1_scale,
            w2_int8,
            w2_scale,
        ) = build_inputs(
            m=m,
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_experts=int(cfg["experts"]),
            topk=int(cfg["topk"]),
            device=device,
            dtype=dtype,
        )
        dispatch = compute_dispatch_plan(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=int(cfg["topk"]),
            renormalize=bool(cfg["renormalize"]),
            num_experts=int(cfg["experts"]),
        )

        torch_bf16_ms = bench_torch_experts(
            hidden_states=hidden_states,
            gating_output=gating_output,
            w1=w1_bf16,
            w1_scale=None,
            w2=w2_bf16,
            w2_scale=None,
            topk=int(cfg["topk"]),
            renormalize=bool(cfg["renormalize"]),
            repetitions=args.repetitions,
        )
        torch_int8_ms = bench_torch_experts(
            hidden_states=hidden_states,
            gating_output=gating_output,
            w1=w1_int8,
            w1_scale=w1_scale,
            w2=w2_int8,
            w2_scale=w2_scale,
            topk=int(cfg["topk"]),
            renormalize=bool(cfg["renormalize"]),
            repetitions=args.repetitions,
        )
        fused_bf16_ms = bench_fused_experts(
            hidden_states=hidden_states,
            w1=w1_bf16,
            w1_scale=None,
            w2=w2_bf16,
            w2_scale=None,
            topk_weights=dispatch.topk_weights,
            topk_ids=dispatch.topk_ids,
            repetitions=args.repetitions,
        )
        fused_int8_ms = bench_fused_experts(
            hidden_states=hidden_states,
            w1=w1_int8,
            w1_scale=w1_scale,
            w2=w2_int8,
            w2_scale=w2_scale,
            topk_weights=dispatch.topk_weights,
            topk_ids=dispatch.topk_ids,
            repetitions=args.repetitions,
        )
        fused_bf16_phase = profile_fused_experts(
            hidden_states=hidden_states,
            w1=w1_bf16,
            w1_scale=None,
            w2=w2_bf16,
            w2_scale=None,
            topk_weights=dispatch.topk_weights,
            topk_ids=dispatch.topk_ids,
            repetitions=args.repetitions,
        )
        fused_int8_phase = profile_fused_experts(
            hidden_states=hidden_states,
            w1=w1_int8,
            w1_scale=w1_scale,
            w2=w2_int8,
            w2_scale=w2_scale,
            topk_weights=dispatch.topk_weights,
            topk_ids=dispatch.topk_ids,
            repetitions=args.repetitions,
        )

        print(
            json.dumps(
                {
                    "tokens": m,
                    "torch_bf16_ms": torch_bf16_ms,
                    "torch_int8_ms": torch_int8_ms,
                    "fused_bf16_ms": fused_bf16_ms,
                    "fused_int8_ms": fused_int8_ms,
                    "torch_int8_speedup_vs_bf16": torch_bf16_ms / torch_int8_ms if torch_int8_ms > 0 else None,
                    "fused_int8_speedup_vs_bf16": fused_bf16_ms / fused_int8_ms if fused_int8_ms > 0 else None,
                    "fused_bf16_w1_ms": fused_bf16_phase["w1_ms"],
                    "fused_bf16_stage2_ms": fused_bf16_phase["stage2_ms"],
                    "fused_bf16_w2_ms": fused_bf16_phase["w2_ms"],
                    "fused_bf16_reduce_ms": fused_bf16_phase["reduce_ms"],
                    "fused_int8_w1_ms": fused_int8_phase["w1_ms"],
                    "fused_int8_stage2_ms": fused_int8_phase["stage2_ms"],
                    "fused_int8_w2_ms": fused_int8_phase["w2_ms"],
                    "fused_int8_reduce_ms": fused_int8_phase["reduce_ms"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
