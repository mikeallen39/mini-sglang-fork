from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import perf_cuda
from minisgl.moe import create_moe_backend
from minisgl.moe.dispatch import build_local_expert_dispatch_plan
from minisgl.moe.fused import (
    fused_experts_impl,
    fused_topk,
    get_fused_moe_profile,
    grouped_topk,
    reset_fused_moe_profile,
)
from minisgl.models.config import ModelConfig
from minisgl.utils import cached_load_hf_config


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
    w1 = torch.randn((num_experts, intermediate_size * 2, hidden_size), device=device, dtype=dtype)
    w2 = torch.randn((num_experts, hidden_size, intermediate_size), device=device, dtype=dtype)
    return hidden_states, gating_output, w1, w2


def resolve_bench_config(args: argparse.Namespace) -> dict[str, object]:
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
    num_expert_group = (
        args.num_expert_group
        if args.num_expert_group is not None
        else (model_cfg.num_expert_group if model_cfg is not None else 0)
    )
    topk_group = (
        args.topk_group
        if args.topk_group is not None
        else (model_cfg.topk_group if model_cfg is not None else 0)
    )
    use_grouped_topk = num_expert_group > 0 and topk_group > 0

    return {
        "model_path": args.model_path,
        "model_type": model_cfg.model_type if model_cfg is not None else None,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "experts": experts,
        "topk": topk,
        "renormalize": renormalize,
        "use_grouped_topk": use_grouped_topk,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
    }


def compute_routing(
    *,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    use_grouped_topk: bool,
    num_expert_group: int,
    topk_group: int,
    num_global_experts: int,
):
    if use_grouped_topk:
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
    else:
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
        num_local_experts=num_global_experts,
        num_global_experts=num_global_experts,
    )


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if activation == "silu":
        return F.silu(gate) * up
    if activation == "gelu":
        return F.gelu(gate) * up
    raise ValueError(f"Unsupported activation: {activation}")


def torch_expert_only(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    output = torch.zeros_like(hidden_states)

    for expert_id in range(w1.shape[0]):
        token_idx, topk_pos = torch.where(topk_ids == expert_id)
        if token_idx.numel() == 0:
            continue

        routed_x = hidden_states[token_idx]
        routed_w = topk_weights[token_idx, topk_pos].to(hidden_states.dtype)
        inter = F.linear(routed_x, w1[expert_id])
        inter = _apply_activation(inter, activation)
        routed_out = F.linear(inter, w2[expert_id])
        routed_out = routed_out * routed_w.unsqueeze(-1)
        output.index_add_(0, token_idx, routed_out)

    assert output.shape == (num_tokens, hidden_size)
    return output


def bench_router(
    *,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    use_grouped_topk: bool,
    num_expert_group: int,
    topk_group: int,
    repetitions: int,
) -> float:
    num_global_experts = gating_output.shape[1]

    def run():
        compute_routing(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_global_experts=num_global_experts,
        )

    return perf_cuda(run, repetitions=repetitions, cuda_graph_repetitions=None)


def bench_expert_only(
    backend_name: str,
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    repetitions: int,
) -> float:
    if backend_name == "torch":
        fn = lambda: torch_expert_only(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    elif backend_name == "fused":
        fn = lambda: fused_experts_impl(
            hidden_states,
            w1,
            w2,
            None,
            None,
            topk_weights,
            topk_ids,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    return perf_cuda(fn, repetitions=repetitions, cuda_graph_repetitions=None)


def profile_fused_expert_only(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    repetitions: int,
) -> dict[str, float]:
    prev_env = os.environ.get("MINISGL_PROFILE_FUSED_MOE")
    os.environ["MINISGL_PROFILE_FUSED_MOE"] = "1"
    from minisgl.env import ENV

    prev_value = ENV.PROFILE_FUSED_MOE.value
    ENV.PROFILE_FUSED_MOE.value = True
    reset_fused_moe_profile()

    try:
        for _ in range(repetitions):
            fused_experts_impl(
                hidden_states,
                w1,
                w2,
                None,
                None,
                topk_weights,
                topk_ids,
            )
        torch.cuda.synchronize()
        prof = get_fused_moe_profile()
    finally:
        reset_fused_moe_profile()
        ENV.PROFILE_FUSED_MOE.value = prev_value
        if prev_env is None:
            os.environ.pop("MINISGL_PROFILE_FUSED_MOE", None)
        else:
            os.environ["MINISGL_PROFILE_FUSED_MOE"] = prev_env

    count = int(prof["count"])
    if count <= 0:
        return {
            "fused_w1_ms": 0.0,
            "fused_stage2_ms": 0.0,
            "fused_w2_ms": 0.0,
            "fused_reduce_ms": 0.0,
        }

    return {
        "fused_w1_ms": float(prof["w1_ms"]) / count,
        "fused_stage2_ms": float(prof["stage2_ms"]) / count,
        "fused_w2_ms": float(prof["w2_ms"]) / count,
        "fused_reduce_ms": float(prof["reduce_ms"]) / count,
    }


def bench_backend(
    backend_name: str,
    *,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    renormalize: bool,
    use_grouped_topk: bool,
    num_expert_group: int,
    topk_group: int,
    repetitions: int,
) -> float:
    backend = create_moe_backend(backend_name)

    def run():
        backend.forward(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=None,
            w2=w2,
            w2_scale=None,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            activation="silu",
            apply_router_weight_on_input=False,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            local_expert_start=0,
            num_global_experts=w1.shape[0],
            num_dispatch_experts=w1.shape[0],
        )

    return perf_cuda(run, repetitions=repetitions, cuda_graph_repetitions=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--model-path")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--renormalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-expert-group", type=int)
    parser.add_argument("--topk-group", type=int)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = resolve_bench_config(args)

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": args.dtype,
                **config,
                "repetitions": args.repetitions,
            },
            ensure_ascii=False,
        )
    )

    for m in args.tokens:
        hidden_states, gating_output, w1, w2 = build_inputs(
            m=m,
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_experts=config["experts"],
            topk=config["topk"],
            device=device,
            dtype=dtype,
        )
        dispatch_plan = compute_routing(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=config["topk"],
            renormalize=config["renormalize"],
            use_grouped_topk=config["use_grouped_topk"],
            num_expert_group=config["num_expert_group"],
            topk_group=config["topk_group"],
            num_global_experts=config["experts"],
        )

        router_ms = bench_router(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=config["topk"],
            renormalize=config["renormalize"],
            use_grouped_topk=config["use_grouped_topk"],
            num_expert_group=config["num_expert_group"],
            topk_group=config["topk_group"],
            repetitions=args.repetitions,
        )
        torch_expert_ms = bench_expert_only(
            "torch",
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=dispatch_plan.topk_weights,
            topk_ids=dispatch_plan.topk_ids,
            repetitions=args.repetitions,
        )
        fused_expert_ms = bench_expert_only(
            "fused",
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=dispatch_plan.topk_weights,
            topk_ids=dispatch_plan.topk_ids,
            repetitions=args.repetitions,
        )
        fused_phase_ms = profile_fused_expert_only(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=dispatch_plan.topk_weights,
            topk_ids=dispatch_plan.topk_ids,
            repetitions=args.repetitions,
        )

        torch_ms = bench_backend(
            "torch",
            hidden_states=hidden_states,
            gating_output=gating_output,
            w1=w1,
            w2=w2,
            topk=config["topk"],
            renormalize=config["renormalize"],
            use_grouped_topk=config["use_grouped_topk"],
            num_expert_group=config["num_expert_group"],
            topk_group=config["topk_group"],
            repetitions=args.repetitions,
        )
        fused_ms = bench_backend(
            "fused",
            hidden_states=hidden_states,
            gating_output=gating_output,
            w1=w1,
            w2=w2,
            topk=config["topk"],
            renormalize=config["renormalize"],
            use_grouped_topk=config["use_grouped_topk"],
            num_expert_group=config["num_expert_group"],
            topk_group=config["topk_group"],
            repetitions=args.repetitions,
        )
        speedup = torch_ms / fused_ms if fused_ms > 0 else float("inf")

        print(
            json.dumps(
                {
                    "tokens": m,
                    "router_ms": router_ms,
                    "torch_expert_ms": torch_expert_ms,
                    "fused_expert_ms": fused_expert_ms,
                    **fused_phase_ms,
                    "torch_ms": torch_ms,
                    "fused_ms": fused_ms,
                    "torch_router_share": router_ms / torch_ms if torch_ms > 0 else None,
                    "fused_router_share": router_ms / fused_ms if fused_ms > 0 else None,
                    "torch_expert_share": torch_expert_ms / torch_ms if torch_ms > 0 else None,
                    "fused_expert_share": fused_expert_ms / fused_ms if fused_ms > 0 else None,
                    "torch_expert_speedup_over_fused": (
                        torch_expert_ms / fused_expert_ms if fused_expert_ms > 0 else float("inf")
                    ),
                    "speedup_torch_over_fused": speedup,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
