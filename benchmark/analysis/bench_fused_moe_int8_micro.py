from __future__ import annotations

import argparse

import torch
from minisgl.benchmark.perf import perf_cuda
from minisgl.moe.fused import fused_experts_impl
from minisgl.quantization import quantize_weight_per_channel_int8_row_major


def build_inputs(
    *,
    m: int,
    k: int,
    n: int,
    e: int,
    topk: int,
    device: torch.device,
):
    hidden = torch.randn((m, k), device=device, dtype=torch.bfloat16)

    w1_fp = torch.randn((e, n, k), device=device, dtype=torch.bfloat16) * 0.02
    w2_fp = torch.randn((e, k, n // 2), device=device, dtype=torch.bfloat16) * 0.02

    w1_q = torch.empty((e, n, k), device=device, dtype=torch.int8)
    w1_s = torch.empty((e, n, 1), device=device, dtype=torch.float32)
    w2_q = torch.empty((e, k, n // 2), device=device, dtype=torch.int8)
    w2_s = torch.empty((e, k, 1), device=device, dtype=torch.float32)

    for expert in range(e):
        q, s = quantize_weight_per_channel_int8_row_major(w1_fp[expert])
        w1_q[expert].copy_(q)
        w1_s[expert].copy_(s)
        q, s = quantize_weight_per_channel_int8_row_major(w2_fp[expert])
        w2_q[expert].copy_(q)
        w2_s[expert].copy_(s)

    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return hidden, w1_q, w1_s, w2_q, w2_s, topk_weights, topk_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--intermediate-size", type=int, default=4608)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    for m in (1, 8, 64):
        hidden, w1_q, w1_s, w2_q, w2_s, topk_weights, topk_ids = build_inputs(
            m=m,
            k=args.hidden_size,
            n=args.intermediate_size * 2,
            e=args.experts,
            topk=args.topk,
            device=device,
        )

        def run():
            fused_experts_impl(
                hidden,
                w1_q,
                w2_q,
                w1_s,
                w2_s,
                topk_weights,
                topk_ids,
                activation="silu",
                apply_router_weight_on_input=False,
                filter_expert=False,
            )

        dur_ms = perf_cuda(run, repetitions=50, cuda_graph_repetitions=None)
        print(f"M={m}: {dur_ms:.4f} ms")


if __name__ == "__main__":
    main()
