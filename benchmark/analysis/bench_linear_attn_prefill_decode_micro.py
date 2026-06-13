from __future__ import annotations

import time

import torch

from minisgl.linear_attention import (
    fused_gdn_gating_sglang,
    fused_linear_attn_decode_sglang,
    fused_linear_attn_prefill_sglang,
)


def bench(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def make_prefill_inputs(device: str = "cuda", t: int = 1024, hv: int = 32, k_dim: int = 128, v_dim: int = 128):
    q = torch.randn(t, hv, k_dim, device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn(t, hv, k_dim, device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn(t, hv, v_dim, device=device, dtype=torch.float32).contiguous()
    gate = torch.randn(t, hv, device=device, dtype=torch.float32).contiguous()
    beta = torch.sigmoid(torch.randn(t, hv, device=device, dtype=torch.float32)).contiguous()
    state = torch.randn(hv, k_dim, v_dim, device=device, dtype=torch.float32).contiguous()
    scale = 1.0 / (k_dim**0.5)
    return q, k, v, gate, beta, state, scale


def make_decode_inputs(device: str = "cuda", batch_size: int = 1, h: int = 16, hv: int = 32, k_dim: int = 128, v_dim: int = 128):
    mixed_qkv = torch.randn(
        batch_size,
        h * k_dim * 2 + hv * v_dim,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    a = torch.randn(batch_size, hv, device=device, dtype=torch.float32).contiguous()
    b = torch.randn(batch_size, hv, device=device, dtype=torch.float32).contiguous()
    A_log = torch.randn(hv, device=device, dtype=torch.float32).contiguous()
    dt_bias = torch.randn(hv, device=device, dtype=torch.float32).contiguous()
    state = torch.randn(batch_size, hv, v_dim, k_dim, device=device, dtype=torch.float32).contiguous()
    state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    scale = 1.0 / (k_dim**0.5)
    return mixed_qkv, a, b, A_log, dt_bias, state, state_indices, scale


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"

    t = 1024
    hv = 32
    k_dim = 128
    v_dim = 128

    q, k, v, gate, beta, state, scale = make_prefill_inputs(device, t, hv, k_dim, v_dim)
    prefill_in_kernel_norm = bench(
        lambda: fused_linear_attn_prefill_sglang(
            q, k, v, gate, beta, state, scale, use_qk_l2norm_in_kernel=True
        )
    )
    qn = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-6).to(q.dtype).contiguous()
    kn = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-6).to(k.dtype).contiguous()
    prefill_outside_norm = bench(
        lambda: fused_linear_attn_prefill_sglang(
            qn, kn, v, gate, beta, state, scale, use_qk_l2norm_in_kernel=False
        )
    )

    mixed_qkv, a, b, A_log, dt_bias, decode_state, state_indices, decode_scale = make_decode_inputs(
        device, 1, 16, hv, k_dim, v_dim
    )
    decode_in_kernel_norm = bench(
        lambda: fused_linear_attn_decode_sglang(
            mixed_qkv, a, b, A_log, dt_bias, decode_state, state_indices, decode_scale,
            use_qk_l2norm_in_kernel=True,
        ),
        iters=500,
    )
    decode_outside_norm = bench(
        lambda: fused_linear_attn_decode_sglang(
            mixed_qkv, a, b, A_log, dt_bias, decode_state, state_indices, decode_scale,
            use_qk_l2norm_in_kernel=False,
        ),
        iters=500,
    )

    print(
        {
            "prefill_in_kernel_norm_ms": prefill_in_kernel_norm,
            "prefill_outside_norm_ms": prefill_outside_norm,
            "prefill_speedup_outside_over_inside": prefill_in_kernel_norm / prefill_outside_norm,
            "decode_in_kernel_norm_ms": decode_in_kernel_norm,
            "decode_outside_norm_ms": decode_outside_norm,
            "decode_speedup_outside_over_inside": decode_in_kernel_norm / decode_outside_norm,
        }
    )

    # Gating microbench for reference.
    a_gate = torch.randn(t, hv, device=device, dtype=torch.float32).contiguous()
    b_gate = torch.randn(t, hv, device=device, dtype=torch.float32).contiguous()
    A_log_gate = torch.randn(hv, device=device, dtype=torch.float32).contiguous()
    dt_bias_gate = torch.randn(hv, device=device, dtype=torch.float32).contiguous()
    gating_ms = bench(lambda: fused_gdn_gating_sglang(A_log_gate, a_gate, b_gate, dt_bias_gate))
    print({"gating_ms": gating_ms})


if __name__ == "__main__":
    main()
