from __future__ import annotations

import torch


def int8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mat_a.is_cuda or not mat_b.is_cuda:
        raise ValueError("int8_scaled_mm expects CUDA tensors")
    if mat_a.dtype != torch.int8 or mat_b.dtype != torch.int8:
        raise ValueError("int8_scaled_mm expects int8 inputs")
    if mat_a.ndim != 2 or mat_b.ndim != 2:
        raise ValueError("int8_scaled_mm expects 2D tensors")

    use_torch_int_mm = (
        hasattr(torch, "_int_mm")
        and mat_a.shape[1] % 16 == 0
        and mat_b.shape[0] % 16 == 0
        and mat_b.shape[1] > 0
        and mat_b.shape[1] % 8 == 0
    )

    if use_torch_int_mm:
        padded_m = max(mat_a.shape[0], 17)
        if padded_m != mat_a.shape[0]:
            padded_a = torch.zeros(
                (padded_m, mat_a.shape[1]),
                device=mat_a.device,
                dtype=mat_a.dtype,
            )
            padded_a[: mat_a.shape[0]] = mat_a
            acc = torch._int_mm(padded_a, mat_b)[: mat_a.shape[0]]
        else:
            acc = torch._int_mm(mat_a, mat_b)
        output = acc.to(torch.float32)
    else:
        output = torch.matmul(mat_a.to(torch.float32), mat_b.to(torch.float32))

    output = output * scales_a.view(-1, 1).to(torch.float32)
    output = output * scales_b.view(1, -1).to(torch.float32)
    if bias is not None:
        output = output + bias.to(torch.float32)
    return output.to(out_dtype)
