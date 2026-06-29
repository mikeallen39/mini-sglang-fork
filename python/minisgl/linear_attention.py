from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is expected on CUDA setups.
    triton = None
    tl = None

from minisgl.fla_vendor import chunk_gated_delta_rule


SUPPORTED_LINEAR_ATTN_BACKENDS = {"torch", "sglang"}

_CURRENT_LINEAR_ATTN_BACKEND = "torch"


def set_linear_attn_backend(backend: str) -> None:
    if backend not in SUPPORTED_LINEAR_ATTN_BACKENDS:
        raise ValueError(f"Unsupported linear attention backend: {backend}")
    global _CURRENT_LINEAR_ATTN_BACKEND
    _CURRENT_LINEAR_ATTN_BACKEND = backend


def get_linear_attn_backend() -> str:
    return _CURRENT_LINEAR_ATTN_BACKEND


def has_sglang_linear_attn_kernel() -> bool:
    return triton is not None and tl is not None


@triton.jit
def _fused_qkvzba_split_reshape_cat_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs = tl.program_id(0)
    i_qk = tl.program_id(1)

    qk_group = tl.arange(0, HEAD_QK)
    v_group = tl.arange(0, HEAD_V)

    TOTAL_Q = NUM_HEADS_QK * HEAD_QK
    TOTAL_K = NUM_HEADS_QK * HEAD_QK
    TOTAL_V = NUM_HEADS_V * HEAD_V
    TOTAL_QKV = TOTAL_Q + TOTAL_K + TOTAL_V
    TOTAL_QKVZ = TOTAL_QKV + TOTAL_V
    TOTAL_BA = NUM_HEADS_V * 2
    V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    q_base = i_bs * TOTAL_QKVZ + i_qk * HEAD_QK
    k_base = i_bs * TOTAL_QKVZ + TOTAL_Q + i_qk * HEAD_QK
    v_base = i_bs * TOTAL_QKVZ + TOTAL_Q + TOTAL_K + i_qk * V_PER_GROUP * HEAD_V
    z_base = i_bs * TOTAL_QKVZ + TOTAL_QKV + i_qk * V_PER_GROUP * HEAD_V

    q_out_base = i_bs * TOTAL_QKV + i_qk * HEAD_QK
    k_out_base = i_bs * TOTAL_QKV + TOTAL_Q + i_qk * HEAD_QK
    v_out_base = i_bs * TOTAL_QKV + TOTAL_Q + TOTAL_K + i_qk * V_PER_GROUP * HEAD_V

    tl.store(mixed_qkv + q_out_base + qk_group, tl.load(mixed_qkvz + q_base + qk_group))
    tl.store(mixed_qkv + k_out_base + qk_group, tl.load(mixed_qkvz + k_base + qk_group))

    for i in tl.static_range(V_PER_GROUP):
        v_offsets = v_group + i * HEAD_V
        tl.store(
            mixed_qkv + v_out_base + v_offsets,
            tl.load(mixed_qkvz + v_base + v_offsets),
        )
        z_out_base = (i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i) * HEAD_V
        tl.store(
            z + z_out_base + v_group,
            tl.load(mixed_qkvz + z_base + v_offsets),
        )
        b_offset = i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(
            b + b_offset,
            tl.load(mixed_ba + i_bs * TOTAL_BA + i_qk * V_PER_GROUP + i),
        )
        tl.store(
            a + b_offset,
            tl.load(mixed_ba + i_bs * TOTAL_BA + NUM_HEADS_V + i_qk * V_PER_GROUP + i),
        )


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for fused_qkvzba_split_reshape_cat_contiguous")

    batch = mixed_qkvz.shape[0]
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty((batch, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    z = torch.empty((batch, num_heads_v, head_v), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    b = torch.empty((batch, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)

    grid = (batch, num_heads_qk)
    _fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


def linear_attn_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = gate.float()
    betaf = beta.float()
    h = state.float()
    t, hv, _ = q.shape
    out = torch.empty((t, hv, v.shape[2]), device=v.device, dtype=torch.float32)
    for i in range(t):
        h.mul_(torch.exp(gf[i]).view(hv, 1, 1))
        value_residual = vf[i] - torch.einsum("hkv,hk->hv", h, kf[i])
        value_residual = value_residual * betaf[i].unsqueeze(-1)
        h.add_(kf[i].unsqueeze(-1) * value_residual.unsqueeze(-2))
        out[i] = torch.einsum("hkv,hk->hv", h, qf[i] * scale)
    state.copy_(h.to(state.dtype))
    return out.to(v.dtype)


def linear_attn_prefill_chunk_scan_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = gate.float()
    betaf = beta.float()
    h = state.float()
    t, hv, _ = q.shape
    v_dim = v.shape[2]
    out = torch.empty((t, hv, v_dim), device=v.device, dtype=torch.float32)
    eye = torch.eye(q.shape[2], device=q.device, dtype=torch.float32).expand(hv, -1, -1)

    for chunk_start in range(0, t, chunk_size):
        chunk_end = min(chunk_start + chunk_size, t)
        chunk_len = chunk_end - chunk_start
        gc = gf[chunk_start:chunk_end]
        kc = kf[chunk_start:chunk_end]
        vc = vf[chunk_start:chunk_end]
        betac = betaf[chunk_start:chunk_end]
        qc = qf[chunk_start:chunk_end]

        gc_cumsum = torch.cumsum(gc, dim=0)
        A = torch.zeros((chunk_len, hv, chunk_len), device=q.device, dtype=torch.float32)
        for i in range(chunk_len):
            for j in range(i):
                A[i, :, j] = (
                    betac[i]
                    * torch.exp(gc_cumsum[i] - gc_cumsum[j])
                    * torch.sum(kc[i] * kc[j], dim=-1)
                )
        A_inv = torch.zeros_like(A)
        for h_idx in range(hv):
            tri = torch.eye(chunk_len, device=q.device, dtype=torch.float32) + A[:, h_idx, :]
            A_inv[:, h_idx, :] = torch.linalg.inv(tri)

        u = torch.einsum("ihj,jhv->ihv", A_inv, vc * betac.unsqueeze(-1))
        w = torch.einsum(
            "ihj,jhk->ihk",
            A_inv,
            kc * betac.unsqueeze(-1) * torch.exp(gc_cumsum).unsqueeze(-1),
        )

        h_state_chunks = torch.empty((chunk_len, hv, q.shape[2], v_dim), device=q.device, dtype=torch.float32)
        running_h = h.clone()
        for i in range(chunk_len):
            h_state_chunks[i] = running_h
            running_h = torch.exp(gc[i]).view(hv, 1, 1) * running_h + kc[i].unsqueeze(-1) * u[i].unsqueeze(-2)

        for i in range(chunk_len):
            cross = torch.einsum("hkv,hk->hv", h_state_chunks[i], qc[i] * scale)
            intra_mask = torch.arange(chunk_len, device=q.device) <= i
            scores = torch.einsum("hk,jhk->hj", qc[i], kc)
            scores = scores * torch.exp(gc_cumsum[i] - gc_cumsum).transpose(0, 1)
            intra = torch.einsum("hj,jhv->hv", scores[:, intra_mask], u[intra_mask])
            out[chunk_start + i] = cross + intra
        h = running_h

    state.copy_(h.to(state.dtype))
    return out.to(v.dtype)


def linear_attn_prefill_hf_chunk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    initial_dtype = q.dtype
    compute_dtype = torch.float64
    qf = q.to(compute_dtype).permute(1, 0, 2).unsqueeze(0).contiguous()
    kf = k.to(compute_dtype).permute(1, 0, 2).unsqueeze(0).contiguous()
    vf = v.to(compute_dtype).permute(1, 0, 2).unsqueeze(0).contiguous()
    betaf = beta.to(compute_dtype).permute(1, 0).unsqueeze(0).contiguous()
    gf = gate.to(compute_dtype).permute(1, 0).unsqueeze(0).contiguous()
    statef = state.to(compute_dtype).unsqueeze(0).contiguous()

    batch_size, num_heads, sequence_length, k_head_dim = kf.shape
    v_head_dim = vf.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    qf = torch.nn.functional.pad(qf, (0, 0, 0, pad_size))
    kf = torch.nn.functional.pad(kf, (0, 0, 0, pad_size))
    vf = torch.nn.functional.pad(vf, (0, 0, 0, pad_size))
    betaf = torch.nn.functional.pad(betaf, (0, pad_size))
    gf = torch.nn.functional.pad(gf, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    qf = qf * scale

    v_beta = vf * betaf.unsqueeze(-1)
    k_beta = kf * betaf.unsqueeze(-1)
    q_chunks, k_chunks, v_chunks, k_beta_chunks, v_beta_chunks = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (qf, kf, vf, k_beta, v_beta)
    ]
    g_chunks = gf.reshape(gf.shape[0], gf.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

    g_chunks = g_chunks.cumsum(dim=-1)
    decay_mask = ((g_chunks.unsqueeze(-1) - g_chunks.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta_chunks @ k_chunks.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value_chunks = attn @ v_beta_chunks
    k_cumdecay = attn @ (k_beta_chunks * g_chunks.exp().unsqueeze(-1))

    last_recurrent_state = statef
    core_attn_out = torch.zeros_like(value_chunks)
    for i in range(total_sequence_length // chunk_size):
        q_i = q_chunks[:, :, i]
        k_i = k_chunks[:, :, i]
        v_i = value_chunks[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_chunks[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_chunks[:, :, i, -1, None, None].exp()
            + (k_i * (g_chunks[:, :, i, -1, None] - g_chunks[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    out = core_attn_out.transpose(1, 2).contiguous().squeeze(0).to(initial_dtype)
    final_state = last_recurrent_state.squeeze(0).contiguous()
    state.copy_(final_state.to(state.dtype))
    return out


def linear_attn_prefill_chunk_stable_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = gate.float()
    betaf = beta.float()
    h = state.float()

    t, hv, k_dim = q.shape
    v_dim = v.shape[2]
    out = torch.empty((t, hv, v_dim), device=v.device, dtype=torch.float32)

    for chunk_start in range(0, t, chunk_size):
        chunk_end = min(chunk_start + chunk_size, t)
        n = chunk_end - chunk_start
        qc = qf[chunk_start:chunk_end]
        kc = kf[chunk_start:chunk_end]
        vc = vf[chunk_start:chunk_end]
        gc = gf[chunk_start:chunk_end]
        betac = betaf[chunk_start:chunk_end]

        g_cumsum = gc.cumsum(dim=0)  # [n, hv]
        # L[i, j] = beta_i * <k_i, k_j> * exp(g_i_cum - g_j_cum), for j < i
        k_dot = torch.einsum("ihk,jhk->ijh", kc, kc)  # [n, n, hv]
        g_diff = g_cumsum[:, None, :] - g_cumsum[None, :, :]
        L = betac[:, None, :] * k_dot * torch.exp(g_diff)
        tril_mask = torch.tril(torch.ones((n, n), device=q.device, dtype=torch.bool), diagonal=-1)
        L = torch.where(tril_mask.unsqueeze(-1), L, torch.zeros_like(L))

        # Solve (I + L_h) U_h = beta * V for each head h, chunk-local.
        rhs_u = betac.unsqueeze(-1) * vc  # [n, hv, v]
        U = torch.empty_like(rhs_u)
        rhs_w = betac.unsqueeze(-1) * kc * torch.exp(g_cumsum).unsqueeze(-1)  # [n, hv, k]
        W = torch.empty_like(rhs_w)
        eye = torch.eye(n, device=q.device, dtype=torch.float32)
        for h_idx in range(hv):
            A_h = eye + L[:, :, h_idx]
            U[:, h_idx] = torch.linalg.solve_triangular(A_h, rhs_u[:, h_idx], upper=False)
            W[:, h_idx] = torch.linalg.solve_triangular(A_h, rhs_w[:, h_idx], upper=False)

        running_h = h
        h_before = torch.empty((n, hv, k_dim, v_dim), device=q.device, dtype=torch.float32)
        for i in range(n):
            h_before[i] = running_h
            running_h = torch.exp(gc[i]).view(hv, 1, 1) * running_h + kc[i].unsqueeze(-1) * U[i].unsqueeze(-2)

        # Cross-chunk contribution from initial state.
        cross = torch.einsum("ihkv,ihk->ihv", h_before, qc * scale)
        # Intra-chunk contribution from solved values.
        for i in range(n):
            scores = torch.einsum("hk,jhk->hj", qc[i] * scale, kc[: i + 1])
            decay = torch.exp(g_cumsum[i] - g_cumsum[: i + 1]).transpose(0, 1)
            intra = torch.einsum("jh,jhv->hv", scores.transpose(0, 1) * decay.transpose(0, 1), U[: i + 1])
            out[chunk_start + i] = cross[i] + intra

        h = running_h

    state.copy_(h.to(state.dtype))
    return out.to(v.dtype)


def linear_attn_chunk_local_cumsum_reference(
    gate: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    gatef = gate.float()
    t, hv = gatef.shape
    out = torch.empty_like(gatef)
    for chunk_start in range(0, t, chunk_size):
        chunk_end = min(chunk_start + chunk_size, t)
        out[chunk_start:chunk_end] = gatef[chunk_start:chunk_end].cumsum(dim=0)
    return out


def linear_attn_chunk_intra_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Placeholder reference that preserves the chunk-stage API shape.
    # For now, reuse the numerically correct HF-style chunk reference components.
    kf = k.float()
    vf = v.float()
    betaf = beta.float()
    gcf = g_cumsum.float()
    t, hv, k_dim = k.shape
    v_dim = v.shape[2]
    u = torch.empty((t, hv, v_dim), device=v.device, dtype=torch.float32)
    w = torch.empty((t, hv, k_dim), device=v.device, dtype=torch.float32)

    for chunk_start in range(0, t, chunk_size):
        chunk_end = min(chunk_start + chunk_size, t)
        n = chunk_end - chunk_start
        kc = kf[chunk_start:chunk_end]
        vc = vf[chunk_start:chunk_end]
        betac = betaf[chunk_start:chunk_end]
        gcc = gcf[chunk_start:chunk_end]

        kc64 = kc.double()
        vc64 = vc.double()
        betac64 = betac.double()
        gcc64 = gcc.double()
        k_dot = torch.einsum("ihk,jhk->ijh", kc64, kc64)
        g_diff = gcc64[:, None, :] - gcc64[None, :, :]
        lower = torch.tril(torch.ones((n, n), device=k.device, dtype=torch.bool), diagonal=-1)
        A = betac64[:, None, :] * k_dot * torch.exp(g_diff)
        A = torch.where(lower.unsqueeze(-1), A, torch.zeros_like(A))
        eye = torch.eye(n, device=k.device, dtype=torch.float64)
        rhs_u = betac64.unsqueeze(-1) * vc64
        rhs_w = betac64.unsqueeze(-1) * kc64 * torch.exp(gcc64).unsqueeze(-1)
        for h_idx in range(hv):
            tri = eye + A[:, :, h_idx]
            u[chunk_start:chunk_end, h_idx] = torch.linalg.solve_triangular(
                tri, rhs_u[:, h_idx], upper=False
            ).float()
            w[chunk_start:chunk_end, h_idx] = torch.linalg.solve_triangular(
                tri, rhs_w[:, h_idx], upper=False
            ).float()

    return w, u


def linear_attn_chunk_state_scan_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kf = k.float()
    wf = w.float()
    uf = u.float()
    gf = gate.float()
    h = state.float()
    t, hv, k_dim = k.shape
    v_dim = u.shape[2]
    num_chunks = (t + chunk_size - 1) // chunk_size
    h_chunk = torch.empty((num_chunks, hv, k_dim, v_dim), device=k.device, dtype=torch.float32)
    v_new = torch.empty((t, hv, v_dim), device=k.device, dtype=torch.float32)
    for chunk_idx, chunk_start in enumerate(range(0, t, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, t)
        h_chunk[chunk_idx] = h
        for i in range(chunk_start, chunk_end):
            v_new[i] = uf[i] - torch.einsum("hk,hkv->hv", wf[i], h)
        for i in range(chunk_start, chunk_end):
            h = torch.exp(gf[i]).view(hv, 1, 1) * h + kf[i].unsqueeze(-1) * v_new[i].unsqueeze(-2)
    return h_chunk, v_new, h


def linear_attn_chunk_output_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_chunk: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    qf = q.float()
    kf = k.float()
    vnf = v_new.float()
    gcf = g_cumsum.float()
    t, hv, _ = q.shape
    v_dim = v_new.shape[2]
    out = torch.empty((t, hv, v_dim), device=q.device, dtype=torch.float32)
    for chunk_idx, chunk_start in enumerate(range(0, t, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, t)
        qc = qf[chunk_start:chunk_end]
        kc = kf[chunk_start:chunk_end]
        vnc = vnf[chunk_start:chunk_end]
        gcc = gcf[chunk_start:chunk_end]
        n = chunk_end - chunk_start
        h0 = h_chunk[chunk_idx]
        cross = torch.einsum("hkv,ihk->ihv", h0, qc * scale * torch.exp(gcc).unsqueeze(-1))
        for i in range(n):
            scores = torch.einsum("hk,jhk->hj", qc[i] * scale, kc[: i + 1])
            decay = torch.exp(gcc[i] - gcc[: i + 1]).transpose(0, 1)
            intra = torch.einsum("hj,jhv->hv", scores * decay, vnc[: i + 1])
            out[chunk_start + i] = cross[i] + intra
    return out.to(q.dtype)


def linear_attn_prefill_chunk_sglang_style_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    # Match sglang chunk kernels: intra/recompute path expects low-precision
    # value tensors entering tl.dot. The surrounding prefill path may still
    # hand us fp32 values for numerical reasons, so keep a local low-precision
    # view for chunk intra only.
    if v.dtype != q.dtype:
        v = v.to(q.dtype)
    g_cumsum = linear_attn_chunk_local_cumsum_reference(gate, chunk_size=chunk_size)
    w, u = linear_attn_chunk_intra_triton(k, v, g_cumsum, beta, chunk_size=chunk_size)
    h_chunk, v_new, h_final = linear_attn_chunk_state_scan_reference(k, w, u, gate, state, chunk_size=chunk_size)
    out = linear_attn_chunk_output_reference(q, k, v_new, h_chunk, g_cumsum, scale, chunk_size=chunk_size)
    state.copy_(h_final.to(state.dtype))
    return out


def linear_attn_chunk_intra_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        return linear_attn_chunk_intra_reference(
            k,
            v,
            g_cumsum,
            beta,
            chunk_size=chunk_size,
        )
    if chunk_size != 64:
        return linear_attn_chunk_intra_reference(
            k,
            v,
            g_cumsum,
            beta,
            chunk_size=chunk_size,
        )
    t, hv, k_dim = k.shape
    v_dim = v.shape[2]
    num_chunks = (t + chunk_size - 1) // chunk_size
    # Stage 1: build chunk-local lower-triangular system M = I + L.
    A = torch.empty((num_chunks, hv, chunk_size, chunk_size), device=k.device, dtype=k.dtype)
    grid = (num_chunks, hv)
    bk = min(64, triton.next_power_of_2(k_dim))
    _linear_attn_chunk_kkt_kernel[grid](
        k,
        g_cumsum,
        beta,
        A,
        stride_k_tok=k.stride(0),
        stride_k_head=k.stride(1),
        stride_g_tok=g_cumsum.stride(0),
        stride_beta_tok=beta.stride(0),
        stride_A_chunk=A.stride(0),
        stride_A_head=A.stride(1),
        stride_A_row=A.stride(2),
        T=t,
        HV=hv,
        K=k_dim,
        CHUNK=chunk_size,
        BK=bk,
        num_warps=4,
        num_stages=3,
    )
    # Stage 2: invert each chunk-local lower-triangular matrix in Triton so the
    # recompute path consumes the same "(I + L)^{-1}" representation as sglang.
    _linear_attn_chunk_inverse_tril_kernel[grid](
        A,
        stride_A_chunk=A.stride(0),
        stride_A_head=A.stride(1),
        stride_A_row=A.stride(2),
        T=t,
        CHUNK=chunk_size,
        num_warps=4,
        num_stages=3,
    )
    u = torch.empty((t, hv, v_dim), device=v.device, dtype=torch.float32)
    w = torch.empty((t, hv, k_dim), device=v.device, dtype=torch.float32)
    grid_recompute = (num_chunks, hv)
    bk_recompute = min(64, triton.next_power_of_2(k_dim))
    bv_recompute = min(64, triton.next_power_of_2(v_dim))
    _linear_attn_chunk_recompute_u_kernel[grid_recompute](
        v,
        beta,
        A,
        u,
        stride_v_tok=v.stride(0),
        stride_v_head=v.stride(1),
        stride_beta_tok=beta.stride(0),
        stride_A_chunk=A.stride(0),
        stride_A_head=A.stride(1),
        stride_A_row=A.stride(2),
        stride_u_tok=u.stride(0),
        stride_u_head=u.stride(1),
        T=t,
        HV=hv,
        V=v_dim,
        CHUNK=chunk_size,
        BV=bv_recompute,
        num_warps=4,
        num_stages=3,
    )
    _linear_attn_chunk_recompute_w_kernel[grid_recompute](
        k,
        beta,
        g_cumsum,
        A,
        w,
        stride_k_tok=k.stride(0),
        stride_k_head=k.stride(1),
        stride_beta_tok=beta.stride(0),
        stride_g_tok=g_cumsum.stride(0),
        stride_A_chunk=A.stride(0),
        stride_A_head=A.stride(1),
        stride_A_row=A.stride(2),
        stride_w_tok=w.stride(0),
        stride_w_head=w.stride(1),
        T=t,
        HV=hv,
        K=k_dim,
        CHUNK=chunk_size,
        BK=bk_recompute,
        num_warps=4,
        num_stages=3,
    )
    return w, u


if triton is not None:

    @triton.jit
    def _linear_attn_chunk_kkt_kernel(
        k,
        g_cumsum,
        beta,
        A,
        stride_k_tok: tl.constexpr,
        stride_k_head: tl.constexpr,
        stride_g_tok: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_A_chunk: tl.constexpr,
        stride_A_head: tl.constexpr,
        stride_A_row: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        CHUNK: tl.constexpr,
        BK: tl.constexpr,
    ):
        chunk_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        row = tl.arange(0, CHUNK)
        col = tl.arange(0, CHUNK)
        tok_row = chunk_idx * CHUNK + row
        tok_col = chunk_idx * CHUNK + col
        row_mask = tok_row < T
        col_mask = tok_col < T
        valid = row[:, None] > col[None, :]
        valid = valid & row_mask[:, None] & col_mask[None, :]

        acc = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for k_block in range(0, K, BK):
            offs_k = k_block + tl.arange(0, BK)
            mask_k = offs_k < K
            p_row = (
                k
                + tok_row[:, None] * stride_k_tok
                + head_idx * stride_k_head
                + offs_k[None, :]
            )
            p_col = (
                k
                + tok_col[:, None] * stride_k_tok
                + head_idx * stride_k_head
                + offs_k[None, :]
            )
            b_row = tl.load(p_row, mask=row_mask[:, None] & mask_k[None, :], other=0).to(tl.float32)
            b_col = tl.load(p_col, mask=col_mask[:, None] & mask_k[None, :], other=0).to(tl.float32)
            acc += tl.dot(b_row, tl.trans(b_col))

        p_g_row = g_cumsum + tok_row * stride_g_tok + head_idx
        p_g_col = g_cumsum + tok_col * stride_g_tok + head_idx
        p_beta_row = beta + tok_row * stride_beta_tok + head_idx
        g_row = tl.load(p_g_row, mask=row_mask, other=0).to(tl.float32)
        g_col = tl.load(p_g_col, mask=col_mask, other=0).to(tl.float32)
        beta_row = tl.load(p_beta_row, mask=row_mask, other=0).to(tl.float32)
        acc = acc * tl.exp(g_row[:, None] - g_col[None, :]) * beta_row[:, None]
        acc = tl.where(valid, acc, 0.0)
        diag = row[:, None] == col[None, :]
        acc = tl.where(diag & row_mask[:, None] & col_mask[None, :], acc + 1.0, acc)

        p_A = (
            A
            + chunk_idx * stride_A_chunk
            + head_idx * stride_A_head
            + row[:, None] * stride_A_row
            + col[None, :]
        )
        tl.store(p_A, acc, mask=row_mask[:, None] & col_mask[None, :])

    @triton.jit
    def _linear_attn_chunk_inverse_tril_kernel(
        A,
        stride_A_chunk: tl.constexpr,
        stride_A_head: tl.constexpr,
        stride_A_row: tl.constexpr,
        T: tl.constexpr,
        CHUNK: tl.constexpr,
    ):
        chunk_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        row = tl.arange(0, CHUNK)
        col = tl.arange(0, CHUNK)
        tok = chunk_idx * CHUNK + row
        valid_row = tok < T
        valid = valid_row[:, None] & valid_row[None, :]
        p_A = (
            A
            + chunk_idx * stride_A_chunk
            + head_idx * stride_A_head
            + row[:, None] * stride_A_row
            + col[None, :]
        )
        b_A = tl.load(p_A, mask=valid, other=0).to(tl.float32)
        b_inv = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i in range(CHUNK):
            a_row = tl.sum(tl.where((row == i)[:, None], b_A, 0.0), axis=0)
            diag = tl.sum(tl.where(col == i, a_row, 0.0), axis=0)
            prev = tl.where(col < i, a_row, 0.0)
            contrib = tl.sum(prev[:, None] * b_inv, axis=0)
            new_row = tl.where(col < i, -contrib / diag, 0.0)
            new_row = tl.where(col == i, 1.0 / diag, new_row)
            b_inv = tl.where((row[:, None] == i), new_row[None, :], b_inv)

        tl.store(p_A, b_inv, mask=valid)

    @triton.jit
    def _linear_attn_chunk_recompute_u_kernel(
        v,
        beta,
        A,
        u,
        stride_v_tok: tl.constexpr,
        stride_v_head: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_A_chunk: tl.constexpr,
        stride_A_head: tl.constexpr,
        stride_A_row: tl.constexpr,
        stride_u_tok: tl.constexpr,
        stride_u_head: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        V: tl.constexpr,
        CHUNK: tl.constexpr,
        BV: tl.constexpr,
    ):
        chunk_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        row = tl.arange(0, CHUNK)
        tok0 = chunk_idx * CHUNK
        row_mask = (tok0 + row) < T
        p_beta = tl.make_block_ptr(
            beta + head_idx,
            (T,),
            (stride_beta_tok,),
            (tok0,),
            (CHUNK,),
            (0,),
        )
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
        p_A = tl.make_block_ptr(
            A + chunk_idx * stride_A_chunk + head_idx * stride_A_head,
            (CHUNK, CHUNK),
            (stride_A_row, 1),
            (0, 0),
            (CHUNK, CHUNK),
            (1, 0),
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))

        for v_block in range(0, V, BV):
            p_v = tl.make_block_ptr(
                v + head_idx * stride_v_head,
                (T, V),
                (stride_v_tok, 1),
                (tok0, v_block),
                (CHUNK, BV),
                (1, 0),
            )
            p_u = tl.make_block_ptr(
                u + head_idx * stride_u_head,
                (T, V),
                (stride_u_tok, 1),
                (tok0, v_block),
                (CHUNK, BV),
                (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
            b_u = tl.dot(b_A, b_vb)
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    @triton.jit
    def _linear_attn_chunk_recompute_w_kernel(
        k,
        beta,
        g_cumsum,
        A,
        w,
        stride_k_tok: tl.constexpr,
        stride_k_head: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_g_tok: tl.constexpr,
        stride_A_chunk: tl.constexpr,
        stride_A_head: tl.constexpr,
        stride_A_row: tl.constexpr,
        stride_w_tok: tl.constexpr,
        stride_w_head: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        CHUNK: tl.constexpr,
        BK: tl.constexpr,
    ):
        chunk_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        tok0 = chunk_idx * CHUNK
        p_beta = tl.make_block_ptr(
            beta + head_idx,
            (T,),
            (stride_beta_tok,),
            (tok0,),
            (CHUNK,),
            (0,),
        )
        p_g = tl.make_block_ptr(
            g_cumsum + head_idx,
            (T,),
            (stride_g_tok,),
            (tok0,),
            (CHUNK,),
            (0,),
        )
        p_A = tl.make_block_ptr(
            A + chunk_idx * stride_A_chunk + head_idx * stride_A_head,
            (CHUNK, CHUNK),
            (stride_A_row, 1),
            (0, 0),
            (CHUNK, CHUNK),
            (1, 0),
        )
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
        b_g = tl.exp(tl.load(p_g, boundary_check=(0,)).to(tl.float32))
        b_A = tl.load(p_A, boundary_check=(0, 1))

        for k_block in range(0, K, BK):
            p_k = tl.make_block_ptr(
                k + head_idx * stride_k_head,
                (T, K),
                (stride_k_tok, 1),
                (tok0, k_block),
                (CHUNK, BK),
                (1, 0),
            )
            p_w = tl.make_block_ptr(
                w + head_idx * stride_w_head,
                (T, K),
                (stride_w_tok, 1),
                (tok0, k_block),
                (CHUNK, BK),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
            b_w = tl.dot(b_A, b_kb)
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    @triton.jit
    def _fused_gdn_gating_kernel(
        gate,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        stride_gate_tok: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        BH: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_h_blk = tl.program_id(1)
        head_off = i_h_blk * BH + tl.arange(0, BH)
        mask = (i_t < T) & (head_off < HV)
        blk_A_log = tl.load(A_log + head_off, mask=head_off < HV, other=0).to(tl.float32)
        blk_a = tl.load(a + i_t * stride_a_tok + head_off, mask=mask, other=0).to(tl.float32)
        blk_b = tl.load(b + i_t * stride_b_tok + head_off, mask=mask, other=0).to(tl.float32)
        blk_bias = tl.load(dt_bias + head_off, mask=head_off < HV, other=0).to(tl.float32)
        x = blk_a + blk_bias
        softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        blk_g = -tl.exp(blk_A_log) * softplus_x
        blk_beta = tl.sigmoid(blk_b)
        tl.store(gate + i_t * stride_gate_tok + head_off, blk_g, mask=mask)
        tl.store(beta_output + i_t * stride_beta_tok + head_off, blk_beta, mask=mask)

    @triton.jit
    def _fused_linear_attn_decode_kv_kernel(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        output,
        state,
        state_indices,
        scale,
        stride_mixed_qkv_tok: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        stride_output_tok: tl.constexpr,
        stride_state_token: tl.constexpr,
        stride_state_head: tl.constexpr,
        stride_state_k: tl.constexpr,
        stride_indices_tok: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        SOFTPLUS_THRESHOLD: tl.constexpr,
        USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
        STATE_LAYOUT_VK: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        if STATE_LAYOUT_VK:
            mask_h = mask_v[:, None] & mask_k[None, :]
        else:
            mask_h = mask_k[:, None] & mask_v[None, :]

        state_idx = tl.load(state_indices + i_n * stride_indices_tok).to(tl.int64)
        if state_idx < 0:
            return

        p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_off = i_h * K + o_k
        k_off = (H * K) + i_h * K + o_k
        v_off = (2 * H * K) + i_hv * V + o_v
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(
            x <= SOFTPLUS_THRESHOLD,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val)

        if STATE_LAYOUT_VK:
            p_state = (
                state
                + state_idx * stride_state_token
                + i_hv * stride_state_head
                + o_v[:, None] * stride_state_k
                + o_k[None, :]
            )
        else:
            p_state = (
                state
                + state_idx * stride_state_token
                + i_hv * stride_state_head
                + o_k[:, None] * stride_state_k
                + o_v[None, :]
            )
        b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)
        b_h *= tl.exp(g_val)
        if STATE_LAYOUT_VK:
            b_v -= tl.sum(b_h * b_k[None, :], axis=1)
        else:
            b_v -= tl.sum(b_h * b_k[:, None], axis=0)
        b_v *= beta_val
        if STATE_LAYOUT_VK:
            b_h += b_v[:, None] * b_k[None, :]
        else:
            b_h += b_k[:, None] * b_v[None, :]

        p_out = output + i_n * stride_output_tok + i_hv * V + o_v
        if STATE_LAYOUT_VK:
            b_o = tl.sum(b_h * b_q[None, :], axis=1)
        else:
            b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_out, b_o.to(output.dtype.element_ty), mask=mask_v)
        tl.store(p_state, b_h.to(state.dtype.element_ty), mask=mask_h)

    @triton.jit
    def _fused_linear_attn_prefill_kv_kernel(
        q,
        k,
        v,
        gate,
        beta,
        output,
        state,
        scale,
        stride_q_tok: tl.constexpr,
        stride_q_head: tl.constexpr,
        stride_k_tok: tl.constexpr,
        stride_k_head: tl.constexpr,
        stride_v_tok: tl.constexpr,
        stride_v_head: tl.constexpr,
        stride_gate_tok: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_output_tok: tl.constexpr,
        stride_output_head: tl.constexpr,
        stride_state_head: tl.constexpr,
        stride_state_k: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        p_state = (
            state
            + i_h * stride_state_head
            + o_k[:, None] * stride_state_k
            + o_v[None, :]
        )
        b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)

        for i_t in range(T):
            p_q = q + i_t * stride_q_tok + i_h * stride_q_head + o_k
            p_k = k + i_t * stride_k_tok + i_h * stride_k_head + o_k
            p_v = v + i_t * stride_v_tok + i_h * stride_v_head + o_v
            p_gate = gate + i_t * stride_gate_tok + i_h
            p_beta = beta + i_t * stride_beta_tok + i_h

            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_gate).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], axis=0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            p_out = output + i_t * stride_output_tok + i_h * stride_output_head + o_v
            b_o = tl.sum(b_h * b_q[:, None], axis=0)
            tl.store(p_out, b_o.to(output.dtype.element_ty), mask=mask_v)

        tl.store(p_state, b_h.to(state.dtype.element_ty), mask=mask_h)


def fused_gdn_gating_sglang(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for the sglang linear attention backend")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected a and b to be 2D, got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if a.shape != b.shape:
        raise ValueError(f"Expected a and b to have same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    t, hv = a.shape
    if A_log.numel() != hv or dt_bias.numel() != hv:
        raise ValueError(
            f"Expected A_log/dt_bias to have {hv} elements, got {A_log.numel()} and {dt_bias.numel()}"
        )
    gate = torch.empty((t, hv), dtype=torch.float32, device=a.device)
    beta = torch.empty((t, hv), dtype=torch.float32, device=a.device)
    bh = 32
    grid = (t, triton.cdiv(hv, bh))
    _fused_gdn_gating_kernel[grid](
        gate,
        beta,
        A_log,
        a,
        b,
        dt_bias,
        stride_gate_tok=gate.stride(0),
        stride_beta_tok=beta.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        T=t,
        HV=hv,
        BH=bh,
        num_warps=1,
        num_stages=1,
    )
    return gate, beta


def fused_linear_attn_decode_sglang(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    *,
    use_qk_l2norm_in_kernel: bool = True,
    state_layout: str = "kv",
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the sglang linear attention backend")
    if mixed_qkv.ndim != 2:
        raise ValueError(f"Expected mixed_qkv to be 2D, got shape={tuple(mixed_qkv.shape)}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected a and b to be 2D, got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if state.ndim != 4:
        raise ValueError(f"Expected state to be 4D, got shape={tuple(state.shape)}")
    if state_layout not in {"kv", "vk"}:
        raise ValueError(f"Unsupported state layout: {state_layout}")

    batch_size = mixed_qkv.shape[0]
    if state_layout == "kv":
        hv, k_dim, v_dim = state.shape[-3:]
    else:
        hv, v_dim, k_dim = state.shape[-3:]
    if a.shape != (batch_size, hv) or b.shape != (batch_size, hv):
        raise ValueError(
            f"Expected a/b shape {(batch_size, hv)}, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if A_log.numel() != hv or dt_bias.numel() != hv:
        raise ValueError(
            f"Expected A_log/dt_bias to have {hv} elements, got {A_log.numel()} and {dt_bias.numel()}"
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * v_dim
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"Invalid packed QKV shape: {tuple(mixed_qkv.shape)} for state {tuple(state.shape)}")
    q_dim = qk_dim // 2
    if q_dim % k_dim != 0:
        raise ValueError(f"Q dimension {q_dim} is not divisible by K dimension {k_dim}")
    h = q_dim // k_dim
    if h <= 0 or hv % h != 0:
        raise ValueError(f"Invalid grouped-head layout: H={h}, HV={hv}")

    bk = triton.next_power_of_2(k_dim)
    if triton.cdiv(k_dim, bk) != 1:
        raise ValueError(f"Only NK=1 is supported, got K={k_dim}, BK={bk}")
    bv = min(triton.next_power_of_2(v_dim), 32)

    output = torch.empty((batch_size, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    grid = (triton.cdiv(v_dim, bv), batch_size * hv)
    _fused_linear_attn_decode_kv_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        output=output,
        state=state,
        state_indices=state_indices,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_output_tok=output.stride(0),
        stride_state_token=state.stride(0),
        stride_state_head=state.stride(1),
        stride_state_k=state.stride(2),
        stride_indices_tok=state_indices.stride(0),
        H=h,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        STATE_LAYOUT_VK=state_layout == "vk",
        num_warps=1,
        num_stages=3,
    )
    return output


def fused_linear_attn_decode_sglang_packed(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    try:
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode as sglang_fused_recurrent_gated_delta_rule_packed_decode,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sglang packed decode kernel is unavailable") from exc
    if state.ndim != 4:
        raise ValueError(f"Expected state to be 4D, got shape={tuple(state.shape)}")
    batch_size = mixed_qkv.shape[0]
    hv, v_dim, _ = state.shape[-3:]
    output = torch.empty((batch_size, 1, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    sglang_fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        out=output,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return output[:, 0]


def fused_linear_attn_prefill_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the sglang linear attention backend")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            f"Expected q/k/v to be 3D, got q.shape={tuple(q.shape)}, k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
        )
    if gate.ndim != 2 or beta.ndim != 2:
        raise ValueError(
            f"Expected gate/beta to be 2D, got gate.shape={tuple(gate.shape)}, beta.shape={tuple(beta.shape)}"
        )
    if state.ndim != 3:
        raise ValueError(f"Expected state to be 3D, got shape={tuple(state.shape)}")

    t, hv, k_dim = q.shape
    if k.shape != (t, hv, k_dim):
        raise ValueError(f"Expected k shape {(t, hv, k_dim)}, got {tuple(k.shape)}")
    if gate.shape != (t, hv) or beta.shape != (t, hv):
        raise ValueError(
            f"Expected gate/beta shape {(t, hv)}, got {tuple(gate.shape)} and {tuple(beta.shape)}"
        )
    if state.shape[0] != hv or state.shape[1] != k_dim:
        raise ValueError(
            f"Expected state leading shape {(hv, k_dim)}, got {tuple(state.shape[:2])}"
        )
    v_dim = state.shape[2]
    if v.shape != (t, hv, v_dim):
        raise ValueError(f"Expected v shape {(t, hv, v_dim)}, got {tuple(v.shape)}")
    initial_state = state.permute(0, 2, 1).unsqueeze(0).contiguous()
    initial_state_indices = torch.zeros((1,), device=state.device, dtype=torch.long)
    out, _, h = chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.to(q.dtype).unsqueeze(0),
        g=gate.unsqueeze(0),
        beta=beta.unsqueeze(0),
        scale=scale,
        initial_state=initial_state.to(q.dtype),
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    state.copy_(initial_state[0].permute(0, 2, 1).to(state.dtype))
    return out.squeeze(0).to(v.dtype)


# ---------------------------------------------------------------------------
#  Fused QKV split for GDN prefill (ported from sglang main)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_qkv_split_gdn_prefill_kernel(
    q,
    k,
    v,
    mixed_qkv,
    MIXED_QKV_STRIDE_T: tl.constexpr,
    MIXED_QKV_STRIDE_D: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused QKV split: one triton kernel replaces 3 aten::slice + copy_ calls."""
    i_t = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
    k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
    qk_dim: tl.constexpr = q_dim + k_dim
    qkv_dim: tl.constexpr = qk_dim + v_dim

    mask = offsets < qkv_dim
    values = tl.load(
        mixed_qkv + i_t * MIXED_QKV_STRIDE_T + offsets * MIXED_QKV_STRIDE_D,
        mask=mask,
    )

    q_mask = offsets < q_dim
    tl.store(q + i_t * q_dim + offsets, values, mask=q_mask)

    k_offsets = offsets - q_dim
    k_mask = (offsets >= q_dim) & (offsets < qk_dim)
    tl.store(k + i_t * k_dim + k_offsets, values, mask=k_mask)

    v_offsets = offsets - qk_dim
    v_mask = (offsets >= qk_dim) & (offsets < qkv_dim)
    tl.store(v + i_t * v_dim + v_offsets, values, mask=v_mask)


def fused_qkv_split_gdn_prefill(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
):
    """Split packed post-conv GDN QKV into contiguous tensors with fused kernel."""
    seq_len = mixed_qkv.shape[0]
    q = torch.empty(seq_len, num_q_heads * head_q, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty(seq_len, num_k_heads * head_k, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    v = torch.empty(seq_len, num_v_heads * head_v, dtype=mixed_qkv.dtype, device=mixed_qkv.device)

    qkv_dim = num_q_heads * head_q + num_k_heads * head_k + num_v_heads * head_v
    _fused_qkv_split_gdn_prefill_kernel[(seq_len,)](
        q, k, v, mixed_qkv,
        mixed_qkv.stride(0), mixed_qkv.stride(1),
        num_q_heads, num_k_heads, num_v_heads,
        head_q, head_k, head_v,
        BLOCK_SIZE=triton.next_power_of_2(qkv_dim),
        num_warps=8,
        num_stages=3,
    )
    return q, k, v
