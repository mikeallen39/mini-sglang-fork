from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


CHUNK_SIZE = 64


def _safe_exp(x):
    return torch.exp(torch.where(x <= 0, x, torch.full_like(x, float("-inf"))))


def _prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat([torch.arange(n, device=cu_seqlens.device) for n in triton.cdiv(lens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@triton.jit
def _safe_exp_kernel(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit(do_not_specialize=["T"])
def _chunk_local_cumsum_scalar_kernel(
    s,
    o,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(g: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    if g.ndim != 3:
        raise ValueError(f"Expected g to be [B, T, H], got {tuple(g.shape)}")
    b, t, h = g.shape
    out = torch.empty_like(g, dtype=torch.float32)
    grid = (triton.cdiv(t, chunk_size), b * h)
    _chunk_local_cumsum_scalar_kernel[grid](
        g.view(b * t, h),
        out.view(b * t, h),
        T=t,
        H=h,
        BT=chunk_size,
        num_warps=8,
        num_stages=3,
    )
    return out


@triton.jit
def _chunk_build_A_kernel(
    k,
    beta,
    g_cumsum,
    A,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)
    o_i = tl.arange(0, BT)
    o_j = tl.arange(0, BT)
    base = i_chunk * BT
    if base >= T:
        return

    acc = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(K):
        p = k + (base * H + i_h) * K + i_k
        b_k = tl.load(p + o_i[:, None] * H * K, mask=(base + o_i[:, None]) < T, other=0.0)
        c_k = tl.load(p + o_j[None, :] * H * K, mask=(base + o_j[None, :]) < T, other=0.0)
        acc += b_k * c_k

    p_beta = beta + base * H + i_h
    p_g = g_cumsum + base * H + i_h
    b_beta = tl.load(p_beta + o_i * H, mask=(base + o_i) < T, other=0.0).to(tl.float32)
    b_gi = tl.load(p_g + o_i * H, mask=(base + o_i) < T, other=0.0).to(tl.float32)
    b_gj = tl.load(p_g + o_j * H, mask=(base + o_j) < T, other=0.0).to(tl.float32)
    acc *= _safe_exp_kernel(b_gi[:, None] - b_gj[None, :])
    mask = o_i[:, None] > o_j[None, :]
    acc = tl.where(mask, acc * b_beta[:, None], 0.0)
    p_A = A + (base * H + i_h) * BT
    tl.store(
        p_A + o_i[:, None] * H * BT + o_j[None, :],
        acc.to(A.dtype.element_ty),
        mask=((base + o_i[:, None]) < T) & (o_j[None, :] < BT),
    )


@triton.jit(do_not_specialize=["T"])
def _chunk_recompute_w_u_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    p_beta = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + i_h * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_h * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + i_h * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (i_h // (H // Hg)) * K, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + i_h * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, hg, k_dim = k.shape
    h = v.shape[-2]
    v_dim = v.shape[-1]
    assert b == 1, "Only equal-length B=1 path is supported in this minimal vendored backend."
    A = torch.empty((b, t, h, CHUNK_SIZE), device=k.device, dtype=k.dtype)
    grid_A = (triton.cdiv(t, CHUNK_SIZE), h)
    _chunk_build_A_kernel[grid_A](
        k[0],
        beta[0],
        g[0],
        A[0],
        T=t,
        H=h,
        K=k_dim,
        BT=CHUNK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    # Convert A rows from strict-lower L into inverse(I+L) exactly like current local path.
    A_fp32 = A.float()
    eye = torch.eye(CHUNK_SIZE, device=k.device, dtype=torch.float32)
    for head in range(h):
        for start in range(0, t, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, t)
            n = end - start
            tri = eye[:n, :n] + A_fp32[0, start:end, head, :n]
            inv = torch.linalg.inv(tri)
            A_fp32[0, start:end, head, :n] = inv
    A = A_fp32.to(k.dtype)

    u = torch.empty_like(v)
    w = k.new_empty(b, t, h, k_dim)
    grid_recompute = (triton.cdiv(t, CHUNK_SIZE), b * h)
    _chunk_recompute_w_u_kernel[grid_recompute](
        k=k[0],
        v=v[0],
        beta=beta[0],
        w=w[0],
        u=u[0],
        A=A[0],
        g=g[0],
        T=t,
        H=h,
        Hg=hg,
        K=k_dim,
        V=v_dim,
        BT=CHUNK_SIZE,
        BK=64,
        BV=64,
        num_warps=4,
        num_stages=3,
    )
    return w, u, A


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Minimal equal-length fallback matching the previously validated reference semantics.
    assert k.shape[0] == 1
    _, t, hg, k_dim = k.shape
    h = w.shape[2]
    v_dim = u.shape[-1]
    state = initial_state[initial_state_indices[0]].permute(0, 2, 1).contiguous().float()
    h_chunks = []
    v_new = torch.empty_like(u[0], dtype=torch.float32)
    for start in range(0, t, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, t)
        h_chunks.append(state.clone())
        g_last = g[0, end - 1].float().exp().view(h, 1, 1)
        cross = torch.einsum("thk,hkv->thv", w[0, start:end].float(), state)
        v_new[start:end] = u[0, start:end].float() - cross
        state = state * g_last + torch.einsum("thk,thv->hkv", k[0, start:end].float() * torch.exp(g[0, end - 1].float().view(1, h) - g[0, start:end].float()).unsqueeze(-1), v_new[start:end])
    h_out = torch.stack(h_chunks, dim=0)  # [NT, H, K, V]
    final_state = state
    return h_out.to(k.dtype), v_new.unsqueeze(0).to(u.dtype), final_state.to(k.dtype)


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Minimal equal-length fallback matching the previously validated reference semantics.
    assert q.shape[0] == 1
    _, t, hg, k_dim = q.shape
    h_heads = v.shape[2]
    out = torch.empty_like(v)
    for start in range(0, t, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, t)
        q_chunk = q[0, start:end].float()
        k_chunk = k[0, start:end].float()
        v_chunk = v[0, start:end].float()
        g_chunk = g[0, start:end].float()
        h_boundary = h[start // CHUNK_SIZE].float()
        cross = torch.einsum("thk,hkv->thv", q_chunk * torch.exp(g_chunk).unsqueeze(-1), h_boundary)
        g_diff = g_chunk[:, None, :] - g_chunk[None, :, :]  # [t, s, h]
        scores = torch.einsum("thk,shk->tsh", q_chunk, k_chunk)  # [t, s, h]
        decay = torch.exp(torch.tril(g_diff.permute(2, 0, 1), diagonal=0)).permute(1, 2, 0)  # [t, s, h]
        mask = torch.tril(torch.ones((end - start, end - start), device=q.device, dtype=torch.bool))
        scores = torch.where(mask.unsqueeze(-1), scores * decay, torch.zeros_like(scores))
        intra = torch.einsum("tsh,shv->thv", scores, v_chunk)
        out[0, start:end] = (cross + intra) * scale
    return out


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
):
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-6).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-6).to(k.dtype)
    g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE)
    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
    )
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale)
    return o, A, final_state
