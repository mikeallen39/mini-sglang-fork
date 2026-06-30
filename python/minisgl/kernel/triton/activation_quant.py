import triton
import triton.language as tl


@triton.jit
def gemma_rmsnorm_quant_int8_kernel(
    input_ptr,
    weight_ptr,
    output_q_ptr,
    output_s_ptr,
    input_stride_0,
    input_stride_1,
    output_q_stride_0,
    output_q_stride_1,
    output_s_stride_0,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row = pid_m
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    num_blocks = tl.cdiv(N, BLOCK_N)
    row_ptr = input_ptr + row * input_stride_0
    acc_sumsq = tl.zeros((1,), dtype=tl.float32)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N
        x = tl.load(
            row_ptr + block_offs * input_stride_1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc_sumsq += tl.sum(x * x, axis=0)

    variance = acc_sumsq / N
    rstd = tl.rsqrt(variance + eps)
    acc_max = tl.zeros((1,), dtype=tl.float32)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N
        x = tl.load(
            row_ptr + block_offs * input_stride_1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight_ptr + block_offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * (1.0 + w)
        acc_max = tl.maximum(acc_max, tl.max(tl.abs(y), axis=0))

    scale = tl.maximum(acc_max / 127.0, 1e-10)
    tl.store(output_s_ptr + row * output_s_stride_0 + tl.arange(0, 1), scale)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N
        x = tl.load(
            row_ptr + block_offs * input_stride_1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight_ptr + block_offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * (1.0 + w)
        q = tl.extra.cuda.libdevice.llrint(y / scale)
        q = tl.maximum(tl.minimum(q, 127.0), -128.0)
        out_ptr = output_q_ptr + row * output_q_stride_0 + block_offs * output_q_stride_1
        tl.store(out_ptr, q.to(tl.int8), mask=mask)


@triton.jit
def per_token_quant_int8_kernel(
    input_ptr,
    output_q_ptr,
    output_s_ptr,
    input_stride_0,
    input_stride_1,
    output_q_stride_0,
    output_q_stride_1,
    output_s_stride_0,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row = pid_m
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    in_ptr = input_ptr + row * input_stride_0 + offs
    acc_max = tl.zeros((1,), dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_N)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N
        x = tl.load(in_ptr + block_idx * BLOCK_N, mask=mask, other=0.0).to(tl.float32)
        acc_max = tl.maximum(acc_max, tl.max(tl.abs(x), axis=0))

    scale = tl.maximum(acc_max / 127.0, 1e-10)
    tl.store(output_s_ptr + row * output_s_stride_0 + tl.arange(0, 1), scale)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N
        x = tl.load(in_ptr + block_idx * BLOCK_N, mask=mask, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.llrint(x / scale)
        q = tl.maximum(tl.minimum(q, 127.0), -128.0)
        out_ptr = output_q_ptr + row * output_q_stride_0 + block_offs
        tl.store(out_ptr, q.to(tl.int8), mask=mask)


@triton.jit
def silu_and_mul_quant_int8_kernel(
    input_ptr,
    output_q_ptr,
    output_s_ptr,
    input_stride_0,
    input_stride_1,
    output_q_stride_0,
    output_q_stride_1,
    output_s_stride_0,
    M,
    N2,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row = pid_m
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    gate_ptr = input_ptr + row * input_stride_0 + offs
    up_ptr = input_ptr + row * input_stride_0 + N2 + offs

    acc_max = tl.zeros((1,), dtype=tl.float32)
    num_blocks = tl.cdiv(N2, BLOCK_N)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N2
        gate = tl.load(gate_ptr + block_idx * BLOCK_N, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + block_idx * BLOCK_N, mask=mask, other=0.0).to(tl.float32)
        silu = gate * tl.sigmoid(gate)
        y = silu * up
        acc_max = tl.maximum(acc_max, tl.max(tl.abs(y), axis=0))

    scale = tl.maximum(acc_max / 127.0, 1e-10)
    tl.store(output_s_ptr + row * output_s_stride_0 + tl.arange(0, 1), scale)

    for block_idx in range(0, num_blocks):
        block_offs = offs + block_idx * BLOCK_N
        mask = block_offs < N2
        gate = tl.load(gate_ptr + block_idx * BLOCK_N, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + block_idx * BLOCK_N, mask=mask, other=0.0).to(tl.float32)
        silu = gate * tl.sigmoid(gate)
        y = silu * up
        q = tl.extra.cuda.libdevice.llrint(y / scale)
        q = tl.maximum(tl.minimum(q, 127.0), -128.0)
        out_ptr = output_q_ptr + row * output_q_stride_0 + block_offs
        tl.store(out_ptr, q.to(tl.int8), mask=mask)


@triton.jit
def decode_quant_int8_gemm_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    weight_stride_0,
    weight_stride_1,
    output_stride_0,
    output_stride_1,
    M,
    N,
    K,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    a_row_ptrs = input_ptr + offs_m[:, None] * input_stride_0
    acc_max = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        curr_k = k0 * BLOCK_K + offs_k
        a_mask = m_mask[:, None] & (curr_k[None, :] < K)
        x = tl.load(
            a_row_ptrs + curr_k[None, :] * input_stride_1,
            mask=a_mask,
            other=0.0,
        ).to(tl.float32)
        acc_max = tl.maximum(acc_max, tl.max(tl.abs(x), axis=1))

    act_scale = tl.maximum(acc_max / 127.0, 1e-10)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        curr_k = k0 * BLOCK_K + offs_k
        a_mask = m_mask[:, None] & (curr_k[None, :] < K)
        x = tl.load(
            a_row_ptrs + curr_k[None, :] * input_stride_1,
            mask=a_mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.extra.cuda.libdevice.llrint(x / act_scale[:, None])
        q = tl.maximum(tl.minimum(q, 127.0), -128.0).to(tl.int8)

        w_ptrs = weight_ptr + curr_k[:, None] * weight_stride_0 + offs_n[None, :] * weight_stride_1
        w_mask = (curr_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.int8)

        acc += tl.dot(q, w, out_dtype=tl.int32)

    w_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * act_scale[:, None] * w_scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out += bias[None, :]

    out_ptrs = output_ptr + offs_m[:, None] * output_stride_0 + offs_n[None, :] * output_stride_1
    out_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=out_mask)


@triton.jit
def weight_only_int8_gemm_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    weight_stride_0,
    weight_stride_1,
    output_stride_0,
    output_stride_1,
    M,
    N,
    K,
    HAS_BIAS: tl.constexpr,
    OUT_DTYPE_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_row_ptrs = input_ptr + offs_m[:, None] * input_stride_0

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        curr_k = k0 * BLOCK_K + offs_k
        a_mask = m_mask[:, None] & (curr_k[None, :] < K)
        a = tl.load(
            a_row_ptrs + curr_k[None, :] * input_stride_1,
            mask=a_mask,
            other=0.0,
        )

        w_ptrs = weight_ptr + offs_n[:, None] * weight_stride_0 + curr_k[None, :] * weight_stride_1
        w_mask = (offs_n[:, None] < N) & (curr_k[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0).to(a.dtype)
        acc += tl.dot(a, tl.trans(w))

    w_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    out = acc * w_scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out += bias[None, :]

    out_ptrs = output_ptr + offs_m[:, None] * output_stride_0 + offs_n[None, :] * output_stride_1
    out_mask = m_mask[:, None] & (offs_n[None, :] < N)
    if OUT_DTYPE_BF16:
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, out.to(tl.float16), mask=out_mask)
