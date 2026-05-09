import triton
import triton.language as tl


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
