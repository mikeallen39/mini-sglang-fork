#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

template <typename OutType>
struct CastHelper;

template <>
struct CastHelper<__half> {
  __device__ static inline __half cast(float value) { return __float2half_rn(value); }
};

template <>
struct CastHelper<__nv_bfloat16> {
  __device__ static inline __nv_bfloat16 cast(float value) { return __float2bfloat16(value); }
};

template <typename OutType>
__global__ void scaled_epilogue_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ scales_a,
    const float* __restrict__ scales_b,
    const float* __restrict__ bias,
    OutType* __restrict__ out,
    int64_t m,
    int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m * n;
  if (idx >= total) {
    return;
  }

  const int64_t row = idx / n;
  const int64_t col = idx - row * n;
  float value = static_cast<float>(acc[idx]) * scales_a[row] * scales_b[col];
  if (bias != nullptr) {
    value += bias[col];
  }
  out[idx] = CastHelper<OutType>::cast(value);
}

template <typename OutType>
at::Tensor run_scaled_epilogue(
    const at::Tensor& acc,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& bias_fp32,
    c10::ScalarType out_dtype) {
  auto out = at::empty({acc.size(0), acc.size(1)}, acc.options().dtype(out_dtype));
  const int threads = 256;
  const int64_t total = acc.numel();
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const float* bias_ptr = bias_fp32.defined() ? bias_fp32.data_ptr<float>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream(acc.get_device());
  scaled_epilogue_kernel<<<blocks, threads, 0, stream.stream()>>>(
      acc.data_ptr<int32_t>(),
      scales_a.data_ptr<float>(),
      scales_b.data_ptr<float>(),
      bias_ptr,
      reinterpret_cast<OutType*>(out.data_ptr()),
      acc.size(0),
      acc.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

at::Tensor launch_scaled_epilogue_cuda(
    const at::Tensor& acc,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    c10::ScalarType out_dtype,
    const c10::optional<at::Tensor>& bias) {
  c10::cuda::CUDAGuard device_guard(acc.device());

  TORCH_CHECK(acc.is_cuda(), "scaled epilogue expects CUDA accumulators");
  TORCH_CHECK(acc.scalar_type() == at::kInt, "scaled epilogue expects int32 accumulators");

  at::Tensor bias_fp32;
  if (bias.has_value()) {
    bias_fp32 = bias->contiguous().to(at::kFloat);
  }

  if (out_dtype == at::kHalf) {
    return run_scaled_epilogue<__half>(
        acc.contiguous(),
        scales_a.contiguous(),
        scales_b.contiguous(),
        bias_fp32,
        out_dtype);
  }
  if (out_dtype == at::kBFloat16) {
    return run_scaled_epilogue<__nv_bfloat16>(
        acc.contiguous(),
        scales_a.contiguous(),
        scales_b.contiguous(),
        bias_fp32,
        out_dtype);
  }

  TORCH_CHECK(false, "Unsupported output dtype for scaled epilogue");
}
