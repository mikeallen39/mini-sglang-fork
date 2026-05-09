#include <ATen/ops/_int_mm.h>
#include <torch/extension.h>

#include <vector>

at::Tensor launch_scaled_epilogue_cuda(
    const at::Tensor& acc,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    c10::ScalarType out_dtype,
    const c10::optional<at::Tensor>& bias);

namespace {

void check_inputs(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "int8_scaled_mm expects mat_a on CUDA");
  TORCH_CHECK(mat_b.is_cuda(), "int8_scaled_mm expects mat_b on CUDA");
  TORCH_CHECK(scales_a.is_cuda(), "int8_scaled_mm expects scales_a on CUDA");
  TORCH_CHECK(scales_b.is_cuda(), "int8_scaled_mm expects scales_b on CUDA");
  TORCH_CHECK(mat_a.scalar_type() == at::kChar, "int8_scaled_mm expects mat_a int8");
  TORCH_CHECK(mat_b.scalar_type() == at::kChar, "int8_scaled_mm expects mat_b int8");
  TORCH_CHECK(scales_a.scalar_type() == at::kFloat, "int8_scaled_mm expects scales_a float32");
  TORCH_CHECK(scales_b.scalar_type() == at::kFloat, "int8_scaled_mm expects scales_b float32");
  TORCH_CHECK(mat_a.dim() == 2, "int8_scaled_mm expects mat_a to be 2D");
  TORCH_CHECK(mat_b.dim() == 2, "int8_scaled_mm expects mat_b to be 2D");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "int8_scaled_mm shape mismatch");
  TORCH_CHECK(mat_a.size(1) % 16 == 0, "int8_scaled_mm requires K % 16 == 0");
  TORCH_CHECK(mat_b.size(0) % 16 == 0, "int8_scaled_mm requires K % 16 == 0");
  TORCH_CHECK(mat_b.size(1) > 0, "int8_scaled_mm expects N > 0");
  TORCH_CHECK(mat_b.size(1) % 8 == 0, "int8_scaled_mm requires N % 8 == 0");
  TORCH_CHECK(
      scales_a.numel() == mat_a.size(0),
      "int8_scaled_mm expects scales_a numel == M, got ",
      scales_a.numel(),
      " vs ",
      mat_a.size(0));
  TORCH_CHECK(
      scales_b.numel() == mat_b.size(1),
      "int8_scaled_mm expects scales_b numel == N, got ",
      scales_b.numel(),
      " vs ",
      mat_b.size(1));
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "int8_scaled_mm expects bias on CUDA");
    TORCH_CHECK(bias->dim() == 1, "int8_scaled_mm expects bias to be 1D");
    TORCH_CHECK(
        bias->numel() == mat_b.size(1),
        "int8_scaled_mm expects bias numel == N, got ",
        bias->numel(),
        " vs ",
        mat_b.size(1));
  }
}

}  // namespace

at::Tensor int8_scaled_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    at::Tensor scales_a,
    at::Tensor scales_b,
    c10::ScalarType out_dtype,
    const c10::optional<at::Tensor>& bias = c10::nullopt) {
  check_inputs(mat_a, mat_b, scales_a, scales_b, bias);
  TORCH_CHECK(
      out_dtype == at::kHalf || out_dtype == at::kBFloat16,
      "int8_scaled_mm only supports fp16/bf16 output");

  scales_a = scales_a.contiguous().view({mat_a.size(0)});
  scales_b = scales_b.contiguous().view({mat_b.size(1)});

  at::Tensor mat_a_for_gemm = mat_a.contiguous();
  const int64_t original_m = mat_a.size(0);
  if (original_m < 17) {
    mat_a_for_gemm = at::zeros(
        {17, mat_a.size(1)},
        mat_a.options());
    mat_a_for_gemm.narrow(0, 0, original_m).copy_(mat_a);
  }

  at::Tensor acc = at::_int_mm(mat_a_for_gemm, mat_b.contiguous());
  if (original_m < 17) {
    acc = acc.narrow(0, 0, original_m).contiguous();
  }

  return launch_scaled_epilogue_cuda(
      acc,
      scales_a,
      scales_b,
      out_dtype,
      bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "int8_scaled_mm",
      &int8_scaled_mm,
      "Minimal int8 GEMM with CUDA epilogue",
      py::arg("mat_a"),
      py::arg("mat_b"),
      py::arg("scales_a"),
      py::arg("scales_b"),
      py::arg("out_dtype"),
      py::arg("bias") = py::none());
}
