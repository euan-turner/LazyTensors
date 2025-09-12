#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"
#include "init/xavier.hpp"

using namespace tensor;
using namespace init;
using Catch::Approx;

Tensor make_test_cpu_tensor(std::vector<size_t> dims) {
  Tensor t(dims, Device::CPU);
  size_t fan_in = 4, fan_out = 3;
  float gain = 1.0f;
  Xavier x(fan_in, fan_out, gain);
  x.initialise(t);

  return t;
}

// Move equality check into Tensor
void require_tensors_equal(const Tensor& t1, const Tensor& t2, float eps = 1e-3f, float marg = 1e-5f) {
  REQUIRE(t1.dims() == t2.dims());

  size_t total = t1.numel();

  std::vector<size_t> dims = t1.dims();

  std::vector<size_t> idx(dims.size(), 0);

  // TODO: Move some of this into TensorImpl so it can be made more efficient
  for (size_t flat = 0; flat < t1.numel(); ++flat) {
    size_t rem = flat;
    for (size_t i = 0; i < t1.dims().size(); ++i) {
      idx[i] = rem / t1.strides()[i];
      rem %= t1.strides()[i];
    }
    REQUIRE(t1(idx) == Approx(t2(idx)).epsilon(eps).margin(marg));
  }
}

TEST_CASE("CPU vs CUDA consistency elementwise unary+scalar", "[CUDA]") {
  std::vector<size_t> dims = {256, 256};

  Tensor cpu_tensor = make_test_cpu_tensor(dims);

  Tensor cuda_tensor = cpu_tensor;
  cuda_tensor.to(Device::CUDA);

  SECTION("Add Scalar Out-of-Place") {
    float s = 5.0f;
    Tensor cpu_result = cpu_tensor.add(s);
    Tensor cuda_result = cuda_tensor.add(s);

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Add Scalar In-Place") {
    float s = 3.0f;
    cpu_tensor.add_(s);
    cuda_tensor.add_(s);

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }

  SECTION("Mul Scalar Out-of-Place") {
    float s = 5.0f;
    Tensor cpu_result = cpu_tensor.mul(s);
    Tensor cuda_result = cuda_tensor.mul(s);

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Mul Scalar In-Place") {
    float s = 3.0f;
    cpu_tensor.mul_(s);
    cuda_tensor.mul_(s);

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }

  SECTION("Sub Scalar Out-of-Place") {
    float s = 5.0f;
    Tensor cpu_result = cpu_tensor.sub(s);
    Tensor cuda_result = cuda_tensor.sub(s);

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
}

  SECTION("Sub Scalar In-Place") {
    float s = 3.0f;
    cpu_tensor.sub_(s);
    cuda_tensor.sub_(s);

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }

  SECTION("Div Scalar Out-of-Place") {
    float s = 5.0f;
    Tensor cpu_result = cpu_tensor.div(s);
    Tensor cuda_result = cuda_tensor.div(s);

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Div Scalar In-Place") {
    float s = 3.0f;
    cpu_tensor.div_(s);
    cuda_tensor.div_(s);

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }

  SECTION("Exp Out-of-Place") {
    Tensor cpu_result = cpu_tensor.exp();
    Tensor cuda_result = cuda_tensor.exp();

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Exp In-Place") {
    cpu_tensor.exp_();
    cuda_tensor.exp_();

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }

  SECTION("Log Out-of-Place") {
    // Shift out negatives
    cpu_tensor.add_(5.0f);
    cuda_tensor.add_(5.0f);
    Tensor cpu_result = cpu_tensor.log();
    Tensor cuda_result = cuda_tensor.log();

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Log In-Place") {
      cpu_tensor.add_(5.0f);
    cuda_tensor.add_(5.0f);
    cpu_tensor.log_();
    cuda_tensor.log_();

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }

  SECTION("Clamp Out-of-Place") {
    float lo = -0.5f;
    float hi = -0.5f;
    Tensor cpu_result = cpu_tensor.clamp(lo, hi);
    Tensor cuda_result = cuda_tensor.clamp(lo, hi);

    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Clamp Scalar In-Place") {
    float lo = -0.5f;
    float hi = -0.5f;
    cpu_tensor.clamp_(lo, hi);
    cuda_tensor.clamp_(lo, hi);

    cuda_tensor.to(Device::CPU);
    require_tensors_equal(cpu_tensor, cuda_tensor);
  }
}

TEST_CASE("CPU vs CUDA matmuls", "[CUDA]") {
  std::vector<size_t> dims = {256, 256};

  Tensor cpu_a = make_test_cpu_tensor(dims);
  Tensor cuda_a = cpu_a;
  cuda_a.to(Device::CUDA);

  SECTION("Square Matrix-Matrix Multiplication") {
    Tensor cpu_b = make_test_cpu_tensor(dims);
    Tensor cuda_b = cpu_b;
    cuda_b.to(Device::CUDA);

    Tensor cpu_result = cpu_a.matmul(cpu_b);
    Tensor cuda_result = cuda_a.matmul(cuda_b);
    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Rectangle Matrix-Matrix Multiplication") {
    Tensor cpu_b = make_test_cpu_tensor({256, 128});
    Tensor cuda_b = cpu_b;
    cuda_b.to(Device::CUDA);

    Tensor cpu_result = cpu_a.matmul(cpu_b);
    Tensor cuda_result = cuda_a.matmul(cuda_b);
    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Matrix-Vector Multiplication") {
    Tensor cpu_b = make_test_cpu_tensor({256});
    Tensor cuda_b = cpu_b;
    cuda_b.to(Device::CUDA);

    Tensor cpu_result = cpu_a.matmul(cpu_b);
    Tensor cuda_result = cuda_a.matmul(cuda_b);
    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Vector-Matrix Multiplication") {
    Tensor cpu_b = make_test_cpu_tensor({256});
    Tensor cuda_b = cpu_b;
    cuda_b.to(Device::CUDA);

    Tensor cpu_result = cpu_a.matmul(cpu_b);
    Tensor cuda_result = cuda_a.matmul(cuda_b);
    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }
}

TEST_CASE("CPU vs CUDA dot products", "[CUDA]") {
  SECTION("Single Block Dot Product") {
    Tensor cpu_a = make_test_cpu_tensor({32});
    Tensor cuda_a = cpu_a;
    cuda_a.to(Device::CUDA);
    Tensor cpu_b = make_test_cpu_tensor({32});
    Tensor cuda_b = cpu_b;
    cuda_b.to(Device::CUDA);

    Tensor cpu_result = cpu_a.matmul(cpu_b);
    Tensor cuda_result = cuda_a.matmul(cuda_b);
    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }

  SECTION("Multi-Block Dot Product") {
    Tensor cpu_a = make_test_cpu_tensor({256});
    Tensor cuda_a = cpu_a;
    cuda_a.to(Device::CUDA);
    Tensor cpu_b = make_test_cpu_tensor({256});
    Tensor cuda_b = cpu_b;
    cuda_b.to(Device::CUDA);

    Tensor cpu_result = cpu_a.matmul(cpu_b);
    Tensor cuda_result = cuda_a.matmul(cuda_b);
    cuda_result.to(Device::CPU);
    require_tensors_equal(cpu_result, cuda_result);
  }
}