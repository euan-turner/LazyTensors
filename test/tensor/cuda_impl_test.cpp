#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"

#include "tensor/test_helpers.hpp"



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

TEST_CASE("CPU vs CUDA broadcasting elementwise binary ops", "[CUDA][Broadcast]") {
  // Case 1: add_ with RHS having a unit inner dimension (2x3) <- (2x1)
  Tensor a1 = Tensor({2,3}, Device::CPU);
  a1.set(0,0, 1.0f); a1.set(0,1, 2.0f); a1.set(0,2, 3.0f);
  a1.set(1,0, 4.0f); a1.set(1,1, 5.0f); a1.set(1,2, 6.0f);
  Tensor b1 = Tensor({2,1}, Device::CPU);
  b1.set(0,0, 10.0f);
  b1.set(1,0, 20.0f);

  SECTION("add_ with b having unit inner dim") {
    // CPU result
    Tensor cpu_a1 = a1.clone();
    cpu_a1.add_(b1);

    // CUDA result
    Tensor cuda_a1 = a1; cuda_a1.to(Device::CUDA);
    Tensor cuda_b1 = b1; cuda_b1.to(Device::CUDA);
    // Make device-local contiguous/broadcast-friendly copy of RHS
    cuda_b1 = cuda_b1.clone();
    Tensor cuda_res1 = cuda_a1.clone();
    cuda_res1.add_(cuda_b1);
    cuda_res1.to(Device::CPU);
    require_tensors_equal(cpu_a1, cuda_res1);
  }

  // Case 2: sub_ with RHS having fewer leading dims (2x3x2) <- (3x2)
  Tensor a2 = Tensor({2,3,2}, Device::CPU);
  a2.set({0,0,0}, 1.0f); a2.set({0,0,1}, 2.0f);
  a2.set({0,1,0}, 3.0f); a2.set({0,1,1}, 4.0f);
  a2.set({0,2,0}, 5.0f); a2.set({0,2,1}, 6.0f);
  a2.set({1,0,0}, 7.0f); a2.set({1,0,1}, 8.0f);
  a2.set({1,1,0}, 9.0f); a2.set({1,1,1}, 10.0f);
  a2.set({1,2,0}, 11.0f); a2.set({1,2,1}, 12.0f);

  Tensor b2 = Tensor({3,2}, Device::CPU);
  b2.set({0,0}, 2.0f); b2.set({0,1}, 4.0f);
  b2.set({1,0}, 6.0f); b2.set({1,1}, 8.0f);
  b2.set({2,0}, 10.0f); b2.set({2,1}, 12.0f);

  SECTION("sub_ with b having fewer leading dims (leading broadcast)") {
    Tensor cpu_a2 = a2.clone(); cpu_a2.sub_(b2);
    Tensor cuda_a2 = a2; cuda_a2.to(Device::CUDA);
    Tensor cuda_b2 = b2; cuda_b2.to(Device::CUDA);
    cuda_b2 = cuda_b2.clone();
    Tensor cuda_res2 = cuda_a2.clone(); cuda_res2.sub_(cuda_b2);
    cuda_res2.to(Device::CPU);
    require_tensors_equal(cpu_a2, cuda_res2);
  }

  // Case 3: mul_ with combined fewer dims and unit dims (2x3x4) <- (1x3x1)
  Tensor a3 = Tensor({2,3,4}, Device::CPU);
  // Fill 1..24 row-major
  float v = 1.0f;
  for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 3; ++j) for (size_t k = 0; k < 4; ++k) { a3.set({i,j,k}, v); v += 1.0f; }

  Tensor b3 = Tensor({1,3,1}, Device::CPU);
  b3.set({0,0,0}, 1.0f); b3.set({0,1,0}, 2.0f); b3.set({0,2,0}, 3.0f);

  SECTION("mul_ with combined leading and unit-dimension broadcasts") {
    Tensor cpu_a3 = a3.clone(); cpu_a3.mul_(b3);
    Tensor cuda_a3 = a3; cuda_a3.to(Device::CUDA);
    Tensor cuda_b3 = b3; cuda_b3.to(Device::CUDA);
    cuda_b3 = cuda_b3.clone();
    Tensor cuda_res3 = cuda_a3.clone(); cuda_res3.mul_(cuda_b3);
    cuda_res3.to(Device::CPU);
    require_tensors_equal(cpu_a3, cuda_res3);
  }

  // Case 4: div_ with scalar-like RHS (vector of length 1) broadcasting to vector
  Tensor a4 = Tensor::vector(4, Device::CPU);
  a4.set(0, 8.0f); a4.set(1, 16.0f); a4.set(2, 32.0f); a4.set(3, 64.0f);
  Tensor b4 = Tensor::vector(1, Device::CPU); b4.set(0, 2.0f);

  SECTION("div_ with scalar-like RHS broadcasting to vector") {
    Tensor cpu_a4 = a4.clone(); cpu_a4.div_(b4);
    Tensor cuda_a4 = a4; cuda_a4.to(Device::CUDA);
    Tensor cuda_b4 = b4; cuda_b4.to(Device::CUDA);
    cuda_b4 = cuda_b4.clone();
    Tensor cuda_res4 = cuda_a4.clone(); cuda_res4.div_(cuda_b4);
    cuda_res4.to(Device::CPU);
    require_tensors_equal(cpu_a4, cuda_res4);
  }
}