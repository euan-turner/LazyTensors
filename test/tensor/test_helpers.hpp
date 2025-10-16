#pragma once
#include "tensor/tensor.hpp"
#include <catch_amalgamated.hpp>
#include <vector>

using namespace tensor;
using Catch::Approx;

inline Tensor make_test_cpu_tensor(std::vector<size_t> dims) {
  Tensor t(dims, Device::CPU);
  // Deterministic initialization pattern so tests are reproducible.
  // Fill with values in range [-5, 5) based on flat index.
  size_t n = t.numel();
  std::vector<size_t> idx(dims.size(), 0);
  for (size_t flat = 0; flat < n; ++flat) {
    size_t rem = flat;
    for (size_t i = 0; i < dims.size(); ++i) {
      idx[i] = rem / t.strides()[i];
      rem %= t.strides()[i];
    }
    float v = static_cast<float>(static_cast<int>(flat % 100)) / 10.0f - 5.0f; // [-5,5)
    t.set(idx, v);
  }
  return t;
}

inline void require_tensors_equal(const Tensor& t1, const Tensor& t2, float eps = 1e-3f, float marg = 1e-5f) {
  REQUIRE(t1.dims() == t2.dims());
  size_t total = t1.numel();
  std::vector<size_t> dims = t1.dims();
  std::vector<size_t> idx(dims.size(), 0);
  for (size_t flat = 0; flat < t1.numel(); ++flat) {
    size_t rem = flat;
    for (size_t i = 0; i < t1.dims().size(); ++i) {
      idx[i] = rem / t1.strides()[i];
      rem %= t1.strides()[i];
    }
    REQUIRE(t1(idx) == Approx(t2(idx)).epsilon(eps).margin(marg));
  }
}
