#pragma once
#include "tensor/tensor.hpp"
#include "init/xavier.hpp"
#include <catch_amalgamated.hpp>
#include <vector>

using namespace tensor;
using namespace init;
using Catch::Approx;

inline Tensor make_test_cpu_tensor(std::vector<size_t> dims) {
  Tensor t(dims, Device::CPU);
  size_t fan_in = 4, fan_out = 3;
  float gain = 1.0f;
  Xavier x(fan_in, fan_out, gain);
  x.initialise(t);
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
