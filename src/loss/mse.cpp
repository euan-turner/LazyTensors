#include "loss/mse.hpp"
#include "tensor/tensor.hpp"

using namespace tensor;

namespace loss {

float MSE::forward(Tensor& pred, Tensor& target) {
  last_pred = pred;
  last_target = target;

  // TODO: Add pow to Tensor
  Tensor res = pred.sub(target).mul_(pred.sub(target)).mean();
  return res(0);
}

Tensor MSE::backward() {
  size_t N = last_pred.numel();
  return last_pred.sub(last_target).mul(2.0f / N);
}

}