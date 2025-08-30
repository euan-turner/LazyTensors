#include "loss/cce.hpp"

using namespace tensor;

namespace loss {

float CCE::forward(Tensor& pred, Tensor& target) {
  last_pred = pred;
  last_target = target;

  Tensor res = pred.clamp(eps, 1.0f - eps).log_().mul_(target).sum();
  float loss = -res(0);

  if (pred.isMatrix()) {
    // batch processing
    size_t batch_size = pred.dim(0);
    loss /= static_cast<float>(batch_size);
  }
  return loss;
}

Tensor CCE::backward() {
  Tensor res = last_target.mul(-1).div_(last_pred.clamp(eps, 1.0f - eps));
  if (last_pred.isMatrix()) {
    size_t batch_size = last_pred.dim(0);
    res.div_(static_cast<float>(batch_size));
  }
  return res;
}

}