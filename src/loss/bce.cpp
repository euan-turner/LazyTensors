#include "loss/bce.hpp"

using namespace tensor;

namespace loss {

float BCE::forward(Tensor& pred, Tensor& target) {
  Tensor pred_clamped = pred.clamp(eps, 1.0f - eps);
  last_pred = pred_clamped;
  last_target = target;

  Tensor l = pred_clamped.log();     // out-of-place
  l.mul_(target);                    // safe in-place on l

  Tensor r1 = target.mul(-1.0f);     // out-of-place
  r1.add_(1.0f);                     // safe in-place on r1

  Tensor r2 = pred_clamped.mul(-1.0f); // out-of-place
  r2.add_(1.0f);                       // safe in-place on r2
  r2.log_();                           // safe in-place on r2

  Tensor r = r1.mul(r2);             // out-of-place multiply
  Tensor res = l.add(r);             // out-of-place add
  return -res(0);

}

Tensor BCE::backward() {
  float N = static_cast<float>(last_pred.numel());
  return last_pred.sub(last_target)
                      .div_(
                        last_pred.mul(
                          last_pred.mul(-1.0f).add_(1.0f)
                        )
                      ).mul_(1.0f / N);
}

}