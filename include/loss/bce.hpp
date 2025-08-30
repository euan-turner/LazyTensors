#pragma once

#include "loss/loss.hpp"
#include "tensor/tensor.hpp"

namespace loss {
/**
 * @brief A Binary Cross-Entropy loss module
 *
 */
class BCE : public Loss {
   private:
    Tensor last_pred;
    Tensor last_target;
    float eps = 1e-7f;

   public:
    BCE() : last_pred(Tensor::vector(1)), last_target(Tensor::vector(1)) {}
    float forward(Tensor& pred, Tensor& target) override;
    Tensor backward() override;
};
}  // namespace loss