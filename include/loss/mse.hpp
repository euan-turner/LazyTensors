#pragma once

#include "loss/loss.hpp"
#include "tensor/tensor.hpp"

using namespace tensor;

namespace loss {
/**
 * @brief A Mean-Square Error loss module
 *
 */

class MSE : public Loss {
   private:
    Tensor last_pred;
    Tensor last_target;

   public:
    MSE() : last_pred(Tensor::vector(1)), last_target(Tensor::vector(1)) {}
    float forward(Tensor& pred, Tensor& target) override;
    Tensor backward() override;
};

}  // namespace loss