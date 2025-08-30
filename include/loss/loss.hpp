#pragma once

#include "tensor/tensor.hpp"

using namespace tensor;

namespace loss {
class Loss {
   public:
    // Compute loss between predictions and targets
    virtual float forward(Tensor& pred, Tensor& target) = 0;
    // Compute gradient updates on predictions and return
    virtual Tensor backward() = 0;
};

}  // namespace loss