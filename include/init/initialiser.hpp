#pragma once

#include "tensor/tensor.hpp"

using namespace tensor;

namespace init {

class Initialiser {
  public:
    virtual ~Initialiser() = default;
    virtual void initialise(Tensor& tensor) = 0;
};

} // namespace init