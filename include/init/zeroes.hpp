#pragma once

#include "init/initialiser.hpp"
#include "tensor/tensor.hpp"

using namespace tensor;

namespace init {

class Zeroes final : public Initialiser {
  public:
    void initialise(Tensor& tensor) override;
};

}