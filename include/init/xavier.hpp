#pragma once

#include <random>
#include "init/initialiser.hpp"
#include "tensor/tensor.hpp"

using namespace tensor;

namespace init {

class Xavier final : public Initialiser {
  private:
    size_t _fan_in, _fan_out;
    float _gain;
    std::mt19937 _gen;

  public:
    Xavier(size_t fan_in, size_t fan_out, float gain, unsigned seed = std::random_device{}());
    void initialise(Tensor& tensor) override;
};

}