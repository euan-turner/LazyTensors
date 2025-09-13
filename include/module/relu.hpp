#pragma once

#include "module/module.hpp"

using namespace tensor;

namespace module {

class ReLU : public Module {
  public:
    bool hasTrainableParameters() const override { return false; }
  
  protected:
    std::shared_ptr<Tensor> computeForward(const std::shared_ptr<Tensor>& input);

    std::shared_ptr<Tensor> computeBackward(const std::shared_ptr<Tensor>& grad_output);
};

}