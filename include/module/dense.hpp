#pragma once

#include "module/module.hpp"
#include "init/initialiser.hpp"

using namespace init;

namespace module {

class Dense : public Module {
  private:
    size_t input_size_;
    size_t output_size_;
  
  public:
    Dense(size_t input_size, size_t output_size,
          std::shared_ptr<Initialiser> weight_init = nullptr, std::shared_ptr<Initialiser> bias_init = nullptr);

    bool hasTrainableParameters() const override { return true; }

  protected:
    std::shared_ptr<Tensor> computeForward(const std::shared_ptr<Tensor>& input) override;

    std::shared_ptr<Tensor> computeBackward(const std::shared_ptr<Tensor>& grad_output) override;


};

}