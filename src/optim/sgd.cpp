#include "optim/sgd.hpp"

namespace optim {

void SGD::step() {
  auto allParameters = getAllTrainableParams();
  auto allGradients = getAllGradients();

  size_t num_modules = getModuleCount();

  for (size_t i = 0; i < num_modules; ++i) {
    const auto& params = allParameters[i];
    const auto& grads = allGradients[i];

    // SGD: params <- params - lr * grad
    for (const auto& namedParam : params) {
      const std::string& name = namedParam.first;
      std::shared_ptr<Tensor> param = namedParam.second;

      if (grads.count(name)) {
        std::shared_ptr<Tensor> grad = grads.at(name);
        param->sub_(grad->mul(lr_));
      } else {
        throw std::runtime_error("SGD: Trainable parameter name not found in grads");
      }
    }
  }
}

}