#include "module/relu.hpp"
#include <limits>

using namespace tensor;

namespace module {
std::shared_ptr<Tensor> ReLU::computeForward(const std::shared_ptr<Tensor>& input) {
  Tensor relu = input->relu();
  auto res = std::make_shared<Tensor>(relu);
  cache("input", input);
  return res;
}

std::shared_ptr<Tensor> ReLU::computeBackward(const std::shared_ptr<Tensor>& grad_output) {
  Tensor grads = get("input")->relu_back(*grad_output);
  return std::make_shared<Tensor>(grads);
}
}