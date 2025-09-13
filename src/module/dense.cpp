#include "module/dense.hpp"
#include "init/xavier.hpp"
#include "init/zeroes.hpp"

namespace module {

Dense::Dense(size_t input_size, size_t output_size, std::shared_ptr<Initialiser> weight_init, std::shared_ptr<Initialiser> bias_init) : input_size_(input_size), output_size_(output_size_) {
  parameters_["weights"] = std::make_shared<Tensor>(std::vector<size_t>{output_size, input_size});
  parameters_["biases"] = std::make_shared<Tensor>(std::vector<size_t>{output_size});

  if (!weight_init) {
    weight_init = std::make_shared<Xavier>(input_size_, output_size_, 1.0f);
  }
  if (!bias_init) {
    bias_init = std::make_shared<Zeroes>();
  }
  weight_init->initialise(*parameters_.at("weights"));
  bias_init->initialise(*parameters_.at("biases"));
}

std::shared_ptr<Tensor> Dense::computeForward(const std::shared_ptr<Tensor>& input) {
  if (input->isVector() || input->isMatrix()) {
    cache("input", input);
    if (input->isVector()) {
      Tensor res = parameters_["weights"]->matmul(*input);
      res.add_(*parameters_["biases"]);
      return std::make_shared<Tensor>(res);
    } else if (input->isMatrix()) {
      throw std::runtime_error("Dense: Batched inputs requires Tensor transpose implementation");
    }
  } else {
    throw std::runtime_error("Dense: Input must be a vector or matrix");
  }
}

std::shared_ptr<Tensor> Dense::computeBackward(const std::shared_ptr<Tensor>& grad_output) {
  if (get("input")->isVector()) {
    throw std::runtime_error("Dense: Backward pass requires Tensor transpose"); 
  } else{
    throw std::runtime_error("Dense: Batched inputs requires Tensor transpose implementation");
  }
}


}