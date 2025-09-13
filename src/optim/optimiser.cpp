#include "optim/optimiser.hpp"

namespace optim {

void Optimiser::registerModule(std::shared_ptr<Module> module) {
  modules_.push_back(module);
}

std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> Optimiser::getAllTrainableParams() const {
  std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> allTrainableParams;

  for (auto module : modules_) {
    allTrainableParams.push_back(module->getTrainableParameters());
  }
}

std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> Optimiser::getAllGradients() const {
  std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> allGradients;

  for (auto module : modules_) {
    allGradients.push_back(module->getGradients());
  }
}

}