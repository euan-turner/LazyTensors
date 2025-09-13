#pragma once

#include "module/module.hpp"
#include "tensor/tensor.hpp"

using namespace module;

namespace optim {

class Optimiser {
  public:
    void registerModule(std::shared_ptr<Module> module);

    virtual void step() = 0;
    size_t getModuleCount() const { return modules_.size(); }

    virtual ~Optimiser() = default;

  protected:
    std::vector<std::shared_ptr<Module>> modules_;

    // for each module, trainable param name -> param tensor
    std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> getAllTrainableParams() const;

    // for each module, trainable param name -> gradient tensor
    std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> getAllGradients() const;
};

}