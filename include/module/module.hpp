#pragma once

#include <unordered_map>
#include "tensor/tensor.hpp"

using namespace tensor;

namespace module {

class Module {
public:
    virtual ~Module() = default; 

    // Forward pass: store any required intermediates
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) {
        cached_tensors_.clear();                     // clear previous caches
        auto output = computeForward(input);        // subclass hook
        // subclass should populate cached_tensors_ via cache() helper in computeForward
        return output;
    }

    // Backward pass: use cached tensors and incoming gradient
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) {
        return computeBackward(grad_output);
    }

    virtual bool hasTrainableParameters() = 0;

    std::unordered_map<std::string, std::shared_ptr<Tensor>>& getParameters() { return parameters_; }
    std::unordered_map<std::string, std::shared_ptr<Tensor>>& getGradients() { return gradients_; }

protected:
    // Hook for subclasses to implement forward computation and cache required tensors
    virtual std::shared_ptr<Tensor> computeForward(const std::shared_ptr<Tensor>& input) = 0;

    // Hook for subclasses to implement backward computation
    virtual std::shared_ptr<Tensor> computeBackward(const std::shared_ptr<Tensor>& grad_output) = 0;

    // Helper for subclasses to store any intermediate tensor
    void cache(const std::string& key, const std::shared_ptr<Tensor>& tensor) {
        cached_tensors_[key] = tensor;
    }

    // Helper for subclasses to retrieve an intermediate tensor
    std::shared_ptr<Tensor> get(const std::string& key) {
        if (cached_tensors_.count(key)) {
            return cached_tensors_[key];
        } else {
            throw std::runtime_error("Unknown cached tensor key");
        }
    }

    // Map of tensors cached during forward for backward use
    std::unordered_map<std::string, std::shared_ptr<Tensor>> cached_tensors_;

    // Parameter storage
    std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> gradients_;
};

}  // namespace module