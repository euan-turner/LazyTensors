#include "tensor/tensor_impl.hpp"
#include "tensor/cpu_impl.hpp"
#include "tensor/cuda_impl.hpp"
#include <stdexcept>
#include <cstring>

namespace tensor {

void TensorImpl::copy_to(TensorImpl& dst) {
  throw std::runtime_error("Not implemented yet");
}

size_t TensorImpl::flatIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != _shape->dims.size()) {
        throw std::runtime_error("Incorrect number of indices");
    }

    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= _shape->dims[i]) {
            throw std::runtime_error("Index out of bounds");
        }
        offset += indices[i] * _shape->strides[i];
    }
    return offset;
}

}