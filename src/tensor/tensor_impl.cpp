#include "tensor/tensor_impl.hpp"
#include "tensor/cpu_impl.hpp"
#include "tensor/cuda_impl.hpp"
#include <stdexcept>
#include <cstring>

namespace tensor {

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

std::unique_ptr<TensorImpl> TensorImpl::create_impl(Device device, std::shared_ptr<TensorShape> shape) {
      switch (device) {
        case Device::CPU:
          return std::make_unique<CPUImpl>(shape);
#ifdef CUDA_AVAILABLE
        case Device::CUDA:
          return std::make_unique<CUDAImpl>(shape);
#endif
        default:
          throw std::runtime_error("Unknown device");
      }
    }
}