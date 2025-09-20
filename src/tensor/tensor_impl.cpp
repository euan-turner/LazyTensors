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

std::shared_ptr<TensorImpl> TensorImpl::create_impl(Device device, std::shared_ptr<TensorShape> shape) {
      switch (device) {
        case Device::CPU:
          return std::make_shared<CPUImpl>(shape);
#ifdef CUDA_AVAILABLE
        case Device::CUDA:
          return std::make_shared<CUDAImpl>(shape);
#endif
        default:
          throw std::runtime_error("Unknown device");
      }
    }

void TensorImpl::transpose(const std::vector<size_t>& axes) {
  if (axes.size() != _shape->strides.size()) {
    throw std::runtime_error("Insufficient number of axes provided");
  }

  std::vector<size_t> new_dims(axes.size());
  for (int i = 0; i < axes.size(); ++i) {
    new_dims[i] = _shape->dims[axes[i]];
  }

  // Replace the shape that this Tensor, and the TensorImpl point at
  // if this impl was cloned, previous version will keep using the old stride pattern,
  // if they are still referenced somewhere
  _shape = _shape->transpose(axes);
}

}