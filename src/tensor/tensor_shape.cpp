#include "tensor/tensor_shape.hpp"
#include <stdexcept>

namespace tensor {
std::shared_ptr<TensorShape> createShape(const std::vector<size_t>& dims) {
    const size_t n = dims.size();
    if (n == 0) throw std::runtime_error("dims cannot be empty for tensor");

    size_t total_size = 1;
    for (const size_t& d : dims) total_size *= d;
    if (total_size == 0) throw std::runtime_error("size of tensor cannot be zero");

    std::vector<size_t> strides(n, 1);
    size_t prod = 1;
    for (size_t i = n; i-- > 0;) {
        strides[i] = prod;
        prod *= dims[i];
    }

    return std::make_shared<TensorShape>(total_size, std::vector<size_t>(dims), std::move(strides));
}
}  // namespace tensor