#include "tensor/tensor_shape.hpp"
#include <stdexcept>
#include <algorithm>

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

    auto res = std::make_shared<TensorShape>(total_size, std::vector<size_t>(dims), std::move(strides));
    return res;
}

// In-place transpose by reordering dims and strides
std::shared_ptr<TensorShape> TensorShape::transpose(const std::vector<size_t>& axes) {
    if (axes.size() != dims.size()) {
        throw std::invalid_argument("axes size must match number of dimensions");
    }

    // sanity check: axes must be a permutation of [0, dims.size())
    std::vector<size_t> check = axes;
    std::sort(check.begin(), check.end());
    for (size_t i = 0; i < check.size(); ++i) {
        if (check[i] != i) {
            throw std::invalid_argument("axes must be a permutation of dimensions");
        }
    }

    std::vector<size_t> newDims(dims.size());
    std::vector<size_t> newStrides(strides.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        newDims[i]    = dims[axes[i]];
        newStrides[i] = strides[axes[i]];
    }

    return std::make_shared<TensorShape>(numel, newDims, newStrides);
}
}  // namespace tensor
