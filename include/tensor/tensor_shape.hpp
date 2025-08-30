#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensor {

struct TensorShape {
    size_t numel;
    std::vector<size_t> dims;
    std::vector<size_t> strides;

    TensorShape(size_t numel_, std::vector<size_t> dims_, std::vector<size_t> strides_)
        : numel(numel_), dims(std::move(dims_)), strides(std::move(strides_)) {}
};

// Helper to create TensorShape from dimensions
std::shared_ptr<TensorShape> createShape(const std::vector<size_t>& dims);
}  // namespace tensor