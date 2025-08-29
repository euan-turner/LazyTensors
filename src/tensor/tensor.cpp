#include "tensor/tensor.hpp"
#include "tensor/tensor_shape.hpp"

namespace tensor {

Tensor::Tensor(std::vector<size_t> dims, std::unique_ptr<TensorImpl> impl) {}

Tensor::Tensor(std::vector<size_t> dims, Device device = Device::CPU) : _device(device) {
    _shape = createShape(dims);
    _impl = TensorImpl::create_impl(_device, *_shape);
}

Tensor Tensor::to(Device device) {}

Tensor::Tensor(const Tensor& other) {}

Tensor& Tensor::operator=(const Tensor& other) {}

Tensor::Tensor(Tensor&& other) noexcept {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {}

Tensor Tensor::clone() const {}

float Tensor::at(std::vector<size_t> indices) const {}

void Tensor::set(std::vector<size_t> indices, float v) {}

float Tensor::operator()(std::vector<size_t> indices) const {}

float Tensor::operator()(size_t row, size_t col) const {}

void Tensor::set(size_t row, size_t col, float v) {}

float Tensor::operator()(size_t idx) const {}

void Tensor::set(size_t idx, float v) {}

size_t Tensor::dim(size_t axis) const {}

std::vector<size_t> Tensor::dims() const {}

size_t Tensor::stride(size_t axis) const {}

std::vector<size_t> Tensor::strides() const {}

size_t Tensor::numel() const {}

bool Tensor::isMatrix() const {}

bool Tensor::isScalar() const {}

size_t Tensor::rows() const {}

size_t Tensor::cols() const {}

size_t Tensor::length() const {}

Tensor Tensor::add(const Tensor& b) const {}

Tensor Tensor::sub(const Tensor& b) const {}

Tensor Tensor::mul(const Tensor& b) const {}

Tensor Tensor::div(const Tensor& b) const {}

Tensor Tensor::add(float s) const {}

Tensor Tensor::mul(float s) const {}

Tensor Tensor::exp() const {}

Tensor Tensor::log() const {}

Tensor Tensor::clamp(float lo, float hi) const {}

Tensor Tensor::matmul(const Tensor& b) const {}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {}

Tensor Tensor::broadcast_to(const TensorShape& target_shape) const {}

Tensor Tensor::expand_as(const Tensor& other) const {}

Tensor& Tensor::add_(const Tensor& b) {}

Tensor& Tensor::sub_(const Tensor& b) {}

Tensor& Tensor::mul_(const Tensor& b) {}

Tensor& Tensor::div_(const Tensor& b) {}

Tensor& Tensor::add_(float s) {}

Tensor& Tensor::mul_(float s) {}

Tensor& Tensor::exp_() {}

Tensor& Tensor::log_() {}

Tensor& Tensor::clamp_(float lo, float hi) {}

Tensor& Tensor::transpose_(std::vector<size_t> axes) {}

Tensor& Tensor::transpose_() {}

} // namespace tensor