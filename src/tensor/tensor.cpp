#include "tensor/tensor.hpp"
#include "tensor/tensor_shape.hpp"
#include "tensor/tensor_ops.hpp"

namespace tensor {

Tensor::Tensor(std::vector<size_t> dims, std::unique_ptr<TensorImpl> impl) {}

Tensor::Tensor(std::vector<size_t> dims, Device device) : _device(device) {
    _shape = createShape(dims);
    _impl = TensorImpl::create_impl(_device, _shape);
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

Tensor Tensor::matmul(const Tensor& other) const {}

Tensor Tensor::sum(size_t dim, bool keepdim) const {}

Tensor Tensor::mean(size_t dim, bool keepdim) const {}

Tensor Tensor::broadcast_to(const TensorShape& target_shape) const {}

Tensor Tensor::expand_as(const Tensor& other) const {}

Tensor& Tensor::add_(const Tensor& other) {
  Op op {OpType::BIN_ADD, std::monostate(), other._impl.get()};
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::sub_(const Tensor& other) {
  Op op {OpType::BIN_SUB, std::monostate(), other._impl.get()};
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::mul_(const Tensor& other) {
  Op op {OpType::BIN_MUL, std::monostate(), other._impl.get()};
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::div_(const Tensor& other) {
  Op op {OpType::BIN_DIV, std::monostate(), other._impl.get()};
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::add_(float s) {
  Op op { OpType::SCAL_ADD, ScalParams{s}, nullptr };
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::mul_(float s) {
  Op op { OpType::SCAL_MUL, ScalParams{s}, nullptr };
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::sub_(float s) {
  Op op { OpType::SCAL_SUB, ScalParams{s}, nullptr };
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::div_(float s) {
  Op op { OpType::SCAL_DIV, ScalParams{s}, nullptr };
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::exp_() {
  Op op { OpType::EXP, std::monostate(), nullptr };
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::log_() {
  Op op { OpType::LOG, std::monostate(), nullptr };
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::clamp_(float lo, float hi) {
  Op op { OpType::CLAMP, ClampParams{lo, hi}, nullptr };
}

Tensor& Tensor::transpose_(std::vector<size_t> axes) {}

Tensor& Tensor::transpose_() {}

} // namespace tensor