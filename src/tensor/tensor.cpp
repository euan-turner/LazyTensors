#include "tensor/tensor.hpp"
#include "tensor/cuda_impl.hpp"
#include "tensor/tensor_shape.hpp"
#include "tensor/tensor_ops.hpp"

namespace tensor {

Tensor::Tensor(std::unique_ptr<TensorImpl> impl) {
  _shape = impl->shape();
  _impl = std::move(impl);
}

Tensor::Tensor(std::vector<size_t> dims, Device device) : _device(device) {
    _shape = createShape(dims);
    _impl = TensorImpl::create_impl(_device, _shape);
}

void Tensor::to(Device target) {
  if (target == device()) return;

  auto cpu_tensor = _impl->to_cpu();
  switch (target) {
    case Device::CPU:
      _impl = std::move(cpu_tensor);
      break;
    case Device::CUDA:
      _impl = CUDAImpl::from_cpu(*cpu_tensor.get());
      break;
    default:
      throw std::runtime_error("Unsupported device");
  }
}

Tensor Tensor::clone() const {
  std::unique_ptr<TensorImpl> new_impl = _impl->clone();
  return Tensor(std::move(new_impl));
}

float Tensor::at(std::vector<size_t> indices) const {
  if (indices.size() != _shape->dims.size())
      throw std::runtime_error("Incorrect number of indices");
  for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= _shape->dims[i]) throw std::runtime_error("Index out of bounds");
  }
  return _impl->at(indices);
}

void Tensor::set(std::vector<size_t> indices, float v) {
  if (indices.size() != _shape->dims.size())
      throw std::runtime_error("Incorrect number of indices");
  for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= _shape->dims[i]) throw std::runtime_error("Index out of bounds");
  }
  _impl->set(indices, v);
}

float Tensor::operator()(std::vector<size_t> indices) const { return at(indices); }

float Tensor::operator()(size_t row, size_t col) const { 
  if (!isMatrix()) {
    throw std::runtime_error("Not a matrix");
  }
  return at({row, col}); 
}

void Tensor::set(size_t row, size_t col, float v) { 
  if (!isMatrix()) {
    throw std::runtime_error("Not a matrix");
  }
  set({row, col}, v); 
}

float Tensor::operator()(size_t idx) const { 
  if (!isVector()) {
    throw std::runtime_error("Not a vector");
  }
  return at(std::vector<size_t>{idx}); 
}

void Tensor::set(size_t idx, float v) { 
  if (!isVector()) {
    throw std::runtime_error("Not a vector");
  }
  set(std::vector<size_t>{idx}, v); 
}

size_t Tensor::dim(size_t axis) const {
  if (axis >= _shape->dims.size()) {
    throw std::runtime_error("Axis out of bounds");
  }
  return _shape->dims[axis];
}

std::vector<size_t> Tensor::dims() const {
  return _shape->dims;
}

size_t Tensor::stride(size_t axis) const {
  if (axis >= _shape->strides.size()) {
    throw std::runtime_error("Axis out of bounds");
  }
  return _shape->strides[axis];
}

std::vector<size_t> Tensor::strides() const {
  return _shape->strides;
}

size_t Tensor::numel() const {
  return _shape->numel;
}

bool Tensor::isMatrix() const {
  return _shape->dims.size() == 2;
}

bool Tensor::isVector() const {
  return _shape->dims.size() == 1;
}

// scalars cannot be multi-dimensional
bool Tensor::isScalar() const {
  return isVector() && numel() == 1;
}

size_t Tensor::rows() const {
  if (!isMatrix()) {
    throw std::runtime_error("Not a matrix");
  }
  return dim(0);
}

size_t Tensor::cols() const {
  if (!isMatrix()) {
    throw std::runtime_error("Not a matrix");
  }
  return dim(1);
}

size_t Tensor::length() const {
  if (!isVector()) {
    throw std::runtime_error("Not a vector");
  }
  return dim(0);
}

Tensor Tensor::matmul(const Tensor& other) const {
  // TODO: Validation checks on shapes
  std::unique_ptr<TensorImpl> res_impl = _impl->matmul(*other._impl);
  return Tensor(std::move(res_impl));
}

Tensor Tensor::sum(int axis, bool keepdim) const {
  std::unique_ptr<TensorImpl> res_impl = _impl->sum(axis, keepdim);
  return Tensor(std::move(res_impl));
}

Tensor Tensor::mean(int axis, bool keepdim) const {
  std::unique_ptr<TensorImpl> res_impl = _impl->mean(axis, keepdim);
  return Tensor(std::move(res_impl));
}

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
  _impl->apply(op);
  return *this;
}

Tensor& Tensor::transpose_(std::vector<size_t> axes) {}

Tensor& Tensor::transpose_() {}

} // namespace tensor