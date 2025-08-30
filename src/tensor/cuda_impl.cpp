#include "tensor/cuda_impl.hpp"

using namespace tensor;

CUDAImpl::CUDAImpl(const std::shared_ptr<TensorShape> shape) : TensorImpl(shape), _data(nullptr) {}
CUDAImpl::~CUDAImpl() {}

float CUDAImpl::at(const std::vector<size_t>& idx) const { return 0.0f; }
void CUDAImpl::set(const std::vector<size_t>& idx, float v) {}

Device CUDAImpl::device() const { return Device(); }

std::unique_ptr<TensorImpl> CUDAImpl::clone() const { return nullptr; }
std::unique_ptr<TensorImpl> CUDAImpl::to(Device target) const { return nullptr; }

void CUDAImpl::apply(const Op& op) {}

std::unique_ptr<TensorImpl> CUDAImpl::matmul(const TensorImpl& b) { return nullptr; }

std::unique_ptr<TensorImpl> CUDAImpl::sum(int axis, bool keepdim) { return nullptr; }
std::unique_ptr<TensorImpl> CUDAImpl::mean(int axis, bool keepdim) { return nullptr; }

std::unique_ptr<TensorImpl> CUDAImpl::transpose(const std::vector<size_t>& axes) const { return nullptr; }

void CUDAImpl::flush() {}
