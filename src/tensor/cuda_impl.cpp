#include "tensor/cuda_impl.hpp"
#include "tensor/cu_macros.hpp"

namespace tensor {

CUDAImpl::CUDAImpl(const std::shared_ptr<TensorShape> shape) : TensorImpl(shape) {
  CUDA_CHECK(cudaMalloc(&_data, numel() * sizeof(float)));
}

CUDAImpl::~CUDAImpl() {
  CUDA_CHECK(cudaFree(_data));
}

CUDAImpl::CUDAImpl(const CUDAImpl& other)
  : TensorImpl(other)
  , _data(nullptr)
{
  if (other._data && _shape) {
    size_t size = numel() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&_data, size));
    CUDA_CHECK(cudaMemcpy(_data, other._data, size, cudaMemcpyDeviceToDevice));
  }
}

CUDAImpl& CUDAImpl::operator=(const CUDAImpl& other) {
  if (this != &other) {
    TensorImpl::operator=(other);

    CUDA_CHECK(cudaFree(_data));
    _data = nullptr;

    if (other._data && _shape) {
      size_t size = numel() * sizeof(float);
      CUDA_CHECK(cudaMalloc(&_data, size));
      CUDA_CHECK(cudaMemcpy(_data, other._data, size, cudaMemcpyDeviceToDevice));
    }
  }
  return *this;
}

CUDAImpl::CUDAImpl(CUDAImpl&& other) noexcept
  : TensorImpl(std::move(other))
  , _data(other._data)
{
  other._data = nullptr;
}

CUDAImpl& CUDAImpl::operator=(CUDAImpl&& other) noexcept {
  if (this != &other) {
    TensorImpl::operator=(std::move(other));

    CUDA_CHECK(cudaFree(_data));
    _data = other._data;
    other._data = nullptr;
  }
  return *this;
}

float CUDAImpl::at(const std::vector<size_t> &idx) const { 
  float res;
  float* addr = _data + flatIndex(idx);
  CUDA_CHECK(cudaMemcpy(&res, addr, sizeof(float), cudaMemcpyDeviceToHost));
  return res;
}

void CUDAImpl::set(const std::vector<size_t> &idx, float v) {
  float* addr = _data + flatIndex(idx);
  CUDA_CHECK(cudaMemcpy(addr, &v, sizeof(float), cudaMemcpyHostToDevice));
}

Device CUDAImpl::device() const { return Device::CUDA; }

std::unique_ptr<TensorImpl> CUDAImpl::clone() const {
  auto other = std::make_unique<CUDAImpl>(_shape);
  CUDA_CHECK(cudaMemcpy(other->_data, _data, numel() * sizeof(float), cudaMemcpyDeviceToDevice));
  return other;
}

std::unique_ptr<CPUImpl> CUDAImpl::to_cpu() const { 
  auto cpu_tensor = std::make_unique<CPUImpl>(_shape);
  CUDA_CHECK(cudaMemcpy(cpu_tensor->raw_data(), _data, numel() * sizeof(float), cudaMemcpyDeviceToHost));
  return cpu_tensor;
}

std::unique_ptr<TensorImpl> CUDAImpl::from_cpu(const CPUImpl& cpu_tensor) {
  auto gpu_tensor = std::make_unique<CUDAImpl>(cpu_tensor.shape());
  CUDA_CHECK(cudaMemcpy(gpu_tensor->_data, cpu_tensor.raw_data(), cpu_tensor.numel() * sizeof(float), cudaMemcpyHostToDevice));
  return gpu_tensor;
}



std::unique_ptr<TensorImpl> CUDAImpl::sum(int axis, bool keepdim) {}

std::unique_ptr<TensorImpl> CUDAImpl::mean(int axis, bool keepdim) {}

std::unique_ptr<TensorImpl> CUDAImpl::transpose(const std::vector<size_t>& axes) const {}

void CUDAImpl::flush() {}

}