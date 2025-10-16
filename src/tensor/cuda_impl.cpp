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
  // Ensure any buffered ops on the source are applied before copying
  const_cast<CUDAImpl&>(other).flush();
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

    // Ensure any buffered ops on the source are applied before copying
    const_cast<CUDAImpl&>(other).flush();

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

float CUDAImpl::at(const std::vector<size_t> &idx) {
  flush();

  float res;
  float* addr = _data + flatIndex(idx);
  CUDA_CHECK(cudaMemcpy(&res, addr, sizeof(float), cudaMemcpyDeviceToHost));
  return res;
}

void CUDAImpl::set(const std::vector<size_t> &idx, float v) {
  flush();

  float* addr = _data + flatIndex(idx);
  CUDA_CHECK(cudaMemcpy(addr, &v, sizeof(float), cudaMemcpyHostToDevice));
}

Device CUDAImpl::device() const { return Device::CUDA; }

std::shared_ptr<TensorImpl> CUDAImpl::clone() const {
  // Ensure any pending ops are applied before cloning
  const_cast<CUDAImpl*>(this)->flush();
  auto other = std::make_shared<CUDAImpl>(_shape);
  CUDA_CHECK(cudaMemcpy(other->_data, _data, numel() * sizeof(float), cudaMemcpyDeviceToDevice));
  return other;
}

std::shared_ptr<CPUImpl> CUDAImpl::to_cpu() const { 
  // Ensure pending ops are applied before transferring to host
  const_cast<CUDAImpl*>(this)->flush();
  auto cpu_tensor = std::make_shared<CPUImpl>(_shape);
  CUDA_CHECK(cudaMemcpy(cpu_tensor->raw_data(), _data, numel() * sizeof(float), cudaMemcpyDeviceToHost));
  return cpu_tensor;
}

std::shared_ptr<TensorImpl> CUDAImpl::from_cpu(const CPUImpl& cpu_tensor) {
  auto gpu_tensor = std::make_shared<CUDAImpl>(cpu_tensor.shape());
  CUDA_CHECK(cudaMemcpy(gpu_tensor->_data, cpu_tensor.raw_data(), cpu_tensor.numel() * sizeof(float), cudaMemcpyHostToDevice));
  return gpu_tensor;
}



std::shared_ptr<TensorImpl> CUDAImpl::sum(int axis, bool keepdim) {
  flush();
  throw std::runtime_error("Not implemented yet");
}

std::shared_ptr<TensorImpl> CUDAImpl::mean(int axis, bool keepdim) {
  flush();
  throw std::runtime_error("Not implemented yet");
}

void CUDAImpl::apply(const Op& op) {
  op_buffer.push_back(op);
  // switch (op.type) {
  //   case OpType::SCAL_ADD:
  //   case OpType::SCAL_SUB:
  //   case OpType::SCAL_MUL:
  //   case OpType::SCAL_DIV:
  //   case OpType::EXP:
  //   case OpType::LOG:
  //   case OpType::CLAMP:
  //   case OpType::BIN_ADD:
  //   case OpType::BIN_SUB:
  //   case OpType::BIN_MUL:
  //   case OpType::BIN_DIV:
  //     _apply(op);
  //     break;
  //   default:
  //     throw std::runtime_error("CUDAImpl::apply: unsupported OpType");
  // }
}

// TODO: Fuse ops
void CUDAImpl::flush() {
  for (auto& op : op_buffer) {
    _apply(op);
  }
  op_buffer.clear();
}

}