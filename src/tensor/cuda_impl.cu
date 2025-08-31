#include "tensor/cuda_impl.hpp"
#include "tensor/tensor_ops.hpp"
#include "tensor/cu_macros.hpp"
#include "tensor/cu_ops.cuh"

namespace tensor {

template <typename ScalOp>
__global__ void elemwise_scalop_kernel(float* a, float s, size_t N, ScalOp op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    a[idx] = op(a[idx], s);
  }
}

template <typename ScalOp>
void launch_elemwise_scalop(float* a, float s, size_t N, ScalOp op) {
  size_t BLOCK_SIZE = 256;
  size_t GRID_SIZE = CEIL_DIV(N, BLOCK_SIZE);
  elemwise_scalop_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, s, N, op);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename UnOp>
__global__ void elemwise_unop_kernel(float* a, size_t N, UnOp op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    a[idx] = op(a[idx]);
  }
}

template <typename UnOp>
void launch_elemwise_unop(float* a, size_t N, UnOp op) {
  size_t BLOCK_SIZE = 256;
  size_t GRID_SIZE = CEIL_DIV(N, BLOCK_SIZE);
  elemwise_unop_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, N, op);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename BinOp>
__global__ void elemwise_binop_kernel(float* a, float* b, size_t N, BinOp op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    a[idx] = op(a[idx], b[idx]);
  }
}

template <typename BinOp>
void launch_elemwise_binop(float* a, float* b, size_t N, BinOp op) {
  size_t BLOCK_SIZE = 256;
  size_t GRID_SIZE = CEIL_DIV(N, BLOCK_SIZE);
  elemwise_binop_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, N, op);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Initial implementation - just naively dispatch every op immediately
// All operations applied in place
void CUDAImpl::apply(const Op& op) {
    switch (op.type) {
        case OpType::SCAL_ADD: {
            auto p = std::get<ScalParams>(op.params);
            launch_elemwise_scalop(_data, p.x, numel(), AddOp{});
            break;
        }
        case OpType::SCAL_SUB: {
            auto p = std::get<ScalParams>(op.params);
            launch_elemwise_scalop(_data, p.x, numel(), SubOp{});
            break;
        }
        case OpType::SCAL_MUL: {
            auto p = std::get<ScalParams>(op.params);
            launch_elemwise_scalop(_data, p.x, numel(), MulOp{});
            break;
        }
        case OpType::SCAL_DIV: {
            auto p = std::get<ScalParams>(op.params);
            if (std::abs(p.x) < std::numeric_limits<float>::epsilon()) {
              throw std::runtime_error("CUDAImpl::apply: SCAL_DIV dividing by zero");
            }
            launch_elemwise_scalop(_data, p.x, numel(), DivOp{});
            break;
        }

        case OpType::EXP:
            launch_elemwise_unop(_data, numel(), ExpOp{});
            break;

        case OpType::LOG:
            launch_elemwise_unop(_data, numel(), LogOp{});
            break;

        case OpType::CLAMP: {
            auto p = std::get<ClampParams>(op.params);
            launch_elemwise_unop(_data, numel(), ClampOp{p.lo, p.hi});
            break;
        }

        case OpType::BIN_ADD: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_ADD expected CUDAImpl");
            launch_elemwise_binop(_data, other->_data, numel(), AddOp{});
            break;
        }
        case OpType::BIN_SUB: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_SUB expected CUDAImpl");
            launch_elemwise_binop(_data, other->_data, numel(), SubOp{});
            break;
        }
        case OpType::BIN_MUL: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_MUL expected CUDAImpl");
            launch_elemwise_binop(_data, other->_data, numel(), MulOp{});
            break;
        }
        case OpType::BIN_DIV: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_DIV expected CUDAImpl");
            launch_elemwise_binop(_data, other->_data, numel(), DivOp{});
            break;
        }

        default:
            throw std::runtime_error("CUDAImpl::apply: unsupported OpType");
    }
}

}