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

// TODO: optimise
__global__ void matvec_kernel(float* mat, float* vec, float* res, size_t M, size_t N) {

}

void launch_matvec(float* mat, float* vec, float* res, size_t M, size_t N) {
  // mat is M x N, vec is N, res is M
}

// TODO: optimise
__global__ void matmat_kernel(float* a, float* b, float* res, size_t M, size_t N, size_t K) {
  // a is M x N, b is N x K, res is M x K
  // one thread per res element, so M x K threads
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // this thread computes res[row][col]
  // so dot products a[row][:] * b[:][col]

  if (row < M && col < K) {
    float v = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      v += a[row * N + i] * b[i * K + col];
    }
    res[row * K + col] = v;
  }

}

void launch_matmat(float* a, float* b, float* res, size_t M, size_t N, size_t K) {
  // a is M x N, b is N x K, res is M x K
  dim3 BLOCK_DIM(32, 32);
  dim3 GRID_DIM(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
  matmat_kernel<<<GRID_DIM, BLOCK_DIM>>>(a, b, res, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

}

// TODO: optimise
__global__ void vecmat_kernel(float* vec, float* mat, float* res, size_t M, size_t N) {

}

void launch_vecmat(float* vec, float* mat, float* res, size_t M, size_t N) {
  // vec is (1 x) M, mat is M x N, res is (1 x) N
}

// TODO: optimise
__global__ void dotprod_kernel(float* a, float* b, float* res, size_t N) {

}

void launch_dotprod(float* a, float* b, float* res, size_t N) {
  // a, b are N, res is 1
}

std::unique_ptr<TensorImpl> CUDAImpl::matmul(const TensorImpl& b) {
  // TODO: Broadcasting
  auto other = dynamic_cast<const CUDAImpl*>(&b);
  if (!other) {
    throw std::runtime_error("CUDAImpl::matmul: expected CUDAImpl");
  }

  const auto& a_dims = _shape->dims;
  const auto& b_dims = other->_shape->dims;

  if (a_dims.size() == 2 && b_dims.size() == 1) {
    // Matrix x Vector = Vector
    if (a_dims[1] != b_dims[0]) {
      throw std::runtime_error("CUDAImpl::matmul: Matrix columns must match vector length");
    }
    size_t rows = a_dims[0];
    size_t cols = a_dims[1];
    auto result = std::make_unique<CUDAImpl>(createShape(std::vector<size_t>{rows}));

    launch_matvec(_data, other->_data, result->_data, rows, cols);
    return result;
  }

  if (a_dims.size() == 2 && b_dims.size() == 2) {
    // Matrix x Matrix = Matrix
    if (a_dims[1] != b_dims[0]) {
      throw std::runtime_error("CUDAImpl::matmul: Matrix inner dimensions must match");
    }
    size_t rows = a_dims[0];
    size_t cols = b_dims[1];
    size_t inner = a_dims[1];
    auto result = std::make_unique<CUDAImpl>(createShape(std::vector<size_t>{rows, cols}));

    launch_matmat(_data, other->_data, result->_data, rows, inner, cols);
    return result;
  }

  if (a_dims.size() == 1 && b_dims.size() == 2) {
    // Vector x Matrix = Vector (row vectors)
    if (a_dims[0] != b_dims[0]) {
      throw std::runtime_error("CUDAImpl::matmul: Vector length must match matrix rows");
    }

    size_t rows = b_dims[0];
    size_t cols = b_dims[1];
    auto result = std::make_unique<CUDAImpl>(createShape(std::vector<size_t>{cols}));

    launch_vecmat(_data, other->_data, result->_data, rows, cols);
    return result;
  }

  if (a_dims.size() == 1 && b_dims.size() == 1) {
    // Vector x Vector = Inner Product Scalar
    if (a_dims[0] != b_dims[0]) {
      throw std::runtime_error("CUDAImpl::matmul: Vector lengths must match");
    }

    size_t len = a_dims[0];
    auto result = std::make_unique<CUDAImpl>(createShape(std::vector<size_t>{1}));

    launch_dotprod(_data, other->_data, result->_data, len);
    return result;
  }
}

}