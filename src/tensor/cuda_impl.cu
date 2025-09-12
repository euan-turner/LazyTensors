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


void CUDAImpl::_apply(const Op& op) {
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
// Matrix multiplication kernel: computes C = A x B
// A is (M x K), B is (K x N), C is (M x N)
// M: number of rows of A and C (output)
// N: number of columns of B and C (output)
// K: inner dimension (columns of A, rows of B)
__global__ void matmat_kernel(float* A, float* B, float* C, size_t M, size_t N, size_t K) {
  // One thread per C element, so M x N threads
  int col = blockIdx.x * blockDim.x + threadIdx.x; // N (output cols)
  int row = blockIdx.y * blockDim.y + threadIdx.y; // M (output rows)

  if (row < M && col < N) {
    float v = 0.0f;
    for (size_t i = 0; i < K; ++i) {
      v += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = v;
  }
}
// Matrix-vector multiplication: mat (M x K) * vec (K) = res (M)
void launch_matvec(float* mat, float* vec, float* res, size_t M, size_t K) {
  dim3 BLOCK_DIM(32, 32);
  dim3 GRID_DIM(1, CEIL_DIV(M, 32));
  matmat_kernel<<<GRID_DIM, BLOCK_DIM>>>(mat, vec, res, M, 1, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Matrix-matrix multiplication: a (M x K) * b (K x N) = res (M x N)
void launch_matmat(float* a, float* b, float* res, size_t M, size_t K, size_t N) {
  dim3 BLOCK_DIM(32, 32);
  dim3 GRID_DIM(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  matmat_kernel<<<GRID_DIM, BLOCK_DIM>>>(a, b, res, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Vector-matrix multiplication: vec (K) * mat (K x N) = res (N)
void launch_vecmat(float* vec, float* mat, float* res, size_t K, size_t N) {
  dim3 BLOCK_DIM(32, 32);
  dim3 GRID_DIM(CEIL_DIV(N, 32), 1);
  matmat_kernel<<<GRID_DIM, BLOCK_DIM>>>(vec, mat, res, 1, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Vector dot product kernel: computes <A, B>
// A is (N), B is (N), res is (1)
__global__ void dotprod_kernel(float* A, float* B, float* res, size_t N) {
  // One thread per A/B element, so N threads
  // One shared memory element per thread in block, so N length array
  extern __shared__ float smem[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x;
  float val = (idx < N) ? (A[idx] * B[idx]) : 0.0f;
  smem[lane] = val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      smem[lane] += smem[lane + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(res, *smem);
  }
}

// Vector-vector dot product: a (N) * b (N) = res (1)
void launch_dotprod(float* a, float* b, float* res, size_t N) {
  int BLOCK_DIM = 32;
  int GRID_DIM = CEIL_DIV(N, 32);
  dotprod_kernel<<<GRID_DIM, BLOCK_DIM, BLOCK_DIM * sizeof(float)>>>(a, b, res, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

std::shared_ptr<TensorImpl> CUDAImpl::matmul(TensorImpl& b) {
  flush();

  // TODO: Broadcasting
  auto other = dynamic_cast<CUDAImpl*>(&b);
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