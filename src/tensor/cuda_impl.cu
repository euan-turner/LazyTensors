#include "tensor/cuda_impl.hpp"
#include "tensor/tensor_ops.hpp"
#include "tensor/cu_macros.hpp"
#include "tensor/cu_ops.cuh"
#include <iostream>

namespace tensor {

template <typename UnOp>
__global__ void unary_op_kernel(float *a, size_t N, UnOp op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    a[idx] = op(a[idx]);
  }
}


template <typename UnOp>
void launch_unary_op_inplace(float *a, size_t N, UnOp op) {
  size_t BLOCK_SIZE = 256;
  size_t GRID_SIZE = CEIL_DIV(N, BLOCK_SIZE);
  unary_op_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, N, op);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// TODO: broadcasting
// template <typename BinOp>
// __global__ void elemwise_binop_kernel_ip(float* a, float* b, size_t N, BinOp op) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < N) {
//     a[idx] = op(a[idx], b[idx]);
//   }
// }

// template <typename BinOp>
// void launch_elemwise_binop_ip(float* a, float* b, size_t N, BinOp op) {
//   size_t BLOCK_SIZE = 256;
//   size_t GRID_SIZE = CEIL_DIV(N, BLOCK_SIZE);
//   elemwise_binop_kernel_ip<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, N, op);
//   CUDA_CHECK(cudaGetLastError());
//   CUDA_CHECK(cudaDeviceSynchronize());
// }

// TODO: Either remove unnecessary broadcasting of a, or make this out of place
template <typename BinOp>
__global__ void elemwise_binop_ip_broadcast(
  float* a,
  const float* b,
  const size_t* shape, // size of each dim
  const size_t* stride_a, // a stride along each dim
  const size_t* stride_b, // b stride along each dim
  int ndim,
  size_t N, // total output size
  BinOp op
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  size_t tmp = idx;
  size_t offset_a = 0;
  size_t offset_b = 0;

  // convert output linear index -> N-D co-ordinates -> offsets
  for (int d = ndim - 1; d >= 0; --d) {
    size_t coord = tmp % shape[d];
    tmp /= shape[d];
    offset_a += coord * stride_a[d];
    offset_b += coord * stride_b[d];
  }
  a[offset_a] = op(a[offset_a], b[offset_b]);
}

template <typename BinOp>
void launch_elemwise_binop_ip_broadcast(
  float* a,
  const float* b,
  TensorShape a_shape,
  TensorShape b_shape,
  size_t N, // size of A
  BinOp op
) {
  size_t ndim = a_shape.ndim();
  size_t b_ndim = b_shape.ndim();

  if (b_ndim > ndim) {
    throw std::runtime_error("CUDAImpl::launch_elemwise_binop_ip_broadcast: only RHS can broadcast for binary in-place op");
  }

  // insert leading broadcast dimensions to b
  if (b_ndim < ndim) {
    int leading = ndim - b_ndim;
    for (int i = 0; i < leading; ++i) {
      b_shape.dims.insert(b_shape.dims.begin(), 1);
      b_shape.strides.insert(b_shape.strides.begin(), 1);
    }
  }

  // validate broadcasting of b's dimensions
  // and ensure that stride is 0 for all dimensions with size 1
  for (size_t d = 0; d < ndim; ++d) {
    size_t d_idx = ndim - 1 - d;
    if (b_shape.dims[d_idx] == 1) {
      b_shape.strides[d_idx] = 0;
    } else if (a_shape.dims[d_idx] != b_shape.dims[d_idx]) {
      throw std::runtime_error("CUDAImpl::launch_elemwise_binop_ip_broadcast: only RHS can broadcast for in-place op");
    }
  }


  size_t* d_shape;
  size_t* d_stride_a;
  size_t* d_stride_b;
  CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_stride_a, ndim * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_stride_b, ndim * sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(d_shape, a_shape.dims.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_stride_a, a_shape.strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_stride_b, b_shape.strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));

  size_t BLOCK_SIZE = 256;
  size_t GRID_SIZE = CEIL_DIV(N, BLOCK_SIZE);
  elemwise_binop_ip_broadcast<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, d_shape, d_stride_a, d_stride_b, ndim, N, op);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(d_shape);
  cudaFree(d_stride_a);
  cudaFree(d_stride_b);
}


void CUDAImpl::_apply(const Op& op) {
    switch (op.type) {
        case OpType::SCAL_ADD: {
            auto p = std::get<ScalParams>(op.params);
            launch_unary_op_inplace(_data, numel(), ScalAddCUDA(p.x));
            break;
        }
        case OpType::SCAL_SUB: {
            auto p = std::get<ScalParams>(op.params);
            launch_unary_op_inplace(_data, numel(), ScalSubCUDA(p.x));
            break;
        }
        case OpType::SCAL_MUL: {
            auto p = std::get<ScalParams>(op.params);
            launch_unary_op_inplace(_data, numel(), ScalMulCUDA(p.x));
            break;
        }
        case OpType::SCAL_DIV: {
            auto p = std::get<ScalParams>(op.params);
            if (std::abs(p.x) < std::numeric_limits<float>::epsilon()) {
              throw std::runtime_error("CUDAImpl::apply: SCAL_DIV dividing by zero");
            }
            launch_unary_op_inplace(_data, numel(), ScalDivCUDA(p.x));
            break;
        }
        case OpType::EXP:
            launch_unary_op_inplace(_data, numel(), UnExpCUDA{});
            break;

        case OpType::LOG:
            launch_unary_op_inplace(_data, numel(), UnLogCUDA{});
            break;

        case OpType::CLAMP: {
            auto p = std::get<ClampParams>(op.params);
            launch_unary_op_inplace(_data, numel(), UnClampCUDA{p.lo, p.hi});
            break;
        }
        case OpType::BIN_ADD: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_ADD expected CUDAImpl");
            launch_elemwise_binop_ip_broadcast(_data, other->_data, *_shape, *other->shape(), numel(), BinAddCUDA{});
            // launch_elemwise_binop_ip(_data, other->_data, numel(), BinAddCUDA{});
            break;
        }
        case OpType::BIN_SUB: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_SUB expected CUDAImpl");
            launch_elemwise_binop_ip_broadcast(_data, other->_data, *_shape, *other->shape(), numel(), BinSubCUDA{});
            break;
        }
        case OpType::BIN_MUL: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_MUL expected CUDAImpl");
            launch_elemwise_binop_ip_broadcast(_data, other->_data, *_shape, *other->shape(), numel(), BinMulCUDA{});
            break;
        }
        case OpType::BIN_DIV: {
            auto other = dynamic_cast<const CUDAImpl*>(op.other);
            if (!other) throw std::runtime_error("CUDAImpl::apply: BIN_DIV expected CUDAImpl");
            launch_elemwise_binop_ip_broadcast(_data, other->_data, *_shape, *other->shape(), numel(), BinDivCUDA{});
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
    // each block writes its partial sum to the single float pointed to by res
    atomicAdd(res, smem[0]);
  }
}

// Vector-vector dot product: a (N) * b (N) = res (1)
void launch_dotprod(float* a, float* b, float* res, size_t N) {
  int BLOCK_DIM = 32;
  int GRID_DIM = CEIL_DIV(N, 32);
  // Ensure the accumulator is zeroed before launching the kernel
  CUDA_CHECK(cudaMemset(res, 0, sizeof(float)));
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

  throw std::runtime_error("Dimensions invalid");
  return nullptr;
}

} // namespace tensor