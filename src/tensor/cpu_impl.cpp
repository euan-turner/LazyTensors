
#include "tensor/cpu_impl.hpp"
#include "tensor/tensor_ops.hpp"
#include "tensor/cpu_ops.hpp"
#include "tensor/tensor_shape.hpp"
#include <cstdlib>
#include <cstring>

namespace tensor {


struct TensorIterator {
    const TensorShape shape;           // actual tensor in memory: numel, dims, strides
    const std::vector<size_t>& broadcast_dims; // broadcasted shape for iteration
    std::vector<size_t> idx;           // current N-dimensional index
    size_t numel;                      // total elements in broadcasted shape
    size_t linear_index;               // how many elements have been visited
    size_t ndim;

    TensorIterator(const TensorShape& shape_,
                   const std::vector<size_t>& broadcast_dims_)
        : shape(shape_), broadcast_dims(broadcast_dims_), linear_index(0) 
    {
        ndim = broadcast_dims.size();
        numel = 1;
        for (auto d : broadcast_dims) numel *= d;
        idx.assign(ndim, 0);
    }

    // Advance iterator and compute memory offset
    // Returns false if iteration is complete
    bool next(size_t& offset) {
        if (linear_index >= numel) return false;

        // Compute offset using actual tensor shape and strides
        offset = 0;
        for (size_t d = 0; d < ndim; ++d) {
            size_t i = (shape.dims[d] == 1) ? 0 : idx[d]; // broadcast dims -> repeated
            offset += i * shape.strides[d];
        }

        // Increment multi-dimensional index (odometer style)
        ++linear_index;
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < broadcast_dims[d]) break;
            idx[d] = 0;
        }

        return true;
    }
};


CPUImpl::CPUImpl(const std::shared_ptr<TensorShape> shape) : TensorImpl(shape) {
  _data = (float*)malloc(numel() * sizeof(float));
}
CPUImpl::~CPUImpl() {
  free(_data);
}

CPUImpl::CPUImpl(const CPUImpl& other) 
    : TensorImpl(other)
    , _data(nullptr)
{
    if (other._data && _shape) {
      size_t size = numel() * sizeof(float);
      _data = (float*)malloc(size);
      memcpy(_data, other._data, size);
    }
}

CPUImpl& CPUImpl::operator=(const CPUImpl& other) {
    if (this != &other) {
        TensorImpl::operator=(other);
        
        // TODO: Need clearer documentation of when allocations and frees occur
        free(_data);
        _data = nullptr;
        
        if (other._data && _shape) {
            size_t size = numel() * sizeof(float);
            _data = (float*)malloc(size);
            memcpy(_data, other._data, size);
        }
    }
    return *this;
}

CPUImpl::CPUImpl(CPUImpl&& other) noexcept 
    : TensorImpl(std::move(other))
    , _data(other._data)
{
    other._data = nullptr;
}

CPUImpl& CPUImpl::operator=(CPUImpl&& other) noexcept {
    if (this != &other) {
        TensorImpl::operator=(std::move(other));
        
        free(_data);
        _data = other._data;
        other._data = nullptr;
    }
    return *this;
}


float CPUImpl::at(const std::vector<size_t>& idx) { flush(); return _data[flatIndex(idx)]; }
void CPUImpl::set(const std::vector<size_t>& idx, float v) { flush(); _data[flatIndex(idx)] = v; }

Device CPUImpl::device() const { return Device::CPU; }

std::shared_ptr<TensorImpl> CPUImpl::clone() const { 
  auto other = std::make_shared<CPUImpl>(_shape);
  std::memcpy(other->_data, _data, numel() * sizeof(float));
  return other;
}

std::shared_ptr<CPUImpl> CPUImpl::to_cpu() const {
  return std::make_shared<CPUImpl>(*this);
}

std::shared_ptr<TensorImpl> CPUImpl::from_cpu(const CPUImpl& cpu_tensor) {
  return std::make_shared<CPUImpl>(cpu_tensor);
}


template <typename UnOp>
void CPUImpl::unary_op_inplace(UnOp op) {
  size_t n = numel();
  // TODO: parallelise
  for (size_t i = 0; i < n; ++i) {
    _data[i] = op(_data[i]);
  }
}

/**
 * @brief Perform an in-place binary operation on two CPU tensors.
 * The operation is defined by the template parameter `op`.
 * The tensor `a` is modified in place by applying the operation with tensor `b`.
 * Broadcasting of `b` dimensions is supported according to the following rules:
 *  - Dimensions are compatible when
 *    - they are equal, or
 *    - one of them is 1 (repeat to match `a`)
 * 
 * @tparam op 
 * @param a 
 * @param b 
 */
template <typename BinOp>
void CPUImpl::binary_op_inplace(const TensorImpl* b, BinOp op) {
  auto b_cpu = dynamic_cast<const CPUImpl*>(b);
  if (!b_cpu) {
    throw std::runtime_error("CPUImpl::binary_op_inplace: expected CPUImpl");
  }

  TensorShape b_shape = *b->shape();

  size_t a_ndim = _shape->ndim();
  size_t b_ndim = b_shape.ndim();
  std::vector<size_t>& a_dims = _shape->dims;
  std::vector<size_t>& b_dims = b_shape.dims;

  if (a_ndim < b_ndim) {
    throw std::runtime_error("CPUImpl::binary_op_inplace: only RHS can broadcast in binary in-place op");
  }
  
  if (b_ndim < a_ndim) {
    // Extend a view of b to broadcast in leading dimensions
    // but don't persist this view within b
    int leading = a_ndim - b_ndim;
    for (int i = 0; i < leading; ++i) {
      b_shape.dims.insert(b_shape.dims.begin(), 1);
      b_shape.strides.insert(b_shape.strides.begin(), 1);
    }
  }

  // 1. Determine broadcasting of b's dimensions
  std::vector<size_t> broadcast_dims(a_ndim, 0);
  for (size_t d = 0; d < a_ndim; ++d) {
    // d is the offset from the back of dims
    size_t d_idx = a_ndim - 1 - d;
    if (a_dims[d_idx] == b_dims[d_idx] || b_dims[d_idx] == 1) {
      broadcast_dims[d_idx] = a_dims[d_idx];
    } else {
      throw std::runtime_error("CPUImpl::binary_op_inplace: only RHS can broadcast in binary in-place op");
    }
  }

  // 2. Construct iterator for each Tensor
  TensorIterator a_it(*_shape, _shape->dims);
  TensorIterator b_it(b_shape, broadcast_dims);

  // 3. Iterate both in tandem
  size_t a_off = 0;
  size_t b_off = 0;
  while (a_it.next(a_off) && b_it.next(b_off)) {
    _data[a_off] = op(_data[a_off], b_cpu->_data[b_off]);
  }
  
}

template void CPUImpl::unary_op_inplace<ScalAddCPU>(ScalAddCPU);
template void CPUImpl::unary_op_inplace<ScalSubCPU>(ScalSubCPU);
template void CPUImpl::unary_op_inplace<ScalMulCPU>(ScalMulCPU);
template void CPUImpl::unary_op_inplace<ScalDivCPU>(ScalDivCPU);
template void CPUImpl::unary_op_inplace<UnExpCPU>(UnExpCPU);
template void CPUImpl::unary_op_inplace<UnLogCPU>(UnLogCPU);
template void CPUImpl::unary_op_inplace<UnClampCPU>(UnClampCPU);
template void CPUImpl::binary_op_inplace<BinAddCPU>(const TensorImpl* b, BinAddCPU);
template void CPUImpl::binary_op_inplace<BinSubCPU>(const TensorImpl* b, BinSubCPU);
template void CPUImpl::binary_op_inplace<BinMulCPU>(const TensorImpl* b, BinMulCPU);
template void CPUImpl::binary_op_inplace<BinDivCPU>(const TensorImpl* b, BinDivCPU);

// Initial implementation - just naively dispatch every op immediately
void CPUImpl::apply(const Op& op) {
    switch (op.type) {
        case OpType::SCAL_ADD: {
            auto p = std::get<ScalParams>(op.params);
            unary_op_inplace(ScalAddCPU(p.x));
            break;
        }
        case OpType::SCAL_SUB: {
            auto p = std::get<ScalParams>(op.params);
            unary_op_inplace(ScalSubCPU(p.x));
            break;
        }
        case OpType::SCAL_MUL: {
            auto p = std::get<ScalParams>(op.params);
            unary_op_inplace(ScalMulCPU(p.x));
            break;
        }
        case OpType::SCAL_DIV: {
            auto p = std::get<ScalParams>(op.params);
            unary_op_inplace(ScalDivCPU(p.x));
            break;
        }

        case OpType::EXP:
            unary_op_inplace(UnExpCPU{});
            break;

        case OpType::LOG:
            unary_op_inplace(UnLogCPU{});
            break;

        case OpType::CLAMP: {
            auto p = std::get<ClampParams>(op.params);
            unary_op_inplace(UnClampCPU(p.lo, p.hi));
            break;
        }
        case OpType::BIN_ADD: {
          binary_op_inplace(op.other, BinAddCPU{});
          break;
        }
        case OpType::BIN_SUB: {
          binary_op_inplace(op.other, BinSubCPU{});
          break;
        }
        case OpType::BIN_MUL: {
          binary_op_inplace(op.other, BinMulCPU{});
          break;
        }
        case OpType::BIN_DIV: {
          binary_op_inplace(op.other, BinDivCPU{});
          break;
        }

        default:
            throw std::runtime_error("CPUImpl::apply: unsupported OpType");
    }
}

std::shared_ptr<TensorImpl> CPUImpl::matmul(TensorImpl& b) {
  flush();

  // TODO: Broadcasting
  auto other = dynamic_cast<CPUImpl*>(&b);
  if (!other) {
    throw std::runtime_error("CPUImpl::matmul: expected CPUImpl");
  }

  const auto& a_dims = _shape->dims;
  const auto& b_dims = other->_shape->dims;

  if (a_dims.size() == 2 && b_dims.size() == 1) {
      // Matrix × Vector = Vector
      if (a_dims[1] != b_dims[0]) {
          throw std::runtime_error("Matrix columns must match vector length");
      }

      size_t rows = a_dims[0];
      auto result = std::make_shared<CPUImpl>(createShape(std::vector<size_t>{rows}));

      for (size_t i = 0; i < rows; ++i) {
          float sum = 0.0f;
          for (size_t k = 0; k < a_dims[1]; ++k) {
            sum += at({i, k}) * other->at({k});
          }
          result->set({i}, sum);
      }
      return result;
  }

  if (a_dims.size() == 2 && b_dims.size() == 2) {
      // Matrix × Matrix = Matrix
      if (a_dims[1] != b_dims[0]) {
          throw std::runtime_error("Matrix inner dimensions must match");
      }

      size_t rows = a_dims[0];
      size_t cols = b_dims[1];
      size_t inner = a_dims[1];

      auto result = std::make_shared<CPUImpl>(createShape(std::vector<size_t>{rows, cols}));

      for (size_t i = 0; i < rows; ++i) {
          for (size_t j = 0; j < cols; ++j) {
              float sum = 0.0f;
              for (size_t k = 0; k < inner; ++k) {
                  sum += at({i, k}) * other->at({k, j});
              }
              result->set({i, j}, sum);
          }
      }
      return result;
  }

  if (a_dims.size() == 1 && b_dims.size() == 2) {
      // Vector × Matrix = Vector (row vectors)
      if (a_dims[0] != b_dims[0]) {
          throw std::runtime_error("Vector length must match matrix rows");
      }

      size_t cols = b_dims[1];
      auto result = std::make_shared<CPUImpl>(createShape(std::vector<size_t>{cols}));

      for (size_t j = 0; j < cols; ++j) {
          float sum = 0.0f;
          for (size_t k = 0; k < a_dims[0]; ++k) {
              sum += at({k}) * other->at({k, j});
          }
          result->set({j}, sum);
      }
      return result;
  }

  if (a_dims.size() == 1 && b_dims.size() == 1) {
      // Vector × Vector = Inner Product (Scalar)
      if (a_dims[0] != b_dims[0]) {
        throw std::runtime_error("Vector lengths must match");
      }
      size_t len = a_dims[0];

      // Scalar result
      auto result = std::make_shared<CPUImpl>(createShape(std::vector<size_t>{1}));

      float v = 0.0f;
      for (size_t i = 0; i < len; ++i) {
        v += at({i}) * other->at({i});
      }
      result->set({0}, v);

      return result;
  }

  throw std::runtime_error("Dimensions invalid");
  return nullptr;
}

std::shared_ptr<TensorImpl> CPUImpl::sum(int axis, bool keepdim) { 
  flush();

  std::vector<size_t> res_dims = _shape->dims;
  if (keepdim) {
    if (axis == -1) {
      std::fill(res_dims.begin(), res_dims.end(), 1);
    } else {
      res_dims[axis] = 1;
    }
  } else {
    if (axis == -1) {
      res_dims = {1};
    } else {
      res_dims.erase(res_dims.begin() + axis);
    }
  }

  auto result = std::make_shared<CPUImpl>(createShape(res_dims));

  if (axis == -1) {
    float s = 0.0f;
    for (size_t i = 0; i < numel(); ++i) {
      s += _data[i];
    }
    *result->_data = s;
  } else {
    size_t ndim = _shape->dims.size();
    size_t reduce_dim = _shape->dims[axis];
    size_t stride = _shape->strides[axis];
    
    // product of dim sizes before axis
    size_t outer = numel() / (reduce_dim * stride);

    for (size_t o = 0; o < outer; ++o) {
      for (size_t i = 0; i < stride; ++i) {
        // along reduction axis
        float s = 0.0f;
        size_t base_idx = o * reduce_dim * stride + i;

        for (size_t r = 0; r < reduce_dim; ++r) {
          s += _data[base_idx + r * stride];
        }
        result->_data[o * stride + i] = s;
      }

    }
  }
  return result;
}

std::shared_ptr<TensorImpl> CPUImpl::mean(int axis, bool keepdim) { 
  flush();

  std::shared_ptr<TensorImpl> result = sum(axis, keepdim);
  CPUImpl* cpu = dynamic_cast<CPUImpl*>(result.get());
  float n;
  if (axis == -1) {
    n = numel();
  } else {
    n = _shape->dims[axis];
  }
  for (size_t i = 0; i < result->numel(); ++i) {
    cpu->_data[i] /= n;
  }
  return result;
}

void CPUImpl::flush() {}

}