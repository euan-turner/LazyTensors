
#include "tensor/cpu_impl.hpp"
#include "tensor/tensor_ops.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace tensor {

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
        
        // TODO: Need better control over when data is allocated and freed
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


float CPUImpl::at(const std::vector<size_t>& idx) const { return _data[flatIndex(idx)]; }
void CPUImpl::set(const std::vector<size_t>& idx, float v) { _data[flatIndex(idx)] = v; }

Device CPUImpl::device() const { return Device::CPU; }

std::unique_ptr<TensorImpl> CPUImpl::clone() const { 
  auto other = std::make_unique<CPUImpl>(_shape);
  std::memcpy(other->_data, _data, numel() * sizeof(float));
  return other;
}

// TODO
std::unique_ptr<CPUImpl> CPUImpl::to_cpu() const {
  return std::make_unique<CPUImpl>(*this);
}

std::unique_ptr<TensorImpl> CPUImpl::from_cpu(const CPUImpl& cpu_tensor) {
  return std::make_unique<CPUImpl>(cpu_tensor);
}

// Initial implementation - just naively dispatch every op immediately
void CPUImpl::apply(const Op& op) {
    switch (op.type) {
        case OpType::SCAL_ADD: {
            auto p = std::get<ScalParams>(op.params);
            for (size_t i = 0; i < numel(); i++)
                _data[i] += p.x;
            break;
        }
        case OpType::SCAL_SUB: {
            auto p = std::get<ScalParams>(op.params);
            for (size_t i = 0; i < numel(); i++)
                _data[i] -= p.x;
            break;
        }
        case OpType::SCAL_MUL: {
            auto p = std::get<ScalParams>(op.params);
            for (size_t i = 0; i < numel(); i++)
                _data[i] *= p.x;
            break;
        }
        case OpType::SCAL_DIV: {
            auto p = std::get<ScalParams>(op.params);
            for (size_t i = 0; i < numel(); i++)
                _data[i] /= p.x;
            break;
        }

        case OpType::EXP:
            for (size_t i = 0; i < numel(); i++)
                _data[i] = std::exp(_data[i]);
            break;

        case OpType::LOG:
            for (size_t i = 0; i < numel(); i++)
                _data[i] = std::log(_data[i]);
            break;

        case OpType::CLAMP: {
            auto p = std::get<ClampParams>(op.params);
            for (size_t i = 0; i < numel(); i++) {
                if (_data[i] < p.lo) _data[i] = p.lo;
                else if (_data[i] > p.hi) _data[i] = p.hi;
            }
            break;
        }

        case OpType::BIN_ADD: {
            auto other = dynamic_cast<const CPUImpl*>(op.other);
            if (!other) throw std::runtime_error("CPUImpl::apply: BIN_ADD expected CPUImpl");
            for (size_t i = 0; i < numel(); i++)
                _data[i] += other->_data[i];
            break;
        }
        case OpType::BIN_SUB: {
            auto other = dynamic_cast<const CPUImpl*>(op.other);
            if (!other) throw std::runtime_error("CPUImpl::apply: BIN_SUB expected CPUImpl");
            for (size_t i = 0; i < numel(); i++)
                _data[i] -= other->_data[i];
            break;
        }
        case OpType::BIN_MUL: {
            auto other = dynamic_cast<const CPUImpl*>(op.other);
            if (!other) throw std::runtime_error("CPUImpl::apply: BIN_MUL expected CPUImpl");
            for (size_t i = 0; i < numel(); i++)
                _data[i] *= other->_data[i];
            break;
        }
        case OpType::BIN_DIV: {
            auto other = dynamic_cast<const CPUImpl*>(op.other);
            if (!other) throw std::runtime_error("CPUImpl::apply: BIN_DIV expected CPUImpl");
            for (size_t i = 0; i < numel(); i++)
                _data[i] /= other->_data[i];
            break;
        }

        default:
            throw std::runtime_error("CPUImpl::apply: unsupported OpType");
    }
}

std::unique_ptr<TensorImpl> CPUImpl::matmul(const TensorImpl& b) {
  flush();

  // TODO: Broadcasting
  auto other = dynamic_cast<const CPUImpl*>(&b);
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
      auto result = std::make_unique<CPUImpl>(createShape(std::vector<size_t>{rows}));

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

      auto result = std::make_unique<CPUImpl>(createShape(std::vector<size_t>{rows, cols}));

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
      auto result = std::make_unique<CPUImpl>(createShape(std::vector<size_t>{cols}));

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
      auto result = std::make_unique<CPUImpl>(createShape(std::vector<size_t>{1}));

      float v = 0.0f;
      for (size_t i = 0; i < len; ++i) {
        v += at({i}) * other->at({i});
      }
      result->set({0}, v);

      return result;
  }

}

std::unique_ptr<TensorImpl> CPUImpl::sum(int axis, bool keepdim) { 
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

  auto result = std::make_unique<CPUImpl>(createShape(res_dims));

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

std::unique_ptr<TensorImpl> CPUImpl::mean(int axis, bool keepdim) { 
  flush();

  std::unique_ptr<TensorImpl> result = sum(axis, keepdim);
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

std::unique_ptr<TensorImpl> CPUImpl::transpose(const std::vector<size_t>& axes) const { 

}

void CPUImpl::flush() {}

}