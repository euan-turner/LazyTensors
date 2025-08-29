
#include "tensor/cpu_impl.hpp"
#include "tensor/tensor_ops.hpp"
#include <cmath>

using namespace tensor;

CPUImpl::CPUImpl(const std::shared_ptr<TensorShape> shape) : TensorImpl(shape), _data(nullptr) {}
CPUImpl::~CPUImpl() {}

float CPUImpl::at(const std::vector<size_t>& idx) const { return 0.0f; }
void CPUImpl::set(const std::vector<size_t>& idx, float v) {}

Device CPUImpl::device() const { return Device(); }

std::unique_ptr<TensorImpl> CPUImpl::clone() const { return nullptr; }
std::unique_ptr<TensorImpl> CPUImpl::to(Device target) const { return nullptr; }

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

std::unique_ptr<TensorImpl> CPUImpl::matmul(const TensorImpl& b) const { return nullptr; }

std::unique_ptr<TensorImpl> CPUImpl::sum(int64_t dim, bool keepdim) const { return nullptr; }
std::unique_ptr<TensorImpl> CPUImpl::mean(int64_t dim, bool keepdim) const { return nullptr; }

std::unique_ptr<TensorImpl> CPUImpl::transpose(const std::vector<size_t>& axes) const { return nullptr; }

void CPUImpl::flush() {}
