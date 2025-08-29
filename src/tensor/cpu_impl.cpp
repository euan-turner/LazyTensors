
#include "tensor/cpu_impl.hpp"

using namespace tensor;

CPUImpl::CPUImpl(const std::shared_ptr<TensorShape> shape) : TensorImpl(shape), _data(nullptr) {}
CPUImpl::~CPUImpl() {}

float CPUImpl::at(const std::vector<size_t>& idx) const { return 0.0f; }
void CPUImpl::set(const std::vector<size_t>& idx, float v) {}

Device CPUImpl::device() const { return Device(); }

std::unique_ptr<TensorImpl> CPUImpl::clone() const { return nullptr; }
std::unique_ptr<TensorImpl> CPUImpl::to(Device target) const { return nullptr; }

void CPUImpl::apply(const Op& op) {}

std::unique_ptr<TensorImpl> CPUImpl::matmul(const TensorImpl& b) const { return nullptr; }

std::unique_ptr<TensorImpl> CPUImpl::sum(int64_t dim, bool keepdim) const { return nullptr; }
std::unique_ptr<TensorImpl> CPUImpl::mean(int64_t dim, bool keepdim) const { return nullptr; }

std::unique_ptr<TensorImpl> CPUImpl::transpose(const std::vector<size_t>& axes) const { return nullptr; }

void CPUImpl::flush() {}
