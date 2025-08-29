#pragma once

#include <functional>
#include <memory>
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_shape.hpp"
#include "tensor/tensor_ops.hpp"

namespace tensor {

class CPUImpl final : public TensorImpl {
    private:
        float* _data;

    public:
        explicit CPUImpl(const std::shared_ptr<TensorShape> shape);
        ~CPUImpl() override;

        // --- Memory access ---
        float at(const std::vector<size_t>& idx) const override;
        void set(const std::vector<size_t>& idx, float v) override;

        // --- Device info ---
        Device device() const override;

        // --- Core cloning / transfers ---
        std::unique_ptr<TensorImpl> clone() const override;
        std::unique_ptr<TensorImpl> to(Device target) const override;

        // --- In-place / fusible elementwise ops ---
        // Single entrypoint for all buffered/fusible ops
        void apply(const Op& op) override;

        // --- Out-of-place operations ---
        std::unique_ptr<TensorImpl> matmul(const TensorImpl& b) const override;

        std::unique_ptr<TensorImpl> sum(int64_t dim, bool keepdim) const override;
        std::unique_ptr<TensorImpl> mean(int64_t dim, bool keepdim) const override;

        std::unique_ptr<TensorImpl> transpose(const std::vector<size_t>& axes) const override;

        // --- Flush buffered operations ---
        void flush() override;
};

}