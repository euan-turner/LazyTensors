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

        CPUImpl(const CPUImpl& other);                    // Copy constructor
        CPUImpl& operator=(const CPUImpl& other);         // Copy assignment
        CPUImpl(CPUImpl&& other) noexcept;               // Move constructor  
        CPUImpl& operator=(CPUImpl&& other) noexcept;    // Move assignment

        // --- Memory access ---
        float at(const std::vector<size_t>& idx) const override;
        void set(const std::vector<size_t>& idx, float v) override;

        // --- Device info ---
        Device device() const override;

        // --- Core cloning / transfers ---
        std::unique_ptr<TensorImpl> clone() const override;
        std::unique_ptr<CPUImpl> to_cpu() const override;
        static std::unique_ptr<TensorImpl> from_cpu(const CPUImpl& cpu_tensor);

        // --- In-place / fusible elementwise ops ---
        // Single entrypoint for all buffered/fusible ops
        void apply(const Op& op) override;

        std::unique_ptr<TensorImpl> transpose(const std::vector<size_t>& axes) const override;

        // --- Flush buffered operations ---
        void flush() override;

        // --- Out-of-place operations ---
        std::unique_ptr<TensorImpl> matmul(const TensorImpl& b) override;

        /**
         * @brief Sums a Tensor along a specified axis
         * 
         * @param axis Axis to sum along, -1 to sum whole Tensor
         * @param keepdim Retain collapsed axis dimension with length 1
         * @return std::unique_ptr<TensorImpl> 
         */
        std::unique_ptr<TensorImpl> sum(int axis, bool keepdim) override;
        std::unique_ptr<TensorImpl> mean(int axis, bool keepdim) override;
};

}