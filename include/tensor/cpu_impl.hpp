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
        // Templated in-place binary op. Defined in cpu_impl.cpp and explicitly
        // instantiated for the supported BinOp functors there.
        template <typename BinOp>
        void binary_op_inplace(const TensorImpl* b, BinOp op);


    public:
        explicit CPUImpl(const std::shared_ptr<TensorShape> shape);
        ~CPUImpl() override;

        CPUImpl(const CPUImpl& other);                    // Copy constructor
        CPUImpl& operator=(const CPUImpl& other);         // Copy assignment
        CPUImpl(CPUImpl&& other) noexcept;               // Move constructor  
        CPUImpl& operator=(CPUImpl&& other) noexcept;    // Move assignment

        // --- Memory access ---
        float at(const std::vector<size_t>& idx) override;
        void set(const std::vector<size_t>& idx, float v) override;
        float* raw_data() const { return _data; } // for moving tensor impls between devices

        // --- Device info ---
        Device device() const override;

        // --- Core cloning / transfers ---
        std::shared_ptr<TensorImpl> clone() const override;
        std::shared_ptr<CPUImpl> to_cpu() const override;
        static std::shared_ptr<TensorImpl> from_cpu(const CPUImpl& cpu_tensor);


        // --- In-place / fusible elementwise ops ---
        // Single entrypoint for all buffered/fusible ops
        void apply(const Op& op) override;

        // --- Flush buffered operations ---
        void flush() override;

        // --- Out-of-place operations ---
        std::shared_ptr<TensorImpl> matmul(TensorImpl& b) override;

        /**
         * @brief Sums a Tensor along a specified axis
         * 
         * @param axis Axis to sum along, -1 to sum whole Tensor
         * @param keepdim Retain collapsed axis dimension with length 1
         * @return std::shared_ptr<TensorImpl> 
         */
        std::shared_ptr<TensorImpl> sum(int axis, bool keepdim) override;
        std::shared_ptr<TensorImpl> mean(int axis, bool keepdim) override;

};

}