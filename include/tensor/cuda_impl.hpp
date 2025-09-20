#pragma once

#include <functional>
#include <memory>
#include <deque>
#include "tensor/cpu_impl.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_shape.hpp"
#include "tensor/tensor_ops.hpp"

namespace tensor {

class CUDAImpl final : public TensorImpl {
    private:
        float* _data;
        std::deque<Op> op_buffer;

        void _apply(const Op& op); // no buffering

    public:
        explicit CUDAImpl(const std::shared_ptr<TensorShape> shape);
        ~CUDAImpl() override;

        CUDAImpl(const CUDAImpl& other);                    // Copy constructor
        CUDAImpl& operator=(const CUDAImpl& other);         // Copy assignment
        CUDAImpl(CUDAImpl&& other) noexcept;               // Move constructor  
        CUDAImpl& operator=(CUDAImpl&& other) noexcept;    // Move assignment

        // --- Memory access ---
        float at(const std::vector<size_t>& idx) override;
        void set(const std::vector<size_t>& idx, float v) override;

        // --- Device info ---
        Device device() const override;

        // --- Core cloning / transfers ---
        std::shared_ptr<TensorImpl> clone() const override;
        std::shared_ptr<CPUImpl> to_cpu() const override;
        static std::shared_ptr<TensorImpl> from_cpu(const CPUImpl& cpu_tensor);

        // --- In-place / fusible elementwise ops ---
        // Single entrypoint for all buffered/fusible ops
        void apply(const Op& op) override; // buffering

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

        std::shared_ptr<TensorImpl> relu_back(TensorImpl& gradients) override;
};

}