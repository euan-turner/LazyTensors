#pragma once

#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_shape.hpp"

namespace tensor {

/**
 * @brief The Tensor abstraction API
 * 
 * Provides both out-of-place ops, and in-place ops. In-place ops are post-fixed 
 * with _ (e.g. _add is in-place, add is out-of-place).
 * To apply custom operations between Tensors (e.g. for custom forward and backward passes),
 * implement them as a chain of in-place operations following an out-of-place operation.
 * 
 */
class Tensor {
    private:
        std::shared_ptr<TensorShape> _shape;
        std::unique_ptr<TensorImpl> _impl;
        Device _device;

        /**
         * @brief Construct a new Tensor object, from an underlying TensorImpl
         * 
         * @param dims Tensor dimensions
         * @param impl TensorImpl with backing memory already allocated
         */
        explicit Tensor(std::vector<size_t> dims, std::unique_ptr<TensorImpl> impl);


    public:
        /**
         * @brief Construct a new Tensor object
         * 
         * @param dims Tensor dimensions
         * @param device Device backend to use
         */
        explicit Tensor(std::vector<size_t> dims, Device device = Device::CPU);

        // Static factory methods

        /**
         * @brief Construct a vector
         * 
         * @param length 
         * @param device 
         * @return Tensor 
         */
        static Tensor vector(size_t length, Device device = Device::CPU) {
            return Tensor({length}, device);
        }

        /**
         * @brief Construct a matrix
         * 
         * @param rows 
         * @param cols 
         * @param device 
         * @return Tensor 
         */
        static Tensor matrix(size_t rows, size_t cols, Device device = Device::CPU) {
            return Tensor({rows, cols}, device);
        }

        /**
         * @brief Construct a scalar
         * 
         * @param value 
         * @param device 
         * @return Tensor 
         */
        static Tensor scalar(float value, Device device = Device::CPU) {
            Tensor t = Tensor::vector(1, device);
            // t.set(0, value);
            return t;
        }
        
        /**
         * @brief Get Device backing this Tensor
         * 
         * @return Device 
         */
        Device device() const { return _device; }

        /**
         * @brief Move this Tensor to a different device
         * 
         * @param device Target device
         * @return Tensor 
         */
        Tensor to(Device device);

        /**
         * @brief Construct a Tensor by copying from this instance
         * Copies all underlying memory and state
         * 
         * @param other 
         */
        Tensor(const Tensor& other);

        /**
         * @brief Copy assign from this Tensor
         * 
         * @param other 
         * @return Tensor& 
         */
        Tensor& operator=(const Tensor& other);

        /**
         * @brief Construct a new Tensor object by moving all state from this instance
         * 
         * @param other 
         */
        Tensor(Tensor&& other) noexcept;

        /**
         * @brief Move assign from this Tensor
         * 
         * @param other 
         * @return Tensor& 
         */
        Tensor& operator=(Tensor&& other) noexcept;

        ~Tensor() = default;

        /**
         * @brief Explicitly clone a Tensor
         * 
         * @return Tensor 
         */
        Tensor clone() const;

        // Element accessors

        /**
         * @brief Retrieve element at index
         * 
         * @param indices Index per Tensor axis
         * @return float 
         */
        float at(std::vector<size_t> indices) const;
        /**
         * @brief Set element at index
         * 
         * @param indices Index per Tensor axis
         * @param v Value
         */
        void set(std::vector<size_t> indices, float v);
        float operator()(std::vector<size_t> indices) const;
        // Convenience accessors for matrices
        float operator()(size_t row, size_t col) const;
        void set(size_t row, size_t col, float v);
        // Convenience accessors for vectors
        float operator()(size_t idx) const;
        void set(size_t idx, float v);

        // Property accessors

        /**
         * @brief Get Tensor dimension along an axis
         * 
         * @param axis 
         * @return size_t 
         */
        size_t dim(size_t axis) const;
        
        /**
         * @brief Get Tensor dimensions
         * 
         * @return std::vector<size_t> 
         */
        std::vector<size_t> dims() const;

        /**
         * @brief Get Tensor stride along an axis
         * 
         * @param axis 
         * @return size_t 
         */
        size_t stride(size_t axis) const;

        /**
         * @brief Get Tensor strides (per axis)
         * 
         * @return std::vector<size_t> 
         */
        std::vector<size_t> strides() const;

        /**
         * @brief Get Tensor size
         * 
         * @return size_t 
         */
        size_t numel() const;

        // Dimension utility methods
        bool isMatrix() const;
        bool isScalar() const;
        size_t rows() const;
        size_t cols() const;
        size_t length() const;

        // ---- out-of-place ops (thin wrappers over clone()+in-place) ----

        // Element-wise ops, implicitly broadcasts
        Tensor add(const Tensor& other) const      { Tensor out = clone(); out.add_(other); return out; }
        Tensor sub(const Tensor& other) const      { Tensor out = clone(); out.sub_(other); return out; }
        Tensor mul(const Tensor& other) const      { Tensor out = clone(); out.mul_(other); return out; }
        Tensor div(const Tensor& other) const      { Tensor out = clone(); out.div_(other); return out; }

        // Scalar ops
        Tensor add(float s) const              { Tensor out = clone(); out.add_(s); return out; }
        Tensor mul(float s) const              { Tensor out = clone(); out.mul_(s); return out; }
        Tensor sub(float s) const              { Tensor out = clone(); out.sub_(s); return out; }
        Tensor div(float s) const              { Tensor out = clone(); out.div_(s); return out; }
        Tensor exp() const                     { Tensor out = clone(); out.exp_(); return out; }
        Tensor log() const                     { Tensor out = clone(); out.log_(); return out; }
        Tensor clamp(float lo, float hi) const { Tensor out = clone(); out.clamp_(lo,hi); return out; }

        // Matrix multiplication, implicitly broadcasts, only out-of-place
        Tensor matmul(const Tensor& other) const;

        // Reductions, returns new Tensor
        Tensor sum(int64_t dim = -1, bool keepdim=false) const;
        Tensor mean(int64_t dim = -1, bool keepdim=false) const;

        // Broadcast
        Tensor broadcast_to(const TensorShape& target_shape) const;
        Tensor expand_as(const Tensor& other) const;

        // ---- in-place ops (backbone for perf / chaining) ----
        // return *this

        // Element-wise ops, implicitly broadcasts
        Tensor& add_(const Tensor& other);
        Tensor& sub_(const Tensor& other);
        Tensor& mul_(const Tensor& other);
        Tensor& div_(const Tensor& other);

        // Scalar ops
        Tensor& add_(float s);
        Tensor& mul_(float s);
        Tensor& sub_(float s);
        Tensor& div_(float s);

        Tensor& exp_();
        Tensor& log_();
        Tensor& clamp_(float lo, float hi);

        // Transpose - in-place by updating the view by strides
        Tensor& transpose_(std::vector<size_t> axes);
        Tensor& transpose_();
};

} // namespace tensor
