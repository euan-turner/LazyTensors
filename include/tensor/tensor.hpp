#pragma once

#include <limits>
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_shape.hpp"

namespace tensor {

/**
 * @brief The Tensor API
 * 
 * Provides both out-of-place ops, and in-place ops. In-place ops are post-fixed 
 * with _ (e.g. add_ is in-place, add is out-of-place).
 * 
 * To apply custom operations between Tensors implement them as a chain of in-place 
 * operations following an out-of-place operation.
 *
 * The abstraction is backed by a device-specific TensorImpl, which handles the actual
 * memory allocation and definition of operations for particular hardware. The abstraction
 * is responsible for the validation of input shapes, types etc., so TensorImpls only need
 * hardware-specific validation.
 * 
 */
class Tensor {
    private:
        /**
         * @brief Contains numel, dims and strides
         * Shared with the TensorImpl
         * 
         */
        std::shared_ptr<TensorShape> _shape;
        std::shared_ptr<TensorImpl> _impl;

        /**
         * @brief Construct a new Tensor object, from an underlying TensorImpl
         * 
         * @param impl TensorImpl with backing memory already allocated
         */
        explicit Tensor(std::shared_ptr<TensorImpl> impl);


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
         * @param length Number of elements
         * @param device Device backend to use
         * @return Tensor 
         */
        static Tensor vector(size_t length, Device device = Device::CPU) {
            return Tensor({length}, device);
        }

        /**
         * @brief Construct a matrix
         * 
         * @param rows Number of rows
         * @param cols Number of columns
         * @param device Device backend to use
         * @return Tensor 
         */
        static Tensor matrix(size_t rows, size_t cols, Device device = Device::CPU) {
            return Tensor({rows, cols}, device);
        }

        /**
         * @brief Construct a scalar
         * 
         * @param value Scalar value
         * @param device Device backend to use
         * @return Tensor 
         */
        static Tensor scalar(float value, Device device = Device::CPU) {
            Tensor t = Tensor::vector(1, device);
            t.set(0, value);
            return t;
        }
        
        /**
         * @brief Get Device backing this Tensor
         * 
         * @return Device 
         */
        Device device() const { return _impl->device(); }

        /**
         * @brief Move this Tensor to a different device
         * 
         * @param target Target device
         */
        void to(Device target);

        /**
         * @brief Explicitly clone a Tensor
         * Performs a deep clone, i.e. clones the underlying
         * TensorImpl as well.
         // TODO: Make sure this happens
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
         * @brief Retrieve element at index
         * 
         * @param indices Index per Tensor axis
         * @return float 
         */
        float operator()(std::vector<size_t> indices) const;


        /**
         * @brief Convenience element accessor for matrices
         * 
         * @param row Row index
         * @param col Column index
         * @return float 
         */
        float operator()(size_t row, size_t col) const;
        
        /**
         * @brief Convenience element accessor for vectors
         * 
         * @param idx Index
         * @return float 
         */
        float operator()(size_t idx) const;

        /**
         * @brief Set element at index
         * 
         * @param indices Index per Tensor axis
         * @param v Value
         */
        void set(std::vector<size_t> indices, float v);

        /**
         * @brief Convenience element setter for matrices
         * 
         * @param row Row index
         * @param col Col index
         * @param v Value
         */
        void set(size_t row, size_t col, float v);

        /**
         * @brief Convenience element setter for vectors
         * 
         * @param idx Index
         * @param v Value
         */
        void set(size_t idx, float v);

        // Property accessors

        /**
         * @brief Get Tensor dimension along an axis
         * 
         * @param axis Axis
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
         * @brief Get Tensor strides
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

        /**
         * @brief Is the tensor 2D?
         * 
         * @return true 
         * @return false 
         */
        bool isMatrix() const;

        /**
         * @brief Is the tensor 1D?
         * 
         * @return true 
         * @return false 
         */
        bool isVector() const;

        /**
         * @brief Is the tensor 1D with size 1?
         * 
         * @return true 
         * @return false 
         */
        bool isScalar() const;

        /**
         * @brief Convenience dim accessor for matrices
         * @throws If isMatrix() == false
         * 
         * @return size_t 
         */
        size_t rows() const;

        /**
         * @brief Convenience dim accessor for matrices
         * @throws If isMatrix() == false
         * 
         * @return size_t 
         */
        size_t cols() const;

        /**
         * @brief Convenience dim accessor for vectors
         * @throws If isVector() == false
         * 
         * @return size_t 
         */
        size_t length() const;

        // ---- out-of-place ops (thin wrappers over clone()+in-place) ----

        // Element-wise ops, implicitly broadcasts
        Tensor add(const Tensor& other) const      { _impl->flush(); Tensor out = clone(); out.add_(other); return out; }
        Tensor sub(const Tensor& other) const      { _impl->flush(); Tensor out = clone(); out.sub_(other); return out; }
        Tensor mul(const Tensor& other) const      { _impl->flush(); Tensor out = clone(); out.mul_(other); return out; }
        Tensor div(const Tensor& other) const      { _impl->flush(); Tensor out = clone(); out.div_(other); return out; }

        // Scalar ops
        Tensor add(float s) const              { _impl->flush(); Tensor out = clone(); out.add_(s); return out; }
        Tensor mul(float s) const              { _impl->flush(); Tensor out = clone(); out.mul_(s); return out; }
        Tensor sub(float s) const              { _impl->flush(); Tensor out = clone(); out.sub_(s); return out; }
        Tensor div(float s) const              { _impl->flush(); Tensor out = clone(); out.div_(s); return out; }
        Tensor exp() const                     { _impl->flush(); Tensor out = clone(); out.exp_(); return out; }
        Tensor log() const                     { _impl->flush(); Tensor out = clone(); out.log_(); return out; }
        Tensor clamp(float lo, float hi) const { _impl->flush(); Tensor out = clone(); out.clamp_(lo,hi); return out; }

        Tensor transpose(std::vector<size_t> axes) const { Tensor out = clone(); out.transpose_(axes); return out; }
        Tensor transpose() const { Tensor out = clone(); out.transpose_(); return out; }

        /**
         * @brief Insert a unitary dimension at axis
         * 
         * @param axis 
         * @return Tensor 
         */
        Tensor unsqueeze(size_t axis) const { Tensor out = clone(); out.unsqueeze_(axis); return out; }

        /**
         * @brief Remove a unitary dimension at axis
         * @throws If axis is not unitary
         * 
         * @param axis 
         * @return Tensor 
         */
        Tensor squeeze(size_t axis) const { Tensor out = clone(); out.squeeze_(axis); return out; }


        /**
         * @brief Matrix multiplication
         * @todo Support broadcasting
         * @throw If tensor dimensions are not compatible for matrix multiplication
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor matmul(const Tensor& other) const;

        // Reductions, returns new Tensor
        /**
         * @brief Sum a Tensor
         * 
         * @param axis Axis to reduce over, defaults as -1 to sum entire tensor
         * @param keepdim Whether to retain axis (as size 1) in the output shape
         * @return Tensor 
         */
        Tensor sum(int axis = -1, bool keepdim = false) const;

        /**
         * @brief Mean of a Tensor
         * 
         * @param axis Axis to take mean over, defaults as -1 to take mean over entire tensor
         * @param keepdim Whether to retain axis (as size 1) in the output shape
         * @return Tensor 
         */
        Tensor mean(int axis = -1, bool keepdim = false) const;

        /**
         * @brief Max of a Tensor
         * 
         * @param axis Axis to take max over, defaults as -1 to take max over entire tensor
         * @param keepdim Whether to retain axis (as size 1) in the output shape
         * @return Tensor 
         */
        Tensor max(int axis = -1, bool keepdim = false) const;

        // Broadcast
        // TODO: Are these necessary
        Tensor broadcast_to(const TensorShape& target_shape) const;
        Tensor expand_as(const Tensor& other) const;

        // ---- in-place ops (backbone for benchmarking / chaining) ----

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

        /**
         * @brief Transpose in-place by updating strides
         * 
         * @param axes Axes ordering after transpose
         * @return Tensor& 
         */
        Tensor& transpose_(std::vector<size_t> axes);
        Tensor& transpose_();

        // Insert a unitary dimension at axis
        Tensor& unsqueeze_(size_t axis);
        Tensor& squeeze_(size_t axis);
};

} // namespace tensor
