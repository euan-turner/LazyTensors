#pragma once

#include "tensor/tensor_shape.hpp"
#include "tensor/tensor_device.hpp"
#include "tensor/tensor_ops.hpp"
#include <memory>
#include <vector>

namespace tensor {

class CPUImpl;

/**
 * @brief Abstract backend implementation of a Tensor.
 * 
 * Each Tensor holds a shared_ptr<TensorImpl> which encapsulates:
 *  - The raw memory buffer
 *  - Device-specific execution (CPU / CUDA)
 *  - Primitive ops (elementwise, matmul, reductions, etc.)
 *
 * Tensor forwards high-level API calls to its Impl. The Impl can
 * assume that dimensions will always match for the operation.
 */
class TensorImpl {
    protected:
    std::shared_ptr<TensorShape> _shape;

    size_t flatIndex(const std::vector<size_t>& indices) const;

    public:
    /**
     * @brief Construct a new Tensor Impl object
     * Sub-classes should perform their memory allocation here
     *
     * @param shape
     */
    explicit TensorImpl(const std::shared_ptr<TensorShape> shape) : _shape(shape) {}

    /**
     * @brief Destroy the Tensor Impl object
     * Sub-classes should override if they need custom logic
     *
     */
    virtual ~TensorImpl() = default;

    // Allow copying - just copies shared_ptr
    TensorImpl(const TensorImpl& other) : _shape(other._shape) {}
    TensorImpl& operator=(const TensorImpl& other) {
        if (this != &other) {
            _shape = other._shape;
        }
        return *this;
    }
    
    // Move operations
    TensorImpl(TensorImpl&&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;


    // --- Memory access ---
    virtual float at(const std::vector<size_t>& idx) = 0;
    virtual void set(const std::vector<size_t>& idx, float v) = 0;

    inline size_t numel() const { return _shape->numel; }
    const std::shared_ptr<TensorShape>& shape() const { return _shape; }

    // --- Device info ---
    virtual Device device() const = 0;
    static std::shared_ptr<TensorImpl> create_impl(Device device, std::shared_ptr<TensorShape> shape);

    // --- Core cloning / transfers ---
    virtual std::shared_ptr<TensorImpl> clone() const = 0;
    virtual std::shared_ptr<CPUImpl> to_cpu() const = 0;
    // Each subclass must implement
    // static std::shared_ptr<TensorImpl> from_cpu(const CPUImpl& cpu_tensor)

    // --- In-place / fusible elementwise ops ---
    // Single entrypoint for all buffered/fusible ops
    virtual void apply(const Op& op) = 0;
    // Creates a new view of the same data
    virtual std::shared_ptr<TensorImpl> transpose(const std::vector<size_t>& axes) const = 0;

    // --- Flush buffered operations ---
    virtual void flush() = 0;

    // --- Out-of-place operations ---
    // Non-const as they will need to flush first
    virtual std::shared_ptr<TensorImpl> matmul(TensorImpl& b) = 0;

    virtual std::shared_ptr<TensorImpl> sum(int axis, bool keepdim) = 0;
    virtual std::shared_ptr<TensorImpl> mean(int axis, bool keepdim) = 0;

    virtual std::shared_ptr<TensorImpl> relu_back(TensorImpl& gradients) = 0;
};

} // namespace tensor