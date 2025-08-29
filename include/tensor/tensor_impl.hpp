#pragma once

#include "tensor/tensor_shape.hpp"
#include "tensor/tensor_device.hpp"
#include "tensor/tensor_ops.hpp"
#include <memory>
#include <vector>

namespace tensor {

/**
 * @brief Abstract backend implementation of a Tensor.
 * 
 * Each Tensor holds a unique_ptr<TensorImpl> which encapsulates:
 *  - The raw memory buffer
 *  - Device-specific execution (CPU / CUDA)
 *  - Primitive ops (elementwise, matmul, reductions, etc.)
 *
 * Tensor forwards high-level API calls to its Impl.
 */
class TensorImpl {
    protected:
    std::shared_ptr<TensorShape> _shape;

    size_t flatIndex(const std::vector<size_t>& indices) const;
    void copy_to(TensorImpl& dst_backend);

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


    // --- Memory access ---
    virtual float at(const std::vector<size_t>& idx) const = 0;
    virtual void set(const std::vector<size_t>& idx, float v) = 0;

    size_t numel() const { return _shape->numel; }
    const std::shared_ptr<TensorShape>& shape() const { return _shape; }

    // --- Device info ---
    virtual Device device() const = 0;

    // --- Core cloning / transfers ---
    virtual std::unique_ptr<TensorImpl> clone() const = 0;
    virtual std::unique_ptr<TensorImpl> to(Device target) const = 0;

    // --- In-place / fusible elementwise ops ---
    // Single entrypoint for all buffered/fusible ops
    virtual void apply(const Op& op) = 0;

    // --- Out-of-place operations ---
    virtual std::unique_ptr<TensorImpl> matmul(const TensorImpl& b) const = 0;

    virtual std::unique_ptr<TensorImpl> sum(int64_t dim, bool keepdim) const = 0;
    virtual std::unique_ptr<TensorImpl> mean(int64_t dim, bool keepdim) const = 0;

    virtual std::unique_ptr<TensorImpl> transpose(const std::vector<size_t>& axes) const = 0;

    // --- Flush buffered operations ---
    virtual void flush() = 0;
};

} // namespace tensor