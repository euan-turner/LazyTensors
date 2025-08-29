#pragma once

#include <cstddef>
#include <variant>

namespace tensor {

// All in-place ops on Tensors are handled by dispatch, allowing for fusion
// All out-of-place ops (e.g. matmul, reductions) are handled directly
// Transpose is also a flush point - treat as a "view" and modify dims and strides accordingly
enum class OpType {
    // Binary operations between Tensors
    BIN_ADD,       // elementwise addition
    BIN_SUB,       // elementwise subtraction
    BIN_MUL,       // elementwise multiplication
    BIN_DIV,       // elementwise division
    // Unary operations on a Tensor
    SCAL_ADD,      // scalar addition
    SCAL_SUB,      // scalar subtraction
    SCAL_MUL,      // scalar multiplication
    SCAL_DIV,      // scalar division
    EXP,           // elementwise exponential
    LOG,           // elementwise logarithm
    CLAMP,         // elementwise clamp
};

// Parameters for ops that require extra data
struct ClampParams { float lo, hi; };
struct ScalParams { float x; };

// Variant type for op parameters
using OpParams = std::variant<std::monostate, ClampParams, ScalParams>;

struct Op {
    OpType type;
    OpParams params;
    const class TensorImpl* other = nullptr;  // optional tensor for binary ops
};

} // namespace tensor