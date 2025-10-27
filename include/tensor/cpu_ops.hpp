#pragma once
#include <cmath>
#include "tensor/tensor_ops.hpp"

namespace tensor {

// ScalarOps
#define DEFINE_SCAL_OP(name, expr) \
struct name { \
    float s; \
    explicit name(float scalar): s(scalar) {} \
    float operator()(float x) const { return (expr); } \
};

DEFINE_SCAL_OP(ScalAddCPU, x + s)
DEFINE_SCAL_OP(ScalSubCPU, x - s)
DEFINE_SCAL_OP(ScalMulCPU, x * s)
DEFINE_SCAL_OP(ScalDivCPU, x / s)

// UnaryOps
#define DEFINE_UNARY_OP(name, expr) \
struct name { \
    float operator()(float x) const { return (expr); } \
};

DEFINE_UNARY_OP(UnExpCPU, std::exp(x))
DEFINE_UNARY_OP(UnLogCPU, std::log(x))

struct UnClampCPU {
    float lo;
    float hi;

    explicit UnClampCPU(float low, float high): lo(low), hi(high) { }

    float operator()(float x) const {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }
};


// BinaryOps
#define DEFINE_BINARY_OP(name, expr) \
struct name { \
    float operator()(float x, float y) const { return (expr); } \
};

DEFINE_BINARY_OP(BinAddCPU, x + y)
DEFINE_BINARY_OP(BinSubCPU, x - y)
DEFINE_BINARY_OP(BinMulCPU, x * y)
DEFINE_BINARY_OP(BinDivCPU, x / y)

#undef DEFINE_SCAL_OP
#undef DEFINE_UNARY_OP
#undef DEFINE_BINARY_OP

} // namespace tensor