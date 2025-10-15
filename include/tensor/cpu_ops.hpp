#pragma once

namespace tensor {

// BinaryOps
#define DEFINE_BINARY_OP(name, expr) \
struct name { \
    float operator()(float x, float y) const { return expr; } \
};

DEFINE_BINARY_OP(AddCPUOp, x + y)
DEFINE_BINARY_OP(SubCPUOp, x - y)
DEFINE_BINARY_OP(MulCPUOp, x * y)
DEFINE_BINARY_OP(DivCPUOp, x / y)
}