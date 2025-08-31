#pragma once

namespace tensor {

// UnaryOps
#define DEFINE_UNARY_OP(name, expr) \
struct name { \
    __device__ float operator()(float x) const { return expr; } \
};

DEFINE_UNARY_OP(ExpOp, __expf(x))
DEFINE_UNARY_OP(LogOp, __log2f(x))

struct ClampOp {
  float _lo, _hi;
  ClampOp(float lo, float hi) : _lo(lo), _hi(hi) {}
  __device__ float operator()(float x) const { return x < _lo ? _lo : (x > _hi ? _hi : x); }
};

// BinaryOps
#define DEFINE_BINARY_OP(name, expr) \
struct name { \
    __device__ float operator()(float x, float y) const { return expr; } \
};

DEFINE_BINARY_OP(AddOp, x + y)
DEFINE_BINARY_OP(SubOp, x - y)
DEFINE_BINARY_OP(MulOp, x * y)
DEFINE_BINARY_OP(DivOp, x / y)



}