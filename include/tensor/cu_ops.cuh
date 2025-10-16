#pragma once

namespace tensor {

// UnaryOps
#define DEFINE_UNARY_OP(name, expr) \
struct name { \
    __device__ float operator()(float x) const { return expr; } \
};

DEFINE_UNARY_OP(ExpCUOp, __expf(x))
DEFINE_UNARY_OP(LogCUOp, __logf(x))

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

DEFINE_BINARY_OP(AddCUOp, x + y)
DEFINE_BINARY_OP(SubCUOp, x - y)
DEFINE_BINARY_OP(MulCUOp, x * y)
DEFINE_BINARY_OP(DivCUOp, x / y)


}