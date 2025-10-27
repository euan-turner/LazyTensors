#pragma once

namespace tensor {

// ScalarOps
#define DEFINE_SCAL_OP(name, expr) \
struct name { \
    float s; \
    explicit name(float scalar) : s(scalar) {} \
    __device__ float operator()(float x) const { return (expr); } \
};

DEFINE_SCAL_OP(ScalAddCUDA, x + s)
DEFINE_SCAL_OP(ScalSubCUDA, x - s)
DEFINE_SCAL_OP(ScalMulCUDA, x * s)
DEFINE_SCAL_OP(ScalDivCUDA, x / s)

// UnaryOps
#define DEFINE_UNARY_OP(name, expr) \
struct name { \
    __device__ float operator()(float x) const { return (expr); } \
};

DEFINE_UNARY_OP(UnExpCUDA, __expf(x))
DEFINE_UNARY_OP(UnLogCUDA, __logf(x))

struct UnClampCUDA {
  float _lo, _hi;
  UnClampCUDA(float lo, float hi) : _lo(lo), _hi(hi) {}
  __device__ float operator()(float x) const { return x < _lo ? _lo : (x > _hi ? _hi : x); }
};

// BinaryOps
#define DEFINE_BINARY_OP(name, expr) \
struct name { \
    __device__ float operator()(float x, float y) const { return expr; } \
};

DEFINE_BINARY_OP(BinAddCUDA, x + y)
DEFINE_BINARY_OP(BinSubCUDA, x - y)
DEFINE_BINARY_OP(BinMulCUDA, x * y)
DEFINE_BINARY_OP(BinDivCUDA, x / y)

#undef DEFINE_SCAL_OP
#undef DEFINE_UNARY_OP
#undef DEFINE_BINARY_OP

}