#define CATCH_CONFIG_MAIN

#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"
#include "tensor/tensor_device.hpp"

#include <vector>
#include "tensor/test_helpers.hpp"
#include <functional>

using namespace tensor;

using Catch::Approx;

TEST_CASE("Tensor in-place and out-of-place elementwise ops (CPU)", "[Tensor][CPUImpl]") {
    Tensor a = Tensor::vector(3, Device::CPU);
    a.set(0, 1.0f);
    a.set(1, 2.0f);
    a.set(2, 3.0f);

    Tensor b = Tensor::vector(3, Device::CPU);
    b.set(0, 4.0f);
    b.set(1, 5.0f);
    b.set(2, 6.0f);

    SECTION("Out-of-place add") {
        Tensor c = a.add(b);
    REQUIRE(c(0) == Catch::Approx(5.0f));
    REQUIRE(c(1) == Catch::Approx(7.0f));
    REQUIRE(c(2) == Catch::Approx(9.0f));
        // Original unchanged
    REQUIRE(a(0) == Catch::Approx(1.0f));
    }
    SECTION("In-place add_") {
        a.add_(b);
    REQUIRE(a(0) == Catch::Approx(5.0f));
    REQUIRE(a(1) == Catch::Approx(7.0f));
    REQUIRE(a(2) == Catch::Approx(9.0f));
    }
    SECTION("Out-of-place sub") {
        Tensor c = b.sub(a);
    REQUIRE(c(0) == Catch::Approx(3.0f));
    REQUIRE(c(1) == Catch::Approx(3.0f));
    REQUIRE(c(2) == Catch::Approx(3.0f));
    }
    SECTION("In-place sub_") {
        b.sub_(a);
    REQUIRE(b(0) == Catch::Approx(3.0f));
    REQUIRE(b(1) == Catch::Approx(3.0f));
    REQUIRE(b(2) == Catch::Approx(3.0f));
    }
    SECTION("Out-of-place mul") {
        Tensor c = a.mul(b);
    REQUIRE(c(0) == Catch::Approx(4.0f));
    REQUIRE(c(1) == Catch::Approx(10.0f));
    REQUIRE(c(2) == Catch::Approx(18.0f));
    }
    SECTION("In-place mul_") {
        a.mul_(b);
    REQUIRE(a(0) == Catch::Approx(4.0f));
    REQUIRE(a(1) == Catch::Approx(10.0f));
    REQUIRE(a(2) == Catch::Approx(18.0f));
    }
    SECTION("Out-of-place div") {
        Tensor c = b.div(a);
    REQUIRE(c(0) == Catch::Approx(4.0f));
    REQUIRE(c(1) == Catch::Approx(2.5f));
    REQUIRE(c(2) == Catch::Approx(2.0f));
    }
    SECTION("In-place div_") {
        b.div_(a);
    REQUIRE(b(0) == Catch::Approx(4.0f));
    REQUIRE(b(1) == Catch::Approx(2.5f));
    REQUIRE(b(2) == Catch::Approx(2.0f));
    }
}

TEST_CASE("Tensor scalar ops (CPU)", "[Tensor][CPUImpl]") {
    Tensor a = Tensor::vector(3, Device::CPU);
    a.set(0, 1.0f);
    a.set(1, 2.0f);
    a.set(2, 3.0f);

    SECTION("Out-of-place add scalar") {
        Tensor b = a.add(2.0f);
    REQUIRE(b(0) == Catch::Approx(3.0f));
    REQUIRE(b(1) == Catch::Approx(4.0f));
    REQUIRE(b(2) == Catch::Approx(5.0f));
    }
    SECTION("In-place add_ scalar") {
        a.add_(2.0f);
    REQUIRE(a(0) == Catch::Approx(3.0f));
    REQUIRE(a(1) == Catch::Approx(4.0f));
    REQUIRE(a(2) == Catch::Approx(5.0f));
    }
    SECTION("Out-of-place mul scalar") {
        Tensor b = a.mul(2.0f);
    REQUIRE(b(0) == Catch::Approx(2.0f));
    REQUIRE(b(1) == Catch::Approx(4.0f));
    REQUIRE(b(2) == Catch::Approx(6.0f));
    }
    SECTION("In-place mul_ scalar") {
        a.mul_(2.0f);
    REQUIRE(a(0) == Catch::Approx(2.0f));
    REQUIRE(a(1) == Catch::Approx(4.0f));
    REQUIRE(a(2) == Catch::Approx(6.0f));
    }
    SECTION("Out-of-place sub scalar") {
        Tensor b = a.sub(1.0f);
    REQUIRE(b(0) == Catch::Approx(0.0f));
    REQUIRE(b(1) == Catch::Approx(1.0f));
    REQUIRE(b(2) == Catch::Approx(2.0f));
    }
    SECTION("In-place sub_ scalar") {
        a.sub_(1.0f);
    REQUIRE(a(0) == Catch::Approx(0.0f));
    REQUIRE(a(1) == Catch::Approx(1.0f));
    REQUIRE(a(2) == Catch::Approx(2.0f));
    }
    SECTION("Out-of-place div scalar") {
        Tensor b = a.div(2.0f);
    REQUIRE(b(0) == Catch::Approx(0.5f));
    REQUIRE(b(1) == Catch::Approx(1.0f));
    REQUIRE(b(2) == Catch::Approx(1.5f));
    }
    SECTION("In-place div_ scalar") {
        a.div_(2.0f);
    REQUIRE(a(0) == Catch::Approx(0.5f));
    REQUIRE(a(1) == Catch::Approx(1.0f));
    REQUIRE(a(2) == Catch::Approx(1.5f));
    }
}

TEST_CASE("Tensor exp, log, clamp ops (CPU)", "[Tensor][CPUImpl]") {
    Tensor a = Tensor::vector(3, Device::CPU);
    a.set(0, 1.0f);
    a.set(1, 2.0f);
    a.set(2, 3.0f);

    SECTION("Out-of-place exp") {
        Tensor b = a.exp();
    REQUIRE(b(0) == Catch::Approx(std::exp(1.0f)));
    REQUIRE(b(1) == Catch::Approx(std::exp(2.0f)));
    REQUIRE(b(2) == Catch::Approx(std::exp(3.0f)));
    }
    SECTION("In-place exp_") {
        a.exp_();
    REQUIRE(a(0) == Catch::Approx(std::exp(1.0f)));
    REQUIRE(a(1) == Catch::Approx(std::exp(2.0f)));
    REQUIRE(a(2) == Catch::Approx(std::exp(3.0f)));
    }
    SECTION("Out-of-place log") {
        Tensor b = a.log();
    REQUIRE(b(0) == Catch::Approx(0.0f));
    REQUIRE(b(1) == Catch::Approx(std::log(2.0f)));
    REQUIRE(b(2) == Catch::Approx(std::log(3.0f)));
    }
    SECTION("In-place log_") {
        a.log_();
    REQUIRE(a(0) == Catch::Approx(0.0f));
    REQUIRE(a(1) == Catch::Approx(std::log(2.0f)));
    REQUIRE(a(2) == Catch::Approx(std::log(3.0f)));
    }
    SECTION("Out-of-place clamp") {
        Tensor b = a.clamp(1.5f, 2.5f);
    REQUIRE(b(0) == Catch::Approx(1.5f));
    REQUIRE(b(1) == Catch::Approx(2.0f));
    REQUIRE(b(2) == Catch::Approx(2.5f));
    }
    SECTION("In-place clamp_") {
        a.clamp_(1.5f, 2.5f);
    REQUIRE(a(0) == Catch::Approx(1.5f));
    REQUIRE(a(1) == Catch::Approx(2.0f));
    REQUIRE(a(2) == Catch::Approx(2.5f));
    }
}

TEST_CASE("Tensor matrix multiplication (CPU)", "[Tensor][CPUImpl]") {
    Tensor a = Tensor::matrix(2, 3, Device::CPU);
    a.set(0, 0, 1.0f); a.set(0, 1, 2.0f); a.set(0, 2, 3.0f);
    a.set(1, 0, 4.0f); a.set(1, 1, 5.0f); a.set(1, 2, 6.0f);

    Tensor b = Tensor::matrix(3, 2, Device::CPU);
    b.set(0, 0, 7.0f);  b.set(0, 1, 8.0f);
    b.set(1, 0, 9.0f);  b.set(1, 1, 10.0f);
    b.set(2, 0, 11.0f); b.set(2, 1, 12.0f);

    Tensor c = a.matmul(b);
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0,0) == Catch::Approx(58.0f));
    REQUIRE(c(0,1) == Catch::Approx(64.0f));
    REQUIRE(c(1,0) == Catch::Approx(139.0f));
    REQUIRE(c(1,1) == Catch::Approx(154.0f));
}

TEST_CASE("Tensor reductions (CPU)", "[Tensor][CPUImpl]") {
    Tensor a = Tensor::matrix(2, 3, Device::CPU);
    a.set(0, 0, 1.0f); a.set(0, 1, 2.0f); a.set(0, 2, 3.0f);
    a.set(1, 0, 4.0f); a.set(1, 1, 5.0f); a.set(1, 2, 6.0f);

    SECTION("Sum over all elements") {
        Tensor s = a.sum();
    REQUIRE(s.isScalar());
    REQUIRE(s(0) == Catch::Approx(21.0f));
    }
    SECTION("Sum over axis 0 (rows)") {
        Tensor s = a.sum(0, false);
    REQUIRE(s.isMatrix() == false);
    REQUIRE(s.length() == 3);
    REQUIRE(s(0) == Catch::Approx(5.0f));
    REQUIRE(s(1) == Catch::Approx(7.0f));
    REQUIRE(s(2) == Catch::Approx(9.0f));
    }
    SECTION("Sum over axis 1 (cols)") {
        Tensor s = a.sum(1, false);
    REQUIRE(s.length() == 2);
    REQUIRE(s(0) == Catch::Approx(6.0f));
    REQUIRE(s(1) == Catch::Approx(15.0f));
    }
    SECTION("Mean over all elements") {
        Tensor m = a.mean();
    REQUIRE(m.isScalar());
    REQUIRE(m(0) == Catch::Approx(3.5f));
    }
    SECTION("Mean over axis 0 (rows)") {
        Tensor m = a.mean(0, false);
    REQUIRE(m.length() == 3);
    REQUIRE(m(0) == Catch::Approx(2.5f));
    REQUIRE(m(1) == Catch::Approx(3.5f));
    REQUIRE(m(2) == Catch::Approx(4.5f));
    }
    SECTION("Mean over axis 1 (cols)") {
        Tensor m = a.mean(1, false);
    REQUIRE(m.length() == 2);
    REQUIRE(m(0) == Catch::Approx(2.0f));
    REQUIRE(m(1) == Catch::Approx(5.0f));
    }
}

TEST_CASE("Tensor transpose changes shapes", "[Tensor]") {
    std::vector<size_t> dims = {16, 8};
    Tensor a = make_test_cpu_tensor(dims);
    Tensor b = make_test_cpu_tensor(dims);

    REQUIRE_THROWS(a.matmul(b));

    SECTION("Explicit axes") {
        b.transpose_({1, 0});
        REQUIRE(b.dim(0) == 8);
        REQUIRE(b.dim(1) == 16);
        Tensor res = a.matmul(b);
        REQUIRE(res.dim(0) == 16);
        REQUIRE(res.dim(1) == 16);
    }

    SECTION("Implicit axes") {
        a.transpose_();
        REQUIRE(a.dim(0) == 8);
        REQUIRE(a.dim(1) == 16);
        Tensor res = a.matmul(b);
        REQUIRE(res.dim(0) == 8);
        REQUIRE(res.dim(1) == 8);
    }
}

TEST_CASE("Matrix multiplication after transpose returns correct values", "[Tensor][Transpose][Matmul]") {
    // 2x3 matrix
    Tensor a = Tensor::matrix(2, 3, Device::CPU);
    a.set(0, 0, 1.0f); a.set(0, 1, 2.0f); a.set(0, 2, 3.0f);
    a.set(1, 0, 4.0f); a.set(1, 1, 5.0f); a.set(1, 2, 6.0f);

    // 2x3 matrix
    Tensor b = Tensor::matrix(2, 3, Device::CPU);
    b.set(0, 0, 7.0f);  b.set(0, 1, 8.0f);  b.set(0, 2, 9.0f);
    b.set(1, 0, 10.0f); b.set(1, 1, 11.0f); b.set(1, 2, 12.0f);

    // Transpose b to 3x2
    b.transpose_();

    // a (2x3) @ b^T (3x2) = c (2x2)
    Tensor c = a.matmul(b);

    // Expected result:
    // c[0,0] = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
    // c[0,1] = 1*10 + 2*11 + 3*12 = 10 + 22 + 36 = 68
    // c[1,0] = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
    // c[1,1] = 4*10 + 5*11 + 6*12 = 40 + 55 + 72 = 167
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0,0) == Approx(50.0f));
    REQUIRE(c(0,1) == Approx(68.0f));
    REQUIRE(c(1,0) == Approx(122.0f));
    REQUIRE(c(1,1) == Approx(167.0f));
}

TEST_CASE("In-place binary ops with broadcasting (CPU)", "[Tensor][CPUImpl][Broadcast]") {

    // Case 1: same number of dims, b has some unit dimensions that should broadcast
    Tensor a1 = Tensor({2,3}, Device::CPU);
    // a1 values: row-major fill
    a1.set(0,0, 1.0f); a1.set(0,1, 2.0f); a1.set(0,2, 3.0f);
    a1.set(1,0, 4.0f); a1.set(1,1, 5.0f); a1.set(1,2, 6.0f);

    // b1 has same ndim but second dim is 1 -> should broadcast across columns
    Tensor b1 = Tensor({2,1}, Device::CPU);
    b1.set(0,0, 10.0f);
    b1.set(1,0, 20.0f);

    SECTION("add_ with b having unit inner dim") {
        Tensor a = a1.clone();
        a.add_(b1);
        REQUIRE(a(0,0) == Approx(11.0f));
        REQUIRE(a(0,1) == Approx(12.0f));
        REQUIRE(a(0,2) == Approx(13.0f));
        REQUIRE(a(1,0) == Approx(24.0f));
        REQUIRE(a(1,1) == Approx(25.0f));
        REQUIRE(a(1,2) == Approx(26.0f));
    }

    // Case 2: b has fewer dimensions and should broadcast into leading dims
    Tensor a2 = Tensor({2,3,2}, Device::CPU);
    // Explicit values for a2 (flat index + 1):
    a2.set({0,0,0}, 1.0f);
    a2.set({0,0,1}, 2.0f);
    a2.set({0,1,0}, 3.0f);
    a2.set({0,1,1}, 4.0f);
    a2.set({0,2,0}, 5.0f);
    a2.set({0,2,1}, 6.0f);
    a2.set({1,0,0}, 7.0f);
    a2.set({1,0,1}, 8.0f);
    a2.set({1,1,0}, 9.0f);
    a2.set({1,1,1}, 10.0f);
    a2.set({1,2,0}, 11.0f);
    a2.set({1,2,1}, 12.0f);

    // b2 is 3x2 (missing leading dim) -> should be broadcast to 2x3x2
    Tensor b2 = Tensor({3,2}, Device::CPU);
    // Explicit values for b2: (flat_index+1)*2
    b2.set({0,0}, 2.0f);
    b2.set({0,1}, 4.0f);
    b2.set({1,0}, 6.0f);
    b2.set({1,1}, 8.0f);
    b2.set({2,0}, 10.0f);
    b2.set({2,1}, 12.0f);

    SECTION("sub_ with b having fewer leading dims (leading broadcast)") {
        Tensor a = a2.clone();
        a.sub_(b2);
        // check every element: a2 - broadcasted b2
        // a2 explicit values: see above; b2 mapped per middle,last indices
        // expected per position computed manually
        REQUIRE(a({0,0,0}) == Approx(1.0f - 2.0f));   // 1 - b[0,0]=2
        REQUIRE(a({0,0,1}) == Approx(2.0f - 4.0f));   // 2 - b[0,1]=4
        REQUIRE(a({0,1,0}) == Approx(3.0f - 6.0f));   // 3 - b[1,0]=6
        REQUIRE(a({0,1,1}) == Approx(4.0f - 8.0f));   // 4 - b[1,1]=8
        REQUIRE(a({0,2,0}) == Approx(5.0f - 10.0f));  // 5 - b[2,0]=10
        REQUIRE(a({0,2,1}) == Approx(6.0f - 12.0f));  // 6 - b[2,1]=12

        REQUIRE(a({1,0,0}) == Approx(7.0f - 2.0f));   // 7 - b[0,0]=2
        REQUIRE(a({1,0,1}) == Approx(8.0f - 4.0f));   // 8 - b[0,1]=4
        REQUIRE(a({1,1,0}) == Approx(9.0f - 6.0f));   // 9 - b[1,0]=6
        REQUIRE(a({1,1,1}) == Approx(10.0f - 8.0f));  // 10 - b[1,1]=8
        REQUIRE(a({1,2,0}) == Approx(11.0f - 10.0f)); // 11 - b[2,0]=10
        REQUIRE(a({1,2,1}) == Approx(12.0f - 12.0f)); // 12 - b[2,1]=12
    }

    // Case 3: combined - b has fewer dims and also unit dimensions
    // a3 is 2x3x4 (explicit values)
    Tensor a3 = Tensor({2,3,4}, Device::CPU);
    // Fill a3 with explicit values 1..24 in row-major order
    a3.set({0,0,0}, 1.0f);  a3.set({0,0,1}, 2.0f);  a3.set({0,0,2}, 3.0f);  a3.set({0,0,3}, 4.0f);
    a3.set({0,1,0}, 5.0f);  a3.set({0,1,1}, 6.0f);  a3.set({0,1,2}, 7.0f);  a3.set({0,1,3}, 8.0f);
    a3.set({0,2,0}, 9.0f);  a3.set({0,2,1}, 10.0f); a3.set({0,2,2}, 11.0f); a3.set({0,2,3}, 12.0f);
    a3.set({1,0,0}, 13.0f); a3.set({1,0,1}, 14.0f); a3.set({1,0,2}, 15.0f); a3.set({1,0,3}, 16.0f);
    a3.set({1,1,0}, 17.0f); a3.set({1,1,1}, 18.0f); a3.set({1,1,2}, 19.0f); a3.set({1,1,3}, 20.0f);
    a3.set({1,2,0}, 21.0f); a3.set({1,2,1}, 22.0f); a3.set({1,2,2}, 23.0f); a3.set({1,2,3}, 24.0f);

    // b3 is 1x3x1 -> will broadcast the leading dim and last dim
    Tensor b3 = Tensor({1,3,1}, Device::CPU);
    // set b3 values per middle dim
    for (size_t i = 0; i < 3; ++i) b3.set({0,i,0}, static_cast<float>(i + 1));

    SECTION("mul_ with combined leading and unit-dimension broadcasts") {
        Tensor a = a3.clone();
        a.mul_(b3);
        // Build expected tensor explicitly (no loops)
        Tensor expected = Tensor({2,3,4}, Device::CPU);
        // i=0, j=0 (multiplier 1)
        expected.set({0,0,0}, 1.0f * 1.0f);
        expected.set({0,0,1}, 2.0f * 1.0f);
        expected.set({0,0,2}, 3.0f * 1.0f);
        expected.set({0,0,3}, 4.0f * 1.0f);
        // i=0, j=1 (multiplier 2)
        expected.set({0,1,0}, 5.0f * 2.0f);
        expected.set({0,1,1}, 6.0f * 2.0f);
        expected.set({0,1,2}, 7.0f * 2.0f);
        expected.set({0,1,3}, 8.0f * 2.0f);
        // i=0, j=2 (multiplier 3)
        expected.set({0,2,0}, 9.0f * 3.0f);
        expected.set({0,2,1}, 10.0f * 3.0f);
        expected.set({0,2,2}, 11.0f * 3.0f);
        expected.set({0,2,3}, 12.0f * 3.0f);
        // i=1, j=0 (multiplier 1)
        expected.set({1,0,0}, 13.0f * 1.0f);
        expected.set({1,0,1}, 14.0f * 1.0f);
        expected.set({1,0,2}, 15.0f * 1.0f);
        expected.set({1,0,3}, 16.0f * 1.0f);
        // i=1, j=1 (multiplier 2)
        expected.set({1,1,0}, 17.0f * 2.0f);
        expected.set({1,1,1}, 18.0f * 2.0f);
        expected.set({1,1,2}, 19.0f * 2.0f);
        expected.set({1,1,3}, 20.0f * 2.0f);
        // i=1, j=2 (multiplier 3)
        expected.set({1,2,0}, 21.0f * 3.0f);
        expected.set({1,2,1}, 22.0f * 3.0f);
        expected.set({1,2,2}, 23.0f * 3.0f);
        expected.set({1,2,3}, 24.0f * 3.0f);

        // Explicit REQUIREs for each element (no loops)
        REQUIRE(a({0,0,0}) == Approx(expected({0,0,0}))); REQUIRE(a({0,0,1}) == Approx(expected({0,0,1}))); REQUIRE(a({0,0,2}) == Approx(expected({0,0,2}))); REQUIRE(a({0,0,3}) == Approx(expected({0,0,3})));
        REQUIRE(a({0,1,0}) == Approx(expected({0,1,0}))); REQUIRE(a({0,1,1}) == Approx(expected({0,1,1}))); REQUIRE(a({0,1,2}) == Approx(expected({0,1,2}))); REQUIRE(a({0,1,3}) == Approx(expected({0,1,3})));
        REQUIRE(a({0,2,0}) == Approx(expected({0,2,0}))); REQUIRE(a({0,2,1}) == Approx(expected({0,2,1}))); REQUIRE(a({0,2,2}) == Approx(expected({0,2,2}))); REQUIRE(a({0,2,3}) == Approx(expected({0,2,3})));
        REQUIRE(a({1,0,0}) == Approx(expected({1,0,0}))); REQUIRE(a({1,0,1}) == Approx(expected({1,0,1}))); REQUIRE(a({1,0,2}) == Approx(expected({1,0,2}))); REQUIRE(a({1,0,3}) == Approx(expected({1,0,3})));
        REQUIRE(a({1,1,0}) == Approx(expected({1,1,0}))); REQUIRE(a({1,1,1}) == Approx(expected({1,1,1}))); REQUIRE(a({1,1,2}) == Approx(expected({1,1,2}))); REQUIRE(a({1,1,3}) == Approx(expected({1,1,3})));
        REQUIRE(a({1,2,0}) == Approx(expected({1,2,0}))); REQUIRE(a({1,2,1}) == Approx(expected({1,2,1}))); REQUIRE(a({1,2,2}) == Approx(expected({1,2,2}))); REQUIRE(a({1,2,3}) == Approx(expected({1,2,3})));
    }

    // Case 4: another variation using div_ for a 1D target with b broadcast
    Tensor a4 = Tensor::vector(4, Device::CPU);
    a4.set(0, 8.0f); a4.set(1, 16.0f); a4.set(2, 32.0f); a4.set(3, 64.0f);
    // b4 is scalar shape (1,) represented as vector of length 1
    Tensor b4 = Tensor::vector(1, Device::CPU);
    b4.set(0, 2.0f);

    SECTION("div_ with scalar-like RHS broadcasting to vector") {
        Tensor a = a4.clone();
        a.div_(b4);
        REQUIRE(a(0) == Approx(4.0f));
        REQUIRE(a(1) == Approx(8.0f));
        REQUIRE(a(2) == Approx(16.0f));
        REQUIRE(a(3) == Approx(32.0f));
    }
}