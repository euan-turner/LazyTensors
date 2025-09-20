#define CATCH_CONFIG_MAIN

#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"
#include "tensor/tensor_device.hpp"

#include <vector>
#include "tensor/test_helpers.hpp"

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