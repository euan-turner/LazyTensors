#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"
#include "loss/cce.hpp"
#include "loss/bce.hpp"
#include "loss/mse.hpp"

using namespace tensor;


TEST_CASE("CCE loss computes correct value for simple case", "[loss][cce]") {
    Tensor y_true = Tensor::vector(3);
    y_true.set(0, 0.0f); y_true.set(1, 1.0f); y_true.set(2, 0.0f);
    Tensor y_pred = Tensor::vector(3);
    y_pred.set(0, 0.1f); y_pred.set(1, 0.8f); y_pred.set(2, 0.1f);
    loss::CCE cce;
    float loss = cce.forward(y_pred, y_true);
    REQUIRE(loss == Catch::Approx(-std::log(0.8f)));
}


TEST_CASE("BCE loss computes correct value for simple case", "[loss][bce]") {
    Tensor y_true = Tensor::vector(2);
    y_true.set(0, 1.0f); y_true.set(1, 0.0f);
    Tensor y_pred = Tensor::vector(2);
    y_pred.set(0, 0.9f); y_pred.set(1, 0.1f);
    loss::BCE bce;
    float loss = bce.forward(y_pred, y_true);
    float expected = -0.5f * (std::log(0.9f) + std::log(1-0.1f));
    REQUIRE(loss == Catch::Approx(expected));
}


TEST_CASE("MSE loss computes correct value for simple case", "[loss][mse]") {
    Tensor y_true = Tensor::vector(2);
    y_true.set(0, 1.0f); y_true.set(1, 0.0f);
    Tensor y_pred = Tensor::vector(2);
    y_pred.set(0, 0.8f); y_pred.set(1, 0.2f);
    loss::MSE mse;
    float loss = mse.forward(y_pred, y_true);
    float expected = 0.5f * ((1.0f-0.8f)*(1.0f-0.8f) + (0.0f-0.2f)*(0.0f-0.2f));
    REQUIRE(loss == Catch::Approx(expected));
}


TEST_CASE("CCE loss backward computes correct gradient for simple case", "[loss][cce][backward]") {
    Tensor y_true = Tensor::vector(3);
    y_true.set(0, 0.0f); y_true.set(1, 1.0f); y_true.set(2, 0.0f);
    Tensor y_pred = Tensor::vector(3);
    y_pred.set(0, 0.1f); y_pred.set(1, 0.8f); y_pred.set(2, 0.1f);
    loss::CCE cce;
    cce.forward(y_pred, y_true);
    Tensor grad = cce.backward();
    // Gradient should be -y_true / y_pred
    REQUIRE(grad(0) == Catch::Approx(0.0f));
    REQUIRE(grad(1) == Catch::Approx(-1.0f / 0.8f));
    REQUIRE(grad(2) == Catch::Approx(0.0f));
}

TEST_CASE("BCE loss backward computes correct gradient for simple case", "[loss][bce][backward]") {
    Tensor y_true = Tensor::vector(2);
    y_true.set(0, 1.0f); y_true.set(1, 0.0f);
    Tensor y_pred = Tensor::vector(2);
    y_pred.set(0, 0.9f); y_pred.set(1, 0.1f);
    loss::BCE bce;
    bce.forward(y_pred, y_true);
    Tensor grad = bce.backward();
    // Gradient: - (y_true / y_pred - (1-y_true)/(1-y_pred)) / N
    float g0 = - (1.0f / 0.9f - 0.0f / (1.0f - 0.9f)) / 2.0f;
    float g1 = - (0.0f / 0.1f - 1.0f / (1.0f - 0.1f)) / 2.0f;
    REQUIRE(grad(0) == Catch::Approx(g0));
    REQUIRE(grad(1) == Catch::Approx(g1));
}

TEST_CASE("MSE loss backward computes correct gradient for simple case", "[loss][mse][backward]") {
    Tensor y_true = Tensor::vector(2);
    y_true.set(0, 1.0f); y_true.set(1, 0.0f);
    Tensor y_pred = Tensor::vector(2);
    y_pred.set(0, 0.8f); y_pred.set(1, 0.2f);
    loss::MSE mse;
    mse.forward(y_pred, y_true);
    Tensor grad = mse.backward();
    // Gradient: 2/N * (y_pred - y_true)
    float g0 = 2.0f/2.0f * (0.8f - 1.0f);
    float g1 = 2.0f/2.0f * (0.2f - 0.0f);
    REQUIRE(grad(0) == Catch::Approx(g0));
    REQUIRE(grad(1) == Catch::Approx(g1));
}
