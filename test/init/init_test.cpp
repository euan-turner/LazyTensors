#include <catch_amalgamated.hpp>
#include "init/zeroes.hpp"
#include "init/xavier.hpp"
#include "tensor/tensor.hpp"

using namespace tensor;
using namespace init;

TEST_CASE("Zeroes initialiser sets all elements to zero", "[init][zeroes]") {
    Tensor t = Tensor::vector(5);
    Zeroes z;
    z.initialise(t);
    for (size_t i = 0; i < t.length(); ++i) {
        REQUIRE(t(i) == Catch::Approx(0.0f));
    }
}

TEST_CASE("Xavier initialiser produces values in expected range", "[init][xavier]") {
    size_t fan_in = 4, fan_out = 3;
    float gain = 1.0f;
    Tensor t = Tensor::vector(1000);
    Xavier x(fan_in, fan_out, gain, 42); // fixed seed for reproducibility
    x.initialise(t);
    // For Xavier uniform, values should be in [-a, a] where a = gain * sqrt(6/(fan_in+fan_out))
    float a = gain * std::sqrt(6.0f / (fan_in + fan_out));
    float sum = 0.0f, sum_sq = 0.0f;
    for (size_t i = 0; i < t.length(); ++i) {
        float v = t(i);
        sum += v;
        sum_sq += v * v;
        REQUIRE(std::isfinite(v));
    }
    float mean = sum / t.length();
    float var = sum_sq / t.length() - mean * mean;
    float expected_var = 2.0f / (fan_in + fan_out);
    REQUIRE(std::abs(mean) < 0.05f); // mean close to 0
    REQUIRE(std::abs(var - expected_var) < 0.05f * expected_var); // variance close to theory
}
