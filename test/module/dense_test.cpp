#include "module/dense.hpp"
#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"
#include <memory>

using Catch::Approx;
using module::Dense;
using tensor::Tensor;

TEST_CASE("Dense Module Tests", "[dense]") {
    SECTION("Dense module construction and basic properties") {
        Dense dense(3, 2); // 3 input, 2 output
        REQUIRE_NOTHROW(Dense(3, 2));
        REQUIRE(dense.hasTrainableParameters());
        auto params = dense.getParameters();
        REQUIRE(params.size() == 2); // weights and bias
        auto gradients = dense.getGradients();
        REQUIRE(gradients.size() == 2);
    }

    SECTION("Forward pass with 1D tensor") {
        Dense dense(3, 2);
        // Set weights and bias for deterministic output
        auto params = dense.getParameters();
        // weights: 2x3, bias: 2
        params[0]->set({0,0}, 1.0f); params[0]->set({0,1}, 2.0f); params[0]->set({0,2}, 3.0f);
        params[0]->set({1,0}, 4.0f); params[0]->set({1,1}, 5.0f); params[0]->set({1,2}, 6.0f);
        params[1]->set(0, 0.5f); params[1]->set(1, -0.5f);
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3});
        input->set(0, 1.0f); input->set(1, 2.0f); input->set(2, 3.0f);
        Tensor output = *dense.forward(input);
        REQUIRE(output.dims() == std::vector<size_t>{2});
        // y = Wx + b
        // [1 2 3] * [[1 2 3],[4 5 6]]^T + [0.5, -0.5]
        // output[0] = 1*1 + 2*2 + 3*3 + 0.5 = 1+4+9+0.5 = 14.5
        // output[1] = 1*4 + 2*5 + 3*6 - 0.5 = 4+10+18-0.5 = 31.5
        REQUIRE(output(0) == Approx(14.5f));
        REQUIRE(output(1) == Approx(31.5f));
    }

    SECTION("Forward pass with 2D tensor (batch)") {
        Dense dense(3, 2);
        auto params = dense.getParameters();
        params[0]->set({0,0}, 1.0f); params[0]->set({0,1}, 2.0f); params[0]->set({0,2}, 3.0f);
        params[0]->set({1,0}, 4.0f); params[0]->set({1,1}, 5.0f); params[0]->set({1,2}, 6.0f);
        params[1]->set(0, 0.5f); params[1]->set(1, -0.5f);
        auto input = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        input->set({0,0}, 1.0f); input->set({0,1}, 2.0f); input->set({0,2}, 3.0f);
        input->set({1,0}, 4.0f); input->set({1,1}, 5.0f); input->set({1,2}, 6.0f);
        Tensor output = *dense.forward(input);
        REQUIRE(output.dims() == std::vector<size_t>{2,2});
        // First row: same as 1D test
        REQUIRE(output(0,0) == Approx(14.5f));
        REQUIRE(output(0,1) == Approx(31.5f));
        // Second row:
        // output[1,0] = 4*1 + 5*2 + 6*3 + 0.5 = 4+10+18+0.5 = 32.5
        // output[1,1] = 4*4 + 5*5 + 6*6 - 0.5 = 16+25+36-0.5 = 76.5
        REQUIRE(output(1,0) == Approx(32.5f));
        REQUIRE(output(1,1) == Approx(76.5f));
    }

    SECTION("Backward pass with 1D tensor") {
        Dense dense(3, 2);
        auto params = dense.getParameters();
        params[0]->set({0,0}, 1.0f); params[0]->set({0,1}, 2.0f); params[0]->set({0,2}, 3.0f);
        params[0]->set({1,0}, 4.0f); params[0]->set({1,1}, 5.0f); params[0]->set({1,2}, 6.0f);
        params[1]->set(0, 0.5f); params[1]->set(1, -0.5f);
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3});
        input->set(0, 1.0f); input->set(1, 2.0f); input->set(2, 3.0f);
        // Forward
        auto output_ptr = dense.forward(input);
        // Simulate gradient from next layer
        Tensor grad_out({2}); grad_out.set(0, 1.0f); grad_out.set(1, 2.0f);
        auto grad_in = dense.backward(input, grad_out);
        // grad_in = W^T * grad_out
        // W^T = [[1,4],[2,5],[3,6]]
        // grad_in = [1*1+4*2, 2*1+5*2, 3*1+6*2] = [1+8, 2+10, 3+12] = [9,12,15]
        REQUIRE(grad_in->dims() == std::vector<size_t>{3});
        REQUIRE((*grad_in)(0) == Approx(9.0f));
        REQUIRE((*grad_in)(1) == Approx(12.0f));
        REQUIRE((*grad_in)(2) == Approx(15.0f));
    }

    SECTION("Backward pass with 2D tensor (batch)") {
        Dense dense(3, 2);
        auto params = dense.getParameters();
        params[0]->set({0,0}, 1.0f); params[0]->set({0,1}, 2.0f); params[0]->set({0,2}, 3.0f);
        params[0]->set({1,0}, 4.0f); params[0]->set({1,1}, 5.0f); params[0]->set({1,2}, 6.0f);
        params[1]->set(0, 0.5f); params[1]->set(1, -0.5f);
        auto input = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        input->set({0,0}, 1.0f); input->set({0,1}, 2.0f); input->set({0,2}, 3.0f);
        input->set({1,0}, 4.0f); input->set({1,1}, 5.0f); input->set({1,2}, 6.0f);
        // Forward
        auto output_ptr = dense.forward(input);
        // Simulate gradient from next layer
        Tensor grad_out({2,2}); grad_out.set({0,0}, 1.0f); grad_out.set({0,1}, 2.0f);
        grad_out.set({1,0}, 3.0f); grad_out.set({1,1}, 4.0f);
        auto grad_in = dense.backward(input, grad_out);
        REQUIRE(grad_in->dims() == std::vector<size_t>{2,3});
        // Only check shape and a few values for illustration
        // (full correctness would require a more detailed check)
        // grad_in[0] = W^T * grad_out[0]
        // grad_in[1] = W^T * grad_out[1]
        // grad_out[0] = [1,2], grad_out[1] = [3,4]
        // grad_in[0,0] = 1*1+4*2 = 1+8=9
        // grad_in[0,1] = 2*1+5*2 = 2+10=12
        // grad_in[0,2] = 3*1+6*2 = 3+12=15
        // grad_in[1,0] = 1*3+4*4 = 3+16=19
        // grad_in[1,1] = 2*3+5*4 = 6+20=26
        // grad_in[1,2] = 3*3+6*4 = 9+24=33
        REQUIRE((*grad_in)(0,0) == Approx(9.0f));
        REQUIRE((*grad_in)(0,1) == Approx(12.0f));
        REQUIRE((*grad_in)(0,2) == Approx(15.0f));
        REQUIRE((*grad_in)(1,0) == Approx(19.0f));
        REQUIRE((*grad_in)(1,1) == Approx(26.0f));
        REQUIRE((*grad_in)(1,2) == Approx(33.0f));
    }
}
