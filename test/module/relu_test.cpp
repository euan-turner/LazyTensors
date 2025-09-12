#include "module/relu.hpp"
#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"

using Catch::Approx;
using module::ReLU;
using tensor::Tensor;

TEST_CASE("ReLU Module Tests", "[relu]") {
    SECTION("ReLU module construction and basic properties") {
        ReLU relu;

        // Constructor should not throw
        REQUIRE_NOTHROW(ReLU());

        // ReLU should have no trainable parameters
        REQUIRE_FALSE(relu.hasTrainableParameters());

        // Should have no parameters
        auto params = relu.getParameters();
        REQUIRE(params.size() == 0);

        // Should have no gradients
        auto gradients = relu.getGradients();
        REQUIRE(gradients.size() == 0);
    }

    SECTION("Forward pass with 1D tensor") {
        ReLU relu;

        // Test input with positive, negative, and zero values
        auto input = std::make_shared<Tensor>(std::vector<size_t>{5});
        input->set(0, -2.0f);
        input->set(1, -1.0f);
        input->set(2, 0.0f);
        input->set(3, 1.0f);
        input->set(4, 2.0f);

        Tensor output = *relu.forward(input);

        // Check output dimensions
        REQUIRE(output.dims() == std::vector<size_t>{5});

        // Check ReLU function: max(0, x)
        REQUIRE(output(0) == Approx(0.0f));  // -2.0 -> 0.0
        REQUIRE(output(1) == Approx(0.0f));  // -1.0 -> 0.0
        REQUIRE(output(2) == Approx(0.0f));  // 0.0 -> 0.0
        REQUIRE(output(3) == Approx(1.0f));  // 1.0 -> 1.0
        REQUIRE(output(4) == Approx(2.0f));  // 2.0 -> 2.0
    }

    SECTION("Forward pass with 2D tensor") {
        ReLU relu;

        // Test 2x3 matrix
        auto input = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        input->set({0, 0}, -3.0f);
        input->set({0, 1}, 0.0f);
        input->set({0, 2}, 2.5f);
        input->set({1, 0}, -1.5f);
        input->set({1, 1}, 4.0f);
        input->set({1, 2}, -0.1f);

        Tensor output = *relu.forward(input);

        // Check output dimensions
        REQUIRE(output.dims() == std::vector<size_t>{2, 3});

        // Check ReLU function
        REQUIRE(output({0, 0}) == Approx(0.0f));  // -3.0 -> 0.0
        REQUIRE(output({0, 1}) == Approx(0.0f));  // 0.0 -> 0.0
        REQUIRE(output({0, 2}) == Approx(2.5f));  // 2.5 -> 2.5
        REQUIRE(output({1, 0}) == Approx(0.0f));  // -1.5 -> 0.0
        REQUIRE(output({1, 1}) == Approx(4.0f));  // 4.0 -> 4.0
        REQUIRE(output({1, 2}) == Approx(0.0f));  // -0.1 -> 0.0
    }

    SECTION("Backward pass with 1D tensor") {
        ReLU relu;

        // Forward pass first to cache input
        auto input = std::make_shared<Tensor>(std::vector<size_t>{4});
        input->set(0, -1.0f);
        input->set(1, 0.0f);
        input->set(2, 2.0f);
        input->set(3, -0.5f);

        relu.forward(input);

        // Gradient from upstream
        auto gradients = std::make_shared<Tensor>(std::vector<size_t>{4});
        gradients->set(0, 1.0f);
        gradients->set(1, 2.0f);
        gradients->set(2, 3.0f);
        gradients->set(3, 4.0f);

        Tensor grad_output = *relu.backward(gradients);

        // Check output dimensions
        REQUIRE(grad_output.dims() == std::vector<size_t>{4});

        // Check ReLU derivative: 1 if input > 0, 0 otherwise
        REQUIRE(grad_output(0) == Approx(0.0f));  // input -1.0 <= 0 -> gradient 0
        REQUIRE(grad_output(1) == Approx(0.0f));  // input 0.0 <= 0 -> gradient 0
        REQUIRE(grad_output(2) == Approx(3.0f));  // input 2.0 > 0 -> gradient passes through
        REQUIRE(grad_output(3) == Approx(0.0f));  // input -0.5 <= 0 -> gradient 0
    }

    SECTION("Backward pass with 2D tensor") {
        ReLU relu;

        // Forward pass first to cache input
        auto input = std::make_shared<Tensor>(std::vector<size_t>{2, 2});
        input->set({0, 0}, -2.0f);
        input->set({0, 1}, 1.0f);
        input->set({1, 0}, 0.0f);
        input->set({1, 1}, 3.0f);

        relu.forward(input);

        // Gradient from upstream
        auto gradients = std::make_shared<Tensor>(std::vector<size_t>{2, 2});
        gradients->set({0, 0}, 1.5f);
        gradients->set({0, 1}, 2.0f);
        gradients->set({1, 0}, 2.5f);
        gradients->set({1, 1}, 3.0f);

        Tensor grad_output = *relu.backward(gradients);

        // Check output dimensions
        REQUIRE(grad_output.dims() == std::vector<size_t>{2, 2});

        // Check ReLU derivative
        REQUIRE(grad_output({0, 0}) == Approx(0.0f));  // input -2.0 <= 0 -> gradient 0
        REQUIRE(grad_output({0, 1}) == Approx(2.0f));  // input 1.0 > 0 -> gradient passes through
        REQUIRE(grad_output({1, 0}) == Approx(0.0f));  // input 0.0 <= 0 -> gradient 0
        REQUIRE(grad_output({1, 1}) == Approx(3.0f));  // input 3.0 > 0 -> gradient passes through
    }

    SECTION("Forward-backward consistency") {
        ReLU relu;

        // Test that input shape is preserved through forward and backward
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3, 4, 2});

        // Fill with random values
        size_t total_elements = input->numel();
        auto dims = input->dims();
        std::vector<size_t> indices(dims.size(), 0);

        for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
            input->set(indices, (float)linear_idx - 12.0f);  // Mix of positive and negative values

            // Increment indices
            for (int d = dims.size() - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < dims[d]) {
                    break;
                }
                indices[d] = 0;
            }
        }

        Tensor output = *relu.forward(input);

        // Output should have same shape as input
        REQUIRE(output.dims() == input->dims());

        // Create dummy gradients
        auto gradients = std::make_shared<Tensor>(output);  // Same shape as output
        Tensor grad_input = *relu.backward(gradients);

        // Gradient input should have same shape as original input
        REQUIRE(grad_input.dims() == input->dims());
    }

    SECTION("Edge cases") {
        ReLU relu;

        SECTION("Single element tensor") {
            auto input = std::make_shared<Tensor>(std::vector<size_t>{1});
            input->set(0, -5.0f);

            Tensor output = *relu.forward(input);
            REQUIRE(output(0) == Approx(0.0f));

            auto gradients = std::make_shared<Tensor>(std::vector<size_t>{1});
            gradients->set(0, 2.0f);
            Tensor grad_output = *relu.backward(gradients);
            REQUIRE(grad_output(0) == Approx(0.0f));
        }

        SECTION("All positive values") {
            auto input = std::make_shared<Tensor>(std::vector<size_t>{3});
            input->set(0, 1.0f);
            input->set(1, 2.0f);
            input->set(2, 3.0f);

            Tensor output = *relu.forward(input);
            REQUIRE(output(0) == Approx(1.0f));
            REQUIRE(output(1) == Approx(2.0f));
            REQUIRE(output(2) == Approx(3.0f));

            auto gradients = std::make_shared<Tensor>(std::vector<size_t>{3});
            gradients->set(0, 1.0f);
            gradients->set(1, 1.0f);
            gradients->set(2, 1.0f);
            Tensor grad_output = *relu.backward(gradients);
            REQUIRE(grad_output(0) == Approx(1.0f));
            REQUIRE(grad_output(1) == Approx(1.0f));
            REQUIRE(grad_output(2) == Approx(1.0f));
        }

        SECTION("All negative values") {
            auto input = std::make_shared<Tensor>(std::vector<size_t>{3});
            input->set(0, -1.0f);
            input->set(1, -2.0f);
            input->set(2, -3.0f);

            Tensor output = *relu.forward(input);
            REQUIRE(output(0) == Approx(0.0f));
            REQUIRE(output(1) == Approx(0.0f));
            REQUIRE(output(2) == Approx(0.0f));

            auto gradients = std::make_shared<Tensor>(std::vector<size_t>{3});
            gradients->set(0, 1.0f);
            gradients->set(1, 1.0f);
            gradients->set(2, 1.0f);
            Tensor grad_output = *relu.backward(gradients);
            REQUIRE(grad_output(0) == Approx(0.0f));
            REQUIRE(grad_output(1) == Approx(0.0f));
            REQUIRE(grad_output(2) == Approx(0.0f));
        }
    }
}