// #include "optim/sgd.hpp"
// #include <catch_amalgamated.hpp>
// #include "module/dense.hpp"
// #include "tensor/tensor.hpp"

// using Catch::Approx;
// using module::Dense;
// using optim::SGD;
// using tensor::Tensor;

// TEST_CASE("SGD Optimizer Tests", "[sgd]") {
//     SECTION("SGD optimizer construction") {
//         float lr = 0.01f;
//         SGD sgd(lr);

//         // Constructor should not throw
//         REQUIRE_NOTHROW(SGD(0.001f));
//         REQUIRE_NOTHROW(SGD(1.0f));
//         REQUIRE_NOTHROW(SGD(0.1f));

//         // Should start with no registered modules
//         REQUIRE(sgd.getModuleCount() == 0);
//     }

//     SECTION("Module registration") {
//         SGD sgd(0.01f);
//         Dense dense1(2, 3);  // Has trainable parameters
//         Dense dense2(3, 1);  // Has trainable parameters

//         // Register modules
//         sgd.registerModule(&dense1);
//         sgd.registerModule(&dense2);

//         REQUIRE(sgd.getModuleCount() == 2);

//         // Test registering nullptr (should be safely ignored)
//         sgd.registerModule(nullptr);
//         REQUIRE(sgd.getModuleCount() == 2);
//     }

//     SECTION("Zero gradients functionality") {
//         SGD sgd(0.01f);
//         Dense dense(2, 2);

//         sgd.registerModule(&dense);

//         // Set up some gradients manually
//         const auto& gradients = dense.getGradients();
//         // Note: In a real scenario, gradients would be set by backward pass
//         // For this test, we'll just verify zeroGradients doesn't crash
//         REQUIRE_NOTHROW(sgd.zeroGradients());
//     }

//     SECTION("SGD step with simple parameter update") {
//         SGD sgd(0.1f);      // Learning rate = 0.1
//         Dense dense(2, 1);  // 2 inputs, 1 output

//         sgd.registerModule(&dense);

//         // Get initial parameters
//         auto params = dense.getParameters();
//         Tensor* weights = params["weights"];
//         Tensor* biases = params["biases"];

//         // Set known initial values
//         (*weights)(0, 0) = 1.0f;
//         (*weights)(0, 1) = 2.0f;
//         (*biases)(0) = 0.5f;

//         // Create input for forward pass
//         Tensor input({2});
//         input(0) = 1.0f;
//         input(1) = 1.0f;

//         // Forward pass
//         Tensor output = dense.forward(input);

//         // Create fake gradients (normally from backward pass)
//         Tensor output_grad({1});
//         output_grad(0) = 1.0f;  // Simple gradient

//         // Backward pass to compute gradients
//         Tensor input_grad = dense.backward(output_grad);

//         // Store original parameter values
//         float orig_w00 = (*weights)(0, 0);
//         float orig_w01 = (*weights)(0, 1);
//         float orig_b0 = (*biases)(0);

//         // Perform SGD step
//         sgd.step();

//         // Check that parameters have been updated
//         // Parameters should change: theta = theta - lr * grad
//         REQUIRE((*weights)(0, 0) != Approx(orig_w00));
//         REQUIRE((*weights)(0, 1) != Approx(orig_w01));
//         REQUIRE((*biases)(0) != Approx(orig_b0));
//     }

//     SECTION("SGD step with multiple modules") {
//         SGD sgd(0.05f);  // Learning rate = 0.05
//         Dense dense1(2, 2);
//         Dense dense2(2, 1);

//         sgd.registerModule(&dense1);
//         sgd.registerModule(&dense2);

//         REQUIRE(sgd.getModuleCount() == 2);

//         // Get parameters for both modules
//         auto params1 = dense1.getParameters();
//         auto params2 = dense2.getParameters();

//         // Set known initial values for module 1
//         (*params1["weights"])(0, 0) = 1.0f;
//         (*params1["weights"])(0, 1) = 2.0f;
//         (*params1["weights"])(1, 0) = 3.0f;
//         (*params1["weights"])(1, 1) = 4.0f;
//         (*params1["biases"])(0) = 0.1f;
//         (*params1["biases"])(1) = 0.2f;

//         // Set known initial values for module 2
//         (*params2["weights"])(0, 0) = 0.5f;
//         (*params2["weights"])(0, 1) = 1.5f;
//         (*params2["biases"])(0) = 0.3f;

//         // Create inputs and perform forward passes
//         Tensor input1({2});
//         input1(0) = 1.0f;
//         input1(1) = 1.0f;

//         Tensor output1 = dense1.forward(input1);
//         Tensor output2 = dense2.forward(output1);

//         // Create gradients and perform backward passes
//         Tensor final_grad({1});
//         final_grad(0) = 1.0f;

//         Tensor grad2 = dense2.backward(final_grad);
//         Tensor grad1 = dense1.backward(grad2);

//         // Store original values
//         float orig_w1_00 = (*params1["weights"])(0, 0);
//         float orig_w2_00 = (*params2["weights"])(0, 0);

//         // Perform SGD step
//         sgd.step();

//         // Check that parameters in both modules have been updated
//         REQUIRE((*params1["weights"])(0, 0) != Approx(orig_w1_00));
//         REQUIRE((*params2["weights"])(0, 0) != Approx(orig_w2_00));
//     }

//     SECTION("SGD step with zero learning rate") {
//         SGD sgd(0.0f);  // Zero learning rate
//         Dense dense(2, 1);

//         sgd.registerModule(&dense);

//         auto params = dense.getParameters();
//         Tensor* weights = params["weights"];

//         // Set initial value
//         (*weights)(0, 0) = 1.0f;
//         float original_value = (*weights)(0, 0);

//         // Perform forward and backward pass to generate gradients
//         Tensor input({2});
//         input(0) = 1.0f;
//         input(1) = 1.0f;

//         Tensor output = dense.forward(input);

//         Tensor output_grad({1});
//         output_grad(0) = 1.0f;

//         dense.backward(output_grad);

//         // Perform SGD step with zero learning rate
//         sgd.step();

//         // Parameters should remain unchanged
//         REQUIRE((*weights)(0, 0) == Approx(original_value));
//     }

//     SECTION("Multiple SGD steps") {
//         SGD sgd(0.1f);
//         Dense dense(1, 1);  // Simple 1x1 network

//         sgd.registerModule(&dense);

//         auto params = dense.getParameters();
//         Tensor* weights = params["weights"];

//         // Set initial values
//         (*weights)(0, 0) = 1.0f;

//         // Perform a few steps to verify parameters are being updated
//         std::vector<float> weight_values;

//         for (int step = 0; step < 3; ++step) {
//             // Zero gradients at start of each step
//             sgd.zeroGradients();

//             Tensor input({1});
//             input(0) = 1.0f;

//             Tensor output = dense.forward(input);

//             Tensor output_grad({1});
//             output_grad(0) = 1.0f;

//             dense.backward(output_grad);

//             // Store weight before update
//             weight_values.push_back((*weights)(0, 0));

//             sgd.step();
//         }

//         // Check that weights changed between steps
//         REQUIRE(weight_values.size() == 3);
//         REQUIRE(weight_values[0] != Approx(weight_values[1]));
//         REQUIRE(weight_values[1] != Approx(weight_values[2]));
//     }

//     SECTION("SGD step with no registered modules") {
//         SGD sgd(0.01f);

//         // Should not crash when stepping with no modules
//         REQUIRE_NOTHROW(sgd.step());
//         REQUIRE(sgd.getModuleCount() == 0);
//     }
// }