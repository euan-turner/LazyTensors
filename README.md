# Lazy Tensors

A simple library for Tensor operations, supporting lazy evaluation (where possible), with kernel fusion at the point operations are executed.

CPU and CUDA back-ends currently being implemented.

Worklist:
[ ] Clean up removal of old code
[ ] Document roles of Tensor and TensorImpl
[ ] Broadcasting of binary operations
[ ] Broadcasting of matrix multiplications
[ ] Elevate lazy buffering of operations to TensorImpl
[ ] Kernel fusion of binary operations
[ ] Kernel fusion of unary operations
