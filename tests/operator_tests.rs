//! Per-operator integration tests.
//!
//! Tests have been split into focused files by category:
//! - op_math_tests.rs  — MatMul, Gemm, Reduce family
//! - op_nn_tests.rs    — Softmax, LayerNorm, BatchNorm, activations
//! - op_conv_tests.rs  — Conv2D variants, MaxPool, AveragePool
//! - op_shape_tests.rs — Concat, Slice, Transpose, Reshape, Split, Identity
//! - op_elementwise_tests.rs — arithmetic, unary math, Clip, Bitwise, Size
