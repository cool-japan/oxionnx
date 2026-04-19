//! Node-level ONNX backend conformance tests — split into focused modules.
//!
//! Tests have been split across:
//!   - conformance_math.rs      (tests  1–15: arithmetic + reductions)
//!   - conformance_nn.rs        (tests 16–21, 36–37, 40: NN activations + norms)
//!   - conformance_shape.rs     (tests 22–28: reshape, transpose, concat, etc.)
//!   - conformance_conv.rs      (tests 29–31: conv2d + pooling)
//!   - conformance_indexing.rs  (tests 32–34, 38–39: gather, where, onehot, clip, identity)
//!   - conformance_quant.rs     (test  35: quantize/dequantize round-trip)
