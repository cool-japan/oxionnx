//! Conformance test 35: Quantization operators.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_close, run_op};

// ═══════════════════════════════════════════════════════════════════════════════
// 35: Quantization conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 35. conformance_quantize_dequantize — round-trip within tolerance
#[test]
fn conformance_quantize_dequantize() {
    // QuantizeLinear: y = clamp(round(x / scale) + zero_point, 0, 255)
    // DequantizeLinear: y = (x - zero_point) * scale
    //
    // x = [0.0, 1.0, 2.0, 3.0, 4.0]
    // scale = 0.1, zero_point = 0
    // Quantized: round([0, 10, 20, 30, 40]) = [0, 10, 20, 30, 40]
    // Dequantized: [0, 1.0, 2.0, 3.0, 4.0]
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5]);
    let scale = Tensor::new(vec![0.1], vec![1]);
    let zero_point = Tensor::new(vec![0.0], vec![1]);

    // Step 1: Quantize
    let quantized = run_op(
        OpKind::QuantizeLinear,
        vec!["x", "scale", "zp"],
        vec!["qout"],
        vec!["x", "scale", "zp"],
        vec![
            ("x", x),
            ("scale", scale.clone()),
            ("zp", zero_point.clone()),
        ],
        vec![],
        Attributes::default(),
    );
    let q = quantized.get("qout").unwrap();

    // Step 2: Dequantize
    let dequantized = run_op(
        OpKind::DequantizeLinear,
        vec!["q", "scale", "zp"],
        vec!["dqout"],
        vec!["q", "scale", "zp"],
        vec![("q", q.clone()), ("scale", scale), ("zp", zero_point)],
        vec![],
        Attributes::default(),
    );
    let dq = dequantized.get("dqout").unwrap();

    // Round-trip should be within scale tolerance
    assert_close(
        &dq.data,
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        0.1,
        "quantize_dequantize_roundtrip",
    );
}
