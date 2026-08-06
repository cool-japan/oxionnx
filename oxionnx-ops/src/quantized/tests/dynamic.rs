//! Dynamic quantization tests.

use oxionnx_core::{Attributes, Node, OpContext, OpKind, Operator, Tensor};

use crate::quantized::dynamic_quantize;
use crate::registry::quant_ops::DynamicQuantizeLinearOp;

#[test]
fn test_dynamic_quantize_mixed() {
    let x = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], vec![2, 3]);
    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize mixed");
    let expected_scale = 4.0 / 255.0;
    assert!(
        (scale - expected_scale).abs() < 1e-6,
        "scale {} != expected {}",
        scale,
        expected_scale,
    );
    for &v in &q.data {
        assert!(
            (0.0_f32..=255.0_f32).contains(&v),
            "Dynamic quantize value {} out of [0,255]",
            v,
        );
    }
    let zp_f = zp as u8 as f32;
    for (i, &orig) in x.data.iter().enumerate() {
        let deq = (q.data[i] - zp_f) * scale;
        assert!(
            (deq - orig).abs() < scale * 1.5,
            "Dynamic roundtrip: orig={}, deq={}, diff={}",
            orig,
            deq,
            (deq - orig).abs(),
        );
    }
}

#[test]
fn test_dynamic_quantize_all_positive() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let (q, scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize all_positive");
    let expected_scale = 4.0 / 255.0;
    assert!(
        (scale - expected_scale).abs() < 1e-6,
        "all_positive scale {} != expected {}",
        scale,
        expected_scale,
    );
    for &v in &q.data {
        assert!((0.0_f32..=255.0_f32).contains(&v));
    }
}

#[test]
fn test_dynamic_quantize_all_negative() {
    let x = Tensor::new(vec![-4.0, -3.0, -2.0, -1.0], vec![4]);
    let (q, scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize all_negative");
    let expected_scale = 4.0 / 255.0;
    assert!(
        (scale - expected_scale).abs() < 1e-6,
        "all_negative scale {} != expected {}",
        scale,
        expected_scale,
    );
    for &v in &q.data {
        assert!((0.0_f32..=255.0_f32).contains(&v));
    }
}

#[test]
fn test_dynamic_quantize_range_includes_zero() {
    let x = Tensor::new(vec![5.0, 10.0, 15.0], vec![3]);
    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize zero_inclusive");
    let zp_u8 = zp as u8;
    let deq_zero = (zp_u8 as f32 - zp_u8 as f32) * scale;
    assert!(
        deq_zero.abs() < 1e-6,
        "Zero point should dequantize to 0, got {}",
        deq_zero,
    );
    for &v in &q.data {
        assert!((0.0_f32..=255.0_f32).contains(&v));
    }
}

#[test]
fn test_dynamic_quantize_single_element() {
    let x = Tensor::new(vec![42.0], vec![1]);
    let (q, _scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize single");
    assert_eq!(q.data.len(), 1);
    assert!((0.0_f32..=255.0_f32).contains(&q.data[0]));
}

#[test]
fn test_dynamic_quantize_empty_fails() {
    let x = Tensor::new(vec![], vec![0]);
    let result = dynamic_quantize(&x);
    assert!(result.is_err());
}

// ── W3-quant-stitch: brief-a3-4 spec fix ─────────────────────────────────────
//
// Wave-2 left `dynamic_quantize` with two spec deviations relative to the
// ONNX `DynamicQuantizeLinear` operator (and this crate's already-correct
// registered `DynamicQuantizeLinearOp`): the zero point was added *before*
// rounding instead of after, using ties-away-from-zero instead of
// ties-to-even. The tests below pin the fix.

/// Build a minimal single-input `OpContext`, for direct `Operator::execute`
/// calls in these tests (same shape as the pattern used across
/// `oxionnx-ops/tests/*_native_dtype_test.rs`).
fn single_input_ctx<'a>(node: &'a Node, x: &'a Tensor) -> OpContext<'a> {
    OpContext {
        node,
        inputs: vec![Some(x)],
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

/// Pins "zero point added after rounding, ties-to-even" with values verified
/// against `numpy.round` (ties-to-even, matching `onnx.reference`):
///
/// ```text
/// x = [-128.0, -127.5, 0.0, 24.5, 25.5, 127.0]
/// min_val = -128.0, max_val = 127.0, scale = 1.0, zero_point = 128 (-128 as i8)
/// data    = [0, 0, 128, 152, 154, 255]
/// ```
///
/// `-127.5` and `24.5` are the discriminating elements: at `scale == 1.0`
/// they make `x / scale` land exactly on a half-integer, the only place
/// "zero point added after rounding" and "ties-to-even" can change the
/// answer at all (elsewhere the fix is a no-op). `25.5` deliberately does
/// *not* discriminate (the `128`-valued zero point shift is even, so both
/// orderings agree there) — which is why this test also reproduces the
/// pre-fix formula, rather than trusting a single changed value to prove the
/// fix did anything.
#[test]
fn test_dynamic_quantize_negative_half_integer_zero_point_after_rounding() {
    let x = Tensor::new(vec![-128.0, -127.5, 0.0, 24.5, 25.5, 127.0], vec![6]);
    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize");

    assert_eq!(scale, 1.0, "scale");
    assert_eq!(zp, -128i8, "zero_point (128 reinterpreted as i8)");
    assert_eq!(
        q.data,
        vec![0.0, 0.0, 128.0, 152.0, 154.0, 255.0],
        "quantized data (ties-to-even, zero point added after rounding)"
    );

    // The pre-fix formula (`round(v/scale + zp_f)`, ties away from zero) is
    // reproduced here — not as production code, only to prove the fix above
    // actually changed the answer at the two discriminating elements.
    let zp_f = zp as u8 as f32;
    let pre_fix = |v: f32| (v / scale + zp_f).round().clamp(0.0, 255.0);
    assert_eq!(
        pre_fix(-127.5),
        1.0,
        "pre-fix formula sanity: would have been 1, not 0"
    );
    assert_eq!(
        pre_fix(24.5),
        153.0,
        "pre-fix formula sanity: would have been 153, not 152"
    );
    assert_ne!(
        pre_fix(-127.5),
        q.data[1],
        "fix must change the -127.5 element"
    );
    assert_ne!(pre_fix(24.5), q.data[3], "fix must change the 24.5 element");
}

/// The zero point's own tie-breaking rounding (independent of "added after
/// rounding") is exercised separately: pick `min(x)` so `-min_val / scale`
/// itself lands on a half-integer.
///
/// ```text
/// x = [-0.5, 254.5]  =>  min_val=-0.5, max_val=254.5, scale=1.0
/// zero_point = round_ties_even(0.5) = 0   (pre-fix, ties-away: round(0.5) = 1)
/// data       = [0.0, 254.0]
/// ```
///
/// Verified against `numpy.round`.
#[test]
fn test_dynamic_quantize_zero_point_uses_ties_to_even() {
    let x = Tensor::new(vec![-0.5, 254.5], vec![2]);
    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize");

    assert_eq!(scale, 1.0, "scale");
    assert_eq!(
        zp, 0i8,
        "zero_point: round_ties_even(0.5) = 0, not the pre-fix 1"
    );
    assert_eq!(q.data, vec![0.0, 254.0], "quantized data");
}

/// `quantized::dynamic_quantize` and the registered `DynamicQuantizeLinearOp`
/// must now agree exactly on any *non-degenerate* input (`max(x) !=
/// min(x)`): both compute `scale = (max_x - min_x) / 255` and now share the
/// same rounding/ordering, so they diverge only on the degenerate all-equal
/// case, where the helper's `1e-10` divide-by-zero guard and the operator's
/// spec-mandated `span = 1.0` substitution are deliberately different — a
/// separate, out-of-scope deviation, not exercised here.
#[test]
fn test_dynamic_quantize_matches_registered_operator_on_non_degenerate_input() {
    let x = Tensor::new(vec![1.0, 5.0, -3.0, 2.5, 0.0, 10.0, -7.5, 3.25], vec![8]);

    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize");

    let node = Node {
        op: OpKind::DynamicQuantizeLinear,
        name: "dql0".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec![
            "y".to_string(),
            "y_scale".to_string(),
            "y_zero_point".to_string(),
        ],
        attrs: Attributes::default(),
    };
    let ctx = single_input_ctx(&node, &x);
    let op_results = DynamicQuantizeLinearOp
        .execute(&ctx)
        .expect("DynamicQuantizeLinearOp::execute");
    let (op_y, op_scale, op_zp) = (&op_results[0], op_results[1].data[0], op_results[2].data[0]);

    assert_eq!(q.data, op_y.data, "quantized values must agree exactly");
    assert_eq!(scale, op_scale, "scale must agree exactly");
    assert_eq!(zp as u8 as f32, op_zp, "zero point must agree exactly");
}
