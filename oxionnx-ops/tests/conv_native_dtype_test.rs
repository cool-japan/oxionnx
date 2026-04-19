//! Tests for ConvOp/ConvTransposeOp native F16/BF16 typed dispatch (D.3 Phase 1).
//!
//! Covers:
//! - native_dtypes() returns F32, F16, BF16
//! - F32 baseline parity (execute_typed vs execute)
//! - F16 parity against f32 reference, tol 1e-2
//! - BF16 parity against f32 reference, tol 5e-2
//! - ConvTranspose F16 parity
//! - ConvTranspose BF16 parity

use oxionnx_core::{
    dtype::{DType, TensorStorage, TypedTensor},
    graph::{Attributes, Node, OpKind},
    operator::{Operator, TypedOpContext},
};
use oxionnx_ops::registry::conv_ops::{ConvOp, ConvTransposeOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn conv_node(strides: [i64; 2], pads: [i64; 4], dilations: [i64; 2], group: i64) -> Node {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), strides.to_vec());
    attrs.int_lists.insert("pads".into(), pads.to_vec());
    attrs
        .int_lists
        .insert("dilations".into(), dilations.to_vec());
    attrs.ints.insert("group".into(), group);
    Node {
        name: "test_conv".into(),
        op: OpKind::Conv,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

fn conv_transpose_node(
    strides: [i64; 2],
    pads: [i64; 4],
    output_padding: [i64; 2],
    dilations: [i64; 2],
    group: i64,
) -> Node {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), strides.to_vec());
    attrs.int_lists.insert("pads".into(), pads.to_vec());
    attrs
        .int_lists
        .insert("output_padding".into(), output_padding.to_vec());
    attrs
        .int_lists
        .insert("dilations".into(), dilations.to_vec());
    attrs.ints.insert("group".into(), group);
    Node {
        name: "test_conv_transpose".into(),
        op: OpKind::ConvTranspose,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

fn make_typed_ctx_2<'a>(
    node: &'a Node,
    input: &'a TypedTensor,
    weight: &'a TypedTensor,
) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs: vec![Some(input), Some(weight)],
        outer_scope: None,
        registry: None,
    }
}

fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

fn f32_to_bf16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

/// Reference: naive f32 2D convolution (no padding, stride=1, dilation=1, group=1).
/// Shapes: input [N,Cin,H,W], weight [Cout,Cin,KH,KW] → output [N,Cout,OH,OW]
fn conv2d_f32_ref(
    input: &[f32],
    in_shape: [usize; 4],
    weight: &[f32],
    w_shape: [usize; 4],
    pads: [usize; 4],
) -> Vec<f32> {
    let (n, c_in, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (c_out, c_per_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    assert_eq!(
        c_in, c_per_g,
        "single-group ref: c_in must equal w_shape[1]"
    );
    let oh = h + pads[0] + pads[2] - (kh - 1);
    let ow = w + pads[1] + pads[3] - (kw - 1);
    let out_len = n * c_out * oh * ow;
    let mut out = vec![0.0f32; out_len];
    for ni in 0..n {
        for oc in 0..c_out {
            for oi in 0..oh {
                for oj in 0..ow {
                    let mut acc = 0.0f32;
                    for ic in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let si = oi + ki;
                                let sj = oj + kj;
                                let pi = si as isize - pads[0] as isize;
                                let pj = sj as isize - pads[1] as isize;
                                if pi >= 0 && pj >= 0 && (pi as usize) < h && (pj as usize) < w {
                                    let iv = input
                                        [((ni * c_in + ic) * h + pi as usize) * w + pj as usize];
                                    let wv = weight[((oc * c_per_g + ic) * kh + ki) * kw + kj];
                                    acc += iv * wv;
                                }
                            }
                        }
                    }
                    out[((ni * c_out + oc) * oh + oi) * ow + oj] = acc;
                }
            }
        }
    }
    out
}

/// Build a deterministic f32 sequence: 0.1, 0.2, … (wraps with small values).
fn det_vals(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i % 20) as f32 + 1.0) * scale) - 1.0)
        .collect()
}

// ── Tests: ConvOp ────────────────────────────────────────────────────────────

/// native_dtypes() must include F32, F16, BF16.
#[test]
fn test_conv_native_dtypes_includes_all_three() {
    let dtypes = ConvOp.native_dtypes();
    for dt in [DType::F32, DType::F16, DType::BF16] {
        assert!(
            dtypes.contains(&dt),
            "ConvOp.native_dtypes() must contain {dt:?}"
        );
    }
    assert_eq!(
        dtypes.len(),
        3,
        "ConvOp.native_dtypes() should have exactly 3 entries, got {}",
        dtypes.len()
    );
}

/// ConvTransposeOp native_dtypes() must include F32, F16, BF16.
#[test]
fn test_conv_transpose_native_dtypes_includes_all_three() {
    let dtypes = ConvTransposeOp.native_dtypes();
    for dt in [DType::F32, DType::BF16, DType::F16] {
        assert!(
            dtypes.contains(&dt),
            "ConvTransposeOp.native_dtypes() must contain {dt:?}"
        );
    }
    assert_eq!(
        dtypes.len(),
        3,
        "ConvTransposeOp.native_dtypes() should have exactly 3 entries"
    );
}

/// F32 baseline: execute_typed(F32) must produce correct output.
/// [1,1,4,4] input, [1,1,3,3] weight, stride=1, pad=0 → [1,1,2,2] output.
#[test]
fn test_conv_f32_baseline() {
    assert!(ConvOp.native_dtypes().contains(&DType::F32));

    // Checkerboard-like input to catch transposition bugs.
    #[rustfmt::skip]
    let input_vals: Vec<f32> = vec![
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    // Identity-like 3×3 kernel (centre=1, rest=0)
    let mut weight_vals = vec![0.0f32; 9];
    weight_vals[4] = 1.0;

    let ref_out = conv2d_f32_ref(
        &input_vals,
        [1, 1, 4, 4],
        &weight_vals,
        [1, 1, 3, 3],
        [0, 0, 0, 0],
    );

    let input = TypedTensor::new(TensorStorage::F32(input_vals), vec![1, 1, 4, 4]);
    let weight = TypedTensor::new(TensorStorage::F32(weight_vals), vec![1, 1, 3, 3]);
    let node = conv_node([1, 1], [0, 0, 0, 0], [1, 1], 1);
    let ctx = make_typed_ctx_2(&node, &input, &weight);
    let result = ConvOp.execute_typed(&ctx).expect("F32 conv failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::F32);
    assert_eq!(result[0].shape, vec![1, 1, 2, 2]);

    let out = result[0].storage.to_f32_vec();
    for (i, (&got, &expected)) in out.iter().zip(ref_out.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "F32 conv output[{i}]: got {got}, expected {expected}"
        );
    }
}

/// F16 parity: [1,4,8,8] input, [4,4,3,3] weight, stride=1, pad=1.
/// Tolerance: absolute 1e-2.
#[test]
fn test_conv_f16_parity() {
    assert!(ConvOp.native_dtypes().contains(&DType::F16));

    let batch = 1usize;
    let c_in = 4usize;
    let c_out = 4usize;
    let h = 8usize;
    let w = 8usize;
    let kh = 3usize;
    let kw = 3usize;

    let in_len = batch * c_in * h * w;
    let w_len = c_out * c_in * kh * kw;
    let input_f32 = det_vals(in_len, 0.1);
    let weight_f32 = det_vals(w_len, 0.05);

    // Compute f32 reference via execute_typed F32 path (full kernel)
    let input_f32_tt = TypedTensor::new(
        TensorStorage::F32(input_f32.clone()),
        vec![batch, c_in, h, w],
    );
    let weight_f32_tt = TypedTensor::new(
        TensorStorage::F32(weight_f32.clone()),
        vec![c_out, c_in, kh, kw],
    );
    let node = conv_node([1, 1], [1, 1, 1, 1], [1, 1], 1);
    let ctx_f32 = make_typed_ctx_2(&node, &input_f32_tt, &weight_f32_tt);
    let ref_result = ConvOp
        .execute_typed(&ctx_f32)
        .expect("F32 reference conv failed");
    let ref_out = ref_result[0].storage.to_f32_vec();
    let ref_shape = ref_result[0].shape.clone();

    // Now run F16 typed path
    let input_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&input_f32)),
        vec![batch, c_in, h, w],
    );
    let weight_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&weight_f32)),
        vec![c_out, c_in, kh, kw],
    );
    let ctx_f16 = make_typed_ctx_2(&node, &input_f16, &weight_f16);
    let result = ConvOp.execute_typed(&ctx_f16).expect("F16 conv failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::F16);
    assert_eq!(result[0].shape, ref_shape);

    if let TensorStorage::F16(ref bits) = result[0].storage {
        let got = f16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-2,
                "F16 conv output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected F16 storage, got {:?}", result[0].dtype());
    }
}

/// BF16 parity: same setup as F16, tolerance 5e-2.
#[test]
fn test_conv_bf16_parity() {
    assert!(ConvOp.native_dtypes().contains(&DType::BF16));

    let batch = 1usize;
    let c_in = 4usize;
    let c_out = 4usize;
    let h = 8usize;
    let w = 8usize;
    let kh = 3usize;
    let kw = 3usize;

    let in_len = batch * c_in * h * w;
    let w_len = c_out * c_in * kh * kw;
    let input_f32 = det_vals(in_len, 0.1);
    let weight_f32 = det_vals(w_len, 0.05);

    let input_f32_tt = TypedTensor::new(
        TensorStorage::F32(input_f32.clone()),
        vec![batch, c_in, h, w],
    );
    let weight_f32_tt = TypedTensor::new(
        TensorStorage::F32(weight_f32.clone()),
        vec![c_out, c_in, kh, kw],
    );
    let node = conv_node([1, 1], [1, 1, 1, 1], [1, 1], 1);
    let ctx_f32 = make_typed_ctx_2(&node, &input_f32_tt, &weight_f32_tt);
    let ref_result = ConvOp
        .execute_typed(&ctx_f32)
        .expect("F32 reference conv failed");
    let ref_out = ref_result[0].storage.to_f32_vec();
    let ref_shape = ref_result[0].shape.clone();

    let input_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&input_f32)),
        vec![batch, c_in, h, w],
    );
    let weight_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&weight_f32)),
        vec![c_out, c_in, kh, kw],
    );
    let ctx_bf16 = make_typed_ctx_2(&node, &input_bf16, &weight_bf16);
    let result = ConvOp.execute_typed(&ctx_bf16).expect("BF16 conv failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::BF16);
    assert_eq!(result[0].shape, ref_shape);

    if let TensorStorage::BF16(ref bits) = result[0].storage {
        let got = bf16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 5e-2,
                "BF16 conv output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected BF16 storage, got {:?}", result[0].dtype());
    }
}

// ── Tests: ConvTransposeOp ───────────────────────────────────────────────────

/// ConvTranspose F16 parity: [1,4,4,4] input, [4,4,2,2] weight, stride=2, pad=0.
/// output shape: [1, 4, 8, 8]. Tolerance 1e-2.
#[test]
fn test_conv_transpose_f16_parity() {
    assert!(ConvTransposeOp.native_dtypes().contains(&DType::F16));

    let batch = 1usize;
    let c_in = 4usize;
    let c_out = 4usize;
    let h = 4usize;
    let w = 4usize;
    let kh = 2usize;
    let kw = 2usize;

    let in_len = batch * c_in * h * w;
    let w_len = c_in * c_out * kh * kw;
    let input_f32 = det_vals(in_len, 0.1);
    let weight_f32 = det_vals(w_len, 0.05);

    // F32 reference
    let input_f32_tt = TypedTensor::new(
        TensorStorage::F32(input_f32.clone()),
        vec![batch, c_in, h, w],
    );
    let weight_f32_tt = TypedTensor::new(
        TensorStorage::F32(weight_f32.clone()),
        vec![c_in, c_out, kh, kw],
    );
    let node = conv_transpose_node([2, 2], [0, 0, 0, 0], [0, 0], [1, 1], 1);
    let ctx_f32 = make_typed_ctx_2(&node, &input_f32_tt, &weight_f32_tt);
    let ref_result = ConvTransposeOp
        .execute_typed(&ctx_f32)
        .expect("F32 reference ConvTranspose failed");
    let ref_out = ref_result[0].storage.to_f32_vec();
    let ref_shape = ref_result[0].shape.clone();

    // F16 typed path
    let input_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&input_f32)),
        vec![batch, c_in, h, w],
    );
    let weight_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&weight_f32)),
        vec![c_in, c_out, kh, kw],
    );
    let ctx_f16 = make_typed_ctx_2(&node, &input_f16, &weight_f16);
    let result = ConvTransposeOp
        .execute_typed(&ctx_f16)
        .expect("F16 ConvTranspose failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::F16);
    assert_eq!(result[0].shape, ref_shape);

    if let TensorStorage::F16(ref bits) = result[0].storage {
        let got = f16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-2,
                "F16 ConvTranspose output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected F16 storage, got {:?}", result[0].dtype());
    }
}

/// Fused relu activation: F16 typed output must match F32 output with relu applied.
///
/// Uses a small [1,1,4,4] input / [1,1,3,3] weight that produces both positive and
/// negative accumulation values. The relu activation is set via `attrs.strings`; the
/// F32 reference is obtained through `execute_typed` with F32 storage (which goes
/// through `default_typed_via_f32` and applies relu inside `execute()`).
#[test]
fn test_conv_f16_relu_activation() {
    use oxionnx_core::graph::OpKind;

    // Build a node that has relu activation set
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), vec![1i64, 1]);
    attrs.int_lists.insert("pads".into(), vec![0i64, 0, 0, 0]);
    attrs.int_lists.insert("dilations".into(), vec![1i64, 1]);
    attrs.ints.insert("group".into(), 1);
    attrs.strings.insert("activation".into(), "relu".into());
    let node = Node {
        name: "test_conv_relu".into(),
        op: OpKind::Conv,
        inputs: vec![],
        outputs: vec![],
        attrs,
    };

    // Values chosen so that all conv outputs are negative before relu.
    // A 3×3 all-negative kernel applied to a positive input gives negative
    // outputs everywhere; relu clamps them all to 0.
    #[rustfmt::skip]
    let input_vals: Vec<f32> = vec![
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    ];
    // Kernel: all −1 → outputs are −9.0 everywhere before relu
    #[rustfmt::skip]
    let weight_vals: Vec<f32> = vec![
        -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
    ];

    // F32 reference via execute_typed (goes through default_typed_via_f32 → execute())
    let input_f32 = TypedTensor::new(TensorStorage::F32(input_vals.clone()), vec![1, 1, 4, 4]);
    let weight_f32 = TypedTensor::new(TensorStorage::F32(weight_vals.clone()), vec![1, 1, 3, 3]);
    let ctx_f32 = make_typed_ctx_2(&node, &input_f32, &weight_f32);
    let ref_result = ConvOp
        .execute_typed(&ctx_f32)
        .expect("F32 relu conv failed");
    let ref_out = ref_result[0].storage.to_f32_vec();

    // Sanity: at least one output must be zero (relu clamps negatives)
    assert!(
        ref_out.contains(&0.0_f32),
        "relu should clamp some values to 0; ref_out: {ref_out:?}"
    );

    // F16 typed path
    let input_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&input_vals)),
        vec![1, 1, 4, 4],
    );
    let weight_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&weight_vals)),
        vec![1, 1, 3, 3],
    );
    let ctx_f16 = make_typed_ctx_2(&node, &input_f16, &weight_f16);
    let result = ConvOp
        .execute_typed(&ctx_f16)
        .expect("F16 relu conv failed");

    assert_eq!(result[0].dtype(), DType::F16);
    assert_eq!(result[0].shape, ref_result[0].shape);

    if let TensorStorage::F16(ref bits) = result[0].storage {
        let got = f16_bits_to_f32(bits);
        // All values must be non-negative (relu applied)
        for (i, &v) in got.iter().enumerate() {
            assert!(v >= 0.0, "F16 relu output[{i}] = {v} is negative");
        }
        // Must be close to f32 reference
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-2,
                "F16 relu conv output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected F16 storage, got {:?}", result[0].dtype());
    }
}

/// ConvTranspose BF16 parity: same setup as F16, tolerance 5e-2.
#[test]
fn test_conv_transpose_bf16_parity() {
    assert!(ConvTransposeOp.native_dtypes().contains(&DType::BF16));

    let batch = 1usize;
    let c_in = 4usize;
    let c_out = 4usize;
    let h = 4usize;
    let w = 4usize;
    let kh = 2usize;
    let kw = 2usize;

    let in_len = batch * c_in * h * w;
    let w_len = c_in * c_out * kh * kw;
    let input_f32 = det_vals(in_len, 0.1);
    let weight_f32 = det_vals(w_len, 0.05);

    // F32 reference
    let input_f32_tt = TypedTensor::new(
        TensorStorage::F32(input_f32.clone()),
        vec![batch, c_in, h, w],
    );
    let weight_f32_tt = TypedTensor::new(
        TensorStorage::F32(weight_f32.clone()),
        vec![c_in, c_out, kh, kw],
    );
    let node = conv_transpose_node([2, 2], [0, 0, 0, 0], [0, 0], [1, 1], 1);
    let ctx_f32 = make_typed_ctx_2(&node, &input_f32_tt, &weight_f32_tt);
    let ref_result = ConvTransposeOp
        .execute_typed(&ctx_f32)
        .expect("F32 reference ConvTranspose failed");
    let ref_out = ref_result[0].storage.to_f32_vec();
    let ref_shape = ref_result[0].shape.clone();

    // BF16 typed path
    let input_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&input_f32)),
        vec![batch, c_in, h, w],
    );
    let weight_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&weight_f32)),
        vec![c_in, c_out, kh, kw],
    );
    let ctx_bf16 = make_typed_ctx_2(&node, &input_bf16, &weight_bf16);
    let result = ConvTransposeOp
        .execute_typed(&ctx_bf16)
        .expect("BF16 ConvTranspose failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::BF16);
    assert_eq!(result[0].shape, ref_shape);

    if let TensorStorage::BF16(ref bits) = result[0].storage {
        let got = bf16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 5e-2,
                "BF16 ConvTranspose output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected BF16 storage, got {:?}", result[0].dtype());
    }
}
