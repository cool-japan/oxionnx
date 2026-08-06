//! Wave-1 correctness tests for the spatial operators: `auto_pad`,
//! `ceil_mode`, pooling `dilations`, MaxPool `Indices`/`storage_order`,
//! ConvTranspose `output_shape`, and the checked output-shape arithmetic that
//! must produce typed errors instead of panicking on malformed models.
//!
//! Reference values were computed with an independent NumPy implementation of
//! the ONNX reference semantics and inlined here as constants.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    OnnxError, Tensor,
};
use oxionnx_ops::registry::conv_ops::{AveragePoolOp, ConvOp, ConvTransposeOp, MaxPoolOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn node_with(op: OpKind, outputs: &[&str]) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn ints(node: &mut Node, name: &str, values: &[i64]) {
    node.attrs.int_lists.insert(name.into(), values.to_vec());
}

fn int(node: &mut Node, name: &str, value: i64) {
    node.attrs.ints.insert(name.into(), value);
}

fn string(node: &mut Node, name: &str, value: &str) {
    node.attrs.strings.insert(name.into(), value.into());
}

fn assert_close(got: &Tensor, want_shape: &[usize], want: &[f32], label: &str) {
    assert_eq!(got.shape, want_shape, "{label}: shape");
    assert_eq!(got.data.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.data.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "{label}[{i}]: got {g}, expected {w}");
    }
}

/// Run an operator through both `execute` and `execute_into_slots` and assert
/// the two agree, returning the `execute` result.
fn run_both(op: &dyn Operator, node: &Node, inputs: Vec<Option<&Tensor>>) -> Vec<Tensor> {
    let context = ctx(node, inputs.clone());
    let direct = op.execute(&context).expect("execute must succeed");

    if op.supports_output_slots() {
        let mut slots: Vec<Tensor> = node
            .outputs
            .iter()
            .map(|_| Tensor::new(Vec::new(), vec![0]))
            .collect();
        let slot_ctx = ctx(node, inputs);
        op.execute_into_slots(&slot_ctx, &mut slots)
            .expect("execute_into_slots must succeed");
        for (i, expected) in direct.iter().enumerate() {
            assert_eq!(slots[i].shape, expected.shape, "slot[{i}] shape parity");
            assert_eq!(slots[i].data, expected.data, "slot[{i}] data parity");
        }
    }
    direct
}

fn iota(count: usize, shape: Vec<usize>) -> Tensor {
    Tensor::new((0..count).map(|v| v as f32).collect(), shape)
}

// ── Conv: auto_pad ───────────────────────────────────────────────────────────

#[test]
fn conv_auto_pad_same_upper_preserves_spatial_size() {
    // 1x1x5x5 iota, 3x3 all-ones kernel, stride 1, auto_pad=SAME_UPPER.
    // ONNX: out = ceil(5/1) = 5, total pad = 4 + 3 - 5 = 2 → (1, 1).
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    string(&mut node, "auto_pad", "SAME_UPPER");

    let out = run_both(&ConvOp, &node, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out[0],
        &[1, 1, 5, 5],
        &[
            12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0,
            81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0,
        ],
        "conv SAME_UPPER",
    );
}

#[test]
fn conv_auto_pad_same_upper_vs_same_lower_odd_padding() {
    // 6x6 input, 3x3 kernel, stride 2 → out = ceil(6/2) = 3,
    // total pad = 2*2 + 3 - 6 = 1 (odd): SAME_UPPER puts it at the end,
    // SAME_LOWER at the beginning — the two must differ.
    let input = iota(36, vec![1, 1, 6, 6]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);

    let mut upper = node_with(OpKind::Conv, &["y"]);
    string(&mut upper, "auto_pad", "SAME_UPPER");
    ints(&mut upper, "strides", &[2, 2]);
    let out_upper = run_both(&ConvOp, &upper, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out_upper[0],
        &[1, 1, 3, 3],
        &[63.0, 81.0, 63.0, 171.0, 189.0, 135.0, 168.0, 180.0, 126.0],
        "conv SAME_UPPER stride 2",
    );

    let mut lower = node_with(OpKind::Conv, &["y"]);
    string(&mut lower, "auto_pad", "SAME_LOWER");
    ints(&mut lower, "strides", &[2, 2]);
    let out_lower = run_both(&ConvOp, &lower, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out_lower[0],
        &[1, 1, 3, 3],
        &[14.0, 30.0, 42.0, 75.0, 126.0, 144.0, 147.0, 234.0, 252.0],
        "conv SAME_LOWER stride 2",
    );
}

#[test]
fn conv_auto_pad_valid_overrides_explicit_pads() {
    // auto_pad=VALID must ignore an explicit `pads` attribute entirely.
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    string(&mut node, "auto_pad", "VALID");
    ints(&mut node, "pads", &[1, 1, 1, 1]);

    let out = run_both(&ConvOp, &node, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out[0],
        &[1, 1, 3, 3],
        &[54.0, 63.0, 72.0, 99.0, 108.0, 117.0, 144.0, 153.0, 162.0],
        "conv VALID",
    );
}

#[test]
fn conv_rejects_unknown_auto_pad() {
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    string(&mut node, "auto_pad", "SAME");

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("unknown auto_pad must be rejected");
    assert!(
        matches!(err, OnnxError::Unsupported(_)),
        "expected Unsupported, got {err:?}"
    );
}

// ── Conv: hardened output-shape arithmetic ───────────────────────────────────

#[test]
fn conv_zero_stride_is_typed_error_not_divide_by_zero() {
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "strides", &[0, 0]);

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("strides=[0,0] must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_zero_group_is_typed_error_not_divide_by_zero() {
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    int(&mut node, "group", 0);

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("group=0 must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_kernel_larger_than_padded_input_is_typed_error() {
    // 1x1x1x1 input with a 3x3 kernel and no padding used to underflow
    // `h + pads - dilation*(k-1) - 1`, panicking in debug and allocating
    // ~usize::MAX elements in release.
    let input = Tensor::new(vec![1.0_f32], vec![1, 1, 1, 1]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let node = node_with(OpKind::Conv, &["y"]);

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("1x1 input with a 3x3 kernel must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_dilation_beyond_input_extent_is_typed_error() {
    // dilations=[4,4] on a 3x3 kernel → effective extent 9 > 5.
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "dilations", &[4, 4]);

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("dilated kernel wider than the input must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_negative_pads_are_typed_error() {
    let input = iota(25, vec![1, 1, 5, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "pads", &[-1, 0, 0, 0]);

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("negative pads must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_non_4d_input_is_typed_error() {
    let input = iota(5, vec![1, 1, 5]);
    let weight = Tensor::new(vec![1.0_f32; 9], vec![1, 1, 3, 3]);
    let node = node_with(OpKind::Conv, &["y"]);

    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("3D input must be rejected, not panic");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

// ── MaxPool: ceil_mode / dilations / auto_pad ────────────────────────────────

#[test]
fn max_pool_ceil_mode_keeps_trailing_window() {
    // 1x1x5x5, kernel 2x2, stride 2.
    // ceil_mode=1 → 3x3 (ONNX), ceil_mode=0 → 2x2.
    let input = iota(25, vec![1, 1, 5, 5]);

    let mut ceil = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut ceil, "kernel_shape", &[2, 2]);
    ints(&mut ceil, "strides", &[2, 2]);
    int(&mut ceil, "ceil_mode", 1);
    let out = run_both(&MaxPoolOp, &ceil, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 3, 3],
        &[6.0, 8.0, 9.0, 16.0, 18.0, 19.0, 21.0, 23.0, 24.0],
        "maxpool ceil_mode=1",
    );

    let mut floor = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut floor, "kernel_shape", &[2, 2]);
    ints(&mut floor, "strides", &[2, 2]);
    let out = run_both(&MaxPoolOp, &floor, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 2, 2],
        &[6.0, 8.0, 16.0, 18.0],
        "maxpool ceil_mode=0",
    );
}

#[test]
fn max_pool_ceil_mode_h4_k3_s2() {
    // The brief's second failing case: H=W=4, kernel 3, stride 2.
    // ceil: ceil((4-3)/2)+1 = 2; floor: floor((4-3)/2)+1 = 1.
    let input = iota(16, vec![1, 1, 4, 4]);

    let mut ceil = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut ceil, "kernel_shape", &[3, 3]);
    ints(&mut ceil, "strides", &[2, 2]);
    int(&mut ceil, "ceil_mode", 1);
    let out = run_both(&MaxPoolOp, &ceil, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 2, 2],
        &[10.0, 11.0, 14.0, 15.0],
        "maxpool H4 K3 S2 ceil",
    );

    let mut floor = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut floor, "kernel_shape", &[3, 3]);
    ints(&mut floor, "strides", &[2, 2]);
    let out = run_both(&MaxPoolOp, &floor, vec![Some(&input)]);
    assert_close(&out[0], &[1, 1, 1, 1], &[10.0], "maxpool H4 K3 S2 floor");
}

#[test]
fn max_pool_ceil_mode_drops_window_starting_in_right_padding() {
    // 5x5, kernel 3, stride 3, pads [0,0,2,2], ceil_mode=1.
    // The raw ceil formula yields ceil((5+2-3)/3)+1 = 3, but the third window
    // would start at index 6 — inside the right padding — so ONNX / ONNX
    // Runtime drop it and the answer is 2.
    let input = iota(25, vec![1, 1, 5, 5]);
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3, 3]);
    ints(&mut node, "strides", &[3, 3]);
    ints(&mut node, "pads", &[0, 0, 2, 2]);
    int(&mut node, "ceil_mode", 1);

    let out = run_both(&MaxPoolOp, &node, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 2, 2],
        &[12.0, 14.0, 22.0, 24.0],
        "maxpool ceil_mode right-pad correction",
    );
}

#[test]
fn max_pool_honors_dilations() {
    // 4x4 input, 2x2 kernel, stride 1, dilations 2 → effective extent 3 → 2x2.
    let input = iota(16, vec![1, 1, 4, 4]);
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2]);
    ints(&mut node, "dilations", &[2, 2]);

    let out = run_both(&MaxPoolOp, &node, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 2, 2],
        &[10.0, 11.0, 14.0, 15.0],
        "maxpool dilations=2",
    );
}

#[test]
fn max_pool_auto_pad_same_upper() {
    let input = iota(25, vec![1, 1, 5, 5]);
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3, 3]);
    string(&mut node, "auto_pad", "SAME_UPPER");

    let out = run_both(&MaxPoolOp, &node, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 5, 5],
        &[
            6.0, 7.0, 8.0, 9.0, 9.0, 11.0, 12.0, 13.0, 14.0, 14.0, 16.0, 17.0, 18.0, 19.0, 19.0,
            21.0, 22.0, 23.0, 24.0, 24.0, 21.0, 22.0, 23.0, 24.0, 24.0,
        ],
        "maxpool SAME_UPPER",
    );
}

// ── MaxPool: Indices output and storage_order ────────────────────────────────

#[test]
fn max_pool_emits_indices_row_major() {
    let input = iota(25, vec![1, 1, 5, 5]);
    let mut node = node_with(OpKind::MaxPool, &["y", "idx"]);
    ints(&mut node, "kernel_shape", &[2, 2]);
    ints(&mut node, "strides", &[2, 2]);
    int(&mut node, "ceil_mode", 1);

    let out = run_both(&MaxPoolOp, &node, vec![Some(&input)]);
    assert_eq!(out.len(), 2, "MaxPool with 2 outputs must emit Indices");
    assert_close(
        &out[1],
        &[1, 1, 3, 3],
        &[6.0, 8.0, 9.0, 16.0, 18.0, 19.0, 21.0, 23.0, 24.0],
        "maxpool indices row major",
    );
}

#[test]
fn max_pool_indices_respect_storage_order() {
    // storage_order=1 encodes the winner as ((n*C+c)*W + w)*H + h.
    let input = iota(25, vec![1, 1, 5, 5]);
    let mut node = node_with(OpKind::MaxPool, &["y", "idx"]);
    ints(&mut node, "kernel_shape", &[3, 3]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    int(&mut node, "storage_order", 1);

    let out = run_both(&MaxPoolOp, &node, vec![Some(&input)]);
    assert_eq!(out.len(), 2);
    assert_close(
        &out[1],
        &[1, 1, 5, 5],
        &[
            6.0, 11.0, 16.0, 21.0, 21.0, 7.0, 12.0, 17.0, 22.0, 22.0, 8.0, 13.0, 18.0, 23.0, 23.0,
            9.0, 14.0, 19.0, 24.0, 24.0, 9.0, 14.0, 19.0, 24.0, 24.0,
        ],
        "maxpool indices column major",
    );
}

#[test]
fn max_pool_single_output_does_not_emit_indices() {
    let input = iota(16, vec![1, 1, 4, 4]);
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2]);
    ints(&mut node, "strides", &[2, 2]);

    let out = run_both(&MaxPoolOp, &node, vec![Some(&input)]);
    assert_eq!(out.len(), 1);
}

#[test]
fn max_pool_rejects_invalid_storage_order() {
    let input = iota(16, vec![1, 1, 4, 4]);
    let mut node = node_with(OpKind::MaxPool, &["y", "idx"]);
    ints(&mut node, "kernel_shape", &[2, 2]);
    int(&mut node, "storage_order", 2);

    let err = MaxPoolOp
        .execute(&ctx(&node, vec![Some(&input)]))
        .expect_err("storage_order=2 must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn max_pool_missing_kernel_shape_is_typed_error() {
    // Previously `ks_v[0]` indexed an empty slice and panicked.
    let input = iota(16, vec![1, 1, 4, 4]);
    let node = node_with(OpKind::MaxPool, &["y"]);

    let err = MaxPoolOp
        .execute(&ctx(&node, vec![Some(&input)]))
        .expect_err("missing kernel_shape must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn max_pool_zero_stride_is_typed_error() {
    let input = iota(16, vec![1, 1, 4, 4]);
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2]);
    ints(&mut node, "strides", &[0, 0]);

    let err = MaxPoolOp
        .execute(&ctx(&node, vec![Some(&input)]))
        .expect_err("strides=[0,0] must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

// ── AveragePool: ceil_mode / count_include_pad / auto_pad ────────────────────

#[test]
fn average_pool_ceil_mode_and_count_include_pad() {
    // 1x1x5x5, kernel 2x2, stride 2, ceil_mode=1 → 3x3.
    // The trailing row/column windows are partly outside the input, so
    // count_include_pad flips the divisor from the live-element count to 4.
    let input = iota(25, vec![1, 1, 5, 5]);

    let mut exclude = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut exclude, "kernel_shape", &[2, 2]);
    ints(&mut exclude, "strides", &[2, 2]);
    int(&mut exclude, "ceil_mode", 1);
    let out = run_both(&AveragePoolOp, &exclude, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 3, 3],
        &[3.0, 5.0, 6.5, 13.0, 15.0, 16.5, 20.5, 22.5, 24.0],
        "avgpool ceil count_include_pad=0",
    );

    let mut include = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut include, "kernel_shape", &[2, 2]);
    ints(&mut include, "strides", &[2, 2]);
    int(&mut include, "ceil_mode", 1);
    int(&mut include, "count_include_pad", 1);
    let out = run_both(&AveragePoolOp, &include, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 3, 3],
        &[3.0, 5.0, 3.25, 13.0, 15.0, 8.25, 10.25, 11.25, 6.0],
        "avgpool ceil count_include_pad=1",
    );
}

#[test]
fn average_pool_auto_pad_same_upper() {
    // 4x4 input, 3x3 kernel, stride 1 → SAME pads (1, 1).
    let input = iota(16, vec![1, 1, 4, 4]);

    let mut exclude = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut exclude, "kernel_shape", &[3, 3]);
    string(&mut exclude, "auto_pad", "SAME_UPPER");
    let out = run_both(&AveragePoolOp, &exclude, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 4, 4],
        &[
            2.5, 3.0, 4.0, 4.5, 4.5, 5.0, 6.0, 6.5, 8.5, 9.0, 10.0, 10.5, 10.5, 11.0, 12.0, 12.5,
        ],
        "avgpool SAME_UPPER count_include_pad=0",
    );

    let mut include = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut include, "kernel_shape", &[3, 3]);
    string(&mut include, "auto_pad", "SAME_UPPER");
    int(&mut include, "count_include_pad", 1);
    let out = run_both(&AveragePoolOp, &include, vec![Some(&input)]);
    assert_close(
        &out[0],
        &[1, 1, 4, 4],
        &[
            // sums 10/9, 18/9, 24/9, 18/9, … divided by the full 3x3 window
            1.1111, 2.0, 2.6667, 2.0, 3.0, 5.0, 6.0, 4.3333, 5.6667, 9.0, 10.0, 7.0, 4.6667, 7.3333,
            8.0, 5.5556,
        ],
        "avgpool SAME_UPPER count_include_pad=1",
    );
}

// ── ConvTranspose: output_shape / auto_pad ───────────────────────────────────

fn conv_transpose_fixture() -> (Tensor, Tensor) {
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = Tensor::new(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );
    (input, weight)
}

#[test]
fn conv_transpose_natural_output_shape_unchanged() {
    // Sanity baseline: stride 2, no pads → 2*(2-1) + 3 = 5.
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "strides", &[2, 2]);

    let out = run_both(&ConvTransposeOp, &node, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out[0],
        &[1, 1, 5, 5],
        &[
            1.0, 2.0, 5.0, 4.0, 6.0, 4.0, 5.0, 14.0, 10.0, 12.0, 10.0, 14.0, 36.0, 24.0, 30.0,
            12.0, 15.0, 34.0, 20.0, 24.0, 21.0, 24.0, 55.0, 32.0, 36.0,
        ],
        "convtranspose natural",
    );
}

#[test]
fn conv_transpose_output_shape_attribute_derives_pads() {
    // output_shape given as the full [N, C, oH, oW] form (what PyTorch/TF emit).
    // total_padding = 5 - 4 = 1; NOTSET puts the extra pad at the beginning.
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "strides", &[2, 2]);
    ints(&mut node, "output_shape", &[1, 1, 4, 4]);

    let out = run_both(&ConvTransposeOp, &node, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out[0],
        &[1, 1, 4, 4],
        &[
            5.0, 14.0, 10.0, 12.0, 14.0, 36.0, 24.0, 30.0, 15.0, 34.0, 20.0, 24.0, 24.0, 55.0,
            32.0, 36.0,
        ],
        "convtranspose output_shape NOTSET",
    );
}

#[test]
fn conv_transpose_output_shape_same_upper_crops_at_the_end() {
    // Same target extent, SAME_UPPER → the extra pad moves to the end.
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "strides", &[2, 2]);
    ints(&mut node, "output_shape", &[4, 4]); // spatial-only form
    string(&mut node, "auto_pad", "SAME_UPPER");

    let out = run_both(&ConvTransposeOp, &node, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out[0],
        &[1, 1, 4, 4],
        &[
            1.0, 2.0, 5.0, 4.0, 4.0, 5.0, 14.0, 10.0, 10.0, 14.0, 36.0, 24.0, 12.0, 15.0, 34.0,
            20.0,
        ],
        "convtranspose output_shape SAME_UPPER",
    );
}

#[test]
fn conv_transpose_auto_pad_same_upper_without_output_shape() {
    // SAME_* without output_shape targets out = in * stride = 4.
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "strides", &[2, 2]);
    string(&mut node, "auto_pad", "SAME_UPPER");

    let out = run_both(&ConvTransposeOp, &node, vec![Some(&input), Some(&weight)]);
    assert_close(
        &out[0],
        &[1, 1, 4, 4],
        &[
            1.0, 2.0, 5.0, 4.0, 4.0, 5.0, 14.0, 10.0, 10.0, 14.0, 36.0, 24.0, 12.0, 15.0, 34.0,
            20.0,
        ],
        "convtranspose SAME_UPPER",
    );
}

#[test]
fn conv_transpose_decoder_shape_matches_skip_connection() {
    // The brief's regression: 32x32 → 64x64 with stride 2 and a 3x3 kernel.
    // Without output_shape support this produced 65x65.
    let input = Tensor::new(vec![0.5_f32; 32 * 32], vec![1, 1, 32, 32]);
    let weight = Tensor::new(vec![0.25_f32; 9], vec![1, 1, 3, 3]);
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "strides", &[2, 2]);
    ints(&mut node, "output_shape", &[1, 1, 64, 64]);

    let out = ConvTransposeOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect("execute");
    assert_eq!(out[0].shape, vec![1, 1, 64, 64]);
}

#[test]
fn conv_transpose_output_shape_larger_than_natural_is_typed_error() {
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "strides", &[2, 2]);
    ints(&mut node, "output_shape", &[1, 1, 9, 9]);

    let err = ConvTransposeOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("an oversized output_shape must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_transpose_zero_group_is_typed_error() {
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    int(&mut node, "group", 0);

    let err = ConvTransposeOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("group=0 must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn conv_transpose_pads_exceeding_output_extent_is_typed_error() {
    // pads larger than the un-cropped extent used to underflow to ~usize::MAX.
    let (input, weight) = conv_transpose_fixture();
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "pads", &[8, 8, 8, 8]);

    let err = ConvTransposeOp
        .execute(&ctx(&node, vec![Some(&input), Some(&weight)]))
        .expect_err("oversized pads must be rejected");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}
