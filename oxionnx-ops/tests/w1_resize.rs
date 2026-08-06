//! Wave-1 correctness tests for the ONNX `Resize` operator.
//!
//! Every numeric constant below was produced by the ONNX **reference
//! implementation** (`onnx.reference.ReferenceEvaluator`, opset 19) on the
//! stated input, then inlined. Where this engine deliberately differs from that
//! reference the divergence is called out in a comment on the test.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::{OnnxError, Tensor};
use oxionnx_ops::registry::conv_ops::ResizeOp;
use oxionnx_ops::resize::{resize, resize_with, ResizeOptions};

// ── Reference constants ─────────────────────────────────────────────────────

// data = [1,2,3,4] as [1,1,1,4], scales = [1,1,1,2], mode = nearest, half_pixel
const NEAREST_DEFAULT_2X: [f32; 8] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
const NEAREST_FLOOR_2X: [f32; 8] = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
const NEAREST_CEIL_2X: [f32; 8] = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
const NEAREST_PREFER_CEIL_2X: [f32; 8] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];

// data = [1,2,3,4] as [1,1,2,2], scales = [1,1,2,2], asymmetric
#[rustfmt::skip]
const NEAREST_ASYM_PREFER_FLOOR: [f32; 16] = [
    1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0,
    3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
];
#[rustfmt::skip]
const NEAREST_ASYM_PREFER_CEIL: [f32; 16] = [
    1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0,
    3.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0,
];

// data = [1,2,3,4] as [1,1,2,2], scales = [1,1,2,2], half_pixel
#[rustfmt::skip]
const LINEAR_HALF_PIXEL_2X: [f32; 16] = [
    1.0, 1.25, 1.75, 2.0,
    1.5, 1.75, 2.25, 2.5,
    2.5, 2.75, 3.25, 3.5,
    3.0, 3.25, 3.75, 4.0,
];
#[rustfmt::skip]
const CUBIC_HALF_PIXEL_2X: [f32; 16] = [
    0.6835938, 1.015625, 1.5625, 1.894531,
    1.347656, 1.679688, 2.226562, 2.558594,
    2.441406, 2.773438, 3.320312, 3.652344,
    3.105469, 3.4375, 3.984375, 4.316406,
];
#[rustfmt::skip]
const CUBIC_A_HALF: [f32; 16] = [
    0.7890625, 1.0625, 1.65625, 1.929688,
    1.335938, 1.609375, 2.203125, 2.476562,
    2.523438, 2.796875, 3.390625, 3.664062,
    3.070312, 3.34375, 3.9375, 4.210938,
];
#[rustfmt::skip]
const CUBIC_EXCLUDE_OUTSIDE: [f32; 16] = [
    0.5909091, 0.9567248, 1.497821, 1.863636,
    1.322541, 1.688356, 2.229452, 2.595268,
    2.404732, 2.770548, 3.311644, 3.677459,
    3.136364, 3.502179, 4.043275, 4.409091,
];

// data = 1..=20 as [1,1,4,5]
const LINEAR_DOWN_45: [f32; 4] = [4.25, 6.75, 14.25, 16.75];
#[rustfmt::skip]
const CUBIC_ALIGN_CORNERS_45: [f32; 20] = [
    1.0, 2.37037, 3.62963, 5.0,
    4.339844, 5.710214, 6.969473, 8.339844,
    8.5, 9.87037, 11.12963, 12.5,
    12.66016, 14.03053, 15.28979, 16.66016,
    16.0, 17.37037, 18.62963, 20.0,
];
const LINEAR_ANTIALIAS_DOWN_45: [f32; 6] = [3.243636, 5.483636, 9.38, 11.62, 15.51636, 17.75636];
const PYTORCH_HALF_PIXEL_DOWN: [f32; 6] = [3.833333, 5.5, 7.166667, 13.83333, 15.5, 17.16667];
#[rustfmt::skip]
const HALF_PIXEL_SYMMETRIC_45: [f32; 42] = [
    1.0, 1.666667, 2.333333, 3.0, 3.666667, 4.333333, 5.0,
    3.5, 4.166667, 4.833333, 5.5, 6.166667, 6.833333, 7.5,
    6.833333, 7.5, 8.166667, 8.833333, 9.5, 10.16667, 10.83333,
    10.16667, 10.83333, 11.5, 12.16667, 12.83333, 13.5, 14.16667,
    13.5, 14.16667, 14.83333, 15.5, 16.16667, 16.83333, 17.5,
    16.0, 16.66667, 17.33333, 18.0, 18.66667, 19.33333, 20.0,
];
const TFCROP_LINEAR: [f32; 9] = [5.95, 7.05, 8.15, 10.825, 11.925, 13.025, 15.7, 16.8, 17.9];
#[rustfmt::skip]
const TFCROP_EQUAL_SIZE: [f32; 20] = [
    5.75, 6.25, 6.75, 7.25, 7.75,
    8.25, 8.75, 9.25, 9.75, 10.25,
    10.75, 11.25, 11.75, 12.25, 12.75,
    13.25, 13.75, 14.25, 14.75, 15.25,
];
#[rustfmt::skip]
const TFCROP_EXTRAP: [f32; 16] = [
    9.0, 9.0, 9.0, 9.0,
    9.0, 3.816666, 6.083333, 9.0,
    9.0, 13.81667, 16.08333, 9.0,
    9.0, 9.0, 9.0, 9.0,
];
#[rustfmt::skip]
const AXES_NOT_LARGER: [f32; 30] = [
    1.0, 1.75, 2.583333, 3.416667, 4.25, 5.0,
    4.75, 5.5, 6.333333, 7.166667, 8.0, 8.75,
    8.916667, 9.666667, 10.5, 11.33333, 12.16667, 12.91667,
    13.08333, 13.83333, 14.66667, 15.5, 16.33333, 17.08333,
    16.0, 16.75, 17.58333, 18.41667, 19.25, 20.0,
];
#[rustfmt::skip]
const AXES_NOT_SMALLER: [f32; 48] = [
    1.0, 1.5, 2.166667, 2.833333, 3.5, 4.166667, 4.833333, 5.0,
    3.5, 4.0, 4.666667, 5.333333, 6.0, 6.666667, 7.333333, 7.5,
    6.833333, 7.333333, 8.0, 8.666667, 9.333333, 10.0, 10.66667, 10.83333,
    10.16667, 10.66667, 11.33333, 12.0, 12.66667, 13.33333, 14.0, 14.16667,
    13.5, 14.0, 14.66667, 15.33333, 16.0, 16.66667, 17.33333, 17.5,
    16.0, 16.5, 17.16667, 17.83333, 18.5, 19.16667, 19.83333, 20.0,
];

// data = 0..8 as [1,1,2,2,2], scales = [1,1,2,2,2]
#[rustfmt::skip]
const TRILINEAR_2X: [f32; 64] = [
    0.0, 0.25, 0.75, 1.0, 0.5, 0.75, 1.25, 1.5,
    1.5, 1.75, 2.25, 2.5, 2.0, 2.25, 2.75, 3.0,
    1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5,
    2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
    3.0, 3.25, 3.75, 4.0, 3.5, 3.75, 4.25, 4.5,
    4.5, 4.75, 5.25, 5.5, 5.0, 5.25, 5.75, 6.0,
    4.0, 4.25, 4.75, 5.0, 4.5, 4.75, 5.25, 5.5,
    5.5, 5.75, 6.25, 6.5, 6.0, 6.25, 6.75, 7.0,
];
#[rustfmt::skip]
const TRINEAREST_2X: [f32; 64] = [
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0,
    4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0,
    6.0, 6.0, 7.0, 7.0, 6.0, 6.0, 7.0, 7.0,
    4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0,
    6.0, 6.0, 7.0, 7.0, 6.0, 6.0, 7.0, 7.0,
];

// ── Helpers ─────────────────────────────────────────────────────────────────

fn d14() -> Tensor {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4])
}
fn d22() -> Tensor {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])
}
fn d45() -> Tensor {
    Tensor::new((1..=20).map(|v| v as f32).collect(), vec![1, 1, 4, 5])
}

fn assert_close(got: &Tensor, expect: &[f32], shape: &[usize], label: &str) {
    assert_eq!(got.shape, shape, "{label}: shape");
    assert_eq!(got.data.len(), expect.len(), "{label}: length");
    for (i, (&a, &b)) in got.data.iter().zip(expect.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
            "{label}[{i}]: got {a}, expected {b}",
        );
    }
}

fn opts_of<'a>(mode: &'a str, ctm: &'a str) -> ResizeOptions<'a> {
    ResizeOptions {
        mode,
        coordinate_transformation_mode: ctm,
        ..ResizeOptions::default()
    }
}

fn node_with(attrs: Attributes) -> Node {
    Node {
        name: "resize".into(),
        op: OpKind::Resize,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs,
    }
}

fn ctx_of<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn empty_tensor() -> Tensor {
    Tensor::new(Vec::new(), vec![0])
}

// ── a0-6 / a1-7: nearest_mode is honoured ───────────────────────────────────

#[test]
fn nearest_default_mode_is_round_prefer_floor() {
    // The exact failing case from the audit: half_pixel 2x nearest upsample of
    // [1,2,3,4].  floor() shifts the whole feature map by half a pixel.
    let out = resize_with(
        &d14(),
        Some(&[1.0, 1.0, 1.0, 2.0]),
        None,
        &ResizeOptions::default(),
    )
    .expect("resize");
    assert_close(&out, &NEAREST_DEFAULT_2X, &[1, 1, 1, 8], "nearest default");
    assert_ne!(
        out.data.as_slice(),
        NEAREST_FLOOR_2X.as_slice(),
        "default must NOT be the old floor() behaviour",
    );
}

#[test]
fn nearest_mode_variants() {
    for (mode, expect) in [
        ("round_prefer_floor", NEAREST_DEFAULT_2X),
        ("round_prefer_ceil", NEAREST_PREFER_CEIL_2X),
        ("floor", NEAREST_FLOOR_2X),
        ("ceil", NEAREST_CEIL_2X),
    ] {
        let opts = ResizeOptions {
            nearest_mode: mode,
            ..ResizeOptions::default()
        };
        let out = resize_with(&d14(), Some(&[1.0, 1.0, 1.0, 2.0]), None, &opts).expect("resize");
        assert_close(&out, &expect, &[1, 1, 1, 8], mode);
    }
}

#[test]
fn nearest_asymmetric_tie_breaking() {
    // asymmetric x2 puts coordinates exactly on .5, which is where
    // round_prefer_floor and round_prefer_ceil disagree.
    let floor_opts = ResizeOptions {
        coordinate_transformation_mode: "asymmetric",
        ..ResizeOptions::default()
    };
    let out = resize_with(&d22(), Some(&[1.0, 1.0, 2.0, 2.0]), None, &floor_opts).expect("resize");
    assert_close(
        &out,
        &NEAREST_ASYM_PREFER_FLOOR,
        &[1, 1, 4, 4],
        "asym prefer_floor",
    );

    let ceil_opts = ResizeOptions {
        coordinate_transformation_mode: "asymmetric",
        nearest_mode: "round_prefer_ceil",
        ..ResizeOptions::default()
    };
    let out = resize_with(&d22(), Some(&[1.0, 1.0, 2.0, 2.0]), None, &ceil_opts).expect("resize");
    assert_close(
        &out,
        &NEAREST_ASYM_PREFER_CEIL,
        &[1, 1, 4, 4],
        "asym prefer_ceil",
    );
}

#[test]
fn unsupported_nearest_mode_is_typed_error() {
    let opts = ResizeOptions {
        nearest_mode: "round_to_taste",
        ..ResizeOptions::default()
    };
    let err = resize_with(&d22(), Some(&[1.0, 1.0, 2.0, 2.0]), None, &opts)
        .expect_err("unknown nearest_mode must error");
    assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
}

// ── a0-7 / a1-8 / a5-5 / a11-4: cubic is a real kernel ──────────────────────

#[test]
fn cubic_matches_reference() {
    let out = resize(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        None,
        "cubic",
        "half_pixel",
    )
    .expect("resize");
    assert_close(&out, &CUBIC_HALF_PIXEL_2X, &[1, 1, 4, 4], "cubic");

    // Cubic must not be nearest-neighbour in disguise.
    let nearest = resize(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        None,
        "nearest",
        "half_pixel",
    )
    .expect("resize");
    assert_ne!(out.data, nearest.data, "cubic must differ from nearest");
}

#[test]
fn cubic_coeff_a_is_honoured() {
    let opts = ResizeOptions {
        mode: "cubic",
        cubic_coeff_a: -0.5,
        ..ResizeOptions::default()
    };
    let out = resize_with(&d22(), Some(&[1.0, 1.0, 2.0, 2.0]), None, &opts).expect("resize");
    assert_close(&out, &CUBIC_A_HALF, &[1, 1, 4, 4], "cubic a=-0.5");
}

#[test]
fn cubic_exclude_outside_renormalises() {
    let opts = ResizeOptions {
        mode: "cubic",
        exclude_outside: true,
        ..ResizeOptions::default()
    };
    let out = resize_with(&d22(), Some(&[1.0, 1.0, 2.0, 2.0]), None, &opts).expect("resize");
    assert_close(
        &out,
        &CUBIC_EXCLUDE_OUTSIDE,
        &[1, 1, 4, 4],
        "cubic exclude_outside",
    );
}

#[test]
fn cubic_align_corners_matches_reference() {
    let out = resize(&d45(), None, Some(&[1, 1, 5, 4]), "cubic", "align_corners").expect("resize");
    assert_close(
        &out,
        &CUBIC_ALIGN_CORNERS_45,
        &[1, 1, 5, 4],
        "cubic align_corners",
    );
}

#[test]
fn unsupported_mode_is_typed_error() {
    for mode in ["quintic", "lanczos", "area"] {
        let err = resize(
            &d22(),
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            mode,
            "half_pixel",
        )
        .expect_err("unknown mode must error");
        assert!(matches!(err, OnnxError::Unsupported(_)), "{mode}: {err:?}");
    }
}

// ── Linear: border handling and N-D coverage ────────────────────────────────

#[test]
fn linear_half_pixel_border_is_clamped_after_flooring() {
    // Regression: the old code clamped the source coordinate to 0 *before*
    // taking its fractional part, so out[0] became 0.25*d0 + 0.75*d1 = 1.75
    // instead of the correct 1.0.
    let out = resize(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        None,
        "linear",
        "half_pixel",
    )
    .expect("resize");
    assert_close(&out, &LINEAR_HALF_PIXEL_2X, &[1, 1, 4, 4], "linear border");
    assert!((out.data[0] - 1.0).abs() < 1e-6, "corner must stay 1.0");
}

#[test]
fn linear_downscale_matches_reference() {
    let out = resize(&d45(), None, Some(&[1, 1, 2, 2]), "linear", "half_pixel").expect("resize");
    assert_close(&out, &LINEAR_DOWN_45, &[1, 1, 2, 2], "linear down");
}

#[test]
fn pytorch_half_pixel_downscale() {
    let out = resize(
        &d45(),
        None,
        Some(&[1, 1, 2, 3]),
        "linear",
        "pytorch_half_pixel",
    )
    .expect("resize");
    assert_close(
        &out,
        &PYTORCH_HALF_PIXEL_DOWN,
        &[1, 1, 2, 3],
        "pytorch_half_pixel",
    );
}

#[test]
fn half_pixel_symmetric_uses_fractional_output_width() {
    let out = resize(
        &d45(),
        Some(&[1.0, 1.0, 1.5, 1.5]),
        None,
        "linear",
        "half_pixel_symmetric",
    )
    .expect("resize");
    assert_close(
        &out,
        &HALF_PIXEL_SYMMETRIC_45,
        &[1, 1, 6, 7],
        "half_pixel_symmetric",
    );
}

#[test]
fn five_d_linear_interpolates_the_depth_axis() {
    // Regression: linear used to interpolate only the last two dims and fall
    // back to nearest on the depth axis of a 5-D tensor.
    let input = Tensor::new((0..8).map(|v| v as f32).collect(), vec![1, 1, 2, 2, 2]);
    let out = resize(
        &input,
        Some(&[1.0, 1.0, 2.0, 2.0, 2.0]),
        None,
        "linear",
        "half_pixel",
    )
    .expect("resize");
    assert_close(&out, &TRILINEAR_2X, &[1, 1, 4, 4, 4], "trilinear");
    assert_ne!(
        out.data.as_slice(),
        TRINEAREST_2X.as_slice(),
        "trilinear must not equal nearest",
    );
}

#[test]
fn five_d_nearest_matches_reference() {
    let input = Tensor::new((0..8).map(|v| v as f32).collect(), vec![1, 1, 2, 2, 2]);
    let out = resize(
        &input,
        Some(&[1.0, 1.0, 2.0, 2.0, 2.0]),
        None,
        "nearest",
        "half_pixel",
    )
    .expect("resize");
    assert_close(&out, &TRINEAREST_2X, &[1, 1, 4, 4, 4], "5d nearest");
}

// ── antialias ───────────────────────────────────────────────────────────────

#[test]
fn antialias_downscale_matches_reference() {
    let opts = ResizeOptions {
        mode: "linear",
        antialias: true,
        ..ResizeOptions::default()
    };
    let out = resize_with(&d45(), None, Some(&[1, 1, 3, 2]), &opts).expect("resize");
    assert_close(
        &out,
        &LINEAR_ANTIALIAS_DOWN_45,
        &[1, 1, 3, 2],
        "linear antialias",
    );

    // Without antialias the same downscale must give a different answer.
    let plain = resize(&d45(), None, Some(&[1, 1, 3, 2]), "linear", "half_pixel").expect("resize");
    assert_ne!(out.data, plain.data, "antialias must change the result");
}

#[test]
fn antialias_upscale_degenerates_to_plain_filter() {
    for mode in ["linear", "cubic"] {
        let aa = ResizeOptions {
            mode,
            antialias: true,
            ..ResizeOptions::default()
        };
        let with_aa = resize_with(&d22(), Some(&[1.0, 1.0, 2.0, 2.0]), None, &aa).expect("resize");
        let plain = resize(
            &d22(),
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            mode,
            "half_pixel",
        )
        .expect("resize");
        assert_close(&with_aa, &plain.data, &plain.shape, mode);
    }
}

#[test]
fn antialias_rejected_for_nearest() {
    let opts = ResizeOptions {
        mode: "nearest",
        antialias: true,
        ..ResizeOptions::default()
    };
    let err = resize_with(&d45(), None, Some(&[1, 1, 2, 2]), &opts)
        .expect_err("antialias+nearest must error");
    assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
}

// ── a11-21: tf_crop_and_resize and coordinate mode validation ───────────────

#[test]
fn tf_crop_and_resize_matches_reference() {
    let roi = [0.0f32, 0.0, 0.25, 0.3, 1.0, 1.0, 0.9, 0.85];
    let opts = ResizeOptions {
        mode: "linear",
        coordinate_transformation_mode: "tf_crop_and_resize",
        roi: Some(&roi),
        ..ResizeOptions::default()
    };
    let out = resize_with(&d45(), None, Some(&[1, 1, 3, 3]), &opts).expect("resize");
    assert_close(&out, &TFCROP_LINEAR, &[1, 1, 3, 3], "tf_crop_and_resize");
}

#[test]
fn tf_crop_and_resize_extrapolation_value() {
    let roi = [0.0f32, 0.0, -0.55, -0.3, 1.0, 1.0, 1.45, 1.4];
    let opts = ResizeOptions {
        mode: "linear",
        coordinate_transformation_mode: "tf_crop_and_resize",
        extrapolation_value: 9.0,
        roi: Some(&roi),
        ..ResizeOptions::default()
    };
    let out = resize_with(&d45(), None, Some(&[1, 1, 4, 4]), &opts).expect("resize");
    assert_close(&out, &TFCROP_EXTRAP, &[1, 1, 4, 4], "tf_crop extrapolation");
}

#[test]
fn tf_crop_and_resize_at_equal_size_is_not_skipped_as_identity() {
    // out spatial == in spatial, but the roi still crops: the axis pass must run.
    let roi = [0.0f32, 0.0, 0.25, 0.25, 1.0, 1.0, 0.75, 0.75];
    let opts = ResizeOptions {
        mode: "linear",
        coordinate_transformation_mode: "tf_crop_and_resize",
        roi: Some(&roi),
        ..ResizeOptions::default()
    };
    let out = resize_with(&d45(), None, Some(&[1, 1, 4, 5]), &opts).expect("resize");
    assert_close(
        &out,
        &TFCROP_EQUAL_SIZE,
        &[1, 1, 4, 5],
        "tf_crop equal size",
    );
    assert_ne!(out.data, d45().data, "the crop must actually be applied");
}

#[test]
fn tf_crop_and_resize_requires_roi() {
    let opts = ResizeOptions {
        mode: "linear",
        coordinate_transformation_mode: "tf_crop_and_resize",
        ..ResizeOptions::default()
    };
    let err = resize_with(&d45(), None, Some(&[1, 1, 3, 3]), &opts)
        .expect_err("tf_crop_and_resize without roi must error");
    assert!(matches!(err, OnnxError::InvalidModel(_)), "got {err:?}");

    let short = [0.0f32, 0.0, 1.0, 1.0];
    let opts = ResizeOptions {
        roi: Some(&short),
        ..opts
    };
    let err =
        resize_with(&d45(), None, Some(&[1, 1, 3, 3]), &opts).expect_err("short roi must error");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}

#[test]
fn unsupported_coordinate_transformation_mode_is_typed_error() {
    let err = resize(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        None,
        "nearest",
        "magic_pixel",
    )
    .expect_err("unknown ctm must error");
    assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
}

#[test]
fn tf_half_pixel_for_nn_is_implemented() {
    // Legacy opset-10 mode: x_original = (x_resized + 0.5) / scale.
    // 1x1x1x4 [1,2,3,4] at scale 2 -> 0.25, 0.75, 1.25, 1.75, 2.25, ...
    // round_prefer_floor -> 0,1,1,2,2,3,3,4(clamped 3) -> [1,2,2,3,3,4,4,4]
    let out = resize(
        &d14(),
        Some(&[1.0, 1.0, 1.0, 2.0]),
        None,
        "nearest",
        "tf_half_pixel_for_nn",
    )
    .expect("resize");
    assert_close(
        &out,
        &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        &[1, 1, 1, 8],
        "tf_half_pixel_for_nn",
    );
}

// ── a0-8: output shape uses floor(input * scale) ────────────────────────────

#[test]
fn output_shape_uses_floor_not_round() {
    // 4x5 spatial.  round() would give 3 for 5*0.5 and 7 for 5*1.3.
    for (scale, expect) in [
        (0.5f32, [2usize, 2]),
        (0.6, [2, 3]),
        (1.3, [5, 6]),
        (2.7, [10, 13]),
    ] {
        let out = resize(
            &d45(),
            Some(&[1.0, 1.0, scale, scale]),
            None,
            "nearest",
            "half_pixel",
        )
        .expect("resize");
        assert_eq!(
            out.shape,
            vec![1, 1, expect[0], expect[1]],
            "scale {scale}: floor(dim * scale)",
        );
        assert_eq!(out.data.len(), expect[0] * expect[1]);
    }
}

// ── axes / keep_aspect_ratio_policy ─────────────────────────────────────────

#[test]
fn axes_with_keep_aspect_ratio_policy() {
    for (policy, expect, shape) in [
        ("not_larger", AXES_NOT_LARGER.as_slice(), [1usize, 1, 5, 6]),
        ("not_smaller", AXES_NOT_SMALLER.as_slice(), [1, 1, 6, 8]),
    ] {
        let axes = [2i64, 3];
        let opts = ResizeOptions {
            mode: "linear",
            keep_aspect_ratio_policy: policy,
            axes: Some(&axes),
            ..ResizeOptions::default()
        };
        let out = resize_with(&d45(), None, Some(&[6, 6]), &opts).expect("resize");
        assert_close(&out, expect, &shape, policy);
    }
}

#[test]
fn negative_axes_match_positive_axes() {
    let pos = [2i64, 3];
    let neg = [-2i64, -1];
    let mk = |axes: &[i64]| {
        let opts = ResizeOptions {
            mode: "cubic",
            axes: Some(axes),
            ..ResizeOptions::default()
        };
        resize_with(&d45(), Some(&[2.0, 3.0]), None, &opts).expect("resize")
    };
    let a = mk(&pos);
    let b = mk(&neg);
    assert_eq!(a.shape, vec![1, 1, 8, 15]);
    assert_eq!(a.data, b.data, "negative axes must alias positive axes");
}

#[test]
fn axes_leave_unlisted_dimensions_alone() {
    let axes = [3i64];
    let opts = ResizeOptions {
        mode: "linear",
        axes: Some(&axes),
        ..ResizeOptions::default()
    };
    let out = resize_with(&d45(), Some(&[2.0]), None, &opts).expect("resize");
    assert_eq!(out.shape, vec![1, 1, 4, 10]);
}

#[test]
fn invalid_axes_are_rejected() {
    let bad = [9i64];
    let opts = ResizeOptions {
        axes: Some(&bad),
        ..ResizeOptions::default()
    };
    let err = resize_with(&d45(), Some(&[2.0]), None, &opts).expect_err("axis 9 out of range");
    assert!(matches!(err, OnnxError::InvalidModel(_)), "got {err:?}");

    let dup = [2i64, 2];
    let opts = ResizeOptions {
        axes: Some(&dup),
        ..ResizeOptions::default()
    };
    let err = resize_with(&d45(), Some(&[2.0, 2.0]), None, &opts).expect_err("duplicate axis");
    assert!(matches!(err, OnnxError::InvalidModel(_)), "got {err:?}");
}

#[test]
fn unsupported_keep_aspect_ratio_policy_is_typed_error() {
    let opts = ResizeOptions {
        keep_aspect_ratio_policy: "squish",
        ..ResizeOptions::default()
    };
    let err = resize_with(&d45(), None, Some(&[1, 1, 2, 2]), &opts).expect_err("bad policy");
    assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
}

// ── a10-17: malformed / degenerate inputs never panic ───────────────────────

#[test]
fn zero_length_axis_does_not_underflow() {
    for mode in ["nearest", "linear", "cubic"] {
        let input = Tensor::new(Vec::new(), vec![1, 1, 0, 4]);
        let out = resize(
            &input,
            Some(&[1.0, 1.0, 1.0, 2.0]),
            None,
            mode,
            "half_pixel",
        )
        .unwrap_or_else(|e| panic!("{mode}: zero-length axis must not fail: {e:?}"));
        assert_eq!(out.shape, vec![1, 1, 0, 8], "{mode}");
        assert!(out.data.is_empty(), "{mode}");
    }
}

#[test]
fn zero_length_axis_via_sizes_is_typed_error_when_growing() {
    let input = Tensor::new(Vec::new(), vec![1, 1, 0, 4]);
    let err = resize(&input, None, Some(&[1, 1, 3, 4]), "nearest", "half_pixel")
        .expect_err("cannot grow a zero-length axis");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}

#[test]
fn scales_and_sizes_are_mutually_exclusive() {
    let err = resize(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        Some(&[1, 1, 4, 4]),
        "nearest",
        "half_pixel",
    )
    .expect_err("both scales and sizes must error");
    assert!(matches!(err, OnnxError::InvalidModel(_)), "got {err:?}");

    let err = resize(&d22(), None, None, "nearest", "half_pixel")
        .expect_err("neither scales nor sizes must error");
    assert!(matches!(err, OnnxError::InvalidModel(_)), "got {err:?}");
}

#[test]
fn invalid_scales_are_rejected() {
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let err = resize(
            &d22(),
            Some(&[1.0, 1.0, bad, 2.0]),
            None,
            "nearest",
            "half_pixel",
        )
        .expect_err("bad scale must error");
        assert!(matches!(err, OnnxError::InvalidModel(_)), "{bad}: {err:?}");
    }
}

#[test]
fn wrong_length_scales_are_rejected() {
    let err = resize(&d22(), Some(&[2.0, 2.0]), None, "nearest", "half_pixel")
        .expect_err("scales length must match rank");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}

#[test]
fn empty_output_is_produced_without_panicking() {
    let out = resize(&d45(), None, Some(&[1, 1, 0, 3]), "cubic", "half_pixel").expect("resize");
    assert_eq!(out.shape, vec![1, 1, 0, 3]);
    assert!(out.data.is_empty());
}

// ── ResizeOp: both dispatch paths agree ─────────────────────────────────────

fn run_op_both_ways(attrs: Attributes, inputs: &[Option<&Tensor>]) -> (Tensor, Tensor) {
    let node = node_with(attrs);
    let ctx = ctx_of(&node, inputs.to_vec());
    let direct = ResizeOp.execute(&ctx).expect("execute");
    let mut slots = vec![Tensor::new(vec![0.0; 3], vec![3])];
    ResizeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots");
    (
        direct.into_iter().next().expect("one output"),
        slots.remove(0),
    )
}

fn attrs_with(pairs: &[(&str, &str)], ints: &[(&str, i64)], floats: &[(&str, f32)]) -> Attributes {
    let mut a = Attributes::default();
    for (k, v) in pairs {
        a.strings.insert((*k).into(), (*v).into());
    }
    for (k, v) in ints {
        a.ints.insert((*k).into(), *v);
    }
    for (k, v) in floats {
        a.floats.insert((*k).into(), *v);
    }
    a
}

#[test]
fn op_execute_and_slots_agree() {
    let input = d45();
    let empty = empty_tensor();
    let sizes = Tensor::new(vec![1.0, 1.0, 7.0, 3.0], vec![4]);
    let scales = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]);

    let cases: Vec<(Attributes, Vec<Option<&Tensor>>)> = vec![
        (
            attrs_with(&[("mode", "cubic")], &[], &[]),
            vec![Some(&input), Some(&empty), Some(&empty), Some(&sizes)],
        ),
        (
            attrs_with(&[("mode", "linear")], &[("antialias", 1)], &[]),
            vec![Some(&input), Some(&empty), Some(&empty), Some(&sizes)],
        ),
        (
            attrs_with(&[("mode", "nearest"), ("nearest_mode", "ceil")], &[], &[]),
            vec![Some(&input), Some(&empty), Some(&scales), None],
        ),
        (
            attrs_with(
                &[("mode", "cubic")],
                &[("exclude_outside", 1)],
                &[("cubic_coeff_a", -0.5)],
            ),
            vec![Some(&input), Some(&empty), Some(&scales), None],
        ),
    ];

    for (i, (attrs, inputs)) in cases.into_iter().enumerate() {
        let (direct, slot) = run_op_both_ways(attrs, &inputs);
        assert_eq!(direct.shape, slot.shape, "case {i}: shape");
        assert_eq!(direct.data, slot.data, "case {i}: data");
    }
}

#[test]
fn op_reads_nearest_mode_and_cubic_attributes() {
    let input = d14();
    let empty = empty_tensor();
    let scales = Tensor::new(vec![1.0, 1.0, 1.0, 2.0], vec![4]);
    let inputs = vec![Some(&input), Some(&empty), Some(&scales), None];

    let node = node_with(attrs_with(&[("mode", "nearest")], &[], &[]));
    let ctx = ctx_of(&node, inputs.clone());
    let out = ResizeOp.execute(&ctx).expect("execute");
    assert_close(&out[0], &NEAREST_DEFAULT_2X, &[1, 1, 1, 8], "op default");

    let node = node_with(attrs_with(
        &[("mode", "nearest"), ("nearest_mode", "floor")],
        &[],
        &[],
    ));
    let ctx = ctx_of(&node, inputs);
    let out = ResizeOp.execute(&ctx).expect("execute");
    assert_close(&out[0], &NEAREST_FLOOR_2X, &[1, 1, 1, 8], "op floor");
}

#[test]
fn op_rejects_unknown_mode() {
    let input = d45();
    let empty = empty_tensor();
    let scales = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]);
    let node = node_with(attrs_with(&[("mode", "cubic_spline")], &[], &[]));
    let ctx = ctx_of(&node, vec![Some(&input), Some(&empty), Some(&scales), None]);
    let err = ResizeOp.execute(&ctx).expect_err("unknown mode must error");
    assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
}

#[test]
fn op_slot_is_reused_across_changing_shapes() {
    let input = d45();
    let empty = empty_tensor();
    let big = Tensor::new(vec![1.0, 1.0, 8.0, 10.0], vec![4]);
    let small = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]);
    let node = node_with(attrs_with(&[("mode", "linear")], &[], &[]));

    let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
    let ctx = ctx_of(
        &node,
        vec![Some(&input), Some(&empty), Some(&empty), Some(&big)],
    );
    ResizeOp.execute_into_slots(&ctx, &mut slots).expect("big");
    assert_eq!(slots[0].shape, vec![1, 1, 8, 10]);
    assert_eq!(slots[0].data.len(), 80);

    let ctx = ctx_of(
        &node,
        vec![Some(&input), Some(&empty), Some(&empty), Some(&small)],
    );
    ResizeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("small");
    assert_eq!(slots[0].shape, vec![1, 1, 2, 2]);
    assert_close(&slots[0], &LINEAR_DOWN_45, &[1, 1, 2, 2], "reused slot");
}

#[test]
fn op_slot_of_the_right_length_is_fully_overwritten() {
    // When the slot already has the correct length the buffer is *not* cleared,
    // so correctness depends on every output element being written. Seed it with
    // NaN and require a bit-identical second run.
    let input = d45();
    let empty = empty_tensor();
    let sizes = Tensor::new(vec![1.0, 1.0, 7.0, 3.0], vec![4]);

    for (mode, ints) in [
        ("nearest", vec![]),
        ("linear", vec![]),
        ("cubic", vec![("exclude_outside", 1i64)]),
        ("linear", vec![("antialias", 1i64)]),
    ] {
        let node = node_with(attrs_with(&[("mode", mode)], &ints, &[]));
        let ctx = ctx_of(
            &node,
            vec![Some(&input), Some(&empty), Some(&empty), Some(&sizes)],
        );
        let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
        ResizeOp
            .execute_into_slots(&ctx, &mut slots)
            .expect("first run");
        let first = slots[0].clone();
        assert_eq!(first.shape, vec![1, 1, 7, 3]);

        slots[0].data.fill(f32::NAN);
        ResizeOp
            .execute_into_slots(&ctx, &mut slots)
            .expect("second run");
        assert_eq!(slots[0].shape, first.shape, "{mode}: shape");
        assert_eq!(
            slots[0].data, first.data,
            "{mode}: stale slot contents leaked into the output",
        );
        assert!(
            slots[0].data.iter().all(|v| v.is_finite()),
            "{mode}: NaN survived",
        );
    }
}

#[test]
fn op_rejects_negative_sizes() {
    let input = d45();
    let empty = empty_tensor();
    let sizes = Tensor::new(vec![1.0, 1.0, -4.0, 4.0], vec![4]);
    let node = node_with(attrs_with(&[("mode", "nearest")], &[], &[]));
    let ctx = ctx_of(
        &node,
        vec![Some(&input), Some(&empty), Some(&empty), Some(&sizes)],
    );
    let err = ResizeOp
        .execute(&ctx)
        .expect_err("negative size must error");
    assert!(matches!(err, OnnxError::InvalidModel(_)), "got {err:?}");
}

#[test]
fn op_accepts_the_opset10_two_input_layout() {
    // Resize-10 is (X, scales); Resize-11+ inserts roi at index 1.
    let input = d14();
    let scales = Tensor::new(vec![1.0, 1.0, 1.0, 2.0], vec![4]);
    let node = node_with(attrs_with(&[("mode", "nearest")], &[], &[]));
    let ctx = ctx_of(&node, vec![Some(&input), Some(&scales)]);
    let out = ResizeOp.execute(&ctx).expect("opset-10 layout");
    assert_close(&out[0], &NEAREST_DEFAULT_2X, &[1, 1, 1, 8], "opset-10");
}

#[test]
fn op_honours_axes_attribute() {
    let input = d45();
    let empty = empty_tensor();
    let scales = Tensor::new(vec![2.0, 2.0], vec![2]);
    let mut attrs = attrs_with(&[("mode", "linear")], &[], &[]);
    attrs.int_lists.insert("axes".into(), vec![2i64, 3]);
    let node = node_with(attrs);
    let ctx = ctx_of(&node, vec![Some(&input), Some(&empty), Some(&scales), None]);
    let out = ResizeOp.execute(&ctx).expect("axes");
    assert_eq!(out[0].shape, vec![1, 1, 8, 10]);
}

#[test]
fn op_rejects_wrong_slot_count() {
    let input = d45();
    let empty = empty_tensor();
    let scales = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]);
    let node = node_with(attrs_with(&[("mode", "linear")], &[], &[]));
    let ctx = ctx_of(&node, vec![Some(&input), Some(&empty), Some(&scales), None]);
    let mut slots = vec![Tensor::zeros(&[1]), Tensor::zeros(&[1])];
    let err = ResizeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect_err("two slots must error");
    assert!(matches!(err, OnnxError::Internal(_)), "got {err:?}");
}

// ── Identity paths ──────────────────────────────────────────────────────────

#[test]
fn scale_one_is_an_exact_copy() {
    for mode in ["nearest", "linear", "cubic"] {
        let out = resize(
            &d45(),
            Some(&[1.0, 1.0, 1.0, 1.0]),
            None,
            mode,
            "half_pixel",
        )
        .expect("resize");
        assert_eq!(out.shape, d45().shape, "{mode}");
        assert_eq!(out.data, d45().data, "{mode}: identity must be exact");
    }
}

#[test]
fn exporter_alias_modes_are_accepted() {
    // "bilinear" / "bicubic" aliases some exporters emit must not error.
    let bil = resize_with(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        None,
        &opts_of("bilinear", "half_pixel"),
    )
    .expect("bilinear");
    assert_close(&bil, &LINEAR_HALF_PIXEL_2X, &[1, 1, 4, 4], "bilinear alias");

    let bic = resize_with(
        &d22(),
        Some(&[1.0, 1.0, 2.0, 2.0]),
        None,
        &opts_of("bicubic", "half_pixel"),
    )
    .expect("bicubic");
    assert_close(&bic, &CUBIC_HALF_PIXEL_2X, &[1, 1, 4, 4], "bicubic alias");
}

// ── Hostile inputs must not panic ───────────────────────────────────────────

#[test]
fn extreme_roi_values_do_not_overflow() {
    // A crop box far outside the tensor drives the transformed coordinate to
    // ~ -1e38 / +1e38.  Casting that straight to i64 saturates, and the tap
    // arithmetic (`base - 1`, `base + offset`) would then overflow.
    for (start, end) in [
        (-1e37f32, 1e37f32),
        (-3.0e38, 3.0e38),
        (f32::MIN, f32::MAX),
        (1e30, -1e30),
    ] {
        let roi = [0.0f32, 0.0, start, start, 1.0, 1.0, end, end];
        for mode in ["nearest", "linear", "cubic"] {
            for exclude_outside in [false, true] {
                let opts = ResizeOptions {
                    mode,
                    coordinate_transformation_mode: "tf_crop_and_resize",
                    extrapolation_value: -1.0,
                    exclude_outside,
                    roi: Some(&roi),
                    ..ResizeOptions::default()
                };
                // Either a typed error (a roi span that overflows f32 makes the
                // transformed coordinate NaN) or a finite tensor — never a panic.
                match resize_with(&d45(), None, Some(&[1, 1, 3, 3]), &opts) {
                    Ok(out) => {
                        assert_eq!(out.shape, vec![1, 1, 3, 3]);
                        for v in &out.data {
                            assert!(v.is_finite(), "{mode}/{start}: produced {v}");
                        }
                    }
                    Err(e) => assert!(
                        matches!(e, OnnxError::Arithmetic(_) | OnnxError::InvalidModel(_)),
                        "{mode}/{start}: unexpected error {e:?}",
                    ),
                }
            }
        }
    }
}

#[test]
fn extreme_scales_do_not_panic() {
    for scale in [1e-30f32, 1e30, f32::MIN_POSITIVE, f32::MAX] {
        for mode in ["nearest", "linear", "cubic"] {
            // Either a typed error or a valid tensor — never a panic.
            let result = resize(
                &d45(),
                Some(&[1.0, 1.0, 1.0, scale]),
                None,
                mode,
                "half_pixel",
            );
            if let Ok(out) = result {
                assert_eq!(out.data.len(), out.shape.iter().product::<usize>());
            }
        }
    }
}
