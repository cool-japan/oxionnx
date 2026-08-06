//! Wave-2 session-level end-to-end tests for the int8/uint8 quantized operator
//! family and the plain `RNN` operator.
//!
//! Before this wave `quantized/functions.rs` held working kernels but
//! `default_registry()` mapped no `op_type` onto them, so *every* model
//! produced by ONNX Runtime's static quantizer (a quantized ResNet50, a
//! quantized BERT) failed at run time with `UnsupportedOp` — and the same was
//! true of the fully-implemented `rnn::simple_rnn` kernel, which had no `RNN`
//! operator and no `OpKind::RNN` variant.
//!
//! Every expected value below is produced by `onnx.reference`
//! (`onnx` 1.21.0, opset 21) — the ONNX project's own reference
//! implementation — via
//! `/private/tmp/.../scratchpad/ref_ops.py`. Quantized tensors are integer
//! values carried in this engine's f32 lanes, so the integer outputs are
//! compared **exactly**.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── helpers ─────────────────────────────────────────────────────────────────

fn node(op: OpKind, inputs: &[&str], outputs: &[&str], attrs: Attributes) -> Node {
    Node {
        op,
        name: "op0".to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs,
    }
}

/// Build a one-node `Session` where every input is a *graph input*, and run it.
fn run_op(
    op: OpKind,
    node_inputs: &[&str],
    node_outputs: &[&str],
    feeds: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let graph = Graph {
        nodes: vec![node(op, node_inputs, node_outputs, attrs)],
        input_names: feeds.iter().map(|(n, _)| (*n).to_string()).collect(),
        output_names: node_outputs.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let feed: HashMap<&str, Tensor> = feeds.into_iter().collect();
    session.run(&feed).expect("run")
}

fn scalar(v: f32) -> Tensor {
    Tensor::new(vec![v], vec![1])
}

fn ints(values: &[i64]) -> Vec<i64> {
    values.to_vec()
}

#[track_caller]
fn assert_exact(actual: &Tensor, expected: &[f32], shape: &[usize], what: &str) {
    assert_eq!(actual.shape, shape, "{what}: shape");
    assert_eq!(
        actual.data, expected,
        "{what}: values (quantized lanes must match exactly)"
    );
}

#[track_caller]
fn assert_close(actual: &Tensor, expected: &[f32], shape: &[usize], tol: f32, what: &str) {
    assert_eq!(actual.shape, shape, "{what}: shape");
    assert_eq!(actual.data.len(), expected.len(), "{what}: element count");
    for (i, (&a, &e)) in actual.data.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}: element {i}: got {a}, expected {e} (tol {tol})"
        );
    }
}

// ── registry / OpKind wiring ────────────────────────────────────────────────

/// Every operator this wave registers must be reachable by its ONNX name *and*
/// parse to a real `OpKind` (not `Unknown`), which is what a loaded model does.
#[test]
fn quantized_family_and_rnn_are_registered() {
    let registry = oxionnx_ops::default_registry();
    for name in [
        "QLinearConv",
        "QLinearMatMul",
        "MatMulInteger",
        "ConvInteger",
        "DynamicQuantizeLinear",
        "RNN",
    ] {
        assert!(registry.contains(name), "{name} must be registered");
        let kind = OpKind::parse(name);
        assert_ne!(
            kind,
            OpKind::Unknown(name.to_string()),
            "{name} must have its own OpKind variant"
        );
        assert_eq!(kind.as_str(), name, "{name}: as_str must round-trip");
    }
}

// ── MatMulInteger ───────────────────────────────────────────────────────────

/// `onnx.reference` MatMulInteger, uint8 with a non-zero `a_zero_point`.
///
/// The output is raw **int32**: no scales, no 8-bit saturation. The reference
/// values go negative, which is exactly what clamping to `[0, 255]` would
/// destroy.
#[test]
fn matmul_integer_u8_with_zero_point() {
    let a = Tensor::new(
        vec![11., 7., 3., 10., 6., 2., 9., 5., 1., 8., 4., 0.],
        vec![4, 3],
    );
    let b = Tensor::new(vec![1., 4., 2., 5., 3., 6.], vec![3, 2]);
    let out = run_op(
        OpKind::MatMulInteger,
        &["a", "b", "a_zp", "b_zp"],
        &["y"],
        vec![
            ("a", a),
            ("b", b),
            ("a_zp", scalar(12.0)),
            ("b_zp", scalar(0.0)),
        ],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[-38., -83., -44., -98., -50., -113., -56., -128.],
        &[4, 2],
        "MatMulInteger u8",
    );
}

/// int8 inputs with the zero-point inputs omitted entirely (both default to 0).
#[test]
fn matmul_integer_i8_without_zero_points() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = Tensor::new(vec![-1., 2., 3., -4., 5., 6.], vec![3, 2]);
    let out = run_op(
        OpKind::MatMulInteger,
        &["a", "b"],
        &["y"],
        vec![("a", a), ("b", b)],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[20., 12., 41., 24.],
        &[2, 2],
        "MatMulInteger i8",
    );
}

// ── QLinearMatMul ───────────────────────────────────────────────────────────

/// The ONNX node-test values for `QLinearMatMul` (uint8 end to end).
#[test]
fn qlinear_matmul_u8_node_test() {
    let a = Tensor::new(vec![208., 236., 0., 238., 3., 214., 255., 29.], vec![2, 4]);
    let b = Tensor::new(
        vec![
            152., 51., 244., 60., 26., 255., 0., 127., 246., 127., 254., 247.,
        ],
        vec![4, 3],
    );
    let out = run_op(
        OpKind::QLinearMatMul,
        &["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"],
        &["y"],
        vec![
            ("a", a),
            ("as", scalar(0.0066)),
            ("azp", scalar(113.0)),
            ("b", b),
            ("bs", scalar(0.00705)),
            ("bzp", scalar(114.0)),
            ("ys", scalar(0.0107)),
            ("yzp", scalar(118.0)),
        ],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[168., 115., 255., 1., 66., 151.],
        &[2, 3],
        "QLinearMatMul u8",
    );
}

/// Symmetric int8 (`a_zp = b_zp = 0`) with a *negative-capable* output.
///
/// This is the case where a naive "assume uint8" saturation would clamp `-2`
/// and `-9` to `0`; see `SatRange::infer` for the documented cascade.
#[test]
fn qlinear_matmul_i8_symmetric_keeps_negative_outputs() {
    let a = Tensor::new(vec![1., -2., 3., 4., 5., -6.], vec![2, 3]);
    let b = Tensor::new(vec![-1., 2., 3., -4., 5., 6.], vec![3, 2]);
    let out = run_op(
        OpKind::QLinearMatMul,
        &["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"],
        &["y"],
        vec![
            ("a", a),
            ("as", scalar(0.5)),
            ("azp", scalar(0.0)),
            ("b", b),
            ("bs", scalar(0.5)),
            ("bzp", scalar(0.0)),
            ("ys", scalar(1.0)),
            ("yzp", scalar(3.0)),
        ],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[5., 10., -2., -9.],
        &[2, 2],
        "QLinearMatMul i8 symmetric",
    );
}

// ── ConvInteger ─────────────────────────────────────────────────────────────

#[test]
fn conv_integer_with_padding() {
    let x = Tensor::new(vec![2., 3., 4., 5., 6., 7., 8., 9., 10.], vec![1, 1, 3, 3]);
    let w = Tensor::new(vec![1., 1., 1., 1.], vec![1, 1, 2, 2]);

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pads".into(), ints(&[0, 0, 0, 0]));
    let out = run_op(
        OpKind::ConvInteger,
        &["x", "w", "xzp"],
        &["y"],
        vec![("x", x.clone()), ("w", w.clone()), ("xzp", scalar(1.0))],
        attrs,
    );
    assert_exact(
        &out["y"],
        &[12., 16., 24., 28.],
        &[1, 1, 2, 2],
        "ConvInteger pads=0",
    );

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pads".into(), ints(&[1, 1, 1, 1]));
    let out = run_op(
        OpKind::ConvInteger,
        &["x", "w", "xzp"],
        &["y"],
        vec![("x", x), ("w", w), ("xzp", scalar(1.0))],
        attrs,
    );
    assert_exact(
        &out["y"],
        &[
            1., 3., 5., 3., 5., 12., 16., 9., 11., 24., 28., 15., 7., 15., 17., 9.,
        ],
        &[1, 1, 4, 4],
        "ConvInteger pads=1",
    );
}

/// uint8 activations with int8 weights and *both* zero points set — the
/// combination ORT's static quantizer produces most often.
#[test]
fn conv_integer_mixed_sign_zero_points() {
    let x = Tensor::new(vec![2., 3., 4., 5., 6., 7., 8., 9., 10.], vec![1, 1, 3, 3]);
    let w = Tensor::new(vec![1., -1., 2., 3.], vec![1, 1, 2, 2]);
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pads".into(), ints(&[0, 0, 0, 0]));
    let out = run_op(
        OpKind::ConvInteger,
        &["x", "w", "xzp", "wzp"],
        &["y"],
        vec![
            ("x", x),
            ("w", w),
            ("xzp", scalar(3.0)),
            ("wzp", scalar(1.0)),
        ],
        attrs,
    );
    assert_exact(
        &out["y"],
        &[8., 9., 11., 12.],
        &[1, 1, 2, 2],
        "ConvInteger u8 x / i8 w",
    );
}

// ── QLinearConv ─────────────────────────────────────────────────────────────

/// The canonical ONNX node test `test_qlinearconv`: a 7x7 uint8 input, a 1x1
/// uint8 weight with `w_zero_point = 255`, and uint8 output.
#[test]
fn qlinear_conv_node_test() {
    let x = Tensor::new(
        vec![
            255., 174., 162., 25., 203., 168., 58., 15., 59., 237., 95., 129., 0., 64., 56., 242.,
            153., 221., 168., 12., 166., 232., 178., 186., 195., 237., 162., 237., 188., 39., 124.,
            77., 80., 102., 43., 127., 230., 21., 83., 41., 40., 134., 255., 154., 92., 141., 42.,
            148., 247.,
        ],
        vec![1, 1, 7, 7],
    );
    let w = Tensor::new(vec![0.], vec![1, 1, 1, 1]);
    let out = run_op(
        OpKind::QLinearConv,
        &["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"],
        &["y"],
        vec![
            ("x", x),
            ("xs", scalar(0.003_692_047)),
            ("xzp", scalar(132.0)),
            ("w", w),
            ("ws", scalar(0.001_727_945_8)),
            ("wzp", scalar(255.0)),
            ("ys", scalar(0.001_626_812_6)),
            ("yzp", scalar(123.0)),
        ],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[
            0., 81., 93., 230., 52., 87., 197., 240., 196., 18., 160., 126., 255., 191., 199., 13.,
            102., 34., 87., 243., 89., 23., 77., 69., 60., 18., 93., 18., 67., 216., 131., 178.,
            175., 153., 212., 128., 25., 234., 172., 214., 215., 121., 0., 101., 163., 114., 213.,
            107., 8.,
        ],
        &[1, 1, 7, 7],
        "QLinearConv node test",
    );
}

/// Per-output-channel `w_scale` plus an **int32 bias**.
///
/// `B` is defined by the spec to live already in the `x_scale * w_scale`
/// domain and to be added straight into the integer accumulator — not divided
/// by a combined scale, and not a float bias. Channel 0's first output is
/// `(-15 + 50) * (0.02 * 0.5 / 0.05) + 100 = 107`, which only comes out right
/// with that convention.
#[test]
fn qlinear_conv_per_channel_scale_and_int32_bias() {
    let x = Tensor::new((1..=32).map(|v| v as f32).collect(), vec![1, 2, 4, 4]);
    let w = Tensor::new(
        vec![
            1., -2., 3., 0., 2., 1., -1., 1., 0., 1., 2., -3., 1., 1., 1., 1.,
        ],
        vec![2, 2, 2, 2],
    );
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), ints(&[1, 1]));
    attrs.int_lists.insert("pads".into(), ints(&[0, 0, 0, 0]));

    let out = run_op(
        OpKind::QLinearConv,
        &["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp", "b"],
        &["y"],
        vec![
            ("x", x),
            ("xs", scalar(0.02)),
            ("xzp", scalar(16.0)),
            ("w", w),
            ("ws", Tensor::new(vec![0.5, 0.25], vec![2])),
            ("wzp", Tensor::new(vec![0.0, 0.0], vec![2])),
            ("ys", scalar(0.05)),
            ("yzp", scalar(100.0)),
            ("b", Tensor::new(vec![50.0, -60.0], vec![2])),
        ],
        attrs,
    );
    assert_exact(
        &out["y"],
        &[
            107., 108., 109., 111., 112., 113., 115., 116., 117., 95., 95., 96., 96., 97., 97.,
            98., 98., 99.,
        ],
        &[1, 2, 3, 3],
        "QLinearConv per-channel + int32 bias",
    );
}

/// Depthwise (`group == C`) with `auto_pad = SAME_UPPER`.
///
/// The `auto_pad` split must match the float `Conv`'s, and the output includes
/// a value above 127 (`134`) that an int8 saturation guess would clip.
#[test]
fn qlinear_conv_depthwise_same_upper() {
    let x = Tensor::new((1..=36).map(|v| v as f32).collect(), vec![1, 4, 3, 3]);
    let w = Tensor::new(
        vec![
            130., 120., 140., 126., 128., 132., 124., 129., 127., 131., 125., 133., 135., 121.,
            138., 122.,
        ],
        vec![4, 1, 2, 2],
    );
    let mut attrs = Attributes::default();
    attrs.ints.insert("group".into(), 4);
    attrs.strings.insert("auto_pad".into(), "SAME_UPPER".into());
    attrs.int_lists.insert("kernel_shape".into(), ints(&[2, 2]));

    let out = run_op(
        OpKind::QLinearConv,
        &["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"],
        &["y"],
        vec![
            ("x", x),
            ("xs", scalar(0.05)),
            ("xzp", scalar(10.0)),
            ("w", w),
            ("ws", scalar(0.4)),
            ("wzp", scalar(128.0)),
            ("ys", scalar(0.1)),
            ("yzp", scalar(50.0)),
        ],
        attrs,
    );
    assert_exact(
        &out["y"],
        &[
            47., 48., 38., 49., 50., 46., 52., 51., 50., 49., 49., 46., 50., 50., 44., 56., 56.,
            50., 60., 61., 39., 62., 63., 37., 57., 57., 47., 64., 65., 124., 67., 67., 134., 49.,
            49., 86.,
        ],
        &[1, 4, 3, 3],
        "QLinearConv depthwise SAME_UPPER",
    );
}

/// int8 activations, int8 weights, int8 output with a **negative** zero point
/// (which pins the saturation range to `[-128, 127]`), and `dilations = 2`.
#[test]
fn qlinear_conv_int8_with_dilations() {
    let x = Tensor::new(
        vec![
            -5., 3., 7., -2., 0., 9., 4., -8., 6., 1., -3., 2., 8., -6., 5., 0.,
        ],
        vec![1, 1, 4, 4],
    );
    let w = Tensor::new(vec![2., -1., 1., 3.], vec![1, 1, 2, 2]);
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("dilations".into(), ints(&[2, 2]));
    attrs.int_lists.insert("pads".into(), ints(&[0, 0, 0, 0]));

    let out = run_op(
        OpKind::QLinearConv,
        &["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"],
        &["y"],
        vec![
            ("x", x),
            ("xs", scalar(0.5)),
            ("xzp", scalar(0.0)),
            ("w", w),
            ("ws", scalar(0.5)),
            ("wzp", scalar(0.0)),
            ("ys", scalar(0.25)),
            ("yzp", scalar(-5.0)),
        ],
        attrs,
    );
    assert_exact(
        &out["y"],
        &[-25., 10., 14., 15.],
        &[1, 1, 2, 2],
        "QLinearConv i8 dilations=2",
    );
}

/// A zero `y_scale` would divide by zero inside the requantization; it must be
/// a typed error, never a NaN tensor or a panic.
#[test]
fn qlinear_conv_rejects_zero_output_scale() {
    let graph = Graph {
        nodes: vec![node(
            OpKind::QLinearConv,
            &["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"],
            &["y"],
            Attributes::default(),
        )],
        input_names: vec![
            "x".into(),
            "xs".into(),
            "xzp".into(),
            "w".into(),
            "ws".into(),
            "wzp".into(),
            "ys".into(),
            "yzp".into(),
        ],
        output_names: vec!["y".into()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let feed: HashMap<&str, Tensor> = [
        ("x", Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2])),
        ("xs", scalar(0.1)),
        ("xzp", scalar(0.0)),
        ("w", Tensor::new(vec![1.], vec![1, 1, 1, 1])),
        ("ws", scalar(0.1)),
        ("wzp", scalar(0.0)),
        ("ys", scalar(0.0)),
        ("yzp", scalar(0.0)),
    ]
    .into_iter()
    .collect();
    let err = session.run(&feed).expect_err("zero y_scale must fail");
    assert!(
        format!("{err}").contains("scale is zero"),
        "unexpected error: {err}"
    );
}

// ── DynamicQuantizeLinear ───────────────────────────────────────────────────

/// Three outputs, uint8 range, and the two subtle numerics: the zero point is
/// added **after** rounding, and the rounding is ties-to-even.
///
/// `-2.5 / (5/255) = -127.49999…` must round to `-127` (giving `26`), and
/// `0.5 / (5/255) = 25.5` must round to `26` (giving `179`).
///
/// `y_scale` and `y_zero_point` are declared **scalars** in the op spec ("it's a
/// scalar, which means a per-tensor/layer quantization"), so their shape is `[]`
/// — rank 0, not the rank-1 `[1]` the engine emitted before it supported rank-0
/// tensors. Only `y` carries the input's shape.
#[test]
fn dynamic_quantize_linear_matches_reference() {
    let x = Tensor::new(vec![0.0, 2.0, -3.0, -2.5, 1.34, 0.5], vec![6]);
    let out = run_op(
        OpKind::DynamicQuantizeLinear,
        &["x"],
        &["y", "y_scale", "y_zp"],
        vec![("x", x)],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[153., 255., 0., 26., 221., 179.],
        &[6],
        "DynamicQuantizeLinear y",
    );
    // Rank 0 (`&[]`), per the spec note in this test's doc comment.
    assert_close(
        &out["y_scale"],
        &[0.019_607_844],
        &[],
        1e-9,
        "DynamicQuantizeLinear y_scale",
    );
    assert_exact(
        &out["y_zp"],
        &[153.],
        &[],
        "DynamicQuantizeLinear y_zero_point",
    );
}

#[test]
fn dynamic_quantize_linear_2d_preserves_shape() {
    let x = Tensor::new(vec![1.5, 2.5, 3.5, -1.0, -2.0, 0.0], vec![2, 3]);
    let out = run_op(
        OpKind::DynamicQuantizeLinear,
        &["x"],
        &["y", "y_scale", "y_zp"],
        vec![("x", x)],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[163., 209., 255., 47., 0., 93.],
        &[2, 3],
        "DynamicQuantizeLinear 2x3 y",
    );
    // `y` keeps the input's [2,3]; the two quantization parameters stay rank 0
    // regardless of the input rank (`&[]`) — that is what "preserves shape"
    // means here, and it is the property this test is named for.
    assert_close(
        &out["y_scale"],
        &[0.021_568_63],
        &[],
        1e-8,
        "DynamicQuantizeLinear 2x3 y_scale",
    );
    assert_exact(&out["y_zp"], &[93.], &[], "DynamicQuantizeLinear 2x3 y_zp");
}

// ── RNN ─────────────────────────────────────────────────────────────────────

/// The plain ONNX `RNN`: `h_t = tanh(x_t W^T + h_{t-1} R^T + Wb + Rb)`.
///
/// The kernel already existed and was unit-tested; what this pins is that a
/// *model* containing an `RNN` node now reaches it instead of failing with
/// `UnsupportedOp`.
#[test]
fn rnn_forward_matches_reference() {
    let x = Tensor::new(
        vec![
            0.0012, 0.2987, -0.2741, -0.8906, -0.4547, -0.9916, 0.0601, 1.3402, -0.4922, -0.6205,
            0.4898, 0.3569,
        ],
        vec![3, 2, 2],
    );
    let w = Tensor::new(
        vec![0.1054, -0.9305, -0.0293, 0.6953, -1.3442, -0.4576],
        vec![1, 3, 2],
    );
    let r = Tensor::new(
        vec![
            -1.9012, -1.2895, -1.8417, -0.2351, -1.2674, 0.2713, 0.1568, -0.1869, -2.5168,
        ],
        vec![1, 3, 3],
    );
    let b = Tensor::new(
        vec![-0.5387, -0.0485, 0.1133, -1.5301, -0.4778, -0.9785],
        vec![1, 6],
    );

    let mut attrs = Attributes::default();
    attrs.ints.insert("hidden_size".into(), 3);

    let out = run_op(
        OpKind::RNN,
        &["X", "W", "R", "B"],
        &["Y", "Yh"],
        vec![("X", x), ("W", w), ("R", r), ("B", b)],
        attrs,
    );

    assert_close(
        &out["Y"],
        &[
            -0.981_852, -0.308_285, -0.763_059, -0.853_523, -0.813_572, -0.088_980, 0.985_947,
            -0.657_218, 0.965_678, -0.441_287, 0.923_370, -0.866_059, -0.999_666, -0.079_990,
            -0.968_815, -0.802_630, -0.920_759, 0.245_876,
        ],
        &[3, 1, 2, 3],
        2e-5,
        "RNN Y",
    );
    assert_close(
        &out["Yh"],
        &[
            -0.999_666, -0.079_990, -0.968_815, -0.802_630, -0.920_759, 0.245_876,
        ],
        &[1, 2, 3],
        2e-5,
        "RNN Y_h",
    );
}

/// The documented **union** saturation range, exercised deliberately.
///
/// When every zero point sits in `0..=127` neither one can say whether the
/// output element type is `uint8` or `int8` (symmetric int8 and post-`ReLU`
/// uint8 both put it at 0), and this runtime's dtype-erased `Tensor` cannot
/// report the declared type. `SatRange::infer` then clamps to the union
/// `[-128, 255]` instead of guessing — see its docs for why.
///
/// This model is engineered so all three candidate ranges disagree:
///
/// | element | union (ours) | uint8 | int8 |
/// |---------|--------------|-------|------|
/// | `200`   | `200`        | `200` | `127`|
/// | `-50`   | `-50`        | `0`   | `-50`|
///
/// A future change that threads the real element type through (see the
/// `deferred` note on `execute_typed` for `QLinearMatMul`) should replace this
/// test with one asserting the true dtype's answer — it is pinned here so the
/// deviation is a known contract rather than a silent surprise.
#[test]
fn qlinear_matmul_ambiguous_zero_points_use_the_union_range() {
    let a = Tensor::new(vec![10., 0., 0., 10.], vec![2, 2]);
    let b = Tensor::new(vec![10., -15., 0., 0.], vec![2, 2]);
    let out = run_op(
        OpKind::QLinearMatMul,
        &["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"],
        &["y"],
        vec![
            ("a", a),
            ("as", scalar(1.0)),
            ("azp", scalar(0.0)),
            ("b", b),
            ("bs", scalar(1.0)),
            ("bzp", scalar(0.0)),
            ("ys", scalar(1.0)),
            ("yzp", scalar(100.0)),
        ],
        Attributes::default(),
    );
    assert_exact(
        &out["y"],
        &[200., -50., 100., 100.],
        &[2, 2],
        "QLinearMatMul union range",
    );
}

/// A zero-point input whose length matches neither 1 nor the axis it may vary
/// along is a malformed model; reading `0` for the missing lanes would produce
/// a plausible-looking but wrong result.
#[test]
fn matmul_integer_rejects_mis_sized_zero_point() {
    let graph = Graph {
        nodes: vec![node(
            OpKind::MatMulInteger,
            &["a", "b", "a_zp"],
            &["y"],
            Attributes::default(),
        )],
        input_names: vec!["a".into(), "b".into(), "a_zp".into()],
        output_names: vec!["y".into()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let feed: HashMap<&str, Tensor> = [
        ("a", Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3])),
        ("b", Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2])),
        // A has 2 rows, so 3 lanes is neither per-tensor nor per-row.
        ("a_zp", Tensor::new(vec![1., 2., 3.], vec![3])),
    ]
    .into_iter()
    .collect();
    let err = session
        .run(&feed)
        .expect_err("mis-sized zero point must fail");
    assert!(
        format!("{err}").contains("a_zero_point has 3 entries"),
        "unexpected error: {err}"
    );
}
