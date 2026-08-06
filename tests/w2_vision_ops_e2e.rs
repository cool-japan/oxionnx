//! Wave-2 session-level end-to-end tests for the classic CNN / vision
//! operators that the ONNX model zoo still needs: `LRN`, `LpPool`,
//! `GlobalLpPool`, `MaxUnpool`, `MaxRoiPool`, `Upsample` and `CastLike`.
//!
//! Each of these previously parsed to `OpKind::Unknown` and failed at run time
//! with `UnsupportedOp`, so AlexNet / CaffeNet / GoogLeNet (`LRN`), SegNet-style
//! decoders (`MaxUnpool`), Fast R-CNN (`MaxRoiPool`) and every opset ≤ 9
//! detection export (`Upsample`) could not load.
//!
//! Reference values come from `onnx.reference` (`onnx` 1.21.0) where it
//! implements the operator, and from an explicit NumPy transcription of the
//! spec where it does not (`GlobalLpPool`, `MaxRoiPool`); the derivation is in
//! `scratchpad/ref_ops.py`. `LRN`'s values are computed from the spec formula
//! rather than `onnx.reference`, whose implementation iterates the *batch*
//! axis where it means the *channel* axis and therefore leaves `square_sum`
//! zero for every channel beyond `N - 1`.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── helpers ─────────────────────────────────────────────────────────────────

fn run_op(
    op: OpKind,
    node_inputs: &[&str],
    node_outputs: &[&str],
    feeds: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let graph = Graph {
        nodes: vec![Node {
            op,
            name: "op0".to_string(),
            inputs: node_inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: node_outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs,
        }],
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

fn try_run_op(
    op: OpKind,
    node_inputs: &[&str],
    node_outputs: &[&str],
    feeds: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> Result<HashMap<String, Tensor>, oxionnx::OnnxError> {
    let graph = Graph {
        nodes: vec![Node {
            op,
            name: "op0".to_string(),
            inputs: node_inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: node_outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs,
        }],
        input_names: feeds.iter().map(|(n, _)| (*n).to_string()).collect(),
        output_names: node_outputs.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let feed: HashMap<&str, Tensor> = feeds.into_iter().collect();
    session.run(&feed)
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

/// `arange(1, 41) * 0.1 - 1.0` reshaped `[2, 5, 2, 2]`, the LRN probe input.
fn lrn_input() -> Tensor {
    let data: Vec<f32> = (1..=40).map(|i| i as f32 * 0.1 - 1.0).collect();
    Tensor::new(data, vec![2, 5, 2, 2])
}

// ── registry / OpKind wiring ────────────────────────────────────────────────

#[test]
fn vision_ops_are_registered() {
    let registry = oxionnx_ops::default_registry();
    for name in [
        "LRN",
        "LpPool",
        "GlobalLpPool",
        "MaxUnpool",
        "MaxRoiPool",
        "Upsample",
        "CastLike",
        // Already present before this wave — asserted so a future refactor
        // cannot quietly drop them while the brief still lists them.
        "LpNormalization",
        "MeanVarianceNormalization",
    ] {
        assert!(registry.contains(name), "{name} must be registered");
        let kind = OpKind::parse(name);
        assert_ne!(
            kind,
            OpKind::Unknown(name.to_string()),
            "{name} must have its own OpKind variant"
        );
    }
}

// ── LRN ─────────────────────────────────────────────────────────────────────

/// Odd `size`: a symmetric 3-channel window.
#[test]
fn lrn_odd_size_window() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".into(), 0.5);
    attrs.floats.insert("beta".into(), 0.75);
    attrs.floats.insert("bias".into(), 1.0);
    attrs.ints.insert("size".into(), 3);

    let out = run_op(OpKind::LRN, &["x"], &["y"], vec![("x", lrn_input())], attrs);
    assert_close(
        &out["y"],
        &[
            -0.796_622_3,
            -0.728_319_2,
            -0.653_193_8,
            -0.571_649_4,
            -0.442_098_4,
            -0.364_159_5,
            -0.279_621_5,
            -0.189_661_4,
            -0.095_836_9,
            0.0,
            0.095_837_0,
            0.189_661_5,
            0.279_621_6,
            0.364_159_5,
            0.442_098_4,
            0.512_673_6,
            0.653_193_9,
            0.728_319_2,
            0.796_622_3,
            0.857_936_3,
            0.781_785_9,
            0.818_078,
            0.849_557,
            0.876_653_5,
            0.836_561_7,
            0.847_495_6,
            0.855_482_5,
            0.860_959_3,
            0.864_308_8,
            0.865_863_3,
            0.865_909_8,
            0.864_694_8,
            0.862_428_8,
            0.859_291_9,
            0.855_436_3,
            0.850_992_2,
            1.156_625_9,
            1.152_336_7,
            1.147_264_8,
            1.141_538_4,
        ],
        &[2, 5, 2, 2],
        2e-6,
        "LRN size=3",
    );
}

/// Even `size`: the window is **asymmetric** — `floor((size-1)/2)` back and
/// `ceil((size-1)/2)` forward, i.e. one channel back and two forward for
/// `size = 4`. A symmetric implementation gets every channel wrong here.
#[test]
fn lrn_even_size_window_is_asymmetric() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".into(), 0.6);
    attrs.floats.insert("beta".into(), 0.5);
    attrs.floats.insert("bias".into(), 2.0);
    attrs.ints.insert("size".into(), 4);

    let out = run_op(OpKind::LRN, &["x"], &["y"], vec![("x", lrn_input())], attrs);
    assert_close(
        &out["y"],
        &[
            -0.612_301_5,
            -0.549_442_3,
            -0.484_374,
            -0.417_432_3,
            -0.339_109_7,
            -0.273_179_2,
            -0.205_749_9,
            -0.137_360_6,
            -0.068_583_3,
            0.0,
            0.067_822_0,
            0.134_352_3,
            0.207_588_9,
            0.274_721_1,
            0.340_167_5,
            0.403_603_7,
            0.484_548,
            0.549_442_3,
            0.612_514_3,
            0.673_587_8,
            0.628_776_9,
            0.670_820_4,
            0.710_424_9,
            0.747_690_9,
            0.764_074,
            0.793_675_8,
            0.821_150_7,
            0.846_648_8,
            0.870_315_3,
            0.892_288_3,
            0.912_698_5,
            0.931_668_3,
            1.092_948,
            1.114_172,
            1.133_964_7,
            1.152_429_8,
            1.369_482_5,
            1.393_052,
            1.415_223_6,
            1.436_080_7,
        ],
        &[2, 5, 2, 2],
        2e-6,
        "LRN size=4 (even)",
    );
}

/// `size = 1` reduces to `x / (bias + alpha * x^2)^beta`, the degenerate case
/// that pins the `alpha / size` division.
#[test]
fn lrn_size_one() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".into(), 1.0);
    attrs.floats.insert("beta".into(), 1.0);
    attrs.floats.insert("bias".into(), 1.0);
    attrs.ints.insert("size".into(), 1);

    let out = run_op(OpKind::LRN, &["x"], &["y"], vec![("x", lrn_input())], attrs);
    // y = x / (1 + x^2); check a couple of positions against that closed form.
    let x = lrn_input();
    for idx in [0_usize, 7, 19, 39] {
        let xv = x.data[idx];
        let expected = xv / (1.0 + xv * xv);
        assert!(
            (out["y"].data[idx] - expected).abs() <= 2e-6,
            "LRN size=1 element {idx}: got {}, expected {expected}",
            out["y"].data[idx]
        );
    }
}

#[test]
fn lrn_rejects_zero_size() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("size".into(), 0);
    let err = try_run_op(OpKind::LRN, &["x"], &["y"], vec![("x", lrn_input())], attrs)
        .expect_err("size = 0 must fail");
    assert!(format!("{err}").contains("size must be >= 1"), "got: {err}");
}

// ── LpPool / GlobalLpPool ───────────────────────────────────────────────────

fn pool_input() -> Tensor {
    Tensor::new(
        vec![
            1., -2., 3., 4., 5., 6., -7., 8., 9., 10., 11., -12., 13., 14., 15., 16.,
        ],
        vec![1, 1, 4, 4],
    )
}

#[test]
fn lp_pool_p2_non_overlapping() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
    attrs.int_lists.insert("strides".into(), vec![2, 2]);
    attrs.ints.insert("p".into(), 2);

    let out = run_op(
        OpKind::LpPool,
        &["x"],
        &["y"],
        vec![("x", pool_input())],
        attrs,
    );
    assert_close(
        &out["y"],
        &[8.124_039, 11.747_34, 23.366_642, 27.313_0],
        &[1, 1, 2, 2],
        1e-4,
        "LpPool p=2",
    );
}

/// `p = 1` with padding: a padded position contributes `|0|^1 == 0`, so the
/// result is the plain sum of absolute values over the in-bounds window —
/// there is **no** division by the window size (which is what would make this
/// an average pool).
#[test]
fn lp_pool_p1_with_padding_sums_without_averaging() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![3, 3]);
    attrs.int_lists.insert("strides".into(), vec![1, 1]);
    attrs.int_lists.insert("pads".into(), vec![1, 1, 1, 1]);
    attrs.ints.insert("p".into(), 1);

    let out = run_op(
        OpKind::LpPool,
        &["x"],
        &["y"],
        vec![("x", pool_input())],
        attrs,
    );
    assert_close(
        &out["y"],
        &[
            14., 24., 30., 22., 33., 54., 63., 45., 57., 90., 99., 69., 46., 72., 78., 54.,
        ],
        &[1, 1, 4, 4],
        1e-4,
        "LpPool p=1 padded",
    );
}

#[test]
fn global_lp_pool_p2_and_p3() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("p".into(), 2);
    let out = run_op(
        OpKind::GlobalLpPool,
        &["x"],
        &["y"],
        vec![("x", pool_input())],
        attrs,
    );
    assert_close(
        &out["y"],
        &[38.678_158],
        &[1, 1, 1, 1],
        1e-3,
        "GlobalLpPool p=2",
    );

    let mut attrs = Attributes::default();
    attrs.ints.insert("p".into(), 3);
    let out = run_op(
        OpKind::GlobalLpPool,
        &["x"],
        &["y"],
        vec![("x", pool_input())],
        attrs,
    );
    assert_close(
        &out["y"],
        &[26.445_955],
        &[1, 1, 1, 1],
        1e-3,
        "GlobalLpPool p=3",
    );
}

#[test]
fn lp_pool_rejects_p_zero() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
    attrs.ints.insert("p".into(), 0);
    let err = try_run_op(
        OpKind::LpPool,
        &["x"],
        &["y"],
        vec![("x", pool_input())],
        attrs,
    )
    .expect_err("p = 0 must fail");
    assert!(format!("{err}").contains("p must be >= 1"), "got: {err}");
}

// ── MaxUnpool ───────────────────────────────────────────────────────────────

#[test]
fn max_unpool_scatters_to_inferred_shape() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
    attrs.int_lists.insert("strides".into(), vec![2, 2]);

    let out = run_op(
        OpKind::MaxUnpool,
        &["x", "i"],
        &["y"],
        vec![
            ("x", Tensor::new(vec![5., 6., 9., 10.], vec![1, 1, 2, 2])),
            ("i", Tensor::new(vec![5., 7., 13., 15.], vec![1, 1, 2, 2])),
        ],
        attrs,
    );
    assert_close(
        &out["y"],
        &[
            0., 0., 0., 0., 0., 5., 0., 6., 0., 0., 0., 0., 0., 9., 0., 10.,
        ],
        &[1, 1, 4, 4],
        0.0,
        "MaxUnpool inferred shape",
    );
}

/// With an explicit `output_shape`, the ONNX node test
/// `test_maxunpool_export_with_output_shape` places the *inferred* 4x4 block at
/// the origin of the 5x5 output — the indices are **not** re-interpreted
/// against the larger shape (which would move `5` from `(1,1)` to `(1,0)`).
#[test]
fn max_unpool_output_shape_reframes_the_inferred_block() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
    attrs.int_lists.insert("strides".into(), vec![2, 2]);

    let out = run_op(
        OpKind::MaxUnpool,
        &["x", "i", "s"],
        &["y"],
        vec![
            ("x", Tensor::new(vec![5., 6., 9., 10.], vec![1, 1, 2, 2])),
            ("i", Tensor::new(vec![5., 7., 13., 15.], vec![1, 1, 2, 2])),
            ("s", Tensor::new(vec![1., 1., 5., 5.], vec![4])),
        ],
        attrs,
    );
    assert_close(
        &out["y"],
        &[
            0., 0., 0., 0., 0., //
            0., 5., 0., 6., 0., //
            0., 0., 0., 0., 0., //
            0., 9., 0., 10., 0., //
            0., 0., 0., 0., 0.,
        ],
        &[1, 1, 5, 5],
        0.0,
        "MaxUnpool with output_shape",
    );
}

/// An index past the end of the inferred tensor is a malformed model: it must
/// be a typed error, never an out-of-bounds write.
#[test]
fn max_unpool_rejects_out_of_range_index() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
    attrs.int_lists.insert("strides".into(), vec![2, 2]);

    let err = try_run_op(
        OpKind::MaxUnpool,
        &["x", "i"],
        &["y"],
        vec![
            ("x", Tensor::new(vec![5., 6., 9., 10.], vec![1, 1, 2, 2])),
            ("i", Tensor::new(vec![5., 7., 13., 999.], vec![1, 1, 2, 2])),
        ],
        attrs,
    )
    .expect_err("out-of-range index must fail");
    assert!(
        format!("{err}").contains("outside the inferred"),
        "got: {err}"
    );
}

// ── MaxRoiPool ──────────────────────────────────────────────────────────────

fn roi_feature_map() -> Tensor {
    Tensor::new((1..=36).map(|v| v as f32).collect(), vec![1, 1, 6, 6])
}

/// Two RoIs over a 6x6 map with `spatial_scale = 1`. RoI rows are
/// `[batch, x1, y1, x2, y2]` — **width first**.
#[test]
fn max_roi_pool_two_rois() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pooled_shape".into(), vec![2, 2]);
    attrs.floats.insert("spatial_scale".into(), 1.0);

    let out = run_op(
        OpKind::MaxRoiPool,
        &["x", "rois"],
        &["y"],
        vec![
            ("x", roi_feature_map()),
            (
                "rois",
                Tensor::new(vec![0., 0., 0., 4., 4., 0., 1., 1., 5., 5.], vec![2, 5]),
            ),
        ],
        attrs,
    );
    assert_close(
        &out["y"],
        &[15., 17., 27., 29., 22., 24., 34., 36.],
        &[2, 1, 2, 2],
        0.0,
        "MaxRoiPool 2 RoIs",
    );
}

/// `spatial_scale != 1` scales (and rounds) the RoI coordinates into
/// feature-map space before binning.
///
/// The rounding is C's `std::round` — **half away from zero**, what Caffe and
/// ONNX Runtime use — so `9 * 0.5 = 4.5` becomes `5`, not the `4` that a
/// ties-to-even `round` would give. The two choices differ visibly here:
/// half-away-from-zero yields a bin width of exactly 2 and the last column
/// `[12, 24, 36]`, ties-to-even yields `5/3` and `[11, 23, 35]`.
#[test]
fn max_roi_pool_with_spatial_scale() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pooled_shape".into(), vec![3, 3]);
    attrs.floats.insert("spatial_scale".into(), 0.5);

    let out = run_op(
        OpKind::MaxRoiPool,
        &["x", "rois"],
        &["y"],
        vec![
            ("x", roi_feature_map()),
            ("rois", Tensor::new(vec![0., 0., 0., 9., 9.], vec![1, 5])),
        ],
        attrs,
    );
    assert_close(
        &out["y"],
        &[8., 10., 12., 20., 22., 24., 32., 34., 36.],
        &[1, 1, 3, 3],
        0.0,
        "MaxRoiPool scale=0.5",
    );
}

#[test]
fn max_roi_pool_rejects_bad_batch_index() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pooled_shape".into(), vec![2, 2]);

    let err = try_run_op(
        OpKind::MaxRoiPool,
        &["x", "rois"],
        &["y"],
        vec![
            ("x", roi_feature_map()),
            ("rois", Tensor::new(vec![7., 0., 0., 4., 4.], vec![1, 5])),
        ],
        attrs,
    )
    .expect_err("batch index 7 with N = 1 must fail");
    assert!(format!("{err}").contains("batch index"), "got: {err}");
}

// ── Upsample ────────────────────────────────────────────────────────────────

/// `Upsample`-9: `scales` arrives as the **second input**.
///
/// The result is `np.repeat` in both spatial axes, which is the `asymmetric`
/// coordinate transformation. `Resize`'s default (`half_pixel`) would give
/// `[1, 1, 1, 2 / …]` instead of `[1, 1, 2, 2 / …]` on this exact input, so
/// this test discriminates the two mappings.
#[test]
fn upsample_nearest_scales_input_is_asymmetric() {
    let mut attrs = Attributes::default();
    attrs.strings.insert("mode".into(), "nearest".into());

    let out = run_op(
        OpKind::Upsample,
        &["x", "scales"],
        &["y"],
        vec![
            ("x", Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2])),
            ("scales", Tensor::new(vec![1., 1., 2., 2.], vec![4])),
        ],
        attrs,
    );
    assert_close(
        &out["y"],
        &[
            1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.,
        ],
        &[1, 1, 4, 4],
        0.0,
        "Upsample-9 nearest",
    );
}

/// `Upsample`-7: `scales` is a **float-list attribute**, and the axes may use
/// different factors.
#[test]
fn upsample_nearest_scales_attribute() {
    let mut attrs = Attributes::default();
    attrs.strings.insert("mode".into(), "nearest".into());
    attrs
        .float_lists
        .insert("scales".into(), vec![1.0, 1.0, 2.0, 3.0]);

    let out = run_op(
        OpKind::Upsample,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2]))],
        attrs,
    );
    assert_close(
        &out["y"],
        &[
            1., 1., 1., 2., 2., 2., //
            1., 1., 1., 2., 2., 2., //
            3., 3., 3., 4., 4., 4., //
            3., 3., 3., 4., 4., 4.,
        ],
        &[1, 1, 4, 6],
        0.0,
        "Upsample-7 nearest (scales attribute)",
    );
}

/// `linear` mode maps onto `Resize(asymmetric, linear)`.
#[test]
fn upsample_linear() {
    let mut attrs = Attributes::default();
    attrs.strings.insert("mode".into(), "linear".into());

    let out = run_op(
        OpKind::Upsample,
        &["x", "scales"],
        &["y"],
        vec![
            ("x", Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2])),
            ("scales", Tensor::new(vec![1., 1., 2., 2.], vec![4])),
        ],
        attrs,
    );
    assert_close(
        &out["y"],
        &[
            1.0, 1.5, 2.0, 2.0, //
            2.0, 2.5, 3.0, 3.0, //
            3.0, 3.5, 4.0, 4.0, //
            3.0, 3.5, 4.0, 4.0,
        ],
        &[1, 1, 4, 4],
        1e-6,
        "Upsample linear",
    );
}

#[test]
fn upsample_requires_scales() {
    let attrs = Attributes::default();
    let err = try_run_op(
        OpKind::Upsample,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2]))],
        attrs,
    )
    .expect_err("Upsample without scales must fail");
    assert!(format!("{err}").contains("no scales given"), "got: {err}");
}

// ── CastLike ────────────────────────────────────────────────────────────────

/// On the f32 execution path `CastLike` is an identity copy: every tensor is
/// already f32 and carries no dtype tag, so there is no target type to read.
/// (The typed path performs the real cast — see `CastLikeOp`'s docs.) What is
/// pinned here is that the node *runs* instead of failing with `UnsupportedOp`,
/// which is what blocked every modern PyTorch export that emits it.
#[test]
fn cast_like_runs_and_preserves_values_on_the_f32_path() {
    let out = run_op(
        OpKind::CastLike,
        &["x", "t"],
        &["y"],
        vec![
            ("x", Tensor::new(vec![1.7, -2.3, 3.9], vec![3])),
            ("t", Tensor::new(vec![0.], vec![1])),
        ],
        Attributes::default(),
    );
    assert_close(
        &out["y"],
        &[1.7, -2.3, 3.9],
        &[3],
        0.0,
        "CastLike f32 identity",
    );
}

#[test]
fn cast_like_requires_the_target_input() {
    let err = try_run_op(
        OpKind::CastLike,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(vec![1.0], vec![1]))],
        Attributes::default(),
    )
    .expect_err("CastLike without target_type must fail");
    assert!(format!("{err}").contains("input[1]"), "got: {err}");
}

/// `MaxPool` → `MaxUnpool` round-trip in one graph — the SegNet decoder shape.
///
/// This is the only check that `MaxUnpool`'s index space agrees with *this
/// engine's* `MaxPool` index encoding (`((n*C + c) * H + h) * W + w`) rather
/// than merely with the ONNX node test's hard-coded constants. With matching
/// `kernel_shape` / `strides` / `pads` the inferred unpool extent equals
/// `MaxPool`'s input extent, so every pooled maximum must land back on the
/// exact position it was taken from and everything else must be zero.
#[test]
fn max_pool_then_max_unpool_round_trips() {
    let mut pool_attrs = Attributes::default();
    pool_attrs
        .int_lists
        .insert("kernel_shape".into(), vec![2, 2]);
    pool_attrs.int_lists.insert("strides".into(), vec![2, 2]);
    let mut unpool_attrs = Attributes::default();
    unpool_attrs
        .int_lists
        .insert("kernel_shape".into(), vec![2, 2]);
    unpool_attrs.int_lists.insert("strides".into(), vec![2, 2]);

    let graph = Graph {
        nodes: vec![
            Node {
                op: OpKind::MaxPool,
                name: "pool".into(),
                inputs: vec!["x".into()],
                outputs: vec!["pooled".into(), "indices".into()],
                attrs: pool_attrs,
            },
            Node {
                op: OpKind::MaxUnpool,
                name: "unpool".into(),
                inputs: vec!["pooled".into(), "indices".into()],
                outputs: vec!["out".into()],
                attrs: unpool_attrs,
            },
        ],
        input_names: vec!["x".into()],
        output_names: vec!["pooled".into(), "out".into()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");

    let x = Tensor::new((1..=16).map(|v| v as f32).collect(), vec![1, 1, 4, 4]);
    let feed: HashMap<&str, Tensor> = [("x", x)].into_iter().collect();
    let out = session.run(&feed).expect("run");

    assert_close(
        &out["pooled"],
        &[6., 8., 14., 16.],
        &[1, 1, 2, 2],
        0.0,
        "MaxPool values",
    );
    assert_close(
        &out["out"],
        &[
            0., 0., 0., 0., //
            0., 6., 0., 8., //
            0., 0., 0., 0., //
            0., 14., 0., 16.,
        ],
        &[1, 1, 4, 4],
        0.0,
        "MaxUnpool round-trip",
    );
}

// ── output-slot path ────────────────────────────────────────────────────────

/// `LRN`, `LpPool`, `GlobalLpPool` and `Upsample` all declare
/// `supports_output_slots()`, which the session takes whenever it can
/// pre-allocate outputs. That path is a *second* implementation of each
/// operator's shape and write logic, so it is checked directly here against
/// the allocating `execute` — including a slot that arrives pre-sized to the
/// **wrong** shape, which is what a reused pool buffer looks like.
#[test]
fn slot_path_agrees_with_execute_for_every_new_slot_operator() {
    use oxionnx::{Node as OpNode, OpContext, Operator};
    use oxionnx_ops::registry::vision_ops::{GlobalLpPoolOp, LRNOp, LpPoolOp, UpsampleOp};

    fn check(op: &dyn Operator, inputs: &[&Tensor], attrs: Attributes) {
        let node = OpNode {
            op: OpKind::Identity, // unused by these operators; only attrs/outputs matter
            name: "slot_probe".into(),
            inputs: (0..inputs.len()).map(|i| format!("in{i}")).collect(),
            outputs: vec!["y".into()],
            attrs,
        };
        let ctx = OpContext {
            node: &node,
            inputs: inputs.iter().map(|t| Some(*t)).collect(),
            outer_scope: None,
            weights: None,
            registry: None,
        };
        let expected = op.execute(&ctx).expect("execute");
        assert!(op.supports_output_slots(), "{}", op.op_type());

        // A deliberately mis-sized slot, as a recycled pool buffer would be.
        let mut slots = vec![Tensor::new(vec![-1.0; 3], vec![3])];
        op.execute_into_slots(&ctx, &mut slots)
            .expect("execute_into_slots");
        assert_eq!(
            slots[0].shape,
            expected[0].shape,
            "{}: slot shape",
            op.op_type()
        );
        assert_eq!(
            slots[0].data.len(),
            expected[0].data.len(),
            "{}: slot length",
            op.op_type()
        );
        for (i, (&a, &e)) in slots[0].data.iter().zip(&expected[0].data).enumerate() {
            assert!(
                (a - e).abs() <= 1e-6,
                "{}: slot element {i}: got {a}, expected {e}",
                op.op_type()
            );
        }
    }

    let x = lrn_input();
    let mut attrs = Attributes::default();
    attrs.ints.insert("size".into(), 3);
    attrs.floats.insert("alpha".into(), 0.5);
    check(&LRNOp, &[&x], attrs);

    let pooled = pool_input();
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
    attrs.int_lists.insert("strides".into(), vec![2, 2]);
    attrs.ints.insert("p".into(), 2);
    check(&LpPoolOp, &[&pooled], attrs);

    let mut attrs = Attributes::default();
    attrs.ints.insert("p".into(), 2);
    check(&GlobalLpPoolOp, &[&pooled], attrs);

    let up_x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2]);
    let scales = Tensor::new(vec![1., 1., 2., 2.], vec![4]);
    let mut attrs = Attributes::default();
    attrs.strings.insert("mode".into(), "nearest".into());
    check(&UpsampleOp, &[&up_x, &scales], attrs);
}
