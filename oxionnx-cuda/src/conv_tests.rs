//! Unit tests for [`crate::conv`].
//!
//! Split out of `conv.rs` for the same reason `dispatch_tests.rs` is split out
//! of `lib.rs`: the module's own code plus its suite ran past this workspace's
//! 2000-line-per-file ceiling, and a test module is the half that can move
//! without changing a single call site. `use super::*` still reaches every
//! private item, so nothing here had to be widened to be testable.

use super::*;

#[test]
fn conv_params_round_trip() {
    let params = ConvParams {
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
        group: 1,
        activation: ConvActivation::None,
    };
    assert_eq!(params.strides, [2, 2]);
    assert_eq!(params.pads, [1, 1, 1, 1]);
    assert_eq!(params.dilations, [1, 1]);
    assert_eq!(params.group, 1);
}

/// `Conv` must be advertised as a CUDA-supported op: [`cuda_conv`]
/// dispatches to a real `oxicuda-dnn` engine for every configuration
/// [`problem_from_params`] accepts, so placement logic must be allowed to
/// route convolutions here.
///
/// The mirror image of this module's original `Conv`-is-unsupported
/// assertion, kept in the same place on purpose: this file is where a
/// future change would most plausibly re-break the link between what
/// [`cuda_conv`] can do and what [`crate::is_supported_op`] says it can
/// do, so the check lives next to the implementation as well as next to
/// the predicate (`crate::tests::conv_is_advertised_as_supported`).
#[test]
fn conv_is_advertised_as_supported() {
    use oxionnx_core::graph::OpKind;
    assert!(
        crate::is_supported_op(&OpKind::Conv),
        "cuda_conv() computes real convolutions on the GPU (Conv1x1 / DepthwiseConv / \
         ImplicitGemmConv); is_supported_op must report Conv as supported so \
         decide_placement routes Conv nodes to CUDA",
    );
}

// ── problem_from_params: pure, GPU-free ─────────────────────────────────

fn base_params() -> ConvParams {
    ConvParams {
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
        group: 1,
        activation: ConvActivation::None,
    }
}

#[test]
fn problem_from_params_rejects_non_4d_input() {
    assert!(problem_from_params(&[1, 3, 5], &[4, 3, 3, 3], &base_params()).is_none());
}

#[test]
fn problem_from_params_rejects_non_4d_weight() {
    assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3], &base_params()).is_none());
}

#[test]
fn problem_from_params_rejects_asymmetric_pad_top_bottom() {
    let mut params = base_params();
    params.pads = [1, 1, 0, 1];
    assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
}

#[test]
fn problem_from_params_rejects_asymmetric_pad_left_right() {
    let mut params = base_params();
    params.pads = [1, 1, 1, 0];
    assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
}

#[test]
fn problem_from_params_accepts_symmetric_pad_and_derives_every_field() {
    let mut params = base_params();
    params.pads = [1, 1, 1, 1];
    params.strides = [2, 3];
    params.dilations = [1, 2];
    let p = problem_from_params(&[1, 3, 5, 7], &[4, 3, 3, 3], &params)
        .expect("well-formed symmetric-pad convolution must be accepted");
    assert_eq!(p.batch, 1);
    assert_eq!(p.in_channels, 3);
    assert_eq!(p.in_dims, vec![5, 7]);
    assert_eq!(p.out_channels, 4);
    assert_eq!(p.filter_dims, vec![3, 3]);
    assert_eq!(p.padding, vec![1, 1]);
    assert_eq!(p.stride, vec![2, 3]);
    assert_eq!(p.dilation, vec![1, 2]);
    assert_eq!(p.groups, 1);
}

#[test]
fn problem_from_params_rejects_zero_group() {
    let mut params = base_params();
    params.group = 0;
    assert!(problem_from_params(&[1, 4, 5, 5], &[4, 4, 3, 3], &params).is_none());
}

#[test]
fn problem_from_params_rejects_in_channels_not_divisible_by_group() {
    let mut params = base_params();
    params.group = 3;
    // in_channels=4 is not divisible by group=3.
    assert!(problem_from_params(&[1, 4, 5, 5], &[6, 2, 3, 3], &params).is_none());
}

#[test]
fn problem_from_params_rejects_out_channels_not_divisible_by_group() {
    let mut params = base_params();
    params.group = 2;
    // in_channels=4 divides group=2 fine, but out_channels=5 does not.
    assert!(problem_from_params(&[1, 4, 5, 5], &[5, 2, 3, 3], &params).is_none());
}

#[test]
fn problem_from_params_rejects_filter_in_channel_mismatch() {
    // in_channels=4, group=1 -> filter dim[1] must be 4; it claims 3.
    assert!(problem_from_params(&[1, 4, 5, 5], &[4, 3, 3, 3], &base_params()).is_none());
}

#[test]
fn problem_from_params_accepts_grouped_conv_with_correct_filter_channels() {
    let mut params = base_params();
    params.group = 2;
    // in_channels=4, group=2 -> filter dim[1] must be 2.
    let p = problem_from_params(&[1, 4, 5, 5], &[6, 2, 3, 3], &params)
        .expect("a correctly-shaped grouped conv must be accepted");
    assert_eq!(p.groups, 2);
}

#[test]
fn problem_from_params_rejects_zero_batch() {
    assert!(problem_from_params(&[0, 3, 5, 5], &[4, 3, 3, 3], &base_params()).is_none());
}

#[test]
fn problem_from_params_rejects_zero_spatial_dim() {
    assert!(problem_from_params(&[1, 3, 0, 5], &[4, 3, 3, 3], &base_params()).is_none());
}

#[test]
fn problem_from_params_rejects_zero_filter_spatial_dim() {
    assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 0, 3], &base_params()).is_none());
}

#[test]
fn problem_from_params_rejects_filter_larger_than_padded_input() {
    // 3x3 input, 5x5 filter, no padding: the filter does not fit.
    assert!(problem_from_params(&[1, 3, 3, 3], &[4, 3, 5, 5], &base_params()).is_none());
}

#[test]
fn problem_from_params_rejects_zero_stride() {
    let mut params = base_params();
    params.strides = [0, 1];
    assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
}

#[test]
fn problem_from_params_rejects_zero_dilation() {
    let mut params = base_params();
    params.dilations = [1, 0];
    assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
}

// ── pick_engine: pure, GPU-free ──────────────────────────────────────────

fn tiny_problem(
    filter: [u32; 2],
    stride: [u32; 2],
    dilation: [u32; 2],
    pad: [u32; 2],
    groups: u32,
    in_channels: u32,
    out_channels: u32,
) -> ConvProblem {
    ConvProblem {
        batch: 1,
        in_channels,
        in_dims: vec![16, 16],
        out_channels,
        filter_dims: filter.to_vec(),
        padding: pad.to_vec(),
        stride: stride.to_vec(),
        dilation: dilation.to_vec(),
        groups,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    }
}

#[test]
fn pick_engine_selects_conv1x1_for_unpadded_unit_stride_1x1() {
    let p = tiny_problem([1, 1], [1, 1], [1, 1], [0, 0], 1, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::Conv1x1);
}

#[test]
fn pick_engine_does_not_select_conv1x1_when_padded() {
    let p = tiny_problem([1, 1], [1, 1], [1, 1], [1, 1], 1, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
}

#[test]
fn pick_engine_does_not_select_conv1x1_when_strided() {
    let p = tiny_problem([1, 1], [2, 2], [1, 1], [0, 0], 1, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
}

#[test]
fn pick_engine_does_not_select_conv1x1_when_dilated() {
    // Dilation is mathematically moot for a 1x1 tap, but rule 1 checks
    // it literally, per spec -- pin that down explicitly.
    let p = tiny_problem([1, 1], [1, 1], [2, 2], [0, 0], 1, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
}

#[test]
fn pick_engine_selects_depthwise_for_true_depthwise_3x3() {
    let p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 8, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::Depthwise);
}

#[test]
fn pick_engine_selects_implicit_gemm_for_grouped_non_depthwise() {
    // groups=2, in=out=8: depthwise would require groups==8.
    let p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 2, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
}

#[test]
fn pick_engine_selects_implicit_gemm_for_plain_3x3() {
    let p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 1, 16, 32);
    assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
}

#[test]
fn pick_engine_prioritises_conv1x1_over_depthwise_when_both_apply() {
    // filter=1x1, groups==in==out==8: matches BOTH the Conv1x1 and the
    // Depthwise conditions -- rule 1 (Conv1x1) must win, per the
    // ordered 3-way dispatch rule in the module docs.
    let p = tiny_problem([1, 1], [1, 1], [1, 1], [0, 0], 8, 8, 8);
    assert_eq!(pick_engine(&p), ConvEngine::Conv1x1);
}

/// Rule 0: a problem big enough for the CTA-tiled implicit-GEMM kernel goes
/// to `ImplicitGemmConv` (which dispatches to that kernel internally) even
/// when it is an unpadded unit-stride 1x1 -- the shape rule 1 would otherwise
/// claim.
///
/// This is a throughput decision with a measured basis, not a preference: on
/// an RTX A4000 `Conv1x1` runs this shape at 480 GFLOPS and the tiled kernel
/// at 7345 GFLOPS. It also moves the bias and any fused activation onto the
/// device, because `Conv1x1::execute` has no bias parameter at all.
#[test]
fn pick_engine_prefers_the_tiled_kernel_over_conv1x1_on_a_large_pointwise() {
    let mut p = tiny_problem([1, 1], [1, 1], [1, 1], [0, 0], 1, 256, 512);
    p.in_dims = vec![64, 64];
    assert!(
        TiledConvPlan::for_problem(&p).is_some(),
        "fixture must be one the tiling claims, or this test proves nothing"
    );
    assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
}

/// ...and a 1x1 the tiling declines still takes the `Conv1x1` path, so rule 0
/// narrows rule 1 rather than replacing it.
#[test]
fn pick_engine_still_selects_conv1x1_when_the_tiling_declines() {
    let p = tiny_problem([1, 1], [1, 1], [1, 1], [0, 0], 1, 8, 8);
    assert!(TiledConvPlan::for_problem(&p).is_none());
    assert_eq!(pick_engine(&p), ConvEngine::Conv1x1);
}

/// Rule 0 must never steal a depthwise problem: the tiling models
/// `groups == 1` only, and a depthwise convolution routed to the general
/// engine would be correct but far slower than `DepthwiseConv`.
#[test]
fn pick_engine_keeps_large_depthwise_on_the_depthwise_engine() {
    let mut p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 512, 512, 512);
    p.in_dims = vec![128, 128];
    assert!(
        TiledConvPlan::for_problem(&p).is_none(),
        "the tiling must decline every grouped convolution"
    );
    assert_eq!(pick_engine(&p), ConvEngine::Depthwise);
}

/// The face pipeline's own dominant convolutions must all land on the engine
/// that dispatches to the tiled kernel -- this is the routing the whole
/// change exists for, asserted on the real shapes rather than on a synthetic
/// one.
#[test]
fn pick_engine_routes_every_dominant_face_pipeline_conv_to_the_tiled_kernel() {
    // (label, C_in, H, W, C_out, pad)
    let shapes = [
        (
            "InSwapper resblock 1024->1024 @34x34",
            1024u32,
            34u32,
            1024u32,
            0u32,
        ),
        ("InSwapper decoder 1024->512 @64x64", 1024, 64, 512, 1),
        ("InSwapper decoder 512->256 @128x128", 512, 128, 256, 1),
        ("SCRFD 28->56 @320x320", 28, 320, 56, 1),
        ("ArcFace 64->64 @112x112", 64, 112, 64, 1),
    ];
    for (label, cin, hw, cout, pad) in shapes {
        let mut p = tiny_problem([3, 3], [1, 1], [1, 1], [pad, pad], 1, cin, cout);
        p.in_dims = vec![hw, hw];
        assert_eq!(
            pick_engine(&p),
            ConvEngine::ImplicitGemm,
            "{label} must route to the tiled implicit-GEMM kernel"
        );
    }
}

// ── add_bias_nchw: pure, GPU-free ────────────────────────────────────────

#[test]
fn add_bias_nchw_broadcasts_per_channel_across_batch_and_spatial() {
    // n=2, channels=2, spatial=3 -> 12 elements, layout
    // [n0c0(3), n0c1(3), n1c0(3), n1c1(3)].
    let mut data = vec![0.0_f32; 12];
    let bias = [10.0_f32, 100.0];
    add_bias_nchw(&mut data, &bias, 2, 2, 3);
    assert_eq!(
        data,
        vec![10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0]
    );
}

// ── The fused-activation contract, GPU-free ─────────────────────────────
//
// The bug these pin down: `oxionnx`'s optimizer rewrites every
// `Conv -> Relu` pair into a single `Conv` node carrying
// `activation="relu"`, and this backend used to read only
// `strides`/`pads`/`dilations`/`group`, so it returned the *raw*
// convolution for 26 of SCRFD det_10g's 58 convolutions. Shadow
// verification could not see it, because `reference::ref_conv` read the
// same `ConvParams` and therefore made the same omission.

/// Build an `Attributes` the way the optimizer's fusion pass does.
fn fused_attrs(activation: &str) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), vec![1, 1]);
    attrs.int_lists.insert("pads".into(), vec![1, 1, 1, 1]);
    attrs.int_lists.insert("dilations".into(), vec![1, 1]);
    attrs.ints.insert("group".into(), 1);
    if !activation.is_empty() {
        attrs
            .strings
            .insert("activation".into(), activation.to_string());
    }
    attrs
}

#[test]
fn a_conv_node_with_no_activation_attribute_parses_to_none() {
    let params = conv_params_from_attrs(&fused_attrs(""), &[1, 3, 8, 8], &[4, 3, 3, 3])
        .expect("a plain 3x3 Conv must be claimable");
    assert_eq!(params.activation, ConvActivation::None);
    assert_eq!(params.strides, [1, 1]);
    assert_eq!(params.pads, [1, 1, 1, 1]);
}

/// The regression test for the corruption: the optimizer's fused `Relu`
/// must reach [`ConvParams`], not be dropped on the floor.
#[test]
fn the_optimizers_fused_relu_reaches_conv_params() {
    let params = conv_params_from_attrs(&fused_attrs("relu"), &[1, 3, 8, 8], &[4, 3, 3, 3])
        .expect("a fused-Relu Conv must still be claimable");
    assert_eq!(
        params.activation,
        ConvActivation::Relu,
        "a Conv_*_fused_activation node's Relu must survive attribute parsing; dropping it \
         is what collapsed every SCRFD detection to a degenerate corner box",
    );
}

#[test]
fn the_optimizers_fused_clip_carries_its_bounds() {
    let mut attrs = fused_attrs("clip");
    attrs.floats.insert("activation_min".into(), 0.0);
    attrs.floats.insert("activation_max".into(), 6.0);
    let params = conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3])
        .expect("a fused-Clip Conv must still be claimable");
    assert_eq!(
        params.activation,
        ConvActivation::Clip { min: 0.0, max: 6.0 }
    );
}

/// The polarity that matters: an activation this backend has no
/// implementation of must **decline**, never be silently ignored.
#[test]
fn an_unrecognised_fused_activation_declines_the_node() {
    for activation in ["sigmoid", "tanh", "leakyrelu", "RELU", "relu6"] {
        assert!(
            conv_params_from_attrs(&fused_attrs(activation), &[1, 3, 8, 8], &[4, 3, 3, 3])
                .is_none(),
            "activation={activation:?} has no CUDA implementation and must decline the node \
             rather than be ignored",
        );
    }
}

#[test]
fn auto_pad_same_upper_is_resolved_not_ignored() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), vec![1, 1]);
    attrs
        .strings
        .insert("auto_pad".into(), "SAME_UPPER".to_string());
    // 3x3 kernel, stride 1: SAME needs 1 pixel on each side.
    let params = conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3])
        .expect("SAME_UPPER is resolvable");
    assert_eq!(
        params.pads,
        [1, 1, 1, 1],
        "auto_pad overrides `pads`; reading only `pads` convolves a SAME model unpadded",
    );
}

#[test]
fn auto_pad_same_puts_the_odd_pixel_on_the_named_side() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), vec![1, 1]);
    attrs
        .strings
        .insert("auto_pad".into(), "SAME_UPPER".to_string());
    // 4x4 kernel, stride 1, input 8: total padding 3, split 1 + 2.
    let upper = conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 4, 4]).unwrap();
    assert_eq!(upper.pads, [1, 1, 2, 2]);
    attrs
        .strings
        .insert("auto_pad".into(), "SAME_LOWER".to_string());
    let lower = conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 4, 4]).unwrap();
    assert_eq!(lower.pads, [2, 2, 1, 1]);
}

#[test]
fn auto_pad_valid_zeroes_the_padding_even_when_pads_says_otherwise() {
    let mut attrs = fused_attrs("");
    attrs.strings.insert("auto_pad".into(), "VALID".to_string());
    let params = conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).unwrap();
    assert_eq!(params.pads, [0, 0, 0, 0]);
}

#[test]
fn an_unknown_auto_pad_value_declines_the_node() {
    let mut attrs = fused_attrs("");
    attrs
        .strings
        .insert("auto_pad".into(), "SAME_MIDDLE".to_string());
    assert!(conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).is_none());
}

/// `as usize` on a negative `i64` wraps to ~1.8e19 and sails past every
/// downstream `!= 0` check, so these must be caught at the parse.
#[test]
fn negative_geometry_attributes_decline_rather_than_wrapping() {
    for (name, value) in [
        ("strides", vec![-1_i64, 1]),
        ("dilations", vec![1, -2]),
        ("pads", vec![-1, 0, 0, 0]),
    ] {
        let mut attrs = fused_attrs("");
        attrs.int_lists.insert(name.into(), value.clone());
        assert!(
            conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).is_none(),
            "{name}={value:?} must decline, not wrap into a colossal usize",
        );
    }
    let mut attrs = fused_attrs("");
    attrs.ints.insert("group".into(), 0);
    assert!(conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).is_none());
    attrs.ints.insert("group".into(), -1);
    assert!(conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).is_none());
}

#[test]
fn a_kernel_shape_contradicting_the_filter_declines_the_node() {
    let mut attrs = fused_attrs("");
    attrs.int_lists.insert("kernel_shape".into(), vec![5, 5]);
    assert!(
        conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).is_none(),
        "the CPU kernel rejects this node; CUDA must not answer where the CPU errors",
    );
    attrs.int_lists.insert("kernel_shape".into(), vec![3, 3]);
    assert!(conv_params_from_attrs(&attrs, &[1, 3, 8, 8], &[4, 3, 3, 3]).is_some());
}

#[test]
fn a_non_2d_spatial_attribute_declines_the_node() {
    let mut attrs = fused_attrs("");
    attrs.int_lists.insert("strides".into(), vec![1, 1, 1]);
    assert!(conv_params_from_attrs(&attrs, &[1, 3, 8, 8, 8], &[4, 3, 3, 3, 3]).is_none());
}

// ── the host activation epilogue ────────────────────────────────────────

#[test]
fn host_relu_epilogue_rectifies_in_place() {
    let mut data = vec![-2.0_f32, -0.0, 0.0, 3.5];
    apply_conv_activation_host(&mut data, ConvActivation::Relu);
    assert_eq!(data, vec![0.0, 0.0, 0.0, 3.5]);
}

#[test]
fn host_none_epilogue_is_the_identity() {
    let mut data = vec![-2.0_f32, 7.0];
    apply_conv_activation_host(&mut data, ConvActivation::None);
    assert_eq!(data, vec![-2.0, 7.0]);
}

#[test]
fn host_clip_epilogue_clamps_to_both_bounds() {
    let mut data = vec![-1.0_f32, 0.5, 9.0];
    apply_conv_activation_host(&mut data, ConvActivation::Clip { min: 0.0, max: 6.0 });
    assert_eq!(data, vec![0.0, 0.5, 6.0]);
}

/// Matches `oxionnx-ops`' `apply_fused_activation`: an inverted range
/// passes the data through rather than tripping `f32::clamp`'s assert.
#[test]
fn host_clip_epilogue_passes_an_inverted_range_through_untouched() {
    let mut data = vec![-1.0_f32, 0.5, 9.0];
    apply_conv_activation_host(&mut data, ConvActivation::Clip { min: 6.0, max: 0.0 });
    assert_eq!(data, vec![-1.0, 0.5, 9.0]);
}

/// Also matches `apply_fused_activation`: a NaN bound is no bound.
#[test]
fn host_clip_epilogue_treats_a_nan_bound_as_unbounded() {
    let mut data = vec![-1.0_f32, 0.5, 9.0];
    apply_conv_activation_host(
        &mut data,
        ConvActivation::Clip {
            min: f32::NAN,
            max: 6.0,
        },
    );
    assert_eq!(data, vec![-1.0, 0.5, 6.0]);
}

// ── the oracle applies it too ───────────────────────────────────────────

/// The second half of the bug: even a correct kernel is unprotected if
/// the oracle it is checked against makes the same omission.
#[test]
fn the_reference_oracle_applies_the_fused_activation() {
    // 1x1x2x2 input, 1x1x1x1 filter of -1: the raw convolution is all
    // negative, so a Relu-fused node's output must be all zero.
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let weight = [-1.0_f32];
    let mut params = base_params();
    let raw =
        crate::reference::ref_conv(&input, &weight, None, &[1, 1, 2, 2], &[1, 1, 1, 1], &params);
    assert_eq!(raw, vec![-1.0, -2.0, -3.0, -4.0]);

    params.activation = ConvActivation::Relu;
    let rectified =
        crate::reference::ref_conv(&input, &weight, None, &[1, 1, 2, 2], &[1, 1, 1, 1], &params);
    assert_eq!(
        rectified,
        vec![0.0, 0.0, 0.0, 0.0],
        "ref_conv must apply the node's fused activation, or it silently agrees with a \
         kernel that also skipped it",
    );

    params.activation = ConvActivation::Clip {
        min: -2.5,
        max: -1.5,
    };
    let clipped =
        crate::reference::ref_conv(&input, &weight, None, &[1, 1, 2, 2], &[1, 1, 1, 1], &params);
    assert_eq!(clipped, vec![-1.5, -2.0, -2.5, -2.5]);
}

// ── On-device numeric validation ────────────────────────────────────────
//
// `cargo test -p oxionnx-cuda --features gpu-tests conv` on a
// CUDA-capable host. `problem_from_params`/`pick_engine` above are
// exercised for free by every case here (`cuda_conv` calls both), so
// these tests additionally confirm the dispatched engine actually
// computes the right numbers, not merely that it runs.
//
// The CPU oracle below (`naive_conv2d_f64`) predates `crate::reference`
// having a `Conv` formula and is kept as its own from-scratch reference
// rather than switched over to `crate::reference::ref_conv` now that the
// latter exists: this module's tests check each of the three dispatched
// engines (`Conv1x1`/`DepthwiseConv`/`ImplicitGemmConv`) directly, and
// two independently-written oracles agreeing is stronger evidence than
// one oracle re-used to check itself. `reference::ref_conv` gets its own
// from-scratch validation, both as unit tests in `reference.rs` and
// end-to-end (through `try_cuda_dispatch`'s `OXIONNX_CUDA_VERIFY=1`
// wiring) in `oxionnx-cuda/tests/verify_path_gpu.rs`. It deliberately
// mirrors the *cross-correlation* convention (no 180-degree kernel
// flip) `oxicuda_dnn`'s shared `emit_standard_conv_body` /
// `emit_depthwise_pixel` kernel bodies use, `f64`-accumulated, O(naive)
// nested loops -- correct over fast.
#[cfg(feature = "gpu-tests")]
#[allow(clippy::needless_range_loop)]
mod gpu_numeric {
    use super::*;
    use crate::context::Activation;

    /// A small deterministic LCG (Knuth/MMIX multiplier) — avoids a
    /// `rand` dependency for throwaway, reproducible test data.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
        /// Uniform `f32` in `[lo, hi)`.
        fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = f64::from(self.next_u32()) / 4_294_967_296.0;
            (f64::from(lo) + (f64::from(hi) - f64::from(lo)) * unit) as f32
        }
    }

    /// Scalar convolution geometry (mirrors the fields a `ConvProblem`
    /// needs, plus the raw ONNX-style asymmetric-capable `pads` this
    /// crate accepts -- kept symmetric here since that is what
    /// `cuda_conv` can claim).
    #[derive(Clone, Copy)]
    struct ConvCase {
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        k: usize,
        r: usize,
        s: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dil_h: usize,
        dil_w: usize,
        group: usize,
        /// The optimizer-fused activation this case exercises. The
        /// independent `naive_conv2d_f64` reference below applies it
        /// itself, so a dispatch that dropped it (the pre-fix behaviour)
        /// fails these tests on real hardware rather than silently
        /// agreeing with an equally-forgetful oracle.
        activation: ConvActivation,
    }

    impl ConvCase {
        fn out_hw(self) -> (usize, usize) {
            let eff_r = self.dil_h * (self.r - 1) + 1;
            let eff_s = self.dil_w * (self.s - 1) + 1;
            let out_h = (self.h + 2 * self.pad_h - eff_r) / self.stride_h + 1;
            let out_w = (self.w + 2 * self.pad_w - eff_s) / self.stride_w + 1;
            (out_h, out_w)
        }

        fn params(self) -> ConvParams {
            ConvParams {
                strides: [self.stride_h, self.stride_w],
                pads: [self.pad_h, self.pad_w, self.pad_h, self.pad_w],
                dilations: [self.dil_h, self.dil_w],
                group: self.group,
                activation: self.activation,
            }
        }

        fn in_ch_per_group(self) -> usize {
            self.c / self.group
        }
    }

    /// Independent `f64`-accumulated NCHW cross-correlation reference.
    /// Indexing follows the same `[K, C/g, R, S]` filter layout ONNX
    /// (and this crate's `cuda_conv`) uses.
    fn naive_conv2d_f64(
        case: ConvCase,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let (out_h, out_w) = case.out_hw();
        let in_ch_per_group = case.in_ch_per_group();
        let out_ch_per_group = case.k / case.group;

        let mut out = vec![0.0_f32; case.n * case.k * out_h * out_w];
        for ni in 0..case.n {
            for ki in 0..case.k {
                let g = ki / out_ch_per_group;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0_f64;
                        for cg in 0..in_ch_per_group {
                            let ci = g * in_ch_per_group + cg;
                            for ri in 0..case.r {
                                let ih = oh as isize * case.stride_h as isize - case.pad_h as isize
                                    + ri as isize * case.dil_h as isize;
                                if ih < 0 || ih as usize >= case.h {
                                    continue;
                                }
                                let ih = ih as usize;
                                for si in 0..case.s {
                                    let iw = ow as isize * case.stride_w as isize
                                        - case.pad_w as isize
                                        + si as isize * case.dil_w as isize;
                                    if iw < 0 || iw as usize >= case.w {
                                        continue;
                                    }
                                    let iw = iw as usize;
                                    let in_idx = ((ni * case.c + ci) * case.h + ih) * case.w + iw;
                                    let f_idx =
                                        ((ki * in_ch_per_group + cg) * case.r + ri) * case.s + si;
                                    acc += f64::from(input[in_idx]) * f64::from(weight[f_idx]);
                                }
                            }
                        }
                        if let Some(bv) = bias {
                            acc += f64::from(bv[ki]);
                        }
                        let o_idx = ((ni * case.k + ki) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc as f32;
                    }
                }
            }
        }
        // The fused activation is part of what the node computes; see
        // `ConvActivation`. Written out longhand here rather than calling
        // `apply_conv_activation_host`, keeping this reference independent
        // of the code it checks.
        match case.activation {
            ConvActivation::None => {}
            ConvActivation::Relu => {
                for v in out.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            ConvActivation::Clip { min, max } => {
                for v in out.iter_mut() {
                    if *v < min {
                        *v = min;
                    } else if *v > max {
                        *v = max;
                    }
                }
            }
        }
        out
    }

    /// Real `CudaContext`, bypassing the `OXIONNX_CUDA` env-var opt-in
    /// gate (unit-tested separately in `context::tests`) -- still
    /// requires a real, working CUDA driver and device 0. Returns
    /// `None` when no driver / device is present, so each case skips
    /// and `--all-features` stays green on a CPU-only host (the
    /// OxiCUDA convention -- see `oxicuda-blas`'s `src/gpu_tests.rs`).
    fn gpu_ctx() -> Option<CudaContext> {
        CudaContext::try_new_with(Activation::Enabled)
    }

    /// Resolves which engine `cuda_conv` will pick for `case`, so each
    /// test can assert it is actually exercising the branch it claims
    /// to (rather than silently drifting onto a different one).
    fn engine_for(case: ConvCase) -> ConvEngine {
        let input_shape = [case.n, case.c, case.h, case.w];
        let weight_shape = [case.k, case.in_ch_per_group(), case.r, case.s];
        let problem = problem_from_params(&input_shape, &weight_shape, &case.params())
            .expect("ConvCase must describe a valid, non-declined convolution");
        pick_engine(&problem)
    }

    /// Runs `case` through the real `cuda_conv` dispatch on-device and
    /// checks the result against `naive_conv2d_f64`, using this
    /// crate's own `reference::compare` tolerance (tight: `ATOL=1e-4`,
    /// `RTOL=1e-3`).
    fn run_case_and_compare(case: ConvCase, with_bias: bool, seed: u64, tag: &str) {
        let Some(ctx) = gpu_ctx() else {
            eprintln!("no CUDA device present, skipping {tag}");
            return;
        };
        let (out_h, out_w) = case.out_hw();
        let in_ch_per_group = case.in_ch_per_group();

        let mut lcg = Lcg::new(seed);
        let in_data: Vec<f32> = (0..case.n * case.c * case.h * case.w)
            .map(|_| lcg.range_f32(-1.0, 1.0))
            .collect();
        let fil_data: Vec<f32> = (0..case.k * in_ch_per_group * case.r * case.s)
            .map(|_| lcg.range_f32(-1.0, 1.0))
            .collect();
        let bias_data: Option<Vec<f32>> =
            with_bias.then(|| (0..case.k).map(|_| lcg.range_f32(-0.5, 0.5)).collect());

        let input = Tensor::new(in_data.clone(), vec![case.n, case.c, case.h, case.w]);
        let weight = Tensor::new(
            fil_data.clone(),
            vec![case.k, in_ch_per_group, case.r, case.s],
        );
        let bias_tensor = bias_data
            .as_ref()
            .map(|b| Tensor::new(b.clone(), vec![case.k]));

        let params = case.params();
        let result = cuda_conv(&ctx, &input, &weight, bias_tensor.as_ref(), &params)
            .unwrap_or_else(|e| panic!("{tag}: cuda_conv hard-errored: {e}"))
            .unwrap_or_else(|| panic!("{tag}: cuda_conv declined a shape it must claim"));

        assert_eq!(
            result.shape,
            vec![case.n, case.k, out_h, out_w],
            "{tag}: output shape mismatch"
        );

        let expected = naive_conv2d_f64(case, &in_data, &fil_data, bias_data.as_deref());
        if let Err(e) = crate::reference::compare(&result.data, &expected) {
            panic!("{tag}: GPU output disagrees with the naive CPU oracle: {e}");
        }
    }

    #[test]
    fn conv1x1_no_bias_matches_naive_cpu_reference() {
        let case = ConvCase {
            n: 1,
            c: 24,
            h: 18,
            w: 26,
            k: 40,
            r: 1,
            s: 1,
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::None,
        };
        assert_eq!(engine_for(case), ConvEngine::Conv1x1);
        run_case_and_compare(case, false, 0x51ED_0000_0000_0001, "conv1x1_no_bias");
    }

    #[test]
    fn conv1x1_with_bias_matches_naive_cpu_reference() {
        // Exercises the host-side `add_bias_nchw` path: `Conv1x1::execute`
        // itself has no bias parameter at all.
        let case = ConvCase {
            n: 1,
            c: 32,
            h: 22,
            w: 15,
            k: 17,
            r: 1,
            s: 1,
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::None,
        };
        assert_eq!(engine_for(case), ConvEngine::Conv1x1);
        run_case_and_compare(case, true, 0x51ED_0000_0000_0002, "conv1x1_with_bias");
    }

    #[test]
    fn depthwise_dilated_with_bias_matches_naive_cpu_reference() {
        // Exercises DepthwiseConv AND the host-side `add_bias_nchw`
        // path (same reason as conv1x1_with_bias: no bias parameter on
        // `DepthwiseConv::execute`). `pad=2, dilation=2` is a
        // "same"-style dilated depthwise conv (output spatial size ==
        // input spatial size).
        let case = ConvCase {
            n: 1,
            c: 28,
            h: 21,
            w: 25,
            k: 28,
            r: 3,
            s: 3,
            pad_h: 2,
            pad_w: 2,
            stride_h: 1,
            stride_w: 1,
            dil_h: 2,
            dil_w: 2,
            group: 28,
            activation: ConvActivation::None,
        };
        assert_eq!(engine_for(case), ConvEngine::Depthwise);
        run_case_and_compare(
            case,
            true,
            0x51ED_0000_0000_0003,
            "depthwise_dilated_with_bias",
        );
    }

    #[test]
    fn implicit_gemm_3x3_stride1_with_bias_matches_naive_cpu_reference() {
        let case = ConvCase {
            n: 2,
            c: 19,
            h: 20,
            w: 28,
            k: 37,
            r: 3,
            s: 3,
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::None,
        };
        assert_eq!(engine_for(case), ConvEngine::ImplicitGemm);
        run_case_and_compare(
            case,
            true,
            0x51ED_0000_0000_0004,
            "implicit_gemm_3x3_stride1_with_bias",
        );
    }

    #[test]
    fn implicit_gemm_3x3_stride2_with_bias_matches_naive_cpu_reference() {
        let case = ConvCase {
            n: 1,
            c: 33,
            h: 27,
            w: 19,
            k: 21,
            r: 3,
            s: 3,
            pad_h: 1,
            pad_w: 1,
            stride_h: 2,
            stride_w: 2,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::None,
        };
        assert_eq!(engine_for(case), ConvEngine::ImplicitGemm);
        // Hand-verified output size: floor((27+2-2-1)/2)+1=14, floor((19+2-2-1)/2)+1=10.
        assert_eq!(case.out_hw(), (14, 10));
        run_case_and_compare(
            case,
            true,
            0x51ED_0000_0000_0005,
            "implicit_gemm_3x3_stride2_with_bias",
        );
    }

    // ── The fused activation, on real hardware ──────────────────────────
    //
    // These are the on-device half of the corruption regression: the
    // random weights make roughly half the raw outputs negative, so a
    // dispatch that dropped the fused `Relu` disagrees with
    // `naive_conv2d_f64` on ~half the tensor and the test fails loudly.

    /// The device epilogue path: `ImplicitGemmConv` applies its bias in
    /// the kernel, so the `Relu` is launched on the device, in place, on
    /// the same stream, before the readback.
    #[test]
    fn implicit_gemm_with_fused_relu_matches_naive_cpu_reference() {
        let case = ConvCase {
            n: 1,
            c: 23,
            h: 24,
            w: 18,
            k: 29,
            r: 3,
            s: 3,
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::Relu,
        };
        assert_eq!(engine_for(case), ConvEngine::ImplicitGemm);
        run_case_and_compare(
            case,
            true,
            0x51ED_0000_0000_0006,
            "implicit_gemm_fused_relu",
        );
    }

    /// The host epilogue path: `Conv1x1` has no bias parameter, so the
    /// bias is added on the host after the readback and the activation
    /// has to follow it there — `Relu(conv + bias)`, never
    /// `Relu(conv) + bias`. A dispatch that applied the activation on the
    /// device here would rectify *before* the bias and disagree wherever
    /// the bias pushes a negative pre-activation back above zero.
    #[test]
    fn conv1x1_with_bias_and_fused_relu_orders_the_epilogue_correctly() {
        let case = ConvCase {
            n: 1,
            c: 30,
            h: 19,
            w: 23,
            k: 26,
            r: 1,
            s: 1,
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::Relu,
        };
        assert_eq!(engine_for(case), ConvEngine::Conv1x1);
        run_case_and_compare(case, true, 0x51ED_0000_0000_0007, "conv1x1_bias_fused_relu");
    }

    /// The host `Clip` epilogue, on the `ImplicitGemmConv` engine (whose
    /// bias is already on the device, so this isolates "no device kernel
    /// for Clip" from "bias still owed on the host").
    #[test]
    fn implicit_gemm_with_fused_clip_matches_naive_cpu_reference() {
        let case = ConvCase {
            n: 1,
            c: 21,
            h: 17,
            w: 22,
            k: 25,
            r: 3,
            s: 3,
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 1,
            activation: ConvActivation::Clip {
                min: -0.5,
                max: 0.75,
            },
        };
        assert_eq!(engine_for(case), ConvEngine::ImplicitGemm);
        run_case_and_compare(
            case,
            true,
            0x51ED_0000_0000_0008,
            "implicit_gemm_fused_clip",
        );
    }

    /// The depthwise engine also owes a host bias add, so it takes the
    /// host epilogue for the same reason `Conv1x1` does.
    #[test]
    fn depthwise_with_bias_and_fused_relu_matches_naive_cpu_reference() {
        let case = ConvCase {
            n: 1,
            c: 26,
            h: 20,
            w: 24,
            k: 26,
            r: 3,
            s: 3,
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            group: 26,
            activation: ConvActivation::Relu,
        };
        assert_eq!(engine_for(case), ConvEngine::Depthwise);
        run_case_and_compare(
            case,
            true,
            0x51ED_0000_0000_0009,
            "depthwise_bias_fused_relu",
        );
    }
}
