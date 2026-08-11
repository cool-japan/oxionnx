//! [c3] Conv2D GPU/CPU parity over the *real* convolution inventory of the
//! three models OxiFace runs, plus the ragged and degenerate shapes those
//! models never produce.
//!
//! # Where this list comes from
//!
//! Not from memory, and not from the task description. Every entry below was
//! read out of the shipped `.onnx` files with a throwaway loader
//! (`oxionnx_proto::model::load` + `oxionnx::optimizer::shape_inference::infer_shapes`,
//! seeded with each graph's declared input shapes), which printed each `Conv`
//! node's resolved input shape, weight shape and spatial attributes. The
//! per-shape `count` fields are that dump's histogram, and
//! [`inventory_counts_match_the_models`] asserts they still add up to the node
//! counts the dump reported:
//!
//! | model | file | Conv nodes |
//! |---|---|---|
//! | InSwapper | `inswapper_128.onnx` | 20 |
//! | ArcFace   | `w600k_r50.onnx` | 53 |
//! | SCRFD     | `web/det_2.5g_fp16.onnx` | **57** |
//!
//! The SCRFD count is 57, not the 58 the task sheet quoted, and the deployed
//! web model is `det_2.5g_fp16`, not `det_10g` — both are reported as measured
//! rather than reconciled.
//!
//! ## The one place this list is a superset
//!
//! SCRFD declares a dynamic input (`[1, 3, None, None]`); seeded at the
//! detector's `640x640` (see `oxiface-detect`'s `ScrfdDetector`), static shape
//! inference resolves 36 of its 57 Conv nodes and gives up at the FPN
//! `Resize`/`Concat` boundary for the remaining 21 — six `[24, 24, 3, 3]` neck
//! convolutions and fifteen head convolutions (three strides x five layers).
//! Those 21 all run at one of the three FPN resolutions, `80x80`, `40x40` or
//! `20x20`, so they are enumerated here at **all three**. That is a superset of
//! the real graph, which is the safe direction: every node the model actually
//! executes is covered, and the extra cases are the same kernel at a different
//! spatial size.
//!
//! # What is compared
//!
//! `oxionnx_gpu::shaders::gpu_conv2d_implicit` against
//! `oxionnx_ops::conv::conv2d` — the rayon im2col+sgemm CPU operator the
//! engine falls back to. Two *independent* implementations: the GPU kernel
//! never materialises a column matrix, the CPU one is built entirely around
//! materialising it, and the unit tests in `shaders/conv2d.rs` additionally
//! check the same kernel against a from-the-definition six-loop reference.
//!
//! Tolerance is `1e-4` relative to the output's own magnitude. Per-*element*
//! relative error is not a meaningful bound for a `K = 9216` f32 reduction
//! whose result legitimately crosses zero; the magnitude-scaled form is.
//!
//! Every test skips (rather than fails) when no adapter is available.

use oxionnx_core::Tensor;
use oxionnx_gpu::shaders::{gpu_conv2d_implicit, ConvActivation};
use oxionnx_gpu::GpuContext;

/// One convolution shape as it appears in a model, with how many nodes share it.
struct ConvCase {
    label: &'static str,
    /// Nodes in the model with this exact shape and attributes.
    count: usize,
    /// `[N, C_in, H, W]`.
    input: [usize; 4],
    /// `[C_out, C_in, kH, kW]`.
    weight: [usize; 4],
    strides: [usize; 2],
    /// ONNX order: `[top, left, bottom, right]`.
    pads: [usize; 4],
    dilations: [usize; 2],
}

/// InSwapper (`inswapper_128.onnx`), 20 Conv nodes, all `group = 1`,
/// `dilation = 1`, all with bias.
///
/// The two `7x7` and the twelve `3x3` bottleneck convolutions carry
/// `pads = 0` because the graph pads reflectively with an explicit `Pad` node
/// first — hence the `134x134` and `34x34` inputs for `128x128` and `32x32`
/// outputs.
const INSWAPPER: &[ConvCase] = &[
    ConvCase {
        label: "inswapper stem 7x7 s1 p0",
        count: 1,
        input: [1, 3, 134, 134],
        weight: [128, 3, 7, 7],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "inswapper enc 3x3 s1 p1 @128",
        count: 1,
        input: [1, 128, 128, 128],
        weight: [256, 128, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "inswapper enc 3x3 s2 p1 @128",
        count: 1,
        input: [1, 256, 128, 128],
        weight: [512, 256, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "inswapper enc 3x3 s2 p1 @64",
        count: 1,
        input: [1, 512, 64, 64],
        weight: [1024, 512, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        // The dominant layer: 9.66 GMAC each, twelve of them per frame.
        label: "inswapper bottleneck 3x3 s1 p0 @32 (x12)",
        count: 12,
        input: [1, 1024, 34, 34],
        weight: [1024, 1024, 3, 3],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "inswapper dec 3x3 s1 p1 @64",
        count: 1,
        input: [1, 1024, 64, 64],
        weight: [512, 1024, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        // 19.33 GMAC — the single most expensive node in the model.
        label: "inswapper dec 3x3 s1 p1 @128",
        count: 1,
        input: [1, 512, 128, 128],
        weight: [256, 512, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "inswapper dec 3x3 s1 p1 @128 (256->128)",
        count: 1,
        input: [1, 256, 128, 128],
        weight: [128, 256, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "inswapper head 7x7 s1 p0",
        count: 1,
        input: [1, 128, 134, 134],
        weight: [3, 128, 7, 7],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
];

/// ArcFace (`w600k_r50.onnx`), 53 Conv nodes: an IR-SE-50 trunk of `3x3`
/// stride-1/stride-2 convolutions plus four `1x1` stride-2 downsample
/// shortcuts. All `group = 1`, `dilation = 1`, all with bias.
const ARCFACE: &[ConvCase] = &[
    ConvCase {
        label: "arcface stem 3x3 s1 p1 @112",
        count: 1,
        input: [1, 3, 112, 112],
        weight: [64, 3, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @112",
        count: 1,
        input: [1, 64, 112, 112],
        weight: [64, 64, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s2 p1 @112",
        count: 1,
        input: [1, 64, 112, 112],
        weight: [64, 64, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface shortcut 1x1 s2 p0 @112",
        count: 1,
        input: [1, 64, 112, 112],
        weight: [64, 64, 1, 1],
        strides: [2, 2],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @56 (x4)",
        count: 4,
        input: [1, 64, 56, 56],
        weight: [64, 64, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @56 (64->128)",
        count: 1,
        input: [1, 64, 56, 56],
        weight: [128, 64, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s2 p1 @56",
        count: 1,
        input: [1, 128, 56, 56],
        weight: [128, 128, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface shortcut 1x1 s2 p0 @56",
        count: 1,
        input: [1, 64, 56, 56],
        weight: [128, 64, 1, 1],
        strides: [2, 2],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @28 (x6)",
        count: 6,
        input: [1, 128, 28, 28],
        weight: [128, 128, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @28 (128->256)",
        count: 1,
        input: [1, 128, 28, 28],
        weight: [256, 128, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s2 p1 @28",
        count: 1,
        input: [1, 256, 28, 28],
        weight: [256, 256, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface shortcut 1x1 s2 p0 @28",
        count: 1,
        input: [1, 128, 28, 28],
        weight: [256, 128, 1, 1],
        strides: [2, 2],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @14 (x26)",
        count: 26,
        input: [1, 256, 14, 14],
        weight: [256, 256, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @14 (256->512)",
        count: 1,
        input: [1, 256, 14, 14],
        weight: [512, 256, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s2 p1 @14",
        count: 1,
        input: [1, 512, 14, 14],
        weight: [512, 512, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface shortcut 1x1 s2 p0 @14",
        count: 1,
        input: [1, 256, 14, 14],
        weight: [512, 256, 1, 1],
        strides: [2, 2],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "arcface 3x3 s1 p1 @7 (x4)",
        count: 4,
        input: [1, 512, 7, 7],
        weight: [512, 512, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
];

/// SCRFD (`web/det_2.5g_fp16.onnx`) at the detector's `640x640` input — the 36
/// Conv nodes static shape inference resolves. All `group = 1`,
/// `dilation = 1`, all with bias.
const SCRFD_RESOLVED: &[ConvCase] = &[
    ConvCase {
        label: "scrfd stem 3x3 s2 p1 @640",
        count: 1,
        input: [1, 3, 640, 640],
        weight: [12, 3, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s1 p1 @320 (12->12)",
        count: 1,
        input: [1, 12, 320, 320],
        weight: [12, 12, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s1 p1 @320 (12->24)",
        count: 1,
        input: [1, 12, 320, 320],
        weight: [24, 12, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s1 p1 @160 (x6)",
        count: 6,
        input: [1, 24, 160, 160],
        weight: [24, 24, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s2 p1 @160 (24->48)",
        count: 1,
        input: [1, 24, 160, 160],
        weight: [48, 24, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s1 p1 @80 (x9)",
        count: 9,
        input: [1, 48, 80, 80],
        weight: [48, 48, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd shortcut 1x1 s1 p0 @80 (24->48)",
        count: 1,
        input: [1, 24, 80, 80],
        weight: [48, 24, 1, 1],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s2 p1 @80 (48->48)",
        count: 1,
        input: [1, 48, 80, 80],
        weight: [48, 48, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s1 p1 @40 (x5)",
        count: 5,
        input: [1, 48, 40, 40],
        weight: [48, 48, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd shortcut 1x1 s1 p0 @40 (48->48)",
        count: 1,
        input: [1, 48, 40, 40],
        weight: [48, 48, 1, 1],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s2 p1 @40 (48->80)",
        count: 1,
        input: [1, 48, 40, 40],
        weight: [80, 48, 3, 3],
        strides: [2, 2],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd 3x3 s1 p1 @20 (x3)",
        count: 3,
        input: [1, 80, 20, 20],
        weight: [80, 80, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd shortcut 1x1 s1 p0 @20 (48->80)",
        count: 1,
        input: [1, 48, 20, 20],
        weight: [80, 48, 1, 1],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd lateral 1x1 s1 p0 @80 (48->24)",
        count: 1,
        input: [1, 48, 80, 80],
        weight: [24, 48, 1, 1],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd lateral 1x1 s1 p0 @40 (48->24)",
        count: 1,
        input: [1, 48, 40, 40],
        weight: [24, 48, 1, 1],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd lateral 1x1 s1 p0 @20 (80->24)",
        count: 1,
        input: [1, 80, 20, 20],
        weight: [24, 80, 1, 1],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
        dilations: [1, 1],
    },
    ConvCase {
        label: "scrfd neck 3x3 s1 p1 @20 (24->24)",
        count: 1,
        input: [1, 24, 20, 20],
        weight: [24, 24, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
    },
];

/// The three FPN resolutions SCRFD's unresolved neck/head convolutions run at.
const SCRFD_FPN_SIZES: [usize; 3] = [80, 40, 20];

/// `(C_out, C_in)` of each SCRFD neck/head convolution whose spatial size
/// static inference could not pin down. All are `3x3 s1 p1`.
///
/// Six `[24, 24]` neck convolutions plus five head layers per stride
/// (`24->64`, `64->64`, and the cls / bbox / kps projections `64->2`,
/// `64->8`, `64->20`) x three strides = 21 nodes.
const SCRFD_HEAD_LAYERS: &[(usize, usize)] =
    &[(24, 24), (64, 24), (64, 64), (2, 64), (8, 64), (20, 64)];

/// Deterministic, signed, non-monotonic fill — a plain `i % small` ramp hides
/// transposition bugs because many wrong indices carry the right value.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

fn tensor(shape: &[usize], seed: u32) -> Tensor {
    Tensor::new(fill(shape.iter().product(), seed), shape.to_vec())
}

/// Largest absolute difference, relative to the reference's own magnitude.
fn relative_error(got: &Tensor, want: &Tensor) -> (f32, usize) {
    let scale = want
        .data
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (g, e)) in got.data.iter().zip(want.data.iter()).enumerate() {
        let d = (g - e).abs() / scale;
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    (worst, worst_at)
}

/// Run one case against `oxionnx-ops` and assert agreement.
///
/// The activation is varied by case index so the fused epilogue is exercised
/// across the real inventory rather than only in the unit tests: the CPU
/// operator has no fused activation, so the reference applies it afterwards —
/// which is exactly the equivalence the fused path claims.
fn check_case(ctx: &GpuContext, case: &ConvCase, act: ConvActivation) {
    let input = tensor(&case.input, 2_654_435_761);
    let weight = tensor(&case.weight, 40_503);
    let bias = tensor(&[case.weight[0]], 97_711);

    let mut want = oxionnx_ops::conv::conv2d(
        &input,
        &weight,
        Some(&bias),
        case.strides,
        case.pads,
        case.dilations,
        1,
    );
    match act {
        ConvActivation::None => {}
        ConvActivation::Relu => {
            for v in want.data.iter_mut() {
                *v = v.max(0.0);
            }
        }
        ConvActivation::LeakyRelu(alpha) => {
            for v in want.data.iter_mut() {
                if *v < 0.0 {
                    *v *= alpha;
                }
            }
        }
        ConvActivation::Clip { min, max } => {
            for v in want.data.iter_mut() {
                *v = v.clamp(min, max);
            }
        }
    }

    let got = gpu_conv2d_implicit(
        ctx,
        &input,
        &weight,
        Some(&bias),
        case.strides,
        case.pads,
        case.dilations,
        1,
        act,
    )
    .unwrap_or_else(|| {
        panic!(
            "{}: the direct kernel declined a shape from a real model \
             (in={:?} w={:?} s={:?} p={:?} d={:?})",
            case.label, case.input, case.weight, case.strides, case.pads, case.dilations
        )
    });

    assert_eq!(got.shape, want.shape, "{}: output shape", case.label);
    let (err, at) = relative_error(&got, &want);
    assert!(
        err <= 1e-4,
        "{}: relative error {err} at index {at} (gpu={}, cpu={}) exceeds 1e-4",
        case.label,
        got.data[at],
        want.data[at],
    );
}

/// Cycle the fused activation across a model's cases so every variant is
/// checked against a real shape without tripling the runtime.
fn activation_for(index: usize) -> ConvActivation {
    match index % 4 {
        0 => ConvActivation::None,
        1 => ConvActivation::Relu,
        2 => ConvActivation::LeakyRelu(0.1),
        _ => ConvActivation::Clip { min: 0.0, max: 6.0 },
    }
}

fn run_model(model: &str, cases: &[ConvCase]) {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[c3] no wgpu adapter — skipping {model} conv parity");
        return;
    };
    for (i, case) in cases.iter().enumerate() {
        check_case(&ctx, case, activation_for(i));
    }
}

#[test]
fn inswapper_conv_inventory_matches_the_cpu_operator() {
    run_model("InSwapper", INSWAPPER);
}

#[test]
fn arcface_conv_inventory_matches_the_cpu_operator() {
    run_model("ArcFace", ARCFACE);
}

#[test]
fn scrfd_resolved_conv_inventory_matches_the_cpu_operator() {
    run_model("SCRFD", SCRFD_RESOLVED);
}

/// The 21 SCRFD neck/head convolutions whose spatial extent static shape
/// inference cannot resolve, run at every FPN resolution they could have.
#[test]
fn scrfd_head_convs_match_the_cpu_operator_at_every_fpn_scale() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[c3] no wgpu adapter — skipping SCRFD head conv parity");
        return;
    };
    let mut i = 0usize;
    for size in SCRFD_FPN_SIZES {
        for &(c_out, c_in) in SCRFD_HEAD_LAYERS {
            let case = ConvCase {
                label: "scrfd head 3x3 s1 p1",
                count: 0,
                input: [1, c_in, size, size],
                weight: [c_out, c_in, 3, 3],
                strides: [1, 1],
                pads: [1, 1, 1, 1],
                dilations: [1, 1],
            };
            check_case(&ctx, &case, activation_for(i));
            i += 1;
        }
    }
}

/// The inventory tables must keep matching the models they were read from.
///
/// A shape list that silently drifts is worse than no shape list: it would
/// keep passing while covering something the models no longer contain. These
/// totals are what the ONNX dump reported, node for node.
#[test]
fn inventory_counts_match_the_models() {
    let inswapper: usize = INSWAPPER.iter().map(|c| c.count).sum();
    let arcface: usize = ARCFACE.iter().map(|c| c.count).sum();
    let scrfd_resolved: usize = SCRFD_RESOLVED.iter().map(|c| c.count).sum();
    let scrfd_unresolved = SCRFD_HEAD_LAYERS.len() * SCRFD_FPN_SIZES.len();

    assert_eq!(inswapper, 20, "InSwapper Conv nodes");
    assert_eq!(arcface, 53, "ArcFace Conv nodes");
    assert_eq!(scrfd_resolved, 36, "SCRFD Conv nodes with resolved shapes");
    // 36 resolved + 21 unresolved = 57 nodes; the 21 are covered by 18
    // enumerated (layer, scale) combinations, a superset of the real graph.
    assert_eq!(scrfd_resolved + 21, 57, "SCRFD Conv nodes");
    assert_eq!(scrfd_unresolved, 18, "enumerated SCRFD head cases");
}

/// Shapes the three models never produce but a fourth model might: ragged in
/// every tiled axis at once, one-pixel outputs, single channels, and a batch.
///
/// The macro-tile is `64x64` with a 16-deep channel tile, so the sizes here
/// are chosen to land one element either side of each of those boundaries.
#[test]
fn ragged_and_degenerate_shapes_match_the_cpu_operator() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[c3] no wgpu adapter — skipping ragged conv parity");
        return;
    };
    let cases: &[ConvCase] = &[
        ConvCase {
            label: "1x1 output, 1 channel in and out",
            count: 0,
            input: [1, 1, 3, 3],
            weight: [1, 1, 3, 3],
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
        },
        ConvCase {
            label: "ragged M (65) and N (65) at once",
            count: 0,
            input: [1, 17, 65, 65],
            weight: [65, 17, 3, 3],
            strides: [1, 1],
            pads: [1, 1, 1, 1],
            dilations: [1, 1],
        },
        ConvCase {
            label: "exactly one macro tile (64 x 64)",
            count: 0,
            input: [1, 16, 8, 8],
            weight: [64, 16, 3, 3],
            strides: [1, 1],
            pads: [1, 1, 1, 1],
            dilations: [1, 1],
        },
        ConvCase {
            label: "one past a macro tile in N only",
            count: 0,
            input: [1, 5, 13, 5],
            weight: [3, 5, 3, 3],
            strides: [1, 1],
            pads: [1, 1, 1, 1],
            dilations: [1, 1],
        },
        ConvCase {
            label: "non-square kernel and stride, asymmetric pads",
            count: 0,
            input: [2, 6, 11, 13],
            weight: [9, 6, 5, 3],
            strides: [2, 3],
            pads: [2, 1, 0, 2],
            dilations: [1, 1],
        },
        ConvCase {
            label: "dilated 3x3, batch of 2",
            count: 0,
            input: [2, 7, 15, 15],
            weight: [5, 7, 3, 3],
            strides: [1, 1],
            pads: [2, 2, 2, 2],
            dilations: [2, 2],
        },
        ConvCase {
            label: "wide input, single output channel",
            count: 0,
            input: [1, 33, 4, 130],
            weight: [1, 33, 1, 1],
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
        },
    ];
    for (i, case) in cases.iter().enumerate() {
        check_case(&ctx, case, activation_for(i));
    }
}

/// Grouped convolution must decline so the hybrid im2col path in `compute.rs`
/// handles it — the direct kernel does not implement `group > 1`.
#[test]
fn grouped_convolution_declines_to_the_hybrid_path() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[c3] no wgpu adapter — skipping grouped conv decline check");
        return;
    };
    let input = tensor(&[1, 16, 8, 8], 5);
    for group in [2usize, 4, 16] {
        let weight = tensor(&[16, 16 / group, 3, 3], 7);
        assert!(
            gpu_conv2d_implicit(
                &ctx,
                &input,
                &weight,
                None,
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                group,
                ConvActivation::None,
            )
            .is_none(),
            "group={group} must decline"
        );
    }
}

/// The whole-path entry point must still produce the same answer as the CPU
/// operator, including its size gate: a convolution below the GEMM-FLOP
/// threshold declines to the CPU exactly as it always did, and one above it
/// now runs on the direct kernel.
#[test]
fn the_dispatch_entry_point_keeps_its_threshold_and_its_answer() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[c3] no wgpu adapter — skipping entry point check");
        return;
    };
    // Below threshold (M*K*N = 1*1*4): declines, as before.
    let tiny_in = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
    let tiny_w = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    assert!(
        oxionnx_gpu::gpu_conv2d(
            &ctx,
            &tiny_in,
            &tiny_w,
            None,
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1
        )
        .is_none(),
        "a below-threshold conv must still decline to the CPU"
    );

    // Above threshold: 64 * 576 * 3136 = 115.6 MFLOP.
    let input = tensor(&[1, 64, 56, 56], 2_654_435_761);
    let weight = tensor(&[64, 64, 3, 3], 40_503);
    let bias = tensor(&[64], 97_711);
    let want = oxionnx_ops::conv::conv2d(
        &input,
        &weight,
        Some(&bias),
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
    );
    let got = oxionnx_gpu::gpu_conv2d(
        &ctx,
        &input,
        &weight,
        Some(&bias),
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
    )
    .expect("an above-threshold conv must dispatch");
    let (err, at) = relative_error(&got, &want);
    assert!(err <= 1e-4, "entry point: relative error {err} at {at}");
}

/// The un-fused entry point must **not** apply an activation.
///
/// `src/session/gpu_dispatch.rs` applies the fused `activation` attribute on
/// the host after calling it, so fusing one in here too would apply it twice —
/// silently squaring a LeakyRelu slope. This pins the contract from the
/// oxionnx-gpu side, where the fix would have to be made.
#[test]
fn the_unfused_entry_point_leaves_negative_outputs_alone() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[c3] no wgpu adapter — skipping un-fused activation check");
        return;
    };
    let input = tensor(&[1, 64, 56, 56], 2_654_435_761);
    let weight = tensor(&[64, 64, 3, 3], 40_503);
    let got = oxionnx_gpu::gpu_conv2d(&ctx, &input, &weight, None, [1, 1], [1, 1, 1, 1], [1, 1], 1)
        .expect("an above-threshold conv must dispatch");
    assert!(
        got.data.iter().any(|v| *v < 0.0),
        "gpu_conv2d must not fuse an activation: the dispatcher applies it"
    );
}
