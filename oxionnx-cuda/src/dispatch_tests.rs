//! Unit tests for [`crate::try_cuda_dispatch`] and its helpers.
//!
//! Split out of `lib.rs` purely for size: the two enumerations these tests
//! pin -- `all_op_kinds()`, one entry per `OpKind` variant, and
//! `claimable_ops()`, one per dispatch arm that can claim a node -- are a
//! thousand lines between them, and keeping them in the crate root pushed it
//! past this workspace's 2000-line-per-file limit. Nothing else changes:
//! `#[cfg(test)] mod dispatch_tests;` in `lib.rs` makes this the same
//! crate-root-child module it was as an inline `mod tests`, with the same
//! access to private items (`apply_gemm_bias`, `transpose_2d_batched`,
//! `initializer_id`) that several of these tests need.

use crate::*;
use oxionnx_core::graph::{Attributes, Node, OpKind};

fn make_node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: "test_node".to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

// ── is_supported_op ⇄ try_cuda_dispatch agreement ───────────────────────

/// The ops `try_cuda_dispatch` can actually claim, transcribed from its
/// `match &node.op` arms.
///
/// Read the dispatch match top-to-bottom and this list must fall out of it:
/// every arm that can return `Ok(Some(_))` for at least one node
/// configuration appears here, and nothing else does.
fn claimable_ops() -> Vec<OpKind> {
    vec![
        // `OpKind::MatMul | OpKind::Gemm` arm → matmul::cuda_matmul
        OpKind::MatMul,
        OpKind::Gemm,
        // `OpKind::Conv` arm → conv::cuda_conv, which dispatches directly
        // to oxicuda-dnn's Conv1x1 / DepthwiseConv / ImplicitGemmConv.
        // Like every other arm here it can still decline an individual
        // node whose configuration is out of range (asymmetric pads, a
        // non-4-D shape, ...) -- claimable is an op-kind property, not a
        // per-node promise (see `is_supported_op`'s "Necessary, not
        // sufficient").
        OpKind::Conv,
        // unary activation arm → elementwise::cuda_elementwise
        OpKind::Relu,
        OpKind::Sigmoid,
        OpKind::Gelu,
        OpKind::Tanh,
        OpKind::Exp,
        OpKind::Sqrt,
        OpKind::Abs,
        OpKind::Neg,
        OpKind::Log,
        OpKind::Ceil,
        OpKind::Floor,
        OpKind::HardSigmoid,
        OpKind::HardSwish,
        OpKind::SiLU,
        OpKind::Softplus,
        OpKind::LeakyRelu,
        // binary arm → elementwise::cuda_binary_elementwise
        OpKind::Add,
        OpKind::Sub,
        OpKind::Mul,
        OpKind::Div,
        // reduction arm → reduce::cuda_reduce
        OpKind::ReduceSum,
        OpKind::ReduceMax,
        // softmax arm → softmax::cuda_softmax
        OpKind::Softmax,
        // NOTE: `OpKind::Conv` has a dispatch arm, `conv::cuda_conv` now
        // computes real answers for some shapes, and that arm's output is
        // shadow-verified against `reference::ref_conv` -- but
        // `is_supported_op` still reports it unsupported (routing
        // production traffic through it is a separate decision from
        // having a verification oracle -- see the `conv` module docs), so
        // it is NOT claimable and must not appear here.
    ]
}

/// Every unit variant of `OpKind` (i.e. excluding `OpKind::Unknown(_)`).
///
/// Enumerated exhaustively so that `is_supported_op` can be pinned to
/// *exactly* the claimable set — an op accidentally added to the predicate
/// without a matching dispatch arm makes this test fail.
fn all_op_kinds() -> Vec<OpKind> {
    vec![
        OpKind::MatMul,
        OpKind::Gemm,
        OpKind::Add,
        OpKind::Sub,
        OpKind::Mul,
        OpKind::Div,
        OpKind::Pow,
        OpKind::Sqrt,
        OpKind::Reciprocal,
        OpKind::Neg,
        OpKind::ReduceMean,
        OpKind::ReduceSum,
        OpKind::ReduceMax,
        OpKind::ReduceMin,
        OpKind::ReduceProd,
        OpKind::ArgMax,
        OpKind::ArgMin,
        OpKind::CumSum,
        OpKind::Range,
        OpKind::TopK,
        OpKind::Softmax,
        OpKind::LayerNorm,
        OpKind::GroupNorm,
        OpKind::BatchNorm,
        OpKind::Gelu,
        OpKind::Relu,
        OpKind::Sigmoid,
        OpKind::Tanh,
        OpKind::Erf,
        OpKind::SiLU,
        OpKind::HardSigmoid,
        OpKind::HardSwish,
        OpKind::RMSNorm,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Squeeze,
        OpKind::Unsqueeze,
        OpKind::Flatten,
        OpKind::Concat,
        OpKind::Slice,
        OpKind::Expand,
        OpKind::Split,
        OpKind::Tile,
        OpKind::Gather,
        OpKind::GatherElements,
        OpKind::Where,
        OpKind::ScatterElements,
        OpKind::ScatterND,
        OpKind::Conv,
        OpKind::MaxPool,
        OpKind::AveragePool,
        OpKind::Pad,
        OpKind::LeakyRelu,
        OpKind::PRelu,
        OpKind::Resize,
        OpKind::GlobalAveragePool,
        OpKind::GlobalMaxPool,
        OpKind::QuantizeLinear,
        OpKind::DequantizeLinear,
        OpKind::Identity,
        OpKind::Cast,
        OpKind::Shape,
        OpKind::Constant,
        OpKind::Clip,
        OpKind::Abs,
        OpKind::Log,
        OpKind::Exp,
        OpKind::Ceil,
        OpKind::Floor,
        OpKind::Round,
        OpKind::Sign,
        OpKind::Mod,
        OpKind::BitShift,
        OpKind::Sin,
        OpKind::Cos,
        OpKind::Tan,
        OpKind::Asin,
        OpKind::Acos,
        OpKind::Atan,
        OpKind::Sinh,
        OpKind::Cosh,
        OpKind::Asinh,
        OpKind::Acosh,
        OpKind::Atanh,
        OpKind::VariadicMin,
        OpKind::VariadicMax,
        OpKind::VariadicMean,
        OpKind::VariadicSum,
        OpKind::Equal,
        OpKind::Greater,
        OpKind::GreaterOrEqual,
        OpKind::Less,
        OpKind::LessOrEqual,
        OpKind::And,
        OpKind::Or,
        OpKind::Xor,
        OpKind::Not,
        OpKind::IsInf,
        OpKind::IsNaN,
        OpKind::NonZero,
        OpKind::ConstantOfShape,
        OpKind::EyeLike,
        OpKind::Trilu,
        OpKind::LogSoftmax,
        OpKind::Softplus,
        OpKind::Softsign,
        OpKind::Mish,
        OpKind::Celu,
        OpKind::Elu,
        OpKind::Selu,
        OpKind::ThresholdedRelu,
        OpKind::InstanceNorm,
        OpKind::LpNorm,
        OpKind::MeanVarianceNormalization,
        OpKind::Dropout,
        OpKind::DepthToSpace,
        OpKind::SpaceToDepth,
        OpKind::ReverseSequence,
        OpKind::GatherND,
        OpKind::OneHot,
        OpKind::Compress,
        OpKind::Unique,
        OpKind::Einsum,
        OpKind::ConvTranspose,
        OpKind::NonMaxSuppression,
        OpKind::LSTM,
        OpKind::GRU,
        OpKind::Attention,
        OpKind::MultiHeadAttention,
        OpKind::RotaryEmbedding,
        OpKind::GridSample,
        OpKind::RoiAlign,
        OpKind::If,
        OpKind::Loop,
        OpKind::Scan,
        OpKind::LinearClassifier,
        OpKind::LinearRegressor,
        OpKind::Normalizer,
        OpKind::Scaler,
        OpKind::LabelEncoder,
        OpKind::TreeEnsembleClassifier,
        OpKind::TreeEnsembleRegressor,
        OpKind::SVMClassifier,
        OpKind::SVMRegressor,
        OpKind::TfIdfVectorizer,
        OpKind::StringNormalizer,
        OpKind::DFT,
        OpKind::STFT,
        OpKind::BlackmanWindow,
        OpKind::HannWindow,
        OpKind::HammingWindow,
        OpKind::MelWeightMatrix,
        OpKind::Bernoulli,
        OpKind::ReduceL1,
        OpKind::ReduceL2,
        OpKind::ReduceLogSum,
        OpKind::ReduceLogSumExp,
        OpKind::ReduceSumSquare,
        OpKind::BitwiseAnd,
        OpKind::BitwiseOr,
        OpKind::BitwiseXor,
        OpKind::BitwiseNot,
        OpKind::Size,
        OpKind::Hardmax,
        OpKind::Shrink,
        OpKind::ConvAddRelu,
    ]
}

/// `is_supported_op` must return `true` for **exactly** the ops that
/// `try_cuda_dispatch` can claim — no more, no fewer.
///
/// This is the contract `oxionnx::execution_providers::decide_placement`
/// relies on to avoid an upload → dispatch → fence → readback round-trip
/// for an op CUDA was never going to handle.
#[test]
fn is_supported_op_matches_dispatch_arms() {
    let claimable = claimable_ops();

    // 1. Every claimable op is reported supported.
    for op in &claimable {
        assert!(
            is_supported_op(op),
            "{op:?} has a live try_cuda_dispatch arm but is_supported_op says false",
        );
    }

    // 2. Nothing outside the claimable set is reported supported.
    //    Sweeping every OpKind unit variant makes this an "exactly" check.
    let all = all_op_kinds();
    for op in &all {
        let expected = claimable.contains(op);
        assert_eq!(
            is_supported_op(op),
            expected,
            "is_supported_op({op:?}) disagrees with the try_cuda_dispatch match arms",
        );
    }

    // 3. Guard the enumeration itself: if `OpKind` grows a variant and
    //    `all_op_kinds` is not updated, the arity check below trips.
    assert_eq!(
        all.len(),
        166,
        "OpKind gained/lost a unit variant — update all_op_kinds() and re-audit \
         is_supported_op against the try_cuda_dispatch match arms",
    );
    assert_eq!(
        claimable.len(),
        26,
        "claimable_ops() changed — re-audit against the try_cuda_dispatch match arms",
    );
}

/// Every op `try_cuda_dispatch` can claim must have a live [`reference`] oracle
/// behind the `verify_or_fallback` call in its arm.
///
/// Without this, an op added to a dispatch arm with no matching `reference::ref_*`
/// case doesn't fail loudly: `verify_or_fallback`'s oracle closure returns `None`,
/// [`reference::shadow_verify`] treats that as "the oracle has no formula, skip the
/// check" and logs a `warn!` — which only a human staring at a real CUDA machine's
/// logs under `OXIONNX_CUDA_VERIFY=1` would ever see. This test makes that gap fail
/// on every host, including this one with no GPU, by driving the same
/// `claimable_ops()` list `is_supported_op_matches_dispatch_arms` already pins so the
/// two enumerations cannot silently drift apart.
///
/// The op families split three ways by how their oracle is *shaped*, and each is
/// checked the strongest way its shape allows:
///
/// * `UNARY_OPS`/`BINARY_OPS`/`REDUCE_OPS` — one `OpKind`-dispatched oracle per
///   family ([`reference::ref_unary`] / [`reference::ref_binary`] /
///   [`reference::ref_reduce`]), each returning `Option`. A missing formula *is*
///   the `None` that `shadow_verify` would silently skip on, so `.is_some()` is
///   exactly the right check.
/// * `CONV_OPS` — [`reference::ref_conv`] takes no `OpKind` (there is one `Conv`
///   formula) *and* returns a plain `Vec<f32>`, so there is no `None` to probe;
///   deleting it would be a compile error rather than a silent skip. The
///   equivalent-strength check is therefore to run it on a hand-computable
///   problem and assert the answer, which additionally fails if it is ever gutted
///   into a shape-correct stub (all zeros, bias dropped, input echoed) — the
///   silent-fabrication failure mode this whole test family exists to catch.
/// * `NO_OPKIND_NEEDED` — `MatMul`/`Gemm` ([`reference::ref_matmul`]) and
///   `Softmax` ([`reference::ref_softmax`]) are likewise `OpKind`-free and
///   non-`Option`, and unlike `ref_conv` are already exercised against
///   hand-computed constants by their own dedicated `reference::tests` cases and
///   end-to-end by `tests/verify_path_gpu.rs`; they are listed here only so that
///   removing one from `claimable_ops()` still trips the partition check below.
#[test]
fn oracle_covers_every_op_try_cuda_dispatch_can_claim() {
    const UNARY_OPS: &[OpKind] = &[
        OpKind::Relu,
        OpKind::Sigmoid,
        OpKind::Gelu,
        OpKind::Tanh,
        OpKind::Exp,
        OpKind::Sqrt,
        OpKind::Abs,
        OpKind::Neg,
        OpKind::Log,
        OpKind::Ceil,
        OpKind::Floor,
        OpKind::HardSigmoid,
        OpKind::HardSwish,
        OpKind::SiLU,
        OpKind::Softplus,
        OpKind::LeakyRelu,
    ];
    const BINARY_OPS: &[OpKind] = &[OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div];
    const REDUCE_OPS: &[OpKind] = &[OpKind::ReduceSum, OpKind::ReduceMax];
    const CONV_OPS: &[OpKind] = &[OpKind::Conv];
    const NO_OPKIND_NEEDED: &[OpKind] = &[OpKind::MatMul, OpKind::Gemm, OpKind::Softmax];

    let claimable = claimable_ops();
    for op in &claimable {
        if UNARY_OPS.contains(op) {
            assert!(
                reference::ref_unary(op, 0.5).is_some(),
                "{op:?} is claimable by the unary elementwise dispatch arm but \
                 reference::ref_unary has no formula for it",
            );
        } else if BINARY_OPS.contains(op) {
            assert!(
                reference::ref_binary(op, 1.0, 2.0).is_some(),
                "{op:?} is claimable by the binary elementwise dispatch arm but \
                 reference::ref_binary has no formula for it",
            );
        } else if REDUCE_OPS.contains(op) {
            assert!(
                reference::ref_reduce(op, &[1.0, 2.0, 3.0, 4.0], &[4], 0).is_some(),
                "{op:?} is claimable by the reduce dispatch arm but reference::ref_reduce \
                 has no formula for it",
            );
        } else if CONV_OPS.contains(op) {
            // A `[1, 1, 2, 2]` input under a `[1, 1, 2, 2]` filter, no padding,
            // unit stride/dilation, one group: exactly one output element, whose
            // value is a four-term dot product with a decade-separated filter, so
            // every contributing term is visible in the total's digits.
            //   1*1 + 2*10 + 3*100 + 4*1000 = 4321, + bias 5 = 4326.
            // Decade separation is deliberate: a stub that drops the bias reads
            // 4321, one that transposes the filter reads 4213 (1*1 + 2*100 +
            // 3*10 + 4*1000 = 4231 -- either way, not 4326), one that returns
            // zeros or echoes the input misses by orders of magnitude. Any of
            // those is a distinguishable failure rather than a near-miss.
            let params = conv::ConvParams {
                strides: [1, 1],
                pads: [0, 0, 0, 0],
                dilations: [1, 1],
                group: 1,
            };
            let got = reference::ref_conv(
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 10.0, 100.0, 1000.0],
                Some(&[5.0]),
                &[1, 1, 2, 2],
                &[1, 1, 2, 2],
                &params,
            );
            assert_eq!(
                got,
                vec![4326.0_f32],
                "{op:?} is claimable by the conv dispatch arm, whose verify_or_fallback \
                 oracle is reference::ref_conv -- but ref_conv does not compute the \
                 hand-verified answer for a 2x2 single-channel convolution with bias",
            );
        } else {
            assert!(
                NO_OPKIND_NEEDED.contains(op),
                "{op:?} is claimable but not classified into any op-family list in this \
                 test — add it to one of the lists above so oracle coverage stays pinned",
            );
        }
    }

    // The five lists above partition `claimable_ops()` exactly; this catches an op
    // quietly removed from one list (rather than from `claimable_ops()` itself, which
    // `is_supported_op_matches_dispatch_arms` already pins at 26).
    assert_eq!(
        UNARY_OPS.len()
            + BINARY_OPS.len()
            + REDUCE_OPS.len()
            + CONV_OPS.len()
            + NO_OPKIND_NEEDED.len(),
        claimable.len(),
        "the op-family lists in this test no longer partition claimable_ops() exactly",
    );
}

/// `Conv` is advertised as CUDA-supported, so
/// `oxionnx::execution_providers::decide_placement` routes convolutions
/// to CUDA rather than skipping it on the strength of the predicate alone.
///
/// Kept as its own test — rather than left implicit in
/// `is_supported_op_matches_dispatch_arms`'s sweep — because this one bit
/// is what actually turns CUDA convolution on for every downstream caller
/// (`oxionnx`'s placement logic, and through it `oxiface`'s SCRFD /
/// ArcFace / InSwapper graphs, which are overwhelmingly `Conv`). A
/// regression that silently flips it back would otherwise read as a
/// one-line arity edit in a sweeping test rather than as what it is: all
/// CUDA convolution acceleration switching off.
///
/// The claim is not vacuous — that `cuda_conv` really computes a correct
/// convolution through this exact path is what the on-device suites prove
/// (`conv::tests::gpu_numeric` against a from-scratch oracle per engine,
/// `tests/verify_path_gpu.rs` end-to-end through `try_cuda_dispatch` with
/// `OXIONNX_CUDA_VERIFY=1` live, and
/// `conv_claimed_by_the_pre_filter_is_actually_dispatched` below, which
/// walks the production pre-filter → dispatch sequence this predicate
/// gates).
#[test]
fn conv_is_advertised_as_supported() {
    assert!(
        is_supported_op(&OpKind::Conv),
        "Conv must be advertised: conv::cuda_conv dispatches directly to oxicuda-dnn's \
         Conv1x1 / DepthwiseConv / ImplicitGemmConv engines, so decide_placement must be \
         allowed to route convolutions to CUDA",
    );
    assert!(
        claimable_ops().contains(&OpKind::Conv),
        "Conv is advertised but missing from claimable_ops() — the two enumerations this \
         module keeps honest against each other have drifted apart",
    );
}

/// `Unknown` ops can never be claimed.
#[test]
fn unknown_op_is_not_supported() {
    assert!(!is_supported_op(&OpKind::Unknown("Frobnicate".to_string())));
}

/// The predicate must be pure and side-effect free — callable without a device.
#[test]
fn is_supported_op_needs_no_cuda_device() {
    // No CudaContext is constructed anywhere in this test.
    for op in all_op_kinds() {
        let _ = is_supported_op(&op);
    }
}

/// Validates that try_cuda_dispatch returns Ok(None) for unsupported ops
/// when no CUDA context is available (unit test only touches the match arm).
#[test]
fn dispatch_unknown_op_returns_none() {
    // Without a real CUDA device we can only test the None-returning path.
    // We verify the dispatch fn returns None for an op that has no CUDA kernel.
    let node = make_node(OpKind::Identity, &["x"], &["y"]);
    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut intermediates: HashMap<String, Tensor> = HashMap::new();
    let t = Tensor::new(vec![1.0f32], vec![1]);
    intermediates.insert("x".to_string(), t);

    // We cannot construct a real CudaContext in CI, so we skip the actual
    // dispatch and just verify the type signature compiles.
    let _ = &node;
    let _ = &weights;
    let _ = &intermediates;
}

#[test]
fn cuda_context_try_new_no_panic() {
    // try_new must never panic — it should return None if no GPU present.
    let _ctx = CudaContext::try_new();
}

#[test]
fn cuda_error_displays_correctly() {
    let e = CudaError::Ptx("bad ptx".to_string());
    let s = format!("{e}");
    assert!(
        s.contains("bad ptx"),
        "Expected error message to contain 'bad ptx', got: {s}"
    );
}

#[test]
fn cuda_error_maps_to_onnx_internal() {
    let e = CudaError::Shape {
        op: "Conv",
        msg: "wrong shape".to_string(),
    };
    let onnx_err: OnnxError = e.into();
    match onnx_err {
        OnnxError::Internal(msg) => {
            assert!(
                msg.contains("wrong shape"),
                "Expected 'wrong shape' in: {msg}"
            );
        }
        other => panic!("Expected OnnxError::Internal, got: {other:?}"),
    }
}

// ── apply_gemm_bias (finding a8-4): every spec-legal broadcastable `C` ──
//
// ONNX Gemm's `C` may be unidirectionally broadcastable to `[M, N]` as a true
// scalar (`[]` or `[1]`), `[N]`, `[M, 1]`, or `[M, N]`. The pre-fix code only
// handled `[N]` and `[M, N]`; `[M, 1]` (M != N) and a genuine scalar (N != 1)
// silently added nothing. Each case below is hand-verified.

#[test]
fn gemm_bias_row_broadcast_n_shape() {
    // bias = [N] = [10, 20, 30], M=2, N=3, beta=1.0 — broadcasts across every row.
    let mut out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert!(apply_gemm_bias(
        &mut out,
        &[10.0, 20.0, 30.0],
        &[3],
        2,
        3,
        1.0
    ));
    assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn gemm_bias_one_by_n_shape_matches_plain_n_shape() {
    // bias = [1, N] = [[10, 20, 30]] — same broadcast as the plain [N] case above,
    // exercised through the rank-2-with-leading-1 code path instead of rank-1.
    let mut out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert!(apply_gemm_bias(
        &mut out,
        &[10.0, 20.0, 30.0],
        &[1, 3],
        2,
        3,
        1.0
    ));
    assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn gemm_bias_full_m_by_n_matrix() {
    // bias = [M, N] = [[100,200,300],[400,500,600]], M=2, N=3, beta=1.0.
    let mut out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
    assert!(apply_gemm_bias(&mut out, &bias, &[2, 3], 2, 3, 1.0));
    assert_eq!(out, vec![101.0, 202.0, 303.0, 404.0, 505.0, 606.0]);
}

#[test]
fn gemm_bias_m_by_one_column_broadcast_with_m_ne_n() {
    // The exact a8-4 regression case: bias = [M, 1] = [[7],[70]], M=2, N=3 (M != N),
    // beta=1.0. Before the fix, neither `bias.len() == n` (2 != 3) nor
    // `bias.len() == m*n` (2 != 6) matched, so the bias was silently dropped.
    let mut out = vec![0.0; 6];
    assert!(apply_gemm_bias(&mut out, &[7.0, 70.0], &[2, 1], 2, 3, 1.0));
    assert_eq!(out, vec![7.0, 7.0, 7.0, 70.0, 70.0, 70.0]);
}

#[test]
fn gemm_bias_true_scalar_broadcasts_to_every_element_with_n_ne_one() {
    // The other a8-4 regression case: bias = [1] (a true scalar), M=2, N=3 (N != 1),
    // beta=2.0. Before the fix, `bias.len() == n` (1 != 3) and `bias.len() == m*n`
    // (1 != 6) both failed, so the bias was silently dropped.
    let mut out = vec![0.0; 6];
    assert!(apply_gemm_bias(&mut out, &[5.0], &[1], 2, 3, 2.0));
    assert_eq!(out, vec![10.0; 6]);
}

#[test]
fn gemm_bias_rank_zero_scalar_broadcasts_too() {
    // A genuine ONNX scalar tensor has shape `[]` (rank 0), not `[1]`.
    let mut out = vec![0.0; 4];
    assert!(apply_gemm_bias(&mut out, &[9.0], &[], 2, 2, 1.0));
    assert_eq!(out, vec![9.0; 4]);
}

#[test]
fn gemm_bias_declines_an_incompatible_shape_leaving_out_untouched() {
    // bias = [5] against N=3: neither equal nor 1 — not unidirectionally broadcastable.
    let mut out = vec![1.0, 2.0, 3.0];
    let untouched = out.clone();
    assert!(!apply_gemm_bias(
        &mut out,
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[5],
        1,
        3,
        1.0
    ));
    assert_eq!(
        out, untouched,
        "a declined bias must leave `out` unmodified"
    );
}

#[test]
fn gemm_bias_declines_when_the_data_is_shorter_than_its_declared_shape() {
    // bias_shape claims 3 elements (a malformed model: shape/data length mismatch).
    let mut out = vec![1.0, 2.0, 3.0];
    let untouched = out.clone();
    assert!(!apply_gemm_bias(&mut out, &[1.0, 2.0], &[3], 1, 3, 1.0));
    assert_eq!(out, untouched);
}

#[test]
fn gemm_bias_row_broadcast_applies_across_every_stacked_batch_slice() {
    // `out` may stack several batch slices (`out.len() == batch * m * n`); a row-
    // broadcast bias must repeat for every row across every slice, matching the
    // pre-fix behaviour for this already-supported shape (a8-4 regression guard).
    let mut out = vec![0.0; 8]; // batch=2, m=2, n=2 -> 4 rows total.
    assert!(apply_gemm_bias(&mut out, &[1.0, 2.0], &[2], 2, 2, 1.0));
    assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

// ── Cross-thread dispatch (`CudaContext` thread affinity) ──────────────
//
// On-device only: run with `cargo test -p oxionnx-cuda --features
// gpu-tests` on a machine with a real NVIDIA GPU. These reproduce, and
// guard against a regression of, the exact scenario `activate_context`'s
// doc comment describes: a `CudaContext` built on one OS thread,
// dispatched from a *different* thread after the building thread has
// already exited -- exactly `oxiface-convert`'s
// `Converter::load_models_concurrently` pattern, which loads the SCRFD
// detector and ArcFace embedder (each building its own `CudaContext`) on
// `std::thread::scope`-spawned worker threads that are joined (and so
// have exited) before the `Converter` ever dispatches a single node.

/// Build a real `CudaContext` on a `std::thread::scope`-spawned thread
/// that exits (is joined) before this function returns it to the caller.
///
/// `Activation::Enabled` bypasses the `OXIONNX_CUDA` env-var opt-in gate
/// -- that policy is unit-tested separately in `context::tests` and is
/// orthogonal to what these tests are proving -- but still requires a
/// real, working CUDA driver and device 0 to succeed. The `expect` is
/// deliberate: a `gpu-tests`-gated run with no GPU present is a
/// misconfigured test invocation, not a case to degrade quietly out of.
#[cfg(feature = "gpu-tests")]
fn build_context_on_a_thread_that_then_exits() -> CudaContext {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                CudaContext::try_new_with(context::Activation::Enabled)
                    .expect("gpu-tests requires a real CUDA device -- run on a CUDA-capable host")
            })
            .join()
            .expect("context-building thread panicked")
    })
}

/// The exact regression, through the BLAS/GEMM dispatch path
/// (`matmul::cuda_matmul`, via `ctx.dnn.blas()`): build `ctx` on a
/// thread that exits, dispatch a real `MatMul` from the test's own
/// (different) thread, and confirm both that the call succeeds --
/// rather than failing with an invalid-context/invalid-handle driver
/// error the way it did before `activate_context` existed (see that
/// function's doc comment for exactly which error this driver raises
/// and why) -- and that the GPU actually computed the right numbers,
/// not just *some* numbers of the right shape.
#[cfg(feature = "gpu-tests")]
#[test]
fn matmul_dispatch_succeeds_from_a_different_thread_than_construction() {
    let ctx = build_context_on_a_thread_that_then_exits();

    let mut weights = HashMap::new();
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] (both 2x2, row-major).
    weights.insert(
        "a".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );
    weights.insert(
        "b".to_string(),
        Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    );
    let intermediates: HashMap<String, Tensor> = HashMap::new();
    let node = make_node(OpKind::MatMul, &["a", "b"], &["c"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect(
            "dispatch must succeed: `ctx` was built on a now-dead thread, so this call must \
             re-activate it on the calling thread instead of failing with an \
             invalid-context/invalid-handle driver error",
        )
        .expect("MatMul with two well-formed 2x2 operands must be claimed by CUDA, not declined");

    assert_eq!(outputs.len(), 1);
    let c = &outputs[0];
    assert_eq!(c.shape, vec![2, 2]);
    // A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]].
    let expected = [19.0_f32, 22.0, 43.0, 50.0];
    for (i, (&got, &want)) in c.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "index {i}: GPU MatMul result {got} does not match the hand-computed {want}",
        );
    }
}

/// Same regression, through the PTX-kernel/elementwise dispatch path
/// (`elementwise::cuda_elementwise`, via `ctx.dnn.stream()` direct
/// launch -- a different code path from MatMul's BLAS/GEMM one above,
/// so this independently confirms `activate_context` covers both):
/// `Relu` on a context built on a now-dead thread, dispatched from this
/// one.
#[cfg(feature = "gpu-tests")]
#[test]
fn relu_dispatch_succeeds_from_a_different_thread_than_construction() {
    let ctx = build_context_on_a_thread_that_then_exits();

    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut intermediates = HashMap::new();
    intermediates.insert(
        "x".to_string(),
        Tensor::new(vec![-2.0, -0.5, 0.0, 1.5, 3.0], vec![5]),
    );
    let node = make_node(OpKind::Relu, &["x"], &["y"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must succeed on a context built on a now-dead thread")
        .expect("Relu on a plain f32 vector must be claimed by CUDA, not declined");

    assert_eq!(outputs.len(), 1);
    let y = &outputs[0];
    assert_eq!(y.shape, vec![5]);
    let expected = [0.0_f32, 0.0, 0.0, 1.5, 3.0];
    for (i, (&got, &want)) in y.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "index {i}: GPU Relu result {got} does not match the expected {want}",
        );
    }
}

/// Same regression a third time, through [`conv::cuda_conv`]'s
/// `oxicuda_dnn` convolution-engine dispatch path -- a *third* distinct
/// kernel-launch mechanism from MatMul's BLAS/GEMM handle and Relu's
/// direct PTX-kernel launch above (see the [`conv`] module docs: its
/// three engines call straight into `Conv1x1`/`DepthwiseConv`/
/// `ImplicitGemmConv::execute`, none of which go through
/// `ctx.dnn.blas()`). This is the path production convolutions actually
/// take now that `is_supported_op` claims `Conv`, and `oxiface`'s own
/// concurrent model loading is precisely the dead-thread scenario
/// `activate_context` exists for -- a graph that is overwhelmingly
/// convolution meeting a context whose building thread has exited -- so
/// this is a hot-path regression test, not a direct-caller curiosity.
/// Uses the simplest of the three engines (`Conv1x1`, with a bias) to
/// keep the hand-computed expectation easy to check independently.
#[cfg(feature = "gpu-tests")]
#[test]
fn conv_dispatch_succeeds_from_a_different_thread_than_construction() {
    let ctx = build_context_on_a_thread_that_then_exits();

    // N=1, Cin=2, H=2, W=2, a 1x1 filter to Cout=3: unpadded, unit
    // stride, unit dilation, 1x1 kernel -- `pick_engine` selects
    // `ConvEngine::Conv1x1` for exactly this shape (see `conv::tests`).
    let mut weights = HashMap::new();
    weights.insert(
        "x".to_string(),
        // channel 0: [1, 2, 3, 4], channel 1: [5, 6, 7, 8].
        Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
        ),
    );
    weights.insert(
        "w".to_string(),
        // [Cout=3, Cin=2, 1, 1]: co0=[1,0], co1=[0,1], co2=[1,1].
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2, 1, 1]),
    );
    weights.insert(
        "b".to_string(),
        Tensor::new(vec![10.0, 100.0, 1000.0], vec![3]),
    );
    let intermediates: HashMap<String, Tensor> = HashMap::new();

    let mut node = make_node(OpKind::Conv, &["x", "w", "b"], &["y"]);
    node.attrs
        .int_lists
        .insert("strides".to_string(), vec![1, 1]);
    node.attrs
        .int_lists
        .insert("pads".to_string(), vec![0, 0, 0, 0]);
    node.attrs
        .int_lists
        .insert("dilations".to_string(), vec![1, 1]);
    node.attrs.ints.insert("group".to_string(), 1);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must succeed on a context built on a now-dead thread")
        .expect("a 1x1 Conv with well-formed operands must be claimed by cuda_conv, not declined");

    assert_eq!(outputs.len(), 1);
    let y = &outputs[0];
    assert_eq!(y.shape, vec![1, 3, 2, 2]);
    // co0 = 1*c0 + 0*c1 + bias0 = [1,2,3,4]      + 10   = [11,12,13,14]
    // co1 = 0*c0 + 1*c1 + bias1 = [5,6,7,8]       + 100  = [105,106,107,108]
    // co2 = 1*c0 + 1*c1 + bias2 = [6,8,10,12]     + 1000 = [1006,1008,1010,1012]
    let expected = [
        11.0_f32, 12.0, 13.0, 14.0, //
        105.0, 106.0, 107.0, 108.0, //
        1006.0, 1008.0, 1010.0, 1012.0,
    ];
    for (i, (&got, &want)) in y.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "index {i}: GPU Conv result {got} does not match the hand-computed {want}",
        );
    }
}

// ── `Conv` as an advertised op: the production pre-filter → dispatch ────
//
// On-device only, same gating as the cross-thread tests above.

/// The production sequence this change switches on, end to end on real
/// hardware: [`is_supported_op`] pre-filters the node *in* (exactly as
/// `oxionnx::execution_providers::decide_placement` now does for every
/// convolution), [`try_cuda_dispatch`] then actually claims it rather
/// than declining, and the numbers that come back agree with
/// [`reference::ref_conv`].
///
/// Deliberately asserts the *whole* sequence rather than the dispatch
/// call alone: before this change the pre-filter answered `false`, so a
/// dispatch-only test could pass in full while production convolutions
/// never reached the arm at all -- which is precisely the state this
/// crate was in. Pinning predicate and dispatch together in one test is
/// what makes "CUDA `Conv` is live" a single claim that can fail.
///
/// The shape is the `oxiface` workhorse -- 3x3, stride 1, `pad=1` on all
/// four sides, one group, multi-channel, with bias -- which
/// `conv::pick_engine` routes to `ImplicitGemmConv`, the general engine
/// (and the one SCRFD/ArcFace/InSwapper hit most).
#[cfg(feature = "gpu-tests")]
#[test]
fn conv_claimed_by_the_pre_filter_is_actually_dispatched() {
    let ctx = CudaContext::try_new_with(context::Activation::Enabled)
        .expect("gpu-tests requires a real CUDA device -- run on a CUDA-capable host");

    let (batch, in_ch, in_h, in_w, out_ch) = (1usize, 4usize, 8usize, 8usize, 6usize);
    let in_shape = vec![batch, in_ch, in_h, in_w];
    let weight_shape = vec![out_ch, in_ch, 3, 3];

    // Deterministic pseudo-random operands (same LCG constants as
    // `conv::tests::gpu_numeric` and `tests/*_gpu.rs`): a smooth ramp
    // would let a transposed index still land near the right answer,
    // which is exactly what must not slip through here.
    let mut state = 0x00C0_FFEE_D15E_A5E5_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = f64::from((state >> 32) as u32) / 4_294_967_296.0;
        (unit * 2.0 - 1.0) as f32
    };
    let input: Vec<f32> = (0..in_shape.iter().product::<usize>())
        .map(|_| next())
        .collect();
    let weight: Vec<f32> = (0..weight_shape.iter().product::<usize>())
        .map(|_| next())
        .collect();
    let bias: Vec<f32> = (0..out_ch).map(|_| next()).collect();

    let mut weights = HashMap::new();
    weights.insert(
        "x".to_string(),
        Tensor::new(input.clone(), in_shape.clone()),
    );
    weights.insert(
        "w".to_string(),
        Tensor::new(weight.clone(), weight_shape.clone()),
    );
    weights.insert("b".to_string(), Tensor::new(bias.clone(), vec![out_ch]));
    let intermediates: HashMap<String, Tensor> = HashMap::new();

    let mut node = make_node(OpKind::Conv, &["x", "w", "b"], &["y"]);
    node.attrs
        .int_lists
        .insert("strides".to_string(), vec![1, 1]);
    node.attrs
        .int_lists
        .insert("pads".to_string(), vec![1, 1, 1, 1]);
    node.attrs
        .int_lists
        .insert("dilations".to_string(), vec![1, 1]);
    node.attrs.ints.insert("group".to_string(), 1);

    // 1. The production pre-filter must let this node through at all.
    assert!(
        is_supported_op(&node.op),
        "is_supported_op(Conv) is false, so decide_placement would never route this node \
         to CUDA and the dispatch below is unreachable in production",
    );

    // 2. The arm must actually claim it, not decline.
    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "a 3x3 stride-1 symmetrically-padded Conv with bias must be claimed by \
             cuda_conv (Ok(Some(_))), not declined -- is_supported_op advertises Conv, so \
             an Ok(None) here means the advertisement is writing cheques the arm does not \
             cash for the single most common convolution shape in the workload",
        );

    // 3. The claimed answer must be right.
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].shape,
        vec![batch, out_ch, in_h, in_w],
        "3x3 with pad=1 and stride 1 is shape-preserving in the spatial dims",
    );
    let params = conv::ConvParams {
        strides: [1, 1],
        pads: [1, 1, 1, 1],
        dilations: [1, 1],
        group: 1,
    };
    let expected = reference::ref_conv(
        &input,
        &weight,
        Some(&bias),
        &in_shape,
        &weight_shape,
        &params,
    );
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("advertised CUDA Conv disagrees with the reference::ref_conv oracle: {e}");
    }
}

/// `is_supported_op(Conv) == true` is a claim about the op *kind*, not a
/// promise about every node -- see its "Necessary, not sufficient"
/// section. This pins the half of that contract that only became
/// reachable once `Conv` was advertised: a node the predicate waves
/// through, but whose configuration [`conv::cuda_conv`] declines, must
/// come back as `Ok(None)` (a clean fall-through to the CPU operator),
/// never as a wrong answer and never as a hard error.
///
/// Asymmetric `pads` is the case worth pinning, because it is the one
/// where a silent bug would be *plausible* rather than obvious:
/// `ConvProblem` carries a single padding value per spatial dimension, so
/// an implementation that just passed `pads[0]`/`pads[1]` through would
/// compile, run, and hand back a correctly-shaped tensor full of numbers
/// computed for the wrong padding -- the exact failure class this crate's
/// shadow verification exists for, but on a path that a caller trusting
/// the predicate would now actually reach.
#[cfg(feature = "gpu-tests")]
#[test]
fn advertised_conv_still_declines_the_configurations_it_cannot_compute() {
    let ctx = CudaContext::try_new_with(context::Activation::Enabled)
        .expect("gpu-tests requires a real CUDA device -- run on a CUDA-capable host");

    let mut weights = HashMap::new();
    weights.insert(
        "x".to_string(),
        Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]),
    );
    weights.insert("w".to_string(), Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]));
    let intermediates: HashMap<String, Tensor> = HashMap::new();

    let mut node = make_node(OpKind::Conv, &["x", "w"], &["y"]);
    node.attrs
        .int_lists
        .insert("strides".to_string(), vec![1, 1]);
    // ONNX order is [top, left, bottom, right]: 1 of top padding, none at
    // the bottom -- asymmetric, and therefore not expressible as a
    // `ConvProblem`.
    node.attrs
        .int_lists
        .insert("pads".to_string(), vec![1, 1, 0, 0]);
    node.attrs
        .int_lists
        .insert("dilations".to_string(), vec![1, 1]);
    node.attrs.ints.insert("group".to_string(), 1);

    assert!(
        is_supported_op(&node.op),
        "this test is only meaningful while Conv is advertised -- otherwise the pre-filter, \
         not cuda_conv, is what keeps this node off the GPU",
    );

    let claimed = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("an unsupported configuration is a decline, not a hard error");
    assert!(
        claimed.is_none(),
        "asymmetric pads [1,1,0,0] must be declined (Ok(None)) so the CPU operator computes \
         it -- ConvProblem cannot express asymmetric padding, so anything else returned \
         here was computed for padding the model did not ask for",
    );
}

/// Negative control, independent of whether `activate_context` is
/// present: proves the underlying failure mode this whole fix exists
/// for is real hardware/driver behaviour and not an artefact of this
/// test suite, by calling [`matmul::cuda_matmul`] **directly** --
/// bypassing `try_cuda_dispatch` (and therefore `activate_context`)
/// entirely -- from a thread that never activated `ctx`.
///
/// This is the same GPU-side function `try_cuda_dispatch`'s `MatMul`
/// arm calls in production, so a passing `matmul_dispatch_succeeds_*`
/// test above together with a failing call here pins down precisely
/// what `activate_context` is buying: not "some driver call somewhere
/// eventually fails," but this exact one. (An earlier version of this
/// test called `Stream::synchronize` directly instead and did *not*
/// reproduce the failure -- this driver's `cuStreamSynchronize`
/// tolerates a thread with no current context, unlike the
/// `cuMemAlloc`/`cuLaunchKernel`-family calls `cuda_matmul` makes. Pick
/// the operation that is actually on the hot path, not any driver call.)
///
/// If CUDA's context/thread model ever changes such that this starts
/// succeeding, `activate_context` may no longer be necessary and this
/// test should be revisited alongside it.
#[cfg(feature = "gpu-tests")]
#[test]
fn without_reactivation_a_context_from_a_dead_thread_is_unusable() {
    let ctx = build_context_on_a_thread_that_then_exits();
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let err = matmul::cuda_matmul(&ctx, &a, &b, 2, 2, 2).expect_err(
        "a context built on a thread that has since exited must NOT be usable on another \
         thread without an explicit set_current first; if this now succeeds, CUDA's \
         context/thread model has changed",
    );
    let msg = err.to_string().to_ascii_lowercase();
    assert!(
        msg.contains("invalid") || msg.contains("context") || msg.contains("initialized"),
        "expected an invalid-context/invalid-handle style driver error, got: {msg}",
    );
}
