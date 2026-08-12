//! On-device proof that `OXIONNX_CUDA_VERIFY=1`'s *live* shadow-verification
//! wiring actually engages, end to end, through [`try_cuda_dispatch`] --
//! across every dispatch-arm family that calls it, not just the one or two
//! that happen to be exercised incidentally by other test files.
//!
//! # Why this file exists, specifically
//!
//! `try_cuda_dispatch`'s six distinct op-family arms (`MatMul`/`Gemm`, `Conv`,
//! the four binary elementwise ops, `ReduceSum`/`ReduceMax`, the sixteen
//! unary activations, and `Softmax`) each make their own separate call to the
//! crate-private `verify_or_fallback`, with their own hand-written oracle
//! closure (`reference::ref_matmul`, `reference::ref_conv`,
//! `reference::ref_binary_vec`, `reference::ref_reduce`,
//! `reference::ref_unary_vec`, `reference::ref_softmax` respectively -- see
//! `lib.rs`'s match arms). That is six independent places a wiring mistake
//! can hide: the wrong `OpKind` passed to a shared oracle function,
//! transposed/swapped operands in the closure, an axis (or, for `Conv`, a
//! stride/pad/dilation/group) computed by one arm but not threaded into the
//! oracle call for that same arm, or (more basic still)
//! `reference::verify_enabled()`'s *live* return value silently failing to
//! reach `verify_or_fallback` at all. None of `oxionnx-cuda`'s existing
//! on-device tests actually run with `OXIONNX_CUDA_VERIFY=1` live in the
//! process environment and check what happens per arm:
//!
//! * `tests/matmul_shape_sweep_gpu.rs` and `conv::tests::gpu_numeric` call
//!   [`oxionnx_cuda::matmul::cuda_matmul`] / [`oxionnx_cuda::conv::cuda_conv`]
//!   **directly**, bypassing [`try_cuda_dispatch`] (and therefore
//!   `verify_or_fallback`) entirely -- `OXIONNX_CUDA_VERIFY` has *zero*
//!   effect on either file, at any setting.
//! * `lib.rs`'s cross-thread regression tests (`matmul_dispatch_succeeds_*`,
//!   `relu_dispatch_succeeds_*`, `conv_dispatch_succeeds_*`) do route through
//!   [`try_cuda_dispatch`], so running them with the env var live does
//!   exercise three of the six arms -- but only incidentally: they were
//!   written to prove thread affinity, not shadow verification, and are not
//!   self-checking about whether the env var was actually set.
//!
//! This file is purpose-built for the gap: one test per dispatch-arm family,
//! every one of which fails loudly (not silently no-ops) if invoked without
//! `OXIONNX_CUDA_VERIFY=1` actually live -- see [`require_verify_enabled`].
//! Every arm covered here is one `oxionnx`'s placement logic actually routes
//! production nodes to, `Conv` included ([`oxionnx_cuda::is_supported_op`]
//! reports `true` for it -- see the `conv` module docs' "Advertised as
//! CUDA-supported"), so a broken verify wiring found here is a broken verify
//! wiring on the hot path.
//!
//! # What a pass here does, and does NOT, prove
//!
//! Honestly, mirroring this codebase's own accounting standard (see e.g.
//! `oxicuda-blas::gpu_tests`'s "Honest kernel accounting" section):
//!
//! * **Proven:** [`reference::verify_enabled`] reads a live, process-
//!   environment `OXIONNX_CUDA_VERIFY=1` correctly (not hard-coded, not
//!   stuck at a stale cached default); every one of the six
//!   `verify_or_fallback` call sites receives that live `true` and, given a
//!   GPU kernel that is (as these all are, per the other on-device suites in
//!   this crate) actually correct, does **not** spuriously discard the
//!   result -- i.e. shadow verification does not "cry wolf" and silently
//!   turn off acceleration for a node it should have accepted. That would be
//!   just as real a bug as the reverse (a wrong kernel sailing through
//!   unverified): a mismatched oracle wiring (wrong `OpKind`, swapped
//!   operands, wrong axis) makes `try_cuda_dispatch` return `Ok(None)`
//!   instead of `Ok(Some(_))` under the default `FailurePolicy::Fallback`,
//!   which is exactly what every `.expect("... must be claimed by CUDA, not
//!   declined")` below would catch.
//! * **NOT proven here:** that `shadow_verify`'s *mismatch* branch (an
//!   actually-wrong GPU result gets caught and discarded) fires correctly --
//!   that needs a deliberately-wrong kernel to compare against, which this
//!   file cannot manufacture without touching production dispatch code. That
//!   branch already has direct, parameterised, environment-independent
//!   coverage in `reference::tests::shadow_verify_mismatch_under_fallback_*`
//!   / `_under_strict_*`. What was missing, and what this file adds, is
//!   proof that the *live* env-var path actually reaches that already-tested
//!   logic in the first place, for every arm that is supposed to reach it.
//!
//! # Running
//!
//! ```text
//! OXIONNX_CUDA_VERIFY=1 cargo test -p oxionnx-cuda --features gpu-tests --test verify_path_gpu
//! ```
//!
//! Gated the same way as every other on-device suite in this crate:
//! `required-features = ["gpu-tests"]` in `Cargo.toml` keeps a plain `cargo
//! test -p oxionnx-cuda` (no feature flag) from touching a GPU at all, and
//! [`gpu_ctx`] `.expect()`s rather than skips when the feature is on but no
//! device is present -- a misconfigured invocation, not a case to degrade
//! quietly out of (same rationale as `lib.rs`'s cross-thread tests and
//! `conv::tests::gpu_numeric`).

use std::collections::HashMap;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;
use oxionnx_cuda::context::{Activation, CudaContext};
use oxionnx_cuda::{conv, reference, try_cuda_dispatch};

// ---------------------------------------------------------------------------
// Fixture & helpers
// ---------------------------------------------------------------------------

/// A real GPU context, bypassing the `OXIONNX_CUDA` env-var opt-in gate
/// (that policy is unit-tested independently in `context::tests`, and is
/// orthogonal to what this file proves). `.expect()`, not a quiet skip --
/// see the module docs' "Running" section.
fn gpu_ctx() -> CudaContext {
    CudaContext::try_new_with(Activation::Enabled)
        .expect("gpu-tests requires a real CUDA device -- run on a CUDA-capable host")
}

/// The one thing every test in this file must confirm before it does
/// anything else: that `OXIONNX_CUDA_VERIFY` is actually live in *this*
/// process's environment. Without this, a forgotten env var would make
/// every test below pass having silently verified nothing -- indistinguishable
/// from a real pass, which is precisely the failure mode this file exists to
/// rule out (see the module docs).
///
/// Safe to call from every test despite `reference::verify_enabled`
/// internally caching its answer in a `OnceLock`: the env var is fixed for
/// the lifetime of this test binary's process (set by whoever invoked
/// `cargo test`, before this process even started), so there is no
/// first-caller-wins race to worry about the way there would be if a test
/// tried to *mutate* it with `std::env::set_var` instead.
fn require_verify_enabled() {
    assert!(
        reference::verify_enabled(),
        "this test file only proves anything with shadow verification live -- rerun as \
         `OXIONNX_CUDA_VERIFY=1 cargo test -p oxionnx-cuda --features gpu-tests --test \
         verify_path_gpu` (see this file's module docs)",
    );
}

fn make_node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: "verify_path_test_node".to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// A small deterministic LCG (same algorithm this crate already uses in
/// `tests/matmul_shape_sweep_gpu.rs` and `conv::tests::gpu_numeric`,
/// duplicated here for the same reason: no `rand` dependency, and a
/// regression test must be bit-reproducible across runs and machines).
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
    fn range_f32(&mut self, lo: f64, hi: f64) -> f32 {
        let unit = f64::from(self.next_u32()) / 4_294_967_296.0;
        (lo + (hi - lo) * unit) as f32
    }
}

fn make_vec(rng: &mut Lcg, len: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..len).map(|_| rng.range_f32(lo, hi)).collect()
}

// ---------------------------------------------------------------------------
// One test per `verify_or_fallback` call site in `try_cuda_dispatch`
// ---------------------------------------------------------------------------

/// `MatMul`/`Gemm` arm: oracle is `reference::ref_matmul`.
#[test]
fn matmul_verify_path_agrees_live_on_real_hardware() {
    require_verify_enabled();
    let ctx = gpu_ctx();

    let (m, k, n) = (5usize, 7usize, 3usize);
    let mut rng = Lcg::new(0x7E71_FADE_7457_0001);
    let a = make_vec(&mut rng, m * k, -1.0, 1.0);
    let b = make_vec(&mut rng, k * n, -1.0, 1.0);

    let mut weights = HashMap::new();
    weights.insert("a".to_string(), Tensor::new(a.clone(), vec![m, k]));
    weights.insert("b".to_string(), Tensor::new(b.clone(), vec![k, n]));
    let intermediates: HashMap<String, Tensor> = HashMap::new();
    let node = make_node(OpKind::MatMul, &["a", "b"], &["c"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "MatMul must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means shadow verification spuriously disagreed with a correct GPU \
             kernel, i.e. the MatMul arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    let expected = reference::ref_matmul(&a, &b, m, k, n);
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("MatMul verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// Binary elementwise arm (`Add`/`Sub`/`Mul`/`Div`): oracle is
/// `reference::ref_binary_vec`. `Add` stands in for the family -- the four
/// ops share one dispatch arm and one `verify_or_fallback` call site.
#[test]
fn add_verify_path_agrees_live_on_real_hardware() {
    require_verify_enabled();
    let ctx = gpu_ctx();

    let shape = vec![3usize, 4, 2];
    let elems: usize = shape.iter().product();
    let mut rng = Lcg::new(0x7E71_FADE_7457_0002);
    let a = make_vec(&mut rng, elems, -5.0, 5.0);
    let b = make_vec(&mut rng, elems, -5.0, 5.0);

    let mut weights = HashMap::new();
    weights.insert("a".to_string(), Tensor::new(a.clone(), shape.clone()));
    weights.insert("b".to_string(), Tensor::new(b.clone(), shape.clone()));
    let intermediates: HashMap<String, Tensor> = HashMap::new();
    let node = make_node(OpKind::Add, &["a", "b"], &["c"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Add must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means the binary-elementwise arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    let expected = reference::ref_binary_vec(&OpKind::Add, &a, &b)
        .expect("ref_binary_vec has a formula for Add");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Add verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// Reduce arm (`ReduceSum`/`ReduceMax`): oracle is `reference::ref_reduce`.
/// `ReduceSum` over a non-trailing axis, `keepdims=0`, so this also exercises
/// the axis-resolution/attrs-reading path the arm does before it ever calls
/// the oracle.
#[test]
fn reduce_sum_verify_path_agrees_live_on_real_hardware() {
    require_verify_enabled();
    let ctx = gpu_ctx();

    let shape = vec![4usize, 3];
    let axis = 0usize;
    let mut rng = Lcg::new(0x7E71_FADE_7457_0003);
    let data = make_vec(&mut rng, 12, -3.0, 3.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(data.clone(), shape.clone()));
    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut node = make_node(OpKind::ReduceSum, &["x"], &["y"]);
    node.attrs
        .int_lists
        .insert("axes".to_string(), vec![axis as i64]);
    node.attrs.ints.insert("keepdims".to_string(), 0);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "ReduceSum must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- \
             an Ok(None) here means the reduce arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![3]);
    let expected = reference::ref_reduce(&OpKind::ReduceSum, &data, &shape, axis)
        .expect("ref_reduce has a formula for ReduceSum over a valid axis");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("ReduceSum verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// Unary elementwise arm: oracle is `reference::ref_unary_vec`. `Relu`
/// stands in for the sixteen-op family -- they share one dispatch arm and
/// one `verify_or_fallback` call site (the three attribute-gated ops --
/// `LeakyRelu`, `HardSigmoid`, `Gelu` -- branch *before* reaching it, so
/// `Relu` exercises the call site itself most directly).
#[test]
fn relu_verify_path_agrees_live_on_real_hardware() {
    require_verify_enabled();
    let ctx = gpu_ctx();

    let mut rng = Lcg::new(0x7E71_FADE_7457_0004);
    let mut data = make_vec(&mut rng, 32, -4.0, 4.0);
    // Plant exact zero and both signed tiny values, same reasoning
    // `oxionnx-directml`'s cross-validation suite documents for its own
    // Relu case: these are where a vectorised kernel's tail handling is
    // most likely to disagree with a scalar oracle.
    data[0] = 0.0;
    data[1] = -0.0;
    data[2] = f32::MIN_POSITIVE;
    data[3] = -f32::MIN_POSITIVE;

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(data.clone(), vec![data.len()]));
    let weights: HashMap<String, Tensor> = HashMap::new();
    let node = make_node(OpKind::Relu, &["x"], &["y"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Relu must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means the unary-elementwise arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    let expected = reference::ref_unary_vec(&OpKind::Relu, &data)
        .expect("ref_unary_vec has a formula for Relu");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Relu verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// `Softmax` arm: oracle is `reference::ref_softmax`. Default `axis`
/// attribute (`-1`) resolves to the last dimension, which is the only shape
/// `cuda_softmax` can claim -- no explicit attrs needed.
#[test]
fn softmax_verify_path_agrees_live_on_real_hardware() {
    require_verify_enabled();
    let ctx = gpu_ctx();

    let shape = vec![3usize, 5];
    let mut rng = Lcg::new(0x7E71_FADE_7457_0005);
    let data = make_vec(&mut rng, 15, -6.0, 6.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(data.clone(), shape.clone()));
    let weights: HashMap<String, Tensor> = HashMap::new();
    let node = make_node(OpKind::Softmax, &["x"], &["y"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Softmax must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means the Softmax arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    let expected =
        reference::ref_softmax(&data, &shape).expect("ref_softmax has a formula for a 2-D shape");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Softmax verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// `Conv` arm: oracle is [`reference::ref_conv`]. A 3x3, stride-1,
/// "same"-padded (`pad=1`), single-group convolution with a bias -- general
/// enough that [`conv::cuda_conv`]'s own dispatch rule routes it through
/// `ImplicitGemmConv` (see the `conv` module docs' "Dispatch rule"), so this
/// exercises the attrs-reading path (`strides`/`pads`/`dilations`/`group`)
/// *and* `ImplicitGemmConv`'s native bias epilogue, ahead of ever reaching
/// `verify_or_fallback`.
///
/// Like the other five, this is a production path: `is_supported_op(Conv)`
/// reports `true`, so `oxionnx`'s placement logic routes real convolutions
/// through this exact arm. Asserted below rather than merely stated, so that
/// re-hiding `Conv` behind the pre-filter cannot leave this file quietly
/// over-claiming what it proves.
#[test]
fn conv_verify_path_agrees_live_on_real_hardware() {
    require_verify_enabled();
    assert!(
        oxionnx_cuda::is_supported_op(&OpKind::Conv),
        "Conv is no longer advertised, so this test no longer covers a path oxionnx's \
         placement logic reaches -- update this file's claims along with the predicate",
    );
    let ctx = gpu_ctx();

    let (n, in_channels, in_h, in_w) = (1usize, 3usize, 6usize, 7usize);
    let (out_channels, filter_h, filter_w) = (4usize, 3usize, 3usize);
    let (pad, stride, dilation, group) = (1usize, 1usize, 1usize, 1usize);

    let mut rng = Lcg::new(0x7E71_FADE_7457_0006);
    let input_data = make_vec(&mut rng, n * in_channels * in_h * in_w, -1.0, 1.0);
    let weight_data = make_vec(
        &mut rng,
        out_channels * in_channels * filter_h * filter_w,
        -1.0,
        1.0,
    );
    let bias_data = make_vec(&mut rng, out_channels, -0.5, 0.5);

    let mut weights = HashMap::new();
    weights.insert(
        "x".to_string(),
        Tensor::new(input_data.clone(), vec![n, in_channels, in_h, in_w]),
    );
    weights.insert(
        "w".to_string(),
        Tensor::new(
            weight_data.clone(),
            vec![out_channels, in_channels, filter_h, filter_w],
        ),
    );
    weights.insert(
        "b".to_string(),
        Tensor::new(bias_data.clone(), vec![out_channels]),
    );
    let intermediates: HashMap<String, Tensor> = HashMap::new();

    let mut node = make_node(OpKind::Conv, &["x", "w", "b"], &["y"]);
    node.attrs
        .int_lists
        .insert("strides".to_string(), vec![stride as i64, stride as i64]);
    node.attrs.int_lists.insert(
        "pads".to_string(),
        vec![pad as i64, pad as i64, pad as i64, pad as i64],
    );
    node.attrs.int_lists.insert(
        "dilations".to_string(),
        vec![dilation as i64, dilation as i64],
    );
    node.attrs.ints.insert("group".to_string(), group as i64);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Conv must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means either the Conv arm's verify wiring is broken, or this shape \
             was unexpectedly declined by cuda_conv itself (it must not be: symmetric pad=1, \
             stride=1, dilation=1, group=1, rank-4 shapes, no zero dims)",
        );

    assert_eq!(outputs.len(), 1);
    // pad=1, stride=1, dilation=1, a 3x3 kernel: "same" padding, so the
    // output spatial size equals the input's.
    assert_eq!(
        outputs[0].shape,
        vec![n, out_channels, in_h, in_w],
        "Conv verify-path output shape mismatch"
    );

    let params = conv::ConvParams {
        strides: [stride, stride],
        pads: [pad, pad, pad, pad],
        dilations: [dilation, dilation],
        group,
    };
    let expected = reference::ref_conv(
        &input_data,
        &weight_data,
        Some(&bias_data),
        &[n, in_channels, in_h, in_w],
        &[out_channels, in_channels, filter_h, filter_w],
        &params,
    );
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Conv verify-path result disagrees with the CPU oracle: {e}");
    }
}
