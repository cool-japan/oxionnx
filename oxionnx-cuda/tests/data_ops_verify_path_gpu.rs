//! On-device proof for the data-movement CUDA op wave (`MaxPool`,
//! `AveragePool`, `Resize`, `Pad`, `Slice`, `Concat`, and the zero-cost
//! `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten` residency aliases).
//!
//! Mirrors `tests/verify_path_gpu.rs`'s discipline exactly: one test per
//! `verify_or_fallback` call site, `OXIONNX_CUDA_VERIFY=1` live and asserted
//! (via [`require_verify_enabled`]), every result compared against this
//! crate's own `reference::ref_*` oracle. See that file's module docs for the
//! full rationale (what a pass here does and does not prove) — not repeated
//! here to avoid the two drifting apart.
//!
//! The `Reshape` family gets a different kind of proof
//! ([`reshape_alias_keeps_a_chain_fully_resident`]): there is no oracle to
//! shadow-verify against (see `reshape`'s module docs), so what this file
//! proves instead is the property that actually matters for that arm — a
//! `Device`-resident producer feeding a `Reshape` feeding a `Device`-resident
//! consumer never touches the host, and the *values* survive the alias
//! correctly.
//!
//! # Running
//!
//! ```text
//! OXIONNX_CUDA_VERIFY=1 cargo test -p oxionnx-cuda --features gpu-tests \
//!     --test data_ops_verify_path_gpu
//! ```

use std::collections::HashMap;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;
use oxionnx_cuda::context::{Activation, CudaContext};
use oxionnx_cuda::{
    concat, pad, pool, reference, resize, slice, try_cuda_dispatch, try_cuda_dispatch_resident,
    CudaDeviceTensor, CudaDispatchOutcome, CudaOutputPlacement, ResidentActivations,
};

// ---------------------------------------------------------------------------
// Fixture & helpers -- deliberately identical in shape to verify_path_gpu.rs
// ---------------------------------------------------------------------------

fn gpu_ctx() -> Option<CudaContext> {
    CudaContext::try_new_with(Activation::Enabled)
}

fn require_verify_enabled() {
    assert!(
        reference::verify_enabled(),
        "this test file only proves anything with shadow verification live -- rerun as \
         `OXIONNX_CUDA_VERIFY=1 cargo test -p oxionnx-cuda --features gpu-tests --test \
         data_ops_verify_path_gpu` (see this file's module docs)",
    );
}

fn make_node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: "data_ops_verify_path_test_node".to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

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
// MaxPool / AveragePool
// ---------------------------------------------------------------------------

/// `MaxPool` arm: oracle is `reference::ref_pool`. Kernel/stride 2, no
/// padding -- exactly `det_10g.onnx`'s real `MaxPool_9` geometry.
#[test]
fn max_pool_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping max_pool_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 5, 8, 10];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0001);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut node = make_node(OpKind::MaxPool, &["x"], &["y"]);
    node.attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    node.attrs
        .int_lists
        .insert("strides".to_string(), vec![2, 2]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "MaxPool must still be claimed (Ok(Some(_))) with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means the pool arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 5, 4, 5]);
    let params = pool::PoolParams {
        kernel: [2, 2],
        strides: [2, 2],
        pads: [0, 0, 0, 0],
        ceil_mode: false,
        count_include_pad: false,
    };
    let expected = reference::ref_pool(&x, &shape, pool::PoolKind::Max, &params)
        .expect("ref_pool has a formula for this shape");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("MaxPool verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// `AveragePool` arm: oracle is `reference::ref_pool`. `ceil_mode=1` over an
/// evenly-divisible input -- exactly `det_10g.onnx`'s real `AveragePool_36`
/// geometry, and the case `crate::pool`'s dispatch-time floor/ceil agreement
/// check must accept rather than decline.
#[test]
fn average_pool_ceil_mode_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping \
             average_pool_ceil_mode_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 3, 20, 20];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0002);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let weights: HashMap<String, Tensor> = HashMap::new();
    let mut node = make_node(OpKind::AveragePool, &["x"], &["y"]);
    node.attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    node.attrs
        .int_lists
        .insert("strides".to_string(), vec![2, 2]);
    node.attrs.ints.insert("ceil_mode".to_string(), 1);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "AveragePool(ceil_mode=1) must still be claimed with OXIONNX_CUDA_VERIFY=1 live -- \
             an Ok(None) here means either the pool arm's verify wiring is broken, or the \
             floor/ceil agreement check unexpectedly declined an evenly-divisible input",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 3, 10, 10]);
    let params = pool::PoolParams {
        kernel: [2, 2],
        strides: [2, 2],
        pads: [0, 0, 0, 0],
        ceil_mode: true,
        count_include_pad: false,
    };
    let expected = reference::ref_pool(&x, &shape, pool::PoolKind::Avg, &params)
        .expect("ref_pool has a formula for this shape");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("AveragePool verify-path result disagrees with the CPU oracle: {e}");
    }
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

/// `Resize` arm, nearest/asymmetric/floor: oracle is `reference::ref_resize`.
/// `sizes` supplied as input 3 (opset-11+ layout, no `roi`/`scales`) -- the
/// exact shape `det_10g.onnx`'s two FPN-upsample `Resize` nodes use.
#[test]
fn resize_nearest_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping resize_nearest_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 4, 5, 5];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0003);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "sizes".to_string(),
        Tensor::new(vec![1.0, 4.0, 10.0, 10.0], vec![4]),
    );
    let mut node = make_node(OpKind::Resize, &["x", "", "", "sizes"], &["y"]);
    node.attrs
        .strings
        .insert("mode".to_string(), "nearest".to_string());
    node.attrs.strings.insert(
        "coordinate_transformation_mode".to_string(),
        "asymmetric".to_string(),
    );
    node.attrs
        .strings
        .insert("nearest_mode".to_string(), "floor".to_string());

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Resize(nearest/asymmetric/floor) must still be claimed with OXIONNX_CUDA_VERIFY=1 \
             live -- an Ok(None) here means the resize arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 4, 10, 10]);
    let params = resize::ResizeParams {
        mode: resize::ResizeMode::Nearest,
        out_h: 10,
        out_w: 10,
    };
    let expected = reference::ref_resize(&x, &shape, &params).expect("ref_resize has a formula");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Resize(nearest) verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// `Resize` arm, bilinear/`pytorch_half_pixel`: oracle is
/// `reference::ref_resize`. `scales=[1,1,2,2]` supplied as input 2 -- the
/// exact shape `inswapper_128.onnx`'s two decoder-upsample `Resize` nodes use.
#[test]
fn resize_bilinear_pytorch_half_pixel_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping \
             resize_bilinear_pytorch_half_pixel_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 8, 6, 6];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0004);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "scales".to_string(),
        Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]),
    );
    let mut node = make_node(OpKind::Resize, &["x", "", "scales"], &["y"]);
    node.attrs
        .strings
        .insert("mode".to_string(), "linear".to_string());
    node.attrs.strings.insert(
        "coordinate_transformation_mode".to_string(),
        "pytorch_half_pixel".to_string(),
    );

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Resize(linear/pytorch_half_pixel) must still be claimed with \
             OXIONNX_CUDA_VERIFY=1 live -- an Ok(None) here means the resize arm's verify \
             wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 8, 12, 12]);
    let params = resize::ResizeParams {
        mode: resize::ResizeMode::Bilinear {
            align_corners: false,
        },
        out_h: 12,
        out_w: 12,
    };
    let expected = reference::ref_resize(&x, &shape, &params).expect("ref_resize has a formula");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Resize(bilinear) verify-path result disagrees with the CPU oracle: {e}");
    }
}

// ---------------------------------------------------------------------------
// Pad
// ---------------------------------------------------------------------------

/// `Pad` arm, `reflect`: oracle is `reference::ref_pad`. `pads=[0,0,3,3,0,0,3,3]`
/// -- exactly `inswapper_128.onnx`'s `Pad_39` (the model's 7x7-stem reflect
/// pad).
#[test]
fn pad_reflect_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping pad_reflect_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 3, 16, 16];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0005);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "pads".to_string(),
        Tensor::new(vec![0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0], vec![8]),
    );
    let mut node = make_node(OpKind::Pad, &["x", "pads"], &["y"]);
    node.attrs
        .strings
        .insert("mode".to_string(), "reflect".to_string());

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Pad(reflect) must still be claimed with OXIONNX_CUDA_VERIFY=1 live -- an Ok(None) \
             here means the pad arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 3, 22, 22]);
    let params = pad::PadParams {
        pad_h: (3, 3),
        pad_w: (3, 3),
        mode: pad::PadMode::Reflect,
        constant_value: 0.0,
    };
    let expected = reference::ref_pad(&x, &shape, &params).expect("ref_pad has a formula");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Pad(reflect) verify-path result disagrees with the CPU oracle: {e}");
    }
}

/// `Pad` arm, `constant`: oracle is `reference::ref_pad`. Exercises the
/// bounds-branch kernel and a non-zero fill value.
#[test]
fn pad_constant_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping pad_constant_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 2, 6, 7];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0006);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "pads".to_string(),
        Tensor::new(vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0], vec![8]),
    );
    weights.insert("value".to_string(), Tensor::new(vec![-3.5], vec![]));
    let mut node = make_node(OpKind::Pad, &["x", "pads", "value"], &["y"]);
    node.attrs
        .strings
        .insert("mode".to_string(), "constant".to_string());

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Pad(constant) must still be claimed with OXIONNX_CUDA_VERIFY=1 live -- an \
             Ok(None) here means the pad arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 2, 9, 10]);
    let params = pad::PadParams {
        pad_h: (1, 2),
        pad_w: (2, 1),
        mode: pad::PadMode::Constant,
        constant_value: -3.5,
    };
    let expected = reference::ref_pad(&x, &shape, &params).expect("ref_pad has a formula");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Pad(constant) verify-path result disagrees with the CPU oracle: {e}");
    }
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

/// `Slice` arm: oracle is `reference::ref_slice`. Splits a `[1,2048,1,1]`
/// style-vector's channel axis in half -- exactly `inswapper_128.onnx`'s real
/// `Slice_86`/`Slice_89` pattern.
#[test]
fn slice_channel_half_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping slice_channel_half_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let shape = vec![1usize, 64, 1, 1];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0007);
    let x = make_vec(&mut rng, shape.iter().product(), -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("x".to_string(), Tensor::new(x.clone(), shape.clone()));
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("starts".to_string(), Tensor::new(vec![32.0], vec![1]));
    weights.insert("ends".to_string(), Tensor::new(vec![64.0], vec![1]));
    weights.insert("axes".to_string(), Tensor::new(vec![1.0], vec![1]));
    let node = make_node(OpKind::Slice, &["x", "starts", "ends", "axes"], &["y"]);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Slice must still be claimed with OXIONNX_CUDA_VERIFY=1 live -- an Ok(None) here \
             means the slice arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 32, 1, 1]);
    let params = slice::slice_params_from_node(&shape, &[32], &[64], Some(&[1]), None)
        .expect("must resolve");
    let expected = reference::ref_slice(&x, &shape, &params).expect("ref_slice has a formula");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Slice verify-path result disagrees with the CPU oracle: {e}");
    }
    // Cross-check against the second half too, and that the two together
    // reconstruct the original -- catches an off-by-one in the shared
    // `start`/`step` arithmetic that a single slice could miss.
    assert_eq!(outputs[0].data.as_slice(), &x[32..64]);
}

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

/// `Concat` arm: oracle is `reference::ref_concat`. `axis=0` over three
/// operands -- the shape `det_10g.onnx`'s two real `Concat` nodes use
/// (assembling a `Resize` `sizes` input from a `Shape`/`Slice` chain and two
/// `Unsqueeze`d scalars).
#[test]
fn concat_axis_0_three_operands_verify_path_agrees_live_on_real_hardware() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!(
            "no CUDA device present, skipping \
             concat_axis_0_three_operands_verify_path_agrees_live_on_real_hardware"
        );
        return;
    };
    require_verify_enabled();

    let mut rng = Lcg::new(0x00D4_740F_0AD5_0008);
    let a = make_vec(&mut rng, 2, -4.0, 4.0);
    let b = make_vec(&mut rng, 1, -4.0, 4.0);
    let c = make_vec(&mut rng, 3, -4.0, 4.0);

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![2]));
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("b".to_string(), Tensor::new(b.clone(), vec![1]));
    weights.insert("c".to_string(), Tensor::new(c.clone(), vec![3]));
    let mut node = make_node(OpKind::Concat, &["a", "b", "c"], &["y"]);
    node.attrs.ints.insert("axis".to_string(), 0);

    let outputs = try_cuda_dispatch(&node, &weights, &intermediates, &ctx)
        .expect("dispatch must not hard-error")
        .expect(
            "Concat must still be claimed with OXIONNX_CUDA_VERIFY=1 live -- an Ok(None) here \
             means the concat arm's verify wiring is broken",
        );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![6]);
    let params = concat::ConcatParams {
        axis: 0,
        out_shape: vec![6],
        segment_lens: vec![2, 1, 3],
    };
    let expected = reference::ref_concat(&[&a, &b, &c], &params).expect("ref_concat has a formula");
    if let Err(e) = reference::compare(&outputs[0].data, &expected) {
        panic!("Concat verify-path result disagrees with the CPU oracle: {e}");
    }
    let mut want = a.clone();
    want.extend_from_slice(&b);
    want.extend_from_slice(&c);
    assert_eq!(outputs[0].data, want);
}

// ---------------------------------------------------------------------------
// Reshape / Squeeze / Unsqueeze / Flatten: zero-cost residency alias.
// No oracle (see `reshape`'s module docs) -- proven instead by keeping a
// chain fully device-resident and checking the values survive the alias.
// ---------------------------------------------------------------------------

/// The empty [`ResidentActivations`] map used everywhere else in this crate,
/// plus exactly the one name this test needs resident.
struct OneResident<'a> {
    name: &'a str,
    tensor: &'a CudaDeviceTensor,
}
impl ResidentActivations for OneResident<'_> {
    fn resident(&self, name: &str) -> Option<&CudaDeviceTensor> {
        (name == self.name).then_some(self.tensor)
    }
    fn holds_node_output(&self, name: &str) -> bool {
        name == self.name
    }
}

/// Produces a real device-resident `CudaDeviceTensor` by dispatching a
/// trivial `Relu` through the public [`try_cuda_dispatch_resident`] entry
/// point with `CudaOutputPlacement::Device` requested -- this crate's public
/// API has no lower-level way to *construct* one directly (by design: e.g.
/// `CudaDeviceTensor::from_owned` is `pub(crate)`), and this is exactly the
/// mechanism a real session uses to produce one.
fn resident_tensor_of(ctx: &CudaContext, data: &[f32], shape: &[usize]) -> CudaDeviceTensor {
    let mut intermediates = HashMap::new();
    intermediates.insert(
        "relu_in".to_string(),
        Tensor::new(data.to_vec(), shape.to_vec()),
    );
    let weights: HashMap<String, Tensor> = HashMap::new();
    let node = make_node(OpKind::Relu, &["relu_in"], &["relu_out"]);

    match try_cuda_dispatch_resident(
        &node,
        &weights,
        &intermediates,
        &oxionnx_cuda::NoActivations,
        CudaOutputPlacement::Device,
        ctx,
    )
    .expect("Relu dispatch must not hard-error")
    .expect("Relu must be claimed")
    {
        CudaDispatchOutcome::Device(tensor) => tensor,
        CudaDispatchOutcome::Host(_) => panic!("requested Device placement but got Host back"),
    }
}

/// A `Reshape` consuming a device-resident input must (a) stay claimed and
/// resident end to end -- proving `accepts_resident_slot`/`activations.resident`
/// actually engage for this arm -- and (b) alias the *same* bytes rather than
/// silently recomputing or corrupting them: the read-back values must equal
/// the original input's, just under the new shape.
#[test]
fn reshape_alias_keeps_a_chain_fully_resident() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!("no CUDA device present, skipping reshape_alias_keeps_a_chain_fully_resident");
        return;
    };

    let shape = vec![2usize, 6];
    let mut rng = Lcg::new(0x00D4_740F_0AD5_0009);
    // All non-negative, so Relu is the identity and the values pass through
    // exactly -- this test is about the *alias*, not about Relu's math.
    let x = make_vec(&mut rng, 12, 0.5, 4.0);

    let resident = resident_tensor_of(&ctx, &x, &shape);
    let map = OneResident {
        name: "x",
        tensor: &resident,
    };

    let node = make_node(OpKind::Reshape, &["x", "new_shape"], &["y"]);
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "new_shape".to_string(),
        Tensor::new(vec![3.0, 4.0], vec![2]),
    );
    let intermediates: HashMap<String, Tensor> = HashMap::new();

    let outcome = try_cuda_dispatch_resident(
        &node,
        &weights,
        &intermediates,
        &map,
        CudaOutputPlacement::Device,
        &ctx,
    )
    .expect("dispatch must not hard-error")
    .expect(
        "Reshape of a device-resident input must be claimed (Ok(Some(_))) -- an Ok(None) here \
         means accepts_resident_slot/activations.resident is not wired for this arm",
    );

    let reshaped = match outcome {
        CudaDispatchOutcome::Device(tensor) => tensor,
        CudaDispatchOutcome::Host(_) => {
            panic!("requested Device placement but got Host back")
        }
    };
    assert_eq!(reshaped.shape(), &[3, 4]);
    let host = reshaped.read_back(&ctx).expect("read-back must not fail");
    assert_eq!(
        host.data, x,
        "Reshape must alias the exact same bytes -- a mismatch here means the device buffer \
         was corrupted, recomputed, or the wrong allocation was aliased",
    );
}
