use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;

/// Normalize a reduction's `axes` list against `rank`: negative axes resolve
/// as `axis + rank` (matching `reduce_output_shape`,
/// oxionnx-ops/src/math/reduce.rs), and anything that still doesn't land in
/// `0..rank` after normalization fails the whole list.
///
/// Declining outright — rather than silently truncating, or letting
/// `axis as usize` wrap a negative `i64` into a huge `usize` the way `-1`
/// used to become `18446744073709551615` here — means an out-of-range axis
/// always falls back to the CPU kernel, which reports it properly instead of
/// indexing off the end of the shape.
#[cfg(feature = "gpu")]
fn normalize_reduce_axes(axes: &[i64], rank: usize) -> Option<Vec<usize>> {
    let rank_i = i64::try_from(rank).ok()?;
    axes.iter()
        .map(|&a| {
            let n = if a < 0 { a + rank_i } else { a };
            usize::try_from(n).ok().filter(|&u| u < rank)
        })
        .collect()
}

/// Like [`normalize_reduce_axes`], but for the single-axis GPU reduce
/// kernels (`gpu_reduce_sum`/`gpu_reduce_max`/`gpu_reduce_min`), which take
/// one `axis: usize` and have no way to express a multi-axis or
/// full/all-axes reduction. Declines whenever `axes` is not exactly one
/// entry.
#[cfg(feature = "gpu")]
fn normalize_single_reduce_axis(axes: &[i64], rank: usize) -> Option<usize> {
    if axes.len() != 1 {
        return None;
    }
    normalize_reduce_axes(axes, rank)?.first().copied()
}

/// Output shape of a single-axis reduction, matching `reduce_output_shape`'s
/// (oxionnx-ops/src/math/reduce.rs) `keepdims` handling exactly: the axis
/// either collapses to `1` or is dropped outright.
///
/// Dropping the only axis of a rank-1 input leaves the **empty** shape — a
/// genuine rank-0 scalar, which is what ONNX (and NumPy:
/// `np.sum(np.arange(5), axis=0, keepdims=False).shape == ()`) specifies for a
/// fully-reduced input with `keepdims=0`. This used to fall back to `[1]`, which
/// made the GPU arm's result disagree with the CPU kernel it is standing in for:
/// the same graph would report a different output *rank* depending on whether the
/// GPU accepted the node, and anything downstream driven by a `Shape` node would
/// then see one dimension too many. Element count is identical either way (the
/// empty shape's product is the empty product 1), so the `Tensor::new` calls at
/// the call sites are unaffected.
///
/// `axis` must already be `< input_shape.len()` (see
/// [`normalize_single_reduce_axis`]) — this indexes it unconditionally.
#[cfg(feature = "gpu")]
fn single_axis_reduce_shape(input_shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        let mut out = input_shape.to_vec();
        out[axis] = 1;
        out
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == axis { None } else { Some(d) })
            .collect()
    }
}

/// [a4-11/a7-1/a7-9] Decide whether the `MatMul` arm may dispatch to the flat
/// 2-D `gpu_matmul` kernel (`m,k,n` only — no batching support at all), and
/// if so, what `(m, k, n, out_shape)` it should be dispatched with.
///
/// Mirrors the CPU `matmul` (oxionnx-ops/src/math/matmul.rs) inner-dimension
/// check and batch broadcast exactly (reusing the very same
/// `Tensor::broadcast_shape` helper), but only *accepts* the trivial case —
/// broadcast batch size `1` — where every leading dim on both operands is 1,
/// so `a`'s and `b`'s data buffers are already contiguous `[m,k]`/`[k,n]`
/// matrices and a flat 2-D matmul is exact, not an approximation. Any real
/// batching on either operand (`a`, `b`, or both) returns `None`: the CPU
/// path implements broadcasting fully and must handle it instead.
#[cfg(feature = "gpu")]
fn matmul_gpu_plan(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Option<(usize, usize, usize, Vec<usize>)> {
    let an = a_shape.len();
    let bn = b_shape.len();
    if an < 2 || bn < 2 {
        return None;
    }
    let m = a_shape[an - 2];
    let k = a_shape[an - 1];
    let k2 = b_shape[bn - 2];
    let n = b_shape[bn - 1];
    if k != k2 {
        return None;
    }
    let out_batch = Tensor::broadcast_shape(&a_shape[..an - 2], &b_shape[..bn - 2]).ok()?;
    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    if batch_size != 1 {
        return None;
    }
    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
    Some((m, k, n, out_shape))
}

/// [a4-12/a7-0] Whether the (possibly negative) ONNX `axis` attribute
/// resolves to the last dimension of a tensor with `rank` dimensions — the
/// only axis `gpu_softmax` (oxionnx-gpu/src/shaders/softmax.rs) can compute,
/// since it hard-codes the last dim as the reduction axis.
#[cfg(feature = "gpu")]
fn softmax_axis_is_last_dim(axis: i64, rank: usize) -> bool {
    if rank == 0 {
        return false;
    }
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    normalized == rank_i - 1
}

/// [a4-18] Whether `Add`/`Mul` may dispatch to the flat elementwise
/// `gpu_add`/`gpu_mul` kernels (oxionnx-gpu/src/shaders/elementwise.rs),
/// which implement no broadcasting at all. Equal element counts do not imply
/// equal shapes — `[1,6]` and `[6,1]` both have 6 elements but broadcast to a
/// 36-element `[6,6]` result — so this checks shape equality, not length.
#[cfg(feature = "gpu")]
fn elementwise_shapes_match(a_shape: &[usize], b_shape: &[usize]) -> bool {
    a_shape == b_shape
}

/// [a7-5] Whether the Conv arm recognises `activation` well enough to decide
/// what to do with it. Only `"relu"`, `"clip"`, or absent (`""`) are ever
/// emitted by the optimizer's Conv+activation fusion passes
/// (src/optimizer/fusion/conv/{relu,relu6}.rs) — anything else must decline
/// the GPU path rather than silently drop an activation it doesn't
/// recognise.
#[cfg(feature = "gpu")]
fn conv_activation_is_recognized(activation: &str) -> bool {
    matches!(activation, "" | "relu" | "clip")
}

/// [a7-5] Apply the optimizer's fused Conv activation in place, mirroring
/// `apply_fused_activation` (oxionnx-ops/src/registry/conv_ops/conv.rs,
/// private to that crate and so not directly reusable here) exactly: `"relu"`
/// clamps to `>= 0`, `"clip"` clamps to `[min_val, max_val]`, anything else
/// (in practice just `""`) is a no-op.
#[cfg(feature = "gpu")]
fn apply_conv_activation(activation: &str, min_val: f32, max_val: f32, data: &mut [f32]) {
    match activation {
        "relu" => {
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
        "clip" => {
            for v in data.iter_mut() {
                *v = v.clamp(min_val, max_val);
            }
        }
        _ => {}
    }
}

/// [conv-pool report] Validate + convert a 2-entry spatial attribute
/// (`strides`, `dilations`) the same way the CPU kernel does
/// (`read_positive_pair`, oxionnx-ops/src/registry/conv_ops/conv.rs, private
/// to that crate and so not directly reusable here): every *present* entry
/// must be `>= 1`; missing entries fall back to `default`. Declines
/// (`None`) otherwise.
///
/// Without this gate, a malformed model's `dilations=[-1, 1]` survives the
/// arm's raw `as usize` cast as `usize::MAX`, which used to reach
/// `conv_same_pad_split`'s `+ 1` on a `saturating_mul`-derived `usize::MAX`
/// and overflow — a debug-build panic, and release-mode garbage pads. The
/// CPU kernel's `gpu_conv2d` counterpart is safe from this class of input
/// because it threads everything through `checked_mul`/`checked_add`, but
/// the SAME_UPPER/SAME_LOWER pad-splitting math introduced by this fix
/// wasn't, so it needs its own gate rather than relying on `gpu_conv2d`'s.
#[cfg(feature = "gpu")]
fn read_positive_pair_gpu(values: &[i64], default: usize) -> Option<[usize; 2]> {
    let mut out = [default; 2];
    for (axis, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(axis) {
            if v < 1 {
                return None;
            }
            *slot = usize::try_from(v).ok()?;
        }
    }
    Some(out)
}

/// [conv-pool report] Validate + convert the `pads` attribute the same way
/// the CPU kernel does (`read_pads_2d`, oxionnx-ops/src/registry/conv_ops/conv.rs,
/// private to that crate): every *present* entry must be `>= 0`. Declines
/// (`None`) on a negative entry instead of letting `as usize` wrap it into
/// an enormous padding amount.
#[cfg(feature = "gpu")]
fn read_pads_gpu(values: &[i64]) -> Option<[usize; 4]> {
    let mut out = [0_usize; 4];
    for (idx, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(idx) {
            if v < 0 {
                return None;
            }
            *slot = usize::try_from(v).ok()?;
        }
    }
    Some(out)
}

/// [conv-pool report] Validate + convert the `group` attribute the same way
/// the CPU kernel does (`read_group`, oxionnx-ops/src/registry/conv_ops/conv.rs,
/// private to that crate): must be `>= 1`. Declines (`None`) otherwise
/// instead of letting `as usize` turn a negative group into `usize::MAX`
/// (which `gpu_conv2d`'s `c_out % group != 0` check happens to reject for
/// every `c_out` except the astronomically unlikely `c_out == usize::MAX`,
/// but that's incidental, not a guarantee this call site should lean on).
#[cfg(feature = "gpu")]
fn read_group_gpu(group: i64) -> Option<usize> {
    if group < 1 {
        return None;
    }
    usize::try_from(group).ok()
}

/// SAME_UPPER / SAME_LOWER padding split for one spatial axis. Mirrors
/// `same_pad_split` (oxionnx-ops/src/registry/conv_ops/conv.rs, private to
/// that crate and so not directly reusable here) exactly: target extent
/// `ceil(in / stride)`, total padding
/// `(out - 1) * stride + ((k - 1) * dilation + 1) - in` clamped at zero, and
/// the odd pixel goes to the end (`SAME_UPPER`) or the beginning
/// (`SAME_LOWER`).
///
/// `kernel`/`stride`/`dilation` are assumed already validated (`>= 1`, via
/// [`read_positive_pair_gpu`]) by every current caller. The arithmetic is
/// nonetheless fully saturating rather than relying on that precondition —
/// including the `eff_k` step, which used to be a bare `+ 1` on a
/// `saturating_mul` result and could overflow-panic in debug builds when a
/// caller ever passed an unvalidated `dilation` — so a future caller that
/// forgets to validate degrades to a saturated (and CPU-declined-anyway,
/// since callers only use this for the SAME_UPPER/SAME_LOWER branch of
/// `resolve_conv_pads_for_gpu`) padding value instead of panicking.
#[cfg(feature = "gpu")]
fn conv_same_pad_split(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    lower: bool,
) -> (usize, usize) {
    let out_dim = in_dim.div_ceil(stride.max(1));
    let eff_k = kernel
        .saturating_sub(1)
        .saturating_mul(dilation)
        .saturating_add(1);
    let needed = out_dim
        .saturating_sub(1)
        .saturating_mul(stride)
        .saturating_add(eff_k)
        .saturating_sub(in_dim);
    let half = needed / 2;
    if lower {
        (needed - half, half)
    } else {
        (half, needed - half)
    }
}

/// [conv-pool report] Resolve Conv's effective `[begin_h, begin_w, end_h,
/// end_w]` padding the same way the CPU kernel does (`resolve_pads_2d` +
/// `parse_auto_pad`, oxionnx-ops/src/registry/conv_ops/conv.rs), or decline
/// (`None`) when the GPU arm cannot: an unrecognized `auto_pad` string, or
/// `SAME_UPPER`/`SAME_LOWER` without a 4-D `[N,C,H,W]` input/weight shape to
/// read `H`/`W`/`kH`/`kW` from.
///
/// Before this, the arm read only `attrs.ints("pads")` and never looked at
/// `auto_pad` at all, so a `SAME_UPPER`/`SAME_LOWER`/`VALID` Conv dispatched
/// to the GPU silently convolved with whatever the (usually absent, so
/// all-zero) explicit `pads` happened to be — wrong output shape, wrong
/// values, no error. Declining here — rather than falling back to the
/// explicit `pads` attribute — routes those models to the CPU kernel, which
/// already resolves `auto_pad` correctly.
///
/// `NotSet` and `Valid` need no shape information at all (the former is the
/// explicit `pads` verbatim, the latter is always all-zero), so both resolve
/// even when the ranks are wrong; `gpu_conv2d`'s own rank check declines the
/// dispatch afterwards, exactly as it already did before this fix.
#[cfg(feature = "gpu")]
fn resolve_conv_pads_for_gpu(
    auto_pad_raw: &str,
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: [usize; 2],
    dilations: [usize; 2],
    explicit: [usize; 4],
) -> Option<[usize; 4]> {
    match auto_pad_raw {
        "" | "NOTSET" => Some(explicit),
        "VALID" => Some([0, 0, 0, 0]),
        "SAME_UPPER" | "SAME_LOWER" => {
            if input_shape.len() != 4 || weight_shape.len() != 4 {
                return None;
            }
            let lower = auto_pad_raw == "SAME_LOWER";
            let mut out = [0_usize; 4];
            for axis in 0..2 {
                let (begin, end) = conv_same_pad_split(
                    input_shape[axis + 2],
                    weight_shape[axis + 2],
                    strides[axis],
                    dilations[axis],
                    lower,
                );
                out[axis] = begin;
                out[axis + 2] = end;
            }
            Some(out)
        }
        // Unrecognized auto_pad value: decline outright so the CPU kernel's
        // `parse_auto_pad` reports the typed error instead of this arm
        // silently guessing.
        _ => None,
    }
}

/// Provides metadata about GPU-accelerated operator support.
#[cfg(feature = "gpu")]
pub struct GpuExecutionProvider;

#[cfg(feature = "gpu")]
impl GpuExecutionProvider {
    /// Return the list of operator types that have GPU acceleration.
    ///
    /// [a7-19] Derived from `crate::execution_providers::GPU_DISPATCH_OPS`
    /// — the same array [`crate::execution_providers::is_gpu_capable`] is
    /// built on — via `OpKind::as_str()`, so this list can no longer
    /// independently drift from either `is_gpu_capable` or the
    /// `try_gpu_dispatch` match arms that `GPU_DISPATCH_OPS` mirrors. It used
    /// to be a third, separately hand-maintained list that omitted ops
    /// `try_gpu_dispatch` actually implements (Gelu, the three single-axis
    /// reduces, and the elementwise unary kernels) while including ops that
    /// happened to agree with the other two lists only by chance.
    ///
    /// Computed once and memoized: `OpKind::as_str()` borrows from `self`, so
    /// turning a `&'static [OpKind]` into a `&'static [&'static str]` needs
    /// somewhere to own the resulting `Vec` for `'static`.
    pub fn supported_ops() -> &'static [&'static str] {
        static OPS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
        OPS.get_or_init(|| {
            crate::execution_providers::GPU_DISPATCH_OPS
                .iter()
                .map(|op| op.as_str())
                .collect()
        })
        .as_slice()
    }

    /// Check whether a given operator type is GPU-accelerated.
    pub fn is_supported(op_type: &str) -> bool {
        Self::supported_ops().contains(&op_type)
    }
}

/// GPU dispatch for ops with hardware acceleration (MatMul, Conv).
/// Returns `Ok(Some(results))` if GPU handled it, `Ok(None)` to fall back to CPU.
#[cfg(feature = "gpu")]
pub(crate) fn try_gpu_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    gpu: &crate::gpu::GpuContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    match &node.op {
        OpKind::MatMul => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // [a4-11/a7-1/a7-9] See `matmul_gpu_plan`: declines whenever
                // either operand carries a real batch dimension the flat
                // `gpu_matmul` kernel cannot express, and otherwise returns
                // the *correct* output shape (batch prefix included) instead
                // of the bare `[m, n]` this arm used to hand back regardless
                // of input rank.
                if let Some((m, k, n, out_shape)) = matmul_gpu_plan(&a.shape, &b.shape) {
                    if let Some(result) = crate::gpu::gpu_matmul(gpu, &a.data, &b.data, m, k, n) {
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Conv => {
            let input = resolve(&node.inputs[0]);
            let weight = resolve(&node.inputs[1]);
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(input), Some(weight)) = (input, weight) {
                let attrs = &node.attrs;
                // [a7-5] The optimizer's Conv+Relu/Conv+Clip(0,6) fusion
                // (src/optimizer/fusion/conv/{relu,relu6}.rs) folds the
                // activation into this same Conv node as the `activation`
                // string attribute ("relu", or "clip" plus
                // activation_min/activation_max); `ConvOp::execute`
                // (oxionnx-ops/src/registry/conv_ops/conv.rs,
                // `apply_fused_activation`) applies it after the
                // convolution. Only "relu"/"clip"/absent are ever emitted by
                // the fusion passes — decline anything else outright rather
                // than silently drop an activation this dispatcher doesn't
                // recognise.
                let activation = attrs.s("activation");
                if conv_activation_is_recognized(activation) {
                    // [conv-pool report] Validate every spatial attribute
                    // the same way the CPU kernel does *before* converting
                    // to `usize` — a negative `strides`/`dilations`/`pads`
                    // entry, or `group < 1`, must decline to the CPU
                    // kernel (which reports the typed error) rather than
                    // silently wrapping into `usize::MAX` and feeding it to
                    // `resolve_conv_pads_for_gpu`'s arithmetic. See
                    // `read_positive_pair_gpu`/`read_pads_gpu`/`read_group_gpu`.
                    let strides_opt = read_positive_pair_gpu(attrs.ints("strides"), 1);
                    let dilations_opt = read_positive_pair_gpu(attrs.ints("dilations"), 1);
                    let explicit_pads_opt = read_pads_gpu(attrs.ints("pads"));
                    let group_opt = read_group_gpu(attrs.i("group", 1));
                    if let (Some(strides), Some(dilations), Some(explicit_pads), Some(group)) =
                        (strides_opt, dilations_opt, explicit_pads_opt, group_opt)
                    {
                        // [conv-pool report] `auto_pad` overrides the explicit
                        // `pads` attribute for every mode but NOTSET — see
                        // `resolve_conv_pads_for_gpu`. Reading only `pads` here
                        // (the pre-fix behaviour) silently convolved SAME_UPPER /
                        // SAME_LOWER models unpadded.
                        if let Some(pads) = resolve_conv_pads_for_gpu(
                            attrs.s("auto_pad"),
                            &input.shape,
                            &weight.shape,
                            strides,
                            dilations,
                            explicit_pads,
                        ) {
                            if let Some(mut result) = crate::gpu::gpu_conv2d(
                                gpu, input, weight, bias, strides, pads, dilations, group,
                            ) {
                                let min_val = attrs.f("activation_min", f32::NEG_INFINITY);
                                let max_val = attrs.f("activation_max", f32::INFINITY);
                                apply_conv_activation(
                                    activation,
                                    min_val,
                                    max_val,
                                    &mut result.data,
                                );
                                return Ok(Some(vec![result]));
                            }
                        }
                    }
                }
            }
            Ok(None)
        }
        OpKind::Softmax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                // [a4-12/a7-0] The CPU kernel (oxionnx-ops/src/nn/normalization.rs)
                // honours `axis` (default -1, the same default used here) for
                // any value, so a model with a non-default axis would
                // silently get a different reduction axis depending only on
                // whether the `gpu` feature and a device happen to be
                // present. See `softmax_axis_is_last_dim`.
                let axis = node.attrs.i("axis", -1);
                if softmax_axis_is_last_dim(axis, input.shape.len()) {
                    if let Some(result) = crate::gpu::gpu_softmax(gpu, &input.data, &input.shape) {
                        return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Relu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_relu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Sigmoid => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_sigmoid(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Gelu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_gelu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::ReduceSum => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                // [a4-17/a7-7] Read `axes` the same way the CPU path does
                // (`axes_from_ctx`,
                // oxionnx-ops/src/registry/math_ops/reduce.rs): prefer the
                // opset-13+ tensor input, fall back to the attribute.
                let axes_raw: Vec<i64> =
                    if let Some(axes_input) = node.inputs.get(1).and_then(|n| resolve(n)) {
                        axes_input.data.iter().map(|&v| v as i64).collect()
                    } else {
                        node.attrs.ints("axes").to_vec()
                    };
                if let Some(axis) = normalize_single_reduce_axis(&axes_raw, input.shape.len()) {
                    if let Some(result) =
                        crate::gpu::gpu_reduce_sum(gpu, &input.data, axis, &input.shape)
                    {
                        let keepdims = node.attrs.i("keepdims", 1) != 0;
                        let out_shape = single_axis_reduce_shape(&input.shape, axis, keepdims);
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::ReduceMax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes_raw: Vec<i64> =
                    if let Some(axes_input) = node.inputs.get(1).and_then(|n| resolve(n)) {
                        axes_input.data.iter().map(|&v| v as i64).collect()
                    } else {
                        node.attrs.ints("axes").to_vec()
                    };
                if let Some(axis) = normalize_single_reduce_axis(&axes_raw, input.shape.len()) {
                    if let Some(result) =
                        crate::gpu::gpu_reduce_max(gpu, &input.data, axis, &input.shape)
                    {
                        let keepdims = node.attrs.i("keepdims", 1) != 0;
                        let out_shape = single_axis_reduce_shape(&input.shape, axis, keepdims);
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::ReduceMin => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes_raw: Vec<i64> =
                    if let Some(axes_input) = node.inputs.get(1).and_then(|n| resolve(n)) {
                        axes_input.data.iter().map(|&v| v as i64).collect()
                    } else {
                        node.attrs.ints("axes").to_vec()
                    };
                if let Some(axis) = normalize_single_reduce_axis(&axes_raw, input.shape.len()) {
                    if let Some(result) =
                        crate::gpu::gpu_reduce_min(gpu, &input.data, axis, &input.shape)
                    {
                        let keepdims = node.attrs.i("keepdims", 1) != 0;
                        let out_shape = single_axis_reduce_shape(&input.shape, axis, keepdims);
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Tanh => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_tanh(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Exp => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_exp(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Sqrt => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_sqrt(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Abs => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_abs(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Neg => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_neg(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Log => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_log(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::SiLU => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_silu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::LeakyRelu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                // [a7-8] The node's `alpha` must reach the kernel: the
                // alpha-less entry point hardcodes the ONNX default 0.01, so a
                // YOLOv3-style `alpha = 0.1` model used to get every negative
                // activation scaled 10x too small — with the correct output
                // shape, so nothing downstream could detect it.
                let alpha = node.attrs.f("alpha", 0.01);
                if let Some(result) = crate::gpu::gpu_leaky_relu_alpha(gpu, &input.data, alpha) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Add => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // [a4-18] See `elementwise_shapes_match`: equal element
                // counts (the old gate) do not imply equal shapes, and
                // `gpu_add` has no broadcasting.
                if elementwise_shapes_match(&a.shape, &b.shape) {
                    if let Some(result) = crate::gpu::gpu_add(gpu, &a.data, &b.data) {
                        return Ok(Some(vec![Tensor::new(result, a.shape.clone())]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Mul => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // [a4-18] Same fix as `Add` above — see `elementwise_shapes_match`.
                if elementwise_shapes_match(&a.shape, &b.shape) {
                    if let Some(result) = crate::gpu::gpu_mul(gpu, &a.data, &b.data) {
                        return Ok(Some(vec![Tensor::new(result, a.shape.clone())]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::LayerNorm => {
            let input = resolve(&node.inputs[0]);
            let scale = node.inputs.get(1).and_then(|n| resolve(n));
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(input), Some(scale), Some(bias)) = (input, scale, bias) {
                let eps = node.attrs.f("epsilon", 1e-5);
                // [a7-6] `gpu_layer_norm` hardcodes `axis = -1` and declines
                // whenever scale/bias do not match that suffix, so a
                // non-last-axis LayerNorm silently fell back to the CPU.
                // Passing the node's `axis` lets the GPU handle it, and the
                // kernel still declines (returns `None`) when the axis is
                // out of range for the input rank.
                let axis = node.attrs.i("axis", -1);
                if let Some(result) = crate::gpu::gpu_layer_norm_axis(
                    gpu,
                    &input.data,
                    &input.shape,
                    &scale.data,
                    &bias.data,
                    eps,
                    axis,
                ) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::BatchNorm => {
            let input = resolve(&node.inputs[0]);
            let scale = node.inputs.get(1).and_then(|n| resolve(n));
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            let mean = node.inputs.get(3).and_then(|n| resolve(n));
            let var = node.inputs.get(4).and_then(|n| resolve(n));
            if let (Some(input), Some(scale), Some(bias), Some(mean), Some(var)) =
                (input, scale, bias, mean, var)
            {
                let eps = node.attrs.f("epsilon", 1e-5);
                if let Some(result) = crate::gpu::gpu_batch_norm(
                    gpu,
                    &input.data,
                    &input.shape,
                    &scale.data,
                    &bias.data,
                    &mean.data,
                    &var.data,
                    eps,
                ) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Transpose => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let perm_attr = node.attrs.ints("perm");
                let perm: Vec<usize> = if perm_attr.is_empty() {
                    // Default: reverse dimensions
                    (0..input.shape.len()).rev().collect()
                } else {
                    perm_attr.iter().map(|&p| p as usize).collect()
                };
                if let Some(result) =
                    crate::gpu::gpu_transpose(gpu, &input.data, &input.shape, &perm)
                {
                    let out_shape: Vec<usize> = perm.iter().map(|&p| input.shape[p]).collect();
                    return Ok(Some(vec![Tensor::new(result, out_shape)]));
                }
            }
            Ok(None)
        }
        OpKind::ReduceMean => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                // [a4-17] Same axes-resolution priority as the other reduce
                // arms: prefer the opset-18 tensor input over the attribute.
                let axes_raw: Vec<i64> =
                    if let Some(axes_input) = node.inputs.get(1).and_then(|n| resolve(n)) {
                        axes_input.data.iter().map(|&v| v as i64).collect()
                    } else {
                        node.attrs.ints("axes").to_vec()
                    };
                let keepdims = node.attrs.i("keepdims", 1) != 0;
                if let Some(axes) = normalize_reduce_axes(&axes_raw, input.shape.len()) {
                    if !axes.is_empty() {
                        if let Some(result) = crate::gpu::gpu_reduce_mean(
                            gpu,
                            &input.data,
                            &input.shape,
                            &axes,
                            keepdims,
                        ) {
                            let mut out_shape = input.shape.clone();
                            if keepdims {
                                for &a in &axes {
                                    out_shape[a] = 1;
                                }
                            } else {
                                let mut sorted_axes = axes.clone();
                                sorted_axes.sort_unstable();
                                // A malformed model could repeat an axis;
                                // collapse duplicates to match
                                // `reduce_output_shape`'s `axes.contains(&i)`
                                // set semantics (each axis position is
                                // removed exactly once) instead of removing
                                // the same shape position twice, which would
                                // underflow `a - offset` on the second
                                // removal and panic.
                                sorted_axes.dedup();
                                for (offset, &a) in sorted_axes.iter().enumerate() {
                                    out_shape.remove(a - offset);
                                }
                                if out_shape.is_empty() {
                                    out_shape.push(1);
                                }
                            }
                            return Ok(Some(vec![Tensor::new(result, out_shape)]));
                        }
                    }
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

#[cfg(all(test, feature = "gpu"))]
mod zzz_probe {
    #[test]
    fn zzz_probe_gpu_availability() {
        let available = crate::gpu::GpuContext::try_new().is_some();
        eprintln!("PROBE_GPU_AVAILABLE={available}");
    }
}

/// Shape/attribute gating unit tests for every decline decision fixed in
/// this file. None of these need a live GPU adapter — they exercise the
/// pure helper functions `try_gpu_dispatch`'s match arms consult *before*
/// ever touching `crate::gpu`, so they run on every CI machine regardless of
/// Metal/Vulkan/DX12 availability.
#[cfg(all(test, feature = "gpu"))]
mod gating_tests {
    use super::*;

    // ── matmul_gpu_plan [a4-11/a7-1/a7-9] ───────────────────────────────────

    #[test]
    fn matmul_plan_preserves_leading_batch_dim_of_one() {
        // The canonical transformer projection: MatMul(A[1,128,768], B[768,768]).
        // Before the fix this returned Tensor::new(result, vec![m, n]) = [128, 768]
        // unconditionally — a silent rank drop from 3 to 2.
        let plan = matmul_gpu_plan(&[1, 128, 768], &[768, 768]).expect("batch of 1 must dispatch");
        assert_eq!(plan, (128, 768, 768, vec![1, 128, 768]));
    }

    #[test]
    fn matmul_plan_plain_2d_has_no_batch_prefix() {
        let plan = matmul_gpu_plan(&[4, 8], &[8, 16]).expect("plain 2-D matmul must dispatch");
        assert_eq!(plan, (4, 8, 16, vec![4, 16]));
    }

    #[test]
    fn matmul_plan_batch_of_one_on_both_operands() {
        let plan = matmul_gpu_plan(&[1, 4, 8], &[1, 8, 16])
            .expect("batch of 1 on both sides must dispatch");
        assert_eq!(plan, (4, 8, 16, vec![1, 4, 16]));
    }

    #[test]
    fn matmul_plan_declines_real_batch_on_a() {
        // a is 2-D with batch_size==1 trivially, but a real batch on `a` alone
        // must still decline: [2,4,8] @ [8,16] is not expressible by the flat
        // gpu_matmul kernel.
        assert_eq!(matmul_gpu_plan(&[2, 4, 8], &[8, 16]), None);
    }

    #[test]
    fn matmul_plan_declines_real_batch_on_b_even_when_a_is_2d() {
        // [a7-9] a=[768,768] alone looks batch-free (an=2), but b carries a
        // real batch of 2 that the flat kernel would silently truncate to
        // the first slice if this weren't checked against the *broadcast*
        // of both operands' batch prefixes.
        assert_eq!(matmul_gpu_plan(&[768, 768], &[2, 768, 512]), None);
    }

    #[test]
    fn matmul_plan_declines_mismatched_inner_dimension() {
        assert_eq!(matmul_gpu_plan(&[4, 8], &[9, 16]), None);
    }

    #[test]
    fn matmul_plan_declines_rank_below_two() {
        assert_eq!(matmul_gpu_plan(&[8], &[8, 16]), None);
        assert_eq!(matmul_gpu_plan(&[4, 8], &[8]), None);
    }

    // ── softmax_axis_is_last_dim [a4-12/a7-0] ───────────────────────────────

    #[test]
    fn softmax_axis_negative_one_is_always_last_dim() {
        assert!(softmax_axis_is_last_dim(-1, 3));
        assert!(softmax_axis_is_last_dim(-1, 1));
    }

    #[test]
    fn softmax_axis_explicit_last_index_matches() {
        assert!(softmax_axis_is_last_dim(2, 3));
    }

    #[test]
    fn softmax_axis_one_on_rank_three_is_not_last_dim() {
        // The exact [a7-0] regression case: Softmax(axis=1) on an [8,4,1024]
        // tensor. The GPU kernel can only reduce the last (1024) axis; axis=1
        // (size 4) must decline, not silently normalize over the wrong axis.
        assert!(!softmax_axis_is_last_dim(1, 3));
    }

    #[test]
    fn softmax_axis_rank_zero_never_matches() {
        assert!(!softmax_axis_is_last_dim(-1, 0));
        assert!(!softmax_axis_is_last_dim(0, 0));
    }

    // ── normalize_reduce_axes / normalize_single_reduce_axis [a4-17/a7-7] ──

    #[test]
    fn normalize_reduce_axes_resolves_negative_axes() {
        assert_eq!(normalize_reduce_axes(&[-1], 4), Some(vec![3]));
        assert_eq!(normalize_reduce_axes(&[1, -1], 4), Some(vec![1, 3]));
    }

    #[test]
    fn normalize_reduce_axes_declines_out_of_range() {
        // `axis as usize` used to turn -1 into 18446744073709551615 and index
        // straight off the end of the shape; this must decline instead.
        assert_eq!(normalize_reduce_axes(&[5], 4), None);
        assert_eq!(normalize_reduce_axes(&[-5], 4), None);
    }

    #[test]
    fn normalize_single_reduce_axis_matches_a7_7_example() {
        // ReduceSum(axes=[-1], keepdims=0) on a [100000, 3] tensor.
        assert_eq!(normalize_single_reduce_axis(&[-1], 2), Some(1));
    }

    #[test]
    fn normalize_single_reduce_axis_declines_non_singleton_lists() {
        assert_eq!(normalize_single_reduce_axis(&[], 2), None);
        assert_eq!(normalize_single_reduce_axis(&[0, 1], 2), None);
    }

    // ── single_axis_reduce_shape [a4-17/a7-7] ───────────────────────────────

    #[test]
    fn single_axis_reduce_shape_matches_a7_7_example() {
        // ReduceSum(axes=[1], keepdims=0) on [100000, 3] must produce [100000],
        // not the keepdims=1 shape [100000, 1] the pre-fix code always emitted.
        assert_eq!(
            single_axis_reduce_shape(&[100_000, 3], 1, false),
            vec![100_000]
        );
        assert_eq!(
            single_axis_reduce_shape(&[100_000, 3], 1, true),
            vec![100_000, 1]
        );
    }

    #[test]
    fn single_axis_reduce_shape_full_reduction_is_rank0() {
        // ONNX `ReduceSum` with `keepdims=0` *removes* the reduced axes, so a
        // fully-reduced rank-1 input is a rank-0 scalar: shape `[]`, not `[1]`
        // (`np.sum(np.arange(5), axis=0, keepdims=False).shape == ()`). This must
        // match `reduce_output_shape`/`reduce_with` in
        // oxionnx-ops/src/math/reduce.rs, which the CPU fallback goes through —
        // otherwise the reported output rank would depend on whether the GPU arm
        // happened to accept the node.
        let rank0: Vec<usize> = Vec::new();
        let got = single_axis_reduce_shape(&[5], 0, false);
        assert_eq!(got, rank0);
        // The element count is unchanged: the empty shape's product is 1.
        assert_eq!(got.iter().product::<usize>(), 1);
        // `keepdims=1` is deliberately untouched by the migration.
        assert_eq!(single_axis_reduce_shape(&[5], 0, true), vec![1]);
    }

    /// The CPU kernel the GPU arm stands in for must agree, dimension for
    /// dimension, on every case `single_axis_reduce_shape` claims to handle —
    /// this is the cross-check that keeps the two from drifting apart again.
    #[test]
    fn single_axis_reduce_shape_agrees_with_the_cpu_reduce_kernel() {
        for (shape, axis) in [
            (vec![5usize], 0usize),
            (vec![4, 3], 0),
            (vec![4, 3], 1),
            (vec![2, 3, 5], 1),
            (vec![1, 1], 0),
        ] {
            for keepdims in [false, true] {
                let n: usize = shape.iter().product();
                let x = Tensor::new(vec![1.0_f32; n], shape.clone());
                let want = oxionnx_ops::math::reduce_sum(&x, &[axis as i64], keepdims)
                    .expect("cpu reduce_sum runs");
                assert_eq!(
                    single_axis_reduce_shape(&shape, axis, keepdims),
                    want.shape,
                    "shape={shape:?} axis={axis} keepdims={keepdims}"
                );
            }
        }
    }

    // ── elementwise_shapes_match [a4-18] ────────────────────────────────────

    #[test]
    fn elementwise_shapes_reject_equal_element_count_unequal_shape() {
        // [1,6] and [6,1] both have 6 elements but must broadcast to a
        // 36-element [6,6] result — the flat kernel cannot do that.
        assert!(!elementwise_shapes_match(&[1, 6], &[6, 1]));
        assert!(!elementwise_shapes_match(&[2, 3], &[3, 2]));
    }

    #[test]
    fn elementwise_shapes_accept_identical_shapes() {
        assert!(elementwise_shapes_match(&[4, 5], &[4, 5]));
        assert!(elementwise_shapes_match(&[], &[]));
    }

    // ── conv_activation_is_recognized / apply_conv_activation [a7-5] ───────

    #[test]
    fn conv_activation_recognizes_only_the_fusion_pass_outputs() {
        assert!(conv_activation_is_recognized(""));
        assert!(conv_activation_is_recognized("relu"));
        assert!(conv_activation_is_recognized("clip"));
        assert!(!conv_activation_is_recognized("sigmoid"));
        assert!(!conv_activation_is_recognized("relu6"));
    }

    #[test]
    fn apply_conv_activation_relu_zeroes_negatives() {
        let mut data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        apply_conv_activation("relu", f32::NEG_INFINITY, f32::INFINITY, &mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn apply_conv_activation_clip_clamps_to_range() {
        let mut data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        apply_conv_activation("clip", 0.0, 1.0, &mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn apply_conv_activation_unrecognized_is_a_no_op() {
        let mut data = vec![-2.0, 0.5];
        apply_conv_activation("bogus", 0.0, 1.0, &mut data);
        assert_eq!(data, vec![-2.0, 0.5]);
    }

    // ── resolve_conv_pads_for_gpu / conv_same_pad_split [conv-pool report] ─

    #[test]
    fn conv_pads_notset_uses_explicit_pads_verbatim() {
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [1, 2, 3, 4],
            ),
            Some([1, 2, 3, 4]),
        );
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "NOTSET",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [1, 2, 3, 4],
            ),
            Some([1, 2, 3, 4]),
        );
    }

    #[test]
    fn conv_pads_valid_is_always_zero_regardless_of_explicit_pads() {
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "VALID",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [9, 9, 9, 9],
            ),
            Some([0, 0, 0, 0]),
        );
    }

    #[test]
    fn conv_pads_same_upper_matches_hand_computed_values() {
        // input 7x7, kernel 3x3, stride 2, dilation 1, SAME_UPPER:
        //   out = ceil(7/2) = 4
        //   eff_k = 3
        //   needed = (4-1)*2 + 3 - 7 = 2  → half=1, split (1,1)
        // Verified: padded = 7+1+1=9, (9-3)/2+1 = 4 == out. The explicit
        // pads attribute [9,9,9,9] must be entirely ignored — this is the
        // exact conv-pool-reported bug (auto_pad was never read at all).
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_UPPER",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [2, 2],
                [1, 1],
                [9, 9, 9, 9],
            ),
            Some([1, 1, 1, 1]),
        );
    }

    #[test]
    fn conv_pads_same_upper_vs_same_lower_split_the_odd_pixel_differently() {
        // input 8x8, kernel 3x3, stride 2, dilation 1:
        //   out = ceil(8/2) = 4, eff_k = 3
        //   needed = (4-1)*2 + 3 - 8 = 1 → half=0
        //   SAME_UPPER: (0,1) → odd pixel at the end
        //   SAME_LOWER: (1,0) → odd pixel at the beginning
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_UPPER",
                &[1, 3, 8, 8],
                &[8, 3, 3, 3],
                [2, 2],
                [1, 1],
                [0, 0, 0, 0],
            ),
            Some([0, 0, 1, 1]),
        );
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_LOWER",
                &[1, 3, 8, 8],
                &[8, 3, 3, 3],
                [2, 2],
                [1, 1],
                [0, 0, 0, 0],
            ),
            Some([1, 1, 0, 0]),
        );
    }

    #[test]
    fn conv_pads_same_upper_declines_when_shape_rank_is_wrong() {
        // NotSet/Valid need no shape info and still resolve; SAME_UPPER /
        // SAME_LOWER need H/W/kH/kW from a 4-D shape and must decline
        // (falling back to the CPU kernel) rather than guess when the model
        // is malformed.
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_UPPER",
                &[1, 3, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [0; 4]
            ),
            None,
        );
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_LOWER",
                &[1, 3, 7, 7],
                &[8, 3, 3],
                [1, 1],
                [1, 1],
                [0; 4]
            ),
            None,
        );
    }

    #[test]
    fn conv_pads_unrecognized_auto_pad_declines() {
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "BOGUS",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [0; 4]
            ),
            None,
        );
    }

    // ── read_positive_pair_gpu / read_pads_gpu / read_group_gpu ────────────
    // [conv-pool report follow-up] These gate every value that flows into
    // `resolve_conv_pads_for_gpu`/`conv_same_pad_split`, closing the panic
    // reported below.

    #[test]
    fn read_positive_pair_gpu_accepts_valid_values_and_defaults() {
        assert_eq!(read_positive_pair_gpu(&[2, 3], 1), Some([2, 3]));
        assert_eq!(read_positive_pair_gpu(&[], 1), Some([1, 1]));
        assert_eq!(read_positive_pair_gpu(&[5], 1), Some([5, 1]));
    }

    #[test]
    fn read_positive_pair_gpu_rejects_non_positive_entries() {
        // The exact reported repro: a malformed `dilations=[-1, 1]`.
        assert_eq!(read_positive_pair_gpu(&[-1, 1], 1), None);
        assert_eq!(read_positive_pair_gpu(&[1, 0], 1), None);
    }

    #[test]
    fn read_pads_gpu_accepts_non_negative_values_and_defaults() {
        assert_eq!(read_pads_gpu(&[1, 2, 3, 4]), Some([1, 2, 3, 4]));
        assert_eq!(read_pads_gpu(&[]), Some([0, 0, 0, 0]));
        assert_eq!(read_pads_gpu(&[0, 0, 0, 0]), Some([0, 0, 0, 0]));
    }

    #[test]
    fn read_pads_gpu_rejects_negative_entries() {
        assert_eq!(read_pads_gpu(&[-1, 0, 0, 0]), None);
    }

    #[test]
    fn read_group_gpu_accepts_positive_values() {
        assert_eq!(read_group_gpu(1), Some(1));
        assert_eq!(read_group_gpu(4), Some(4));
    }

    #[test]
    fn read_group_gpu_rejects_non_positive_values() {
        assert_eq!(read_group_gpu(0), None);
        assert_eq!(read_group_gpu(-1), None);
    }

    // ── conv_same_pad_split panic regression ────────────────────────────────

    #[test]
    fn conv_same_pad_split_saturates_instead_of_panicking_on_extreme_dilation() {
        // Regression for a real bug caught in review: a malformed
        // `dilations=[-1, 1]` attribute, after the arm's raw `as usize`
        // cast, becomes `usize::MAX`. Before this fix, `conv_same_pad_split`
        // computed `eff_k` as a `saturating_mul` result plus a *bare* `+ 1`
        // — `usize::MAX + 1` panics in debug builds ("attempt to add with
        // overflow") and silently wraps to `0` in release. Every current
        // caller now validates `dilation >= 1` first
        // (`read_positive_pair_gpu`), so this value can no longer reach
        // here in practice — this test is the defense-in-depth backstop,
        // proving the arithmetic itself is safe regardless of caller
        // discipline. The result's value is unimportant (a caller that
        // reaches this with an invalid dilation has already failed to
        // validate); what matters is that it returns instead of panicking.
        let (begin, end) = conv_same_pad_split(7, 3, 1, usize::MAX, false);
        let _ = (begin, end);
    }
}

/// End-to-end regression tests that drive `try_gpu_dispatch` itself (not the
/// extracted pure helpers) through a live `GpuContext`, so they exercise the
/// real compute-shader path — not just the CPU-side shape/attribute gating.
///
/// Every test skips (rather than fails) when no adapter is available, the
/// same convention `zzz_probe` and every test in
/// `oxionnx-gpu/src/shaders/tests.rs` already use, so this suite is a no-op
/// on headless CI and a real regression check wherever Metal/Vulkan/DX12 is
/// present (confirmed available here: `PROBE_GPU_AVAILABLE=true`).
///
/// Every tensor is sized to clear the relevant kernel's own GPU_THRESHOLD
/// (oxionnx-gpu/src/compute.rs, oxionnx-gpu/src/shaders/common.rs) so a
/// `try_gpu_dispatch(...).unwrap()` that got `Ok(None)` back would mean the
/// dispatch was wrongly declined, not "too small to bother" — the `.expect`
/// messages below record which threshold each shape was chosen to clear.
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_tests {
    use super::*;
    use crate::graph::Attributes;

    #[test]
    fn matmul_e2e_preserves_batch_shape_and_computes_correct_values() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // a = [1, M, K] all-ones; b = [K, N] with column `l` holding the
        // constant `l + 1` in every row. output[i, l] = sum_j a[i,j]*b[j,l]
        // = K * (l + 1), independent of `i` — checking it across every row
        // confirms the batch dim was broadcast rather than only the first
        // row being computed from misaligned memory (the exact a7-9
        // failure mode when `b` carries an unexamined batch dimension).
        //
        // M*K*N = 101*100*1000 = 10,100,000, above wgpu's 10M FLOP
        // GPU_THRESHOLD, so this is genuinely claimed by the compute shader.
        const M: usize = 101;
        const K: usize = 100;
        const N: usize = 1000;

        let a_data = vec![1.0_f32; M * K];
        let mut b_data = vec![0.0_f32; K * N];
        for j in 0..K {
            for l in 0..N {
                b_data[j * N + l] = (l + 1) as f32;
            }
        }

        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), Tensor::new(a_data, vec![1, M, K]));
        intermediates.insert("b".to_string(), Tensor::new(b_data, vec![K, N]));

        let node = Node {
            op: OpKind::MatMul,
            name: "matmul0".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("10.1M FLOPs is above GPU_THRESHOLD (10M); must be claimed");
        assert_eq!(outputs.len(), 1);
        let y = &outputs[0];

        // [a4-11/a7-1] The batch prefix must survive — this used to come
        // back as the bare 2-D [M, N] regardless of `a`'s rank.
        assert_eq!(y.shape, vec![1, M, N]);
        assert_eq!(y.data.len(), M * N);
        for l in 0..N {
            let expected = K as f32 * (l as f32 + 1.0);
            for i in 0..M {
                let got = y.data[i * N + l];
                assert!(
                    (got - expected).abs() < 1e-2,
                    "output[{i},{l}] = {got}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn conv_e2e_same_upper_pads_correctly_when_pads_attribute_is_absent() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [conv-pool report] All-ones input/weight, no bias, stride 1, no
        // dilation: with SAME_UPPER padding, an *interior* output pixel sees
        // a full c_in(16) * 3 * 3 = 144 window of ones, while the top-left
        // *corner* pixel sees only the 2x2 in-bounds portion of its window
        // (the other row/col falls in the zero pad) = 16 * 2 * 2 = 64.
        //
        // Before this fix the arm read only the (absent, so all-zero)
        // `pads` attribute and never looked at `auto_pad` — indistinguishable
        // from VALID padding, which has *no* boundary effect at all (every
        // pixel, corner included, would see the full 144). This test fails
        // against the pre-fix code on both the output shape (66x66 vs
        // VALID's 64x64) and the corner value (64 vs a wrongly-uniform 144).
        const C: usize = 16;
        const HW: usize = 66;

        let input = Tensor::new(vec![1.0_f32; C * HW * HW], vec![1, C, HW, HW]);
        let weight = Tensor::new(vec![1.0_f32; C * C * 3 * 3], vec![C, C, 3, 3]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);
        intermediates.insert("w".to_string(), weight);

        let mut attrs = Attributes::default();
        attrs
            .strings
            .insert("auto_pad".to_string(), "SAME_UPPER".to_string());
        // `pads` deliberately left unset: a real SAME_UPPER-exported model
        // never emits it, and silently reading it as all-zero is the bug.

        let node = Node {
            op: OpKind::Conv,
            name: "conv0".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("16*144*4356=10,036,224 FLOPs is above GPU_THRESHOLD (10M); must be claimed");
        let y = &outputs[0];

        // SAME_UPPER must preserve the spatial extent (ceil(66/1) = 66),
        // not VALID's shrunk (66-3)/1+1 = 64.
        assert_eq!(y.shape, vec![1, C, HW, HW]);

        let at = |co: usize, row: usize, col: usize| y.data[co * HW * HW + row * HW + col];
        // Interior pixel: full 3x3x16 window of ones.
        assert!(
            (at(0, 33, 33) - 144.0).abs() < 1e-2,
            "interior = {}",
            at(0, 33, 33)
        );
        // Top-left corner: only a 2x2x16 window is in-bounds — this is the
        // value that proves zero-padding was actually applied.
        assert!(
            (at(0, 0, 0) - 64.0).abs() < 1e-2,
            "corner = {}",
            at(0, 0, 0)
        );
        // A different output channel must agree (the weight is uniform, so
        // every channel sees the same sum) — catches a channel-stride mixup.
        assert!(
            (at(7, 33, 33) - 144.0).abs() < 1e-2,
            "channel 7 interior = {}",
            at(7, 33, 33)
        );
    }

    #[test]
    fn conv_e2e_declines_malformed_negative_dilation_instead_of_panicking() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // Regression for a real bug caught in review: `dilations=[-1, 1]`
        // is invalid per the ONNX spec (the CPU kernel's `read_positive_pair`
        // rejects it with a typed error), but the arm's raw `as usize` cast
        // used to turn `-1_i64` into `usize::MAX` and feed it straight to
        // `resolve_conv_pads_for_gpu` → `conv_same_pad_split`, whose `eff_k
        // = kernel.saturating_sub(1).saturating_mul(dilation) + 1` overflowed
        // on the bare `+ 1` — a debug-build panic (`test result: ok` below
        // *is* the regression check: pre-fix, this test never reached the
        // `assert!` at all).
        //
        // The shape is deliberately the same C=16/HW=66 fixture as
        // `conv_e2e_same_upper_pads_correctly_when_pads_attribute_is_absent`
        // — large enough to clear `gpu_conv2d`'s own GPU_THRESHOLD — so
        // `outputs.is_none()` below is *also* discriminating on its own
        // merits: with a fixture too small for GPU_THRESHOLD, declining
        // would be a foregone conclusion regardless of whether the
        // dilation was ever validated, proving nothing about this fix
        // specifically. At this size, a hypothetical partial fix that
        // merely made `conv_same_pad_split` saturate (without the
        // `read_positive_pair_gpu` validation gate added alongside it)
        // would stop panicking but would still compute nonsense pads from
        // the saturated `dilation` and *dispatch* — `outputs.is_none()`
        // would then correctly fail, catching that gap too.
        const C: usize = 16;
        const HW: usize = 66;
        let input = Tensor::new(vec![1.0_f32; C * HW * HW], vec![1, C, HW, HW]);
        let weight = Tensor::new(vec![1.0_f32; C * C * 3 * 3], vec![C, C, 3, 3]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);
        intermediates.insert("w".to_string(), weight);

        let mut attrs = Attributes::default();
        attrs
            .strings
            .insert("auto_pad".to_string(), "SAME_UPPER".to_string());
        attrs.int_lists.insert("dilations".to_string(), vec![-1, 1]);

        let node = Node {
            op: OpKind::Conv,
            name: "conv_bad_dilation".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        // Must not panic (see above), and must decline — the malformed
        // attribute means the CPU kernel is the one that should report the
        // typed error, not the GPU arm computing on saturated garbage.
        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error");
        assert!(
            outputs.is_none(),
            "a malformed negative dilation must decline to CPU, not dispatch",
        );
    }

    #[test]
    fn reduce_sum_e2e_normalizes_negative_axis_and_honours_keepdims_false() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [a4-17/a7-7] The exact reported example: ReduceSum(axes=[-1],
        // keepdims=0) on a [100000, 3] tensor. out_count = 100000 >=
        // REDUCE_GPU_THRESHOLD (50_000), so this is claimed by the GPU arm.
        // Before the fix, `axes[0] as usize` on `-1_i64` wrapped to
        // `usize::MAX` instead of normalizing to `1`, and `out_shape[axis] =
        // 1` never consulted `keepdims`.
        const ROWS: usize = 100_000;
        let data: Vec<f32> = (0..ROWS).flat_map(|_| [1.0_f32, 2.0, 3.0]).collect();
        let input = Tensor::new(data, vec![ROWS, 3]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);

        let mut attrs = Attributes::default();
        attrs.int_lists.insert("axes".to_string(), vec![-1]);
        attrs.ints.insert("keepdims".to_string(), 0);

        let node = Node {
            op: OpKind::ReduceSum,
            name: "reduce0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("out_count 100000 is above REDUCE_GPU_THRESHOLD (50000); must be claimed");
        let y = &outputs[0];

        // keepdims=0 must drop the axis entirely: [100000], not the
        // pre-fix, always-emitted keepdims=1 shape [100000, 1].
        assert_eq!(y.shape, vec![ROWS]);
        assert_eq!(y.data.len(), ROWS);
        for (i, &v) in y.data.iter().enumerate() {
            assert!((v - 6.0).abs() < 1e-2, "row {i}: {v} != 6.0");
        }
    }

    #[test]
    fn softmax_e2e_axis_last_dim_dispatches_and_computes_uniform_distribution() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [a4-12/a7-0] last_dim = 1024 >= SOFTMAX_DIM_THRESHOLD (1000). This
        // is the positive-path counterpart to the pure `softmax_axis_*`
        // decline tests: axis=-1 on a rank-2 tensor *is* the last dim, so
        // this must still dispatch and compute correctly through the real
        // kernel. All-zero input makes softmax exactly uniform: exp(0)=1 for
        // all 1024 entries, sum=1024.0 (exact in f32), so every output is
        // exactly 1/1024 = 2^-10.
        const ROWS: usize = 2;
        const LAST: usize = 1024;
        let input = Tensor::new(vec![0.0_f32; ROWS * LAST], vec![ROWS, LAST]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);

        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), -1);

        let node = Node {
            op: OpKind::Softmax,
            name: "softmax0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("last_dim 1024 is above SOFTMAX_DIM_THRESHOLD (1000); must be claimed");
        let y = &outputs[0];

        assert_eq!(y.shape, vec![ROWS, LAST]);
        let expected = 1.0_f32 / LAST as f32;
        for &v in &y.data {
            assert!((v - expected).abs() < 1e-6, "{v} != {expected}");
        }
    }

    #[test]
    fn add_e2e_dispatches_when_shapes_match_exactly() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [a4-18] BINARY_EW_GPU_THRESHOLD (100_000) requires a fairly large
        // tensor; this is the positive-path counterpart to
        // `elementwise_shapes_*` — equal shapes above threshold must still
        // dispatch and compute correctly through the real kernel.
        const LEN: usize = 100_000;
        let a = Tensor::new(vec![2.0_f32; LEN], vec![LEN]);
        let b = Tensor::new(vec![3.0_f32; LEN], vec![LEN]);

        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), a);
        intermediates.insert("b".to_string(), b);

        let node = Node {
            op: OpKind::Add,
            name: "add0".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("100000 elements is at BINARY_EW_GPU_THRESHOLD; must be claimed");
        let y = &outputs[0];
        assert_eq!(y.shape, vec![LEN]);
        assert!(y.data.iter().all(|&v| (v - 5.0).abs() < 1e-4));
    }
}
