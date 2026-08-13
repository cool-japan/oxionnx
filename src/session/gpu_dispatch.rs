use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use super::gpu_activations::GpuActivations;
#[cfg(feature = "gpu")]
use oxionnx_gpu::{DeviceTensor, GpuOutput, OutputPlacement, TensorSource};

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
///
/// [r3a] This is now a *fast path* selector rather than a gate: an unequal
/// pair no longer declines the GPU outright, it routes to
/// `gpu_broadcast_*` (oxionnx-gpu/src/shaders/broadcast_binary.rs) instead.
/// The flat kernel is still preferred when it applies because its index math
/// is a bare `output[i] = a[i] op b[i]` against the broadcast kernel's four
/// strided decodes per element.
#[cfg(feature = "gpu")]
fn elementwise_shapes_match(a_shape: &[usize], b_shape: &[usize]) -> bool {
    a_shape == b_shape
}

/// [r3a] Output shape of a broadcasting binary op, or `None` when the
/// operands do not broadcast at all.
///
/// Delegates to the same `Tensor::broadcast_shape` the CPU binary kernels and
/// [`matmul_gpu_plan`] use, so the GPU arm cannot invent a shape the CPU
/// fallback would disagree with.
///
/// # Why this replaced `a.shape.clone()`
///
/// The pre-r3a `Add`/`Mul` arms returned `Tensor::new(result, a.shape.clone())`,
/// which was only correct *because* those arms were gated on shape equality.
/// Removing that gate without also removing the assumption would mis-shape
/// every node whose second operand is the larger one — InSwapper's AdaIN
/// affine pairs are exactly `Mul([1,C,1,1], [1,C,H,W])` in one operand order
/// and the mirror in the other.
#[cfg(feature = "gpu")]
fn broadcast_binary_out_shape(a_shape: &[usize], b_shape: &[usize]) -> Option<Vec<usize>> {
    Tensor::broadcast_shape(a_shape, b_shape).ok()
}

/// [r3a] Map ONNX `Pad`'s `mode` attribute onto the two modes
/// `gpu_pad` (oxionnx-gpu/src/shaders/pad.rs) implements.
///
/// `edge` and `wrap` are CPU-only (`oxionnx-ops::shape::sequence::pad_axes`
/// supports them; no WGSL entry point exists), so they decline. An absent
/// `mode` is `"constant"`, matching `PadOp::execute`.
#[cfg(feature = "gpu")]
fn pad_mode_for_gpu(mode: &str) -> Option<crate::gpu::PadMode> {
    match mode {
        "" | "constant" => Some(crate::gpu::PadMode::Constant),
        "reflect" => Some(crate::gpu::PadMode::Reflect),
        _ => None,
    }
}

/// [r3a] Decide whether a `Pad` node maps onto `gpu_pad`, and if so return
/// `[pad_top, pad_bottom, pad_left, pad_right]`.
///
/// The kernel pads **only** the last two axes of a rank-4 `[N,C,H,W]` tensor.
/// ONNX's `pads` is `[x0_begin, x1_begin, …, x0_end, x1_end, …]`, so for
/// rank 4 the spatial entries are indices 2/3 (begin) and 6/7 (end), and a
/// non-zero pad on `N` or `C` must decline — the kernel would silently ignore
/// it and produce a correctly-shaped, wrongly-sized tensor.
///
/// Verified against the model rather than assumed: all 14 of InSwapper's Pad
/// nodes carry `pads = [0,0,p,p,0,0,p,p]` with `mode="reflect"`.
#[cfg(feature = "gpu")]
fn pad_gpu_plan(input_rank: usize, pads: &[i64]) -> Option<[i64; 4]> {
    if input_rank != 4 || pads.len() != 8 {
        return None;
    }
    if pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0 {
        return None;
    }
    Some([pads[2], pads[6], pads[3], pads[7]])
}

/// [r3a] Which `gpu_resize_*` entry point (if any) implements a `Resize`
/// node's exact interpolation configuration.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResizeGpuKind {
    /// `mode="linear"`, `coordinate_transformation_mode="pytorch_half_pixel"`.
    BilinearPytorchHalfPixel,
    /// `mode="nearest"`, `coordinate_transformation_mode="asymmetric"`,
    /// `nearest_mode="round_prefer_floor"` (the ONNX default).
    NearestAsymmetric,
}

/// [r3a] Match a `Resize` node's attributes against the two kernels
/// oxionnx-gpu implements, declining every other configuration.
///
/// This is deliberately an exact match on all five interpolation attributes
/// rather than a "close enough" one. `Resize` has a large configuration space
/// (`mode` x `coordinate_transformation_mode` x `nearest_mode` x
/// `exclude_outside` x `antialias` x `keep_aspect_ratio_policy`) and every
/// combination the kernels do *not* implement produces a plausible-looking
/// but numerically different image, with the right shape — the failure mode
/// no downstream assertion can catch.
///
/// One consequence, measured not assumed: SCRFD's two `Resize` nodes are
/// `mode="nearest"` with **`nearest_mode="floor"`**, not the ONNX-default
/// `round_prefer_floor` the kernel implements, so they decline to the CPU.
/// They agree for the integer scale factors SCRFD happens to use, but the
/// kernel's contract is the default rule and this gate reflects the contract.
#[cfg(feature = "gpu")]
fn resize_kind_for_gpu(
    mode: &str,
    coord_mode: &str,
    nearest_mode: &str,
    exclude_outside: i64,
    antialias: i64,
    keep_aspect_ratio_policy: &str,
    axes: &[i64],
) -> Option<ResizeGpuKind> {
    if exclude_outside != 0 || antialias != 0 || !axes.is_empty() {
        return None;
    }
    if !matches!(keep_aspect_ratio_policy, "" | "stretch") {
        return None;
    }
    match mode {
        "linear" if coord_mode == "pytorch_half_pixel" => {
            Some(ResizeGpuKind::BilinearPytorchHalfPixel)
        }
        "nearest"
            if coord_mode == "asymmetric" && matches!(nearest_mode, "" | "round_prefer_floor") =>
        {
            Some(ResizeGpuKind::NearestAsymmetric)
        }
        _ => None,
    }
}

/// [r3a] Resolve a `Resize` node's output `(out_h, out_w)` from its `scales`
/// or `sizes` operand, declining anything the spatial-only kernels cannot
/// express.
///
/// Mirrors `resolve_plan` (oxionnx-ops/src/resize.rs) exactly on the arm it
/// accepts: `output_dimension = floor(input_dimension * scale)` with the
/// product evaluated in `f32` (matching onnxruntime), and `scales`/`sizes`
/// mutually exclusive. `N`/`C` must be untouched — scale `1.0`, or a size
/// equal to the input dim — because the kernels pass those axes through.
#[cfg(feature = "gpu")]
fn resize_spatial_extent(
    input_shape: &[usize],
    scales: Option<&[f32]>,
    sizes: Option<&[f32]>,
) -> Option<(usize, usize)> {
    let [n, c, h, w]: [usize; 4] = input_shape.try_into().ok()?;
    match (scales, sizes) {
        // `resolve_plan` errors on both-present; decline so the CPU kernel
        // reports it.
        (Some(_), Some(_)) | (None, None) => None,
        (Some(scales), None) => {
            let [sn, sc, sh, sw]: [f32; 4] = scales.try_into().ok()?;
            if sn != 1.0 || sc != 1.0 {
                return None;
            }
            if !sh.is_finite() || !sw.is_finite() || sh <= 0.0 || sw <= 0.0 {
                return None;
            }
            let out_h = (h as f32 * sh).floor();
            let out_w = (w as f32 * sw).floor();
            if !(0.0..=(usize::MAX as f32)).contains(&out_h)
                || !(0.0..=(usize::MAX as f32)).contains(&out_w)
            {
                return None;
            }
            Some((out_h as usize, out_w as usize))
        }
        (None, Some(sizes)) => {
            let [zn, zc, zh, zw]: [f32; 4] = sizes.try_into().ok()?;
            let to_dim = |v: f32| -> Option<usize> {
                if !v.is_finite() || v < 0.0 || v > usize::MAX as f32 {
                    None
                } else {
                    Some(v as usize)
                }
            };
            if to_dim(zn)? != n || to_dim(zc)? != c {
                return None;
            }
            Some((to_dim(zh)?, to_dim(zw)?))
        }
    }
}

/// [r3a] Decide whether a `Gemm` node maps onto `gpu_gemm_nt`
/// (oxionnx-gpu/src/shaders/gemm.rs) — the only access pattern implemented is
/// `transA=0, transB=1` — and return `(m, k, n)`.
///
/// `alpha`/`beta` are real uniforms in that kernel, so they are *not*
/// constrained here; `transA=1` or `transB=0` decline, since the kernel reads
/// `B` as `[N, K]` row-major (the PyTorch `nn.Linear` weight layout) and
/// would silently mis-index either other layout.
///
/// Verified against the model: all 12 of InSwapper's Gemm nodes and ArcFace's
/// single one are `transB=1, transA` absent, `A=[M,K]`, `B=[N,K]`.
#[cfg(feature = "gpu")]
fn gemm_gpu_plan(
    a_shape: &[usize],
    b_shape: &[usize],
    trans_a: bool,
    trans_b: bool,
) -> Option<(usize, usize, usize)> {
    if trans_a || !trans_b {
        return None;
    }
    let [m, k]: [usize; 2] = a_shape.try_into().ok()?;
    let [n, k_b]: [usize; 2] = b_shape.try_into().ok()?;
    if k != k_b {
        return None;
    }
    Some((m, k, n))
}

/// [r3a] Translate the optimizer's fused Conv `activation` attribute into the
/// kernel-side [`crate::gpu::ConvActivation`] the implicit-GEMM conv fuses
/// into its epilogue.
///
/// Returns `None` for anything [`conv_activation_is_recognized`] rejects, so
/// the two stay in lockstep by construction.
///
/// # This must not be paired with a host-side pass
///
/// `apply_conv_activation` used to run on the read-back result *after*
/// `gpu_conv2d_async`. Now that the activation is fused into the kernel, that
/// call is gone. Doing both would double-apply: `relu`/`clip` are idempotent
/// so the bug would hide, but a future `leaky_relu` mapping would silently
/// square its slope for negative inputs.
#[cfg(feature = "gpu")]
fn conv_activation_for_gpu(
    activation: &str,
    min_val: f32,
    max_val: f32,
) -> Option<crate::gpu::ConvActivation> {
    match activation {
        "" => Some(crate::gpu::ConvActivation::None),
        "relu" => Some(crate::gpu::ConvActivation::Relu),
        "clip" => Some(crate::gpu::ConvActivation::Clip {
            min: min_val,
            max: max_val,
        }),
        _ => None,
    }
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

// [r3a] `apply_conv_activation` — the host-side pass that used to run on the
// read-back Conv result — was **removed**, not merely left unused. The
// activation is now fused into the kernel's epilogue (see
// `conv_activation_for_gpu`), and keeping a second implementation around
// would be an invitation to reintroduce the double-apply bug. Its semantics
// are still pinned by `conv_activation_for_gpu_matches_the_old_host_pass`.

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

/// The residency identity of `name`, or `None` when it does not name a graph
/// initializer.
///
/// This is the *whole* of what the session contributes to weight residency, and
/// it is the part `oxionnx-gpu` must not contain: that crate takes an opaque
/// key and has no concept of an initializer. Two properties make the name a
/// legitimate key, and both are properties of this layer:
///
/// * **Stable across runs.** `weights` is built once when the session is loaded
///   (`session::loading`) and is never mutated afterwards — `Session::run` takes
///   `&self` and no path anywhere reinserts into it. So a name denotes the same
///   bytes for the whole life of the session, which is the life of the
///   `GpuContext` the cache lives in (one context per session,
///   `session::gpu_owner`).
/// * **Unambiguous.** `resolve` prefers a graph intermediate over an
///   initializer of the same name, so a name that a node has produced this run
///   is *not* an initializer here, whatever the weight map holds. Keying such a
///   name would cache one tensor's bytes and then serve them for another's.
#[cfg(feature = "gpu")]
fn initializer_key<'a>(
    name: &'a str,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    activations: &GpuActivations,
) -> Option<&'a str> {
    if name.is_empty() || intermediates.contains_key(name) || activations.holds_node_output(name) {
        return None;
    }
    weights.contains_key(name).then_some(name)
}

/// Whether this dispatcher hands slot `index` of a `node` to its kernel as a
/// *keyed* operand — one the kernel may bind from the residency cache.
///
/// Only two arms key anything: `Conv`'s weight and bias, and `Gemm`'s `B` and
/// `C`. Residency is a property of the (kernel, slot) pair, not of the tensor:
/// an initializer that some convolution has made resident is still uploaded per
/// dispatch when an `Add` or a `PRelu` consumes it, because those arms pass no
/// keys. The census has to agree with that, or a node would be credited with
/// residency its own kernel never uses — and a node credited with *full*
/// residency would be promoted to [`ResidencyTier::Resident`], swapping
/// `MEMORY_BOUND_TRANSFER_FLOOR` for a floor nobody has calibrated.
///
/// [`ResidencyTier::Resident`]: crate::session::gpu_residency::ResidencyTier
/// [`MEMORY_BOUND_TRANSFER_FLOOR`]: crate::session::gpu_residency::MEMORY_BOUND_TRANSFER_FLOOR
#[cfg(feature = "gpu")]
fn keyed_operand_slot(op: &OpKind, index: usize) -> bool {
    matches!(op, OpKind::Conv | OpKind::Gemm) && (index == 1 || index == 2)
}

/// Whether the GPU arm for `op` can bind a device buffer in slot `index`.
///
/// The capability table `RunActivations::new` folds over the graph to decide
/// which values may be produced straight onto the device. It lists (op, slot)
/// pairs rather than ops because the two are genuinely different questions: the
/// `Pad` arm reads its input activation from slot 0 but its `pads` list, its
/// constant value and its `axes` from the *host*, since those become uniform
/// fields and control flow rather than bindings. A value landing in one of
/// those slots has to be on the host whatever the op is.
///
/// Keeping this in step with the match arms below is the one maintenance
/// obligation residency adds: an arm that gains a `TensorSource` operand must
/// gain its slot here, and an arm that loses one must lose it. Getting it wrong
/// is not a correctness bug — an unlisted slot only means the value is read
/// back one node earlier, and a wrongly-listed one means the arm declines and
/// the caller reads it back — but both give up the traffic this wave removes.
#[cfg(feature = "gpu")]
pub(crate) fn op_accepts_resident_slot(op: &OpKind, index: usize) -> bool {
    match op {
        // Slot 0 is the activation; weights and attributes are host-side.
        OpKind::Conv
        | OpKind::Pad
        | OpKind::Resize
        | OpKind::Gemm
        | OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Gelu
        | OpKind::Tanh
        | OpKind::Exp
        | OpKind::Sqrt
        | OpKind::Abs
        | OpKind::Neg
        | OpKind::Log
        | OpKind::SiLU
        | OpKind::LeakyRelu
        | OpKind::OxiInstanceNorm => index == 0,
        // Both operands are real bindings.
        OpKind::Add | OpKind::Mul | OpKind::Sub | OpKind::Div | OpKind::PRelu => index < 2,
        _ => false,
    }
}

/// What a node's operands cost this dispatch: how many there are, how many are
/// already on the device, how many elements still have to cross the bus, and
/// how wide the dispatch itself is.
#[cfg(feature = "gpu")]
struct OperandCensus {
    operands: usize,
    resident: usize,
    transferred_elements: usize,
    /// The largest operand's element count — the size the *resident* tier's
    /// floor is measured against, since its transferred count is always zero.
    /// See `gpu_residency::tier_gate_elements`.
    dispatch_elements: usize,
    /// Operands bound from a run-scoped activation rather than uploaded. Weight
    /// cache hits are counted separately (`GpuRunStats::weight_cache_hits`) and
    /// deliberately excluded here.
    resident_activations: usize,
    /// Their total element count — the upload this dispatch did not perform.
    resident_activation_elements: usize,
}

/// What a dispatch produced.
///
/// The dispatcher used to return `Vec<Tensor>` unconditionally, which was the
/// same statement as "every GPU node ends in a read-back". It no longer does:
/// a node whose result nothing on the host needs hands back the device buffer
/// itself, and the run loop stores it under the output's name until its last
/// consumer.
#[cfg(feature = "gpu")]
pub(crate) enum DispatchOutcome {
    /// Read back into host tensors, positionally aligned with `node.outputs`.
    Host(Vec<Tensor>),
    /// Left in a device buffer. Single-output nodes only — see
    /// `node_output_placement`.
    Device(DeviceTensor),
}

#[cfg(feature = "gpu")]
impl From<GpuOutput> for DispatchOutcome {
    fn from(output: GpuOutput) -> Self {
        match output {
            GpuOutput::Host(tensor) => Self::Host(vec![tensor]),
            GpuOutput::Device(tensor) => Self::Device(tensor),
        }
    }
}

/// Where this node's result should be left.
///
/// [`OutputPlacement::Device`] is a *request*: a kernel that had to take a
/// host-only fallback (the im2col conv path, say) still answers with a host
/// tensor, and the caller stores whichever it gets. Only three things make the
/// request in the first place — the context switch is on, the node has exactly
/// one output, and the graph plan says that output may stay resident.
///
/// The single-output restriction is not a kernel limitation; it is that
/// `DispatchOutcome::Device` carries one buffer, and no op with a resident-
/// capable arm produces more than one result.
#[cfg(feature = "gpu")]
fn node_output_placement(
    node: &Node,
    activations: &GpuActivations,
    gpu: &crate::gpu::GpuContext,
) -> OutputPlacement {
    let single_output = node.outputs.len() == 1;
    let keepable = node
        .outputs
        .first()
        .is_some_and(|name| activations.may_keep(name));
    if single_output && keepable && gpu.activation_residency_enabled() {
        OutputPlacement::Device
    } else {
        OutputPlacement::Host
    }
}

/// GPU dispatch for ops with hardware acceleration (MatMul, Conv).
/// Returns `Ok(Some(results))` if GPU handled it, `Ok(None)` to fall back to CPU.
///
/// Wraps [`dispatch_node_async`], which is the dispatcher proper, to difference
/// the context's cumulative weight-cache counters around it. That difference is
/// what `GpuRunStats` reports per run, and it has to be taken here: the
/// counters on the context are session-cumulative by design (see
/// `oxionnx_gpu::ResidentCounters`).
///
/// # Why this is the async one
///
/// This function *is* the dispatcher; [`try_gpu_dispatch`] is a
/// `pollster::block_on` wrapper around it. Writing it the other way round would
/// mean maintaining two ~450-line copies of the same attribute validation and
/// shape gating — one per target — and the first divergence between them would
/// be a model that computes different values in a browser than on a server.
///
/// On native every `gpu_*_async` call below completes in a single `poll` (its
/// read-back is the same blocking one it always was), so the wrapper is exact:
/// the native engine runs the identical kernels in the identical order. In a
/// browser each `.await` yields to the event loop while the GPU works, which is
/// the whole point — see `oxionnx_gpu`'s crate docs.
///
/// Awaiting one node at a time is a requirement, not an implementation detail:
/// wgpu error scopes are a per-thread LIFO stack. Callers must not run two of
/// these concurrently against the same device.
#[cfg(feature = "gpu")]
pub(crate) async fn try_gpu_dispatch_async(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    activations: &GpuActivations,
    gpu: &crate::gpu::GpuContext,
) -> Result<Option<DispatchOutcome>, OnnxError> {
    // [r3a] Residency-aware size gate. See
    // `gpu_residency::gpu_min_transfer_elements` for the measured table this
    // implements and why several op types decline at *every* size while their
    // operands still cross the bus.
    //
    // Which number the floor is compared against depends on the tier, and that
    // is the whole point: a transferring node is gated on the elements it
    // uploads, a resident one on how wide its dispatch is. Comparing a resident
    // node against its transferred elements would compare every one of them
    // against zero. See `gpu_residency::tier_gate_elements`.
    let census = operand_census(node, weights, intermediates, activations, gpu);
    {
        use crate::session::gpu_residency::{
            gpu_min_transfer_elements, node_residency_tier, tier_gate_elements,
        };
        let tier = node_residency_tier(census.operands, census.resident);
        if let Some(floor) = gpu_min_transfer_elements(&node.op, tier) {
            let measured =
                tier_gate_elements(tier, census.transferred_elements, census.dispatch_elements);
            if measured < floor {
                return Ok(None);
            }
        }
    }

    let placement = node_output_placement(node, activations, gpu);
    let before = gpu.resident_counters();
    let dispatched =
        dispatch_node_async(node, weights, intermediates, activations, gpu, placement).await;
    crate::session::gpu_residency::note_weight_cache(gpu.resident_counters().since(before));
    if let Ok(Some(outcome)) = &dispatched {
        let output_elements = match outcome {
            DispatchOutcome::Device(tensor) => tensor.len(),
            DispatchOutcome::Host(_) => 0,
        };
        crate::session::gpu_residency::note_activation_dispatch(
            census.resident_activations,
            census.resident_activation_elements,
            output_elements,
        );
    }
    dispatched
}

/// Count what this node's operands cost: how many there are, how many are
/// already on the device (as a run-scoped activation or a cached weight), how
/// many elements still cross the bus, and how wide the dispatch is.
///
/// A weight is only counted resident when the arm that will run it actually
/// *keys* that slot ([`keyed_operand_slot`]) — residency is a property of the
/// (kernel, slot) pair, not of the tensor, and a node credited with residency
/// its own kernel never uses would be promoted into a tier whose cost model it
/// does not pay.
#[cfg(feature = "gpu")]
fn operand_census(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    activations: &GpuActivations,
    gpu: &crate::gpu::GpuContext,
) -> OperandCensus {
    let mut census = OperandCensus {
        operands: 0,
        resident: 0,
        transferred_elements: 0,
        dispatch_elements: 0,
        resident_activations: 0,
        resident_activation_elements: 0,
    };
    for (index, name) in node.inputs.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let (len, on_device_activation) = match activations.get(name) {
            Some(tensor) if op_accepts_resident_slot(&node.op, index) => (tensor.len(), true),
            _ => match intermediates.get(name).or_else(|| weights.get(name)) {
                Some(tensor) => (tensor.data.len(), false),
                None => continue,
            },
        };
        census.operands += 1;
        census.dispatch_elements = census.dispatch_elements.max(len);
        let cached_weight = keyed_operand_slot(&node.op, index)
            && initializer_key(name, weights, intermediates, activations)
                .is_some_and(|key| gpu.is_resident(key));
        if on_device_activation {
            census.resident += 1;
            census.resident_activations += 1;
            census.resident_activation_elements =
                census.resident_activation_elements.saturating_add(len);
        } else if cached_weight {
            census.resident += 1;
        } else {
            census.transferred_elements = census.transferred_elements.saturating_add(len);
        }
    }
    census
}

/// The dispatcher proper — see [`try_gpu_dispatch_async`], which is this plus
/// the weight-cache accounting.
#[cfg(feature = "gpu")]
async fn dispatch_node_async(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    activations: &GpuActivations,
    gpu: &crate::gpu::GpuContext,
    placement: OutputPlacement,
) -> Result<Option<DispatchOutcome>, OnnxError> {
    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };
    // The residency-aware operand lookup: a value this run left on the device
    // is bound in place, everything else uploads exactly as before. Arms that
    // read an operand's *contents* on the host (a `pads` list, a `scales`
    // vector, an `axes` tensor) keep using `resolve` — which is why
    // `op_accepts_resident_slot` names slots and not just ops.
    let source = |name: &str| -> Option<TensorSource<'_>> {
        if name.is_empty() {
            return None;
        }
        if let Some(tensor) = activations.get(name) {
            return Some(TensorSource::Device(tensor));
        }
        resolve(name).map(TensorSource::tensor)
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
                    if let Some(result) =
                        crate::gpu::gpu_matmul_async(gpu, &a.data, &b.data, m, k, n).await
                    {
                        return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                            result, out_shape,
                        )])));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Conv => {
            let input = source(&node.inputs[0]);
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
                            input.shape(),
                            &weight.shape,
                            strides,
                            dilations,
                            explicit_pads,
                        ) {
                            // [r3a] The activation is now **fused into the
                            // kernel's epilogue** (C3's implicit-GEMM conv),
                            // not applied host-side after read-back. The
                            // former `apply_conv_activation` call is
                            // deliberately gone: running both would
                            // double-apply it. See `conv_activation_for_gpu`.
                            let min_val = attrs.f("activation_min", f32::NEG_INFINITY);
                            let max_val = attrs.f("activation_max", f32::INFINITY);
                            if let Some(act) = conv_activation_for_gpu(activation, min_val, max_val)
                            {
                                // The weight and the bias are the graph's own
                                // initializers in every convolution network
                                // this engine runs, so they upload once per
                                // session instead of once per frame. The input
                                // activation is deliberately not keyed: it is
                                // different bytes every frame.
                                let keys = crate::gpu::WeightKeys::new(
                                    node.inputs.get(1).and_then(|n| {
                                        initializer_key(n, weights, intermediates, activations)
                                    }),
                                    node.inputs.get(2).and_then(|n| {
                                        initializer_key(n, weights, intermediates, activations)
                                    }),
                                );
                                if let Some(result) = crate::gpu::gpu_conv2d_fused_placed_async(
                                    gpu, input, weight, bias, keys, strides, pads, dilations,
                                    group, act, placement,
                                )
                                .await
                                {
                                    return Ok(Some(result.into()));
                                }
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
                    if let Some(result) =
                        crate::gpu::gpu_softmax_async(gpu, &input.data, &input.shape).await
                    {
                        return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                            result,
                            input.shape.clone(),
                        )])));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Relu => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_relu_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Sigmoid => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) =
                    crate::gpu::gpu_sigmoid_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Gelu => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_gelu_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
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
                        crate::gpu::gpu_reduce_sum_async(gpu, &input.data, axis, &input.shape).await
                    {
                        let keepdims = node.attrs.i("keepdims", 1) != 0;
                        let out_shape = single_axis_reduce_shape(&input.shape, axis, keepdims);
                        return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                            result, out_shape,
                        )])));
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
                        crate::gpu::gpu_reduce_max_async(gpu, &input.data, axis, &input.shape).await
                    {
                        let keepdims = node.attrs.i("keepdims", 1) != 0;
                        let out_shape = single_axis_reduce_shape(&input.shape, axis, keepdims);
                        return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                            result, out_shape,
                        )])));
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
                        crate::gpu::gpu_reduce_min_async(gpu, &input.data, axis, &input.shape).await
                    {
                        let keepdims = node.attrs.i("keepdims", 1) != 0;
                        let out_shape = single_axis_reduce_shape(&input.shape, axis, keepdims);
                        return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                            result, out_shape,
                        )])));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Tanh => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_tanh_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Exp => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_exp_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Sqrt => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_sqrt_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Abs => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_abs_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Neg => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_neg_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::Log => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_log_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::SiLU => {
            if let Some(input) = source(&node.inputs[0]) {
                if let Some(result) = crate::gpu::gpu_silu_placed_async(gpu, input, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        OpKind::LeakyRelu => {
            if let Some(input) = source(&node.inputs[0]) {
                // [a7-8] The node's `alpha` must reach the kernel: the
                // alpha-less entry point hardcodes the ONNX default 0.01, so a
                // YOLOv3-style `alpha = 0.1` model used to get every negative
                // activation scaled 10x too small — with the correct output
                // shape, so nothing downstream could detect it.
                let alpha = node.attrs.f("alpha", 0.01);
                if let Some(result) =
                    crate::gpu::gpu_leaky_relu_placed_async(gpu, input, alpha, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        // [r3a] The four broadcasting binary ops share one body: the flat
        // equal-shape kernel when it applies, otherwise R3b's rank-4
        // broadcast kernel. Before r3a, `Add`/`Mul` *declined* every unequal
        // pair and `Sub`/`Div` had no arm at all — 49 of InSwapper's
        // Add/Mul nodes (its AdaIN affine pairs, `[1,C,1,1]` against
        // `[1,C,H,W]`) went to the CPU for that reason alone.
        OpKind::Add | OpKind::Mul | OpKind::Sub | OpKind::Div => {
            let a = source(&node.inputs[0]);
            let b = source(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // The output shape must come from the broadcast of both
                // operands, never from `a` alone — see
                // `broadcast_binary_out_shape`.
                if let Some(out_shape) = broadcast_binary_out_shape(a.shape(), b.shape()) {
                    // [a4-18] `gpu_add`/`gpu_mul` implement no broadcasting;
                    // they stay the preferred path when the shapes are
                    // already equal because their indexing is one load per
                    // operand instead of four strided decodes.
                    if elementwise_shapes_match(a.shape(), b.shape()) {
                        let flat = match &node.op {
                            OpKind::Add => {
                                crate::gpu::gpu_add_placed_async(gpu, a, b, placement).await
                            }
                            OpKind::Mul => {
                                crate::gpu::gpu_mul_placed_async(gpu, a, b, placement).await
                            }
                            // No flat `Sub`/`Div` kernel exists; fall through
                            // to the broadcast one, which handles the
                            // equal-shape case correctly (its stride
                            // resolution simply never zeroes a stride).
                            _ => None,
                        };
                        if let Some(result) = flat {
                            return Ok(Some(result.into()));
                        }
                    }
                    let kind = match &node.op {
                        OpKind::Add => crate::gpu::BroadcastKind::Add,
                        OpKind::Mul => crate::gpu::BroadcastKind::Mul,
                        OpKind::Sub => crate::gpu::BroadcastKind::Sub,
                        _ => crate::gpu::BroadcastKind::Div,
                    };
                    if let Some(result) = crate::gpu::gpu_broadcast_placed_async(
                        gpu, a, b, &out_shape, kind, placement,
                    )
                    .await
                    {
                        return Ok(Some(result.into()));
                    }
                }
            }
            Ok(None)
        }
        // [r3a] PRelu — ArcFace's 25 nodes, slope `[C,1,1]`, input `[N,C,H,W]`.
        // The kernel derives `channels` from `shape[1]` and accepts a slope of
        // length `C` or `1`, declining anything else itself.
        OpKind::PRelu => {
            let input = source(&node.inputs[0]);
            let slope = node.inputs.get(1).and_then(|n| source(n));
            if let (Some(input), Some(slope)) = (input, slope) {
                if let Some(result) =
                    crate::gpu::gpu_prelu_placed_async(gpu, input, slope, placement).await
                {
                    return Ok(Some(result.into()));
                }
            }
            Ok(None)
        }
        // [r3a] Pad — InSwapper's 14 reflect-pad nodes ahead of each conv.
        OpKind::Pad => {
            let input = source(&node.inputs[0]);
            let pads_tensor = node.inputs.get(1).and_then(|n| resolve(n));
            // `axes` (opset-18 input 3) restricts which axes `pads` refers to;
            // the kernel has no way to express that, so its presence declines.
            let has_axes = node
                .inputs
                .get(3)
                .and_then(|n| resolve(n))
                .is_some_and(|t| !t.data.is_empty());
            if let (Some(input), Some(pads_tensor), false) = (input, pads_tensor, has_axes) {
                let pads: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
                if let (Some(mode), Some([top, bottom, left, right])) = (
                    pad_mode_for_gpu(node.attrs.s("mode")),
                    pad_gpu_plan(input.shape().len(), &pads),
                ) {
                    // `constant_value` is input 2 (a scalar tensor); absent
                    // means 0.0, matching `read_optional_pad_inputs`.
                    let constant_value = node
                        .inputs
                        .get(2)
                        .and_then(|n| resolve(n))
                        .and_then(|t| t.data.first().copied())
                        .unwrap_or(0.0);
                    if let Some(result) = crate::gpu::gpu_pad_placed_async(
                        gpu,
                        input,
                        top,
                        bottom,
                        left,
                        right,
                        mode,
                        constant_value,
                        placement,
                    )
                    .await
                    {
                        return Ok(Some(result.into()));
                    }
                }
            }
            Ok(None)
        }
        // [r3a] Resize — InSwapper's two 2x bilinear upsamples.
        OpKind::Resize => {
            let input = source(&node.inputs[0]);
            if let Some(input) = input {
                let attrs = &node.attrs;
                let kind = resize_kind_for_gpu(
                    attrs.s("mode"),
                    attrs.s("coordinate_transformation_mode"),
                    attrs.s("nearest_mode"),
                    attrs.i("exclude_outside", 0),
                    attrs.i("antialias", 0),
                    attrs.s("keep_aspect_ratio_policy"),
                    attrs.ints("axes"),
                );
                // Opset-11+ positional layout only: `(X, roi, scales, sizes)`.
                // The opset-10 two-input `(X, scales)` form is ambiguous with
                // `roi` (see `read_tensor_inputs`, oxionnx-ops), so a node
                // with fewer than three inputs declines rather than guessing.
                let non_empty = |idx: usize| -> Option<&Tensor> {
                    node.inputs
                        .get(idx)
                        .and_then(|n| resolve(n))
                        .filter(|t| !t.data.is_empty())
                };
                let roi_present = non_empty(1).is_some();
                if let (Some(kind), true, false) = (kind, node.inputs.len() >= 3, roi_present) {
                    let scales = non_empty(2).map(|t| t.data.as_slice());
                    let sizes = non_empty(3).map(|t| t.data.as_slice());
                    if let Some((out_h, out_w)) =
                        resize_spatial_extent(input.shape(), scales, sizes)
                    {
                        let kernel_kind = match kind {
                            ResizeGpuKind::BilinearPytorchHalfPixel => {
                                crate::gpu::ResizeKind::BilinearPytorchHalfPixel
                            }
                            ResizeGpuKind::NearestAsymmetric => {
                                crate::gpu::ResizeKind::NearestAsymmetric
                            }
                        };
                        if let Some(result) = crate::gpu::gpu_resize_placed_async(
                            gpu,
                            input,
                            out_h,
                            out_w,
                            kernel_kind,
                            placement,
                        )
                        .await
                        {
                            return Ok(Some(result.into()));
                        }
                    }
                }
            }
            Ok(None)
        }
        // [r3a] Gemm — InSwapper's 12 AdaIN style heads and ArcFace's
        // embedding head, all `alpha=1, beta=1, transA=0, transB=1`.
        OpKind::Gemm => {
            let a = source(&node.inputs[0]);
            let b = node.inputs.get(1).and_then(|n| resolve(n));
            let c = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(a), Some(b)) = (a, b) {
                let attrs = &node.attrs;
                let trans_a = attrs.i("transA", 0) != 0;
                let trans_b = attrs.i("transB", 0) != 0;
                if let Some((m, k, n)) = gemm_gpu_plan(a.shape(), &b.shape, trans_a, trans_b) {
                    // `B` and `C` are initializers in every Gemm this engine
                    // dispatches — ArcFace's embedding head alone is a 51.4 MB
                    // `B`. `A` is the activation and is never keyed.
                    let weight_key = node
                        .inputs
                        .get(1)
                        .and_then(|n| initializer_key(n, weights, intermediates, activations));
                    let keys = crate::gpu::WeightKeys::new(
                        weight_key,
                        node.inputs
                            .get(2)
                            .and_then(|n| initializer_key(n, weights, intermediates, activations)),
                    );
                    // [r3a] `gpu_gemm_nt` carries no size gate of its own —
                    // `kernel_support`'s convention puts the placement
                    // heuristic at the session call site, and this is that
                    // site. Without it, InSwapper's 12 2.1-MFLOP AdaIN heads
                    // all dispatched and ran 3.07x slower than the CPU kernel.
                    //
                    // The gate is device- *and* shape-aware, and it needs
                    // `weight_key` to be decided first: a `Gemm` whose `B` is a
                    // graph initializer binds it from the residency cache on
                    // every frame after the first, which is a different cost
                    // model from one that re-uploads `k*n` floats per call. See
                    // `gpu_residency::gemm_gpu_admits`.
                    if !crate::session::gpu_residency::gemm_gpu_admits(
                        gpu,
                        m,
                        k,
                        n,
                        weight_key.is_some(),
                    ) {
                        return Ok(None);
                    }
                    let alpha = attrs.f("alpha", 1.0);
                    let beta = attrs.f("beta", 1.0);
                    if let Some(result) = crate::gpu::gpu_gemm_nt_placed_async(
                        gpu,
                        a,
                        m,
                        k,
                        &b.data,
                        n,
                        c.map(|t| t.data.as_slice()),
                        alpha,
                        beta,
                        keys,
                        placement,
                    )
                    .await
                    {
                        return Ok(Some(result.into()));
                    }
                }
            }
            Ok(None)
        }
        // [r3a] OxiInstanceNorm — the 12 AdaIN normalisations F3's fusion
        // pass creates. One input, one `epsilon` attribute, shape-preserving.
        OpKind::OxiInstanceNorm => {
            if let Some(input) = source(&node.inputs[0]) {
                let eps = node.attrs.f("epsilon", 1e-5);
                if let Some(result) =
                    crate::gpu::gpu_instance_norm_placed_async(gpu, input, eps, placement).await
                {
                    return Ok(Some(result.into()));
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
                if let Some(result) = crate::gpu::gpu_layer_norm_axis_async(
                    gpu,
                    &input.data,
                    &input.shape,
                    &scale.data,
                    &bias.data,
                    eps,
                    axis,
                )
                .await
                {
                    return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                        result,
                        input.shape.clone(),
                    )])));
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
                if let Some(result) = crate::gpu::gpu_batch_norm_async(
                    gpu,
                    &input.data,
                    &input.shape,
                    &scale.data,
                    &bias.data,
                    &mean.data,
                    &var.data,
                    eps,
                )
                .await
                {
                    return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                        result,
                        input.shape.clone(),
                    )])));
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
                    crate::gpu::gpu_transpose_async(gpu, &input.data, &input.shape, &perm).await
                {
                    let out_shape: Vec<usize> = perm.iter().map(|&p| input.shape[p]).collect();
                    return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                        result, out_shape,
                    )])));
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
                        if let Some(result) = crate::gpu::gpu_reduce_mean_async(
                            gpu,
                            &input.data,
                            &input.shape,
                            &axes,
                            keepdims,
                        )
                        .await
                        {
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
                            return Ok(Some(DispatchOutcome::Host(vec![Tensor::new(
                                result, out_shape,
                            )])));
                        }
                    }
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Blocking form of [`try_gpu_dispatch_async`], used by the synchronous run
/// loops.
///
/// # wasm32
///
/// Declines every node. The browser cannot block on a GPU fence, so the
/// synchronous run loop has nothing it could usefully do with a WebGPU device;
/// declining here sends the node to the CPU operator without allocating a
/// buffer or encoding a pass, which is exactly the "GPU as pure overhead"
/// failure this dispatcher used to have on that target. Browser callers use
/// [`crate::Session::run_gpu_async`], which drives
/// [`try_gpu_dispatch_async`] directly.
#[cfg(feature = "gpu")]
pub(crate) fn try_gpu_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    gpu: &crate::gpu::GpuContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // The synchronous loops have no run-scoped activation map — they hold
        // host tensors between nodes and always have. An empty plan makes every
        // placement decision come out `Host`, so this path is byte-for-byte the
        // dispatcher it was before residency existed.
        let activations = GpuActivations::default();
        let outcome = pollster::block_on(try_gpu_dispatch_async(
            node,
            weights,
            intermediates,
            &activations,
            gpu,
        ))?;
        Ok(match outcome {
            Some(DispatchOutcome::Host(tensors)) => Some(tensors),
            // Unreachable: an empty plan never requests `OutputPlacement::Device`.
            Some(DispatchOutcome::Device(_)) | None => None,
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (node, weights, intermediates, gpu);
        Ok(None)
    }
}

// The unit/e2e test modules for this file live in `gpu_dispatch_tests.rs`
// (split out to keep this source file under the 2000-line policy limit).
// They are child modules of `gpu_dispatch`, so they retain access to the
// private gating helpers they exercise.
#[cfg(all(test, feature = "gpu"))]
#[path = "gpu_dispatch_tests.rs"]
mod gpu_dispatch_tests;
