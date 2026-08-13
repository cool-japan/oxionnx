//! CUDA-accelerated ReduceSum / ReduceMax dispatch.
//!
//! Delegates to [`oxicuda_blas::reduction::reduce_axis`], which views the
//! tensor as `[outer, axis_len, inner]` around the reduced axis and launches
//! one thread block per `(outer_idx, inner_idx)` output pair; each block
//! accumulates `axis_len` elements with a strided per-thread loop
//! (`k = tid, tid + block_size, tid + 2*block_size, ...`) followed by a
//! shared-memory tree reduction. That loop is what makes this correct for
//! *any* `axis_len` — not just axis lengths up to the block size — and for
//! any `outer`/`inner` combination, not just the whole-tensor-reduction
//! case the previous hand-rolled integration
//! (`oxicuda_ptx::templates::reduction::ReductionTemplate`, a single-block
//! kernel with exactly one element read per thread and no accumulation
//! loop) was limited to.
//!
//! [`cuda_reduce_mean_bound`] extends the same machinery to `ReduceMean`:
//! [`ReductionOp::Sum`] over a *range* of one or more contiguous axes (see
//! [`resolve_contiguous_axes`] — the ONNX `axes=[2,3]` shape InSwapper's
//! un-fused `InstanceNorm` decomposition emits, exactly as much generality as
//! that real pattern needs and no more), followed by an in-place device-side
//! divide by `axis_len` so the whole op — sum, then scale — never leaves the
//! GPU. The scale step reuses `oxicuda_ptx::templates::elementwise`'s
//! existing `Scale` kernel (`b[i] = alpha * a[i]`, `alpha` a genuine runtime
//! parameter, not baked in like `LeakyRelu`'s) rather than adding a new one.

use oxicuda_blas::reduction::{reduce_axis, ReductionOp};
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_ptx::{
    ir::PtxType,
    templates::elementwise::{ElementwiseOp, ElementwiseTemplate},
};

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency-cache slot label for the reduction operand. A reduction input is
/// always an activation, so this only ever tags a transient pooled upload.
const INPUT_LABEL: &str = "reduce_in";

/// Threads per block for the post-sum scale kernel `cuda_reduce_mean_bound`
/// launches.
const SCALE_BLOCK_SIZE: u32 = 256;

/// Decompose `shape` around a single `axis` into `(outer, axis_len, inner)`.
/// A thin wrapper over [`reduce_plan_range`] with `start == end == axis`; see
/// its docs for the decline rules.
fn reduce_plan(shape: &[usize], axis: usize) -> Option<(usize, usize, usize)> {
    reduce_plan_range(shape, axis, axis)
}

/// Decompose `shape` around the contiguous axis range `[start, end]`
/// (inclusive both ends) into `(outer, axis_len, inner)`, and decide whether
/// this is a configuration the CUDA reduction machinery will attempt (as
/// opposed to declining to the CPU).
///
/// `axis_len` is the *product* of every dimension in `start..=end` — the
/// generalisation [`cuda_reduce_mean_bound`] needs to reduce ONNX's
/// `axes=[2,3]` (InstanceNorm's un-fused spatial mean) in one launch, exactly
/// as if the two trailing dimensions had been flattened into one before the
/// call. Row-major contiguity is what makes that legal: `outer` covers every
/// dimension strictly before `start`, `inner` every dimension strictly after
/// `end`, and nothing in between is skipped.
///
/// Pure and allocation-free, so the axis/shape bookkeeping is unit-testable
/// without a CUDA device — unlike the GPU launch itself, which cannot be
/// exercised on a host with no CUDA device.
///
/// Declines (`None`) when:
/// - `end` is out of range for `shape`, or `start > end` (a malformed model
///   or a caller bug in the range resolved by [`resolve_contiguous_axes`]).
/// - The reduction would touch zero elements (`outer`, `axis_len`, or
///   `inner` is `0`): a degenerate edge case left to the CPU kernel's
///   identity-element handling rather than special-cased here.
fn reduce_plan_range(shape: &[usize], start: usize, end: usize) -> Option<(usize, usize, usize)> {
    if end >= shape.len() || start > end {
        return None;
    }
    let outer: usize = shape[..start].iter().product();
    let axis_len: usize = shape[start..=end].iter().product();
    let inner: usize = shape[end + 1..].iter().product();
    if outer == 0 || axis_len == 0 || inner == 0 {
        return None;
    }
    Some((outer, axis_len, inner))
}

/// Resolve an ONNX `axes` attribute list (as `ReduceMean`'s node carries it —
/// see `lib.rs`'s `OpKind::ReduceMean` arm) against `rank`, and check that it
/// names a single *contiguous* run of axes once negative indices are
/// resolved and duplicates removed.
///
/// Returns `(start, end)` inclusive, suitable for [`reduce_plan_range`].
/// `None` — decline to the CPU kernel, which has no such restriction —
/// covers every case this narrow CUDA path does not model: an empty `axes`
/// list (ONNX's "reduce every axis", or `noop_with_empty_axes`, neither of
/// which this function can express as one contiguous range in general), an
/// out-of-range or duplicate-free-but-non-contiguous axis set (e.g. `[0,
/// 2]` on a rank-4 tensor, skipping axis `1`), or an axis outside `[-rank,
/// rank)`.
///
/// Pure and allocation-free, so unit-testable without a CUDA device.
#[must_use]
pub(crate) fn resolve_contiguous_axes(rank: usize, raw_axes: &[i64]) -> Option<(usize, usize)> {
    if raw_axes.is_empty() {
        return None;
    }
    let mut resolved: Vec<usize> = Vec::with_capacity(raw_axes.len());
    for &raw in raw_axes {
        let r = if raw < 0 { raw + rank as i64 } else { raw };
        if r < 0 || r as usize >= rank {
            return None;
        }
        resolved.push(r as usize);
    }
    resolved.sort_unstable();
    resolved.dedup();
    for pair in resolved.windows(2) {
        if pair[1] != pair[0] + 1 {
            return None;
        }
    }
    let start = *resolved.first()?;
    let end = *resolved.last()?;
    Some((start, end))
}

/// GPU reduction for a single axis.
///
/// `shape` is decomposed as `[outer, axis_len, inner]` around `axis` (see
/// `reduce_plan` in this module). Returns `Ok(None)` — deferring to the
/// CPU — when the plan declines, or when a dimension doesn't fit the
/// kernel's `u32` launch parameters (the CPU path has no such limit).
pub fn cuda_reduce(
    ctx: &CudaContext,
    data: &[f32],
    shape: &[usize],
    axis: usize,
    op_name: &str,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    match cuda_reduce_bound(
        ctx,
        InputBinding::Host(data),
        shape,
        axis,
        op_name,
        &[],
        CudaOutputPlacement::Host,
    )? {
        Some(KernelOutput::Host(out)) => Ok(Some(out)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "reduce",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

/// [`cuda_reduce`] over an operand that may already be on the device, leaving
/// the result there when the caller asks for it.
///
/// `out_shape` is the ONNX output shape (which `keepdims` decides, and this
/// module deliberately does not); it is only consulted on the device path,
/// where it becomes the resident tensor's shape.
///
/// # Errors
///
/// As [`cuda_reduce`].
pub(crate) fn cuda_reduce_bound(
    ctx: &CudaContext,
    input: InputBinding<'_>,
    shape: &[usize],
    axis: usize,
    op_name: &str,
    out_shape: &[usize],
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    let Some((outer, axis_len, inner)) = reduce_plan(shape, axis) else {
        return Ok(None);
    };

    let reduce_op = match op_name {
        "ReduceSum" => ReductionOp::Sum,
        "ReduceMax" => ReductionOp::Max,
        other => {
            return Err(CudaDispatchError::Unsupported {
                op: "reduce",
                reason: format!("no CUDA reduction kernel for ONNX op '{other}'"),
            });
        }
    };

    let (Ok(outer_u32), Ok(axis_len_u32), Ok(inner_u32)) = (
        u32::try_from(outer),
        u32::try_from(axis_len),
        u32::try_from(inner),
    ) else {
        // A dimension too large for a u32-indexed CUDA kernel launch.
        return Ok(None);
    };
    let Some(output_len) = outer.checked_mul(inner) else {
        return Ok(None);
    };

    // `reduce_axis` launches on `handle.stream()` where `handle` is
    // `ctx.dnn.blas()`. Since `DnnHandle::build` collapsed the BLAS sub-handle
    // onto the DNN handle's own stream, that *is* `ctx.dnn.stream()` — one
    // queue for both op families, which is what lets a `Conv` output feed this
    // reduction with neither an event nor a host fence between them. Issuing
    // the upload, the zero-fill and the readback on the same stream as the
    // kernel is what orders them against it; the predecessor of this code had
    // to `synchronize_all()` mid-dispatch precisely because its copies rode a
    // different stream from its kernel.
    let stream = ctx.dnn.blas().stream();

    let Some(mut d_input) = input.bind(ctx, INPUT_LABEL, input.len(), stream)? else {
        return Ok(None);
    };
    // Zero-filled, exactly as the `DeviceBuffer::zeroed` this replaces was —
    // the output is a recycled allocation now, so whatever the previous
    // borrower left in it must not be visible to the reduction kernel or to
    // the readback. Stream-ordered, so it costs a queued memset rather than
    // `zeroed`'s context-wide fence.
    let mut d_output = ctx.scratch(output_len)?;
    d_output.zero_fill(stream)?;

    reduce_axis(
        ctx.dnn.blas(),
        reduce_op,
        outer_u32,
        axis_len_u32,
        inner_u32,
        d_input.buffer(),
        d_output.buffer_mut(),
    )
    .map_err(|e| CudaDispatchError::Blas(e.to_string()))?;

    // The one fence in the dispatch on the host path — everything above was
    // enqueued on `stream`, in order — and none at all on the device path.
    let out = finish_output(ctx, d_output, output_len, out_shape, placement, stream)?;
    // ...and only now may the input borrow go back to the pool. See
    // `PooledBuffer`'s "a borrow is only recycled once its stream work is
    // known to be done".
    match &out {
        KernelOutput::Host(_) => d_input.retire(),
        KernelOutput::Device(_) => retire_queued(ctx, &mut d_input),
    }
    Ok(Some(out))
}

/// Fetch — compiling on first use — the `Scale` kernel (`b[i] = alpha *
/// a[i]`) [`cuda_reduce_mean_bound`] uses for its post-sum divide.
///
/// `alpha` is a genuine runtime `.param` in this kernel (unlike
/// `LeakyRelu`'s baked-in constant slope), so one compiled module serves
/// every `axis_len` a `ReduceMean` node could ever present — no per-shape
/// PTX regeneration, and no cache-key subtlety like
/// `crate::norm`'s `epsilon` (which *is* baked in).
fn scale_kernel(ctx: &CudaContext) -> Result<Kernel, CudaDispatchError> {
    let template =
        ElementwiseTemplate::new(ElementwiseOp::Scale, PtxType::F32, ctx.dnn.sm_version());
    let kernel_name = template.kernel_name();
    let module = ctx.module(&kernel_name, || {
        template
            .generate()
            .map_err(|e| CudaDispatchError::Ptx(e.to_string()))
    })?;
    Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)
}

/// Multiply every element at `device_ptr` by `alpha`, in place, queued on
/// `ctx.dnn.blas().stream()` (the same stream [`cuda_reduce_mean_bound`]'s
/// `reduce_axis` call ran on) so stream order alone sequences the scale
/// behind the sum — no fence between them.
///
/// In-place aliasing (`a_ptr == b_ptr`) is safe: `Scale`'s kernel gives
/// thread `i` exactly one read of `a[i]` and one write of `b[i]`, the same
/// discipline `elementwise::launch_unary_in_place`'s doc comment establishes
/// for the ops it covers.
fn launch_scale_in_place(
    ctx: &CudaContext,
    device_ptr: CUdeviceptr,
    alpha: f32,
    n: usize,
) -> Result<(), CudaDispatchError> {
    if n == 0 {
        return Ok(());
    }
    let Ok(n_u32) = u32::try_from(n) else {
        return Err(CudaDispatchError::Shape {
            op: "reduce_mean_scale",
            msg: format!("{n} elements exceed a u32 kernel launch"),
        });
    };
    let kernel = scale_kernel(ctx)?;
    let grid = grid_size_for(n_u32, SCALE_BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(SCALE_BLOCK_SIZE));
    let args = (device_ptr, device_ptr, alpha, n_u32);
    kernel
        .launch(&params, ctx.dnn.blas().stream(), &args)
        .map_err(CudaDispatchError::Driver)
}

/// GPU `ReduceMean` over one or more contiguous axes: [`ReductionOp::Sum`]
/// via the same [`reduce_axis`] machinery [`cuda_reduce_bound`] uses,
/// immediately followed by an in-place device-side divide by `axis_len` (see
/// [`launch_scale_in_place`]) — so a `ReduceMean` whose result stays on the
/// device never round-trips through the host to apply the `1/axis_len`
/// scale.
///
/// `shape` is decomposed as `[outer, axis_len, inner]` around the *inclusive*
/// axis range `[start_axis, end_axis]` (see [`reduce_plan_range`] and
/// [`resolve_contiguous_axes`], which `lib.rs`'s `OpKind::ReduceMean` arm
/// uses to turn an ONNX `axes` list into this range). Returns `Ok(None)` —
/// deferring to the CPU — when the plan declines, or when a dimension
/// doesn't fit the kernel's `u32` launch parameters.
///
/// # Errors
///
/// As [`cuda_reduce_bound`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_reduce_mean_bound(
    ctx: &CudaContext,
    input: InputBinding<'_>,
    shape: &[usize],
    start_axis: usize,
    end_axis: usize,
    out_shape: &[usize],
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    let Some((outer, axis_len, inner)) = reduce_plan_range(shape, start_axis, end_axis) else {
        return Ok(None);
    };

    let (Ok(outer_u32), Ok(axis_len_u32), Ok(inner_u32)) = (
        u32::try_from(outer),
        u32::try_from(axis_len),
        u32::try_from(inner),
    ) else {
        return Ok(None);
    };
    let Some(output_len) = outer.checked_mul(inner) else {
        return Ok(None);
    };
    // `axis_len` came from real (checked) shape dims and is >= 1 by
    // `reduce_plan_range`'s own zero-guard, so this division is exact and
    // never by zero.
    #[allow(clippy::cast_precision_loss)]
    let alpha = 1.0_f32 / axis_len as f32;

    let stream = ctx.dnn.blas().stream();

    let Some(mut d_input) = input.bind(ctx, INPUT_LABEL, input.len(), stream)? else {
        return Ok(None);
    };
    let mut d_output = ctx.scratch(output_len)?;
    d_output.zero_fill(stream)?;

    reduce_axis(
        ctx.dnn.blas(),
        ReductionOp::Sum,
        outer_u32,
        axis_len_u32,
        inner_u32,
        d_input.buffer(),
        d_output.buffer_mut(),
    )
    .map_err(|e| CudaDispatchError::Blas(e.to_string()))?;

    // Same stream as the sum above: no fence between "sum" and "divide".
    launch_scale_in_place(ctx, d_output.device_ptr(), alpha, output_len)?;

    let out = finish_output(ctx, d_output, output_len, out_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => d_input.retire(),
        KernelOutput::Device(_) => retire_queued(ctx, &mut d_input),
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── reduce_plan: pure decomposition/decline logic, no CUDA needed ──────

    #[test]
    fn plan_rejects_out_of_range_axis() {
        assert_eq!(reduce_plan(&[2, 3], 5), None);
        assert_eq!(reduce_plan(&[2, 3], 2), None);
    }

    #[test]
    fn plan_decomposes_a_middle_axis() {
        // [2, 3, 4], axis=1 -> outer=2 (dims before), axis_len=3, inner=4 (dims after).
        assert_eq!(reduce_plan(&[2, 3, 4], 1), Some((2, 3, 4)));
    }

    #[test]
    fn plan_decomposes_the_first_axis() {
        // [2, 3, 4], axis=0 -> outer=1 (empty product), axis_len=2, inner=12.
        assert_eq!(reduce_plan(&[2, 3, 4], 0), Some((1, 2, 12)));
    }

    #[test]
    fn plan_decomposes_the_last_axis() {
        // [2, 3, 4], axis=2 -> outer=6, axis_len=4, inner=1 (empty product).
        assert_eq!(reduce_plan(&[2, 3, 4], 2), Some((6, 4, 1)));
    }

    /// The exact motivating case from the a8-1 finding: a 1-D tensor with
    /// more than `REDUCE_BLOCK_SIZE` (256) elements. The previous
    /// hand-rolled single-block-of-256 kernel silently truncated this to
    /// elements `0..255`; the plan itself now has no such ceiling — the
    /// truncation is gone at the decomposition level (the strided
    /// accumulation loop inside `oxicuda_blas::reduction::reduce_axis`,
    /// which cannot be exercised on this host, is what proves the *kernel*
    /// side; this proves the *decision to attempt it at all* is no longer
    /// gated on `axis_len <= 256`).
    #[test]
    fn plan_no_longer_caps_axis_len_at_the_block_size() {
        let axis_len = 1024;
        assert_eq!(reduce_plan(&[axis_len], 0), Some((1, axis_len, 1)));
    }

    /// Before this fix, `cuda_reduce` only ever attempted the
    /// whole-tensor-reduction case (`outer == 1 && inner == 1`). The plan
    /// now accepts general `outer`/`inner`, which is what lets CUDA claim
    /// e.g. a per-channel reduction over an NCHW tensor.
    #[test]
    fn plan_no_longer_requires_a_trivial_outer_and_inner() {
        assert_eq!(reduce_plan(&[5, 7, 9], 1), Some((5, 7, 9)));
    }

    #[test]
    fn plan_declines_a_zero_length_axis() {
        assert_eq!(reduce_plan(&[2, 0, 4], 1), None);
    }

    #[test]
    fn plan_declines_when_another_dimension_is_zero() {
        // outer becomes 0 even though the reduced axis itself is non-empty.
        assert_eq!(reduce_plan(&[0, 3, 4], 1), None);
        // inner becomes 0.
        assert_eq!(reduce_plan(&[2, 3, 0], 1), None);
    }

    #[test]
    fn cuda_context_construction_never_panics_even_though_unavailable_here() {
        // No CUDA device exists on this host; this only asserts try_new()
        // itself does not panic (cuda_reduce cannot be exercised without a
        // live context, which is why the plan/decline logic above is
        // factored out into a pure function instead).
        let _ = CudaContext::try_new();
    }

    // ── reduce_plan_range: the multi-axis generalisation ReduceMean needs ──

    #[test]
    fn range_of_one_axis_matches_reduce_plan() {
        assert_eq!(
            reduce_plan_range(&[2, 3, 4], 1, 1),
            reduce_plan(&[2, 3, 4], 1)
        );
    }

    #[test]
    fn range_merges_two_trailing_axes_into_one_axis_len() {
        // The OxiInstanceNorm decomposition's `ReduceMean(axes=[2,3])` on a
        // [N,C,H,W] tensor: outer=N*C, axis_len=H*W, inner=1.
        assert_eq!(reduce_plan_range(&[2, 3, 4, 5], 2, 3), Some((6, 20, 1)));
    }

    #[test]
    fn range_merges_a_leading_pair_of_axes() {
        assert_eq!(reduce_plan_range(&[2, 3, 4, 5], 0, 1), Some((1, 6, 20)));
    }

    #[test]
    fn range_declines_when_end_is_out_of_bounds() {
        assert_eq!(reduce_plan_range(&[2, 3], 0, 5), None);
    }

    #[test]
    fn range_declines_when_start_exceeds_end() {
        assert_eq!(reduce_plan_range(&[2, 3, 4], 2, 1), None);
    }

    #[test]
    fn range_declines_a_zero_dimension_inside_the_range() {
        assert_eq!(reduce_plan_range(&[2, 0, 4, 5], 1, 2), None);
    }

    // ── resolve_contiguous_axes ──────────────────────────────────────────────

    #[test]
    fn resolves_a_single_positive_axis() {
        assert_eq!(resolve_contiguous_axes(4, &[2]), Some((2, 2)));
    }

    #[test]
    fn resolves_the_trailing_pair_instance_norm_uses() {
        assert_eq!(resolve_contiguous_axes(4, &[2, 3]), Some((2, 3)));
    }

    #[test]
    fn resolves_negative_axes_against_rank() {
        // axes=[-2, -1] on a rank-4 tensor == [2, 3].
        assert_eq!(resolve_contiguous_axes(4, &[-2, -1]), Some((2, 3)));
    }

    #[test]
    fn resolves_out_of_order_axes_by_sorting() {
        assert_eq!(resolve_contiguous_axes(4, &[3, 2]), Some((2, 3)));
    }

    #[test]
    fn deduplicates_a_repeated_axis() {
        assert_eq!(resolve_contiguous_axes(4, &[2, 2, 2]), Some((2, 2)));
    }

    #[test]
    fn declines_an_empty_axes_list() {
        assert_eq!(resolve_contiguous_axes(4, &[]), None);
    }

    #[test]
    fn declines_a_non_contiguous_axis_set() {
        // [0, 2] on a rank-4 tensor skips axis 1 -- not expressible as one
        // [outer, axis_len, inner] range.
        assert_eq!(resolve_contiguous_axes(4, &[0, 2]), None);
    }

    #[test]
    fn declines_an_axis_out_of_range() {
        assert_eq!(resolve_contiguous_axes(4, &[4]), None);
        assert_eq!(resolve_contiguous_axes(4, &[-5]), None);
    }

    #[test]
    fn resolves_every_axis_of_a_rank_one_tensor() {
        assert_eq!(resolve_contiguous_axes(1, &[0]), Some((0, 0)));
        assert_eq!(resolve_contiguous_axes(1, &[-1]), Some((0, 0)));
    }

    #[test]
    fn resolves_all_axes_when_every_one_is_named_contiguously() {
        assert_eq!(resolve_contiguous_axes(3, &[0, 1, 2]), Some((0, 2)));
    }
}
