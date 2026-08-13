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

use oxicuda_blas::reduction::{reduce_axis, ReductionOp};

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency-cache slot label for the reduction operand. A reduction input is
/// always an activation, so this only ever tags a transient pooled upload.
const INPUT_LABEL: &str = "reduce_in";

/// Decompose `shape` around `axis` into `(outer, axis_len, inner)`, and
/// decide whether this is a configuration [`cuda_reduce`] will attempt (as
/// opposed to declining to the CPU).
///
/// Pure and allocation-free, so the axis/shape bookkeeping is unit-testable
/// without a CUDA device — unlike the GPU launch itself, which cannot be
/// exercised on a host with no CUDA device.
///
/// Declines (`None`) when:
/// - `axis` is out of range for `shape` (a malformed model).
/// - The reduction would touch zero elements (`outer`, `axis_len`, or
///   `inner` is `0`): a degenerate edge case left to the CPU kernel's
///   identity-element handling rather than special-cased here.
fn reduce_plan(shape: &[usize], axis: usize) -> Option<(usize, usize, usize)> {
    if axis >= shape.len() {
        return None;
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    if outer == 0 || axis_len == 0 || inner == 0 {
        return None;
    }
    Some((outer, axis_len, inner))
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
}
