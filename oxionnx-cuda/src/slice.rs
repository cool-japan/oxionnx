//! CUDA `Slice` dispatch: a rank-4 strided-copy kernel.
//!
//! # Status
//!
//! No `oxicuda-dnn` kernel exists for this (a `grep` across every module
//! under `oxicuda-dnn/src` finds none), so — like [`crate::pad`] — this
//! module generates its own PTX with `oxicuda_ptx`'s [`KernelBuilder`].
//!
//! # Why every output index is provably in-bounds
//!
//! [`slice_params_from_node`] resolves each axis' `(start, step)` with the
//! *exact* clamp rule ONNX `Slice` (opset 10+) defines — mirrored from
//! `oxionnx-ops::shape::sequence::slice`, not called from it (see
//! [`mod@crate::reference`]'s "why this does not depend on `oxionnx-ops`"):
//! a negative `start`/`end` counts from the end of the axis, then the result
//! is clamped to `[0, dim]` for a positive step or `[-1, dim-1]` for a
//! negative one, and the output extent is derived from *that* clamped range.
//! The consequence the kernel leans on: for every output coordinate
//! `o` in `[0, out_shape[d])`, `start[d] + o * step[d]` is *always* inside
//! `[0, dim[d])` by construction — there is no configuration
//! [`slice_params_from_node`] accepts where it would not be. So
//! [`generate_slice_ptx`]'s kernel body needs no bounds check at all, unlike
//! [`crate::pad`]'s `constant` mode: it computes the mapped input index and
//! loads it unconditionally.
//!
//! # Scope: rank-4 only
//!
//! Every real `Slice` node in this workspace's target models is either a
//! genuine rank-4 NCHW-shaped tensor (`inswapper_128.onnx`'s 24 style-vector
//! channel splits, `[1, 2048, 1, 1] -> [1, 1024, 1, 1]`) or a tiny 1-D
//! shape-arithmetic vector (`det_10g.onnx`'s 2 `Resize`-sizing slices,
//! operating on a handful of `int64` values). This module claims only the
//! former — the CPU already computes the latter in negligible time, so
//! declining costs nothing observed, and it keeps this kernel's launch
//! geometry (a fixed 4-level `div`/`mod` decompose, mirroring
//! `oxicuda_dnn::pool`/`resize`'s own `(n, c, oh, ow)` decomposition) as
//! simple as every other kernel in this crate.
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::Slice`; a
//! non-rank-4 node, or one whose `axes`/`steps` this module cannot resolve,
//! still declines to `Ok(None)`. Shadow-verifiable via
//! [`crate::reference::ref_slice`] through the same `verify_or_fallback` gate
//! every other claimable op uses.

use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_ptx::prelude::*;

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency slot label for `Slice`'s data operand. `starts`/`ends`/`axes`/
/// `steps` are never bound this way — their host bytes are read directly by
/// [`slice_params_from_node`]'s caller.
pub(crate) const INPUT_LABEL: &str = "slice_input";

const SLICE_BLOCK: u32 = 256;
const SLICE_KERNEL_NAME: &str = "oxionnx_cuda_slice_f32";

/// Resolved geometry for one rank-4 `Slice` dispatch: per-axis `(start,
/// step)` and the output shape they produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceParams {
    /// Per-axis start index into the input (already sign/clamp-resolved —
    /// see the [module docs](self) for why every `start[d] + o*step[d]` for
    /// `o` in `[0, out_shape[d])` is guaranteed in `[0, in_shape[d])`).
    pub start: [i32; 4],
    /// Per-axis step (may be negative — a reversed walk).
    pub step: [i32; 4],
    /// Output shape.
    pub out_shape: [usize; 4],
}

/// Normalizes a possibly-negative ONNX axis against `rank`. A local duplicate
/// — see [`crate::pad`]'s identically-named helper's doc comment for why.
#[must_use]
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..r).contains(&a).then_some(a as usize)
}

/// Builds [`SliceParams`] for an ONNX `Slice` node from its already-resolved
/// `starts`/`ends`/`axes`/`steps` operands and the input shape, or declines.
///
/// `axes`/`steps` default exactly as the ONNX spec (and
/// `oxionnx-ops::shape::sequence::slice`) do: `axes` absent means
/// `starts`/`ends` name axes `0..starts.len()` in order; `steps` absent means
/// every named axis steps by `1`.
///
/// Pure and allocation-light: unit-testable without a CUDA device.
#[must_use]
pub fn slice_params_from_node(
    input_shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Option<SliceParams> {
    if input_shape.len() != 4 {
        return None;
    }
    let default_axes: Vec<i64> = (0..starts.len() as i64).collect();
    let axes = axes.unwrap_or(&default_axes);
    let default_steps: Vec<i64> = vec![1; starts.len()];
    let steps = steps.unwrap_or(&default_steps);
    if starts.len() != axes.len() || ends.len() != axes.len() || steps.len() != axes.len() {
        return None;
    }

    let mut dim_start = [0_i64; 4];
    let mut dim_end: [i64; 4] = [
        input_shape[0] as i64,
        input_shape[1] as i64,
        input_shape[2] as i64,
        input_shape[3] as i64,
    ];
    let mut dim_step = [1_i64; 4];

    for (i, &raw_ax) in axes.iter().enumerate() {
        let ax = normalize_axis(raw_ax, 4)?;
        let dim = input_shape[ax] as i64;
        let step = steps[i];
        if step == 0 || step == i64::MIN {
            return None;
        }
        let mut start = starts[i];
        let mut end = ends[i];
        // `dim` is bounded by a real tensor size, so `start + dim` cannot
        // overflow even at `start == i64::MIN`; `saturating_add` is
        // defense-in-depth, mirroring `oxionnx-ops::shape::sequence::slice`.
        if start < 0 {
            start = start.saturating_add(dim);
        }
        if end < 0 {
            end = end.saturating_add(dim);
        }
        if step > 0 {
            start = start.clamp(0, dim);
            end = end.clamp(0, dim);
        } else {
            start = start.clamp(-1, dim - 1);
            end = end.clamp(-1, dim - 1);
        }
        dim_start[ax] = start;
        dim_end[ax] = end;
        dim_step[ax] = step;
    }

    let mut out_shape = [0_usize; 4];
    for d in 0..4 {
        let (start, end, step) = (dim_start[d], dim_end[d], dim_step[d]);
        out_shape[d] = if step > 0 {
            if end <= start {
                0
            } else {
                ((end - start + step - 1) / step) as usize
            }
        } else {
            let neg_step = -step;
            if start <= end {
                0
            } else {
                ((start - end + neg_step - 1) / neg_step) as usize
            }
        };
    }

    let mut start_i32 = [0_i32; 4];
    let mut step_i32 = [0_i32; 4];
    for d in 0..4 {
        start_i32[d] = i32::try_from(dim_start[d]).ok()?;
        step_i32[d] = i32::try_from(dim_step[d]).ok()?;
    }

    Some(SliceParams {
        start: start_i32,
        step: step_i32,
        out_shape,
    })
}

// ─── PTX generation ─────────────────────────────────────────────────────────

/// Computes one axis' contribution to the flattened input index:
/// `(start + out_coord * step) * stride`, entirely in registers, with no
/// bounds check (see the [module docs](self) for why none is needed).
#[allow(clippy::too_many_arguments)]
fn axis_contribution(
    b: &mut BodyBuilder<'_>,
    out_coord: Register,
    start_param: &str,
    step_param: &str,
    stride_param: &str,
) -> Register {
    let start_bits = b.load_param_u32(start_param);
    let start = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {start}, {start_bits};"));
    let step_bits = b.load_param_u32(step_param);
    let step = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {step}, {step_bits};"));

    let out_coord_s32 = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {out_coord_s32}, {out_coord};"));
    let scaled = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mul.lo.s32 {scaled}, {out_coord_s32}, {step};"));
    let in_coord = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("add.s32 {in_coord}, {scaled}, {start};"));

    // Provably non-negative (see the module docs): a lossless reinterpret.
    let in_coord_u = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.b32 {in_coord_u}, {in_coord};"));

    let stride = b.load_param_u32(stride_param);
    b.mul_lo_u32(in_coord_u, stride)
}

/// Generates the rank-4 `Slice` kernel.
fn generate_slice_ptx(sm: SmVersion) -> Result<String, CudaDispatchError> {
    let ptx = KernelBuilder::new(SLICE_KERNEL_NAME)
        .target(sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("out_d0", PtxType::U32)
        .param("out_d1", PtxType::U32)
        .param("out_d2", PtxType::U32)
        .param("out_d3", PtxType::U32)
        .param("start0", PtxType::U32)
        .param("start1", PtxType::U32)
        .param("start2", PtxType::U32)
        .param("start3", PtxType::U32)
        .param("step0", PtxType::U32)
        .param("step1", PtxType::U32)
        .param("step2", PtxType::U32)
        .param("step3", PtxType::U32)
        .param("stride0", PtxType::U32)
        .param("stride1", PtxType::U32)
        .param("stride2", PtxType::U32)
        .param("stride3", PtxType::U32)
        .param("total", PtxType::U32)
        .max_threads_per_block(SLICE_BLOCK)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total");
            b.if_lt_u32(gid.clone(), total, move |b| {
                // Decompose gid -> (o0, o1, o2, o3), row-major (`out_d3`
                // fastest-varying) -- the same 4-level div/mod chain
                // `oxicuda_dnn::pool`/`resize` use to decompose `(n, c, oh,
                // ow)`, generalised to arbitrary per-axis extents.
                let out_d3 = b.load_param_u32("out_d3");
                let out_d2 = b.load_param_u32("out_d2");
                let out_d1 = b.load_param_u32("out_d1");

                let o3 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {o3}, {gid}, {out_d3};"));
                let t1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {t1}, {gid}, {out_d3};"));
                let o2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {o2}, {t1}, {out_d2};"));
                let t2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {t2}, {t1}, {out_d2};"));
                let o1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {o1}, {t2}, {out_d1};"));
                let o0 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {o0}, {t2}, {out_d1};"));

                let c0 = axis_contribution(b, o0, "start0", "step0", "stride0");
                let c1 = axis_contribution(b, o1, "start1", "step1", "stride1");
                let c2 = axis_contribution(b, o2, "start2", "step2", "stride2");
                let c3 = axis_contribution(b, o3, "start3", "step3", "stride3");
                let c01 = b.add_u32(c0, c1);
                let c012 = b.add_u32(c01, c2);
                let in_idx = b.add_u32(c012, c3);

                let in_ptr = b.load_param_u64("in_ptr");
                let addr = b.f32_elem_addr(in_ptr, in_idx);
                let val = b.load_global_f32(addr);

                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.f32_elem_addr(out_ptr, gid.clone());
                b.store_global_f32(out_addr, val);
            });
            b.ret();
        })
        .build()
        .map_err(|e| CudaDispatchError::Ptx(e.to_string()))?;
    Ok(ptx)
}

/// Fetches — compiling on first use — the `Slice` kernel.
fn kernel_for(ctx: &CudaContext) -> Result<Kernel, CudaDispatchError> {
    let sm = ctx.dnn.sm_version();
    let module = ctx.module(SLICE_KERNEL_NAME, || generate_slice_ptx(sm))?;
    Kernel::from_module(module, SLICE_KERNEL_NAME).map_err(CudaDispatchError::Driver)
}

// ─── dispatch ───────────────────────────────────────────────────────────────

/// Row-major strides (in elements) for a rank-4 shape.
#[must_use]
fn row_major_strides(shape: [usize; 4]) -> [usize; 4] {
    let mut strides = [1_usize; 4];
    for d in (0..3).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// ONNX `Slice` forward on the GPU, over an operand that may already be on
/// the device, leaving the result there when the caller asks for it.
///
/// # Returns
/// * `Ok(Some(_))` — computed on the GPU.
/// * `Ok(None)` — not accelerated; see the [module docs](self).
/// * `Err(_)` — a real failure after dispatch was already committed to.
///
/// # Errors
/// See "Returns" above.
pub(crate) fn cuda_slice_bound(
    ctx: &CudaContext,
    input: InputBinding<'_>,
    input_shape: &[usize],
    params: &SliceParams,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    if input_shape.len() != 4 {
        return Ok(None);
    }
    let in_shape: [usize; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    if in_shape.contains(&0) {
        return Ok(None);
    }
    let Some(in_needed) = in_shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
    else {
        return Ok(None);
    };
    let Some(out_needed) = params
        .out_shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
    else {
        return Ok(None);
    };
    if input.len() < in_needed {
        return Ok(None);
    }

    let in_stride = row_major_strides(in_shape);

    let to_u32_4 = |v: [usize; 4]| -> Option<[u32; 4]> {
        Some([
            u32::try_from(v[0]).ok()?,
            u32::try_from(v[1]).ok()?,
            u32::try_from(v[2]).ok()?,
            u32::try_from(v[3]).ok()?,
        ])
    };
    let Some(out_d) = to_u32_4(params.out_shape) else {
        return Ok(None);
    };
    let Some(stride_u32) = to_u32_4(in_stride) else {
        return Ok(None);
    };
    let Ok(total_u32) = u32::try_from(out_needed) else {
        return Ok(None);
    };
    // Bit-cast, matching `decode_pad_coords`'/`axis_contribution`'s
    // convention: the kernel parameter is declared `u32` and reinterpreted
    // to `s32` inside the kernel body.
    let start_bits: [u32; 4] = params.start.map(|v| v as u32);
    let step_bits: [u32; 4] = params.step.map(|v| v as u32);

    let stream = ctx.dnn.stream();
    let Some(mut d_input) = input.bind(ctx, INPUT_LABEL, in_needed, stream)? else {
        return Ok(None);
    };
    let d_output = ctx.scratch(out_needed)?;
    // No zero-fill: every one of the `out_needed` output elements is written
    // by exactly one thread when `total_u32 > 0` (same reasoning as
    // `cuda_pool_bound`/`cuda_pad_bound`); when it is `0` there is nothing to
    // read back.

    if total_u32 > 0 {
        let kernel = kernel_for(ctx)?;
        let grid = grid_size_for(total_u32, SLICE_BLOCK);
        let launch_params = LaunchParams::new(Dim3::from(grid), Dim3::from(SLICE_BLOCK));
        let args = (
            d_input.device_ptr(),
            d_output.device_ptr(),
            out_d[0],
            out_d[1],
            out_d[2],
            out_d[3],
            start_bits[0],
            start_bits[1],
            start_bits[2],
            start_bits[3],
            step_bits[0],
            step_bits[1],
            step_bits[2],
            step_bits[3],
            stride_u32[0],
            stride_u32[1],
            stride_u32[2],
            stride_u32[3],
            total_u32,
        );
        kernel
            .launch(&launch_params, stream, &args)
            .map_err(CudaDispatchError::Driver)?;
    }

    let out_shape = params.out_shape.to_vec();
    let out = finish_output(ctx, d_output, out_needed, &out_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => d_input.retire(),
        KernelOutput::Device(_) => retire_queued(ctx, &mut d_input),
    }
    Ok(Some(out))
}

/// [`cuda_slice_bound`] over plain host slices, always reading the result
/// back. The non-resident entry point this module's own tests use.
///
/// # Errors
/// As [`cuda_slice_bound`].
#[must_use = "the slice result is only computed if this is consumed"]
pub fn cuda_slice(
    ctx: &CudaContext,
    input: &[f32],
    input_shape: &[usize],
    params: &SliceParams,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    match cuda_slice_bound(
        ctx,
        InputBinding::Host(input),
        input_shape,
        params,
        CudaOutputPlacement::Host,
    )? {
        Some(KernelOutput::Host(data)) => Ok(Some(data)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "Slice",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── slice_params_from_node ──────────────────────────────────────────────

    #[test]
    fn inswapper_style_channel_half_split_is_claimed() {
        // Real pattern: [1, 2048, 1, 1] sliced [0:1024] and [1024:2048] along
        // axis 1 (see `Slice_86`/`Slice_89` in inswapper_128.onnx).
        let first = slice_params_from_node(&[1, 2048, 1, 1], &[0], &[1024], Some(&[1]), None)
            .expect("must be claimable");
        assert_eq!(first.out_shape, [1, 1024, 1, 1]);
        assert_eq!(first.start, [0, 0, 0, 0]);
        assert_eq!(first.step, [1, 1, 1, 1]);

        let second = slice_params_from_node(&[1, 2048, 1, 1], &[1024], &[2048], Some(&[1]), None)
            .expect("must be claimable");
        assert_eq!(second.out_shape, [1, 1024, 1, 1]);
        assert_eq!(second.start, [0, 1024, 0, 0]);
    }

    #[test]
    fn negative_start_counts_from_the_end() {
        let params = slice_params_from_node(&[1, 8, 4, 4], &[-4], &[8], Some(&[1]), None)
            .expect("must be claimable");
        assert_eq!(params.out_shape, [1, 4, 4, 4]);
        assert_eq!(params.start[1], 4);
    }

    #[test]
    fn a_negative_step_reverses_the_axis() {
        // ONNX's own convention for "reverse the whole axis": `start=-1`
        // (the last real index) and `end` more negative than `-dim` (here
        // `-5` for `dim=4`) so it clamps to the "one before index 0"
        // sentinel `-1`, not to `dim-1` the way a merely-negative-but-in-
        // range `end` would (a plain `end=-1` computes to `dim-1=3`,
        // *equal* to `start`, and yields an *empty* slice -- the mistake
        // this test used to make before being hand-corrected against
        // `oxionnx-ops::shape::sequence::slice`'s own doc comment).
        let params = slice_params_from_node(&[1, 4, 1, 1], &[-1], &[-5], Some(&[1]), Some(&[-1]))
            .expect("must be claimable");
        assert_eq!(params.out_shape, [1, 4, 1, 1]);
        assert_eq!(params.start[1], 3);
        assert_eq!(params.step[1], -1);
    }

    #[test]
    fn a_zero_step_declines() {
        assert!(
            slice_params_from_node(&[1, 4, 1, 1], &[0], &[4], Some(&[1]), Some(&[0])).is_none()
        );
    }

    #[test]
    fn default_axes_name_zero_upward() {
        let params = slice_params_from_node(&[8, 1, 1, 1], &[2], &[6], None, None)
            .expect("must be claimable");
        assert_eq!(params.out_shape, [4, 1, 1, 1]);
    }

    #[test]
    fn a_full_range_no_op_slice_is_claimed() {
        let params = slice_params_from_node(&[1, 16, 4, 4], &[0], &[16], Some(&[1]), None)
            .expect("must be claimable");
        assert_eq!(params.out_shape, [1, 16, 4, 4]);
        assert_eq!(params.start[1], 0);
        assert_eq!(params.step[1], 1);
    }

    #[test]
    fn an_empty_slice_range_yields_a_zero_output_axis() {
        let params = slice_params_from_node(&[1, 16, 4, 4], &[10], &[5], Some(&[1]), None)
            .expect("must be claimable");
        assert_eq!(params.out_shape[1], 0);
    }

    #[test]
    fn non_4d_input_declines() {
        assert!(slice_params_from_node(&[16], &[0], &[8], Some(&[0]), None).is_none());
    }

    #[test]
    fn mismatched_lengths_decline() {
        assert!(slice_params_from_node(&[1, 16, 4, 4], &[0, 1], &[8], Some(&[1]), None).is_none());
    }

    // ── row_major_strides ───────────────────────────────────────────────────

    #[test]
    fn row_major_strides_match_nchw_layout() {
        assert_eq!(row_major_strides([1, 2048, 1, 1]), [2048, 1, 1, 1]);
        assert_eq!(row_major_strides([2, 3, 4, 5]), [60, 20, 5, 1]);
    }

    // ── PTX generation ───────────────────────────────────────────────────────

    #[test]
    fn slice_ptx_generates_and_validates() {
        let ptx = generate_slice_ptx(SmVersion::Sm86).expect("must generate");
        assert!(ptx.contains(SLICE_KERNEL_NAME));
        let report = validate_ptx(&ptx);
        assert!(
            report.is_ok(),
            "generated slice PTX failed validation: {report:?}"
        );
    }
}
