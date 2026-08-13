//! CUDA `Pad` dispatch: `reflect` and `constant` modes over the spatial axes
//! of a 4-D NCHW tensor.
//!
//! # Status
//!
//! Unlike [`crate::pool`]/[`crate::resize`], `oxicuda-dnn` has no existing
//! pad kernel to dispatch to (`grep`-confirmed against every module under
//! `oxicuda-dnn/src`), so this module generates its own PTX directly with
//! `oxicuda_ptx`'s [`KernelBuilder`] — the same low-level DSL
//! `oxicuda-dnn`'s own kernels are built from, and the same one
//! [`crate::elementwise`]/[`crate::softmax`] already use for kernels that
//! live in *this* crate rather than `oxicuda-dnn`.
//!
//! The reflect formula mirrors `oxionnx-ops::shape::sequence::pad_axes`'s
//! `"reflect"` arm bit-for-bit (not called from it — see
//! [`mod@crate::reference`]'s "why this does not depend on `oxionnx-ops`"),
//! and the WGSL precedent already validated in `oxionnx-gpu/src/shaders/pad.rs`
//! uses the identical formula for the identical reason:
//! `c = (out_coord - begin).rem_euclid(2*(dim-1)); if c >= dim { c = 2*(dim-1) - c }`.
//! Because that formula is a *periodic* fold, it produces a valid `[0, dim)`
//! index unconditionally — including for pads several multiples of `dim`
//! wide — so [`generate_pad_reflect_ptx`]'s kernel body needs no bounds
//! branch at all, only the fold. `constant` mode is the opposite: it is
//! *exactly* a bounds branch (in-range → load, out-of-range → the constant),
//! so the two modes get two separate kernels rather than one kernel with a
//! runtime mode flag, mirroring the cip/nocip and ac/noac split
//! `oxicuda_dnn::pool`/`oxicuda_dnn::resize` already use for the same reason:
//! no wasted branches, no unused parameters.
//!
//! # Only `N`/`C`-unpadded, spatial-only padding is claimed
//!
//! Mirrors the WGSL precedent's own scope note: `N`/`C` always pass through
//! unpadded, matching how `inswapper_128.onnx`'s 14 reflect-pad nodes are
//! actually used (padding the spatial extent ahead of a convolution).
//! [`pad_params_from_node`] declines any node whose resolved per-dimension
//! padding is nonzero on axis 0 or 1, whichever way the node named its axes
//! attribute (whether it is absent — the default, every axis in declaration
//! order — or an explicit `[2, 3]` naming only the spatial ones).
//!
//! `pads` may be negative (an ONNX-legal crop) since opset 11; this dispatch
//! supports that too — `constant` mode's bounds branch and `reflect` mode's
//! periodic fold are both already correct for a negative `pad_begin`, so
//! nothing about the kernel needs to special-case it.
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::Pad`; a node whose
//! `mode`/`axes`/resolved-padding falls outside the whitelist above still
//! declines to `Ok(None)`. Shadow-verifiable via
//! [`crate::reference::ref_pad`] through the same `verify_or_fallback` gate
//! every other claimable op uses.

use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_ptx::prelude::*;

use oxionnx_core::Attributes;

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency slot label for `Pad`'s data operand. The `pads`/`constant_value`/
/// `axes` inputs are never bound this way — their host bytes are read
/// directly by [`pad_params_from_node`]'s caller, the same treatment
/// [`crate::conv`] gives a convolution's weight/bias.
pub(crate) const INPUT_LABEL: &str = "pad_input";

const PAD_BLOCK: u32 = 256;
const PAD_REFLECT_KERNEL_NAME: &str = "oxionnx_cuda_pad_reflect_f32";
const PAD_CONSTANT_KERNEL_NAME: &str = "oxionnx_cuda_pad_constant_f32";

/// Which ONNX `Pad` mode a dispatch computes. `edge`/`wrap` (which
/// `oxionnx-ops::shape::sequence::pad_axes` also supports on the CPU) have no
/// kernel here — there is nothing to select if a node asks for one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Mirror the input across each edge, indefinitely (`c =
    /// (out_coord - begin).rem_euclid(2*(dim-1))`, folded once more if it
    /// lands in the upper half).
    Reflect,
    /// Fill every out-of-range position with a single constant value.
    Constant,
}

/// Resolved geometry for one `Pad` dispatch.
///
/// Padding is expressed as signed `i32` (matching the kernel parameter's
/// bit-cast representation — see [`generate_pad_reflect_ptx`]'s body) so a
/// negative value (crop) needs no separate code path anywhere downstream of
/// [`pad_params_from_node`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadParams {
    /// Height padding: `(begin, end)`.
    pub pad_h: (i32, i32),
    /// Width padding: `(begin, end)`.
    pub pad_w: (i32, i32),
    /// `reflect` or `constant`.
    pub mode: PadMode,
    /// The fill value for `constant` mode. Ignored by `reflect`.
    pub constant_value: f32,
}

/// Normalizes a possibly-negative ONNX axis against `rank`. Independent of
/// (mirrors, does not call) `oxionnx-ops::shape::basic::normalize_axis` — see
/// [`mod@crate::reference`]'s "why this does not depend on `oxionnx-ops`".
#[must_use]
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..r).contains(&a).then_some(a as usize)
}

/// Builds [`PadParams`] for an ONNX `Pad` node from its `mode` attribute, its
/// already-resolved `pads` (and optional `axes`) operand, and the input
/// shape — or declines. See the [module docs](self) for the full whitelist.
///
/// `pads` is `2 * axes.len()` (or `2 * input_shape.len()` when `axes` is
/// `None`) signed values, `[begin_0, ..., begin_{k-1}, end_0, ..., end_{k-1}]`
/// — the layout `oxionnx-ops::shape::sequence::pad_axes` documents and the
/// one the node's `pads` tensor input carries verbatim once cast from `f32`
/// to `i64`.
///
/// Pure and allocation-light: unit-testable without a CUDA device.
#[must_use]
pub fn pad_params_from_node(
    attrs: &Attributes,
    input_shape: &[usize],
    pads: &[i64],
    axes: Option<&[i64]>,
    constant_value: f32,
) -> Option<PadParams> {
    let ndim = input_shape.len();
    if ndim != 4 {
        return None;
    }
    let raw_mode = attrs.s("mode");
    let mode = match if raw_mode.is_empty() {
        "constant"
    } else {
        raw_mode
    } {
        "constant" => PadMode::Constant,
        "reflect" => PadMode::Reflect,
        // "edge"/"wrap"/anything else: no kernel.
        _ => return None,
    };

    let axes_norm: Vec<usize> = match axes {
        Some(raw) => raw
            .iter()
            .map(|&ax| normalize_axis(ax, ndim))
            .collect::<Option<_>>()?,
        None => (0..ndim).collect(),
    };
    if pads.len() != 2 * axes_norm.len() {
        return None;
    }

    let mut begin = [0_i64; 4];
    let mut end = [0_i64; 4];
    for (i, &ax) in axes_norm.iter().enumerate() {
        begin[ax] = pads[i];
        end[ax] = pads[axes_norm.len() + i];
    }
    // Only H/W (axes 2, 3) may be padded -- N/C passing through unpadded is
    // what keeps this a 2-D spatial pad rather than a batch/channel
    // reinterpretation the launch geometry below has no way to express.
    if begin[0] != 0 || end[0] != 0 || begin[1] != 0 || end[1] != 0 {
        return None;
    }

    let in_h = input_shape[2];
    let in_w = input_shape[3];
    if mode == PadMode::Reflect && (in_h <= 1 || in_w <= 1) {
        // The periodic-fold formula's period is `2*(dim-1)`; a dimension of
        // 0 or 1 has nothing to reflect across. Not reachable by either
        // target model (every reflect-pad axis in `inswapper_128.onnx` is a
        // real, multi-pixel feature-map extent), so declining here costs
        // nothing observed, only a formula this module would otherwise have
        // to special-case for a configuration nothing exercises.
        return None;
    }

    let pad_h_begin = i32::try_from(begin[2]).ok()?;
    let pad_h_end = i32::try_from(end[2]).ok()?;
    let pad_w_begin = i32::try_from(begin[3]).ok()?;
    let pad_w_end = i32::try_from(end[3]).ok()?;

    // A crop must not remove more than the whole axis (a spec violation the
    // CPU kernel reports properly); checked here so `cuda_pad_bound` can
    // trust `out_h`/`out_w` never underflow.
    let out_h = (in_h as i64) + i64::from(pad_h_begin) + i64::from(pad_h_end);
    let out_w = (in_w as i64) + i64::from(pad_w_begin) + i64::from(pad_w_end);
    if out_h < 0 || out_w < 0 || usize::try_from(out_h).is_err() || usize::try_from(out_w).is_err()
    {
        return None;
    }

    Some(PadParams {
        pad_h: (pad_h_begin, pad_h_end),
        pad_w: (pad_w_begin, pad_w_end),
        mode,
        constant_value,
    })
}

/// The `[N, C, out_H, out_W]` shape a claimed `Pad` node produces from
/// `input_shape` and an already-resolved [`PadParams`].
///
/// Exposed so a caller that needs the shape — `lib.rs`'s `Pad` arm, which
/// must attach one to a `Host`-placement result — does not have to re-derive
/// the `in_dim + pad_begin + pad_end` arithmetic itself; [`cuda_pad_bound`]
/// answers the identical question internally. `None` only when `input_shape`
/// is not rank-4 or the resolved extent would be negative — neither should
/// arise for a `params` this module itself produced via
/// [`pad_params_from_node`] against the same `input_shape`, since that
/// function already checked the crop does not underflow.
#[must_use]
pub fn pad_output_shape(input_shape: &[usize], params: &PadParams) -> Option<[usize; 4]> {
    if input_shape.len() != 4 {
        return None;
    }
    let n = input_shape[0];
    let c = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];
    let out_h = in_h as i64 + i64::from(params.pad_h.0) + i64::from(params.pad_h.1);
    let out_w = in_w as i64 + i64::from(params.pad_w.0) + i64::from(params.pad_w.1);
    let out_h = usize::try_from(out_h).ok()?;
    let out_w = usize::try_from(out_w).ok()?;
    Some([n, c, out_h, out_w])
}

// ─── PTX generation ─────────────────────────────────────────────────────────

/// Folds a signed coordinate into `[0, dim)` by reflection with period
/// `2*(dim-1)`, entirely branchless (two `rem`/`selp` pairs). `dim > 1` is a
/// precondition the caller (a claimed [`PadMode::Reflect`] node) has already
/// checked in [`pad_params_from_node`], so `period > 0` always holds here.
fn reflect_fold(
    b: &mut BodyBuilder<'_>,
    offset: Register,
    period: Register,
    dim: Register,
) -> Register {
    // `rem.s32` (truncating remainder, C/Rust `%` semantics) yields a value
    // in `(-period, period)`; add `period` back on when it came out negative
    // to land in `[0, period)`, matching `i64::rem_euclid`.
    let rem = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("rem.s32 {rem}, {offset}, {period};"));
    let rem_plus_period = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("add.s32 {rem_plus_period}, {rem}, {period};"));
    let is_negative = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lt.s32 {is_negative}, {rem}, 0;"));
    let euclid = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!(
        "selp.s32 {euclid}, {rem_plus_period}, {rem}, {is_negative};"
    ));

    // `euclid` is now in `[0, period)`; the upper half (`[dim, period)`)
    // mirrors back down to `[1, dim)`.
    let mirrored = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("sub.s32 {mirrored}, {period}, {euclid};"));
    let in_upper_half = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.s32 {in_upper_half}, {euclid}, {dim};"));
    let folded = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!(
        "selp.s32 {folded}, {mirrored}, {euclid}, {in_upper_half};"
    ));
    folded
}

/// Shared prologue for both pad kernels: decomposes `gid` into
/// `(nc_idx, oh_idx, ow_idx)` and computes the signed input coordinates
/// `(ih, iw) = (oh_idx, ow_idx) - (pad_h_begin, pad_w_begin)`.
///
/// `N`/`C` are never padded (checked in [`pad_params_from_node`]), so input
/// and output share one `nc_idx` -- there is no separate `n`/`c` decompose to
/// do, unlike `oxicuda_dnn::pool`/`resize`'s kernels.
struct PadCoords {
    nc_idx: Register,
    ih: Register,
    iw: Register,
}

fn decode_pad_coords(b: &mut BodyBuilder<'_>, gid: Register) -> PadCoords {
    let out_w = b.load_param_u32("out_w");
    let out_h = b.load_param_u32("out_h");

    let ow_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow_idx}, {gid}, {out_w};"));
    let tmp1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {tmp1}, {gid}, {out_w};"));
    let oh_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {oh_idx}, {tmp1}, {out_h};"));
    let nc_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {nc_idx}, {tmp1}, {out_h};"));

    // Kernel parameters carry the pad offsets as `u32` bit patterns (the
    // host side bit-casts `i32 as u32`); reinterpret back to `s32` here.
    let pad_h_begin_bits = b.load_param_u32("pad_h_begin");
    let pad_w_begin_bits = b.load_param_u32("pad_w_begin");
    let pad_h_begin = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {pad_h_begin}, {pad_h_begin_bits};"));
    let pad_w_begin = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {pad_w_begin}, {pad_w_begin_bits};"));

    let oh_s32 = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {oh_s32}, {oh_idx};"));
    let ow_s32 = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {ow_s32}, {ow_idx};"));

    let ih = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("sub.s32 {ih}, {oh_s32}, {pad_h_begin};"));
    let iw = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("sub.s32 {iw}, {ow_s32}, {pad_w_begin};"));

    PadCoords { nc_idx, ih, iw }
}

/// Generates the `reflect`-mode pad kernel.
fn generate_pad_reflect_ptx(sm: SmVersion) -> Result<String, CudaDispatchError> {
    let ptx = KernelBuilder::new(PAD_REFLECT_KERNEL_NAME)
        .target(sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("pad_h_begin", PtxType::U32)
        .param("pad_w_begin", PtxType::U32)
        .param("total", PtxType::U32)
        .max_threads_per_block(PAD_BLOCK)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total");
            b.if_lt_u32(gid.clone(), total, move |b| {
                let coords = decode_pad_coords(b, gid.clone());
                let in_h = b.load_param_u32("in_h");
                let in_w = b.load_param_u32("in_w");
                let in_h_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.b32 {in_h_s32}, {in_h};"));
                let in_w_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.b32 {in_w_s32}, {in_w};"));

                let in_h_m1 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {in_h_m1}, {in_h_s32}, 1;"));
                let period_h = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mul.lo.s32 {period_h}, {in_h_m1}, 2;"));
                let in_w_m1 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("sub.s32 {in_w_m1}, {in_w_s32}, 1;"));
                let period_w = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mul.lo.s32 {period_w}, {in_w_m1}, 2;"));

                let ih_folded = reflect_fold(b, coords.ih, period_h, in_h_s32);
                let iw_folded = reflect_fold(b, coords.iw, period_w, in_w_s32);
                let ih_u = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {ih_u}, {ih_folded};"));
                let iw_u = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {iw_u}, {iw_folded};"));

                let in_hw = b.mul_lo_u32(in_h.clone(), in_w.clone());
                let plane_off = b.mul_lo_u32(coords.nc_idx, in_hw);
                let row_off = b.mul_lo_u32(ih_u, in_w);
                let hw_off = b.add_u32(row_off, iw_u);
                let in_idx = b.add_u32(plane_off, hw_off);

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

/// Generates the `constant`-mode pad kernel.
fn generate_pad_constant_ptx(sm: SmVersion) -> Result<String, CudaDispatchError> {
    let ptx = KernelBuilder::new(PAD_CONSTANT_KERNEL_NAME)
        .target(sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("pad_h_begin", PtxType::U32)
        .param("pad_w_begin", PtxType::U32)
        .param("fill_value", PtxType::F32)
        .param("total", PtxType::U32)
        .max_threads_per_block(PAD_BLOCK)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let total = b.load_param_u32("total");
            b.if_lt_u32(gid.clone(), total, move |b| {
                let coords = decode_pad_coords(b, gid.clone());
                let in_h = b.load_param_u32("in_h");
                let in_w = b.load_param_u32("in_w");
                let in_h_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.b32 {in_h_s32}, {in_h};"));
                let in_w_s32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.b32 {in_w_s32}, {in_w};"));

                // In-range predicate, one compound `setp` per axis exactly
                // like `oxicuda_dnn::pool::max_pool2d`'s bounds check.
                let h_ge0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {h_ge0}, {}, 0;", coords.ih));
                let h_ok = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.lt.and.s32 {h_ok}, {}, {in_h_s32}, {h_ge0};",
                    coords.ih
                ));
                let w_ge0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {w_ge0}, {}, 0;", coords.iw));
                let w_ok = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.lt.and.s32 {w_ok}, {}, {in_w_s32}, {w_ge0};",
                    coords.iw
                ));
                let in_bounds = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("and.pred {in_bounds}, {h_ok}, {w_ok};"));

                let fill = b.load_param_f32("fill_value");
                let result = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {result}, {fill};"));

                let skip = b.fresh_label("pad_const_oob");
                let out_of_bounds = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {out_of_bounds}, {in_bounds};"));
                b.branch_if(out_of_bounds, &skip);

                let ih_u = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {ih_u}, {};", coords.ih));
                let iw_u = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {iw_u}, {};", coords.iw));
                let in_hw = b.mul_lo_u32(in_h.clone(), in_w.clone());
                let plane_off = b.mul_lo_u32(coords.nc_idx, in_hw);
                let row_off = b.mul_lo_u32(ih_u, in_w);
                let hw_off = b.add_u32(row_off, iw_u);
                let in_idx = b.add_u32(plane_off, hw_off);
                let in_ptr = b.load_param_u64("in_ptr");
                let addr = b.f32_elem_addr(in_ptr, in_idx);
                let val = b.load_global_f32(addr);
                b.raw_ptx(&format!("mov.f32 {result}, {val};"));

                b.label(&skip);
                let out_ptr = b.load_param_u64("out_ptr");
                let out_addr = b.f32_elem_addr(out_ptr, gid.clone());
                b.store_global_f32(out_addr, result);
            });
            b.ret();
        })
        .build()
        .map_err(|e| CudaDispatchError::Ptx(e.to_string()))?;
    Ok(ptx)
}

/// Fetches — compiling on first use — the kernel for one [`PadMode`].
fn kernel_for(ctx: &CudaContext, mode: PadMode) -> Result<Kernel, CudaDispatchError> {
    let name = match mode {
        PadMode::Reflect => PAD_REFLECT_KERNEL_NAME,
        PadMode::Constant => PAD_CONSTANT_KERNEL_NAME,
    };
    let sm = ctx.dnn.sm_version();
    let module = ctx.module(name, || match mode {
        PadMode::Reflect => generate_pad_reflect_ptx(sm),
        PadMode::Constant => generate_pad_constant_ptx(sm),
    })?;
    Kernel::from_module(module, name).map_err(CudaDispatchError::Driver)
}

// ─── dispatch ───────────────────────────────────────────────────────────────

/// ONNX `Pad` (`reflect`/`constant`, spatial-only) forward on the GPU, over an
/// operand that may already be on the device, leaving the result there when
/// the caller asks for it.
///
/// Mirrors [`crate::pool::cuda_pool_bound`]'s shape: a single operand, no
/// epilogue.
///
/// # Returns
/// * `Ok(Some(_))` — computed on the GPU.
/// * `Ok(None)` — not accelerated; see the [module docs](self).
/// * `Err(_)` — a real failure after dispatch was already committed to.
///
/// # Errors
/// See "Returns" above.
pub(crate) fn cuda_pad_bound(
    ctx: &CudaContext,
    input: InputBinding<'_>,
    input_shape: &[usize],
    params: &PadParams,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    if input_shape.len() != 4 {
        return Ok(None);
    }
    let n = input_shape[0];
    let c = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];
    if n == 0 || c == 0 || in_h == 0 || in_w == 0 {
        return Ok(None);
    }

    let out_h_i64 = in_h as i64 + i64::from(params.pad_h.0) + i64::from(params.pad_h.1);
    let out_w_i64 = in_w as i64 + i64::from(params.pad_w.0) + i64::from(params.pad_w.1);
    let (Ok(out_h), Ok(out_w)) = (usize::try_from(out_h_i64), usize::try_from(out_w_i64)) else {
        return Ok(None);
    };

    let (Some(in_needed), Some(out_needed)) = (
        n.checked_mul(c)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w)),
        n.checked_mul(c)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w)),
    ) else {
        return Ok(None);
    };
    if input.len() < in_needed {
        return Ok(None);
    }

    let (Ok(in_h_u32), Ok(in_w_u32), Ok(out_h_u32), Ok(out_w_u32), Ok(total_u32)) = (
        u32::try_from(in_h),
        u32::try_from(in_w),
        u32::try_from(out_h),
        u32::try_from(out_w),
        u32::try_from(out_needed),
    ) else {
        return Ok(None);
    };
    let pad_h_begin_bits = params.pad_h.0 as u32;
    let pad_w_begin_bits = params.pad_w.0 as u32;

    let stream = ctx.dnn.stream();
    let Some(mut d_input) = input.bind(ctx, INPUT_LABEL, in_needed, stream)? else {
        return Ok(None);
    };
    let d_output = ctx.scratch(out_needed)?;
    // No zero-fill: below, every one of the `out_needed` output elements is
    // written by exactly one thread when `total_u32 > 0` (same reasoning as
    // `cuda_pool_bound`); when it is `0` there is nothing to read back at all.

    if total_u32 > 0 {
        let kernel = kernel_for(ctx, params.mode)?;
        let grid = grid_size_for(total_u32, PAD_BLOCK);
        let launch_params = LaunchParams::new(Dim3::from(grid), Dim3::from(PAD_BLOCK));
        match params.mode {
            PadMode::Reflect => {
                let args = (
                    d_input.device_ptr(),
                    d_output.device_ptr(),
                    in_h_u32,
                    in_w_u32,
                    out_h_u32,
                    out_w_u32,
                    pad_h_begin_bits,
                    pad_w_begin_bits,
                    total_u32,
                );
                kernel
                    .launch(&launch_params, stream, &args)
                    .map_err(CudaDispatchError::Driver)?;
            }
            PadMode::Constant => {
                let args = (
                    d_input.device_ptr(),
                    d_output.device_ptr(),
                    in_h_u32,
                    in_w_u32,
                    out_h_u32,
                    out_w_u32,
                    pad_h_begin_bits,
                    pad_w_begin_bits,
                    params.constant_value,
                    total_u32,
                );
                kernel
                    .launch(&launch_params, stream, &args)
                    .map_err(CudaDispatchError::Driver)?;
            }
        }
    }

    let out_shape = vec![n, c, out_h, out_w];
    let out = finish_output(ctx, d_output, out_needed, &out_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => d_input.retire(),
        KernelOutput::Device(_) => retire_queued(ctx, &mut d_input),
    }
    Ok(Some(out))
}

/// [`cuda_pad_bound`] over plain host slices, always reading the result back.
/// The non-resident entry point this module's own tests use.
///
/// # Errors
/// As [`cuda_pad_bound`].
#[must_use = "the pad result is only computed if this is consumed"]
pub fn cuda_pad(
    ctx: &CudaContext,
    input: &[f32],
    input_shape: &[usize],
    params: &PadParams,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    match cuda_pad_bound(
        ctx,
        InputBinding::Host(input),
        input_shape,
        params,
        CudaOutputPlacement::Host,
    )? {
        Some(KernelOutput::Host(data)) => Ok(Some(data)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "Pad",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_mode(mode: &str) -> Attributes {
        let mut a = Attributes::default();
        if !mode.is_empty() {
            a.strings.insert("mode".into(), mode.to_string());
        }
        a
    }

    // ── pad_params_from_node ────────────────────────────────────────────────

    #[test]
    fn inswapper_style_reflect_pad_3_is_claimed() {
        // Real node: Pad_39, mode=reflect, pads=[0,0,3,3,0,0,3,3] (no axes
        // input -> full-rank pads).
        let attrs = attrs_mode("reflect");
        let pads = [0, 0, 3, 3, 0, 0, 3, 3];
        let params = pad_params_from_node(&attrs, &[1, 3, 128, 128], &pads, None, 0.0)
            .expect("must be claimable");
        assert_eq!(params.mode, PadMode::Reflect);
        assert_eq!(params.pad_h, (3, 3));
        assert_eq!(params.pad_w, (3, 3));
    }

    #[test]
    fn inswapper_style_reflect_pad_1_is_claimed() {
        // Real node: Pad_61 (and eleven siblings), pads=[0,0,1,1,0,0,1,1].
        let attrs = attrs_mode("reflect");
        let pads = [0, 0, 1, 1, 0, 0, 1, 1];
        let params = pad_params_from_node(&attrs, &[1, 64, 64, 64], &pads, None, 0.0)
            .expect("must be claimable");
        assert_eq!(params.pad_h, (1, 1));
        assert_eq!(params.pad_w, (1, 1));
    }

    #[test]
    fn default_mode_is_constant() {
        let attrs = attrs_mode("");
        let pads = [0, 0, 1, 1, 0, 0, 1, 1];
        let params = pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 2.5)
            .expect("must be claimable");
        assert_eq!(params.mode, PadMode::Constant);
        assert_eq!(params.constant_value, 2.5);
    }

    #[test]
    fn edge_mode_declines() {
        let attrs = attrs_mode("edge");
        let pads = [0, 0, 1, 1, 0, 0, 1, 1];
        assert!(pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 0.0).is_none());
    }

    #[test]
    fn wrap_mode_declines() {
        let attrs = attrs_mode("wrap");
        let pads = [0, 0, 1, 1, 0, 0, 1, 1];
        assert!(pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 0.0).is_none());
    }

    #[test]
    fn padding_batch_or_channels_declines() {
        let attrs = attrs_mode("constant");
        let pads = [0, 1, 1, 1, 0, 1, 1, 1]; // begin[N]=1 as well
        assert!(pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 0.0).is_none());
    }

    #[test]
    fn axes_2_3_naming_only_the_spatial_dims_is_accepted() {
        let attrs = attrs_mode("constant");
        let pads = [2, 3, 2, 3]; // begin_h, begin_w, end_h, end_w
        let axes = [2_i64, 3];
        let params = pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, Some(&axes), 0.0)
            .expect("spatial-only axes must be claimable");
        assert_eq!(params.pad_h, (2, 2));
        assert_eq!(params.pad_w, (3, 3));
    }

    #[test]
    fn negative_pads_crop_and_are_still_claimed() {
        let attrs = attrs_mode("constant");
        let pads = [0, 0, -2, -2, 0, 0, -2, -2];
        let params = pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 0.0)
            .expect("a crop must still be claimable");
        assert_eq!(params.pad_h, (-2, -2));
    }

    #[test]
    fn a_crop_larger_than_the_axis_declines() {
        let attrs = attrs_mode("constant");
        let pads = [0, 0, -10, -10, 0, 0, -10, -10];
        assert!(pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 0.0).is_none());
    }

    #[test]
    fn pad_output_shape_matches_pads() {
        let attrs = attrs_mode("reflect");
        let pads = [0, 0, 3, 3, 0, 0, 3, 3];
        let params = pad_params_from_node(&attrs, &[1, 3, 128, 128], &pads, None, 0.0)
            .expect("must be claimable");
        assert_eq!(
            pad_output_shape(&[1, 3, 128, 128], &params),
            Some([1, 3, 134, 134])
        );
        assert_eq!(pad_output_shape(&[1, 3, 128], &params), None);
    }

    #[test]
    fn reflect_declines_on_a_1_pixel_axis() {
        let attrs = attrs_mode("reflect");
        let pads = [0, 0, 1, 1, 0, 0, 1, 1];
        assert!(pad_params_from_node(&attrs, &[1, 3, 1, 8], &pads, None, 0.0).is_none());
    }

    #[test]
    fn non_4d_input_declines() {
        let attrs = attrs_mode("reflect");
        let pads = [1, 1, 1, 1];
        assert!(pad_params_from_node(&attrs, &[3, 8, 8], &pads, None, 0.0).is_none());
    }

    #[test]
    fn mismatched_pads_length_declines() {
        let attrs = attrs_mode("constant");
        let pads = [1, 1, 1]; // needs 8 for a rank-4 default-axes pad
        assert!(pad_params_from_node(&attrs, &[1, 3, 8, 8], &pads, None, 0.0).is_none());
    }

    // ── reflect index formula, checked against `oxionnx-ops`'s own doc'd
    //    example arithmetic (independently, without invoking a GPU) ─────────

    /// Host-side re-implementation of the exact formula
    /// [`reflect_fold`]'s PTX computes, used only to cross-check the PTX by
    /// hand for a few concrete cases -- the GPU-side numeric agreement itself
    /// is checked by [`crate::reference::ref_pad`] under
    /// `OXIONNX_CUDA_VERIFY=1` on real hardware.
    fn reflect_index(coord: i64, dim: i64) -> i64 {
        let period = 2 * (dim - 1);
        let mut c = coord.rem_euclid(period);
        if c >= dim {
            c = period - c;
        }
        c
    }

    // ── PTX generation (no GPU required — this is pure text generation and
    //    validation, exactly like `oxicuda_dnn::pool::max_pool.rs`'s own
    //    `*_ptx_generates_f32` tests) ───────────────────────────────────────

    #[test]
    fn reflect_ptx_generates_and_validates() {
        let ptx = generate_pad_reflect_ptx(SmVersion::Sm86).expect("must generate");
        assert!(ptx.contains(PAD_REFLECT_KERNEL_NAME));
        assert!(ptx.contains("rem.s32"));
        assert!(ptx.contains("selp.s32"));
        let report = validate_ptx(&ptx);
        assert!(
            report.is_ok(),
            "generated reflect-pad PTX failed validation: {report:?}"
        );
    }

    #[test]
    fn constant_ptx_generates_and_validates() {
        let ptx = generate_pad_constant_ptx(SmVersion::Sm86).expect("must generate");
        assert!(ptx.contains(PAD_CONSTANT_KERNEL_NAME));
        assert!(ptx.contains("fill_value"));
        let report = validate_ptx(&ptx);
        assert!(
            report.is_ok(),
            "generated constant-pad PTX failed validation: {report:?}"
        );
    }

    #[test]
    fn reflect_index_matches_hand_worked_examples() {
        // dim=5 (period=8): indices ..., -1, 0, 1, 2, 3, 4, 5, 6, ...
        // reflect:          ...,  1, 0, 1, 2, 3, 4, 3, 2, ...
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(4, 5), 4);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(-5, 5), 3);
    }
}
