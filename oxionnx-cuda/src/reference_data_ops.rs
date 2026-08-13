//! CPU oracles for this wave's data-movement ops (`MaxPool`, `AveragePool`,
//! `Resize`, `Pad`, `Slice`, `Concat`).
//!
//! Kept in a file of its own — included into [`mod@crate::reference`]
//! verbatim via `#[path]` — rather than appended to `reference.rs` itself,
//! purely to avoid growing an already-large file further (this workspace's
//! own "single file under 2000 lines" refactor policy). Mirrors `lib.rs`'s
//! own `#[path = "dispatch_tests.rs"] mod dispatch_tests;` and `conv.rs`'s
//! `#[path = "conv_tests.rs"] mod tests;` — the established way a companion
//! file joins a module in this crate without becoming its own top-level
//! module.  Every function here is reachable as `crate::reference::ref_*`,
//! exactly like every oracle `reference.rs` defines directly.
//!
//! # Independent of `oxionnx-ops`, and of the kernels under test
//!
//! Same rule as every oracle in `reference.rs` itself (see that module's own
//! "why this does not depend on `oxionnx-ops`"): every formula below is
//! reimplemented from the ONNX specification (or, where the dispatch module
//! itself narrows the spec to what a specific `oxicuda_dnn`/hand-written PTX
//! kernel computes — see [`crate::pool`]/[`crate::resize`]'s own module
//! docs — from that narrowed formula), never called from the CPU operator
//! library and never shared with the CUDA kernel/PTX it grades.
//!
//! # Deliberately serial
//!
//! Unlike `ref_conv`/`ref_matmul`/`ref_reduce`/`ref_softmax`, none of the
//! functions below has an inner accumulation loop of consequence: the
//! costliest is [`ref_pool`]'s `kernel_h * kernel_w` window (`4` terms for
//! every real pooling node in this workspace), and the rest are pure
//! `O(output size)` gathers/copies. None of them are the "349 GFLOP of
//! scalar work per frame" pathology `rayon` was added to `reference.rs` to
//! fix (see that module's own "Parallelism" section), so none of them pull
//! `rayon` in at all — a plain loop over even InSwapper's largest pad output
//! (a few million elements) finishes in low tens of milliseconds, nowhere
//! near where a progress reporter or a parallel split would earn its keep.

use crate::concat::ConcatParams;
use crate::pad::{PadMode, PadParams};
use crate::pool::{PoolKind, PoolParams};
use crate::resize::{ResizeMode, ResizeParams};
use crate::slice::SliceParams;

// ── MaxPool / AveragePool ───────────────────────────────────────────────────

/// Naive NCHW `MaxPool`/`AveragePool`, matching exactly what
/// `oxicuda_dnn::pool::{max_pool2d, avg_pool2d}` compute: floor-mode output
/// extents (`(in + pad_begin + pad_end - kernel) / stride + 1`), symmetric
/// padding, `f64` accumulation for the average.
///
/// `params.ceil_mode` is not consulted here — by the time a dispatch reaches
/// this oracle, [`crate::pool::pool_params_from_attrs`]'s caller has already
/// confirmed (in [`crate::pool`]'s private `problem_from_params`) that the
/// floor- and ceil-mode output extents for this exact input agree, so the
/// kernel's floor-mode formula and the node's declared output shape are the
/// same question either way.
///
/// # Panics
/// Never — returns `None` for a shape this oracle cannot resolve rather than
/// indexing out of bounds.
#[must_use]
pub fn ref_pool(
    data: &[f32],
    in_shape: &[usize],
    kind: PoolKind,
    params: &PoolParams,
) -> Option<Vec<f32>> {
    if in_shape.len() != 4 {
        return None;
    }
    let n = in_shape[0];
    let c = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let [kh, kw] = params.kernel;
    let [sh, sw] = params.strides;
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;
    if sh == 0 || sw == 0 || kh == 0 || kw == 0 {
        return None;
    }
    let padded_h = in_h.checked_add(pad_top)?.checked_add(pad_bottom)?;
    let padded_w = in_w.checked_add(pad_left)?.checked_add(pad_right)?;
    if padded_h < kh || padded_w < kw {
        return None;
    }
    let out_h = (padded_h - kh) / sh + 1;
    let out_w = (padded_w - kw) / sw + 1;
    let needed = n.checked_mul(c)?.checked_mul(in_h)?.checked_mul(in_w)?;
    if data.len() < needed {
        return None;
    }

    let mut out = vec![0.0_f32; n * c * out_h * out_w];
    for ni in 0..n {
        for ci in 0..c {
            let plane = (ni * c + ci) * in_h * in_w;
            let out_plane = (ni * c + ci) * out_h * out_w;
            for oh in 0..out_h {
                let h_start = (oh * sh) as isize - pad_top as isize;
                for ow in 0..out_w {
                    let w_start = (ow * sw) as isize - pad_left as isize;
                    let mut max_val = f64::NEG_INFINITY;
                    let mut sum = 0.0_f64;
                    let mut count = 0_usize;
                    for ky in 0..kh {
                        let ih = h_start + ky as isize;
                        if ih < 0 || ih as usize >= in_h {
                            continue;
                        }
                        for kx in 0..kw {
                            let iw = w_start + kx as isize;
                            if iw < 0 || iw as usize >= in_w {
                                continue;
                            }
                            let v = f64::from(data[plane + ih as usize * in_w + iw as usize]);
                            max_val = max_val.max(v);
                            sum += v;
                            count += 1;
                        }
                    }
                    let value = match kind {
                        PoolKind::Max => max_val,
                        PoolKind::Avg => {
                            let divisor = if params.count_include_pad {
                                (kh * kw) as f64
                            } else {
                                count as f64
                            };
                            if divisor > 0.0 {
                                sum / divisor
                            } else {
                                0.0
                            }
                        }
                    };
                    out[out_plane + oh * out_w + ow] = value as f32;
                }
            }
        }
    }
    Some(out)
}

// ── Resize ───────────────────────────────────────────────────────────────────

/// One nearest-neighbour output row's source height index:
/// `floor(oh * in_h / out_h)`, computed as an **exact** unsigned integer
/// division — bit-for-bit the same formula
/// `oxicuda_dnn::resize::resize_nearest`'s PTX kernel evaluates with
/// `div.u32`, so this half of [`ref_resize`] agrees with the GPU exactly,
/// not merely within tolerance.
#[must_use]
fn nearest_index(o: usize, in_dim: usize, out_dim: usize) -> usize {
    (o * in_dim) / out_dim
}

/// The bilinear source coordinate for one output index, matching
/// `oxicuda_dnn::resize::resize_bilinear`'s two coordinate-transform
/// formulas (`align_corners` true/false) — computed in `f64` here rather
/// than replaying the kernel's `f32` instruction sequence, since the two are
/// only required to agree within [`crate::reference`]'s tolerance, not
/// bit-for-bit (see that module's own "not the ONNX spec in the abstract").
#[must_use]
fn bilinear_coord(o: usize, in_dim: usize, out_dim: usize, align_corners: bool) -> f64 {
    if align_corners {
        if out_dim <= 1 {
            0.0
        } else {
            o as f64 * (in_dim as f64 - 1.0) / (out_dim as f64 - 1.0)
        }
    } else {
        (o as f64 + 0.5) * in_dim as f64 / out_dim as f64 - 0.5
    }
}

/// Naive NCHW `Resize`, dispatching on [`ResizeParams::mode`] to the nearest
/// or bilinear formula. `N`/`C` pass through unchanged (`params` never
/// describes anything else — see [`crate::resize`]'s module docs).
///
/// # Panics
/// Never — returns `None` for a shape this oracle cannot resolve.
#[must_use]
pub fn ref_resize(data: &[f32], in_shape: &[usize], params: &ResizeParams) -> Option<Vec<f32>> {
    if in_shape.len() != 4 {
        return None;
    }
    let n = in_shape[0];
    let c = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let out_h = params.out_h;
    let out_w = params.out_w;
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return None;
    }
    let needed = n.checked_mul(c)?.checked_mul(in_h)?.checked_mul(in_w)?;
    if data.len() < needed {
        return None;
    }

    let mut out = vec![0.0_f32; n * c * out_h * out_w];
    for ni in 0..n {
        for ci in 0..c {
            let plane = (ni * c + ci) * in_h * in_w;
            let out_plane = (ni * c + ci) * out_h * out_w;
            match params.mode {
                ResizeMode::Nearest => {
                    for oh in 0..out_h {
                        let ih = nearest_index(oh, in_h, out_h);
                        for ow in 0..out_w {
                            let iw = nearest_index(ow, in_w, out_w);
                            out[out_plane + oh * out_w + ow] = data[plane + ih * in_w + iw];
                        }
                    }
                }
                ResizeMode::Bilinear { align_corners } => {
                    for oh in 0..out_h {
                        let src_h = bilinear_coord(oh, in_h, out_h, align_corners).max(0.0);
                        let h0 = (src_h.floor() as usize).min(in_h - 1);
                        let fh = src_h - src_h.floor();
                        let h1 = (h0 + 1).min(in_h - 1);
                        for ow in 0..out_w {
                            let src_w = bilinear_coord(ow, in_w, out_w, align_corners).max(0.0);
                            let w0 = (src_w.floor() as usize).min(in_w - 1);
                            let fw = src_w - src_w.floor();
                            let w1 = (w0 + 1).min(in_w - 1);

                            let v00 = f64::from(data[plane + h0 * in_w + w0]);
                            let v01 = f64::from(data[plane + h0 * in_w + w1]);
                            let v10 = f64::from(data[plane + h1 * in_w + w0]);
                            let v11 = f64::from(data[plane + h1 * in_w + w1]);
                            let value = (1.0 - fh) * (1.0 - fw) * v00
                                + (1.0 - fh) * fw * v01
                                + fh * (1.0 - fw) * v10
                                + fh * fw * v11;
                            out[out_plane + oh * out_w + ow] = value as f32;
                        }
                    }
                }
            }
        }
    }
    Some(out)
}

// ── Pad ──────────────────────────────────────────────────────────────────────

/// Folds `coord` into `[0, dim)` by reflection with period `2*(dim-1)`.
/// `dim > 1` is a precondition ([`crate::pad::pad_params_from_node`] already
/// declined `PadMode::Reflect` on a `<= 1`-sized axis before this can run).
/// Mirrors `oxionnx-ops::shape::sequence::pad_axes`'s `"reflect"` arm and
/// [`crate::pad::reflect_fold`]'s PTX — three independent expressions of the
/// same formula, none calling either of the other two.
#[must_use]
fn reflect_index(coord: i64, dim: i64) -> i64 {
    let period = 2 * (dim - 1);
    let mut c = coord.rem_euclid(period);
    if c >= dim {
        c = period - c;
    }
    c
}

/// Naive NCHW `Pad` (`reflect`/`constant`, spatial-only), matching
/// [`crate::pad::cuda_pad_bound`]'s claimed scope exactly.
///
/// # Panics
/// Never — returns `None` for a shape this oracle cannot resolve.
#[must_use]
pub fn ref_pad(data: &[f32], in_shape: &[usize], params: &PadParams) -> Option<Vec<f32>> {
    if in_shape.len() != 4 {
        return None;
    }
    let n = in_shape[0];
    let c = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let out_h_i64 = in_h as i64 + i64::from(params.pad_h.0) + i64::from(params.pad_h.1);
    let out_w_i64 = in_w as i64 + i64::from(params.pad_w.0) + i64::from(params.pad_w.1);
    if out_h_i64 < 0 || out_w_i64 < 0 {
        return None;
    }
    let out_h = out_h_i64 as usize;
    let out_w = out_w_i64 as usize;
    let needed = n.checked_mul(c)?.checked_mul(in_h)?.checked_mul(in_w)?;
    if data.len() < needed {
        return None;
    }

    let mut out = vec![params.constant_value; n * c * out_h * out_w];
    for ni in 0..n {
        for ci in 0..c {
            let plane = (ni * c + ci) * in_h * in_w;
            let out_plane = (ni * c + ci) * out_h * out_w;
            for oh in 0..out_h {
                let ih = oh as i64 - i64::from(params.pad_h.0);
                let ih_final = match params.mode {
                    PadMode::Reflect => Some(reflect_index(ih, in_h as i64)),
                    PadMode::Constant => (ih >= 0 && ih < in_h as i64).then_some(ih),
                };
                let Some(ih_final) = ih_final else { continue };
                for ow in 0..out_w {
                    let iw = ow as i64 - i64::from(params.pad_w.0);
                    let iw_final = match params.mode {
                        PadMode::Reflect => Some(reflect_index(iw, in_w as i64)),
                        PadMode::Constant => (iw >= 0 && iw < in_w as i64).then_some(iw),
                    };
                    let Some(iw_final) = iw_final else { continue };
                    out[out_plane + oh * out_w + ow] =
                        data[plane + ih_final as usize * in_w + iw_final as usize];
                }
            }
        }
    }
    Some(out)
}

// ── Slice ────────────────────────────────────────────────────────────────────

/// Row-major strides (in elements) for a rank-4 shape.
#[must_use]
fn row_major_strides(shape: [usize; 4]) -> [usize; 4] {
    let mut strides = [1_usize; 4];
    for d in (0..3).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Naive rank-4 `Slice`: for every output coordinate, `start[d] + o[d] *
/// step[d]` per axis, no bounds check (see [`crate::slice`]'s module docs
/// for why every mapped coordinate is provably in range).
///
/// # Panics
/// Never — returns `None` for a shape this oracle cannot resolve.
#[must_use]
pub fn ref_slice(data: &[f32], in_shape: &[usize], params: &SliceParams) -> Option<Vec<f32>> {
    if in_shape.len() != 4 {
        return None;
    }
    let in_shape4: [usize; 4] = [in_shape[0], in_shape[1], in_shape[2], in_shape[3]];
    let needed = in_shape4
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))?;
    if data.len() < needed {
        return None;
    }
    let in_stride = row_major_strides(in_shape4);
    let out_stride = row_major_strides(params.out_shape);
    let out_n: usize = params.out_shape.iter().product();

    let mut out = vec![0.0_f32; out_n];
    for (idx, out_val) in out.iter_mut().enumerate() {
        let mut rem = idx;
        let mut in_idx: i64 = 0;
        for d in 0..4 {
            let coord = rem / out_stride[d];
            rem %= out_stride[d];
            let in_coord = i64::from(params.start[d]) + coord as i64 * i64::from(params.step[d]);
            in_idx += in_coord * in_stride[d] as i64;
        }
        *out_val = data[in_idx as usize];
    }
    Some(out)
}

// ── Concat ───────────────────────────────────────────────────────────────────

/// Naive `Concat` along `params.axis`: `[outer, axis_len, inner]` blocks
/// copied per input, in node-input order.
///
/// # Panics
/// Never — returns `None` for shapes this oracle cannot resolve.
#[must_use]
pub fn ref_concat(inputs: &[&[f32]], params: &ConcatParams) -> Option<Vec<f32>> {
    if inputs.is_empty() || inputs.len() != params.segment_lens.len() {
        return None;
    }
    let ndim = params.out_shape.len();
    if params.axis >= ndim {
        return None;
    }
    let outer: usize = params.out_shape[..params.axis].iter().product();
    let inner: usize = params.out_shape[params.axis + 1..].iter().product();
    let out_axis_len = params.out_shape[params.axis];

    for (&input, &seg) in inputs.iter().zip(&params.segment_lens) {
        let needed = outer.checked_mul(seg)?.checked_mul(inner)?;
        if input.len() < needed {
            return None;
        }
    }

    let mut out = vec![0.0_f32; outer.checked_mul(out_axis_len)?.checked_mul(inner)?];
    let mut axis_offset = 0_usize;
    for (&input, &seg) in inputs.iter().zip(&params.segment_lens) {
        for o in 0..outer {
            let src_off = (o * seg) * inner;
            let dst_off = (o * out_axis_len + axis_offset) * inner;
            out[dst_off..dst_off + seg * inner]
                .copy_from_slice(&input[src_off..src_off + seg * inner]);
        }
        axis_offset += seg;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ref_pool ─────────────────────────────────────────────────────────────

    fn unit_pool_params() -> PoolParams {
        PoolParams {
            kernel: [2, 2],
            strides: [2, 2],
            pads: [0, 0, 0, 0],
            ceil_mode: false,
            count_include_pad: false,
        }
    }

    #[test]
    fn max_pool_hand_verified_4x4() {
        // 4x4 input, 2x2 kernel/stride -> 2x2 output.
        // [[ 1, 2, 3, 4],
        //  [ 5, 6, 7, 8],
        //  [ 9,10,11,12],
        //  [13,14,15,16]]
        // window (0,0): {1,2,5,6} -> max 6; window (0,1): {3,4,7,8} -> max 8
        // window (1,0): {9,10,13,14} -> max 14; window (1,1): {11,12,15,16} -> max 16
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let out = ref_pool(&data, &[1, 1, 4, 4], PoolKind::Max, &unit_pool_params())
            .expect("must resolve");
        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn avg_pool_hand_verified_4x4() {
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let out = ref_pool(&data, &[1, 1, 4, 4], PoolKind::Avg, &unit_pool_params())
            .expect("must resolve");
        // (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
        assert_eq!(out, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn avg_pool_count_include_pad_divides_by_the_full_window() {
        let params = PoolParams {
            kernel: [2, 2],
            strides: [2, 2],
            pads: [0, 0, 1, 1],
            ceil_mode: false,
            count_include_pad: true,
        };
        // 2x2 input padded to 3x3 on the bottom/right with zeros, window
        // (0,0) is the full unpadded input: {1,2,3,4} -> mean 2.5.
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = ref_pool(&data, &[1, 1, 2, 2], PoolKind::Avg, &params).expect("must resolve");
        assert_eq!(out[0], 2.5);
    }

    // ── ref_resize ───────────────────────────────────────────────────────────

    #[test]
    fn nearest_asymmetric_floor_2x_upsample() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let params = ResizeParams {
            mode: ResizeMode::Nearest,
            out_h: 4,
            out_w: 4,
        };
        let out = ref_resize(&data, &[1, 1, 2, 2], &params).expect("must resolve");
        // Each source pixel repeats 2x2: row0 = [1,1,2,2], row1 = [1,1,2,2],
        // row2 = [3,3,4,4], row3 = [3,3,4,4].
        assert_eq!(
            out,
            vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
        );
    }

    #[test]
    fn bilinear_half_pixel_hand_verified_midpoint() {
        // 2x1 input [0, 10], resized to 4x1 with half_pixel: src = (o+0.5)*2/4 - 0.5.
        // o=0 -> src=0.0 -> exactly 0; o=3 -> src=1.25 -> clamp handling picks w1=1
        let data = vec![0.0_f32, 10.0];
        let params = ResizeParams {
            mode: ResizeMode::Bilinear {
                align_corners: false,
            },
            out_h: 1,
            out_w: 4,
        };
        let out = ref_resize(&data, &[1, 1, 1, 2], &params).expect("must resolve");
        assert_eq!(out[0], 0.0);
        // o=1: src=(1.5*0.5)-0.5=0.25 -> 0.25*10=2.5
        assert!((out[1] - 2.5).abs() < 1e-4);
    }

    #[test]
    fn bilinear_align_corners_matches_exact_endpoints() {
        let data = vec![0.0_f32, 10.0];
        let params = ResizeParams {
            mode: ResizeMode::Bilinear {
                align_corners: true,
            },
            out_h: 1,
            out_w: 3,
        };
        let out = ref_resize(&data, &[1, 1, 1, 2], &params).expect("must resolve");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[2], 10.0);
        assert!((out[1] - 5.0).abs() < 1e-4);
    }

    // ── ref_pad ──────────────────────────────────────────────────────────────

    #[test]
    fn reflect_pad_hand_verified_w_only() {
        // Input [[1,2,3],[4,5,6]] (H=2, W=3; H>1 is a precondition
        // `crate::pad::pad_params_from_node` already enforces for
        // `PadMode::Reflect`, so this oracle is never asked to fold a
        // 1-or-0-sized axis in practice -- see `reflect_pad_declines_on_a_1_pixel_axis_upstream`),
        // reflect-padded 2 on the left of W only:
        // ow=0 -> iw=-2 -> reflect(-2,3)=2 -> col 2
        // ow=1 -> iw=-1 -> reflect(-1,3)=1 -> col 1
        // ow=2 -> iw=0  -> col 0
        // ow=3 -> iw=1  -> col 1
        // ow=4 -> iw=2  -> col 2
        // matching `crate::pad`'s own `reflect_index_matches_hand_worked_examples`.
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let params = PadParams {
            pad_h: (0, 0),
            pad_w: (2, 0),
            mode: PadMode::Reflect,
            constant_value: 0.0,
        };
        let out = ref_pad(&data, &[1, 1, 2, 3], &params).expect("must resolve");
        assert_eq!(out, vec![3.0, 2.0, 1.0, 2.0, 3.0, 6.0, 5.0, 4.0, 5.0, 6.0]);
    }

    /// Documents (rather than exercises — this crate's own dispatch-side
    /// decline is tested directly in `crate::pad`'s own test suite) why the
    /// test above uses `H=2`, not `H=1`: `ref_pad` has no bounds/precondition
    /// check of its own for a `<=1`-sized reflect axis (`reflect_index`
    /// divides by `2*(dim-1)`, which is `0` at `dim<=1`), exactly mirroring
    /// `crate::pad::reflect_fold`'s PTX, whose own precondition comment says
    /// the same thing. The oracle is never asked to violate it because
    /// `crate::pad::pad_params_from_node` already declined the node before
    /// any oracle call happens.
    #[test]
    fn reflect_pad_declines_on_a_1_pixel_axis_upstream() {
        let mut attrs = oxionnx_core::Attributes::default();
        attrs
            .strings
            .insert("mode".to_string(), "reflect".to_string());
        assert!(crate::pad::pad_params_from_node(
            &attrs,
            &[1, 1, 1, 3],
            &[0, 0, 1, 1, 0, 0, 1, 1],
            None,
            0.0,
        )
        .is_none());
    }

    #[test]
    fn constant_pad_fills_outside_the_input() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let params = PadParams {
            pad_h: (1, 0),
            pad_w: (0, 1),
            mode: PadMode::Constant,
            constant_value: -1.0,
        };
        let out = ref_pad(&data, &[1, 1, 2, 2], &params).expect("must resolve");
        // Output is 3x3: row0 all -1, row1 = [1,2,-1], row2 = [3,4,-1].
        assert_eq!(out, vec![-1.0, -1.0, -1.0, 1.0, 2.0, -1.0, 3.0, 4.0, -1.0]);
    }

    #[test]
    fn negative_pad_crops() {
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect(); // 4x4
        let params = PadParams {
            pad_h: (-1, -1),
            pad_w: (-1, -1),
            mode: PadMode::Constant,
            constant_value: 0.0,
        };
        let out = ref_pad(&data, &[1, 1, 4, 4], &params).expect("must resolve");
        // Crops to the inner 2x2: [[6,7],[10,11]].
        assert_eq!(out, vec![6.0, 7.0, 10.0, 11.0]);
    }

    // ── ref_slice ────────────────────────────────────────────────────────────

    #[test]
    fn slice_channel_half_matches_inswapper_pattern() {
        let data: Vec<f32> = (0..8).map(|v| v as f32).collect(); // [1,8,1,1]
        let params = SliceParams {
            start: [0, 4, 0, 0],
            step: [1, 1, 1, 1],
            out_shape: [1, 4, 1, 1],
        };
        let out = ref_slice(&data, &[1, 8, 1, 1], &params).expect("must resolve");
        assert_eq!(out, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn slice_reverse_step() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0]; // [1,4,1,1]
        let params = SliceParams {
            start: [0, 3, 0, 0],
            step: [1, -1, 1, 1],
            out_shape: [1, 4, 1, 1],
        };
        let out = ref_slice(&data, &[1, 4, 1, 1], &params).expect("must resolve");
        assert_eq!(out, vec![4.0, 3.0, 2.0, 1.0]);
    }

    // ── ref_concat ───────────────────────────────────────────────────────────

    #[test]
    fn concat_axis_0_of_shape_vectors() {
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32];
        let params = ConcatParams {
            axis: 0,
            out_shape: vec![3],
            segment_lens: vec![2, 1],
        };
        let out = ref_concat(&[&a, &b], &params).expect("must resolve");
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_axis_1_of_feature_maps() {
        // [1,1,2,2] and [1,1,2,2] concatenated on channel -> [1,2,2,2].
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let params = ConcatParams {
            axis: 1,
            out_shape: vec![1, 2, 2, 2],
            segment_lens: vec![1, 1],
        };
        let out = ref_concat(&[&a, &b], &params).expect("must resolve");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn concat_interleaves_outer_blocks_correctly() {
        // axis=0 is NOT the outermost concat case when outer>1: exercise
        // axis=1 with outer=2 (N=2) so each input contributes two separate
        // (non-contiguous-across-N) blocks.
        // a: [2,1,1,1] = [1,2], b: [2,2,1,1] = [10,11, 20,21]
        let a = [1.0_f32, 2.0];
        let b = [10.0_f32, 11.0, 20.0, 21.0];
        let params = ConcatParams {
            axis: 1,
            out_shape: vec![2, 3, 1, 1],
            segment_lens: vec![1, 2],
        };
        let out = ref_concat(&[&a, &b], &params).expect("must resolve");
        // n=0: [a0, b0, b1] = [1, 10, 11]; n=1: [a1, b2, b3] = [2, 20, 21]
        assert_eq!(out, vec![1.0, 10.0, 11.0, 2.0, 20.0, 21.0]);
    }
}
