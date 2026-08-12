//! Direct NCHW Conv2D — implicit GEMM in WGSL, with fused bias and activation.
//!
//! # Why this exists
//!
//! [c3] `compute.rs`'s [`gpu_conv2d_async`](crate::compute::gpu_conv2d_async)
//! was a *hybrid*: im2col on the CPU, GEMM on the GPU, bias and activation
//! back on the CPU. That shape is wrong for a convolution network in three
//! separate ways, all of them measured:
//!
//! * **The column matrix is `kH*kW` times the input.** InSwapper's twelve
//!   `Conv[1024, 1024, 3, 3]` layers at `34x34` see a `[1024, 32, 32]` input
//!   (4 MB) but a `[9216, 1024]` column matrix (37.7 MB) — and the hybrid
//!   uploaded that column matrix *per call*, so one frame pushed ~595 MB of
//!   im2col across the bus that the input alone would have covered in ~50 MB.
//! * **The im2col itself is CPU work** on the critical path, serialized ahead
//!   of every dispatch.
//! * **Bias and activation were separate CPU passes** over the whole output.
//!
//! The end-to-end result was a GPU conv measurably *slower* than the rayon CPU
//! operator it was meant to accelerate (0.33-0.58x on Metal).
//!
//! This kernel uploads the input and the weights **as they are**, and never
//! materialises a column matrix anywhere — the im2col index arithmetic happens
//! inside the shader, in registers, as the input tile is staged into workgroup
//! memory. Bias and activation are folded into the epilogue, so the output is
//! read back already finished.
//!
//! # The implicit-GEMM mapping
//!
//! For one batch element, a `group = 1` convolution is exactly the GEMM
//!
//! ```text
//!   C[M, N] = A[M, K] * B[K, N]
//!   M = C_out                       (output channels)
//!   N = OH * OW                     (output pixels, row-major)
//!   K = C_in * kH * kW              (the "reduction" axis)
//!   A = the weight tensor, verbatim: A[oc, (ic*kH + ky)*kW + kx] = w[oc][ic][ky][kx]
//!   B = the column matrix, never materialised:
//!       B[(ic*kH + ky)*kW + kx, oy*OW + ox] = in[n][ic][oy*sh + ky*dh - pt][ox*sw + kx*dw - pl]
//!   C = the output, verbatim: C[oc, oy*OW + ox] = out[n][oc][oy][ox]
//! ```
//!
//! So `A` and `C` need no index translation at all — only `B`'s gather does,
//! and that is what makes the kernel "implicit".
//!
//! # Why the K loop is a `(ky, kx, ic)` nest rather than a flat `k` loop
//!
//! The obvious implicit-GEMM writes one loop over `k` and recovers
//! `(ic, ky, kx)` by dividing. That costs **four integer divisions per staged
//! element**, and this kernel stages 8 elements per thread per K-tile against
//! 256 FMAs — divisions at that density are comparable in cost to the
//! arithmetic they feed, i.e. a tax on the whole kernel rather than a detail.
//!
//! Instead the K axis is walked as an explicit `ky -> kx -> ic-tile` nest.
//! `ky` and `kx` are then *loop variables*, uniform across the workgroup, so
//! the decomposition never happens: the weight column is `ic*kH*kW + ky*kW +
//! kx` (one multiply-add) and the input row/column offsets are `base + ky*dh`
//! / `base + kx*dw` (one multiply-add each, hoisted to the `ky` and `kx` loop
//! headers respectively). Only `ic` is tiled into shared memory, and only the
//! *output-pixel* decomposition `j -> (oy, ox)` still needs a division — which
//! is loop-invariant, so it is computed **once per thread** in the prologue
//! (8 divisions for the whole kernel) rather than once per staged element.
//!
//! # Tile shape
//!
//! The macro-tile geometry is the register-tiled GEMM's in
//! `context/functions.rs` (`TILED_MATMUL_SHADER`, 242-434 GFLOP/s natively on
//! M3): a `64x64` macro-tile per workgroup, `16x16` threads, a `4x4` register
//! tile per thread, and a 16-deep shared tile of each operand. That shape is
//! right for this workload for the same reason it is right there —
//! InSwapper's dominant layer is `M = 1024, N = 1024, K = 9216` and the
//! decoder's is `M = 256, N = 16384, K = 4608`, both large in every dimension,
//! so the 16x arithmetic intensity of a `4x4` register tile is what keeps the
//! kernel off the shared-memory bandwidth wall.
//!
//! The one deliberate departure is that both shared tiles are **`vec4`
//! blocked** — `array<array<vec4<f32>, 16>, 16>`, block `b` holding elements
//! `b*4 .. b*4+3`. A thread's `4x4` register tile spans exactly rows
//! `ty*4..ty*4+3` and columns `tx*4..tx*4+3`, so one block *is* one thread's
//! operand set: the innermost loop reads two `vec4`s per 16 FMAs where the
//! scalar-tiled form read eight `f32`s, and the cooperative load writes one
//! whole block per thread instead of four scattered scalars. Measured against
//! the scalar-tiled version of this same kernel on M3
//! (`examples/c3_conv2d_gflops.rs`): the `128x128` decoder layer went 69-76 ms
//! to 55-56 ms (~507 -> ~692 GFLOP/s). The `32x32` bottleneck layer, which is
//! bound elsewhere, moved 40.0 -> ~35 ms.
//!
//! The 16 accumulators are individually-named scalars, not `array<f32, 16>` —
//! see `TILED_MATMUL_SHADER`'s doc comment for the measurement behind that
//! (naga's WGSL->MSL lowering did not promote the small loop-indexed arrays to
//! registers, making an array version *slower* than a naive kernel). The
//! `vec4` tiles above are the sanctioned exception: they are fixed-width
//! vectors with static component access, not dynamically-indexed arrays.
//!
//! # What declines
//!
//! `group > 1` and any shape the device cannot bind or dispatch return `None`,
//! which sends the caller to the hybrid path that is still in `compute.rs`.
//! Grouped convolution is not a gap in the mapping — it is a deliberate
//! omission: none of the three models this kernel exists for uses it, and a
//! depthwise conv (`C_out/group = 1`) against a `64`-row macro-tile would
//! waste 63/64 of the M tile, so it belongs on a differently-shaped kernel,
//! not this one. Dilation *is* supported (it is one multiply in the `ky`/`kx`
//! loop headers).
//!
//! Unlike the element-wise kernels, this file applies **no minimum-size
//! threshold** — see [`kernel_support`](super::kernel_support)'s module docs
//! for why a CPU/GPU placement heuristic belongs at the call site. The
//! threshold that used to gate the hybrid conv still gates both paths, in
//! `compute.rs`.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::{GpuContext, WeightFormat, WeightKeys};
use crate::device_guard::{block_on_gpu, checked_storage_bytes, finish_output_async};
use crate::device_guard::{ErrorScope, GpuLimits};
use oxionnx_core::Tensor;

use super::kernel_support::{
    bgl_ro, bgl_rw, bgl_uniform, build_pipeline, insert_for_current_device,
};

/// Output rows (channels) and columns (pixels) owned by one workgroup.
///
/// Mirrors `MACRO_M` / `MACRO_N` in the WGSL below; the host uses it to size
/// the dispatch grid, so the two must move together.
const MACRO_TILE: u32 = 64;

/// An activation folded into the convolution's epilogue.
///
/// The variants mirror what the optimizer's Conv fusion passes actually emit
/// (`src/optimizer/fusion/conv/{relu,relu6}.rs` produce `"relu"` and
/// `"clip"`), plus the leaky variant the task list calls for. `None` is the
/// identity and is what the plain, un-fused entry point uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvActivation {
    /// No activation — the convolution's raw output (plus bias).
    None,
    /// `max(x, 0)`.
    Relu,
    /// `x` for `x >= 0`, `alpha * x` otherwise.
    LeakyRelu(f32),
    /// `min(max(x, min), max)` — the fused form of `Clip`, including ReLU6.
    ///
    /// Written as a `max` then a `min` rather than a `clamp` in **both** the
    /// WGSL and [`Self::apply_host`], and that is not a style choice. An
    /// inverted range (`min > max`) is expressible here — `Clip`'s bounds come
    /// from a model file — and neither `clamp` handles it: `f32::clamp`
    /// *panics*, and WGSL's `clamp` is undefined when `low > high`. The
    /// `max`-then-`min` form is total on both sides and agrees on both sides
    /// (an inverted range saturates to `max`), so no caller can panic and the
    /// GPU and the hybrid fallback cannot diverge.
    Clip {
        /// Lower clamp bound.
        min: f32,
        /// Upper clamp bound.
        max: f32,
    },
}

impl ConvActivation {
    /// The `act_mode` discriminant the WGSL `activate()` switch reads.
    #[inline]
    fn mode(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Relu => 1,
            Self::LeakyRelu(_) => 2,
            Self::Clip { .. } => 3,
        }
    }

    /// `(alpha, min, max)` as the uniform block carries them. Unused slots are
    /// zero rather than `NaN` so a shader that reads them anyway is harmless.
    #[inline]
    fn scalars(self) -> (f32, f32, f32) {
        match self {
            Self::None | Self::Relu => (0.0, 0.0, 0.0),
            Self::LeakyRelu(alpha) => (alpha, 0.0, 0.0),
            Self::Clip { min, max } => (0.0, min, max),
        }
    }

    /// True when every scalar this variant carries is finite.
    ///
    /// A `NaN` slope or clamp bound would make the GPU and CPU paths disagree
    /// in ways no tolerance can bridge, so such a call declines instead.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        match self {
            Self::None | Self::Relu => true,
            Self::LeakyRelu(alpha) => alpha.is_finite(),
            Self::Clip { min, max } => min.is_finite() && max.is_finite(),
        }
    }

    /// Apply this activation on the host.
    ///
    /// Used by the fallback path in `compute.rs`: when the direct kernel
    /// declines and the hybrid im2col path runs instead, the caller still owes
    /// its contract an activated result. Written to match the WGSL
    /// `activate()` below expression-for-expression.
    pub fn apply_host(self, data: &mut [f32]) {
        match self {
            Self::None => {}
            Self::Relu => {
                for v in data.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            Self::LeakyRelu(alpha) => {
                for v in data.iter_mut() {
                    if *v < 0.0 {
                        *v *= alpha;
                    }
                }
            }
            Self::Clip { min, max } => {
                // Not `v.clamp(min, max)`: that panics on an inverted range.
                // See the `Clip` variant's docs.
                for v in data.iter_mut() {
                    *v = v.max(min).min(max);
                }
            }
        }
    }
}

/// Uniform block — must match the WGSL `Params` struct field for field.
///
/// Every member is a 4-byte scalar, so WGSL's uniform layout rules put them at
/// their natural offsets and `#[repr(C)]` agrees. The trailing pad keeps the
/// whole block a multiple of 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    dil_h: u32,
    dil_w: u32,
    pad_t: i32,
    pad_l: i32,
    /// `oh * ow` — the GEMM's `N`.
    n_out: u32,
    /// `kh * kw` — the weight's per-input-channel stride.
    k_stride: u32,
    /// `c_in * kh * kw` — the weight's per-output-channel stride (the GEMM's `K`).
    k_total: u32,
    /// `h * w` — the input's per-channel stride.
    hw: u32,
    has_bias: u32,
    act_mode: u32,
    act_alpha: f32,
    act_min: f32,
    act_max: f32,
    _pad0: u32,
}

/// The kernel, in `f32`.
///
/// [w2-f16] Deliberately **unchanged** by the half-precision work: the `f16`
/// variant is derived from this text at runtime by
/// [`super::f16_variant::conv2d_f16`], so this constant stays byte-for-byte
/// what it was and a toggle-off dispatch compiles the very same shader it
/// always did. `pub(super)` only so that module can read it.
pub(super) const CONV2D_SHADER: &str = r#"
struct Params {
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    dil_h: u32,
    dil_w: u32,
    pad_t: i32,
    pad_l: i32,
    n_out: u32,
    k_stride: u32,
    k_total: u32,
    hw: u32,
    has_bias: u32,
    act_mode: u32,
    act_alpha: f32,
    act_min: f32,
    act_max: f32,
    _pad0: u32,
}

// Output channels / output pixels owned by one workgroup.
const MACRO_M: u32 = 64u;
const MACRO_N: u32 = 64u;
// Input channels staged into shared memory per inner iteration.
const CTILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> wgt: array<f32>;
@group(0) @binding(2) var<storage, read> bias_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// c-major and `vec4`-blocked: `tile[c][b]` holds the four consecutive output
// channels `b*4 .. b*4+3` (weights) or output pixels (inputs) for channel `c`.
// 16 blocks x 4 = the 64-wide macro tile.
//
// The `vec4` blocking is the point, not decoration: a thread's 4x4 register
// tile spans exactly rows `ty*4..ty*4+3` and columns `tx*4..tx*4+3`, so one
// block *is* one thread's operand set and the innermost loop reads **two**
// values from workgroup memory per 16 FMAs instead of eight. On a kernel whose
// inner loop is 8 shared loads to 16 FMAs, that is the difference between 24
// and 18 instructions per 16 multiply-adds.
var<workgroup> tile_w: array<array<vec4<f32>, 16>, 16>;
var<workgroup> tile_x: array<array<vec4<f32>, 16>, 16>;

fn activate(v: f32) -> f32 {
    if (params.act_mode == 1u) {
        return max(v, 0.0);
    }
    if (params.act_mode == 2u) {
        return select(params.act_alpha * v, v, v >= 0.0);
    }
    if (params.act_mode == 3u) {
        // `max` then `min`, not `clamp`: WGSL's clamp is undefined when
        // low > high, and a model file can carry an inverted Clip range.
        return min(max(v, params.act_min), params.act_max);
    }
    return v;
}

// Zero-padding load of one weight element. `channel_ok` is the input-channel
// bound (the last c-tile is ragged whenever `c_in % 16 != 0`) and is checked
// here as well as on the input side: without it the tail of the last tile
// would read past the weight tensor.
fn load_weight(oc: u32, a_col: u32, channel_ok: bool) -> f32 {
    if (channel_ok && oc < params.c_out) {
        return wgt[oc * params.k_total + a_col];
    }
    return 0.0;
}

// Zero-padding gather of one input element. `ok` folds together the output
// column bound, the input-channel bound, and the spatial in-bounds test that
// implements zero padding; `spatial` is only ever indexed when all three hold,
// so it is never negative here.
fn load_input(plane: u32, spatial: i32, ok: bool) -> f32 {
    if (ok) {
        return inp[plane + u32(spatial)];
    }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn conv2d_implicit(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m_total = params.c_out;
    let n_total = params.n_out;

    // `wid` is uniform across the workgroup, so this early exit is not a
    // divergent return ahead of the workgroupBarrier() calls below. The host
    // dispatches exactly ceil(M/64) x ceil(N/64) x batch, so this only fires
    // if a future call site over-provisions.
    let tiles_n = (n_total + MACRO_N - 1u) / MACRO_N;
    let tiles_m = (m_total + MACRO_M - 1u) / MACRO_M;
    if (wid.x >= tiles_n || wid.y >= tiles_m) {
        return;
    }

    let tile_row = wid.y * MACRO_M;
    let tile_col = wid.x * MACRO_N;
    let tx = lid.x;
    let ty = lid.y;
    let batch = wid.z;

    let h_i = i32(params.h);
    let w_i = i32(params.w);
    let sh = i32(params.stride_h);
    let sw = i32(params.stride_w);

    // --- prologue: this thread's four input-tile columns, decomposed once ---
    //
    // The cooperative load below fills the whole `vec4` block tile_x[ty][tx],
    // i.e. output pixels `tile_col + tx*4 .. +3`, and it fills the same four
    // for every (ky, kx, ic-tile) iteration. Their (oy, ox) decomposition —
    // the only division left in the kernel — is therefore hoisted here. They
    // are decomposed independently rather than incremented, because four
    // consecutive `j` can straddle an output row boundary.
    let j0 = tile_col + tx * 4u;
    let j1 = j0 + 1u;
    let j2 = j0 + 2u;
    let j3 = j0 + 3u;
    let col_ok0 = j0 < n_total;
    let col_ok1 = j1 < n_total;
    let col_ok2 = j2 < n_total;
    let col_ok3 = j3 < n_total;
    // Top-left of each column's receptive window, before the kernel offset.
    let base_y0 = i32(j0 / params.ow) * sh - params.pad_t;
    let base_x0 = i32(j0 % params.ow) * sw - params.pad_l;
    let base_y1 = i32(j1 / params.ow) * sh - params.pad_t;
    let base_x1 = i32(j1 % params.ow) * sw - params.pad_l;
    let base_y2 = i32(j2 / params.ow) * sh - params.pad_t;
    let base_x2 = i32(j2 % params.ow) * sw - params.pad_l;
    let base_y3 = i32(j3 / params.ow) * sh - params.pad_t;
    let base_x3 = i32(j3 % params.ow) * sw - params.pad_l;

    // 4x4 register tile. Named scalars on purpose — see the module docs.
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc02: f32 = 0.0;
    var acc03: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;
    var acc12: f32 = 0.0;
    var acc13: f32 = 0.0;
    var acc20: f32 = 0.0;
    var acc21: f32 = 0.0;
    var acc22: f32 = 0.0;
    var acc23: f32 = 0.0;
    var acc30: f32 = 0.0;
    var acc31: f32 = 0.0;
    var acc32: f32 = 0.0;
    var acc33: f32 = 0.0;

    let num_c_tiles = (params.c_in + CTILE - 1u) / CTILE;

    // Every loop bound below is a uniform value, so all invocations execute the
    // same number of workgroupBarrier() calls.
    for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
        let dy = i32(ky * params.dil_h);
        let y0 = base_y0 + dy;
        let y1 = base_y1 + dy;
        let y2 = base_y2 + dy;
        let y3 = base_y3 + dy;
        let row_ok0 = y0 >= 0 && y0 < h_i;
        let row_ok1 = y1 >= 0 && y1 < h_i;
        let row_ok2 = y2 >= 0 && y2 < h_i;
        let row_ok3 = y3 >= 0 && y3 < h_i;
        let row_off0 = y0 * w_i;
        let row_off1 = y1 * w_i;
        let row_off2 = y2 * w_i;
        let row_off3 = y3 * w_i;
        let ky_col = ky * params.kw;

        for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
            let dx = i32(kx * params.dil_w);
            let x0 = base_x0 + dx;
            let x1 = base_x1 + dx;
            let x2 = base_x2 + dx;
            let x3 = base_x3 + dx;
            let in0 = col_ok0 && row_ok0 && x0 >= 0 && x0 < w_i;
            let in1 = col_ok1 && row_ok1 && x1 >= 0 && x1 < w_i;
            let in2 = col_ok2 && row_ok2 && x2 >= 0 && x2 < w_i;
            let in3 = col_ok3 && row_ok3 && x3 >= 0 && x3 < w_i;
            let sp0 = row_off0 + x0;
            let sp1 = row_off1 + x1;
            let sp2 = row_off2 + x2;
            let sp3 = row_off3 + x3;
            let k_off = ky_col + kx;

            for (var ct: u32 = 0u; ct < num_c_tiles; ct = ct + 1u) {
                // CTILE == the workgroup's Y extent, so each thread row owns
                // exactly one input channel of the tile: no loop needed.
                let ic = ct * CTILE + ty;
                let ic_ok = ic < params.c_in;

                // One `vec4` block per thread per tile: 256 threads fill the
                // 16x16 blocks of each tile exactly once, no inner loop.
                let a_col = ic * params.k_stride + k_off;
                let m0 = tile_row + tx * 4u;
                tile_w[ty][tx] = vec4<f32>(
                    load_weight(m0, a_col, ic_ok),
                    load_weight(m0 + 1u, a_col, ic_ok),
                    load_weight(m0 + 2u, a_col, ic_ok),
                    load_weight(m0 + 3u, a_col, ic_ok),
                );

                let plane = (batch * params.c_in + ic) * params.hw;
                tile_x[ty][tx] = vec4<f32>(
                    load_input(plane, sp0, in0 && ic_ok),
                    load_input(plane, sp1, in1 && ic_ok),
                    load_input(plane, sp2, in2 && ic_ok),
                    load_input(plane, sp3, in3 && ic_ok),
                );

                workgroupBarrier();

                for (var cc: u32 = 0u; cc < CTILE; cc = cc + 1u) {
                    // `tile_w[cc][ty]` *is* this thread's four output channels
                    // and `tile_x[cc][tx]` its four output pixels — the block
                    // index and the register tile were chosen to coincide.
                    let a4 = tile_w[cc][ty];
                    let b4 = tile_x[cc][tx];
                    acc00 = acc00 + a4.x * b4.x;
                    acc01 = acc01 + a4.x * b4.y;
                    acc02 = acc02 + a4.x * b4.z;
                    acc03 = acc03 + a4.x * b4.w;
                    acc10 = acc10 + a4.y * b4.x;
                    acc11 = acc11 + a4.y * b4.y;
                    acc12 = acc12 + a4.y * b4.z;
                    acc13 = acc13 + a4.y * b4.w;
                    acc20 = acc20 + a4.z * b4.x;
                    acc21 = acc21 + a4.z * b4.y;
                    acc22 = acc22 + a4.z * b4.z;
                    acc23 = acc23 + a4.z * b4.w;
                    acc30 = acc30 + a4.w * b4.x;
                    acc31 = acc31 + a4.w * b4.y;
                    acc32 = acc32 + a4.w * b4.z;
                    acc33 = acc33 + a4.w * b4.w;
                }

                workgroupBarrier();
            }
        }
    }

    // --- epilogue: fused bias + activation, then one store per element ---
    let out_base = batch * m_total * n_total;
    let r0 = tile_row + ty * 4u + 0u;
    let r1 = tile_row + ty * 4u + 1u;
    let r2 = tile_row + ty * 4u + 2u;
    let r3 = tile_row + ty * 4u + 3u;
    let c0 = tile_col + tx * 4u + 0u;
    let c1 = tile_col + tx * 4u + 1u;
    let c2 = tile_col + tx * 4u + 2u;
    let c3 = tile_col + tx * 4u + 3u;

    if (r0 < m_total) {
        var bv: f32 = 0.0;
        if (params.has_bias != 0u) { bv = bias_buf[r0]; }
        let row_base = out_base + r0 * n_total;
        if (c0 < n_total) { outp[row_base + c0] = activate(acc00 + bv); }
        if (c1 < n_total) { outp[row_base + c1] = activate(acc01 + bv); }
        if (c2 < n_total) { outp[row_base + c2] = activate(acc02 + bv); }
        if (c3 < n_total) { outp[row_base + c3] = activate(acc03 + bv); }
    }
    if (r1 < m_total) {
        var bv: f32 = 0.0;
        if (params.has_bias != 0u) { bv = bias_buf[r1]; }
        let row_base = out_base + r1 * n_total;
        if (c0 < n_total) { outp[row_base + c0] = activate(acc10 + bv); }
        if (c1 < n_total) { outp[row_base + c1] = activate(acc11 + bv); }
        if (c2 < n_total) { outp[row_base + c2] = activate(acc12 + bv); }
        if (c3 < n_total) { outp[row_base + c3] = activate(acc13 + bv); }
    }
    if (r2 < m_total) {
        var bv: f32 = 0.0;
        if (params.has_bias != 0u) { bv = bias_buf[r2]; }
        let row_base = out_base + r2 * n_total;
        if (c0 < n_total) { outp[row_base + c0] = activate(acc20 + bv); }
        if (c1 < n_total) { outp[row_base + c1] = activate(acc21 + bv); }
        if (c2 < n_total) { outp[row_base + c2] = activate(acc22 + bv); }
        if (c3 < n_total) { outp[row_base + c3] = activate(acc23 + bv); }
    }
    if (r3 < m_total) {
        var bv: f32 = 0.0;
        if (params.has_bias != 0u) { bv = bias_buf[r3]; }
        let row_base = out_base + r3 * n_total;
        if (c0 < n_total) { outp[row_base + c0] = activate(acc30 + bv); }
        if (c1 < n_total) { outp[row_base + c1] = activate(acc31 + bv); }
        if (c2 < n_total) { outp[row_base + c2] = activate(acc32 + bv); }
        if (c3 < n_total) { outp[row_base + c3] = activate(acc33 + bv); }
    }
}
"#;

/// Build this kernel's shader module, bind group layout and pipeline.
pub(crate) fn build_conv2d_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        device,
        "conv2d_implicit",
        CONV2D_SHADER,
        "conv2d_implicit",
        &[bgl_ro(0), bgl_ro(1), bgl_ro(2), bgl_rw(3), bgl_uniform(4)],
    )
}

// ========================================================================
// Pipeline cache
// ========================================================================

/// Thread-local one-entry cache of the compiled conv pipeline.
///
/// The kernels in this directory otherwise rebuild their pipeline per call
/// (see [`kernel_support`](super::kernel_support)), which is fine for a kernel
/// that runs once. This one runs 20 times per InSwapper frame and 53 times per
/// ArcFace frame, and a WGSL compile is milliseconds — the same order as the
/// dispatch it is preparing. Caching it here rather than on `GpuContext` keeps
/// the change inside this file; the eventual hoist into
/// `GpuContext::build_from_device_queue` is a call-site move, exactly as
/// `kernel_support` describes.
///
/// * **Thread-local, not `static`** — `wgpu::Device` is neither `Send` nor
///   `Sync` on wasm32, so a `static Mutex<_>` would not compile there. Native
///   worker threads each keep their own copy, which costs one extra compile
///   per thread and nothing else.
/// * **The `Device` is stored, not just compared** — `wgpu`'s handle equality
///   is `Arc` identity, so holding the handle is what guarantees a later,
///   different device cannot compare equal by reusing a freed slot. It also
///   means every entry keeps its device alive, which is why insertion goes
///   through [`insert_for_current_device`]: it drops the other devices'
///   entries first, so a session that drops its `GpuContext` really does
///   release the device rather than leaving it pinned until the thread exits.
///   The `f32`/`f16` pair for the *current* device survives, because the
///   predicate tests the device alone.
///
/// [w2-f16] `f16` is part of the identity, not a property of the entry: a
/// context that flips the toggle must never be handed the other variant's
/// pipeline. The two are compiled from different sources and read the weight
/// binding at different widths, so sharing a slot would not be slow — it would
/// reinterpret pairs of halves as single floats.
struct CachedPipeline {
    device: wgpu::Device,
    f16: bool,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

thread_local! {
    /// At most one entry per `(device, f16)`. A `Vec` rather than two `Option`s
    /// so the lookup reads the same for both and adding a third variant later
    /// is not a restructure; it holds two entries at most in practice.
    static CONV2D_PIPELINE: std::cell::RefCell<Vec<CachedPipeline>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Devices on which the `f16` pipeline failed to compile.
    ///
    /// A negative cache, and it is load-bearing. Compiling a shader the driver
    /// rejects raises a validation error, and the dispatch's own `ErrorScope`
    /// would turn that into `mark_degraded` — killing the whole session's GPU
    /// path over an *optional* fast path. So the first `f16` compile on a
    /// device happens inside its own scope (see [`conv2d_pipeline_f16_async`]),
    /// and a failure is remembered here so it is never retried and never
    /// escalated: the kernel simply keeps taking the `f32` path.
    ///
    /// The entries are device handles, so they are inserted through
    /// [`insert_for_current_device`] for the same lifetime reason as the
    /// pipeline cache above: remembering a verdict must not pin the device it
    /// was reached on.
    static CONV2D_F16_UNAVAILABLE: std::cell::RefCell<Vec<wgpu::Device>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The compiled `f32` pipeline for `device`, building and caching it on first
/// use.
fn conv2d_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    cached_pipeline(device, false, || build_conv2d_pipeline(device))
}

/// Look `(device, f16)` up in the thread-local cache, building with `build` on
/// a miss.
fn cached_pipeline(
    device: &wgpu::Device,
    f16: bool,
    build: impl FnOnce() -> (wgpu::ComputePipeline, wgpu::BindGroupLayout),
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let hit = CONV2D_PIPELINE.with(|cell| {
        cell.borrow()
            .iter()
            .find(|c| c.f16 == f16 && &c.device == device)
            .map(|c| (c.pipeline.clone(), c.layout.clone()))
    });
    if let Some(found) = hit {
        return found;
    }
    let (pipeline, layout) = build();
    CONV2D_PIPELINE.with(|cell| {
        insert_for_current_device(
            &mut cell.borrow_mut(),
            |c| &c.device == device,
            CachedPipeline {
                device: device.clone(),
                f16,
                pipeline: pipeline.clone(),
                layout: layout.clone(),
            },
        );
    });
    (pipeline, layout)
}

/// The compiled `f16` pipeline for this context, or `None` when half precision
/// is unavailable here.
///
/// `None` for any of: the toggle is off or the device lacks `SHADER_F16`; the
/// `f32` source has drifted from the derivation's anchors
/// (`super::f16_variant`); or this device already failed to compile the
/// variant once.
///
/// # Why the first compile gets its own error scope
///
/// `ErrorScope::finish_async` marks the context **degraded** on any captured
/// validation error, which sends every subsequent node — of every op, not just
/// this one — to the CPU for the rest of the session. That is the right
/// response to a broken dispatch and the wrong one to a driver declining an
/// optional shader extension. Compiling here, before the dispatch's own scope
/// is opened, and popping this scope ourselves keeps an `f16` rejection what it
/// actually is: this kernel has no half-precision variant on this device, so it
/// uses the `f32` one.
async fn conv2d_pipeline_f16_async(
    ctx: &GpuContext,
) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    if !ctx.f16_compute_enabled() {
        return None;
    }
    let device = &ctx.device;
    let known_bad = CONV2D_F16_UNAVAILABLE.with(|cell| cell.borrow().iter().any(|d| d == device));
    if known_bad {
        return None;
    }
    if let Some(found) = CONV2D_PIPELINE.with(|cell| {
        cell.borrow()
            .iter()
            .find(|c| c.f16 && &c.device == device)
            .map(|c| (c.pipeline.clone(), c.layout.clone()))
    }) {
        return Some(found);
    }
    let src = super::f16_variant::conv2d_f16(CONV2D_SHADER)?;

    // This crate's scopes are a per-thread LIFO stack: this one is pushed and
    // popped entirely before the caller opens the dispatch's scope, so the
    // ordering contract in `device_guard::ErrorScope` is preserved.
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let built = build_pipeline(
        device,
        "conv2d_implicit_f16",
        src,
        "conv2d_implicit",
        &[bgl_ro(0), bgl_ro(1), bgl_ro(2), bgl_rw(3), bgl_uniform(4)],
    );
    if guard.pop().await.is_some() {
        // Deliberately not `ctx.mark_degraded`: an optional shader extension
        // this device will not compile is a decline, not a dead device.
        CONV2D_F16_UNAVAILABLE.with(|cell| {
            insert_for_current_device(&mut cell.borrow_mut(), |d| d == device, device.clone());
        });
        return None;
    }
    Some(cached_pipeline(device, true, || built))
}

/// \[w4\] Drop this thread's `CONV2D_PIPELINE` and `CONV2D_F16_UNAVAILABLE`
/// entries for a device that is going away.
///
/// Both caches, because both hold a `wgpu::Device` handle and both therefore
/// keep it alive — the negative cache stores the device precisely so that a
/// later, different device cannot land in a freed slot and compare equal. See
/// `kernel_support::purge_thread_local` for why this is same-thread
/// best-effort on top of the retain-on-insert rule rather than instead of it.
pub(super) fn purge_device(device: &wgpu::Device) {
    super::kernel_support::purge_thread_local(&CONV2D_PIPELINE, |cached| &cached.device == device);
    super::kernel_support::purge_thread_local(&CONV2D_F16_UNAVAILABLE, |cached| cached == device);
}

/// \[w4\] Entries this thread holds for `device` across both caches. Test-only.
#[cfg(test)]
pub(super) fn cached_entries_for_device(device: &wgpu::Device) -> usize {
    super::kernel_support::thread_local_matches(&CONV2D_PIPELINE, |cached| &cached.device == device)
        + super::kernel_support::thread_local_matches(&CONV2D_F16_UNAVAILABLE, |cached| {
            cached == device
        })
}

// ========================================================================
// Shape planning
// ========================================================================

/// Everything the dispatch needs, derived once from the operand shapes.
///
/// Building this is the whole validation pass: if it returns `Some`, every
/// index the kernel computes is in range, every buffer fits the device, and
/// every field below fits the `u32` the shader indexes with.
struct ConvPlan {
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    n_out: usize,
    out_len: usize,
    in_bytes: u64,
    bias_bytes: u64,
    out_bytes: u64,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
}

/// Validate a `group = 1` NCHW convolution and derive its dispatch.
///
/// Returns `None` — i.e. "let the hybrid path try" — for a grouped
/// convolution, a malformed or degenerate shape, an operand the device cannot
/// bind, or a dispatch grid wider than the device allows. Never panics: every
/// product is checked, because these numbers come straight from a model file.
#[allow(clippy::too_many_arguments)]
fn plan_conv(
    limits: &GpuLimits,
    input: TensorSource<'_>,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Option<ConvPlan> {
    if input.shape().len() != 4 || weight.shape.len() != 4 {
        return None;
    }
    // Grouped convolution is a deliberate decline, not an oversight — see the
    // module docs. `group == 0` is malformed and declines the same way.
    if group != 1 {
        return None;
    }
    let input_shape = input.shape();
    let (n, c_in, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (c_out, c_w, kh, kw) = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    );
    if c_w != c_in || c_in == 0 || c_out == 0 || kh == 0 || kw == 0 || h == 0 || w == 0 {
        return None;
    }
    if strides[0] == 0 || strides[1] == 0 || dilations[0] == 0 || dilations[1] == 0 {
        return None;
    }

    // Output extent, with the same checked arithmetic the hybrid path uses: a
    // dilated window wider than the padded input would underflow `usize`.
    let padded_h = h.checked_add(pads[0])?.checked_add(pads[2])?;
    let padded_w = w.checked_add(pads[1])?.checked_add(pads[3])?;
    let span_h = dilations[0].checked_mul(kh - 1)?.checked_add(1)?;
    let span_w = dilations[1].checked_mul(kw - 1)?.checked_add(1)?;
    let oh = padded_h.checked_sub(span_h)? / strides[0] + 1;
    let ow = padded_w.checked_sub(span_w)? / strides[1] + 1;
    let n_out = oh.checked_mul(ow)?;
    if n_out == 0 {
        return None;
    }

    // Operand lengths, exactly as the kernel will index them.
    let in_len = n.checked_mul(c_in)?.checked_mul(h)?.checked_mul(w)?;
    let k_total = c_in.checked_mul(kh)?.checked_mul(kw)?;
    let weight_len = c_out.checked_mul(k_total)?;
    let out_len = n.checked_mul(c_out)?.checked_mul(n_out)?;
    if input.len() < in_len || weight.data.len() < weight_len {
        return None;
    }
    if let Some(b) = bias {
        if b.data.len() < c_out {
            return None;
        }
    }

    // Every `u32` the uniform block carries must actually fit.
    u32::try_from(c_in).ok()?;
    u32::try_from(c_out).ok()?;
    u32::try_from(h).ok()?;
    u32::try_from(w).ok()?;
    u32::try_from(kh).ok()?;
    u32::try_from(kw).ok()?;
    u32::try_from(strides[0]).ok()?;
    u32::try_from(strides[1]).ok()?;
    u32::try_from(dilations[0]).ok()?;
    u32::try_from(dilations[1]).ok()?;
    i32::try_from(pads[0]).ok()?;
    i32::try_from(pads[1]).ok()?;
    u32::try_from(n_out).ok()?;
    u32::try_from(k_total).ok()?;
    u32::try_from(h.checked_mul(w)?).ok()?;
    u32::try_from(kh.checked_mul(kw)?).ok()?;

    // A zero batch has no dispatch at all; the caller short-circuits it, but
    // guard here too so `wg_z` is never zero.
    if n == 0 {
        return None;
    }

    // Bindings. `checked_storage_bytes` also enforces the `u32` element bound
    // every index in the shader relies on.
    let in_bytes = checked_storage_bytes(limits, in_len)?;
    // [w2-f16] The weight's *byte* figure now depends on its on-device format,
    // so it is derived at the call site rather than carried here. This call is
    // kept for its other job: enforcing the `u32` element bound every index in
    // the shader relies on, which is a property of the element count and is
    // therefore the same for both formats.
    checked_storage_bytes(limits, weight_len)?;
    let bias_bytes = checked_storage_bytes(limits, bias.map_or(1, |_| c_out).max(1))?;
    let out_bytes = checked_storage_bytes(limits, out_len)?;
    if !limits.buffer_fits(out_bytes) {
        return None;
    }

    // Dispatch grid: one workgroup per 64x64 output tile, per batch element.
    let wg_x = u32::try_from(n_out).ok()?.div_ceil(MACRO_TILE);
    let wg_y = u32::try_from(c_out).ok()?.div_ceil(MACRO_TILE);
    let wg_z = u32::try_from(n).ok()?;
    // `dispatch_2d_fits` in `device_guard` only covers X and Y; this kernel is
    // the first with a real Z extent, so all three are checked here.
    let max_dim = limits.max_workgroups_per_dimension;
    if wg_x == 0 || wg_y == 0 || wg_z == 0 || wg_x > max_dim || wg_y > max_dim || wg_z > max_dim {
        return None;
    }

    Some(ConvPlan {
        n,
        c_in,
        h,
        w,
        c_out,
        kh,
        kw,
        oh,
        ow,
        n_out,
        out_len,
        in_bytes,
        bias_bytes,
        out_bytes,
        wg_x,
        wg_y,
        wg_z,
    })
}

// ========================================================================
// Entry points
// ========================================================================

/// Direct NCHW Conv2D on the GPU: implicit GEMM, fused bias, fused activation.
///
/// `input` is `[N, C_in, H, W]`, `weight` is `[C_out, C_in, kH, kW]`, `bias` is
/// `[C_out]` or absent. `pads` is ONNX order — `[top, left, bottom, right]`.
/// The result is `[N, C_out, OH, OW]` with `bias` added and `act` applied.
///
/// Returns `None` when this kernel declines: `group != 1`, a malformed or
/// degenerate shape, a non-finite activation scalar, a degraded context, or an
/// operand/dispatch the device cannot handle. Callers fall back to the hybrid
/// im2col path in `compute.rs` (and must apply `act` themselves when they do).
///
/// **No size threshold.** A one-pixel convolution dispatches happily. The
/// CPU/GPU placement decision belongs at the call site — see the module docs.
///
/// Uploads the weight and bias on every call. A caller whose weight is a graph
/// initializer — the same bytes on every frame — should use
/// [`gpu_conv2d_implicit_resident_async`] instead and pass their identities.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_implicit_async(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: ConvActivation,
) -> Option<Tensor> {
    gpu_conv2d_implicit_resident_async(
        ctx,
        input,
        weight,
        bias,
        WeightKeys::default(),
        strides,
        pads,
        dilations,
        group,
        act,
    )
    .await
}

/// [`gpu_conv2d_implicit_async`] with the weight and bias kept on the device.
///
/// `keys` names the two invariant operands. A named operand is uploaded the
/// first time this context sees it and bound from the residency cache
/// thereafter — for InSwapper-128 that is 502.7 MB of convolution weights that
/// used to cross the bus on every frame. A `None` slot behaves exactly as it
/// always did, so the input activation (which changes every frame) and the
/// params block are untouched by any of this.
///
/// Numerically identical to the un-keyed form by construction: the same bytes
/// reach the same binding, the shader is the same, and nothing about the
/// dispatch grid or the accumulation order depends on where the buffer came
/// from.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_implicit_resident_async(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    keys: WeightKeys<'_>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: ConvActivation,
) -> Option<Tensor> {
    gpu_conv2d_implicit_placed_async(
        ctx,
        TensorSource::tensor(input),
        weight,
        bias,
        keys,
        strides,
        pads,
        dilations,
        group,
        act,
        OutputPlacement::Host,
    )
    .await?
    .into_tensor()
}

/// [`gpu_conv2d_implicit_resident_async`] with the *activation* free to stay on
/// the device as well.
///
/// This is the implementation the other two entry points delegate to, and the
/// only one that closes the loop the residency work exists for: `input` may be
/// the previous node's output still sitting in its device buffer, and
/// `placement` decides whether this node's result stays in one for the next.
/// With `TensorSource::Host` + `OutputPlacement::Host` it is the pre-residency
/// kernel, instruction for instruction — the same pipeline over the same
/// bytes, with the same accumulation order — so promoting a convolution into
/// the resident regime cannot change a single bit of its result.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_implicit_placed_async(
    ctx: &GpuContext,
    input: TensorSource<'_>,
    weight: &Tensor,
    bias: Option<&Tensor>,
    keys: WeightKeys<'_>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: ConvActivation,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if ctx.is_degraded() || !act.is_finite() {
        return None;
    }
    let plan = plan_conv(
        &ctx.limits,
        input,
        weight,
        bias,
        strides,
        pads,
        dilations,
        group,
    )?;
    // The WGSL placeholder that stands in for an absent bias is not the
    // caller's tensor, so it must never be cached under the caller's identity.
    let bias_key = bias.and(keys.bias);

    // [w2-f16] Resolve the half-precision variant *before* the dispatch's error
    // scope opens (see `conv2d_pipeline_f16_async`), and let its answer decide
    // the weight's on-device format. `None` here is the ordinary f32 path,
    // unchanged in every respect.
    let device = &ctx.device;
    let f16_pipeline = conv2d_pipeline_f16_async(ctx).await;
    let weight_format = if f16_pipeline.is_some() {
        WeightFormat::F16
    } else {
        WeightFormat::F32
    };
    let weight_len = plan.c_out * plan.c_in * plan.kh * plan.kw;
    let weight_bytes = weight_format.byte_len(weight_len);

    // Input, weight, bias, output and read-back staging — minus whatever is
    // already resident *in this format*, whose bytes the budget is counting
    // already. The format qualifier matters: a weight resident as f32 has not
    // paid for its f16 copy.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(input, plan.in_bytes),
        ctx.operand_admission_bytes_for(keys.weight, weight_format, weight_bytes),
        ctx.operand_admission_bytes(bias_key, plan.bias_bytes),
        plan.out_bytes,
        placement.staging_bytes(plan.out_bytes),
    ]) {
        return None;
    }

    let scope = ErrorScope::begin(ctx);
    let (pipeline, bgl) = match f16_pipeline {
        Some(pair) => pair,
        None => conv2d_pipeline(device),
    };

    let in_len = plan.n * plan.c_in * plan.h * plan.w;
    let k_total = plan.c_in * plan.kh * plan.kw;

    let input_buf = ctx.operand_source(
        "conv2d_input",
        input.truncated(in_len)?,
        wgpu::BufferUsages::STORAGE,
    )?;
    // The bias is *not* narrowed: the epilogue adds it to an f32 accumulator.
    let weight_buf = ctx.operand_buffer_typed(
        keys.weight,
        "conv2d_weight",
        weight.data.get(..weight_len)?,
        weight_format,
        wgpu::BufferUsages::STORAGE,
    )?;
    // WGSL cannot drop a binding, so a 1-element placeholder stands in when
    // there is no bias. `has_bias == 0` makes the shader's `if` skip it, so it
    // is bound but never read (WGSL `if` is real control flow, not `select`).
    let bias_placeholder = [0.0f32];
    let bias_slice: &[f32] = match bias {
        Some(b) => b.data.get(..plan.c_out)?,
        None => &bias_placeholder,
    };
    let bias_buf = ctx.operand_buffer(
        bias_key,
        "conv2d_bias",
        bytemuck::cast_slice(bias_slice),
        wgpu::BufferUsages::STORAGE,
    )?;

    let output_buf = ctx.pooled_buffer(
        plan.out_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;

    let (act_alpha, act_min, act_max) = act.scalars();
    let params = ConvParams {
        c_in: u32::try_from(plan.c_in).ok()?,
        h: u32::try_from(plan.h).ok()?,
        w: u32::try_from(plan.w).ok()?,
        c_out: u32::try_from(plan.c_out).ok()?,
        oh: u32::try_from(plan.oh).ok()?,
        ow: u32::try_from(plan.ow).ok()?,
        kh: u32::try_from(plan.kh).ok()?,
        kw: u32::try_from(plan.kw).ok()?,
        stride_h: u32::try_from(strides[0]).ok()?,
        stride_w: u32::try_from(strides[1]).ok()?,
        dil_h: u32::try_from(dilations[0]).ok()?,
        dil_w: u32::try_from(dilations[1]).ok()?,
        pad_t: i32::try_from(pads[0]).ok()?,
        pad_l: i32::try_from(pads[1]).ok()?,
        n_out: u32::try_from(plan.n_out).ok()?,
        k_stride: u32::try_from(plan.kh * plan.kw).ok()?,
        k_total: u32::try_from(k_total).ok()?,
        hw: u32::try_from(plan.h * plan.w).ok()?,
        has_bias: u32::from(bias.is_some()),
        act_mode: act.mode(),
        act_alpha,
        act_min,
        act_max,
        _pad0: 0,
    };
    let params_buf = ctx.upload_buffer(
        "conv2d_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("conv2d_staging", plan.out_bytes)?),
        OutputPlacement::Device => None,
    };

    // The pooled output buffer may be *larger* than this call needs (the pool
    // hands back anything within 2x), and `as_entire_binding()` would then bind
    // that larger size — which can exceed `max_storage_buffer_binding_size`
    // even though the request did not. Bind the exact range instead.
    let output_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &output_buf,
        offset: 0,
        size: wgpu::BufferSize::new(plan.out_bytes),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv2d_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("conv2d_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv2d_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(plan.wg_x, plan.wg_y, plan.wg_z);
    }
    if let Some(staging) = &staging_buf {
        encoder.copy_buffer_to_buffer(&output_buf, 0, staging, 0, plan.out_bytes);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }
    finish_output_async(
        ctx,
        placement,
        staging_buf,
        output_buf,
        plan.out_len,
        plan.out_bytes,
        vec![plan.n, plan.c_out, plan.oh, plan.ow],
    )
    .await
}

/// Blocking form of [`gpu_conv2d_implicit_async`].
///
/// Declines outright on wasm32 — see
/// `block_on_gpu` and the crate docs.
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d_implicit(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: ConvActivation,
) -> Option<Tensor> {
    block_on_gpu(gpu_conv2d_implicit_async(
        ctx, input, weight, bias, strides, pads, dilations, group, act,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight-from-the-definition NCHW convolution — six nested loops, no
    /// im2col, no GEMM. Deliberately *not* `oxionnx-ops`'s implementation:
    /// this is the independent oracle the WGSL is checked against at the small
    /// shapes where an indexing bug is still localizable. The full-model
    /// inventory sweep in `tests/c3_conv2d_parity.rs` checks against
    /// `oxionnx-ops` instead, so both directions are covered.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn reference_conv2d(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        strides: [usize; 2],
        pads: [usize; 4],
        dilations: [usize; 2],
        act: ConvActivation,
    ) -> Tensor {
        let (n, c_in, h, w) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (c_out, kh, kw) = (weight.shape[0], weight.shape[2], weight.shape[3]);
        let oh = (h + pads[0] + pads[2] - (dilations[0] * (kh - 1) + 1)) / strides[0] + 1;
        let ow = (w + pads[1] + pads[3] - (dilations[1] * (kw - 1) + 1)) / strides[1] + 1;
        let mut out = vec![0.0f32; n * c_out * oh * ow];
        for b in 0..n {
            for oc in 0..c_out {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0.0f32;
                        for ic in 0..c_in {
                            for ky in 0..kh {
                                let iy = (oy * strides[0] + ky * dilations[0]) as isize
                                    - pads[0] as isize;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ix = (ox * strides[1] + kx * dilations[1]) as isize
                                        - pads[1] as isize;
                                    if ix < 0 || ix >= w as isize {
                                        continue;
                                    }
                                    let iv = input.data
                                        [((b * c_in + ic) * h + iy as usize) * w + ix as usize];
                                    let wv = weight.data[((oc * c_in + ic) * kh + ky) * kw + kx];
                                    acc += iv * wv;
                                }
                            }
                        }
                        if let Some(bias) = bias {
                            acc += bias.data[oc];
                        }
                        out[((b * c_out + oc) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
        act.apply_host(&mut out);
        Tensor::new(out, vec![n, c_out, oh, ow])
    }

    /// Deterministic, signed, non-monotonic fill. A plain `i % small` ramp
    /// hides transposition bugs (many wrong indices carry the right value);
    /// this does not.
    pub(super) fn fill(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
                ((x % 23) as f32) * 0.037 - 0.4
            })
            .collect()
    }

    /// Compare against the reference with a tolerance scaled to the output's
    /// own magnitude — a per-element *relative* bound is meaningless near a
    /// zero crossing, where a K-deep f32 reduction legitimately lands.
    pub(super) fn assert_close(got: &Tensor, want: &Tensor, rel_tol: f32, case: &str) {
        assert_eq!(got.shape, want.shape, "{case}: shape");
        assert_eq!(got.data.len(), want.data.len(), "{case}: length");
        let scale = want
            .data
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()))
            .max(1e-6);
        let tol = rel_tol * scale;
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for (i, (g, e)) in got.data.iter().zip(want.data.iter()).enumerate() {
            let d = (g - e).abs();
            if d > worst {
                worst = d;
                worst_at = i;
            }
        }
        assert!(
            worst <= tol,
            "{case}: max abs error {worst} at index {worst_at} \
             (got {}, want {}) exceeds {tol} (rel_tol {rel_tol} x scale {scale})",
            got.data[worst_at],
            want.data[worst_at],
        );
    }

    fn tensor(shape: Vec<usize>, seed: u32) -> Tensor {
        let len = shape.iter().product();
        Tensor::new(fill(len, seed), shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_case(
        ctx: &GpuContext,
        case: &str,
        in_shape: [usize; 4],
        w_shape: [usize; 4],
        with_bias: bool,
        strides: [usize; 2],
        pads: [usize; 4],
        dilations: [usize; 2],
        act: ConvActivation,
    ) {
        let input = tensor(in_shape.to_vec(), 2_654_435_761);
        let weight = tensor(w_shape.to_vec(), 40_503);
        let bias = with_bias.then(|| tensor(vec![w_shape[0]], 97_711));
        let want = reference_conv2d(
            &input,
            &weight,
            bias.as_ref(),
            strides,
            pads,
            dilations,
            act,
        );
        let Some(got) = gpu_conv2d_implicit(
            ctx,
            &input,
            &weight,
            bias.as_ref(),
            strides,
            pads,
            dilations,
            1,
            act,
        ) else {
            panic!("{case}: the kernel declined a shape it must support");
        };
        assert_close(&got, &want, 1e-4, case);
    }

    /// The smallest shape that still exercises every index in the kernel:
    /// ragged in M (`c_out = 2` against a 64-row macro-tile), ragged in N
    /// (`5x5` output = 25 columns), ragged in K (`c_in = 2` against a 16-deep
    /// c-tile), and every kind of padded edge.
    #[test]
    fn tiny_3x3_padded_matches_the_reference() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "tiny 3x3 s1 p1",
            [1, 2, 5, 5],
            [2, 2, 3, 3],
            true,
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            ConvActivation::None,
        );
    }

    #[test]
    fn tiny_3x3_unpadded_and_strided_match_the_reference() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "tiny 3x3 s1 p0",
            [1, 3, 7, 7],
            [4, 3, 3, 3],
            true,
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            ConvActivation::None,
        );
        check_case(
            &ctx,
            "tiny 3x3 s2 p1",
            [1, 3, 8, 8],
            [5, 3, 3, 3],
            true,
            [2, 2],
            [1, 1, 1, 1],
            [1, 1],
            ConvActivation::None,
        );
    }

    /// `c_in = 17` and `c_in = 33` straddle the 16-deep c-tile, so the last
    /// tile is ragged and *both* staged operands must zero-pad it. Dropping
    /// the guard on the weight tile is the failure this pins.
    #[test]
    fn ragged_input_channel_tile_is_zero_padded() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        for c_in in [1usize, 15, 16, 17, 33] {
            check_case(
                &ctx,
                &format!("ragged c_in={c_in}"),
                [1, c_in, 6, 6],
                [3, c_in, 3, 3],
                true,
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                ConvActivation::None,
            );
        }
    }

    /// Ragged in both macro-tile axes at once: `c_out = 65` needs two M tiles
    /// with the second holding one row, `9x9` output = 81 columns needs two N
    /// tiles with the second holding 17.
    #[test]
    fn ragged_macro_tiles_are_bounds_checked() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "ragged macro tiles",
            [1, 4, 9, 9],
            [65, 4, 3, 3],
            true,
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            ConvActivation::None,
        );
    }

    #[test]
    fn pointwise_and_large_kernels_match_the_reference() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "1x1 s1 p0",
            [1, 8, 6, 6],
            [7, 8, 1, 1],
            true,
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            ConvActivation::None,
        );
        check_case(
            &ctx,
            "1x1 s2 p0",
            [1, 8, 7, 7],
            [7, 8, 1, 1],
            true,
            [2, 2],
            [0, 0, 0, 0],
            [1, 1],
            ConvActivation::None,
        );
        check_case(
            &ctx,
            "7x7 s1 p3",
            [1, 3, 12, 12],
            [4, 3, 7, 7],
            true,
            [1, 1],
            [3, 3, 3, 3],
            [1, 1],
            ConvActivation::None,
        );
    }

    /// Padding arithmetic must be signed end to end: computed in `u32` a
    /// negative row index wraps to ~4e9, which passes an `iy < h` test in the
    /// wrong direction and silently reads garbage instead of zero. Asymmetric
    /// pads make a sign error visible as a shifted output rather than a
    /// uniformly wrong one.
    #[test]
    fn asymmetric_and_lopsided_padding_matches_the_reference() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "asymmetric pads",
            [1, 3, 7, 9],
            [4, 3, 3, 3],
            true,
            [1, 1],
            [2, 0, 0, 3],
            [1, 1],
            ConvActivation::None,
        );
    }

    #[test]
    fn dilation_matches_the_reference() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "3x3 dilation 2",
            [1, 4, 11, 11],
            [6, 4, 3, 3],
            true,
            [1, 1],
            [2, 2, 2, 2],
            [2, 2],
            ConvActivation::None,
        );
    }

    #[test]
    fn batches_are_independent() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "batch 3",
            [3, 5, 6, 6],
            [4, 5, 3, 3],
            true,
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            ConvActivation::None,
        );
    }

    #[test]
    fn missing_bias_is_not_read() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        check_case(
            &ctx,
            "no bias",
            [1, 4, 6, 6],
            [5, 4, 3, 3],
            false,
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            ConvActivation::None,
        );
    }

    /// The fused activations, against the same reference applying them on the
    /// host. `leaky_relu` is the one that cannot be applied twice by mistake
    /// without changing the answer, so it is the one that matters most.
    #[test]
    fn fused_activations_match_the_reference() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        for (name, act) in [
            ("relu", ConvActivation::Relu),
            ("leaky_relu", ConvActivation::LeakyRelu(0.1)),
            ("clip(0,6)", ConvActivation::Clip { min: 0.0, max: 6.0 }),
            (
                "clip(-1,1)",
                ConvActivation::Clip {
                    min: -1.0,
                    max: 1.0,
                },
            ),
            // Inverted range: `f32::clamp` would panic and WGSL's `clamp` is
            // undefined. Both sides must saturate to `max` instead.
            (
                "clip(6,0) inverted",
                ConvActivation::Clip { min: 6.0, max: 0.0 },
            ),
        ] {
            check_case(
                &ctx,
                name,
                [1, 6, 7, 7],
                [8, 6, 3, 3],
                true,
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                act,
            );
        }
    }

    #[test]
    fn grouped_convolution_declines_to_the_hybrid_path() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        let input = tensor(vec![1, 8, 6, 6], 11);
        let weight = tensor(vec![8, 4, 3, 3], 13);
        assert!(
            gpu_conv2d_implicit(
                &ctx,
                &input,
                &weight,
                None,
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                2,
                ConvActivation::None,
            )
            .is_none(),
            "group > 1 must decline so the hybrid path runs"
        );
    }

    #[test]
    fn malformed_shapes_decline_instead_of_panicking() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        /// One rejected call, named so the assertion message says which.
        struct Malformed {
            why: &'static str,
            in_shape: &'static [usize],
            w_shape: &'static [usize],
            strides: [usize; 2],
            dilations: [usize; 2],
        }
        let cases = [
            Malformed {
                why: "window wider than the padded input",
                in_shape: &[1, 2, 3, 3],
                w_shape: &[4, 2, 7, 7],
                strides: [1, 1],
                dilations: [1, 1],
            },
            Malformed {
                why: "zero stride",
                in_shape: &[1, 2, 8, 8],
                w_shape: &[4, 2, 3, 3],
                strides: [0, 1],
                dilations: [1, 1],
            },
            Malformed {
                why: "zero dilation",
                in_shape: &[1, 2, 8, 8],
                w_shape: &[4, 2, 3, 3],
                strides: [1, 1],
                dilations: [0, 1],
            },
            Malformed {
                why: "channel mismatch between input and weight",
                in_shape: &[1, 3, 8, 8],
                w_shape: &[4, 2, 3, 3],
                strides: [1, 1],
                dilations: [1, 1],
            },
            Malformed {
                why: "rank 3 input",
                in_shape: &[1, 2, 8],
                w_shape: &[4, 2, 3, 3],
                strides: [1, 1],
                dilations: [1, 1],
            },
        ];
        for case in cases {
            let Malformed {
                why,
                in_shape,
                w_shape,
                strides,
                dilations,
            } = case;
            let pads = [0, 0, 0, 0];
            let input = Tensor::new(vec![0.5; in_shape.iter().product()], in_shape.to_vec());
            let weight = Tensor::new(vec![0.5; w_shape.iter().product()], w_shape.to_vec());
            assert!(
                gpu_conv2d_implicit(
                    &ctx,
                    &input,
                    &weight,
                    None,
                    strides,
                    pads,
                    dilations,
                    1,
                    ConvActivation::None,
                )
                .is_none(),
                "malformed case ({why}): in={in_shape:?} w={w_shape:?} must decline"
            );
        }
    }

    #[test]
    fn non_finite_activation_scalars_decline() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        let input = tensor(vec![1, 4, 6, 6], 17);
        let weight = tensor(vec![4, 4, 3, 3], 19);
        assert!(gpu_conv2d_implicit(
            &ctx,
            &input,
            &weight,
            None,
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            1,
            ConvActivation::LeakyRelu(f32::NAN),
        )
        .is_none());
    }

    /// The host activation must be expression-for-expression what the WGSL
    /// does, because the hybrid fallback applies it on the CPU.
    #[test]
    fn host_activation_matches_the_shader_definition() {
        let mut data = vec![-2.0f32, -0.5, 0.0, 0.5, 7.0];
        ConvActivation::Relu.apply_host(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.5, 7.0]);

        let mut data = vec![-2.0f32, -0.5, 0.0, 0.5, 7.0];
        ConvActivation::LeakyRelu(0.25).apply_host(&mut data);
        assert_eq!(data, vec![-0.5, -0.125, 0.0, 0.5, 7.0]);

        let mut data = vec![-2.0f32, -0.5, 0.0, 0.5, 7.0];
        ConvActivation::Clip { min: 0.0, max: 6.0 }.apply_host(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.5, 6.0]);

        // An inverted range must saturate, not panic — `f32::clamp` would.
        let mut data = vec![-2.0f32, 1.0, 7.0];
        ConvActivation::Clip { min: 6.0, max: 0.0 }.apply_host(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);

        let mut data = vec![-2.0f32, 7.0];
        ConvActivation::None.apply_host(&mut data);
        assert_eq!(data, vec![-2.0, 7.0]);
    }

    #[test]
    fn activation_scalars_round_trip_their_discriminants() {
        assert_eq!(ConvActivation::None.mode(), 0);
        assert_eq!(ConvActivation::Relu.mode(), 1);
        assert_eq!(ConvActivation::LeakyRelu(0.2).mode(), 2);
        assert_eq!(ConvActivation::Clip { min: 0.0, max: 6.0 }.mode(), 3);
        assert_eq!(ConvActivation::LeakyRelu(0.2).scalars().0, 0.2);
        let (_, min, max) = ConvActivation::Clip {
            min: -1.5,
            max: 6.0,
        }
        .scalars();
        assert_eq!((min, max), (-1.5, 6.0));
        assert!(!ConvActivation::Clip {
            min: f32::NAN,
            max: 1.0
        }
        .is_finite());
    }
}
