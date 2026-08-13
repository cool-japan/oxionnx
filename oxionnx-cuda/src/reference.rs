//! A minimal, deliberately-naive CPU oracle for shadow-verifying CUDA kernel
//! output, gated behind `OXIONNX_CUDA_VERIFY=1`.
//!
//! # Why this exists
//!
//! `try_cuda_dispatch` returns `Ok(Some(wrong_data))` — a *successful* answer
//! — when a kernel is wrong; nothing downstream can tell the difference from
//! a correct one.  This module is the mechanism that can: it recomputes the
//! same op on the CPU with the simplest, most obviously-correct
//! implementation available (naive loops, `f64` accumulation, no batching,
//! no SIMD) and diffs it against what the GPU returned.  See
//! [`crate::context`] for the full activation/verify/strict story.
//!
//! # Why this does not depend on `oxionnx-ops`
//!
//! Mirroring `oxionnx-directml`'s identical layering rule: an execution
//! provider must not depend on the CPU operator library it exists to bypass.
//! A normal dependency on `oxionnx-ops` here would invert that, and would
//! make it possible for a bug to accidentally use `oxionnx-ops` as the
//! *dispatch fallback* instead of returning `Ok(None)` and letting the
//! session runner do it.  So every oracle below is reimplemented from
//! scratch, small enough to read in one sitting, and cross-checked against
//! hand-computed constants in this module's own tests.
//!
//! # What the oracle models
//!
//! Deliberately **not** the ONNX spec in the abstract — the exact formula
//! each `oxicuda_ptx` kernel template computes, including the two ops
//! (`LeakyRelu`, `HardSigmoid`) that hard-code the ONNX *default* constants
//! with no launch-time override, and `Gelu`, whose kernel computes the
//! `tanh` approximation rather than the exact/erf form.  This matches
//! `try_cuda_dispatch`'s dispatch-time guards (see `lib.rs`), which decline
//! to the CPU whenever a node's actual attributes would disagree with these
//! constants — so by the time any of these oracle functions runs, the
//! constants below are exactly what the kernel is computing.  Verified
//! directly against `oxicuda_ptx::templates::elementwise`'s
//! `generate_leaky_relu` / `generate_hard_sigmoid` / `generate_gelu` doc
//! comments and PTX literals (`0f3C23D70A` = 0.01, `0f3E4CCCCD` = 0.2,
//! `0f3F000000` = 0.5).
//!
//! # Parallelism
//!
//! `ref_conv`/`ref_matmul`/`ref_reduce`/`ref_softmax` split their work
//! across `rayon` above a size threshold (`PAR_MIN_MACS` /
//! `PAR_MIN_ELEMENTWISE_LEN`): naive single-threaded `f64` conv is the
//! entire reason `OXIONNX_CUDA_VERIFY=1` used to turn a 7.7s swap into one
//! that had not finished after 45 minutes (349 GFLOP of it per InSwapper
//! frame alone). Every oracle's per-output-unit formula (one conv row, one
//! matmul row, one reduce/softmax row) is written exactly once and called
//! identically from the serial and parallel branches, splitting only
//! across *independent* output rows/elements -- never within a single
//! output element's own accumulation loop, whose `f64` summation order
//! (and therefore its rounding) is untouched by parallelism. See the
//! `*_parallel_matches_serial_on_*` property tests below, which assert
//! this directly (bit-for-bit, not just "close").

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use oxionnx_core::graph::OpKind;
use rayon::prelude::*;

use crate::context::{parse_env_flag, FailurePolicy};
use crate::conv::{ConvActivation, ConvParams};
use crate::error::CudaDispatchError;

/// The data-movement wave's oracles (`MaxPool`/`AveragePool`/`Resize`/`Pad`/
/// `Slice`/`Concat`) live in a companion file rather than growing this one
/// further -- see that file's own header for why -- but are re-exported here
/// so every oracle in the crate is reachable uniformly as `reference::ref_*`.
#[path = "reference_data_ops.rs"]
mod reference_data_ops;
pub use reference_data_ops::{ref_concat, ref_pad, ref_pool, ref_resize, ref_slice};

/// The elementwise/normalization wave's oracles (the `[1,C,1,1]`/scalar
/// broadcast path of `Add`/`Sub`/`Mul`/`Div`, `PRelu`, `BatchNormalization`,
/// `OxiInstanceNorm`) live in a companion file for the identical reason
/// [`reference_data_ops`] does — see that module's doc comment just above,
/// and [`reference_norm_ops`]'s own header.
#[path = "reference_norm_ops.rs"]
mod reference_norm_ops;
pub use reference_norm_ops::{
    ref_batch_norm, ref_binary_broadcast, ref_oxi_instance_norm, ref_prelu,
};

// ─── the verify gate ────────────────────────────────────────────────────────

/// Set this to shadow-verify every CUDA-dispatched op against this module's
/// CPU oracle: `OXIONNX_CUDA_VERIFY=1`.
///
/// Off by default: computing the oracle roughly doubles the cost of every
/// claimed node, so this is a diagnostic mode for development/CI on real
/// hardware, not something a production build should carry.
pub const VERIFY_ENV_VAR: &str = "OXIONNX_CUDA_VERIFY";

/// Is shadow verification on?
///
/// Read once and cached: the value cannot change within a process, and this
/// is consulted on the dispatch path of every claimed node.
#[must_use]
pub fn verify_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_flag(std::env::var(VERIFY_ENV_VAR).ok().as_deref()))
}

// ─── comparison ─────────────────────────────────────────────────────────────

/// Absolute tolerance for the GPU-vs-oracle comparison.
///
/// `oxicuda_ptx`'s transcendental kernels use `ex2.approx` / `lg2.approx` /
/// `rcp.approx` (PTX's reduced-precision fast-math intrinsics, documented to
/// roughly 22-23 bits of accuracy) rather than a correctly-rounded `exp`, so
/// bit-exact agreement is not the bar — a value that is *wrong for the
/// formula*, not merely differently-rounded, is.
const ATOL: f32 = 1.0e-4;
/// Relative tolerance, applied against the oracle's magnitude. Wide enough to
/// absorb a longer MatMul/ReduceSum accumulation's rounding drift without
/// absorbing a genuinely wrong element.
const RTOL: f32 = 1.0e-3;

/// Do `gpu` and `cpu` agree within tolerance?
///
/// Two `NaN`s agree (both engines saturated/undefined the same way); one
/// `NaN` and one finite value never do. `+inf`/`-inf` must match exactly.
fn nearly_eq(gpu: f32, cpu: f32) -> bool {
    if gpu == cpu {
        return true; // exact match, including matching +-inf and +-0.0.
    }
    if gpu.is_nan() || cpu.is_nan() {
        return gpu.is_nan() && cpu.is_nan();
    }
    if !gpu.is_finite() || !cpu.is_finite() {
        return false; // one side is infinite, the other is not (and not NaN either).
    }
    let diff = (gpu - cpu).abs();
    diff <= ATOL + RTOL * cpu.abs()
}

/// Compare a GPU result against the oracle's, element by element.
///
/// # Errors
/// A message naming the first disagreement (index, both values, and the raw
/// difference) — the single most diagnostic thing a mismatch report can
/// carry: an early index says "the first thread group is wrong"; a late one
/// says "only the tail is wrong".
pub fn compare(gpu: &[f32], cpu: &[f32]) -> Result<(), String> {
    if gpu.len() != cpu.len() {
        return Err(format!(
            "length mismatch: GPU returned {} elements, the CPU oracle computed {}",
            gpu.len(),
            cpu.len()
        ));
    }
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if !nearly_eq(g, c) {
            return Err(format!(
                "element {i}: GPU={g}, CPU-oracle={c} (|diff|={:.3e}, tolerance={:.3e})",
                (g - c).abs(),
                ATOL + RTOL * c.abs()
            ));
        }
    }
    Ok(())
}

// ─── shadow_verify: the glue between a kernel call and the policy ─────────

/// Shadow-verify a GPU kernel's output against this module's CPU oracle, and
/// decide what the caller should do about it.
///
/// `verify_on` and `policy` are taken as plain parameters — rather than read
/// internally from [`verify_enabled`] / [`FailurePolicy::current`] — purely
/// so this function is unit-testable without mutating the process
/// environment (global, racy under a threaded test runner, and cached by
/// both of those `OnceLock`s besides). Real call sites in `lib.rs` pass the
/// live values.
///
/// `oracle` is a closure — not a plain `&[f32]` — so the (potentially
/// expensive) CPU computation is skipped entirely when `verify_on` is
/// `false`, which is the default and every production dispatch.
///
/// # Returns
/// * `Ok(true)` — verification is off, or on and passed: the caller should
///   trust `gpu` and proceed normally.
/// * `Ok(false)` — verification is on, the comparison disagreed, and
///   `policy` is [`FailurePolicy::Fallback`] (the default). The mismatch has
///   already been logged at `error!`. The caller **must** discard `gpu` and
///   return `Ok(None)` so the real CPU operator recomputes the node.
/// * `Ok(true)` (with a `warn!` logged) — the oracle has no formula for this
///   op/configuration (`oracle` returned `None`). This is a gap in the
///   *oracle*, not a proven GPU bug, so it is never promoted to `Err`, but a
///   silent skip would defeat the purpose of `OXIONNX_CUDA_VERIFY` — it is
///   always audible.
///
/// # Errors
/// [`CudaDispatchError::Verify`] only when `policy` is
/// [`FailurePolicy::Strict`] and the comparison disagreed.
pub(crate) fn shadow_verify(
    op: &str,
    gpu: &[f32],
    verify_on: bool,
    policy: FailurePolicy,
    oracle: impl FnOnce() -> Option<Vec<f32>>,
) -> Result<bool, CudaDispatchError> {
    if !verify_on {
        return Ok(true);
    }
    let Some(cpu) = oracle() else {
        tracing::warn!(
            op,
            "CUDA shadow verification: the CPU oracle has no formula for this op/configuration; \
             skipping the check for this node (this is a gap in the oracle, not a proven GPU bug)"
        );
        return Ok(true);
    };
    match compare(gpu, &cpu) {
        Ok(()) => {
            tracing::debug!(op, "CUDA shadow verification passed");
            Ok(true)
        }
        Err(reason) => {
            tracing::error!(
                op,
                %reason,
                strict = policy == FailurePolicy::Strict,
                "CUDA kernel VERIFY MISMATCH (a GPU kernel bug, not a decline) — the GPU's \
                 output has been discarded, not returned. Set {} to make this fatal.",
                crate::context::STRICT_ENV_VAR,
            );
            match policy {
                FailurePolicy::Strict => Err(CudaDispatchError::Verify(reason)),
                FailurePolicy::Fallback => Ok(false),
            }
        }
    }
}

// ─── parallelism plumbing ───────────────────────────────────────────────────
//
// `ref_conv`, `ref_matmul`, `ref_reduce`, and `ref_softmax` all follow the
// same shape: a small per-output-unit formula (one conv row, one matmul
// row, one reduce/softmax row) written exactly once, called identically
// from a serial branch (small shapes, where a `rayon` split would cost
// more than it saves) and a `rayon`-parallel branch (real shapes).
// Splitting only ever happens across *independent* output rows/elements --
// never within a single output element's own `f64` accumulation loop, so
// every element's accumulation order (and therefore its rounding) is
// bit-for-bit identical to the pre-parallelisation serial code, whichever
// branch computed it. The `*_parallel_matches_serial_on_*` property tests
// below assert this directly against randomised shapes, not just this
// comment.

/// Below this many scalar multiply-adds, a `rayon` split costs more than
/// the serial loop it would replace: a `rayon::join` still pays a
/// work-deque push/pop even when nothing gets stolen, and this crate's own
/// hand-verified unit tests below call `ref_conv`/`ref_matmul`/`ref_reduce`
/// on 4-to-25-element toy shapes hundreds of times. `1 << 16` sits far
/// below any real conv/matmul in this pipeline (the smallest is a 1x1
/// projection, still tens of thousands of MACs) and far above every
/// hand-verified test shape below. Mirrors the identical small-input guard
/// `oxionnx-ops`'s `matmul_nt_into_par`/`parallel_sgemm` use for the same
/// reason (`attention/gemm.rs::PAR_MIN_MACS`,
/// `conv/conv2d.rs::PARALLEL_GEMM_THRESHOLD`).
const PAR_MIN_MACS: u64 = 1 << 16;

/// The elementwise-formula equivalent of [`PAR_MIN_MACS`]: below this many
/// elements, `ref_unary_vec`/`ref_binary_vec`/`ref_softmax` stay serial.
const PAR_MIN_ELEMENTWISE_LEN: usize = 1 << 14;

/// Is a `rayon`-parallel split worth it for `total_macs` worth of scalar
/// multiply-adds?
///
/// Also declines whenever the global `rayon` pool has been configured down
/// to a single thread (`RAYON_NUM_THREADS=1`, or a custom single-threaded
/// pool): splitting work that can never run concurrently only adds
/// overhead for nothing.
fn parallel_worthwhile(total_macs: u64) -> bool {
    total_macs >= PAR_MIN_MACS && rayon::current_num_threads() > 1
}

/// How often a single long-running oracle call may emit a `tracing::info!`
/// progress line, in nanoseconds -- frequent enough that a user watching an
/// `OXIONNX_CUDA_VERIFY=1` run sees liveness, infrequent enough that a
/// multi-second op does not flood the log. "A few seconds", per this
/// module's brief.
const PROGRESS_LOG_INTERVAL_NANOS: u64 = 2_000_000_000;

/// Rate-limited, thread-safe progress reporter shared by every `rayon`
/// worker computing one oracle call.
///
/// Workers call [`ProgressReporter::advance`] once per independent unit of
/// work they finish (a conv/matmul/reduce output row or `rayon`-chunk, a
/// softmax row); at most one `tracing::info!` line escapes per
/// [`PROGRESS_LOG_INTERVAL_NANOS`] *total*, regardless of how many threads
/// call `advance` concurrently -- the gate is a single atomic
/// compare-exchange on a shared "nanoseconds of the last log" timestamp, so
/// only the worker that wins the race logs, and every other call pays one
/// atomic add plus one atomic load. Only ever constructed on the
/// large-shape `rayon` branch (see [`PAR_MIN_MACS`]/[`PAR_MIN_ELEMENTWISE_LEN`]
/// above): a call too small to be worth parallelising is also too small to
/// ever run long enough to need a progress line.
struct ProgressReporter {
    op: &'static str,
    shape: String,
    total_units: usize,
    done: AtomicUsize,
    last_log_nanos: AtomicU64,
    start: Instant,
}

impl ProgressReporter {
    /// `total_units` is whatever [`ProgressReporter::advance`] counts in
    /// (output rows, or `rayon`-chunks for `ref_reduce`) -- only used to
    /// compute a percentage for the log line, so an approximate unit is
    /// fine.
    fn new(op: &'static str, shape: String, total_units: usize) -> Self {
        Self {
            op,
            shape,
            total_units,
            done: AtomicUsize::new(0),
            // Starts at 0 ("due immediately"), but `advance` only logs once
            // *elapsed* time clears the interval, so the very first line
            // still cannot appear before `PROGRESS_LOG_INTERVAL_NANOS` has
            // actually passed -- a call that finishes faster than that
            // never logs at all, which is the point: liveness only matters
            // once a call has already run long enough to make a user
            // wonder.
            last_log_nanos: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    /// Record `n` newly finished units of work.
    fn advance(&self, n: usize) {
        let done = self.done.fetch_add(n, Ordering::Relaxed) + n;
        let elapsed = self.start.elapsed();
        let elapsed_nanos = elapsed.as_nanos() as u64;
        let last = self.last_log_nanos.load(Ordering::Relaxed);
        if elapsed_nanos.saturating_sub(last) < PROGRESS_LOG_INTERVAL_NANOS {
            return;
        }
        // CAS, not a plain store: if two threads cross the interval at
        // once, exactly one updates the timestamp and logs; the loser's
        // `compare_exchange` fails and it returns without logging.
        if self
            .last_log_nanos
            .compare_exchange(last, elapsed_nanos, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            let percent = if self.total_units == 0 {
                100.0
            } else {
                100.0 * (done.min(self.total_units) as f64) / (self.total_units as f64)
            };
            tracing::info!(
                op = self.op,
                shape = %self.shape,
                percent = format!("{percent:.1}"),
                elapsed_s = format!("{:.1}", elapsed.as_secs_f64()),
                "OXIONNX_CUDA_VERIFY CPU oracle still computing",
            );
        }
    }
}

// ─── the oracle itself ──────────────────────────────────────────────────────

/// Compute one output row `out[i, 0..n]` of a naive `[m,k] x [k,n] ->
/// [m,n]` row-major matmul.
///
/// The single place the per-element formula is written down -- both of
/// [`ref_matmul`]'s branches call exactly this, once per row `i`, so there
/// is no second copy that could silently drift from the first. Each output
/// element's own `f64` accumulation order (`p` ascending from 0) is
/// unaffected by which branch calls it or which thread runs it.
fn matmul_row(row_out: &mut [f32], a_row: &[f32], b: &[f32], k: usize, n: usize) {
    for (j, out_val) in row_out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for p in 0..k {
            acc += f64::from(a_row[p]) * f64::from(b[p * n + j]);
        }
        *out_val = acc as f32;
    }
}

/// Fill every row of `out` serially -- no `rayon` involvement at all. See
/// [`ref_matmul`].
fn ref_matmul_fill_serial(out: &mut [f32], a: &[f32], b: &[f32], k: usize, n: usize) {
    for (i, row) in out.chunks_mut(n).enumerate() {
        matmul_row(row, &a[i * k..i * k + k], b, k, n);
    }
}

/// Fill every row of `out` via a `rayon` split across rows. See
/// [`ref_matmul`].
///
/// Exposed as its own function (rather than inlined into [`ref_matmul`]) so
/// the `*_parallel_matches_serial_on_*` property tests below can call it
/// directly against [`ref_matmul_fill_serial`] on the *same* large shape --
/// [`ref_matmul`] itself only ever runs one branch or the other for a given
/// `m`/`k`/`n`, decided by [`parallel_worthwhile`], so there is no other way
/// to force both branches over identical data.
fn ref_matmul_fill_parallel(
    out: &mut [f32],
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    reporter: &ProgressReporter,
) {
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        matmul_row(row, &a[i * k..i * k + k], b, k, n);
        reporter.advance(1);
    });
}

/// Naive `[m, k] x [k, n] -> [m, n]` row-major matmul.
///
/// `O(m*k*n)` with an `f64` accumulator per output element — deliberately
/// unoptimised; this exists to be obviously correct, not fast, and is only
/// ever called behind [`verify_enabled`]. Above `PAR_MIN_MACS` total
/// multiply-adds, the `m` output rows are split across `rayon` (see
/// `ref_matmul_fill_parallel` and the [module-level parallelism
/// note](self)); below it, runs as a single serial loop with no `rayon`
/// involvement at all (`ref_matmul_fill_serial`).
#[must_use]
pub fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    if n == 0 {
        return out;
    }

    let total_macs = (m as u64).saturating_mul(k as u64).saturating_mul(n as u64);
    if parallel_worthwhile(total_macs) {
        let reporter =
            ProgressReporter::new("MatMul", format!("[{m},{k}]x[{k},{n}]->[{m},{n}]"), m);
        ref_matmul_fill_parallel(&mut out, a, b, k, n, &reporter);
    } else {
        ref_matmul_fill_serial(&mut out, a, b, k, n);
    }
    out
}

/// Per-row geometry for [`ref_conv`]'s naive cross-correlation: everything
/// that stays fixed across every output row of one call, resolved once so
/// [`conv_row`] only does index arithmetic and never re-derives strides or
/// group boundaries. Also carries the output shape (`n`/`out_channels`/
/// `out_h`/`out_w`) so [`ref_conv_fill_serial`]/[`ref_conv_fill_parallel`]
/// need no parameters beyond `out` itself and this struct.
#[derive(Clone, Copy)]
struct ConvGeometry {
    n: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    out_h: usize,
    out_w: usize,
    in_ch_per_group: usize,
    filter_h: usize,
    filter_w: usize,
    out_ch_per_group: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
}

/// The operands [`conv_row`] needs, bundled so its own parameter list stays
/// under `clippy::too_many_arguments` -- every field is a borrow or a
/// `Copy` value, so grouping them costs nothing at the call site.
struct ConvRowInputs<'a> {
    input: &'a [f32],
    weight: &'a [f32],
    bias: Option<&'a [f32]>,
    geom: ConvGeometry,
}

/// Compute one output row `out[n=ni, k=ki, h=oh, 0..out_w]` of a naive NCHW
/// cross-correlation.
///
/// The single place the per-element formula is written down -- [`ref_conv`]'s
/// serial (small-shape) and `rayon`-parallel (large-shape) branches both
/// call exactly this function, once per `(ni, ki, oh)`, so there is no
/// second copy that could silently drift from the first. Each output
/// element's own `f64` accumulation order (`cg` outermost, then `ri`, then
/// `si` -- byte for byte the original fully-serial loop nest) is
/// unaffected by which branch calls it or which thread runs it: only
/// *which rows* run concurrently changes, never the arithmetic inside one
/// row.
#[allow(clippy::needless_range_loop)]
fn conv_row(row_out: &mut [f32], ops: &ConvRowInputs<'_>, ni: usize, ki: usize, oh: usize) {
    let geom = &ops.geom;
    // Which group this output channel belongs to -- `weight`'s leading
    // `[K, ...]` dim is laid out group-major (the first `out_ch_per_group`
    // filters belong to group 0, the next `out_ch_per_group` to group 1,
    // and so on), matching ONNX's `Conv` spec and `oxicuda_dnn`'s own
    // kernel bodies.
    let g = ki / geom.out_ch_per_group;
    for (ow, out_val) in row_out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for cg in 0..geom.in_ch_per_group {
            let ci = g * geom.in_ch_per_group + cg;
            for ri in 0..geom.filter_h {
                // Implicit zero-padding: an input coordinate that lands
                // outside `[0, in_h)`/`[0, in_w)` contributes nothing,
                // rather than being an error -- this is what makes the
                // padding "same"-style output sizes correct.
                let ih = oh as isize * geom.stride_h as isize - geom.pad_h as isize
                    + ri as isize * geom.dil_h as isize;
                if ih < 0 || ih as usize >= geom.in_h {
                    continue;
                }
                let ih = ih as usize;
                for si in 0..geom.filter_w {
                    let iw = ow as isize * geom.stride_w as isize - geom.pad_w as isize
                        + si as isize * geom.dil_w as isize;
                    if iw < 0 || iw as usize >= geom.in_w {
                        continue;
                    }
                    let iw = iw as usize;
                    let in_idx = ((ni * geom.in_channels + ci) * geom.in_h + ih) * geom.in_w + iw;
                    let f_idx = ((ki * geom.in_ch_per_group + cg) * geom.filter_h + ri)
                        * geom.filter_w
                        + si;
                    acc += f64::from(ops.input[in_idx]) * f64::from(ops.weight[f_idx]);
                }
            }
        }
        if let Some(bv) = ops.bias {
            acc += f64::from(bv[ki]);
        }
        *out_val = acc as f32;
    }
}

/// Fill every row of `out` serially -- no `rayon` involvement at all. See
/// [`ref_conv`].
fn ref_conv_fill_serial(out: &mut [f32], ops: &ConvRowInputs<'_>) {
    let geom = &ops.geom;
    for ni in 0..geom.n {
        for ki in 0..geom.out_channels {
            let row_base = (ni * geom.out_channels + ki) * geom.out_h;
            for oh in 0..geom.out_h {
                let row = &mut out[(row_base + oh) * geom.out_w..(row_base + oh + 1) * geom.out_w];
                conv_row(row, ops, ni, ki, oh);
            }
        }
    }
}

/// Fill every row of `out` via a `rayon` split across `(ni, ki, oh)` rows.
/// See [`ref_conv`].
///
/// Exposed as its own function (rather than inlined into [`ref_conv`]) so
/// the `*_parallel_matches_serial_on_*` property tests below can call it
/// directly against [`ref_conv_fill_serial`] on the *same* large shape --
/// [`ref_conv`] itself only ever runs one branch or the other for a given
/// shape, decided by [`parallel_worthwhile`], so there is no other way to
/// force both branches over identical data.
fn ref_conv_fill_parallel(out: &mut [f32], ops: &ConvRowInputs<'_>, reporter: &ProgressReporter) {
    let geom = ops.geom;
    out.par_chunks_mut(geom.out_w)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let oh = row_idx % geom.out_h;
            let ki = (row_idx / geom.out_h) % geom.out_channels;
            let ni = row_idx / (geom.out_h * geom.out_channels);
            conv_row(row, ops, ni, ki, oh);
            reporter.advance(1);
        });
}

/// Naive NCHW cross-correlation (the `cuDNN`/ONNX `Conv` convention — no
/// 180-degree kernel flip), honouring padding, stride, dilation, and
/// groups, with an optional per-output-channel bias.
///
/// `input` is `[N, C, H, W]` (`in_shape`), `weight` is `[K, C/groups, R, S]`
/// (`weight_shape`) — the exact layout the ONNX `Conv` operator (and
/// [`crate::conv::cuda_conv`], which uploads this same layout to the GPU)
/// uses. `bias`, when present, is `[K]`, added once per output channel and
/// broadcast across every batch and spatial position.
///
/// Only `params.pads`' first two elements (`[pad_top, pad_left]`) are read.
/// That is not a shortcut: every call site in this crate reaches this
/// oracle only after [`crate::conv::cuda_conv`] has already declined any
/// node whose padding is asymmetric (`pads[0] != pads[2] || pads[1] !=
/// pads[3]` — see the [`crate::conv`] module docs' "What still declines"
/// section), so `pads[2]`/`pads[3]` are guaranteed to equal
/// `pads[0]`/`pads[1]` by the time this oracle ever runs in this crate.
///
/// `O(N*K*P*Q*(C/groups)*R*S)` with an `f64` accumulator per output
/// element, cast to `f32` only once, at the very end — deliberately
/// unoptimised, the same discipline as [`ref_matmul`]: this exists to be
/// obviously correct, not fast. Above `PAR_MIN_MACS` total multiply-adds,
/// the `N*K*out_h` output rows are split across `rayon`
/// (`ref_conv_fill_parallel`, see also the [module-level parallelism
/// note](self)); below it, runs as a single serial loop with no `rayon`
/// involvement at all (`ref_conv_fill_serial`).
///
/// # Panics
/// Indexes `input`/`weight`/`bias`/`in_shape`/`weight_shape` directly with
/// no bounds- or length-checking, the same discipline as [`ref_matmul`]: a
/// caller-supplied shape/data-length mismatch (or a `params.group` of `0`)
/// panics rather than silently computing garbage. Every call site in this
/// crate derives its arguments from a `ConvProblem` that
/// [`crate::conv::cuda_conv`] has already validated end-to-end (rank-4
/// shapes, non-zero dims, `group` dividing both channel counts, the filter
/// fitting the padded input) before this oracle ever runs, so a panic here
/// means the caller passed shapes it never validated — a bug in the caller,
/// not a normal outcome.
#[must_use]
pub fn ref_conv(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_shape: &[usize],
    weight_shape: &[usize],
    params: &ConvParams,
) -> Vec<f32> {
    let n = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let out_channels = weight_shape[0];
    let in_ch_per_group = weight_shape[1];
    let filter_h = weight_shape[2];
    let filter_w = weight_shape[3];

    let [stride_h, stride_w] = params.strides;
    let [pad_h, pad_w, _pad_bottom, _pad_right] = params.pads;
    let [dil_h, dil_w] = params.dilations;
    let group = params.group;
    let out_ch_per_group = out_channels / group;

    // Standard convolution output-size formula (matches
    // `oxicuda_dnn::conv::descriptor::ConvProblem::output_dims`'s
    // `ConvolutionDescriptor::output_size`, and this crate's own
    // `conv::tests::gpu_numeric::ConvCase::out_hw`):
    // `floor((dim + 2*pad - dilation*(k-1) - 1) / stride) + 1`.
    let eff_h = dil_h * (filter_h - 1) + 1;
    let eff_w = dil_w * (filter_w - 1) + 1;
    let out_h = (in_h + 2 * pad_h - eff_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - eff_w) / stride_w + 1;

    let mut out = vec![0.0_f32; n * out_channels * out_h * out_w];
    if out_w == 0 {
        // Only reachable when `out` is already empty (a 0-width row can
        // only coexist with a 0-length `out`, since `out.len()` is a
        // product that includes `out_w`) -- guarded explicitly, before the
        // `par_chunks_mut(out_w)` below, because `chunks_mut(0)` panics
        // even on an empty slice, where a 0-length `out` would otherwise be
        // the (already correct) answer.
        return out;
    }

    let ops = ConvRowInputs {
        input,
        weight,
        bias,
        geom: ConvGeometry {
            n,
            in_channels,
            in_h,
            in_w,
            out_channels,
            out_h,
            out_w,
            in_ch_per_group,
            filter_h,
            filter_w,
            out_ch_per_group,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        },
    };

    let total_macs = (n as u64)
        .saturating_mul(out_channels as u64)
        .saturating_mul(out_h as u64)
        .saturating_mul(out_w as u64)
        .saturating_mul(in_ch_per_group as u64)
        .saturating_mul(filter_h as u64)
        .saturating_mul(filter_w as u64);

    if parallel_worthwhile(total_macs) {
        let reporter = ProgressReporter::new(
            "Conv",
            format!(
                "[{n},{in_channels},{in_h},{in_w}]*[{out_channels},{in_ch_per_group},\
                 {filter_h},{filter_w}]->[{n},{out_channels},{out_h},{out_w}]"
            ),
            n * out_channels * out_h,
        );
        ref_conv_fill_parallel(&mut out, &ops, &reporter);
    } else {
        ref_conv_fill_serial(&mut out, &ops);
    }

    // The optimizer's fused activation is part of what the *node* computes,
    // not a separate node — see [`crate::conv::ConvActivation`]. An oracle
    // that skipped it would agree with a GPU dispatch that also skipped it,
    // which is exactly how a dropped `Relu` survived `OXIONNX_CUDA_VERIFY=1`
    // and corrupted every SCRFD detection while every verified node "passed".
    apply_conv_activation_ref(&mut out, params.activation);
    out
}

/// The oracle's own implementation of the fused activation [`ref_conv`]
/// applies.
///
/// Written from the ONNX / `oxionnx-ops` semantics rather than by calling
/// `crate::conv`'s dispatch-side helper, for the same reason [`ref_conv`] does
/// not call the CUDA kernel: an oracle that shares an implementation with the
/// thing it checks checks nothing.
fn apply_conv_activation_ref(out: &mut [f32], activation: ConvActivation) {
    match activation {
        ConvActivation::None => {}
        ConvActivation::Relu => {
            for v in out.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        ConvActivation::Clip { min, max } => {
            // ONNX `Clip`: a NaN bound is no bound on that side, and an
            // inverted `[min, max]` leaves the data alone.
            let lo = if min.is_nan() { f32::NEG_INFINITY } else { min };
            let hi = if max.is_nan() { f32::INFINITY } else { max };
            if lo > hi {
                return;
            }
            for v in out.iter_mut() {
                if *v < lo {
                    *v = lo;
                } else if *v > hi {
                    *v = hi;
                }
            }
        }
    }
}

/// One `ReduceSum` output element at flattened `(o, i)` -- see [`ref_reduce`]
/// for the `[outer, axis_len, inner]` layout this indexes into.
fn reduce_sum_at(data: &[f32], axis_len: usize, inner: usize, o: usize, i: usize) -> f32 {
    let mut acc = 0.0_f64;
    for a in 0..axis_len {
        acc += f64::from(data[(o * axis_len + a) * inner + i]);
    }
    acc as f32
}

/// One `ReduceMax` output element. See [`reduce_sum_at`].
fn reduce_max_at(data: &[f32], axis_len: usize, inner: usize, o: usize, i: usize) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for a in 0..axis_len {
        acc = acc.max(data[(o * axis_len + a) * inner + i]);
    }
    acc
}

/// One `ReduceMean` output element: the same `f64` accumulation as
/// [`reduce_sum_at`], divided by `axis_len` *before* the single cast to
/// `f32` — not `reduce_sum_at(..) / axis_len as f32`, which would round
/// twice. `axis_len` is always `>= 1` here (`ref_reduce` already declined a
/// zero-length axis before this is ever called), so the division is exact
/// arithmetic, never by zero.
#[allow(clippy::cast_precision_loss)]
fn reduce_mean_at(data: &[f32], axis_len: usize, inner: usize, o: usize, i: usize) -> f32 {
    let mut acc = 0.0_f64;
    for a in 0..axis_len {
        acc += f64::from(data[(o * axis_len + a) * inner + i]);
    }
    (acc / axis_len as f64) as f32
}

/// Fill every output element of `out` serially -- no `rayon` involvement at
/// all. See [`ref_reduce`].
fn ref_reduce_fill_serial(
    out: &mut [f32],
    data: &[f32],
    compute: fn(&[f32], usize, usize, usize, usize) -> f32,
    axis_len: usize,
    inner: usize,
    outer: usize,
) {
    for o in 0..outer {
        for (i, out_val) in out[o * inner..(o + 1) * inner].iter_mut().enumerate() {
            *out_val = compute(data, axis_len, inner, o, i);
        }
    }
}

/// Fill every output element of `out` via a `rayon` split into flat chunks
/// of `chunk_len` elements. See [`ref_reduce`].
///
/// Exposed as its own function (rather than inlined into [`ref_reduce`]),
/// and taking `chunk_len` as a parameter rather than deriving it internally
/// the way [`ref_conv_fill_parallel`]/[`ref_matmul_fill_parallel`] do, so
/// the `*_parallel_matches_serial_on_*` property tests below can call it
/// directly against [`ref_reduce_fill_serial`] on the *same* large shape at
/// several different chunk sizes -- the per-element `compute` call does not
/// depend on chunk boundaries at all, so this doubles as a check that the
/// result is chunk-size-invariant, not just "matches one specific `rayon`
/// config".
fn ref_reduce_fill_parallel(
    out: &mut [f32],
    data: &[f32],
    compute: fn(&[f32], usize, usize, usize, usize) -> f32,
    axis_len: usize,
    inner: usize,
    chunk_len: usize,
    reporter: &ProgressReporter,
) {
    out.par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * chunk_len;
            for (offset, out_val) in chunk.iter_mut().enumerate() {
                let idx = base + offset;
                *out_val = compute(data, axis_len, inner, idx / inner, idx % inner);
            }
            reporter.advance(1);
        });
}

/// Naive per-axis `ReduceSum` / `ReduceMax` / `ReduceMean`.
///
/// `shape` is decomposed as `[outer, axis_len, inner]` around `axis`,
/// matching [`crate::reduce::cuda_reduce`]'s own layout. A multi-axis
/// `ReduceMean` (`crate::reduce::cuda_reduce_mean_bound`, one or more
/// *contiguous* axes) calls this with a synthetic 3-element `shape` and
/// `axis = 1` — the merged-axis view [`crate::reduce::reduce_plan_range`]'s
/// docs describe — rather than this function growing a second axis
/// parameter. Returns `None` for an out-of-range axis or an op this oracle
/// has no formula for (the caller treats that as "skip the check", not "the
/// GPU is wrong"). Above
/// `PAR_MIN_MACS` total multiply-adds, the `outer*inner` output elements
/// are split across `rayon` in `rayon::current_num_threads() * 4`-ish flat
/// chunks (`ref_reduce_fill_parallel` -- finer than "one chunk per `o`",
/// so a reduce over a small `outer` with a large `inner` -- or vice versa --
/// still parallelises well); below it, runs as a single serial loop with no
/// `rayon` involvement at all (`ref_reduce_fill_serial`).
#[must_use]
pub fn ref_reduce(op: &OpKind, data: &[f32], shape: &[usize], axis: usize) -> Option<Vec<f32>> {
    if axis >= shape.len() {
        return None;
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    if outer == 0 || axis_len == 0 || inner == 0 {
        return None;
    }

    // Resolved once, outside every loop below, rather than re-matched on
    // `op` for every one of `outer*inner` output elements: a bare function
    // pointer costs nothing to call (no captured state, `Send + Sync` for
    // free), and re-matching on every element is exactly the per-element
    // cost the original code paid.
    let compute: fn(&[f32], usize, usize, usize, usize) -> f32 = match op {
        OpKind::ReduceSum => reduce_sum_at,
        OpKind::ReduceMax => reduce_max_at,
        OpKind::ReduceMean => reduce_mean_at,
        _ => return None,
    };

    let total = outer * inner;
    let mut out = vec![0.0_f32; total];
    let total_macs = (total as u64).saturating_mul(axis_len as u64);

    if parallel_worthwhile(total_macs) {
        // Chunked flat over `total`, not by `inner` (one chunk per `o`):
        // unlike a conv/matmul row, a reduce output element's cost does not
        // depend on its position, so a flat chunking that ignores the
        // `outer`/`inner` split costs nothing in locality and stays
        // balanced even when `outer` is small (e.g. axis 0) and `inner` is
        // large, where "one chunk per `o`" would degenerate to a single
        // chunk.
        let chunk_len = total
            .div_ceil((rayon::current_num_threads() * 4).max(1))
            .max(1);
        let reporter = ProgressReporter::new(
            "Reduce",
            format!("outer={outer} axis_len={axis_len} inner={inner}"),
            total.div_ceil(chunk_len),
        );
        ref_reduce_fill_parallel(
            &mut out, data, compute, axis_len, inner, chunk_len, &reporter,
        );
    } else {
        ref_reduce_fill_serial(&mut out, data, compute, axis_len, inner, outer);
    }
    Some(out)
}

/// One softmax row: max-subtraction, `f64`-accumulated `exp`-sum, then
/// normalise.
///
/// `exps` is scratch space of length `in_row.len()`, supplied by the
/// caller so [`ref_softmax`]'s serial branch allocates it once for the
/// whole call and its `rayon`-parallel branch allocates it once per worker
/// (via `for_each_init`) rather than once per row.
fn softmax_row(out_row: &mut [f32], in_row: &[f32], exps: &mut [f64]) {
    let max = in_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for (i, &x) in in_row.iter().enumerate() {
        let e = f64::from(x - max).exp();
        exps[i] = e;
        sum += e;
    }
    for (i, &e) in exps.iter().enumerate() {
        out_row[i] = (e / sum) as f32;
    }
}

/// Fill every row of `out` serially -- no `rayon` involvement at all. See
/// [`ref_softmax`].
fn ref_softmax_fill_serial(out: &mut [f32], data: &[f32], row: usize, rows: usize) {
    let mut exps = vec![0.0_f64; row];
    for r in 0..rows {
        let base = r * row;
        softmax_row(
            &mut out[base..base + row],
            &data[base..base + row],
            &mut exps,
        );
    }
}

/// Fill every row of `out` via a `rayon` split across rows. See
/// [`ref_softmax`].
///
/// Exposed as its own function (rather than inlined into [`ref_softmax`])
/// so the `*_parallel_matches_serial_on_*` property tests below can call it
/// directly against [`ref_softmax_fill_serial`] on the *same* large shape --
/// [`ref_softmax`] itself only ever runs one branch or the other for a
/// given shape, decided by its own size guard, so there is no other way to
/// force both branches over identical data.
fn ref_softmax_fill_parallel(
    out: &mut [f32],
    data: &[f32],
    row: usize,
    reporter: &ProgressReporter,
) {
    out.par_chunks_mut(row)
        .zip(data.par_chunks(row))
        .for_each_init(
            || vec![0.0_f64; row],
            |exps, (out_row, in_row)| {
                softmax_row(out_row, in_row, exps);
                reporter.advance(1);
            },
        );
}

/// Naive Softmax over the last dimension: the standard
/// max-subtraction-then-normalise formula, `f64` accumulation for the
/// denominator.
///
/// `shape` must be non-empty (mirrors [`crate::softmax::cuda_softmax`]'s own
/// precondition; the caller already declined an empty shape before this
/// would ever run). Above `PAR_MIN_ELEMENTWISE_LEN` total elements (and
/// given at least two rows to split across), the rows are independent and
/// split across `rayon` (`ref_softmax_fill_parallel`); below it, runs as
/// a single serial loop with no `rayon` involvement at all
/// (`ref_softmax_fill_serial`).
#[must_use]
pub fn ref_softmax(data: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
    let &row = shape.last()?;
    if row == 0 {
        return Some(Vec::new());
    }
    let rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    if rows.checked_mul(row) != Some(data.len()) {
        return None;
    }

    let mut out = vec![0.0_f32; data.len()];
    if rows < 2 || data.len() < PAR_MIN_ELEMENTWISE_LEN || rayon::current_num_threads() < 2 {
        ref_softmax_fill_serial(&mut out, data, row, rows);
        return Some(out);
    }

    let reporter = ProgressReporter::new("Softmax", format!("rows={rows} row_len={row}"), rows);
    ref_softmax_fill_parallel(&mut out, data, row, &reporter);
    Some(out)
}

/// The elementwise unary formula matching exactly what the corresponding
/// `oxicuda_ptx` kernel computes — see the [module docs](self) for why this
/// is not always the plain ONNX-spec formula.
///
/// `f64` intermediate arithmetic throughout; the final cast to `f32` is the
/// only place precision is dropped, so the oracle's own rounding never
/// competes with the GPU's approximate transcendentals for `compare`'s
/// (crate-private) tolerance budget.
///
/// Returns `None` for any `op` this oracle has no formula for.
#[must_use]
pub fn ref_unary(op: &OpKind, x: f32) -> Option<f32> {
    let xf = f64::from(x);
    let y: f64 = match op {
        OpKind::Relu => xf.max(0.0),
        OpKind::Sigmoid => 1.0 / (1.0 + (-xf).exp()),
        // GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) — the tanh
        // approximation, matching `oxicuda_ptx::generate_gelu`'s own doc comment, NOT
        // ONNX's exact/erf default.
        OpKind::Gelu => {
            let x3 = xf * xf * xf;
            let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (xf + 0.044_715 * x3);
            0.5 * xf * (1.0 + inner.tanh())
        }
        OpKind::Tanh => xf.tanh(),
        OpKind::Exp => xf.exp(),
        OpKind::Sqrt => xf.sqrt(),
        OpKind::Abs => xf.abs(),
        OpKind::Neg => -xf,
        OpKind::Log => xf.ln(),
        OpKind::Ceil => xf.ceil(),
        OpKind::Floor => xf.floor(),
        // clamp(0.2*x + 0.5, 0, 1) — ONNX's default alpha/beta, hard-coded in the kernel.
        OpKind::HardSigmoid => (0.2 * xf + 0.5).clamp(0.0, 1.0),
        // x * clamp(x + 3, 0, 6) / 6.
        OpKind::HardSwish => xf * (xf + 3.0).clamp(0.0, 6.0) / 6.0,
        OpKind::SiLU => xf * (1.0 / (1.0 + (-xf).exp())),
        OpKind::Softplus => (1.0 + xf.exp()).ln(),
        // x >= 0 ? x : 0.01*x — ONNX's default alpha, hard-coded in the kernel.
        OpKind::LeakyRelu => {
            if xf >= 0.0 {
                xf
            } else {
                0.01 * xf
            }
        }
        _ => return None,
    };
    Some(y as f32)
}

/// Map [`ref_unary`] over every element of `data`, in parallel above
/// `PAR_MIN_ELEMENTWISE_LEN` elements.
///
/// `None` if `op` has no formula — `Option<Vec<_>>`'s `FromIterator` (below
/// the threshold) or `FromParallelIterator` (above it) impl short-circuits
/// the whole collection to `None` on the first element [`ref_unary`] cannot
/// compute, rather than silently passing that element through unchanged.
///
/// No `ProgressReporter` here (unlike `ref_conv`/`ref_matmul`/
/// `ref_reduce`/`ref_softmax`): every [`ref_unary`] formula is a handful of
/// scalar FLOPs, so even the largest real activation tensor in this
/// pipeline (InSwapper's ~1.18M-element feature maps) finishes in tens of
/// milliseconds — far under `PROGRESS_LOG_INTERVAL_NANOS`, so a progress
/// line could structurally never fire and the plumbing would be dead
/// weight.
#[must_use]
pub fn ref_unary_vec(op: &OpKind, data: &[f32]) -> Option<Vec<f32>> {
    if data.len() < PAR_MIN_ELEMENTWISE_LEN || rayon::current_num_threads() < 2 {
        return data.iter().map(|&x| ref_unary(op, x)).collect();
    }
    data.par_iter().map(|&x| ref_unary(op, x)).collect()
}

/// The elementwise binary formula for `Add` / `Sub` / `Mul` / `Div`.
///
/// Returns `None` for any other `op`.
#[must_use]
pub fn ref_binary(op: &OpKind, a: f32, b: f32) -> Option<f32> {
    match op {
        OpKind::Add => Some(a + b),
        OpKind::Sub => Some(a - b),
        OpKind::Mul => Some(a * b),
        OpKind::Div => Some(a / b),
        _ => None,
    }
}

/// Map [`ref_binary`] over two equal-length operand slices, in parallel
/// above `PAR_MIN_ELEMENTWISE_LEN` elements.
///
/// `None` if the lengths disagree or `op` has no formula. See
/// [`ref_unary_vec`] for why this has no `ProgressReporter`.
#[must_use]
pub fn ref_binary_vec(op: &OpKind, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    if a.len() != b.len() {
        return None;
    }
    if a.len() < PAR_MIN_ELEMENTWISE_LEN || rayon::current_num_threads() < 2 {
        return a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| ref_binary(op, x, y))
            .collect();
    }
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| ref_binary(op, x, y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── verify_enabled / parse_env_flag plumbing ───────────────────────────

    #[test]
    fn verify_enabled_never_panics() {
        let _ = verify_enabled();
    }

    // ── nearly_eq / compare ─────────────────────────────────────────────────

    #[test]
    fn nearly_eq_accepts_exact_and_small_rounding_drift() {
        assert!(nearly_eq(1.0, 1.0));
        assert!(nearly_eq(0.0, 0.0));
        assert!(nearly_eq(1.000_05, 1.0)); // well within ATOL.
        assert!(nearly_eq(1000.5, 1000.0)); // within RTOL of a large magnitude.
    }

    #[test]
    fn nearly_eq_rejects_a_genuine_mismatch() {
        assert!(!nearly_eq(1.0, 2.0));
        assert!(!nearly_eq(0.0, 1.0));
    }

    #[test]
    fn nearly_eq_treats_matching_non_finite_values_as_agreement() {
        assert!(nearly_eq(f32::NAN, f32::NAN));
        assert!(nearly_eq(f32::INFINITY, f32::INFINITY));
        assert!(nearly_eq(f32::NEG_INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn nearly_eq_rejects_mismatched_non_finite_values() {
        assert!(!nearly_eq(f32::NAN, 1.0));
        assert!(!nearly_eq(1.0, f32::NAN));
        assert!(!nearly_eq(f32::INFINITY, f32::NEG_INFINITY));
        assert!(!nearly_eq(f32::INFINITY, 1.0e30));
    }

    #[test]
    fn compare_reports_the_first_mismatched_index() {
        let gpu = [1.0_f32, 2.0, 99.0, 4.0];
        let cpu = [1.0_f32, 2.0, 3.0, 4.0];
        let err = compare(&gpu, &cpu).unwrap_err();
        assert!(err.contains("element 2"), "got: {err}");
        assert!(err.contains("99"), "got: {err}");
    }

    #[test]
    fn compare_reports_a_length_mismatch_distinctly() {
        let err = compare(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(err.contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn compare_passes_identical_slices() {
        let v = [1.0_f32, -2.5, 0.0, 3.75];
        assert!(compare(&v, &v).is_ok());
    }

    // ── ref_matmul ───────────────────────────────────────────────────────────

    #[test]
    fn ref_matmul_identity() {
        // [[1,0],[0,1]] @ [[5,6],[7,8]] = [[5,6],[7,8]].
        let a = [1.0_f32, 0.0, 0.0, 1.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        assert_eq!(ref_matmul(&a, &b, 2, 2, 2), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn ref_matmul_hand_verified_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2).
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [28+45+66, 32+50+72] = [139, 154]
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(ref_matmul(&a, &b, 2, 3, 2), vec![58.0, 64.0, 139.0, 154.0]);
    }

    // ── ref_conv ─────────────────────────────────────────────────────────────
    //
    // Every expected output below was hand-derived AND independently
    // cross-checked with a from-scratch Python re-implementation of the same
    // naive algorithm before being pasted in here, catching (during that
    // cross-check) one hand-arithmetic mistake of my own on a discarded
    // extra case -- a reminder of exactly why this whole module exists.

    fn unit_params() -> ConvParams {
        ConvParams {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
            group: 1,
            activation: ConvActivation::None,
        }
    }

    #[test]
    fn ref_conv_is_cross_correlation_not_true_convolution() {
        // input (3x3): [[1,2,3],[4,5,6],[7,8,9]], weight (2x2): [[1,2],[3,4]].
        // Deliberately asymmetric so a 180-degree kernel flip would change
        // the answer -- proving this computes cross-correlation (the ONNX
        // `Conv` / cuDNN convention), not textbook convolution.
        //   out[0][0] = 1*1+2*2+4*3+5*4 = 1+4+12+20 = 37
        //   out[0][1] = 2*1+3*2+5*3+6*4 = 2+6+15+24 = 47
        //   out[1][0] = 4*1+5*2+7*3+8*4 = 4+10+21+32 = 67
        //   out[1][1] = 5*1+6*2+8*3+9*4 = 5+12+24+36 = 77
        // (a flipped kernel [[4,3],[2,1]] would instead give 23 at [0][0].)
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0_f32, 2.0, 3.0, 4.0];
        let out = ref_conv(
            &input,
            &weight,
            None,
            &[1, 1, 3, 3],
            &[1, 1, 2, 2],
            &unit_params(),
        );
        assert_eq!(out, vec![37.0, 47.0, 67.0, 77.0]);
    }

    #[test]
    fn ref_conv_adds_bias_once_per_output_channel() {
        // Same geometry as above, plus bias=[100] on the single output channel:
        // every element of the no-bias result shifts by exactly 100.
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0_f32, 2.0, 3.0, 4.0];
        let bias = [100.0_f32];
        let out = ref_conv(
            &input,
            &weight,
            Some(&bias),
            &[1, 1, 3, 3],
            &[1, 1, 2, 2],
            &unit_params(),
        );
        assert_eq!(out, vec![137.0, 147.0, 167.0, 177.0]);
    }

    #[test]
    fn ref_conv_honours_stride() {
        // 5x5 input (row-major 1..25), 2x2 kernel [[1,0],[0,1]], stride=2, no
        // padding: out[oh][ow] = in[2*oh][2*ow] + in[2*oh+1][2*ow+1].
        //   out[0][0] = in[0][0]+in[1][1] =  1+ 7 =  8
        //   out[0][1] = in[0][2]+in[1][3] =  3+ 9 = 12
        //   out[1][0] = in[2][0]+in[3][1] = 11+17 = 28
        //   out[1][1] = in[2][2]+in[3][3] = 13+19 = 32
        let input: Vec<f32> = (1..=25).map(|v| v as f32).collect();
        let weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut params = unit_params();
        params.strides = [2, 2];
        let out = ref_conv(&input, &weight, None, &[1, 1, 5, 5], &[1, 1, 2, 2], &params);
        assert_eq!(out, vec![8.0, 12.0, 28.0, 32.0]);
    }

    #[test]
    fn ref_conv_zero_pads_the_border_with_correct_offset_direction() {
        // 2x2 input [[1,2],[3,4]], 3x3 kernel that is a one-hot at tap
        // (r=0, s=0) (all other taps zero), pad=1, stride=1 ("same" output
        // size, 2x2). With `ih = oh - 1 + 0` / `iw = ow - 1 + 0`, the
        // top-left tap only ever lands inside the real 2x2 image (rather
        // than the zero border) when oh=ow=1, reading `input[0][0]=1`;
        // every other output position reads purely padding, i.e. 0. This
        // pins down both that out-of-range taps contribute zero AND that
        // `pad` is subtracted (not added) when computing the input
        // coordinate -- a sign error here would shift which corner is
        // nonzero instead of merely failing everywhere.
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let weight = [
            1.0_f32, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let mut params = unit_params();
        params.pads = [1, 1, 1, 1];
        let out = ref_conv(&input, &weight, None, &[1, 1, 2, 2], &[1, 1, 3, 3], &params);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn ref_conv_honours_dilation() {
        // Same 5x5 input as the stride test, 2x2 kernel [[1,0],[0,1]],
        // dilation=2 (effective kernel size 3), stride=1, no padding -> 3x3
        // output. out[oh][ow] = in[oh][ow] + in[oh+2][ow+2] (the dilated tap
        // skips one row/col instead of the stride test's adjacent one):
        //   out[0][0]=in[0][0]+in[2][2]= 1+13=14   out[0][1]= 2+14=16   out[0][2]= 3+15=18
        //   out[1][0]=in[1][0]+in[3][2]= 6+18=24   out[1][1]= 7+19=26   out[1][2]= 8+20=28
        //   out[2][0]=in[2][0]+in[4][2]=11+23=34   out[2][1]=12+24=36   out[2][2]=13+25=38
        let input: Vec<f32> = (1..=25).map(|v| v as f32).collect();
        let weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut params = unit_params();
        params.dilations = [2, 2];
        let out = ref_conv(&input, &weight, None, &[1, 1, 5, 5], &[1, 1, 2, 2], &params);
        assert_eq!(
            out,
            vec![14.0, 16.0, 18.0, 24.0, 26.0, 28.0, 34.0, 36.0, 38.0]
        );
    }

    #[test]
    fn ref_conv_honours_groups_keeping_each_groups_channels_independent() {
        // 1x1 spatial, in_channels=4, out_channels=4, groups=2: group 0 owns
        // input channels [0,1] and output channels [0,1]; group 1 owns input
        // channels [2,3] and output channels [2,3]. input=[1,2,3,4],
        // weight[k][cg] = k0:[1,1] k1:[2,2] k2:[3,3] k3:[4,4].
        //   out[0] = in[0]*1+in[1]*1 = 1+2 =  3   (group 0, sees channels 0,1)
        //   out[1] = in[0]*2+in[1]*2 = 2+4 =  6   (group 0, sees channels 0,1)
        //   out[2] = in[2]*3+in[3]*3 = 9+12= 21   (group 1, sees channels 2,3)
        //   out[3] = in[2]*4+in[3]*4 =12+16= 28   (group 1, sees channels 2,3)
        // A broken group offset (e.g. every output channel reading input
        // channels [0,1] regardless of its group) would instead give
        // [3, 6, 9, 12] for out[2..4] -- a different, wrong answer, so this
        // discriminates a real groups bug rather than passing vacuously.
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let mut params = unit_params();
        params.group = 2;
        let out = ref_conv(&input, &weight, None, &[1, 4, 1, 1], &[4, 2, 1, 1], &params);
        assert_eq!(out, vec![3.0, 6.0, 21.0, 28.0]);
    }

    // ── ref_reduce ───────────────────────────────────────────────────────────

    #[test]
    fn ref_reduce_sum_whole_1d_tensor_over_256_elements() {
        // The exact motivating shape from finding a8-1: a 1-D axis longer than the old
        // 256-thread block size. data[i] = i, sum_{i=0}^{1023} i = 1023*1024/2 = 523776.
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let out = ref_reduce(&OpKind::ReduceSum, &data, &[1024], 0).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 523_776.0).abs() < 1.0);
    }

    #[test]
    fn ref_reduce_max_over_a_middle_axis() {
        // shape [2, 3, 2], axis=1 (outer=2, axis_len=3, inner=2).
        // data laid out row-major: [[[0,1],[2,3],[4,5]], [[6,7],[8,9],[10,11]]]
        // max over axis 1 -> [[4,5],[10,11]]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = ref_reduce(&OpKind::ReduceMax, &data, &[2, 3, 2], 1).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn ref_reduce_declines_out_of_range_axis_and_unknown_op() {
        assert_eq!(ref_reduce(&OpKind::ReduceSum, &[1.0, 2.0], &[2], 5), None);
        assert_eq!(ref_reduce(&OpKind::Relu, &[1.0, 2.0], &[2], 0), None);
    }

    #[test]
    fn ref_reduce_mean_over_a_middle_axis() {
        // Same shape/data as `ref_reduce_max_over_a_middle_axis`: [2,3,2],
        // axis=1. mean over axis 1 of [[0,1],[2,3],[4,5]] is [2,3]; of
        // [[6,7],[8,9],[10,11]] is [8,9].
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = ref_reduce(&OpKind::ReduceMean, &data, &[2, 3, 2], 1).unwrap();
        assert_eq!(out, vec![2.0, 3.0, 8.0, 9.0]);
    }

    #[test]
    fn ref_reduce_mean_whole_1d_tensor_matches_sum_over_n() {
        // Same data as `ref_reduce_sum_whole_1d_tensor_over_256_elements`:
        // mean = 523776 / 1024 = 511.5.
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let out = ref_reduce(&OpKind::ReduceMean, &data, &[1024], 0).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 511.5).abs() < 1.0e-3, "got {}", out[0]);
    }

    /// The multi-axis path `crate::reduce::cuda_reduce_mean_bound` (and its
    /// `lib.rs` call site) actually uses: a synthetic `[outer, axis_len,
    /// inner]` shape standing in for a merged contiguous axis range, exactly
    /// as `OpKind::ReduceMean`'s dispatch arm builds it.
    #[test]
    fn ref_reduce_mean_over_a_synthetic_merged_axis_matches_hand_computation() {
        // [N=1,C=2,H=2,W=2] flattened to the OxiInstanceNorm decomposition's
        // view: outer=N*C=2, axis_len=H*W=4, inner=1.
        // Plane 0: [1,2,3,4] -> mean 2.5. Plane 1: [10,20,30,40] -> mean 25.
        let data = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = ref_reduce(&OpKind::ReduceMean, &data, &[2, 4, 1], 1).unwrap();
        assert_eq!(out, vec![2.5, 25.0]);
    }

    // ── ref_softmax ──────────────────────────────────────────────────────────

    #[test]
    fn ref_softmax_uniform_row_is_uniform() {
        let out = ref_softmax(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        for v in out {
            assert!((v - 0.25).abs() < 1.0e-6, "got {v}");
        }
    }

    #[test]
    fn ref_softmax_rows_sum_to_one() {
        let out = ref_softmax(&[1.0, 2.0, 3.0, -1.0, 0.0, 5.0], &[2, 3]).unwrap();
        let row0: f32 = out[0..3].iter().sum();
        let row1: f32 = out[3..6].iter().sum();
        assert!((row0 - 1.0).abs() < 1.0e-6, "row0 sums to {row0}");
        assert!((row1 - 1.0).abs() < 1.0e-6, "row1 sums to {row1}");
    }

    #[test]
    fn ref_softmax_is_shift_invariant_and_does_not_overflow() {
        // Large inputs would overflow a naive exp() without the max-subtraction; this must
        // not produce NaN/Inf.
        let out = ref_softmax(&[1000.0, 1000.0, 1000.0], &[3]).unwrap();
        for v in &out {
            assert!(
                v.is_finite(),
                "softmax must stay finite on large shifted inputs"
            );
        }
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
    }

    // ── ref_unary: hand-verified constants ──────────────────────────────────

    #[test]
    fn ref_unary_relu() {
        assert_eq!(ref_unary(&OpKind::Relu, -3.0), Some(0.0));
        assert_eq!(ref_unary(&OpKind::Relu, 3.0), Some(3.0));
    }

    #[test]
    fn ref_unary_leaky_relu_matches_the_hard_coded_kernel_alpha() {
        // oxicuda_ptx::generate_leaky_relu hard-codes alpha=0.01 (0f3C23D70A).
        assert_eq!(ref_unary(&OpKind::LeakyRelu, 2.0), Some(2.0));
        let y = ref_unary(&OpKind::LeakyRelu, -2.0).unwrap();
        assert!((y - (-0.02)).abs() < 1.0e-6, "got {y}");
    }

    #[test]
    fn ref_unary_hard_sigmoid_matches_the_hard_coded_kernel_constants() {
        // clamp(0.2*x + 0.5, 0, 1): x=0 -> 0.5; x=-10 -> clamp(-1.5,0,1)=0; x=10 -> clamp(2.5,0,1)=1.
        assert!((ref_unary(&OpKind::HardSigmoid, 0.0).unwrap() - 0.5).abs() < 1.0e-6);
        assert_eq!(ref_unary(&OpKind::HardSigmoid, -10.0), Some(0.0));
        assert_eq!(ref_unary(&OpKind::HardSigmoid, 10.0), Some(1.0));
    }

    #[test]
    fn ref_unary_sigmoid_hand_verified_at_zero() {
        let y = ref_unary(&OpKind::Sigmoid, 0.0).unwrap();
        assert!((y - 0.5).abs() < 1.0e-6, "sigmoid(0) must be 0.5, got {y}");
    }

    #[test]
    fn ref_unary_gelu_is_the_tanh_approximation_not_the_exact_form() {
        // At x=0 both the exact erf-based GELU and the tanh approximation give exactly 0,
        // so this checks a nonzero point where they'd disagree if the formula were wrong.
        // GELU_tanh(1.0) = 0.5*1*(1+tanh(sqrt(2/pi)*(1+0.044715))) ~= 0.8411919906...
        let y = ref_unary(&OpKind::Gelu, 1.0).unwrap();
        assert!((y - 0.841_192).abs() < 1.0e-4, "got {y}");
    }

    #[test]
    fn ref_unary_softplus_hand_verified_at_zero() {
        // softplus(0) = ln(1 + e^0) = ln(2) ~= 0.693147.
        let y = ref_unary(&OpKind::Softplus, 0.0).unwrap();
        assert!((y - std::f32::consts::LN_2).abs() < 1.0e-5, "got {y}");
    }

    #[test]
    fn ref_unary_unknown_op_is_none() {
        assert_eq!(ref_unary(&OpKind::MatMul, 1.0), None);
    }

    #[test]
    fn ref_unary_vec_maps_every_element() {
        let out = ref_unary_vec(&OpKind::Neg, &[1.0, -2.0, 3.0]).unwrap();
        assert_eq!(out, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn ref_unary_vec_unknown_op_is_none() {
        assert_eq!(ref_unary_vec(&OpKind::MatMul, &[1.0, 2.0]), None);
    }

    // ── ref_binary ───────────────────────────────────────────────────────────

    #[test]
    fn ref_binary_all_four_ops() {
        assert_eq!(ref_binary(&OpKind::Add, 2.0, 3.0), Some(5.0));
        assert_eq!(ref_binary(&OpKind::Sub, 2.0, 3.0), Some(-1.0));
        assert_eq!(ref_binary(&OpKind::Mul, 2.0, 3.0), Some(6.0));
        assert_eq!(ref_binary(&OpKind::Div, 6.0, 3.0), Some(2.0));
    }

    #[test]
    fn ref_binary_unknown_op_is_none() {
        assert_eq!(ref_binary(&OpKind::MatMul, 1.0, 2.0), None);
    }

    #[test]
    fn ref_binary_vec_maps_pairwise() {
        let out = ref_binary_vec(&OpKind::Mul, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(out, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn ref_binary_vec_rejects_length_mismatch() {
        assert_eq!(ref_binary_vec(&OpKind::Add, &[1.0], &[1.0, 2.0]), None);
    }

    // ── shadow_verify: the FailurePolicy branching ──────────────────────────

    #[test]
    fn shadow_verify_skips_the_oracle_entirely_when_verify_is_off() {
        let called = std::cell::Cell::new(false);
        let result = shadow_verify("Test", &[1.0], false, FailurePolicy::Strict, || {
            called.set(true);
            Some(vec![999.0]) // would fail comparison if it were ever run
        });
        // `CudaDispatchError` deliberately does not derive `PartialEq` (it wraps an opaque
        // external driver error type), so `Result<bool, _>` is compared by pattern, not
        // `assert_eq!`, throughout this test group.
        assert!(matches!(result, Ok(true)), "got {result:?}");
        assert!(
            !called.get(),
            "the oracle closure must not run when verify_on is false"
        );
    }

    #[test]
    fn shadow_verify_passes_a_matching_result() {
        let gpu = [1.0_f32, 2.0, 3.0];
        let result = shadow_verify("Add", &gpu, true, FailurePolicy::Fallback, || {
            Some(vec![1.0, 2.0, 3.0])
        });
        assert!(matches!(result, Ok(true)), "got {result:?}");
    }

    #[test]
    fn shadow_verify_mismatch_under_fallback_discards_without_erroring() {
        let gpu = [1.0_f32, 2.0, 99.0];
        let result = shadow_verify("Add", &gpu, true, FailurePolicy::Fallback, || {
            Some(vec![1.0, 2.0, 3.0])
        });
        assert!(
            matches!(result, Ok(false)),
            "a mismatch under Fallback must be Ok(false) (discard GPU numbers, run on CPU), not \
             Err; got {result:?}"
        );
    }

    #[test]
    fn shadow_verify_mismatch_under_strict_is_a_hard_error() {
        let gpu = [1.0_f32, 2.0, 99.0];
        let result = shadow_verify("Add", &gpu, true, FailurePolicy::Strict, || {
            Some(vec![1.0, 2.0, 3.0])
        });
        match result {
            Err(CudaDispatchError::Verify(msg)) => {
                assert!(msg.contains("element 2"), "got: {msg}");
            }
            other => panic!("expected Err(Verify(_)) under Strict, got {other:?}"),
        }
    }

    #[test]
    fn shadow_verify_an_oracle_that_declines_is_ok_true_not_a_silent_pass_disguised_as_failure() {
        // The oracle returning `None` (no formula for this op) must never be treated as a
        // GPU failure, under either policy -- it is a gap in the oracle.
        for policy in [FailurePolicy::Fallback, FailurePolicy::Strict] {
            let result = shadow_verify("Unknown", &[1.0], true, policy, || None);
            assert!(
                matches!(result, Ok(true)),
                "policy {policy:?} got {result:?}"
            );
        }
    }

    // ── honest wall-clock timing ─────────────────────────────────────────────
    //
    // Measures whatever `ref_conv` currently is (serial or `rayon`-parallel)
    // on two real shapes, so a future change to this module's parallelism
    // can be re-measured the same way it was validated when the `rayon`
    // split was first added. Run explicitly (never part of a plain
    // `cargo test`, and only meaningful in `--release`):
    //   cargo test -p oxionnx-cuda --release -- --ignored --nocapture ref_conv_oracle_timing
    //
    // Shapes come straight from the CUDA performance audit's real per-layer
    // measurements: InSwapper's 1024ch residual-block conv (3x3, stride 1,
    // pad 1, 34x34) and SCRFD's C28->C56 stage (3x3, stride 1, pad 1,
    // 320x320) -- both symmetric "same"-padding convs, so `out_h == h` and
    // `out_w == w` and the naive `2*N*K*C*R*S*out_h*out_w` FLOP count below
    // is exact, not approximate.
    fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = f64::from((state >> 32) as u32) / 4_294_967_296.0;
                (unit * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn time_conv_shape(
        label: &str,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        k: usize,
        r: usize,
        s: usize,
    ) {
        let input = pseudo_random(n * c * h * w, 1);
        let weight = pseudo_random(k * c * r * s, 2);
        let params = ConvParams {
            strides: [1, 1],
            pads: [1, 1, 1, 1],
            dilations: [1, 1],
            group: 1,
            activation: ConvActivation::None,
        };
        let start = std::time::Instant::now();
        let out = ref_conv(&input, &weight, None, &[n, c, h, w], &[k, c, r, s], &params);
        let elapsed = start.elapsed();
        // Symmetric pad=1, k=3, stride=1 => out spatial size equals input's.
        let macs = (n * k * h * w * c * r * s) as f64;
        let gflop = 2.0 * macs / 1.0e9;
        println!(
            "{label}: in=[{n},{c},{h},{w}] w=[{k},{c},{r},{s}] -> {} elems, {:.4}s, {gflop:.3} GFLOP, {:.3} GFLOP/s",
            out.len(),
            elapsed.as_secs_f64(),
            gflop / elapsed.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "wall-clock timing, not correctness -- run explicitly with --ignored --nocapture"]
    fn ref_conv_oracle_timing() {
        time_conv_shape("InSwapper-class", 1, 1024, 34, 34, 1024, 3, 3);
        time_conv_shape("SCRFD-class", 1, 28, 320, 320, 56, 3, 3);
    }

    // ── parallel-vs-serial identity ──────────────────────────────────────────
    //
    // Every oracle parallelised in this module exposes its two branches as
    // separately callable `*_fill_serial`/`*_fill_parallel` functions (see
    // their doc comments) purely so the tests below can force *both* over
    // the *same* randomised large-shape data and assert the outputs are
    // bit-for-bit identical (`assert_eq!` on `Vec<f32>`, not an epsilon
    // comparison) -- proving the `rayon` split changed nothing about the
    // arithmetic, only which thread ran which independent row/element.
    // `ref_conv`/`ref_matmul`/`ref_reduce`/`ref_softmax` themselves only
    // ever run one branch or the other for a given shape (picked by
    // [`parallel_worthwhile`]), so this direct-call approach is the only way
    // to exercise both over identical inputs.
    //
    // Every generated shape is asserted to clear the relevant parallel
    // threshold itself (`total_macs >= PAR_MIN_MACS` / `total >=
    // PAR_MIN_ELEMENTWISE_LEN`) -- a shape that accidentally fell under the
    // threshold would make the test pass vacuously (both "branches" being
    // the exact same serial call), which would defeat the point.

    /// A small deterministic LCG (Knuth/MMIX multiplier), matching
    /// `conv.rs`'s own `gpu_numeric::Lcg` -- reproducible randomised shapes
    /// and data without a `rand` dependency.
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

        /// Uniform `usize` in `[lo, hi]` (inclusive both ends).
        fn range_usize(&mut self, lo: usize, hi: usize) -> usize {
            debug_assert!(lo <= hi);
            let span = u64::from(self.next_u32());
            lo + (span % (hi - lo + 1) as u64) as usize
        }

        /// Uniform `f32` in `[-1, 1)`.
        fn unit_f32(&mut self) -> f32 {
            let unit = f64::from(self.next_u32()) / 4_294_967_296.0;
            (unit * 2.0 - 1.0) as f32
        }

        fn vec_f32(&mut self, len: usize) -> Vec<f32> {
            (0..len).map(|_| self.unit_f32()).collect()
        }
    }

    #[test]
    fn ref_conv_parallel_matches_serial_on_randomized_shapes() {
        let mut rng = Lcg::new(0xC0FF_EE01_u64);
        for case in 0..6_usize {
            let group = [1_usize, 2, 4][case % 3];
            let in_ch_per_group = rng.range_usize(6, 8);
            let out_ch_per_group = rng.range_usize(6, 8);
            let in_channels = in_ch_per_group * group;
            let out_channels = out_ch_per_group * group;
            let n = 1 + case % 2;
            let in_h = rng.range_usize(48, 56);
            let in_w = rng.range_usize(48, 56);
            let filter_h = 3;
            let filter_w = 3;
            let stride_h = 1 + (case / 3) % 2;
            let stride_w = stride_h;
            let dil_h = 1 + (case / 2) % 2;
            let dil_w = dil_h;
            let pad_h = 1;
            let pad_w = 1;

            let input = rng.vec_f32(n * in_channels * in_h * in_w);
            let weight = rng.vec_f32(out_channels * in_ch_per_group * filter_h * filter_w);
            let bias_data = rng.vec_f32(out_channels);
            let bias: Option<&[f32]> = if case % 2 == 0 {
                Some(&bias_data)
            } else {
                None
            };

            let eff_h = dil_h * (filter_h - 1) + 1;
            let eff_w = dil_w * (filter_w - 1) + 1;
            let out_h = (in_h + 2 * pad_h - eff_h) / stride_h + 1;
            let out_w = (in_w + 2 * pad_w - eff_w) / stride_w + 1;

            let geom = ConvGeometry {
                n,
                in_channels,
                in_h,
                in_w,
                out_channels,
                out_h,
                out_w,
                in_ch_per_group,
                filter_h,
                filter_w,
                out_ch_per_group,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
            };
            let total_macs = (n as u64)
                .saturating_mul(out_channels as u64)
                .saturating_mul(out_h as u64)
                .saturating_mul(out_w as u64)
                .saturating_mul(in_ch_per_group as u64)
                .saturating_mul((filter_h * filter_w) as u64);
            assert!(
                total_macs >= PAR_MIN_MACS,
                "case {case}: shape too small to actually exercise the rayon branch \
                 ({total_macs} MACs, need >= {PAR_MIN_MACS}) -- widen the generator's ranges"
            );

            let ops = ConvRowInputs {
                input: &input,
                weight: &weight,
                bias,
                geom,
            };
            let total_out = n * out_channels * out_h * out_w;
            let mut out_serial = vec![0.0_f32; total_out];
            let mut out_parallel = vec![0.0_f32; total_out];
            ref_conv_fill_serial(&mut out_serial, &ops);
            let reporter = ProgressReporter::new("Conv", "test".to_string(), 1);
            ref_conv_fill_parallel(&mut out_parallel, &ops, &reporter);

            assert_eq!(
                out_serial,
                out_parallel,
                "case {case}: n={n} group={group} in=[{in_channels},{in_h},{in_w}] \
                 out_ch_per_group={out_ch_per_group} stride=({stride_h},{stride_w}) \
                 dilation=({dil_h},{dil_w}) bias={}",
                bias.is_some()
            );
        }
    }

    #[test]
    fn ref_matmul_parallel_matches_serial_on_randomized_shapes() {
        let mut rng = Lcg::new(0xFEED_BEEF_u64);
        for case in 0..6_usize {
            let m = rng.range_usize(45, 90);
            let k = rng.range_usize(45, 90);
            let n = rng.range_usize(45, 90);
            let total_macs = (m as u64) * (k as u64) * (n as u64);
            assert!(
                total_macs >= PAR_MIN_MACS,
                "case {case}: {m}x{k}x{n} = {total_macs} MACs, too small -- widen the ranges"
            );

            let a = rng.vec_f32(m * k);
            let b = rng.vec_f32(k * n);

            let mut out_serial = vec![0.0_f32; m * n];
            ref_matmul_fill_serial(&mut out_serial, &a, &b, k, n);
            let mut out_parallel = vec![0.0_f32; m * n];
            let reporter = ProgressReporter::new("MatMul", "test".to_string(), 1);
            ref_matmul_fill_parallel(&mut out_parallel, &a, &b, k, n, &reporter);

            assert_eq!(out_serial, out_parallel, "case {case}: m={m} k={k} n={n}");
        }
    }

    #[test]
    fn ref_reduce_parallel_matches_serial_on_randomized_shapes_and_chunk_sizes() {
        let mut rng = Lcg::new(0xDEAD_10CC_u64);
        for case in 0..6_usize {
            let outer = rng.range_usize(40, 60);
            let inner = rng.range_usize(40, 60);
            let axis_len = rng.range_usize(80, 120);
            let total = outer * inner;
            let total_macs = (total as u64) * (axis_len as u64);
            assert!(
                total_macs >= PAR_MIN_MACS,
                "case {case}: outer={outer} inner={inner} axis_len={axis_len} = {total_macs} \
                 MACs, too small -- widen the ranges"
            );

            let data = rng.vec_f32(outer * axis_len * inner);
            let compute: fn(&[f32], usize, usize, usize, usize) -> f32 = if case % 2 == 0 {
                reduce_sum_at
            } else {
                reduce_max_at
            };

            let mut out_serial = vec![0.0_f32; total];
            ref_reduce_fill_serial(&mut out_serial, &data, compute, axis_len, inner, outer);

            // Deliberately awkward chunk sizes too (1, a small prime, and a
            // size larger than `total` so `par_chunks_mut` yields a single
            // chunk) -- `compute`'s per-index arithmetic never reads
            // `chunk_len`, so the result must be identical regardless of how
            // the work was cut up.
            for &chunk_len in &[1, 3, 17, total.div_ceil(5).max(1), total, total * 2] {
                let mut out_parallel = vec![0.0_f32; total];
                let reporter = ProgressReporter::new("Reduce", "test".to_string(), 1);
                ref_reduce_fill_parallel(
                    &mut out_parallel,
                    &data,
                    compute,
                    axis_len,
                    inner,
                    chunk_len,
                    &reporter,
                );
                assert_eq!(
                    out_serial, out_parallel,
                    "case {case} chunk_len={chunk_len}: outer={outer} inner={inner} \
                     axis_len={axis_len}"
                );
            }
        }
    }

    #[test]
    fn ref_softmax_parallel_matches_serial_on_randomized_shapes() {
        let mut rng = Lcg::new(0xABCD_1234_u64);
        for case in 0..6_usize {
            let row = rng.range_usize(16, 64);
            let rows = rng.range_usize(1200, 2000);
            let total = rows * row;
            assert!(
                total >= PAR_MIN_ELEMENTWISE_LEN && rows >= 2,
                "case {case}: rows={rows} row={row} total={total}, too small -- widen the ranges"
            );
            let data = rng.vec_f32(total);

            let mut out_serial = vec![0.0_f32; total];
            ref_softmax_fill_serial(&mut out_serial, &data, row, rows);
            let mut out_parallel = vec![0.0_f32; total];
            let reporter = ProgressReporter::new("Softmax", "test".to_string(), 1);
            ref_softmax_fill_parallel(&mut out_parallel, &data, row, &reporter);

            assert_eq!(
                out_serial, out_parallel,
                "case {case}: rows={rows} row={row}"
            );
        }
    }

    #[test]
    fn ref_unary_vec_parallel_matches_serial_on_large_input() {
        // Non-negative data: `Sqrt`/`Log` would otherwise produce `NaN` for
        // negative inputs, and `NaN != NaN` would make `assert_eq!` fail
        // even when both sides computed the identical bit pattern. The
        // negative-input branch of each formula is already covered by
        // `ref_unary`'s own hand-verified tests above; this test's job is
        // only to prove the parallel *mapping* doesn't drop/reorder/corrupt
        // elements, which needs no negative coverage of its own.
        let mut rng = Lcg::new(0x5EED_0001_u64);
        let len = PAR_MIN_ELEMENTWISE_LEN * 3 + 17; // comfortably above the threshold, not a round multiple
        let data: Vec<f32> = rng.vec_f32(len).into_iter().map(f32::abs).collect();

        for op in [
            OpKind::Relu,
            OpKind::Sigmoid,
            OpKind::Softplus,
            OpKind::Sqrt,
            OpKind::HardSwish,
        ] {
            let expected: Vec<f32> = data.iter().map(|&x| ref_unary(&op, x).unwrap()).collect();
            let got = ref_unary_vec(&op, &data).unwrap();
            assert_eq!(expected, got, "op={op:?}");
        }
    }

    #[test]
    fn ref_binary_vec_parallel_matches_serial_on_large_input() {
        let mut rng = Lcg::new(0x5EED_0002_u64);
        let len = PAR_MIN_ELEMENTWISE_LEN * 3 + 17;
        let a = rng.vec_f32(len);
        // Shifted well away from 0 so `Div` never divides by (or produces)
        // something that could make two mathematically-identical results
        // round differently, and never hits an exact `0.0/0.0 = NaN` that
        // would trip the same `NaN != NaN` `assert_eq!` gotcha noted above.
        let b: Vec<f32> = rng.vec_f32(len).into_iter().map(|x| x + 2.0).collect();

        for op in [OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div] {
            let expected: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| ref_binary(&op, x, y).unwrap())
                .collect();
            let got = ref_binary_vec(&op, &a, &b).unwrap();
            assert_eq!(expected, got, "op={op:?}");
        }
    }
}
