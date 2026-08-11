//! Shared f32 GEMM kernels for the attention / flash-attention kernels.
//!
//! Before this module every attention matmul was a hand-rolled triple loop
//! (`mm` / `mm_a_bt`): a serial FP-add chain that reaches roughly one FMA per
//! four cycles instead of the 8–16 FLOP/cycle a blocked, register-tiled kernel
//! achieves.  All of them now route through [`matrixmultiply::sgemm`], which is
//! Pure Rust (no BLAS, no C) and already a workspace dependency.
//!
//! ## Two families — never nest rayon
//!
//! * `*_into` — strictly **serial**. Use inside an already-parallel loop (one
//!   task per `(batch, head)` slice) so a rayon task never spawns more
//!   parallel work.
//! * `*_into_par` — splits the output **rows** across rayon workers. Use only
//!   where the surrounding loop is trivial — e.g. a QKV projection with
//!   `batch == 1`, where row splitting is the only available parallelism.
//!
//! ## Numerics
//!
//! `sgemm` blocks and vectorises the `k` reduction, so results differ from the
//! previous strictly-sequential accumulation by floating-point reassociation
//! (observed ≤ 1e-6 absolute on attention-sized problems, well inside the 1e-5
//! parity budget documented for this module).  Shapes with fewer than
//! [`SGEMM_MIN_ROWS`] output rows keep the original scalar loops **in their
//! original accumulation order**, so small-shape results stay bit-identical —
//! which is why the existing 1e-6 attention assertions did not have to move.

/// Minimum number of output rows before `sgemm`'s packing/blocking overhead
/// pays for itself.  Mirrors the same threshold used by `math::matmul`.
pub(crate) const SGEMM_MIN_ROWS: usize = 4;

/// Minimum multiply-accumulate count before splitting GEMM rows across rayon
/// workers is worth the task-dispatch overhead (~10 µs).
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const PAR_MIN_MACS: usize = 1 << 17;

// ── C = A[m,k] · B[k,n] ──────────────────────────────────────────────────────

/// `out[m, n] = a[m, k] · b[k, n]` (row-major, assign semantics).
///
/// Serial. Safe to call from inside a rayon task.
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub(crate) fn matmul_nn_into(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    let out = &mut out[..m * n];
    if k == 0 {
        out.fill(0.0);
        return;
    }
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    if m >= SGEMM_MIN_ROWS {
        // SAFETY: `a`, `b` and `out` are checked above to hold at least
        // `m*k`, `k*n` and `m*n` elements; the strides passed describe exactly
        // those row-major extents, so every access stays in bounds.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    } else {
        // Original i–k–j accumulation order, preserved bit-for-bit.
        out.fill(0.0);
        for i in 0..m {
            for kk in 0..k {
                let a_val = a[i * k + kk];
                let b_row = &b[kk * n..kk * n + n];
                let o_row = &mut out[i * n..i * n + n];
                for (o, &bv) in o_row.iter_mut().zip(b_row.iter()) {
                    *o += a_val * bv;
                }
            }
        }
    }
}

/// Allocating wrapper around [`matmul_nn_into`]: `[m, k] · [k, n] -> [m, n]`.
pub(crate) fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    matmul_nn_into(a, b, m, k, n, &mut out);
    out
}

// ── C = A[m,k] · B[n,k]^T ────────────────────────────────────────────────────

/// `out[m, n] = a[m, k] · b[n, k]^T` (both operands row-major, assign
/// semantics).
///
/// The transpose is free: `sgemm` takes explicit row/column strides, so `B^T`
/// is expressed as row-stride 1 / column-stride `k` over the same buffer.
///
/// Serial. Safe to call from inside a rayon task.
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub(crate) fn matmul_nt_into(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    let out = &mut out[..m * n];
    if k == 0 {
        out.fill(0.0);
        return;
    }
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= n * k);
    if m >= SGEMM_MIN_ROWS {
        // SAFETY: as in `matmul_nn_into`; `b` is read as the transpose of an
        // `n × k` row-major matrix (rsb = 1, csb = k), which touches exactly
        // the `n*k` elements checked above.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                out.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    } else {
        // Original i–j–k accumulation order, preserved bit-for-bit.
        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            for j in 0..n {
                let b_row = &b[j * k..j * k + k];
                let mut s = 0.0f32;
                for (&av, &bv) in a_row.iter().zip(b_row.iter()) {
                    s += av * bv;
                }
                out[i * n + j] = s;
            }
        }
    }
}

/// Allocating wrapper around [`matmul_nt_into`]: `[m, k] · [n, k]^T -> [m, n]`.
pub(crate) fn matmul_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    matmul_nt_into(a, b, m, k, n, &mut out);
    out
}

// ── Row-parallel variants ────────────────────────────────────────────────────

/// Split `out`'s rows across rayon workers, running [`matmul_nt_into`] on each
/// row block.
///
/// Only for call sites whose surrounding loop offers no parallelism of its own
/// (typically `batch == 1` projections). Falls back to the serial kernel when
/// the problem is too small to amortise task dispatch.
///
/// The row partition is derived from `rayon::current_num_threads()` and each
/// block is computed independently, so the result is deterministic for a given
/// machine and does not depend on scheduling order.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
pub(crate) fn matmul_nt_into_par(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    use rayon::prelude::*;

    let threads = rayon::current_num_threads();
    if threads < 2
        || n == 0
        || m < 2 * SGEMM_MIN_ROWS
        || m.saturating_mul(k).saturating_mul(n) < PAR_MIN_MACS
    {
        matmul_nt_into(a, b, m, k, n, out);
        return;
    }
    let row_block = m.div_ceil(threads).max(SGEMM_MIN_ROWS);
    out[..m * n]
        .par_chunks_mut(row_block * n)
        .enumerate()
        .for_each(|(blk, dst)| {
            let r0 = blk * row_block;
            let rows = (m - r0).min(row_block);
            matmul_nt_into(&a[r0 * k..], b, rows, k, n, dst);
        });
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
pub(crate) fn matmul_nt_into_par(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    matmul_nt_into(a, b, m, k, n, out);
}

// ── Parallel-decision helper for (batch × head) loops ────────────────────────

/// Whether a loop of `units` independent `(batch, head)` slices, each costing
/// roughly `macs_per_unit` multiply-accumulates, is worth handing to rayon.
///
/// Not defined on a serial wasm32 build (no rayon there — every caller stays
/// on its serial path); present again under `wasm-threads`.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[inline]
pub(crate) fn should_parallelize(units: usize, macs_per_unit: usize) -> bool {
    /// Task dispatch costs a few microseconds; below this much total work the
    /// serial loop wins.
    const PAR_MIN_TOTAL_MACS: usize = 1 << 15;
    units >= 2
        && rayon::current_num_threads() >= 2
        && units.saturating_mul(macs_per_unit) >= PAR_MIN_TOTAL_MACS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference triple loop used to pin the sgemm path.
    fn ref_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for kk in 0..k {
                    s += f64::from(a[i * k + kk]) * f64::from(b[kk * n + j]);
                }
                out[i * n + j] = s as f32;
            }
        }
        out
    }

    fn ref_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for kk in 0..k {
                    s += f64::from(a[i * k + kk]) * f64::from(b[j * k + kk]);
                }
                out[i * n + j] = s as f32;
            }
        }
        out
    }

    fn seq(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * seed + 0.13).sin()).collect()
    }

    /// numpy check (python3):
    /// ```text
    /// a = np.arange(6, dtype=np.float32).reshape(2, 3)
    /// b = np.arange(12, dtype=np.float32).reshape(3, 4)
    /// (a @ b).ravel()
    /// -> [20., 23., 26., 29., 56., 68., 80., 92.]
    /// ```
    #[test]
    fn nn_matches_numpy_small() {
        let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let got = matmul_nn(&a, &b, 2, 3, 4);
        let want = [20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
        }
    }

    /// numpy check (python3):
    /// ```text
    /// a = np.arange(6, dtype=np.float32).reshape(2, 3)
    /// b = np.arange(12, dtype=np.float32).reshape(4, 3)
    /// (a @ b.T).ravel()
    /// -> [ 5., 14., 23., 32., 14., 50., 86., 122.]
    /// ```
    #[test]
    fn nt_matches_numpy_small() {
        let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let got = matmul_nt(&a, &b, 2, 3, 4);
        let want = [5.0, 14.0, 23.0, 32.0, 14.0, 50.0, 86.0, 122.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
        }
    }

    #[test]
    fn nn_matches_reference_across_shapes() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (3, 5, 7),
            (4, 4, 4),
            (17, 23, 11),
            (64, 33, 48),
            (5, 1, 9),
        ] {
            let a = seq(m * k, 0.31);
            let b = seq(k * n, 0.17);
            let got = matmul_nn(&a, &b, m, k, n);
            let want = ref_nn(&a, &b, m, k, n);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() < 1e-5, "nn {m}x{k}x{n} idx {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn nt_matches_reference_across_shapes() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (3, 5, 7),
            (4, 4, 4),
            (17, 23, 11),
            (64, 33, 48),
            (5, 1, 9),
        ] {
            let a = seq(m * k, 0.29);
            let b = seq(n * k, 0.41);
            let got = matmul_nt(&a, &b, m, k, n);
            let want = ref_nt(&a, &b, m, k, n);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() < 1e-5, "nt {m}x{k}x{n} idx {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn row_parallel_matches_serial() {
        for &(m, k, n) in &[(4, 8, 8), (128, 96, 64), (257, 33, 65)] {
            let a = seq(m * k, 0.11);
            let b = seq(n * k, 0.23);
            let mut par = vec![0.0f32; m * n];
            matmul_nt_into_par(&a, &b, m, k, n, &mut par);
            let ser = matmul_nt(&a, &b, m, k, n);
            for (i, (p, s)) in par.iter().zip(ser.iter()).enumerate() {
                assert!((p - s).abs() < 1e-5, "par {m}x{k}x{n} idx {i}: {p} vs {s}");
            }
        }
    }

    #[test]
    fn zero_dims_are_no_ops() {
        let mut out = vec![7.0f32; 4];
        matmul_nn_into(&[], &[], 0, 3, 2, &mut out);
        assert_eq!(out, vec![7.0; 4]);
        matmul_nt_into(&[], &[], 2, 0, 2, &mut out);
        assert_eq!(out, vec![0.0; 4]);
        let mut out2 = vec![7.0f32; 4];
        matmul_nn_into(&[1.0, 2.0], &[], 2, 0, 2, &mut out2);
        assert_eq!(out2, vec![0.0; 4]);
    }
}
