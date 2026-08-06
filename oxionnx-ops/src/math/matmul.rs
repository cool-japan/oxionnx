use oxionnx_core::Tensor;

use super::broadcast::broadcast_to;

// ── MatMul / Gemm ───────────────────────────────────────────────────────────
//
// The core `A @ B` computation for both MatMul and Gemm is
// `crate::math_typed::sgemm_strided` — the single stride-aware
// `matrixmultiply::sgemm` call, shared with the typed F32 dispatch arms in
// `crate::registry::math_ops::matmul_gemm` (see `math_typed`'s module doc
// comment for why it lives there rather than here: it must be reachable from
// both this module and that one, which are not in an ancestor/descendant
// relationship). `matmul`/`gemm` delegate to their `_into` counterparts so
// the batching/parallelisation decision exists in exactly one place.

/// Matrix multiplication supporting batched tensors.
/// Last two dims: [M, K] @ [K, N] = [M, N]
///
/// When `batch_size >= 4` and not targeting wasm32, batch iterations are
/// parallelised with rayon for throughput. Every batch slice — including
/// M<4, e.g. the M=1 decode-phase GEMM of an autoregressive transformer's
/// `[1,1,d] @ [d,d]` projections — is computed via `matrixmultiply::sgemm`;
/// there is no naive scalar fallback loop.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let mut out = Vec::new();
    let out_shape = matmul_into(a, b, &mut out)?;
    Ok(Tensor::new(out, out_shape))
}

/// Gemm: Y = alpha * A' @ B' + beta * C
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<Tensor, String> {
    let mut out = Vec::new();
    let out_shape = gemm_into(a, b, c, alpha, beta, trans_a, trans_b, &mut out)?;
    Ok(Tensor::new(out, out_shape))
}

/// Matrix multiplication that writes the result directly into `out`.
///
/// Resizes `out` to the exact output length, then writes every element in
/// place: no temporary output allocation, and (for `batch_size >= 4`) no
/// per-batch `Vec` collected and copied back either — each rayon task writes
/// straight into its slice of `out` (see `crate::math_typed::matmul_f32_into`).
///
/// Returns the output shape `[batch..., M, N]`.
pub fn matmul_into(a: &Tensor, b: &Tensor, out: &mut Vec<f32>) -> Result<Vec<usize>, String> {
    let an = a.ndim();
    let bn = b.ndim();

    if an < 2 || bn < 2 {
        return Err(format!(
            "matmul_into requires at least 2D tensors, got {}D and {}D",
            an, bn
        ));
    }

    let k = a.shape[an - 1];
    let k2 = b.shape[bn - 2];
    if k != k2 {
        return Err(format!(
            "matmul_into: inner dimensions mismatch {k} != {k2}"
        ));
    }

    crate::math_typed::matmul_f32_into(&a.data, &a.shape, &b.data, &b.shape, out)
}

/// Gemm that writes Y = alpha * A' @ B' + beta * C directly into `out`.
///
/// Resizes `out` to M*N elements, then writes in place.
/// Returns the output shape `[M, N]`.
///
/// When both `a` and `b` are exactly 2D — the ONNX-spec shape for Gemm, and
/// what every fusion pass and every hand-authored `Gemm` node actually
/// carries — a transposed operand is expressed by swapping `sgemm`'s
/// row/column strides (see `sgemm_strided` in `crate::math_typed`)
/// instead of being materialised as a physical copy. `transB=1` is the
/// common case (a PyTorch `nn.Linear` weight is stored
/// `[out_features, in_features]`), so for a `[4096, 4096]` weight this
/// removes a 64 MB copy on every `run()` call, once per layer.
///
/// A non-2D `a`/`b` — only reachable from the untyped `Gemm::execute`, which
/// (unlike `execute_typed`) does not validate rank; a rank-3+ Gemm input is
/// malformed-model territory per the ONNX spec, not the hot path — falls
/// back to the batched `transpose_2d` + `matmul_into` path, unchanged from
/// before this optimisation.
#[allow(clippy::too_many_arguments)]
pub fn gemm_into(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    out: &mut Vec<f32>,
) -> Result<Vec<usize>, String> {
    if a.ndim() == 2 && b.ndim() == 2 {
        return gemm_2d_into(a, b, c, alpha, beta, trans_a, trans_b, out);
    }
    let a_eff = if trans_a { transpose_2d(a)? } else { a.clone() };
    let b_eff = if trans_b { transpose_2d(b)? } else { b.clone() };
    let out_shape = matmul_into(&a_eff, &b_eff, out)?;
    if alpha != 1.0 {
        out.iter_mut().for_each(|v| *v *= alpha);
    }
    if let Some(c_tensor) = c {
        let c_bcast = broadcast_to(c_tensor, &out_shape);
        for (r, &cv) in out.iter_mut().zip(c_bcast.data.iter()) {
            *r += beta * cv;
        }
    }
    Ok(out_shape)
}

/// The rank-2 fast path for [`gemm_into`]: a single, stride-transposed
/// `sgemm_strided` call with no operand copy.
///
/// `alpha`/`beta` are applied exactly as [`gemm_into`]'s general path
/// applies them — as separate passes over `out` after the raw `A @ B` is
/// written — rather than folded into `sgemm_strided`'s native `alpha`/`beta`
/// parameters. That fold is possible (prefill `out` with the broadcast `C`
/// and let `sgemm` combine it), but it would change how `alpha`/`beta` are
/// rounded relative to the pre-optimisation code, for a saving of O(MN)
/// against an O(MNK) kernel — not worth the numerical-parity risk. This way,
/// the only thing that changes relative to before is how the raw `A @ B`
/// term itself is computed, and per `sgemm_strided`'s doc comment that is
/// bit-identical to transpose-then-multiply (strides don't affect
/// `matrixmultiply`'s packing/accumulation order).
#[allow(clippy::too_many_arguments)]
fn gemm_2d_into(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    out: &mut Vec<f32>,
) -> Result<Vec<usize>, String> {
    let (m, k) = if trans_a {
        (a.shape[1], a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    let (k2, n) = if trans_b {
        (b.shape[1], b.shape[0])
    } else {
        (b.shape[0], b.shape[1])
    };
    if k != k2 {
        return Err(format!("gemm: inner dimensions mismatch {k} != {k2}"));
    }
    out.resize(m * n, 0.0f32);

    // A is stored `[m,k]` normally (row stride k, col stride 1) and `[k,m]`
    // when transposed (row stride 1, col stride m) so that `A[i,p]` still
    // reads the logical (post-transpose) element at `(i,p)`; B mirrors this.
    let (rsa, csa) = if trans_a {
        (1isize, m as isize)
    } else {
        (k as isize, 1isize)
    };
    let (rsb, csb) = if trans_b {
        (1isize, k as isize)
    } else {
        (n as isize, 1isize)
    };
    crate::math_typed::sgemm_strided(
        m, k, n, 1.0, &a.data, rsa, csa, &b.data, rsb, csb, 0.0, out, n as isize, 1,
    );

    if alpha != 1.0 {
        out.iter_mut().for_each(|v| *v *= alpha);
    }
    if let Some(c_tensor) = c {
        let c_bcast = broadcast_to(c_tensor, &[m, n]);
        for (r, &cv) in out.iter_mut().zip(c_bcast.data.iter()) {
            *r += beta * cv;
        }
    }
    Ok(vec![m, n])
}

fn transpose_2d(t: &Tensor) -> Result<Tensor, String> {
    let nd = t.ndim();
    if nd < 2 {
        return Err(format!("transpose_2d: expected at least 2D, got {nd}D"));
    }
    let rows = t.shape[nd - 2];
    let cols = t.shape[nd - 1];
    let batch: usize = t.shape[..nd - 2].iter().product::<usize>().max(1);
    let slice = rows * cols;
    let mut out = vec![0.0f32; t.data.len()];
    for b in 0..batch {
        let base = b * slice;
        for r in 0..rows {
            for c in 0..cols {
                out[base + c * rows + r] = t.data[base + r * cols + c];
            }
        }
    }
    let mut new_shape = t.shape[..nd - 2].to_vec();
    new_shape.push(cols);
    new_shape.push(rows);
    Ok(Tensor::new(out, new_shape))
}

// ── Perf smoke tests: before/after timing notes for W2-perf-matmul ─────────
//
// These are `#[ignore]`d (they don't run under the normal correctness gate)
// and print wall-clock timings for the exact shapes called out in the
// a6-0/a6-6/a6-17 findings. Run with:
//   cargo test -p oxionnx-ops --release --lib math::matmul::perf_smoke -- --ignored --nocapture
#[cfg(test)]
mod perf_smoke {
    use super::*;
    use std::time::Instant;

    fn timed(label: &str, iters: u32, mut f: impl FnMut()) {
        // one warm-up call, untimed
        f();
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        println!(
            "{label}: {:.3} ms/call ({iters} calls, {:.3} ms total)",
            elapsed.as_secs_f64() * 1000.0 / f64::from(iters),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    #[test]
    #[ignore]
    fn perf_m1_gemm_decode() {
        // The decode-phase LLM GEMM from a6-0/a6-6: [1,4096] @ [4096,4096]^T
        // (transB=1, matching a PyTorch nn.Linear weight layout).
        let a = Tensor::new(vec![0.01f32; 4096], vec![1, 4096]);
        let w = Tensor::new(vec![0.001f32; 4096 * 4096], vec![4096, 4096]);
        timed("gemm M=1 [1,4096]x[4096,4096]^T (transB)", 20, || {
            let _ = gemm(&a, &w, None, 1.0, 0.0, false, true).expect("gemm");
        });
        timed("matmul M=1 [1,4096]x[4096,4096]", 20, || {
            let _ = matmul(&a, &w).expect("matmul");
        });
    }

    #[test]
    #[ignore]
    fn perf_batched_matmul_decode() {
        // Batched-attention decode matmul: [B*H,1,d] @ [B*H,d,S]
        let bh = 32;
        let d = 64;
        let s = 128;
        let a = Tensor::new(vec![0.01f32; bh * d], vec![bh, 1, d]);
        let b = Tensor::new(vec![0.001f32; bh * d * s], vec![bh, d, s]);
        timed("matmul batched [32,1,64]x[32,64,128]", 200, || {
            let _ = matmul(&a, &b).expect("matmul");
        });
    }
}
