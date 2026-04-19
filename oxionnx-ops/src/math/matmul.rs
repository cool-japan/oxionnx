use oxionnx_core::Tensor;

use super::broadcast::broadcast_to;

// ── Shared helper: compute one batch slice into `dst` ───────────────────────

/// Write the product `a[a_off..] @ b[b_off..]` (shape M×K × K×N) into `dst`.
#[inline]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn matmul_batch_slice(
    a_data: &[f32],
    b_data: &[f32],
    dst: &mut [f32],
    a_off: usize,
    b_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    if m >= 4 {
        #[allow(unsafe_code)]
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a_data[a_off..].as_ptr(),
                k as isize,
                1,
                b_data[b_off..].as_ptr(),
                n as isize,
                1,
                0.0,
                dst.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    } else {
        for i in 0..m {
            let a_row = &a_data[a_off + i * k..a_off + (i + 1) * k];
            for j in 0..n {
                let mut s = 0.0f32;
                for (kk, &a_val) in a_row.iter().enumerate() {
                    s += a_val * b_data[b_off + kk * n + j];
                }
                dst[i * n + j] = s;
            }
        }
    }
}

// ── MatMul / Gemm ───────────────────────────────────────────────────────────

/// Matrix multiplication supporting batched tensors.
/// Last two dims: [M, K] @ [K, N] = [M, N]
///
/// When `batch_size >= 4` and not targeting wasm32, batch iterations are
/// parallelised with rayon for throughput.
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let an = a.ndim();
    let bn = b.ndim();

    if an < 2 || bn < 2 {
        return Err(format!(
            "matmul requires at least 2D tensors, got {}D and {}D",
            an, bn
        ));
    }

    let m = a.shape[an - 2];
    let k = a.shape[an - 1];
    let k2 = b.shape[bn - 2];
    let n = b.shape[bn - 1];

    if k != k2 {
        return Err(format!("matmul: inner dimensions mismatch {k} != {k2}"));
    }

    let a_batch: Vec<usize> = a.shape[..an - 2].to_vec();
    let b_batch: Vec<usize> = b.shape[..bn - 2].to_vec();
    let out_batch = Tensor::broadcast_shape(&a_batch, &b_batch)?;

    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let mn = m * n;
    let out_size = batch_size * mn;

    let a_batches = a.numel() / (m * k);
    let b_batches = b.numel() / (k * n);

    #[cfg(not(target_arch = "wasm32"))]
    let out = if batch_size >= 4 {
        use rayon::prelude::*;
        let a_data = &a.data;
        let b_data = &b.data;
        let results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b_idx| {
                let a_off = (b_idx % a_batches) * a_batch_stride;
                let b_off = (b_idx % b_batches) * b_batch_stride;
                let mut buf = vec![0.0f32; mn];
                matmul_batch_slice(a_data, b_data, &mut buf, a_off, b_off, m, k, n);
                buf
            })
            .collect();
        let mut out = Vec::with_capacity(out_size);
        for r in results {
            out.extend_from_slice(&r);
        }
        out
    } else {
        let mut out = vec![0.0f32; out_size];
        for b_idx in 0..batch_size {
            let a_off = (b_idx % a_batches) * a_batch_stride;
            let b_off = (b_idx % b_batches) * b_batch_stride;
            let c_off = b_idx * mn;
            matmul_batch_slice(
                &a.data,
                &b.data,
                &mut out[c_off..c_off + mn],
                a_off,
                b_off,
                m,
                k,
                n,
            );
        }
        out
    };

    #[cfg(target_arch = "wasm32")]
    let out = {
        let mut out = vec![0.0f32; out_size];
        for b_idx in 0..batch_size {
            let a_off = (b_idx % a_batches) * a_batch_stride;
            let b_off = (b_idx % b_batches) * b_batch_stride;
            let c_off = b_idx * mn;
            matmul_batch_slice(
                &a.data,
                &b.data,
                &mut out[c_off..c_off + mn],
                a_off,
                b_off,
                m,
                k,
                n,
            );
        }
        out
    };

    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
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
    let a_eff = if trans_a { transpose_2d(a)? } else { a.clone() };
    let b_eff = if trans_b { transpose_2d(b)? } else { b.clone() };
    let mut result = matmul(&a_eff, &b_eff)?;
    if alpha != 1.0 {
        result.data.iter_mut().for_each(|v| *v *= alpha);
    }
    if let Some(c) = c {
        let c_bcast = broadcast_to(c, &result.shape);
        for (r, &cv) in result.data.iter_mut().zip(c_bcast.data.iter()) {
            *r += beta * cv;
        }
    }
    Ok(result)
}

/// Matrix multiplication that writes the result directly into `out`.
///
/// Resizes `out` to the exact output length, then writes every element in
/// place — no temporary allocation for the output buffer.
///
/// Returns the output shape `[batch..., M, N]`.
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub fn matmul_into(a: &Tensor, b: &Tensor, out: &mut Vec<f32>) -> Result<Vec<usize>, String> {
    let an = a.ndim();
    let bn = b.ndim();

    if an < 2 || bn < 2 {
        return Err(format!(
            "matmul_into requires at least 2D tensors, got {}D and {}D",
            an, bn
        ));
    }

    let m = a.shape[an - 2];
    let k = a.shape[an - 1];
    let k2 = b.shape[bn - 2];
    let n = b.shape[bn - 1];

    if k != k2 {
        return Err(format!(
            "matmul_into: inner dimensions mismatch {k} != {k2}"
        ));
    }

    let a_batch: Vec<usize> = a.shape[..an - 2].to_vec();
    let b_batch: Vec<usize> = b.shape[..bn - 2].to_vec();
    let out_batch = Tensor::broadcast_shape(&a_batch, &b_batch)?;

    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let mn = m * n;
    let out_size = batch_size * mn;

    let a_batches = a.numel() / (m * k);
    let b_batches = b.numel() / (k * n);

    // Pre-size the slot buffer — zero-copy target.
    out.resize(out_size, 0.0_f32);

    #[cfg(not(target_arch = "wasm32"))]
    if batch_size >= 4 {
        use rayon::prelude::*;
        let a_data = &a.data;
        let b_data = &b.data;
        out.par_chunks_mut(mn).enumerate().for_each(|(b_idx, dst)| {
            let a_off = (b_idx % a_batches) * a_batch_stride;
            let b_off = (b_idx % b_batches) * b_batch_stride;
            matmul_batch_slice(a_data, b_data, dst, a_off, b_off, m, k, n);
        });
    } else {
        for b_idx in 0..batch_size {
            let a_off = (b_idx % a_batches) * a_batch_stride;
            let b_off = (b_idx % b_batches) * b_batch_stride;
            let c_off = b_idx * mn;
            matmul_batch_slice(
                &a.data,
                &b.data,
                &mut out[c_off..c_off + mn],
                a_off,
                b_off,
                m,
                k,
                n,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    for b_idx in 0..batch_size {
        let a_off = (b_idx % a_batches) * a_batch_stride;
        let b_off = (b_idx % b_batches) * b_batch_stride;
        let c_off = b_idx * mn;
        matmul_batch_slice(
            &a.data,
            &b.data,
            &mut out[c_off..c_off + mn],
            a_off,
            b_off,
            m,
            k,
            n,
        );
    }

    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
    Ok(out_shape)
}

/// Gemm that writes Y = alpha * A' @ B' + beta * C directly into `out`.
///
/// Resizes `out` to M*N elements, then writes in place.
/// Returns the output shape `[M, N]`.
///
/// Note: `trans_a`/`trans_b` require a transposed copy of A/B (unavoidable
/// without a tiled write-side transpose), so only the final result write is
/// zero-copy.
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
    let a_eff = if trans_a { transpose_2d(a)? } else { a.clone() };
    let b_eff = if trans_b { transpose_2d(b)? } else { b.clone() };
    let out_shape = matmul_into(&a_eff, &b_eff, out)?;
    if alpha != 1.0 {
        out.iter_mut().for_each(|v| *v *= alpha);
    }
    if let Some(c_tensor) = c {
        // Build a temporary Tensor view over the current `out` for shape broadcast.
        // We work on the raw slice to avoid cloning `out` again.
        let c_bcast = broadcast_to(c_tensor, &out_shape);
        for (r, &cv) in out.iter_mut().zip(c_bcast.data.iter()) {
            *r += beta * cv;
        }
    }
    Ok(out_shape)
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
