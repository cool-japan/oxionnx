use oxionnx_core::Tensor;

/// Check if the reduction covers all dimensions (full-tensor reduction).
/// This is true when axes is empty (ONNX convention: all axes)
/// or when axes lists every dimension index.
#[cfg(feature = "simd")]
pub(super) fn is_full_reduction(x: &Tensor, axes: &[i64]) -> bool {
    if axes.is_empty() {
        return true;
    }
    let ndim = x.ndim();
    if axes.len() < ndim {
        return false;
    }
    let mut seen = vec![false; ndim];
    for &a in axes {
        let idx = if a < 0 {
            (a + ndim as i64) as usize
        } else {
            a as usize
        };
        if idx < ndim {
            seen[idx] = true;
        }
    }
    seen.iter().all(|&s| s)
}

/// Broadcast a tensor to the given shape.
pub fn broadcast_to(t: &Tensor, out_shape: &[usize]) -> Tensor {
    if t.shape == out_shape {
        return t.clone();
    }
    let n_out: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; n_out];

    // Pad t.shape on the left with 1s to match out_shape.ndim
    let n = out_shape.len();
    let pad = n - t.shape.len();
    let padded: Vec<usize> = (0..pad).map(|_| 1).chain(t.shape.iter().copied()).collect();

    // Compute strides for padded shape (0 for broadcast dims)
    let mut strides = vec![0usize; n];
    let mut stride = 1usize;
    for i in (0..n).rev() {
        if padded[i] == 1 && out_shape[i] != 1 {
            strides[i] = 0; // broadcast
        } else {
            strides[i] = stride;
        }
        stride *= padded[i];
    }

    // Fill output
    let mut out_strides = vec![0usize; n];
    let mut s = 1usize;
    for i in (0..n).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    for (out_idx, out_val) in data.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut src_idx = 0usize;
        for i in 0..n {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            src_idx += coord * strides[i];
        }
        *out_val = t.data[src_idx];
    }

    Tensor::new(data, out_shape.to_vec())
}

/// Left-pad `shape` with 1s to `n` dimensions (NumPy broadcast padding).
fn pad_shape(shape: &[usize], n: usize) -> Vec<usize> {
    let mut padded = vec![1usize; n - shape.len()];
    padded.extend_from_slice(shape);
    padded
}

/// If `padded` (already left-padded to `out_shape`'s rank) is a "trailing
/// vector" — some prefix of all-1 dims, followed by a suffix that exactly
/// matches the corresponding suffix of `out_shape` — return the element
/// count of that suffix: the operand can then be reproduced by simply
/// tiling (repeating) its flat data `out_numel / count` times, with no
/// index arithmetic at all. Covers both the common bias-add case
/// (`[768]` against `[1,512,768]`) and a leading-broadcast row vector
/// (`[1,768]` against `[512,768]`), which both reduce to the same tiling.
///
/// Returns `None` when the operand broadcasts some other way (e.g. `[m,1]`
/// against `[m,n]`, where the tile boundary doesn't align with any suffix of
/// `out_shape`), which must fall back to the general stride walk.
fn trailing_vector_len(padded: &[usize], out_shape: &[usize]) -> Option<usize> {
    let n = out_shape.len();
    let start = padded.iter().position(|&d| d != 1).unwrap_or(n);
    if padded[start..] == out_shape[start..] {
        Some(out_shape[start..].iter().product())
    } else {
        None
    }
}

/// Per-dimension stride for walking `shape` (broadcast against `out_shape`)
/// via increment-and-carry instead of a div/mod per output element: 0 on a
/// broadcast dimension, the standard row-major stride otherwise. Same
/// definition `broadcast_to` uses internally, factored out so
/// `elementwise_binary`'s general path can walk two operands' indices in one
/// pass instead of materialising two full broadcast copies first.
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let n = out_shape.len();
    let padded = pad_shape(shape, n);
    let mut strides = vec![0usize; n];
    let mut stride = 1usize;
    for i in (0..n).rev() {
        if padded[i] == 1 && out_shape[i] != 1 {
            strides[i] = 0; // broadcast dimension: same source element for every step
        } else {
            strides[i] = stride;
        }
        stride *= padded[i];
    }
    strides
}

pub(super) fn elementwise_binary(
    a: &Tensor,
    b: &Tensor,
    op: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let out_numel: usize = out_shape.iter().product();

    // Degenerate: nothing to compute. Also sidesteps `chunks(0)` panics
    // below for a broadcast operand paired with a 0-sized output dimension.
    if out_numel == 0 {
        return Ok(Tensor::new(Vec::new(), out_shape));
    }

    // Fast path 1: identical shapes — no broadcasting at all. Zip both data
    // slices directly, no clone of either operand. This is the path that
    // matters most in the default (non-`simd`) build, where `add`/`sub`/
    // `mul`/`div`'s SIMD same-shape fast paths are compiled out and every
    // same-shape elementwise op (e.g. a residual `Add`) would otherwise
    // fall all the way through to `broadcast_to`'s full materialise-and-copy.
    if a.shape == out_shape && b.shape == out_shape {
        let data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        return Ok(Tensor::new(data, out_shape));
    }

    let n = out_shape.len();

    // Fast path 2: one operand already has the output shape, and the other
    // is a trailing-contiguous "vector" (or scalar) that tiles across it —
    // e.g. the bias-add `[1,512,768] + [768]`. Each inner loop zips two
    // contiguous slices with no per-element index arithmetic, so it is as
    // vectorisable as the equal-shape case above.
    if a.shape == out_shape {
        let b_padded = pad_shape(&b.shape, n);
        if let Some(chunk) = trailing_vector_len(&b_padded, &out_shape) {
            let data: Vec<f32> = a
                .data
                .chunks(chunk)
                .flat_map(|a_chunk| a_chunk.iter().zip(b.data.iter()).map(|(&x, &y)| op(x, y)))
                .collect();
            return Ok(Tensor::new(data, out_shape));
        }
    } else if b.shape == out_shape {
        let a_padded = pad_shape(&a.shape, n);
        if let Some(chunk) = trailing_vector_len(&a_padded, &out_shape) {
            let data: Vec<f32> = b
                .data
                .chunks(chunk)
                .flat_map(|b_chunk| a.data.iter().zip(b_chunk.iter()).map(|(&x, &y)| op(x, y)))
                .collect();
            return Ok(Tensor::new(data, out_shape));
        }
    }

    // General path: neither operand alone spans the full output (e.g.
    // `[m,1] op [1,n]`), so walk an "odometer" over `out_shape`'s
    // coordinates instead of `broadcast_to`-ing both operands in full and
    // instead of computing `idx / out_strides[i] % out_shape[i]` (a div and
    // a mod per dimension, per output element): increment the innermost
    // coordinate and each operand's flat index together, carrying into
    // outer dimensions only when a coordinate wraps — amortised O(1) per
    // output element rather than O(ndim) divisions/modulos per element.
    let a_strides = broadcast_strides(&a.shape, &out_shape);
    let b_strides = broadcast_strides(&b.shape, &out_shape);
    let mut data = Vec::with_capacity(out_numel);
    let mut coord = vec![0usize; n];
    let mut a_idx = 0usize;
    let mut b_idx = 0usize;
    for _ in 0..out_numel {
        data.push(op(a.data[a_idx], b.data[b_idx]));
        for i in (0..n).rev() {
            coord[i] += 1;
            a_idx += a_strides[i];
            b_idx += b_strides[i];
            if coord[i] < out_shape[i] {
                break;
            }
            // Carry: dimension i wrapped, undo its full contribution and
            // roll into dimension i-1 on the next loop iteration.
            a_idx -= a_strides[i] * out_shape[i];
            b_idx -= b_strides[i] * out_shape[i];
            coord[i] = 0;
        }
    }
    Ok(Tensor::new(data, out_shape))
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_add(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x + y)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_sub(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x - y)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_mul(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x * y)
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_div(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x / y)
}

pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    elementwise_binary(a, b, |x, y| x.powf(y))
}

pub fn sqrt(a: &Tensor) -> Tensor {
    #[cfg(feature = "simd")]
    {
        let mut data = a.data.clone();
        crate::simd_ops::simd_sqrt(&mut data);
        Tensor::new(data, a.shape.clone())
    }
    #[cfg(not(feature = "simd"))]
    Tensor::new(a.data.iter().map(|x| x.sqrt()).collect(), a.shape.clone())
}

pub fn reciprocal(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.recip()).collect(), a.shape.clone())
}

pub fn neg(a: &Tensor) -> Tensor {
    #[cfg(feature = "simd")]
    {
        let mut data = a.data.clone();
        crate::simd_ops::simd_neg(&mut data);
        Tensor::new(data, a.shape.clone())
    }
    #[cfg(not(feature = "simd"))]
    Tensor::new(a.data.iter().map(|x| -x).collect(), a.shape.clone())
}

// ── Perf smoke test: before/after timing note for W2-perf-matmul a6-3 ──────
//
// `#[ignore]`d; run with:
//   cargo test -p oxionnx-ops --release --lib math::broadcast::perf_smoke -- --ignored --nocapture
#[cfg(test)]
mod perf_smoke {
    use super::*;
    use std::time::Instant;

    fn timed(label: &str, iters: u32, mut f: impl FnMut()) {
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
    fn perf_broadcast_add_bias() {
        // The bias-add case from a6-3: [1,512,768] + [768], default (non-simd) build.
        let a = Tensor::new(vec![0.5f32; 512 * 768], vec![1, 512, 768]);
        let bias = Tensor::new(vec![0.25f32; 768], vec![768]);
        timed("add [1,512,768]+[768]", 2000, || {
            let _ = add(&a, &bias).expect("add");
        });
    }
}
