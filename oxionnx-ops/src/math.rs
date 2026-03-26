use oxionnx_core::Tensor;

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

fn elementwise_binary(
    a: &Tensor,
    b: &Tensor,
    op: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let a = broadcast_to(a, &out_shape);
    let b = broadcast_to(b, &out_shape);
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
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
    elementwise_binary(a, b, |x, y| x / y)
}

pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    elementwise_binary(a, b, |x, y| x.powf(y))
}

pub fn sqrt(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.sqrt()).collect(), a.shape.clone())
}

pub fn reciprocal(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.recip()).collect(), a.shape.clone())
}

pub fn neg(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| -x).collect(), a.shape.clone())
}

// ── Generic reduction helper ────────────────────────────────────────────────

fn reduce_with<F, G>(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    init: f32,
    accumulate: F,
    finalize: G,
) -> Result<Tensor, String>
where
    F: Fn(f32, f32) -> f32,
    G: Fn(f32, u32) -> f32,
{
    let ndim = x.ndim();
    let axes: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter()
            .map(|&a| {
                if a < 0 {
                    (a + ndim as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    };

    let mut out_shape: Vec<usize> = x.shape.clone();
    for &ax in &axes {
        out_shape[ax] = 1;
    }

    let out_n: usize = out_shape.iter().product();
    let mut acc = vec![init; out_n];
    let mut counts = vec![0u32; out_n];

    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= x.shape[i];
    }
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    for in_idx in 0..x.numel() {
        let mut rem = in_idx;
        let mut out_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            if !axes.contains(&d) {
                out_idx += coord * out_strides[d];
            }
        }
        acc[out_idx] = accumulate(acc[out_idx], x.data[in_idx]);
        counts[out_idx] += 1;
    }

    let data: Vec<f32> = acc
        .iter()
        .zip(counts.iter())
        .map(|(&a, &c)| finalize(a, c))
        .collect();

    if keepdims {
        Ok(Tensor::new(data, out_shape))
    } else {
        let final_shape: Vec<usize> = out_shape
            .into_iter()
            .enumerate()
            .filter_map(|(i, d)| if axes.contains(&i) { None } else { Some(d) })
            .collect();
        let final_shape = if final_shape.is_empty() {
            vec![1]
        } else {
            final_shape
        };
        Ok(Tensor::new(data, final_shape))
    }
}

pub fn reduce_mean(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, c| s / c as f32)
}

pub fn reduce_sum(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s)
}

pub fn reduce_max(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(
        x,
        axes,
        keepdims,
        f32::NEG_INFINITY,
        |a, v| a.max(v),
        |s, _| s,
    )
}

pub fn reduce_min(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, f32::INFINITY, |a, v| a.min(v), |s, _| s)
}

pub fn reduce_prod(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 1.0, |a, v| a * v, |s, _| s)
}

// ── ArgMax / ArgMin ─────────────────────────────────────────────────────────

/// Index of the maximum value along `axis`.
pub fn arg_max(x: &Tensor, axis: i64, keepdims: bool) -> Result<Tensor, String> {
    arg_reduce(x, axis, keepdims, true)
}

/// Index of the minimum value along `axis`.
pub fn arg_min(x: &Tensor, axis: i64, keepdims: bool) -> Result<Tensor, String> {
    arg_reduce(x, axis, keepdims, false)
}

fn arg_reduce(x: &Tensor, axis: i64, keepdims: bool, find_max: bool) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "arg_reduce: axis {ax} out of range for {ndim}D tensor"
        ));
    }

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
    let axis_len = x.shape[ax];

    let mut result = vec![0.0f32; outer * inner];

    for o in 0..outer {
        for i in 0..inner {
            let base = x.data[o * axis_len * inner + i];
            let (mut best_val, mut best_idx) = (base, 0usize);
            for j in 1..axis_len {
                let v = x.data[o * axis_len * inner + j * inner + i];
                let better = if find_max { v > best_val } else { v < best_val };
                if better {
                    best_val = v;
                    best_idx = j;
                }
            }
            result[o * inner + i] = best_idx as f32;
        }
    }

    if keepdims {
        let mut out_shape = x.shape.clone();
        out_shape[ax] = 1;
        Ok(Tensor::new(result, out_shape))
    } else {
        let mut final_shape = x.shape.clone();
        final_shape.remove(ax);
        let final_shape = if final_shape.is_empty() {
            vec![1]
        } else {
            final_shape
        };
        Ok(Tensor::new(result, final_shape))
    }
}

// ── CumSum ──────────────────────────────────────────────────────────────────

/// Prefix sum (inclusive or exclusive) along `axis`.
pub fn cumsum(x: &Tensor, axis: i64, exclusive: bool, reverse: bool) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("cumsum: axis {ax} out of range for {ndim}D tensor"));
    }

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
    let axis_len = x.shape[ax];

    let mut data = x.data.clone();

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
            if !reverse {
                for j in 0..axis_len {
                    let idx = o * axis_len * inner + j * inner + i;
                    let val = x.data[idx];
                    if exclusive {
                        data[idx] = acc;
                        acc += val;
                    } else {
                        acc += val;
                        data[idx] = acc;
                    }
                }
            } else {
                for j in (0..axis_len).rev() {
                    let idx = o * axis_len * inner + j * inner + i;
                    let val = x.data[idx];
                    if exclusive {
                        data[idx] = acc;
                        acc += val;
                    } else {
                        acc += val;
                        data[idx] = acc;
                    }
                }
            }
        }
    }

    Ok(Tensor::new(data, x.shape.clone()))
}

// ── Range ───────────────────────────────────────────────────────────────────

/// Generate `[start, start+delta, ...]` up to (not including) `limit`.
pub fn range(start: f32, limit: f32, delta: f32) -> Result<Tensor, String> {
    if delta == 0.0 {
        return Err("range: delta cannot be zero".into());
    }
    let count = if delta > 0.0 {
        ((limit - start) / delta).ceil().max(0.0) as usize
    } else {
        ((start - limit) / (-delta)).ceil().max(0.0) as usize
    };
    let data: Vec<f32> = (0..count).map(|i| start + i as f32 * delta).collect();
    Ok(Tensor::new(data, vec![count]))
}

// ── TopK ────────────────────────────────────────────────────────────────────

/// Top-k values and indices along `axis`.
/// Returns `(values, indices)`.
pub fn top_k(
    x: &Tensor,
    k: usize,
    axis: i64,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor, Tensor), String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("top_k: axis {ax} out of range for {ndim}D tensor"));
    }
    let k = k.min(x.shape[ax]);

    let mut out_shape = x.shape.clone();
    out_shape[ax] = k;
    let out_n: usize = out_shape.iter().product();

    let mut values = vec![0.0f32; out_n];
    let mut indices = vec![0.0f32; out_n];

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
    let axis_len = x.shape[ax];

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(f32, usize)> = (0..axis_len)
                .map(|j| (x.data[o * axis_len * inner + j * inner + i], j))
                .collect();

            if largest {
                pairs.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                pairs.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            // If not sorted by value, sort top-k by original index
            if !sorted {
                pairs[..k].sort_unstable_by_key(|p| p.1);
            }

            for (ki, (v, j)) in pairs[..k].iter().enumerate() {
                let dst = o * k * inner + ki * inner + i;
                values[dst] = *v;
                indices[dst] = *j as f32;
            }
        }
    }

    Ok((
        Tensor::new(values, out_shape.clone()),
        Tensor::new(indices, out_shape),
    ))
}

// ── MatMul / Gemm ───────────────────────────────────────────────────────────

/// Matrix multiplication supporting batched tensors.
/// Last two dims: [M, K] @ [K, N] = [M, N]
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
    let out_size = batch_size * m * n;

    let mut out = vec![0.0f32; out_size];

    if m >= 4 {
        for b_idx in 0..batch_size {
            let a_off = (b_idx % (a.numel() / (m * k))) * a_batch_stride;
            let b_off = (b_idx % (b.numel() / (k * n))) * b_batch_stride;
            let c_off = b_idx * m * n;
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    a.data[a_off..].as_ptr(),
                    k as isize,
                    1,
                    b.data[b_off..].as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    out[c_off..].as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        }
    } else {
        for b_idx in 0..batch_size {
            let a_off = (b_idx % (a.numel() / (m * k))) * a_batch_stride;
            let b_off = (b_idx % (b.numel() / (k * n))) * b_batch_stride;
            let c_off = b_idx * m * n;
            for i in 0..m {
                let a_row = &a.data[a_off + i * k..a_off + (i + 1) * k];
                for j in 0..n {
                    let mut s = 0.0f32;
                    for (kk, &a_val) in a_row.iter().enumerate() {
                        s += a_val * b.data[b_off + kk * n + j];
                    }
                    out[c_off + i * n + j] = s;
                }
            }
        }
    }

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

fn transpose_2d(t: &Tensor) -> Result<Tensor, String> {
    if t.ndim() != 2 {
        return Err(format!("transpose_2d: expected 2D, got {}D", t.ndim()));
    }
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = t.data[r * cols + c];
        }
    }
    Ok(Tensor::new(out, vec![cols, rows]))
}

// ── Unary element-wise: trig & rounding ─────────────────────────────────────

pub fn ceil(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.ceil()).collect(), x.shape.clone())
}

pub fn floor_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.floor()).collect(), x.shape.clone())
}

/// Round half to even (banker's rounding), matching ONNX spec.
pub fn round_op(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| {
                let rounded = v.round();
                // When exactly halfway, round to even
                if (v - v.floor() - 0.5).abs() < f32::EPSILON {
                    if rounded as i64 % 2 != 0 {
                        rounded - v.signum()
                    } else {
                        rounded
                    }
                } else {
                    rounded
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

/// Sign function: -1 for negative, 0 for zero, 1 for positive (ONNX convention).
pub fn sign(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| {
                if *v > 0.0 {
                    1.0
                } else if *v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

pub fn sin_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.sin()).collect(), x.shape.clone())
}

pub fn cos_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.cos()).collect(), x.shape.clone())
}

pub fn tan_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.tan()).collect(), x.shape.clone())
}

pub fn asin_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.asin()).collect(), x.shape.clone())
}

pub fn acos_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.acos()).collect(), x.shape.clone())
}

pub fn atan_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.atan()).collect(), x.shape.clone())
}

pub fn sinh_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.sinh()).collect(), x.shape.clone())
}

pub fn cosh_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.cosh()).collect(), x.shape.clone())
}

pub fn asinh_op(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| (*v + (v * v + 1.0).sqrt()).ln())
            .collect(),
        x.shape.clone(),
    )
}

pub fn acosh_op(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| (*v + (v * v - 1.0).sqrt()).ln())
            .collect(),
        x.shape.clone(),
    )
}

pub fn atanh_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.atanh()).collect(), x.shape.clone())
}

// ── Binary element-wise: mod, bitshift ──────────────────────────────────────

/// Mod operation. fmod=1 uses floating-point remainder, fmod=0 uses truncated integer mod.
pub fn mod_op(a: &Tensor, b: &Tensor, fmod: i64) -> Result<Tensor, String> {
    let target = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let ab = broadcast_to(a, &target);
    let bb = broadcast_to(b, &target);
    let data: Vec<f32> = if fmod != 0 {
        ab.data
            .iter()
            .zip(bb.data.iter())
            .map(|(x, y)| x % y)
            .collect()
    } else {
        ab.data
            .iter()
            .zip(bb.data.iter())
            .map(|(x, y)| {
                let t = (x / y).trunc();
                x - t * y
            })
            .collect()
    };
    Ok(Tensor::new(data, target))
}

/// Bit shift left or right. `direction` must be `"LEFT"` or `"RIGHT"`.
pub fn bit_shift(x: &Tensor, y: &Tensor, direction: &str) -> Result<Tensor, String> {
    let target = Tensor::broadcast_shape(&x.shape, &y.shape)?;
    let xb = broadcast_to(x, &target);
    let yb = broadcast_to(y, &target);
    let data: Vec<f32> = if direction == "LEFT" {
        xb.data
            .iter()
            .zip(yb.data.iter())
            .map(|(a, b)| {
                let ai = *a as u32;
                let bi = *b as u32;
                (ai << bi) as f32
            })
            .collect()
    } else {
        xb.data
            .iter()
            .zip(yb.data.iter())
            .map(|(a, b)| {
                let ai = *a as u32;
                let bi = *b as u32;
                (ai >> bi) as f32
            })
            .collect()
    };
    Ok(Tensor::new(data, target))
}

// ── Variadic operators ──────────────────────────────────────────────────────

/// Element-wise minimum across multiple tensors.
pub fn variadic_min(tensors: &[&Tensor]) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("variadic_min: no inputs".into());
    }
    let mut result = tensors[0].clone();
    for t in &tensors[1..] {
        let target = Tensor::broadcast_shape(&result.shape, &t.shape)?;
        let rb = broadcast_to(&result, &target);
        let tb = broadcast_to(t, &target);
        let data: Vec<f32> = rb
            .data
            .iter()
            .zip(tb.data.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        result = Tensor::new(data, target);
    }
    Ok(result)
}

/// Element-wise maximum across multiple tensors.
pub fn variadic_max(tensors: &[&Tensor]) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("variadic_max: no inputs".into());
    }
    let mut result = tensors[0].clone();
    for t in &tensors[1..] {
        let target = Tensor::broadcast_shape(&result.shape, &t.shape)?;
        let rb = broadcast_to(&result, &target);
        let tb = broadcast_to(t, &target);
        let data: Vec<f32> = rb
            .data
            .iter()
            .zip(tb.data.iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        result = Tensor::new(data, target);
    }
    Ok(result)
}

/// Element-wise sum across multiple tensors.
pub fn variadic_sum(tensors: &[&Tensor]) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("variadic_sum: no inputs".into());
    }
    let mut result = tensors[0].clone();
    for t in &tensors[1..] {
        let target = Tensor::broadcast_shape(&result.shape, &t.shape)?;
        let rb = broadcast_to(&result, &target);
        let tb = broadcast_to(t, &target);
        let data: Vec<f32> = rb
            .data
            .iter()
            .zip(tb.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        result = Tensor::new(data, target);
    }
    Ok(result)
}

/// Element-wise mean across multiple tensors.
pub fn variadic_mean(tensors: &[&Tensor]) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("variadic_mean: no inputs".into());
    }
    let sum = variadic_sum(tensors)?;
    let count = tensors.len() as f32;
    let data: Vec<f32> = sum.data.iter().map(|v| v / count).collect();
    Ok(Tensor::new(data, sum.shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_broadcast_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0], vec![1]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_matmul_2x3_3x4() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![3, 4],
        );
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, vec![2, 4]);
        assert!((c.data[0] - 1.0).abs() < 1e-5);
        assert!((c.data[1] - 2.0).abs() < 1e-5);
        assert!((c.data[4] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_mean() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let m = reduce_mean(&t, &[1], false).unwrap();
        assert_eq!(m.shape, vec![2]);
        assert!((m.data[0] - 2.0).abs() < 1e-5);
        assert!((m.data[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_sum() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = reduce_sum(&t, &[1], false).unwrap();
        assert_eq!(s.shape, vec![2]);
        assert!((s.data[0] - 6.0).abs() < 1e-5);
        assert!((s.data[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_max_keepdims() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
        let m = reduce_max(&t, &[1], true).unwrap();
        assert_eq!(m.shape, vec![2, 1]);
        assert_eq!(m.data, vec![5.0, 3.0]);
    }

    #[test]
    fn test_reduce_min() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
        let m = reduce_min(&t, &[1], false).unwrap();
        assert_eq!(m.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_reduce_prod() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let p = reduce_prod(&t, &[1], false).unwrap();
        assert_eq!(p.data, vec![2.0, 12.0]);
    }

    #[test]
    fn test_arg_max() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 4.0, 0.0], vec![2, 3]);
        let idx = arg_max(&t, 1, false).unwrap();
        assert_eq!(idx.shape, vec![2]);
        assert_eq!(idx.data[0], 1.0); // max of [1,5,3] is at index 1
        assert_eq!(idx.data[1], 1.0); // max of [2,4,0] is at index 1
    }

    #[test]
    fn test_arg_min() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 4.0, 0.0], vec![2, 3]);
        let idx = arg_min(&t, 1, false).unwrap();
        assert_eq!(idx.data[0], 0.0); // min of [1,5,3] is at index 0
        assert_eq!(idx.data[1], 2.0); // min of [2,4,0] is at index 2
    }

    #[test]
    fn test_cumsum_inclusive() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let c = cumsum(&t, 0, false, false).unwrap();
        assert_eq!(c.data, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_exclusive() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let c = cumsum(&t, 0, true, false).unwrap();
        assert_eq!(c.data, vec![0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_range() {
        let r = range(0.0, 5.0, 1.0).unwrap();
        assert_eq!(r.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.shape, vec![5]);
    }

    #[test]
    fn test_range_negative_delta() {
        let r = range(5.0, 0.0, -1.0).unwrap();
        assert_eq!(r.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_top_k_largest() {
        let t = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], vec![8]);
        let (vals, idxs) = top_k(&t, 3, 0, true, true).unwrap();
        assert_eq!(vals.shape, vec![3]);
        assert!((vals.data[0] - 9.0).abs() < 1e-5);
        assert!((vals.data[1] - 6.0).abs() < 1e-5);
        assert!((vals.data[2] - 5.0).abs() < 1e-5);
        assert_eq!(idxs.data[0], 5.0); // 9 is at index 5
    }

    #[test]
    fn test_top_k_smallest() {
        let t = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
        let (vals, _) = top_k(&t, 2, 0, false, true).unwrap();
        assert_eq!(vals.shape, vec![2]);
        assert!((vals.data[0] - 1.0).abs() < 1e-5);
        assert!((vals.data[1] - 1.0).abs() < 1e-5);
    }

    // ── Unary math ops tests ────────────────────────────────────────────────

    #[test]
    fn test_ceil() {
        let t = Tensor::new(vec![1.5, -1.5, 0.0, 2.3], vec![4]);
        let r = ceil(&t);
        assert_eq!(r.data, vec![2.0, -1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_floor_op() {
        let t = Tensor::new(vec![1.5, -1.5, 0.0, 2.9], vec![4]);
        let r = floor_op(&t);
        assert_eq!(r.data, vec![1.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_round_op() {
        let t = Tensor::new(vec![1.5, 2.5, 0.4, -0.6], vec![4]);
        let r = round_op(&t);
        // Rust uses banker's rounding: 1.5 -> 2.0, 2.5 -> 2.0
        assert_eq!(r.data, vec![2.0, 2.0, 0.0, -1.0]);
    }

    #[test]
    fn test_sign() {
        let t = Tensor::new(vec![-3.0, 0.0, 5.0, -0.5], vec![4]);
        let r = sign(&t);
        assert_eq!(r.data, vec![-1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_sin_cos_tan() {
        let t = Tensor::new(vec![0.0], vec![1]);
        let s = sin_op(&t);
        let c = cos_op(&t);
        let ta = tan_op(&t);
        assert!((s.data[0] - 0.0).abs() < 1e-5);
        assert!((c.data[0] - 1.0).abs() < 1e-5);
        assert!((ta.data[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_asin_acos_atan() {
        let t = Tensor::new(vec![0.0, 0.5], vec![2]);
        let as_r = asin_op(&t);
        let ac_r = acos_op(&t);
        let at_r = atan_op(&t);
        assert!((as_r.data[0] - 0.0).abs() < 1e-5);
        assert!((ac_r.data[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!((at_r.data[0] - 0.0).abs() < 1e-5);
        // asin(0.5) = pi/6
        assert!((as_r.data[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-5);
    }

    #[test]
    fn test_sinh_cosh() {
        let t = Tensor::new(vec![0.0, 1.0], vec![2]);
        let sh = sinh_op(&t);
        let ch = cosh_op(&t);
        assert!((sh.data[0] - 0.0).abs() < 1e-5);
        assert!((ch.data[0] - 1.0).abs() < 1e-5);
        assert!((sh.data[1] - 1.0_f32.sinh()).abs() < 1e-5);
        assert!((ch.data[1] - 1.0_f32.cosh()).abs() < 1e-5);
    }

    #[test]
    fn test_asinh_acosh_atanh() {
        let t_asinh = Tensor::new(vec![0.0, 1.0], vec![2]);
        let r = asinh_op(&t_asinh);
        assert!((r.data[0] - 0.0).abs() < 1e-4);
        assert!((r.data[1] - 1.0_f32.asinh()).abs() < 1e-4);

        let t_acosh = Tensor::new(vec![1.0, 2.0], vec![2]);
        let r2 = acosh_op(&t_acosh);
        assert!((r2.data[0] - 0.0).abs() < 1e-4);
        assert!((r2.data[1] - 2.0_f32.acosh()).abs() < 1e-4);

        let t_atanh = Tensor::new(vec![0.0, 0.5], vec![2]);
        let r3 = atanh_op(&t_atanh);
        assert!((r3.data[0] - 0.0).abs() < 1e-5);
        assert!((r3.data[1] - 0.5_f32.atanh()).abs() < 1e-5);
    }

    // ── Binary math ops tests ───────────────────────────────────────────────

    #[test]
    fn test_mod_op_fmod() {
        let a = Tensor::new(vec![7.0, -7.0], vec![2]);
        let b = Tensor::new(vec![3.0], vec![1]);
        let r = mod_op(&a, &b, 1).expect("mod_op fmod failed");
        assert!((r.data[0] - 1.0).abs() < 1e-5); // 7 % 3 = 1
        assert!((r.data[1] - (-1.0)).abs() < 1e-5); // -7 % 3 = -1
    }

    #[test]
    fn test_mod_op_truncated() {
        let a = Tensor::new(vec![7.0, -7.0], vec![2]);
        let b = Tensor::new(vec![3.0], vec![1]);
        let r = mod_op(&a, &b, 0).expect("mod_op truncated failed");
        assert!((r.data[0] - 1.0).abs() < 1e-5);
        assert!((r.data[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_bit_shift_left() {
        let x = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]);
        let y = Tensor::new(vec![2.0], vec![1]);
        let r = bit_shift(&x, &y, "LEFT").expect("bit_shift left failed");
        assert_eq!(r.data, vec![4.0, 8.0, 16.0]);
    }

    #[test]
    fn test_bit_shift_right() {
        let x = Tensor::new(vec![16.0, 8.0, 4.0], vec![3]);
        let y = Tensor::new(vec![2.0], vec![1]);
        let r = bit_shift(&x, &y, "RIGHT").expect("bit_shift right failed");
        assert_eq!(r.data, vec![4.0, 2.0, 1.0]);
    }

    // ── Variadic ops tests ──────────────────────────────────────────────────

    #[test]
    fn test_variadic_min() {
        let a = Tensor::new(vec![5.0, 2.0, 8.0], vec![3]);
        let b = Tensor::new(vec![3.0, 6.0, 1.0], vec![3]);
        let c = Tensor::new(vec![4.0, 1.0, 9.0], vec![3]);
        let r = variadic_min(&[&a, &b, &c]).expect("variadic_min failed");
        assert_eq!(r.data, vec![3.0, 1.0, 1.0]);
    }

    #[test]
    fn test_variadic_max() {
        let a = Tensor::new(vec![5.0, 2.0, 8.0], vec![3]);
        let b = Tensor::new(vec![3.0, 6.0, 1.0], vec![3]);
        let c = Tensor::new(vec![4.0, 1.0, 9.0], vec![3]);
        let r = variadic_max(&[&a, &b, &c]).expect("variadic_max failed");
        assert_eq!(r.data, vec![5.0, 6.0, 9.0]);
    }

    #[test]
    fn test_variadic_sum() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]);
        let r = variadic_sum(&[&a, &b, &c]).expect("variadic_sum failed");
        assert_eq!(r.data, vec![12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_variadic_mean() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]);
        let r = variadic_mean(&[&a, &b, &c]).expect("variadic_mean failed");
        assert!((r.data[0] - 4.0).abs() < 1e-5);
        assert!((r.data[1] - 5.0).abs() < 1e-5);
        assert!((r.data[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_variadic_empty() {
        assert!(variadic_min(&[]).is_err());
        assert!(variadic_max(&[]).is_err());
        assert!(variadic_sum(&[]).is_err());
        assert!(variadic_mean(&[]).is_err());
    }
}
