use oxionnx_core::Tensor;

#[cfg(feature = "simd")]
use super::broadcast::is_full_reduction;
use super::broadcast::sub;

pub(super) fn reduce_with<F, G>(
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
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_mean(&x.data);
            let shape = if keepdims { vec![1; x.ndim()] } else { vec![1] };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, c| s / c as f32)
}

pub fn reduce_sum(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_sum(&x.data);
            let shape = if keepdims { vec![1; x.ndim()] } else { vec![1] };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s)
}

pub fn reduce_max(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_max(&x.data);
            let shape = if keepdims { vec![1; x.ndim()] } else { vec![1] };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
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
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_min(&x.data);
            let shape = if keepdims { vec![1; x.ndim()] } else { vec![1] };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(x, axes, keepdims, f32::INFINITY, |a, v| a.min(v), |s, _| s)
}

pub fn reduce_prod(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 1.0, |a, v| a * v, |s, _| s)
}

/// ReduceL1: sum(|x|)
pub fn reduce_l1(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v.abs(), |s, _| s)
}

/// ReduceL2: sqrt(sum(x^2))
pub fn reduce_l2(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v * v, |s, _| s.sqrt())
}

/// ReduceLogSum: log(sum(x))
pub fn reduce_log_sum(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s.ln())
}

/// ReduceSumSquare: sum(x^2)
pub fn reduce_sum_square(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v * v, |s, _| s)
}

/// ReduceLogSumExp: log(sum(exp(x))) — numerically stable via max-subtract trick.
pub fn reduce_log_sum_exp(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    // Step 1: max with keepdims=true for broadcasting
    let max_keep = reduce_max(x, axes, true)?;
    // Step 2: x - max (sub handles broadcasting internally)
    let shifted = sub(x, &max_keep)?;
    // Step 3: sum(exp(shifted)) with requested keepdims
    let exp_data: Vec<f32> = shifted.data.iter().map(|v| v.exp()).collect();
    let exp_tensor = Tensor::new(exp_data, shifted.shape.clone());
    let sum_exp = reduce_sum(&exp_tensor, axes, keepdims)?;
    // Step 4: log(sum_exp) + max_final
    let max_final = if keepdims {
        max_keep
    } else {
        reduce_max(x, axes, false)?
    };
    let out_data: Vec<f32> = sum_exp
        .data
        .iter()
        .zip(max_final.data.iter())
        .map(|(&s, &m)| s.ln() + m)
        .collect();
    Ok(Tensor::new(out_data, sum_exp.shape.clone()))
}
