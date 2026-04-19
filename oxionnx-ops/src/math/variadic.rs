use oxionnx_core::Tensor;

use super::broadcast::broadcast_to;

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
