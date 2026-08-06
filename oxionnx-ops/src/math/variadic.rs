use oxionnx_core::Tensor;

use super::broadcast::broadcast_to;

// ── Binary element-wise: mod, bitshift ──────────────────────────────────────

/// Mod operation.
///
/// - `fmod=1`: C `fmod` semantics -- the result takes the sign of the
///   *dividend*. Rust's `%` on floats already implements this directly.
/// - `fmod=0` (default): ONNX specifies numpy `mod` semantics -- the result
///   takes the sign of the *divisor* (Python's `%`), computed here as
///   `x - floor(x/y)*y`. This is NOT the same as C's truncated modulo: e.g.
///   `Mod(-7, 3)` must be `2` (sign of divisor `3`), not `-1` (sign of
///   dividend `-7`, which truncated modulo would give).
///
/// Division by zero (either branch) naturally yields NaN: `y == 0.0` drives
/// `x / y` to +/-inf or NaN, and `0.0 * inf == NaN` per IEEE 754, so no
/// special-casing is needed to avoid a panic here (float division/remainder
/// by zero never traps, unlike integer division).
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
            .map(|(x, y)| x - (x / y).floor() * y)
            .collect()
    };
    Ok(Tensor::new(data, target))
}

/// Bit shift left or right. `direction` must be `"LEFT"` or `"RIGHT"`.
///
/// ONNX BitShift operates on *unsigned* integer types only, so operands are
/// recovered as u64 (a `v as u32` round-trip previously truncated any
/// logical value above 2^32-1). The shift amount is bounds-checked with
/// `checked_shl`/`checked_shr` instead of the raw `<<`/`>>` operators: a
/// shift amount >= the operand's bit width is a Rust panic ("attempt to
/// shift left/right with overflow" in debug builds), which a malformed or
/// adversarial model could otherwise trigger; shifting all bits out
/// saturates to 0, matching what shifting a 64-bit value by >=64 positions
/// means mathematically.
pub fn bit_shift(x: &Tensor, y: &Tensor, direction: &str) -> Result<Tensor, String> {
    let target = Tensor::broadcast_shape(&x.shape, &y.shape)?;
    let xb = broadcast_to(x, &target);
    let yb = broadcast_to(y, &target);
    let shift_amount = |b: f32| -> u32 { (b as u64).try_into().unwrap_or(u32::MAX) };
    let data: Vec<f32> = if direction == "LEFT" {
        xb.data
            .iter()
            .zip(yb.data.iter())
            .map(|(a, b)| {
                let ai = *a as u64;
                let bi = shift_amount(*b);
                ai.checked_shl(bi).unwrap_or(0) as f32
            })
            .collect()
    } else {
        xb.data
            .iter()
            .zip(yb.data.iter())
            .map(|(a, b)| {
                let ai = *a as u64;
                let bi = shift_amount(*b);
                ai.checked_shr(bi).unwrap_or(0) as f32
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
