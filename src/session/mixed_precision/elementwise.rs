//! Native f16 element-wise operation implementations.
//!
//! Converts inputs to `half::f16`, performs the operation, and returns
//! results as standard f32 `Tensor` values with f16-rounded precision.

use crate::tensor::Tensor;
use crate::OnnxError;

use super::broadcast::broadcast_binary_f16;

/// Execute an element-wise operation natively in f16 precision.
///
/// Converts inputs to `half::f16`, performs the operation, and returns
/// the result as a standard f32 `Tensor` with f16-rounded values.
///
/// Returns `None` if the op is not supported for native f16 execution,
/// in which case the caller should fall back to the normal f32 path
/// (optionally with f16 rounding of outputs).
pub fn execute_elementwise_f16(
    op_type: &str,
    inputs: &[&Tensor],
) -> Option<Result<Vec<Tensor>, OnnxError>> {
    match op_type {
        "Relu" => Some(execute_relu_f16(inputs)),
        "Add" => Some(execute_add_f16(inputs)),
        "Mul" => Some(execute_mul_f16(inputs)),
        "Sub" => Some(execute_sub_f16(inputs)),
        "Sigmoid" => Some(execute_sigmoid_f16(inputs)),
        "Tanh" => Some(execute_tanh_f16(inputs)),
        "Neg" => Some(execute_neg_f16(inputs)),
        "Abs" => Some(execute_abs_f16(inputs)),
        _ => None,
    }
}

fn execute_relu_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Relu f16: expected at least 1 input".to_string())
    })?;
    let zero = half::f16::ZERO;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            let h = half::f16::from_f32(v);
            if h < zero { zero } else { h }.to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_add_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "Add f16: expected 2 inputs".to_string(),
        ));
    }
    let a = inputs[0];
    let b = inputs[1];

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(|e| OnnxError::ShapeMismatch(format!("Add f16 broadcast: {e}")))?;
    let out_size: usize = out_shape.iter().product();

    let data = if a.shape == b.shape {
        // Fast path: same shape, no broadcasting needed
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&va, &vb)| {
                let ha = half::f16::from_f32(va);
                let hb = half::f16::from_f32(vb);
                (ha + hb).to_f32()
            })
            .collect()
    } else {
        broadcast_binary_f16(
            &a.data,
            &a.shape,
            &b.data,
            &b.shape,
            &out_shape,
            out_size,
            |ha, hb| ha + hb,
        )
    };

    Ok(vec![Tensor::new(data, out_shape)])
}

fn execute_mul_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "Mul f16: expected 2 inputs".to_string(),
        ));
    }
    let a = inputs[0];
    let b = inputs[1];

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(|e| OnnxError::ShapeMismatch(format!("Mul f16 broadcast: {e}")))?;
    let out_size: usize = out_shape.iter().product();

    let data = if a.shape == b.shape {
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&va, &vb)| {
                let ha = half::f16::from_f32(va);
                let hb = half::f16::from_f32(vb);
                (ha * hb).to_f32()
            })
            .collect()
    } else {
        broadcast_binary_f16(
            &a.data,
            &a.shape,
            &b.data,
            &b.shape,
            &out_shape,
            out_size,
            |ha, hb| ha * hb,
        )
    };

    Ok(vec![Tensor::new(data, out_shape)])
}

fn execute_sub_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "Sub f16: expected 2 inputs".to_string(),
        ));
    }
    let a = inputs[0];
    let b = inputs[1];

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(|e| OnnxError::ShapeMismatch(format!("Sub f16 broadcast: {e}")))?;
    let out_size: usize = out_shape.iter().product();

    let data = if a.shape == b.shape {
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&va, &vb)| {
                let ha = half::f16::from_f32(va);
                let hb = half::f16::from_f32(vb);
                (ha - hb).to_f32()
            })
            .collect()
    } else {
        broadcast_binary_f16(
            &a.data,
            &a.shape,
            &b.data,
            &b.shape,
            &out_shape,
            out_size,
            |ha, hb| ha - hb,
        )
    };

    Ok(vec![Tensor::new(data, out_shape)])
}

fn execute_sigmoid_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Sigmoid f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            // Compute sigmoid in f16: 1 / (1 + exp(-x))
            let h = half::f16::from_f32(v);
            let neg_h = -h;
            let exp_neg = half::f16::from_f32(neg_h.to_f32().exp());
            let one = half::f16::ONE;
            let denom = one + exp_neg;
            half::f16::from_f32(one.to_f32() / denom.to_f32()).to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_tanh_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Tanh f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            let h = half::f16::from_f32(v);
            half::f16::from_f32(h.to_f32().tanh()).to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_neg_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Neg f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| (-half::f16::from_f32(v)).to_f32())
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_abs_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Abs f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            let h = half::f16::from_f32(v);
            half::f16::from_f32(h.to_f32().abs()).to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}
