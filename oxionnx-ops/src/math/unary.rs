use oxionnx_core::Tensor;

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
