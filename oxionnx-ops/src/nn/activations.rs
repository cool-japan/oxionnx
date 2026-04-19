//! Activation functions: relu, sigmoid, tanh, gelu, silu and variants.

use oxionnx_core::Tensor;

pub fn relu(x: &Tensor) -> Tensor {
    relu_impl(x)
}

#[cfg(feature = "simd")]
fn relu_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_relu(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn relu_impl(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| v.max(0.0)).collect(),
        x.shape.clone(),
    )
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    sigmoid_impl(x)
}

#[cfg(feature = "simd")]
fn sigmoid_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_sigmoid(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn sigmoid_impl(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
        x.shape.clone(),
    )
}

pub fn tanh_op(x: &Tensor) -> Tensor {
    tanh_op_impl(x)
}

#[cfg(feature = "simd")]
fn tanh_op_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_tanh(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn tanh_op_impl(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|&v| v.tanh()).collect(), x.shape.clone())
}

/// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Tensor) -> Tensor {
    gelu_impl(x)
}

#[cfg(feature = "simd")]
fn gelu_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_gelu(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn gelu_impl(x: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF: f32 = 0.044_715;
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
                0.5 * v * (1.0 + inner.tanh())
            })
            .collect(),
        x.shape.clone(),
    )
}

/// LeakyRelu: f(x) = x if x >= 0, alpha * x if x < 0
pub fn leaky_relu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect(),
        x.shape.clone(),
    )
}

/// PRelu: `f(x) = x if x >= 0, slope[c] * x if x < 0`
/// slope shape is typically \[C\] or \[1, C, 1, 1\] -- broadcast per-channel
pub fn prelu(x: &Tensor, slope: &Tensor) -> Tensor {
    let slope_numel = slope.numel();
    if slope_numel == 1 {
        // scalar slope
        let alpha = slope.data[0];
        return Tensor::new(
            x.data
                .iter()
                .map(|&v| if v >= 0.0 { v } else { alpha * v })
                .collect(),
            x.shape.clone(),
        );
    }

    // Per-channel: x is [N, C, ...], slope is [C] or [1, C, 1, 1]
    let c = slope_numel;

    if x.ndim() >= 2 {
        let spatial: usize = if x.ndim() > 2 {
            x.shape[2..].iter().product()
        } else {
            1
        };
        let n = x.shape[0];
        let x_c = x.shape[1];

        let mut data = x.data.clone();
        if x_c == c {
            for ni in 0..n {
                for ci in 0..c {
                    let alpha = slope.data[ci];
                    for si in 0..spatial {
                        let idx = ni * c * spatial + ci * spatial + si;
                        if data[idx] < 0.0 {
                            data[idx] *= alpha;
                        }
                    }
                }
            }
        } else {
            // Fallback: broadcast element-wise
            for (i, v) in data.iter_mut().enumerate() {
                if *v < 0.0 {
                    *v *= slope.data[i % slope_numel];
                }
            }
        }
        Tensor::new(data, x.shape.clone())
    } else {
        // 1D case: broadcast
        Tensor::new(
            x.data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if v >= 0.0 {
                        v
                    } else {
                        slope.data[i % slope_numel] * v
                    }
                })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Softplus: ln(1 + exp(x)), with numerical stability for large x.
pub fn softplus(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                if v > 20.0 {
                    v
                } else if v < -20.0 {
                    0.0
                } else {
                    (1.0 + v.exp()).ln()
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

/// Softsign: x / (1 + |x|)
pub fn softsign(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| v / (1.0 + v.abs())).collect(),
        x.shape.clone(),
    )
}

/// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
pub fn mish(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                let sp = if v > 20.0 {
                    v
                } else if v < -20.0 {
                    0.0
                } else {
                    (1.0 + v.exp()).ln()
                };
                v * sp.tanh()
            })
            .collect(),
        x.shape.clone(),
    )
}

/// CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))
pub fn celu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                if v >= 0.0 {
                    v
                } else {
                    alpha * ((v / alpha).exp() - 1.0)
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

/// ELU: x if x >= 0, alpha*(exp(x)-1) if x < 0
pub fn elu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })
            .collect(),
        x.shape.clone(),
    )
}

/// SELU: gamma * (x if x > 0, alpha*exp(x) - alpha if x <= 0)
/// Default alpha=1.6732632423543772, gamma=1.0507009873554805
pub fn selu(x: &Tensor, alpha: f32, gamma: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| gamma * if v > 0.0 { v } else { alpha * v.exp() - alpha })
            .collect(),
        x.shape.clone(),
    )
}

/// ThresholdedRelu: x if x > alpha, 0 otherwise
pub fn thresholded_relu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| if v > alpha { v } else { 0.0 })
            .collect(),
        x.shape.clone(),
    )
}

/// Dropout: identity in inference mode (passthrough).
pub fn dropout(x: &Tensor) -> Tensor {
    x.clone()
}

/// SiLU / Swish: y = x * sigmoid(x)
pub fn silu(x: &Tensor) -> Tensor {
    silu_impl(x)
}

#[cfg(feature = "simd")]
fn silu_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_silu(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn silu_impl(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect(),
        x.shape.clone(),
    )
}

/// HardSigmoid: y = clamp(alpha * x + beta, 0, 1)
pub fn hard_sigmoid(x: &Tensor, alpha: f32, beta: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| (alpha * v + beta).clamp(0.0, 1.0))
            .collect(),
        x.shape.clone(),
    )
}

/// HardSwish: y = x * HardSigmoid(x, 1/6, 1/2) = x * clamp(x/6 + 0.5, 0, 1)
pub fn hard_swish(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| v * (v / 6.0 + 0.5).clamp(0.0, 1.0))
            .collect(),
        x.shape.clone(),
    )
}

/// Shrink activation: y = x + bias if x < -lambd; x - bias if x > lambd; else 0.
///
/// ONNX spec defaults: bias=0.0, lambd=0.5.
pub fn shrink(x: &Tensor, bias: f32, lambd: f32) -> Tensor {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&v| {
            if v < -lambd {
                v + bias
            } else if v > lambd {
                v - bias
            } else {
                0.0
            }
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}
