use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;

/// Provides metadata about GPU-accelerated operator support.
#[cfg(feature = "gpu")]
pub struct GpuExecutionProvider;

#[cfg(feature = "gpu")]
impl GpuExecutionProvider {
    /// Return the list of operator types that have GPU acceleration.
    pub fn supported_ops() -> &'static [&'static str] {
        &[
            "MatMul",
            "Conv",
            "Softmax",
            "Relu",
            "Sigmoid",
            "Gelu",
            "ReduceSum",
            "ReduceMax",
            "ReduceMean",
            "LayerNormalization",
            "BatchNormalization",
            "Transpose",
        ]
    }

    /// Check whether a given operator type is GPU-accelerated.
    pub fn is_supported(op_type: &str) -> bool {
        Self::supported_ops().contains(&op_type)
    }
}

/// GPU dispatch for ops with hardware acceleration (MatMul, Conv).
/// Returns `Ok(Some(results))` if GPU handled it, `Ok(None)` to fall back to CPU.
#[cfg(feature = "gpu")]
pub(crate) fn try_gpu_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    gpu: &crate::gpu::GpuContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    match &node.op {
        OpKind::MatMul => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                let an = a.ndim();
                let bn = b.ndim();
                if an >= 2 && bn >= 2 {
                    let m = a.shape[an - 2];
                    let k = a.shape[an - 1];
                    let n = b.shape[bn - 1];
                    let batch_size: usize = a.shape[..an - 2].iter().product::<usize>().max(1);
                    if batch_size == 1 {
                        if let Some(result) = crate::gpu::gpu_matmul(gpu, &a.data, &b.data, m, k, n)
                        {
                            return Ok(Some(vec![Tensor::new(result, vec![m, n])]));
                        }
                    }
                }
            }
            Ok(None)
        }
        OpKind::Conv => {
            let input = resolve(&node.inputs[0]);
            let weight = resolve(&node.inputs[1]);
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(input), Some(weight)) = (input, weight) {
                let attrs = &node.attrs;
                let strides_v = attrs.ints("strides");
                let strides = [
                    strides_v.first().copied().unwrap_or(1) as usize,
                    strides_v.get(1).copied().unwrap_or(1) as usize,
                ];
                let pads_v = attrs.ints("pads");
                let pads = [
                    pads_v.first().copied().unwrap_or(0) as usize,
                    pads_v.get(1).copied().unwrap_or(0) as usize,
                    pads_v.get(2).copied().unwrap_or(0) as usize,
                    pads_v.get(3).copied().unwrap_or(0) as usize,
                ];
                let dilations_v = attrs.ints("dilations");
                let dilations = [
                    dilations_v.first().copied().unwrap_or(1) as usize,
                    dilations_v.get(1).copied().unwrap_or(1) as usize,
                ];
                let group = attrs.i("group", 1) as usize;
                if let Some(result) = crate::gpu::gpu_conv2d(
                    gpu, input, weight, bias, strides, pads, dilations, group,
                ) {
                    return Ok(Some(vec![result]));
                }
            }
            Ok(None)
        }
        OpKind::Softmax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_softmax(gpu, &input.data, &input.shape) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Relu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_relu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Sigmoid => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_sigmoid(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Gelu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_gelu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::ReduceSum => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let axis = axes[0] as usize;
                    if let Some(result) =
                        crate::gpu::gpu_reduce_sum(gpu, &input.data, axis, &input.shape)
                    {
                        let mut out_shape = input.shape.clone();
                        if axis < out_shape.len() {
                            out_shape[axis] = 1;
                        }
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::ReduceMax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let axis = axes[0] as usize;
                    if let Some(result) =
                        crate::gpu::gpu_reduce_max(gpu, &input.data, axis, &input.shape)
                    {
                        let mut out_shape = input.shape.clone();
                        if axis < out_shape.len() {
                            out_shape[axis] = 1;
                        }
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::ReduceMin => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let axis = axes[0] as usize;
                    if let Some(result) =
                        crate::gpu::gpu_reduce_min(gpu, &input.data, axis, &input.shape)
                    {
                        let mut out_shape = input.shape.clone();
                        if axis < out_shape.len() {
                            out_shape[axis] = 1;
                        }
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Tanh => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_tanh(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Exp => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_exp(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Sqrt => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_sqrt(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Abs => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_abs(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Neg => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_neg(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Log => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_log(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::SiLU => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_silu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::LeakyRelu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_leaky_relu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Add => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                if a.data.len() == b.data.len() {
                    if let Some(result) = crate::gpu::gpu_add(gpu, &a.data, &b.data) {
                        return Ok(Some(vec![Tensor::new(result, a.shape.clone())]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::Mul => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                if a.data.len() == b.data.len() {
                    if let Some(result) = crate::gpu::gpu_mul(gpu, &a.data, &b.data) {
                        return Ok(Some(vec![Tensor::new(result, a.shape.clone())]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::LayerNorm => {
            let input = resolve(&node.inputs[0]);
            let scale = node.inputs.get(1).and_then(|n| resolve(n));
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(input), Some(scale), Some(bias)) = (input, scale, bias) {
                let eps = node.attrs.f("epsilon", 1e-5);
                if let Some(result) = crate::gpu::gpu_layer_norm(
                    gpu,
                    &input.data,
                    &input.shape,
                    &scale.data,
                    &bias.data,
                    eps,
                ) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::BatchNorm => {
            let input = resolve(&node.inputs[0]);
            let scale = node.inputs.get(1).and_then(|n| resolve(n));
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            let mean = node.inputs.get(3).and_then(|n| resolve(n));
            let var = node.inputs.get(4).and_then(|n| resolve(n));
            if let (Some(input), Some(scale), Some(bias), Some(mean), Some(var)) =
                (input, scale, bias, mean, var)
            {
                let eps = node.attrs.f("epsilon", 1e-5);
                if let Some(result) = crate::gpu::gpu_batch_norm(
                    gpu,
                    &input.data,
                    &input.shape,
                    &scale.data,
                    &bias.data,
                    &mean.data,
                    &var.data,
                    eps,
                ) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Transpose => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let perm_attr = node.attrs.ints("perm");
                let perm: Vec<usize> = if perm_attr.is_empty() {
                    // Default: reverse dimensions
                    (0..input.shape.len()).rev().collect()
                } else {
                    perm_attr.iter().map(|&p| p as usize).collect()
                };
                if let Some(result) =
                    crate::gpu::gpu_transpose(gpu, &input.data, &input.shape, &perm)
                {
                    let out_shape: Vec<usize> = perm.iter().map(|&p| input.shape[p]).collect();
                    return Ok(Some(vec![Tensor::new(result, out_shape)]));
                }
            }
            Ok(None)
        }
        OpKind::ReduceMean => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes_attr = node.attrs.ints("axes");
                let axes: Vec<usize> = axes_attr.iter().map(|&a| a as usize).collect();
                let keepdims = node.attrs.i("keepdims", 1) != 0;
                if !axes.is_empty() {
                    if let Some(result) =
                        crate::gpu::gpu_reduce_mean(gpu, &input.data, &input.shape, &axes, keepdims)
                    {
                        let mut out_shape = input.shape.clone();
                        if keepdims {
                            for &a in &axes {
                                if a < out_shape.len() {
                                    out_shape[a] = 1;
                                }
                            }
                        } else {
                            let mut sorted_axes = axes.clone();
                            sorted_axes.sort_unstable();
                            for (offset, &a) in sorted_axes.iter().enumerate() {
                                if a >= offset && (a - offset) < out_shape.len() {
                                    out_shape.remove(a - offset);
                                }
                            }
                            if out_shape.is_empty() {
                                out_shape.push(1);
                            }
                        }
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}
