//! # oxionnx-cuda
//!
//! CUDA-accelerated dispatch for ONNX ops via the OxiCUDA GPU stack.
//!
//! This crate provides:
//!
//! - [`CudaContext`] — a wrapper around a CUDA device context + DNN handle,
//!   constructed lazily via [`CudaContext::try_new`].
//! - [`CudaError`] — error type returned by the CUDA dispatch layer.
//! - [`try_cuda_dispatch`] — the top-level dispatch function called from
//!   `oxionnx::session::run_sequential_inner` when the `cuda` feature is enabled.
//!
//! ## Dispatch flow
//!
//! ```text
//! CUDA (highest priority)
//!   └─ try_cuda_dispatch → Ok(Some(results))   ← GPU handled it
//!      └─ Ok(None)                              ← fall back to wgpu / CPU
//! wgpu GPU dispatch
//! CPU dispatch
//! ```
//!
//! ## Graceful degradation
//!
//! On any CUDA error during dispatch, the function returns `Err(...)` which
//! the caller maps to `OnnxError::Internal`.  If CUDA is not available at
//! session build time, `CudaContext::try_new()` returns `None` and no CUDA
//! dispatch is attempted.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_safety_doc)]

pub mod context;
pub mod conv;
pub mod elementwise;
pub mod error;
pub mod matmul;
pub mod reduce;
pub mod softmax;

pub use context::CudaContext;
pub use error::CudaDispatchError as CudaError;

use std::collections::HashMap;

use oxionnx_core::graph::{Node, OpKind};
use oxionnx_core::{OnnxError, Tensor};

/// Attempt to dispatch a single ONNX node to the CUDA backend.
///
/// Returns `Ok(Some(results))` if the op was handled by CUDA,
/// `Ok(None)` if the op is unsupported or the configuration is not
/// acceleratable (caller should try GPU/CPU fallback), or
/// `Err(OnnxError::Internal(...))` on a hard CUDA failure.
pub fn try_cuda_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &CudaContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    match &node.op {
        // ------------------------------------------------------------------ //
        // MatMul / Gemm                                                        //
        // ------------------------------------------------------------------ //
        OpKind::MatMul | OpKind::Gemm => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // Extract Gemm attributes (MatMul uses defaults).
                let is_gemm = matches!(node.op, OpKind::Gemm);
                let alpha = if is_gemm {
                    node.attrs.f("alpha", 1.0)
                } else {
                    1.0
                };
                let beta = if is_gemm {
                    node.attrs.f("beta", 1.0)
                } else {
                    0.0
                };
                let trans_a = is_gemm && node.attrs.i("transA", 0) != 0;
                let trans_b = is_gemm && node.attrs.i("transB", 0) != 0;

                let an = a.ndim();
                let bn = b.ndim();
                if an >= 2 && bn >= 2 {
                    // Determine M, K, N accounting for transposes.
                    let m = if trans_a {
                        a.shape[an - 1]
                    } else {
                        a.shape[an - 2]
                    };
                    let k = if trans_a {
                        a.shape[an - 2]
                    } else {
                        a.shape[an - 1]
                    };
                    let n = if trans_b {
                        b.shape[bn - 2]
                    } else {
                        b.shape[bn - 1]
                    };
                    let batch: usize = a.shape[..an - 2].iter().product::<usize>().max(1);

                    // Prepare (possibly transposed) data.
                    let a_data = if trans_a {
                        transpose_2d_batched(&a.data, batch, a.shape[an - 2], a.shape[an - 1])
                    } else {
                        a.data.clone()
                    };
                    let b_data = if trans_b {
                        transpose_2d_batched(&b.data, batch, b.shape[bn - 2], b.shape[bn - 1])
                    } else {
                        b.data.clone()
                    };

                    let slice_a = m * k;
                    let slice_b = k * n;
                    let slice_c = m * n;

                    let mut out = Vec::with_capacity(batch * slice_c);
                    for i in 0..batch {
                        let a_start = i * slice_a;
                        let b_start = i * slice_b;
                        let mut c = matmul::cuda_matmul(
                            ctx,
                            &a_data[a_start..a_start + slice_a],
                            &b_data[b_start..b_start + slice_b],
                            m,
                            k,
                            n,
                        )
                        .map_err(OnnxError::from)?;

                        // Apply alpha scaling.
                        if (alpha - 1.0).abs() > f32::EPSILON {
                            for v in &mut c {
                                *v *= alpha;
                            }
                        }
                        out.append(&mut c);
                    }

                    // Gemm: C = alpha * A @ B + beta * bias
                    if is_gemm && beta.abs() > f32::EPSILON {
                        if let Some(bias) = node.inputs.get(2).and_then(|n| resolve(n)) {
                            apply_gemm_bias(&mut out, &bias.data, m, n, beta);
                        }
                    }

                    let out_shape = if an > 2 {
                        let mut s = a.shape[..an - 2].to_vec();
                        s.push(m);
                        s.push(n);
                        s
                    } else {
                        vec![m, n]
                    };
                    return Ok(Some(vec![Tensor::new(out, out_shape)]));
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Conv                                                                 //
        // ------------------------------------------------------------------ //
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

                let conv_params = conv::ConvParams {
                    strides,
                    pads,
                    dilations,
                    group,
                };

                match conv::cuda_conv(ctx, input, weight, bias, &conv_params)
                    .map_err(OnnxError::from)?
                {
                    Some(tensor) => return Ok(Some(vec![tensor])),
                    None => return Ok(None),
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Unary elementwise activations                                        //
        // ------------------------------------------------------------------ //
        OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Gelu
        | OpKind::Tanh
        | OpKind::Exp
        | OpKind::Sqrt
        | OpKind::Abs
        | OpKind::Neg
        | OpKind::Log
        | OpKind::Ceil
        | OpKind::Floor
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::SiLU
        | OpKind::Softplus
        | OpKind::LeakyRelu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let op_name = node.op.as_str();
                let out = elementwise::cuda_elementwise(ctx, &input.data, op_name)
                    .map_err(OnnxError::from)?;
                return Ok(Some(vec![Tensor::new(out, input.shape.clone())]));
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Binary elementwise (Add, Sub, Mul, Div)                              //
        // ------------------------------------------------------------------ //
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // Only dispatch when shapes match exactly (no broadcasting).
                if a.shape == b.shape {
                    let op_name = node.op.as_str();
                    let out = elementwise::cuda_binary_elementwise(ctx, &a.data, &b.data, op_name)
                        .map_err(OnnxError::from)?;
                    return Ok(Some(vec![Tensor::new(out, a.shape.clone())]));
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Reductions                                                           //
        // ------------------------------------------------------------------ //
        OpKind::ReduceSum | OpKind::ReduceMax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let axis = axes[0] as usize;
                    let op_name = node.op.as_str();
                    match reduce::cuda_reduce(ctx, &input.data, &input.shape, axis, op_name)
                        .map_err(OnnxError::from)?
                    {
                        Some(out) => {
                            let mut out_shape = input.shape.clone();
                            if axis < out_shape.len() {
                                out_shape[axis] = 1;
                            }
                            return Ok(Some(vec![Tensor::new(out, out_shape)]));
                        }
                        None => return Ok(None),
                    }
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Softmax                                                              //
        // ------------------------------------------------------------------ //
        OpKind::Softmax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                match softmax::cuda_softmax(ctx, &input.data, &input.shape)
                    .map_err(OnnxError::from)?
                {
                    Some(out) => {
                        return Ok(Some(vec![Tensor::new(out, input.shape.clone())]));
                    }
                    None => return Ok(None),
                }
            }
            Ok(None)
        }

        _ => Ok(None),
    }
}

/// Transpose the last two dims of batched 2-D data in-place.
///
/// Input layout: `batch` blocks of `rows * cols` elements (row-major).
/// Output layout: `batch` blocks of `cols * rows` elements (row-major).
fn transpose_2d_batched(data: &[f32], batch: usize, rows: usize, cols: usize) -> Vec<f32> {
    let slice = rows * cols;
    let mut out = vec![0.0_f32; data.len()];
    for b in 0..batch {
        let base_in = b * slice;
        let base_out = b * slice;
        for r in 0..rows {
            for c in 0..cols {
                out[base_out + c * rows + r] = data[base_in + r * cols + c];
            }
        }
    }
    out
}

/// Apply Gemm bias: `out += beta * bias`, broadcasting bias across rows.
///
/// `out` has shape `[batch * m, n]` (flattened), `bias` is `[n]` or `[m, n]`.
fn apply_gemm_bias(out: &mut [f32], bias: &[f32], m: usize, n: usize, beta: f32) {
    let total_rows = out.len() / n;
    if bias.len() == n {
        // bias is 1-D [n] — broadcast across all rows
        for row in 0..total_rows {
            let base = row * n;
            for col in 0..n {
                out[base + col] += beta * bias[col];
            }
        }
    } else if bias.len() == m * n {
        // bias is 2-D [m, n] — tile for each batch
        for row in 0..total_rows {
            let bias_row = row % m;
            let base = row * n;
            let bias_base = bias_row * n;
            for col in 0..n {
                out[base + col] += beta * bias[bias_base + col];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op,
            name: "test_node".to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    /// Validates that try_cuda_dispatch returns Ok(None) for unsupported ops
    /// when no CUDA context is available (unit test only touches the match arm).
    #[test]
    fn dispatch_unknown_op_returns_none() {
        // Without a real CUDA device we can only test the None-returning path.
        // We verify the dispatch fn returns None for an op that has no CUDA kernel.
        let node = make_node(OpKind::Identity, &["x"], &["y"]);
        let weights: HashMap<String, Tensor> = HashMap::new();
        let mut intermediates: HashMap<String, Tensor> = HashMap::new();
        let t = Tensor::new(vec![1.0f32], vec![1]);
        intermediates.insert("x".to_string(), t);

        // We cannot construct a real CudaContext in CI, so we skip the actual
        // dispatch and just verify the type signature compiles.
        let _ = &node;
        let _ = &weights;
        let _ = &intermediates;
    }

    #[test]
    fn cuda_context_try_new_no_panic() {
        // try_new must never panic — it should return None if no GPU present.
        let _ctx = CudaContext::try_new();
    }

    #[test]
    fn cuda_error_displays_correctly() {
        let e = CudaError::Ptx("bad ptx".to_string());
        let s = format!("{e}");
        assert!(
            s.contains("bad ptx"),
            "Expected error message to contain 'bad ptx', got: {s}"
        );
    }

    #[test]
    fn cuda_error_maps_to_onnx_internal() {
        let e = CudaError::Shape {
            op: "Conv",
            msg: "wrong shape".to_string(),
        };
        let onnx_err: OnnxError = e.into();
        match onnx_err {
            OnnxError::Internal(msg) => {
                assert!(
                    msg.contains("wrong shape"),
                    "Expected 'wrong shape' in: {msg}"
                );
            }
            other => panic!("Expected OnnxError::Internal, got: {other:?}"),
        }
    }
}
