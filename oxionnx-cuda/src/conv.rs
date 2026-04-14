//! CUDA-accelerated 2-D convolution dispatch.
//!
//! Implements ONNX `Conv` for 2-D spatial inputs via `oxicuda_dnn`.
//! Only f32 precision and `group == 1` are supported; other configurations
//! return `Ok(None)` to fall back to CPU.

use oxicuda_dnn::conv::api::conv_forward;
use oxicuda_dnn::types::{ConvolutionDescriptor, TensorDesc, TensorDescMut};
use oxicuda_memory::DeviceBuffer;
use oxionnx_core::Tensor;

use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Grouped convolution parameters extracted from ONNX node attributes.
pub struct ConvParams {
    /// Stride for [H, W].
    pub strides: [usize; 2],
    /// Padding for [pad_top, pad_left, pad_bottom, pad_right].
    pub pads: [usize; 4],
    /// Dilation for [H, W].
    pub dilations: [usize; 2],
    /// Convolution groups.
    pub group: usize,
}

/// ONNX Conv forward on GPU.
///
/// * `input`  — ONNX input tensor, shape `[N, C_in, H, W]`.
/// * `weight` — ONNX filter tensor, shape `[C_out, C_in/group, kH, kW]`.
/// * `bias`   — Optional bias tensor, shape `[C_out]`.
/// * `params` — Convolution strides, pads, dilations, group from ONNX attrs.
///
/// Returns the output tensor `[N, C_out, P, Q]` on success, or `Err` on CUDA
/// failure (mapped to `OnnxError::Internal` by the caller).
///
/// Returns `Ok(None)` for unsupported configurations (non-2-D, non-f32, etc.).
pub fn cuda_conv(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvParams,
) -> Result<Option<Tensor>, CudaDispatchError> {
    // DNN conv engines are not yet fully implemented (GEMM phases are stubbed),
    // so always fall back to CPU for correctness.
    if true {
        return Ok(None);
    }

    // Only support 4-D NCHW inputs.
    if input.shape.len() != 4 || weight.shape.len() != 4 {
        return Ok(None);
    }

    let strides = params.strides;
    let pads = params.pads;
    let dilations = params.dilations;
    let group = params.group;

    let n = input.shape[0];
    let c_in = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];

    let c_out = weight.shape[0];
    let kh = weight.shape[2];
    let kw = weight.shape[3];

    // Compute output spatial dims.
    let p = (h_in + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
    let q = (w_in + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;

    // Upload tensors to device.
    let mut d_input: DeviceBuffer<f32> = DeviceBuffer::alloc(input.data.len())?;
    d_input.copy_from_host(&input.data)?;

    let mut d_weight: DeviceBuffer<f32> = DeviceBuffer::alloc(weight.data.len())?;
    d_weight.copy_from_host(&weight.data)?;

    let mut d_output: DeviceBuffer<f32> = DeviceBuffer::zeroed(n * c_out * p * q)?;

    // Build descriptors.
    let desc_input =
        TensorDesc::<f32>::nchw(&d_input, n as u32, c_in as u32, h_in as u32, w_in as u32)
            .map_err(|e| CudaDispatchError::Dnn(e.to_string()))?;
    let desc_filter = TensorDesc::<f32>::nchw(
        &d_weight,
        c_out as u32,
        (c_in / group) as u32,
        kh as u32,
        kw as u32,
    )
    .map_err(|e| CudaDispatchError::Dnn(e.to_string()))?;
    let mut desc_output =
        TensorDescMut::<f32>::nchw(&mut d_output, n as u32, c_out as u32, p as u32, q as u32)
            .map_err(|e| CudaDispatchError::Dnn(e.to_string()))?;

    let conv_desc = ConvolutionDescriptor::conv2d(
        pads[0] as u32,
        pads[1] as u32,
        strides[0] as u32,
        strides[1] as u32,
        dilations[0] as u32,
        dilations[1] as u32,
        group as u32,
    )
    .map_err(|e| CudaDispatchError::Dnn(e.to_string()))?;

    conv_forward(
        &ctx.dnn,
        &desc_input,
        &desc_filter,
        &mut desc_output,
        &conv_desc,
        None,
    )
    .map_err(|e| CudaDispatchError::Dnn(e.to_string()))?;

    // Synchronize the stream to ensure the convolution completes before
    // reading results back to the host.
    ctx.dnn
        .stream()
        .synchronize()
        .map_err(CudaDispatchError::Driver)?;

    let mut out_data = vec![0.0_f32; n * c_out * p * q];
    d_output.copy_to_host(&mut out_data)?;

    // Add bias if present.
    if let Some(bias_t) = bias {
        if bias_t.data.len() == c_out {
            for n_idx in 0..n {
                for c_idx in 0..c_out {
                    let base = (n_idx * c_out + c_idx) * p * q;
                    let b = bias_t.data[c_idx];
                    for elem in &mut out_data[base..base + p * q] {
                        *elem += b;
                    }
                }
            }
        }
    }

    Ok(Some(Tensor::new(out_data, vec![n, c_out, p, q])))
}
