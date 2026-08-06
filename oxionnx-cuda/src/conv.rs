//! CUDA 2-D convolution dispatch — **not implemented**.
//!
//! # Status
//!
//! ONNX `Conv` has **no working CUDA path** in this crate.  [`cuda_conv`]
//! unconditionally returns `Ok(None)`, which instructs
//! [`crate::try_cuda_dispatch`] to decline the node so the caller falls back
//! to the CPU (or wgpu) implementation.
//!
//! ## Why
//!
//! The natural implementation routes through `oxicuda_dnn::conv::api::conv_forward`
//! with `TensorDesc`/`ConvolutionDescriptor` NCHW descriptors.  That code was
//! written, but the `oxicuda-dnn` convolution engines it depends on have stubbed
//! GEMM phases, so the kernel produced numerically wrong results.  Rather than
//! ship a silently-wrong GPU convolution, the body was removed: a `Conv` node is
//! declined here and executed correctly on the CPU.
//!
//! Consequently [`crate::is_supported_op`] reports `false` for [`OpKind::Conv`],
//! and placement logic in `oxionnx` will never route a `Conv` to CUDA.
//!
//! ## Re-enabling
//!
//! When `oxicuda-dnn` gains a correct f32 NCHW forward convolution:
//!
//! 1. Implement [`cuda_conv`] against `conv_forward`, honouring [`ConvParams`].
//! 2. Add `OpKind::Conv` to the `matches!` list in [`crate::is_supported_op`].
//! 3. The agreement test `is_supported_op_matches_dispatch_arms` in
//!    `crate::tests` will then require `Conv` to be claimable.
//!
//! [`OpKind::Conv`]: oxionnx_core::graph::OpKind::Conv

use oxionnx_core::Tensor;

use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Grouped convolution parameters extracted from ONNX node attributes.
///
/// Constructed by [`crate::try_cuda_dispatch`] from the node's `strides`,
/// `pads`, `dilations`, and `group` attributes and handed to [`cuda_conv`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvParams {
    /// Stride for `[H, W]`.
    pub strides: [usize; 2],
    /// Padding for `[pad_top, pad_left, pad_bottom, pad_right]`.
    pub pads: [usize; 4],
    /// Dilation for `[H, W]`.
    pub dilations: [usize; 2],
    /// Convolution groups.
    pub group: usize,
}

/// ONNX `Conv` forward on the GPU — **always declines**.
///
/// * `ctx`    — live CUDA context (device + DNN handle).
/// * `input`  — ONNX input tensor, shape `[N, C_in, H, W]`.
/// * `weight` — ONNX filter tensor, shape `[C_out, C_in/group, kH, kW]`.
/// * `bias`   — optional bias tensor, shape `[C_out]`.
/// * `params` — strides, pads, dilations and group from the ONNX node attrs.
///
/// # Returns
///
/// Always `Ok(None)`, meaning "CUDA declines this node; run it on the CPU".
/// There is no CUDA convolution kernel — see the [module docs](self) for why.
/// This function never returns `Ok(Some(_))` and never returns `Err`.
pub fn cuda_conv(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvParams,
) -> Result<Option<Tensor>, CudaDispatchError> {
    // The arguments are accepted (and validated by the caller) so that the
    // signature is stable for the eventual real implementation, but no CUDA
    // convolution kernel exists: decline unconditionally.
    let _ = (ctx, input, weight, bias, params);
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_params_round_trip() {
        let params = ConvParams {
            strides: [2, 2],
            pads: [1, 1, 1, 1],
            dilations: [1, 1],
            group: 1,
        };
        assert_eq!(params.strides, [2, 2]);
        assert_eq!(params.pads, [1, 1, 1, 1]);
        assert_eq!(params.dilations, [1, 1]);
        assert_eq!(params.group, 1);
    }

    /// `Conv` must not be advertised as a CUDA-supported op, because
    /// [`cuda_conv`] unconditionally declines.  Placement logic depends on this.
    #[test]
    fn conv_is_not_advertised_as_supported() {
        use oxionnx_core::graph::OpKind;
        assert!(
            !crate::is_supported_op(&OpKind::Conv),
            "cuda_conv() always returns Ok(None); is_supported_op must report Conv as unsupported \
             so decide_placement never routes a Conv node to CUDA",
        );
    }
}
