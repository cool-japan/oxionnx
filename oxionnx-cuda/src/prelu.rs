//! CUDA dispatch for `PRelu`: `f(x) = x if x >= 0, slope[c] * x if x < 0`,
//! `slope` broadcasting per-channel (`[C]`) or as a scalar (`[1]`).
//!
//! Launches [`oxicuda_ptx::templates::channel_broadcast::PReluTemplate`] —
//! the per-channel-slope generalisation of `elementwise.rs`'s `LeakyRelu`
//! (whose `alpha` is a single baked-in constant) sharing its channel-index
//! addressing with [`crate::broadcast`]'s binary kernels.
//!
//! # Weight residency
//!
//! Unlike `elementwise.rs`'s binary operands and `crate::broadcast`'s "small"
//! operand, `slope` genuinely is (in every ONNX `PRelu` this pipeline emits)
//! a learned per-channel parameter — a graph initializer, invariant for the
//! session — so this module weight-caches it exactly as [`crate::conv`]
//! caches a convolution's filter/bias: the caller resolves its
//! [`crate::residency::WeightId`] once via `lib.rs`'s `initializer_id`, and
//! [`cuda_prelu_bound`] threads it through to [`CudaContext::operand`].

use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_ptx::{ir::PtxType, templates::channel_broadcast::PReluTemplate};

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;
use crate::residency::WeightId;

/// Residency-cache slot label for `PRelu`'s activation input.
///
/// The activation is never weight-cached (its bytes change every frame), but
/// the binding API takes a label uniformly — see `elementwise.rs`'s
/// `INPUT_LABEL` for the identical rationale.
const INPUT_LABEL: &str = "prelu_input";
/// Residency-cache slot label for `PRelu`'s per-channel slope.
const SLOPE_LABEL: &str = "prelu_slope";

const BLOCK_SIZE: u32 = 256;

/// Shape check and `(channels, spatial, total_len)` derivation for a `PRelu`
/// node.
///
/// Mirrors `oxionnx-gpu`'s `gpu_prelu_placed_async` decline rule exactly:
/// `x_shape` must have rank `>= 2`, and `slope_len` must equal
/// `x_shape[1]` (per-channel) or be `1` (scalar broadcast). Anything else —
/// including a zero channel count, which would make the kernel's `%
/// channels` divide by zero — declines (`None`) to the CPU kernel's more
/// permissive fallback (see `oxionnx-ops::nn::activations::prelu`'s "x_c !=
/// c" branch).
///
/// Pure and allocation-free, so this is unit-testable without a CUDA device.
#[must_use]
pub(crate) fn prelu_plan(x_shape: &[usize], slope_len: usize) -> Option<(usize, usize, usize)> {
    if x_shape.len() < 2 {
        return None;
    }
    let channels = x_shape[1];
    if channels == 0 || (slope_len != channels && slope_len != 1) {
        return None;
    }
    let spatial: usize = x_shape[2..].iter().product::<usize>().max(1);
    let total_len: usize = x_shape.iter().product();
    Some((channels, spatial, total_len))
}

fn kernel_for(ctx: &CudaContext) -> Result<Kernel, CudaDispatchError> {
    let template = PReluTemplate {
        precision: PtxType::F32,
        target: ctx.dnn.sm_version(),
    };
    let kernel_name = template.kernel_name();
    let module = ctx.module(&kernel_name, || {
        template
            .generate()
            .map_err(|e| CudaDispatchError::Ptx(e.to_string()))
    })?;
    Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)
}

/// Launch a `PRelu` kernel over an activation that may already be on the
/// device, leaving the result there when the caller asks for it.
///
/// `slope` is always read on the host: even when `slope_id` names a resident
/// weight, this crate needs the host bytes' length to run [`prelu_plan`]
/// (and the shadow-verification oracle needs them to build its own answer).
/// Only the *upload* is skipped on a cache hit — see [`CudaContext::operand`].
///
/// Returns `Ok(None)` when [`prelu_plan`] declines the shape.
///
/// # Errors
///
/// [`CudaDispatchError::Shape`] when the input cannot supply the elements
/// its shape declares or the launch width exceeds a `u32`, or a driver error
/// from PTX compilation, allocation, upload, launch or readback.
pub(crate) fn cuda_prelu_bound(
    ctx: &CudaContext,
    x: InputBinding<'_>,
    x_shape: &[usize],
    slope: &[f32],
    slope_id: Option<WeightId<'_>>,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    let Some((channels, spatial, total_len)) = prelu_plan(x_shape, slope.len()) else {
        return Ok(None);
    };

    let (Ok(total_u32), Ok(channels_u32), Ok(spatial_u32), Ok(slope_len_u32)) = (
        u32::try_from(total_len),
        u32::try_from(channels),
        u32::try_from(spatial),
        u32::try_from(slope.len()),
    ) else {
        return Ok(None);
    };

    let kernel = kernel_for(ctx)?;
    let stream = ctx.dnn.stream();

    let Some(mut d_input) = x.bind(ctx, INPUT_LABEL, total_len, stream)? else {
        return Err(CudaDispatchError::Shape {
            op: "prelu",
            msg: format!("input cannot supply the {total_len} elements its shape declares"),
        });
    };
    let mut d_slope = ctx.operand(slope_id, SLOPE_LABEL, slope, stream)?;
    let d_output = ctx.scratch(total_len)?;

    let grid = grid_size_for(total_u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));
    let args = (
        d_input.device_ptr(),
        d_slope.device_ptr(),
        d_output.device_ptr(),
        total_u32,
        channels_u32,
        spatial_u32,
        slope_len_u32,
    );
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    let out = finish_output(ctx, d_output, total_len, x_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => {
            d_input.retire();
            d_slope.retire();
        }
        KernelOutput::Device(_) => {
            retire_queued(ctx, &mut d_input);
            retire_queued(ctx, &mut d_slope);
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_accepts_per_channel_slope() {
        let (channels, spatial, total) = prelu_plan(&[1, 3, 4, 4], 3).expect("must plan");
        assert_eq!(channels, 3);
        assert_eq!(spatial, 16);
        assert_eq!(total, 48);
    }

    #[test]
    fn plan_accepts_scalar_slope() {
        let (channels, spatial, total) = prelu_plan(&[1, 3, 4, 4], 1).expect("must plan");
        assert_eq!(channels, 3);
        assert_eq!(spatial, 16);
        assert_eq!(total, 48);
    }

    #[test]
    fn plan_declines_mismatched_slope_length() {
        assert!(prelu_plan(&[1, 3, 4, 4], 4).is_none());
    }

    #[test]
    fn plan_declines_rank_below_two() {
        assert!(prelu_plan(&[8], 1).is_none());
    }

    #[test]
    fn plan_declines_zero_channels() {
        assert!(prelu_plan(&[1, 0, 4, 4], 1).is_none());
    }

    #[test]
    fn plan_handles_rank_two_with_no_spatial_dims() {
        let (channels, spatial, total) = prelu_plan(&[2, 5], 5).expect("must plan");
        assert_eq!(channels, 5);
        assert_eq!(spatial, 1);
        assert_eq!(total, 10);
    }

    #[test]
    fn cuda_context_construction_never_panics_even_though_unavailable_here() {
        let _ = CudaContext::try_new();
    }
}
