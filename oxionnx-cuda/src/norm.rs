//! CUDA dispatch for `BatchNormalization` (inference) and `OxiInstanceNorm`.
//!
//! Both reuse [`oxicuda_ptx::templates::batch_norm::BatchNormTemplate`]
//! unmodified — no new PTX in this wave — but drive its two modes very
//! differently:
//!
//! * **`BatchNormalization`** maps onto [`BnMode::Inference`] directly: the
//!   template already takes a runtime `batch_count`, so the whole `[N, C,
//!   H, W]` tensor is one kernel launch, one block per channel.
//! * **`OxiInstanceNorm`** has no analogue of `BatchNormalization`'s
//!   precomputed running mean/var — it computes its own mean/var per
//!   `(n, c)` plane, which is exactly what [`BnMode::Training`] does *for one
//!   sample* when `batch_count = 1`. [`BnMode::Training`]'s per-channel
//!   `sample_stride` arithmetic is baked into the PTX from `channels` at
//!   generation time, so a single launch cannot address `N > 1` distinct
//!   samples through it — [`cuda_oxi_instance_norm_bound`] instead launches
//!   the *same* compiled kernel once per sample, pointer-offset by that
//!   sample's byte stride, with `batch_count = 1` every time. `N` is `1` for
//!   every real graph in this pipeline (SCRFD/ArcFace/InSwapper all run one
//!   image per forward pass), so in practice this is exactly one launch; the
//!   loop exists so a batched caller still gets a correct answer rather than
//!   a declined node.
//!
//! `OxiInstanceNorm` has no affine term (see
//! `oxionnx-ops::registry::oxi_instance_norm`'s module docs for why: AdaIN's
//! scale/shift are runtime tensors, not initialisers, so the fused op
//! deliberately excludes them and leaves the trailing `Mul`/`Add` as ordinary
//! graph nodes). [`BnMode::Training`]'s kernel always applies `gamma *
//! normalized + beta`, so this dispatch supplies `gamma = 1`, `beta = 0` —
//! the identity affine — computed fresh (a few hundred bytes) rather than
//! weight-cached: unlike `BatchNormalization`'s scale/bias/mean/var, these
//! are not values the *graph* holds anywhere, so there is no stable host
//! address to key a cache entry on, and the payload is negligible next to
//! the activation traffic this op exists to keep on the device.
//!
//! # Why the module cache key is not `BatchNormTemplate::kernel_name()`
//!
//! `BatchNormTemplate::kernel_name()` encodes `mode`/`precision`/`channels`/
//! `spatial_size`/`block_size` but **not** `epsilon` — `epsilon` is baked
//! into the generated PTX as a literal (see the template's `eps_hex`), yet
//! two nodes with the same shape and a different `epsilon` would otherwise
//! collide on one cache entry and one of them would silently run with the
//! other's epsilon. `entry_name()` in this module is the real PTX symbol
//! (passed to `Kernel::from_module`, unchanged); `cache_key()` appends
//! epsilon's bit pattern and is used **only** as the `CudaContext::module`
//! lookup key. The two need not match — see [`CudaContext::module`]'s doc
//! comment, which keys purely on whatever string a caller passes.

use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_ptx::{
    ir::PtxType,
    templates::batch_norm::{BatchNormTemplate, BnMode},
};

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;
use crate::residency::WeightId;

const INPUT_LABEL: &str = "norm_input";
const SCALE_LABEL: &str = "batch_norm_scale";
const BIAS_LABEL: &str = "batch_norm_bias";
const MEAN_LABEL: &str = "batch_norm_mean";
const VAR_LABEL: &str = "batch_norm_var";
/// Never cached across calls (see the [module docs](self)): a fresh
/// per-dispatch upload, but still routed through the labelled operand API
/// like every other buffer here.
const IDENTITY_GAMMA_LABEL: &str = "oxi_instance_norm_gamma_one";
const IDENTITY_BETA_LABEL: &str = "oxi_instance_norm_beta_zero";

/// Threads per block for both kernels. `BatchNormTemplate::validate`
/// requires a power of two in `[32, 1024]`; 256 matches this crate's other
/// elementwise/reduction launches.
const BLOCK_SIZE: u32 = 256;

/// The module-cache key for one `(mode, channels, spatial, block_size,
/// epsilon)` instantiation. See the [module docs](self) for why this must
/// differ from the PTX entry-point name whenever two nodes share every field
/// but `epsilon`.
fn cache_key(entry_name: &str, epsilon: f32) -> String {
    format!("{entry_name}_eps{:08x}", epsilon.to_bits())
}

/// Compile (or fetch, on a cache hit) the kernel for one `BatchNormTemplate`
/// instantiation.
fn kernel_for(
    ctx: &CudaContext,
    template: &BatchNormTemplate,
) -> Result<Kernel, CudaDispatchError> {
    let entry_name = template.kernel_name();
    let key = cache_key(&entry_name, template.epsilon);
    let sm = ctx.dnn.sm_version();
    let module = ctx.module(&key, || {
        template
            .generate(sm)
            .map_err(|e| CudaDispatchError::Ptx(e.to_string()))
    })?;
    Kernel::from_module(module, &entry_name).map_err(CudaDispatchError::Driver)
}

/// `epsilon` must be finite and strictly positive: `BatchNormTemplate`'s own
/// `validate()` rejects anything else (dividing by `sqrt(var + eps)` with a
/// non-positive `eps` is exactly the numerical-stability guard `epsilon`
/// exists to provide), and checking it here turns a malformed model into a
/// clean decline instead of a hard [`CudaDispatchError::Ptx`] from PTX
/// generation.
fn epsilon_ok(eps: f32) -> bool {
    eps.is_finite() && eps > 0.0
}

// ─── BatchNormalization (inference) ────────────────────────────────────────

/// `(n, channels, spatial)` for a `BatchNormalization` node, or `None` to
/// decline — a zero `n`/`channels` would make the kernel's per-channel
/// indexing (or the grid launch itself) meaningless.
///
/// Pure and allocation-free, so unit-testable without a CUDA device.
#[must_use]
pub(crate) fn batch_norm_plan(x_shape: &[usize]) -> Option<(usize, usize, usize)> {
    if x_shape.len() < 2 {
        return None;
    }
    let n = x_shape[0];
    let channels = x_shape[1];
    if n == 0 || channels == 0 {
        return None;
    }
    let spatial: usize = x_shape[2..].iter().product::<usize>().max(1);
    Some((n, channels, spatial))
}

/// The identities of a `BatchNormalization` node's four invariant operands.
///
/// Mirrors `conv::ConvWeightIds`: `None` in any slot means "not a graph
/// initializer, upload for this dispatch only".
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BatchNormWeightIds<'a> {
    pub(crate) scale: Option<WeightId<'a>>,
    pub(crate) bias: Option<WeightId<'a>>,
    pub(crate) mean: Option<WeightId<'a>>,
    pub(crate) var: Option<WeightId<'a>>,
}

/// Launch a `BatchNormalization` (inference-mode) kernel over an activation
/// that may already be on the device, leaving the result there when the
/// caller asks for it.
///
/// `scale`/`bias`/`mean`/`var` must each hold at least `x_shape[1]`
/// elements; only the first `channels` of each are read (mirrors
/// `oxionnx-ops::nn::normalization::batch_norm`, which never reads past
/// `x.shape[1]` either — a longer operand is not an error, just as it is not
/// there).
///
/// Returns `Ok(None)` when [`batch_norm_plan`] declines the shape, an
/// operand is too short, or `epsilon` is non-positive/non-finite.
///
/// # Errors
///
/// [`CudaDispatchError::Shape`] when the input cannot supply the elements
/// its shape declares or a dimension exceeds a `u32` kernel launch, or a
/// driver error from PTX compilation, allocation, upload, launch or
/// readback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_batch_norm_bound(
    ctx: &CudaContext,
    x: InputBinding<'_>,
    x_shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    epsilon: f32,
    ids: BatchNormWeightIds<'_>,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    let Some((n, channels, spatial)) = batch_norm_plan(x_shape) else {
        return Ok(None);
    };
    if !epsilon_ok(epsilon) {
        return Ok(None);
    }
    if scale.len() < channels
        || bias.len() < channels
        || mean.len() < channels
        || var.len() < channels
    {
        return Ok(None);
    }
    let (Some(total_len), Ok(channels_u32), Ok(spatial_u32), Ok(n_u32)) = (
        n.checked_mul(channels).and_then(|v| v.checked_mul(spatial)),
        u32::try_from(channels),
        u32::try_from(spatial),
        u32::try_from(n),
    ) else {
        return Ok(None);
    };

    let template = BatchNormTemplate::new(
        PtxType::F32,
        BnMode::Inference,
        channels_u32,
        spatial_u32,
        epsilon,
        BLOCK_SIZE,
    );
    let kernel = kernel_for(ctx, &template)?;
    let stream = ctx.dnn.stream();

    let Some(mut d_input) = x.bind(ctx, INPUT_LABEL, total_len, stream)? else {
        return Err(CudaDispatchError::Shape {
            op: "batch_norm",
            msg: format!("input cannot supply the {total_len} elements its shape declares"),
        });
    };
    let mut d_scale = ctx.operand(ids.scale, SCALE_LABEL, &scale[..channels], stream)?;
    let mut d_bias = ctx.operand(ids.bias, BIAS_LABEL, &bias[..channels], stream)?;
    let mut d_mean = ctx.operand(ids.mean, MEAN_LABEL, &mean[..channels], stream)?;
    let mut d_var = ctx.operand(ids.var, VAR_LABEL, &var[..channels], stream)?;
    let d_output = ctx.scratch(total_len)?;

    // One block per channel; the kernel's own runtime `batch_count` loop
    // covers every sample of the batch in this single launch.
    let params = LaunchParams::new(Dim3::from(channels_u32), Dim3::from(BLOCK_SIZE));
    let args = (
        d_input.device_ptr(),
        d_output.device_ptr(),
        d_scale.device_ptr(),
        d_bias.device_ptr(),
        d_mean.device_ptr(),
        d_var.device_ptr(),
        n_u32,
    );
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    let out = finish_output(ctx, d_output, total_len, x_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => {
            d_input.retire();
            d_scale.retire();
            d_bias.retire();
            d_mean.retire();
            d_var.retire();
        }
        KernelOutput::Device(_) => {
            retire_queued(ctx, &mut d_input);
            retire_queued(ctx, &mut d_scale);
            retire_queued(ctx, &mut d_bias);
            retire_queued(ctx, &mut d_mean);
            retire_queued(ctx, &mut d_var);
        }
    }
    Ok(Some(out))
}

// ─── OxiInstanceNorm ────────────────────────────────────────────────────────

/// `(n, channels, spatial)` for an `OxiInstanceNorm` node, or `None` to
/// decline. Requires rank `>= 3` (`[N, C, d1, ...]`), matching
/// `oxionnx-ops::registry::oxi_instance_norm::spatial_size`'s own rank
/// floor.
#[must_use]
pub(crate) fn oxi_instance_norm_plan(x_shape: &[usize]) -> Option<(usize, usize, usize)> {
    if x_shape.len() < 3 {
        return None;
    }
    let n = x_shape[0];
    let channels = x_shape[1];
    let spatial: usize = x_shape[2..].iter().product();
    if n == 0 || channels == 0 || spatial == 0 {
        return None;
    }
    Some((n, channels, spatial))
}

/// Launch an `OxiInstanceNorm` kernel over an activation that may already be
/// on the device, leaving the result there when the caller asks for it.
///
/// See the [module docs](self) for why this issues `n` launches of the same
/// [`BnMode::Training`] kernel — one per sample, pointer-offset — rather
/// than one launch covering the whole batch.
///
/// # Errors
///
/// [`CudaDispatchError::Shape`] when the input cannot supply the elements
/// its shape declares or a dimension exceeds a `u32` kernel launch, or a
/// driver error from PTX compilation, allocation, upload, launch or
/// readback.
pub(crate) fn cuda_oxi_instance_norm_bound(
    ctx: &CudaContext,
    x: InputBinding<'_>,
    x_shape: &[usize],
    epsilon: f32,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    let Some((n, channels, spatial)) = oxi_instance_norm_plan(x_shape) else {
        return Ok(None);
    };
    if !epsilon_ok(epsilon) {
        return Ok(None);
    }
    let (Some(sample_elems), Ok(channels_u32), Ok(spatial_u32)) = (
        channels.checked_mul(spatial),
        u32::try_from(channels),
        u32::try_from(spatial),
    ) else {
        return Ok(None);
    };
    let Some(total_len) = n.checked_mul(sample_elems) else {
        return Ok(None);
    };
    // `batch_count = 1` is baked into every launch below (see the module
    // docs): the template's own runtime batch loop is not what covers `n`
    // here, this dispatch's own `for sample in 0..n` loop is.
    let one: u32 = 1;

    let template = BatchNormTemplate::new(
        PtxType::F32,
        BnMode::Training,
        channels_u32,
        spatial_u32,
        epsilon,
        BLOCK_SIZE,
    );
    let kernel = kernel_for(ctx, &template)?;
    let stream = ctx.dnn.stream();

    let Some(mut d_input) = x.bind(ctx, INPUT_LABEL, total_len, stream)? else {
        return Err(CudaDispatchError::Shape {
            op: "oxi_instance_norm",
            msg: format!("input cannot supply the {total_len} elements its shape declares"),
        });
    };
    let d_output = ctx.scratch(total_len)?;

    // The identity affine: gamma=1, beta=0 for every channel. Not weight-
    // cached -- see the module docs' "Weight residency".
    let ones = vec![1.0_f32; channels];
    let zeros = vec![0.0_f32; channels];
    let mut d_gamma = ctx.operand(None, IDENTITY_GAMMA_LABEL, &ones, stream)?;
    let mut d_beta = ctx.operand(None, IDENTITY_BETA_LABEL, &zeros, stream)?;

    let params = LaunchParams::new(Dim3::from(channels_u32), Dim3::from(BLOCK_SIZE));
    let elem_bytes: u64 = u64::from(std::mem::size_of::<f32>() as u32);
    let Some(sample_stride_bytes) = (sample_elems as u64).checked_mul(elem_bytes) else {
        return Ok(None);
    };
    for sample in 0..n {
        let Some(offset) = (sample as u64).checked_mul(sample_stride_bytes) else {
            return Ok(None);
        };
        let in_ptr = d_input.device_ptr() + offset;
        let out_ptr = d_output.device_ptr() + offset;
        let args = (
            in_ptr,
            out_ptr,
            d_gamma.device_ptr(),
            d_beta.device_ptr(),
            one,
        );
        kernel
            .launch(&params, stream, &args)
            .map_err(CudaDispatchError::Driver)?;
    }

    let out = finish_output(ctx, d_output, total_len, x_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => {
            d_input.retire();
            d_gamma.retire();
            d_beta.retire();
        }
        KernelOutput::Device(_) => {
            retire_queued(ctx, &mut d_input);
            retire_queued(ctx, &mut d_gamma);
            retire_queued(ctx, &mut d_beta);
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── batch_norm_plan ─────────────────────────────────────────────────────

    #[test]
    fn batch_norm_plan_reads_n_and_c_from_the_first_two_dims() {
        let (n, c, spatial) = batch_norm_plan(&[2, 8, 4, 4]).expect("must plan");
        assert_eq!((n, c, spatial), (2, 8, 16));
    }

    #[test]
    fn batch_norm_plan_defaults_spatial_to_one_below_rank_three() {
        let (n, c, spatial) = batch_norm_plan(&[1, 8]).expect("must plan");
        assert_eq!((n, c, spatial), (1, 8, 1));
    }

    #[test]
    fn batch_norm_plan_declines_rank_below_two() {
        assert!(batch_norm_plan(&[8]).is_none());
    }

    #[test]
    fn batch_norm_plan_declines_zero_batch_or_channels() {
        assert!(batch_norm_plan(&[0, 8, 4, 4]).is_none());
        assert!(batch_norm_plan(&[1, 0, 4, 4]).is_none());
    }

    // ── oxi_instance_norm_plan ──────────────────────────────────────────────

    #[test]
    fn instance_norm_plan_reads_n_c_and_flattens_the_spatial_tail() {
        let (n, c, spatial) = oxi_instance_norm_plan(&[2, 3, 4, 5]).expect("must plan");
        assert_eq!((n, c, spatial), (2, 3, 20));
    }

    #[test]
    fn instance_norm_plan_declines_rank_below_three() {
        assert!(oxi_instance_norm_plan(&[1, 8]).is_none());
    }

    #[test]
    fn instance_norm_plan_declines_a_zero_dimension() {
        assert!(oxi_instance_norm_plan(&[1, 8, 0, 4]).is_none());
    }

    #[test]
    fn instance_norm_plan_accepts_rank_three() {
        let (n, c, spatial) = oxi_instance_norm_plan(&[4, 6, 50]).expect("must plan");
        assert_eq!((n, c, spatial), (4, 6, 50));
    }

    // ── epsilon_ok ───────────────────────────────────────────────────────────

    #[test]
    fn epsilon_ok_accepts_the_onnx_default() {
        assert!(epsilon_ok(1e-5));
    }

    #[test]
    fn epsilon_ok_rejects_non_positive_and_non_finite() {
        assert!(!epsilon_ok(0.0));
        assert!(!epsilon_ok(-1e-5));
        assert!(!epsilon_ok(f32::NAN));
        assert!(!epsilon_ok(f32::INFINITY));
    }

    // ── cache_key vs. entry name ────────────────────────────────────────────

    #[test]
    fn cache_key_differs_for_different_epsilons_at_the_same_shape() {
        let entry = "batch_norm_infer_f32_c8_s16_bs256";
        assert_ne!(cache_key(entry, 1e-5), cache_key(entry, 1e-3));
    }

    #[test]
    fn cache_key_is_stable_for_the_same_epsilon() {
        let entry = "batch_norm_infer_f32_c8_s16_bs256";
        assert_eq!(cache_key(entry, 1e-5), cache_key(entry, 1e-5));
    }

    #[test]
    fn cuda_context_construction_never_panics_even_though_unavailable_here() {
        let _ = CudaContext::try_new();
    }
}
