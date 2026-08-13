//! CUDA dispatch for the channel/scalar broadcast patterns of `Add`/`Sub`/
//! `Mul`/`Div`: `[1,C,1,1]`-vs-`[1,C,H,W]` and scalar-vs-tensor.
//!
//! # Why this exists as a separate module from `elementwise.rs`
//!
//! `elementwise.rs`'s binary arm only ever dispatches when the two operand
//! shapes are *exactly* equal — the op-coverage audit that motivated this
//! module found that every one of the 86 CUDA-declined nodes across the
//! three real face-pipeline models was this same narrow broadcast shape
//! falling through that equality check. Rather than turn
//! `elementwise.rs`'s binary arm into a general N-D broadcast engine (which
//! nothing in these models needs), this module adds exactly the two
//! recognised patterns via [`classify`], a pure, GPU-free shape classifier,
//! and [`cuda_broadcast_bound`], which launches
//! [`oxicuda_ptx::templates::channel_broadcast::ChannelBroadcastTemplate`].
//!
//! # Addressing
//!
//! One thread per element of the *larger* ("full") operand. Thread `i` reads
//! `full[i]` and `small[(i / spatial) % channels]` (or `small[0]` when
//! `small` is a true scalar) — see the [`oxicuda_ptx`] template's module docs
//! for the exact trick. `spatial` is the product of every dimension of
//! `full`'s shape *after* the channel axis (`shape[2..]`), matching the ONNX
//! `[N, C, H, W]` layout.
//!
//! # Operand order and `Sub`/`Div`
//!
//! Unlike `Add`/`Mul`, `Sub`/`Div` are not commutative, so which ONNX input
//! (`node.inputs[0]` vs `[1]`) is the broadcast ("small") operand changes the
//! answer. [`classify`] records this as [`BroadcastPlan::lhs_is_small`]; the
//! caller combines it with the op to pick
//! [`oxicuda_ptx::templates::channel_broadcast::ChannelBroadcastTemplate::reverse`]
//! — see [`reverse_for`].
//!
//! # Residency and weight-caching
//!
//! Operands here are **not** weight-cached, mirroring `elementwise.rs`'s
//! binary arm and for the identical reason documented there: the "small"
//! operand in this pipeline's motivating case (InSwapper's AdaIN blocks) is
//! itself a *run-scoped activation* — a per-channel scale/shift produced by a
//! `Gemm` off the identity embedding, not a graph initializer — so caching it
//! by name would either miss every time or (worse, if the caller ever reused
//! a name across frames) silently serve a stale value. A genuine initializer
//! operand still benefits from this wave: it is small (a few hundred
//! floats), and `operand()`'s residency-aware binding already avoids the
//! upload entirely when it arrives as a device-resident activation from a
//! prior CUDA node.

use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_ptx::{
    ir::PtxType,
    templates::channel_broadcast::{ChannelBroadcastOp, ChannelBroadcastTemplate},
};
use oxionnx_core::graph::OpKind;

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency-cache slot label for the full (larger) operand.
const FULL_LABEL: &str = "channel_broadcast_full";
/// Residency-cache slot label for the small (broadcast) operand.
const SMALL_LABEL: &str = "channel_broadcast_small";

/// A recognised "full tensor `OP` per-channel-or-scalar operand" shape pair.
///
/// Produced by [`classify`]; everything a dispatcher needs to launch the
/// kernel without re-deriving shape arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BroadcastPlan {
    /// `true` when `node.inputs[0]` (the ONNX left-hand operand) is the
    /// *small* one, i.e. the kernel must compute `small OP full`, not `full
    /// OP small`, for a non-commutative op. See [`reverse_for`].
    pub(crate) lhs_is_small: bool,
    /// The full operand's channel count (`full.shape[1]`), or `1` when
    /// `small` is a true scalar (any rank, exactly one element).
    pub(crate) channels: usize,
    /// Product of every dimension of `full`'s shape after the channel axis.
    /// Meaningless (but never read as anything but a nonzero divisor) when
    /// `channels == 1`.
    pub(crate) spatial: usize,
    /// Element count of the full operand — the kernel's launch width.
    pub(crate) total_len: usize,
    /// Element count of the small operand, as uploaded (`1` for a scalar,
    /// `channels` for a per-channel operand).
    pub(crate) small_len: usize,
}

/// Classify a pair of operand shapes into a channel/scalar broadcast plan.
///
/// Pure and allocation-free, so this is unit-testable without a CUDA device
/// and cheap enough to call on every `Add`/`Sub`/`Mul`/`Div` node whose
/// shapes are not already exactly equal (the caller's cheaper, existing
/// check).
///
/// Returns `None` when neither operand order is a recognised broadcast — the
/// caller declines the node to the CPU exactly as it did before this module
/// existed.
///
/// Two shapes qualify as `(full, small)` when:
/// - `small` is a plain scalar — `small.iter().product() == 1`, covering
///   any rank (`[]`, `[1]`, `[1,1,1,1]`, ...) — or
/// - `full` and `small` share a rank `>= 2`, `small.shape[1] ==
///   full.shape[1]` (the channel count matches), and every other dimension
///   of `small` is `1` or matches `full`'s corresponding dimension (ONNX/
///   numpy unidirectional broadcasting, restricted to "only axis 1 may
///   differ" — the `[1,C,1,1]`-vs-`[1,C,H,W]` pattern this module targets,
///   generalised only as far as real models in this pipeline need: a
///   leading batch dim of `1` broadcasting against `N`).
#[must_use]
pub(crate) fn classify(a_shape: &[usize], b_shape: &[usize]) -> Option<BroadcastPlan> {
    if let Some((channels, spatial, total_len, small_len)) = as_full_small(a_shape, b_shape) {
        return Some(BroadcastPlan {
            lhs_is_small: false,
            channels,
            spatial,
            total_len,
            small_len,
        });
    }
    if let Some((channels, spatial, total_len, small_len)) = as_full_small(b_shape, a_shape) {
        return Some(BroadcastPlan {
            lhs_is_small: true,
            channels,
            spatial,
            total_len,
            small_len,
        });
    }
    None
}

/// One direction of [`classify`]'s test: is `small` broadcastable onto
/// `full` by this module's narrow rule? Returns `(channels, spatial,
/// total_len, small_len)`.
fn as_full_small(full: &[usize], small: &[usize]) -> Option<(usize, usize, usize, usize)> {
    let total_len: usize = full.iter().product();
    if total_len == 0 {
        return None;
    }
    let small_len: usize = small.iter().product();
    if small_len == 1 {
        // A true scalar broadcasts against every element regardless of
        // `full`'s own shape: channels=1 forces `small_idx` to 0 for every
        // thread (see the kernel's `(i / spatial) % channels`), so `spatial`
        // itself is never actually divided-by-zero-adjacent — any nonzero
        // value works, and `total_len` is the natural, already-computed one.
        return Some((1, total_len, total_len, 1));
    }
    if full.len() < 2 || full.len() != small.len() {
        return None;
    }
    let channels = full[1];
    if channels == 0 || small[1] != channels {
        return None;
    }
    for (d, (&f, &s)) in full.iter().zip(small.iter()).enumerate() {
        if d == 1 {
            continue;
        }
        if s != 1 && s != f {
            return None;
        }
    }
    let spatial: usize = full[2..].iter().product::<usize>().max(1);
    Some((channels, spatial, total_len, channels))
}

/// Whether the generated kernel must take
/// [`ChannelBroadcastTemplate::reverse`] for `op` given `plan`.
///
/// `Add`/`Mul` are commutative — the *values* combined are correct
/// regardless of which operand the kernel calls `full`/`small`, so this is
/// always `false` for them, letting every `Add`/`Mul` broadcast share one
/// compiled kernel instead of two. `Sub`/`Div` depend on order: ONNX
/// `Sub(a, b)` computes `a - b`, so if `a` (`node.inputs[0]`) is the small
/// operand (`plan.lhs_is_small`), the kernel must compute `small - full`,
/// which is the *reverse* form.
#[must_use]
pub(crate) fn reverse_for(op: &OpKind, plan: BroadcastPlan) -> bool {
    matches!(op, OpKind::Sub | OpKind::Div) && plan.lhs_is_small
}

/// Map an ONNX binary op to [`ChannelBroadcastOp`].
fn channel_broadcast_op_for(op: &OpKind) -> Result<ChannelBroadcastOp, CudaDispatchError> {
    match op {
        OpKind::Add => Ok(ChannelBroadcastOp::Add),
        OpKind::Sub => Ok(ChannelBroadcastOp::Sub),
        OpKind::Mul => Ok(ChannelBroadcastOp::Mul),
        OpKind::Div => Ok(ChannelBroadcastOp::Div),
        other => Err(CudaDispatchError::Unsupported {
            op: "channel_broadcast",
            reason: format!(
                "no CUDA channel-broadcast kernel for ONNX op '{}'",
                other.as_str()
            ),
        }),
    }
}

/// Fetch — compiling on first use — the kernel for one (op, direction) pair.
fn kernel_for(
    ctx: &CudaContext,
    op: ChannelBroadcastOp,
    reverse: bool,
) -> Result<Kernel, CudaDispatchError> {
    let template = ChannelBroadcastTemplate {
        op,
        reverse,
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

/// Launch a channel/scalar broadcast binary kernel, over operands that may
/// already be on the device, leaving the result there when the caller asks
/// for it.
///
/// `full`/`small` must already be arranged so that `full` is the larger
/// operand `plan` was classified against — i.e. the caller resolves
/// [`BroadcastPlan::lhs_is_small`] into the right (full, small) binding order
/// *before* calling this function; `reverse` (from [`reverse_for`]) is what
/// then restores the correct `Sub`/`Div` operand order inside the kernel.
///
/// # Errors
///
/// [`CudaDispatchError::Unsupported`] for an op with no channel-broadcast
/// kernel, [`CudaDispatchError::Shape`] if an operand cannot supply the
/// elements `plan` declares or the launch width exceeds a `u32`, or a driver
/// error from PTX compilation, allocation, upload, launch or readback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_broadcast_bound(
    ctx: &CudaContext,
    full: InputBinding<'_>,
    small: InputBinding<'_>,
    plan: BroadcastPlan,
    op: &OpKind,
    out_shape: &[usize],
    placement: CudaOutputPlacement,
) -> Result<KernelOutput, CudaDispatchError> {
    let reverse = reverse_for(op, plan);
    let kernel = kernel_for(ctx, channel_broadcast_op_for(op)?, reverse)?;

    let Ok(total_u32) = u32::try_from(plan.total_len) else {
        return Err(CudaDispatchError::Shape {
            op: "channel_broadcast",
            msg: format!("{} elements exceed a u32 kernel launch", plan.total_len),
        });
    };
    let (Ok(channels_u32), Ok(spatial_u32), Ok(small_len_u32)) = (
        u32::try_from(plan.channels),
        u32::try_from(plan.spatial),
        u32::try_from(plan.small_len),
    ) else {
        return Err(CudaDispatchError::Shape {
            op: "channel_broadcast",
            msg: format!(
                "channels={} spatial={} small_len={} do not fit a u32 kernel launch",
                plan.channels, plan.spatial, plan.small_len
            ),
        });
    };

    let stream = ctx.dnn.stream();
    let (Some(mut d_full), Some(mut d_small)) = (
        full.bind(ctx, FULL_LABEL, plan.total_len, stream)?,
        small.bind(ctx, SMALL_LABEL, plan.small_len, stream)?,
    ) else {
        return Err(CudaDispatchError::Shape {
            op: "channel_broadcast",
            msg: "an operand cannot supply the elements its shape declares".to_string(),
        });
    };
    let d_output = ctx.scratch(plan.total_len)?;

    let grid = grid_size_for(total_u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));
    let args = (
        d_full.device_ptr(),
        d_small.device_ptr(),
        d_output.device_ptr(),
        total_u32,
        channels_u32,
        spatial_u32,
        small_len_u32,
    );
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    let out = finish_output(ctx, d_output, plan.total_len, out_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => {
            d_full.retire();
            d_small.retire();
        }
        KernelOutput::Device(_) => {
            retire_queued(ctx, &mut d_full);
            retire_queued(ctx, &mut d_small);
        }
    }
    Ok(out)
}

const BLOCK_SIZE: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    // ── classify: exactly the shapes real models present ───────────────────

    #[test]
    fn classifies_the_onnx_channel_pattern_rhs_small() {
        // [1,C,1,1] vs [1,C,H,W], small on the right (node.inputs[1]).
        let plan = classify(&[1, 64, 32, 32], &[1, 64, 1, 1]).expect("must classify");
        assert!(!plan.lhs_is_small);
        assert_eq!(plan.channels, 64);
        assert_eq!(plan.spatial, 32 * 32);
        assert_eq!(plan.total_len, 64 * 32 * 32);
        assert_eq!(plan.small_len, 64);
    }

    #[test]
    fn classifies_the_onnx_channel_pattern_lhs_small() {
        let plan = classify(&[1, 64, 1, 1], &[1, 64, 32, 32]).expect("must classify");
        assert!(plan.lhs_is_small);
        assert_eq!(plan.channels, 64);
        assert_eq!(plan.spatial, 32 * 32);
    }

    #[test]
    fn classifies_a_true_scalar_against_a_tensor() {
        let plan = classify(&[1, 3, 8, 8], &[1]).expect("must classify");
        assert!(!plan.lhs_is_small);
        assert_eq!(plan.channels, 1);
        assert_eq!(plan.small_len, 1);
        assert_eq!(plan.total_len, 3 * 8 * 8);
    }

    #[test]
    fn classifies_a_rank_zero_scalar() {
        let plan = classify(&[1, 3, 8, 8], &[]).expect("must classify");
        assert_eq!(plan.small_len, 1);
    }

    #[test]
    fn classifies_a_scalar_on_the_left() {
        let plan = classify(&[1], &[1, 3, 8, 8]).expect("must classify");
        assert!(plan.lhs_is_small);
        assert_eq!(plan.channels, 1);
    }

    #[test]
    fn classifies_rank_two_with_the_channel_axis_at_index_one() {
        // [N,C] vs [1,C]: rank 2, no trailing spatial dims (spatial defaults
        // to 1) -- axis 1 is always the channel axis, matching NCHW and
        // every other plan function in this crate (`prelu_plan`,
        // `batch_norm_plan`, ...).
        let plan = classify(&[2, 5], &[1, 5]).expect("must classify");
        assert_eq!(plan.channels, 5);
        assert_eq!(plan.spatial, 1);
        assert_eq!(plan.total_len, 10);
        assert_eq!(plan.small_len, 5);
    }

    #[test]
    fn declines_mismatched_channel_counts() {
        assert!(classify(&[1, 64, 8, 8], &[1, 32, 1, 1]).is_none());
    }

    #[test]
    fn declines_rank_mismatch_that_is_not_a_scalar() {
        // A bare [C] vector is NOT the [1,C,1,1] pattern (it would numpy-
        // broadcast against the *last* axis, a different rule this module
        // deliberately does not implement).
        assert!(classify(&[1, 3, 4, 4], &[3]).is_none());
    }

    #[test]
    fn declines_a_non_channel_dim_that_disagrees_and_is_not_one() {
        // small's spatial dim (5) is neither 1 nor equal to full's (4).
        assert!(classify(&[1, 3, 4, 4], &[1, 3, 5, 4]).is_none());
    }

    #[test]
    fn declines_transposed_equal_size_shapes() {
        assert!(classify(&[2, 3], &[3, 2]).is_none());
    }

    #[test]
    fn declines_a_zero_sized_full_candidate() {
        assert!(classify(&[0, 3], &[1]).is_none());
    }

    #[test]
    fn declines_rank_below_two_for_the_channel_pattern() {
        // Rank-1 non-scalar operands: no channel axis to match.
        assert!(classify(&[4], &[4]).is_none());
    }

    #[test]
    fn batch_broadcast_within_the_channel_pattern_is_accepted() {
        // full's batch is 2, small's is 1: axis 0 != channel axis, so it may
        // broadcast too (the "only axis 1 may differ" rule permits any
        // non-channel axis to be 1-vs-N).
        let plan = classify(&[2, 8, 4, 4], &[1, 8, 1, 1]).expect("must classify");
        assert_eq!(plan.channels, 8);
        assert_eq!(plan.spatial, 16);
        assert_eq!(plan.total_len, 2 * 8 * 4 * 4);
    }

    // ── reverse_for: commutative ops never need it ──────────────────────────

    #[test]
    fn add_and_mul_never_reverse() {
        let plan_lhs = BroadcastPlan {
            lhs_is_small: true,
            channels: 4,
            spatial: 4,
            total_len: 16,
            small_len: 4,
        };
        let plan_rhs = BroadcastPlan {
            lhs_is_small: false,
            ..plan_lhs
        };
        for plan in [plan_lhs, plan_rhs] {
            assert!(!reverse_for(&OpKind::Add, plan));
            assert!(!reverse_for(&OpKind::Mul, plan));
        }
    }

    #[test]
    fn sub_and_div_reverse_exactly_when_the_small_operand_is_on_the_left() {
        let lhs_small = BroadcastPlan {
            lhs_is_small: true,
            channels: 4,
            spatial: 4,
            total_len: 16,
            small_len: 4,
        };
        let rhs_small = BroadcastPlan {
            lhs_is_small: false,
            ..lhs_small
        };
        assert!(reverse_for(&OpKind::Sub, lhs_small));
        assert!(reverse_for(&OpKind::Div, lhs_small));
        assert!(!reverse_for(&OpKind::Sub, rhs_small));
        assert!(!reverse_for(&OpKind::Div, rhs_small));
    }

    #[test]
    fn channel_broadcast_op_for_covers_exactly_the_four_binary_ops() {
        assert!(channel_broadcast_op_for(&OpKind::Add).is_ok());
        assert!(channel_broadcast_op_for(&OpKind::Sub).is_ok());
        assert!(channel_broadcast_op_for(&OpKind::Mul).is_ok());
        assert!(channel_broadcast_op_for(&OpKind::Div).is_ok());
        assert!(channel_broadcast_op_for(&OpKind::Relu).is_err());
    }

    #[test]
    fn cuda_context_construction_never_panics_even_though_unavailable_here() {
        let _ = CudaContext::try_new();
    }
}
