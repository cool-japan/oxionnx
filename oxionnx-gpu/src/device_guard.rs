//! Device-level safety guards shared by every GPU entry point in this crate.
//!
//! Every `gpu_*` function here has the same contract: it returns `Option<_>`,
//! and `None` means "decline, let the CPU operator handle this node". A GPU
//! backend that can abort the host process breaks that contract, so all of the
//! situations that wgpu turns into a panic by default are funnelled through the
//! helpers in this module and converted into a decline:
//!
//! * a dispatch wider than [`wgpu::Limits::max_compute_workgroups_per_dimension`]
//!   (split into a 2-D grid, or declined when even that is not enough),
//! * a buffer larger than `max_storage_buffer_binding_size` / `max_buffer_size`,
//! * any wgpu validation / out-of-memory / device-lost error (captured by an
//!   error scope, or by the uncaptured-error handler installed on the device),
//! * a read-back that never completes (bounded wait instead of an infinite one).

#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use crate::context::GpuContext;

/// Bytes occupied by one `f32`.
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

/// Upper bound on how long a blocking read-back may wait before the context is
/// declared degraded and the operation falls back to the CPU.
///
/// Only the native read-back blocks, so this is native-only; on wasm32 there is
/// no blocking path at all (and no context — see `GpuContext::try_new_async`).
#[cfg(not(target_arch = "wasm32"))]
const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Byte size of `count` `f32` values, or `None` on overflow.
#[inline]
pub(crate) fn f32_bytes(count: usize) -> Option<u64> {
    (count as u64).checked_mul(F32_BYTES)
}

// ========================================================================
// Cached device limits
// ========================================================================

/// The subset of [`wgpu::Limits`] the dispatch paths need, cached once at
/// context creation so hot paths never clone the full limits struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLimits {
    /// Largest byte size that may be bound as a storage buffer.
    pub max_storage_buffer_binding_size: u64,
    /// Largest byte size a single buffer allocation may have.
    pub max_buffer_size: u64,
    /// Largest workgroup count along any one dispatch dimension.
    pub max_workgroups_per_dimension: u32,
}

impl GpuLimits {
    /// Read the relevant limits out of a live device.
    pub(crate) fn from_device(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// True when a buffer of `bytes` can be created *and* bound as storage.
    #[inline]
    #[must_use]
    pub fn storage_fits(&self, bytes: u64) -> bool {
        bytes <= self.max_storage_buffer_binding_size && bytes <= self.max_buffer_size
    }

    /// True when a buffer of `bytes` can be created (staging / upload buffers
    /// that are never bound as storage).
    #[inline]
    #[must_use]
    pub fn buffer_fits(&self, bytes: u64) -> bool {
        bytes <= self.max_buffer_size
    }

    /// True when every entry of `byte_sizes` can be bound as a storage buffer.
    #[inline]
    #[must_use]
    pub fn all_storage_fit(&self, byte_sizes: &[u64]) -> bool {
        byte_sizes.iter().all(|&b| self.storage_fits(b))
    }
}

/// Storage-buffer byte size of an `f32` tensor with `count` elements, or `None`
/// when the allocation would overflow, exceed the device limits, or hold more
/// elements than the `u32` flat index every kernel here uses can address.
///
/// The `u32` bound matters on adapters that advertise a very large
/// `max_storage_buffer_binding_size` (some report the full `max_buffer_size`):
/// without it a >4G-element binding would wrap the index arithmetic inside the
/// shader and produce silently wrong values instead of a decline.
#[inline]
pub(crate) fn checked_storage_bytes(limits: &GpuLimits, count: usize) -> Option<u64> {
    if u64::try_from(count).ok()? > u64::from(u32::MAX) {
        return None;
    }
    let bytes = f32_bytes(count)?;
    if limits.storage_fits(bytes) {
        Some(bytes)
    } else {
        None
    }
}

// ========================================================================
// Dispatch grid planning
// ========================================================================

/// A workgroup grid that covers a flat thread range without exceeding the
/// device's per-dimension workgroup limit.
///
/// Shaders reconstruct the flat element index as
/// `gid.y * threads_per_row + gid.x`, which degenerates to `gid.x` whenever the
/// grid fits in one dimension (`y == 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DispatchGrid {
    /// Workgroups along X.
    pub x: u32,
    /// Workgroups along Y.
    pub y: u32,
    /// Threads covered by one full row of X workgroups (`x * workgroup_size`).
    pub threads_per_row: u32,
}

/// Plan a 1-D-or-2-D workgroup grid covering `total_threads`, given an explicit
/// per-dimension limit. Split out from [`plan_dispatch`] so the arithmetic is
/// testable without a GPU.
///
/// Returns `None` when the work cannot be expressed within the limit, when the
/// flat index would not fit in a `u32`, or when `total_threads` is zero.
pub(crate) fn plan_dispatch_with_limit(
    total_threads: u64,
    workgroup_size: u32,
    max_per_dimension: u32,
) -> Option<DispatchGrid> {
    if total_threads == 0 || workgroup_size == 0 || max_per_dimension == 0 {
        return None;
    }
    // The shader indexes with `u32`, so the flat range must fit.
    if total_threads > u64::from(u32::MAX) {
        return None;
    }

    let wg_total = total_threads.div_ceil(u64::from(workgroup_size));

    // Never let `x * workgroup_size` overflow the u32 the shader uses.
    let max_x = max_per_dimension.min(u32::MAX / workgroup_size).max(1);
    let max_x_u64 = u64::from(max_x);

    let (x, y) = if wg_total <= max_x_u64 {
        (wg_total, 1u64)
    } else {
        let y = wg_total.div_ceil(max_x_u64);
        if y > u64::from(max_per_dimension) {
            return None;
        }
        (max_x_u64, y)
    };

    let threads_per_row = x.checked_mul(u64::from(workgroup_size))?;
    // Largest index the shader can compute is `y * threads_per_row - 1`.
    if threads_per_row.checked_mul(y)? > u64::from(u32::MAX) {
        return None;
    }

    Some(DispatchGrid {
        x: u32::try_from(x).ok()?,
        y: u32::try_from(y).ok()?,
        threads_per_row: u32::try_from(threads_per_row).ok()?,
    })
}

/// Plan a workgroup grid for this device.
#[inline]
pub(crate) fn plan_dispatch(
    limits: &GpuLimits,
    total_threads: u64,
    workgroup_size: u32,
) -> Option<DispatchGrid> {
    plan_dispatch_with_limit(
        total_threads,
        workgroup_size,
        limits.max_workgroups_per_dimension,
    )
}

/// Validate an explicitly 2-D dispatch (the matmul kernels already index with
/// `gid.x` / `gid.y`, so they only need the limit check).
#[inline]
pub(crate) fn dispatch_2d_fits(limits: &GpuLimits, wg_x: u32, wg_y: u32) -> bool {
    wg_x > 0
        && wg_y > 0
        && wg_x <= limits.max_workgroups_per_dimension
        && wg_y <= limits.max_workgroups_per_dimension
}

// ========================================================================
// Error scopes
// ========================================================================

/// RAII wrapper around a wgpu validation error scope.
///
/// Created before the buffers of one dispatch are allocated and consumed by
/// [`ErrorScope::finish`] right after the submit. Any validation or
/// out-of-memory error raised in between is reported as a `false` result
/// instead of reaching wgpu's default handler (which panics).
pub(crate) struct ErrorScope {
    guard: Option<wgpu::ErrorScopeGuard>,
}

impl ErrorScope {
    /// Begin capturing validation errors on the calling thread.
    pub(crate) fn begin(ctx: &GpuContext) -> Self {
        Self {
            guard: Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation)),
        }
    }

    /// Pop the scope. Returns `true` when nothing went wrong; on an error the
    /// context is marked degraded so later nodes go straight to the CPU.
    ///
    /// A `Validation` filter does not capture [`wgpu::Error::OutOfMemory`] or
    /// [`wgpu::Error::Internal`]; those reach the uncaptured-error handler
    /// installed in `build_from_device_queue`, which sets the degraded flag. So
    /// the flag is consulted here too — otherwise an allocation failure would
    /// leave this dispatch reading back a dead buffer.
    pub(crate) fn finish(mut self, ctx: &GpuContext) -> bool {
        let Some(guard) = self.guard.take() else {
            return !ctx.is_degraded();
        };
        #[cfg(target_arch = "wasm32")]
        {
            // Cannot block on the pop future in the browser; dropping the guard
            // pops the scope and discards whatever it captured.
            drop(guard);
            !ctx.is_degraded()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            match pollster::block_on(guard.pop()) {
                Some(err) => {
                    ctx.mark_degraded(err.to_string());
                    false
                }
                None => !ctx.is_degraded(),
            }
        }
    }
}

// ========================================================================
// Read-back
// ========================================================================

/// Read a mapped staging buffer back into a `Vec<f32>`.
///
/// The wait is bounded: a device that never completes the submission (driver
/// reset, GPU hang, lost device) marks the context degraded and yields `None`
/// rather than blocking the calling thread forever.
///
/// On wasm32 blocking is not possible at all, so this always returns `None`.
pub(crate) fn read_back(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    count: usize,
) -> Option<Vec<f32>> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (ctx, staging, count);
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Bounded wait: a lost device or a timeout is an error, not a hang.
        if let Err(err) = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(READBACK_TIMEOUT),
        }) {
            ctx.mark_degraded(format!("gpu readback poll failed: {err}"));
            return None;
        }

        match receiver.recv_timeout(READBACK_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                // Mapping itself failed — decline, but the device is still usable.
                let _ = err;
                return None;
            }
            Err(err) => {
                ctx.mark_degraded(format!("gpu readback did not complete: {err}"));
                return None;
            }
        }

        let data = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        let result = values.get(..count).map(<[f32]>::to_vec);
        drop(data);
        staging.unmap();
        result
    }
}

/// Read `count` f32s back from `staging`, then hand `output` to the context's
/// buffer pool — but **only** if the read-back actually completed.
///
/// [a7-10] Every kernel used to recycle its output buffer unconditionally,
/// right after `queue.submit`. On the failure paths of [`read_back`] the
/// submission may still be executing (the poll timed out, or the device was
/// lost mid-flight), so returning the buffer there lets a later dispatch pull
/// it out of the pool and bind it while the GPU is still writing to it. When
/// the read-back fails the buffer is simply dropped instead; wgpu keeps the
/// allocation alive until the submission it belongs to retires.
///
/// A poisoned pool mutex only costs the recycling, never the result — the old
/// `ctx.pool.lock().ok()?` discarded a perfectly good tensor in that case.
pub(crate) fn read_back_and_recycle(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    count: usize,
    output: wgpu::Buffer,
) -> Option<Vec<f32>> {
    let result = read_back(ctx, staging, count);
    if result.is_some() {
        if let Ok(mut pool) = ctx.pool.lock() {
            pool.return_buffer(output);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const WGPU_DEFAULT_MAX_DIM: u32 = 65535;

    #[test]
    fn plan_dispatch_one_dimensional_below_limit() {
        // 100_000 elements at 256 threads/workgroup => 391 workgroups.
        let grid = plan_dispatch_with_limit(100_000, 256, WGPU_DEFAULT_MAX_DIM)
            .expect("small dispatch must be plannable");
        assert_eq!(grid.x, 391);
        assert_eq!(grid.y, 1);
        assert_eq!(grid.threads_per_row, 391 * 256);
        // Every element is covered.
        assert!(u64::from(grid.x) * u64::from(grid.y) * 256 >= 100_000);
    }

    #[test]
    fn plan_dispatch_exactly_at_the_limit_stays_one_dimensional() {
        // 65535 * 256 = 16_776_960 threads is the largest 1-D dispatch.
        let total = 65_535u64 * 256;
        let grid = plan_dispatch_with_limit(total, 256, WGPU_DEFAULT_MAX_DIM)
            .expect("boundary dispatch must be plannable");
        assert_eq!(grid.x, 65_535);
        assert_eq!(grid.y, 1);
    }

    #[test]
    fn plan_dispatch_splits_into_two_dimensions_past_the_limit() {
        // One element past the 1-D capacity: 16_776_961 threads => 65_536 workgroups.
        let grid = plan_dispatch_with_limit(16_776_961, 256, WGPU_DEFAULT_MAX_DIM)
            .expect("oversized dispatch must split into a 2-D grid");
        assert_eq!(grid.x, 65_535);
        assert_eq!(grid.y, 2);
        assert_eq!(grid.threads_per_row, 65_535 * 256);
        assert!(grid.x <= WGPU_DEFAULT_MAX_DIM && grid.y <= WGPU_DEFAULT_MAX_DIM);
        // Covers everything: 2 * 65535 * 256 = 33_553_920 >= 16_776_961.
        assert!(u64::from(grid.x) * u64::from(grid.y) * 256 >= 16_776_961);
    }

    #[test]
    fn plan_dispatch_relu_16m_case_from_the_audit() {
        // [1, 64, 512, 512] Relu activation = 16_777_216 elements.
        let grid = plan_dispatch_with_limit(16_777_216, 256, WGPU_DEFAULT_MAX_DIM)
            .expect("16M-element relu must be plannable");
        assert_eq!((grid.x, grid.y), (65_535, 2));
    }

    #[test]
    fn plan_dispatch_one_workgroup_per_instance_layer_norm() {
        // LayerNorm dispatches one workgroup per instance: 65_536 instances.
        let grid = plan_dispatch_with_limit(65_536, 1, WGPU_DEFAULT_MAX_DIM)
            .expect("65536 instances must be plannable");
        assert_eq!(grid.x, 65_535);
        assert_eq!(grid.y, 2);
        assert_eq!(grid.threads_per_row, 65_535);
    }

    #[test]
    fn plan_dispatch_declines_zero_and_overflow() {
        assert!(plan_dispatch_with_limit(0, 256, WGPU_DEFAULT_MAX_DIM).is_none());
        assert!(plan_dispatch_with_limit(1000, 0, WGPU_DEFAULT_MAX_DIM).is_none());
        assert!(plan_dispatch_with_limit(1000, 256, 0).is_none());
        // Beyond u32 addressing.
        assert!(
            plan_dispatch_with_limit(u64::from(u32::MAX) + 1, 256, WGPU_DEFAULT_MAX_DIM).is_none()
        );
    }

    #[test]
    fn plan_dispatch_declines_when_two_dimensions_are_not_enough() {
        // A tiny per-dimension limit makes even a 2-D grid insufficient.
        assert!(plan_dispatch_with_limit(4_000_000, 1, 4).is_none());
        // ... but the same work fits with a larger limit.
        assert!(plan_dispatch_with_limit(4_000_000, 1, 65_535).is_some());
    }

    #[test]
    fn plan_dispatch_never_overflows_threads_per_row() {
        // A device advertising an enormous per-dimension limit must not make
        // `x * workgroup_size` wrap around.
        let grid = plan_dispatch_with_limit(1_000_000_000, 256, u32::MAX)
            .expect("large limit must still plan");
        assert!(u64::from(grid.threads_per_row) * u64::from(grid.y) <= u64::from(u32::MAX));
        assert!(u64::from(grid.x) * u64::from(grid.y) * 256 >= 1_000_000_000);
    }

    #[test]
    fn limits_reject_oversized_buffers() {
        let limits = GpuLimits {
            max_storage_buffer_binding_size: 128 << 20,
            max_buffer_size: 256 << 20,
            max_workgroups_per_dimension: 65_535,
        };
        // The lm_head projection from the audit: b = [4096, 32000] f32 = 524 MB.
        let lm_head_bytes = f32_bytes(4096 * 32_000).expect("no overflow");
        assert_eq!(lm_head_bytes, 524_288_000);
        assert!(!limits.storage_fits(lm_head_bytes));
        assert!(!limits.buffer_fits(lm_head_bytes));
        // 32 MiB is fine.
        assert!(limits.storage_fits(32 << 20));
        assert!(limits.all_storage_fit(&[1024, 32 << 20]));
        assert!(!limits.all_storage_fit(&[1024, lm_head_bytes]));
        assert!(checked_storage_bytes(&limits, 4096 * 32_000).is_none());
        assert_eq!(checked_storage_bytes(&limits, 1024), Some(4096));
    }

    #[test]
    fn checked_storage_bytes_rejects_counts_beyond_u32_indexing() {
        // An adapter that advertises a huge storage binding limit must still not
        // be handed a tensor the shaders' u32 flat index cannot address.
        let huge = GpuLimits {
            max_storage_buffer_binding_size: u64::MAX,
            max_buffer_size: u64::MAX,
            max_workgroups_per_dimension: 65_535,
        };
        let max_addressable = u32::MAX as usize;
        assert_eq!(
            checked_storage_bytes(&huge, max_addressable),
            Some(u64::from(u32::MAX) * 4)
        );
        assert!(checked_storage_bytes(&huge, max_addressable + 1).is_none());
    }

    #[test]
    fn f32_bytes_is_checked() {
        assert_eq!(f32_bytes(0), Some(0));
        assert_eq!(f32_bytes(10), Some(40));
        assert_eq!(f32_bytes(usize::MAX), None);
    }

    #[test]
    fn dispatch_2d_fits_respects_the_limit() {
        let limits = GpuLimits {
            max_storage_buffer_binding_size: 128 << 20,
            max_buffer_size: 256 << 20,
            max_workgroups_per_dimension: 65_535,
        };
        assert!(dispatch_2d_fits(&limits, 65_535, 65_535));
        assert!(!dispatch_2d_fits(&limits, 65_536, 1));
        assert!(!dispatch_2d_fits(&limits, 1, 65_536));
        assert!(!dispatch_2d_fits(&limits, 0, 1));
    }
}
