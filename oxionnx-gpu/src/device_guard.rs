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
//!
//! # Sync and async
//!
//! Every kernel in this crate is written **once**, as an `async fn` ending in
//! `read_back_and_recycle_async`. The synchronous `gpu_*` entry points are
//! one-line wrappers around `block_on_gpu`:
//!
//! * **native** — `block_on_gpu` is `pollster::block_on`, and the awaited
//!   read-back is the same `poll(Wait)` + `recv_timeout` pair it always was
//!   (see `read_back_blocking`), so the future completes in a single `poll`
//!   and native behaviour is unchanged, instruction for instruction.
//! * **wasm32** — `block_on_gpu` drops the future without polling it and
//!   returns `None`, because a browser thread cannot block. Callers there use
//!   the `*_async` entry points, whose read-back is a real `map_async` future
//!   driven by the JS microtask queue (`read_back_web`).

#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use crate::context::activation::{DeviceTensor, GpuOutput, OutputPlacement};
use crate::context::{GpuContext, TrackedBuffer};
use oxionnx_core::Tensor;

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
    ///
    /// # Why this is the only variant
    ///
    /// There used to be a synchronous `finish` too, whose wasm32 arm simply
    /// **dropped** the guard — which pops the scope and discards whatever it
    /// captured, so every validation error in the browser was invisible.
    /// `ErrorScopeGuard::pop` returns a future on both backends
    /// (`wgpu/src/api/device.rs`), and on native that future is
    /// `ready(scope.error)` — already complete before the first `poll`
    /// (`wgpu/src/backend/wgpu_core.rs:1883`). So awaiting it costs a native
    /// caller nothing, and there is no reason to keep a second, weaker
    /// implementation alive next to it.
    ///
    /// # Ordering contract
    ///
    /// wgpu error scopes are a per-thread LIFO stack and the native backend
    /// *panics* if they are popped out of order. Two GPU dispatches from this
    /// crate must therefore never be in flight at the same time on one device —
    /// which is exactly what the sequential, one-op-at-a-time async run loop
    /// guarantees. Do not `join!` two `gpu_*_async` calls.
    pub(crate) async fn finish_async(mut self, ctx: &GpuContext) -> bool {
        let Some(guard) = self.guard.take() else {
            return !ctx.is_degraded();
        };
        match guard.pop().await {
            Some(err) => {
                ctx.mark_degraded(err.to_string());
                false
            }
            None => !ctx.is_degraded(),
        }
    }

    /// Blocking form of [`Self::finish_async`], for kernels that are still
    /// written synchronously.
    ///
    /// On wasm32 this reports `false` — "treat this dispatch as failed" — which
    /// makes a synchronous kernel decline to the CPU, the only correct outcome
    /// there. The scope is still popped: the unpolled future owns `self`, so
    /// dropping it runs `ErrorScopeGuard`'s own `Drop`, and the per-thread
    /// scope stack stays balanced.
    ///
    /// # Do not reach for this when wiring a kernel into the async dispatcher
    ///
    /// This shim (and [`read_back_and_recycle`]) exists so kernels written
    /// against the old synchronous API keep compiling while they are being
    /// integrated — nothing more. Calling a *synchronous* kernel from
    /// `try_gpu_dispatch_async` compiles and works natively, and in a browser
    /// it silently never runs: `block_on_gpu` drops the future unpolled and
    /// yields `None`, so the op quietly falls back to the CPU on exactly the
    /// target the async dispatcher was built for, with no error anywhere. That
    /// is the "GPU as pure overhead in the browser" failure this wave removed.
    ///
    /// Integrating a kernel means converting its body to an `async fn` ending
    /// in [`read_back_and_recycle_async`] and adding a one-line
    /// `block_on_gpu` wrapper for the synchronous name — the shape every
    /// kernel in `shaders/` and `compute.rs` already has.
    ///
    /// [R3b] That conversion is now complete for every kernel this crate
    /// ships (the K2 batch — `broadcast_binary`/`gemm`/`pad`/`prelu`/`resize`
    /// — was the last holdout), so nothing calls this synchronous shim
    /// anymore; `cargo check` proves it (a `dead_code` warning here without
    /// the `#[allow]` below), which is a stronger guarantee than any test
    /// could give that no kernel can silently no-op in a browser. Left in
    /// place rather than deleted: this file is outside the K2-conversion
    /// wave's file ownership, and removing `pub(crate)` infrastructure is a
    /// bigger edit than that wave's scope warrants. A follow-up cleanup wave
    /// that owns this file can delete both this method and
    /// [`read_back_and_recycle`] once that is confirmed stable.
    #[allow(dead_code)]
    pub(crate) fn finish(self, ctx: &GpuContext) -> bool {
        block_on_gpu(async move { Some(self.finish_async(ctx).await) }).unwrap_or(false)
    }
}

// ========================================================================
// Blocking bridge
// ========================================================================

/// Drive `future` to completion on a target that is allowed to block.
///
/// This is the single seam between the crate's `async` kernels and its
/// synchronous public API. On native it is `pollster::block_on` — the same
/// pattern `GpuContext::try_new` already used for adapter acquisition. On
/// wasm32 it declines: the future is dropped **unpolled** (an `async fn` body
/// runs only when polled, so nothing is submitted, allocated or uploaded), and
/// the caller falls back to the CPU operator.
///
/// That decline is not a limitation of this function, it is the correct answer:
/// blocking a browser's only thread on a GPU fence deadlocks the page. Browser
/// callers use the `*_async` entry points instead.
#[inline]
pub(crate) fn block_on_gpu<T>(future: impl core::future::Future<Output = Option<T>>) -> Option<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(future)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _unpolled = future;
        None
    }
}

// ========================================================================
// Read-back
// ========================================================================

/// Copy the first `count` `f32`s out of a mapped byte range.
///
/// Deliberately **not** `bytemuck::cast_slice`, which panics on a misaligned
/// source. On the WebGPU backend `get_mapped_range` hands back a range whose
/// backing pointer carries no `f32` alignment guarantee at all (it is
/// materialized out of a JS `ArrayBuffer` copy), so the aligned cast is a
/// best-effort fast path and the byte-wise decode is the contract. Both are
/// little-endian, which is what WebGPU specifies and what every target this
/// crate builds for uses.
fn decode_f32(bytes: &[u8], count: usize) -> Option<Vec<f32>> {
    let needed = count.checked_mul(std::mem::size_of::<f32>())?;
    let src = bytes.get(..needed)?;
    if let Ok(values) = bytemuck::try_cast_slice::<u8, f32>(src) {
        return Some(values.to_vec());
    }
    let mut out = Vec::with_capacity(count);
    for chunk in src.chunks_exact(std::mem::size_of::<f32>()) {
        let quad: [u8; 4] = chunk.try_into().ok()?;
        out.push(f32::from_le_bytes(quad));
    }
    Some(out)
}

/// Read a mapped staging buffer back into a `Vec<f32>`, blocking the calling
/// thread until the submission retires.
///
/// The wait is bounded: a device that never completes the submission (driver
/// reset, GPU hang, lost device) marks the context degraded and yields `None`
/// rather than blocking the calling thread forever.
#[cfg(not(target_arch = "wasm32"))]
fn read_back_blocking(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    count: usize,
    bytes: u64,
) -> Option<Vec<f32>> {
    let slice = staging.slice(0..bytes);
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
    let result = decode_f32(&data, count);
    drop(data);
    staging.unmap();
    result
}

/// Browser read-back: register the map and `.await` its completion.
///
/// `Device::poll` is a **no-op** on the WebGPU backend — it returns
/// `Ok(QueueEmpty)` without touching the queue (`wgpu/src/backend/webgpu.rs`,
/// `WebDevice::poll`) — so [`read_back_blocking`]'s `poll(Wait)` +
/// `recv_timeout` pair would simply hang the page's only thread forever. What
/// actually completes a WebGPU map is the JS `GPUBuffer.mapAsync` promise, and
/// the only way to observe it is to yield to the event loop. Hence: no poll, no
/// channel, no timeout — one future whose waker is armed from the map callback
/// (which wgpu invokes from the promise's `then`).
///
/// Registering `map_async` immediately after `queue.submit` is the same thing
/// `CommandEncoder::map_buffer_on_submit` does — that API just defers the very
/// same `buffer.map_async` call to submit time (`wgpu`'s
/// `DeferredCommandBufferActions::execute`, `api/command_buffer_actions.rs:33`)
/// — so the kernels here call it directly and keep one code path for both
/// targets.
#[cfg(target_arch = "wasm32")]
async fn read_back_web(staging: &wgpu::Buffer, count: usize, bytes: u64) -> Option<Vec<f32>> {
    let slice = staging.slice(0..bytes);
    let signal = map_signal::MapSignal::new();
    {
        let signal = signal.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| signal.complete(result));
    }
    // A mapping failure is a decline, not a degradation: the device is still
    // perfectly usable, this one buffer just never became readable.
    signal.await.ok()?;

    let data = slice.get_mapped_range();
    let result = decode_f32(&data, count);
    drop(data);
    staging.unmap();
    result
}

/// The one-shot future [`read_back_web`] waits on.
///
/// Hand-rolled rather than pulled from `futures`: it needs no dependency, and
/// `wasm32-unknown-unknown` lets the map callback be `!Send` (wgpu bounds it by
/// `WasmNotSend`, which is an empty bound there), so a plain `Rc` is enough.
///
/// Both cells are `Cell`, not `RefCell`: `Cell::take`/`Cell::set` cannot panic,
/// so there is no borrow-conflict failure mode to reason about even if wgpu
/// were to invoke the callback re-entrantly from inside `map_async`.
#[cfg(target_arch = "wasm32")]
mod map_signal {
    use core::cell::Cell;
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll, Waker};
    use std::rc::Rc;

    type MapResult = Result<(), wgpu::BufferAsyncError>;

    #[derive(Default)]
    struct Slot {
        result: Cell<Option<MapResult>>,
        waker: Cell<Option<Waker>>,
    }

    /// Cloneable handle to one pending `map_async`.
    pub(super) struct MapSignal(Rc<Slot>);

    impl Clone for MapSignal {
        fn clone(&self) -> Self {
            Self(Rc::clone(&self.0))
        }
    }

    impl MapSignal {
        pub(super) fn new() -> Self {
            Self(Rc::new(Slot::default()))
        }

        /// Called from wgpu's map callback: record the outcome and wake the
        /// awaiting task. Waking after the store is what makes the `poll`
        /// below observe a result rather than re-arming its waker forever.
        pub(super) fn complete(&self, result: MapResult) {
            self.0.result.set(Some(result));
            if let Some(waker) = self.0.waker.take() {
                waker.wake();
            }
        }
    }

    impl Future for MapSignal {
        type Output = MapResult;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if let Some(result) = self.0.result.take() {
                return Poll::Ready(result);
            }
            self.0.waker.set(Some(cx.waker().clone()));
            // Re-check: the callback may have fired between the `take` above
            // and the waker being stored, in which case nothing will wake us.
            match self.0.result.take() {
                Some(result) => Poll::Ready(result),
                None => Poll::Pending,
            }
        }
    }
}

/// Read a mapped staging buffer back into a `Vec<f32>`.
///
/// The single read-back every kernel in this crate ends in. Native blocks
/// ([`read_back_blocking`], unchanged); wasm32 awaits the `mapAsync` promise
/// ([`read_back_web`]).
///
/// Exactly `count` f32s are mapped, never the whole buffer: on the WebGPU
/// backend `get_mapped_range` materializes its range as a `Vec<u8>` in wasm
/// linear memory, so mapping a staging buffer larger than the result would copy
/// the slack too. A range wider than the buffer is a decline rather than a
/// panic — `Buffer::slice` panics on an out-of-range range, and this crate does
/// not panic on any input.
pub(crate) async fn read_back_async(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    count: usize,
) -> Option<Vec<f32>> {
    let bytes = f32_bytes(count)?;
    if bytes == 0 {
        return Some(Vec::new());
    }
    if bytes > staging.size() {
        return None;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        read_back_blocking(ctx, staging, count, bytes)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ctx;
        read_back_web(staging, count, bytes).await
    }
}

/// Read `count` f32s back from `staging`, then hand `output` to the context's
/// buffer pool — but **only** if the read-back actually completed.
///
/// [a7-10] Every kernel used to recycle its output buffer unconditionally,
/// right after `queue.submit`. On the failure paths of [`read_back_async`] the
/// submission may still be executing (the poll timed out, or the device was
/// lost mid-flight), so returning the buffer there lets a later dispatch pull
/// it out of the pool and bind it while the GPU is still writing to it. When
/// the read-back fails the buffer is dropped instead — which now *destroys* it
/// rather than merely releasing the handle. That is still safe with a
/// submission in flight: both backends defer the underlying free until the last
/// submission referencing the buffer retires (`wgpu-core`'s `Buffer::destroy`
/// schedules a `TempResource` against that submission index; WebGPU specifies
/// the same). What it is not safe to do is hand it to another *dispatch*, which
/// is why the a7-10 rule is unchanged.
///
/// A poisoned pool mutex only costs the recycling, never the result — the old
/// `ctx.pool.lock().ok()?` discarded a perfectly good tensor in that case.
///
/// The pool lock is taken **after** the await and released before returning, so
/// no `MutexGuard` is ever held across a suspension point.
pub(crate) async fn read_back_and_recycle_async(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    count: usize,
    output: TrackedBuffer,
) -> Option<Vec<f32>> {
    let result = read_back_async(ctx, staging, count).await;
    if result.is_some() {
        if let Ok(mut pool) = ctx.pool.lock() {
            pool.return_buffer(output);
        }
    }
    result
}

/// Finish a dispatch, either by reading its output back or by handing the
/// output buffer to the caller as a run-scoped activation.
///
/// The single tail every residency-aware kernel ends in, so "what happens after
/// submit" is written once rather than once per kernel:
///
/// * [`OutputPlacement::Host`] — the pre-residency behaviour exactly, including
///   the a7-10 rule that the output buffer is recycled *only* when the read-back
///   completed. `staging` must be the buffer the kernel copied into; a `None`
///   here is a kernel that allocated no staging for a host result, which is a
///   decline rather than a panic.
/// * [`OutputPlacement::Device`] — no map, no copy, no wait. The output buffer
///   becomes a [`DeviceTensor`] the caller owns and destroys at the activation's
///   last consumer.
///
/// # There is no fence in the `Device` arm, and that is correct
///
/// A host result implies a `map_async` that cannot complete until the
/// submission retires, so the pre-residency path always ended with the GPU
/// caught up. Keeping a result on the device ends with work still in flight —
/// which is safe for every consumer of the value, because WebGPU orders one
/// queue's submissions and inserts the buffer barriers between them, so the
/// next dispatch's reads are ordered after this dispatch's writes. It is *not*
/// safe to assume anything about elapsed time: a per-node duration measured
/// around this arm measures encode-and-submit, not execution.
pub(crate) async fn finish_output_async(
    ctx: &GpuContext,
    placement: OutputPlacement,
    staging: Option<TrackedBuffer>,
    output: TrackedBuffer,
    count: usize,
    bytes: u64,
    shape: Vec<usize>,
) -> Option<GpuOutput> {
    match placement {
        OutputPlacement::Host => {
            let staging = staging?;
            let data = read_back_and_recycle_async(ctx, &staging, count, output).await?;
            Some(GpuOutput::Host(Tensor::new(data, shape)))
        }
        OutputPlacement::Device => Some(GpuOutput::Device(DeviceTensor::new(
            output, shape, count, bytes,
        ))),
    }
}

/// Read a run-scoped activation back into host memory.
///
/// The lazy half of the residency contract: a value that stayed on the device
/// because its producer's consumers looked GPU-bound, and then met a consumer
/// that declined. One encoder, one copy, one submit, one map — the same
/// sequence a kernel's own read-back performs, minus the compute pass.
///
/// The activation itself is untouched: `tensor` is borrowed, not consumed, so a
/// later GPU consumer can still bind it in place. Callers memoize the result so
/// this happens at most once per tensor per run.
pub async fn read_device_tensor_async(ctx: &GpuContext, tensor: &DeviceTensor) -> Option<Tensor> {
    if ctx.is_degraded() {
        return None;
    }
    if tensor.is_empty() {
        return Some(Tensor::new(Vec::new(), tensor.shape().to_vec()));
    }
    let bytes = tensor.byte_len();
    if !ctx.budget_admits(&[bytes]) {
        return None;
    }
    let scope = ErrorScope::begin(ctx);
    let staging = ctx.staging_buffer("activation_readback_staging", bytes)?;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("activation_readback_enc"),
        });
    encoder.copy_buffer_to_buffer(tensor.buffer(), 0, &staging, 0, bytes);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    if !scope.finish_async(ctx).await {
        return None;
    }
    let data = read_back_async(ctx, &staging, tensor.len()).await?;
    Some(Tensor::new(data, tensor.shape().to_vec()))
}

/// Blocking form of [`read_back_and_recycle_async`], for kernels that are still
/// written synchronously.
///
/// Declines on wasm32 without submitting anything further: the future is
/// dropped unpolled, so `output` is destroyed rather than recycled (the
/// underlying free still waits for its submission to retire) and the caller
/// falls back to the CPU operator.
///
/// **A kernel that still calls this cannot run in a browser.** See
/// [`ErrorScope::finish`] for the full warning and the one-line conversion a
/// kernel needs before it is wired into `try_gpu_dispatch_async`.
///
/// [R3b] Unused since the K2 kernel batch's conversion to the async contract
/// removed its last caller — see [`ErrorScope::finish`]'s `[R3b]` note for
/// why this is suppressed here rather than deleted.
#[allow(dead_code)]
pub(crate) fn read_back_and_recycle(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    count: usize,
    output: TrackedBuffer,
) -> Option<Vec<f32>> {
    block_on_gpu(read_back_and_recycle_async(ctx, staging, count, output))
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

    /// The mapped range handed back by the WebGPU backend carries no `f32`
    /// alignment guarantee, so the decoder must produce the same values from an
    /// aligned and a deliberately misaligned view of identical bytes — where
    /// `bytemuck::cast_slice` would have panicked on the second.
    #[test]
    fn decode_f32_agrees_on_aligned_and_misaligned_input() {
        let values = [1.0f32, -2.5, 3.25, 0.0, f32::MIN_POSITIVE];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let aligned = decode_f32(&bytes, values.len()).expect("aligned decode");
        assert_eq!(aligned, values);

        // Offset by one byte so the slice can no longer be a `&[f32]`.
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&bytes);
        let misaligned = decode_f32(&shifted[1..], values.len()).expect("misaligned decode");
        assert_eq!(misaligned, values);
    }

    #[test]
    fn decode_f32_declines_a_short_range_instead_of_panicking() {
        let bytes = [0u8; 7];
        assert!(
            decode_f32(&bytes, 2).is_none(),
            "7 bytes cannot hold 2 f32s"
        );
        assert!(decode_f32(&bytes, 1).is_some());
        assert_eq!(decode_f32(&bytes, 0), Some(Vec::new()));
        // A count whose byte size overflows `usize` declines rather than wraps.
        assert!(decode_f32(&bytes, usize::MAX).is_none());
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
