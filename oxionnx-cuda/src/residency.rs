//! Session-lifetime reuse of device memory: a scratch-buffer pool and a
//! weight-residency cache, both owned by the [`CudaContext`](crate::CudaContext).
//!
//! # The cost this removes
//!
//! Every op in this crate used to build its device memory from scratch on
//! *every* dispatch: `DeviceBuffer::alloc` for each operand and the output, a
//! host upload for each operand, the kernel, a fence, a readback, and then
//! `cuMemFree` for all of them as the buffers dropped at the end of the
//! function. Nothing survived the call. For a train-once workload that is
//! invisible; for the workload this crate exists for — three ONNX graphs run
//! once per video frame, for hundreds of frames, with identical shapes and
//! identical weights every time — it is most of the cost:
//!
//! * **`cuMemAlloc`/`cuMemFree` are not cheap and are not asynchronous.** They
//!   serialise against the device and against each other. A graph with 150
//!   nodes pays several hundred of them per frame to hand back the same size
//!   classes it asked for on the previous frame.
//! * **Weight uploads are pure waste after the first frame.** A graph
//!   initializer holds the same bytes on frame 1 and frame 10 000. ArcFace's
//!   embedding head alone is a 49 MiB matrix; InSwapper's twelve AdaIN
//!   projections are 4 MiB each. Re-uploading them every frame is tens of
//!   milliseconds of bus time buying nothing.
//! * **`DeviceBuffer::copy_from_host` ends in a full `cuCtxSynchronize`** (see
//!   its own doc comment for why: it makes the "pageable upload has landed"
//!   postcondition hold against a non-blocking stream). Correct, but it is a
//!   *context*-wide fence paid per operand. Uploading onto the stream the
//!   kernel will run on gets the same ordering guarantee from stream order
//!   alone, with no fence at all.
//!
//! `DevicePool` closes the first, `ResidentWeights` the second, and
//! `PooledBuffer::upload` the third.
//!
//! # This is the CUDA analogue of what the wgpu backend already does
//!
//! `oxionnx`'s own wgpu path solved exactly this problem in
//! `oxionnx_gpu::context::resident` (weights uploaded once per session and
//! bound from a device-side cache) and `oxionnx::session::gpu_activations`
//! (a GPU-produced value staying on the device for its next GPU consumer).
//! This module is the first of those two, for CUDA, and deliberately mirrors
//! its shape — including the part that matters most: a cached identity is
//! **checked, not trusted** (see `ResidentWeights::acquire`).
//!
//! The activation half is *not* here, and that is a scope statement rather
//! than an omission: `try_cuda_dispatch` returns a host [`Tensor`](oxionnx_core::Tensor)
//! per node because its caller (`oxionnx`'s sequential/parallel runners) hands
//! it host tensors and expects host tensors back. Keeping an activation on the
//! device across a node boundary requires the *session* to own the
//! name→buffer map and the last-use schedule, exactly as `gpu_activations`
//! does for wgpu; there is nothing this crate can do about it from inside a
//! single dispatch. What this module does buy on the activation side is that
//! the buffers those round trips land in are recycled rather than
//! reallocated.
//!
//! # Why the caches live on the context
//!
//! One [`CudaContext`](crate::CudaContext) is built per `oxionnx::Session`, so
//! context-scoped is session-scoped — which is the exact lifetime over which a
//! graph initializer's bytes are invariant. It also puts every buffer's `Drop`
//! (a `cuMemFree` against this context) inside the lifetime of the
//! `Arc<Context>` the same struct holds, so the frees cannot outlive the
//! context they belong to.
//!
//! # Thread safety
//!
//! `oxionnx::Session` is `Send + Sync` (a hard, asserted requirement — see its
//! auto-trait invariant), and `oxionnx`'s parallel runner hands `&Session`,
//! and therefore `&CudaContext`, to rayon workers. Both caches are therefore
//! `Mutex`-guarded, and every lock is released before any driver call that
//! could block. A poisoned lock is never fatal here: it degrades to the
//! pre-cache behaviour (allocate fresh, upload every time), because nothing
//! about *correctness* depends on a cache hit.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use oxicuda_driver::{ffi::CUdeviceptr, Stream};
use oxicuda_memory::DeviceBuffer;

use crate::error::CudaDispatchError;

// ─── size classes ──────────────────────────────────────────────────────────

/// Smallest pooled allocation, in `f32` elements (1 KiB).
///
/// Below this the allocation is rounding error but the `cuMemAlloc` is not, so
/// every request under it shares one class rather than fragmenting the pool
/// into dozens of near-empty ones.
const MIN_CLASS_ELEMENTS: usize = 256;

/// Total device memory the pool may hold *while idle* — i.e. summed over
/// buffers that are checked in and available for reuse.
///
/// Buffers currently lent out are not counted (they are live tensors, not
/// cache), so this bounds the cache, not the workload. 512 MiB is comfortably
/// above the working set of the graphs this crate targets — InSwapper's
/// largest activation is ~33 MiB and its largest weight ~4 MiB — while staying
/// a small fraction of a mid-range card's memory.
///
/// A checked-in buffer that would push the pool past this is freed instead of
/// retained. That is the *only* eviction rule: no LRU, no timers. The pool is
/// a free-list for a workload that asks for the same handful of size classes
/// every frame, and for that workload the bound is never reached.
const POOL_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Buffers retained per size class.
///
/// A single dispatch borrows at most four (two operands, an output, a bias),
/// and `oxionnx`'s parallel runner can have several dispatches in flight, so
/// eight leaves headroom without letting one class monopolise the budget.
const MAX_PER_CLASS: usize = 8;

/// The pooled capacity that serves a request for `len` elements.
///
/// Powers of two above [`MIN_CLASS_ELEMENTS`]: a request is served by an
/// allocation at most 2x its size, which bounds the pool's internal waste
/// while keeping the number of distinct classes logarithmic in the largest
/// tensor. The same convention `oxionnx-gpu`'s buffer pool uses, for the same
/// reason.
///
/// Every consumer of a pooled buffer must therefore treat the allocation as
/// *at least* the requested length and never as exactly it — see
/// [`PooledBuffer`].
#[must_use]
fn size_class(len: usize) -> usize {
    len.max(MIN_CLASS_ELEMENTS).next_power_of_two()
}

// ─── counters ──────────────────────────────────────────────────────────────

/// Cumulative, monotonic counters describing what one context's caches have
/// done.
///
/// Monotonic deliberately: "what did this frame upload" is the difference
/// between two snapshots, and a counter that could fall would make that
/// difference meaningless. [`Self::weight_bytes_uploaded`] is the number the
/// whole residency claim rests on — once every initializer has been seen once,
/// it must stop growing no matter how many more frames run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheCounters {
    /// Scratch requests served by a buffer the pool already held.
    pub pool_hits: u64,
    /// Scratch requests that had to `cuMemAlloc`.
    pub pool_allocs: u64,
    /// Buffers freed on check-in because the pool was already at its budget
    /// or class limit.
    pub pool_evictions: u64,
    /// Weight lookups served by a device copy already resident.
    pub weight_hits: u64,
    /// Weight lookups that had to upload — a first sighting, or a conflict.
    pub weight_misses: u64,
    /// Bytes those misses handed to the driver.
    pub weight_bytes_uploaded: u64,
    /// **Every** byte this context has pushed host → device, weights and
    /// activations alike.
    ///
    /// [`Self::weight_bytes_uploaded`] is the invariant-bytes subset; the
    /// difference between the two is what this frame's activations cost.
    /// Counted at the copy itself rather than inferred from shapes, so it
    /// cannot drift from what the driver was actually asked to move.
    pub host_to_device_bytes: u64,
    /// Every byte this context has pulled device → host.
    ///
    /// The number activation residency exists to drive to "graph outputs
    /// only": a claimed node that keeps its result on the device adds nothing
    /// here.
    pub device_to_host_bytes: u64,
    /// Blocking `stream.synchronize()` calls this context has performed on a
    /// dispatch path.
    ///
    /// One per claimed node before residency (237/frame across the three
    /// face-pipeline models); one per *host-visible* result after it.
    pub stream_syncs: u64,
    /// Operands bound straight from a device-resident activation — uploads
    /// that did not happen.
    pub resident_activation_binds: u64,
    /// Node outputs handed back as device buffers rather than read back.
    pub device_handoffs: u64,
    /// Activation buffers returned to the scratch pool by the session's
    /// activation map.
    pub activation_recycles: u64,
}

impl CacheCounters {
    /// The activity between the `earlier` snapshot and this one.
    ///
    /// Saturating, so a snapshot taken against a different context yields zero
    /// rather than a wrapped number that would read as an enormous upload.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            pool_hits: self.pool_hits.saturating_sub(earlier.pool_hits),
            pool_allocs: self.pool_allocs.saturating_sub(earlier.pool_allocs),
            pool_evictions: self.pool_evictions.saturating_sub(earlier.pool_evictions),
            weight_hits: self.weight_hits.saturating_sub(earlier.weight_hits),
            weight_misses: self.weight_misses.saturating_sub(earlier.weight_misses),
            weight_bytes_uploaded: self
                .weight_bytes_uploaded
                .saturating_sub(earlier.weight_bytes_uploaded),
            host_to_device_bytes: self
                .host_to_device_bytes
                .saturating_sub(earlier.host_to_device_bytes),
            device_to_host_bytes: self
                .device_to_host_bytes
                .saturating_sub(earlier.device_to_host_bytes),
            stream_syncs: self.stream_syncs.saturating_sub(earlier.stream_syncs),
            resident_activation_binds: self
                .resident_activation_binds
                .saturating_sub(earlier.resident_activation_binds),
            device_handoffs: self.device_handoffs.saturating_sub(earlier.device_handoffs),
            activation_recycles: self
                .activation_recycles
                .saturating_sub(earlier.activation_recycles),
        }
    }

    /// Whether nothing happened at all.
    #[must_use]
    pub fn is_idle(self) -> bool {
        self == Self::default()
    }
}

/// The atomic backing for [`CacheCounters`], shared by both caches so a caller
/// gets one consistent snapshot rather than two that drifted apart.
#[derive(Default)]
struct Counters {
    pool_hits: AtomicU64,
    pool_allocs: AtomicU64,
    pool_evictions: AtomicU64,
    weight_hits: AtomicU64,
    weight_misses: AtomicU64,
    weight_bytes_uploaded: AtomicU64,
    host_to_device_bytes: AtomicU64,
    device_to_host_bytes: AtomicU64,
    stream_syncs: AtomicU64,
    resident_activation_binds: AtomicU64,
    device_handoffs: AtomicU64,
    activation_recycles: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> CacheCounters {
        CacheCounters {
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            pool_allocs: self.pool_allocs.load(Ordering::Relaxed),
            pool_evictions: self.pool_evictions.load(Ordering::Relaxed),
            weight_hits: self.weight_hits.load(Ordering::Relaxed),
            weight_misses: self.weight_misses.load(Ordering::Relaxed),
            weight_bytes_uploaded: self.weight_bytes_uploaded.load(Ordering::Relaxed),
            host_to_device_bytes: self.host_to_device_bytes.load(Ordering::Relaxed),
            device_to_host_bytes: self.device_to_host_bytes.load(Ordering::Relaxed),
            stream_syncs: self.stream_syncs.load(Ordering::Relaxed),
            resident_activation_binds: self.resident_activation_binds.load(Ordering::Relaxed),
            device_handoffs: self.device_handoffs.load(Ordering::Relaxed),
            activation_recycles: self.activation_recycles.load(Ordering::Relaxed),
        }
    }
}

// ─── the scratch pool ──────────────────────────────────────────────────────

/// A free-list of `f32` device buffers, keyed by [`size_class`].
///
/// Buffers are lent out by [`Self::acquire`] and returned automatically when
/// the [`PooledBuffer`] guard drops, so a caller cannot forget to check one
/// back in, and an early `?` return on a driver error checks it back in too.
pub(crate) struct DevicePool {
    /// class capacity (elements) → buffers of exactly that capacity.
    free: Mutex<HashMap<usize, Vec<DeviceBuffer<f32>>>>,
    /// Bytes currently held in `free`. Tracked rather than recomputed so
    /// check-in stays O(1).
    idle_bytes: Mutex<usize>,
    counters: Arc<Counters>,
}

impl DevicePool {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            free: Mutex::new(HashMap::new()),
            idle_bytes: Mutex::new(0),
            counters,
        }
    }

    /// Borrow a buffer of **at least** `len` elements.
    ///
    /// The contents are *undefined*: whatever the previous borrower left, or
    /// whatever `cuMemAlloc` handed back. Every caller either overwrites the
    /// whole logical range with an upload or zeroes it with
    /// [`PooledBuffer::zero_fill`] — see [`PooledBuffer`]'s docs for why that
    /// obligation is not optional.
    ///
    /// # Errors
    ///
    /// Propagates the driver's error when a fresh allocation is needed and
    /// fails (typically out of memory).
    fn acquire(&self, len: usize) -> Result<PooledBuffer<'_>, CudaDispatchError> {
        let class = size_class(len);

        let reused = match self.free.lock() {
            Ok(mut free) => free.get_mut(&class).and_then(Vec::pop),
            // A poisoned lock costs reuse, not correctness: allocate fresh.
            Err(_) => None,
        };

        let buffer = match reused {
            Some(buffer) => {
                if let Ok(mut idle) = self.idle_bytes.lock() {
                    *idle = idle.saturating_sub(buffer.byte_size());
                }
                self.counters.pool_hits.fetch_add(1, Ordering::Relaxed);
                buffer
            }
            None => {
                self.counters.pool_allocs.fetch_add(1, Ordering::Relaxed);
                DeviceBuffer::<f32>::alloc(class)?
            }
        };

        Ok(PooledBuffer {
            buffer: Some(buffer),
            pool: self,
            len,
            in_flight: false,
        })
    }

    /// Take a buffer back, or free it if the pool is already full enough.
    fn release(&self, buffer: DeviceBuffer<f32>) {
        let bytes = buffer.byte_size();
        let class = buffer.len();

        let Ok(mut free) = self.free.lock() else {
            // Poisoned: drop the buffer (freeing it) rather than leaking it.
            self.counters.pool_evictions.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let Ok(mut idle) = self.idle_bytes.lock() else {
            self.counters.pool_evictions.fetch_add(1, Ordering::Relaxed);
            return;
        };

        let slot = free.entry(class).or_default();
        if slot.len() >= MAX_PER_CLASS || *idle + bytes > POOL_BUDGET_BYTES {
            self.counters.pool_evictions.fetch_add(1, Ordering::Relaxed);
            return;
        }
        slot.push(buffer);
        *idle += bytes;
    }

    /// Account a buffer dropped with work still in flight: freed rather than
    /// recycled. Counted as an eviction, so a dispatch path that stopped
    /// retiring its buffers shows up as a pool that never warms.
    fn note_unretired(&self) {
        self.counters.pool_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Free every idle buffer, returning the bytes released.
    ///
    /// Buffers currently lent out are untouched — they are live tensors, and
    /// the borrow checker guarantees none can be in the free list anyway.
    fn clear(&self) -> usize {
        match (self.free.lock(), self.idle_bytes.lock()) {
            (Ok(mut free), Ok(mut idle)) => {
                free.clear();
                std::mem::replace(&mut *idle, 0)
            }
            _ => 0,
        }
    }

    /// Bytes currently held idle in the pool.
    fn idle_bytes(&self) -> usize {
        self.idle_bytes.lock().map(|idle| *idle).unwrap_or(0)
    }
}

/// A device buffer borrowed from a [`DevicePool`], returned on drop.
///
/// # The allocation is bigger than the tensor, and that is load-bearing
///
/// The *logical length* is what the caller asked for; the underlying
/// [`DeviceBuffer`] is a whole [`size_class`], up to twice as large. Two
/// consequences every consumer has to respect:
///
/// * **Descriptors must be built from the logical length.** `MatrixDesc`,
///   `TensorDesc` and friends all validate "the buffer holds *at least* this
///   many elements", so passing the real dimensions is both correct and
///   necessary — a descriptor built from the buffer's capacity would describe
///   a matrix the caller never wrote.
/// * **The tail is stale.** It holds a previous dispatch's numbers. Reading
///   it back would be silent, plausible-looking garbage, which is why
///   [`Self::download`] refuses to copy more than the logical length.
///
/// # A borrow is only recycled once its stream work is known to be done
///
/// Every copy and memset this type issues is *asynchronous*: queued on a
/// stream, returning immediately. Recycling the allocation while one is still
/// pending would be a real data race, and not a theoretical one — this crate
/// uses **two** streams (`DnnHandle`'s own and its `BlasHandle`'s), so a
/// buffer released by a convolution and picked up by a GEMM changes stream,
/// and stream order stops protecting it.
///
/// On the happy path that cannot arise: every dispatch synchronises before it
/// returns, so by the time the guard drops there is nothing in flight. The gap
/// is the *error* path — a kernel launch failing after its operands were
/// uploaded — which drops the guards with copies still queued. So the guard
/// tracks it: [`Self::retire`] records "the stream has been synchronised", and
/// a guard dropped without it is **freed instead of pooled**. Correct either
/// way; the cost of getting it wrong is a lost allocation, never a corrupted
/// tensor.
///
/// Forgetting [`Self::retire`] on a success path is therefore fail-safe — it
/// costs reuse, not correctness — and it does not go unnoticed either:
/// `tests/batched_matmul_gpu.rs` asserts that a steady-state dispatch performs
/// **zero** allocations, which is exactly what would silently stop holding.
pub(crate) struct PooledBuffer<'p> {
    /// `None` only transiently inside `Drop`, between the `take` and the
    /// hand-off to the pool. Never observable through the public methods.
    buffer: Option<DeviceBuffer<f32>>,
    pool: &'p DevicePool,
    /// Elements the caller asked for; `<=` the allocation's capacity.
    len: usize,
    /// Whether asynchronous work has been queued against this allocation
    /// without the caller yet confirming it completed. See the type docs.
    in_flight: bool,
}

impl PooledBuffer<'_> {
    /// The underlying allocation, for descriptor construction.
    pub(crate) fn buffer(&self) -> &DeviceBuffer<f32> {
        self.buffer
            .as_ref()
            .expect("PooledBuffer used after its allocation was returned to the pool")
    }

    /// The underlying allocation, mutably — for the `*Mut` descriptor
    /// constructors, which take `&mut DeviceBuffer` to encode that the kernel
    /// will write through them.
    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer<f32> {
        self.buffer
            .as_mut()
            .expect("PooledBuffer used after its allocation was returned to the pool")
    }

    /// The raw device pointer, for kernel launch arguments.
    pub(crate) fn device_ptr(&self) -> CUdeviceptr {
        self.buffer().as_device_ptr()
    }

    /// Upload `data` onto `stream`, filling the logical range.
    ///
    /// # Stream ordering is the whole point
    ///
    /// This enqueues the copy on the same stream the consuming kernel will be
    /// launched on, so the kernel is ordered after it by stream semantics
    /// alone — no fence, no `cuCtxSynchronize`. That is what makes this
    /// cheaper than [`DeviceBuffer::copy_from_host`], which is correct against
    /// *any* stream precisely because it ends in a context-wide synchronise
    /// (see its own doc comment).
    ///
    /// The caller must therefore pass the stream its kernel will use, and must
    /// keep `data` alive until that stream has been synchronised — both of
    /// which every call site in this crate does, since `data` is borrowed from
    /// the caller's `Tensor` for the whole dispatch and the dispatch ends with
    /// a synchronise before it returns.
    ///
    /// # Errors
    ///
    /// [`CudaDispatchError::Shape`] if `data` is longer than this borrow's
    /// logical length (a caller bug: the buffer was acquired too small), or
    /// the driver's error from the copy.
    pub(crate) fn upload(
        &mut self,
        data: &[f32],
        stream: &Stream,
    ) -> Result<(), CudaDispatchError> {
        if data.len() > self.len {
            return Err(CudaDispatchError::Shape {
                op: "device_pool",
                msg: format!(
                    "upload of {} elements into a borrow of {}",
                    data.len(),
                    self.len
                ),
            });
        }
        // SAFETY: `view` borrows the allocation this `PooledBuffer` owns for
        // the duration of the copy and nothing else touches it meanwhile; its
        // length is `data.len()`, which the check above proved is within the
        // borrow's logical length and therefore within the allocation. A view
        // built by `from_raw` is non-owning, so its drop does not free the
        // allocation this buffer will return to the pool.
        let mut view = unsafe { DeviceBuffer::<f32>::from_raw(self.device_ptr(), data.len()) };
        self.in_flight = true;
        view.copy_from_host_async(data, stream)?;
        self.pool
            .counters
            .host_to_device_bytes
            .fetch_add(byte_len(data), Ordering::Relaxed);
        Ok(())
    }

    /// Read the logical range back into `out` from `stream`.
    ///
    /// Enqueues the copy on `stream` (so it is ordered after the kernel that
    /// produced the data) and does **not** wait: the caller synchronises once,
    /// after issuing every readback it needs, rather than once per buffer.
    ///
    /// # Errors
    ///
    /// [`CudaDispatchError::Shape`] if `out` is longer than the logical
    /// length, or the driver's error from the copy.
    pub(crate) fn download(
        &mut self,
        out: &mut [f32],
        stream: &Stream,
    ) -> Result<(), CudaDispatchError> {
        if out.len() > self.len {
            return Err(CudaDispatchError::Shape {
                op: "device_pool",
                msg: format!(
                    "readback of {} elements from a borrow of {}",
                    out.len(),
                    self.len
                ),
            });
        }
        // SAFETY: as in `upload` — a non-owning view of exactly `out.len()`
        // elements, which is within this borrow's logical length.
        let view = unsafe { DeviceBuffer::<f32>::from_raw(self.device_ptr(), out.len()) };
        self.in_flight = true;
        view.copy_to_host_async(out, stream)?;
        self.pool
            .counters
            .device_to_host_bytes
            .fetch_add(byte_len(out), Ordering::Relaxed);
        Ok(())
    }

    /// Zero the logical range on `stream`.
    ///
    /// # Why a pooled output buffer must be zeroed at all
    ///
    /// A GEMM with `beta = 0` still evaluates `alpha * A@B + beta * C`, and
    /// `0.0 * NaN` is `NaN`, not `0.0`. A freshly `cuMemAlloc`ed buffer holds
    /// arbitrary bit patterns, some of which are `NaN`. The pre-pool code
    /// avoided this by allocating outputs with `DeviceBuffer::zeroed`; this is
    /// the same guarantee for a recycled buffer, minus the context-wide fence
    /// that `zeroed` performs (`cuMemsetD32Async` is stream-ordered, so the
    /// kernel launched next on the same stream sees the zeros).
    ///
    /// It is deliberately **not** conditional on "was this allocation fresh":
    /// a recycled buffer holds finite numbers, so `0.0 * old` would be `0.0`
    /// and skipping the fill would *usually* work — which is exactly the kind
    /// of reasoning that turns into a heisenbug the first time a previous
    /// dispatch legitimately produces an infinity.
    ///
    /// # Errors
    ///
    /// The driver's error from the memset, including
    /// [`CudaError::NotSupported`](oxicuda_driver::CudaError::NotSupported) on
    /// a driver with no async memset entry point — in which case the caller
    /// falls back to the synchronous form.
    pub(crate) fn zero_fill(&mut self, stream: &Stream) -> Result<(), CudaDispatchError> {
        let ptr = self.device_ptr();
        self.in_flight = true;
        match oxicuda_driver::memory_info::memset_d32_async(ptr, 0, self.len, stream) {
            Ok(()) => Ok(()),
            // Ancient driver with no `cuMemsetD32Async`: the synchronous form
            // is a correctness-preserving fallback, and one extra fence on a
            // driver this old is not the interesting case.
            Err(oxicuda_driver::CudaError::NotSupported) => {
                stream.synchronize()?;
                oxicuda_driver::memory_info::memset_d32(ptr, 0, self.len)?;
                Ok(())
            }
            Err(e) => Err(CudaDispatchError::Driver(e)),
        }
    }

    /// Record that the stream this buffer's work was queued on has been
    /// synchronised, so the allocation is safe to recycle.
    ///
    /// Called by every dispatch immediately after its one `synchronize()`.
    /// See the type docs for what happens to a buffer dropped without it.
    pub(crate) fn retire(&mut self) {
        self.in_flight = false;
    }

    /// Take the allocation out of the pool for good, disarming this guard.
    ///
    /// The handover an activation needs: a [`PooledBuffer`] dies at the end of
    /// the dispatch that borrowed it, and an activation must outlive its
    /// producing node. After this the pool has no claim on the allocation at
    /// all — it comes back only through
    /// [`CudaContext::recycle_activation`](crate::CudaContext::recycle_activation),
    /// when the session's activation map releases the value.
    ///
    /// The `in_flight` flag is deliberately *not* consulted: the caller is
    /// handing the allocation to a consumer that will read it on the same
    /// queue, which is the ordering guarantee, so "has the stream been
    /// synchronised" is the wrong question here. See
    /// [`mod@crate::activation`]'s header.
    pub(crate) fn into_owned(mut self) -> DeviceBuffer<f32> {
        self.buffer
            .take()
            .expect("PooledBuffer used after its allocation was returned to the pool")
    }
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if self.in_flight {
            // Work may still be queued against this allocation (see the type
            // docs: only an error path reaches here). Freeing it is safe --
            // `cuMemFree` on a pointer with pending stream work is defined to
            // block until that work completes -- whereas handing it to the
            // next dispatch, which may run on the *other* stream, is not.
            self.pool.note_unretired();
            return;
        }
        self.pool.release(buffer);
    }
}

// ─── weight residency ──────────────────────────────────────────────────────

/// Which derived form of a host tensor's bytes a device copy holds.
///
/// A `Gemm` with `transB=1` uploads the *transpose* of its weight, which is a
/// different byte sequence from the weight itself — and the same initializer
/// can legitimately be consumed both ways by two different nodes of one graph.
/// Caching them under one key would serve one node the other's bytes, so the
/// form is part of the identity.
///
/// Two variants rather than a map: there are exactly two, so a fixed-size
/// array costs no allocation and no hashing per lookup. Mirrors how
/// `oxionnx_gpu::context::resident` keys its `f32`/`f16` copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandForm {
    /// The host tensor's bytes as they are.
    Raw,
    /// The last two dimensions transposed, per batch slice.
    Transposed,
}

impl OperandForm {
    /// Index into [`Entry::slots`].
    fn index(self) -> usize {
        match self {
            Self::Raw => 0,
            Self::Transposed => 1,
        }
    }
}

/// Number of [`OperandForm`] variants.
const FORMS: usize = 2;

/// The stable identity of an operand whose bytes do not change between
/// dispatches.
///
/// # What makes this safe to cache
///
/// The name alone is what `oxionnx`'s wgpu path keys on, and it is sound there
/// for a documented reason: `weights` is built once when the session loads and
/// never mutated, so a name denotes one byte sequence for the session's whole
/// life — which is also the [`CudaContext`](crate::CudaContext)'s life.
///
/// This crate's entry point is `pub`, though, and a caller outside `oxionnx`
/// could hand the same context two different weight maps that share a name. So
/// the identity carries two more fields that make an accidental violation
/// *detectable* rather than merely forbidden: the address and length of the
/// host allocation the bytes were read from. A caller who swaps in different
/// weights has, almost certainly, a different `Vec` behind them, and the
/// mismatch is caught as a conflict (upload transiently, change nothing)
/// instead of silently serving stale numbers.
///
/// "Almost certainly" is the honest strength of the check: an allocator can
/// hand a freed address straight back for a same-length `Vec`. It is a
/// backstop for a contract, not a substitute for one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WeightId<'a> {
    /// The graph-initializer name — unique within one session's graph.
    pub(crate) name: &'a str,
    /// Address of the host allocation the bytes came from.
    pub(crate) origin: usize,
    /// Element count of that host allocation.
    pub(crate) origin_len: usize,
    /// Which derived form of those bytes this identity denotes.
    pub(crate) form: OperandForm,
}

impl<'a> WeightId<'a> {
    /// The identity of `tensor`'s bytes in `form`, under `name`.
    pub(crate) fn new(name: &'a str, tensor_data: &[f32], form: OperandForm) -> Self {
        Self {
            name,
            origin: tensor_data.as_ptr() as usize,
            origin_len: tensor_data.len(),
            form,
        }
    }
}

/// One device copy the context keeps for its whole lifetime.
struct Slot {
    buffer: Arc<DeviceBuffer<f32>>,
    /// The kernel slot this was uploaded for (`"matmul_a"`, `"conv_weight"`,
    /// …). Two kernels asking for the same identity with different slot
    /// semantics must not share bytes.
    label: &'static str,
    /// Address of the host allocation, and its length — see [`WeightId`].
    origin: usize,
    origin_len: usize,
    /// Elements actually uploaded (`<= origin_len`: a dispatch may upload only
    /// the prefix its declared shape covers).
    uploaded: usize,
}

/// Everything held under one name: at most one copy per [`OperandForm`].
#[derive(Default)]
struct Entry {
    slots: [Option<Slot>; FORMS],
}

/// Device copies of invariant operands, held for the lifetime of one
/// [`CudaContext`](crate::CudaContext).
pub(crate) struct ResidentWeights {
    entries: Mutex<HashMap<String, Entry>>,
    counters: Arc<Counters>,
}

/// What a resident operand acquisition produced.
///
/// The two variants differ only in who frees the memory and when: a
/// [`Self::Resident`] copy lives until the context drops, a [`Self::Pooled`]
/// one goes back to the scratch pool at the end of this dispatch. Both hand
/// out the same `&DeviceBuffer<f32>`, so every call site treats them alike.
pub(crate) enum Operand<'p> {
    /// Served from (or just added to) the residency cache.
    Resident(Arc<DeviceBuffer<f32>>),
    /// Uploaded for this dispatch only — no identity was supplied, or the one
    /// supplied conflicted with what the cache already holds.
    Pooled(PooledBuffer<'p>),
}

impl Operand<'_> {
    /// The device allocation backing this operand.
    pub(crate) fn buffer(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Resident(buffer) => buffer,
            Self::Pooled(buffer) => buffer.buffer(),
        }
    }

    /// Record that this operand's stream has been synchronised.
    ///
    /// A no-op for a resident copy, and provably so rather than by omission:
    /// a resident buffer is never recycled, and its one upload is issued on --
    /// and every later read of it comes from -- the same stream, because a
    /// residency entry is keyed by kernel-slot label, one label belongs to one
    /// op family, and one op family uses one stream. Only *pooled* buffers can
    /// migrate between the two streams this crate uses, so only they need the
    /// bookkeeping. See [`PooledBuffer`].
    pub(crate) fn retire(&mut self) {
        if let Self::Pooled(buffer) = self {
            buffer.retire();
        }
    }

    /// The raw device pointer, for kernel launch arguments.
    pub(crate) fn device_ptr(&self) -> CUdeviceptr {
        match self {
            Self::Resident(buffer) => buffer.as_device_ptr(),
            Self::Pooled(buffer) => buffer.device_ptr(),
        }
    }
}

impl ResidentWeights {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            counters,
        }
    }

    /// Hand back a device copy of `data`, reusing a resident one when `id`
    /// names bytes this context has already uploaded.
    ///
    /// `label` names the kernel slot; two dispatches that would use the same
    /// identity for different slot semantics conflict rather than sharing.
    ///
    /// # The four outcomes
    ///
    /// 1. `id` is `None` — the bytes are not invariant (an activation).
    ///    Upload into a pooled buffer for this dispatch alone.
    /// 2. `id` is known, and label/origin/length all agree — a **hit**: no
    ///    upload at all, which is the entire point of the cache.
    /// 3. `id` is new — upload once into a dedicated allocation and keep it.
    /// 4. `id` is known but something disagrees — a **conflict**. Upload
    ///    transiently and leave the existing entry alone. Overwriting would be
    ///    the worse failure: two nodes fighting over one key would re-upload
    ///    every frame *and* invalidate each other, while a hit-rate metric
    ///    reported progress.
    ///
    /// # Errors
    ///
    /// Propagates allocation and upload failures.
    fn acquire<'p>(
        &self,
        pool: &'p DevicePool,
        id: Option<WeightId<'_>>,
        label: &'static str,
        data: &[f32],
        stream: &Stream,
    ) -> Result<Operand<'p>, CudaDispatchError> {
        let Some(id) = id else {
            return Ok(Operand::Pooled(self.transient(pool, data, stream)?));
        };

        // 1. Fast path: is it already here, and does it still describe these
        //    exact bytes?
        {
            let Ok(entries) = self.entries.lock() else {
                // Poisoned lock: nothing about correctness depends on a hit.
                return Ok(Operand::Pooled(self.transient(pool, data, stream)?));
            };
            match entries
                .get(id.name)
                .and_then(|entry| entry.slots[id.form.index()].as_ref())
            {
                Some(slot)
                    if slot.label == label
                        && slot.origin == id.origin
                        && slot.origin_len == id.origin_len
                        && slot.uploaded == data.len() =>
                {
                    self.counters.weight_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(Operand::Resident(Arc::clone(&slot.buffer)));
                }
                // Occupied by something this request does not match: upload
                // for this dispatch only and leave the entry untouched.
                Some(_) => {
                    self.counters.weight_misses.fetch_add(1, Ordering::Relaxed);
                    self.counters
                        .weight_bytes_uploaded
                        .fetch_add(byte_len(data), Ordering::Relaxed);
                    return Ok(Operand::Pooled(self.transient(pool, data, stream)?));
                }
                None => {}
            }
        }

        // 2. First sighting: a dedicated, exactly-sized allocation rather than
        //    a pooled one. A resident buffer is never recycled, so paying the
        //    size class's rounding for its whole lifetime would be waste, and
        //    an exact length lets `MatrixDesc::from_buffer` describe it without
        //    any capacity/length distinction to get wrong.
        if data.is_empty() {
            // `DeviceBuffer::alloc(0)` is an error, and an empty operand has
            // nothing to cache; let the transient path reject it uniformly.
            return Ok(Operand::Pooled(self.transient(pool, data, stream)?));
        }
        let mut buffer = DeviceBuffer::<f32>::alloc(data.len())?;
        buffer.copy_from_host_async(data, stream)?;
        let shared = Arc::new(buffer);

        self.counters.weight_misses.fetch_add(1, Ordering::Relaxed);
        self.counters
            .weight_bytes_uploaded
            .fetch_add(byte_len(data), Ordering::Relaxed);
        // This copy does not go through `PooledBuffer::upload` (a resident
        // allocation is exact-sized and never pooled), so the bus counter has
        // to be bumped here or a session's first frame would under-report its
        // uploads by the whole weight set.
        self.counters
            .host_to_device_bytes
            .fetch_add(byte_len(data), Ordering::Relaxed);

        if let Ok(mut entries) = self.entries.lock() {
            let entry = entries.entry(id.name.to_string()).or_default();
            // Re-check rather than assume the slot is still vacant: the lock
            // was released between the lookup above and here, so a concurrent
            // dispatch may have filled it. Whichever insert lands last wins;
            // both `Arc`s stay valid for as long as anyone holds them.
            entry.slots[id.form.index()] = Some(Slot {
                buffer: Arc::clone(&shared),
                label,
                origin: id.origin,
                origin_len: id.origin_len,
                uploaded: data.len(),
            });
        }
        Ok(Operand::Resident(shared))
    }

    /// Look an identity up **without** the host bytes.
    ///
    /// The point is what the caller is spared. Building the bytes for a
    /// cacheable operand is not always free: a `transB=1` weight has to be
    /// transposed into a fresh host buffer before it could be uploaded, and
    /// that transpose is `O(k*n)` of pure waste on every frame after the first
    /// if the device already holds its result. Asking first turns it into
    /// `O(1)`.
    ///
    /// Returns `None` for a miss *and* for a conflict — the caller then builds
    /// the bytes and goes through [`Self::acquire`], which distinguishes the
    /// two and counts them. A hit is counted here, exactly once, because a hit
    /// here means the caller does not call `acquire` at all.
    fn resident(
        &self,
        id: WeightId<'_>,
        label: &'static str,
        uploaded: usize,
    ) -> Option<Arc<DeviceBuffer<f32>>> {
        let entries = self.entries.lock().ok()?;
        let slot = entries
            .get(id.name)
            .and_then(|entry| entry.slots[id.form.index()].as_ref())?;
        if slot.label == label
            && slot.origin == id.origin
            && slot.origin_len == id.origin_len
            && slot.uploaded == uploaded
        {
            self.counters.weight_hits.fetch_add(1, Ordering::Relaxed);
            return Some(Arc::clone(&slot.buffer));
        }
        None
    }

    /// Upload `data` into a pooled buffer for this dispatch only.
    fn transient<'p>(
        &self,
        pool: &'p DevicePool,
        data: &[f32],
        stream: &Stream,
    ) -> Result<PooledBuffer<'p>, CudaDispatchError> {
        let mut buffer = pool.acquire(data.len())?;
        buffer.upload(data, stream)?;
        Ok(buffer)
    }

    /// Whether `name` has a device copy in any form — the "have these bytes
    /// stopped crossing the bus?" question.
    fn contains(&self, name: &str) -> bool {
        self.entries.lock().is_ok_and(|entries| {
            entries
                .get(name)
                .is_some_and(|entry| entry.slots.iter().any(Option::is_some))
        })
    }

    /// Device bytes currently pinned by resident weights.
    fn bytes(&self) -> usize {
        self.entries
            .lock()
            .map(|entries| {
                entries
                    .values()
                    .flat_map(|entry| entry.slots.iter().flatten())
                    .map(|slot| slot.buffer.byte_size())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Drop every resident copy, returning the bytes released.
    ///
    /// A copy an in-flight dispatch is still holding survives until that
    /// dispatch drops its `Arc`; the cache simply stops serving it.
    fn clear(&self) -> usize {
        match self.entries.lock() {
            Ok(mut entries) => {
                let released = entries
                    .values()
                    .flat_map(|entry| entry.slots.iter().flatten())
                    .map(|slot| slot.buffer.byte_size())
                    .sum();
                entries.clear();
                released
            }
            Err(_) => 0,
        }
    }
}

/// `data`'s size in bytes, as a `u64` for the counters.
fn byte_len(data: &[f32]) -> u64 {
    (data.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
}

// ─── the pair, as one object ───────────────────────────────────────────────

/// Both caches plus their shared counters — the unit a
/// [`CudaContext`](crate::CudaContext) holds.
pub(crate) struct DeviceCaches {
    pool: DevicePool,
    weights: ResidentWeights,
    counters: Arc<Counters>,
}

impl DeviceCaches {
    /// Empty caches. No device memory is touched until the first
    /// [`Self::scratch`] or [`Self::operand`] call.
    pub(crate) fn new() -> Self {
        let counters = Arc::new(Counters::default());
        Self {
            pool: DevicePool::new(Arc::clone(&counters)),
            weights: ResidentWeights::new(Arc::clone(&counters)),
            counters,
        }
    }

    /// Borrow an uninitialised scratch buffer of at least `len` elements.
    ///
    /// # Errors
    ///
    /// Propagates an allocation failure.
    pub(crate) fn scratch(&self, len: usize) -> Result<PooledBuffer<'_>, CudaDispatchError> {
        self.pool.acquire(len)
    }

    /// Upload `data` for this dispatch, reusing a resident copy when `id`
    /// names invariant bytes this context has seen before.
    ///
    /// # Errors
    ///
    /// Propagates allocation and upload failures.
    pub(crate) fn operand(
        &self,
        id: Option<WeightId<'_>>,
        label: &'static str,
        data: &[f32],
        stream: &Stream,
    ) -> Result<Operand<'_>, CudaDispatchError> {
        self.weights.acquire(&self.pool, id, label, data, stream)
    }

    /// A device copy the cache already holds for `id`, if any — resolved
    /// without the host bytes, so a caller that would have to *build* them can
    /// skip that work entirely.
    ///
    /// `uploaded` is the element count the caller would have uploaded; a
    /// resident copy of a different length is not a match.
    pub(crate) fn resident(
        &self,
        id: WeightId<'_>,
        label: &'static str,
        uploaded: usize,
    ) -> Option<Operand<'_>> {
        self.weights
            .resident(id, label, uploaded)
            .map(Operand::Resident)
    }

    /// A consistent snapshot of both caches' counters.
    pub(crate) fn counters(&self) -> CacheCounters {
        self.counters.snapshot()
    }

    /// Take an activation's allocation back into the scratch pool.
    ///
    /// The return leg of [`PooledBuffer::into_owned`]: the session's
    /// activation map calls it when a resident value's last consumer has run.
    /// Recycling rather than freeing is what keeps a steady-state frame at
    /// zero `cuMemAlloc`s — the very next node that needs an output buffer of
    /// that size class takes this one.
    ///
    /// No fence is taken, and none is needed on a unified queue: see
    /// [`mod@crate::activation`]'s header for the argument.
    pub(crate) fn recycle(&self, buffer: DeviceBuffer<f32>) {
        self.counters
            .activation_recycles
            .fetch_add(1, Ordering::Relaxed);
        self.pool.release(buffer);
    }

    /// Count a blocking fence taken on a dispatch path.
    pub(crate) fn note_sync(&self) {
        self.counters.stream_syncs.fetch_add(1, Ordering::Relaxed);
    }

    /// Count an operand bound straight from a device-resident activation.
    pub(crate) fn note_resident_bind(&self) {
        self.counters
            .resident_activation_binds
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Count a node output handed back as a device buffer.
    pub(crate) fn note_device_handoff(&self) {
        self.counters
            .device_handoffs
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Count a device → host copy issued outside a [`PooledBuffer`] — the
    /// read-back of a resident activation for a CPU consumer.
    pub(crate) fn note_download(&self, elements: usize) {
        self.counters.device_to_host_bytes.fetch_add(
            (elements as u64).saturating_mul(std::mem::size_of::<f32>() as u64),
            Ordering::Relaxed,
        );
    }

    /// Whether `name` is resident in any form.
    pub(crate) fn is_resident(&self, name: &str) -> bool {
        self.weights.contains(name)
    }

    /// Device bytes held: resident weights plus idle pooled buffers.
    pub(crate) fn bytes(&self) -> usize {
        self.weights.bytes() + self.pool.idle_bytes()
    }

    /// Release everything both caches hold, returning the bytes freed.
    pub(crate) fn clear(&self) -> usize {
        self.weights.clear() + self.pool.clear()
    }
}

// A floor under the pool's own constants, checked at compile time because it is
// a statement about constants rather than about behaviour. The pool must be able
// to hold several of the largest activations this workload produces (InSwapper's
// ~33 MiB feature maps) and enough buffers per class for one dispatch's four
// (two operands, an output, a bias) -- otherwise it would evict on every frame
// and never warm up, which is a silent performance cliff rather than a failure.
const _: () = {
    const LARGEST_ACTIVATION_BYTES: usize = 33 * 1024 * 1024;
    assert!(
        POOL_BUDGET_BYTES >= LARGEST_ACTIVATION_BYTES * 4,
        "the pool budget cannot hold four of the workload's largest activations",
    );
    assert!(
        MAX_PER_CLASS >= 4,
        "a single dispatch borrows up to four pooled buffers",
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise the pure, device-free logic: size classing, identity
    // construction and comparison, and counter arithmetic. Everything that
    // needs a real allocation is covered by the on-device suites (see
    // `tests/batched_matmul_gpu.rs` and this crate's `gpu-tests` module),
    // because `DeviceBuffer::alloc` cannot run on a host with no GPU.

    #[test]
    fn size_classes_never_return_less_than_requested() {
        for len in [
            1usize,
            2,
            255,
            256,
            257,
            1023,
            1024,
            4095,
            1 << 20,
            (1 << 20) + 1,
        ] {
            assert!(
                size_class(len) >= len,
                "class {} is smaller than the request {len}",
                size_class(len),
            );
        }
    }

    #[test]
    fn size_classes_waste_at_most_a_factor_of_two_above_the_floor() {
        for len in [256usize, 257, 1000, 65_537, 1 << 20] {
            let class = size_class(len);
            assert!(
                class < len * 2,
                "class {class} wastes more than 2x on a request of {len}",
            );
        }
    }

    #[test]
    fn small_requests_all_share_one_class() {
        // Every request below the floor must land in the same bucket, or the
        // pool fragments into a class per tiny tensor.
        let classes: Vec<usize> = [1usize, 7, 64, 255, 256]
            .iter()
            .map(|&l| size_class(l))
            .collect();
        assert!(
            classes.iter().all(|&c| c == MIN_CLASS_ELEMENTS),
            "small requests landed in different classes: {classes:?}",
        );
    }

    #[test]
    fn size_classes_are_powers_of_two_so_the_class_count_stays_logarithmic() {
        for len in [300usize, 5000, 100_000, 3 << 20] {
            assert!(size_class(len).is_power_of_two());
        }
    }

    #[test]
    fn an_identity_records_the_host_allocation_it_was_built_from() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let id = WeightId::new("w", &data, OperandForm::Raw);
        assert_eq!(id.name, "w");
        assert_eq!(id.origin, data.as_ptr() as usize);
        assert_eq!(id.origin_len, 3);
        assert_eq!(id.form, OperandForm::Raw);
    }

    #[test]
    fn the_two_operand_forms_occupy_different_slots() {
        assert_ne!(
            OperandForm::Raw.index(),
            OperandForm::Transposed.index(),
            "a transposed copy sharing the raw copy's slot would serve one node the other's bytes",
        );
        assert!(OperandForm::Raw.index() < FORMS);
        assert!(OperandForm::Transposed.index() < FORMS);
    }

    #[test]
    fn two_tensors_with_the_same_name_but_different_storage_are_different_identities() {
        // The check that turns "the caller promised these bytes never change"
        // from an unverified contract into a detectable conflict.
        let first = vec![1.0_f32; 8];
        let second = vec![2.0_f32; 8];
        let a = WeightId::new("w", &first, OperandForm::Raw);
        let b = WeightId::new("w", &second, OperandForm::Raw);
        assert_eq!(a.name, b.name);
        assert_ne!(
            a.origin, b.origin,
            "two live Vecs cannot share an address, so the origins must differ",
        );
    }

    #[test]
    fn counter_deltas_are_saturating_rather_than_wrapping() {
        let earlier = CacheCounters {
            weight_hits: 10,
            weight_bytes_uploaded: 1024,
            ..CacheCounters::default()
        };
        let later = CacheCounters {
            weight_hits: 25,
            weight_bytes_uploaded: 1024,
            ..CacheCounters::default()
        };
        let delta = later.since(earlier);
        assert_eq!(delta.weight_hits, 15);
        assert_eq!(
            delta.weight_bytes_uploaded, 0,
            "a frame that uploaded nothing must read as zero, which is the whole residency claim",
        );

        // A snapshot from a different (or reset) context must not wrap.
        let backwards = earlier.since(later);
        assert_eq!(backwards.weight_hits, 0);
        assert!(backwards.is_idle());
    }

    #[test]
    fn fresh_counters_are_idle() {
        assert!(CacheCounters::default().is_idle());
        assert!(!CacheCounters {
            pool_allocs: 1,
            ..CacheCounters::default()
        }
        .is_idle());
    }

    #[test]
    fn caches_start_empty_without_touching_a_device() {
        // Construction must be allocation-free on the device side, so a
        // session on a GPU-less host pays nothing for holding these.
        let caches = DeviceCaches::new();
        assert!(caches.counters().is_idle());
        assert_eq!(caches.bytes(), 0);
        assert!(!caches.is_resident("anything"));
        assert_eq!(caches.clear(), 0);
    }
}
