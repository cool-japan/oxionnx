//! Live-byte accounting for every GPU buffer this crate allocates, plus the
//! owning handle that keeps the accounting honest.
//!
//! # Why a buffer needs an owner at all
//!
//! Dropping a `wgpu::Buffer` frees nothing on the WebGPU backend: `WebBuffer`'s
//! `Drop` impl is empty (`wgpu-29.0.4`, `src/backend/webgpu.rs`), so the device
//! memory behind a dropped buffer is released only once the JS garbage
//! collector finalizes a `GPUBuffer` object it has no particular reason to
//! consider urgent. The only reliable release is `GPUBuffer.destroy()`, reached
//! through [`wgpu::Buffer::destroy`]. A dispatch loop that allocates a few
//! buffers per node and drops them therefore accumulates driver memory until
//! `createBuffer` starts failing — and a failed `createBuffer` is not
//! recoverable inside wgpu's web backend, it unwraps.
//!
//! Native backends do free on drop (`wgpu-core` reference-counts the raw
//! allocation), so on native this module changes *when* memory is returned, not
//! whether. `destroy` is defined there too: it snatches the raw buffer and
//! schedules its destruction against the last submission that used it
//! (`wgpu-core-29.0.4`, `resource.rs`'s `Buffer::destroy` →
//! `schedule_resource_destruction`), so destroying right after `queue.submit`
//! is safe on both targets.
//!
//! [`TrackedBuffer`] is that owner: it destroys on drop and releases its bytes
//! from the [`GpuMemoryBudget`] it was allocated against. Every buffer this
//! crate creates is one, which makes "allocated bytes are eventually released"
//! a property of the type rather than of each kernel's error paths.
//!
//! # Why a budget on top of that
//!
//! Destroying promptly bounds the *steady state*, but a single oversized graph
//! can still ask for more than the device has in one node. On the WebGPU
//! backend a rejected `createBuffer` becomes a wasm trap inside wgpu, which no
//! caller can catch — so the only defense is never to ask. Reserving bytes
//! before allocating turns "device out of memory" into this crate's ordinary
//! decline: the node runs on the CPU instead.

use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Default ceiling on the bytes this crate keeps allocated on the device at
/// once: 1.5 GiB.
///
/// Sized for the tightest target rather than the roomiest: a browser tab shares
/// one GPU process with every other tab, and a `GPUDevice` that has handed out
/// several gigabytes starts rejecting allocations *of any size* — a 4 KiB
/// uniform buffer fails exactly like a 100 MiB one, because what is exhausted
/// is the process budget, not the request. One node's working set in this crate
/// is a few operand buffers plus one output, so 1.5 GiB leaves room for the
/// largest graphs it can dispatch while staying far below the point where a
/// browser starts refusing.
pub const DEFAULT_LIVE_BYTE_BUDGET: u64 = 1536 << 20;

/// Byte accounting shared by a [`crate::GpuContext`] and its buffer pool.
///
/// Counts bytes that are *allocated on the device*, whether they are in use by
/// a dispatch or sitting idle in the pool. Both states cost the same driver
/// memory, so both must be visible to the check that decides whether the next
/// allocation is safe.
#[derive(Debug)]
pub struct GpuMemoryBudget {
    /// Bytes currently allocated (in use plus pooled).
    live: AtomicU64,
    /// Ceiling on `live`. Runtime-settable so a caller can tighten it for a
    /// constrained page without rebuilding the context.
    limit: AtomicU64,
}

impl GpuMemoryBudget {
    /// A budget that admits up to `limit` live bytes.
    #[must_use]
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            live: AtomicU64::new(0),
            limit: AtomicU64::new(limit),
        })
    }

    /// A budget that never declines — for a pool used outside a context.
    #[must_use]
    pub fn unlimited() -> Arc<Self> {
        Self::new(u64::MAX)
    }

    /// Bytes currently allocated on the device through this budget.
    #[must_use]
    pub fn live_bytes(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }

    /// The current ceiling.
    #[must_use]
    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Relaxed)
    }

    /// Move the ceiling. Already-live bytes are never revoked; a lowered limit
    /// takes effect on the next allocation.
    pub fn set_limit(&self, limit: u64) {
        self.limit.store(limit, Ordering::Relaxed);
    }

    /// Whether `bytes` more would still fit under the ceiling.
    ///
    /// Advisory: a kernel calls this once with its whole requirement so a node
    /// that cannot fit declines *before* allocating anything, rather than
    /// discovering it three buffers in. `TrackedBuffer::create` is the
    /// authority, and it re-checks atomically.
    #[must_use]
    pub fn admits(&self, bytes: u64) -> bool {
        self.live_bytes().saturating_add(bytes) <= self.limit()
    }

    /// Whether all of `byte_sizes` together would still fit.
    #[must_use]
    pub fn admits_all(&self, byte_sizes: &[u64]) -> bool {
        self.admits(
            byte_sizes
                .iter()
                .fold(0u64, |acc, &b| acc.saturating_add(b)),
        )
    }

    /// Claim `bytes`, or report that they do not fit.
    ///
    /// A compare-and-swap loop rather than `fetch_add` + compensating
    /// `fetch_sub`: the latter transiently over-reports, which would make a
    /// concurrent [`Self::admits`] decline a node that in fact fits.
    fn try_reserve(&self, bytes: u64) -> bool {
        let limit = self.limit();
        self.live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                let next = live.checked_add(bytes)?;
                (next <= limit).then_some(next)
            })
            .is_ok()
    }

    /// Give `bytes` back. Saturating: the count can never wrap below zero, even
    /// if a release were ever to arrive twice.
    fn release(&self, bytes: u64) {
        let _ = self
            .live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                Some(live.saturating_sub(bytes))
            });
    }
}

/// A `wgpu::Buffer` that destroys itself and releases its bytes when dropped.
///
/// Derefs to the buffer, so it binds, copies and maps exactly like one --
/// every kernel in this crate builds its bind group entries, its
/// `copy_buffer_to_buffer` calls and its mapped-range reads by calling
/// straight through this `Deref`, close to ninety call sites across
/// `shaders/` and `compute.rs`.
///
/// # `Deref` is not a safe way to obtain an owned `wgpu::Buffer`
///
/// `wgpu::Buffer` derives [`Clone`] (`wgpu-29.0.4`, `src/api/buffer.rs`), and
/// that clone is a cheap handle copy, not a new device allocation: both
/// handles name the *same* underlying resource. So `(*tracked).clone()` --
/// or any other spelling that derefs first, `tracked.deref().clone()`
/// included -- compiles cleanly and yields a second, perfectly ordinary
/// `wgpu::Buffer` this module has never heard of. Two things go wrong once
/// that clone exists:
///
/// * The [`GpuMemoryBudget`] this handle reserved against is never told
///   about it. Only the original `TrackedBuffer`'s [`Drop`] impl releases
///   bytes back to it -- the clone holds no budget reference at all -- so
///   the accounting this type exists to make reliable (see the module docs)
///   silently stops describing reality the moment the clone is made, not
///   when anything is next allocated.
/// * When the original *does* drop, [`Drop::drop`] calls
///   [`wgpu::Buffer::destroy`] on the shared resource -- which the clone
///   still points to. The clone is now a handle to a destroyed buffer.
///   Binding, copying or mapping it afterwards is a wgpu validation error,
///   caught by this crate's error scopes, which respond to *any* validation
///   error by marking the whole [`crate::GpuContext`] degraded. One escaped
///   clone therefore does not just misbehave on its own next use -- it pushes
///   every later node in the session onto the CPU fallback for good.
///
/// Cloning the `Deref` target is forbidden -- not "avoid where possible", an
/// invariant every caller must uphold, the same way "never hold two
/// `TrackedBuffer`s over one physical allocation" is. No caller in this crate
/// does it today.
///
/// ## Why nothing here stops it mechanically
///
/// `TrackedBuffer` deliberately does not implement [`Clone`] itself. That
/// blocks the *common* spelling, `tracked.clone()`: Rust tries the receiver's
/// own inherent and trait methods (by value, then `&`, then `&mut`) before
/// following `Deref`, so with no `Clone` impl here that lookup keeps going
/// and would, absent this note, eventually reach `wgpu::Buffer::clone`
/// through the `Deref` chain. It does **not** block `(*tracked).clone()`:
/// writing the deref explicitly makes the receiver expression's static type
/// `wgpu::Buffer` *before* method resolution ever starts, so the call goes
/// straight to `wgpu::Buffer::clone` with `TrackedBuffer` never consulted. No
/// inherent method, trait impl or attribute on this type can intercept a call
/// already resolved against a different, concrete type -- and that is
/// exactly what "derefs to the buffer" (this type's whole purpose, and the
/// reason the call sites above need no ceremony) has to mean.
///
/// The tempting fix -- implement [`Clone`] for `TrackedBuffer` itself, doing
/// a real device-side copy so `tracked.clone()` is safe -- is not the answer
/// either. It would still leave `(*tracked).clone()` reaching
/// `wgpu::Buffer::clone` unguarded, for the reason just given, while making
/// the spelling it *does* catch perform a fallible, budget-gated allocation
/// and a GPU copy behind a trait signature (`fn clone(&self) -> Self`) that
/// promises neither -- an infallible-looking call that would have to panic
/// or silently hand back a degenerate buffer on a budget decline. Removing
/// `Deref` instead would close the hole completely, at the cost of rewriting
/// every bind/copy/map call site above to go through an explicit accessor;
/// that is a real option for a future crate-wide pass, not one this type can
/// make unilaterally without breaking all of them today.
#[derive(Debug)]
pub struct TrackedBuffer {
    buffer: wgpu::Buffer,
    budget: Arc<GpuMemoryBudget>,
}

impl TrackedBuffer {
    /// Reserve `desc.size` bytes and create the buffer, or decline.
    ///
    /// `None` means the budget is full — a decline, never a device error, so
    /// the caller falls back to the CPU and the context stays usable.
    pub(crate) fn create(
        device: &wgpu::Device,
        budget: &Arc<GpuMemoryBudget>,
        desc: &wgpu::BufferDescriptor<'_>,
    ) -> Option<Self> {
        if !budget.try_reserve(desc.size) {
            return None;
        }
        Some(Self {
            buffer: device.create_buffer(desc),
            budget: Arc::clone(budget),
        })
    }

    /// Bytes this handle has reserved — the allocation wgpu made, which is what
    /// `Drop` releases.
    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        self.buffer.size()
    }
}

impl Deref for TrackedBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl Drop for TrackedBuffer {
    fn drop(&mut self) {
        // Safe at any point after `queue.submit`: both backends defer the
        // actual free until the submissions referencing the buffer retire (see
        // the module docs).
        self.buffer.destroy();
        self.budget.release(self.buffer.size());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arithmetic is the whole contract, and none of it needs a device.
    #[test]
    fn a_budget_admits_up_to_its_limit_and_no_further() {
        let budget = GpuMemoryBudget::new(1024);
        assert_eq!(budget.live_bytes(), 0);
        assert_eq!(budget.limit(), 1024);
        assert!(budget.admits(1024));
        assert!(!budget.admits(1025));

        assert!(budget.try_reserve(768));
        assert_eq!(budget.live_bytes(), 768);
        assert!(budget.admits(256));
        assert!(!budget.admits(257));
        // A refused reservation must not move the counter.
        assert!(!budget.try_reserve(257));
        assert_eq!(budget.live_bytes(), 768);

        budget.release(768);
        assert_eq!(budget.live_bytes(), 0);
        // Releasing more than is live saturates rather than wrapping.
        budget.release(4096);
        assert_eq!(budget.live_bytes(), 0);
    }

    #[test]
    fn admits_all_sums_without_overflowing() {
        let budget = GpuMemoryBudget::new(4096);
        assert!(budget.admits_all(&[1024, 1024, 2048]));
        assert!(!budget.admits_all(&[1024, 1024, 2049]));
        assert!(!budget.admits_all(&[u64::MAX, u64::MAX]));
        assert!(budget.admits_all(&[]));
    }

    #[test]
    fn a_reservation_that_would_overflow_u64_is_refused() {
        let budget = GpuMemoryBudget::unlimited();
        assert!(budget.try_reserve(1));
        assert!(!budget.try_reserve(u64::MAX));
        assert_eq!(budget.live_bytes(), 1);
    }

    #[test]
    fn lowering_the_limit_takes_effect_on_the_next_reservation() {
        let budget = GpuMemoryBudget::new(4096);
        assert!(budget.try_reserve(4096));
        budget.set_limit(1024);
        assert_eq!(budget.live_bytes(), 4096, "live bytes are never revoked");
        assert!(!budget.try_reserve(1));
        budget.release(4096);
        assert!(budget.try_reserve(1024));
    }
}
