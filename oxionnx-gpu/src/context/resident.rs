//! Session-lifetime residency for operands whose bytes never change.
//!
//! # What is resident, and what it replaces
//!
//! A graph's initializers — a convolution's weight and bias, `Gemm`'s `B` and
//! `C` — hold the same bytes on frame 1 and on frame 10_000. Every dispatch
//! nevertheless used to rebuild their device buffers from the host slice:
//! InSwapper-128 re-uploaded 502.7 MB of invariant convolution weights on
//! *every* forward pass. That is bus time on a path that is already the
//! frame's bottleneck, and it is allocation traffic on a `GPUDevice` whose
//! memory this crate must keep bounded (see `super::budget`).
//!
//! `ResidentBuffers` keeps one [`TrackedBuffer`] per identity for as long as
//! the [`crate::GpuContext`] lives, so those bytes cross the bus once per
//! session instead of once per node per frame.
//!
//! # Why the cache belongs to the context rather than to the caller
//!
//! Two reasons; the second is the binding one.
//!
//! * A resident buffer is a budget-accounted allocation, and the only place in
//!   this crate that may create one is `TrackedBuffer::create`, reached through
//!   `GpuContext`'s upload helpers. A cache owned outside the context would
//!   need those helpers made public — which is precisely the invariant they
//!   exist to hold.
//! * A [`TrackedBuffer`] calls `wgpu::Buffer::destroy` when it drops. oxionnx's
//!   `session::gpu_owner` exists because a `GpuContext` must be created *and
//!   destroyed on one dedicated thread*: destroying a device from a different
//!   thread than created it reproduced a driver `SIGSEGV`. A residency map held
//!   beside the session, rather than inside the context, would destroy its
//!   buffers on whatever thread happened to drop the session's last `Arc` —
//!   exactly the cross-thread teardown that module was built to close. As a
//!   field of `GpuContext`, resident buffers die with the device, on the
//!   device's own thread.
//!
//! # The identity is the caller's, and this crate never interprets it
//!
//! Keys are opaque strings. This crate has no idea what an ONNX initializer is
//! and must not learn: the caller — oxionnx's session layer, which knows which
//! tensors are graph initializers and that their names are unique and fixed for
//! the session's graph — promises that one key denotes one byte sequence for
//! the lifetime of the context.
//!
//! That promise is checked rather than trusted. Each entry records the kernel
//! slot label and the byte length it was uploaded for, and a lookup that
//! disagrees with either is a `Lookup::Conflict`: the operand uploads for
//! that dispatch alone and the existing entry is left untouched. Overwriting
//! would be the worse failure — every frame would re-upload while the cache
//! still reported a hit rate.
//!
//! # One identity, two on-device formats
//!
//! [w2-f16] A weight may be resident as `f32` bytes, as `f16` bytes, or as
//! both, depending on whether [`crate::GpuContext::set_f16_compute`] was on
//! when each kernel asked for it. Those are *different bytes for the same
//! identity*, so the format is part of the key: each entry holds an
//! independent `Slot` per [`WeightFormat`], and a lookup names the format it
//! needs.
//!
//! Keying rather than replacing is what makes a mid-session toggle flip safe.
//! Handing `f16` bytes to the `f32` kernel would not be a slow path or a
//! degraded result — the shader would reinterpret pairs of halves as single
//! floats and compute garbage. (Today's length check would in fact catch that
//! particular case, since an `f16` copy is half the bytes and would read as a
//! `Lookup::Conflict`; relying on that would mean a permanent per-frame
//! re-upload of every weight the moment a caller flipped the toggle, plus an
//! `uploaded_bytes` counter that never settles. The format in the key is the
//! actual fix, and the length check goes back to catching what it was written
//! for.)

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::activation::DeviceTensor;
use super::budget::TrackedBuffer;
use super::weight_format::WeightFormat;

/// Cumulative, monotonic counters describing what one context's residency
/// cache has done.
///
/// Monotonic deliberately: "what did this frame upload" is the difference
/// between two snapshots ([`Self::since`]), and a counter that could fall would
/// make that difference meaningless. [`Self::uploaded_bytes`] is the number the
/// whole residency claim rests on — once every initializer has been seen it
/// must stop growing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentCounters {
    /// Lookups served by a buffer that was already on the device.
    pub hits: u64,
    /// Lookups that had to upload — a first sight of a key, or a conflict.
    pub misses: u64,
    /// Bytes those misses handed to the driver.
    pub uploaded_bytes: u64,
}

impl ResidentCounters {
    /// The activity between the `earlier` snapshot and this one.
    ///
    /// Saturating, so a snapshot taken against a different context (or a
    /// counter that has been reset by dropping the context) yields zero rather
    /// than a wrapped number that would read as an enormous upload.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            uploaded_bytes: self.uploaded_bytes.saturating_sub(earlier.uploaded_bytes),
        }
    }

    /// Whether nothing happened at all — the early-out for a caller that only
    /// records non-empty deltas.
    #[must_use]
    pub fn is_idle(self) -> bool {
        self == Self::default()
    }
}

/// Stable identities for the invariant operands of one dispatch.
///
/// A slot left `None` means "these bytes are not invariant; upload them for
/// this dispatch only". That is what every un-keyed entry point passes, so the
/// kernels behave exactly as they did before residency existed.
///
/// The slots are named for the convolution they were built for. `Gemm` maps its
/// `B` onto [`Self::weight`] and its `C` onto [`Self::bias`]: the same
/// distinction between the matrix operand and the vector one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightKeys<'a> {
    /// Identity of the matrix operand — conv `W`, gemm `B`.
    pub weight: Option<&'a str>,
    /// Identity of the vector operand — conv `B`, gemm `C`.
    pub bias: Option<&'a str>,
}

impl<'a> WeightKeys<'a> {
    /// Keys for both slots; pass `None` for an operand that is not invariant.
    #[must_use]
    pub fn new(weight: Option<&'a str>, bias: Option<&'a str>) -> Self {
        Self { weight, bias }
    }
}

/// The device buffer a kernel binds for one operand.
///
/// Derefs to `wgpu::Buffer`, so a kernel binds it exactly as it bound the
/// per-dispatch [`TrackedBuffer`] it used to allocate. The variants differ in
/// who destroys the buffer: a [`Self::Transient`] one when this value drops, at
/// the end of the dispatch; a [`Self::Resident`] one when the context's cache
/// releases it, at the end of the session; a [`Self::Device`] one when the
/// caller drops the [`DeviceTensor`] it borrowed, at that activation's last
/// consumer within the run.
///
/// `Resident` holds an `Arc`, and that is what makes it *impossible* for a
/// resident buffer to be swallowed by the reusable-buffer pool:
/// `read_back_and_recycle_async` — the only path into `GpuBufferPool` — takes a
/// `TrackedBuffer` **by value**, which an `Arc` cannot produce. `Device` holds a
/// shared reference, which cannot produce one either.
pub(crate) enum OperandBuffer<'a> {
    /// Uploaded for this dispatch, destroyed when the dispatch ends.
    Transient(TrackedBuffer),
    /// Borrowed from the context's residency cache.
    Resident(Arc<TrackedBuffer>),
    /// Borrowed from a run-scoped activation the caller owns.
    Device(&'a DeviceTensor),
}

impl OperandBuffer<'_> {
    /// The binding for this operand.
    ///
    /// `Transient` and `Resident` buffers are allocated at exactly the size
    /// their contents needed, so `as_entire_binding` is exact for them — which
    /// is what every kernel already did for these two. A `Device` operand may
    /// sit in a pooled allocation up to 2x its tensor, so it binds its own
    /// range explicitly; see [`DeviceTensor::binding`].
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        match self {
            Self::Transient(buffer) => buffer.as_entire_binding(),
            Self::Resident(buffer) => buffer.as_entire_binding(),
            Self::Device(tensor) => tensor.binding(),
        }
    }
}

impl Deref for OperandBuffer<'_> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        match self {
            Self::Transient(buffer) => buffer,
            Self::Resident(buffer) => buffer,
            Self::Device(tensor) => tensor.buffer(),
        }
    }
}

/// One buffer the context keeps for its whole lifetime.
struct Slot {
    buffer: Arc<TrackedBuffer>,
    /// The kernel slot this was uploaded for (`"conv2d_weight"`, `"gemm_b"`, …).
    /// Two kernels asking for the same key with different slot semantics must
    /// not share bytes; see the module docs.
    label: &'static str,
    /// Bytes handed to the upload — the length a hit has to agree with.
    byte_len: u64,
}

/// Everything held under one caller identity: at most one buffer per on-device
/// format.
///
/// Two `Option`s rather than a map keyed by [`WeightFormat`]: there are exactly
/// two formats and there always will be exactly as many as the enum has
/// variants, so this costs no allocation and no hashing per lookup, and
/// [`ResidentBuffers::bytes`] / [`ResidentBuffers::clear`] stay obvious.
#[derive(Default)]
struct Resident {
    f32: Option<Slot>,
    f16: Option<Slot>,
}

impl Resident {
    /// The slot for one format.
    fn slot(&self, format: WeightFormat) -> &Option<Slot> {
        match format {
            WeightFormat::F32 => &self.f32,
            WeightFormat::F16 => &self.f16,
        }
    }

    /// The slot for one format, mutably.
    fn slot_mut(&mut self, format: WeightFormat) -> &mut Option<Slot> {
        match format {
            WeightFormat::F32 => &mut self.f32,
            WeightFormat::F16 => &mut self.f16,
        }
    }

    /// Every slot that holds a buffer.
    fn slots(&self) -> impl Iterator<Item = &Slot> {
        self.f32.iter().chain(self.f16.iter())
    }
}

/// What a lookup found.
pub(crate) enum Lookup {
    /// A buffer already on the device, matching label, length and usage.
    Hit(Arc<TrackedBuffer>),
    /// The key has never been seen: upload, then keep the result.
    Vacant,
    /// The key is taken by an upload this one does not match — or the cache's
    /// lock is poisoned. Either way: upload for this dispatch only, and change
    /// nothing.
    Conflict,
}

/// Buffers held for the lifetime of one [`crate::GpuContext`], keyed by an
/// identity the caller chooses.
#[derive(Default)]
pub(crate) struct ResidentBuffers {
    entries: Mutex<HashMap<String, Resident>>,
    hits: AtomicU64,
    misses: AtomicU64,
    uploaded_bytes: AtomicU64,
}

impl ResidentBuffers {
    /// Look `key` up for a `byte_len`-byte upload of `format` bytes into slot
    /// `label` needing `usage`.
    ///
    /// A hit is recorded here; a miss is recorded by [`Self::insert`] or
    /// [`Self::note_conflict`], after the upload it describes has actually
    /// succeeded — a lookup whose upload then declines on the byte budget must
    /// not show up as bytes that crossed the bus.
    ///
    /// `format` selects which of the identity's slots is consulted, so a key
    /// resident only as `f32` is [`Lookup::Vacant`] for an `f16` request — an
    /// upload of the other copy, never a hit on the wrong bytes.
    pub(crate) fn lookup(
        &self,
        key: &str,
        label: &str,
        byte_len: u64,
        usage: wgpu::BufferUsages,
        format: WeightFormat,
    ) -> Lookup {
        let Ok(entries) = self.entries.lock() else {
            // Nothing about correctness depends on a hit, and a poisoned lock
            // cannot be inserted into either, so this degrades to the
            // pre-residency behaviour: upload every time.
            return Lookup::Conflict;
        };
        match entries.get(key).map(|entry| entry.slot(format)) {
            None | Some(None) => Lookup::Vacant,
            Some(Some(slot))
                if slot.label == label
                    && slot.byte_len == byte_len
                    && slot.buffer.usage().contains(usage) =>
            {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Lookup::Hit(Arc::clone(&slot.buffer))
            }
            Some(Some(_)) => Lookup::Conflict,
        }
    }

    /// Keep `buffer` under `(key, format)` and hand back a shared handle to it.
    ///
    /// A second insert for the same key *and format* replaces that slot and
    /// leaves the other format's slot alone; the handle already handed out
    /// stays valid until its holder drops it. Two dispatches never race here in
    /// practice — this crate's contract is one dispatch at a time per device —
    /// so the replacement path exists for well-definedness, not for a case that
    /// occurs.
    pub(crate) fn insert(
        &self,
        key: &str,
        label: &'static str,
        byte_len: u64,
        buffer: TrackedBuffer,
        format: WeightFormat,
    ) -> Arc<TrackedBuffer> {
        let shared = Arc::new(buffer);
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.uploaded_bytes.fetch_add(byte_len, Ordering::Relaxed);
        if let Ok(mut entries) = self.entries.lock() {
            *entries.entry(key.to_string()).or_default().slot_mut(format) = Some(Slot {
                buffer: Arc::clone(&shared),
                label,
                byte_len,
            });
        }
        shared
    }

    /// Account an upload that had to be per-dispatch because the key was taken
    /// by different bytes. Counted as a miss with its bytes, so a cache that is
    /// silently thrashing shows up as upload bytes that keep growing.
    pub(crate) fn note_conflict(&self, byte_len: u64) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.uploaded_bytes.fetch_add(byte_len, Ordering::Relaxed);
    }

    /// Whether an operand with this identity is on the device in *any* format.
    ///
    /// The question `GpuContext::is_resident` asks on the caller's behalf —
    /// "have these bytes stopped crossing the bus?" — which a copy in either
    /// format answers.
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.entries
            .lock()
            .is_ok_and(|entries| entries.get(key).is_some_and(|e| e.slots().next().is_some()))
    }

    /// Whether this identity is on the device *in this format*.
    ///
    /// The question the byte budget asks, and it is a different one: a weight
    /// resident as `f32` still has to allocate and upload its `f16` copy the
    /// first time an `f16` dispatch wants it, so counting it as free would
    /// admit a node the device has no room for.
    pub(crate) fn contains_format(&self, key: &str, format: WeightFormat) -> bool {
        self.entries.lock().is_ok_and(|entries| {
            entries
                .get(key)
                .is_some_and(|entry| entry.slot(format).is_some())
        })
    }

    /// This cache's cumulative counters.
    pub(crate) fn counters(&self) -> ResidentCounters {
        ResidentCounters {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            uploaded_bytes: self.uploaded_bytes.load(Ordering::Relaxed),
        }
    }

    /// Device bytes currently pinned by resident buffers.
    ///
    /// The *reserved* size, not the uploaded length: that is what the byte
    /// budget counts, so this number is directly comparable with
    /// `GpuContext::live_gpu_bytes`.
    /// Sums *every* format's slot, so a key resident in both is counted twice —
    /// which is what the device is actually holding.
    pub(crate) fn bytes(&self) -> u64 {
        self.entries.lock().map_or(0, |entries| {
            entries
                .values()
                .flat_map(Resident::slots)
                .fold(0u64, |acc, slot| {
                    acc.saturating_add(slot.buffer.reserved_bytes())
                })
        })
    }

    /// Number of distinct identities held.
    ///
    /// Identities, not buffers: an initializer resident in both formats is one
    /// operand that has been uploaded twice, and this counts the operand. (The
    /// bytes of both copies do show up in [`Self::bytes`].)
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().map_or(0, |entries| entries.len())
    }

    /// Release every resident buffer, destroying those nothing else still
    /// holds. Counters are cumulative and are deliberately left alone.
    pub(crate) fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Snapshot arithmetic is the whole measurement contract, and none of it
    /// needs a device.
    #[test]
    fn a_delta_is_the_difference_between_two_snapshots() {
        let earlier = ResidentCounters {
            hits: 3,
            misses: 2,
            uploaded_bytes: 4096,
        };
        let later = ResidentCounters {
            hits: 9,
            misses: 2,
            uploaded_bytes: 4096,
        };
        let delta = later.since(earlier);
        assert_eq!(delta.hits, 6);
        assert_eq!(
            delta.misses, 0,
            "a frame that uploaded nothing must report no misses"
        );
        assert_eq!(
            delta.uploaded_bytes, 0,
            "this zero is the whole point of the cache"
        );
        assert!(!delta.is_idle(), "six hits is not idleness");
        assert!(later.since(later).is_idle());
    }

    /// A snapshot from a different context must not wrap into a huge delta.
    #[test]
    fn a_delta_against_a_larger_snapshot_saturates_to_zero() {
        let bigger = ResidentCounters {
            hits: 100,
            misses: 100,
            uploaded_bytes: 100,
        };
        let delta = ResidentCounters::default().since(bigger);
        assert!(delta.is_idle());
    }

    #[test]
    fn keys_default_to_no_residency_at_all() {
        let keys = WeightKeys::default();
        assert_eq!(keys.weight, None);
        assert_eq!(keys.bias, None);
        assert_eq!(WeightKeys::new(Some("w"), None).weight, Some("w"));
        assert_eq!(WeightKeys::new(None, Some("b")).bias, Some("b"));
    }

    /// An empty cache answers every query without a device.
    #[test]
    fn an_empty_cache_holds_nothing_and_has_done_nothing() {
        let cache = ResidentBuffers::default();
        assert!(!cache.contains("conv1.weight"));
        assert!(!cache.contains_format("conv1.weight", WeightFormat::F32));
        assert!(!cache.contains_format("conv1.weight", WeightFormat::F16));
        assert!(matches!(
            cache.lookup(
                "conv1.weight",
                "conv2d_weight",
                4096,
                wgpu::BufferUsages::STORAGE,
                WeightFormat::F16,
            ),
            Lookup::Vacant
        ));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.bytes(), 0);
        assert!(cache.counters().is_idle());
        cache.clear();
        assert_eq!(cache.len(), 0);

        // A conflict is accounted even though no entry exists to conflict with
        // — the bytes really did cross the bus.
        cache.note_conflict(64);
        let counters = cache.counters();
        assert_eq!(counters.misses, 1);
        assert_eq!(counters.uploaded_bytes, 64);
        assert_eq!(counters.hits, 0);
    }
}
