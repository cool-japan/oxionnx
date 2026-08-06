//! D3D12 buffers, and the barrier / copy sequences around them.
//!
//! | Heap | **Required** initial state | CPU `Map`? | Role |
//! |------|---------------------------|------------|------|
//! | UPLOAD | `GENERIC_READ` | write-only | host → device staging |
//! | DEFAULT | `COMMON` | no | what the shader touches |
//! | READBACK | `COPY_DEST` | read-only | device → host staging |
//!
//! UPLOAD buffers stay in `GENERIC_READ` **forever**, and READBACK buffers stay in
//! `COPY_DEST` **forever** — D3D12 does not merely discourage transitioning them, it
//! *forbids* it.  Only DEFAULT buffers move.  [`GpuBuffer::barrier_to`] therefore records
//! nothing for a staging heap, and that is not a defensive fudge: `GENERIC_READ` is the
//! bitwise union that already contains `COPY_SOURCE`, and `COPY_DEST` is already
//! `COPY_DEST`, so a staging buffer is *always* in the state the canonical sequence below
//! wants it in.  The no-op is the correct barrier, not a suppressed one.
//!
//! # The canonical sequence
//!
//! Every dispatch in this crate — HLSL and DirectML alike — is exactly this, and both
//! engines are expected to follow it verbatim:
//!
//! ```text
//! core.begin()
//!   → dst.barrier_to(COPY_DEST)              // DEFAULT input buffers
//!   → dst.record_copy_from(list, &upload, n) // upload → default
//!   → inputs.barrier_to(NON_PIXEL_SHADER_RESOURCE)   [HLSL: they are SRVs]
//!                     or (UNORDERED_ACCESS)          [DML: it binds inputs as UAVs too]
//!   → output.barrier_to(UNORDERED_ACCESS)
//!   → Dispatch(es)
//!   → output.record_uav_barrier()            ← THE ONE EVERYONE OMITS
//!   → output.barrier_to(COPY_SOURCE)
//!   → readback.record_copy_from(list, &output, n)
//! core.submit_and_wait()
//!   → readback.read_f32(count)
//! ```
//!
//! ## Why the UAV barrier is not optional
//!
//! A transition barrier orders *state*, not *memory*.  `UNORDERED_ACCESS → COPY_SOURCE`
//! tells the driver the resource is about to be read by the copy engine; it does **not**
//! promise that the UAV writes of the preceding `Dispatch` have landed.  Only a
//! `D3D12_RESOURCE_BARRIER_TYPE_UAV` barrier does that.
//!
//! Some implementations flush UAV writes on a transition anyway, which is why omitting it
//! is correct-by-luck on a lot of NVIDIA hardware and produces garbage on a lot of AMD
//! hardware.  There is no Windows host and no GPU in this repository, so this is precisely
//! the bug we cannot catch here — hence [`GpuBuffer::record_uav_barrier`] exists, is
//! documented at both ends of the sequence, and is impossible to *reach* the readback
//! without having been offered.
//!
//! ## Why the resource state is tracked rather than assumed
//!
//! A transition barrier whose `StateBefore` does not match the resource's real state is,
//! in a retail runtime, silent corruption.  So every [`GpuBuffer`] carries its own state
//! in a [`Cell`] and [`GpuBuffer::barrier_to`] transitions *from what it knows*, never
//! from what the caller guessed.  A no-op transition (`StateBefore == StateAfter`) is
//! separately illegal — D3D12 rejects it — so it is elided.
//!
//! ## Why `Map`'s read range matters
//!
//! An **empty** read range (`Begin == End`) tells the driver "the CPU will not read this
//! memory", which on a discrete GPU permits it to skip the invalidation of the CPU cache
//! line — and the readback comes back as stale garbage.  [`GpuBuffer::read_f32`] passes
//! `D3D12_RANGE { Begin: 0, End: count * 4 }`.  Symmetrically, a *write* map passes an
//! empty **read** range (the CPU genuinely reads nothing) and a `None` **written** range
//! to `Unmap` ("assume I wrote everything"), which is the conservative, correct direction.

use core::cell::Cell;
use core::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::{
    ID3D12GraphicsCommandList, ID3D12Resource, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RANGE,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use super::device::D3d12Core;
use crate::error::{DirectMLError, HrExt, Result};
use crate::plan::{align_up, CBV_ALIGNMENT, ELEM_SIZE};

/// Every buffer allocation is rounded up to this many bytes.
///
/// Not a D3D12 *requirement* for a plain buffer — `Alignment: 0` already gives the
/// implicit 64 KiB placement alignment — but it is the granularity everything downstream
/// wants: a root CBV would need a 256-byte `SizeInBytes`
/// ([`crate::plan::CBV_ALIGNMENT`]), DirectML wants bindings aligned to
/// [`crate::plan::DML_BUFFER_ALIGNMENT`] (16) with a `TotalTensorSizeInBytes` that is a
/// multiple of 4, and 256 is a multiple of both.  Rounding once, here, means no caller
/// ever has to think about it — and `size_bytes()` always reports a size that is `>=` the
/// bytes the caller asked to store.
const BUFFER_SIZE_GRANULARITY: usize = CBV_ALIGNMENT;

/// Which D3D12 heap a [`GpuBuffer`] lives on.
///
/// This is not decoration: it is what makes "never barrier a staging buffer" a property
/// of the type rather than of the caller's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeapKind {
    /// `D3D12_HEAP_TYPE_DEFAULT` — device-local, shader-accessible, transitions freely.
    Default,
    /// `D3D12_HEAP_TYPE_UPLOAD` — CPU-write / GPU-read, pinned in `GENERIC_READ`.
    Upload,
    /// `D3D12_HEAP_TYPE_READBACK` — GPU-write / CPU-read, pinned in `COPY_DEST`.
    Readback,
}

impl HeapKind {
    /// The heap's `D3D12_HEAP_TYPE`.
    fn heap_type(self) -> D3D12_HEAP_TYPE {
        match self {
            Self::Default => D3D12_HEAP_TYPE_DEFAULT,
            Self::Upload => D3D12_HEAP_TYPE_UPLOAD,
            Self::Readback => D3D12_HEAP_TYPE_READBACK,
        }
    }

    /// The **only** state D3D12 will accept at creation for this heap.  For UPLOAD and
    /// READBACK it is also the only state the resource will ever be in.
    fn initial_state(self) -> D3D12_RESOURCE_STATES {
        match self {
            Self::Default => D3D12_RESOURCE_STATE_COMMON,
            Self::Upload => D3D12_RESOURCE_STATE_GENERIC_READ,
            Self::Readback => D3D12_RESOURCE_STATE_COPY_DEST,
        }
    }

    /// Only DEFAULT buffers are ever bound as UAVs, and only DEFAULT buffers may be
    /// transitioned.
    fn is_transitionable(self) -> bool {
        matches!(self, Self::Default)
    }
}

/// A committed D3D12 buffer plus the state tracking needed to barrier it correctly.
///
/// `!Send + !Sync` (it owns a COM interface and a [`Cell`]), which is what keeps it
/// behind `context.rs`'s mutex along with the rest of the backend.
pub(crate) struct GpuBuffer {
    resource: ID3D12Resource,
    /// The **allocated** size, i.e. the caller's request rounded up to
    /// [`BUFFER_SIZE_GRANULARITY`].  Always `>=` the bytes the caller wanted to store.
    size_bytes: u64,
    /// The state this resource is currently in, as far as the *command list being
    /// recorded* is concerned.
    ///
    /// A transition barrier whose `StateBefore` disagrees with reality is a debug-layer
    /// error and, in a retail runtime, silent corruption.  So it is tracked, not guessed.
    /// `Cell` because recording goes through `&self`; the whole backend already sits
    /// behind `DirectMLContext`'s mutex, which is what makes that sound.
    state: Cell<D3D12_RESOURCE_STATES>,
    heap: HeapKind,
}

impl GpuBuffer {
    /// DEFAULT heap, `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`, created in
    /// `D3D12_RESOURCE_STATE_COMMON`.  `size_bytes` is rounded up to
    /// [`BUFFER_SIZE_GRANULARITY`].
    ///
    /// This is the only kind of buffer a shader ever touches, and the only kind that may
    /// be transitioned.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `size_bytes` is 0 (D3D12 has no zero-length
    /// resource; [`crate::plan`] declines empty tensors upstream so this should be
    /// unreachable) or when rounding it up overflows `usize`.
    /// [`DirectMLError::Win32`] when `CreateCommittedResource` fails.
    pub(crate) fn new_default(core: &D3d12Core, size_bytes: u64) -> Result<Self> {
        Self::create(core, size_bytes, HeapKind::Default)
    }

    /// UPLOAD heap, created in `GENERIC_READ` — D3D12 rejects any other initial state for
    /// this heap, and rejects every subsequent transition away from it.
    ///
    /// # Errors
    /// As [`Self::new_default`].
    pub(crate) fn new_upload(core: &D3d12Core, size_bytes: u64) -> Result<Self> {
        Self::create(core, size_bytes, HeapKind::Upload)
    }

    /// READBACK heap, created in `COPY_DEST` — again, the only legal state, now and
    /// forever.
    ///
    /// # Errors
    /// As [`Self::new_default`].
    pub(crate) fn new_readback(core: &D3d12Core, size_bytes: u64) -> Result<Self> {
        Self::create(core, size_bytes, HeapKind::Readback)
    }

    /// Allocate an UPLOAD buffer sized for `data` and memcpy into it.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `data` is empty or its byte length overflows.
    /// [`DirectMLError::Win32`] on allocation or `Map` failure.
    /// [`DirectMLError::TransferError`] when `Map` succeeds but hands back a null
    /// pointer.
    pub(crate) fn upload_from_f32(core: &D3d12Core, data: &[f32]) -> Result<Self> {
        let bytes = data.len().checked_mul(ELEM_SIZE).ok_or_else(|| {
            DirectMLError::Declined(format!(
                "upload of {} f32s overflows a byte count",
                data.len()
            ))
        })?;

        // SAFETY: `f32` has no padding, no invalid bit patterns and no `Drop`, so
        // `[f32]` and the `[u8]` of the same allocation are the same bytes; the
        // resulting slice borrows `data` and has exactly `bytes` elements, which is
        // `data.len() * size_of::<f32>()` by construction above.  `f32`'s alignment (4)
        // exceeds `u8`'s (1), so the pointer is trivially aligned for the target type.
        let raw = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), bytes) };
        Self::upload_from_bytes(core, raw)
    }

    /// As [`Self::upload_from_f32`], for raw bytes — e.g. an already-256-padded constant
    /// block from [`crate::plan::MatMulConstants::const_buffer_bytes`].
    ///
    /// The `memcpy` runs under a [`MapGuard`], so `Unmap` cannot be skipped on an error
    /// path.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `bytes` is empty.
    /// [`DirectMLError::Win32`] on allocation or `Map` failure.
    /// [`DirectMLError::TransferError`] when `Map` hands back a null pointer.
    pub(crate) fn upload_from_bytes(core: &D3d12Core, bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(DirectMLError::Declined(
                "cannot upload an empty buffer: D3D12 has no zero-length resource".into(),
            ));
        }

        // `usize` → `u64` is a value-preserving widening on every target this builds for.
        let buffer = Self::new_upload(core, bytes.len() as u64)?;

        // The guard's scope ends before `buffer` is returned, so the resource is
        // unmapped by the time anybody else can see it.
        {
            let guard = MapGuard::for_write(&buffer.resource)?;
            // SAFETY: `guard.ptr()` is the base of a mapping of `buffer.resource`, whose
            // allocated size is `align_up(bytes.len(), 256) >= bytes.len()`, so the
            // destination range `[ptr, ptr + bytes.len())` is entirely inside it.  The
            // source is a live borrow of `bytes`.  Host memory and an UPLOAD-heap mapping
            // are distinct allocations, so they cannot overlap.  Both pointers are `u8`,
            // which has alignment 1, so no alignment precondition applies — this is why
            // the f32 path casts to bytes rather than copying `f32`s into a mapping whose
            // alignment D3D12 does not promise.
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), guard.ptr(), bytes.len());
            }
        }

        Ok(buffer)
    }

    /// Map, copy out `count` f32s, unmap.
    ///
    /// Legal only on a READBACK buffer, and only after [`D3d12Core::submit_and_wait`] has
    /// *returned `Ok`* — that fence wait is the entire reason the bytes read here are the
    /// bytes the GPU wrote.
    ///
    /// # Errors
    /// [`DirectMLError::TransferError`] when this is not a READBACK buffer, when `count`
    /// exceeds the buffer, or when `Map` hands back a null pointer.
    /// [`DirectMLError::Win32`] when `Map` fails.
    pub(crate) fn read_f32(&self, count: usize) -> Result<Vec<f32>> {
        if self.heap != HeapKind::Readback {
            return Err(DirectMLError::TransferError(
                "read_f32 is only valid on a READBACK buffer: DEFAULT and UPLOAD heaps are \
                 not CPU-readable, and mapping them for a read yields undefined contents"
                    .into(),
            ));
        }

        let bytes = count.checked_mul(ELEM_SIZE).ok_or_else(|| {
            DirectMLError::TransferError(format!("readback of {count} f32s overflows a byte count"))
        })?;
        // `usize` → `u64` is a value-preserving widening.
        if bytes as u64 > self.size_bytes {
            return Err(DirectMLError::TransferError(format!(
                "readback of {count} f32s ({bytes} bytes) exceeds the {} byte buffer",
                self.size_bytes
            )));
        }

        let mut out = vec![0.0_f32; count];

        // Non-empty read range: an *empty* one tells the driver the CPU reads nothing,
        // and some drivers then skip the cache invalidation — which is exactly how a
        // readback comes back as stale garbage.
        let guard = MapGuard::for_read(&self.resource, bytes)?;

        // SAFETY: `guard.ptr()` maps at least `self.size_bytes >= bytes` bytes of
        // `self.resource`, so the source range `[ptr, ptr + bytes)` is in bounds.  `out`
        // is a freshly allocated `Vec<f32>` of exactly `count` elements, i.e. exactly
        // `bytes` bytes of destination, and it cannot alias the GPU mapping.  Both sides
        // are treated as `u8`, so no alignment precondition applies.  Every byte of `out`
        // is overwritten, so no uninitialised or stale value survives the copy.
        unsafe {
            core::ptr::copy_nonoverlapping(guard.ptr(), out.as_mut_ptr().cast::<u8>(), bytes);
        }

        drop(guard);
        Ok(out)
    }

    /// Record a transition from the **tracked** current state to `to`, and update the
    /// tracker.
    ///
    /// Records nothing in two cases, both of which D3D12 would reject:
    ///
    /// * `StateBefore == StateAfter` — an explicitly illegal barrier;
    /// * a staging heap — an UPLOAD buffer is pinned in `GENERIC_READ` and a READBACK
    ///   buffer in `COPY_DEST` for their entire lifetimes.  This is not a suppressed
    ///   error: `GENERIC_READ` already *contains* `COPY_SOURCE`, and `COPY_DEST` already
    ///   *is* `COPY_DEST`, so a staging buffer is permanently in the state the canonical
    ///   sequence asks for, and the correct barrier is no barrier.
    ///
    /// This is the only sanctioned way to transition a buffer in this crate.  Constructing
    /// a `D3D12_RESOURCE_BARRIER` anywhere else re-opens the `StateBefore` hazard, and a
    /// `StateBefore` that disagrees with reality is silent corruption in a retail runtime.
    pub(crate) fn barrier_to(&self, list: &ID3D12GraphicsCommandList, to: D3D12_RESOURCE_STATES) {
        if !self.heap.is_transitionable() {
            return;
        }
        let from = self.state.get();
        if from == to {
            return;
        }

        let barrier = BarrierRef::transition(&self.resource, from, to);
        // SAFETY: `list` is an open compute command list on the same device as
        // `self.resource` (both come from the one `D3d12Core`).  `barrier.as_slice()` is
        // a one-element slice of a fully initialised `D3D12_RESOURCE_BARRIER` whose
        // `Transition` arm is the union's live field; the slice is valid for the duration
        // of the call, which is all `ResourceBarrier` needs — it copies the barrier into
        // the command list.  `StateBefore` is `self.state`, which is the state this
        // buffer was created in updated by every transition recorded through this very
        // function, so it matches reality.  The COM reference the barrier carries is
        // released exactly once, by `barrier`'s `Drop`, after this call.
        unsafe {
            list.ResourceBarrier(barrier.as_slice());
        }

        self.state.set(to);
    }

    /// Record a `D3D12_RESOURCE_BARRIER_TYPE_UAV` barrier on this buffer.
    ///
    /// **Required** between a `Dispatch` that writes this buffer through a UAV and the
    /// `CopyBufferRegion` that reads it back.  A transition barrier orders *state*, not
    /// *memory*: it does not promise that the dispatch's UAV writes have landed.  Some
    /// implementations flush them anyway, which is why omitting this is correct on much
    /// NVIDIA hardware and garbage on much AMD hardware — a bug that cannot be reproduced
    /// on the machine that wrote it.
    ///
    /// Records nothing on a staging heap: those resources are created without
    /// `ALLOW_UNORDERED_ACCESS` and are never bound as UAVs, so a UAV barrier on one is an
    /// invalid call with no meaning to suppress.
    pub(crate) fn record_uav_barrier(&self, list: &ID3D12GraphicsCommandList) {
        if !self.heap.is_transitionable() {
            return;
        }

        let barrier = BarrierRef::uav(&self.resource);
        // SAFETY: as `barrier_to` — a one-element slice of a fully initialised barrier
        // whose `UAV` arm is the union's live field, valid for the duration of a call that
        // only reads it, on an open command list from the same device.  `self.resource` is
        // a DEFAULT-heap buffer created with `ALLOW_UNORDERED_ACCESS`, which is what makes
        // a UAV barrier on it legal.  The COM reference is released once, in `Drop`.
        unsafe {
            list.ResourceBarrier(barrier.as_slice());
        }
    }

    /// `CopyBufferRegion(dst = self, 0, src = src, 0, bytes)`.
    ///
    /// The caller must already have transitioned `self` to `COPY_DEST` and `src` to
    /// `COPY_SOURCE` — for a staging heap both are free, per [`Self::barrier_to`].
    ///
    /// # The clamp
    ///
    /// `bytes` is clamped to the smaller of the two allocated sizes.  In every correct
    /// call the clamp is inert: [`crate::plan`] sizes both buffers from the same numbers
    /// and every allocation is rounded *up*, so `bytes <= min(sizes)` already holds.  It
    /// exists because this call records into a command list and cannot report an error —
    /// and an out-of-bounds `CopyBufferRegion` is GPU memory corruption, whereas a clamped
    /// one is at worst a short copy that the caller's own length checks will catch on
    /// readback.  Given no way to fail loudly, failing *inside* the allocation is strictly
    /// the better of the two.
    pub(crate) fn record_copy_from(
        &self,
        list: &ID3D12GraphicsCommandList,
        src: &GpuBuffer,
        bytes: u64,
    ) {
        let bytes = bytes.min(self.size_bytes).min(src.size_bytes);
        if bytes == 0 {
            return;
        }

        // SAFETY: `list` is an open command list on the same device as both resources.
        // Both are buffers (`D3D12_RESOURCE_DIMENSION_BUFFER`), which is what
        // `CopyBufferRegion` requires, and the clamp above guarantees the source range
        // `[0, bytes)` and the destination range `[0, bytes)` are both entirely within
        // their respective allocations.  Neither reference is retained past the call; the
        // command list references the resources by pointer, and the caller keeps both
        // `GpuBuffer`s alive until after `submit_and_wait` returns, which is what keeps
        // the GPU from reading a freed resource.
        unsafe {
            list.CopyBufferRegion(&self.resource, 0, &src.resource, 0, bytes);
        }
    }

    /// The underlying resource, for `DML_BUFFER_BINDING` and root-descriptor binding.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    /// GPU virtual address, for `SetComputeRootShaderResourceView` /
    /// `SetComputeRootUnorderedAccessView`.
    pub(crate) fn gpu_address(&self) -> u64 {
        // SAFETY: `self.resource` is a live committed buffer.  `GetGPUVirtualAddress` is a
        // pure accessor with no preconditions beyond that, and returns 0 for a resource
        // that has none (which a buffer always does).
        unsafe { self.resource.GetGPUVirtualAddress() }
    }

    /// The **allocated** size in bytes, i.e. the requested size rounded up to
    /// [`BUFFER_SIZE_GRANULARITY`].
    pub(crate) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// The one place a `CreateCommittedResource` happens.
    fn create(core: &D3d12Core, size_bytes: u64, heap: HeapKind) -> Result<Self> {
        if size_bytes == 0 {
            return Err(DirectMLError::Declined(
                "D3D12 has no zero-length buffer resource; an empty tensor must be declined \
                 upstream in `plan`"
                    .into(),
            ));
        }

        // `u64` → `usize` and back.  The round-up itself is `plan::align_up`, so the
        // alignment rule lives in exactly one place, is unit-tested on Linux, and is not
        // re-derived here.
        let requested = usize::try_from(size_bytes).map_err(|_| {
            DirectMLError::Declined(format!(
                "a {size_bytes}-byte buffer does not fit this target's address space"
            ))
        })?;
        let aligned = align_up(requested, BUFFER_SIZE_GRANULARITY).ok_or_else(|| {
            DirectMLError::Declined(format!(
                "rounding a {size_bytes}-byte buffer up to {BUFFER_SIZE_GRANULARITY} bytes \
                 overflows"
            ))
        })?;
        // `usize` → `u64` is a value-preserving widening.
        let width = aligned as u64;

        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: heap.heap_type(),
            // UNKNOWN for both: with a non-CUSTOM heap type, D3D12 derives the page
            // property and the memory pool itself.  Setting them explicitly alongside a
            // named heap type is an invalid-argument error, not a refinement.
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            // 0 means "let D3D12 pick", which for a buffer is the implicit 64 KiB.
            Alignment: 0,
            Width: width,
            // A buffer is a 1-D resource: Height, DepthOrArraySize, MipLevels and the
            // sample count are all fixed at 1, and the format must be UNKNOWN.  Any other
            // value is an invalid-argument error.
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if heap.is_transitionable() {
                // Only the DEFAULT heap gets UAV access: it is the only one a shader (or
                // DirectML, which binds *every* tensor as a UAV) ever writes.  Setting
                // this on an UPLOAD or READBACK buffer is rejected outright.
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };

        let initial_state = heap.initial_state();
        let mut resource: Option<ID3D12Resource> = None;

        // SAFETY: `core.device` is live.  `&heap_properties` and `&desc` coerce to
        // `*const`s that are valid for the call and only read by it — D3D12 copies both.
        // `None` for the optimised clear value is required for a buffer (clear values are
        // a texture concept; passing one is an invalid-argument error).  `resource` is a
        // caller-owned slot that receives a +1 reference on success and is left `None` on
        // failure, so the reference is owned exactly once and released by
        // `ID3D12Resource`'s `Drop`.
        unsafe {
            core.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                initial_state,
                None,
                &mut resource,
            )
        }
        .ctx("ID3D12Device::CreateCommittedResource")?;

        let resource = resource.ok_or_else(|| {
            DirectMLError::DeviceInitFailed(
                "CreateCommittedResource reported success but produced no resource".into(),
            )
        })?;

        Ok(Self {
            resource,
            size_bytes: width,
            state: Cell::new(initial_state),
            heap,
        })
    }
}

/// RAII for the COM reference inside a `D3D12_RESOURCE_BARRIER`.
///
/// # Why this type exists
///
/// `D3D12_RESOURCE_TRANSITION_BARRIER::pResource` and `D3D12_RESOURCE_UAV_BARRIER::pResource`
/// are both `ManuallyDrop<Option<ID3D12Resource>>`.  `ManuallyDrop` means:
///
/// * built from a `clone()` (an AddRef) and never dropped → **one leaked COM reference per
///   barrier, per dispatch, forever**;
/// * built from a borrow and dropped → **double release** → use-after-free the next time
///   anything touches the resource.
///
/// Neither is caught by rustc, by clippy, by Miri (this is FFI), or by any test that can
/// run in this repository.  So the AddRef and the Release are put in one type, at one
/// place, and the union arm that must be dropped is recorded in a plain Rust enum rather
/// than re-derived from `barrier.Type` — reading the wrong arm of a union is exactly the
/// mistake this is guarding against.
struct BarrierRef {
    barrier: D3D12_RESOURCE_BARRIER,
    /// Which arm of `barrier.Anonymous` is live.  The single source of truth for `Drop`.
    kind: BarrierKind,
}

/// The live arm of a [`BarrierRef`]'s union.
#[derive(Clone, Copy)]
enum BarrierKind {
    Transition,
    Uav,
}

impl BarrierRef {
    /// A transition barrier `before → after` on `resource`.
    ///
    /// The caller must have established that `before` is the resource's real state and
    /// that `before != after`; [`GpuBuffer::barrier_to`] is the only caller and does both.
    fn transition(
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) -> Self {
        Self {
            barrier: D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        // An explicit AddRef, balanced in `Drop`.
                        pResource: ManuallyDrop::new(Some(resource.clone())),
                        // A buffer has exactly one subresource, and it is index 0.
                        Subresource: 0,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            },
            kind: BarrierKind::Transition,
        }
    }

    /// A UAV barrier on `resource`.
    fn uav(resource: &ID3D12Resource) -> Self {
        Self {
            barrier: D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        // An explicit AddRef, balanced in `Drop`.
                        pResource: ManuallyDrop::new(Some(resource.clone())),
                    }),
                },
            },
            kind: BarrierKind::Uav,
        }
    }

    /// The one-element slice `ResourceBarrier` wants.
    fn as_slice(&self) -> &[D3D12_RESOURCE_BARRIER] {
        core::slice::from_ref(&self.barrier)
    }
}

impl Drop for BarrierRef {
    fn drop(&mut self) {
        // SAFETY: `self.kind` records which arm of the union was written, and it is set
        // only by `transition` and `uav` above, each of which initialises exactly the
        // matching arm.  So the field read here is the field that was written — this is
        // not a guess about the union's discriminant, it is a value we carried alongside
        // it precisely so it need not be guessed.
        //
        // Each arm's `pResource` was built as `ManuallyDrop::new(Some(resource.clone()))`,
        // i.e. an explicit AddRef.  Dropping it here Releases exactly that one reference.
        // `BarrierRef` is not `Clone` and `Drop` runs at most once per value, so the
        // Release is neither doubled nor skipped: the refcount is unchanged across the
        // barrier's whole lifetime, which is what `ResourceBarrier` expects (it copies the
        // barrier into the command list and does not take ownership of the reference).
        //
        // The explicit `*` is not cosmetic: rustc refuses to auto-`DerefMut` through a
        // `ManuallyDrop` *union* field, precisely because doing so silently would be the
        // easiest way in the language to run a destructor nobody asked for.
        unsafe {
            match self.kind {
                BarrierKind::Transition => {
                    let transition = &mut *self.barrier.Anonymous.Transition;
                    ManuallyDrop::drop(&mut transition.pResource);
                }
                BarrierKind::Uav => {
                    let uav = &mut *self.barrier.Anonymous.UAV;
                    ManuallyDrop::drop(&mut uav.pResource);
                }
            }
        }
    }
}

/// RAII for `ID3D12Resource::Map` / `Unmap` — the crate's second paired-FFI guard.
///
/// The guard is only ever constructed *after* a successful `Map`, so `Drop` can never
/// unmap a resource that was not mapped, and `Unmap` cannot be skipped by an early return
/// or a panic between the map and the copy.
struct MapGuard<'a> {
    resource: &'a ID3D12Resource,
    ptr: *mut u8,
    subresource: u32,
    /// The range handed to `Unmap` as "what the CPU wrote".
    ///
    /// `None` → "the CPU may have written anywhere", the conservative answer for a write
    /// map.  `Some(empty)` → "the CPU wrote nothing", which is true for a read map and
    /// lets the driver skip a flush it would otherwise have to perform.
    written: Option<D3D12_RANGE>,
}

impl<'a> MapGuard<'a> {
    /// Map subresource 0 for CPU **writes**.
    ///
    /// The *read* range is empty, which is the truth — an upload never reads back — and
    /// on a discrete GPU it lets the driver skip invalidating the CPU cache for memory we
    /// are about to overwrite in full.
    fn for_write(resource: &'a ID3D12Resource) -> Result<Self> {
        // Must outlive the `Map` call below; it does, being a local of this function.
        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
        let ptr = Self::map(resource, Some(&no_read as *const D3D12_RANGE), "write")?;
        Ok(Self {
            resource,
            ptr,
            subresource: 0,
            written: None,
        })
    }

    /// Map subresource 0 for CPU **reads** of `[0, read_bytes)`.
    ///
    /// The read range is *not* empty, and that is load-bearing: an empty read range tells
    /// the driver the CPU will not read the memory, and some drivers then skip the cache
    /// invalidation — which is how a readback silently returns stale garbage.
    fn for_read(resource: &'a ID3D12Resource, read_bytes: usize) -> Result<Self> {
        let read = D3D12_RANGE {
            Begin: 0,
            End: read_bytes,
        };
        let ptr = Self::map(resource, Some(&read as *const D3D12_RANGE), "read")?;
        Ok(Self {
            resource,
            ptr,
            subresource: 0,
            // The CPU wrote nothing.  Saying so lets the driver skip a write-back.
            written: Some(D3D12_RANGE { Begin: 0, End: 0 }),
        })
    }

    /// The shared `Map` call.  Returns `Err` — and therefore builds no guard — on failure,
    /// which is what makes `Drop`'s `Unmap` unconditionally paired.
    fn map(
        resource: &ID3D12Resource,
        read_range: Option<*const D3D12_RANGE>,
        purpose: &'static str,
    ) -> Result<*mut u8> {
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();

        // SAFETY: `resource` is a live committed buffer, so subresource 0 is its only
        // subresource and is always a valid index.  `read_range`, when `Some`, points at a
        // `D3D12_RANGE` local to the caller that outlives this call, and D3D12 only reads
        // it.  `&mut ptr` is a live slot that receives the mapped base address.  On
        // failure D3D12 leaves `ptr` untouched — hence the explicit null check below,
        // rather than trusting a null-plus-success driver quirk.
        unsafe { resource.Map(0, read_range, Some(&mut ptr)) }.ctx("ID3D12Resource::Map")?;

        if ptr.is_null() {
            // Nothing to unmap: a failed/degenerate `Map` did not produce a mapping, and
            // no guard is constructed, so `Drop` will not run.
            return Err(DirectMLError::TransferError(format!(
                "ID3D12Resource::Map for {purpose} reported success but returned a null pointer"
            )));
        }

        Ok(ptr.cast::<u8>())
    }

    /// The base of the mapping.  Valid for the guard's lifetime and no longer.
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for MapGuard<'_> {
    fn drop(&mut self) {
        // `self.written` is a field of `self`, so the pointer taken from it is valid for
        // the whole of this call.
        let written = self
            .written
            .as_ref()
            .map(|range| range as *const D3D12_RANGE);

        // SAFETY: this guard exists only because `MapGuard::map` returned `Ok`, i.e. a
        // successful `Map` of exactly `self.subresource` on `self.resource`, and
        // `MapGuard` is neither `Clone` nor `Copy`, so this `Unmap` pairs that one `Map`
        // exactly once.  `self.resource` outlives the guard by the `'a` borrow.  `written`
        // is either null (`None` — "the CPU may have written anywhere") or points at a
        // `D3D12_RANGE` living in `self`, which is alive for the duration of the call.
        // `Unmap` returns nothing and cannot fail in a way we could react to.
        unsafe {
            self.resource.Unmap(self.subresource, written);
        }
    }
}
