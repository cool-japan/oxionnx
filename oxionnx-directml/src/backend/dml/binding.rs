//! DirectML binding tables — **the most dangerous module in this crate**.
//!
//! # The `ManuallyDrop` hazard
//!
//! `DML_BUFFER_BINDING::Buffer` is a `ManuallyDrop<Option<ID3D12Resource>>`, and
//! `DML_BINDING_TABLE_DESC::Dispatchable` is a `ManuallyDrop<Option<IDMLDispatchable>>`.
//! `ManuallyDrop` disables the automatic `Release()` that every other COM handle in this
//! crate gets for free, which leaves exactly two ways to be wrong:
//!
//! * built from a **clone** (an `AddRef`) and never dropped → **leaks** one COM reference
//!   per binding, per dispatch, forever;
//! * built from a **borrow** (no `AddRef`) and then dropped → **double-release** → a
//!   use-after-free the next time DirectML, or anyone else, touches the resource.
//!
//! Neither is caught by rustc, by clippy, by Miri (which does not see across FFI), or by
//! any test that can run on a machine without a GPU.  Each is caught only by a Windows
//! debug-layer session, or by a crash in production.
//!
//! ## The rule, and how this module enforces it
//!
//! > `ManuallyDrop::new(Some(x.clone()))` on the way in — an **explicit `AddRef`** —
//! > balanced by exactly one `ManuallyDrop::drop` in a `Drop` impl.
//!
//! Both `ManuallyDrop` fields are wrapped in an owning RAII type that pairs the two
//! halves so that no early return, no `?`, and no panic can separate them:
//! [`BufferBindings`] owns the `DML_BUFFER_BINDING`s, and the private `BindingTableDesc`
//! owns the `DML_BINDING_TABLE_DESC`.  This is the same paired-FFI guard pattern as
//! `oxionnx-coreml`'s `PixelBufferLockGuard` and this crate's `EventHandle` / `MapGuard`.
//!
//! **No code outside this module may construct a `DML_BUFFER_BINDING` or a
//! `DML_BINDING_TABLE_DESC`**, and nothing anywhere may `.clone()` a
//! `DML_BUFFER_BINDING` — the derived `Clone` `AddRef`s the inner resource, producing a
//! reference this module's `Drop` will never see, and therefore never release.
//!
//! ### The ordering rule inside `BindingTableDesc::new`
//!
//! Every fallible step (the descriptor-count checks, the two heap-handle lookups) runs
//! **before** the `dispatchable.clone()` that performs the `AddRef`.  If a `?` fired
//! *after* the clone but *before* `Self` was constructed, the already-evaluated
//! `ManuallyDrop` field would be dropped as part of unwinding the partially-built struct
//! literal — and `ManuallyDrop`'s drop is a no-op, so the `AddRef` would leak.  Doing the
//! AddRef last makes that unrepresentable.
//!
//! # Descriptor-count correctness
//!
//! `SizeInDescriptors` must be **at least** the dispatchable's
//! `GetBindingProperties().RequiredDescriptorCount`, and the shader-visible heap must
//! have at least that capacity.  A mismatch is memory corruption, not an error —
//! DirectML writes descriptors into the heap slots it was told it has.  Both bounds are
//! therefore *checked here*, against DirectML's own answer, on every table creation and
//! every reset.  This is the one hazard in this file that this crate can genuinely
//! enforce rather than merely document.
//!
//! Note that the *initializer's* required count and the *compiled operator's* differ, so
//! the heap must be sized from the **max** of the two.  That sizing is
//! [`super::dml_backend`]'s job; the check here catches it if it gets it wrong.

use core::ffi::c_void;
use core::mem::ManuallyDrop;
use core::ptr;

use windows::Win32::AI::MachineLearning::DirectML::{
    IDMLBindingTable, IDMLDevice, IDMLDispatchable, DML_BINDING_DESC, DML_BINDING_TABLE_DESC,
    DML_BINDING_TYPE_BUFFER, DML_BUFFER_BINDING,
};

use crate::backend::d3d12::buffer::GpuBuffer;
use crate::backend::d3d12::device::DescriptorHeap;
use crate::error::{DirectMLError, HrExt, Result};

// ─── BufferBindings ──────────────────────────────────────────────────────────

/// Owns the `ManuallyDrop` COM references inside a set of `DML_BUFFER_BINDING`s, and
/// releases each exactly once.
///
/// # Why a `Vec` and not an array
///
/// [`Self::descs`] hands out `DML_BINDING_DESC`s whose `Desc` field is a raw pointer
/// **into** `self.bindings`.  A `Vec`'s elements live on the heap, so they keep their
/// addresses even when the `BufferBindings` value itself is moved (only the `Vec` header
/// moves).  An inline array would relocate with the struct, and every `DML_BINDING_DESC`
/// previously handed out would dangle the moment the caller returned one from a function.
///
/// `self.bindings` is never mutated after [`Self::new`] — no `push`, no `remove`, no
/// `&mut` accessor exists — so no reallocation can ever move the elements either.  These
/// two facts together are the entire soundness argument for [`Self::descs`].
pub(crate) struct BufferBindings {
    /// One `DML_BUFFER_BINDING` per bound tensor, in DirectML's binding order.
    ///
    /// Each `Buffer` field holds **one owned COM reference**, taken by `clone()` in
    /// [`Self::new`] and released by exactly one `ManuallyDrop::drop` in [`Drop`].
    bindings: Vec<DML_BUFFER_BINDING>,
}

impl BufferBindings {
    /// One `DML_BUFFER_BINDING` per `(buffer, size_in_bytes)` entry, in order.
    ///
    /// `Offset` is always 0 — which is what makes DirectML's 16-byte
    /// [`DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT`](crate::plan::DML_BUFFER_ALIGNMENT)
    /// requirement on binding offsets trivially satisfied, and is why this crate has no
    /// sub-buffer suballocator.
    ///
    /// # Caller contract
    ///
    /// `size_in_bytes` must be **exactly** the corresponding tensor's
    /// [`DmlTensorLayout::total_bytes`](crate::layout::DmlTensorLayout::total_bytes), and
    /// the `GpuBuffer` must be at least that large.  Both are guaranteed when the caller
    /// allocates each buffer with `GpuBuffer::new_default(core, layout.total_bytes)`
    /// (which rounds the D3D12 allocation up to 256 bytes) and binds it with that same
    /// `layout.total_bytes`.
    ///
    /// Passing a size **larger** than the resource makes DirectML read past the end of the
    /// allocation; passing one **smaller** than the tensor's footprint makes it reject the
    /// binding.  Neither is checkable here — `GpuBuffer` knows its own size but not which
    /// tensor it is about to hold — so it is checked where both facts are in scope, in
    /// [`super::dml_backend`].
    ///
    /// Note the deliberate absence of a `size == 0` guard: a zero-element tensor is
    /// [`DirectMLError::Declined`] all the way back in [`crate::plan`], so a 0-byte
    /// binding cannot be constructed from a plan that reached this far.
    pub(crate) fn new(entries: &[(&GpuBuffer, u64)]) -> Self {
        let mut bindings = Vec::with_capacity(entries.len());
        for &(buffer, size_in_bytes) in entries {
            bindings.push(DML_BUFFER_BINDING {
                // The explicit `AddRef`.  Balanced by the single `ManuallyDrop::drop` in
                // this type's `Drop`, below.  `GpuBuffer` keeps its own reference for as
                // long as it lives, so this one is genuinely additional and genuinely
                // needed: DirectML holds the binding for the duration of the dispatch.
                Buffer: ManuallyDrop::new(Some(buffer.resource().clone())),
                Offset: 0,
                SizeInBytes: size_in_bytes,
            });
        }
        Self { bindings }
    }

    /// Fresh `DML_BINDING_DESC`s pointing **into** `self.bindings`, in binding order.
    ///
    /// Built on demand rather than stored, because storing them would make `self`
    /// self-referential — and a self-referential struct that is then moved out of its
    /// constructor dangles immediately, which is the exact bug this whole module exists
    /// to prevent.
    ///
    /// # Lifetime
    ///
    /// The returned `DML_BINDING_DESC`s are raw pointers and carry **no** lifetime; the
    /// borrow checker will not stop you from outliving `self` with them.  They are valid
    /// for as long as `self` is alive, and are invalidated by nothing else (see this
    /// type's documentation for why a move of `self` is harmless).  Every caller in this
    /// crate uses them within a single statement —
    /// `table.bind_inputs(&inputs.descs())` — where `inputs` provably outlives the
    /// temporary.
    pub(crate) fn descs(&self) -> Vec<DML_BINDING_DESC> {
        self.bindings.iter().map(buffer_binding_desc).collect()
    }

    /// A single `DML_BINDING_DESC` for entry `i`, for
    /// [`BindingTable::bind_temporary`] / [`BindingTable::bind_persistent`], which take
    /// one binding rather than a slice.
    ///
    /// `None` when `i` is out of range — the caller asked for a resource it never bound,
    /// which for the temporary and persistent slots is precisely how "DirectML asked for
    /// 0 bytes, so nothing was allocated" is expressed.
    pub(crate) fn desc_at(&self, i: usize) -> Option<DML_BINDING_DESC> {
        self.bindings.get(i).map(buffer_binding_desc)
    }
}

impl Drop for BufferBindings {
    fn drop(&mut self) {
        for binding in &mut self.bindings {
            // SAFETY: every `Buffer` was built by `ManuallyDrop::new(Some(res.clone()))`
            // in `new` — one owned COM reference each — and nothing between `new` and here
            // can have dropped it: `bindings` is private, is never mutated after `new`,
            // and no method hands out a `&mut DML_BUFFER_BINDING`.  `Drop::drop` runs
            // exactly once per value, so each reference is released exactly once.
            unsafe { ManuallyDrop::drop(&mut binding.Buffer) };
        }
    }
}

/// The `DML_BINDING_DESC` for one already-owned `DML_BUFFER_BINDING`.
///
/// Free-standing rather than a method so that it can be passed to `Iterator::map` by
/// name in both [`BufferBindings::descs`] and [`BufferBindings::desc_at`], keeping the
/// `Type` ↔ `Desc` correspondence — `DML_BINDING_TYPE_BUFFER` ⇒ the payload *is* a
/// `DML_BUFFER_BINDING` — in one place.
fn buffer_binding_desc(binding: &DML_BUFFER_BINDING) -> DML_BINDING_DESC {
    DML_BINDING_DESC {
        Type: DML_BINDING_TYPE_BUFFER,
        // Borrowed from the caller's `&DML_BUFFER_BINDING`, which lives in the
        // `BufferBindings`' heap-allocated `Vec` and therefore has a stable address.
        Desc: ptr::addr_of!(*binding).cast::<c_void>(),
    }
}

// ─── DML_BINDING_TABLE_DESC, and its ManuallyDrop ────────────────────────────

/// RAII owner of a `DML_BINDING_TABLE_DESC` and of the one COM reference held by its
/// `Dispatchable: ManuallyDrop<Option<IDMLDispatchable>>` field.
///
/// Exists only as a local inside [`BindingTable::new`] / [`BindingTable::reset`]: both
/// `CreateBindingTable` and `Reset` read the descriptor during the call and keep no
/// pointer to it, so it dies at the end of the function that built it — and takes its
/// `AddRef` with it.
struct BindingTableDesc {
    /// The descriptor.  Its `Dispatchable` holds exactly one owned COM reference, taken
    /// in [`Self::new`] and released in [`Drop`].
    raw: DML_BINDING_TABLE_DESC,
}

impl BindingTableDesc {
    /// Validate `descriptor_count` against **both** DirectML's requirement and the heap's
    /// capacity, then take the one `AddRef` this descriptor owns.
    ///
    /// # Errors
    ///
    /// [`DirectMLError::DispatchFailed`] when `descriptor_count` is below the
    /// dispatchable's `RequiredDescriptorCount` (DirectML would write descriptors it has
    /// no room for), or above the heap's capacity (the same, one level down).  Both are
    /// corruption rather than a recoverable condition, so they are surfaced as hard
    /// errors and never as [`DirectMLError::Declined`].
    /// [`DirectMLError::DispatchFailed`] from [`DescriptorHeap::cpu_handle`] /
    /// [`DescriptorHeap::gpu_handle`] when slot 0 is somehow out of range (a zero-capacity
    /// heap).
    fn new(
        heap: &DescriptorHeap,
        dispatchable: &IDMLDispatchable,
        descriptor_count: u32,
    ) -> Result<Self> {
        // SAFETY: `dispatchable` is a live COM interface pointer; `GetBindingProperties`
        // is an infallible vtable call that fills a caller-allocated, `Copy`, POD
        // `DML_BINDING_PROPERTIES` and returns it by value.  It borrows nothing.
        let required = unsafe { dispatchable.GetBindingProperties() }.RequiredDescriptorCount;

        if descriptor_count < required {
            return Err(DirectMLError::DispatchFailed(format!(
                "binding table sized for {descriptor_count} descriptors, but the dispatchable \
                 requires {required}; DirectML would write past the end of the table"
            )));
        }
        if descriptor_count > heap.capacity() {
            return Err(DirectMLError::DispatchFailed(format!(
                "binding table needs {descriptor_count} descriptors, but the shader-visible heap \
                 holds only {}; size the heap from max(initializer, compiled) \
                 RequiredDescriptorCount",
                heap.capacity()
            )));
        }

        // Every fallible step happens BEFORE the `AddRef` below — see this module's
        // documentation.  A `?` after the clone would leak the reference it took.
        let cpu = heap.cpu_handle(0)?;
        let gpu = heap.gpu_handle(0)?;

        Ok(Self {
            raw: DML_BINDING_TABLE_DESC {
                // The explicit `AddRef`, balanced by the single `ManuallyDrop::drop` in
                // this type's `Drop`.
                Dispatchable: ManuallyDrop::new(Some(dispatchable.clone())),
                CPUDescriptorHandle: cpu,
                GPUDescriptorHandle: gpu,
                SizeInDescriptors: descriptor_count,
            },
        })
    }

    /// Pointer for `CreateBindingTable` / `Reset`.  Valid while `self` lives.
    fn as_ptr(&self) -> *const DML_BINDING_TABLE_DESC {
        ptr::addr_of!(self.raw)
    }
}

impl Drop for BindingTableDesc {
    fn drop(&mut self) {
        // SAFETY: `Dispatchable` was built by `ManuallyDrop::new(Some(d.clone()))` in
        // `new` — one owned COM reference — and `raw` is private, never handed out by
        // `&mut`, and never reassigned.  `Drop::drop` runs exactly once, so the reference
        // is released exactly once.
        unsafe { ManuallyDrop::drop(&mut self.raw.Dispatchable) };
    }
}

// ─── BindingTable ────────────────────────────────────────────────────────────

/// An `IDMLBindingTable` over a shader-visible descriptor heap.
///
/// The table itself owns no `ManuallyDrop` field — `IDMLBindingTable` is an ordinary
/// `windows` COM handle with an ordinary `Drop` — so this type needs no `Drop` impl of
/// its own.  All of the `ManuallyDrop` danger lives in the *descriptor* used to create
/// and to reset it, which is why that descriptor is [`BindingTableDesc`] and never a bare
/// struct literal.
pub(crate) struct BindingTable {
    /// The DirectML binding table.  Released by `windows`' own `Drop`.
    table: IDMLBindingTable,
}

impl BindingTable {
    /// `IDMLDevice::CreateBindingTable` over slot 0 of `heap`.
    ///
    /// `heap` **must** be a `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV` heap created with
    /// `D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE` — DirectML writes its descriptors into
    /// it and the GPU reads them from there.  [`DescriptorHeap::new`] creates no other
    /// kind, so the requirement is satisfied by construction; it cannot be re-checked
    /// here, because D3D12 exposes no way to query a heap's flags back.
    ///
    /// `descriptor_count` is validated against the dispatchable's
    /// `RequiredDescriptorCount` and against the heap's capacity — see
    /// [`BindingTableDesc::new`].
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] on a descriptor-count mismatch;
    /// [`DirectMLError::Win32`] when `CreateBindingTable` fails.
    pub(crate) fn new(
        dml: &IDMLDevice,
        heap: &DescriptorHeap,
        dispatchable: &IDMLDispatchable,
        descriptor_count: u32,
    ) -> Result<Self> {
        let desc = BindingTableDesc::new(heap, dispatchable, descriptor_count)?;

        // SAFETY: `desc` is a live, fully-initialised `DML_BINDING_TABLE_DESC` local; it
        // outlives this call and is released — exactly once, `ManuallyDrop` and all — when
        // it is dropped at the end of this function.  `CreateBindingTable` reads the
        // descriptor during the call and retains no pointer into it (it does `AddRef` the
        // dispatchable for itself, independently of the reference `desc` holds).
        let table: IDMLBindingTable = unsafe { dml.CreateBindingTable(Some(desc.as_ptr())) }
            .ctx("IDMLDevice::CreateBindingTable")?;

        Ok(Self { table })
    }

    /// `IDMLBindingTable::Reset` onto a different dispatchable, reusing the same heap.
    ///
    /// # This rewrites heap descriptors from the CPU, **immediately**
    ///
    /// Not at GPU-execution time — at the moment this function is called.  So an
    /// initializer dispatch that has been *recorded* but not yet *executed* still has its
    /// descriptors in the heap, and resetting onto the compiled operator overwrites them
    /// out from under it.  The initializer's submission must therefore be waited on before
    /// this is called; see [`super::dml_backend`]'s two-submission rule.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] on a descriptor-count mismatch;
    /// [`DirectMLError::Win32`] when `Reset` fails.
    pub(crate) fn reset(
        &self,
        heap: &DescriptorHeap,
        dispatchable: &IDMLDispatchable,
        descriptor_count: u32,
    ) -> Result<()> {
        let desc = BindingTableDesc::new(heap, dispatchable, descriptor_count)?;

        // SAFETY: as in `new` — `desc` is a live local for the whole call and releases its
        // own `AddRef` on drop.  `Reset` reads the descriptor during the call and retains
        // no pointer into it.
        unsafe { self.table.Reset(Some(desc.as_ptr())) }.ctx("IDMLBindingTable::Reset")
    }

    /// Bind the operator's inputs, in DirectML's declared input order.
    ///
    /// For a GEMM that is `[A, B]` or `[A, B, C]`; for a binary elementwise op `[A, B]`;
    /// for a unary op `[A]`.  The count and order must match the operator's descriptor —
    /// DirectML validates the count, but *not* the order, so a swapped `A`/`B` on a
    /// `Subtract` or a `Divide` is a wrong answer with no diagnostic.
    ///
    /// `descs` must have been produced by [`BufferBindings::descs`] on a
    /// [`BufferBindings`] that is still alive; see its lifetime note.
    pub(crate) fn bind_inputs(&self, descs: &[DML_BINDING_DESC]) {
        // SAFETY: `descs` is a live, initialised slice for the duration of the call.  Each
        // element's `Desc` points at a `DML_BUFFER_BINDING` owned by a live
        // `BufferBindings` (the caller's contract, discharged at every call site by
        // passing a `&BufferBindings::descs()` temporary that cannot outlive its owner).
        // `BindInputs` copies what it needs into the table's heap descriptors and retains
        // no pointer into the slice.  It returns no `HRESULT`: DirectML reports a bad
        // binding through the debug layer and through the *next* `RecordDispatch`.
        unsafe { self.table.BindInputs(Some(descs)) };
    }

    /// Bind the operator's outputs.  Every operator this crate compiles has exactly one.
    ///
    /// Same lifetime contract as [`Self::bind_inputs`].
    pub(crate) fn bind_outputs(&self, descs: &[DML_BINDING_DESC]) {
        // SAFETY: as `bind_inputs`.
        unsafe { self.table.BindOutputs(Some(descs)) };
    }

    /// Bind the scratch buffer DirectML asked for, or nothing.
    ///
    /// Pass `None` **exactly when** the dispatchable's
    /// `GetBindingProperties().TemporaryResourceSize` is 0 — DirectML then expects no
    /// binding at all, and a 0-byte D3D12 buffer cannot be created anyway
    /// (`CreateCommittedResource` rejects `Width == 0`), so there is nothing to pass.
    ///
    /// The temporary is scratch space for one dispatch: DirectML does not read it at the
    /// start and does not preserve it at the end, so it may be reallocated freely between
    /// dispatches — unlike the persistent resource.  Its size at *initialise* time and at
    /// *execute* time are different numbers from different `GetBindingProperties` calls;
    /// using one where the other is meant is a heap overflow.
    pub(crate) fn bind_temporary(&self, desc: Option<&DML_BINDING_DESC>) {
        // SAFETY: `desc`, when `Some`, is a live `&DML_BINDING_DESC` for the duration of
        // the call, whose `Desc` points at a `DML_BUFFER_BINDING` owned by a live
        // `BufferBindings`.  `None` becomes a null pointer, which is DirectML's documented
        // encoding for "no temporary resource" and is the *required* encoding when
        // `TemporaryResourceSize == 0`.
        unsafe {
            self.table
                .BindTemporaryResource(desc.map(|d| ptr::addr_of!(*d)));
        }
    }

    /// Bind the persistent buffer, or nothing.
    ///
    /// Pass `None` exactly when `GetBindingProperties().PersistentResourceSize` is 0, for
    /// the same reasons as [`Self::bind_temporary`].
    ///
    /// **The persistent resource is not scratch.**  The operator initializer *writes* it
    /// (as the initializer's output), and every subsequent execution of that compiled
    /// operator *reads* it.  It must therefore be allocated once, kept alive for as long
    /// as the compiled operator is cached, and bound — with the same contents — to every
    /// dispatch.  Reallocating or zeroing it between dispatches destroys state DirectML
    /// put there at initialisation time, which surfaces as wrong numbers rather than as an
    /// error.
    pub(crate) fn bind_persistent(&self, desc: Option<&DML_BINDING_DESC>) {
        // SAFETY: as `bind_temporary`.
        unsafe {
            self.table
                .BindPersistentResource(desc.map(|d| ptr::addr_of!(*d)));
        }
    }

    /// The raw table, for `IDMLCommandRecorder::RecordDispatch`.
    pub(crate) fn raw(&self) -> &IDMLBindingTable {
        &self.table
    }
}
