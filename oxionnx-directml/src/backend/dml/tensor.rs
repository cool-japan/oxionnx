//! `DML_BUFFER_TENSOR_DESC` construction — and the lifetime rule that keeps it sound.
//!
//! # No math here
//!
//! `sizes`, `strides` and `total_bytes` are computed once, on **every** platform, in
//! [`crate::layout::DmlTensorLayout`], and unit-tested on Linux.  This module copies those
//! numbers into DirectML's struct and recomputes nothing.
//!
//! That is not tidiness, it is the containment for two hazards that neither `rustc` nor
//! `clippy` nor any test we can run here would catch:
//!
//! * **`TotalTensorSizeInBytes` is not `product(sizes) * 4`.**  It is
//!   `(1 + Σᵢ (sizes[i] - 1) · strides[i]) · 4`, rounded up to a multiple of 4 — the true
//!   footprint *given the strides*.  For a packed tensor the two formulas agree, which is
//!   exactly why the wrong one survives review.  For a **0-stride broadcast** tensor they
//!   diverge enormously: a `[1, 4]` operand broadcast to `[2, 3, 4]` occupies **16** bytes,
//!   not 96, because DirectML reads the *original, un-expanded* buffer through the
//!   0-strides.  Declaring 96 makes DirectML either reject the binding or read 80 bytes
//!   past the end of a 16-byte allocation.  [`crate::layout`] gets this right and pins it
//!   with `broadcast_to_total_bytes_is_the_source_size`; [`DmlTensorStorage::buffer_desc`]
//!   below reads the field and does not touch it.
//! * **`u32` truncation.**  Every size and stride crosses the FFI as a `u32`.  Each one is
//!   range-checked exactly once, up front, by [`crate::plan`] / [`crate::layout`] via
//!   `u32::try_from`.  Nothing in this file writes a bare `as u32` on a shape-derived
//!   value — there is no shape-derived value here to cast, only already-`u32` fields to
//!   copy.
//!
//! # Lifetime hazard
//!
//! `DML_BUFFER_TENSOR_DESC::{Sizes, Strides}` are raw `*const u32`, and
//! `DML_TENSOR_DESC::Desc` is a raw `*const c_void`.  **None of them carries a Rust
//! lifetime, so the borrow checker will not save you.**  DirectML copies everything it
//! needs during `IDMLDevice::CreateOperator`, so the pointers only need to be live *for
//! that call* — but they must be, and a self-referential struct that stores its own
//! `as_ptr()` and is then `move`d out of a constructor dangles immediately.
//!
//! The rule this API enforces **by shape**:
//!
//! > A [`DmlTensorStorage`] stores only the arrays.  The descriptors are built as **stack
//! > locals inside a single function** — [`super::op`]'s `compile_*` — and die when that
//! > function returns, while the storage that backs them is a local in the *same* function,
//! > declared *before* them.
//!
//! Concretely, and this ordering is load-bearing:
//!
//! ```ignore
//! let a_store = DmlTensorStorage::new(&layout.a);   // owns Sizes[] / Strides[]
//! let a_buf   = a_store.buffer_desc();              // points into a_store
//! let a_desc  = BoundTensorDesc::new(&a_buf);       // points at a_buf
//! let op = DML_GEMM_OPERATOR_DESC { ATensor: a_desc.as_ptr(), .. };
//! unsafe { dml.CreateOperator(&DML_OPERATOR_DESC { .. }, &mut out) }?;   // copies it all
//! // every local above dies here, together, in reverse order — after the copy.
//! ```
//!
//! **No function in this subtree may return a `DML_BUFFER_TENSOR_DESC`, a
//! `DML_TENSOR_DESC` or a `DML_*_OPERATOR_DESC` derived from a temporary**, and no
//! pointer obtained from [`BoundTensorDesc::as_ptr`] may outlive the `BoundTensorDesc`
//! binding it was taken from.  In particular `opt.map(|d| d.as_ptr())` on an
//! `Option<BoundTensorDesc>` **moves** the descriptor into the closure and returns a
//! pointer to a value that dies with it — write `opt.as_ref().map(BoundTensorDesc::as_ptr)`
//! instead, so the pointer refers to the long-lived local.

use core::ffi::c_void;
use core::marker::PhantomData;

use windows::Win32::AI::MachineLearning::DirectML::{
    DML_BUFFER_TENSOR_DESC, DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_DESC, DML_TENSOR_FLAG_NONE,
    DML_TENSOR_TYPE_BUFFER,
};

use crate::layout::{DmlTensorLayout, DML_RANK};

/// `DML_BUFFER_TENSOR_DESC::DimensionCount`.
///
/// This is *not* a shape-derived value: it is the crate-wide constant [`DML_RANK`], and
/// every layout [`crate::layout`] produces has exactly this many entries by construction.
/// `u32::try_from` is not `const`, so the width check is made below instead.
const DIMENSION_COUNT: u32 = DML_RANK as u32;

/// Compile-time proof that [`DIMENSION_COUNT`] and [`DML_RANK`] are the same number.
///
/// If [`DML_RANK`] ever changes to something that does not survive the `as u32` above, the
/// two array types stop matching and the **build** fails — rather than DirectML being told
/// a dimension count that disagrees with the length of the `Sizes` array it is handed.
const _: [(); DML_RANK] = [(); DIMENSION_COUNT as usize];

/// Owns the `Sizes` / `Strides` storage that a `DML_BUFFER_TENSOR_DESC` points at.
///
/// These are plain copies of [`DmlTensorLayout`]'s already-validated, already-
/// range-checked numbers.  Nothing here is recomputed; see the module docs for why that
/// is the whole point of the type.
pub(crate) struct DmlTensorStorage {
    /// `DML_BUFFER_TENSOR_DESC::Sizes`.
    sizes: [u32; DML_RANK],
    /// `DML_BUFFER_TENSOR_DESC::Strides`, in elements.  A 0 entry means "broadcast".
    strides: [u32; DML_RANK],
    /// When `true`, `Strides` may be passed as null — DirectML's fast path.
    packed: bool,
    /// `DML_BUFFER_TENSOR_DESC::TotalTensorSizeInBytes`, straight from
    /// [`DmlTensorLayout::total_bytes`].  **Never recomputed here.**
    total_bytes: u64,
}

impl DmlTensorStorage {
    /// Copy a neutral layout into FFI-shaped storage.
    ///
    /// `DataType` is always `DML_TENSOR_DATA_TYPE_FLOAT32` — `oxionnx_core::Tensor::data`
    /// is a `Vec<f32>` and this crate is f32-only.  `Flags` is always
    /// `DML_TENSOR_FLAG_NONE`, never `DML_TENSOR_FLAG_OWNED_BY_DML`: the latter would
    /// require routing the weight through the operator initializer's *input* bindings so
    /// DirectML could preprocess and keep its own copy, which this crate does not do — it
    /// binds every operand fresh at execute time.  Claiming `OWNED_BY_DML` without doing
    /// that would leave the operator reading a tensor that was never delivered.
    pub(crate) fn new(layout: &DmlTensorLayout) -> Self {
        Self {
            sizes: layout.sizes,
            strides: layout.strides,
            packed: layout.is_packed,
            total_bytes: layout.total_bytes,
        }
    }

    /// A `DML_BUFFER_TENSOR_DESC` whose `Sizes` / `Strides` point **into `self`**.
    ///
    /// The returned struct carries raw pointers with no lifetime, so it is only valid
    /// while `self` is alive **and has not moved**.  Keep both as locals in the function
    /// that calls `CreateOperator`, `self` declared first — see the module docs.
    ///
    /// `Strides` is left **null** when the layout is packed.  That is DirectML's
    /// documented fast path ("if `Strides` is null, the tensor is assumed to be packed in
    /// C-contiguous order"), and it is not merely an optimisation: an explicit stride
    /// vector makes DirectML take its general strided kernel even when the strides happen
    /// to be contiguous.  [`DmlTensorLayout::is_packed`] decides, on Linux, whether the
    /// stride vector *is* the packed one — this file does not re-derive that judgement.
    pub(crate) fn buffer_desc(&self) -> DML_BUFFER_TENSOR_DESC {
        DML_BUFFER_TENSOR_DESC {
            DataType: DML_TENSOR_DATA_TYPE_FLOAT32,
            Flags: DML_TENSOR_FLAG_NONE,
            DimensionCount: DIMENSION_COUNT,
            Sizes: self.sizes.as_ptr(),
            Strides: if self.packed {
                core::ptr::null()
            } else {
                self.strides.as_ptr()
            },
            // Hazard 7.  Copied, never computed.  For a 0-stride broadcast operand this is
            // the *source* buffer's footprint, which is far smaller than
            // `product(sizes) * 4` — and it is the buffer that will actually be bound.
            TotalTensorSizeInBytes: self.total_bytes,
            // 0 means "no guarantee about the base offset's alignment".  We could promise
            // `plan::DML_BUFFER_ALIGNMENT` here — every binding this crate makes is at
            // offset 0 of a committed D3D12 buffer, which the runtime aligns far more
            // strongly than 16 — but the field is a promise about *every future binding of
            // this descriptor*, not about the one we are about to make, and a promise
            // DirectML may then rely on when it lays out its kernels.  0 costs nothing and
            // cannot be violated.
            GuaranteedBaseOffsetAlignment: 0,
        }
    }
}

/// A `DML_TENSOR_DESC` branded with the lifetime of the `DML_BUFFER_TENSOR_DESC` it points
/// at, so that at least the *first* level of the two-level raw-pointer chain is
/// borrow-checked.
///
/// The second level — `DML_BUFFER_TENSOR_DESC::Sizes` into [`DmlTensorStorage`] — cannot
/// be branded, because `DML_BUFFER_TENSOR_DESC` is a foreign `#[repr(C)]` struct with no
/// lifetime parameter.  That one is held by convention (module docs), not by the compiler.
///
/// # `Copy`, and the one way to misuse it
///
/// [`Self::as_ptr`] returns a pointer to `self.raw`, i.e. into *this* copy.  A copy that
/// dies while the pointer is still in flight leaves the pointer dangling.  The only shape
/// in which that realistically happens is `option.map(|d| d.as_ptr())`, which moves the
/// descriptor into the closure; use `option.as_ref().map(BoundTensorDesc::as_ptr)`.
#[derive(Clone, Copy)]
pub(crate) struct BoundTensorDesc<'a> {
    /// The descriptor itself.  `Desc` points at the `'a` buffer descriptor.
    raw: DML_TENSOR_DESC,
    /// Ties `'a` — the borrow of the `DML_BUFFER_TENSOR_DESC` — to this value, so the
    /// buffer descriptor provably outlives it.
    _marker: PhantomData<&'a DML_BUFFER_TENSOR_DESC>,
}

impl<'a> BoundTensorDesc<'a> {
    /// Wrap a buffer descriptor in the tagged union DirectML's operator descriptions take.
    ///
    /// `Type` is always `DML_TENSOR_TYPE_BUFFER`: it is the only tensor type DirectML
    /// defines besides `DML_TENSOR_TYPE_INVALID`, and it selects `DML_BUFFER_TENSOR_DESC`
    /// as the type `Desc` points at.  Getting that tag wrong would have DirectML reinterpret
    /// our struct as a different one.
    pub(crate) fn new(buffer: &'a DML_BUFFER_TENSOR_DESC) -> Self {
        Self {
            raw: DML_TENSOR_DESC {
                Type: DML_TENSOR_TYPE_BUFFER,
                // The cast is spelled out from the `&'a` binding, not from a temporary, so
                // the resulting pointer carries exactly the provenance `'a` brands.
                // (`core::ptr::from_ref` would read better but is stable only since 1.76,
                // and this workspace's MSRV is 1.75.)
                Desc: (buffer as *const DML_BUFFER_TENSOR_DESC).cast::<c_void>(),
            },
            _marker: PhantomData,
        }
    }

    /// The pointer a `DML_*_OPERATOR_DESC`'s tensor field wants.
    ///
    /// Valid for as long as **this binding** lives — not for `'a`.  See the type's docs.
    pub(crate) fn as_ptr(&self) -> *const DML_TENSOR_DESC {
        &self.raw as *const DML_TENSOR_DESC
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{BoundTensorDesc, DmlTensorStorage, DIMENSION_COUNT};
    use crate::layout::{DmlTensorLayout, DML_RANK};

    /// Read back what `buffer_desc` would hand DirectML.
    ///
    /// # Safety
    /// `desc.Sizes` points into `storage`, which the caller keeps alive.
    unsafe fn sizes_of(
        desc: &windows::Win32::AI::MachineLearning::DirectML::DML_BUFFER_TENSOR_DESC,
    ) -> [u32; DML_RANK] {
        let mut out = [0u32; DML_RANK];
        for (i, slot) in out.iter_mut().enumerate() {
            // SAFETY: `Sizes` is a valid, non-null array of `DimensionCount == DML_RANK`
            // `u32`s owned by the `DmlTensorStorage` the caller is holding.
            *slot = unsafe { *desc.Sizes.add(i) };
        }
        out
    }

    #[test]
    fn packed_layout_leaves_strides_null_and_copies_the_sizes() {
        let layout = DmlTensorLayout::packed(&[2, 3]).unwrap();
        let storage = DmlTensorStorage::new(&layout);
        let desc = storage.buffer_desc();

        assert_eq!(desc.DimensionCount, DIMENSION_COUNT);
        assert!(
            desc.Strides.is_null(),
            "a packed tensor must take DirectML's null-Strides fast path"
        );
        // SAFETY: `storage` is alive for the whole test, so `desc.Sizes` is live.
        assert_eq!(unsafe { sizes_of(&desc) }, [1, 1, 2, 3]);
        assert_eq!(desc.TotalTensorSizeInBytes, 24);
        assert_eq!(desc.GuaranteedBaseOffsetAlignment, 0);
    }

    /// **The hazard-7 regression test, on the FFI side of the boundary.**
    ///
    /// `layout.rs` proves the number is 16; this proves the number that actually reaches
    /// `DML_BUFFER_TENSOR_DESC` is that same 16, and not `product(sizes) * 4 == 96`.
    #[test]
    fn broadcast_desc_declares_the_source_footprint_not_the_expanded_one() {
        let layout = DmlTensorLayout::broadcast_to(&[1, 4], &[2, 3, 4]).unwrap();
        let storage = DmlTensorStorage::new(&layout);
        let desc = storage.buffer_desc();

        // SAFETY: `storage` outlives `desc` for the whole test.
        assert_eq!(unsafe { sizes_of(&desc) }, [1, 2, 3, 4]);
        assert!(
            !desc.Strides.is_null(),
            "a 0-stride broadcast tensor must send its strides explicitly"
        );
        assert_eq!(
            desc.TotalTensorSizeInBytes, 16,
            "must be the SOURCE's 4 floats, NOT product(sizes) * 4 = 96"
        );
        assert_ne!(desc.TotalTensorSizeInBytes, 96);
    }

    #[test]
    fn bound_tensor_desc_points_at_the_buffer_desc() {
        let layout = DmlTensorLayout::packed(&[4]).unwrap();
        let storage = DmlTensorStorage::new(&layout);
        let buffer = storage.buffer_desc();
        let bound = BoundTensorDesc::new(&buffer);

        assert_eq!(
            bound.raw.Type,
            windows::Win32::AI::MachineLearning::DirectML::DML_TENSOR_TYPE_BUFFER
        );
        assert!(core::ptr::eq(
            bound
                .raw
                .Desc
                .cast::<windows::Win32::AI::MachineLearning::DirectML::DML_BUFFER_TENSOR_DESC>(),
            &buffer as *const windows::Win32::AI::MachineLearning::DirectML::DML_BUFFER_TENSOR_DESC,
        ));
        assert!(!bound.as_ptr().is_null());
    }
}
