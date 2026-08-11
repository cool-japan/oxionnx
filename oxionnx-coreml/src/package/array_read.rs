//! Output extraction — turning an `MLMultiArray` into a tightly packed,
//! C-contiguous [`RawArray`] (dtype preserved) or [`Tensor`] (`f32`), in
//! **one** pass over CoreML's backing buffer.
//!
//! ## Why this is its own module
//!
//! Reading outputs is the only part of the runtime that is measurably
//! hot per prediction: a SCRFD detection frame pulls nine outputs of
//! ~252 k elements each back across the FFI boundary, and every one of
//! them arrives on the strided path (see [`CopyPlan`]).  The extraction
//! logic accordingly grew a real layout planner, which does not belong
//! inline among `macos_impl`'s FFI plumbing.
//!
//! ## Stride correctness (the SCRFD bug)
//!
//! CoreML may allocate output buffers with non-C-contiguous strides for
//! ANE / GPU alignment.  In practice this shows up on SCRFD: an output
//! declared as shape `[800, 1]` is laid out with strides `[32, 1]` —
//! each row padded to 32 elements for 64-byte cache-line alignment.  A
//! naive `copy_nonoverlapping(N)` from `dataPointer()` reads padding
//! bytes and silently scrambles the data; the symptom in OxiFace's
//! `--device coreml` SCRFD detector was zero detections on real face
//! images even though `coremltools`' Python `predict()` returned correct
//! values for the same model.
//!
//! Every reader here therefore addresses source elements as
//! `Σ idx[d] · strides[d]` — the C-major destination index decomposed
//! via the *declared* `shape`, multiplied with the array's *reported*
//! `strides()` — never as a flat run.  [`CopyPlan`] is purely an
//! evaluation strategy for that same expression; the regression suite in
//! `tests.rs` checks it against a deliberately naive implementation of
//! the formula over deliberately padded fixtures.
//!
//! ## Single pass
//!
//! `tensor_from_multi_array` used to be built on `read_raw_bytes`: one
//! pass gathered elements into a `Vec<u8>`, then a second pass converted
//! that whole buffer to `Vec<f32>` via `chunks_exact(4)`.  Both passes
//! are now fused — the `f32` reader writes final `f32` values straight
//! into its destination from inside the `getBytesWithHandler:` block, so
//! the intermediate byte buffer (one full-size allocation and one full
//! extra traversal per output, per prediction) is gone.
//!
//! ## Alignment
//!
//! Source addressing is always done on a `*const u8` derived from the
//! block's own pointer, which carries no alignment guarantee at an
//! arbitrary strided row offset.  So:
//!
//! * `Float32` runs are copied as **bytes** (`copy_nonoverlapping::<u8>`)
//!   into a `Vec<f32>` destination.  The destination *is* `f32`-aligned
//!   (it is a real `Vec<f32>`), the source needs no alignment for a
//!   `u8` copy, and a native-endian byte copy is bit-for-bit what
//!   `f32::from_ne_bytes` computed before.
//! * `Float16` elements are read with `read_unaligned::<u16>`, which is
//!   defined for any address, then widened via `half::f16::to_f32`.
//! * The source is **never** cast to `*const f32` / `*const u16` and
//!   dereferenced normally.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};

use super::{MlArrayDtype, RawArray};
use crate::error::{CoreMLError, Result};
use oxionnx_core::Tensor;

/// A stride-normalizing traversal plan: how to walk an `MLMultiArray`'s
/// backing buffer once, in C-major destination order, emitting the
/// longest contiguous element runs the source layout permits.
///
/// The plan splits the declared shape into an *outer* odometer plus a
/// trailing contiguous *row*.  That split is what turns the former
/// per-element slow path — which recomputed the full
/// `Σ idx[d] · strides[d]` sum from scratch for every single element —
/// into one `memcpy` per row with an incrementally maintained row
/// offset.  For a fully C-contiguous array the split degenerates to a
/// single run covering the whole buffer, so the contiguous fast path
/// needs no separate branch.
pub(super) struct CopyPlan {
    /// Elements per contiguous run.
    pub(super) row_len: usize,
    /// Number of runs — `total / row_len`, or 0 for an empty array.
    pub(super) row_count: usize,
    /// Extents of the dimensions the outer odometer iterates (every
    /// dimension not folded into the row).
    pub(super) outer_shape: Vec<usize>,
    /// Source strides, in elements, matching `outer_shape` positionally.
    pub(super) outer_strides: Vec<isize>,
    /// Total C-contiguous element count (`shape.iter().product()`).
    pub(super) total: usize,
}

impl CopyPlan {
    /// Build the plan for `shape` / `strides` (both in the CoreML
    /// convention: extents outermost-first, strides in *elements*).
    ///
    /// `shape` and `strides` must have the same length; callers get that
    /// from [`read_layout`], which rejects a mismatch up front.
    pub(super) fn build(shape: &[usize], strides: &[isize]) -> Self {
        let total: usize = shape.iter().product();
        let rank = shape.len().min(strides.len());

        // Fold as many trailing dimensions as possible into one
        // contiguous run, innermost first.  A dimension joins the run
        // when stepping it by one advances the source by exactly the
        // number of elements already in the run.
        let mut row_len: usize = 1;
        let mut split = rank;
        for d in (0..rank).rev() {
            if shape[d] == 1 {
                // A degenerate dimension is only ever indexed at 0, so
                // it contributes no offset and cannot break contiguity
                // — fold it in whatever stride CoreML reports for it.
                split = d;
                continue;
            }
            if strides[d] == row_len as isize {
                row_len *= shape[d];
                split = d;
            } else {
                break;
            }
        }

        // `row_len` is 0 exactly when some extent is 0, i.e. the array
        // holds no elements at all and nothing is copied.
        let row_count = total.checked_div(row_len).unwrap_or(0);
        Self {
            row_len,
            row_count,
            outer_shape: shape[..split].to_vec(),
            outer_strides: strides[..split].to_vec(),
            total,
        }
    }

    /// Invoke `emit(source_element_offset, destination_element_index)`
    /// once per contiguous run, in C-major destination order.
    ///
    /// The source offset is carried incrementally: each odometer step
    /// adds that dimension's stride, and each wrap subtracts exactly the
    /// offset that dimension accumulated (`extent · stride`).  This is
    /// algebraically identical to recomputing `Σ idx[d] · strides[d]`,
    /// which is what the pre-fusion implementation did per element.
    pub(super) fn for_each_run(&self, mut emit: impl FnMut(isize, usize)) {
        let outer_rank = self.outer_shape.len();
        let mut idx = vec![0usize; outer_rank];
        let mut src_offset: isize = 0;
        let mut dst_index: usize = 0;
        for _ in 0..self.row_count {
            emit(src_offset, dst_index);
            dst_index += self.row_len;
            for d in (0..outer_rank).rev() {
                idx[d] += 1;
                src_offset += self.outer_strides[d];
                if idx[d] < self.outer_shape[d] {
                    break;
                }
                idx[d] = 0;
                src_offset -= self.outer_shape[d] as isize * self.outer_strides[d];
            }
        }
    }
}

/// Map a source `MLMultiArrayDataType` to the portable
/// [`MlArrayDtype`] tag plus its per-element byte width, or reject it
/// with [`CoreMLError::UnsupportedOutputDtype`].
///
/// Only `Float32` and `Float16` are supported today — exactly the set
/// `tensor_from_multi_array` has always accepted.  `MLMultiArrayDataType`
/// is a newtype-wrapped integer with associated consts (not a real
/// Rust `enum`), so a catch-all arm is required for exhaustiveness.
fn dtype_and_width(dt: MLMultiArrayDataType) -> Result<(MlArrayDtype, usize)> {
    match dt {
        MLMultiArrayDataType::Float32 => Ok((MlArrayDtype::F32, core::mem::size_of::<f32>())),
        MLMultiArrayDataType::Float16 => Ok((MlArrayDtype::F16, core::mem::size_of::<u16>())),
        _ => Err(CoreMLError::UnsupportedOutputDtype(format!("{dt:?}"))),
    }
}

/// Everything both readers need before touching the backing buffer:
/// the declared shape, the portable dtype tag and its byte width, and
/// the traversal plan.
struct Layout {
    shape: Vec<usize>,
    dtype: MlArrayDtype,
    elem_bytes: usize,
    plan: CopyPlan,
}

/// Read shape/strides/dtype off `arr` and build its [`Layout`].
///
/// A shape/strides rank mismatch is impossible for a well-formed
/// `MLMultiArray`, but it is surfaced as an error rather than being
/// allowed to index out of bounds (which is what the pre-fusion walk
/// would have done).
fn read_layout(arr: &MLMultiArray) -> Result<Layout> {
    let shape = read_shape(arr);
    let strides = read_strides(arr);
    if shape.len() != strides.len() {
        return Err(CoreMLError::Internal(format!(
            "MLMultiArray reported {} shape dimensions but {} strides",
            shape.len(),
            strides.len()
        )));
    }
    let (dtype, elem_bytes) = dtype_and_width(unsafe { arr.dataType() })?;
    let plan = CopyPlan::build(&shape, &strides);
    Ok(Layout {
        shape,
        dtype,
        elem_bytes,
        plan,
    })
}

/// Run `body` inside `getBytesWithHandler:`, returning an error if the
/// framework does not invoke the handler.
///
/// `getBytesWithHandler:` is documented to call its handler
/// synchronously exactly once, but we do not trust that blindly — and
/// `block2::StackBlock` requires `Fn`, so the "did it run" signal goes
/// through a `Cell<bool>` rather than a captured `bool`.
fn with_bytes(arr: &MLMultiArray, body: impl Fn(*const u8)) -> Result<()> {
    let invoked: core::cell::Cell<bool> = core::cell::Cell::new(false);
    let invoked_ref: &core::cell::Cell<bool> = &invoked;
    let handler = block2::StackBlock::new(|bytes: NonNull<c_void>, _size: isize| {
        body(bytes.as_ptr() as *const u8);
        invoked_ref.set(true);
    });
    unsafe { arr.getBytesWithHandler(&handler) };
    if invoked.get() {
        Ok(())
    } else {
        Err(CoreMLError::Internal(
            "getBytesWithHandler did not invoke its handler".to_string(),
        ))
    }
}

/// Extract an `MLMultiArray`'s contents verbatim into a portable
/// [`RawArray`] — dtype preserved (no `Float16` → `f32`
/// up-conversion), shape normalized to a tightly-packed C-contiguous
/// byte run regardless of CoreML's internal strides.
///
/// Backs [`MlPackageModel::predict_raw`](super::MlPackageModel::predict_raw).
/// The `f32`-producing [`tensor_from_multi_array`] deliberately does
/// *not* build on this function any more: it would pay for a full
/// intermediate `Vec<u8>` plus a second conversion pass it can avoid
/// entirely (see this module's header).
///
/// Every copy here is `u8`-to-`u8` — never a typed pointer cast — so
/// neither the source nor the destination needs any alignment beyond
/// `align_of::<u8>() == 1`, whatever `dtype`'s natural alignment is.
pub(super) fn read_raw_bytes(arr: &MLMultiArray) -> Result<RawArray> {
    let Layout {
        shape,
        dtype,
        elem_bytes,
        plan,
    } = read_layout(arr)?;

    // Allocate the destination outside the closure; the closure writes
    // through a raw pointer.  The buffer is alive for the entire
    // function, and therefore for the entire synchronous handler call.
    let mut out: Vec<u8> = vec![0u8; plan.total * elem_bytes];
    let dst_ptr: *mut u8 = out.as_mut_ptr();
    let row_bytes = plan.row_len * elem_bytes;
    let elem_stride = elem_bytes as isize;

    with_bytes(arr, |base| {
        plan.for_each_run(|src_offset, dst_index| {
            // SAFETY: `src_offset` addresses an element the declared
            // (shape, strides) pair covers, so `base + src_offset` and
            // the `row_len` elements after it lie inside CoreML's
            // buffer; `dst_index + row_len <= plan.total` by
            // construction of the odometer, so the destination range
            // lies inside `out`.  Both pointers are `u8`.
            unsafe {
                let src = base.offset(src_offset * elem_stride);
                let dst = dst_ptr.add(dst_index * elem_bytes);
                core::ptr::copy_nonoverlapping(src, dst, row_bytes);
            }
        });
    })?;

    Ok(RawArray {
        shape,
        dtype,
        data: out,
    })
}

/// Convert an output `MLMultiArray` to a tightly-packed C-contiguous
/// `f32` [`Tensor`] in a single pass: the stride walk and the dtype
/// conversion happen together, inside one `getBytesWithHandler:` block,
/// writing final `f32` values straight into the returned buffer.
///
/// `Float32` sources are copied run-wise as raw bytes into an
/// `f32`-aligned destination (a native-endian byte copy is exactly what
/// `f32::from_ne_bytes` used to compute); `Float16` sources are widened
/// element-wise via an unaligned `u16` read during the same walk.
pub(super) fn tensor_from_multi_array(arr: &MLMultiArray) -> Result<Tensor> {
    let Layout {
        shape,
        dtype,
        elem_bytes,
        plan,
    } = read_layout(arr)?;

    let mut out: Vec<f32> = vec![0.0f32; plan.total];
    let dst_ptr: *mut f32 = out.as_mut_ptr();
    let row_len = plan.row_len;
    let elem_stride = elem_bytes as isize;

    match dtype {
        MlArrayDtype::F32 => with_bytes(arr, |base| {
            plan.for_each_run(|src_offset, dst_index| {
                // SAFETY: same bounds argument as `read_raw_bytes`.
                // The copy is `u8`-to-`u8`, so the unaligned source is
                // fine; the destination is a real `Vec<f32>` and
                // therefore correctly aligned for the values it will
                // hold.
                unsafe {
                    let src = base.offset(src_offset * elem_stride);
                    let dst = dst_ptr.add(dst_index).cast::<u8>();
                    core::ptr::copy_nonoverlapping(src, dst, row_len * 4);
                }
            });
        })?,
        MlArrayDtype::F16 => with_bytes(arr, |base| {
            plan.for_each_run(|src_offset, dst_index| {
                // SAFETY: same bounds argument as `read_raw_bytes`.
                // `read_unaligned` is defined for any source address,
                // which is what a strided row offset gives us.
                unsafe {
                    let src = base.offset(src_offset * elem_stride);
                    for k in 0..row_len {
                        let bits = core::ptr::read_unaligned(src.add(k * 2).cast::<u16>());
                        dst_ptr
                            .add(dst_index + k)
                            .write(half::f16::from_bits(bits).to_f32());
                    }
                }
            });
        })?,
        MlArrayDtype::F64 | MlArrayDtype::I32 | MlArrayDtype::I8 => {
            // Unreachable in practice: `read_layout` calls
            // `dtype_and_width`, which only ever returns `Ok` for
            // F32/F16 — anything else already returned
            // `Err(UnsupportedOutputDtype)` above. Surfaced as an
            // `Internal` error rather than `unreachable!()` so a
            // future change to `dtype_and_width`'s coverage fails
            // loudly as a `Result`, not a panic.
            return Err(CoreMLError::Internal(format!(
                "read_layout produced dtype {dtype:?}, but only F32/F16 are ever returned \
                 for MLMultiArray sources — this indicates an internal invariant violation",
            )));
        }
    }

    Ok(Tensor::new(out, shape))
}

/// Read `MLMultiArray::shape()` as a plain `Vec<usize>`.
fn read_shape(arr: &MLMultiArray) -> Vec<usize> {
    let s = unsafe { arr.shape() };
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = s.objectAtIndex(i);
        // NSNumber::longLongValue is the safe path for any integer
        // underlying type.
        let iv = v.longLongValue();
        out.push(if iv < 0 { 0 } else { iv as usize });
    }
    out
}

/// Read `MLMultiArray::strides()` as a plain `Vec<isize>` (in elements,
/// not bytes — same convention CoreML uses).
fn read_strides(arr: &MLMultiArray) -> Vec<isize> {
    let s = unsafe { arr.strides() };
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = s.objectAtIndex(i);
        out.push(v.longLongValue() as isize);
    }
    out
}
