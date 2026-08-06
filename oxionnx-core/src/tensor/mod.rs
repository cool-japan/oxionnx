//! Tensor types and utilities for the oxionnx ONNX inference engine.
//!
//! # Rank-0 (scalar) tensors
//!
//! ONNX distinguishes a **rank-0** tensor — shape `[]`, exactly one element —
//! from the rank-1 single-element tensor `[1]`. The distinction is observable:
//! `Shape` of a rank-0 tensor is the empty (length-0) vector while `Shape` of a
//! `[1]` tensor is the length-1 vector `[1]`, so any `Reshape`/`Concat` driven
//! by that vector produces a different rank. Likewise `ReduceSum(keepdims=0)`
//! over all axes is rank 0, and `Unsqueeze(axes=[0])` on it must give `[1]`, not
//! `[1, 1]`.
//!
//! ## The core contract
//!
//! Shape `[]` is a first-class value throughout this module:
//!
//! * **Element count is 1, not 0.** The empty shape's product is the empty
//!   product. [`Tensor::numel`] reads the data buffer and [`TensorView::numel`]
//!   takes the shape product, so both report 1. Only a genuine zero-size
//!   dimension (`[0, 3]`) gives 0, and the two cases must not be conflated —
//!   this is why nothing in this module clamps an element count with `.max(1)`.
//! * **Construction.** [`Tensor::rank0`] builds one directly, and
//!   `Tensor::zeros(&[])` / `Tensor::new(vec![v], vec![])` /
//!   `Tensor::try_new(vec![v], vec![])` all accept the empty shape (`try_new`'s
//!   validation and `new`'s `debug_assert` both compute the same empty
//!   product).
//! * **Broadcasting.** [`Tensor::broadcast_shape`] treats `[]` as the identity:
//!   `[]` broadcast with `[d0, …, dn]` is `[d0, …, dn]`, and `[]` with `[]` is
//!   `[]`. [`BroadcastIter`] over two rank-0 tensors yields exactly one pair.
//! * **Views and iteration.** [`Tensor::view`] on a rank-0 tensor gives a
//!   rank-0 [`TensorView`] with empty strides; `get(&[])` reads the element,
//!   `iter()` yields exactly one value, `is_contiguous()` is true, and
//!   `to_tensor()` round-trips the empty shape. `squeeze` of an all-size-1
//!   shape lands on `[]`, and `unsqueeze(&[0])` lifts `[]` back to `[1]`.
//! * **Axis-taking view methods are no-ops at rank 0, not panics.** A rank-0
//!   view has no axis, so *every* axis index is out of range for
//!   [`TensorView::transpose`], [`TensorView::slice`] and
//!   [`TensorView::select`] — each used to panic on a scalar unconditionally
//!   (`self.shape[p]`, `new_shape[axis]`, `Vec::remove(axis)`). Each now
//!   returns the scalar view unchanged, matching the graceful-degrade rule
//!   [`TensorView::unsqueeze`] already followed.
//!
//!   Beyond rank 0 the three differ deliberately. `slice` clamps an
//!   out-of-range range and `select` ignores an out-of-range axis, neither of
//!   which can lose an element the view could address; `transpose` still
//!   panics on an out-of-range `perm` entry, because silently dropping it
//!   would return a lower-rank view holding *fewer* elements. The `try_*`
//!   variants ([`TensorView::try_transpose`], [`TensorView::try_slice`],
//!   [`TensorView::try_select`]) report any out-of-range request as `None` for
//!   callers that want it rejected rather than degraded.
//!
//! ## Reading scalars across the ongoing migration
//!
//! Much of the operator layer still emits shape `[1]` where ONNX specifies rank
//! 0 — [`Tensor::scalar`] is that legacy constructor and is deliberately
//! unchanged. Consumers of a scalar-typed input (`If`'s condition, `Loop`'s
//! trip count and condition, `Clip`'s min/max, …) should therefore read it
//! through [`Tensor::to_scalar`], which accepts the rank-0 and `[1]` forms
//! alike. That keeps a consumer correct both before and after the producer
//! migrates, which is what allows the operator layer to move over op by op
//! rather than in one breaking change.

pub mod broadcastiter_traits;
pub mod functions;
pub mod tensorviewiter_traits;
pub mod types;

// Re-export all public items so the original `tensor::*` usage is preserved.
pub use functions::*;
pub use types::*;
