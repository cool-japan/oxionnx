//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// `alloc`-backed types/macros, imported unconditionally: `alloc` is always
// linked by the crate root (see lib.rs), and `alloc::vec::Vec` /
// `alloc::string::String` are the exact same items as `std::vec::Vec` /
// `std::string::String`, so this resolves identically whether or not the
// `std` feature is enabled.
use alloc::{format, string::String, vec, vec::Vec};

use super::functions::{broadcast_strides, compute_strides};

/// A read-only view into a tensor's data with stride-based indexing.
///
/// Enables zero-copy transpose, slice, squeeze, and unsqueeze operations
/// by manipulating shape, strides, and offset without copying data.
///
/// # Rank-0 views
///
/// An empty `shape` (and therefore an empty `strides`) describes a **rank-0**
/// (scalar) view: one element, addressed by the empty index list. See the
/// module documentation of [`crate::tensor`] for the engine-wide rank-0
/// contract. Every method here is defined for `ndim() == 0`:
/// [`numel`](TensorView::numel) is 1, [`get(&[])`](TensorView::get) reads the
/// single element, [`iter`](TensorView::iter) yields exactly one value, and the
/// axis-taking methods ([`transpose`](TensorView::transpose),
/// [`slice`](TensorView::slice), [`select`](TensorView::select)) degrade to a
/// no-op rather than panicking, since a rank-0 view has no axis any index could
/// name.
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}
impl<'a> TensorView<'a> {
    /// Create a view from a data slice, shape, and strides.
    pub fn new(data: &'a [f32], shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }
    /// Shape of this view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Strides of this view.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /// Number of dimensions. `0` for a rank-0 (scalar) view.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// Total number of elements.
    ///
    /// This is the product of the shape, which for the empty shape of a rank-0
    /// view is the empty product **1** — a scalar holds exactly one element,
    /// not zero. Only a genuine zero-size dimension (e.g. `[0, 3]`) yields 0.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// Whether this is a rank-0 (scalar, shape `[]`) view.
    pub fn is_rank0(&self) -> bool {
        self.shape.is_empty()
    }
    /// Check if the view is contiguous (C-order, row-major).
    ///
    /// Contiguous when `strides[i] == product(shape[i+1..])` for all `i`.
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape);
        self.strides == expected && self.offset == 0
    }
    /// Access a single element by multi-dimensional indices.
    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        if indices.len() != self.shape.len() {
            return None;
        }
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
        }
        let flat_idx: usize = self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>();
        self.data.get(flat_idx).copied()
    }
    /// Transpose: permute dimensions and their strides.
    ///
    /// A **rank-0** view is returned unchanged whatever `perm` says: it has no
    /// axis any index could name, so the only permutation of its axes is the
    /// identity. Without this, `self.shape[p]` indexes out of bounds for
    /// *every* `p` and `transpose(&[0])` on a scalar panics unconditionally.
    ///
    /// At rank >= 1 an out-of-range entry still panics, deliberately. Unlike
    /// the out-of-range degrades in [`slice`](TensorView::slice),
    /// [`select`](TensorView::select) and [`unsqueeze`](TensorView::unsqueeze)
    /// — none of which can lose an element the view could address — quietly
    /// dropping a `perm` entry here would return a *lower-rank view holding
    /// fewer elements*, turning a caller's bug into silent data loss. Failing
    /// loudly is the safer degrade; use
    /// [`try_transpose`](TensorView::try_transpose) to get the rejection as a
    /// `None` instead.
    ///
    /// # Panics
    /// Panics at rank >= 1 if any entry of `perm` is not a valid axis index.
    pub fn transpose(&self, perm: &[usize]) -> Self {
        if self.shape.is_empty() {
            return self.clone();
        }
        let new_shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| self.strides[p]).collect();
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }
    /// Checked [`transpose`](TensorView::transpose): `None` unless `perm` is a
    /// genuine permutation of `0..ndim()` (every axis named exactly once).
    ///
    /// For a rank-0 view the only valid permutation is the empty one, so
    /// `try_transpose(&[])` returns the scalar view and anything else is `None`.
    pub fn try_transpose(&self, perm: &[usize]) -> Option<Self> {
        let ndim = self.shape.len();
        if perm.len() != ndim {
            return None;
        }
        let mut seen = vec![false; ndim];
        for &p in perm {
            let slot = seen.get_mut(p)?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        Some(self.transpose(perm))
    }
    /// Slice along one axis: select a range `[start, end)` along the given axis.
    ///
    /// `start`/`end` are clamped into `[0, shape[axis]]` (and `end` to at least
    /// `start`), so an over-long or reversed range yields a correspondingly
    /// shorter or empty view instead of an out-of-bounds view or a `usize`
    /// underflow on `end - start`. An `axis` that does not name a dimension of
    /// this view — which for a rank-0 view is *every* axis — leaves the view
    /// unchanged rather than panicking; see
    /// [`transpose`](TensorView::transpose) for why this degrades instead of
    /// erroring, and use [`try_slice`](TensorView::try_slice) when the request
    /// must be validated.
    ///
    /// Requests already within range behave exactly as before.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Self {
        let Some(&dim) = self.shape.get(axis) else {
            return self.clone();
        };
        let start = start.min(dim);
        let end = end.clamp(start, dim);
        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;
        let axis_stride = self.strides.get(axis).copied().unwrap_or(0);
        Self {
            data: self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: self.offset + start * axis_stride,
        }
    }
    /// Checked [`slice`](TensorView::slice): `None` if `axis` does not name a
    /// dimension of this view, or if `start > end`, or if `end > shape[axis]`.
    pub fn try_slice(&self, axis: usize, start: usize, end: usize) -> Option<Self> {
        let &dim = self.shape.get(axis)?;
        if start > end || end > dim {
            return None;
        }
        Some(self.slice(axis, start, end))
    }
    /// Select a single index along an axis, reducing rank by 1.
    ///
    /// An `axis` that does not name a dimension of this view — which for a
    /// rank-0 view is *every* axis, there being no rank left to reduce —
    /// leaves the view unchanged rather than panicking; see
    /// [`transpose`](TensorView::transpose) for why this degrades instead of
    /// erroring.
    ///
    /// `index` is **not** bounds-checked: an out-of-range index produces a view
    /// whose offset points past the elements this view can legally address, so
    /// [`get`](TensorView::get) reads return `None` and
    /// [`to_tensor`](TensorView::to_tensor) zero-fills. Use
    /// [`try_select`](TensorView::try_select) to reject it up front.
    pub fn select(&self, axis: usize, index: usize) -> Self {
        if axis >= self.shape.len() {
            return self.clone();
        }
        let axis_stride = self.strides.get(axis).copied().unwrap_or(0);
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(axis);
        // `strides` is parallel to `shape` for every view this module builds,
        // but `TensorView::new` is public and cannot enforce that, so a shorter
        // `strides` must not turn into a `Vec::remove` panic here.
        if axis < new_strides.len() {
            new_strides.remove(axis);
        }
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset + index * axis_stride,
        }
    }
    /// Checked [`select`](TensorView::select): `None` if `axis` does not name a
    /// dimension of this view or `index` is out of range along that axis.
    ///
    /// Selecting along the sole axis of a rank-1 view is how a rank-0 view is
    /// normally produced: `t.view().try_select(0, i)` on a `[n]` tensor gives a
    /// scalar view of element `i`.
    pub fn try_select(&self, axis: usize, index: usize) -> Option<Self> {
        let &dim = self.shape.get(axis)?;
        if index >= dim {
            return None;
        }
        Some(self.select(axis, index))
    }
    /// Squeeze: remove dimensions of size 1.
    pub fn squeeze(&self, axes: &[usize]) -> Self {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        for (i, (&s, &st)) in self.shape.iter().zip(self.strides.iter()).enumerate() {
            if axes.contains(&i) && s == 1 {
                continue;
            }
            new_shape.push(s);
            new_strides.push(st);
        }
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }
    /// Unsqueeze: insert dimensions of size 1.
    pub fn unsqueeze(&self, axes: &[usize]) -> Self {
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_unstable();
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        for (offset, &ax) in sorted_axes.iter().enumerate() {
            // Clamp a caller-supplied out-of-range axis to the current (growing)
            // shape's length instead of letting it reach `Vec::insert` unchecked:
            // `insert` panics whenever `index > len`, and nothing upstream of this
            // internal view helper guarantees `axes` stays within the eventual
            // output rank -- a malformed model or a buggy caller can supply an
            // axis arbitrarily larger than `self.shape.len() + axes.len()`. This
            // method returns `Self`, not `Result`, so (matching the duplicate-axis
            // handling directly below, and the same "no panic on attacker/model
            // controlled input" rule that motivated it) we degrade gracefully --
            // clamping to "insert at the end" -- rather than crash the process.
            let pos = ax.min(new_shape.len());
            // `pos + 1 - offset` must not underflow: with duplicate entries in the
            // caller-supplied `axes` (e.g. `axes=[0,0,0]`), `offset` (the loop
            // counter) can exceed `pos + 1`. Compute it with checked arithmetic and
            // treat anything that doesn't fit (underflow, or a result at/beyond
            // `self.strides.len()`) as out-of-range, falling back to a stride of 1
            // rather than subtracting unconditionally and panicking.
            let stride_val = match pos.checked_add(1).and_then(|v| v.checked_sub(offset)) {
                Some(idx) if idx < self.strides.len() => self.strides[idx].max(1),
                _ => 1,
            };
            new_shape.insert(pos, 1);
            new_strides.insert(pos, stride_val);
        }
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }
    /// Materialize to a contiguous Tensor.
    ///
    /// If already contiguous, copies data directly. Otherwise, iterates
    /// through all elements using strided indexing.
    ///
    /// The returned tensor always satisfies the `Tensor` invariant
    /// `data.len() == shape.iter().product()`. A view that cannot actually
    /// reach every element its shape claims — reachable both through the public
    /// [`TensorView::new`] and by rank-reducing off the end of a zero-size
    /// dimension, e.g. `Tensor::zeros(&[0]).view().select(0, 0)`, which yields a
    /// rank-0 (one-element) view over an *empty* backing slice — is completed
    /// with zeros rather than panicking on the slice or, worse, silently
    /// returning a short-data tensor that later reads out of bounds.
    pub fn to_tensor(&self) -> Tensor {
        let n = self.numel();
        if self.is_contiguous() {
            if let Some(head) = self.data.get(..n) {
                return Tensor::new(head.to_vec(), self.shape.clone());
            }
            // Backing slice shorter than the shape claims: fall through to the
            // element-wise path, which zero-fills the unreachable tail.
        }
        let mut data: Vec<f32> = self.iter().collect();
        if data.len() != n {
            data.resize(n, 0.0f32);
        }
        Tensor::new(data, self.shape.clone())
    }
    /// Iterate over all elements in logical (row-major) order.
    pub fn iter(&self) -> TensorViewIter<'_> {
        let ndim = self.shape.len();
        let exhausted = self.numel() == 0;
        TensorViewIter {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            indices: vec![0; ndim],
            exhausted,
        }
    }
}
/// Iterator over a `TensorView` in logical (row-major) order.
pub struct TensorViewIter<'a> {
    data: &'a [f32],
    pub(super) shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    pub(super) indices: Vec<usize>,
    pub(super) exhausted: bool,
}
impl TensorViewIter<'_> {
    pub(super) fn get_at(&self, indices: &[usize]) -> Option<f32> {
        let flat_idx: usize = self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>();
        self.data.get(flat_idx).copied()
    }
}
/// N-dimensional tensor with f32 data and a shape vector.
/// Layout: row-major (C order), last dimension varies fastest.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
impl Tensor {
    /// Create a tensor from owned data and a shape vector.
    ///
    /// # Panics
    /// Panics in debug builds only (via `debug_assert_eq!`) if
    /// `data.len() != shape.iter().product()`. **Release builds do not
    /// validate this invariant** and will silently construct an inconsistent
    /// `Tensor`, which typically surfaces much later as an unrelated
    /// out-of-bounds panic deep inside an operator, or as silently wrong
    /// output. Callers that cannot statically guarantee the data/shape
    /// pairing agrees -- in particular anything constructing a tensor from
    /// parsed/untrusted model bytes -- should use [`Tensor::try_new`]
    /// instead, which validates unconditionally (including in release
    /// builds) and returns a typed error rather than risking a panic or
    /// silent corruption.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }

    /// Fallibly create a tensor from owned data and a shape vector.
    ///
    /// Unlike [`Tensor::new`], this validates `data.len() ==
    /// shape.iter().product()` **unconditionally, including in release
    /// builds**, and returns a typed error instead of constructing an
    /// inconsistent tensor. Prefer this constructor whenever the data/shape
    /// pairing is not statically guaranteed to agree -- e.g. when building a
    /// tensor from a parsed ONNX model's raw initializer bytes and declared
    /// shape, where a malformed model file could supply a mismatched pair.
    ///
    /// The element-count computation itself uses checked multiplication, so
    /// a shape whose product overflows `usize` (another way a malformed
    /// model can misbehave) is also reported as an error instead of
    /// wrapping or panicking.
    ///
    /// # Errors
    /// Returns [`crate::error::OnnxError::ShapeMismatch`] if `data.len() !=
    /// shape.iter().product()`, or if that product overflows `usize`.
    pub fn try_new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, crate::error::OnnxError> {
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                crate::error::OnnxError::ShapeMismatch(format!(
                    "Tensor::try_new: shape {shape:?} overflows usize when computing its element count"
                ))
            })?;
        if data.len() != expected {
            return Err(crate::error::OnnxError::ShapeMismatch(format!(
                "Tensor::try_new: data has {} elements but shape {:?} implies {}",
                data.len(),
                shape,
                expected
            )));
        }
        Ok(Self { data, shape })
    }
    /// Create a zero-filled tensor with the given shape.
    ///
    /// `Tensor::zeros(&[])` builds a **rank-0** (scalar) tensor holding a
    /// single `0.0`: the empty shape's element-count product is the empty
    /// product 1, not 0. See the [`crate::tensor`] module documentation.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0f32; n],
            shape: shape.to_vec(),
        }
    }
    /// Create a **rank-1** single-element tensor (shape `[1]`).
    ///
    /// This is the engine's legacy scalar representation and is deliberately
    /// left unchanged: much of the operator layer still emits and consumes
    /// shape `[1]` where ONNX specifies rank 0 (see the migration notes in the
    /// [`crate::tensor`] module documentation). Use [`Tensor::rank0`] when the
    /// ONNX-correct rank-0 value is required — most visibly when a `Shape` node
    /// consumes the result, since `Shape` of a rank-0 tensor is the empty
    /// (length-0) vector while `Shape` of a `[1]` tensor is `[1]`.
    pub fn scalar(val: f32) -> Self {
        Self {
            data: vec![val],
            shape: vec![1],
        }
    }
    /// Create a true **rank-0** (scalar) tensor: shape `[]`, one element.
    ///
    /// This is what ONNX calls a scalar, as distinct from the rank-1
    /// single-element tensor [`Tensor::scalar`] returns. `Shape` of this
    /// tensor is an empty vector, `Size` is 1, and it broadcasts against any
    /// other shape.
    pub fn rank0(val: f32) -> Self {
        Self {
            data: vec![val],
            shape: Vec::new(),
        }
    }
    /// Total number of elements in this tensor.
    ///
    /// Read off the data buffer, so a rank-0 tensor correctly reports 1.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    /// Number of dimensions (rank) of this tensor. `0` for a rank-0 scalar.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// Whether this is a true rank-0 (shape `[]`) tensor.
    ///
    /// Note this is a statement about *rank*, not about element count: a `[1]`
    /// tensor also holds one element but is rank 1. Use
    /// [`to_scalar`](Tensor::to_scalar) when either representation is
    /// acceptable.
    pub fn is_rank0(&self) -> bool {
        self.shape.is_empty()
    }
    /// The single value of a one-element tensor, whatever its rank.
    ///
    /// Returns `Some` for shape `[]`, `[1]`, `[1, 1]`, … — i.e. whenever
    /// exactly one element is present — and `None` otherwise. This is the
    /// predicate the operator layer should read scalar-typed inputs through
    /// (Loop's trip-count and condition, `If`'s condition, `Clip`'s min/max,
    /// `Pad`'s constant value, …): it accepts the ONNX-correct rank-0 form and
    /// the legacy [`Tensor::scalar`] `[1]` form alike, so op-by-op migration to
    /// rank 0 cannot break a consumer that has already been migrated.
    pub fn to_scalar(&self) -> Option<f32> {
        match self.data.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }
    /// Return a new tensor with the same data but a different shape.
    ///
    /// Reshaping to or from rank 0 is well-defined and supported:
    /// `t.reshape(&[])` on a one-element tensor yields the rank-0 form, and
    /// `scalar.reshape(&[1])` yields the rank-1 form.
    ///
    /// # Panics
    /// Panics if the element count changes. Prefer
    /// [`try_reshape`](Tensor::try_reshape) whenever the target shape is not
    /// statically known to preserve the element count — in particular when it
    /// comes from a parsed model.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.numel(),
            "reshape: element count mismatch"
        );
        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        }
    }
    /// Fallible [`reshape`](Tensor::reshape): returns a typed error instead of
    /// panicking when `new_shape` does not preserve the element count, and when
    /// the shape's element count overflows `usize`.
    ///
    /// # Errors
    /// Returns [`crate::error::OnnxError::ShapeMismatch`] if
    /// `new_shape.iter().product() != self.numel()`, or if that product
    /// overflows `usize`.
    pub fn try_reshape(&self, new_shape: &[usize]) -> Result<Self, crate::error::OnnxError> {
        Self::try_new(self.data.clone(), new_shape.to_vec())
    }
    /// Compute the broadcast shape of two tensors (NumPy rules).
    /// Returns Err if shapes are incompatible.
    ///
    /// A rank-0 operand is the identity for broadcasting: `[]` against
    /// `[d0, …, dn]` gives `[d0, …, dn]` (it right-aligns to zero axes, so
    /// every output axis is taken from the other operand), and `[]` against
    /// `[]` gives `[]`. Unlike a `[1]` operand, a rank-0 operand therefore
    /// never raises the output rank.
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
        let n = a.len().max(b.len());
        let mut out = vec![0usize; n];
        let a_pad = n - a.len();
        let b_pad = n - b.len();
        for i in 0..n {
            let ai = if i < a_pad { 1 } else { a[i - a_pad] };
            let bi = if i < b_pad { 1 } else { b[i - b_pad] };
            if ai == bi {
                out[i] = ai;
            } else if ai == 1 {
                out[i] = bi;
            } else if bi == 1 {
                out[i] = ai;
            } else {
                return Err(format!("Cannot broadcast {:?} with {:?}", a, b));
            }
        }
        Ok(out)
    }
    /// Retrieve a single element by flat index (bounds checked in debug).
    #[inline(always)]
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }
}
#[cfg(feature = "ndarray")]
impl Tensor {
    /// Convert an owned `ndarray::ArrayBase` with `f32` elements into a `Tensor`.
    ///
    /// The array is converted to C-order (row-major) contiguous layout before
    /// copying into the `Tensor` backing store.
    pub fn from_ndarray<S, D>(arr: ndarray::ArrayBase<S, D>) -> Self
    where
        S: ndarray::Data<Elem = f32>,
        D: ndarray::Dimension,
    {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr.iter().copied().collect();
        Self::new(data, shape)
    }
    /// Convert a borrowed `ndarray::ArrayView<'_, f32, D>` into a `Tensor`.
    pub fn from_ndarray_view<D>(view: ndarray::ArrayView<'_, f32, D>) -> Self
    where
        D: ndarray::Dimension,
    {
        let shape: Vec<usize> = view.shape().to_vec();
        let data: Vec<f32> = view.iter().copied().collect();
        Self::new(data, shape)
    }
    /// Convert this `Tensor` into an `ndarray::ArrayD<f32>`.
    ///
    /// Returns a new heap-allocated dynamic-rank array with a copy of this
    /// tensor's data.
    pub fn to_ndarray(&self) -> ndarray::ArrayD<f32> {
        let shape = ndarray::IxDyn(&self.shape);
        ndarray::ArrayD::from_shape_vec(shape, self.data.clone())
            .unwrap_or_else(|_| ndarray::ArrayD::zeros(ndarray::IxDyn(&self.shape)))
    }
    /// Return a borrowed `ndarray::ArrayViewD<f32>` into this tensor's data.
    ///
    /// The view is valid for the lifetime of `self`.
    ///
    /// Returns `Err` if the shape is inconsistent with the data length.
    pub fn as_ndarray_view(&self) -> Result<ndarray::ArrayViewD<'_, f32>, crate::error::OnnxError> {
        let shape = ndarray::IxDyn(&self.shape);
        ndarray::ArrayViewD::from_shape(shape, &self.data)
            .map_err(|e| crate::error::OnnxError::Internal(format!("ndarray view error: {e}")))
    }
    /// `ort`-compatible method: extract `(shape, data)` from this tensor.
    ///
    /// Returns a `(&[usize], &[f32])` tuple.  The type parameter `T` is ignored
    /// (it acts as a phantom type for ort API compatibility; oxionnx tensors are
    /// always `f32` internally).
    pub fn try_extract_tensor<T: ?Sized>(
        &self,
    ) -> Result<(&[usize], &[f32]), crate::error::OnnxError> {
        Ok((&self.shape, &self.data))
    }
    /// `ort`-compatible method: extract tensor data as an `ndarray::ArrayViewD<f32>`.
    ///
    /// The type parameter `T` is ignored — see [`Tensor::try_extract_tensor`].
    pub fn try_extract_array<T: ?Sized>(
        &self,
    ) -> Result<ndarray::ArrayViewD<'_, f32>, crate::error::OnnxError> {
        let shape = ndarray::IxDyn(&self.shape);
        ndarray::ArrayViewD::from_shape(shape, &self.data)
            .map_err(|e| crate::error::OnnxError::Internal(format!("ndarray view error: {e}")))
    }
}
impl Tensor {
    /// Create a contiguous view of this tensor.
    pub fn view(&self) -> TensorView<'_> {
        let strides = compute_strides(&self.shape);
        TensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides,
            offset: 0,
        }
    }
    /// Create a transposed view without copying data.
    pub fn transpose_view(&self, perm: &[usize]) -> TensorView<'_> {
        self.view().transpose(perm)
    }
    /// Create a sliced view without copying data.
    pub fn slice_view(&self, axis: usize, start: usize, end: usize) -> TensorView<'_> {
        self.view().slice(axis, start, end)
    }
}
impl Tensor {
    /// Create a broadcast iterator pairing this tensor with another.
    pub fn broadcast_iter<'a>(&'a self, other: &'a Tensor) -> Option<BroadcastIter<'a>> {
        BroadcastIter::new(self, other)
    }
}
/// Iterator that broadcasts two tensors together, yielding `(a_val, b_val)` pairs.
/// Does NOT allocate an expanded tensor — computes indices on the fly using strides.
pub struct BroadcastIter<'a> {
    pub(super) a_data: &'a [f32],
    pub(super) b_data: &'a [f32],
    pub(super) a_strides: Vec<usize>,
    pub(super) b_strides: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) output_strides: Vec<usize>,
    pub(super) total: usize,
    pub(super) idx: usize,
}
impl<'a> BroadcastIter<'a> {
    /// Create a broadcast iterator for two tensors.
    /// Returns `None` if shapes are not broadcast-compatible.
    pub fn new(a: &'a Tensor, b: &'a Tensor) -> Option<Self> {
        let output_shape = Tensor::broadcast_shape(&a.shape, &b.shape).ok()?;
        let a_strides = broadcast_strides(&a.shape, &output_shape);
        let b_strides = broadcast_strides(&b.shape, &output_shape);
        let output_strides = compute_strides(&output_shape);
        let total: usize = output_shape.iter().product();
        Some(Self {
            a_data: &a.data,
            b_data: &b.data,
            a_strides,
            b_strides,
            output_shape,
            output_strides,
            total,
            idx: 0,
        })
    }
    /// The output shape of the broadcast.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.total
    }
    /// Whether the iterator is empty.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
}
/// Tensor memory layout for image/convolution data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TensorLayout {
    /// Batch, Channels, Height, Width (default for ONNX/PyTorch).
    NCHW,
    /// Batch, Height, Width, Channels (used by TensorFlow, often faster on CPU).
    NHWC,
    /// Generic row-major layout (non-image tensors).
    #[default]
    RowMajor,
}

#[cfg(test)]
mod w1_hardening_tests {
    use super::*;
    use crate::error::OnnxError;

    /// [a10-18] regression: `axes` containing duplicate entries used to compute
    /// `pos + 1 - offset` unconditionally, underflowing `usize` (e.g. `1 - 2`) and
    /// panicking "attempt to subtract with overflow" before the `< self.strides.len()`
    /// guard ever ran. Reference values hand-traced (and cross-checked with an
    /// independent Python re-implementation of the same loop) for
    /// `shape=[2,3]` (strides `[3,1]`) with `axes=[0,0,0]`:
    ///   offset=0, ax=0 -> pos+1-offset=1 -> strides[1]=1            -> insert(0,1)
    ///   offset=1, ax=0 -> pos+1-offset=0 -> strides[0]=3            -> insert(0,3)
    ///   offset=2, ax=0 -> pos+1-offset underflows -> out-of-range=1 -> insert(0,1)
    /// giving shape=[1,1,1,2,3], strides=[1,3,1,3,1].
    #[test]
    fn unsqueeze_duplicate_axes_does_not_underflow() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = t.view();

        // Must not panic.
        let out = view.unsqueeze(&[0, 0, 0]);

        assert_eq!(out.shape(), &[1, 1, 1, 2, 3]);
        assert_eq!(out.strides(), &[1, 3, 1, 3, 1]);
    }

    /// Extension of the [a10-18] regression found while re-verifying the fix above:
    /// an out-of-range (not just duplicate) caller-supplied axis reaches
    /// `Vec::insert(pos, ..)` unclamped, and `insert` panics whenever `index > len`.
    /// For `shape=[2,3]` (len 2) and `axes=[9]`, the pre-fix code would call
    /// `new_shape.insert(9, 1)` on a 2-element vec and panic "insertion index (is 9)
    /// should be <= len (is 2)" -- the same "caller/model-controlled axis, no
    /// internal bounds check, panic instead of graceful degrade" defect class as the
    /// duplicate-axis case, just not triggered by duplicates. With the fix, `pos` is
    /// clamped to the shape's length at the time of insertion (hand-traced: `pos =
    /// min(9, 2) = 2`, appending the new size-1 dim at the end), so this must not
    /// panic.
    #[test]
    fn unsqueeze_out_of_range_axis_does_not_panic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = t.view();

        // Must not panic.
        let out = view.unsqueeze(&[9]);

        assert_eq!(out.shape(), &[2, 3, 1]);
        assert_eq!(out.numel(), 6, "clamping must not change the element count");
    }

    /// Sanity / non-regression check: unsqueezing at a single, non-duplicated axis
    /// still produces the textbook result (hand-verified: inserting a size-1 dim at
    /// axis 1 of a [2,3] tensor gives shape [2,1,3] with the new dim's stride equal
    /// to the original stride at that position, 1, since nothing after it shifts).
    #[test]
    fn unsqueeze_single_axis_unaffected_by_fix() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = t.view();

        let out = view.unsqueeze(&[1]);

        assert_eq!(out.shape(), &[2, 1, 3]);
        assert_eq!(out.strides(), &[3, 1, 1]);
    }

    /// Sanity / non-regression check: two distinct axes (the common, spec-valid
    /// case) still produce the correct interleaved shape/strides.
    #[test]
    fn unsqueeze_two_distinct_axes_unaffected_by_fix() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = t.view();

        let out = view.unsqueeze(&[0, 2]);

        assert_eq!(out.shape(), &[1, 2, 1, 3]);
        assert_eq!(out.strides(), &[1, 3, 1, 1]);
    }

    /// [a9-7]: `Tensor::try_new` must accept a matching data/shape pair.
    #[test]
    fn try_new_accepts_matching_data_and_shape() {
        let t = Tensor::try_new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("should succeed");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// [a9-7]: unlike `Tensor::new` (debug-assert only), `try_new` must reject a
    /// mismatched data/shape pair with a typed error in *every* build profile,
    /// including release -- this is the exact scenario from the finding
    /// (`Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2])`: 3 elements vs. a shape
    /// implying 4).
    #[test]
    fn try_new_rejects_mismatched_data_and_shape() {
        let err = Tensor::try_new(vec![1.0, 2.0, 3.0], vec![2, 2])
            .expect_err("3 elements cannot satisfy a shape implying 4");
        assert!(matches!(err, OnnxError::ShapeMismatch(_)));
    }

    /// [a9-7]: a shape whose element-count product overflows `usize` must be
    /// reported as a typed error, not wrap around (which could then pass the
    /// `data.len() == expected` check with a nonsensical `expected`) or panic.
    #[test]
    fn try_new_rejects_overflowing_shape_product() {
        let err = Tensor::try_new(vec![1.0], vec![usize::MAX, 2])
            .expect_err("shape product must be reported as overflow, not wrapped");
        assert!(matches!(err, OnnxError::ShapeMismatch(_)));
    }
}

/// [a0-21] Rank-0 (shape `[]`, one element) tensors are a first-class value.
///
/// Reference values come from NumPy, whose rank-0 arrays implement exactly the
/// semantics ONNX specifies. Computed with
/// `python3 -c "import numpy as np; a0 = np.array(7.0, dtype=np.float32); ..."`:
///
/// ```text
/// a0.shape                          -> ()      a0.ndim -> 0   a0.size -> 1
/// np.broadcast_shapes((), (2,3))    -> (2, 3)
/// np.broadcast_shapes((), ())       -> ()
/// (a0 + np.arange(6).reshape(2,3))  -> shape (2,3), [7,8,9,10,11,12]
/// np.squeeze(np.ones((1,1,1)))      -> shape ()
/// np.sum(np.arange(24).reshape(2,3,4), axis=None, keepdims=False) -> shape (), 276.0
/// np.expand_dims(a0, 0).shape       -> (1,)
/// a0.reshape(1).shape               -> (1,)      np.array([7.0]).reshape(()).shape -> ()
/// len(a0.shape)                     -> 0        # i.e. ONNX Shape(a0) is length-0
/// ```
#[cfg(test)]
mod w2_rank0_tests {
    use super::*;
    use crate::error::OnnxError;

    // ── Construction and element count ───────────────────────────────────────

    /// The empty shape's element count is the *empty product* 1, not 0: a
    /// scalar holds one element. NumPy: `np.array(7.0).size == 1`.
    #[test]
    fn rank0_holds_exactly_one_element() {
        let t = Tensor::rank0(7.0);
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data, vec![7.0]);
        assert!(t.is_rank0());
    }

    /// `zeros(&[])` must land on the scalar, not on an empty buffer -- the
    /// `shape.iter().product()` it sizes from is 1 for the empty shape.
    #[test]
    fn zeros_of_empty_shape_is_a_rank0_zero() {
        let t = Tensor::zeros(&[]);
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data, vec![0.0]);
    }

    /// A zero-size dimension is the *other* case and must stay distinct: 0
    /// elements, rank 1. Conflating it with rank 0 is what `.max(1)` clamps do.
    #[test]
    fn zero_size_dim_is_not_rank0() {
        let t = Tensor::zeros(&[0]);
        assert_eq!(t.numel(), 0);
        assert_eq!(t.ndim(), 1);
        assert!(!t.is_rank0());
        assert!(t.to_scalar().is_none());
    }

    /// Both constructors validate the rank-0 pairing against the same empty
    /// product, so a one-element buffer with shape `[]` is accepted and a
    /// two-element one is rejected.
    #[test]
    fn try_new_accepts_rank0_and_rejects_overfull_rank0() {
        let t = Tensor::try_new(vec![7.0], Vec::new()).expect("[] with 1 element is valid");
        assert_eq!(t.numel(), 1);

        let err = Tensor::try_new(vec![7.0, 8.0], Vec::new())
            .expect_err("shape [] implies exactly 1 element");
        assert!(matches!(err, OnnxError::ShapeMismatch(_)));
    }

    /// `Tensor::scalar` is the *legacy* rank-1 form and must stay byte-identical
    /// while the operator layer migrates; `rank0` is the ONNX-correct one. The
    /// observable difference is the rank, which is what `Shape` reports on.
    #[test]
    fn scalar_stays_rank1_and_is_distinct_from_rank0() {
        let legacy = Tensor::scalar(7.0);
        let rank0 = Tensor::rank0(7.0);

        assert_eq!(legacy.shape, vec![1]);
        assert_eq!(rank0.shape, Vec::<usize>::new());
        assert_eq!(legacy.data, rank0.data);
        // What a following `Shape` node would emit: length 1 vs. length 0.
        assert_eq!(legacy.ndim(), 1);
        assert_eq!(rank0.ndim(), 0);
    }

    /// `to_scalar` is the migration-safe reader: it accepts every one-element
    /// representation and rejects everything else.
    #[test]
    fn to_scalar_accepts_both_scalar_representations() {
        assert_eq!(Tensor::rank0(7.0).to_scalar(), Some(7.0));
        assert_eq!(Tensor::scalar(7.0).to_scalar(), Some(7.0));
        assert_eq!(Tensor::new(vec![7.0], vec![1, 1]).to_scalar(), Some(7.0));
        assert_eq!(Tensor::new(vec![7.0, 8.0], vec![2]).to_scalar(), None);
        assert_eq!(Tensor::zeros(&[0]).to_scalar(), None);
    }

    // ── Reshape across the rank boundary ─────────────────────────────────────

    /// NumPy: `np.array([7.0]).reshape(()).shape == ()` and
    /// `np.array(7.0).reshape(1).shape == (1,)`.
    #[test]
    fn reshape_round_trips_across_the_rank0_boundary() {
        let rank1 = Tensor::scalar(7.0);
        let rank0 = rank1.reshape(&[]);
        assert_eq!(rank0.shape, Vec::<usize>::new());
        assert_eq!(rank0.data, vec![7.0]);

        let back = rank0.reshape(&[1]);
        assert_eq!(back.shape, vec![1]);
        assert_eq!(back.data, vec![7.0]);
    }

    /// `try_reshape` reports a count-changing target as a typed error rather
    /// than panicking the way `reshape`'s `assert_eq!` does.
    #[test]
    fn try_reshape_reports_mismatch_instead_of_panicking() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]);
        assert!(
            t.try_reshape(&[]).is_err(),
            "2 elements cannot become rank 0"
        );
        assert_eq!(
            t.try_reshape(&[1, 2]).expect("count preserved").shape,
            vec![1, 2]
        );
    }

    // ── Broadcasting ─────────────────────────────────────────────────────────

    /// A rank-0 operand right-aligns to zero axes, so every output axis comes
    /// from the other operand. NumPy: `np.broadcast_shapes((), (2,3)) == (2,3)`,
    /// `np.broadcast_shapes((), ()) == ()`.
    #[test]
    fn broadcast_shape_treats_rank0_as_the_identity() {
        let empty: Vec<usize> = Vec::new();
        assert_eq!(
            Tensor::broadcast_shape(&[], &[2, 3]).expect("rank-0 broadcasts against anything"),
            vec![2, 3]
        );
        assert_eq!(
            Tensor::broadcast_shape(&[2, 3], &[]).expect("order must not matter"),
            vec![2, 3]
        );
        assert_eq!(
            Tensor::broadcast_shape(&[], &[]).expect("scalar with scalar is a scalar"),
            empty
        );
    }

    /// Unlike a `[1]` operand -- which right-aligns to one axis and therefore
    /// *raises* a rank-0 output to rank 1 -- a rank-0 operand never changes the
    /// output rank. This is the pair of cases the `[1]`-promotion bug conflates.
    #[test]
    fn rank0_and_rank1_scalars_broadcast_to_different_ranks() {
        let empty: Vec<usize> = Vec::new();
        assert_eq!(
            Tensor::broadcast_shape(&[], &[]).expect("rank 0 with rank 0"),
            empty
        );
        assert_eq!(
            Tensor::broadcast_shape(&[1], &[]).expect("rank 1 with rank 0"),
            vec![1]
        );
    }

    /// NumPy: `np.array(7.0, dtype=np.float32) + np.arange(6, dtype=np.float32).reshape(2,3)`
    /// gives shape (2,3) with values [7,8,9,10,11,12].
    #[test]
    fn broadcast_iter_pairs_rank0_against_a_matrix() {
        let scalar = Tensor::rank0(7.0);
        let mat = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);

        let iter = BroadcastIter::new(&scalar, &mat).expect("rank-0 is broadcast-compatible");
        assert_eq!(iter.output_shape(), &[2, 3]);
        assert_eq!(iter.len(), 6);

        let sums: Vec<f32> = iter.map(|(a, b)| a + b).collect();
        assert_eq!(sums, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    /// Two rank-0 operands yield an output of *one* pair, not zero: the
    /// iterator's total is the empty product. NumPy: `a0 * b0` is a scalar 21.0.
    #[test]
    fn broadcast_iter_of_two_rank0_yields_exactly_one_pair() {
        let a = Tensor::rank0(7.0);
        let b = Tensor::rank0(3.0);

        let iter = BroadcastIter::new(&a, &b).expect("rank-0 with rank-0");
        let empty: &[usize] = &[];
        assert_eq!(iter.output_shape(), empty);
        assert_eq!(iter.len(), 1);
        assert!(!iter.is_empty());

        let pairs: Vec<(f32, f32)> = iter.collect();
        assert_eq!(pairs, vec![(7.0, 3.0)]);
        assert!((pairs[0].0 * pairs[0].1 - 21.0).abs() < 1e-6);
    }

    // ── Views and iteration ──────────────────────────────────────────────────

    /// A rank-0 view has empty strides, is trivially contiguous, holds one
    /// element addressed by the empty index list, and round-trips its rank.
    #[test]
    fn rank0_view_is_addressable_and_round_trips() {
        let t = Tensor::rank0(7.0);
        let v = t.view();

        let empty: &[usize] = &[];
        assert_eq!(v.shape(), empty);
        assert_eq!(v.strides(), empty);
        assert_eq!(v.ndim(), 0);
        assert_eq!(v.numel(), 1);
        assert!(v.is_rank0());
        assert!(v.is_contiguous());

        assert_eq!(v.get(&[]), Some(7.0));
        // An index list of the wrong length names no element.
        assert_eq!(v.get(&[0]), None);

        let back = v.to_tensor();
        assert_eq!(back.shape, Vec::<usize>::new());
        assert_eq!(back.data, vec![7.0]);
    }

    /// The odometer in `TensorViewIter` runs over the empty coordinate list
    /// exactly once, so a rank-0 view yields one element and then stops.
    #[test]
    fn rank0_view_iterates_exactly_once() {
        let t = Tensor::rank0(7.0);
        let v = t.view();

        let mut it = v.iter();
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next(), Some(7.0));
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);

        assert_eq!(v.iter().collect::<Vec<f32>>(), vec![7.0]);
    }

    /// Squeezing every size-1 axis lands on rank 0, and unsqueezing lifts it
    /// back to `[1]`. NumPy: `np.squeeze(np.ones((1,1,1))).shape == ()` and
    /// `np.expand_dims(np.array(7.0), 0).shape == (1,)`.
    #[test]
    fn view_squeeze_to_rank0_and_unsqueeze_back() {
        let t = Tensor::new(vec![7.0], vec![1, 1, 1]);

        let squeezed = t.view().squeeze(&[0, 1, 2]);
        let empty: &[usize] = &[];
        assert_eq!(squeezed.shape(), empty);
        assert_eq!(squeezed.numel(), 1);
        assert_eq!(squeezed.get(&[]), Some(7.0));

        let lifted = squeezed.unsqueeze(&[0]);
        assert_eq!(lifted.shape(), &[1]);
        assert_eq!(lifted.get(&[0]), Some(7.0));
    }

    /// Selecting along the sole axis of a rank-1 view is the normal way to
    /// produce a rank-0 view, and it must keep pointing at the right element.
    #[test]
    fn select_off_a_rank1_view_produces_a_rank0_view() {
        let t = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        let picked = t.view().select(0, 1);
        let empty: &[usize] = &[];
        assert_eq!(picked.shape(), empty);
        assert_eq!(picked.numel(), 1);
        assert_eq!(picked.get(&[]), Some(20.0));
        assert_eq!(picked.to_tensor().data, vec![20.0]);

        assert!(t.view().try_select(0, 1).is_some());
        assert!(
            t.view().try_select(0, 3).is_none(),
            "index 3 is out of range for a length-3 axis"
        );
        assert!(
            t.view().try_select(1, 0).is_none(),
            "axis 1 does not exist on a rank-1 view"
        );
    }

    // ── Axis-taking view methods on a rank-0 view degrade, never panic ───────

    /// Every axis index is out of range for a rank-0 view, so `transpose`,
    /// `slice` and `select` used to panic unconditionally on one
    /// (`self.shape[p]`, `new_shape[axis]`, `Vec::remove(axis)`). They must
    /// degrade to the identity instead. Note this rank-0 identity is the *only*
    /// out-of-range degrade `transpose` performs: see
    /// `transpose_still_panics_on_a_bad_perm_at_rank1_and_above`.
    #[test]
    fn rank0_view_axis_methods_degrade_instead_of_panicking() {
        let t = Tensor::rank0(7.0);
        let v = t.view();
        let empty: &[usize] = &[];

        // Must not panic.
        let transposed = v.transpose(&[0]);
        assert_eq!(transposed.shape(), empty);
        assert_eq!(transposed.get(&[]), Some(7.0));

        let sliced = v.slice(0, 0, 1);
        assert_eq!(sliced.shape(), empty);
        assert_eq!(sliced.get(&[]), Some(7.0));

        let selected = v.select(0, 0);
        assert_eq!(selected.shape(), empty);
        assert_eq!(selected.get(&[]), Some(7.0));

        // The empty permutation is the only valid one at rank 0.
        assert!(v.try_transpose(&[]).is_some());
        assert!(v.try_transpose(&[0]).is_none());
        assert!(v.try_slice(0, 0, 1).is_none());
        assert!(v.try_select(0, 0).is_none());
    }

    /// Non-regression: for in-range requests the three methods behave exactly
    /// as they did before the rank-0 degrade paths were added.
    #[test]
    fn in_range_view_axis_methods_are_unchanged() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::new(data, vec![3, 4]);

        let tr = t.view().transpose(&[1, 0]);
        assert_eq!(tr.shape(), &[4, 3]);
        assert_eq!(tr.strides(), &[1, 4]);
        assert_eq!(tr.get(&[2, 1]), Some(6.0));
        assert_eq!(
            t.view().try_transpose(&[1, 0]).expect("valid perm").shape(),
            &[4, 3]
        );
        assert!(
            t.view().try_transpose(&[0, 0]).is_none(),
            "a repeated axis is not a permutation"
        );

        let sl = t.view().slice(0, 1, 3);
        assert_eq!(sl.shape(), &[2, 4]);
        assert_eq!(sl.get(&[0, 0]), Some(4.0));

        let se = t.view().select(1, 2);
        assert_eq!(se.shape(), &[3]);
        assert_eq!(se.get(&[1]), Some(6.0));
    }

    /// `transpose` is the one axis-taking method that keeps failing loudly on an
    /// out-of-range `perm` entry once there *is* an axis to name: quietly
    /// dropping the entry would hand back a lower-rank view holding fewer
    /// elements, which is silent data loss rather than a graceful degrade.
    /// `try_transpose` is the way to get that rejection without a panic.
    #[test]
    fn transpose_still_panics_on_a_bad_perm_at_rank1_and_above() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let bad = std::panic::catch_unwind(|| t.view().transpose(&[0, 9]));
        assert!(
            bad.is_err(),
            "an out-of-range perm entry must not be ignored"
        );

        // The checked variant reports it instead, without unwinding, and never
        // returns a view with a different element count.
        assert!(t.view().try_transpose(&[0, 9]).is_none());
        assert!(t.view().try_transpose(&[0]).is_none(), "dropped axis");
        assert_eq!(
            t.view()
                .try_transpose(&[1, 0])
                .expect("a genuine permutation")
                .numel(),
            6
        );
    }

    /// A reversed or over-long range clamps to an in-bounds (possibly empty)
    /// view instead of underflowing `end - start` or claiming elements the
    /// backing slice does not have.
    #[test]
    fn slice_clamps_reversed_and_overlong_ranges() {
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]);

        // Must not panic: 1 - 3 would underflow `usize`.
        let reversed = t.view().slice(0, 3, 1);
        assert_eq!(reversed.shape(), &[0]);
        assert_eq!(reversed.numel(), 0);
        assert_eq!(reversed.to_tensor().data, Vec::<f32>::new());

        let overlong = t.view().slice(0, 2, 99);
        assert_eq!(overlong.shape(), &[2]);
        assert_eq!(overlong.to_tensor().data, vec![2.0, 3.0]);

        assert!(t.view().try_slice(0, 3, 1).is_none());
        assert!(t.view().try_slice(0, 2, 99).is_none());
        assert!(t.view().try_slice(0, 2, 4).is_some());
    }

    // ── `to_tensor` keeps the Tensor invariant ───────────────────────────────

    /// A rank-0 view over an *empty* backing slice is reachable two ways: the
    /// public `TensorView::new`, and rank-reducing off a zero-size dimension.
    /// The contiguous fast path used to index `self.data[..1]` on a zero-length
    /// slice (panic), and the strided path used to collect 0 elements under a
    /// shape claiming 1, violating `data.len() == shape.product()` and tripping
    /// `Tensor::new`'s debug assert. The result must instead be a well-formed
    /// one-element tensor.
    #[test]
    fn to_tensor_of_rank0_view_over_empty_data_keeps_the_invariant() {
        // Route 1: constructed directly.
        let direct = TensorView::new(&[], Vec::new(), Vec::new(), 0);
        assert!(direct.is_contiguous());
        let out = direct.to_tensor();
        assert_eq!(out.shape, Vec::<usize>::new());
        assert_eq!(out.data.len(), 1, "shape [] implies exactly 1 element");
        assert_eq!(out.data, vec![0.0]);

        // Route 2: rank-reduced off a zero-size dimension.
        let empty = Tensor::zeros(&[0]);
        let picked = empty.view().select(0, 0);
        assert!(picked.is_rank0());
        let out = picked.to_tensor();
        assert_eq!(out.shape, Vec::<usize>::new());
        assert_eq!(out.data.len(), 1);
    }

    /// The same invariant for a higher-rank view whose offset walks it past the
    /// end of the backing slice: the tail is zero-filled to the declared shape
    /// rather than silently producing a short-data tensor.
    #[test]
    fn to_tensor_zero_fills_an_unreachable_tail() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        // Offset 3 with a length-3 shape reaches elements 3, 4, 5 -- only the
        // first exists.
        let over = TensorView::new(&t.data, vec![3], vec![1], 3);
        let out = over.to_tensor();
        assert_eq!(out.shape, vec![3]);
        assert_eq!(
            out.data.len(),
            3,
            "invariant: data.len() == shape.product()"
        );
        assert_eq!(out.data, vec![4.0, 0.0, 0.0]);
    }
}
