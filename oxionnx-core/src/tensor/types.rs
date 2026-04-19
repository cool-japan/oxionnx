//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{broadcast_strides, compute_strides};

/// A read-only view into a tensor's data with stride-based indexing.
///
/// Enables zero-copy transpose, slice, squeeze, and unsqueeze operations
/// by manipulating shape, strides, and offset without copying data.
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
    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
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
    pub fn transpose(&self, perm: &[usize]) -> Self {
        let new_shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| self.strides[p]).collect();
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }
    /// Slice along one axis: select a range `[start, end)` along the given axis.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;
        Self {
            data: self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[axis],
        }
    }
    /// Select a single index along an axis, reducing rank by 1.
    pub fn select(&self, axis: usize, index: usize) -> Self {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(axis);
        new_strides.remove(axis);
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset + index * self.strides[axis],
        }
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
            let pos = ax;
            let stride_val = if pos + 1 - offset < self.strides.len() {
                self.strides[pos + 1 - offset].max(1)
            } else {
                1
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
    pub fn to_tensor(&self) -> Tensor {
        if self.is_contiguous() {
            let n = self.numel();
            let data = self.data[..n].to_vec();
            return Tensor::new(data, self.shape.clone());
        }
        let data: Vec<f32> = self.iter().collect();
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
    /// Panics (debug-only) if `data.len() != shape.product()`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }
    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0f32; n],
            shape: shape.to_vec(),
        }
    }
    /// Create a scalar tensor (shape `[1]`) containing a single value.
    pub fn scalar(val: f32) -> Self {
        Self {
            data: vec![val],
            shape: vec![1],
        }
    }
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    /// Number of dimensions (rank) of this tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// Return a new tensor with the same data but a different shape.
    /// Panics if the element count changes.
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
    /// Compute the broadcast shape of two tensors (NumPy rules).
    /// Returns Err if shapes are incompatible.
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
