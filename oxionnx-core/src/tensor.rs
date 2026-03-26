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

/// Convert a tensor from NCHW to NHWC layout.
/// Input shape: [N, C, H, W] -> Output shape: [N, H, W, C]
pub fn nchw_to_nhwc(tensor: &Tensor) -> Result<Tensor, String> {
    if tensor.shape.len() != 4 {
        return Err(format!(
            "nchw_to_nhwc: expected 4D tensor, got {}D",
            tensor.shape.len()
        ));
    }
    let (n, c, h, w) = (
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    );
    let mut out = vec![0.0f32; tensor.data.len()];

    for batch in 0..n {
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let src_idx = batch * c * h * w + ch * h * w + row * w + col;
                    let dst_idx = batch * h * w * c + row * w * c + col * c + ch;
                    out[dst_idx] = tensor.data[src_idx];
                }
            }
        }
    }

    Ok(Tensor::new(out, vec![n, h, w, c]))
}

/// Convert a tensor from NHWC to NCHW layout.
/// Input shape: [N, H, W, C] -> Output shape: [N, C, H, W]
pub fn nhwc_to_nchw(tensor: &Tensor) -> Result<Tensor, String> {
    if tensor.shape.len() != 4 {
        return Err(format!(
            "nhwc_to_nchw: expected 4D tensor, got {}D",
            tensor.shape.len()
        ));
    }
    let (n, h, w, c) = (
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    );
    let mut out = vec![0.0f32; tensor.data.len()];

    for batch in 0..n {
        for row in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    let src_idx = batch * h * w * c + row * w * c + col * c + ch;
                    let dst_idx = batch * c * h * w + ch * h * w + row * w + col;
                    out[dst_idx] = tensor.data[src_idx];
                }
            }
        }
    }

    Ok(Tensor::new(out, vec![n, c, h, w]))
}

/// Convert between tensor layouts.
pub fn convert_layout(
    tensor: &Tensor,
    from: TensorLayout,
    to: TensorLayout,
) -> Result<Tensor, String> {
    match (from, to) {
        (TensorLayout::NCHW, TensorLayout::NHWC) => nchw_to_nhwc(tensor),
        (TensorLayout::NHWC, TensorLayout::NCHW) => nhwc_to_nchw(tensor),
        (a, b) if a == b => Ok(tensor.clone()),
        _ => Err(format!(
            "Unsupported layout conversion: {:?} -> {:?}",
            from, to
        )),
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

// ---------------------------------------------------------------------------
// TensorView — zero-copy strided view
// ---------------------------------------------------------------------------

/// Compute C-order (row-major) strides from shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

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
        // Sort axes so we can insert from left to right with offset tracking.
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_unstable();

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        for (offset, &ax) in sorted_axes.iter().enumerate() {
            let pos = ax; // axes refer to positions in the *output* shape
                          // For stride of a size-1 dim, any value works; use the stride of
                          // the next dim (or 1 if at end).
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
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    indices: Vec<usize>,
    exhausted: bool,
}

impl TensorViewIter<'_> {
    fn get_at(&self, indices: &[usize]) -> Option<f32> {
        let flat_idx: usize = self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>();
        self.data.get(flat_idx).copied()
    }
}

impl Iterator for TensorViewIter<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.exhausted {
            return None;
        }
        let val = self.get_at(&self.indices);

        // Increment indices (rightmost first, carry over)
        let ndim = self.shape.len();
        let mut carry = true;
        for i in (0..ndim).rev() {
            if carry {
                self.indices[i] += 1;
                if self.indices[i] < self.shape[i] {
                    carry = false;
                } else {
                    self.indices[i] = 0;
                }
            }
        }
        if carry {
            self.exhausted = true;
        }

        val
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.exhausted {
            return (0, Some(0));
        }
        let total: usize = self.shape.iter().product();
        let mut consumed = 0usize;
        let logical_strides = compute_strides(&self.shape);
        for (i, &idx) in self.indices.iter().enumerate() {
            consumed += idx * logical_strides[i];
        }
        let remaining = total.saturating_sub(consumed);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TensorViewIter<'_> {}

// ---------------------------------------------------------------------------
// Tensor — view methods
// ---------------------------------------------------------------------------

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

// ===========================================================================

/// Build a Tensor from raw f16 little-endian bytes (ONNX `raw_data` with float16 dtype).
pub fn from_f16_bytes(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    let data: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();
    Tensor::new(data, shape)
}

/// Build a Tensor from raw f32 little-endian bytes.
pub fn from_f32_bytes(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Tensor::new(data, shape)
}

/// Build a Tensor from raw i64 little-endian bytes (index tensors).
pub fn from_i64_bytes(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    let data: Vec<f32> = bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
        .collect();
    Tensor::new(data, shape)
}

/// Build a Tensor from repeated float_data values.
pub fn from_f32_vec(floats: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::new(floats, shape)
}

// ---------------------------------------------------------------------------
// BroadcastIter — zero-allocation broadcasting iterator
// ---------------------------------------------------------------------------

/// Iterator that broadcasts two tensors together, yielding `(a_val, b_val)` pairs.
/// Does NOT allocate an expanded tensor — computes indices on the fly using strides.
pub struct BroadcastIter<'a> {
    a_data: &'a [f32],
    b_data: &'a [f32],
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
    output_shape: Vec<usize>,
    output_strides: Vec<usize>,
    total: usize,
    idx: usize,
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

impl<'a> Iterator for BroadcastIter<'a> {
    type Item = (f32, f32);

    fn next(&mut self) -> Option<(f32, f32)> {
        if self.idx >= self.total {
            return None;
        }

        // Convert flat index to multi-dimensional indices, then to source indices
        let mut a_flat = 0usize;
        let mut b_flat = 0usize;
        let mut remaining = self.idx;

        for dim in 0..self.output_shape.len() {
            let coord = remaining / self.output_strides[dim];
            remaining %= self.output_strides[dim];
            a_flat += coord * self.a_strides[dim];
            b_flat += coord * self.b_strides[dim];
        }

        self.idx += 1;
        Some((self.a_data[a_flat], self.b_data[b_flat]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BroadcastIter<'_> {}

/// Compute broadcast strides: if the original dim is 1 (broadcasted), stride is 0.
fn broadcast_strides(original_shape: &[usize], broadcast_shape: &[usize]) -> Vec<usize> {
    let ndim = broadcast_shape.len();
    let pad = ndim - original_shape.len();
    let orig_strides = compute_strides(original_shape);

    (0..ndim)
        .map(|i| {
            if i < pad {
                0 // prepended dimension, broadcast
            } else {
                let orig_idx = i - pad;
                if original_shape[orig_idx] == 1 {
                    0 // broadcast this dim
                } else {
                    orig_strides[orig_idx]
                }
            }
        })
        .collect()
}

impl Tensor {
    /// Create a broadcast iterator pairing this tensor with another.
    pub fn broadcast_iter<'a>(&'a self, other: &'a Tensor) -> Option<BroadcastIter<'a>> {
        BroadcastIter::new(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(
            Tensor::broadcast_shape(&[3, 1], &[1, 4]).expect("broadcast should succeed"),
            vec![3, 4]
        );
        assert_eq!(
            Tensor::broadcast_shape(&[1], &[4, 3]).expect("broadcast should succeed"),
            vec![4, 3]
        );
        assert!(Tensor::broadcast_shape(&[2], &[3]).is_err());
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::zeros(&[2, 3]);
        let r = t.reshape(&[6]);
        assert_eq!(r.shape, vec![6]);
    }

    // -----------------------------------------------------------------------
    // TensorView tests
    // -----------------------------------------------------------------------

    fn make_seq_tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::new(data, shape.to_vec())
    }

    #[test]
    fn test_view_basic() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.strides(), &[3, 1]);
        assert_eq!(v.ndim(), 2);
        assert_eq!(v.numel(), 6);
    }

    #[test]
    fn test_view_get() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view();
        // [0,1,2; 3,4,5]
        assert_eq!(v.get(&[0, 0]), Some(0.0));
        assert_eq!(v.get(&[0, 2]), Some(2.0));
        assert_eq!(v.get(&[1, 0]), Some(3.0));
        assert_eq!(v.get(&[1, 2]), Some(5.0));
        // out of bounds
        assert_eq!(v.get(&[2, 0]), None);
        assert_eq!(v.get(&[0]), None);
    }

    #[test]
    fn test_view_is_contiguous() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view();
        assert!(v.is_contiguous());

        let tv = v.transpose(&[1, 0]);
        assert!(!tv.is_contiguous());
    }

    #[test]
    fn test_view_transpose() {
        // Shape [2,3] -> transpose to [3,2]
        let t = make_seq_tensor(&[2, 3]); // [0,1,2,3,4,5]
        let v = t.view().transpose(&[1, 0]);
        assert_eq!(v.shape(), &[3, 2]);
        // Transposed: [[0,3],[1,4],[2,5]]
        assert_eq!(v.get(&[0, 0]), Some(0.0));
        assert_eq!(v.get(&[0, 1]), Some(3.0));
        assert_eq!(v.get(&[1, 0]), Some(1.0));
        assert_eq!(v.get(&[1, 1]), Some(4.0));
        assert_eq!(v.get(&[2, 0]), Some(2.0));
        assert_eq!(v.get(&[2, 1]), Some(5.0));
    }

    #[test]
    fn test_view_transpose_3d() {
        // Shape [2,3,4], perm [2,0,1] -> [4,2,3]
        let t = make_seq_tensor(&[2, 3, 4]);
        let v = t.view().transpose(&[2, 0, 1]);
        assert_eq!(v.shape(), &[4, 2, 3]);
        // Element at original [i,j,k] is at transposed [k,i,j]
        // Original [0,0,0]=0, [0,1,2]=6, [1,2,3]=23
        assert_eq!(v.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(v.get(&[2, 0, 1]), Some(6.0));
        assert_eq!(v.get(&[3, 1, 2]), Some(23.0));
    }

    #[test]
    fn test_view_slice() {
        // Shape [4,3], slice axis=0, [1,3) -> shape [2,3]
        let t = make_seq_tensor(&[4, 3]); // 0..12
        let v = t.view().slice(0, 1, 3);
        assert_eq!(v.shape(), &[2, 3]);
        // Row 1: [3,4,5], Row 2: [6,7,8]
        assert_eq!(v.get(&[0, 0]), Some(3.0));
        assert_eq!(v.get(&[0, 2]), Some(5.0));
        assert_eq!(v.get(&[1, 0]), Some(6.0));
        assert_eq!(v.get(&[1, 2]), Some(8.0));
    }

    #[test]
    fn test_view_select() {
        // Shape [3,4], select axis=0, index=1 -> shape [4]
        let t = make_seq_tensor(&[3, 4]); // 0..12
        let v = t.view().select(0, 1);
        assert_eq!(v.shape(), &[4]);
        assert_eq!(v.get(&[0]), Some(4.0));
        assert_eq!(v.get(&[1]), Some(5.0));
        assert_eq!(v.get(&[2]), Some(6.0));
        assert_eq!(v.get(&[3]), Some(7.0));
    }

    #[test]
    fn test_view_squeeze() {
        // Shape [1,3,1,4] -> squeeze axes [0,2] -> [3,4]
        let t = make_seq_tensor(&[1, 3, 1, 4]);
        let v = t.view().squeeze(&[0, 2]);
        assert_eq!(v.shape(), &[3, 4]);
        assert_eq!(v.numel(), 12);
        assert_eq!(v.get(&[0, 0]), Some(0.0));
        assert_eq!(v.get(&[2, 3]), Some(11.0));
    }

    #[test]
    fn test_view_unsqueeze() {
        // Shape [3,4] -> unsqueeze axis 0 -> [1,3,4]
        let t = make_seq_tensor(&[3, 4]);
        let v = t.view().unsqueeze(&[0]);
        assert_eq!(v.shape(), &[1, 3, 4]);
        assert_eq!(v.numel(), 12);
        assert_eq!(v.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(v.get(&[0, 2, 3]), Some(11.0));
    }

    #[test]
    fn test_view_to_tensor() {
        // Transpose then materialize
        let t = make_seq_tensor(&[2, 3]); // [0,1,2,3,4,5]
        let v = t.view().transpose(&[1, 0]); // [3,2]
        let mat = v.to_tensor();
        assert_eq!(mat.shape, vec![3, 2]);
        // Transposed row-major: [0,3,1,4,2,5]
        assert_eq!(mat.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_view_iter() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view().transpose(&[1, 0]); // [3,2]
        let elems: Vec<f32> = v.iter().collect();
        assert_eq!(elems, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_view_chained_ops() {
        // Shape [4,6]: transpose to [6,4], then slice axis=0 [1,4) -> [3,4]
        let t = make_seq_tensor(&[4, 6]);
        let v = t.view().transpose(&[1, 0]).slice(0, 1, 4);
        assert_eq!(v.shape(), &[3, 4]);
        let mat = v.to_tensor();
        assert_eq!(mat.shape, vec![3, 4]);
        // Original layout row-major [4,6]:
        //   row0: 0..6, row1: 6..12, row2: 12..18, row3: 18..24
        // Transposed [6,4]: row i of transposed = column i of original
        //   trow0: [0,6,12,18], trow1: [1,7,13,19], ...
        // Slice axis=0 [1,4): trow1,trow2,trow3
        assert_eq!(
            mat.data,
            vec![1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0,]
        );
    }

    // -----------------------------------------------------------------------
    // BroadcastIter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_broadcast_iter_same_shape() {
        // [2,3] x [2,3] — no actual broadcasting, just element-wise pairs
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[2, 3]);
        assert_eq!(iter.len(), 6);
        assert!(!iter.is_empty());
        let pairs: Vec<(f32, f32)> = iter.collect();
        assert_eq!(
            pairs,
            vec![
                (1.0, 10.0),
                (2.0, 20.0),
                (3.0, 30.0),
                (4.0, 40.0),
                (5.0, 50.0),
                (6.0, 60.0),
            ]
        );
    }

    #[test]
    fn test_broadcast_iter_scalar() {
        // [2,3] x [1] — scalar broadcast
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![100.0], vec![1]);
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[2, 3]);
        let pairs: Vec<(f32, f32)> = iter.collect();
        for (i, (av, bv)) in pairs.iter().enumerate() {
            assert!((*av - (i as f32 + 1.0)).abs() < 1e-6);
            assert!((*bv - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_broadcast_iter_row_col() {
        // [3,1] x [1,4] -> [3,4]
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]);
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[3, 4]);
        assert_eq!(iter.len(), 12);
        let pairs: Vec<(f32, f32)> = iter.collect();
        // Row 0: a=1, b cycles [10,20,30,40]
        // Row 1: a=2, b cycles [10,20,30,40]
        // Row 2: a=3, b cycles [10,20,30,40]
        let expected = vec![
            (1.0, 10.0),
            (1.0, 20.0),
            (1.0, 30.0),
            (1.0, 40.0),
            (2.0, 10.0),
            (2.0, 20.0),
            (2.0, 30.0),
            (2.0, 40.0),
            (3.0, 10.0),
            (3.0, 20.0),
            (3.0, 30.0),
            (3.0, 40.0),
        ];
        assert_eq!(pairs, expected);
    }

    #[test]
    fn test_broadcast_iter_3d() {
        // [2,1,4] x [1,3,4] -> [2,3,4]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 1, 4]);
        let b = Tensor::new(
            vec![
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
            ],
            vec![1, 3, 4],
        );
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[2, 3, 4]);
        assert_eq!(iter.len(), 24);

        let pairs: Vec<(f32, f32)> = iter.collect();
        // At [0,0,0]: a[0,0,0]=1, b[0,0,0]=10
        assert_eq!(pairs[0], (1.0, 10.0));
        // At [0,1,0]: a[0,0,0]=1 (dim 1 broadcast), b[0,1,0]=50
        assert_eq!(pairs[4], (1.0, 50.0));
        // At [1,0,0]: a[1,0,0]=5, b[0,0,0]=10 (dim 0 broadcast)
        assert_eq!(pairs[12], (5.0, 10.0));
        // At [1,2,3]: a[1,0,3]=8, b[0,2,3]=120
        assert_eq!(pairs[23], (8.0, 120.0));
    }

    #[test]
    fn test_broadcast_iter_incompatible() {
        // [2,3] x [4,3] — incompatible
        let a = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let b = Tensor::new(vec![1.0; 12], vec![4, 3]);
        assert!(BroadcastIter::new(&a, &b).is_none());
    }

    // -----------------------------------------------------------------------
    // Layout conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nchw_to_nhwc() {
        // [1,2,3,4] tensor: N=1, C=2, H=3, W=4
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(data, vec![1, 2, 3, 4]);
        let nhwc = nchw_to_nhwc(&t).expect("conversion should succeed");
        assert_eq!(nhwc.shape, vec![1, 3, 4, 2]);
        // NCHW [0,0,0,0]=0 -> NHWC [0,0,0,0]=0
        assert!((nhwc.data[0] - 0.0).abs() < 1e-6);
        // NCHW [0,1,0,0]=12 -> NHWC [0,0,0,1]=12
        assert!((nhwc.data[1] - 12.0).abs() < 1e-6);
        // NCHW [0,0,0,1]=1 -> NHWC [0,0,1,0]=1
        assert!((nhwc.data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nhwc_to_nchw() {
        // Build NHWC [1,3,4,2] and convert back
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(data, vec![1, 3, 4, 2]);
        let nchw = nhwc_to_nchw(&t).expect("conversion should succeed");
        assert_eq!(nchw.shape, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_layout_roundtrip() {
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let original = Tensor::new(data.clone(), vec![2, 3, 2, 4]);
        let nhwc = nchw_to_nhwc(&original).expect("nchw_to_nhwc");
        let back = nhwc_to_nchw(&nhwc).expect("nhwc_to_nchw");
        assert_eq!(back.shape, original.shape);
        assert_eq!(back.data, original.data);
    }

    #[test]
    fn test_convert_layout_same() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let result =
            convert_layout(&t, TensorLayout::NCHW, TensorLayout::NCHW).expect("same layout");
        assert_eq!(result.data, t.data);
        assert_eq!(result.shape, t.shape);
    }

    #[test]
    fn test_non_4d_error() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(nchw_to_nhwc(&t).is_err());
        assert!(nhwc_to_nchw(&t).is_err());

        let t3d = Tensor::new(vec![1.0; 12], vec![2, 3, 2]);
        assert!(nchw_to_nhwc(&t3d).is_err());
    }
}
