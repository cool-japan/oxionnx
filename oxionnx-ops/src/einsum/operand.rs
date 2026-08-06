//! Strided operand view used by every einsum execution path.
//!
//! An [`Operand`] is a *label-indexed* strided view over a flat `f32` buffer.
//! Expressing operands this way collapses three separate einsum features into
//! one representation, so no execution path has to special-case any of them:
//!
//! | feature | encoding |
//! |---|---|
//! | axis permutation | choose the axis order when materialising |
//! | diagonal (`ii`)  | one label whose stride is the **sum** of the repeated axes' strides |
//! | broadcast (extent 1 against extent *n*, incl. ellipsis) | stride `0` with the label's full extent |
//!
//! Because a diagonal is a stride fold, `ii->i` never copies the off-diagonal
//! elements, and a broadcast axis never materialises its repeats until (and
//! unless) some path needs a contiguous buffer.

use oxionnx_core::Tensor;
use std::borrow::Cow;

/// Multiply `dims` together, reporting overflow instead of wrapping.
///
/// `[usize]::iter().product()` panics in debug builds and silently wraps in
/// release builds, either of which a malformed model could trigger with a
/// crafted shape; this returns a typed error on both.
pub(crate) fn checked_product(dims: &[usize]) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("einsum: element count of shape {dims:?} overflows usize"))
}

/// `checked_product` in `u128`, for cost estimates that may legitimately exceed
/// `usize` without being an error.
pub(crate) fn wide_product(dims: &[usize]) -> u128 {
    dims.iter()
        .fold(1u128, |acc, &d| acc.saturating_mul(d as u128))
}

/// True when `strides` is the row-major (C order) stride vector for `dims`.
///
/// Axes of extent 1 are ignored: their stride is never added to any offset, so
/// a broadcast placeholder stride of `0` does not make a buffer non-contiguous.
fn is_c_contiguous(dims: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1usize;
    for i in (0..dims.len()).rev() {
        if dims[i] != 1 && strides[i] != expected {
            return false;
        }
        expected = expected.saturating_mul(dims[i]);
    }
    true
}

/// A label-indexed strided view over a flat buffer.
#[derive(Debug, Clone)]
pub(crate) struct Operand<'a> {
    /// Distinct labels, one per axis of this view.
    pub labels: Vec<usize>,
    /// Extent of each axis, already broadcast to the label's global extent.
    pub dims: Vec<usize>,
    /// Element (not byte) stride of each axis; `0` marks a broadcast axis.
    pub strides: Vec<usize>,
    /// Backing storage: borrowed for graph inputs, owned for intermediates.
    pub data: Cow<'a, [f32]>,
}

impl<'a> Operand<'a> {
    /// Build a view of `tensor` under `subs` (one label per tensor axis).
    ///
    /// Repeated labels are folded into a single diagonal axis and axes of
    /// extent 1 whose label is wider are turned into stride-0 broadcasts.
    ///
    /// # Errors
    /// Returns an error if `tensor`'s data length disagrees with its shape (a
    /// possibility because [`Tensor::new`] only checks that in debug builds),
    /// if a repeated label's extents differ, or if an axis is neither the
    /// label's extent nor 1.
    pub(crate) fn from_tensor(
        tensor: &'a Tensor,
        subs: &[usize],
        label_sizes: &[usize],
    ) -> Result<Self, String> {
        let expected = checked_product(&tensor.shape)?;
        if tensor.data.len() != expected {
            return Err(format!(
                "einsum: input tensor has {} elements but shape {:?} implies {expected}",
                tensor.data.len(),
                tensor.shape
            ));
        }
        if subs.len() != tensor.shape.len() {
            return Err(format!(
                "einsum: subscript has {} labels but tensor has {} dims",
                subs.len(),
                tensor.shape.len()
            ));
        }

        let ndim = tensor.shape.len();
        let mut raw_strides = vec![1usize; ndim];
        for axis in (0..ndim.saturating_sub(1)).rev() {
            raw_strides[axis] = raw_strides[axis + 1].saturating_mul(tensor.shape[axis + 1]);
        }

        let mut labels: Vec<usize> = Vec::with_capacity(ndim);
        let mut dims: Vec<usize> = Vec::with_capacity(ndim);
        let mut strides: Vec<usize> = Vec::with_capacity(ndim);
        // label -> axis index in this view, or `usize::MAX` when absent.
        let mut axis_of: Vec<usize> = vec![usize::MAX; label_sizes.len()];
        for (axis, &label) in subs.iter().enumerate() {
            if label >= axis_of.len() {
                return Err(format!("einsum: label index {label} out of range"));
            }
            let dim = tensor.shape[axis];
            let existing = axis_of[label];
            if existing == usize::MAX {
                axis_of[label] = labels.len();
                labels.push(label);
                dims.push(dim);
                strides.push(raw_strides[axis]);
            } else {
                if dims[existing] != dim {
                    return Err(format!(
                        "einsum: dimensions for a repeated label don't match ({} != {dim})",
                        dims[existing]
                    ));
                }
                // Walking one step along the fused axis advances every folded
                // axis by one, i.e. it walks the diagonal.
                strides[existing] = strides[existing].saturating_add(raw_strides[axis]);
            }
        }

        for i in 0..labels.len() {
            let want = label_sizes[labels[i]];
            if dims[i] != want {
                if dims[i] == 1 {
                    dims[i] = want;
                    strides[i] = 0;
                } else {
                    return Err(format!(
                        "einsum: axis extent {} is neither 1 nor the label extent {want}",
                        dims[i]
                    ));
                }
            }
        }

        Ok(Self {
            labels,
            dims,
            strides,
            data: Cow::Borrowed(&tensor.data),
        })
    }

    /// Build an operand over a freshly computed contiguous buffer.
    pub(crate) fn from_owned(labels: Vec<usize>, dims: Vec<usize>, data: Vec<f32>) -> Self {
        let mut strides = vec![1usize; dims.len()];
        for axis in (0..dims.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1].saturating_mul(dims[axis + 1]);
        }
        Self {
            labels,
            dims,
            strides,
            data: Cow::Owned(data),
        }
    }

    /// Axis index holding `label`, if this operand has it.
    pub(crate) fn axis_of(&self, label: usize) -> Option<usize> {
        self.labels.iter().position(|&l| l == label)
    }

    /// Materialise a contiguous C-order buffer whose axes are `keep_axes` in
    /// that order, summing over `reduce_axes`.
    ///
    /// `keep_axes` and `reduce_axes` together must be a permutation of this
    /// operand's axes. Summing an axis here is valid whenever that axis's label
    /// appears in no other operand and not in the output, because
    /// `sum_l (a[..,l] * b[..]) == (sum_l a[..,l]) * b[..]`.
    pub(crate) fn materialize(
        &self,
        keep_axes: &[usize],
        reduce_axes: &[usize],
    ) -> Result<Vec<f32>, String> {
        let pick = |axes: &[usize]| -> Result<(Vec<usize>, Vec<usize>), String> {
            let mut d = Vec::with_capacity(axes.len());
            let mut s = Vec::with_capacity(axes.len());
            for &axis in axes {
                let dim = *self
                    .dims
                    .get(axis)
                    .ok_or_else(|| format!("einsum: axis {axis} out of range"))?;
                let stride = *self
                    .strides
                    .get(axis)
                    .ok_or_else(|| format!("einsum: axis {axis} out of range"))?;
                d.push(dim);
                s.push(stride);
            }
            Ok((d, s))
        };
        let (keep_dims, keep_strides) = pick(keep_axes)?;
        let (reduce_dims, reduce_strides) = pick(reduce_axes)?;

        let out_len = checked_product(&keep_dims)?;
        let mut out = vec![0.0f32; out_len];
        gather_into(
            &keep_dims,
            &keep_strides,
            &reduce_dims,
            &reduce_strides,
            &self.data,
            &mut out,
        )?;
        Ok(out)
    }
}

/// Largest element offset any odometer walk over `dims`/`strides` can produce.
fn max_offset(dims: &[usize], strides: &[usize]) -> Result<usize, String> {
    let mut total = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim == 0 {
            return Ok(0);
        }
        let span = (dim - 1)
            .checked_mul(stride)
            .ok_or_else(|| "einsum: strided offset overflows usize".to_string())?;
        total = total
            .checked_add(span)
            .ok_or_else(|| "einsum: strided offset overflows usize".to_string())?;
    }
    Ok(total)
}

/// Core gather: walk `keep` axes in order writing one output element each,
/// accumulating over the `reduce` axes.
fn gather_into(
    keep_dims: &[usize],
    keep_strides: &[usize],
    reduce_dims: &[usize],
    reduce_strides: &[usize],
    data: &[f32],
    out: &mut [f32],
) -> Result<(), String> {
    if out.is_empty() {
        return Ok(());
    }
    let reduce_len = checked_product(reduce_dims)?;
    if reduce_len == 0 {
        // Summing over an empty axis yields zero, which `out` already holds.
        return Ok(());
    }

    // Bounds are proven once here so the inner loops can index directly: every
    // offset the odometers produce is a sub-sum of `max_offset`.
    let reach = max_offset(keep_dims, keep_strides)?
        .checked_add(max_offset(reduce_dims, reduce_strides)?)
        .ok_or_else(|| "einsum: strided offset overflows usize".to_string())?;
    if reach >= data.len() {
        return Err(format!(
            "einsum: strided access reaches element {reach} of a {}-element buffer",
            data.len()
        ));
    }

    if reduce_dims.is_empty() && is_c_contiguous(keep_dims, keep_strides) {
        let n = out.len();
        if data.len() < n {
            return Err("einsum: source buffer shorter than the gathered result".to_string());
        }
        out.copy_from_slice(&data[..n]);
        return Ok(());
    }

    let mut keep_index = vec![0usize; keep_dims.len()];
    let mut reduce_index = vec![0usize; reduce_dims.len()];
    let mut base = 0usize;
    for slot in out.iter_mut() {
        if reduce_dims.is_empty() {
            *slot = data[base];
        } else {
            reduce_index.iter_mut().for_each(|v| *v = 0);
            let mut offset = base;
            let mut acc = 0.0f32;
            for _ in 0..reduce_len {
                acc += data[offset];
                for axis in (0..reduce_dims.len()).rev() {
                    reduce_index[axis] += 1;
                    offset += reduce_strides[axis];
                    if reduce_index[axis] < reduce_dims[axis] {
                        break;
                    }
                    offset -= reduce_strides[axis] * reduce_dims[axis];
                    reduce_index[axis] = 0;
                }
            }
            *slot = acc;
        }
        for axis in (0..keep_dims.len()).rev() {
            keep_index[axis] += 1;
            base += keep_strides[axis];
            if keep_index[axis] < keep_dims[axis] {
                break;
            }
            base -= keep_strides[axis] * keep_dims[axis];
            keep_index[axis] = 0;
        }
    }
    Ok(())
}
