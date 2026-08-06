//! Platform-neutral DirectML tensor-descriptor layout math.
//!
//! Everything DirectML needs in order to describe an f32 buffer tensor — `Sizes[]`,
//! `Strides[]`, `TotalTensorSizeInBytes` — is computed **here**, on every platform,
//! and unit-tested on Linux.  `backend/dml/tensor.rs` does nothing but memcpy these
//! numbers into a `DML_BUFFER_TENSOR_DESC`; it never recomputes any of them.
//!
//! # The `TotalTensorSizeInBytes` hazard
//!
//! It is **not** `product(sizes) * 4`.  It is
//!
//! ```text
//! (1 + Σᵢ (sizes[i] - 1) * strides[i]) * 4      rounded up to a multiple of 4
//! ```
//!
//! — the true memory footprint *given the strides*.  For a packed tensor the two
//! formulas agree, which is exactly why the wrong one survives review.  For a
//! **broadcast (0-stride)** tensor they diverge enormously: a `[1, 4]` operand
//! broadcast to `[2, 3, 4]` occupies 16 bytes, not 96, because DirectML reads the
//! *original, un-expanded* buffer through the 0-strides.  Declaring 96 makes
//! DirectML either reject the binding outright or read 80 bytes past the end of a
//! 16-byte allocation.  Neither is caught by rustc, by clippy, or by any test that
//! can run without a GPU — so it is caught *here*, by
//! `broadcast_to_total_bytes_is_the_source_size`, on Linux.
//!
//! Nothing in this file has a `#[cfg]`.

use crate::error::{DirectMLError, Result};
use crate::plan::{
    align_up, checked_u32, numel, BinaryOp, ConvPlan, ElementwisePlan, MatMulPlan, ReduceKind,
    ReducePlan, SoftmaxPlan, UnaryOp, DML_TENSOR_SIZE_GRANULARITY, ELEM_SIZE,
};

/// Every tensor this crate hands DirectML is normalised to rank 4 (`[N, C, H, W]`).
///
/// `DML_TENSOR_DIMENSION_COUNT_MAX` is 5, but `DML_OPERATOR_GEMM` requires exactly
/// 4, and 4 is accepted by every operator this crate uses — so 4 it is, uniformly.
/// Shorter shapes are **left-padded with 1s**; anything longer is
/// [`DirectMLError::Declined`].
pub const DML_RANK: usize = 4;

/// The exact numbers that go into one `DML_BUFFER_TENSOR_DESC` for an f32 tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DmlTensorLayout {
    /// `DML_BUFFER_TENSOR_DESC::Sizes` — always [`DML_RANK`] entries.
    pub sizes: [u32; DML_RANK],
    /// `DML_BUFFER_TENSOR_DESC::Strides`, in **elements** (not bytes).
    ///
    /// A `0` entry means "broadcast along this axis".  DirectML supports that
    /// natively, and it is how the DML backend broadcasts without copying a single
    /// byte.
    pub strides: [u32; DML_RANK],
    /// `true` when [`Self::strides`] is exactly the packed C-contiguous stride
    /// vector for [`Self::sizes`], in which case DirectML permits
    /// `Strides = nullptr` — its fast path.
    pub is_packed: bool,
    /// `DML_BUFFER_TENSOR_DESC::TotalTensorSizeInBytes`.
    ///
    /// See this module's documentation.  This is the stride-aware footprint, **not**
    /// `product(sizes) * 4`.
    pub total_bytes: u64,
}

impl DmlTensorLayout {
    /// Packed (C-contiguous) rank-4 layout from an ONNX shape, left-padded with 1s.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `shape.len() > DML_RANK`, when any dim or
    /// the element count exceeds `u32::MAX`, or when the element count is 0.
    pub fn packed(shape: &[usize]) -> Result<Self> {
        let sizes = pad_to_rank4(shape)?;
        let strides = Self::packed_strides(&sizes)?;
        let total_bytes = total_tensor_size_in_bytes(&sizes, &strides)?;
        Ok(Self {
            sizes,
            strides,
            is_packed: true,
            total_bytes,
        })
    }

    /// A rank-4 layout that numpy-broadcasts `shape` up to `target`, using
    /// **0-strides** on every axis where the source's (padded) dim is 1.
    ///
    /// The descriptor reads the *original, un-expanded* buffer, so
    /// [`Self::total_bytes`] is the **source's** footprint, not the target's.
    ///
    /// Convention (pinned by the tests): a stride is `0` whenever the source's
    /// padded dim is 1; otherwise it is the source's own packed stride, and the
    /// source's dim must equal the target's.
    ///
    /// # Errors
    /// [`DirectMLError::ShapeMismatch`] when `shape` does not broadcast to `target`.
    /// [`DirectMLError::Declined`] on the rank / size limits of [`Self::packed`].
    pub fn broadcast_to(shape: &[usize], target: &[usize]) -> Result<Self> {
        let src_sizes = pad_to_rank4(shape)?;
        let dst_sizes = pad_to_rank4(target)?;
        let src_strides = Self::packed_strides(&src_sizes)?;

        let mut strides = [0u32; DML_RANK];
        for i in 0..DML_RANK {
            if src_sizes[i] == 1 {
                // Broadcast (or a genuinely size-1 axis, where a 0 stride is
                // harmless: the footprint formula multiplies it by `size - 1 == 0`).
                strides[i] = 0;
            } else if src_sizes[i] == dst_sizes[i] {
                strides[i] = src_strides[i];
            } else {
                return Err(DirectMLError::ShapeMismatch(format!(
                    "cannot broadcast {shape:?} to {target:?} (rank-4 axis {i}: {} vs {})",
                    src_sizes[i], dst_sizes[i]
                )));
            }
        }

        let packed = Self::packed_strides(&dst_sizes)?;
        let total_bytes = total_tensor_size_in_bytes(&dst_sizes, &strides)?;
        Ok(Self {
            sizes: dst_sizes,
            strides,
            is_packed: strides == packed,
            total_bytes,
        })
    }

    /// Collapse an arbitrary-rank matrix operand into rank 4 by folding every
    /// leading dim into the batch axis: `[d₀ … dₙ₋₂, M, N] → [batch, 1, M, N]`.
    ///
    /// This is the projection `DML_OPERATOR_GEMM` needs: it wants exactly rank 4,
    /// with the matrix in the trailing two axes.
    ///
    /// `batch_stride_zero` forces `strides[0] = 0`, which expresses a
    /// batch-broadcast operand (the [`MatMulPlan::a_batch_stride`] `== 0` case)
    /// without copying anything.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `shape` has fewer than 2 dims, when `batch`
    /// is 0, or on the usual size limits.
    pub fn collapse_to_batched_matrix(
        shape: &[usize],
        batch: u32,
        batch_stride_zero: bool,
    ) -> Result<Self> {
        if shape.len() < 2 {
            return Err(DirectMLError::Declined(format!(
                "collapse_to_batched_matrix: {shape:?} has rank < 2"
            )));
        }
        if batch == 0 {
            return Err(DirectMLError::Declined(
                "collapse_to_batched_matrix: batch is 0".into(),
            ));
        }
        let rows = checked_u32(shape[shape.len() - 2], "matrix rows")?;
        let cols = checked_u32(shape[shape.len() - 1], "matrix cols")?;
        if rows == 0 || cols == 0 {
            return Err(DirectMLError::Declined(format!(
                "collapse_to_batched_matrix: empty matrix {rows} x {cols}"
            )));
        }

        let sizes = [batch, 1, rows, cols];
        let packed = Self::packed_strides(&sizes)?;
        let mut strides = packed;
        if batch_stride_zero {
            strides[0] = 0;
        }
        let total_bytes = total_tensor_size_in_bytes(&sizes, &strides)?;
        Ok(Self {
            sizes,
            strides,
            is_packed: strides == packed,
            total_bytes,
        })
    }

    /// Packed C-contiguous strides, in elements, for `sizes`.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when a stride would overflow `u32` — i.e. when
    /// the tensor has more than `u32::MAX` elements, which DirectML cannot index
    /// anyway.
    ///
    /// (The design document typed this as infallible.  It cannot be: `sizes` is
    /// four independent `u32`s, whose product need not fit in a `u32`, and a
    /// silently wrapped stride is exactly the class of bug this module exists to
    /// prevent.)
    pub fn packed_strides(sizes: &[u32; DML_RANK]) -> Result<[u32; DML_RANK]> {
        let mut strides = [0u32; DML_RANK];
        let mut acc: u64 = 1;
        for i in (0..DML_RANK).rev() {
            strides[i] = u32::try_from(acc).map_err(|_| {
                DirectMLError::Declined(format!("packed stride for sizes {sizes:?} exceeds u32"))
            })?;
            acc *= u64::from(sizes[i]);
        }
        // `acc` is now the element count; it must be addressable too.
        u32::try_from(acc).map_err(|_| {
            DirectMLError::Declined(format!(
                "sizes {sizes:?} hold {acc} elements, which exceeds u32::MAX"
            ))
        })?;
        Ok(strides)
    }

    /// The element count implied by [`Self::sizes`].
    #[must_use]
    pub fn elem_count(&self) -> u64 {
        self.sizes.iter().map(|&s| u64::from(s)).product()
    }
}

/// Left-pad an ONNX shape with 1s to exactly [`DML_RANK`] dims.
///
/// # Errors
/// [`DirectMLError::Declined`] when the rank is above [`DML_RANK`], when any dim
/// exceeds `u32::MAX`, or when the tensor is empty.
fn pad_to_rank4(shape: &[usize]) -> Result<[u32; DML_RANK]> {
    if shape.len() > DML_RANK {
        return Err(DirectMLError::Declined(format!(
            "rank {} of shape {shape:?} exceeds DML_RANK = {DML_RANK}",
            shape.len()
        )));
    }
    if numel(shape)? == 0 {
        return Err(DirectMLError::Declined(format!(
            "empty tensor {shape:?}; DirectML cannot describe a 0-element buffer"
        )));
    }
    let mut sizes = [1u32; DML_RANK];
    let pad = DML_RANK - shape.len();
    for (i, &d) in shape.iter().enumerate() {
        sizes[pad + i] = checked_u32(d, "tensor dimension")?;
    }
    Ok(sizes)
}

/// `DML_BUFFER_TENSOR_DESC::TotalTensorSizeInBytes`, computed the way DirectML
/// documents it.
///
/// **Not** `product(sizes) * ELEM_SIZE`.  See this module's documentation.
///
/// # Errors
/// [`DirectMLError::Declined`] on `u64` overflow, or when the rounded size cannot
/// be represented.
fn total_tensor_size_in_bytes(sizes: &[u32; DML_RANK], strides: &[u32; DML_RANK]) -> Result<u64> {
    let mut index_of_last_element: u64 = 0;
    for i in 0..DML_RANK {
        // `(sizes[i] - 1) * strides[i]`.  A 0-stride axis contributes nothing,
        // which is the whole point: a broadcast axis costs no memory.
        let extent = u64::from(sizes[i].saturating_sub(1)) * u64::from(strides[i]);
        index_of_last_element = index_of_last_element.checked_add(extent).ok_or_else(|| {
            DirectMLError::Declined(format!(
                "tensor footprint for sizes {sizes:?} strides {strides:?} overflows u64"
            ))
        })?;
    }
    let elems = index_of_last_element
        .checked_add(1)
        .ok_or_else(|| DirectMLError::Declined("tensor footprint overflows u64".into()))?;
    let bytes = elems
        .checked_mul(ELEM_SIZE as u64)
        .ok_or_else(|| DirectMLError::Declined("tensor byte size overflows u64".into()))?;

    // A multiple of 4 already (we just multiplied by 4), but round explicitly so
    // that a future non-f32 dtype cannot quietly violate the granularity rule.
    let rounded = align_up(
        usize::try_from(bytes)
            .map_err(|_| DirectMLError::Declined("tensor byte size exceeds usize".into()))?,
        DML_TENSOR_SIZE_GRANULARITY,
    )
    .ok_or_else(|| DirectMLError::Declined("tensor byte size rounding overflows".into()))?;
    Ok(rounded as u64)
}

/// The four rank-4 layouts for one DirectML GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DmlGemmLayout {
    /// The `A` operand, described **as stored** (see [`Self::from_plan`]).
    pub a: DmlTensorLayout,
    /// The `B` operand, described **as stored**.
    pub b: DmlTensorLayout,
    /// The optional `C` bias, broadcast against the output's `[.., .., m, n]`.
    pub c: Option<DmlTensorLayout>,
    /// The `[batch, 1, m, n]` output.
    pub output: DmlTensorLayout,
}

impl DmlGemmLayout {
    /// Derive the four descriptors from a [`MatMulPlan`].
    ///
    /// `DML_OPERATOR_GEMM` wants `A: [batch, 1, ?, ?]`, `B: [batch, 1, ?, ?]`,
    /// `Output: [batch, 1, M, N]`.
    ///
    /// **`sizes` describe the buffer *as it sits in memory*.**
    /// `DML_GEMM_OPERATOR_DESC::TransA` / `TransB` tell DirectML to *interpret* it
    /// transposed; they do not change what is stored.  Since [`MatMulPlan`]'s
    /// `m`/`k`/`n` are the **post**-transpose logical dims:
    ///
    /// | | `trans_a == false` | `trans_a == true` |
    /// |-|-|-|
    /// | `a.sizes` | `[batch, 1, m, k]` | `[batch, 1, k, m]` |
    ///
    /// and symmetrically for `B` with `trans_b`.  Deriving `a.sizes` from `m`/`k`
    /// instead of from `a_stored_shape` would hand DirectML a transposed
    /// description of an untransposed buffer — a wrong answer, not an error.  So
    /// this reads `plan.a_stored_shape` / `plan.b_stored_shape` directly.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] or [`DirectMLError::ShapeMismatch`] from the
    /// underlying [`DmlTensorLayout`] constructors.
    pub fn from_plan(plan: &MatMulPlan) -> Result<Self> {
        let a = DmlTensorLayout::collapse_to_batched_matrix(
            &plan.a_stored_shape,
            plan.batch,
            plan.a_batch_stride == 0,
        )?;
        let b = DmlTensorLayout::collapse_to_batched_matrix(
            &plan.b_stored_shape,
            plan.batch,
            plan.b_batch_stride == 0,
        )?;
        let output = DmlTensorLayout::collapse_to_batched_matrix(
            &[plan.m as usize, plan.n as usize],
            plan.batch,
            false,
        )?;

        let c = match plan.c_shape.as_ref() {
            Some(cs) if plan.has_bias() => {
                // `C` is broadcast against the *output*, so its rank-4 target is the
                // output's own sizes.  A row-vector bias `[n]` becomes
                // `[batch, 1, m, n]` with strides `[0, 0, 0, 1]` — 4·n bytes, read
                // once per row, copied never.
                let target = [
                    plan.batch as usize,
                    1usize,
                    plan.m as usize,
                    plan.n as usize,
                ];
                Some(DmlTensorLayout::broadcast_to(cs, &target)?)
            }
            _ => None,
        };

        Ok(Self { a, b, c, output })
    }
}

/// The layouts for one DirectML elementwise op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DmlElementwiseLayout {
    /// The `A` operand.
    pub a: DmlTensorLayout,
    /// The `B` operand; `None` for unary ops.
    pub b: Option<DmlTensorLayout>,
    /// The output.
    pub output: DmlTensorLayout,
}

impl DmlElementwiseLayout {
    /// Derive from an [`ElementwisePlan`].
    ///
    /// An operand that already matches the output shape gets a **packed** layout
    /// (DirectML's `Strides = nullptr` fast path).  An operand that needs
    /// broadcasting gets 0-strides via [`DmlTensorLayout::broadcast_to`] — the
    /// DirectML path never CPU-expands.  In this wave
    /// [`ElementwisePlan::binary`] declines every non-identical pair, so the
    /// broadcast arm is exercised only by [`DmlGemmLayout`]'s bias and by this
    /// module's tests; it is kept correct and tested so that lifting the
    /// restriction is a one-line change in `plan.rs`, not a redesign.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] or [`DirectMLError::ShapeMismatch`] from the
    /// underlying [`DmlTensorLayout`] constructors.
    pub fn from_plan(plan: &ElementwisePlan) -> Result<Self> {
        let a = Self::operand(&plan.a_shape, &plan.output_shape)?;
        let b = match plan.b_shape.as_ref() {
            Some(bs) => Some(Self::operand(bs, &plan.output_shape)?),
            None => None,
        };
        let output = DmlTensorLayout::packed(&plan.output_shape)?;
        Ok(Self { a, b, output })
    }

    /// Packed when the shapes already agree, 0-strided broadcast otherwise.
    fn operand(shape: &[usize], output_shape: &[usize]) -> Result<DmlTensorLayout> {
        if shape == output_shape {
            DmlTensorLayout::packed(shape)
        } else {
            DmlTensorLayout::broadcast_to(shape, output_shape)
        }
    }
}

/// The three things a DirectML `DML_OPERATOR_REDUCE` needs from a [`ReducePlan`], as the
/// **single source of truth** shared by [`OpCacheKey::reduce`] and
/// [`crate::backend::dml::op::compile_reduce`] — so the cache key and the compiled
/// descriptor can never disagree about a reduction (the invariant this crate lives by).
///
/// Returns `(input, rank4_axis, output)`:
///
/// * `input` — the packed rank-4 layout of the input tensor.
/// * `rank4_axis` — the reduce axis **as DirectML indexes it**.  [`ReducePlan::axis`] is
///   relative to the original ONNX shape; [`DmlTensorLayout::packed`] left-pads that shape
///   to [`DML_RANK`], so the axis DirectML sees is shifted right by the pad width.  This is
///   the value that goes into `DML_REDUCE_OPERATOR_DESC::Axes`.
/// * `output` — the DirectML output layout: the padded input sizes with the reduced axis
///   **collapsed to 1**, which is what `DML_REDUCE_OPERATOR_DESC` requires *regardless* of
///   ONNX `keepdims`.  Dropping a size-1 axis changes only the logical rank, never the
///   buffer, so the byte footprint — and therefore the readback length — is identical
///   either way; `keepdims` is a router concern, not a descriptor one.
///
/// # Errors
/// [`DirectMLError::Declined`] when the input is not rank-4-describable (rank > 4, which
/// the DirectML path cannot express and the CPU kernel handles), or on the usual size
/// limits from [`DmlTensorLayout::packed`].
pub(crate) fn dml_reduce_layouts(
    plan: &ReducePlan,
) -> Result<(DmlTensorLayout, u32, DmlTensorLayout)> {
    // `packed` declines rank > 4, so after this succeeds the pad width cannot underflow and
    // `rank4_axis` cannot reach `DML_RANK`.
    let input = DmlTensorLayout::packed(&plan.input_shape)?;
    let rank = plan.input_shape.len();
    let pad = DML_RANK.checked_sub(rank).ok_or_else(|| {
        DirectMLError::Declined(format!(
            "Reduce: rank {rank} of {:?} exceeds DML_RANK = {DML_RANK}",
            plan.input_shape
        ))
    })?;
    let axis = usize::try_from(plan.axis)
        .map_err(|_| DirectMLError::Declined(format!("Reduce axis {} exceeds usize", plan.axis)))?;
    // `axis < rank` (the plan guarantees it) and `pad + rank == DML_RANK`, so this indexes
    // strictly inside `out_shape`.
    let rank4_axis = pad + axis;

    let mut out_shape = [1usize; DML_RANK];
    for (i, &d) in plan.input_shape.iter().enumerate() {
        out_shape[pad + i] = d;
    }
    out_shape[rank4_axis] = 1;
    let output = DmlTensorLayout::packed(&out_shape)?;

    Ok((
        input,
        checked_u32(rank4_axis, "Reduce rank-4 axis")?,
        output,
    ))
}

/// The DirectML bias layout for a [`ConvPlan`]: `[1, C_out, 1, 1]`, the shape
/// `DML_CONVOLUTION_OPERATOR_DESC::BiasTensor` requires — one value per output channel,
/// broadcast across batch and space by the operator itself.  Shared by
/// [`OpCacheKey::conv`] and [`crate::backend::dml::op::compile_conv`] so the key and the
/// descriptor describe the same bias tensor.
///
/// The bias is packed (`C_out` contiguous f32), **not** 0-strided: unlike a GEMM bias, the
/// convolution operator performs the channel/space broadcast internally from the
/// `[1, C_out, 1, 1]` shape, so the buffer holds exactly `C_out` elements.
///
/// # Errors
/// [`DirectMLError::Declined`] when `C_out` exceeds `usize`, or on the size limits from
/// [`DmlTensorLayout::packed`].
pub(crate) fn dml_conv_bias_layout(plan: &ConvPlan) -> Result<DmlTensorLayout> {
    let c_out = usize::try_from(plan.c_out)
        .map_err(|_| DirectMLError::Declined(format!("Conv C_out {} exceeds usize", plan.c_out)))?;
    DmlTensorLayout::packed(&[1, c_out, 1, 1])
}

/// Cache key for a compiled DirectML operator: everything that affects the
/// compiled binary, and nothing that does not.
///
/// Derived purely from platform-neutral types, so the *keying* logic — the thing
/// that decides whether a cache hit is safe — is unit-testable on Linux.
///
/// `alpha` / `beta` are stored as `f32::to_bits()` so the key can be `Hash + Eq`.
/// That makes `-0.0` and `+0.0` distinct keys and `NaN` equal to itself, both of
/// which are *conservative*: at worst they miss a cache hit and recompile, which
/// is slow rather than wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCacheKey {
    /// A `DML_OPERATOR_GEMM`.
    Gemm {
        /// The `A` operand's layout.
        a: DmlTensorLayout,
        /// The `B` operand's layout.
        b: DmlTensorLayout,
        /// The optional bias's layout.
        c: Option<DmlTensorLayout>,
        /// The output's layout.
        out: DmlTensorLayout,
        /// `DML_GEMM_OPERATOR_DESC::TransA`.
        trans_a: bool,
        /// `DML_GEMM_OPERATOR_DESC::TransB`.
        trans_b: bool,
        /// `alpha.to_bits()`.
        alpha_bits: u32,
        /// `beta.to_bits()`.
        beta_bits: u32,
    },
    /// A `DML_OPERATOR_ELEMENT_WISE_*`.
    Binary {
        /// Which operator.
        op: BinaryOp,
        /// The `A` operand's layout.
        a: DmlTensorLayout,
        /// The `B` operand's layout.
        b: DmlTensorLayout,
        /// The output's layout.
        out: DmlTensorLayout,
    },
    /// A `DML_OPERATOR_ACTIVATION_*`.
    Unary {
        /// Which operator.
        op: UnaryOp,
        /// The `A` operand's layout.
        a: DmlTensorLayout,
        /// The output's layout.
        out: DmlTensorLayout,
    },
    /// A `DML_OPERATOR_ACTIVATION_SOFTMAX` — the axis-less operator, which normalises the
    /// **innermost** dimension.  The engine only reaches here when the softmax axis *is*
    /// innermost ([`crate::plan::SoftmaxPlan::reduces_last_axis`]); a non-terminal axis is
    /// declined to the CPU/HLSL path before a key is built, so the layout alone keys it.
    Softmax {
        /// The (input == output) tensor layout; softmax is shape-preserving.
        tensor: DmlTensorLayout,
    },
    /// A `DML_OPERATOR_REDUCE`.
    Reduce {
        /// Which reduction (→ `DML_REDUCE_FUNCTION`).
        kind: ReduceKind,
        /// The input tensor's packed layout.
        input: DmlTensorLayout,
        /// The output layout: the input sizes with the reduced axis collapsed to 1.
        out: DmlTensorLayout,
        /// The reduce axis as DirectML indexes it (`DML_REDUCE_OPERATOR_DESC::Axes[0]`),
        /// i.e. the rank-4 axis after [`DmlTensorLayout::packed`]'s left-pad.
        axis: u32,
    },
    /// A `DML_OPERATOR_CONVOLUTION` — forward, cross-correlation, 2-D.
    Conv {
        /// The `[N, C_in, H, W]` input layout.
        input: DmlTensorLayout,
        /// The `[C_out, C_in/group, kH, kW]` filter layout.
        filter: DmlTensorLayout,
        /// The `[1, C_out, 1, 1]` bias layout, or `None` when the plan carries no bias.
        bias: Option<DmlTensorLayout>,
        /// The `[N, C_out, out_h, out_w]` output layout.
        out: DmlTensorLayout,
        /// `DML_CONVOLUTION_OPERATOR_DESC::Strides[0]`.
        stride_h: u32,
        /// `::Strides[1]`.
        stride_w: u32,
        /// `::Dilations[0]`.
        dilation_h: u32,
        /// `::Dilations[1]`.
        dilation_w: u32,
        /// `::StartPadding[0]`.
        pad_top: u32,
        /// `::StartPadding[1]`.
        pad_left: u32,
        /// `::EndPadding[0]`.
        pad_bottom: u32,
        /// `::EndPadding[1]`.
        pad_right: u32,
        /// `::GroupCount`.
        group: u32,
    },
}

impl OpCacheKey {
    /// Key a GEMM.
    #[must_use]
    pub fn gemm(plan: &MatMulPlan, layout: &DmlGemmLayout) -> Self {
        Self::Gemm {
            a: layout.a,
            b: layout.b,
            c: layout.c,
            out: layout.output,
            trans_a: plan.trans_a,
            trans_b: plan.trans_b,
            alpha_bits: plan.alpha.to_bits(),
            beta_bits: plan.beta.to_bits(),
        }
    }

    /// Key a binary elementwise op.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `layout.b` is `None` — a binary op with no
    /// second operand is a programming error in the caller, not a shape problem.
    pub fn binary(op: BinaryOp, layout: &DmlElementwiseLayout) -> Result<Self> {
        let b = layout.b.ok_or_else(|| {
            DirectMLError::Declined(format!(
                "{}: binary op has no B operand layout",
                op.as_str()
            ))
        })?;
        Ok(Self::Binary {
            op,
            a: layout.a,
            b,
            out: layout.output,
        })
    }

    /// Key a unary elementwise op.  `layout.b`, if present, is ignored.
    #[must_use]
    pub fn unary(op: UnaryOp, layout: &DmlElementwiseLayout) -> Self {
        Self::Unary {
            op,
            a: layout.a,
            out: layout.output,
        }
    }

    /// Key a softmax.  Softmax is shape-preserving, so the packed tensor layout is the
    /// whole key; the axis is not stored because a non-innermost axis never reaches the
    /// DirectML path (the engine declines it first), and an innermost axis is fixed by the
    /// shape.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on the rank / size limits of [`DmlTensorLayout::packed`]
    /// — a rank-5 tensor the DirectML path cannot describe is declined to the CPU.
    pub fn softmax(plan: &SoftmaxPlan) -> Result<Self> {
        Ok(Self::Softmax {
            tensor: DmlTensorLayout::packed(&plan.shape)?,
        })
    }

    /// Key a reduce, through the same `dml_reduce_layouts` derivation `compile_reduce`
    /// uses — so an identical plan keys identically, and a differing axis, shape or kind
    /// misses.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] from `dml_reduce_layouts`.
    pub fn reduce(plan: &ReducePlan) -> Result<Self> {
        let (input, axis, out) = dml_reduce_layouts(plan)?;
        Ok(Self::Reduce {
            kind: plan.kind,
            input,
            out,
            axis,
        })
    }

    /// Key a conv.  Captures every field `compile_conv` folds into the compiled operator:
    /// the four tensor layouts and the strides / dilations / paddings / group.  `Mode`
    /// (cross-correlation) and `Direction` (forward) are compile-time constants, identical
    /// for every conv, so they are deliberately **not** keyed.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] from [`DmlTensorLayout::packed`] or
    /// `dml_conv_bias_layout`.
    pub fn conv(plan: &ConvPlan) -> Result<Self> {
        let input = DmlTensorLayout::packed(&plan.input_shape)?;
        let filter = DmlTensorLayout::packed(&plan.weight_shape)?;
        let out = DmlTensorLayout::packed(&plan.output_shape)?;
        let bias = if plan.has_bias {
            Some(dml_conv_bias_layout(plan)?)
        } else {
            None
        };
        Ok(Self::Conv {
            input,
            filter,
            bias,
            out,
            stride_h: plan.stride_h,
            stride_w: plan.stride_w,
            dilation_h: plan.dilation_h,
            dilation_w: plan.dilation_w,
            pad_top: plan.pad_top,
            pad_left: plan.pad_left,
            pad_bottom: plan.pad_bottom,
            pad_right: plan.pad_right,
            group: plan.group,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn declined(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::Declined(_))
    }

    #[test]
    fn packed_left_pads_to_rank_four() {
        let l = DmlTensorLayout::packed(&[2, 3]).unwrap();
        assert_eq!(l.sizes, [1, 1, 2, 3]);
        assert_eq!(l.strides, [6, 6, 3, 1]);
        assert!(l.is_packed);
        assert_eq!(l.total_bytes, 24, "6 elements * 4 bytes");
        assert_eq!(l.elem_count(), 6);
    }

    #[test]
    fn packed_rank_four() {
        let l = DmlTensorLayout::packed(&[2, 3, 4, 5]).unwrap();
        assert_eq!(l.sizes, [2, 3, 4, 5]);
        assert_eq!(l.strides, [60, 20, 5, 1]);
        assert!(l.is_packed);
        assert_eq!(l.total_bytes, 480, "120 elements * 4 bytes");
    }

    #[test]
    fn packed_scalar() {
        let l = DmlTensorLayout::packed(&[]).unwrap();
        assert_eq!(l.sizes, [1, 1, 1, 1]);
        assert_eq!(l.strides, [1, 1, 1, 1]);
        assert_eq!(l.total_bytes, 4);
    }

    #[test]
    fn packed_declines_rank_five_and_empty() {
        assert!(declined(
            &DmlTensorLayout::packed(&[2, 2, 2, 2, 2]).unwrap_err()
        ));
        assert!(declined(&DmlTensorLayout::packed(&[0, 128]).unwrap_err()));
    }

    /// **The hazard-7 regression test.**
    ///
    /// A `[1, 4]` tensor broadcast to `[2, 3, 4]` occupies 16 bytes.  If
    /// `TotalTensorSizeInBytes` were computed as `product(sizes) * 4` it would be
    /// 96 — six times the buffer that is actually bound.
    #[test]
    fn broadcast_to_total_bytes_is_the_source_size() {
        let l = DmlTensorLayout::broadcast_to(&[1, 4], &[2, 3, 4]).unwrap();
        assert_eq!(l.sizes, [1, 2, 3, 4]);
        assert_eq!(l.strides, [0, 0, 0, 1]);
        assert!(!l.is_packed, "0-strides are not the packed stride vector");
        assert_eq!(
            l.total_bytes, 16,
            "must be the SOURCE's 4 elements * 4 bytes, NOT product(sizes) * 4 = 96"
        );
        // Guard the exact wrong answer, by name.
        let product_times_four = l.elem_count() * ELEM_SIZE as u64;
        assert_eq!(product_times_four, 96);
        assert_ne!(l.total_bytes, product_times_four);
    }

    #[test]
    fn broadcast_to_a_row_vector_bias() {
        // `[n]` bias against a `[1, 1, m, n]` output.
        let l = DmlTensorLayout::broadcast_to(&[4], &[1, 1, 3, 4]).unwrap();
        assert_eq!(l.sizes, [1, 1, 3, 4]);
        assert_eq!(l.strides, [0, 0, 0, 1]);
        assert_eq!(l.total_bytes, 16, "4 elements * 4 bytes, read 3 times");
    }

    #[test]
    fn broadcast_to_a_column_vector_bias() {
        // `[m, 1]` bias against `[1, 1, m, n]`.
        let l = DmlTensorLayout::broadcast_to(&[3, 1], &[1, 1, 3, 4]).unwrap();
        assert_eq!(l.sizes, [1, 1, 3, 4]);
        assert_eq!(l.strides, [0, 0, 1, 0]);
        assert_eq!(l.total_bytes, 12, "3 elements * 4 bytes");
    }

    #[test]
    fn broadcast_to_identical_shapes_is_the_packed_footprint() {
        let l = DmlTensorLayout::broadcast_to(&[2, 3, 4], &[2, 3, 4]).unwrap();
        assert_eq!(l.sizes, [1, 2, 3, 4]);
        // Axis 0 is a genuine size-1 axis; a 0 stride there is harmless because the
        // footprint formula multiplies it by `size - 1 == 0`.
        assert_eq!(l.strides, [0, 12, 4, 1]);
        assert_eq!(l.total_bytes, 96, "24 elements * 4 bytes");
        assert_eq!(
            l.total_bytes,
            DmlTensorLayout::packed(&[2, 3, 4]).unwrap().total_bytes,
            "a no-op broadcast must agree with the packed footprint"
        );
    }

    #[test]
    fn broadcast_to_rejects_the_impossible() {
        let e = DmlTensorLayout::broadcast_to(&[3, 4], &[2, 3, 5]).unwrap_err();
        assert!(matches!(e, DirectMLError::ShapeMismatch(_)), "got {e:?}");
        // Shrinking is not broadcasting.
        let e = DmlTensorLayout::broadcast_to(&[2, 3, 4], &[3, 4]).unwrap_err();
        assert!(matches!(e, DirectMLError::ShapeMismatch(_)), "got {e:?}");
    }

    #[test]
    fn collapse_to_batched_matrix_folds_leading_dims() {
        let l = DmlTensorLayout::collapse_to_batched_matrix(&[2, 3, 4, 5], 6, false).unwrap();
        assert_eq!(l.sizes, [6, 1, 4, 5]);
        assert_eq!(l.strides, [20, 20, 5, 1]);
        assert!(l.is_packed);
        assert_eq!(l.total_bytes, 6 * 4 * 5 * 4);

        // Batch-broadcast: the operand is 2-D but the output has 6 batches.
        let l = DmlTensorLayout::collapse_to_batched_matrix(&[4, 5], 6, true).unwrap();
        assert_eq!(l.sizes, [6, 1, 4, 5]);
        assert_eq!(l.strides, [0, 20, 5, 1]);
        assert!(!l.is_packed);
        assert_eq!(
            l.total_bytes,
            4 * 5 * 4,
            "a batch-broadcast operand costs ONE matrix, not six"
        );

        assert!(declined(
            &DmlTensorLayout::collapse_to_batched_matrix(&[4], 1, false).unwrap_err()
        ));
        assert!(declined(
            &DmlTensorLayout::collapse_to_batched_matrix(&[4, 5], 0, false).unwrap_err()
        ));
    }

    #[test]
    fn packed_strides_decline_on_overflow() {
        // 2^16 * 2^16 * 2^16 elements is far past u32.
        let e = DmlTensorLayout::packed_strides(&[65_536, 65_536, 65_536, 1]).unwrap_err();
        assert!(declined(&e), "got {e:?}");
    }

    // ── GEMM layout ──────────────────────────────────────────────────────────

    #[test]
    fn gemm_layout_describes_the_stored_buffer_not_the_logical_one() {
        // trans_b: B is *stored* as [n, k] = [4, 3], logically [3, 4].
        let plan = MatMulPlan::gemm(&[2, 3], &[4, 3], None, 1.0, 0.0, false, true).unwrap();
        assert_eq!((plan.m, plan.k, plan.n), (2, 3, 4));

        let l = DmlGemmLayout::from_plan(&plan).unwrap();
        assert_eq!(l.a.sizes, [1, 1, 2, 3], "A stored [m, k]");
        assert_eq!(
            l.b.sizes,
            [1, 1, 4, 3],
            "B must be described as STORED [n, k], with TransB doing the transpose"
        );
        assert_eq!(l.output.sizes, [1, 1, 2, 4]);
        assert_eq!(l.c, None);

        // trans_a: A is stored as [k, m] = [3, 2].
        let plan = MatMulPlan::gemm(&[3, 2], &[3, 4], None, 1.0, 0.0, true, false).unwrap();
        let l = DmlGemmLayout::from_plan(&plan).unwrap();
        assert_eq!(l.a.sizes, [1, 1, 3, 2], "A stored [k, m]");
    }

    #[test]
    fn gemm_layout_broadcasts_the_bias_without_copying() {
        let plan = MatMulPlan::gemm(&[2, 3], &[3, 4], Some(&[4]), 1.0, 1.0, false, false).unwrap();
        let l = DmlGemmLayout::from_plan(&plan).unwrap();
        let c = l.c.expect("bias present");
        assert_eq!(c.sizes, [1, 1, 2, 4]);
        assert_eq!(c.strides, [0, 0, 0, 1]);
        assert_eq!(c.total_bytes, 16, "the bias is 4 floats, read twice");
    }

    #[test]
    fn gemm_layout_output_is_always_packed() {
        let plan = MatMulPlan::matmul(&[7, 5], &[5, 9]).unwrap();
        let l = DmlGemmLayout::from_plan(&plan).unwrap();
        assert!(l.output.is_packed);
        assert_eq!(l.output.total_bytes, 7 * 9 * 4);
    }

    // ── elementwise layout ───────────────────────────────────────────────────

    #[test]
    fn elementwise_layout_is_packed_for_identical_shapes() {
        let plan = ElementwisePlan::binary(&[2, 3, 4], &[2, 3, 4]).unwrap();
        let l = DmlElementwiseLayout::from_plan(&plan).unwrap();
        assert!(l.a.is_packed);
        assert!(l.b.expect("binary").is_packed);
        assert!(l.output.is_packed);
        assert_eq!(l.a.sizes, [1, 2, 3, 4]);
        assert_eq!(l.a.total_bytes, 96);
    }

    #[test]
    fn elementwise_layout_unary_has_no_b() {
        let plan = ElementwisePlan::unary(&[5]).unwrap();
        let l = DmlElementwiseLayout::from_plan(&plan).unwrap();
        assert_eq!(l.b, None);
        assert_eq!(l.a.sizes, [1, 1, 1, 5]);
    }

    // ── cache key ────────────────────────────────────────────────────────────

    #[test]
    fn op_cache_key_distinguishes_what_matters() {
        let mut keys: HashSet<OpCacheKey> = HashSet::new();

        let p1 = MatMulPlan::gemm(&[2, 3], &[3, 4], None, 1.0, 0.0, false, false).unwrap();
        let l1 = DmlGemmLayout::from_plan(&p1).unwrap();
        let p2 = MatMulPlan::gemm(&[2, 3], &[3, 4], None, 2.0, 0.0, false, false).unwrap();
        let l2 = DmlGemmLayout::from_plan(&p2).unwrap();

        // Identical plans key identically — that is what makes the cache useful.
        assert_eq!(OpCacheKey::gemm(&p1, &l1), OpCacheKey::gemm(&p1, &l1));
        // A different alpha compiles a different operator, so it must key differently.
        assert_ne!(OpCacheKey::gemm(&p1, &l1), OpCacheKey::gemm(&p2, &l2));

        keys.insert(OpCacheKey::gemm(&p1, &l1));
        keys.insert(OpCacheKey::gemm(&p1, &l1));
        assert_eq!(keys.len(), 1, "the same plan must hit the cache");
        keys.insert(OpCacheKey::gemm(&p2, &l2));
        assert_eq!(keys.len(), 2);

        // A different shape keys differently.
        let p3 = MatMulPlan::matmul(&[2, 3], &[3, 5]).unwrap();
        let l3 = DmlGemmLayout::from_plan(&p3).unwrap();
        keys.insert(OpCacheKey::gemm(&p3, &l3));
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn op_cache_key_separates_the_binary_ops() {
        let plan = ElementwisePlan::binary(&[4], &[4]).unwrap();
        let l = DmlElementwiseLayout::from_plan(&plan).unwrap();
        let add = OpCacheKey::binary(BinaryOp::Add, &l).unwrap();
        let mul = OpCacheKey::binary(BinaryOp::Mul, &l).unwrap();
        assert_ne!(add, mul);
        assert_eq!(add, OpCacheKey::binary(BinaryOp::Add, &l).unwrap());

        let relu = OpCacheKey::unary(UnaryOp::Relu, &l);
        let sigmoid = OpCacheKey::unary(UnaryOp::Sigmoid, &l);
        assert_ne!(relu, sigmoid);
    }

    #[test]
    fn op_cache_key_binary_declines_a_missing_b() {
        let plan = ElementwisePlan::unary(&[4]).unwrap();
        let l = DmlElementwiseLayout::from_plan(&plan).unwrap();
        assert!(declined(
            &OpCacheKey::binary(BinaryOp::Add, &l).unwrap_err()
        ));
    }

    // ── reduce layout derivation ─────────────────────────────────────────────

    #[test]
    fn dml_reduce_layouts_shifts_the_axis_and_collapses_the_output() {
        // Reduce [2, 3, 4] over the ONNX axis 1.  `packed` left-pads to [1, 2, 3, 4], so
        // DirectML's axis is 2, and the output is the input sizes with axis 2 → 1.
        let plan = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], false).unwrap();
        let (input, rank4_axis, out) = dml_reduce_layouts(&plan).unwrap();
        assert_eq!(input.sizes, [1, 2, 3, 4]);
        assert_eq!(rank4_axis, 2, "ONNX axis 1 + left-pad 1 = rank-4 axis 2");
        assert_eq!(out.sizes, [1, 2, 1, 4], "the reduced axis collapses to 1");
        // The output footprint equals the plan's out_count, whatever keepdims says.
        assert_eq!(
            out.total_bytes,
            u64::from(plan.out_count) * ELEM_SIZE as u64
        );
    }

    #[test]
    fn dml_reduce_layouts_output_ignores_keepdims() {
        // keepdims changes only the ONNX logical shape; the DirectML descriptor and its
        // byte footprint are identical.
        let squeeze = ReducePlan::reduce(ReduceKind::Mean, &[2, 3, 4], &[2], false).unwrap();
        let keep = ReducePlan::reduce(ReduceKind::Mean, &[2, 3, 4], &[2], true).unwrap();
        let (_, ax_s, out_s) = dml_reduce_layouts(&squeeze).unwrap();
        let (_, ax_k, out_k) = dml_reduce_layouts(&keep).unwrap();
        assert_eq!(ax_s, ax_k);
        assert_eq!(
            out_s, out_k,
            "keepdims must not change the DirectML output layout"
        );
        assert_eq!(out_s.sizes, [1, 2, 3, 1]);
    }

    #[test]
    fn dml_reduce_layouts_declines_rank_five() {
        let plan = ReducePlan::reduce(ReduceKind::Sum, &[2, 2, 2, 2, 2], &[0], false).unwrap();
        assert!(declined(&dml_reduce_layouts(&plan).unwrap_err()));
    }

    // ── conv bias layout ─────────────────────────────────────────────────────

    #[test]
    fn dml_conv_bias_layout_is_channel_shaped_and_packed() {
        let plan =
            ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], Some(&[4]), &[], &[], &[], 1).unwrap();
        let bias = dml_conv_bias_layout(&plan).unwrap();
        assert_eq!(bias.sizes, [1, 4, 1, 1], "DirectML wants [1, C_out, 1, 1]");
        assert!(
            bias.is_packed,
            "the conv bias is a packed C_out-vector, not 0-strided"
        );
        assert_eq!(bias.total_bytes, 4 * ELEM_SIZE as u64);
    }

    // ── softmax / reduce / conv cache keys ───────────────────────────────────

    #[test]
    fn op_cache_key_softmax_keys_on_the_shape_only() {
        let mut keys: HashSet<OpCacheKey> = HashSet::new();
        let same_a = SoftmaxPlan::softmax(&[2, 4], 1).unwrap();
        let same_b = SoftmaxPlan::softmax(&[2, 4], 1).unwrap();
        let other = SoftmaxPlan::softmax(&[2, 5], 1).unwrap();
        assert_eq!(
            OpCacheKey::softmax(&same_a).unwrap(),
            OpCacheKey::softmax(&same_b).unwrap()
        );
        keys.insert(OpCacheKey::softmax(&same_a).unwrap());
        keys.insert(OpCacheKey::softmax(&same_b).unwrap());
        assert_eq!(keys.len(), 1, "identical softmax shapes must hit the cache");
        keys.insert(OpCacheKey::softmax(&other).unwrap());
        assert_eq!(keys.len(), 2, "a different shape keys differently");
        // A rank-5 tensor cannot be described to DirectML.
        let big = SoftmaxPlan::softmax(&[2, 2, 2, 2, 2], 4).unwrap();
        assert!(declined(&OpCacheKey::softmax(&big).unwrap_err()));
    }

    #[test]
    fn op_cache_key_reduce_distinguishes_kind_axis_and_shape() {
        let sum = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], false).unwrap();
        let max = ReducePlan::reduce(ReduceKind::Max, &[2, 3, 4], &[1], false).unwrap();
        let axis2 = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[2], false).unwrap();
        assert_eq!(
            OpCacheKey::reduce(&sum).unwrap(),
            OpCacheKey::reduce(&sum).unwrap()
        );
        assert_ne!(
            OpCacheKey::reduce(&sum).unwrap(),
            OpCacheKey::reduce(&max).unwrap(),
            "a different reduction function is a different operator"
        );
        assert_ne!(
            OpCacheKey::reduce(&sum).unwrap(),
            OpCacheKey::reduce(&axis2).unwrap(),
            "a different axis is a different operator"
        );
        // keepdims does not change the compiled operator, so it must not change the key.
        let keep = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], true).unwrap();
        assert_eq!(
            OpCacheKey::reduce(&sum).unwrap(),
            OpCacheKey::reduce(&keep).unwrap(),
            "keepdims is a logical-shape concern, not a descriptor one"
        );
    }

    #[test]
    fn op_cache_key_conv_captures_every_operator_field() {
        let base = ConvPlan::conv(
            &[1, 2, 8, 8],
            &[4, 2, 3, 3],
            Some(&[4]),
            &[1, 1],
            &[0, 0, 0, 0],
            &[1, 1],
            1,
        )
        .unwrap();
        assert_eq!(
            OpCacheKey::conv(&base).unwrap(),
            OpCacheKey::conv(&base).unwrap()
        );

        // Each attribute that reaches DML_CONVOLUTION_OPERATOR_DESC must move the key.
        let stride = ConvPlan::conv(
            &[1, 2, 8, 8],
            &[4, 2, 3, 3],
            Some(&[4]),
            &[2, 2],
            &[0, 0, 0, 0],
            &[1, 1],
            1,
        )
        .unwrap();
        let dilation = ConvPlan::conv(
            &[1, 2, 8, 8],
            &[4, 2, 3, 3],
            Some(&[4]),
            &[1, 1],
            &[0, 0, 0, 0],
            &[2, 2],
            1,
        )
        .unwrap();
        let pad = ConvPlan::conv(
            &[1, 2, 8, 8],
            &[4, 2, 3, 3],
            Some(&[4]),
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
        )
        .unwrap();
        let no_bias = ConvPlan::conv(
            &[1, 2, 8, 8],
            &[4, 2, 3, 3],
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            &[1, 1],
            1,
        )
        .unwrap();
        for other in [&stride, &dilation, &pad, &no_bias] {
            assert_ne!(
                OpCacheKey::conv(&base).unwrap(),
                OpCacheKey::conv(other).unwrap()
            );
        }
        // A grouped conv keys differently from the ungrouped one of the same tensor shapes.
        let grouped = ConvPlan::conv(
            &[1, 2, 8, 8],
            &[4, 1, 3, 3],
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            &[1, 1],
            2,
        )
        .unwrap();
        assert_ne!(
            OpCacheKey::conv(&no_bias).unwrap(),
            OpCacheKey::conv(&grouped).unwrap()
        );
    }

    /// The three Wave-4 keys must be disjoint from each other and from the Wave-3 ones, so
    /// a softmax and a reduce of the same shape can never collide in the cache.
    #[test]
    fn op_cache_key_wave4_variants_do_not_collide() {
        let mut keys: HashSet<OpCacheKey> = HashSet::new();
        let softmax = SoftmaxPlan::softmax(&[2, 3, 4], 2).unwrap();
        let reduce = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[2], false).unwrap();
        let conv = ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[], &[], &[], 1).unwrap();
        keys.insert(OpCacheKey::softmax(&softmax).unwrap());
        keys.insert(OpCacheKey::reduce(&reduce).unwrap());
        keys.insert(OpCacheKey::conv(&conv).unwrap());
        assert_eq!(
            keys.len(),
            3,
            "the three op families must never share a key"
        );
    }
}
