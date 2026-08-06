//! Platform-neutral planning for the Wave-4 neural-network ops: `Softmax`,
//! `Reduce{Sum,Mean,Max,Min}` and `Conv`.
//!
//! This is a child of [`crate::plan`] and obeys every rule its parent does: not a
//! single `#[cfg]`, every shape-derived value range-checked exactly once through
//! [`crate::plan::checked_u32`], and a *decline* (→ CPU) wherever the GPU path could only
//! guess.  It is compiled and unit-tested on every target, including the Linux host
//! that has no D3D12 device.
//!
//! # What each op supports, and where it declines
//!
//! * **[`SoftmaxPlan`]** — a single softmax axis, resolved from the ONNX `axis`
//!   attribute and collapsed to an `outer × axis_len × inner` decomposition.  Both
//!   the HLSL and the genuine-DML paths consume it.  It declines (→ CPU) only when a
//!   count overflows `u32` or the row grid exceeds D3D12's per-dimension limit.
//! * **[`ReducePlan`]** — a **single** reduction axis for `ReduceSum` / `ReduceMean`
//!   / `ReduceMax` / `ReduceMin`.  A *multi*-axis reduce cannot be indexed correctly
//!   by the flat [`crate::hlsl::REDUCE_HLSL`] shader, so it is declined outright
//!   rather than mis-executed — exactly as [`crate::plan::MatMulPlan::matmul`] declines a
//!   batched matmul.  The boundary is stated on [`ReducePlan::reduce`].
//! * **[`ConvPlan`]** — a 2-D `[N, C_in, H, W] × [C_out, C_in/group, kH, kW]`
//!   convolution for the **DirectML path only**.  There is deliberately no Conv HLSL
//!   shader; the HLSL engine declines Conv (see [`ConvPlan::conv`]).  The plan
//!   computes the ONNX output spatial dims and carries every attribute
//!   `DML_CONVOLUTION_OPERATOR_DESC` needs.

use crate::error::{DirectMLError, Result};

use super::{
    ceil_div, checked_u32, numel, pad_to_cbv, DispatchGrid, ELEM_SIZE, ROOT_CONSTANT_COUNT,
};

// ─── constants ───────────────────────────────────────────────────────────────

/// The Softmax and Reduce HLSL kernels are `[numthreads(256, 1, 1)]`.
///
/// One thread handles one **output row** for Softmax (a full length-`axis_len`
/// reduction plus normalisation) and one **output element** for Reduce (a single
/// length-`axis_len` reduction).  This mirrors the elementwise family's group size
/// so the same 2-D group-index folding (`gid.y * GroupsX + gid.x`) applies.
pub const REDUCTION_THREADS_PER_GROUP: u32 = 256;

// ─── reduce kind ─────────────────────────────────────────────────────────────

/// The four reductions this crate plans, each with a `DML_OPERATOR_REDUCE`
/// `Function` and a [`crate::hlsl::REDUCE_HLSL`] entry point.
///
/// The DirectML `DML_REDUCE_FUNCTION` values are `_SUM`, `_AVERAGE`, `_MAX` and
/// `_MIN`; the HLSL entry points are `main_sum`, `main_mean`, `main_max`,
/// `main_min`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    /// `out = Σ x` — `DML_REDUCE_FUNCTION_SUM`, `main_sum`.
    Sum,
    /// `out = (Σ x) / axis_len` — `DML_REDUCE_FUNCTION_AVERAGE`, `main_mean`.
    Mean,
    /// `out = max x` — `DML_REDUCE_FUNCTION_MAX`, `main_max`.
    Max,
    /// `out = min x` — `DML_REDUCE_FUNCTION_MIN`, `main_min`.
    Min,
}

impl ReduceKind {
    /// The ONNX op name, and the stable tag a [`crate::reference::ComparisonReport`]
    /// carries.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "ReduceSum",
            Self::Mean => "ReduceMean",
            Self::Max => "ReduceMax",
            Self::Min => "ReduceMin",
        }
    }

    /// `true` for the two reductions that do **no arithmetic** — `Max` and `Min`
    /// merely *select* an input element, so the GPU must reproduce the oracle bit for
    /// bit.  `Sum` and `Mean` accumulate in f32 and are allowed the documented,
    /// `√axis_len`-scaled drift; see [`crate::reference::Tolerance::for_reduce`].
    #[must_use]
    pub fn is_exact(self) -> bool {
        matches!(self, Self::Max | Self::Min)
    }
}

// ─── SoftmaxPlan ─────────────────────────────────────────────────────────────

/// A validated, backend-agnostic softmax plan.
///
/// ONNX (opset 13+) `Softmax` normalises along **one** axis.  This plan resolves a
/// possibly-negative `axis` and collapses the shape into the three sizes both
/// backends need:
///
/// ```text
/// shape = [ d₀ … d_{axis-1} , d_axis , d_{axis+1} … d_{n-1} ]
///          └──── outer ────┘  axis_len  └───── inner ──────┘
/// ```
///
/// `outer × inner` softmax **rows** are computed independently; each is a length
/// `axis_len` reduction (row-max, then Σ exp, then divide).  Row `r = o·inner + i`
/// lives at buffer offset `base = o·axis_len·inner + i`, striding by `inner`.
///
/// Every field is already range-checked to fit `u32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftmaxPlan {
    /// The resolved, non-negative softmax axis.
    pub axis: u32,
    /// Product of the dims **before** `axis`; `1` when `axis == 0`.
    pub outer: u32,
    /// Length of the softmax axis — the reduction size.
    pub axis_len: u32,
    /// Product of the dims **after** `axis`; `1` when `axis` is the last axis.
    pub inner: u32,
    /// `outer × inner` — the number of independent softmax rows, i.e. the number of
    /// HLSL threads and the value guarded against in the shader.
    pub rows: u32,
    /// The tensor shape; softmax is shape-preserving, so this is both input and
    /// output.
    pub shape: Vec<usize>,
}

impl SoftmaxPlan {
    /// Plan an ONNX `Softmax` over `axis` (opset-13 single-axis semantics).
    ///
    /// A negative `axis` counts from the end (`axis + rank`).
    ///
    /// # The numerically-stable form both paths must use
    ///
    /// The reference ([`crate::reference::ref_softmax`]) and
    /// [`crate::hlsl::SOFTMAX_HLSL`] both compute the **max-subtracted** softmax:
    /// `m = max_k x_k`, then `e_k = exp(x_k − m)`, then `y_k = e_k / Σ e`.  The
    /// subtraction is what keeps `exp` from overflowing to `+inf`; a shader that
    /// skipped it would disagree with the oracle on any row containing a large
    /// positive value.  If you change the shader you must change the oracle in the
    /// same commit.
    ///
    /// # Errors
    /// - [`DirectMLError::ShapeMismatch`] — `axis` is out of range for the rank (the
    ///   CPU operator errors on the same input); this also rejects a rank-0 scalar,
    ///   which has no axis to normalise.
    /// - [`DirectMLError::Declined`] — the tensor is empty (`numel() == 0`), or a
    ///   size exceeds `u32::MAX`.  (The row-grid limit is enforced lazily by
    ///   [`Self::hlsl_grid`], exactly as [`super::MatMulPlan::hlsl_grid`] does.)
    pub fn softmax(shape: &[usize], axis: i64) -> Result<Self> {
        let ax = resolve_axis(axis, shape.len(), "Softmax")?;

        if numel(shape)? == 0 {
            return Err(DirectMLError::Declined(format!(
                "Softmax: empty tensor {shape:?}; a D3D12 buffer of Width = 0 cannot be created"
            )));
        }

        let outer = checked_u32(numel(&shape[..ax])?, "Softmax outer size")?;
        let axis_len = checked_u32(shape[ax], "Softmax axis length")?;
        let inner = checked_u32(numel(&shape[ax + 1..])?, "Softmax inner size")?;
        let rows = checked_u32(
            (outer as usize)
                .checked_mul(inner as usize)
                .ok_or_else(|| DirectMLError::Declined("Softmax outer * inner overflows".into()))?,
            "Softmax row count",
        )?;

        Ok(Self {
            axis: checked_u32(ax, "Softmax axis")?,
            outer,
            axis_len,
            inner,
            rows,
            shape: shape.to_vec(),
        })
    }

    /// The output shape — identical to the input shape.
    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total element count of the (input == output) tensor.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn output_elems(&self) -> Result<usize> {
        numel(&self.shape)
    }

    /// Byte size of each of the input and output buffers.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn buffer_bytes(&self) -> Result<usize> {
        self.output_elems()?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("Softmax buffer size overflows usize".into()))
    }

    /// `true` when the softmax axis is the **last** axis (`inner == 1`).
    ///
    /// The classic `DML_ACTIVATION_SOFTMAX_OPERATOR_DESC` has no axis field and
    /// normalises the trailing dimension only, so the DirectML backend uses this to
    /// decide whether it can express the node with that operator or must decline the
    /// non-terminal-axis case to the CPU.  The HLSL path has no such restriction —
    /// its `inner`-strided loop handles any axis — so it ignores this predicate.
    #[must_use]
    pub fn reduces_last_axis(&self) -> bool {
        self.inner == 1
    }

    /// Thread-group grid for [`crate::hlsl::SOFTMAX_HLSL`]: one thread per softmax
    /// **row**, folded into a 2-D grid exactly like the elementwise kernels.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the row count needs more than
    /// `65535 × 65535` thread groups.
    pub fn hlsl_grid(&self) -> Result<DispatchGrid> {
        let groups = ceil_div(self.rows, REDUCTION_THREADS_PER_GROUP)
            .ok_or_else(|| DirectMLError::Declined("REDUCTION_THREADS_PER_GROUP is 0".into()))?;
        DispatchGrid::linear(groups)
    }

    /// Root constants.  `groups_x` is taken from [`Self::hlsl_grid`]'s `x`, so the
    /// shader's `gid.y · GroupsX + gid.x` row index can never disagree with the grid
    /// actually dispatched.
    ///
    /// # Errors
    /// As [`Self::hlsl_grid`].
    pub fn constants(&self) -> Result<SoftmaxConstants> {
        let grid = self.hlsl_grid()?;
        Ok(SoftmaxConstants {
            rows: self.rows,
            groups_x: grid.x,
            axis_len: self.axis_len,
            inner: self.inner,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        })
    }
}

/// The Softmax kernel's `b0` block.  **Field order is load-bearing** — it must match
/// the `cbuffer` in [`crate::hlsl::SOFTMAX_HLSL`] and is exactly
/// [`ROOT_CONSTANT_COUNT`] `u32`s wide.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SoftmaxConstants {
    /// `outer × inner` — the softmax row count the shader guards against.
    pub rows: u32,
    /// [`DispatchGrid::x`] from [`SoftmaxPlan::hlsl_grid`].
    pub groups_x: u32,
    /// The reduction length (length of the softmax axis).
    pub axis_len: u32,
    /// The stride, in elements, between successive axis entries of one row.
    pub inner: u32,
    /// Padding, so the block is exactly [`ROOT_CONSTANT_COUNT`] `u32`s wide.
    pub pad0: u32,
    /// Padding.
    pub pad1: u32,
    /// Padding.
    pub pad2: u32,
    /// Padding.
    pub pad3: u32,
}

impl SoftmaxConstants {
    /// The eight `u32`s, in `cbuffer` order, for `SetComputeRoot32BitConstants`.
    #[must_use]
    pub fn to_root_constants(self) -> [u32; ROOT_CONSTANT_COUNT] {
        [
            self.rows,
            self.groups_x,
            self.axis_len,
            self.inner,
            self.pad0,
            self.pad1,
            self.pad2,
            self.pad3,
        ]
    }

    /// The same payload padded to [`super::CBV_ALIGNMENT`], for a CBV-based variant.
    #[must_use]
    pub fn const_buffer_bytes(self) -> [u8; super::CBV_ALIGNMENT] {
        pad_to_cbv(&self.to_root_constants())
    }
}

// ─── ReducePlan ──────────────────────────────────────────────────────────────

/// A validated, backend-agnostic single-axis reduction plan.
///
/// Uses the same `outer × axis_len × inner` decomposition as [`SoftmaxPlan`], but
/// the output is **smaller** than the input: the reduced axis collapses to one
/// element per `(outer, inner)` position.  Output element `j = o·inner + i` reduces
/// the `axis_len` inputs at `base = o·axis_len·inner + i`, striding by `inner`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducePlan {
    /// Which reduction.
    pub kind: ReduceKind,
    /// The resolved, non-negative reduction axis.
    pub axis: u32,
    /// Product of the dims before `axis`.
    pub outer: u32,
    /// Length of the reduced axis.
    pub axis_len: u32,
    /// Product of the dims after `axis`.
    pub inner: u32,
    /// `outer × inner` — the number of output elements, and the number of HLSL
    /// threads.
    pub out_count: u32,
    /// ONNX `keepdims`: `true` leaves the reduced axis as a size-1 dim, `false`
    /// removes it.
    pub keepdims: bool,
    /// The input shape.
    pub input_shape: Vec<usize>,
    /// The output shape, per `keepdims`.  Follows ONNX: fully collapsing a rank-1
    /// input with `keepdims=0` leaves the **empty** shape (a rank-0 scalar), not
    /// `[1]` — see `reduce_output_shape`.
    pub output_shape: Vec<usize>,
}

impl ReducePlan {
    /// Plan an ONNX `Reduce{Sum,Mean,Max,Min}` over a **single** axis.
    ///
    /// `axes` follows ONNX: an empty list means "reduce over every axis", and
    /// negative entries count from the end.  Duplicates are ignored.
    ///
    /// # This backend reduces one axis. Full stop.
    ///
    /// A flat, index-parallel shader cannot walk *multiple* independent reduced axes
    /// without a per-thread nested loop over a shape it does not carry — the same
    /// class of "plausible numbers from an out-of-bounds read" trap that
    /// [`super::MatMulPlan::matmul`] refuses batched matmul for.  So any `axes` that
    /// resolves to more than one distinct axis is [`DirectMLError::Declined`] → the
    /// CPU kernel (which handles it correctly) runs.  Note this makes an *empty*
    /// `axes` on a rank ≥ 2 tensor a decline (it means "all axes"), while an empty
    /// `axes` on a rank-1 tensor is the single-axis case and is accepted.
    ///
    /// To lift this you must, in one commit: give the shader the reduced-axis strides
    /// as root constants, walk them in a nested loop, and prove it on hardware with
    /// `DirectMLContext::self_check`.  Not before.
    ///
    /// # Errors
    /// - [`DirectMLError::ShapeMismatch`] — an axis is out of range for the rank; the
    ///   CPU operator errors on the same input.
    /// - [`DirectMLError::Declined`] — the resolved axis set is not exactly one axis;
    ///   the tensor is empty; or a size exceeds `u32::MAX`.
    pub fn reduce(kind: ReduceKind, shape: &[usize], axes: &[i64], keepdims: bool) -> Result<Self> {
        let ndim = shape.len();

        let mut resolved: Vec<usize> = if axes.is_empty() {
            (0..ndim).collect()
        } else {
            let mut v = Vec::with_capacity(axes.len());
            for &a in axes {
                v.push(resolve_axis(a, ndim, kind.as_str())?);
            }
            v
        };
        resolved.sort_unstable();
        resolved.dedup();

        if resolved.len() != 1 {
            return Err(DirectMLError::Declined(format!(
                "{}: this backend reduces a single axis only, but axes {axes:?} over a rank-{ndim} \
                 tensor resolve to {resolved:?}; declining to the CPU kernel",
                kind.as_str()
            )));
        }
        let ax = resolved[0];

        if numel(shape)? == 0 {
            return Err(DirectMLError::Declined(format!(
                "{}: empty tensor {shape:?}; a D3D12 buffer of Width = 0 cannot be created",
                kind.as_str()
            )));
        }

        let outer = checked_u32(numel(&shape[..ax])?, "Reduce outer size")?;
        let axis_len = checked_u32(shape[ax], "Reduce axis length")?;
        let inner = checked_u32(numel(&shape[ax + 1..])?, "Reduce inner size")?;
        let out_count = checked_u32(
            (outer as usize)
                .checked_mul(inner as usize)
                .ok_or_else(|| DirectMLError::Declined("Reduce outer * inner overflows".into()))?,
            "Reduce output element count",
        )?;

        let output_shape = reduce_output_shape(shape, ax, keepdims);

        Ok(Self {
            kind,
            axis: checked_u32(ax, "Reduce axis")?,
            outer,
            axis_len,
            inner,
            out_count,
            keepdims,
            input_shape: shape.to_vec(),
            output_shape,
        })
    }

    /// Total element count of the output.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn output_elems(&self) -> Result<usize> {
        numel(&self.output_shape)
    }

    /// Byte size of the input buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn input_bytes(&self) -> Result<usize> {
        numel(&self.input_shape)?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("Reduce input size overflows usize".into()))
    }

    /// Byte size of the output buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn output_bytes(&self) -> Result<usize> {
        self.output_elems()?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("Reduce output size overflows usize".into()))
    }

    /// Thread-group grid for [`crate::hlsl::REDUCE_HLSL`]: one thread per **output
    /// element**, folded into a 2-D grid like the elementwise kernels.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the output count needs more than
    /// `65535 × 65535` thread groups.
    pub fn hlsl_grid(&self) -> Result<DispatchGrid> {
        let groups = ceil_div(self.out_count, REDUCTION_THREADS_PER_GROUP)
            .ok_or_else(|| DirectMLError::Declined("REDUCTION_THREADS_PER_GROUP is 0".into()))?;
        DispatchGrid::linear(groups)
    }

    /// Root constants; `groups_x` comes straight from [`Self::hlsl_grid`].
    ///
    /// # Errors
    /// As [`Self::hlsl_grid`].
    pub fn constants(&self) -> Result<ReduceConstants> {
        let grid = self.hlsl_grid()?;
        Ok(ReduceConstants {
            out_count: self.out_count,
            groups_x: grid.x,
            axis_len: self.axis_len,
            inner: self.inner,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        })
    }
}

/// The Reduce kernels' `b0` block.  Same layout warning as [`SoftmaxConstants`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReduceConstants {
    /// The number of output elements the shader guards against.
    pub out_count: u32,
    /// [`DispatchGrid::x`] from [`ReducePlan::hlsl_grid`].
    pub groups_x: u32,
    /// The reduction length (length of the reduced axis).
    pub axis_len: u32,
    /// The stride, in elements, between successive axis entries of one reduction.
    pub inner: u32,
    /// Padding, so the block is exactly [`ROOT_CONSTANT_COUNT`] `u32`s wide.
    pub pad0: u32,
    /// Padding.
    pub pad1: u32,
    /// Padding.
    pub pad2: u32,
    /// Padding.
    pub pad3: u32,
}

impl ReduceConstants {
    /// The eight `u32`s, in `cbuffer` order, for `SetComputeRoot32BitConstants`.
    #[must_use]
    pub fn to_root_constants(self) -> [u32; ROOT_CONSTANT_COUNT] {
        [
            self.out_count,
            self.groups_x,
            self.axis_len,
            self.inner,
            self.pad0,
            self.pad1,
            self.pad2,
            self.pad3,
        ]
    }

    /// The same payload padded to [`super::CBV_ALIGNMENT`], for a CBV-based variant.
    #[must_use]
    pub fn const_buffer_bytes(self) -> [u8; super::CBV_ALIGNMENT] {
        pad_to_cbv(&self.to_root_constants())
    }
}

// ─── ConvPlan ────────────────────────────────────────────────────────────────

/// A validated, backend-agnostic 2-D convolution plan.
///
/// Maps an ONNX `Conv` node onto everything `DML_CONVOLUTION_OPERATOR_DESC` needs:
/// the resolved strides / pads / dilations / group, and the output spatial dims
/// computed with the standard formula
///
/// ```text
/// out = floor( (in + pad_begin + pad_end − dilation·(k − 1) − 1) / stride ) + 1.
/// ```
///
/// # Conv is DirectML-only. The HLSL engine declines it.
///
/// A correct, performant convolution shader is a wholly different animal from the
/// naive kernels in [`crate::hlsl`] (tiling, im2col, shared memory).  Rather than
/// ship a slow-and-fragile one that would still have to be proven on hardware, the
/// HLSL path returns [`DirectMLError::Declined`] for `Conv`, and only the DirectML
/// backend — which maps this plan to a Microsoft-validated metacommand — handles it.
/// [`crate::reference::ref_conv`] is the CPU oracle that metacommand is diffed
/// against.
///
/// Every field is already range-checked to fit `u32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvPlan {
    /// Batch size `N`.
    pub batch: u32,
    /// Input channel count `C_in`.
    pub c_in: u32,
    /// Input height `H`.
    pub in_h: u32,
    /// Input width `W`.
    pub in_w: u32,
    /// Output channel count `C_out` — the weight's first dim.
    pub c_out: u32,
    /// Input channels per group — the weight's second dim (`C_in / group`).
    pub c_in_per_group: u32,
    /// Output channels per group (`C_out / group`).
    pub c_out_per_group: u32,
    /// Kernel height `kH`.
    pub kernel_h: u32,
    /// Kernel width `kW`.
    pub kernel_w: u32,
    /// Output height.
    pub out_h: u32,
    /// Output width.
    pub out_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Top padding.
    pub pad_top: u32,
    /// Left padding.
    pub pad_left: u32,
    /// Bottom padding.
    pub pad_bottom: u32,
    /// Right padding.
    pub pad_right: u32,
    /// Vertical dilation.
    pub dilation_h: u32,
    /// Horizontal dilation.
    pub dilation_w: u32,
    /// Group count.
    pub group: u32,
    /// `true` when a bias operand was supplied.
    pub has_bias: bool,
    /// The input shape `[N, C_in, H, W]`.
    pub input_shape: Vec<usize>,
    /// The weight shape `[C_out, C_in/group, kH, kW]`.
    pub weight_shape: Vec<usize>,
    /// The output shape `[N, C_out, out_h, out_w]`.
    pub output_shape: Vec<usize>,
}

impl ConvPlan {
    /// Plan an ONNX 2-D `Conv`.
    ///
    /// `input_shape` is `[N, C_in, H, W]`; `weight_shape` is
    /// `[C_out, C_in/group, kH, kW]`; `bias_shape`, when present, must be `[C_out]`.
    /// `strides` / `dilations` accept an empty slice (ONNX default `1`) or exactly
    /// two entries `[h, w]`; `pads` accepts empty (default `0`) or exactly four
    /// `[top, left, bottom, right]`.
    ///
    /// # Errors
    /// - [`DirectMLError::ShapeMismatch`] — `C_in` / `C_out` are not divisible by
    ///   `group`; the weight's in-channels dim is not `C_in / group`; or the bias
    ///   is not `[C_out]`.  A CPU `Conv` fails on the same inputs.
    /// - [`DirectMLError::Declined`] — input or weight is not rank 4 (this planner
    ///   only expresses the 2-D `[N, C_in, H, W]` case; the CPU `Conv` operator is
    ///   rank-generic — `oxionnx_ops::conv::conv` — and handles Conv1D/Conv3D, so a
    ///   non-rank-4 shape is this backend's capability gap, not a malformed model);
    ///   an attribute list has an unexpected length; a stride or dilation is 0; the
    ///   dilated kernel is larger than the padded input (the output would be
    ///   empty); the output is empty; or a size exceeds `u32::MAX`.
    pub fn conv(
        input_shape: &[usize],
        weight_shape: &[usize],
        bias_shape: Option<&[usize]>,
        strides: &[usize],
        pads: &[usize],
        dilations: &[usize],
        group: usize,
    ) -> Result<Self> {
        if input_shape.len() != 4 {
            return Err(DirectMLError::Declined(format!(
                "Conv: this planner only expresses rank-4 [N, C_in, H, W]; got \
                 {input_shape:?}. The CPU Conv operator is rank-generic and handles \
                 Conv1D/Conv3D."
            )));
        }
        if weight_shape.len() != 4 {
            return Err(DirectMLError::Declined(format!(
                "Conv: this planner only expresses rank-4 [C_out, C_in/group, kH, kW]; got \
                 {weight_shape:?}. The CPU Conv operator is rank-generic and handles \
                 Conv1D/Conv3D."
            )));
        }
        if group == 0 {
            return Err(DirectMLError::Declined("Conv: group is 0".into()));
        }

        let [n, c_in, in_h, in_w] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        let [c_out, w_c_in, kernel_h, kernel_w] = [
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        ];

        if c_in % group != 0 || c_out % group != 0 {
            return Err(DirectMLError::ShapeMismatch(format!(
                "Conv: group {group} does not divide C_in={c_in} and C_out={c_out}"
            )));
        }
        let c_in_per_group = c_in / group;
        let c_out_per_group = c_out / group;
        if w_c_in != c_in_per_group {
            return Err(DirectMLError::ShapeMismatch(format!(
                "Conv: weight in-channels {w_c_in} must equal C_in/group = {c_in_per_group} \
                 (C_in={c_in}, group={group})"
            )));
        }

        let (stride_h, stride_w) = read_pair("Conv strides", strides, 1)?;
        let (dilation_h, dilation_w) = read_pair("Conv dilations", dilations, 1)?;
        let (pad_top, pad_left, pad_bottom, pad_right) = read_quad("Conv pads", pads)?;
        if stride_h == 0 || stride_w == 0 {
            return Err(DirectMLError::Declined(format!(
                "Conv: zero stride [{stride_h}, {stride_w}]"
            )));
        }
        if dilation_h == 0 || dilation_w == 0 {
            return Err(DirectMLError::Declined(format!(
                "Conv: zero dilation [{dilation_h}, {dilation_w}]"
            )));
        }

        let out_h = conv_out_dim(
            "H", in_h, pad_top, pad_bottom, dilation_h, kernel_h, stride_h,
        )?;
        let out_w = conv_out_dim(
            "W", in_w, pad_left, pad_right, dilation_w, kernel_w, stride_w,
        )?;

        let output_shape = vec![n, c_out, out_h, out_w];
        if numel(&output_shape)? == 0 {
            return Err(DirectMLError::Declined(format!(
                "Conv: empty output {output_shape:?}; a D3D12 buffer of Width = 0 cannot be created"
            )));
        }
        // A convolution over an empty input or an empty kernel is equally
        // unrepresentable on the GPU.
        if numel(input_shape)? == 0 || numel(weight_shape)? == 0 {
            return Err(DirectMLError::Declined(format!(
                "Conv: empty input {input_shape:?} or weight {weight_shape:?}"
            )));
        }

        let has_bias = match bias_shape {
            None => false,
            Some(bs) => {
                if bs != [c_out] {
                    return Err(DirectMLError::ShapeMismatch(format!(
                        "Conv: bias must be [C_out] = [{c_out}], got {bs:?}"
                    )));
                }
                true
            }
        };

        Ok(Self {
            batch: checked_u32(n, "Conv N")?,
            c_in: checked_u32(c_in, "Conv C_in")?,
            in_h: checked_u32(in_h, "Conv H")?,
            in_w: checked_u32(in_w, "Conv W")?,
            c_out: checked_u32(c_out, "Conv C_out")?,
            c_in_per_group: checked_u32(c_in_per_group, "Conv C_in/group")?,
            c_out_per_group: checked_u32(c_out_per_group, "Conv C_out/group")?,
            kernel_h: checked_u32(kernel_h, "Conv kH")?,
            kernel_w: checked_u32(kernel_w, "Conv kW")?,
            out_h: checked_u32(out_h, "Conv out_h")?,
            out_w: checked_u32(out_w, "Conv out_w")?,
            stride_h: checked_u32(stride_h, "Conv stride_h")?,
            stride_w: checked_u32(stride_w, "Conv stride_w")?,
            pad_top: checked_u32(pad_top, "Conv pad_top")?,
            pad_left: checked_u32(pad_left, "Conv pad_left")?,
            pad_bottom: checked_u32(pad_bottom, "Conv pad_bottom")?,
            pad_right: checked_u32(pad_right, "Conv pad_right")?,
            dilation_h: checked_u32(dilation_h, "Conv dilation_h")?,
            dilation_w: checked_u32(dilation_w, "Conv dilation_w")?,
            group: checked_u32(group, "Conv group")?,
            has_bias,
            input_shape: input_shape.to_vec(),
            weight_shape: weight_shape.to_vec(),
            output_shape,
        })
    }

    /// Total element count of the output.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn output_elems(&self) -> Result<usize> {
        numel(&self.output_shape)
    }

    /// Byte size of the input buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn input_bytes(&self) -> Result<usize> {
        numel(&self.input_shape)?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("Conv input size overflows usize".into()))
    }

    /// Byte size of the weight buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn weight_bytes(&self) -> Result<usize> {
        numel(&self.weight_shape)?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("Conv weight size overflows usize".into()))
    }

    /// Byte size of the output buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn output_bytes(&self) -> Result<usize> {
        self.output_elems()?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("Conv output size overflows usize".into()))
    }

    /// The length of the dot product that produces one output element:
    /// `c_in_per_group × kH × kW`.
    ///
    /// This is the convolution's "`K`", and it scales
    /// [`crate::reference::Tolerance::for_conv`] the same way `K` scales a matmul's:
    /// the DirectML metacommand accumulates a sum of this length and may contract to
    /// `mad`, so the permitted drift grows like `√depth`.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn reduction_depth(&self) -> Result<usize> {
        (self.c_in_per_group as usize)
            .checked_mul(self.kernel_h as usize)
            .and_then(|v| v.checked_mul(self.kernel_w as usize))
            .ok_or_else(|| DirectMLError::Declined("Conv reduction depth overflows usize".into()))
    }
}

// ─── free helpers ────────────────────────────────────────────────────────────

/// Resolve a possibly-negative ONNX axis against `rank`.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when the axis is out of `[-rank, rank)` — a
/// malformed node the CPU operator would reject too.  A rank-0 tensor rejects every
/// axis.
fn resolve_axis(axis: i64, rank: usize, op: &str) -> Result<usize> {
    let resolved = if axis < 0 { axis + rank as i64 } else { axis };
    if resolved < 0 || resolved as usize >= rank {
        return Err(DirectMLError::ShapeMismatch(format!(
            "{op}: axis {axis} is out of range for a rank-{rank} tensor"
        )));
    }
    Ok(resolved as usize)
}

/// The ONNX reduce output shape: reduced axis → 1 with `keepdims`, else removed.
///
/// Reducing away the only axis of a rank-1 input with `keepdims=0` leaves the
/// **empty** shape — a genuine rank-0 scalar, which is what the ONNX
/// `ReduceSum`/`ReduceMean`/`ReduceMax`/`ReduceMin` spec (and NumPy:
/// `np.sum(np.arange(5), axis=0, keepdims=False).shape == ()`) specifies — not
/// the rank-1 `[1]` this used to promote it to. That promotion made this planner
/// disagree with the CPU kernel it must be interchangeable with
/// (`reduce_output_shape` / `reduce_with` in oxionnx-ops/src/math/reduce.rs, and
/// `single_axis_reduce_shape` in oxionnx's `src/session/gpu_dispatch.rs`), which
/// all report `[]`.
///
/// Nothing downstream is sized differently by this: the empty shape's product is
/// the empty product 1 — exactly the one element a full reduction writes — so
/// [`ReducePlan::output_elems`], [`ReducePlan::output_bytes`] and `out_count`
/// (`outer × inner`, both 1 here) are unchanged, and `DmlTensorLayout::packed`
/// pads `[]` to the same `[1,1,1,1]` DirectML descriptor it padded `[1]` to.
fn reduce_output_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        let mut out = shape.to_vec();
        out[axis] = 1;
        out
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == axis { None } else { Some(d) })
            .collect()
    }
}

/// Read an ONNX `[h, w]` attribute list: empty → `[default, default]`, exactly two →
/// as given, anything else declined.
///
/// # Errors
/// [`DirectMLError::Declined`] on any other length.
fn read_pair(name: &str, values: &[usize], default: usize) -> Result<(usize, usize)> {
    match values {
        [] => Ok((default, default)),
        [h, w] => Ok((*h, *w)),
        other => Err(DirectMLError::Declined(format!(
            "{name}: expected 0 or 2 entries, got {}",
            other.len()
        ))),
    }
}

/// Read an ONNX `[top, left, bottom, right]` pad list: empty → all `0`, exactly four
/// → as given, anything else declined.
///
/// # Errors
/// [`DirectMLError::Declined`] on any other length.
fn read_quad(name: &str, values: &[usize]) -> Result<(usize, usize, usize, usize)> {
    match values {
        [] => Ok((0, 0, 0, 0)),
        [a, b, c, d] => Ok((*a, *b, *c, *d)),
        other => Err(DirectMLError::Declined(format!(
            "{name}: expected 0 or 4 entries, got {}",
            other.len()
        ))),
    }
}

/// One spatial output dimension:
/// `floor((in + pad_begin + pad_end − dilation·(k − 1) − 1) / stride) + 1`.
///
/// The numerator is evaluated in `i64` precisely so that a dilated kernel larger than
/// the padded input produces a **decline**, not a `usize` underflow that would wrap to
/// an enormous, wrong output extent.
///
/// # Errors
/// [`DirectMLError::Declined`] when the numerator is negative (kernel does not fit).
fn conv_out_dim(
    which: &str,
    input: usize,
    pad_begin: usize,
    pad_end: usize,
    dilation: usize,
    kernel: usize,
    stride: usize,
) -> Result<usize> {
    let effective = 1i64 + (dilation as i64) * (kernel as i64 - 1);
    let numerator = input as i64 + pad_begin as i64 + pad_end as i64 - effective;
    if numerator < 0 {
        return Err(DirectMLError::Declined(format!(
            "Conv: dilated kernel does not fit the padded {which} extent \
             (in={input}, pads={pad_begin}+{pad_end}, dilation={dilation}, kernel={kernel})"
        )));
    }
    Ok((numerator / stride as i64) as usize + 1)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn declined(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::Declined(_))
    }
    fn mismatch(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::ShapeMismatch(_))
    }

    // ── ReduceKind ───────────────────────────────────────────────────────────

    #[test]
    fn reduce_kind_names_and_exactness() {
        assert_eq!(ReduceKind::Sum.as_str(), "ReduceSum");
        assert_eq!(ReduceKind::Mean.as_str(), "ReduceMean");
        assert_eq!(ReduceKind::Max.as_str(), "ReduceMax");
        assert_eq!(ReduceKind::Min.as_str(), "ReduceMin");
        assert!(!ReduceKind::Sum.is_exact());
        assert!(!ReduceKind::Mean.is_exact());
        assert!(ReduceKind::Max.is_exact());
        assert!(ReduceKind::Min.is_exact());
    }

    // ── SoftmaxPlan ──────────────────────────────────────────────────────────

    #[test]
    fn softmax_last_axis_decomposition() {
        let p = SoftmaxPlan::softmax(&[2, 3, 4], 2).unwrap();
        assert_eq!(p.axis, 2);
        assert_eq!(p.outer, 6, "2 * 3");
        assert_eq!(p.axis_len, 4);
        assert_eq!(p.inner, 1);
        assert_eq!(p.rows, 6);
        assert!(p.reduces_last_axis());
        assert_eq!(p.output_shape(), &[2, 3, 4]);
        assert_eq!(p.output_elems().unwrap(), 24);
        assert_eq!(p.buffer_bytes().unwrap(), 24 * 4);
    }

    #[test]
    fn softmax_middle_axis_has_nonunit_inner_and_is_not_last() {
        let p = SoftmaxPlan::softmax(&[2, 3, 4], 1).unwrap();
        assert_eq!(p.axis, 1);
        assert_eq!(p.outer, 2);
        assert_eq!(p.axis_len, 3);
        assert_eq!(p.inner, 4);
        assert_eq!(p.rows, 8, "outer 2 * inner 4");
        assert!(
            !p.reduces_last_axis(),
            "axis 1 of a rank-3 tensor is not terminal"
        );
    }

    #[test]
    fn softmax_resolves_a_negative_axis() {
        let p = SoftmaxPlan::softmax(&[5, 7], -1).unwrap();
        assert_eq!(p.axis, 1);
        assert_eq!(p.axis_len, 7);
        assert_eq!(p.inner, 1);
        let p = SoftmaxPlan::softmax(&[5, 7], -2).unwrap();
        assert_eq!(p.axis, 0);
        assert_eq!(p.axis_len, 5);
        assert_eq!(p.inner, 7);
    }

    #[test]
    fn softmax_out_of_range_axis_is_a_shape_error() {
        assert!(mismatch(&SoftmaxPlan::softmax(&[2, 3], 2).unwrap_err()));
        assert!(mismatch(&SoftmaxPlan::softmax(&[2, 3], -3).unwrap_err()));
        // A scalar has no axis at all.
        assert!(mismatch(&SoftmaxPlan::softmax(&[], 0).unwrap_err()));
    }

    #[test]
    fn softmax_empty_tensor_is_declined() {
        assert!(declined(&SoftmaxPlan::softmax(&[0, 4], 1).unwrap_err()));
        assert!(declined(&SoftmaxPlan::softmax(&[3, 0], 0).unwrap_err()));
    }

    #[test]
    fn softmax_grid_and_constants_agree_on_groups_x() {
        for shape in [vec![1usize], vec![256, 4], vec![100_000], vec![4, 50_000]] {
            let axis = (shape.len() - 1) as i64;
            let p = SoftmaxPlan::softmax(&shape, axis).unwrap();
            let grid = p.hlsl_grid().unwrap();
            let consts = p.constants().unwrap();
            assert_eq!(
                consts.groups_x, grid.x,
                "SoftmaxConstants::groups_x must equal the dispatched grid's x"
            );
            assert_eq!(consts.rows, p.rows);
            assert_eq!(consts.axis_len, p.axis_len);
            assert_eq!(consts.inner, p.inner);
            assert!(
                grid.total_groups() * u64::from(REDUCTION_THREADS_PER_GROUP) >= u64::from(p.rows),
                "grid {grid:?} under-covers {} rows",
                p.rows
            );
        }
    }

    #[test]
    fn softmax_constants_serialise_in_cbuffer_order() {
        let c = SoftmaxConstants {
            rows: 6,
            groups_x: 1,
            axis_len: 4,
            inner: 1,
            ..SoftmaxConstants::default()
        };
        assert_eq!(c.to_root_constants(), [6, 1, 4, 1, 0, 0, 0, 0]);
        assert_eq!(
            core::mem::size_of::<SoftmaxConstants>(),
            ROOT_CONSTANT_COUNT * 4
        );
        assert_eq!(c.const_buffer_bytes().len(), super::super::CBV_ALIGNMENT);
    }

    #[test]
    fn softmax_declines_a_row_count_beyond_u32() {
        // The reduction grid can never *itself* exceed the D3D12 limit: with 256 threads
        // per group, even the largest representable row count (`u32::MAX`) folds into at
        // most `ceil(u32::MAX / 256) ≈ 16.7M` groups, comfortably inside `65535 × 65535`.
        // So "beyond what the shader can index" is the `u32` range check at construction —
        // a shape whose outer size or row count overflows `u32` is declined *here*, before
        // any grid is built.
        assert!(declined(
            &SoftmaxPlan::softmax(&[5_000_000_000, 4], 1).unwrap_err()
        ));
        // And every representable plan yields a grid that fits.
        let huge = SoftmaxPlan::softmax(&[u32::MAX as usize, 2], 1).unwrap();
        let grid = huge.hlsl_grid().unwrap();
        assert!(grid.x <= DispatchGrid::MAX_GROUPS_PER_DIM);
        assert!(grid.y <= DispatchGrid::MAX_GROUPS_PER_DIM);
    }

    // ── ReducePlan ───────────────────────────────────────────────────────────

    #[test]
    fn reduce_single_axis_keepdims_and_squeeze() {
        let keep = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], true).unwrap();
        assert_eq!(keep.axis, 1);
        assert_eq!(keep.outer, 2);
        assert_eq!(keep.axis_len, 3);
        assert_eq!(keep.inner, 4);
        assert_eq!(keep.out_count, 8);
        assert_eq!(keep.output_shape, vec![2, 1, 4]);
        assert_eq!(keep.output_elems().unwrap(), 8);

        let squeeze = ReducePlan::reduce(ReduceKind::Mean, &[2, 3, 4], &[1], false).unwrap();
        assert_eq!(squeeze.output_shape, vec![2, 4]);
        assert_eq!(squeeze.out_count, 8);
    }

    #[test]
    fn reduce_resolves_a_negative_axis() {
        let p = ReducePlan::reduce(ReduceKind::Max, &[2, 3, 4], &[-1], false).unwrap();
        assert_eq!(p.axis, 2);
        assert_eq!(p.axis_len, 4);
        assert_eq!(p.inner, 1);
        assert_eq!(p.output_shape, vec![2, 3]);
    }

    #[test]
    fn reduce_full_collapse_of_a_vector_is_rank0() {
        // Empty axes on a rank-1 tensor is the single-axis case.
        //
        // ONNX `ReduceSum`: with `keepdims=0` the reduced axes are *removed*, so
        // reducing the only axis of a rank-1 input leaves a rank-0 scalar —
        // `np.sum(np.arange(5), axis=0, keepdims=False).shape == ()`. This used to
        // assert `[1]`, which made the planner disagree with the CPU kernel
        // (`oxionnx_ops::math::reduce_sum`) it must be a drop-in replacement for;
        // `tests/reference_vs_ops.rs` cross-checks exactly that agreement.
        let squeeze = ReducePlan::reduce(ReduceKind::Sum, &[5], &[], false).unwrap();
        assert_eq!(squeeze.axis, 0);
        let rank0: Vec<usize> = Vec::new();
        assert_eq!(
            squeeze.output_shape, rank0,
            "a fully collapsed shape is [] (rank 0), not [1]"
        );
        // The element count is unaffected — the empty shape's product is 1.
        assert_eq!(squeeze.output_elems().unwrap(), 1);
        assert_eq!(squeeze.out_count, 1);

        // `keepdims=1` is unchanged: the reduced axis stays as a size-1 dim, so a
        // rank-1 input keeps rank 1.
        let keep = ReducePlan::reduce(ReduceKind::Sum, &[5], &[], true).unwrap();
        assert_eq!(keep.output_shape, vec![1]);
    }

    #[test]
    fn reduce_multi_axis_is_declined_not_silently_wrong() {
        // Two explicit axes.
        let e = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[0, 2], false).unwrap_err();
        assert!(declined(&e), "got {e:?}");
        // Empty axes on rank >= 2 means "all axes" == multi-axis.
        let e = ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[], false).unwrap_err();
        assert!(declined(&e), "got {e:?}");
        // Duplicate axes collapse to one and are accepted.
        let p = ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[1, 1, -1], false).unwrap();
        assert_eq!(p.axis, 1);
    }

    #[test]
    fn reduce_out_of_range_axis_is_a_shape_error() {
        assert!(mismatch(
            &ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[2], false).unwrap_err()
        ));
        assert!(mismatch(
            &ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[-3], false).unwrap_err()
        ));
    }

    #[test]
    fn reduce_empty_tensor_is_declined() {
        assert!(declined(
            &ReducePlan::reduce(ReduceKind::Sum, &[0, 4], &[1], false).unwrap_err()
        ));
    }

    #[test]
    fn reduce_grid_and_constants_agree_on_groups_x() {
        let p = ReducePlan::reduce(ReduceKind::Sum, &[300, 128], &[1], false).unwrap();
        assert_eq!(p.out_count, 300);
        let grid = p.hlsl_grid().unwrap();
        let consts = p.constants().unwrap();
        assert_eq!(consts.groups_x, grid.x);
        assert_eq!(consts.out_count, 300);
        assert_eq!(consts.axis_len, 128);
        assert_eq!(consts.inner, 1);
        assert_eq!(
            consts.to_root_constants(),
            [300, grid.x, 128, 1, 0, 0, 0, 0]
        );
        assert_eq!(
            core::mem::size_of::<ReduceConstants>(),
            ROOT_CONSTANT_COUNT * 4
        );
    }

    // ── ConvPlan ─────────────────────────────────────────────────────────────

    #[test]
    fn conv_basic_output_shape() {
        // 1x1x5x5 input, 4 filters of 3x3, stride 1, no pad, dilation 1.
        let p = ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[], &[], &[], 1).unwrap();
        assert_eq!(p.out_h, 3, "(5 - (3-1) - 1)/1 + 1 = 3");
        assert_eq!(p.out_w, 3);
        assert_eq!(p.output_shape, vec![1, 4, 3, 3]);
        assert_eq!(p.reduction_depth().unwrap(), 9, "1 * 3 * 3");
        assert!(!p.has_bias);
    }

    #[test]
    fn conv_stride_pad_dilation() {
        // 1x3x8x8, 6 filters 3x3, stride 2, pad 1 all sides, dilation 1.
        let p = ConvPlan::conv(
            &[1, 3, 8, 8],
            &[6, 3, 3, 3],
            None,
            &[2, 2],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
        )
        .unwrap();
        // (8 + 1 + 1 - 2 - 1)/2 + 1 = 7/2 + 1 = 3 + 1 = 4.
        assert_eq!(p.out_h, 4);
        assert_eq!(p.out_w, 4);
        assert_eq!(p.output_shape, vec![1, 6, 4, 4]);
        assert_eq!(p.reduction_depth().unwrap(), 27, "3 * 3 * 3");

        // Dilation widens the effective kernel: dilation 2, kernel 3 → effective 5.
        let p =
            ConvPlan::conv(&[1, 1, 9, 9], &[1, 1, 3, 3], None, &[1, 1], &[], &[2, 2], 1).unwrap();
        // (9 - 2*(3-1) - 1)/1 + 1 = (9 - 4 - 1) + 1 = 5.
        assert_eq!(p.out_h, 5);
        assert_eq!(p.out_w, 5);
    }

    #[test]
    fn conv_grouped() {
        // group=2: C_in=4, C_out=6, weight in-channels = 4/2 = 2.
        let p = ConvPlan::conv(&[1, 4, 5, 5], &[6, 2, 3, 3], None, &[], &[], &[], 2).unwrap();
        assert_eq!(p.group, 2);
        assert_eq!(p.c_in_per_group, 2);
        assert_eq!(p.c_out_per_group, 3);
        assert_eq!(p.reduction_depth().unwrap(), 18, "2 * 3 * 3");
        assert_eq!(p.output_shape, vec![1, 6, 3, 3]);
    }

    #[test]
    fn conv_bias_must_be_c_out() {
        let ok =
            ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], Some(&[4]), &[], &[], &[], 1).unwrap();
        assert!(ok.has_bias);
        let bad =
            ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], Some(&[3]), &[], &[], &[], 1).unwrap_err();
        assert!(mismatch(&bad), "got {bad:?}");
    }

    #[test]
    fn conv_group_and_channel_mismatches_are_shape_errors() {
        // C_in not divisible by group.
        assert!(mismatch(
            &ConvPlan::conv(&[1, 5, 5, 5], &[6, 2, 3, 3], None, &[], &[], &[], 2).unwrap_err()
        ));
        // Weight in-channels != C_in/group.
        assert!(mismatch(
            &ConvPlan::conv(&[1, 4, 5, 5], &[6, 3, 3, 3], None, &[], &[], &[], 2).unwrap_err()
        ));
    }

    #[test]
    fn conv_non_rank4_is_declined_not_a_shape_error() {
        // This planner only expresses the 2-D case; the CPU `Conv` operator is
        // rank-generic (Conv1D/Conv3D), so a non-rank-4 shape is a capability
        // gap this backend declines to the CPU — not a malformed model — now
        // that CPU `Conv` no longer rejects rank 3/5 either.
        assert!(declined(
            &ConvPlan::conv(&[1, 5, 5], &[4, 1, 3, 3], None, &[], &[], &[], 1).unwrap_err()
        ));
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3], None, &[], &[], &[], 1).unwrap_err()
        ));
    }

    #[test]
    fn conv_kernel_larger_than_padded_input_is_declined() {
        // 3x3 kernel over a 2x2 input with no padding: numerator (2 - 2 - 1) < 0.
        let e = ConvPlan::conv(&[1, 1, 2, 2], &[1, 1, 3, 3], None, &[], &[], &[], 1).unwrap_err();
        assert!(declined(&e), "got {e:?}");
    }

    #[test]
    fn conv_bad_attribute_lengths_and_zero_stride_are_declined() {
        // strides must be 0 or 2 long.
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[1], &[], &[], 1).unwrap_err()
        ));
        // pads must be 0 or 4 long.
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[], &[1, 1], &[], 1).unwrap_err()
        ));
        // zero stride.
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[0, 1], &[], &[], 1).unwrap_err()
        ));
        // zero group.
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[], &[], &[], 0).unwrap_err()
        ));
    }

    #[test]
    fn conv_out_dim_matches_the_standard_formula() {
        // Directly exercise the helper on the ONNX reference numbers.
        assert_eq!(conv_out_dim("H", 5, 0, 0, 1, 3, 1).unwrap(), 3);
        assert_eq!(conv_out_dim("H", 7, 1, 1, 1, 3, 2).unwrap(), 4);
        assert_eq!(conv_out_dim("H", 9, 0, 0, 2, 3, 1).unwrap(), 5);
        assert_eq!(conv_out_dim("H", 1, 0, 0, 1, 1, 1).unwrap(), 1);
        assert!(conv_out_dim("H", 2, 0, 0, 1, 3, 1).is_err());
    }
}
