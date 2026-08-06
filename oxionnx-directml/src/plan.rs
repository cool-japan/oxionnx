//! Platform-neutral planning: shape validation, dispatch-grid math, buffer
//! sizing, and root-constant payloads.
//!
//! There is not a single `#[cfg]` in this file.  It is compiled on every target
//! and fully unit-tested on Linux, where no D3D12 device and no Windows host
//! exist.  Both backends consume these types *unchanged* — a [`MatMulPlan`]
//! produced on Linux and one produced on Windows are bit-identical, which is
//! precisely what makes the Linux tests meaningful rather than decorative.
//!
//! # The `u32` rule
//!
//! Every dimension, element count and buffer offset in this crate crosses the FFI
//! boundary as a `u32` (HLSL's index type; DirectML's `Sizes`/`Strides` element
//! type).  **Every shape-derived value is range-checked exactly once, here, with
//! [`checked_u32`].**  Downstream code reads the pre-checked `u32` fields of
//! [`MatMulPlan`] / [`ElementwisePlan`] / [`crate::layout::DmlTensorLayout`] and
//! **never writes a bare `as u32`** on anything derived from a shape.  A silent
//! truncation here would not crash: it would dispatch a grid that covers a
//! fraction of the output and leave the rest of the buffer as whatever the last
//! allocation left behind.
//!
//! # What this wave deliberately refuses
//!
//! * **MatMul is 2-D × 2-D only** — see [`MatMulPlan::matmul`].
//! * **Binary elementwise requires identical shapes** — see [`ElementwisePlan::binary`].
//! * **Every operand must be non-empty** (`numel() > 0`).
//!
//! Each refusal is a [`DirectMLError::Declined`], which the router turns into
//! `Ok(None)` → the tuned CPU kernel runs and produces the right answer.  A
//! declined op is a *correct* op.

use std::borrow::Cow;

use crate::error::{DirectMLError, Result};

/// Wave-4 neural-network plans (`Softmax`, `Reduce`, `Conv`), kept in a child module
/// so this file stays under the 2000-line refactor ceiling.  Everything it exports is
/// re-exported here, so the public contract is `plan::SoftmaxPlan`, not
/// `plan::nn::SoftmaxPlan`.
pub mod nn;

pub use nn::{
    ConvPlan, ReduceConstants, ReduceKind, ReducePlan, SoftmaxConstants, SoftmaxPlan,
    REDUCTION_THREADS_PER_GROUP,
};

// ─── constants ───────────────────────────────────────────────────────────────

/// This crate is f32-only: `oxionnx_core::Tensor::data` is a `Vec<f32>`.
pub const ELEM_SIZE: usize = core::mem::size_of::<f32>();

/// D3D12 requires a constant-buffer view's `BufferLocation` **and** `SizeInBytes`
/// to be 256-byte aligned.  A 16-byte CBV is a hard debug-layer error and, in
/// retail, a read of garbage.
///
/// The HLSL backend avoids CBVs entirely (it uses `SetComputeRoot32BitConstants`,
/// so there *is* no constant buffer to misalign), but
/// [`MatMulConstants::const_buffer_bytes`] and
/// [`ElementwiseConstants::const_buffer_bytes`] are the correct-by-construction
/// path for anyone who later adds one.  **No code in this crate may create a CBV
/// without routing its payload through one of those two functions.**
pub const CBV_ALIGNMENT: usize = 256;

/// `DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT` — every DirectML buffer *binding* offset
/// must be a multiple of this.  This crate always binds at offset 0, so the
/// constraint is satisfied trivially; the constant exists so that a future
/// sub-buffer allocator has the number to hand.
pub const DML_BUFFER_ALIGNMENT: usize = 16;

/// `DML_BUFFER_TENSOR_DESC::TotalTensorSizeInBytes` must be a multiple of 4.
pub const DML_TENSOR_SIZE_GRANULARITY: usize = 4;

/// The elementwise HLSL kernels are `[numthreads(256, 1, 1)]`.
pub const ELEMENTWISE_THREADS_PER_GROUP: u32 = 256;

/// The MatMul HLSL kernel is `[numthreads(16, 16, 1)]`.
pub const MATMUL_TILE: u32 = 16;

/// Number of 32-bit root constants in the single shared compute root signature.
///
/// Both [`MatMulConstants`] and [`ElementwiseConstants`] are exactly this wide, so
/// one root signature serves all of the HLSL entry points in [`crate::hlsl`].
pub const ROOT_CONSTANT_COUNT: usize = 8;

// ─── free functions ──────────────────────────────────────────────────────────

/// The **only** sanctioned `usize` → `u32` conversion in this crate.
///
/// `what` names the quantity, so a decline reads
/// `"DirectML backend declined op: MatMul K = 5000000000 exceeds u32::MAX"`
/// rather than producing a silently truncated dispatch.
///
/// # Errors
/// [`DirectMLError::Declined`] when `value > u32::MAX`.
pub fn checked_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| DirectMLError::Declined(format!("{what} = {value} exceeds u32::MAX")))
}

/// The product of `shape`'s dimensions, as a `usize`, without overflowing.
///
/// # Errors
/// [`DirectMLError::Declined`] when the product overflows `usize`.
pub fn numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| DirectMLError::Declined(format!("shape {shape:?} overflows usize")))
    })
}

/// Round `n` up to the next multiple of `align`.
///
/// `None` on overflow, or when `align == 0`.
#[must_use]
pub fn align_up(n: usize, align: usize) -> Option<usize> {
    if align == 0 {
        return None;
    }
    match n % align {
        0 => Some(n),
        rem => n.checked_add(align - rem),
    }
}

/// Ceiling division.  `ceil_div(0, d) == Some(0)`; `d == 0` → `None`.
#[must_use]
pub fn ceil_div(n: u32, d: u32) -> Option<u32> {
    if d == 0 {
        return None;
    }
    // `n / d` is at most `n`, and the `+ 1` only happens when `n % d != 0`, which
    // implies `d >= 2` and therefore `n / d <= n / 2 < u32::MAX`.  No overflow.
    Some(n / d + u32::from(n % d != 0))
}

/// Numpy broadcast of two shapes: right-align, then per axis take the larger of
/// the two dims when the other is 1.
///
/// This is a *helper*, not a policy: [`ElementwisePlan::binary`] declines every
/// pair of shapes that is not already identical (see its documentation).  This
/// function exists because [`crate::reference`]'s CPU oracle and any future
/// broadcast-capable wave need the same, single, tested implementation.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when the shapes are not broadcastable — the
/// CPU operator would fail on the same inputs.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        // Right-aligned: axis `i` of the output corresponds to axis
        // `i - (rank - a.len())` of `a`, and 1 when that is out of range.
        let da = axis_or_one(a, rank, i);
        let db = axis_or_one(b, rank, i);
        let d = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(DirectMLError::ShapeMismatch(format!(
                "shapes {a:?} and {b:?} are not broadcastable (axis {i}: {da} vs {db})"
            )));
        };
        out.push(d);
    }
    Ok(out)
}

/// Dimension `i` of `shape` after left-padding it with 1s to `rank`.
fn axis_or_one(shape: &[usize], rank: usize, i: usize) -> usize {
    let pad = rank - shape.len();
    if i < pad {
        1
    } else {
        shape[i - pad]
    }
}

/// Densely expand `src` (of shape `src_shape`) to `dst_shape` by numpy rules.
///
/// Returns `Cow::Borrowed(src)` when no expansion is needed — the common case, so
/// the caller pays nothing when the shapes already match.
///
/// Used only where a *materialised* dense operand is required (the CPU oracle,
/// and — if a later wave lifts the no-broadcast restriction — the HLSL backend).
/// The DirectML backend expresses the same broadcast with 0-strides in the tensor
/// descriptor and copies nothing; see [`crate::layout::DmlTensorLayout::broadcast_to`].
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `src.len()` disagrees with `src_shape`,
/// or when `src_shape` does not broadcast to `dst_shape`.
/// [`DirectMLError::Declined`] when the destination element count overflows `usize`.
pub fn broadcast_expand<'a>(
    src: &'a [f32],
    src_shape: &[usize],
    dst_shape: &[usize],
) -> Result<Cow<'a, [f32]>> {
    let src_elems = numel(src_shape)?;
    if src.len() != src_elems {
        return Err(DirectMLError::ShapeMismatch(format!(
            "buffer of {} elements does not match shape {src_shape:?} ({src_elems} elements)",
            src.len()
        )));
    }
    if src_shape == dst_shape {
        return Ok(Cow::Borrowed(src));
    }

    let rank = dst_shape.len();
    if src_shape.len() > rank {
        return Err(DirectMLError::ShapeMismatch(format!(
            "cannot broadcast rank-{} shape {src_shape:?} down to rank-{rank} {dst_shape:?}",
            src_shape.len()
        )));
    }

    // Padded source dims, and the source's own packed strides — zeroed on every
    // axis that is being broadcast, which makes the index arithmetic below a
    // plain dot product.
    let padded: Vec<usize> = (0..rank).map(|i| axis_or_one(src_shape, rank, i)).collect();
    for (i, (&pd, &dd)) in padded.iter().zip(dst_shape.iter()).enumerate() {
        if pd != dd && pd != 1 {
            return Err(DirectMLError::ShapeMismatch(format!(
                "cannot broadcast {src_shape:?} to {dst_shape:?} (axis {i}: {pd} vs {dd})"
            )));
        }
    }
    let mut strides = vec![0usize; rank];
    let mut acc = 1usize;
    for (stride, &pd) in strides.iter_mut().zip(padded.iter()).rev() {
        *stride = if pd == 1 { 0 } else { acc };
        acc *= pd;
    }

    let dst_elems = numel(dst_shape)?;
    let mut out = Vec::with_capacity(dst_elems);
    let mut coord = vec![0usize; rank];
    for _ in 0..dst_elems {
        let off: usize = coord.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum();
        out.push(src[off]);
        // Odometer increment, least-significant axis first.
        for (c, &dim) in coord.iter_mut().zip(dst_shape.iter()).rev() {
            *c += 1;
            if *c < dim {
                break;
            }
            *c = 0;
        }
    }
    Ok(Cow::Owned(out))
}

/// Transpose a 2-D `rows × cols` row-major buffer into a `cols × rows` one.
///
/// Only ever needed for `Gemm` with `transA` / `transB`, whose transposed operand
/// is almost always a constant weight.  The DirectML backend never calls this —
/// `DML_GEMM_OPERATOR_DESC::TransA` / `TransB` do it on-device.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `src.len() != rows * cols`.
/// [`DirectMLError::Declined`] when `rows * cols` overflows `usize`.
pub fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| DirectMLError::Declined(format!("{rows} x {cols} overflows usize")))?;
    if src.len() != expected {
        return Err(DirectMLError::ShapeMismatch(format!(
            "transpose_2d: buffer of {} elements is not {rows} x {cols}",
            src.len()
        )));
    }
    let mut out = vec![0.0f32; expected];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    Ok(out)
}

/// Apply `Gemm`'s `alpha` scale and `beta * C` bias to a finished `[batch, m, n]`
/// buffer, in place.
///
/// Used only by the HLSL backend, whose shader computes the bare product; the
/// DirectML backend folds both into `DML_GEMM_OPERATOR_DESC` and never calls this.
/// `c` is broadcast against `[m, n]` per ONNX rules and re-applied to every batch
/// slice.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `out` is not `plan.output_elems()` long,
/// when `plan.has_bias()` but `c` is `None`, or when `c` does not broadcast to
/// `[m, n]`.
pub fn apply_gemm_epilogue(plan: &MatMulPlan, out: &mut [f32], c: Option<&[f32]>) -> Result<()> {
    let expected = plan.output_elems()?;
    if out.len() != expected {
        return Err(DirectMLError::ShapeMismatch(format!(
            "gemm epilogue: output buffer of {} elements is not {expected}",
            out.len()
        )));
    }

    if plan.alpha != 1.0 {
        for v in out.iter_mut() {
            *v *= plan.alpha;
        }
    }

    if !plan.has_bias() {
        return Ok(());
    }

    let c_shape = plan.c_shape.as_ref().ok_or_else(|| {
        DirectMLError::ShapeMismatch("gemm epilogue: beta != 0 but no C shape in plan".into())
    })?;
    let c_data = c.ok_or_else(|| {
        DirectMLError::ShapeMismatch("gemm epilogue: beta != 0 but no C buffer supplied".into())
    })?;

    let m = plan.m as usize;
    let n = plan.n as usize;
    let slice_elems = m * n; // `m * n <= u32::MAX` was checked when the plan was built.
    let expanded = broadcast_expand(c_data, c_shape, &[m, n])?;

    for slice in out.chunks_exact_mut(slice_elems) {
        for (dst, &bias) in slice.iter_mut().zip(expanded.iter()) {
            *dst += plan.beta * bias;
        }
    }
    Ok(())
}

// ─── dispatch grid ───────────────────────────────────────────────────────────

/// A D3D12 `Dispatch(x, y, z)` thread-group count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchGrid {
    /// Thread groups along X.
    pub x: u32,
    /// Thread groups along Y.
    pub y: u32,
    /// Thread groups along Z.
    pub z: u32,
}

impl DispatchGrid {
    /// D3D12's hard per-dimension limit
    /// (`D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION`).
    pub const MAX_GROUPS_PER_DIM: u32 = 65_535;

    /// Fold `groups` linear thread-groups into a 2-D grid that respects the
    /// per-dimension limit: `x = min(groups, 65535)`, `y = ceil(groups / x)`.
    ///
    /// The shader recovers the linear group index as `gid.y * GroupsX + gid.x`,
    /// where `GroupsX` is [`ElementwiseConstants::groups_x`] — which **must** be
    /// exactly this function's `x`.  Getting those two out of sync computes the
    /// wrong elements *silently*, so [`ElementwisePlan::hlsl_grid`] and
    /// [`ElementwisePlan::constants`] both derive from this one call rather than
    /// each re-deriving it.
    ///
    /// `y` may overshoot: `x * y >= groups`.  The shader's `if (i >= N) return;`
    /// guard absorbs the surplus, which is why that guard is not optional.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `groups == 0`, or when `groups` needs more
    /// than `65535 * 65535` thread groups.
    pub fn linear(groups: u32) -> Result<Self> {
        if groups == 0 {
            return Err(DirectMLError::Declined(
                "dispatch grid: zero thread groups".into(),
            ));
        }
        let x = groups.min(Self::MAX_GROUPS_PER_DIM);
        // `x >= 1` here, so `ceil_div` cannot return `None`.
        let y = ceil_div(groups, x)
            .ok_or_else(|| DirectMLError::Declined("dispatch grid: zero groups along X".into()))?;
        if y > Self::MAX_GROUPS_PER_DIM {
            return Err(DirectMLError::Declined(format!(
                "dispatch grid: {groups} thread groups exceed the D3D12 limit of {} x {}",
                Self::MAX_GROUPS_PER_DIM,
                Self::MAX_GROUPS_PER_DIM
            )));
        }
        Ok(Self { x, y, z: 1 })
    }

    /// A 2-D tile grid: `x = ceil(cols / tile)`, `y = ceil(rows / tile)`, `z = 1`.
    ///
    /// Note the order.  `x` counts **columns** and `y` counts **rows**, because
    /// the MatMul shader reads `row = tid.y; col = tid.x`.  See
    /// [`MatMulPlan::hlsl_grid`] for why this is the single easiest thing in the
    /// whole crate to get backwards.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `tile == 0`, when `rows` or `cols` is 0, or
    /// when either resulting dimension exceeds [`Self::MAX_GROUPS_PER_DIM`].
    pub fn tiled_2d(rows: u32, cols: u32, tile: u32) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(DirectMLError::Declined(format!(
                "dispatch grid: empty tile grid {rows} x {cols}"
            )));
        }
        let x = ceil_div(cols, tile)
            .ok_or_else(|| DirectMLError::Declined("dispatch grid: tile size 0".into()))?;
        let y = ceil_div(rows, tile)
            .ok_or_else(|| DirectMLError::Declined("dispatch grid: tile size 0".into()))?;
        if x > Self::MAX_GROUPS_PER_DIM || y > Self::MAX_GROUPS_PER_DIM {
            return Err(DirectMLError::Declined(format!(
                "dispatch grid: {rows} x {cols} in tiles of {tile} needs {x} x {y} groups, \
                 above the D3D12 limit of {}",
                Self::MAX_GROUPS_PER_DIM
            )));
        }
        Ok(Self { x, y, z: 1 })
    }

    /// `x * y * z`, widened so it cannot overflow.
    #[must_use]
    pub fn total_groups(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }
}

// ─── root-constant payloads ──────────────────────────────────────────────────

/// Serialise `values` into a 256-byte, zero-padded constant block.
fn pad_to_cbv(values: &[u32; ROOT_CONSTANT_COUNT]) -> [u8; CBV_ALIGNMENT] {
    let mut bytes = [0u8; CBV_ALIGNMENT];
    for (i, v) in values.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// The MatMul kernel's `b0` block.
///
/// **Field order is load-bearing.**  `SetComputeRoot32BitConstants` copies these
/// eight `u32`s verbatim into `b0`, so this `#[repr(C)]` layout must match the
/// `cbuffer` declaration in [`crate::hlsl::MATMUL_HLSL`] exactly.  Reordering a
/// field here compiles cleanly and silently computes garbage.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatMulConstants {
    /// Rows of the output slice.
    pub m: u32,
    /// Shared inner dimension.
    pub k: u32,
    /// Columns of the output slice.
    pub n: u32,
    /// Element offset of this batch slice into the `A` buffer.
    pub a_offset: u32,
    /// Element offset of this batch slice into the `B` buffer.
    pub b_offset: u32,
    /// Element offset of this batch slice into the `C` (output) buffer.
    pub c_offset: u32,
    /// Padding, so the block is exactly [`ROOT_CONSTANT_COUNT`] `u32`s wide.
    pub pad0: u32,
    /// Padding, so the block is exactly [`ROOT_CONSTANT_COUNT`] `u32`s wide.
    pub pad1: u32,
}

impl MatMulConstants {
    /// The eight `u32`s, in `cbuffer` order, for `SetComputeRoot32BitConstants`.
    #[must_use]
    pub fn to_root_constants(self) -> [u32; ROOT_CONSTANT_COUNT] {
        [
            self.m,
            self.k,
            self.n,
            self.a_offset,
            self.b_offset,
            self.c_offset,
            self.pad0,
            self.pad1,
        ]
    }

    /// The same payload padded to [`CBV_ALIGNMENT`], for a CBV-based variant.
    #[must_use]
    pub fn const_buffer_bytes(self) -> [u8; CBV_ALIGNMENT] {
        pad_to_cbv(&self.to_root_constants())
    }
}

/// The elementwise kernels' `b0` block.  Same layout warning as [`MatMulConstants`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ElementwiseConstants {
    /// Total element count of the output.
    pub n: u32,
    /// [`DispatchGrid::x`] from [`DispatchGrid::linear`].  The shader multiplies
    /// `gid.y` by this to recover its linear group index, so it must equal the `x`
    /// actually passed to `Dispatch`.
    pub groups_x: u32,
    /// Padding, so the block is exactly [`ROOT_CONSTANT_COUNT`] `u32`s wide.
    pub pad0: u32,
    /// Padding.
    pub pad1: u32,
    /// Padding.
    pub pad2: u32,
    /// Padding.
    pub pad3: u32,
    /// Padding.
    pub pad4: u32,
    /// Padding.
    pub pad5: u32,
}

impl ElementwiseConstants {
    /// The eight `u32`s, in `cbuffer` order, for `SetComputeRoot32BitConstants`.
    #[must_use]
    pub fn to_root_constants(self) -> [u32; ROOT_CONSTANT_COUNT] {
        [
            self.n,
            self.groups_x,
            self.pad0,
            self.pad1,
            self.pad2,
            self.pad3,
            self.pad4,
            self.pad5,
        ]
    }

    /// The same payload padded to [`CBV_ALIGNMENT`], for a CBV-based variant.
    #[must_use]
    pub fn const_buffer_bytes(self) -> [u8; CBV_ALIGNMENT] {
        pad_to_cbv(&self.to_root_constants())
    }
}

// ─── op enums ────────────────────────────────────────────────────────────────

/// Binary elementwise ops with both an HLSL entry point and a DirectML operator.
///
/// The DirectML operator IDs are `DML_OPERATOR_ELEMENT_WISE_ADD`, `_SUBTRACT`,
/// `_MULTIPLY` and `_DIVIDE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `out = a + b`.
    Add,
    /// `out = a - b`.
    Sub,
    /// `out = a * b`.
    Mul,
    /// `out = a / b`.
    Div,
}

/// Unary elementwise ops with both an HLSL entry point and a DirectML operator.
///
/// The DirectML operator IDs are `DML_OPERATOR_ACTIVATION_RELU`, `_SIGMOID` and
/// `_TANH`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// `out = max(0, a)`.
    Relu,
    /// `out = 1 / (1 + exp(-a))`.
    Sigmoid,
    /// `out = tanh(a)`.
    Tanh,
}

impl BinaryOp {
    /// Stable tag for logs and error messages.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
        }
    }
}

impl UnaryOp {
    /// Stable tag for logs and error messages.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
        }
    }
}

// ─── MatMulPlan ──────────────────────────────────────────────────────────────

/// A fully validated, backend-agnostic MatMul / Gemm plan.
///
/// Every field is already range-checked to fit `u32`, so downstream code reads
/// them directly and never writes a bare `as u32`.
///
/// # Why `batch` is always 1 in this wave
///
/// The struct carries the full batched vocabulary (`batch`, `batch_shape`,
/// `a_batch_stride`, `b_batch_stride`) because the *layout* and *root-constant*
/// machinery around it is already batch-correct, and because a later wave will
/// lift the restriction without changing a single signature.  But
/// [`MatMulPlan::matmul`] declines everything that is not 2-D × 2-D today, so in
/// practice `batch == 1` and `batch_shape == []`.  See that function for why.
#[derive(Debug, Clone, PartialEq)]
pub struct MatMulPlan {
    /// Output rows — `a`'s second-to-last dim **after** any `trans_a`.
    pub m: u32,
    /// Shared inner dimension, after any transposes.
    pub k: u32,
    /// Output columns — `b`'s last dim after any `trans_b`.
    pub n: u32,
    /// Product of the broadcast batch dims; always `>= 1`.
    pub batch: u32,
    /// The broadcast batch shape (leading dims of the output); possibly empty.
    pub batch_shape: Vec<usize>,
    /// `batch_shape ++ [m, n]`.
    pub output_shape: Vec<usize>,
    /// Element stride to advance one batch in `a`'s buffer.  **`0` means `a` is
    /// batch-broadcast**, which the shader honours by leaving `a_offset` at 0 —
    /// that is the entire batch-broadcast implementation, with no CPU-side
    /// expansion at all.
    pub a_batch_stride: u32,
    /// As [`Self::a_batch_stride`], for `b`.
    pub b_batch_stride: u32,
    /// `a`'s shape exactly as stored in the caller's tensor (**pre**-transpose).
    /// DirectML's tensor descriptor needs the *stored* shape; `trans_a` tells
    /// DirectML to read it transposed.
    pub a_stored_shape: Vec<usize>,
    /// As [`Self::a_stored_shape`], for `b`.
    pub b_stored_shape: Vec<usize>,
    /// `Gemm`'s `C` operand's shape, when present **and** `beta != 0`.
    pub c_shape: Option<Vec<usize>>,
    /// `Gemm`'s `transA`.
    pub trans_a: bool,
    /// `Gemm`'s `transB`.
    pub trans_b: bool,
    /// `Gemm`'s `alpha`; `1.0` for `MatMul`.
    pub alpha: f32,
    /// `Gemm`'s `beta`; `0.0` for `MatMul` and for a `Gemm` with no `C` input.
    pub beta: f32,
}

impl MatMulPlan {
    /// Plan an ONNX `MatMul`.
    ///
    /// # This backend accepts 2-D × 2-D only. Full stop.
    ///
    /// ONNX `MatMul` is defined over N-D operands with numpy-broadcast batch dims,
    /// and it is *tempting* to accept them here — the shape math is easy and the
    /// output shape falls out.  Do not.  [`crate::hlsl::MATMUL_HLSL`] indexes
    /// `A[AOff + row * K + k]`, and until a backend actually walks the batch
    /// offsets and is *shown to be right on hardware*, accepting a `[8,128,64] ×
    /// [64,32]` node means returning a buffer of the wrong length filled with the
    /// wrong numbers — which is strictly worse than not accepting it, because the
    /// CPU kernel one line away computes it correctly.
    ///
    /// Everything higher-rank is [`DirectMLError::Declined`] → `Ok(None)` → CPU.
    ///
    /// To lift this restriction you must, in one commit: (a) relax the rank check
    /// here, (b) verify the shader honours `AOff`/`BOff`/`COff` per slice, and
    /// (c) run `DirectMLContext::self_check` on real hardware against a batched
    /// case and paste the report.  Not before.
    ///
    /// # Errors
    /// - [`DirectMLError::ShapeMismatch`] — the inner dims disagree.  The CPU
    ///   operator would fail on the same inputs.
    /// - [`DirectMLError::Declined`] — an operand is not exactly 2-D; any of
    ///   `m`/`k`/`n` is 0 (an empty tensor: `CreateCommittedResource` with
    ///   `Width = 0` fails, and a `[0, 128]` activation is routine after an empty
    ///   batch); or a dim or element count exceeds `u32::MAX`.
    pub fn matmul(a_shape: &[usize], b_shape: &[usize]) -> Result<Self> {
        Self::build(a_shape, b_shape, None, 1.0, 0.0, false, false, "MatMul")
    }

    /// Plan an ONNX `Gemm` (which is 2-D-only by definition).
    ///
    /// `c_shape` is `None` when the node has no third input, in which case `beta`
    /// is forced to `0.0`.  A supplied `C` with `beta == 0.0` is likewise dropped:
    /// ONNX says `beta * C` and `0 * C` is nothing, and carrying it would make the
    /// operator cache key spuriously distinct.
    ///
    /// # Errors
    /// As [`Self::matmul`], plus [`DirectMLError::ShapeMismatch`] when `C` does not
    /// broadcast to `[m, n]`.
    #[allow(clippy::too_many_arguments)] // Mirrors ONNX `Gemm`'s attribute set 1:1.
    pub fn gemm(
        a_shape: &[usize],
        b_shape: &[usize],
        c_shape: Option<&[usize]>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<Self> {
        Self::build(
            a_shape, b_shape, c_shape, alpha, beta, trans_a, trans_b, "Gemm",
        )
    }

    /// The shared validation core of [`Self::matmul`] and [`Self::gemm`].
    #[allow(clippy::too_many_arguments)]
    fn build(
        a_shape: &[usize],
        b_shape: &[usize],
        c_shape: Option<&[usize]>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
        op: &str,
    ) -> Result<Self> {
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(DirectMLError::Declined(format!(
                "{op}: this backend accepts 2-D x 2-D only, got {a_shape:?} x {b_shape:?}"
            )));
        }

        // Logical (post-transpose) dims.  The *stored* shapes stay untouched — the
        // DirectML descriptor describes the buffer as it sits in memory and lets
        // `TransA`/`TransB` reinterpret it.
        let (m, k_a) = if trans_a {
            (a_shape[1], a_shape[0])
        } else {
            (a_shape[0], a_shape[1])
        };
        let (k_b, n) = if trans_b {
            (b_shape[1], b_shape[0])
        } else {
            (b_shape[0], b_shape[1])
        };

        if k_a != k_b {
            return Err(DirectMLError::ShapeMismatch(format!(
                "{op}: inner dimension mismatch — a[-1]={k_a} vs b[-2]={k_b} \
                 (a={a_shape:?} transA={trans_a}, b={b_shape:?} transB={trans_b})"
            )));
        }
        if m == 0 || k_a == 0 || n == 0 {
            return Err(DirectMLError::Declined(format!(
                "{op}: empty operand — m={m} k={k_a} n={n}; a D3D12 buffer of \
                 Width = 0 cannot be created"
            )));
        }

        let m = checked_u32(m, "M")?;
        let k = checked_u32(k_a, "K")?;
        let n = checked_u32(n, "N")?;

        // Range-check every element count that will be turned into a buffer size or
        // a shader offset.  These are the only `u32` conversions of shape-derived
        // values anywhere in the MatMul path.
        let a_elems = checked_u32(numel(a_shape)?, "A element count")?;
        let b_elems = checked_u32(numel(b_shape)?, "B element count")?;
        let _out_elems = checked_u32(
            (m as usize)
                .checked_mul(n as usize)
                .ok_or_else(|| DirectMLError::Declined("M * N overflows usize".into()))?,
            "output element count",
        )?;

        let c_shape = match c_shape {
            // `beta == 0` means `C` contributes nothing; drop it rather than carry a
            // shape that would fragment the operator cache for no reason.
            Some(_) if beta == 0.0 => None,
            Some(cs) => {
                // Must broadcast to `[m, n]` — this is the *bias* broadcast, which is
                // core ONNX `Gemm` semantics and is handled entirely on the CPU by
                // `apply_gemm_epilogue` (HLSL path) or by `DML_GEMM_OPERATOR_DESC`
                // (DirectML path).  It is not the elementwise broadcast that
                // `ElementwisePlan::binary` declines.
                let bc = broadcast_shape(cs, &[m as usize, n as usize])?;
                if bc != vec![m as usize, n as usize] {
                    return Err(DirectMLError::ShapeMismatch(format!(
                        "{op}: C shape {cs:?} does not broadcast to [{m}, {n}]"
                    )));
                }
                if numel(cs)? == 0 {
                    return Err(DirectMLError::Declined(format!("{op}: empty C operand")));
                }
                Some(cs.to_vec())
            }
            None => None,
        };
        let beta = if c_shape.is_some() { beta } else { 0.0 };

        Ok(Self {
            m,
            k,
            n,
            batch: 1,
            batch_shape: Vec::new(),
            output_shape: vec![m as usize, n as usize],
            // With `batch == 1` these are never multiplied by a non-zero slice
            // index, so their value is unobservable today; they carry the honest
            // packed strides so that lifting the 2-D restriction is a pure
            // relaxation of the check above rather than a hunt for the right number.
            a_batch_stride: a_elems,
            b_batch_stride: b_elems,
            a_stored_shape: a_shape.to_vec(),
            b_stored_shape: b_shape.to_vec(),
            c_shape,
            trans_a,
            trans_b,
            alpha,
            beta,
        })
    }

    /// `batch * m * n`.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the product overflows `usize` (impossible
    /// on a 64-bit host given the `u32` field bounds, but checked rather than
    /// assumed).
    pub fn output_elems(&self) -> Result<usize> {
        (self.batch as usize)
            .checked_mul(self.m as usize)
            .and_then(|v| v.checked_mul(self.n as usize))
            .ok_or_else(|| DirectMLError::Declined("batch * M * N overflows usize".into()))
    }

    /// Exact (unaligned) byte size of the `A` buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn a_bytes(&self) -> Result<usize> {
        numel(&self.a_stored_shape)?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("A byte size overflows usize".into()))
    }

    /// Exact (unaligned) byte size of the `B` buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn b_bytes(&self) -> Result<usize> {
        numel(&self.b_stored_shape)?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("B byte size overflows usize".into()))
    }

    /// Exact (unaligned) byte size of the output buffer.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn output_bytes(&self) -> Result<usize> {
        self.output_elems()?
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("output byte size overflows usize".into()))
    }

    /// `true` when `beta != 0.0` **and** a `C` operand was supplied.
    #[must_use]
    pub fn has_bias(&self) -> bool {
        self.beta != 0.0 && self.c_shape.is_some()
    }

    /// `true` when the HLSL backend must CPU-transpose an operand first.  The
    /// DirectML backend never needs this.
    #[must_use]
    pub fn needs_cpu_transpose(&self) -> bool {
        self.trans_a || self.trans_b
    }

    /// Thread-group grid for **one batch slice** of [`crate::hlsl::MATMUL_HLSL`].
    ///
    /// # The transposition trap
    ///
    /// The shader is `[numthreads(16, 16, 1)]` and reads
    ///
    /// ```text
    /// uint row = tid.y;
    /// uint col = tid.x;
    /// ```
    ///
    /// so **X covers columns and Y covers rows**:
    ///
    /// ```text
    /// x = ceil(N / 16)      ← NOT ceil(M / 16)
    /// y = ceil(M / 16)      ← NOT ceil(N / 16)
    /// z = 1
    /// ```
    ///
    /// The scaffold this crate grew out of documented the opposite
    /// (`Dispatch(ceil(M/16), ceil(N/16), 1)`), and following that comment on any
    /// non-square matrix leaves part of the output matrix as whatever the buffer
    /// happened to contain — a wrong answer, not a crash.  `hlsl_grid_is_not_transposed`
    /// in this module's tests pins the correct orientation.
    ///
    /// The shader is not batched: the HLSL backend records `batch` dispatches with
    /// this same grid, varying only the root constants.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `m` or `n` exceeds `65535 * 16 = 1_048_560`.
    pub fn hlsl_grid(&self) -> Result<DispatchGrid> {
        DispatchGrid::tiled_2d(self.m, self.n, MATMUL_TILE)
    }

    /// Root constants for batch slice `slice` (`0 <= slice < batch`).
    ///
    /// `a_offset = slice * a_batch_stride`, `b_offset = slice * b_batch_stride`,
    /// `c_offset = slice * m * n`.  A zero batch-stride keeps the offset at 0 —
    /// that *is* the batch-broadcast implementation.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when `slice >= batch`, or on `u32` overflow of
    /// any offset.
    pub fn constants_for_slice(&self, slice: u32) -> Result<MatMulConstants> {
        if slice >= self.batch {
            return Err(DirectMLError::Declined(format!(
                "batch slice {slice} out of range for batch {}",
                self.batch
            )));
        }
        let offset = |stride: u32, what: &str| -> Result<u32> {
            slice.checked_mul(stride).ok_or_else(|| {
                DirectMLError::Declined(format!(
                    "{what} offset overflows u32 at slice {slice} (stride {stride})"
                ))
            })
        };
        let slice_elems = self
            .m
            .checked_mul(self.n)
            .ok_or_else(|| DirectMLError::Declined("M * N overflows u32".into()))?;
        Ok(MatMulConstants {
            m: self.m,
            k: self.k,
            n: self.n,
            a_offset: offset(self.a_batch_stride, "A")?,
            b_offset: offset(self.b_batch_stride, "B")?,
            c_offset: offset(slice_elems, "C")?,
            pad0: 0,
            pad1: 0,
        })
    }
}

// ─── ElementwisePlan ─────────────────────────────────────────────────────────

/// A validated elementwise plan (binary or unary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementwisePlan {
    /// The output shape.  In this wave it always equals [`Self::a_shape`].
    pub output_shape: Vec<usize>,
    /// `output_shape.iter().product()` — guaranteed `> 0` and `<= u32::MAX`.
    pub elem_count: u32,
    /// Operand `A`'s shape as supplied.
    pub a_shape: Vec<usize>,
    /// Operand `B`'s shape as supplied; `None` for unary ops.
    pub b_shape: Option<Vec<usize>>,
    /// `true` when `A` must be materially expanded to reach `output_shape`.
    /// **Always `false` in this wave** — see [`Self::binary`].
    pub a_needs_broadcast: bool,
    /// As [`Self::a_needs_broadcast`], for `B`.
    pub b_needs_broadcast: bool,
}

impl ElementwisePlan {
    /// Plan a binary elementwise op.
    ///
    /// # This backend accepts identical shapes only
    ///
    /// If `a_shape != b_shape` — even when the two are perfectly numpy-broadcastable,
    /// e.g. `[2, 3, 4]` and `[1, 4]` — this returns [`DirectMLError::Declined`] and
    /// the op runs on the CPU, correctly.
    ///
    /// ## Do not "improve" this by dispatching over `max(a.numel(), b.numel())`
    ///
    /// That is the obvious-looking fix and it is wrong in a way that produces
    /// *plausible numbers*, which is the worst kind of wrong.  The HLSL kernels are
    /// index-parallel — `C[i] = A[i] + B[i]` — with no notion of a shape at all.
    /// Dispatching `max(numel)` threads over a `[2, 3, 4]` + `[1, 4]` pair reads
    /// `B[0..24]` out of a 4-element buffer: past the end of the operand, into
    /// whatever the allocator left there, and the result is a 24-element tensor of
    /// the right *shape* full of the wrong *values*.  No bounds check fires; no
    /// test that only checks shapes catches it.
    ///
    /// The two correct ways to lift this restriction are (a) CPU-expand the
    /// operands with [`broadcast_expand`] before upload, or (b) express the
    /// broadcast as 0-strides in the DirectML tensor descriptor
    /// ([`crate::layout::DmlTensorLayout::broadcast_to`], which copies nothing).
    /// Both are already implemented and tested; neither is *wired in*, because
    /// neither has been run on hardware.  Wire one in, verify it with
    /// `DirectMLContext::self_check`, and relax the check below — in that order.
    ///
    /// # Errors
    /// - [`DirectMLError::ShapeMismatch`] — the shapes are not even broadcastable;
    ///   the CPU operator would fail too.
    /// - [`DirectMLError::Declined`] — the shapes are broadcastable but not
    ///   identical; the rank exceeds [`crate::layout::DML_RANK`]; the element count
    ///   is 0 or exceeds `u32::MAX`.
    pub fn binary(a_shape: &[usize], b_shape: &[usize]) -> Result<Self> {
        // Broadcastability first, so a genuinely malformed model gets a
        // `ShapeMismatch` (a real error) rather than a `Declined` (a silent, and in
        // that case misleading, CPU fallback).
        let _ = broadcast_shape(a_shape, b_shape)?;

        if a_shape != b_shape {
            return Err(DirectMLError::Declined(format!(
                "elementwise: this backend requires identical shapes, got {a_shape:?} and \
                 {b_shape:?} (broadcastable, but not implemented — read ElementwisePlan::binary)"
            )));
        }

        let (output_shape, elem_count) = Self::validate(a_shape)?;
        Ok(Self {
            output_shape,
            elem_count,
            a_shape: a_shape.to_vec(),
            b_shape: Some(b_shape.to_vec()),
            a_needs_broadcast: false,
            b_needs_broadcast: false,
        })
    }

    /// Plan a unary elementwise op.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the rank exceeds [`crate::layout::DML_RANK`],
    /// or the element count is 0 or exceeds `u32::MAX`.
    pub fn unary(a_shape: &[usize]) -> Result<Self> {
        let (output_shape, elem_count) = Self::validate(a_shape)?;
        Ok(Self {
            output_shape,
            elem_count,
            a_shape: a_shape.to_vec(),
            b_shape: None,
            a_needs_broadcast: false,
            b_needs_broadcast: false,
        })
    }

    /// The rank / emptiness / `u32` checks shared by both constructors.
    fn validate(shape: &[usize]) -> Result<(Vec<usize>, u32)> {
        if shape.len() > crate::layout::DML_RANK {
            return Err(DirectMLError::Declined(format!(
                "elementwise: rank {} exceeds DML_RANK = {}",
                shape.len(),
                crate::layout::DML_RANK
            )));
        }
        let elems = numel(shape)?;
        if elems == 0 {
            // A `[0, 128]` tensor is routine after an empty batch, and
            // `CreateCommittedResource` with `Width = 0` fails outright.  Decline it
            // here rather than discover it three FFI calls deep.
            return Err(DirectMLError::Declined(format!(
                "elementwise: empty tensor {shape:?}; a D3D12 buffer of Width = 0 \
                 cannot be created"
            )));
        }
        let elem_count = checked_u32(elems, "element count")?;
        Ok((shape.to_vec(), elem_count))
    }

    /// `elem_count * 4` — the byte size of each operand and of the output.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] on overflow.
    pub fn buffer_bytes(&self) -> Result<usize> {
        (self.elem_count as usize)
            .checked_mul(ELEM_SIZE)
            .ok_or_else(|| DirectMLError::Declined("buffer size overflows usize".into()))
    }

    /// Grid for the `[numthreads(256, 1, 1)]` shaders:
    /// `DispatchGrid::linear(ceil_div(elem_count, 256))`.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the group count exceeds `65535 * 65535`.
    pub fn hlsl_grid(&self) -> Result<DispatchGrid> {
        let groups = ceil_div(self.elem_count, ELEMENTWISE_THREADS_PER_GROUP)
            .ok_or_else(|| DirectMLError::Declined("ELEMENTWISE_THREADS_PER_GROUP is 0".into()))?;
        DispatchGrid::linear(groups)
    }

    /// Root constants.  `groups_x` is taken from [`Self::hlsl_grid`]'s `x`, so the
    /// shader's `gid.y * GroupsX + gid.x` can never disagree with the grid actually
    /// dispatched.
    ///
    /// # Errors
    /// As [`Self::hlsl_grid`].
    pub fn constants(&self) -> Result<ElementwiseConstants> {
        let grid = self.hlsl_grid()?;
        Ok(ElementwiseConstants {
            n: self.elem_count,
            groups_x: grid.x,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
            pad4: 0,
            pad5: 0,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn declined(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::Declined(_))
    }
    fn mismatch(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::ShapeMismatch(_))
    }

    // ── scalar helpers ───────────────────────────────────────────────────────

    #[test]
    fn align_up_covers_the_boundaries() {
        assert_eq!(align_up(0, 256), Some(0));
        assert_eq!(align_up(1, 256), Some(256));
        assert_eq!(align_up(256, 256), Some(256));
        assert_eq!(align_up(257, 256), Some(512));
        assert_eq!(align_up(4, 4), Some(4));
        assert_eq!(align_up(5, 4), Some(8));
        assert_eq!(align_up(10, 0), None, "align 0 must not divide by zero");
        assert_eq!(align_up(usize::MAX, 256), None, "must not wrap");
    }

    #[test]
    fn ceil_div_covers_the_boundaries() {
        assert_eq!(ceil_div(0, 256), Some(0));
        assert_eq!(ceil_div(1, 256), Some(1));
        assert_eq!(ceil_div(256, 256), Some(1));
        assert_eq!(ceil_div(257, 256), Some(2));
        assert_eq!(ceil_div(u32::MAX, 1), Some(u32::MAX));
        assert_eq!(ceil_div(u32::MAX, 2), Some(2_147_483_647 + 1));
        assert_eq!(ceil_div(5, 0), None);
    }

    // ── dispatch grid ────────────────────────────────────────────────────────

    #[test]
    fn linear_grid_never_under_covers() {
        for groups in [
            1u32,
            2,
            255,
            256,
            65_534,
            65_535,
            65_536,
            100_000,
            4_294_836_225,
        ] {
            let g = DispatchGrid::linear(groups).expect("within limits");
            assert!(
                g.total_groups() >= u64::from(groups),
                "grid {g:?} under-covers {groups} groups"
            );
            assert!(g.x <= DispatchGrid::MAX_GROUPS_PER_DIM);
            assert!(g.y <= DispatchGrid::MAX_GROUPS_PER_DIM);
            assert_eq!(g.z, 1);
        }
    }

    #[test]
    fn linear_grid_boundaries() {
        assert_eq!(
            DispatchGrid::linear(1).unwrap(),
            DispatchGrid { x: 1, y: 1, z: 1 }
        );
        assert_eq!(
            DispatchGrid::linear(65_535).unwrap(),
            DispatchGrid {
                x: 65_535,
                y: 1,
                z: 1
            }
        );
        assert_eq!(
            DispatchGrid::linear(65_536).unwrap(),
            DispatchGrid {
                x: 65_535,
                y: 2,
                z: 1
            }
        );
        // Exactly 65_535 x 65_535 is the last representable grid.
        let max = 65_535u32 * 65_535;
        assert_eq!(
            DispatchGrid::linear(max).unwrap(),
            DispatchGrid {
                x: 65_535,
                y: 65_535,
                z: 1
            }
        );
        // One more group needs a third dimension we do not use.
        let e = DispatchGrid::linear(max + 1).unwrap_err();
        assert!(declined(&e), "got {e:?}");

        let e = DispatchGrid::linear(0).unwrap_err();
        assert!(declined(&e), "got {e:?}");
    }

    #[test]
    fn tiled_2d_puts_columns_on_x() {
        // 32 rows, 48 cols, 16x16 tiles → 3 groups across (cols), 2 down (rows).
        assert_eq!(
            DispatchGrid::tiled_2d(32, 48, 16).unwrap(),
            DispatchGrid { x: 3, y: 2, z: 1 }
        );
        assert_eq!(
            DispatchGrid::tiled_2d(1, 1, 16).unwrap(),
            DispatchGrid { x: 1, y: 1, z: 1 }
        );
        assert!(declined(&DispatchGrid::tiled_2d(0, 4, 16).unwrap_err()));
        assert!(declined(&DispatchGrid::tiled_2d(4, 0, 16).unwrap_err()));
        assert!(declined(&DispatchGrid::tiled_2d(4, 4, 0).unwrap_err()));
        // 65_535 * 16 = 1_048_560 tiles' worth of rows is the last legal grid.
        assert!(DispatchGrid::tiled_2d(1_048_560, 16, 16).is_ok());
        assert!(declined(
            &DispatchGrid::tiled_2d(1_048_561, 16, 16).unwrap_err()
        ));
    }

    // ── root constants ───────────────────────────────────────────────────────

    #[test]
    fn matmul_constants_serialise_in_cbuffer_order() {
        let c = MatMulConstants {
            m: 4,
            k: 3,
            n: 5,
            a_offset: 12,
            b_offset: 15,
            c_offset: 20,
            pad0: 0,
            pad1: 0,
        };
        assert_eq!(c.to_root_constants(), [4, 3, 5, 12, 15, 20, 0, 0]);

        let bytes = c.const_buffer_bytes();
        assert_eq!(bytes.len(), CBV_ALIGNMENT);
        assert_eq!(&bytes[0..4], &4u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &5u32.to_le_bytes());
        assert!(bytes[32..].iter().all(|&b| b == 0), "tail must be zeroed");
    }

    #[test]
    fn elementwise_constants_serialise_in_cbuffer_order() {
        let c = ElementwiseConstants {
            n: 1000,
            groups_x: 4,
            ..ElementwiseConstants::default()
        };
        assert_eq!(c.to_root_constants(), [1000, 4, 0, 0, 0, 0, 0, 0]);
        assert_eq!(c.const_buffer_bytes().len(), CBV_ALIGNMENT);
        assert_eq!(&c.const_buffer_bytes()[0..4], &1000u32.to_le_bytes());
    }

    #[test]
    fn root_constant_structs_are_exactly_eight_u32_wide() {
        assert_eq!(
            core::mem::size_of::<MatMulConstants>(),
            ROOT_CONSTANT_COUNT * 4
        );
        assert_eq!(
            core::mem::size_of::<ElementwiseConstants>(),
            ROOT_CONSTANT_COUNT * 4
        );
    }

    // ── MatMulPlan ───────────────────────────────────────────────────────────

    #[test]
    fn matmul_2d_is_accepted() {
        let p = MatMulPlan::matmul(&[2, 3], &[3, 4]).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert_eq!(p.batch, 1);
        assert!(p.batch_shape.is_empty());
        assert_eq!(p.output_shape, vec![2, 4]);
        assert_eq!(p.output_elems().unwrap(), 8);
        assert_eq!(p.a_bytes().unwrap(), 6 * 4);
        assert_eq!(p.b_bytes().unwrap(), 12 * 4);
        assert_eq!(p.output_bytes().unwrap(), 8 * 4);
        assert!(!p.has_bias());
        assert!(!p.needs_cpu_transpose());
        assert_eq!(p.alpha, 1.0);
        assert_eq!(p.beta, 0.0);
    }

    /// The regression that motivates the whole 2-D restriction: today's scaffold
    /// validates only `ndim >= 2`, so `[8,128,64] x [64,32]` sails through
    /// validation and the un-batched shader returns the wrong shape AND the wrong
    /// data.  It must be declined, not "supported".
    #[test]
    fn matmul_batched_is_declined_not_silently_wrong() {
        let e = MatMulPlan::matmul(&[8, 128, 64], &[64, 32]).unwrap_err();
        assert!(declined(&e), "got {e:?}");
        assert!(format!("{e}").contains("2-D"), "got {e}");

        assert!(declined(
            &MatMulPlan::matmul(&[5, 2, 3], &[5, 3, 4]).unwrap_err()
        ));
        assert!(declined(
            &MatMulPlan::matmul(&[1, 2, 3], &[5, 3, 4]).unwrap_err()
        ));
    }

    #[test]
    fn matmul_1d_is_declined() {
        assert!(declined(&MatMulPlan::matmul(&[3], &[3, 4]).unwrap_err()));
        assert!(declined(&MatMulPlan::matmul(&[2, 3], &[3]).unwrap_err()));
        assert!(declined(&MatMulPlan::matmul(&[], &[]).unwrap_err()));
    }

    #[test]
    fn matmul_inner_dim_mismatch_is_a_shape_error_not_a_decline() {
        // The CPU operator would fail on these inputs too, so this must NOT be
        // silently swallowed as a CPU fallback.
        let e = MatMulPlan::matmul(&[2, 3], &[7, 4]).unwrap_err();
        assert!(mismatch(&e), "got {e:?}");
    }

    #[test]
    fn matmul_empty_operand_is_declined() {
        // `[0, 128]` is what an empty batch produces, and `CreateCommittedResource`
        // with `Width = 0` fails.
        assert!(declined(
            &MatMulPlan::matmul(&[0, 128], &[128, 4]).unwrap_err()
        ));
        assert!(declined(&MatMulPlan::matmul(&[4, 0], &[0, 4]).unwrap_err()));
        assert!(declined(
            &MatMulPlan::matmul(&[4, 128], &[128, 0]).unwrap_err()
        ));
    }

    /// The single most important test in this file.
    ///
    /// `MATMUL_HLSL` does `row = tid.y; col = tid.x`, so X must count **columns**.
    /// The scaffold's doc comment said `Dispatch(ceil(M/16), ceil(N/16), 1)` — the
    /// exact transposition — which on any non-square matrix leaves part of the
    /// output untouched.
    #[test]
    fn hlsl_grid_is_not_transposed() {
        // M = 32 rows, N = 48 cols.
        let p = MatMulPlan::matmul(&[32, 64], &[64, 48]).unwrap();
        let g = p.hlsl_grid().unwrap();
        assert_eq!(
            g.x, 3,
            "x must be ceil(N/16) = ceil(48/16) = 3, NOT ceil(M/16)"
        );
        assert_eq!(
            g.y, 2,
            "y must be ceil(M/16) = ceil(32/16) = 2, NOT ceil(N/16)"
        );
        assert_eq!(g.z, 1);

        // A grid that covers every output element: x*16 >= N and y*16 >= M.
        assert!(u64::from(g.x) * 16 >= 48);
        assert!(u64::from(g.y) * 16 >= 32);

        // Strongly asymmetric: 1 row, 4096 cols.  A transposed grid would dispatch
        // x = 1 group and cover only the first 16 of 4096 columns.
        let p = MatMulPlan::matmul(&[1, 8], &[8, 4096]).unwrap();
        let g = p.hlsl_grid().unwrap();
        assert_eq!(g.x, 256);
        assert_eq!(g.y, 1);
    }

    #[test]
    fn hlsl_grid_declines_past_the_d3d12_limit() {
        // 65_535 * 16 = 1_048_560 is the largest coverable dimension.
        let p = MatMulPlan::matmul(&[1_048_560, 2], &[2, 16]).unwrap();
        assert!(p.hlsl_grid().is_ok());
        let p = MatMulPlan::matmul(&[1_048_561, 2], &[2, 16]).unwrap();
        assert!(declined(&p.hlsl_grid().unwrap_err()));
    }

    #[test]
    fn constants_for_slice_bounds_and_offsets() {
        let p = MatMulPlan::matmul(&[4, 3], &[3, 5]).unwrap();
        let c = p.constants_for_slice(0).unwrap();
        assert_eq!(c.m, 4);
        assert_eq!(c.k, 3);
        assert_eq!(c.n, 5);
        assert_eq!(c.a_offset, 0);
        assert_eq!(c.b_offset, 0);
        assert_eq!(c.c_offset, 0);
        // batch == 1, so slice 1 does not exist.
        assert!(declined(&p.constants_for_slice(1).unwrap_err()));
    }

    #[test]
    fn gemm_transposes_are_folded_into_the_logical_dims() {
        // A stored [3, 2] read transposed is logically [2, 3]; B is [3, 4].
        let p = MatMulPlan::gemm(&[3, 2], &[3, 4], None, 1.0, 0.0, true, false).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert_eq!(
            p.a_stored_shape,
            vec![3, 2],
            "stored shape must NOT be rewritten"
        );
        assert!(p.needs_cpu_transpose());

        // B stored [4, 3] read transposed is logically [3, 4].
        let p = MatMulPlan::gemm(&[2, 3], &[4, 3], None, 1.0, 0.0, false, true).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert_eq!(p.b_stored_shape, vec![4, 3]);
    }

    #[test]
    fn gemm_bias_and_beta() {
        // Row-vector bias broadcasts against [m, n].
        let p = MatMulPlan::gemm(&[2, 3], &[3, 4], Some(&[4]), 2.0, 3.0, false, false).unwrap();
        assert!(p.has_bias());
        assert_eq!(p.c_shape, Some(vec![4]));
        assert_eq!(p.alpha, 2.0);
        assert_eq!(p.beta, 3.0);

        // beta == 0 drops C entirely.
        let p = MatMulPlan::gemm(&[2, 3], &[3, 4], Some(&[4]), 1.0, 0.0, false, false).unwrap();
        assert!(!p.has_bias());
        assert_eq!(p.c_shape, None);

        // No C at all forces beta to 0.
        let p = MatMulPlan::gemm(&[2, 3], &[3, 4], None, 1.0, 7.0, false, false).unwrap();
        assert!(!p.has_bias());
        assert_eq!(p.beta, 0.0);

        // A C that does not broadcast to [m, n] is a genuine shape error.
        let e = MatMulPlan::gemm(&[2, 3], &[3, 4], Some(&[5]), 1.0, 1.0, false, false).unwrap_err();
        assert!(mismatch(&e), "got {e:?}");
    }

    #[test]
    fn gemm_epilogue_applies_alpha_and_broadcast_beta_c() {
        let p = MatMulPlan::gemm(&[2, 3], &[3, 2], Some(&[2]), 2.0, 3.0, false, false).unwrap();
        // Bare product of a 2x2 output.
        let mut out = vec![1.0f32, 2.0, 3.0, 4.0];
        let c = [10.0f32, 20.0];
        apply_gemm_epilogue(&p, &mut out, Some(&c)).unwrap();
        // alpha*x + beta*c, with c broadcast along rows.
        assert_eq!(out, vec![2.0 + 30.0, 4.0 + 60.0, 6.0 + 30.0, 8.0 + 60.0]);
    }

    #[test]
    fn gemm_epilogue_rejects_a_wrongly_sized_output() {
        let p = MatMulPlan::matmul(&[2, 3], &[3, 2]).unwrap();
        let mut out = vec![0.0f32; 3];
        assert!(mismatch(
            &apply_gemm_epilogue(&p, &mut out, None).unwrap_err()
        ));
    }

    // ── ElementwisePlan ──────────────────────────────────────────────────────

    #[test]
    fn elementwise_identical_shapes_are_accepted() {
        let p = ElementwisePlan::binary(&[2, 3, 4], &[2, 3, 4]).unwrap();
        assert_eq!(p.elem_count, 24);
        assert_eq!(p.output_shape, vec![2, 3, 4]);
        assert_eq!(p.b_shape, Some(vec![2, 3, 4]));
        assert!(!p.a_needs_broadcast);
        assert!(!p.b_needs_broadcast);
        assert_eq!(p.buffer_bytes().unwrap(), 24 * 4);

        // A scalar (rank 0) is one element.
        let p = ElementwisePlan::unary(&[]).unwrap();
        assert_eq!(p.elem_count, 1);
        assert_eq!(p.b_shape, None);
    }

    #[test]
    fn elementwise_broadcastable_but_unequal_is_declined() {
        // Perfectly broadcastable — and perfectly refused.  See ElementwisePlan::binary.
        let e = ElementwisePlan::binary(&[2, 3, 4], &[1, 4]).unwrap_err();
        assert!(declined(&e), "got {e:?}");
        let e = ElementwisePlan::binary(&[2, 3], &[3]).unwrap_err();
        assert!(declined(&e), "got {e:?}");
        let e = ElementwisePlan::binary(&[2, 3], &[]).unwrap_err();
        assert!(declined(&e), "got {e:?}");
    }

    #[test]
    fn elementwise_incompatible_shapes_are_a_shape_error() {
        let e = ElementwisePlan::binary(&[2, 3], &[4, 5]).unwrap_err();
        assert!(mismatch(&e), "got {e:?}");
    }

    #[test]
    fn elementwise_empty_and_over_rank_are_declined() {
        assert!(declined(
            &ElementwisePlan::binary(&[0, 3], &[0, 3]).unwrap_err()
        ));
        assert!(declined(&ElementwisePlan::unary(&[0, 3]).unwrap_err()));
        // Rank 5 exceeds DML_RANK.
        assert!(declined(
            &ElementwisePlan::unary(&[2, 2, 2, 2, 2]).unwrap_err()
        ));
    }

    #[test]
    fn elementwise_grid_and_constants_agree_on_groups_x() {
        for elems in [1usize, 255, 256, 257, 100_000, 16_777_216, 20_000_000] {
            let p = ElementwisePlan::unary(&[elems]).unwrap();
            let grid = p.hlsl_grid().unwrap();
            let consts = p.constants().unwrap();
            assert_eq!(
                consts.groups_x, grid.x,
                "ElementwiseConstants::groups_x must equal the dispatched grid's x, \
                 or the shader computes the wrong elements"
            );
            assert_eq!(consts.n, p.elem_count);
            // The grid must cover every element.
            assert!(
                grid.total_groups() * u64::from(ELEMENTWISE_THREADS_PER_GROUP)
                    >= u64::from(p.elem_count),
                "grid {grid:?} under-covers {elems} elements"
            );
        }
    }

    #[test]
    fn elementwise_grid_goes_two_dimensional_past_the_cliff() {
        // 65_535 * 256 = 16_776_960 elements is the last 1-D grid.
        let p = ElementwisePlan::unary(&[16_776_960]).unwrap();
        assert_eq!(p.hlsl_grid().unwrap().y, 1);
        let p = ElementwisePlan::unary(&[16_776_961]).unwrap();
        let g = p.hlsl_grid().unwrap();
        assert_eq!(g.y, 2, "must fold into a second dimension, not overflow");
        assert_eq!(g.x, 65_535);
    }

    // ── broadcast / transpose helpers ────────────────────────────────────────

    #[test]
    fn broadcast_shape_rules() {
        assert_eq!(broadcast_shape(&[2, 3, 4], &[1, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast_shape(&[1, 4], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[5, 1], &[1, 6]).unwrap(), vec![5, 6]);
        assert!(mismatch(&broadcast_shape(&[2, 3], &[4, 5]).unwrap_err()));
    }

    #[test]
    fn broadcast_expand_borrows_when_there_is_nothing_to_do() {
        let src = [1.0f32, 2.0, 3.0];
        let out = broadcast_expand(&src, &[3], &[3]).unwrap();
        assert!(matches!(out, Cow::Borrowed(_)), "no-op must not allocate");
        assert_eq!(&*out, &src);
    }

    #[test]
    fn broadcast_expand_materialises() {
        // [2, 1] -> [2, 3]
        let src = [1.0f32, 2.0];
        let out = broadcast_expand(&src, &[2, 1], &[2, 3]).unwrap();
        assert_eq!(&*out, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // [3] -> [2, 3] (left-padded)
        let src = [1.0f32, 2.0, 3.0];
        let out = broadcast_expand(&src, &[3], &[2, 3]).unwrap();
        assert_eq!(&*out, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Scalar -> anything.
        let src = [7.0f32];
        let out = broadcast_expand(&src, &[], &[2, 2]).unwrap();
        assert_eq!(&*out, &[7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn broadcast_expand_rejects_a_mis_sized_buffer() {
        let src = [1.0f32, 2.0];
        assert!(mismatch(&broadcast_expand(&src, &[3], &[3]).unwrap_err()));
    }

    #[test]
    fn transpose_2d_round_trips() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let t = transpose_2d(&src, 2, 3).unwrap();
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // 3x2
        let back = transpose_2d(&t, 3, 2).unwrap();
        assert_eq!(back, src.to_vec());

        assert!(mismatch(&transpose_2d(&src, 3, 3).unwrap_err()));
    }

    // ── properties ───────────────────────────────────────────────────────────

    proptest::proptest! {
        /// Any accepted 2-D MatMul has output shape exactly `[m, n]`, and a grid
        /// that covers every output element.
        #[test]
        fn prop_matmul_output_shape_and_grid_cover(
            m in 1usize..64, k in 1usize..64, n in 1usize..64,
        ) {
            let p = MatMulPlan::matmul(&[m, k], &[k, n]).expect("valid 2-D matmul");
            proptest::prop_assert_eq!(p.output_shape.clone(), vec![m, n]);
            proptest::prop_assert_eq!(p.output_elems().expect("no overflow"), m * n);

            let g = p.hlsl_grid().expect("small grid");
            proptest::prop_assert!(u64::from(g.x) * u64::from(MATMUL_TILE) >= n as u64);
            proptest::prop_assert!(u64::from(g.y) * u64::from(MATMUL_TILE) >= m as u64);
        }

        /// `broadcast_expand` always produces exactly `product(dst_shape)` elements,
        /// and every element it produces came from the source buffer.
        #[test]
        fn prop_broadcast_expand_length(
            rows in 1usize..8, cols in 1usize..8,
        ) {
            let src: Vec<f32> = (0..rows).map(|i| i as f32).collect();
            let out = broadcast_expand(&src, &[rows, 1], &[rows, cols])
                .expect("a size-1 axis always broadcasts");
            proptest::prop_assert_eq!(out.len(), rows * cols);
            for (i, v) in out.iter().enumerate() {
                proptest::prop_assert_eq!(*v, src[i / cols]);
            }
        }

        /// A linear grid never under-covers, and never exceeds the per-dim limit.
        #[test]
        fn prop_linear_grid_covers(groups in 1u32..4_000_000) {
            let g = DispatchGrid::linear(groups).expect("within limits");
            proptest::prop_assert!(g.total_groups() >= u64::from(groups));
            proptest::prop_assert!(g.x <= DispatchGrid::MAX_GROUPS_PER_DIM);
            proptest::prop_assert!(g.y <= DispatchGrid::MAX_GROUPS_PER_DIM);
        }
    }
}
