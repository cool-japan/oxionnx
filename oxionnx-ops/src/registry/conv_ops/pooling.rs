//! Pooling operator implementations: MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool.
//!
//! `MaxPool` and `AveragePool` are rank-generic — 1D (audio), 2D (vision) and
//! 3D (video / volumetric) all resolve `kernel_shape`, `strides`, `pads`,
//! `auto_pad`, `dilations` and `ceil_mode` through [`crate::conv::spatial`]
//! and run the single shared kernel in [`crate::conv`]; there is no
//! rank-2-only code path left.

use oxionnx_core::{Attributes, OnnxError, OpContext, Operator, Tensor};

use crate::conv;
use crate::conv::spatial::{
    parse_auto_pad, read_kernel_shape, read_pads, read_positive_spatial, resolve_pads, spatial_rank,
};

// ── Shared pooling geometry ─────────────────────────────────────────────────

/// Resolve the full N-D pooling geometry of a `MaxPool` / `AveragePool` node.
///
/// Reads and validates every spatial attribute the ONNX spec defines for the
/// two pooling operators — `kernel_shape`, `strides`, `pads`, `auto_pad`,
/// `dilations` and `ceil_mode` — and computes the resulting output extent with
/// the same formula the engine's shape-inference pass uses, so the planner and
/// the kernel can never disagree.
fn pool_geometry(
    attrs: &Attributes,
    input_shape: &[usize],
    op: &str,
) -> Result<conv::PoolGeometry, OnnxError> {
    let rank = spatial_rank(input_shape, op, "input")?;
    // `kernel_shape` is required for both pooling operators and must cover
    // every spatial axis of the input.
    let kernel = read_kernel_shape(attrs.ints("kernel_shape"), rank, op)?;
    let strides = read_positive_spatial(attrs.ints("strides"), rank, 1, "strides", op)?;
    let dilations = read_positive_spatial(attrs.ints("dilations"), rank, 1, "dilations", op)?;
    let explicit = read_pads(attrs.ints("pads"), rank, op)?;
    let auto_pad = parse_auto_pad(attrs.s("auto_pad"), op)?;
    let ceil_mode = attrs.i("ceil_mode", 0) != 0;

    let input_spatial = &input_shape[2..];
    let pads = resolve_pads(
        auto_pad,
        input_spatial,
        &kernel,
        &strides,
        &dilations,
        &explicit,
    );
    conv::PoolGeometry::resolve(
        op,
        input_spatial,
        kernel,
        strides,
        pads,
        dilations,
        ceil_mode,
    )
}

/// Read and validate the MaxPool `storage_order` attribute.
///
/// Returns `true` for column-major (`storage_order == 1`), which only affects
/// how the optional `Indices` output encodes an n-tuple index into one
/// integer. ONNX defines that encoding for the 2D case (and it is trivially
/// the same as row-major for 1D), so a column-major request at spatial rank
/// ≥ 3 is a typed [`OnnxError::Unsupported`] rather than an invented
/// generalisation no reference implementation would agree with.
fn read_storage_order(attrs: &Attributes, rank: usize) -> Result<bool, OnnxError> {
    match attrs.i("storage_order", 0) {
        0 => Ok(false),
        1 => {
            if rank > 2 {
                return Err(OnnxError::Unsupported(format!(
                    "MaxPool: storage_order=1 (column major) is only defined for 1D/2D pooling, \
                     got spatial rank {rank}"
                )));
            }
            Ok(true)
        }
        other => Err(OnnxError::ShapeMismatch(format!(
            "MaxPool: storage_order must be 0 (row major) or 1 (column major), got {other}"
        ))),
    }
}

// ── MaxPool ─────────────────────────────────────────────────────────────────

pub struct MaxPoolOp;
impl Operator for MaxPoolOp {
    fn op_type(&self) -> &str {
        "MaxPool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let geo = pool_geometry(attrs, &input.shape, "MaxPool")?;
        let column_major = read_storage_order(attrs, geo.rank())?;
        let want_indices = ctx.node.outputs.get(1).is_some_and(|name| !name.is_empty());

        let out_shape = geo.out_shape(&input.shape);
        let total: usize = out_shape.iter().product();
        let mut values = vec![f32::NEG_INFINITY; total];
        let mut indices = if want_indices {
            vec![0.0_f32; total]
        } else {
            Vec::new()
        };
        conv::max_pool_into(
            &input.data,
            &input.shape,
            &geo,
            &mut values,
            &mut indices,
            column_major,
        );

        let mut results = vec![Tensor::new(values, out_shape.clone())];
        if want_indices {
            results.push(Tensor::new(indices, out_shape));
        }
        Ok(results)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() || slots.len() > 2 {
            return Err(OnnxError::Internal(format!(
                "MaxPoolOp: expected 1 or 2 output slots, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let geo = pool_geometry(attrs, &input.shape, "MaxPool")?;
        let column_major = read_storage_order(attrs, geo.rank())?;
        let want_indices =
            slots.len() > 1 && ctx.node.outputs.get(1).is_some_and(|name| !name.is_empty());

        let out_shape = geo.out_shape(&input.shape);
        let total: usize = out_shape.iter().product();

        let (head, tail) = slots.split_at_mut(1);
        let values = &mut head[0];
        if values.data.len() != total {
            values.data.resize(total, f32::NEG_INFINITY);
        }
        values.shape.clone_from(&out_shape);

        let mut no_indices: [f32; 0] = [];
        let indices: &mut [f32] = match tail.first_mut() {
            Some(slot) if want_indices => {
                if slot.data.len() != total {
                    slot.data.resize(total, 0.0_f32);
                }
                slot.shape.clone_from(&out_shape);
                slot.data.as_mut_slice()
            }
            _ => &mut no_indices,
        };

        conv::max_pool_into(
            &input.data,
            &input.shape,
            &geo,
            &mut values.data,
            indices,
            column_major,
        );
        Ok(())
    }
}

// ── AveragePool ─────────────────────────────────────────────────────────────

pub struct AveragePoolOp;
impl Operator for AveragePoolOp {
    fn op_type(&self) -> &str {
        "AveragePool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let geo = pool_geometry(attrs, &input.shape, "AveragePool")?;
        let count_include_pad = attrs.i("count_include_pad", 0) != 0;

        let out_shape = geo.out_shape(&input.shape);
        let total: usize = out_shape.iter().product();
        let mut values = vec![0.0_f32; total];
        conv::avg_pool_into(
            &input.data,
            &input.shape,
            &geo,
            count_include_pad,
            &mut values,
        );
        Ok(vec![Tensor::new(values, out_shape)])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() != 1 {
            return Err(OnnxError::Internal(format!(
                "AveragePoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let geo = pool_geometry(attrs, &input.shape, "AveragePool")?;
        let count_include_pad = attrs.i("count_include_pad", 0) != 0;

        let out_shape = geo.out_shape(&input.shape);
        let total: usize = out_shape.iter().product();
        if slots[0].data.len() != total {
            slots[0].data.resize(total, 0.0_f32);
        }
        slots[0].shape.clone_from(&out_shape);
        conv::avg_pool_into(
            &input.data,
            &input.shape,
            &geo,
            count_include_pad,
            &mut slots[0].data,
        );
        Ok(())
    }
}

// ── GlobalAveragePool / GlobalMaxPool ───────────────────────────────────────

pub struct GlobalAveragePoolOp;
impl Operator for GlobalAveragePoolOp {
    fn op_type(&self) -> &str {
        "GlobalAveragePool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![conv::global_avg_pool(ctx.input(0)?)])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() != 1 {
            return Err(OnnxError::Internal(format!(
                "GlobalAveragePoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let x = ctx.input(0)?;
        // Degenerate case: fewer than 3 dims — copy input directly into slot
        if x.ndim() < 3 {
            slots[0].data.resize(x.data.len(), 0.0_f32);
            slots[0].data.copy_from_slice(&x.data);
            slots[0].shape = x.shape.clone();
            return Ok(());
        }
        let n = x.shape[0];
        let c = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        let total = n * c;
        if slots[0].data.len() != total {
            slots[0].data.resize(total, 0.0_f32);
        }
        let mut out_shape = vec![n, c];
        out_shape.extend(vec![1_usize; x.ndim() - 2]);
        slots[0].shape = out_shape;
        for ni in 0..n {
            for ci in 0..c {
                let base = ni * c * spatial + ci * spatial;
                let sum: f32 = x.data[base..base + spatial].iter().sum();
                slots[0].data[ni * c + ci] = sum / spatial as f32;
            }
        }
        Ok(())
    }
}

pub struct GlobalMaxPoolOp;
impl Operator for GlobalMaxPoolOp {
    fn op_type(&self) -> &str {
        "GlobalMaxPool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![conv::global_max_pool(ctx.input(0)?)])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() != 1 {
            return Err(OnnxError::Internal(format!(
                "GlobalMaxPoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let x = ctx.input(0)?;
        if x.ndim() < 3 {
            slots[0].data.resize(x.data.len(), 0.0_f32);
            slots[0].data.copy_from_slice(&x.data);
            slots[0].shape = x.shape.clone();
            return Ok(());
        }
        let n = x.shape[0];
        let c = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        let total = n * c;
        if slots[0].data.len() != total {
            slots[0].data.resize(total, f32::NEG_INFINITY);
        }
        let mut out_shape = vec![n, c];
        out_shape.extend(vec![1_usize; x.ndim() - 2]);
        slots[0].shape = out_shape;
        for ni in 0..n {
            for ci in 0..c {
                let base = ni * c * spatial + ci * spatial;
                let max_val = x.data[base..base + spatial]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                slots[0].data[ni * c + ci] = max_val;
            }
        }
        Ok(())
    }
}
