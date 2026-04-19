//! Normalization operator implementations: LayerNorm, GroupNorm, BatchNorm,
//! RmsNorm, InstanceNorm, LpNorm, MeanVarianceNormalization.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::nn;

// ── LayerNorm ───────────────────────────────────────────────────────────────

pub struct LayerNormOp;
impl Operator for LayerNormOp {
    fn op_type(&self) -> &str {
        "LayerNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let eps = attrs.f("epsilon", 1e-5);
        let axis = attrs.i("axis", -1);
        Ok(vec![nn::layer_norm(x, scale, bias, eps, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── GroupNorm ───────────────────────────────────────────────────────────────

pub struct GroupNormOp;
impl Operator for GroupNormOp {
    fn op_type(&self) -> &str {
        "GroupNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let num_groups = attrs.i("num_groups", 1) as usize;
        let eps = attrs.f("epsilon", 1e-5);
        Ok(vec![nn::group_norm(x, scale, bias, num_groups, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── BatchNorm ───────────────────────────────────────────────────────────────

pub struct BatchNormOp;
impl Operator for BatchNormOp {
    fn op_type(&self) -> &str {
        "BatchNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let mean = ctx.input(3)?;
        let var = ctx.input(4)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![nn::batch_norm(x, scale, bias, mean, var, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────

pub struct RmsNormOp;
impl Operator for RmsNormOp {
    fn op_type(&self) -> &str {
        "SimplifiedLayerNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let eps = ctx.attrs().f("epsilon", 1e-6);
        Ok(vec![nn::rms_norm(x, scale, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── InstanceNorm ────────────────────────────────────────────────────────────

pub struct InstanceNormOp;
impl Operator for InstanceNormOp {
    fn op_type(&self) -> &str {
        "InstanceNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![nn::instance_norm(x, scale, bias, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── LpNorm ──────────────────────────────────────────────────────────────────

pub struct LpNormOp;
impl Operator for LpNormOp {
    fn op_type(&self) -> &str {
        "LpNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", -1);
        let p = attrs.i("p", 2);
        Ok(vec![nn::lp_norm(ctx.input(0)?, axis, p)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let axis = ctx.attrs().i("axis", -1);
        let p = ctx.attrs().i("p", 2);
        let ndim = input.ndim();
        let ax = if axis < 0 {
            (axis + ndim as i64) as usize
        } else {
            axis as usize
        };
        if ax >= ndim {
            return Err(OnnxError::from(format!(
                "lp_norm: axis {ax} out of range for ndim {ndim}"
            )));
        }
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        slots[0].data.copy_from_slice(&input.data);

        let axis_size = input.shape[ax];
        let outer: usize = input.shape[..ax].iter().product();
        let inner: usize = input.shape[ax + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                let norm: f32 = if p == 1 {
                    (0..axis_size)
                        .map(|a| input.data[(o * axis_size + a) * inner + i].abs())
                        .sum()
                } else {
                    (0..axis_size)
                        .map(|a| {
                            let v = input.data[(o * axis_size + a) * inner + i];
                            v * v
                        })
                        .sum::<f32>()
                        .sqrt()
                };
                let norm = if norm == 0.0 { 1.0 } else { norm };
                for a in 0..axis_size {
                    slots[0].data[(o * axis_size + a) * inner + i] /= norm;
                }
            }
        }
        Ok(())
    }
}

// ── MeanVarianceNormalization ───────────────────────────────────────────────

pub struct MeanVarianceNormalizationOp;
impl Operator for MeanVarianceNormalizationOp {
    fn op_type(&self) -> &str {
        "MeanVarianceNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axes_list = ctx.attrs().ints("axes");
        let axes = if axes_list.is_empty() {
            vec![0, 2, 3]
        } else {
            axes_list.to_vec()
        };
        Ok(vec![nn::mean_variance_normalization(ctx.input(0)?, &axes)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let axes_list = ctx.attrs().ints("axes");
        let axes: Vec<i64> = if axes_list.is_empty() {
            vec![0, 2, 3]
        } else {
            axes_list.to_vec()
        };

        let ndim = input.ndim();
        if ndim == 0 {
            return Err(OnnxError::from(
                "mean_variance_normalization: input must have at least 1 dimension".to_string(),
            ));
        }

        let norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + ndim as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        for &ax in &norm_axes {
            if ax >= ndim {
                return Err(OnnxError::from(format!(
                    "mean_variance_normalization: axis {ax} out of range for {ndim}D tensor"
                )));
            }
        }

        let total = input.numel();
        let n = total;
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        slots[0].data.copy_from_slice(&input.data);

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * input.shape[d + 1];
        }

        let mut is_reduced = vec![false; ndim];
        for &ax in &norm_axes {
            is_reduced[ax] = true;
        }

        let reduced_size: usize = (0..ndim)
            .filter(|&d| is_reduced[d])
            .map(|d| input.shape[d])
            .product::<usize>()
            .max(1);
        let non_reduced_size = total / reduced_size;

        let non_reduced_dims: Vec<usize> = (0..ndim).filter(|&d| !is_reduced[d]).collect();
        let reduced_dims: Vec<usize> = (0..ndim).filter(|&d| is_reduced[d]).collect();

        for nr_idx in 0..non_reduced_size {
            let mut nr_coords = vec![0usize; non_reduced_dims.len()];
            let mut rem = nr_idx;
            for j in (0..non_reduced_dims.len()).rev() {
                let dim = non_reduced_dims[j];
                nr_coords[j] = rem % input.shape[dim];
                rem /= input.shape[dim];
            }

            let mut flat_indices = Vec::with_capacity(reduced_size);
            nn::collect_reduced_indices(
                input,
                &strides,
                &reduced_dims,
                &non_reduced_dims,
                &nr_coords,
                0,
                0,
                &mut flat_indices,
            );

            let mean: f32 =
                flat_indices.iter().map(|&fi| input.data[fi]).sum::<f32>() / reduced_size as f32;
            let var: f32 = flat_indices
                .iter()
                .map(|&fi| (input.data[fi] - mean) * (input.data[fi] - mean))
                .sum::<f32>()
                / reduced_size as f32;
            let inv_std = 1.0 / (var + 1e-9_f32).sqrt();

            for &fi in &flat_indices {
                slots[0].data[fi] = (input.data[fi] - mean) * inv_std;
            }
        }
        Ok(())
    }
}
