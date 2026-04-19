//! ResizeOp operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::resize::{nearest_index, transform_coord};

// ── Resize ──────────────────────────────────────────────────────────────────

pub struct ResizeOp;
impl Operator for ResizeOp {
    fn op_type(&self) -> &str {
        "Resize"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let scales: Option<Vec<f32>> = ctx.optional_input(2).and_then(|t| {
            if t.numel() > 0 {
                Some(t.data.clone())
            } else {
                None
            }
        });
        let sizes: Option<Vec<usize>> = ctx.optional_input(3).and_then(|t| {
            if t.numel() > 0 {
                Some(t.data.iter().map(|&v| v as usize).collect())
            } else {
                None
            }
        });
        let attrs = ctx.attrs();
        let mode = attrs.s("mode");
        let mode = if mode.is_empty() { "nearest" } else { mode };
        let coord_transform = attrs.s("coordinate_transformation_mode");
        let coord_transform = if coord_transform.is_empty() {
            "half_pixel"
        } else {
            coord_transform
        };
        Ok(vec![crate::resize::resize(
            input,
            scales.as_deref(),
            sizes.as_deref(),
            mode,
            coord_transform,
        )])
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
                "ResizeOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let scales: Option<Vec<f32>> = ctx.optional_input(2).and_then(|t| {
            if t.numel() > 0 {
                Some(t.data.clone())
            } else {
                None
            }
        });
        let sizes: Option<Vec<usize>> = ctx.optional_input(3).and_then(|t| {
            if t.numel() > 0 {
                Some(t.data.iter().map(|&v| v as usize).collect())
            } else {
                None
            }
        });
        let attrs = ctx.attrs();
        let mode = attrs.s("mode");
        let mode = if mode.is_empty() { "nearest" } else { mode };
        let coord_transform = attrs.s("coordinate_transformation_mode");
        let coord_transform = if coord_transform.is_empty() {
            "half_pixel"
        } else {
            coord_transform
        };

        let ndim = input.ndim();
        let out_shape: Vec<usize> = if let Some(ref sizes) = sizes {
            sizes.clone()
        } else if let Some(ref sc) = scales {
            input
                .shape
                .iter()
                .zip(sc.iter())
                .map(|(&d, &s)| (d as f32 * s).round() as usize)
                .collect()
        } else {
            // No resize info: copy input data into slot unchanged
            let n = input.data.len();
            if slots[0].data.len() != n {
                slots[0].data.resize(n, 0.0_f32);
            }
            slots[0].data.copy_from_slice(&input.data);
            slots[0].shape = input.shape.clone();
            return Ok(());
        };

        let out_n: usize = out_shape.iter().product();
        if slots[0].data.len() != out_n {
            slots[0].data.resize(out_n, 0.0_f32);
        }
        slots[0].shape = out_shape.clone();

        // Compute strides
        let mut in_strides = vec![0_usize; ndim];
        let mut s = 1_usize;
        for i in (0..ndim).rev() {
            in_strides[i] = s;
            s *= input.shape[i];
        }
        let mut out_strides = vec![0_usize; ndim];
        let mut s = 1_usize;
        for i in (0..ndim).rev() {
            out_strides[i] = s;
            s *= out_shape[i];
        }

        let dim_scales: Vec<f32> = (0..ndim)
            .map(|d| {
                if out_shape[d] == input.shape[d] {
                    1.0
                } else if let Some(ref sc) = scales {
                    sc[d]
                } else {
                    out_shape[d] as f32 / input.shape[d] as f32
                }
            })
            .collect();

        match mode {
            "linear" | "bilinear" => {
                let spatial_start = ndim.saturating_sub(2);
                for (out_idx, out_val) in slots[0].data.iter_mut().enumerate() {
                    let mut rem = out_idx;
                    let mut out_coords = vec![0_usize; ndim];
                    for d in 0..ndim {
                        out_coords[d] = rem / out_strides[d];
                        rem %= out_strides[d];
                    }
                    let mut base_idx = 0_usize;
                    for d in 0..spatial_start {
                        let src = transform_coord(
                            out_coords[d],
                            dim_scales[d],
                            input.shape[d],
                            out_shape[d],
                            coord_transform,
                        );
                        let idx = src.round().max(0.0).min((input.shape[d] - 1) as f32) as usize;
                        base_idx += idx * in_strides[d];
                    }
                    if ndim - spatial_start == 2 {
                        let d0 = spatial_start;
                        let d1 = spatial_start + 1;
                        let sy = transform_coord(
                            out_coords[d0],
                            dim_scales[d0],
                            input.shape[d0],
                            out_shape[d0],
                            coord_transform,
                        );
                        let sx = transform_coord(
                            out_coords[d1],
                            dim_scales[d1],
                            input.shape[d1],
                            out_shape[d1],
                            coord_transform,
                        );
                        let y0 = sy.floor().max(0.0) as usize;
                        let x0 = sx.floor().max(0.0) as usize;
                        let y1 = (y0 + 1).min(input.shape[d0] - 1);
                        let x1 = (x0 + 1).min(input.shape[d1] - 1);
                        let fy = sy - sy.floor();
                        let fx = sx - sx.floor();
                        let v00 = input.data[base_idx + y0 * in_strides[d0] + x0 * in_strides[d1]];
                        let v01 = input.data[base_idx + y0 * in_strides[d0] + x1 * in_strides[d1]];
                        let v10 = input.data[base_idx + y1 * in_strides[d0] + x0 * in_strides[d1]];
                        let v11 = input.data[base_idx + y1 * in_strides[d0] + x1 * in_strides[d1]];
                        *out_val = v00 * (1.0 - fy) * (1.0 - fx)
                            + v01 * (1.0 - fy) * fx
                            + v10 * fy * (1.0 - fx)
                            + v11 * fy * fx;
                    } else if ndim - spatial_start == 1 {
                        let d = spatial_start;
                        let sx = transform_coord(
                            out_coords[d],
                            dim_scales[d],
                            input.shape[d],
                            out_shape[d],
                            coord_transform,
                        );
                        let x0 = sx.floor().max(0.0) as usize;
                        let x1 = (x0 + 1).min(input.shape[d] - 1);
                        let fx = sx - sx.floor();
                        let v0 = input.data[base_idx + x0 * in_strides[d]];
                        let v1 = input.data[base_idx + x1 * in_strides[d]];
                        *out_val = v0 * (1.0 - fx) + v1 * fx;
                    } else {
                        *out_val = input.data[base_idx];
                    }
                }
            }
            // "nearest" and fallback
            _ => {
                for (out_idx, out_val) in slots[0].data.iter_mut().enumerate() {
                    let mut rem = out_idx;
                    let mut in_idx = 0_usize;
                    for d in 0..ndim {
                        let out_coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        let src = transform_coord(
                            out_coord,
                            dim_scales[d],
                            input.shape[d],
                            out_shape[d],
                            coord_transform,
                        );
                        let src_idx = nearest_index(src, input.shape[d]);
                        in_idx += src_idx * in_strides[d];
                    }
                    *out_val = input.data[in_idx];
                }
            }
        }
        Ok(())
    }
}
