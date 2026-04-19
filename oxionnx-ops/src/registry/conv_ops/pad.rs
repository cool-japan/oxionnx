//! PadOp operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

// ── Pad ─────────────────────────────────────────────────────────────────────

pub struct PadOp;
impl Operator for PadOp {
    fn op_type(&self) -> &str {
        "Pad"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let pads_tensor = ctx.input(1)?;
        let pads_vals: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let constant_value = ctx.optional_input(2).map(|t| t.data[0]).unwrap_or(0.0);
        let mode = ctx.attrs().s("mode");
        let mode = if mode.is_empty() { "constant" } else { mode };
        Ok(vec![crate::shape::pad(
            input,
            &pads_vals,
            mode,
            constant_value,
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
                "PadOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let pads_tensor = ctx.input(1)?;
        let pads_vals: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let constant_value = ctx.optional_input(2).map(|t| t.data[0]).unwrap_or(0.0);
        let mode = ctx.attrs().s("mode");
        let mode_str: &str = if mode.is_empty() { "constant" } else { mode };

        let ndim = input.ndim();
        if pads_vals.len() != 2 * ndim {
            return Err(OnnxError::Internal(format!(
                "PadOp: pads length {} != 2 * ndim {}",
                pads_vals.len(),
                2 * ndim
            )));
        }

        let begin: Vec<usize> = pads_vals[..ndim]
            .iter()
            .map(|&p| p.max(0) as usize)
            .collect();
        let end: Vec<usize> = pads_vals[ndim..]
            .iter()
            .map(|&p| p.max(0) as usize)
            .collect();

        let out_shape: Vec<usize> = (0..ndim)
            .map(|d| input.shape[d] + begin[d] + end[d])
            .collect();
        let out_n: usize = out_shape.iter().product();

        if slots[0].data.len() != out_n {
            slots[0].data.resize(out_n, constant_value);
        }
        // Fill entire buffer with constant_value first (needed for "constant" mode,
        // and also ensures padding regions are correct for "reflect"/"edge" modes
        // which overwrite every position anyway).
        slots[0].data.fill(constant_value);
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

        match mode_str {
            "reflect" => {
                for (out_idx, out_val) in slots[0].data.iter_mut().enumerate() {
                    let mut rem = out_idx;
                    let mut in_idx = 0_usize;
                    let mut valid = true;
                    for d in 0..ndim {
                        let out_coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        let in_coord_signed = out_coord as isize - begin[d] as isize;
                        let dim = input.shape[d] as isize;
                        let mut c = in_coord_signed;
                        if dim <= 1 {
                            c = 0;
                        } else {
                            let period = 2 * (dim - 1);
                            c = c.rem_euclid(period);
                            if c >= dim {
                                c = period - c;
                            }
                        }
                        if c < 0 || c >= dim {
                            valid = false;
                            break;
                        }
                        in_idx += c as usize * in_strides[d];
                    }
                    if valid {
                        *out_val = input.data[in_idx];
                    }
                }
            }
            "edge" => {
                for (out_idx, out_val) in slots[0].data.iter_mut().enumerate() {
                    let mut rem = out_idx;
                    let mut in_idx = 0_usize;
                    for d in 0..ndim {
                        let out_coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        let in_coord = (out_coord as isize - begin[d] as isize)
                            .max(0)
                            .min(input.shape[d] as isize - 1)
                            as usize;
                        in_idx += in_coord * in_strides[d];
                    }
                    *out_val = input.data[in_idx];
                }
            }
            _ => {
                // "constant" mode: fill already done above; copy input into interior
                for (out_idx, out_val) in slots[0].data.iter_mut().enumerate() {
                    let mut rem = out_idx;
                    let mut in_idx = 0_usize;
                    let mut inside = true;
                    for d in 0..ndim {
                        let out_coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        let in_coord = out_coord as isize - begin[d] as isize;
                        if in_coord < 0 || in_coord >= input.shape[d] as isize {
                            inside = false;
                            break;
                        }
                        in_idx += in_coord as usize * in_strides[d];
                    }
                    if inside {
                        *out_val = input.data[in_idx];
                    }
                }
            }
        }
        Ok(())
    }
}
