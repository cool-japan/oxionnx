//! GRU operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::{rnn, rnn_typed};

pub struct GRUOp;
impl Operator for GRUOp {
    fn op_type(&self) -> &str {
        "GRU"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let r = ctx.input(2)?;
        let b = ctx.optional_input(3);
        let sequence_lens = ctx.optional_input(4);
        let initial_h = ctx.optional_input(5);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let linear_before_reset = attrs.i("linear_before_reset", 0) != 0;
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        let (y, y_h) = rnn::gru(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            linear_before_reset,
            activations,
        )?;

        Ok(vec![y, y_h])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() < 2 {
            return Err(OnnxError::Internal(format!(
                "GRUOp: expected at least 2 output slots, got {}",
                slots.len()
            )));
        }

        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let r = ctx.input(2)?;
        let b = ctx.optional_input(3);
        let sequence_lens = ctx.optional_input(4);
        let initial_h = ctx.optional_input(5);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let linear_before_reset = attrs.i("linear_before_reset", 0) != 0;
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        // Compute output shapes before mutably borrowing slots.
        let seq_len = x.shape[0];
        let batch = x.shape[1];
        let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
        let y_shape = vec![seq_len, num_dir, batch, hidden_size];
        let y_h_shape = vec![num_dir, batch, hidden_size];

        // Resize and set shapes before the 2-way destructure.
        let y_len: usize = y_shape.iter().product();
        let yh_len: usize = y_h_shape.iter().product();
        if slots[0].data.len() != y_len {
            slots[0].data.resize(y_len, 0.0f32);
        }
        slots[0].shape.clone_from(&y_shape);
        if slots[1].data.len() != yh_len {
            slots[1].data.resize(yh_len, 0.0f32);
        }
        slots[1].shape.clone_from(&y_h_shape);

        // Destructure into 2 disjoint mutable borrows.
        let (head, _) = slots.split_at_mut(2);
        let [y_slot, y_h_slot] = head else {
            return Err(OnnxError::Internal(
                "GRUOp: failed to destructure 2 output slots".into(),
            ));
        };

        rnn::gru_into(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            linear_before_reset,
            activations,
            rnn::GruOutputSlots {
                y: &mut y_slot.data,
                y_h: &mut y_h_slot.data,
            },
        )?;

        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, Tensor, TypedTensor};

        let x_tt = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("GRUOp: missing input X".into()))?;
        let w_tt = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("GRUOp: missing input W".into()))?;
        let r_tt = ctx
            .input(2)
            .ok_or_else(|| OnnxError::TensorNotFound("GRUOp: missing input R".into()))?;
        let b_tt = ctx.input(3);
        let sequence_lens_tt = ctx.input(4);
        let initial_h_tt = ctx.input(5);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let linear_before_reset = attrs.i("linear_before_reset", 0) != 0;
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations: Option<&[&str]> = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        // sequence_lens is I32; convert to f32 Tensor for the existing gru kernel.
        let sequence_lens_f32: Option<Tensor> =
            sequence_lens_tt.map(|tt| Tensor::new(tt.storage.to_f32_vec(), tt.shape.clone()));

        match &x_tt.storage {
            TensorStorage::F32(_) => oxionnx_core::default_typed_via_f32(self, ctx),

            TensorStorage::F16(xb) => {
                let w_bits = match &w_tt.storage {
                    TensorStorage::F16(b) => b.as_slice(),
                    _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                };
                let r_bits = match &r_tt.storage {
                    TensorStorage::F16(b) => b.as_slice(),
                    _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                };
                let b_f16: Option<(&[u16], &[usize])> = b_tt.and_then(|tt| match &tt.storage {
                    TensorStorage::F16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                    _ => None,
                });
                let ih_f16: Option<(&[u16], &[usize])> =
                    initial_h_tt.and_then(|tt| match &tt.storage {
                        TensorStorage::F16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                        _ => None,
                    });

                let out = rnn_typed::gru_f16(rnn_typed::GruTypedArgs {
                    x: (xb, &x_tt.shape),
                    w: (w_bits, &w_tt.shape),
                    r: (r_bits, &r_tt.shape),
                    b: b_f16,
                    sequence_lens_f32,
                    initial_h: ih_f16,
                    hidden_size,
                    direction,
                    linear_before_reset,
                    activations,
                })?;

                Ok(vec![
                    TypedTensor::new(TensorStorage::F16(out.y_bits), out.y_shape),
                    TypedTensor::new(TensorStorage::F16(out.y_h_bits), out.y_h_shape),
                ])
            }

            TensorStorage::BF16(xb) => {
                let w_bits = match &w_tt.storage {
                    TensorStorage::BF16(b) => b.as_slice(),
                    _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                };
                let r_bits = match &r_tt.storage {
                    TensorStorage::BF16(b) => b.as_slice(),
                    _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                };
                let b_bf16: Option<(&[u16], &[usize])> = b_tt.and_then(|tt| match &tt.storage {
                    TensorStorage::BF16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                    _ => None,
                });
                let ih_bf16: Option<(&[u16], &[usize])> =
                    initial_h_tt.and_then(|tt| match &tt.storage {
                        TensorStorage::BF16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                        _ => None,
                    });

                let out = rnn_typed::gru_bf16(rnn_typed::GruTypedArgs {
                    x: (xb, &x_tt.shape),
                    w: (w_bits, &w_tt.shape),
                    r: (r_bits, &r_tt.shape),
                    b: b_bf16,
                    sequence_lens_f32,
                    initial_h: ih_bf16,
                    hidden_size,
                    direction,
                    linear_before_reset,
                    activations,
                })?;

                Ok(vec![
                    TypedTensor::new(TensorStorage::BF16(out.y_bits), out.y_shape),
                    TypedTensor::new(TensorStorage::BF16(out.y_h_bits), out.y_h_shape),
                ])
            }

            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}
