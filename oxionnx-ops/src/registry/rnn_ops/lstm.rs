//! LSTM operator implementation.

use oxionnx_core::{Attributes, OnnxError, OpContext, Operator, Tensor};

use crate::{rnn, rnn_typed};

/// Read the optional ONNX RNN attributes (`clip`, `layout`, `activation_alpha`,
/// `activation_beta`) that the plain kernel entry points leave at their defaults.
///
/// `clip` must be strictly positive to have any meaning; a missing, non-positive
/// or non-finite value disables clipping, matching ONNX Runtime.
pub(super) fn rnn_extras(attrs: &Attributes) -> rnn::RnnExtras<'_> {
    let clip = attrs
        .floats
        .get("clip")
        .copied()
        .filter(|c| c.is_finite() && *c > 0.0)
        .unwrap_or(f32::INFINITY);
    rnn::RnnExtras {
        clip,
        layout: attrs.i("layout", 0),
        activation_alpha: attrs
            .float_lists
            .get("activation_alpha")
            .map_or(&[][..], |v| v.as_slice()),
        activation_beta: attrs
            .float_lists
            .get("activation_beta")
            .map_or(&[][..], |v| v.as_slice()),
    }
}

pub struct LSTMOp;
impl Operator for LSTMOp {
    fn op_type(&self) -> &str {
        "LSTM"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let r = ctx.input(2)?;
        let b = ctx.optional_input(3);
        let sequence_lens = ctx.optional_input(4);
        let initial_h = ctx.optional_input(5);
        let initial_c = ctx.optional_input(6);
        let peephole = ctx.optional_input(7);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        let (y, y_h, y_c) = rnn::lstm_ext(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            initial_c,
            peephole,
            hidden_size,
            direction,
            activations,
            rnn_extras(attrs),
        )?;

        Ok(vec![y, y_h, y_c])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() < 3 {
            return Err(OnnxError::Internal(format!(
                "LSTMOp: expected at least 3 output slots, got {}",
                slots.len()
            )));
        }

        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let r = ctx.input(2)?;
        let b = ctx.optional_input(3);
        let sequence_lens = ctx.optional_input(4);
        let initial_h = ctx.optional_input(5);
        let initial_c = ctx.optional_input(6);
        let peephole = ctx.optional_input(7);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        let extras = rnn_extras(attrs);

        // Compute output shapes before mutably borrowing slots.
        // With layout=1 the model hands us X as [batch, seq, input_size].
        if x.ndim() != 3 {
            return Err(OnnxError::ShapeMismatch(format!(
                "LSTMOp: X must be 3D, got {:?}",
                x.shape
            )));
        }
        let (seq_len, batch) = if extras.layout == 1 {
            (x.shape[1], x.shape[0])
        } else {
            (x.shape[0], x.shape[1])
        };
        let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
        let (y_shape, state_shape) = if extras.layout == 1 {
            (
                vec![batch, seq_len, num_dir, hidden_size],
                vec![batch, num_dir, hidden_size],
            )
        } else {
            (
                vec![seq_len, num_dir, batch, hidden_size],
                vec![num_dir, batch, hidden_size],
            )
        };
        let y_h_shape = state_shape.clone();
        let y_c_shape = state_shape;

        // Resize and set shapes before the 3-way destructure (avoids aliasing).
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
        if slots[2].data.len() != yh_len {
            slots[2].data.resize(yh_len, 0.0f32);
        }
        slots[2].shape.clone_from(&y_c_shape);

        // Destructure into 3 disjoint mutable borrows.
        let (head, _) = slots.split_at_mut(3);
        let [y_slot, y_h_slot, y_c_slot] = head else {
            return Err(OnnxError::Internal(
                "LSTMOp: failed to destructure 3 output slots".into(),
            ));
        };

        rnn::lstm_into_ext(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            initial_c,
            peephole,
            hidden_size,
            direction,
            activations,
            extras,
            rnn::LstmOutputSlots {
                y: &mut y_slot.data,
                y_h: &mut y_h_slot.data,
                y_c: &mut y_c_slot.data,
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
            .ok_or_else(|| OnnxError::TensorNotFound("LSTMOp: missing input X".into()))?;
        let w_tt = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("LSTMOp: missing input W".into()))?;
        let r_tt = ctx
            .input(2)
            .ok_or_else(|| OnnxError::TensorNotFound("LSTMOp: missing input R".into()))?;
        let b_tt = ctx.input(3);
        let sequence_lens_tt = ctx.input(4);
        let initial_h_tt = ctx.input(5);
        let initial_c_tt = ctx.input(6);
        let peephole_tt = ctx.input(7);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations: Option<&[&str]> = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        // The half-precision kernels below do not carry `clip` / `layout` /
        // `activation_alpha` / `activation_beta`. When any of them is set, route
        // through the f32 path (which is where those attributes are honoured);
        // `rnn_typed` is an f32 round-trip anyway, so no fidelity is lost.
        let extras = rnn_extras(attrs);
        if extras.clip.is_finite()
            || extras.layout != 0
            || !extras.activation_alpha.is_empty()
            || !extras.activation_beta.is_empty()
        {
            return oxionnx_core::default_typed_via_f32(self, ctx);
        }

        // sequence_lens is I32 regardless of the main dtype — convert to f32 Tensor
        // (the existing lstm kernel reads it as f32 and truncates to usize internally).
        let sequence_lens_f32: Option<Tensor> =
            sequence_lens_tt.map(|tt| Tensor::new(tt.storage.to_f32_vec(), tt.shape.clone()));

        match &x_tt.storage {
            TensorStorage::F32(_) => oxionnx_core::default_typed_via_f32(self, ctx),

            TensorStorage::F16(xb) => {
                // Extract F16 bits for each optional input that must share the dtype.
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
                let ic_f16: Option<(&[u16], &[usize])> =
                    initial_c_tt.and_then(|tt| match &tt.storage {
                        TensorStorage::F16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                        _ => None,
                    });
                let ph_f16: Option<(&[u16], &[usize])> =
                    peephole_tt.and_then(|tt| match &tt.storage {
                        TensorStorage::F16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                        _ => None,
                    });

                let out = rnn_typed::lstm_f16(rnn_typed::LstmTypedArgs {
                    x: (xb, &x_tt.shape),
                    w: (w_bits, &w_tt.shape),
                    r: (r_bits, &r_tt.shape),
                    b: b_f16,
                    sequence_lens_f32,
                    initial_h: ih_f16,
                    initial_c: ic_f16,
                    peephole: ph_f16,
                    hidden_size,
                    direction,
                    activations,
                })?;

                Ok(vec![
                    TypedTensor::new(TensorStorage::F16(out.y_bits), out.y_shape),
                    TypedTensor::new(TensorStorage::F16(out.y_h_bits), out.y_h_shape),
                    TypedTensor::new(TensorStorage::F16(out.y_c_bits), out.y_c_shape),
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
                let ic_bf16: Option<(&[u16], &[usize])> =
                    initial_c_tt.and_then(|tt| match &tt.storage {
                        TensorStorage::BF16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                        _ => None,
                    });
                let ph_bf16: Option<(&[u16], &[usize])> =
                    peephole_tt.and_then(|tt| match &tt.storage {
                        TensorStorage::BF16(b) => Some((b.as_slice(), tt.shape.as_slice())),
                        _ => None,
                    });

                let out = rnn_typed::lstm_bf16(rnn_typed::LstmTypedArgs {
                    x: (xb, &x_tt.shape),
                    w: (w_bits, &w_tt.shape),
                    r: (r_bits, &r_tt.shape),
                    b: b_bf16,
                    sequence_lens_f32,
                    initial_h: ih_bf16,
                    initial_c: ic_bf16,
                    peephole: ph_bf16,
                    hidden_size,
                    direction,
                    activations,
                })?;

                Ok(vec![
                    TypedTensor::new(TensorStorage::BF16(out.y_bits), out.y_shape),
                    TypedTensor::new(TensorStorage::BF16(out.y_h_bits), out.y_h_shape),
                    TypedTensor::new(TensorStorage::BF16(out.y_c_bits), out.y_c_shape),
                ])
            }

            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}
