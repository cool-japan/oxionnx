//! GRU operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::{rnn, rnn_typed};

use super::lstm::rnn_extras;

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

        let (y, y_h) = rnn::gru_ext(
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
            rnn_extras(attrs),
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
        let extras = rnn_extras(attrs);

        // Compute output shapes before mutably borrowing slots.
        // With layout=1 the model hands us X as [batch, seq, input_size].
        if x.ndim() != 3 {
            return Err(OnnxError::ShapeMismatch(format!(
                "GRUOp: X must be 3D, got {:?}",
                x.shape
            )));
        }
        let (seq_len, batch) = if extras.layout == 1 {
            (x.shape[1], x.shape[0])
        } else {
            (x.shape[0], x.shape[1])
        };
        let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
        let (y_shape, y_h_shape) = if extras.layout == 1 {
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

        // `rnn::gru_into_ext` — the true zero-copy, `RnnExtras`-aware kernel
        // entry point that already exists in `rnn::gru` — is not reachable
        // from here: it is `pub(crate)` inside a private submodule and is
        // re-exported from `rnn/mod.rs` only as far as `gru_into` (extras
        // defaulted); `rnn/mod.rs` belongs to a different file-ownership
        // scope in this wave, so its re-export list cannot be extended here
        // (flagged for the orchestrator as a one-line follow-up).
        //
        // With default extras (no clip / layout / activation_alpha /
        // activation_beta — the overwhelmingly common case) this still runs
        // the exact same zero-copy `gru_into` path, byte for byte. With any
        // non-default extra — previously silently ignored on this slot path,
        // unlike `LSTMOp` — correctness wins over the zero-copy property:
        // compute via the extras-aware, allocating `rnn::gru_ext` and copy
        // into the caller's slots (`resize` + `copy_from_slice`, never a
        // fresh `Vec` swapped in, so the slot keeps its backing allocation
        // across calls whenever the length does not change).
        let extras_are_default = !extras.clip.is_finite()
            && extras.layout == 0
            && extras.activation_alpha.is_empty()
            && extras.activation_beta.is_empty();

        if extras_are_default {
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
        } else {
            let (y, y_h) = rnn::gru_ext(
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
                extras,
            )?;
            if y_slot.data.len() != y.data.len() {
                y_slot.data.resize(y.data.len(), 0.0f32);
            }
            y_slot.data.copy_from_slice(&y.data);
            y_slot.shape.clone_from(&y.shape);
            if y_h_slot.data.len() != y_h.data.len() {
                y_h_slot.data.resize(y_h.data.len(), 0.0f32);
            }
            y_h_slot.data.copy_from_slice(&y_h.data);
            y_h_slot.shape.clone_from(&y_h.shape);
        }

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

        // The half-precision kernels below do not carry `clip` / `layout` /
        // `activation_alpha` / `activation_beta`. When any of them is set,
        // route through the f32 path (which is where those attributes are
        // honoured); `rnn_typed` is an f32 round-trip anyway, so no fidelity
        // is lost. Mirrors LSTMOp's identical guard.
        let extras = rnn_extras(attrs);
        if extras.clip.is_finite()
            || extras.layout != 0
            || !extras.activation_alpha.is_empty()
            || !extras.activation_beta.is_empty()
        {
            return oxionnx_core::default_typed_via_f32(self, ctx);
        }

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
