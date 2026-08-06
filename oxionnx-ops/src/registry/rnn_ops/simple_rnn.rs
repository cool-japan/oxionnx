//! Plain ONNX `RNN` (Elman) operator implementation.
//!
//! A thin registry wrapper over the fully-featured [`crate::rnn::simple_rnn_ext`]
//! kernel (forward / reverse / bidirectional, `sequence_lens`, per-direction
//! `activations`, `clip`, `layout`), which until now had no registered operator
//! and was therefore unreachable from a real `.onnx` file.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::rnn;

use super::lstm::rnn_extras;

/// ONNX `RNN` (opset 7+): `h_t = f(clip(x_t W^T + h_{t-1} R^T + Wb + Rb))`.
///
/// Inputs: `X, W, R[, B][, sequence_lens][, initial_h]`.
/// Outputs: `Y` (all hidden states) and `Y_h` (the last one).
pub struct RNNOp;

impl Operator for RNNOp {
    fn op_type(&self) -> &str {
        "RNN"
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
        // `activations` holds one entry per direction; absent means the ONNX
        // default (`Tanh`), which the kernel supplies itself.
        let act_strs = attrs.string_list("activations");
        let act_refs: Vec<&str> = act_strs.iter().map(|s| s.as_str()).collect();
        let activations = if act_refs.is_empty() {
            None
        } else {
            Some(act_refs.as_slice())
        };

        let (y, y_h) = rnn::simple_rnn_ext(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            activations,
            rnn_extras(attrs),
        )?;

        Ok(vec![y, y_h])
    }
}
