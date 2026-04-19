//! Typed (F16/BF16) RNN kernels for LSTMOp and GRUOp.
//!
//! Computation delegates to the existing f32 kernels (`rnn::lstm` / `rnn::gru`).
//! On the F16/BF16 paths every input tensor is cast from half-precision bits to
//! an f32 `Tensor`, the f32 kernel is called, and the resulting f32 outputs are
//! cast back to the requested half-precision format.  This ensures numerical
//! fidelity is identical to the f32 path while advertising native dtype support
//! so that the typed dispatch layer does not need to perform an implicit
//! promotion round-trip at a higher level.

use oxionnx_core::{OnnxError, Tensor};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build an f32 `Tensor` from F16 bits.
fn f16_bits_to_tensor(bits: &[u16], shape: &[usize]) -> Tensor {
    let data: Vec<f32> = bits
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    Tensor::new(data, shape.to_vec())
}

/// Build an f32 `Tensor` from BF16 bits.
fn bf16_bits_to_tensor(bits: &[u16], shape: &[usize]) -> Tensor {
    let data: Vec<f32> = bits
        .iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect();
    Tensor::new(data, shape.to_vec())
}

/// Cast f32 values to F16 bits.
pub(crate) fn f32_to_f16_bits(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

/// Cast f32 values to BF16 bits.
pub(crate) fn f32_to_bf16_bits(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

// ── LSTM typed output ────────────────────────────────────────────────────────

/// Typed output from an F16 or BF16 LSTM kernel.
pub(crate) struct LstmTypedOutput {
    /// Half-precision bits for Y   (`[seq_len, num_dir, batch, hidden_size]`)
    pub y_bits: Vec<u16>,
    pub y_shape: Vec<usize>,
    /// Half-precision bits for Y_h (`[num_dir, batch, hidden_size]`)
    pub y_h_bits: Vec<u16>,
    pub y_h_shape: Vec<usize>,
    /// Half-precision bits for Y_c (`[num_dir, batch, hidden_size]`)
    pub y_c_bits: Vec<u16>,
    pub y_c_shape: Vec<usize>,
}

/// Bundled arguments for a typed LSTM kernel call.
///
/// Mandatory float tensors are represented as `(bits, shape)` pairs.
/// `sequence_lens_f32` is already converted from I32 by the caller so it
/// arrives as an `Option<Tensor>` instead.
pub(crate) struct LstmTypedArgs<'a> {
    pub x: (&'a [u16], &'a [usize]),
    pub w: (&'a [u16], &'a [usize]),
    pub r: (&'a [u16], &'a [usize]),
    pub b: Option<(&'a [u16], &'a [usize])>,
    pub sequence_lens_f32: Option<Tensor>,
    pub initial_h: Option<(&'a [u16], &'a [usize])>,
    pub initial_c: Option<(&'a [u16], &'a [usize])>,
    pub peephole: Option<(&'a [u16], &'a [usize])>,
    pub hidden_size: usize,
    pub direction: &'a str,
    pub activations: Option<&'a [&'a str]>,
}

/// Internal shared LSTM computation: convert typed inputs → f32, call `rnn::lstm`,
/// convert outputs back with the provided `cast_out` function.
///
/// `cast_in` converts the bit-level half-precision tensors to f32 `Tensor` values.
/// `cast_out` converts f32 output data back to half-precision bits.
fn lstm_typed_inner(
    args: LstmTypedArgs<'_>,
    cast_in: impl Fn(&[u16], &[usize]) -> Tensor,
    cast_out: impl Fn(&[f32]) -> Vec<u16>,
) -> Result<LstmTypedOutput, OnnxError> {
    let x = cast_in(args.x.0, args.x.1);
    let w = cast_in(args.w.0, args.w.1);
    let r = cast_in(args.r.0, args.r.1);
    let b = args.b.map(|(bits, shape)| cast_in(bits, shape));
    let initial_h = args.initial_h.map(|(bits, shape)| cast_in(bits, shape));
    let initial_c = args.initial_c.map(|(bits, shape)| cast_in(bits, shape));
    let peephole = args.peephole.map(|(bits, shape)| cast_in(bits, shape));

    let (y, y_h, y_c) = crate::rnn::lstm(
        &x,
        &w,
        &r,
        b.as_ref(),
        args.sequence_lens_f32.as_ref(),
        initial_h.as_ref(),
        initial_c.as_ref(),
        peephole.as_ref(),
        args.hidden_size,
        args.direction,
        args.activations,
    )?;

    Ok(LstmTypedOutput {
        y_bits: cast_out(&y.data),
        y_shape: y.shape,
        y_h_bits: cast_out(&y_h.data),
        y_h_shape: y_h.shape,
        y_c_bits: cast_out(&y_c.data),
        y_c_shape: y_c.shape,
    })
}

// ── Public LSTM kernel entry-points ─────────────────────────────────────────

/// F16 LSTM kernel.  Inputs are F16 bits; output bits are also F16.
pub(crate) fn lstm_f16(args: LstmTypedArgs<'_>) -> Result<LstmTypedOutput, OnnxError> {
    lstm_typed_inner(args, f16_bits_to_tensor, f32_to_f16_bits)
}

/// BF16 LSTM kernel.  Inputs are BF16 bits; output bits are also BF16.
pub(crate) fn lstm_bf16(args: LstmTypedArgs<'_>) -> Result<LstmTypedOutput, OnnxError> {
    lstm_typed_inner(args, bf16_bits_to_tensor, f32_to_bf16_bits)
}

// ── GRU typed output ─────────────────────────────────────────────────────────

/// Typed output from an F16 or BF16 GRU kernel.
pub(crate) struct GruTypedOutput {
    /// Half-precision bits for Y   (`[seq_len, num_dir, batch, hidden_size]`)
    pub y_bits: Vec<u16>,
    pub y_shape: Vec<usize>,
    /// Half-precision bits for Y_h (`[num_dir, batch, hidden_size]`)
    pub y_h_bits: Vec<u16>,
    pub y_h_shape: Vec<usize>,
}

/// Bundled arguments for a typed GRU kernel call.
pub(crate) struct GruTypedArgs<'a> {
    pub x: (&'a [u16], &'a [usize]),
    pub w: (&'a [u16], &'a [usize]),
    pub r: (&'a [u16], &'a [usize]),
    pub b: Option<(&'a [u16], &'a [usize])>,
    pub sequence_lens_f32: Option<Tensor>,
    pub initial_h: Option<(&'a [u16], &'a [usize])>,
    pub hidden_size: usize,
    pub direction: &'a str,
    pub linear_before_reset: bool,
    pub activations: Option<&'a [&'a str]>,
}

/// Internal shared GRU computation.
fn gru_typed_inner(
    args: GruTypedArgs<'_>,
    cast_in: impl Fn(&[u16], &[usize]) -> Tensor,
    cast_out: impl Fn(&[f32]) -> Vec<u16>,
) -> Result<GruTypedOutput, OnnxError> {
    let x = cast_in(args.x.0, args.x.1);
    let w = cast_in(args.w.0, args.w.1);
    let r = cast_in(args.r.0, args.r.1);
    let b = args.b.map(|(bits, shape)| cast_in(bits, shape));
    let initial_h = args.initial_h.map(|(bits, shape)| cast_in(bits, shape));

    let (y, y_h) = crate::rnn::gru(
        &x,
        &w,
        &r,
        b.as_ref(),
        args.sequence_lens_f32.as_ref(),
        initial_h.as_ref(),
        args.hidden_size,
        args.direction,
        args.linear_before_reset,
        args.activations,
    )?;

    Ok(GruTypedOutput {
        y_bits: cast_out(&y.data),
        y_shape: y.shape,
        y_h_bits: cast_out(&y_h.data),
        y_h_shape: y_h.shape,
    })
}

// ── Public GRU kernel entry-points ───────────────────────────────────────────

/// F16 GRU kernel.
pub(crate) fn gru_f16(args: GruTypedArgs<'_>) -> Result<GruTypedOutput, OnnxError> {
    gru_typed_inner(args, f16_bits_to_tensor, f32_to_f16_bits)
}

/// BF16 GRU kernel.
pub(crate) fn gru_bf16(args: GruTypedArgs<'_>) -> Result<GruTypedOutput, OnnxError> {
    gru_typed_inner(args, bf16_bits_to_tensor, f32_to_bf16_bits)
}
