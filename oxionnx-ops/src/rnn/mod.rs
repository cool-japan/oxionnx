//! LSTM, GRU, and simple RNN kernels (ONNX spec).
//!
//! Supports:
//! - Variable-length sequences via `sequence_lens`
//! - LSTM peephole connections (optional 8th input P)
//! - Per-gate activation functions (ONNX `activations` attribute)
//! - Forward, reverse, and bidirectional modes

mod common;
mod gru;
mod lstm;
mod simple_rnn;

pub use gru::gru;
pub use lstm::lstm;
pub use simple_rnn::simple_rnn;

pub(crate) use gru::{gru_into, GruOutputSlots};
pub(crate) use lstm::{lstm_into, LstmOutputSlots};

#[cfg(test)]
#[allow(clippy::identity_op, clippy::needless_range_loop)]
mod tests;
