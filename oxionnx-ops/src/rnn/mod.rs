//! LSTM, GRU, and simple RNN kernels (ONNX spec).
//!
//! Supports:
//! - Variable-length sequences via `sequence_lens`
//! - LSTM peephole connections (optional 8th input P)
//! - The full ONNX `activations` list with `activation_alpha` / `activation_beta`
//!   (an activation outside that list is a typed error, never a silent `Tanh`)
//! - `clip` — clamping of every activation input to `[-clip, +clip]`
//! - `layout` — 0 = `[seq, batch, …]` (default), 1 = `[batch, seq, …]`
//! - Forward, reverse, and bidirectional modes
//!
//! The `*_ext` entry points take an [`RnnExtras`] carrying the optional
//! attributes; the plain `lstm` / `gru` / `simple_rnn` functions keep their
//! historical signatures and use the defaults.

mod common;
mod gru;
mod layout;
mod lstm;
mod simple_rnn;

pub use common::RnnExtras;
pub use gru::{gru, gru_ext};
pub use lstm::{lstm, lstm_ext};
pub use simple_rnn::{simple_rnn, simple_rnn_ext};

pub(crate) use gru::{gru_into, GruOutputSlots};
pub(crate) use lstm::{lstm_into_ext, LstmOutputSlots};

#[cfg(test)]
#[allow(clippy::identity_op, clippy::needless_range_loop)]
mod tests;
