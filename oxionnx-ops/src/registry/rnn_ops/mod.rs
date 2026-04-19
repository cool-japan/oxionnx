//! Operator trait implementations for RNN, attention, and spatial ops.

mod attention;
mod gru;
mod lstm;
mod spatial;

pub use attention::{AttentionOp, MultiHeadAttentionOp};
pub use gru::GRUOp;
pub use lstm::LSTMOp;
pub use spatial::{GridSampleOp, RoiAlignOp, RotaryEmbeddingOp};
