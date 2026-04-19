//! Operator trait implementations for convolution and pooling operations.

mod conv;
mod pad;
mod pooling;
mod resize;

pub use conv::{ConvOp, ConvTransposeOp};
pub use pad::PadOp;
pub use pooling::{AveragePoolOp, GlobalAveragePoolOp, GlobalMaxPoolOp, MaxPoolOp};
pub use resize::ResizeOp;
