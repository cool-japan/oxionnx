//! Control flow operators: If, Loop, Scan.
//!
//! These operators execute subgraphs conditionally or iteratively, enabling
//! dynamic control flow within ONNX models.

mod functions;
mod types;

// Trait impl modules — no public items, but pull impl Operator into scope
mod ifop_traits;
mod loopop_traits;
mod scanop_traits;

// Re-export the public operator types
pub use types::{IfOp, LoopOp, ScanOp};
