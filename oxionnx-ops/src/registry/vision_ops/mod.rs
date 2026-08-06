//! Operator trait implementations for the classic CNN / vision operators that
//! the ONNX model zoo still relies on.
//!
//! * [`LRNOp`] — AlexNet / CaffeNet / GoogLeNet local response normalization.
//! * [`LpPoolOp`] / [`GlobalLpPoolOp`] — p-norm pooling.
//! * [`MaxUnpoolOp`] — SegNet-style decoders.
//! * [`MaxRoiPoolOp`] — Fast R-CNN region pooling.
//! * [`UpsampleOp`] — deprecated since opset 10 but present in essentially
//!   every opset ≤ 9 detection / segmentation export.

mod lp_pool;
mod lrn;
mod max_roi_pool;
mod max_unpool;
mod upsample;

pub use lp_pool::{GlobalLpPoolOp, LpPoolOp};
pub use lrn::LRNOp;
pub use max_roi_pool::MaxRoiPoolOp;
pub use max_unpool::MaxUnpoolOp;
pub use upsample::UpsampleOp;
