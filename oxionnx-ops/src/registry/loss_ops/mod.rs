//! `NegativeLogLikelihoodLoss` and `SoftmaxCrossEntropyLoss` -- the ONNX
//! training-capable loss operators, both built on the same reduction core
//! (this module's private `nll_core` submodule).

mod nll_core;
mod nll_loss;
mod softmax_cross_entropy_loss;

pub use nll_loss::NegativeLogLikelihoodLossOp;
pub use softmax_cross_entropy_loss::SoftmaxCrossEntropyLossOp;
