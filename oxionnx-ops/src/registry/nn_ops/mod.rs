//! Operator trait implementations for neural network operations.
//!
//! Submodules:
//!
//! - `activations`: typed/core activations (Relu, Sigmoid, Tanh, Gelu,
//!   SiLU, HardSwish, Softplus, Softsign, Mish, Dropout, Erf, Abs, Log, Exp)
//! - `parameterized`: parameterized activations (Clip, Softmax, LogSoftmax,
//!   LeakyRelu, PRelu, HardSigmoid, Celu, Elu, Selu, ThresholdedRelu,
//!   Hardmax, Shrink)
//! - `normalization`: normalization ops (LayerNorm, GroupNorm, BatchNorm,
//!   RmsNorm, InstanceNorm, LpNorm, MeanVarianceNorm)

mod activations;
mod normalization;
mod parameterized;

pub use activations::{
    AbsOp, DropoutOp, ErfOp, ExpOp, GeluOp, HardSwishOp, LogOp, MishOp, ReluOp, SiLUOp, SigmoidOp,
    SoftplusOp, SoftsignOp, TanhOp,
};
pub use normalization::{
    BatchNormOp, GroupNormOp, InstanceNormOp, LayerNormOp, LpNormOp, MeanVarianceNormalizationOp,
    RmsNormOp,
};
pub use parameterized::{
    CeluOp, ClipOp, EluOp, HardSigmoidOp, HardmaxOp, LeakyReluOp, LogSoftmaxOp, PReluOp, SeluOp,
    ShrinkOp, SoftmaxOp, ThresholdedReluOp,
};
