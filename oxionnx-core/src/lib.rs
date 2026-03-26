//! oxionnx-core — Core types for the oxionnx ONNX inference engine.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod dtype;
pub mod error;
pub mod graph;
pub mod operator;
pub mod tensor;

pub use dtype::{promote, DType, TensorStorage, TypedTensor};
pub use error::OnnxError;
pub use graph::{Attributes, Graph, Node, OpKind};
pub use operator::{OpContext, Operator, OperatorRegistry};
pub use tensor::{
    compute_strides, convert_layout, nchw_to_nhwc, nhwc_to_nchw, BroadcastIter, Tensor,
    TensorLayout,
};
