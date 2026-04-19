//! oxionnx-core — Core types for the oxionnx ONNX inference engine.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod dtype;
pub mod error;
pub mod graph;
pub mod operator;
pub mod operator_slots;
pub mod operator_typed;
pub mod tensor;

pub use dtype::{promote, DType, TensorStorage, TypedTensor};
pub use error::OnnxError;
pub use graph::{Attributes, Dim, Graph, Node, OpKind, TensorInfo};
pub use operator::{OpContext, Operator, OperatorRegistry, TypedOpContext};
pub use operator_slots::default_into_slots;
pub use operator_typed::default_typed_via_f32;
pub use tensor::{
    compute_strides, convert_layout, nchw_to_nhwc, nhwc_to_nchw, BroadcastIter, Tensor,
    TensorLayout,
};
