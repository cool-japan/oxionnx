//! Conv-related fusion passes:
//! - Conv + BatchNorm folding (weight baking)
//! - Conv + Relu / Clip activation fusion
//! - Conv + Clip(0,6) → Conv with ReLU6 activation
//! - Conv + Add + ReLU fusion (ResNet residual block pattern)
//! - Standalone BatchNorm folding (Mul + Add replacement)

mod add_relu;
mod batchnorm;
mod relu;
mod relu6;

pub use add_relu::fuse_conv_add_relu;
pub use batchnorm::{fold_batch_norm_inference, fuse_conv_batchnorm};
pub use relu::fuse_conv_relu;
pub use relu6::fuse_conv_clip_to_conv_relu6;

#[cfg(test)]
mod tests;
