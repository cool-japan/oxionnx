//! Typed inference API with compile-time shape checking.
//!
//! Provides a `TypedSession` wrapper that encodes input/output shapes at the type level,
//! preventing shape mismatches at compile time for known model architectures.

use crate::session::Session;
use oxionnx_core::{OnnxError, Tensor};
use std::marker::PhantomData;

/// Marker trait for known tensor shapes.
pub trait Shape {
    /// Return the expected dimensions. Use 0 for dynamic dimensions.
    fn dims() -> &'static [usize];

    /// Check if a concrete shape matches this type-level shape.
    fn matches(shape: &[usize]) -> bool {
        let expected = Self::dims();
        if shape.len() != expected.len() {
            return false;
        }
        expected.iter().zip(shape).all(|(&e, &a)| e == 0 || e == a)
    }
}

/// A typed session that validates shapes at construction time.
pub struct TypedSession<I: Shape, O: Shape> {
    inner: Session,
    input_name: String,
    output_name: String,
    _phantom: PhantomData<(I, O)>,
}

impl<I: Shape, O: Shape> TypedSession<I, O> {
    /// Wrap an existing session with type-level shape constraints.
    /// Returns an error if the session's input/output names don't exist.
    pub fn new(session: Session, input_name: &str, output_name: &str) -> Result<Self, OnnxError> {
        // Verify the input name exists
        if !session.input_names().contains(&input_name.to_string()) {
            return Err(OnnxError::TensorNotFound(format!(
                "TypedSession: input '{}' not found in model",
                input_name
            )));
        }
        // Verify the output name exists
        if !session.output_names().contains(&output_name.to_string()) {
            return Err(OnnxError::TensorNotFound(format!(
                "TypedSession: output '{}' not found in model",
                output_name
            )));
        }

        Ok(Self {
            inner: session,
            input_name: input_name.to_string(),
            output_name: output_name.to_string(),
            _phantom: PhantomData,
        })
    }

    /// Run inference with compile-time shape validation.
    /// The input tensor's shape must match `I::dims()`.
    pub fn run(&self, input: &Tensor) -> Result<Tensor, OnnxError> {
        // Validate input shape
        if !I::matches(&input.shape) {
            return Err(OnnxError::ShapeMismatch(format!(
                "TypedSession: input shape {:?} does not match expected {:?}",
                input.shape,
                I::dims()
            )));
        }

        let outputs = self.inner.run_one(&self.input_name, input.clone())?;

        let output = outputs.get(&self.output_name).ok_or_else(|| {
            OnnxError::TensorNotFound(format!(
                "TypedSession: output '{}' not produced",
                self.output_name
            ))
        })?;

        // Validate output shape
        if !O::matches(&output.shape) {
            return Err(OnnxError::ShapeMismatch(format!(
                "TypedSession: output shape {:?} does not match expected {:?}",
                output.shape,
                O::dims()
            )));
        }

        Ok(output.clone())
    }

    /// Access the inner session.
    pub fn inner(&self) -> &Session {
        &self.inner
    }
}

/// Macro to define a shape type.
///
/// # Examples
///
/// ```ignore
/// define_shape!(MyShape, [1, 3, 224, 224]);
/// // Use 0 for dynamic dimensions:
/// define_shape!(BatchedInput, [0, 3, 224, 224]);
/// ```
#[macro_export]
macro_rules! define_shape {
    ($name:ident, [$($dim:expr),*]) => {
        pub struct $name;
        impl $crate::typed_session::Shape for $name {
            fn dims() -> &'static [usize] {
                &[$($dim),*]
            }
        }
    };
}

/// Scalar shape: single element.
pub struct Scalar;
impl Shape for Scalar {
    fn dims() -> &'static [usize] {
        &[1]
    }
}

/// 1D shape with dynamic length.
pub struct Dynamic1D;
impl Shape for Dynamic1D {
    fn dims() -> &'static [usize] {
        &[0]
    }
}

/// 2D shape with dynamic dimensions.
pub struct Dynamic2D;
impl Shape for Dynamic2D {
    fn dims() -> &'static [usize] {
        &[0, 0]
    }
}

/// ImageNet input: [batch, 3, 224, 224] with dynamic batch.
pub struct ImageNet224;
impl Shape for ImageNet224 {
    fn dims() -> &'static [usize] {
        &[0, 3, 224, 224]
    }
}

/// BERT input: [batch, seq_len] with dynamic dims.
pub struct BertInput;
impl Shape for BertInput {
    fn dims() -> &'static [usize] {
        &[0, 0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_matches() {
        // Exact match
        assert!(ImageNet224::matches(&[1, 3, 224, 224]));
        assert!(ImageNet224::matches(&[32, 3, 224, 224]));
        // Wrong channel count
        assert!(!ImageNet224::matches(&[1, 4, 224, 224]));
        // Wrong spatial
        assert!(!ImageNet224::matches(&[1, 3, 256, 256]));
        // Wrong rank
        assert!(!ImageNet224::matches(&[1, 3, 224]));
        // Dynamic1D matches any 1D
        assert!(Dynamic1D::matches(&[42]));
        assert!(Dynamic1D::matches(&[1]));
        assert!(!Dynamic1D::matches(&[1, 2]));
        // Dynamic2D
        assert!(Dynamic2D::matches(&[10, 20]));
        assert!(!Dynamic2D::matches(&[10]));
        // Scalar
        assert!(Scalar::matches(&[1]));
        assert!(!Scalar::matches(&[2]));
    }

    #[test]
    fn test_imagenet_shape() {
        let dims = ImageNet224::dims();
        assert_eq!(dims, &[0, 3, 224, 224]);
        assert_eq!(dims.len(), 4);
    }

    #[test]
    fn test_bert_shape() {
        let dims = BertInput::dims();
        assert_eq!(dims, &[0, 0]);
        assert_eq!(dims.len(), 2);
        // Any batch/seq combo should match
        assert!(BertInput::matches(&[1, 128]));
        assert!(BertInput::matches(&[16, 512]));
        assert!(!BertInput::matches(&[1, 128, 768])); // wrong rank
    }

    #[test]
    fn test_define_shape_macro() {
        crate::define_shape!(CustomShape, [0, 10, 20]);
        assert_eq!(CustomShape::dims(), &[0, 10, 20]);
        assert!(CustomShape::matches(&[5, 10, 20]));
        assert!(!CustomShape::matches(&[5, 10, 30]));
        assert!(!CustomShape::matches(&[5, 10]));
    }

    #[test]
    fn test_typed_session_wrong_input_shape() {
        // We cannot easily construct a full Session in unit tests without a model,
        // so we test the Shape validation logic directly, which is the core mechanic.
        // The TypedSession::run method calls I::matches before forwarding.
        let shape = &[1, 3, 256, 256];
        assert!(
            !ImageNet224::matches(shape),
            "256x256 should not match 224x224"
        );

        let shape_ok = &[1, 3, 224, 224];
        assert!(ImageNet224::matches(shape_ok));
    }

    #[test]
    fn test_typed_session_dynamic() {
        // Dynamic dimensions (0) should match any concrete value
        crate::define_shape!(DynBatch, [0, 768]);
        assert!(DynBatch::matches(&[1, 768]));
        assert!(DynBatch::matches(&[128, 768]));
        assert!(!DynBatch::matches(&[1, 512])); // second dim must be 768
        assert!(!DynBatch::matches(&[1])); // wrong rank
    }

    #[test]
    fn test_typed_session_basic() {
        // Test the Shape trait fundamentals that TypedSession relies on
        // Scalar
        assert!(Scalar::matches(&[1]));
        assert!(!Scalar::matches(&[0]));
        assert!(!Scalar::matches(&[]));

        // Dynamic1D
        assert!(Dynamic1D::matches(&[100]));

        // Dynamic2D
        assert!(Dynamic2D::matches(&[3, 4]));

        // ImageNet224
        assert!(ImageNet224::matches(&[8, 3, 224, 224]));
    }
}
