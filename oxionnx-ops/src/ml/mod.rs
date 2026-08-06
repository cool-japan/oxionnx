//! ONNX-ML domain operator implementations.
//!
//! Covers LinearClassifier, LinearRegressor, Normalizer, Scaler, LabelEncoder,
//! TreeEnsembleClassifier, TreeEnsembleRegressor, SVMClassifier, and SVMRegressor.

mod label_encoder;
mod linear;
mod normalizer;
mod post_transform;
mod scaler;
mod shape;
mod string_normalizer;
mod tfidf;

#[cfg(test)]
mod tests;

pub(crate) use shape::batch_dims;

pub use label_encoder::label_encoder;
pub use linear::{linear_classifier, linear_regressor};
pub use normalizer::normalizer;
pub use post_transform::{apply_post_transform, PostTransform};
pub use scaler::scaler;
pub use string_normalizer::string_normalizer;
pub use tfidf::tfidf_vectorizer;
