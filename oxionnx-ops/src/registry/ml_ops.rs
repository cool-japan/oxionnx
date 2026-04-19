//! Operator trait implementations for ONNX-ML domain operators.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

pub struct LinearClassifierOp;
impl Operator for LinearClassifierOp {
    fn op_type(&self) -> &str {
        "LinearClassifier"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::linear_classifier(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct LinearRegressorOp;
impl Operator for LinearRegressorOp {
    fn op_type(&self) -> &str {
        "LinearRegressor"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::linear_regressor(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct NormalizerOp;
impl Operator for NormalizerOp {
    fn op_type(&self) -> &str {
        "Normalizer"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::normalizer(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct ScalerOp;
impl Operator for ScalerOp {
    fn op_type(&self) -> &str {
        "Scaler"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::scaler(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct LabelEncoderOp;
impl Operator for LabelEncoderOp {
    fn op_type(&self) -> &str {
        "LabelEncoder"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::label_encoder(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct TreeEnsembleClassifierOp;
impl Operator for TreeEnsembleClassifierOp {
    fn op_type(&self) -> &str {
        "TreeEnsembleClassifier"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml_tree::tree_ensemble_classifier(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct TreeEnsembleRegressorOp;
impl Operator for TreeEnsembleRegressorOp {
    fn op_type(&self) -> &str {
        "TreeEnsembleRegressor"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml_tree::tree_ensemble_regressor(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct SVMClassifierOp;
impl Operator for SVMClassifierOp {
    fn op_type(&self) -> &str {
        "SVMClassifier"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml_svm::svm_classifier(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct SVMRegressorOp;
impl Operator for SVMRegressorOp {
    fn op_type(&self) -> &str {
        "SVMRegressor"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml_svm::svm_regressor(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct TfIdfVectorizerOp;
impl Operator for TfIdfVectorizerOp {
    fn op_type(&self) -> &str {
        "TfIdfVectorizer"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::tfidf_vectorizer(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

pub struct StringNormalizerOp;
impl Operator for StringNormalizerOp {
    fn op_type(&self) -> &str {
        "StringNormalizer"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        crate::ml::string_normalizer(ctx)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
