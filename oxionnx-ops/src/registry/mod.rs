//! Operator registry: maps ONNX op_type strings to `Operator` trait objects.
//!
//! Each sub-module implements the `Operator` trait for a group of related ops.
//! `default_registry()` creates a registry pre-populated with all supported ops.

pub mod cast_like;
pub mod center_crop_pad;
pub mod col2im;
pub mod conv_ops;
pub mod det;
pub mod indexing_ops;
pub mod loss_ops;
pub mod math_ops;
pub mod misc_ops;
pub mod ml_ops;
pub mod nn_ops;
pub mod oxi_instance_norm;
pub mod quant_ops;
pub mod random_ops;
pub mod rnn_ops;
pub mod shape_ops;
pub mod vision_ops;

use oxionnx_core::OperatorRegistry;

/// Build the default operator registry containing all supported ONNX operators.
pub fn default_registry() -> OperatorRegistry {
    let mut r = OperatorRegistry::new();

    // ── Math ops ────────────────────────────────────────────────────────────
    r.register(Box::new(math_ops::MatMulOp));
    r.register(Box::new(math_ops::GemmOp));
    r.register(Box::new(math_ops::AddOp));
    r.register(Box::new(math_ops::SubOp));
    r.register(Box::new(math_ops::MulOp));
    r.register(Box::new(math_ops::DivOp));
    r.register(Box::new(math_ops::PowOp));
    r.register(Box::new(math_ops::SqrtOp));
    r.register(Box::new(math_ops::ReciprocalOp));
    r.register(Box::new(math_ops::NegOp));
    r.register(Box::new(math_ops::CeilOp));
    r.register(Box::new(math_ops::FloorOp));
    r.register(Box::new(math_ops::RoundOp));
    r.register(Box::new(math_ops::SignOp));
    r.register(Box::new(math_ops::SinOp));
    r.register(Box::new(math_ops::CosOp));
    r.register(Box::new(math_ops::TanOp));
    r.register(Box::new(math_ops::AsinOp));
    r.register(Box::new(math_ops::AcosOp));
    r.register(Box::new(math_ops::AtanOp));
    r.register(Box::new(math_ops::SinhOp));
    r.register(Box::new(math_ops::CoshOp));
    r.register(Box::new(math_ops::AsinhOp));
    r.register(Box::new(math_ops::AcoshOp));
    r.register(Box::new(math_ops::AtanhOp));
    r.register(Box::new(math_ops::ReduceMeanOp));
    r.register(Box::new(math_ops::ReduceSumOp));
    r.register(Box::new(math_ops::ReduceMaxOp));
    r.register(Box::new(math_ops::ReduceMinOp));
    r.register(Box::new(math_ops::ReduceProdOp));
    r.register(Box::new(math_ops::ArgMaxOp));
    r.register(Box::new(math_ops::ArgMinOp));
    r.register(Box::new(math_ops::CumSumOp));
    r.register(Box::new(math_ops::RangeOp));
    r.register(Box::new(math_ops::TopKOp));
    r.register(Box::new(math_ops::ModOp));
    r.register(Box::new(math_ops::BitShiftOp));
    r.register(Box::new(math_ops::VariadicMinOp));
    r.register(Box::new(math_ops::VariadicMaxOp));
    r.register(Box::new(math_ops::VariadicMeanOp));
    r.register(Box::new(math_ops::VariadicSumOp));

    // ── NN ops ──────────────────────────────────────────────────────────────
    r.register(Box::new(nn_ops::ReluOp));
    r.register(Box::new(nn_ops::SigmoidOp));
    r.register(Box::new(nn_ops::TanhOp));
    r.register(Box::new(nn_ops::GeluOp));
    r.register(Box::new(nn_ops::SiLUOp));
    r.register(Box::new(nn_ops::HardSwishOp));
    r.register(Box::new(nn_ops::SoftplusOp));
    r.register(Box::new(nn_ops::SoftsignOp));
    r.register(Box::new(nn_ops::MishOp));
    r.register(Box::new(nn_ops::DropoutOp));
    r.register(Box::new(nn_ops::ErfOp));
    r.register(Box::new(nn_ops::AbsOp));
    r.register(Box::new(nn_ops::LogOp));
    r.register(Box::new(nn_ops::ExpOp));
    r.register(Box::new(nn_ops::ClipOp));
    r.register(Box::new(nn_ops::SoftmaxOp));
    r.register(Box::new(nn_ops::LogSoftmaxOp));
    r.register(Box::new(nn_ops::LayerNormOp));
    r.register(Box::new(nn_ops::GroupNormOp));
    r.register(Box::new(nn_ops::BatchNormOp));
    r.register(Box::new(nn_ops::RmsNormOp));
    r.register(Box::new(nn_ops::LeakyReluOp));
    r.register(Box::new(nn_ops::PReluOp));
    r.register(Box::new(nn_ops::HardSigmoidOp));
    r.register(Box::new(nn_ops::CeluOp));
    r.register(Box::new(nn_ops::EluOp));
    r.register(Box::new(nn_ops::SeluOp));
    r.register(Box::new(nn_ops::ThresholdedReluOp));
    r.register(Box::new(nn_ops::InstanceNormOp));
    r.register(Box::new(nn_ops::LpNormOp));
    r.register(Box::new(nn_ops::MeanVarianceNormalizationOp));

    // ── Conv / Pool ops ─────────────────────────────────────────────────────
    r.register(Box::new(conv_ops::ConvOp));
    r.register(Box::new(conv_ops::ConvTransposeOp));
    r.register(Box::new(conv_ops::MaxPoolOp));
    r.register(Box::new(conv_ops::AveragePoolOp));
    r.register(Box::new(conv_ops::GlobalAveragePoolOp));
    r.register(Box::new(conv_ops::GlobalMaxPoolOp));
    r.register(Box::new(conv_ops::PadOp));
    r.register(Box::new(conv_ops::ResizeOp));

    // ── Shape ops ───────────────────────────────────────────────────────────
    r.register(Box::new(shape_ops::ReshapeOp));
    r.register(Box::new(shape_ops::TransposeOp));
    r.register(Box::new(shape_ops::SqueezeOp));
    r.register(Box::new(shape_ops::UnsqueezeOp));
    r.register(Box::new(shape_ops::FlattenOp));
    r.register(Box::new(shape_ops::ConcatOp));
    r.register(Box::new(shape_ops::SliceOp));
    r.register(Box::new(shape_ops::ExpandOp));
    r.register(Box::new(shape_ops::SplitOp));
    r.register(Box::new(shape_ops::TileOp));
    r.register(Box::new(shape_ops::DepthToSpaceOp));
    r.register(Box::new(shape_ops::SpaceToDepthOp));
    r.register(Box::new(shape_ops::ReverseSequenceOp));

    // ── Indexing ops ────────────────────────────────────────────────────────
    r.register(Box::new(indexing_ops::GatherOp));
    r.register(Box::new(indexing_ops::GatherElementsOp));
    r.register(Box::new(indexing_ops::GatherNDOp));
    r.register(Box::new(indexing_ops::WhereOp));
    r.register(Box::new(indexing_ops::ScatterElementsOp));
    r.register(Box::new(indexing_ops::ScatterNDOp));
    r.register(Box::new(indexing_ops::QuantizeLinearOp));
    r.register(Box::new(indexing_ops::DequantizeLinearOp));
    r.register(Box::new(indexing_ops::OneHotOp));
    r.register(Box::new(indexing_ops::CompressOp));
    r.register(Box::new(indexing_ops::UniqueOp));

    // ── Misc / comparison / construction ops ────────────────────────────────
    r.register(Box::new(misc_ops::EqualOp));
    r.register(Box::new(misc_ops::GreaterOp));
    r.register(Box::new(misc_ops::GreaterOrEqualOp));
    r.register(Box::new(misc_ops::LessOp));
    r.register(Box::new(misc_ops::LessOrEqualOp));
    r.register(Box::new(misc_ops::AndOp));
    r.register(Box::new(misc_ops::OrOp));
    r.register(Box::new(misc_ops::XorOp));
    r.register(Box::new(misc_ops::NotOp));
    r.register(Box::new(misc_ops::IsInfOp));
    r.register(Box::new(misc_ops::IsNaNOp));
    r.register(Box::new(misc_ops::NonZeroOp));
    r.register(Box::new(misc_ops::ConstantOfShapeOp));
    r.register(Box::new(misc_ops::EyeLikeOp));
    r.register(Box::new(misc_ops::TriluOp));
    r.register(Box::new(misc_ops::IdentityOp));
    r.register(Box::new(misc_ops::CastOp));
    r.register(Box::new(misc_ops::ShapeOp));
    r.register(Box::new(misc_ops::ConstantOp));
    r.register(Box::new(misc_ops::EinsumOp));
    r.register(Box::new(misc_ops::NonMaxSuppressionOp));

    // ── RNN / Attention / Spatial ops ──────────────────────────────────────
    r.register(Box::new(rnn_ops::LSTMOp));
    r.register(Box::new(rnn_ops::GRUOp));
    r.register(Box::new(rnn_ops::RNNOp));
    r.register(Box::new(rnn_ops::AttentionOp));
    r.register(Box::new(rnn_ops::MultiHeadAttentionOp));
    r.register(Box::new(rnn_ops::RotaryEmbeddingOp));
    r.register(Box::new(rnn_ops::GridSampleOp));
    r.register(Box::new(rnn_ops::RoiAlignOp));

    // ── Aliases (ONNX names that differ from our canonical op_type) ────────
    r.register_as("LayerNorm", Box::new(nn_ops::LayerNormOp));
    r.register_as("SimplifiedLayerNormalization", Box::new(nn_ops::RmsNormOp));
    r.register_as("RMSNorm", Box::new(nn_ops::RmsNormOp));
    r.register_as("BatchNormalization", Box::new(nn_ops::BatchNormOp));
    r.register_as("GroupNormalization", Box::new(nn_ops::GroupNormOp));
    r.register_as("InstanceNormalization", Box::new(nn_ops::InstanceNormOp));
    r.register_as("LpNormalization", Box::new(nn_ops::LpNormOp));
    r.register_as("Silu", Box::new(nn_ops::SiLUOp));
    r.register_as("CeLU", Box::new(nn_ops::CeluOp));
    r.register_as("Expand", Box::new(shape_ops::ExpandOp));

    // ── Audio / DSP ops ────────────────────────────────────────────────────
    r.register(Box::new(crate::dsp::DFTOp));
    r.register(Box::new(crate::dsp::STFTOp));
    r.register(Box::new(crate::dsp::HannWindowOp));
    r.register(Box::new(crate::dsp::HammingWindowOp));
    r.register(Box::new(crate::dsp::BlackmanWindowOp));
    r.register(Box::new(crate::dsp::MelWeightMatrixOp));
    r.register(Box::new(crate::dsp::BernoulliOp));

    // ── Control flow ops ───────────────────────────────────────────────────
    r.register(Box::new(crate::control_flow::IfOp));
    r.register(Box::new(crate::control_flow::LoopOp));
    r.register(Box::new(crate::control_flow::ScanOp));

    // ── J-phase additions ──────────────────────────────────────────────────
    // Reduce (J-phase additions)
    r.register(Box::new(math_ops::ReduceL1Op));
    r.register(Box::new(math_ops::ReduceL2Op));
    r.register(Box::new(math_ops::ReduceLogSumOp));
    r.register(Box::new(math_ops::ReduceLogSumExpOp));
    r.register(Box::new(math_ops::ReduceSumSquareOp));
    // Bitwise (J-phase additions)
    r.register(Box::new(misc_ops::BitwiseAndOp));
    r.register(Box::new(misc_ops::BitwiseOrOp));
    r.register(Box::new(misc_ops::BitwiseXorOp));
    r.register(Box::new(misc_ops::BitwiseNotOp));
    r.register(Box::new(misc_ops::SizeOp));
    // Utility (J-phase additions)
    r.register(Box::new(nn_ops::HardmaxOp));
    r.register(Box::new(nn_ops::ShrinkOp));

    // ── Quantized (int8/uint8) ops ─────────────────────────────────────────
    // The operator set ONNX Runtime's static quantizer emits (QOperator
    // format) plus the dynamic-quantization entry point. Until now the
    // kernels existed but nothing mapped an op_type onto them, so every
    // quantized .onnx file failed at load with UnsupportedOp.
    r.register(Box::new(quant_ops::QLinearConvOp));
    r.register(Box::new(quant_ops::QLinearMatMulOp));
    r.register(Box::new(quant_ops::MatMulIntegerOp));
    r.register(Box::new(quant_ops::ConvIntegerOp));
    r.register(Box::new(quant_ops::DynamicQuantizeLinearOp));

    // ── Classic CNN / vision ops ───────────────────────────────────────────
    r.register(Box::new(vision_ops::LRNOp));
    r.register(Box::new(vision_ops::LpPoolOp));
    r.register(Box::new(vision_ops::GlobalLpPoolOp));
    r.register(Box::new(vision_ops::MaxUnpoolOp));
    r.register(Box::new(vision_ops::MaxRoiPoolOp));
    r.register(Box::new(vision_ops::UpsampleOp));
    r.register(Box::new(cast_like::CastLikeOp));

    // ── W2 registry lane B additions ───────────────────────────────────────
    // Linear algebra / tensor reconstruction / cropping.
    r.register(Box::new(det::DetOp));
    r.register(Box::new(col2im::Col2ImOp));
    r.register(Box::new(center_crop_pad::CenterCropPadOp));
    // Random generators + Multinomial (pure-Rust seeded PRNG; see
    // `random_ops::rng` for the non-bitwise-ORT-compatibility note).
    r.register(Box::new(random_ops::RandomNormalOp));
    r.register(Box::new(random_ops::RandomUniformOp));
    r.register(Box::new(random_ops::RandomNormalLikeOp));
    r.register(Box::new(random_ops::RandomUniformLikeOp));
    r.register(Box::new(random_ops::MultinomialOp));
    // Loss ops.
    r.register(Box::new(loss_ops::NegativeLogLikelihoodLossOp));
    r.register(Box::new(loss_ops::SoftmaxCrossEntropyLossOp));

    // ── Fused ops (optimizer-generated) ────────────────────────────────────
    // `fuse_instance_norm` only rewrites a graph when this lookup succeeds, so
    // registering the kernel here is what turns the pass on for the default
    // registry.
    r.register(Box::new(oxi_instance_norm::OxiInstanceNormOp));

    // ── ML ops ──────────────────────────────────────────────────────────────
    r.register(Box::new(ml_ops::LinearClassifierOp));
    r.register(Box::new(ml_ops::LinearRegressorOp));
    r.register(Box::new(ml_ops::NormalizerOp));
    r.register(Box::new(ml_ops::ScalerOp));
    r.register(Box::new(ml_ops::LabelEncoderOp));
    r.register(Box::new(ml_ops::TreeEnsembleClassifierOp));
    r.register(Box::new(ml_ops::TreeEnsembleRegressorOp));
    r.register(Box::new(ml_ops::SVMClassifierOp));
    r.register(Box::new(ml_ops::SVMRegressorOp));
    r.register(Box::new(ml_ops::TfIdfVectorizerOp));
    r.register(Box::new(ml_ops::StringNormalizerOp));

    // ML domain aliases (ai.onnx.ml.*)
    r.register_as(
        "ai.onnx.ml.LinearClassifier",
        Box::new(ml_ops::LinearClassifierOp),
    );
    r.register_as(
        "ai.onnx.ml.LinearRegressor",
        Box::new(ml_ops::LinearRegressorOp),
    );
    r.register_as("ai.onnx.ml.Normalizer", Box::new(ml_ops::NormalizerOp));
    r.register_as("ai.onnx.ml.Scaler", Box::new(ml_ops::ScalerOp));
    r.register_as("ai.onnx.ml.LabelEncoder", Box::new(ml_ops::LabelEncoderOp));
    r.register_as(
        "ai.onnx.ml.TreeEnsembleClassifier",
        Box::new(ml_ops::TreeEnsembleClassifierOp),
    );
    r.register_as(
        "ai.onnx.ml.TreeEnsembleRegressor",
        Box::new(ml_ops::TreeEnsembleRegressorOp),
    );
    r.register_as(
        "ai.onnx.ml.SVMClassifier",
        Box::new(ml_ops::SVMClassifierOp),
    );
    r.register_as("ai.onnx.ml.SVMRegressor", Box::new(ml_ops::SVMRegressorOp));
    r.register_as(
        "ai.onnx.ml.TfIdfVectorizer",
        Box::new(ml_ops::TfIdfVectorizerOp),
    );
    r.register_as(
        "ai.onnx.ml.StringNormalizer",
        Box::new(ml_ops::StringNormalizerOp),
    );

    r
}
