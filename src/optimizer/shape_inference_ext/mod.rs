//! Extended shape inference for additional ONNX operators.
//!
//! Covers comparison/logic, activations, normalization, pooling, reduce,
//! construction, indexing, quantization, RNN, and advanced ops.

mod advanced;
mod construction;
mod elementwise;
mod indexing;
mod reduce;
mod spatial;

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

use crate::optimizer::shape_inference::get_input_shape;

use advanced::{
    infer_conv_transpose_shape, infer_einsum_shape, infer_gru_shape, infer_linear_classifier_shape,
    infer_linear_regressor_shape, infer_lstm_shape,
};
use construction::{
    infer_constant_of_shape, infer_expand_shape, infer_global_pool_shape, infer_pad_shape,
    infer_pool_shape, infer_resize_shape, infer_tile_shape,
};
use elementwise::infer_variadic_broadcast;
use indexing::{infer_gather_nd_shape, infer_onehot_shape, infer_topk_shape};
use reduce::{infer_arg_reduce_shape, infer_reduce_shape};
use spatial::{
    infer_depth_to_space_shape, infer_grid_sample_shape, infer_roi_align_shape,
    infer_space_to_depth_shape,
};

/// Try to infer output shapes for operators not handled by the core
/// shape inference module. Returns `None` if the op is unrecognized
/// or if required input shapes are unavailable.
pub(crate) fn infer_ext_node_shapes(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    match node.op {
        // ── Unary element-wise (same shape as input[0]) ─────────────
        OpKind::Reciprocal
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::Tan
        | OpKind::Asin
        | OpKind::Acos
        | OpKind::Atan
        | OpKind::Sinh
        | OpKind::Cosh
        | OpKind::Asinh
        | OpKind::Acosh
        | OpKind::Atanh
        | OpKind::Clip => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── Activation ops (same shape as input[0]) ─────────────────
        OpKind::LogSoftmax
        | OpKind::Softplus
        | OpKind::Softsign
        | OpKind::Mish
        | OpKind::Celu
        | OpKind::Elu
        | OpKind::Selu
        | OpKind::ThresholdedRelu
        | OpKind::LeakyRelu
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::BitwiseNot
        | OpKind::Hardmax
        | OpKind::Shrink => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // PRelu: broadcast of input and slope
        OpKind::PRelu => {
            let a = get_input_shape(node, 0, known)?;
            let b = get_input_shape(node, 1, known)?;
            let out = Tensor::broadcast_shape(&a, &b).ok()?;
            Some(vec![out])
        }

        // ── Normalization ops (same shape as input[0]) ──────────────
        OpKind::InstanceNorm | OpKind::LpNorm | OpKind::MeanVarianceNormalization => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // Dropout: two outputs - output (same shape), mask (same shape)
        OpKind::Dropout => {
            let shape = get_input_shape(node, 0, known)?;
            let mask = shape.clone();
            Some(vec![shape, mask])
        }

        // ── Binary comparison ops (broadcast, output bool same shape) ──
        OpKind::Equal
        | OpKind::Greater
        | OpKind::GreaterOrEqual
        | OpKind::Less
        | OpKind::LessOrEqual => {
            let a = get_input_shape(node, 0, known)?;
            let b = get_input_shape(node, 1, known)?;
            let out = Tensor::broadcast_shape(&a, &b).ok()?;
            Some(vec![out])
        }

        // ── Logic ops (broadcast for binary, same shape for unary) ──
        OpKind::And | OpKind::Or | OpKind::Xor => {
            let a = get_input_shape(node, 0, known)?;
            let b = get_input_shape(node, 1, known)?;
            let out = Tensor::broadcast_shape(&a, &b).ok()?;
            Some(vec![out])
        }

        OpKind::Not | OpKind::IsInf | OpKind::IsNaN => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── Binary element-wise: Mod, BitShift, Bitwise ops ────────
        OpKind::Mod
        | OpKind::BitShift
        | OpKind::BitwiseAnd
        | OpKind::BitwiseOr
        | OpKind::BitwiseXor => {
            let a = get_input_shape(node, 0, known)?;
            let b = get_input_shape(node, 1, known)?;
            let out = Tensor::broadcast_shape(&a, &b).ok()?;
            Some(vec![out])
        }

        // ── Where: broadcast of condition, X, Y ─────────────────────
        OpKind::Where => {
            let cond = get_input_shape(node, 0, known)?;
            let x = get_input_shape(node, 1, known)?;
            let y = get_input_shape(node, 2, known)?;
            let tmp = Tensor::broadcast_shape(&cond, &x).ok()?;
            let out = Tensor::broadcast_shape(&tmp, &y).ok()?;
            Some(vec![out])
        }

        // ── Variadic ops (broadcast of all inputs) ──────────────────
        OpKind::VariadicMin | OpKind::VariadicMax | OpKind::VariadicMean | OpKind::VariadicSum => {
            infer_variadic_broadcast(node, known)
        }

        // ── Reduce ops ──────────────────────────────────────────────
        OpKind::ReduceMean
        | OpKind::ReduceSum
        | OpKind::ReduceMax
        | OpKind::ReduceMin
        | OpKind::ReduceProd
        | OpKind::ReduceL1
        | OpKind::ReduceL2
        | OpKind::ReduceLogSum
        | OpKind::ReduceLogSumExp
        | OpKind::ReduceSumSquare => infer_reduce_shape(node, known),

        // ArgMax, ArgMin: reduce one axis to 1 (keepdims) or remove it
        OpKind::ArgMax | OpKind::ArgMin => infer_arg_reduce_shape(node, known),

        // ── Construction ops ────────────────────────────────────────
        OpKind::ConstantOfShape => infer_constant_of_shape(node, known, weights),

        OpKind::EyeLike | OpKind::Trilu => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── Pooling ops ─────────────────────────────────────────────
        OpKind::GlobalAveragePool | OpKind::GlobalMaxPool => infer_global_pool_shape(node, known),

        OpKind::AveragePool | OpKind::MaxPool => infer_pool_shape(node, known),

        // ── Shape ops ───────────────────────────────────────────────
        OpKind::Size => Some(vec![vec![]]), // scalar output

        OpKind::Shape => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![vec![shape.len()]])
        }

        OpKind::Constant => {
            // If tensor attribute is present, use its shape
            if let Some(t) = node.attrs.tensors.get("value") {
                Some(vec![t.shape.clone()])
            } else {
                // Scalar constant
                Some(vec![vec![]])
            }
        }

        OpKind::Expand => infer_expand_shape(node, known, weights),

        OpKind::Tile => infer_tile_shape(node, known, weights),

        OpKind::Pad => infer_pad_shape(node, known, weights),

        OpKind::Resize => {
            // Resize shape depends on scales or sizes input - complex
            // Best effort: if sizes input is available as constant, use it
            infer_resize_shape(node, known, weights)
        }

        OpKind::DepthToSpace => infer_depth_to_space_shape(node, known),

        OpKind::SpaceToDepth => infer_space_to_depth_shape(node, known),

        OpKind::ReverseSequence => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── Indexing ops ────────────────────────────────────────────
        OpKind::GatherElements => {
            // Output shape = indices shape
            let indices_shape = get_input_shape(node, 1, known)?;
            Some(vec![indices_shape])
        }

        OpKind::GatherND => infer_gather_nd_shape(node, known),

        OpKind::ScatterElements | OpKind::ScatterND => {
            // Output shape = data shape (input[0])
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        OpKind::OneHot => infer_onehot_shape(node, known, weights),

        OpKind::NonZero => {
            // Output is [rank, num_nonzero] - num_nonzero is data-dependent
            None
        }

        OpKind::Compress | OpKind::Unique => {
            // Data-dependent output shapes
            None
        }

        // ── Quantization ops ────────────────────────────────────────
        OpKind::QuantizeLinear | OpKind::DequantizeLinear => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── CumSum: same shape as input ─────────────────────────────
        OpKind::CumSum => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── Range: output length depends on start/limit/delta values ─
        OpKind::Range => {
            let start_name = node.inputs.first()?;
            let limit_name = node.inputs.get(1)?;
            let delta_name = node.inputs.get(2)?;
            // Only resolvable when all 3 inputs are constant initializers
            let start = *weights.get(start_name)?.data.first()?;
            let limit = *weights.get(limit_name)?.data.first()?;
            let delta = *weights.get(delta_name)?.data.first()?;
            if delta == 0.0 {
                return None;
            }
            let n = ((limit - start) / delta).ceil().max(0.0) as usize;
            Some(vec![vec![n]])
        }

        // ── TopK: two outputs, both with reduced axis ───────────────
        OpKind::TopK => infer_topk_shape(node, known, weights),

        // ── ConvTranspose ───────────────────────────────────────────
        OpKind::ConvTranspose => infer_conv_transpose_shape(node, known),

        // ── Einsum ──────────────────────────────────────────────────
        OpKind::Einsum => infer_einsum_shape(node, known),

        // ── NonMaxSuppression ───────────────────────────────────────
        OpKind::NonMaxSuppression => {
            // Output: [num_selected_indices, 3] - data-dependent count
            None
        }

        // ── RNN ops ─────────────────────────────────────────────────
        OpKind::LSTM => infer_lstm_shape(node, known),

        OpKind::GRU => infer_gru_shape(node, known),

        // ── Attention ops (same shape as input for standard attention) ──
        OpKind::Attention | OpKind::MultiHeadAttention => {
            // Output shape typically matches query shape
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        OpKind::RotaryEmbedding => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // ── Spatial ops ─────────────────────────────────────────────
        OpKind::GridSample => infer_grid_sample_shape(node, known),

        OpKind::RoiAlign => infer_roi_align_shape(node, known),

        // ── Control flow: cannot infer statically ───────────────────
        OpKind::If | OpKind::Loop | OpKind::Scan => None,

        // ── ML ops ──────────────────────────────────────────────────
        OpKind::LinearClassifier => infer_linear_classifier_shape(node, known),

        OpKind::LinearRegressor => infer_linear_regressor_shape(node, known),

        OpKind::Normalizer | OpKind::Scaler => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // Complex ML ops - skip
        OpKind::LabelEncoder
        | OpKind::TreeEnsembleClassifier
        | OpKind::TreeEnsembleRegressor
        | OpKind::SVMClassifier
        | OpKind::SVMRegressor
        | OpKind::TfIdfVectorizer
        | OpKind::StringNormalizer => None,

        // ── Audio / DSP ops ──────────────────────────────────────────────
        //
        // Window functions (HannWindow, HammingWindow, BlackmanWindow):
        //   Output is 1-D [size] where `size` is the runtime scalar inputs[0].
        //   The value is not available at graph-construction time in the general
        //   case, so we cannot compute a concrete static shape here.
        OpKind::HannWindow | OpKind::HammingWindow | OpKind::BlackmanWindow => {
            let size_name = node.inputs.first()?;
            if size_name.is_empty() {
                return None;
            }
            let size_t = weights.get(size_name)?;
            let size = *size_t.data.first()? as usize;
            Some(vec![vec![size]])
        }

        // DFT:
        //   Input shape is [B, L] or [B, L, 1|2]; output is [B, out_len, 2].
        //   `out_len` = L when onesided=0, or L/2+1 when onesided=1.
        //   When inputs[1] (optional dft_length override) is absent we know the
        //   DFT length equals the signal length L and can produce a static shape.
        //   If inputs[1] is present as a weight constant we can resolve that too,
        //   but the common case is that it's absent.
        OpKind::DFT => {
            let in_shape = get_input_shape(node, 0, known)?;
            if in_shape.len() < 2 {
                return None;
            }
            let batch = in_shape[0];
            let signal_len = in_shape[1];
            // Only infer when the optional dft_length input is absent.
            // (If it's present we'd need to read the weight tensor value.)
            let has_dft_length_input = node.inputs.get(1).is_some_and(|s| !s.is_empty());
            if has_dft_length_input {
                // Cannot determine DFT length statically without the tensor value.
                return None;
            }
            let n = signal_len;
            let inverse = node.attrs.i("inverse", 0) != 0;
            let onesided = if inverse {
                false // ONNX spec: onesided is ignored for inverse DFT
            } else {
                node.attrs.i("onesided", 0) != 0
            };
            let out_len = if onesided { n / 2 + 1 } else { n };
            Some(vec![vec![batch, out_len, 2]])
        }

        // STFT:
        //   Output shape is [B, n_frames, n_dft, 2].
        //   n_frames = (T - frame_length) / frame_step + 1.
        //   n_dft    = frame_length/2+1 (onesided=1) or frame_length (onesided=0).
        //   frame_step comes from inputs[1] (runtime scalar) and frame_length
        //   from inputs[3] (runtime scalar) — these are not statically known.
        //   Return None; shape is fully runtime-determined.
        OpKind::STFT => {
            // Inputs: signal[0], frame_step[1], window[2] (optional), frame_length[3] (optional)
            // Output: [batch, n_frames, n_dft/2+1 or n_dft, 2]
            let signal_shape = get_input_shape(node, 0, known)?;
            if signal_shape.len() < 2 {
                return None;
            }
            let batch = signal_shape[0];
            let t_len = signal_shape[1];
            let frame_step_name = node.inputs.get(1)?;
            let frame_step = *weights.get(frame_step_name)?.data.first()? as usize;
            if frame_step == 0 {
                return None;
            }
            // Resolve frame_length: prefer explicit input[3], then window shape from input[2]
            let frame_length: usize = {
                let fl_name = node.inputs.get(3).map(|s| s.as_str()).unwrap_or("");
                if !fl_name.is_empty() {
                    if let Some(t) = weights.get(fl_name) {
                        *t.data.first()? as usize
                    } else {
                        return None;
                    }
                } else {
                    let w_name = node.inputs.get(2).map(|s| s.as_str()).unwrap_or("");
                    if !w_name.is_empty() {
                        get_input_shape(node, 2, known)?.first().copied()?
                    } else {
                        return None;
                    }
                }
            };
            if frame_length == 0 || t_len < frame_length {
                return None;
            }
            let n_frames = (t_len - frame_length) / frame_step + 1;
            let onesided = node.attrs.i("onesided", 1) != 0;
            let n_dft = if onesided {
                frame_length / 2 + 1
            } else {
                frame_length
            };
            Some(vec![vec![batch, n_frames, n_dft, 2]])
        }

        // MelWeightMatrix:
        //   Output shape [num_spectrogram_bins, num_mel_bins].
        //   All five inputs are runtime scalar tensors; nothing is known statically.
        OpKind::MelWeightMatrix => {
            // Inputs: num_mel_bins[0], dft_length[1], ...
            // Output: [dft_length/2+1, num_mel_bins]
            let num_mel_name = node.inputs.first()?;
            let dft_len_name = node.inputs.get(1)?;
            let num_mel_bins = *weights.get(num_mel_name)?.data.first()? as usize;
            let dft_length = *weights.get(dft_len_name)?.data.first()? as usize;
            Some(vec![vec![dft_length / 2 + 1, num_mel_bins]])
        }

        // Bernoulli:
        //   Output shape = input shape (direct pass-through). This IS static.
        OpKind::Bernoulli => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests;
