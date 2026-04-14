//! Extended shape inference for additional ONNX operators.
//!
//! Covers comparison/logic, activations, normalization, pooling, reduce,
//! construction, indexing, quantization, RNN, and advanced ops.

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::shape_inference::get_input_shape;

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

// ═══════════════════════════════════════════════════════════════════
// Helper functions for extended shape inference
// ═══════════════════════════════════════════════════════════════════

/// Broadcast all variadic inputs together.
fn infer_variadic_broadcast(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    if node.inputs.is_empty() {
        return None;
    }
    let mut result = get_input_shape(node, 0, known)?;
    for i in 1..node.inputs.len() {
        let shape = get_input_shape(node, i, known)?;
        result = Tensor::broadcast_shape(&result, &shape).ok()?;
    }
    Some(vec![result])
}

/// Reduce ops: apply axes reduction with keepdims.
fn infer_reduce_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let keepdims = node.attrs.i("keepdims", 1) != 0;

    let axes_attr: Vec<i64> = node.attrs.ints("axes").to_vec();
    let axes: Vec<usize> = if axes_attr.is_empty() {
        // Default: reduce all axes
        (0..input_shape.len()).collect()
    } else {
        axes_attr
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + rank) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    };

    let mut out = Vec::new();
    for (i, &dim) in input_shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }

    Some(vec![out])
}

/// ArgMax/ArgMin: reduce one axis, output index along that axis.
fn infer_arg_reduce_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let keepdims = node.attrs.i("keepdims", 1) != 0;
    let axis_raw = node.attrs.i("axis", 0);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= input_shape.len() {
        return None;
    }

    let mut out = Vec::new();
    for (i, &dim) in input_shape.iter().enumerate() {
        if i == axis {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }

    Some(vec![out])
}

/// ConstantOfShape: output shape from the input tensor's data values.
fn infer_constant_of_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    // The input is a 1-D tensor whose values define the output shape
    let shape_name = node.inputs.first()?;
    if shape_name.is_empty() {
        return None;
    }

    // Try from weights first
    if let Some(shape_tensor) = weights.get(shape_name) {
        let out: Vec<usize> = shape_tensor.data.iter().map(|&v| v as usize).collect();
        return Some(vec![out]);
    }

    // If shape data is known as a shape (e.g., it's a 1-D tensor of known size)
    // we can't determine the actual values without the data
    let _shape = known.get(shape_name)?;
    None
}

/// GlobalAveragePool / GlobalMaxPool: [N, C, ...] -> [N, C, 1, 1, ...]
fn infer_global_pool_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() < 2 {
        return None;
    }

    let mut out = vec![input_shape[0], input_shape[1]];
    out.extend(std::iter::repeat(1).take(input_shape.len() - 2));
    Some(vec![out])
}

/// AveragePool / MaxPool shape inference (similar to Conv).
fn infer_pool_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() < 3 {
        return None;
    }

    let n = input_shape[0];
    let c = input_shape[1];
    let spatial_dims = input_shape.len() - 2;

    let kernel_shape_attr: Vec<i64> = node.attrs.ints("kernel_shape").to_vec();
    if kernel_shape_attr.is_empty() {
        return None;
    }
    let kernel_shape: Vec<usize> = kernel_shape_attr.iter().map(|&k| k as usize).collect();

    if kernel_shape.len() != spatial_dims {
        return None;
    }

    let strides_attr: Vec<i64> = node.attrs.ints("strides").to_vec();
    let strides: Vec<usize> = if strides_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        strides_attr.iter().map(|&s| s as usize).collect()
    };

    let dilations_attr: Vec<i64> = node.attrs.ints("dilations").to_vec();
    let dilations: Vec<usize> = if dilations_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        dilations_attr.iter().map(|&d| d as usize).collect()
    };

    let pads_attr: Vec<i64> = node.attrs.ints("pads").to_vec();
    let pads: Vec<usize> = if pads_attr.is_empty() {
        vec![0; spatial_dims * 2]
    } else {
        pads_attr.iter().map(|&p| p as usize).collect()
    };

    if pads.len() != spatial_dims * 2 {
        return None;
    }

    let ceil_mode = node.attrs.i("ceil_mode", 0) != 0;

    let mut out_shape = vec![n, c];
    for d in 0..spatial_dims {
        let input_dim = input_shape[d + 2];
        let effective_kernel = (kernel_shape[d] - 1) * dilations[d] + 1;
        let padded = input_dim + pads[d] + pads[d + spatial_dims];
        if padded < effective_kernel {
            return None;
        }
        let out_dim = if ceil_mode {
            (padded - effective_kernel).div_ceil(strides[d]) + 1
        } else {
            (padded - effective_kernel) / strides[d] + 1
        };
        out_shape.push(out_dim);
    }

    Some(vec![out_shape])
}

/// Expand: broadcast input to target shape.
fn infer_expand_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let shape_name = node.inputs.get(1)?;
    if shape_name.is_empty() {
        return None;
    }

    if let Some(shape_tensor) = weights.get(shape_name) {
        let target: Vec<usize> = shape_tensor.data.iter().map(|&v| v as usize).collect();
        let out = Tensor::broadcast_shape(&input_shape, &target).ok()?;
        return Some(vec![out]);
    }

    None
}

/// Tile: repeat input along each axis.
fn infer_tile_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let repeats_name = node.inputs.get(1)?;
    if repeats_name.is_empty() {
        return None;
    }

    if let Some(repeats_tensor) = weights.get(repeats_name) {
        let repeats: Vec<usize> = repeats_tensor.data.iter().map(|&v| v as usize).collect();
        if repeats.len() != input_shape.len() {
            return None;
        }
        let out: Vec<usize> = input_shape
            .iter()
            .zip(repeats.iter())
            .map(|(&d, &r)| d * r)
            .collect();
        return Some(vec![out]);
    }

    None
}

/// Pad: add padding to spatial dims.
fn infer_pad_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;

    // pads input: [begin_0, begin_1, ..., end_0, end_1, ...]
    let pads_name = node.inputs.get(1)?;
    if pads_name.is_empty() {
        return None;
    }

    if let Some(pads_tensor) = weights.get(pads_name) {
        let pads: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let rank = input_shape.len();
        if pads.len() != rank * 2 {
            return None;
        }

        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            let padded = input_shape[i] as i64 + pads[i] + pads[i + rank];
            if padded < 0 {
                return None;
            }
            out.push(padded as usize);
        }
        return Some(vec![out]);
    }

    None
}

/// Resize: infer from sizes or scales constant input.
fn infer_resize_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;

    // inputs: X, roi, scales, sizes (opset 11+)
    // Check sizes input first (index 3)
    if let Some(sizes_name) = node.inputs.get(3) {
        if !sizes_name.is_empty() {
            if let Some(sizes_tensor) = weights.get(sizes_name) {
                let out: Vec<usize> = sizes_tensor.data.iter().map(|&v| v as usize).collect();
                if out.len() == input_shape.len() {
                    return Some(vec![out]);
                }
            }
        }
    }

    // Check scales input (index 2)
    if let Some(scales_name) = node.inputs.get(2) {
        if !scales_name.is_empty() {
            if let Some(scales_tensor) = weights.get(scales_name) {
                if scales_tensor.data.len() == input_shape.len() {
                    let out: Vec<usize> = input_shape
                        .iter()
                        .zip(scales_tensor.data.iter())
                        .map(|(&d, &s)| (d as f32 * s) as usize)
                        .collect();
                    return Some(vec![out]);
                }
            }
        }
    }

    None
}

/// DepthToSpace: [N, C, H, W] -> [N, C/(block^2), H*block, W*block]
fn infer_depth_to_space_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() != 4 {
        return None;
    }

    let blocksize = node.attrs.i("blocksize", 0);
    if blocksize <= 0 {
        return None;
    }
    let bs = blocksize as usize;

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    if c % (bs * bs) != 0 {
        return None;
    }

    Some(vec![vec![n, c / (bs * bs), h * bs, w * bs]])
}

/// SpaceToDepth: [N, C, H, W] -> [N, C*(block^2), H/block, W/block]
fn infer_space_to_depth_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() != 4 {
        return None;
    }

    let blocksize = node.attrs.i("blocksize", 0);
    if blocksize <= 0 {
        return None;
    }
    let bs = blocksize as usize;

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    if h % bs != 0 || w % bs != 0 {
        return None;
    }

    Some(vec![vec![n, c * bs * bs, h / bs, w / bs]])
}

/// GatherND shape inference.
fn infer_gather_nd_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let data_shape = get_input_shape(node, 0, known)?;
    let indices_shape = get_input_shape(node, 1, known)?;
    let batch_dims = node.attrs.i("batch_dims", 0) as usize;

    if indices_shape.is_empty() {
        return None;
    }

    let last_idx_dim = *indices_shape.last()?;
    let data_rank = data_shape.len();

    if batch_dims + last_idx_dim > data_rank {
        return None;
    }

    // output shape = indices_shape[:-1] + data_shape[batch_dims + last_idx_dim:]
    let mut out = Vec::new();
    // batch dims from indices
    out.extend_from_slice(&indices_shape[..indices_shape.len() - 1]);
    // remaining data dims
    out.extend_from_slice(&data_shape[batch_dims + last_idx_dim..]);

    Some(vec![out])
}

/// OneHot: indices_shape + [depth]
fn infer_onehot_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let indices_shape = get_input_shape(node, 0, known)?;
    let axis = node.attrs.i("axis", -1);

    // depth is input[1]
    let depth_name = node.inputs.get(1)?;
    if depth_name.is_empty() {
        return None;
    }

    let depth = if let Some(depth_tensor) = weights.get(depth_name) {
        if depth_tensor.data.is_empty() {
            return None;
        }
        depth_tensor.data[0] as usize
    } else {
        return None;
    };

    let rank = indices_shape.len() as i64 + 1;
    let norm_axis = if axis < 0 {
        (axis + rank) as usize
    } else {
        axis as usize
    };

    let mut out = indices_shape;
    if norm_axis > out.len() {
        return None;
    }
    out.insert(norm_axis, depth);

    Some(vec![out])
}

/// TopK: two outputs with the k-dim replaced.
fn infer_topk_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let axis_raw = node.attrs.i("axis", -1);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= input_shape.len() {
        return None;
    }

    // k comes from input[1]
    let k_name = node.inputs.get(1)?;
    if k_name.is_empty() {
        return None;
    }

    let k = if let Some(k_tensor) = weights.get(k_name) {
        if k_tensor.data.is_empty() {
            return None;
        }
        k_tensor.data[0] as usize
    } else {
        return None;
    };

    let mut out = input_shape;
    out[axis] = k;
    // TopK has two outputs: values and indices, both same shape
    Some(vec![out.clone(), out])
}

/// ConvTranspose shape inference.
fn infer_conv_transpose_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let weight_shape = get_input_shape(node, 1, known)?;

    if input_shape.len() < 3 || weight_shape.len() < 3 {
        return None;
    }

    let n = input_shape[0];
    // For ConvTranspose, weight is [C_in, C_out/group, kH, kW, ...]
    let group = node.attrs.i("group", 1) as usize;
    let c_out = weight_shape[1] * group;
    let spatial_dims = input_shape.len() - 2;

    // Check for explicit output_shape attribute
    let output_shape_attr: Vec<i64> = node.attrs.ints("output_shape").to_vec();
    if !output_shape_attr.is_empty() && output_shape_attr.len() == spatial_dims {
        let mut out = vec![n, c_out];
        for &s in &output_shape_attr {
            out.push(s as usize);
        }
        return Some(vec![out]);
    }

    let kernel_shape_attr: Vec<i64> = node.attrs.ints("kernel_shape").to_vec();
    let kernel_shape: Vec<usize> = if kernel_shape_attr.is_empty() {
        weight_shape[2..].to_vec()
    } else {
        kernel_shape_attr.iter().map(|&k| k as usize).collect()
    };

    let strides_attr: Vec<i64> = node.attrs.ints("strides").to_vec();
    let strides: Vec<usize> = if strides_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        strides_attr.iter().map(|&s| s as usize).collect()
    };

    let dilations_attr: Vec<i64> = node.attrs.ints("dilations").to_vec();
    let dilations: Vec<usize> = if dilations_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        dilations_attr.iter().map(|&d| d as usize).collect()
    };

    let pads_attr: Vec<i64> = node.attrs.ints("pads").to_vec();
    let pads: Vec<usize> = if pads_attr.is_empty() {
        vec![0; spatial_dims * 2]
    } else {
        pads_attr.iter().map(|&p| p as usize).collect()
    };

    let output_padding_attr: Vec<i64> = node.attrs.ints("output_padding").to_vec();
    let output_padding: Vec<usize> = if output_padding_attr.is_empty() {
        vec![0; spatial_dims]
    } else {
        output_padding_attr.iter().map(|&p| p as usize).collect()
    };

    if pads.len() != spatial_dims * 2 {
        return None;
    }

    let mut out_shape = vec![n, c_out];
    for d in 0..spatial_dims {
        // ConvTranspose formula:
        // out = (input - 1) * stride - pad_begin - pad_end + dilation * (kernel - 1) + output_padding + 1
        let out_dim = (input_shape[d + 2] - 1) * strides[d]
            + dilations[d] * (kernel_shape[d] - 1)
            + output_padding[d]
            + 1
            - pads[d]
            - pads[d + spatial_dims];
        out_shape.push(out_dim);
    }

    Some(vec![out_shape])
}

/// Einsum: parse equation to determine output shape.
fn infer_einsum_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let equation = node.attrs.s("equation");
    if equation.is_empty() {
        return None;
    }

    // Parse equation: "ij,jk->ik" style
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return None;
    }

    let input_specs: Vec<&str> = parts[0].split(',').collect();
    let output_spec = parts[1].trim();

    if input_specs.len() != node.inputs.len() {
        return None;
    }

    // Build label -> dimension mapping from inputs
    let mut label_dims: HashMap<u8, usize> = HashMap::new();
    for (i, spec) in input_specs.iter().enumerate() {
        let shape = get_input_shape(node, i, known)?;
        let labels: Vec<u8> = spec
            .trim()
            .bytes()
            .filter(|b| b.is_ascii_alphabetic())
            .collect();
        if labels.len() != shape.len() {
            return None;
        }
        for (j, &label) in labels.iter().enumerate() {
            label_dims.entry(label).or_insert(shape[j]);
        }
    }

    // Build output shape from output spec
    let output_labels: Vec<u8> = output_spec
        .bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .collect();
    let mut out = Vec::new();
    for &label in &output_labels {
        let dim = label_dims.get(&label)?;
        out.push(*dim);
    }

    Some(vec![out])
}

/// LSTM shape inference.
/// Inputs: X [seq_len, batch, input_size], W, R, B, sequence_lens, initial_h, initial_c, P
/// Outputs: Y [seq_len, num_directions, batch, hidden_size],
///          Y_h [num_directions, batch, hidden_size],
///          Y_c [num_directions, batch, hidden_size]
fn infer_lstm_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let x_shape = get_input_shape(node, 0, known)?;
    let w_shape = get_input_shape(node, 1, known)?;

    if x_shape.len() != 3 || w_shape.len() != 3 {
        return None;
    }

    let seq_len = x_shape[0];
    let batch = x_shape[1];
    let num_directions = w_shape[0];
    // W shape: [num_directions, 4*hidden_size, input_size]
    let hidden_size_x4 = w_shape[1];
    if hidden_size_x4 % 4 != 0 {
        return None;
    }
    let hidden_size = hidden_size_x4 / 4;

    let y = vec![seq_len, num_directions, batch, hidden_size];
    let y_h = vec![num_directions, batch, hidden_size];
    let y_c = vec![num_directions, batch, hidden_size];

    Some(vec![y, y_h, y_c])
}

/// GRU shape inference.
/// Similar to LSTM but W has 3*hidden_size and only two state outputs.
fn infer_gru_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let x_shape = get_input_shape(node, 0, known)?;
    let w_shape = get_input_shape(node, 1, known)?;

    if x_shape.len() != 3 || w_shape.len() != 3 {
        return None;
    }

    let seq_len = x_shape[0];
    let batch = x_shape[1];
    let num_directions = w_shape[0];
    // W shape: [num_directions, 3*hidden_size, input_size]
    let hidden_size_x3 = w_shape[1];
    if hidden_size_x3 % 3 != 0 {
        return None;
    }
    let hidden_size = hidden_size_x3 / 3;

    let y = vec![seq_len, num_directions, batch, hidden_size];
    let y_h = vec![num_directions, batch, hidden_size];

    Some(vec![y, y_h])
}

/// GridSample: output shape [N, C, H_out, W_out] where H_out, W_out come from grid.
fn infer_grid_sample_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let grid_shape = get_input_shape(node, 1, known)?;

    if input_shape.len() != 4 || grid_shape.len() != 4 {
        return None;
    }

    // Input: [N, C, H_in, W_in], Grid: [N, H_out, W_out, 2]
    Some(vec![vec![
        input_shape[0],
        input_shape[1],
        grid_shape[1],
        grid_shape[2],
    ]])
}

/// RoiAlign: output shape [num_rois, C, output_height, output_width]
fn infer_roi_align_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rois_shape = get_input_shape(node, 1, known)?;

    if input_shape.len() != 4 || rois_shape.len() != 2 {
        return None;
    }

    let num_rois = rois_shape[0];
    let c = input_shape[1];
    let output_height = node.attrs.i("output_height", 1) as usize;
    let output_width = node.attrs.i("output_width", 1) as usize;

    Some(vec![vec![num_rois, c, output_height, output_width]])
}

/// LinearClassifier: output labels [N] and scores [N, num_classes].
fn infer_linear_classifier_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.is_empty() {
        return None;
    }

    let n = input_shape[0];

    // Try to determine number of classes from coefficients attribute
    let coefficients = node.attrs.float_lists.get("coefficients");
    let num_features = if input_shape.len() > 1 {
        input_shape[1]
    } else {
        1
    };

    let num_classes = if let Some(coeffs) = coefficients {
        coeffs.len().checked_div(num_features)?
    } else {
        return None;
    };

    // Two outputs: labels [N], scores [N, num_classes]
    Some(vec![vec![n], vec![n, num_classes]])
}

/// LinearRegressor: output [N, num_targets].
fn infer_linear_regressor_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.is_empty() {
        return None;
    }

    let n = input_shape[0];

    let coefficients = node.attrs.float_lists.get("coefficients");
    let num_features = if input_shape.len() > 1 {
        input_shape[1]
    } else {
        1
    };

    let targets = node.attrs.i("targets", 1) as usize;

    let num_targets = if let Some(coeffs) = coefficients {
        coeffs.len().checked_div(num_features).unwrap_or(targets)
    } else {
        targets
    };

    Some(vec![vec![n, num_targets]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::shape_inference::{infer_shapes, infer_shapes_with_diagnostics};
    use crate::optimizer::test_utils::make_node;

    fn shapes_map(pairs: &[(&str, Vec<usize>)]) -> HashMap<String, Vec<usize>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn weights_map(pairs: &[(&str, Vec<f32>, Vec<usize>)]) -> HashMap<String, Tensor> {
        pairs
            .iter()
            .map(|(k, data, shape)| (k.to_string(), Tensor::new(data.clone(), shape.clone())))
            .collect()
    }

    #[test]
    fn test_shape_inference_global_avg_pool() {
        let node = make_node(OpKind::GlobalAveragePool, "gap", vec!["x"], vec!["y"]);
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![1, 3, 8, 8])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![1, 3, 1, 1]));
    }

    #[test]
    fn test_shape_inference_depth_to_space() {
        let mut node = make_node(OpKind::DepthToSpace, "d2s", vec!["x"], vec!["y"]);
        node.attrs.ints.insert("blocksize".to_string(), 2);

        let nodes = vec![node];
        let weights = HashMap::new();
        // C=12, block=2 -> C/(2^2)=3, H*2=4, W*2=6
        let input_shapes = shapes_map(&[("x", vec![1, 12, 2, 3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![1, 3, 4, 6]));
    }

    #[test]
    fn test_shape_inference_comparison_ops() {
        // Equal with broadcast: [2, 3] and [3] -> [2, 3]
        let node = make_node(OpKind::Equal, "eq", vec!["a", "b"], vec!["c"]);
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("a", vec![2, 3]), ("b", vec![3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("c"), Some(&vec![2, 3]));
    }

    #[test]
    fn test_shape_inference_dropout() {
        let node = make_node(OpKind::Dropout, "drop", vec!["x"], vec!["y", "mask"]);
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![4, 5, 6])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![4, 5, 6]));
        assert_eq!(result.get("mask"), Some(&vec![4, 5, 6]));
    }

    #[test]
    fn test_shape_inference_lstm() {
        // X: [seq_len=10, batch=2, input_size=8]
        // W: [num_directions=1, 4*hidden_size=16, input_size=8]
        // R: [num_directions=1, 4*hidden_size=16, hidden_size=4]
        let node = make_node(
            OpKind::LSTM,
            "lstm",
            vec!["x", "w", "r"],
            vec!["y", "y_h", "y_c"],
        );
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[
            ("x", vec![10, 2, 8]),
            ("w", vec![1, 16, 8]),
            ("r", vec![1, 16, 4]),
        ]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // Y: [10, 1, 2, 4]
        assert_eq!(result.get("y"), Some(&vec![10, 1, 2, 4]));
        // Y_h: [1, 2, 4]
        assert_eq!(result.get("y_h"), Some(&vec![1, 2, 4]));
        // Y_c: [1, 2, 4]
        assert_eq!(result.get("y_c"), Some(&vec![1, 2, 4]));
    }

    #[test]
    fn test_shape_inference_reduce_mean() {
        let mut node = make_node(OpKind::ReduceMean, "rm", vec!["x"], vec!["y"]);
        node.attrs.int_lists.insert("axes".to_string(), vec![1, 2]);
        node.attrs.ints.insert("keepdims".to_string(), 1);

        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4, 5])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 1, 1, 5]));
    }

    #[test]
    fn test_shape_inference_reduce_no_keepdims() {
        let mut node = make_node(OpKind::ReduceSum, "rs", vec!["x"], vec!["y"]);
        node.attrs.int_lists.insert("axes".to_string(), vec![1]);
        node.attrs.ints.insert("keepdims".to_string(), 0);

        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 4]));
    }

    #[test]
    fn test_shape_inference_pool() {
        let mut node = make_node(OpKind::MaxPool, "mp", vec!["x"], vec!["y"]);
        node.attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![2, 2]);
        node.attrs
            .int_lists
            .insert("strides".to_string(), vec![2, 2]);
        node.attrs
            .int_lists
            .insert("pads".to_string(), vec![0, 0, 0, 0]);

        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![1, 3, 8, 8])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![1, 3, 4, 4]));
    }

    #[test]
    fn test_shape_inference_space_to_depth() {
        let mut node = make_node(OpKind::SpaceToDepth, "s2d", vec!["x"], vec!["y"]);
        node.attrs.ints.insert("blocksize".to_string(), 2);

        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![1, 3, 4, 6])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // C*4=12, H/2=2, W/2=3
        assert_eq!(result.get("y"), Some(&vec![1, 12, 2, 3]));
    }

    #[test]
    fn test_shape_inference_variadic_broadcast() {
        let node = make_node(OpKind::VariadicMax, "vmax", vec!["a", "b", "c"], vec!["y"]);
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[
            ("a", vec![2, 1, 4]),
            ("b", vec![1, 3, 1]),
            ("c", vec![2, 1, 1]),
        ]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 3, 4]));
    }

    #[test]
    fn test_shape_inference_einsum() {
        let mut node = make_node(OpKind::Einsum, "ein", vec!["a", "b"], vec!["y"]);
        node.attrs
            .strings
            .insert("equation".to_string(), "ij,jk->ik".to_string());

        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("a", vec![2, 3]), ("b", vec![3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 4]));
    }

    #[test]
    fn test_shape_inference_conv_transpose() {
        let mut node = make_node(OpKind::ConvTranspose, "ct", vec!["x", "w"], vec!["y"]);
        node.attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![3, 3]);
        node.attrs
            .int_lists
            .insert("strides".to_string(), vec![2, 2]);
        node.attrs
            .int_lists
            .insert("pads".to_string(), vec![1, 1, 1, 1]);

        let nodes = vec![node];
        let weights = HashMap::new();
        // Input: [1, 16, 4, 4], Weight: [16, 3, 3, 3] (C_in=16, C_out/group=3)
        let input_shapes = shapes_map(&[("x", vec![1, 16, 4, 4]), ("w", vec![16, 3, 3, 3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // out = (4-1)*2 - 1 - 1 + 1*(3-1) + 0 + 1 = 6-2+2+1 = 7
        assert_eq!(result.get("y"), Some(&vec![1, 3, 7, 7]));
    }

    #[test]
    fn test_shape_inference_quantize_dequantize() {
        let node = make_node(
            OpKind::QuantizeLinear,
            "ql",
            vec!["x", "scale", "zp"],
            vec!["y"],
        );
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4]), ("scale", vec![1]), ("zp", vec![1])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 3, 4]));
    }

    #[test]
    fn test_shape_inference_diagnostics_ext() {
        // Test that ext ops produce proper diagnostics when inputs are missing
        let node = make_node(
            OpKind::GlobalAveragePool,
            "gap",
            vec!["missing_input"],
            vec!["y"],
        );
        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = HashMap::new();

        let (_shapes, diagnostics) = infer_shapes_with_diagnostics(&nodes, &weights, &input_shapes);

        assert!(!diagnostics.is_empty());
        let diag = &diagnostics[0];
        assert_eq!(diag.node_name, "gap");
        assert!(diag.message.contains("missing_input"));
    }

    #[test]
    fn test_shape_inference_gather_nd() {
        let node = make_node(OpKind::GatherND, "gnd", vec!["data", "indices"], vec!["y"]);
        let nodes = vec![node];
        let weights = HashMap::new();
        // data: [2, 3, 4], indices: [2, 2] (last dim=2, so gather 2 dims from data)
        let input_shapes = shapes_map(&[("data", vec![2, 3, 4]), ("indices", vec![2, 2])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // output = indices[:-1] + data[0+2:] = [2] + [4] = [2, 4]
        assert_eq!(result.get("y"), Some(&vec![2, 4]));
    }

    // ── Stage-3 J-tests: shape inference ─────────────────────────────────────

    #[test]
    fn test_range_shape_inference() {
        // Range(start=0, limit=5, delta=1) → output shape [5]
        let node = make_node(
            OpKind::Range,
            "range",
            vec!["start", "limit", "delta"],
            vec!["y"],
        );
        let nodes = vec![node];
        let weights = weights_map(&[
            ("start", vec![0.0], vec![1]),
            ("limit", vec![5.0], vec![1]),
            ("delta", vec![1.0], vec![1]),
        ]);
        let input_shapes = HashMap::new();

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![5]));
    }

    #[test]
    fn test_hann_window_shape_inference() {
        // HannWindow with size=8 → output shape [8]
        let node = make_node(OpKind::HannWindow, "hann", vec!["size"], vec!["y"]);
        let nodes = vec![node];
        let weights = weights_map(&[("size", vec![8.0], vec![1])]);
        let input_shapes = HashMap::new();

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![8]));
    }

    #[test]
    fn test_stft_shape_inference() {
        // signal shape [1, 16], frame_step=4, frame_length=8, onesided=1
        // n_frames = (16 - 8) / 4 + 1 = 3
        // n_dft   = 8 / 2 + 1 = 5
        // output  = [1, 3, 5, 2]
        let mut node = make_node(
            OpKind::STFT,
            "stft",
            vec!["signal", "frame_step", "", "frame_length"],
            vec!["y"],
        );
        node.attrs.ints.insert("onesided".to_string(), 1);
        let nodes = vec![node];
        let weights = weights_map(&[
            ("frame_step", vec![4.0], vec![1]),
            ("frame_length", vec![8.0], vec![1]),
        ]);
        let input_shapes = shapes_map(&[("signal", vec![1, 16])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![1, 3, 5, 2]));
    }

    #[test]
    fn test_mel_weight_matrix_shape_inference() {
        // MelWeightMatrix: num_mel_bins=40, dft_length=512
        // output: [dft_length/2+1, num_mel_bins] = [257, 40]
        let node = make_node(
            OpKind::MelWeightMatrix,
            "mel",
            vec!["num_mel", "dft_len"],
            vec!["y"],
        );
        let nodes = vec![node];
        let weights = weights_map(&[
            ("num_mel", vec![40.0], vec![1]),
            ("dft_len", vec![512.0], vec![1]),
        ]);
        let input_shapes = HashMap::new();

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![257, 40]));
    }

    #[test]
    fn test_reduce_l1_shape_inference() {
        // ReduceL1 on shape [2, 3] with axes=[1], keepdims=false → [2]
        let mut node = make_node(OpKind::ReduceL1, "rl1", vec!["x"], vec!["y"]);
        node.attrs.int_lists.insert("axes".to_string(), vec![1]);
        node.attrs.ints.insert("keepdims".to_string(), 0);

        let nodes = vec![node];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2]));
    }

    // ── End Stage-3 J-tests ───────────────────────────────────────────────────

    #[test]
    fn test_shape_inference_onehot() {
        let mut node = make_node(
            OpKind::OneHot,
            "oh",
            vec!["indices", "depth", "values"],
            vec!["y"],
        );
        node.attrs.ints.insert("axis".to_string(), -1);

        let nodes = vec![node];
        let weights = weights_map(&[("depth", vec![5.0], vec![1])]);
        let input_shapes = shapes_map(&[("indices", vec![3, 4]), ("values", vec![2])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // indices [3, 4] + depth 5 at axis -1 (=2) -> [3, 4, 5]
        assert_eq!(result.get("y"), Some(&vec![3, 4, 5]));
    }

    #[test]
    fn test_shape_inference_gru() {
        let node = make_node(OpKind::GRU, "gru", vec!["x", "w", "r"], vec!["y", "y_h"]);
        let nodes = vec![node];
        let weights = HashMap::new();
        // X: [8, 2, 6], W: [1, 12, 6] (3*hidden=12, hidden=4)
        let input_shapes = shapes_map(&[
            ("x", vec![8, 2, 6]),
            ("w", vec![1, 12, 6]),
            ("r", vec![1, 12, 4]),
        ]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![8, 1, 2, 4]));
        assert_eq!(result.get("y_h"), Some(&vec![1, 2, 4]));
    }
}
