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
