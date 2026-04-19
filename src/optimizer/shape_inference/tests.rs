use super::{infer_shapes, infer_shapes_with_diagnostics};
use crate::tensor::Tensor;
use std::collections::HashMap;

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
fn test_shape_inference_elementwise() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let nodes = vec![make_node(OpKind::Add, "add", vec!["a", "b"], vec!["c"])];
    let weights = HashMap::new();
    // a: [2, 3], b: [3] -> broadcast to [2, 3]
    let input_shapes = shapes_map(&[("a", vec![2, 3]), ("b", vec![3])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("c"), Some(&vec![2, 3]));
}

#[test]
fn test_shape_inference_matmul() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let nodes = vec![make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["c"])];
    let weights = HashMap::new();
    // [2, 3] x [3, 4] -> [2, 4]
    let input_shapes = shapes_map(&[("a", vec![2, 3]), ("b", vec![3, 4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("c"), Some(&vec![2, 4]));
}

#[test]
fn test_shape_inference_matmul_batched() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let nodes = vec![make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["c"])];
    let weights = HashMap::new();
    // [8, 2, 3] x [8, 3, 4] -> [8, 2, 4]
    let input_shapes = shapes_map(&[("a", vec![8, 2, 3]), ("b", vec![8, 3, 4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("c"), Some(&vec![8, 2, 4]));
}

#[test]
fn test_shape_inference_conv() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    // Conv2D: input [1, 3, 8, 8], weight [16, 3, 3, 3], stride=1, pad=1
    let mut conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["y"]);
    conv.attrs
        .int_lists
        .insert("strides".to_string(), vec![1, 1]);
    conv.attrs
        .int_lists
        .insert("pads".to_string(), vec![1, 1, 1, 1]);
    conv.attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![3, 3]);

    let nodes = vec![conv];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![1, 3, 8, 8]), ("w", vec![16, 3, 3, 3])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    // Output: [1, 16, 8, 8] with pad=1, kernel=3, stride=1
    assert_eq!(result.get("y"), Some(&vec![1, 16, 8, 8]));
}

#[test]
fn test_shape_inference_conv_no_pad() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["y"]);
    conv.attrs
        .int_lists
        .insert("strides".to_string(), vec![2, 2]);
    conv.attrs
        .int_lists
        .insert("pads".to_string(), vec![0, 0, 0, 0]);
    conv.attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![3, 3]);

    let nodes = vec![conv];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![1, 3, 8, 8]), ("w", vec![16, 3, 3, 3])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    // (8 - 3) / 2 + 1 = 3
    assert_eq!(result.get("y"), Some(&vec![1, 16, 3, 3]));
}

#[test]
fn test_shape_inference_reshape() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let nodes = vec![make_node(
        OpKind::Reshape,
        "reshape",
        vec!["x", "shape"],
        vec!["y"],
    )];
    // shape tensor: [2, -1] meaning reshape [2, 3, 4] -> [2, 12]
    let weights = weights_map(&[("shape", vec![2.0, -1.0], vec![2])]);
    let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("y"), Some(&vec![2, 12]));
}

#[test]
fn test_shape_inference_transpose() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut t = make_node(OpKind::Transpose, "t", vec!["x"], vec!["y"]);
    t.attrs.int_lists.insert("perm".to_string(), vec![0, 2, 1]);

    let nodes = vec![t];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("y"), Some(&vec![2, 4, 3]));
}

#[test]
fn test_shape_inference_transpose_default() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let t = make_node(OpKind::Transpose, "t", vec!["x"], vec!["y"]);
    // No perm attribute -> reverse
    let nodes = vec![t];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("y"), Some(&vec![4, 3, 2]));
}

#[test]
fn test_shape_inference_concat() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut cat = make_node(OpKind::Concat, "cat", vec!["a", "b", "c"], vec!["y"]);
    cat.attrs.ints.insert("axis".to_string(), 1);

    let nodes = vec![cat];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[
        ("a", vec![2, 3, 4]),
        ("b", vec![2, 5, 4]),
        ("c", vec![2, 7, 4]),
    ]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("y"), Some(&vec![2, 15, 4]));
}

#[test]
fn test_shape_inference_flatten() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut flat = make_node(OpKind::Flatten, "flat", vec!["x"], vec!["y"]);
    flat.attrs.ints.insert("axis".to_string(), 2);

    let nodes = vec![flat];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 3, 4, 5])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    // axis=2: [2*3, 4*5] = [6, 20]
    assert_eq!(result.get("y"), Some(&vec![6, 20]));
}

#[test]
fn test_shape_inference_squeeze() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut sq = make_node(OpKind::Squeeze, "sq", vec!["x"], vec!["y"]);
    sq.attrs.int_lists.insert("axes".to_string(), vec![1, 3]);

    let nodes = vec![sq];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 1, 3, 1, 4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("y"), Some(&vec![2, 3, 4]));
}

#[test]
fn test_shape_inference_unsqueeze() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut usq = make_node(OpKind::Unsqueeze, "usq", vec!["x"], vec!["y"]);
    usq.attrs.int_lists.insert("axes".to_string(), vec![0, 3]);

    let nodes = vec![usq];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 3])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    // [2, 3] -> insert at 0, 3 -> [1, 2, 3, 1]
    assert_eq!(result.get("y"), Some(&vec![1, 2, 3, 1]));
}

#[test]
fn test_shape_inference_gemm() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut gemm = make_node(OpKind::Gemm, "gemm", vec!["a", "b", "c"], vec!["y"]);
    gemm.attrs.ints.insert("transB".to_string(), 1);

    let nodes = vec![gemm];
    let weights = HashMap::new();
    // A: [4, 3], B: [5, 3] (transB=1 -> use B[0]=5 as N)
    let input_shapes = shapes_map(&[("a", vec![4, 3]), ("b", vec![5, 3]), ("c", vec![5])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("y"), Some(&vec![4, 5]));
}

#[test]
fn test_shape_inference_split() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut split = make_node(OpKind::Split, "split", vec!["x"], vec!["a", "b", "c"]);
    split.attrs.ints.insert("axis".to_string(), 1);
    split
        .attrs
        .int_lists
        .insert("split".to_string(), vec![2, 3, 5]);

    let nodes = vec![split];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![4, 10, 6])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("a"), Some(&vec![4, 2, 6]));
    assert_eq!(result.get("b"), Some(&vec![4, 3, 6]));
    assert_eq!(result.get("c"), Some(&vec![4, 5, 6]));
}

#[test]
fn test_shape_inference_gather() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    let mut gather = make_node(OpKind::Gather, "gather", vec!["data", "indices"], vec!["y"]);
    gather.attrs.ints.insert("axis".to_string(), 0);

    let nodes = vec![gather];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("data", vec![10, 5]), ("indices", vec![3, 2])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    // Gather axis=0: [10, 5] with indices [3, 2] -> [3, 2, 5]
    assert_eq!(result.get("y"), Some(&vec![3, 2, 5]));
}

#[test]
fn test_shape_inference_chain() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    // Chain: MatMul -> Relu -> Add
    let nodes = vec![
        make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
        make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["relu_out"]),
        make_node(OpKind::Add, "add", vec!["relu_out", "bias"], vec!["out"]),
    ];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 3]), ("w", vec![3, 4]), ("bias", vec![4])]);

    let result = infer_shapes(&nodes, &weights, &input_shapes);
    assert_eq!(result.get("mm_out"), Some(&vec![2, 4]));
    assert_eq!(result.get("relu_out"), Some(&vec![2, 4]));
    assert_eq!(result.get("out"), Some(&vec![2, 4]));
}

#[test]
fn test_shape_diagnostics() {
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    // Create a graph where the second node references an unknown input,
    // and a third node uses an unsupported op (Unknown).
    let nodes = vec![
        make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]),
        make_node(
            OpKind::Add,
            "add_missing",
            vec!["r1", "missing_input"],
            vec!["a1"],
        ),
        make_node(
            OpKind::Unknown("CustomOp".to_string()),
            "custom",
            vec!["r1"],
            vec!["c1"],
        ),
    ];
    let weights = HashMap::new();
    let input_shapes = shapes_map(&[("x", vec![2, 3])]);

    let (shapes, diagnostics) = infer_shapes_with_diagnostics(&nodes, &weights, &input_shapes);

    // relu1 should succeed
    assert_eq!(shapes.get("r1"), Some(&vec![2, 3]));

    // add_missing should fail due to missing input
    assert!(diagnostics.len() >= 2);

    let add_diag = diagnostics.iter().find(|d| d.node_name == "add_missing");
    assert!(add_diag.is_some(), "Expected diagnostic for add_missing");
    let add_diag = add_diag.expect("checked above");
    assert_eq!(add_diag.op_type, "Add");
    assert!(
        add_diag.message.contains("missing_input"),
        "Diagnostic should mention the missing input, got: {}",
        add_diag.message
    );

    // custom should fail due to unsupported op
    let custom_diag = diagnostics.iter().find(|d| d.node_name == "custom");
    assert!(custom_diag.is_some(), "Expected diagnostic for custom");
    let custom_diag = custom_diag.expect("checked above");
    assert_eq!(custom_diag.op_type, "CustomOp");
    assert!(
        custom_diag.message.contains("not supported"),
        "Diagnostic should mention unsupported op, got: {}",
        custom_diag.message
    );
}
