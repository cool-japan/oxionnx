//! Shape manipulation operator integration tests: Concat, Slice, Transpose,
//! Reshape, Squeeze/Unsqueeze, Flatten, Split, Identity.

mod common;

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, OpKind, OptLevel, Session, Tensor};

use common::{
    assert_tensor_approx, make_node_with_attrs, run_single_op, run_single_op_multi_output,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Shape ops
// ═══════════════════════════════════════════════════════════════════════════════

// 13. test_concat_axis0 - Concat two [2,3] tensors along axis 0 = [4,3]
#[test]
fn test_concat_axis0() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 0);

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]);

    let node = make_node_with_attrs(OpKind::Concat, "concat0", &["a", "b"], &["out"], attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["a".to_string(), "b".to_string()],
        output_names: vec!["out".to_string()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    feed.insert("a", a);
    feed.insert("b", b);
    let outputs = session.run(&feed).expect("run");

    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![4, 3]);
    assert_tensor_approx(
        out,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        1e-5,
    );
}

// test_concat_axis1
#[test]
fn test_concat_axis1() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

    let node = make_node_with_attrs(OpKind::Concat, "concat0", &["a", "b"], &["out"], attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["a".to_string(), "b".to_string()],
        output_names: vec!["out".to_string()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    feed.insert("a", a);
    feed.insert("b", b);
    let outputs = session.run(&feed).expect("run");

    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 5]);
    // Row 0: [1,2, 5,6,7], Row 1: [3,4, 8,9,10]
    assert_tensor_approx(
        out,
        &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0],
        1e-5,
    );
}

// 14. test_slice_steps - Slice with steps > 1
#[test]
fn test_slice_steps() {
    // x = [0, 1, 2, 3, 4, 5, 6, 7] shape [8]
    // Slice: starts=[0], ends=[8], axes=[0], steps=[2]
    // Expected: [0, 2, 4, 6]
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![8]);
    let starts = Tensor::new(vec![0.0], vec![1]);
    let ends = Tensor::new(vec![8.0], vec![1]);
    let axes = Tensor::new(vec![0.0], vec![1]);
    let steps = Tensor::new(vec![2.0], vec![1]);

    let outputs = run_single_op(
        OpKind::Slice,
        vec![("x", x)],
        vec![
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ],
        vec!["x"],
        vec!["x", "starts", "ends", "axes", "steps"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![4]);
    assert_tensor_approx(out, &[0.0, 2.0, 4.0, 6.0], 1e-5);
}

// 15. test_transpose_3d - Transpose [2,3,4] with perm [2,0,1]
#[test]
fn test_transpose_3d() {
    // x has shape [2,3,4] with sequential values
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = Tensor::new(data, vec![2, 3, 4]);

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("perm".to_string(), vec![2, 0, 1]);

    let outputs = run_single_op(
        OpKind::Transpose,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    // perm [2,0,1]: out[k,i,j] = x[i,j,k]
    // out shape = [4, 2, 3]
    assert_eq!(out.shape, vec![4, 2, 3]);

    // Verify some values:
    // x[0,0,0] = 0 => out[0,0,0] = 0
    // x[0,0,1] = 1 => out[1,0,0] = 1
    // x[0,1,0] = 4 => out[0,0,1] = 4
    // x[1,0,0] = 12 => out[0,1,0] = 12

    // out is [4, 2, 3]: index = k * (2*3) + i * 3 + j
    // out[0,0,0] = 0*6 + 0*3 + 0 = idx 0
    assert!((out.data[0] - 0.0).abs() < 1e-5);
    // out[1,0,0] = 1*6 + 0*3 + 0 = idx 6
    assert!((out.data[6] - 1.0).abs() < 1e-5);
    // out[0,0,1] = 0*6 + 0*3 + 1 = idx 1
    assert!((out.data[1] - 4.0).abs() < 1e-5);
    // out[0,1,0] = 0*6 + 1*3 + 0 = idx 3
    assert!((out.data[3] - 12.0).abs() < 1e-5);
}

// 23. test_reshape_with_minus_one - Reshape with -1 (infer dimension)
#[test]
fn test_reshape_with_minus_one() {
    // x shape [2,3] = 6 elements => reshape to [3, -1] => [3, 2]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let shape_tensor = Tensor::new(vec![3.0, -1.0], vec![2]);

    let outputs = run_single_op(
        OpKind::Reshape,
        vec![("x", x)],
        vec![("shape", shape_tensor)],
        vec!["x"],
        vec!["x", "shape"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    assert_tensor_approx(out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}

// 24. test_identity_preserves_data - Identity returns exact copy
#[test]
fn test_identity_preserves_data() {
    let data = vec![
        std::f32::consts::PI,
        -2.71,
        0.0,
        1e10,
        -1e-10,
        f32::INFINITY,
    ];
    let x = Tensor::new(data.clone(), vec![2, 3]);
    let outputs = run_single_op(
        OpKind::Identity,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, data);
}

// test_squeeze_unsqueeze
#[test]
fn test_squeeze_unsqueeze() {
    // Unsqueeze [3] at axis 0 => [1,3]
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let axes = Tensor::new(vec![0.0], vec![1]);

    let outputs = run_single_op(
        OpKind::Unsqueeze,
        vec![("x", x)],
        vec![("axes", axes)],
        vec!["x"],
        vec!["x", "axes"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    assert_tensor_approx(out, &[1.0, 2.0, 3.0], 1e-5);
}

// test_flatten
#[test]
fn test_flatten() {
    // x shape [2,3,4] flatten at axis=1 => [2, 12]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = Tensor::new(data.clone(), vec![2, 3, 4]);

    let outputs = run_single_op(
        OpKind::Flatten,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 12]);
    assert_tensor_approx(out, &data, 1e-5);
}

// test_split_equal
#[test]
fn test_split_equal() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 0);
    attrs.int_lists.insert("split".to_string(), vec![2, 2]);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);

    let outputs = run_single_op_multi_output(
        OpKind::Split,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        vec!["a", "b"],
        attrs,
    );
    let a = outputs.get("a").unwrap();
    let b = outputs.get("b").unwrap();
    assert_eq!(a.shape, vec![2, 2]);
    assert_eq!(b.shape, vec![2, 2]);
    assert_tensor_approx(a, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    assert_tensor_approx(b, &[5.0, 6.0, 7.0, 8.0], 1e-5);
}
