//! Conformance tests 22–28: Shape manipulation operators.

mod common;

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, OpKind, OptLevel, Session, Tensor};

use common::{assert_close, assert_shape, make_node_with_attrs, run_op};

// ═══════════════════════════════════════════════════════════════════════════════
// 22–28: Shape conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 22. conformance_reshape — [6] -> [2,3]
#[test]
fn conformance_reshape() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    let shape = Tensor::new(vec![2.0, 3.0], vec![2]);
    let out = run_op(
        OpKind::Reshape,
        vec!["x", "shape"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![("shape", shape)],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "reshape");
    assert_close(&t.data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, "reshape");
}

/// 23. conformance_transpose — [2,3] perm=[1,0] -> [3,2]
#[test]
fn conformance_transpose() {
    // x = [[1,2,3],[4,5,6]] shape [2,3]
    // transpose perm=[1,0] => [[1,4],[2,5],[3,6]] shape [3,2]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("perm".to_string(), vec![1, 0]);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_op(
        OpKind::Transpose,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[3, 2], "transpose");
    assert_close(&t.data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5, "transpose");
}

/// 24. conformance_concat_axis1 — along axis 1
#[test]
fn conformance_concat_axis1() {
    // A = [[1,2],[3,4]] shape [2,2]
    // B = [[5],[6]] shape [2,1]
    // concat axis=1 => [[1,2,5],[3,4,6]] shape [2,3]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0], vec![2, 1]);

    let node = make_node_with_attrs(OpKind::Concat, "op0", &["a", "b"], &["out"], attrs);
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
    let out = session.run(&feed).expect("run");

    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "concat_axis1");
    assert_close(
        &t.data,
        &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0],
        1e-5,
        "concat_axis1",
    );
}

/// 25. conformance_squeeze — remove dim-1 axes
#[test]
fn conformance_squeeze() {
    // x shape [1,3,1] => squeeze axes=[0,2] => [3]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![0, 2]);

    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
    let out = run_op(
        OpKind::Squeeze,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[3], "squeeze");
    assert_close(&t.data, &[1.0, 2.0, 3.0], 1e-5, "squeeze");
}

/// 26. conformance_unsqueeze — add dim-1 axes
#[test]
fn conformance_unsqueeze() {
    // x shape [3] => unsqueeze axes=[0,2] => [1,3,1]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![0, 2]);

    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let out = run_op(
        OpKind::Unsqueeze,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 3, 1], "unsqueeze");
    assert_close(&t.data, &[1.0, 2.0, 3.0], 1e-5, "unsqueeze");
}

/// 27. conformance_flatten — [2,3,4] -> [2,12]
#[test]
fn conformance_flatten() {
    // axis=1 => flatten dims 1..end => [2, 3*4] = [2, 12]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = Tensor::new(data.clone(), vec![2, 3, 4]);
    let out = run_op(
        OpKind::Flatten,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 12], "flatten");
    assert_close(&t.data, &data, 1e-5, "flatten");
}

/// 28. conformance_slice — slice with start/end/step
#[test]
fn conformance_slice() {
    // x = [0,1,2,3,4,5,6,7] shape [8]
    // starts=[1], ends=[7], axes=[0], steps=[2]
    // Expected: [1, 3, 5]
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![8]);
    let starts = Tensor::new(vec![1.0], vec![1]);
    let ends = Tensor::new(vec![7.0], vec![1]);
    let axes = Tensor::new(vec![0.0], vec![1]);
    let steps = Tensor::new(vec![2.0], vec![1]);

    let out = run_op(
        OpKind::Slice,
        vec!["x", "starts", "ends", "axes", "steps"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[1.0, 3.0, 5.0], 1e-5, "slice");
}
