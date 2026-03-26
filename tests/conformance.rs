//! Node-level ONNX backend conformance tests.
//!
//! Each test builds a minimal single-op graph, runs it through `Session`,
//! and compares the output against hand-computed reference values.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_node_with_attrs(
    op: OpKind,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Attributes,
) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs,
    }
}

/// Run a single-op graph and return the output map.
fn run_op(
    op: OpKind,
    node_inputs: Vec<&str>,
    node_outputs: Vec<&str>,
    graph_inputs: Vec<&str>,
    input_tensors: Vec<(&str, Tensor)>,
    weights: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let node = make_node_with_attrs(op, "op0", &node_inputs, &node_outputs, attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: graph_inputs.iter().map(|s| s.to_string()).collect(),
        output_names: node_outputs.iter().map(|s| s.to_string()).collect(),
    };
    let mut w: HashMap<String, Tensor> = HashMap::new();
    for (name, tensor) in weights {
        w.insert(name.to_string(), tensor);
    }
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, w)
        .expect("build session");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    for (name, tensor) in input_tensors {
        feed.insert(name, tensor);
    }
    session.run(&feed).expect("run")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch (got {} expected {})",
        msg,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{}: idx {} got {} expected {} (tol {})",
            msg,
            i,
            a,
            e,
            tol
        );
    }
}

fn assert_shape(tensor: &Tensor, expected: &[usize], msg: &str) {
    assert_eq!(tensor.shape, expected, "{}: shape mismatch", msg);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1–10: Math conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 1. conformance_add_broadcast — [2,3] + [3] = broadcast add
#[test]
fn conformance_add_broadcast() {
    // A = [[1,2,3],[4,5,6]], B = [10,20,30]
    // Expected: [[11,22,33],[14,25,36]]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let out = run_op(
        OpKind::Add,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "add_broadcast");
    assert_close(
        &t.data,
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        1e-5,
        "add_broadcast",
    );
}

/// 2. conformance_sub — [4] - [4]
#[test]
fn conformance_sub() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let b = Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4]);
    let out = run_op(
        OpKind::Sub,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[4], "sub");
    assert_close(&t.data, &[9.0, 17.0, 25.0, 33.0], 1e-5, "sub");
}

/// 3. conformance_mul_scalar — [3,3] * scalar
#[test]
fn conformance_mul_scalar() {
    let a = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    let b = Tensor::new(vec![3.0], vec![1]);
    let out = run_op(
        OpKind::Mul,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[3, 3], "mul_scalar");
    assert_close(
        &t.data,
        &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0],
        1e-5,
        "mul_scalar",
    );
}

/// 4. conformance_div — element-wise division
#[test]
fn conformance_div() {
    let a = Tensor::new(vec![10.0, 21.0, 36.0, 4.0], vec![4]);
    let b = Tensor::new(vec![2.0, 3.0, 4.0, 8.0], vec![4]);
    let out = run_op(
        OpKind::Div,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[5.0, 7.0, 9.0, 0.5], 1e-5, "div");
}

/// 5. conformance_pow — x^2
#[test]
fn conformance_pow() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let b = Tensor::new(vec![2.0], vec![1]);
    let out = run_op(
        OpKind::Pow,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[1.0, 4.0, 9.0, 16.0, 25.0], 1e-5, "pow");
}

/// 6. conformance_sqrt — sqrt of known values
#[test]
fn conformance_sqrt() {
    let x = Tensor::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0], vec![6]);
    let out = run_op(
        OpKind::Sqrt,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 1e-5, "sqrt");
}

/// 7. conformance_exp — exp(0)=1, exp(1)≈2.718
#[test]
fn conformance_exp() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
    let out = run_op(
        OpKind::Exp,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    let expected = [1.0, std::f32::consts::E, (-1.0_f32).exp(), (2.0_f32).exp()];
    assert_close(&t.data, &expected, 1e-5, "exp");
}

/// 8. conformance_log — log(1)=0, log(e)=1
#[test]
fn conformance_log() {
    let x = Tensor::new(
        vec![
            1.0,
            std::f32::consts::E,
            std::f32::consts::E * std::f32::consts::E,
            10.0,
        ],
        vec![4],
    );
    let out = run_op(
        OpKind::Log,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    let expected = [0.0, 1.0, 2.0, (10.0_f32).ln()];
    assert_close(&t.data, &expected, 1e-5, "log");
}

/// 9. conformance_abs — absolute value of negatives
#[test]
fn conformance_abs() {
    let x = Tensor::new(vec![-3.0, -1.5, 0.0, 2.5, -7.0], vec![5]);
    let out = run_op(
        OpKind::Abs,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[3.0, 1.5, 0.0, 2.5, 7.0], 1e-5, "abs");
}

/// 10. conformance_neg — negation
#[test]
fn conformance_neg() {
    let x = Tensor::new(vec![1.0, -2.0, 0.0, 3.5, -0.5], vec![5]);
    let out = run_op(
        OpKind::Neg,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[-1.0, 2.0, 0.0, -3.5, 0.5], 1e-5, "neg");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11–15: Reduction conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 11. conformance_reduce_mean_keepdims — ReduceMean axis=1, keepdims=1
#[test]
fn conformance_reduce_mean_keepdims() {
    // x = [[1,2,3],[4,5,6]] shape [2,3]
    // mean axis=1 keepdims => [[2],[5]] shape [2,1]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 1);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceMean,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 1], "reduce_mean_keepdims");
    assert_close(&t.data, &[2.0, 5.0], 1e-5, "reduce_mean_keepdims");
}

/// 12. conformance_reduce_sum_no_keepdims — ReduceSum axis=0, keepdims=0
#[test]
fn conformance_reduce_sum_no_keepdims() {
    // x = [[1,2,3],[4,5,6]] shape [2,3]
    // sum axis=0 => [5,7,9] shape [3]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![0]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceSum,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[5.0, 7.0, 9.0], 1e-5, "reduce_sum_no_keepdims");
}

/// 13. conformance_reduce_max — ReduceMax axis=1
#[test]
fn conformance_reduce_max() {
    // x = [[3,1,2],[6,4,5]] shape [2,3]
    // max axis=1, keepdims=0 => [3, 6]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceMax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[3.0, 6.0], 1e-5, "reduce_max");
}

/// 14. conformance_reduce_min — ReduceMin axis=1
#[test]
fn conformance_reduce_min() {
    // x = [[3,1,2],[6,4,5]] shape [2,3]
    // min axis=1, keepdims=0 => [1, 4]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceMin,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[1.0, 4.0], 1e-5, "reduce_min");
}

/// 15. conformance_argmax — ArgMax axis=1
#[test]
fn conformance_argmax() {
    // x = [[3,1,2],[6,4,5]] shape [2,3]
    // argmax axis=1, keepdims=0 => [0, 0] (index of max in each row)
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3]);
    let out = run_op(
        OpKind::ArgMax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    // argmax returns indices: row0 max=3 at idx 0, row1 max=6 at idx 0
    assert_close(&t.data, &[0.0, 0.0], 1e-5, "argmax");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 16–21: NN conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 16. conformance_relu — clamp negatives
#[test]
fn conformance_relu() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0, -0.5], vec![2, 3]);
    let out = run_op(
        OpKind::Relu,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "relu");
    assert_close(&t.data, &[0.0, 0.0, 0.0, 1.0, 3.0, 0.0], 1e-5, "relu");
}

/// 17. conformance_sigmoid — sigmoid(0)=0.5
#[test]
fn conformance_sigmoid() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 100.0, -100.0], vec![5]);
    let out = run_op(
        OpKind::Sigmoid,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    // sigmoid(0)=0.5, sigmoid(1)=1/(1+e^-1)≈0.7311, sigmoid(-1)≈0.2689
    // sigmoid(100)≈1.0, sigmoid(-100)≈0.0
    let expected = [
        0.5,
        1.0 / (1.0 + (-1.0_f32).exp()),
        1.0 / (1.0 + 1.0_f32.exp()),
        1.0,
        0.0,
    ];
    assert_close(&t.data, &expected, 1e-4, "sigmoid");
}

/// 18. conformance_tanh — tanh(0)=0
#[test]
fn conformance_tanh() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
    let out = run_op(
        OpKind::Tanh,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    let expected = [
        0.0_f32.tanh(),
        1.0_f32.tanh(),
        (-1.0_f32).tanh(),
        2.0_f32.tanh(),
    ];
    assert_close(&t.data, &expected, 1e-5, "tanh");
}

/// 19. conformance_softmax — sum=1, non-negative
#[test]
fn conformance_softmax() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    // Two rows to test batched behavior
    // row0: [1,2,3], row1: [0,0,0]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], vec![2, 3]);
    let out = run_op(
        OpKind::Softmax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "softmax");

    // Row 0: softmax([1,2,3])
    let denom0 = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp();
    let expected_row0 = [
        1.0_f32.exp() / denom0,
        2.0_f32.exp() / denom0,
        3.0_f32.exp() / denom0,
    ];
    assert_close(&t.data[0..3], &expected_row0, 1e-5, "softmax_row0");

    // Row 1: softmax([0,0,0]) = [1/3, 1/3, 1/3]
    let third = 1.0 / 3.0;
    assert_close(&t.data[3..6], &[third, third, third], 1e-5, "softmax_row1");

    // All values non-negative and each row sums to 1
    let sum0: f32 = t.data[0..3].iter().sum();
    let sum1: f32 = t.data[3..6].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-5, "softmax row0 sum = {}", sum0);
    assert!((sum1 - 1.0).abs() < 1e-5, "softmax row1 sum = {}", sum1);
}

/// 20. conformance_layer_norm — normalize + scale + bias
#[test]
fn conformance_layer_norm() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    // x = [[1,2,3,4]] shape [1,4]
    // mean=2.5, var=1.25, inv_std = 1/sqrt(1.25+1e-5)
    // scale=[2,2,2,2], bias=[1,1,1,1]
    // output = (x - mean) * inv_std * scale + bias
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let bias = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);

    let out = run_op(
        OpKind::LayerNorm,
        vec!["x", "scale", "bias"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 4], "layer_norm");

    let mean = 2.5_f32;
    let var = 1.25_f32;
    let inv_std = (var + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| (v - mean) * inv_std * 2.0 + 1.0)
        .collect();
    assert_close(&t.data, &expected, 1e-4, "layer_norm");
}

/// 21. conformance_batch_norm — (x-mean)/sqrt(var+eps)*gamma+beta
#[test]
fn conformance_batch_norm() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    // x = [[[[1,2],[3,4]]]] shape [1,1,2,2]
    // scale=[2], bias=[1], mean=[2.5], var=[1.25]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let scale = Tensor::new(vec![2.0], vec![1]);
    let bias = Tensor::new(vec![1.0], vec![1]);
    let bn_mean = Tensor::new(vec![2.5], vec![1]);
    let bn_var = Tensor::new(vec![1.25], vec![1]);

    let out = run_op(
        OpKind::BatchNorm,
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![
            ("scale", scale),
            ("bias", bias),
            ("mean", bn_mean),
            ("var", bn_var),
        ],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 1, 2, 2], "batch_norm");

    let m = 2.5_f32;
    let v = 1.25_f32;
    let inv_std = (v + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&x| (x - m) * inv_std * 2.0 + 1.0)
        .collect();
    assert_close(&t.data, &expected, 1e-4, "batch_norm");
}

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

// ═══════════════════════════════════════════════════════════════════════════════
// 29–31: Conv conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 29. conformance_conv2d_1x1 — 1x1 convolution (equivalent to per-pixel matmul)
#[test]
fn conformance_conv2d_1x1() {
    // Input: [1,2,2,2] (batch=1, 2 channels, 2x2 spatial)
    // Kernel: [3,2,1,1] (3 output channels, 2 input channels, 1x1)
    // Each output pixel = dot product of kernel row with input channels at that pixel
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    // Input: channel 0 = [[1,2],[3,4]], channel 1 = [[5,6],[7,8]]
    let input = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 2, 2, 2],
    );
    // Kernel: 3 filters, each [2,1,1]
    // filter0 = [1, 0], filter1 = [0, 1], filter2 = [1, 1]
    let kernel = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2, 1, 1]);

    let out = run_op(
        OpKind::Conv,
        vec!["input", "kernel"],
        vec!["out"],
        vec!["input"],
        vec![("input", input)],
        vec![("kernel", kernel)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 3, 2, 2], "conv2d_1x1");
    // filter0 (takes ch0): [1,2,3,4]
    // filter1 (takes ch1): [5,6,7,8]
    // filter2 (ch0+ch1):   [6,8,10,12]
    assert_close(
        &t.data,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 8.0, 10.0, 12.0],
        1e-5,
        "conv2d_1x1",
    );
}

/// 30. conformance_conv2d_3x3_pad1 — 3x3 with padding=1 (output same spatial dims)
#[test]
fn conformance_conv2d_3x3_pad1() {
    // Input: [1,1,3,3] all ones
    // Kernel: [1,1,3,3] all ones, pad=1
    // With padding, output is [1,1,3,3]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![1, 1, 1, 1]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    let input = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
    let kernel = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);

    let out = run_op(
        OpKind::Conv,
        vec!["input", "kernel"],
        vec!["out"],
        vec!["input"],
        vec![("input", input)],
        vec![("kernel", kernel)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 1, 3, 3], "conv2d_3x3_pad1");
    // Corner (0,0): 4 elements in receptive field that overlap with input => 4
    // Edge (0,1): 6 elements => 6
    // Center (1,1): all 9 => 9
    assert_close(
        &t.data,
        &[4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0],
        1e-5,
        "conv2d_3x3_pad1",
    );
}

/// 31. conformance_maxpool_2x2 — 2x2 pool stride 2
#[test]
fn conformance_maxpool_2x2() {
    // Input: [1,1,4,4] with values 1..16
    // kernel_shape=[2,2], strides=[2,2] => [1,1,2,2]
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);

    let out = run_op(
        OpKind::MaxPool,
        vec!["input"],
        vec!["out"],
        vec!["input"],
        vec![("input", input)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 1, 2, 2], "maxpool_2x2");
    // max of each 2x2 block:
    // (0,0): max(1,2,5,6)=6
    // (0,1): max(3,4,7,8)=8
    // (1,0): max(9,10,13,14)=14
    // (1,1): max(11,12,15,16)=16
    assert_close(&t.data, &[6.0, 8.0, 14.0, 16.0], 1e-5, "maxpool_2x2");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 32–34: Indexing conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 32. conformance_gather_axis0 — gather rows
#[test]
fn conformance_gather_axis0() {
    // data = [[1,2],[3,4],[5,6]] shape [3,2]
    // indices = [2, 0] shape [2]
    // gather axis=0 => [[5,6],[1,2]] shape [2,2]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 0);

    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let indices = Tensor::new(vec![2.0, 0.0], vec![2]);

    let out = run_op(
        OpKind::Gather,
        vec!["data", "indices"],
        vec!["out"],
        vec!["data", "indices"],
        vec![("data", data), ("indices", indices)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 2], "gather_axis0");
    assert_close(&t.data, &[5.0, 6.0, 1.0, 2.0], 1e-5, "gather_axis0");
}

/// 33. conformance_where_condition — ternary select
#[test]
fn conformance_where_condition() {
    // condition = [1, 0, 1, 0] (treated as bool)
    // x = [10, 20, 30, 40]
    // y = [1, 2, 3, 4]
    // where(cond, x, y) = [10, 2, 30, 4]
    let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    let out = run_op(
        OpKind::Where,
        vec!["cond", "x", "y"],
        vec!["out"],
        vec!["cond", "x", "y"],
        vec![("cond", cond), ("x", x), ("y", y)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[10.0, 2.0, 30.0, 4.0], 1e-5, "where");
}

/// 34. conformance_onehot — one-hot encoding
#[test]
fn conformance_onehot() {
    // indices = [0, 1, 2] shape [3]
    // depth = 4
    // values = [0, 1] (off_value=0, on_value=1)
    // axis = -1 (default)
    // Expected shape [3,4]:
    // [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);

    let indices = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let depth = Tensor::new(vec![4.0], vec![1]);
    let values = Tensor::new(vec![0.0, 1.0], vec![2]);

    let out = run_op(
        OpKind::OneHot,
        vec!["indices", "depth", "values"],
        vec!["out"],
        vec!["indices", "depth", "values"],
        vec![("indices", indices), ("depth", depth), ("values", values)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[3, 4], "onehot");
    assert_close(
        &t.data,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1e-5,
        "onehot",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 35: Quantization conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 35. conformance_quantize_dequantize — round-trip within tolerance
#[test]
fn conformance_quantize_dequantize() {
    // QuantizeLinear: y = clamp(round(x / scale) + zero_point, 0, 255)
    // DequantizeLinear: y = (x - zero_point) * scale
    //
    // x = [0.0, 1.0, 2.0, 3.0, 4.0]
    // scale = 0.1, zero_point = 0
    // Quantized: round([0, 10, 20, 30, 40]) = [0, 10, 20, 30, 40]
    // Dequantized: [0, 1.0, 2.0, 3.0, 4.0]
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5]);
    let scale = Tensor::new(vec![0.1], vec![1]);
    let zero_point = Tensor::new(vec![0.0], vec![1]);

    // Step 1: Quantize
    let quantized = run_op(
        OpKind::QuantizeLinear,
        vec!["x", "scale", "zp"],
        vec!["qout"],
        vec!["x", "scale", "zp"],
        vec![
            ("x", x),
            ("scale", scale.clone()),
            ("zp", zero_point.clone()),
        ],
        vec![],
        Attributes::default(),
    );
    let q = quantized.get("qout").unwrap();

    // Step 2: Dequantize
    let dequantized = run_op(
        OpKind::DequantizeLinear,
        vec!["q", "scale", "zp"],
        vec!["dqout"],
        vec!["q", "scale", "zp"],
        vec![("q", q.clone()), ("scale", scale), ("zp", zero_point)],
        vec![],
        Attributes::default(),
    );
    let dq = dequantized.get("dqout").unwrap();

    // Round-trip should be within scale tolerance
    assert_close(
        &dq.data,
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        0.1,
        "quantize_dequantize_roundtrip",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 36–37: Numerical stability conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 36. conformance_softmax_large_input — softmax with values > 100
#[test]
fn conformance_softmax_large_input() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    // Large values that would overflow naive exp()
    let x = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]);
    let out = run_op(
        OpKind::Softmax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 3], "softmax_large");

    // All values should be finite
    for (i, &v) in t.data.iter().enumerate() {
        assert!(v.is_finite(), "softmax_large[{}] = {} not finite", i, v);
        assert!(v >= 0.0, "softmax_large[{}] = {} negative", i, v);
    }

    // Sum should be 1
    let sum: f32 = t.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax_large sum = {}", sum);

    // Largest input should dominate: output[2] should be close to 1.0
    assert!(
        t.data[2] > 0.99,
        "softmax_large: max input should dominate, got {}",
        t.data[2]
    );
}

/// 37. conformance_layernorm_zero_var — near-zero variance (epsilon test)
#[test]
fn conformance_layernorm_zero_var() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    // All same values => variance = 0
    let x = Tensor::new(vec![7.0, 7.0, 7.0, 7.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let bias = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);

    let out = run_op(
        OpKind::LayerNorm,
        vec!["x", "scale", "bias"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 4], "layernorm_zero_var");

    // (7 - 7) / sqrt(0 + eps) * 1 + 0 = 0 for all
    for (i, &v) in t.data.iter().enumerate() {
        assert!(v.is_finite(), "layernorm_zero_var[{}] not finite: {}", i, v);
        assert!(
            v.abs() < 1e-2,
            "layernorm_zero_var[{}] should be near zero: {}",
            i,
            v
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 38+: Additional conformance tests
// ═══════════════════════════════════════════════════════════════════════════════

/// 38. conformance_clip — clamp values within [min, max]
#[test]
fn conformance_clip() {
    // Clip uses min/max as additional inputs
    let x = Tensor::new(vec![-5.0, -1.0, 0.0, 3.0, 10.0], vec![5]);
    let min_val = Tensor::new(vec![-2.0], vec![1]);
    let max_val = Tensor::new(vec![5.0], vec![1]);

    let out = run_op(
        OpKind::Clip,
        vec!["x", "min", "max"],
        vec!["out"],
        vec!["x", "min", "max"],
        vec![("x", x), ("min", min_val), ("max", max_val)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[-2.0, -1.0, 0.0, 3.0, 5.0], 1e-5, "clip");
}

/// 39. conformance_identity — passthrough
#[test]
fn conformance_identity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::new(data.clone(), vec![2, 3]);
    let out = run_op(
        OpKind::Identity,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "identity");
    assert_close(&t.data, &data, 0.0, "identity");
}

/// 40. conformance_gelu — GELU activation
#[test]
fn conformance_gelu() {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    // GELU(0) = 0, GELU(large) ≈ x
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 3.0], vec![4]);
    let out = run_op(
        OpKind::Gelu,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();

    // GELU(0) = 0
    assert!(
        (t.data[0]).abs() < 1e-5,
        "gelu(0) should be 0, got {}",
        t.data[0]
    );
    // GELU(1) ≈ 0.8413
    assert!(
        (t.data[1] - 0.8413).abs() < 0.01,
        "gelu(1) ≈ 0.8413, got {}",
        t.data[1]
    );
    // GELU(-1) ≈ -0.1587
    assert!(
        (t.data[2] - (-0.1587)).abs() < 0.01,
        "gelu(-1) ≈ -0.1587, got {}",
        t.data[2]
    );
    // GELU(3) ≈ 2.9960
    assert!(
        (t.data[3] - 2.9960).abs() < 0.01,
        "gelu(3) ≈ 2.9960, got {}",
        t.data[3]
    );
}
