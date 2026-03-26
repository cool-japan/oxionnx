//! Comprehensive per-operator unit tests with reference values, batch dims, and edge cases.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn make_node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

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

/// Run a single-op graph and return outputs.
fn run_single_op(
    op: OpKind,
    inputs: Vec<(&str, Tensor)>,
    weights: Vec<(&str, Tensor)>,
    input_names: Vec<&str>,
    node_inputs: Vec<&str>,
    node_output: &str,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let node = make_node_with_attrs(op, "op0", &node_inputs, &[node_output], attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: input_names.iter().map(|s| s.to_string()).collect(),
        output_names: vec![node_output.to_string()],
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
    for (name, tensor) in inputs {
        feed.insert(name, tensor);
    }
    session.run(&feed).expect("run")
}

/// Run a single-op graph with multiple outputs.
fn run_single_op_multi_output(
    op: OpKind,
    inputs: Vec<(&str, Tensor)>,
    weights: Vec<(&str, Tensor)>,
    input_names: Vec<&str>,
    node_inputs: Vec<&str>,
    node_outputs: Vec<&str>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let node = make_node_with_attrs(op, "op0", &node_inputs, &node_outputs, attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: input_names.iter().map(|s| s.to_string()).collect(),
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
    for (name, tensor) in inputs {
        feed.insert(name, tensor);
    }
    session.run(&feed).expect("run")
}

fn assert_tensor_approx(actual: &Tensor, expected: &[f32], tol: f32) {
    assert_eq!(
        actual.data.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        actual.data.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.data.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {}: {} vs {} (tol={})",
            i,
            a,
            e,
            tol
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Math ops
// ═══════════════════════════════════════════════════════════════════════════════

// 1. test_matmul_2d - [2,3] x [3,4] = [2,4]
#[test]
fn test_matmul_2d() {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  shape [2,3]
    // B = [[1, 0, 1, 0],
    //      [0, 1, 0, 1],
    //      [1, 1, 1, 1]] shape [3,4]
    // C = A @ B =
    //   row0: [1*1+2*0+3*1, 1*0+2*1+3*1, 1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5, 4, 5]
    //   row1: [4*1+5*0+6*1, 4*0+5*1+6*1, 4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11, 10, 11]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![3, 4],
    );
    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    assert_tensor_approx(out, &[4.0, 5.0, 4.0, 5.0, 10.0, 11.0, 10.0, 11.0], 1e-5);
}

// 2. test_matmul_batched - [2,3,4] x [2,4,5] = [2,3,5]
#[test]
fn test_matmul_batched() {
    // batch=2, M=3, K=4, N=5
    // Use identity-like patterns for verifiability
    // A[0] = ones(3,4), A[1] = 2*ones(3,4)
    let mut a_data = vec![1.0f32; 12]; // batch 0
    a_data.extend(vec![2.0f32; 12]); // batch 1
    let a = Tensor::new(a_data, vec![2, 3, 4]);

    // B[0] = ones(4,5), B[1] = ones(4,5)
    let b_data = vec![1.0f32; 40]; // 2*4*5
    let b = Tensor::new(b_data, vec![2, 4, 5]);

    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3, 5]);

    // batch 0: ones(3,4) @ ones(4,5) = 4*ones(3,5)
    for i in 0..15 {
        assert!(
            (out.data[i] - 4.0).abs() < 1e-5,
            "batch0 idx {}: {} vs 4.0",
            i,
            out.data[i]
        );
    }
    // batch 1: 2*ones(3,4) @ ones(4,5) = 8*ones(3,5)
    for i in 15..30 {
        assert!(
            (out.data[i] - 8.0).abs() < 1e-5,
            "batch1 idx {}: {} vs 8.0",
            i,
            out.data[i]
        );
    }
}

// 3. test_gemm_transB - Gemm with transB=1, alpha, beta
#[test]
fn test_gemm_trans_b() {
    // A = [[1, 2],
    //      [3, 4]] shape [2,2]
    // B = [[1, 3],
    //      [2, 4]] shape [2,2] => transB => B^T = [[1, 2], [3, 4]]
    // C = [10, 20] shape [2]
    // Y = alpha * A @ B^T + beta * C  with alpha=0.5, beta=2.0
    // A @ B^T = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7, 10], [15, 22]]
    // 0.5 * [[7, 10], [15, 22]] = [[3.5, 5.0], [7.5, 11.0]]
    // + 2.0 * [10, 20] broadcast = + [[20, 40], [20, 40]]
    // = [[23.5, 45.0], [27.5, 51.0]]
    let mut attrs = Attributes::default();
    attrs.ints.insert("transB".to_string(), 1);
    attrs.floats.insert("alpha".to_string(), 0.5);
    attrs.floats.insert("beta".to_string(), 2.0);

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
    let c = Tensor::new(vec![10.0, 20.0], vec![2]);

    let outputs = run_single_op(
        OpKind::Gemm,
        vec![("a", a), ("b", b)],
        vec![("c", c)],
        vec!["a", "b"],
        vec!["a", "b", "c"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert_tensor_approx(out, &[23.5, 45.0, 27.5, 51.0], 1e-4);
}

// 4. test_reduce_mean_axis - ReduceMean along axis 1
#[test]
fn test_reduce_mean_axis() {
    // x = [[1, 2, 3],
    //      [4, 5, 6]] shape [2,3]
    // ReduceMean axis=1, keepdims=0 => [2, 5]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceMean,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[2.0, 5.0], 1e-5);
}

// 5. test_reduce_sum_keepdims - ReduceSum with keepdims=1
#[test]
fn test_reduce_sum_keepdims() {
    // x = [[1, 2, 3],
    //      [4, 5, 6]] shape [2,3]
    // ReduceSum axis=1, keepdims=1 => [[6], [15]] shape [2,1]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 1);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceSum,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 1]);
    assert_tensor_approx(out, &[6.0, 15.0], 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NN ops
// ═══════════════════════════════════════════════════════════════════════════════

// 6. test_softmax_axis1 - Softmax([1,4]) along axis 1, verify sum=1
#[test]
fn test_softmax_axis1() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let outputs = run_single_op(
        OpKind::Softmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    // Sum should be 1
    let sum: f32 = out.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum should be 1.0, got {}",
        sum
    );

    // Values should be monotonically increasing
    for i in 0..3 {
        assert!(
            out.data[i] < out.data[i + 1],
            "softmax should be monotonic: {} >= {}",
            out.data[i],
            out.data[i + 1]
        );
    }

    // Check specific values: softmax([1,2,3,4])
    let expected_denom = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp() + 4.0_f32.exp();
    let expected = [
        1.0_f32.exp() / expected_denom,
        2.0_f32.exp() / expected_denom,
        3.0_f32.exp() / expected_denom,
        4.0_f32.exp() / expected_denom,
    ];
    assert_tensor_approx(out, &expected, 1e-5);
}

// 7. test_layer_norm - LayerNorm with scale+bias
#[test]
fn test_layer_norm() {
    // x = [[1, 2, 3, 4]] shape [1,4]
    // mean = 2.5, var = 1.25
    // normalized = [-1.3416, -0.4472, 0.4472, 1.3416] (approx)
    // scale = [2, 2, 2, 2], bias = [1, 1, 1, 1]
    // output = normalized * scale + bias
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let bias = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);

    let outputs = run_single_op(
        OpKind::LayerNorm,
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        vec!["x"],
        vec!["x", "scale", "bias"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    // Compute expected: mean=2.5, var=1.25, inv_std = 1/sqrt(1.25+1e-5) ~ 0.89442
    let mean = 2.5_f32;
    let var = 1.25_f32;
    let inv_std = (var + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| (v - mean) * inv_std * 2.0 + 1.0)
        .collect();
    assert_tensor_approx(out, &expected, 1e-4);
}

// 8. test_batch_norm_inference
#[test]
fn test_batch_norm_inference() {
    // x = [[[[1, 2], [3, 4]]]] shape [1,1,2,2]
    // scale=[2], bias=[1], mean=[2.5], var=[1.25], eps=1e-5
    // BN: (x - mean) / sqrt(var + eps) * scale + bias
    let mut attrs = Attributes::default();
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let scale = Tensor::new(vec![2.0], vec![1]);
    let bias = Tensor::new(vec![1.0], vec![1]);
    let bn_mean = Tensor::new(vec![2.5], vec![1]);
    let bn_var = Tensor::new(vec![1.25], vec![1]);

    let outputs = run_single_op(
        OpKind::BatchNorm,
        vec![("x", x)],
        vec![
            ("scale", scale),
            ("bias", bias),
            ("mean", bn_mean),
            ("var", bn_var),
        ],
        vec!["x"],
        vec!["x", "scale", "bias", "mean", "var"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);

    let mean_val = 2.5_f32;
    let var_val = 1.25_f32;
    let inv_std = (var_val + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| (v - mean_val) * inv_std * 2.0 + 1.0)
        .collect();
    assert_tensor_approx(out, &expected, 1e-4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conv ops
// ═══════════════════════════════════════════════════════════════════════════════

// 9. test_conv2d_dilated - Conv2D with dilation=[2,2]
#[test]
fn test_conv2d_dilated() {
    // Input: [1,1,5,5] all ones
    // Kernel: [1,1,2,2] all ones, dilation=[2,2]
    // With dilation=2, effective kernel is 3x3 (2 + (2-1)*2 = 3 for each dim, but actually
    // dilated kernel covers positions: (0,0),(0,2),(2,0),(2,2) in a 3x3 receptive field)
    // Output size: (5 + 0 + 0 - 2*(2-1) - 1)/1 + 1 = (5 - 2 - 1)/1 + 1 = 3
    // Each output = sum of 4 input values (all 1s) = 4
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![2, 2]);
    attrs.ints.insert("group".to_string(), 1);

    let input = Tensor::new(vec![1.0; 25], vec![1, 1, 5, 5]);
    let kernel = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    assert_tensor_approx(out, &[4.0; 9], 1e-5);
}

// 10. test_conv2d_grouped - Conv2D with group=2
#[test]
fn test_conv2d_grouped() {
    // Input: [1,4,3,3] (4 channels)
    // Kernel: [4,2,1,1] (4 output channels, 2 input channels per group, 1x1 kernel)
    // group=2: group0 reads channels 0,1 -> outputs 0,1; group1 reads channels 2,3 -> outputs 2,3
    // With all-ones input and all-ones kernel:
    // Each output channel = sum of 2 input channels = 2
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 2);

    let input = Tensor::new(vec![1.0; 36], vec![1, 4, 3, 3]); // 4 channels, 3x3
    let kernel = Tensor::new(vec![1.0; 8], vec![4, 2, 1, 1]); // 4 out, 2 in/group, 1x1

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4, 3, 3]);
    // Each output = sum of 2 channels * 1x1 kernel with 1s = 2
    assert_tensor_approx(out, &[2.0; 36], 1e-5);
}

// 11. test_conv2d_stride2 - Conv2D with stride=[2,2]
#[test]
fn test_conv2d_stride2() {
    // Input: [1,1,4,4] with values 1..16
    // Kernel: [1,1,2,2] all ones, stride=2
    // Output shape: (4-2)/2 + 1 = 2 => [1,1,2,2]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);
    let kernel = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Position (0,0): 1+2+5+6=14
    // Position (0,1): 3+4+7+8=22
    // Position (1,0): 9+10+13+14=46
    // Position (1,1): 11+12+15+16=54
    assert_tensor_approx(out, &[14.0, 22.0, 46.0, 54.0], 1e-5);
}

// 12. test_maxpool - MaxPool with known values
#[test]
fn test_maxpool() {
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

    let outputs = run_single_op(
        OpKind::MaxPool,
        vec![("input", input)],
        vec![],
        vec!["input"],
        vec!["input"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Max of each 2x2 block:
    // (0,0): max(1,2,5,6)=6
    // (0,1): max(3,4,7,8)=8
    // (1,0): max(9,10,13,14)=14
    // (1,1): max(11,12,15,16)=16
    assert_tensor_approx(out, &[6.0, 8.0, 14.0, 16.0], 1e-5);
}

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

// ═══════════════════════════════════════════════════════════════════════════════
// Batch dimension tests
// ═══════════════════════════════════════════════════════════════════════════════

// 16. test_matmul_batch1 - batch=1 MatMul
#[test]
fn test_matmul_batch1() {
    // [1,2,3] @ [1,3,2] = [1,2,2]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    // row0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // row1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]

    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 2, 2]);
    assert_tensor_approx(out, &[4.0, 5.0, 10.0, 11.0], 1e-5);
}

// 17. test_matmul_batch4 - batch=4 MatMul
#[test]
fn test_matmul_batch4() {
    // [4,2,2] @ [4,2,2] = [4,2,2]
    // Use identity matrices for all batches => output = input A
    let eye = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let a_data: Vec<f32> = (0..4)
        .flat_map(|batch| {
            let b = batch as f32;
            vec![b + 1.0, b + 2.0, b + 3.0, b + 4.0]
        })
        .collect();
    let b_data: Vec<f32> = eye.iter().copied().cycle().take(16).collect();

    let a = Tensor::new(a_data.clone(), vec![4, 2, 2]);
    let b = Tensor::new(b_data, vec![4, 2, 2]);

    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![4, 2, 2]);
    // A @ I = A
    assert_tensor_approx(out, &a_data, 1e-5);
}

// 18. test_conv2d_batch_n - Conv2D with batch=4
#[test]
fn test_conv2d_batch_n() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    // Input: [4,1,3,3] all ones
    let input = Tensor::new(vec![1.0; 36], vec![4, 1, 3, 3]);
    // Kernel: [1,1,2,2] all ones
    let kernel = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    // Output: [4,1,2,2] each element = sum of 2x2 = 4
    assert_eq!(out.shape, vec![4, 1, 2, 2]);
    assert_tensor_approx(out, &[4.0; 16], 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Edge case tests
// ═══════════════════════════════════════════════════════════════════════════════

// 19. test_add_scalar - scalar + tensor broadcasting
#[test]
fn test_add_scalar() {
    // scalar [1] + tensor [2,3] => broadcast to [2,3]
    let scalar = Tensor::new(vec![10.0], vec![1]);
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let outputs = run_single_op(
        OpKind::Add,
        vec![("a", scalar), ("b", tensor)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_tensor_approx(out, &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0], 1e-5);
}

// 20. test_relu_empty - ReLU on empty tensor (0 elements)
#[test]
fn test_relu_empty() {
    let x = Tensor::new(vec![], vec![0]);
    let outputs = run_single_op(
        OpKind::Relu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert!(out.data.is_empty());
}

// 21. test_softmax_single_element - Softmax on [1,1]
#[test]
fn test_softmax_single_element() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let x = Tensor::new(vec![42.0], vec![1, 1]);
    let outputs = run_single_op(
        OpKind::Softmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1]);
    // Softmax of single element = 1.0
    assert_tensor_approx(out, &[1.0], 1e-5);
}

// 22. test_matmul_small - [1,3] x [3,4] = [1,4] (smallest useful 2D matmul)
#[test]
fn test_matmul_small() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![3, 4],
    );
    // [1,2,3] @ partial-identity => [1, 2, 3, 0]
    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);
    assert_tensor_approx(out, &[1.0, 2.0, 3.0, 0.0], 1e-5);
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

// ═══════════════════════════════════════════════════════════════════════════════
// Numerical precision tests
// ═══════════════════════════════════════════════════════════════════════════════

// 25. test_softmax_large_values - Softmax with large values (numerical stability)
#[test]
fn test_softmax_large_values() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    // Large values that would overflow naive exp() computation
    let x = Tensor::new(vec![1000.0, 1001.0, 1002.0, 1003.0], vec![1, 4]);
    let outputs = run_single_op(
        OpKind::Softmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();

    // Should still sum to 1.0 (numerically stable implementation subtracts max)
    let sum: f32 = out.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax of large values should sum to 1.0, got {}",
        sum
    );

    // No NaN or Inf
    for (i, &v) in out.data.iter().enumerate() {
        assert!(v.is_finite(), "softmax output[{}] = {} is not finite", i, v);
        assert!(v > 0.0, "softmax output[{}] = {} should be positive", i, v);
    }

    // Values should be monotonically increasing
    for i in 0..3 {
        assert!(out.data[i] < out.data[i + 1]);
    }
}

// 26. test_layer_norm_epsilon - LayerNorm with near-zero variance
#[test]
fn test_layer_norm_epsilon() {
    // All same values => variance = 0, relying on epsilon for stability
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    let x = Tensor::new(vec![5.0, 5.0, 5.0, 5.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let bias = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);

    let outputs = run_single_op(
        OpKind::LayerNorm,
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        vec!["x"],
        vec!["x", "scale", "bias"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    // (5 - 5) / sqrt(0 + 1e-5) * 1 + 0 = 0 for all elements
    for (i, &v) in out.data.iter().enumerate() {
        assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        assert!(v.abs() < 1e-2, "output[{}] = {} should be near zero", i, v);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional operator tests
// ═══════════════════════════════════════════════════════════════════════════════

// test_sub_elementwise
#[test]
fn test_sub_elementwise() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Sub,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[9.0, 18.0, 27.0], 1e-5);
}

// test_mul_broadcast
#[test]
fn test_mul_broadcast() {
    // [2,3] * [1,3] => broadcast to [2,3]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
    let outputs = run_single_op(
        OpKind::Mul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_tensor_approx(out, &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0], 1e-5);
}

// test_div_elementwise
#[test]
fn test_div_elementwise() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let b = Tensor::new(vec![2.0, 4.0, 5.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Div,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[5.0, 5.0, 6.0], 1e-5);
}

// test_pow
#[test]
fn test_pow() {
    let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let b = Tensor::new(vec![3.0, 2.0, 0.5], vec![3]);
    let outputs = run_single_op(
        OpKind::Pow,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    // 2^3=8, 3^2=9, 4^0.5=2
    assert_tensor_approx(out, &[8.0, 9.0, 2.0], 1e-5);
}

// test_sigmoid
#[test]
fn test_sigmoid() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 100.0, -100.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Sigmoid,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    // sigmoid(0) = 0.5, sigmoid(1) ~ 0.7310, sigmoid(-1) ~ 0.2689
    // sigmoid(100) ~ 1.0, sigmoid(-100) ~ 0.0
    assert!((out.data[0] - 0.5).abs() < 1e-5);
    assert!((out.data[1] - 0.7310586).abs() < 1e-4);
    assert!((out.data[2] - 0.2689414).abs() < 1e-4);
    assert!((out.data[3] - 1.0).abs() < 1e-5);
    assert!(out.data[4].abs() < 1e-5);
}

// test_tanh
#[test]
fn test_tanh() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Tanh,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert!((out.data[0] - 0.0).abs() < 1e-5);
    assert!((out.data[1] - 1.0_f32.tanh()).abs() < 1e-5);
    assert!((out.data[2] - (-1.0_f32).tanh()).abs() < 1e-5);
}

// test_relu_mixed
#[test]
fn test_relu_mixed() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Relu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[0.0, 0.0, 0.0, 1.0, 3.0], 1e-5);
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

// test_reduce_max
#[test]
fn test_reduce_max() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    // x = [[1, 5, 3], [4, 2, 6]] shape [2,3]
    let x = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceMax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[5.0, 6.0], 1e-5);
}

// test_reduce_min
#[test]
fn test_reduce_min() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceMin,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[1.0, 2.0], 1e-5);
}

// test_neg
#[test]
fn test_neg() {
    let x = Tensor::new(vec![1.0, -2.0, 0.0, 3.5], vec![4]);
    let outputs = run_single_op(
        OpKind::Neg,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[-1.0, 2.0, 0.0, -3.5], 1e-5);
}

// test_sqrt
#[test]
fn test_sqrt() {
    let x = Tensor::new(vec![0.0, 1.0, 4.0, 9.0, 16.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Sqrt,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-5);
}

// test_gelu
#[test]
fn test_gelu() {
    // GELU(0) = 0, GELU(x) ~ x for large x, GELU(x) ~ 0 for large negative x
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 3.0, -3.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Gelu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    // GELU(0) = 0
    assert!(out.data[0].abs() < 1e-5, "gelu(0) = {}", out.data[0]);
    // GELU(1) ~ 0.8412
    assert!(
        (out.data[1] - 0.8412).abs() < 0.01,
        "gelu(1) = {}",
        out.data[1]
    );
    // GELU(-1) ~ -0.1588
    assert!(
        (out.data[2] - (-0.1588)).abs() < 0.01,
        "gelu(-1) = {}",
        out.data[2]
    );
    // GELU(3) ~ 2.9960
    assert!(
        (out.data[3] - 3.0).abs() < 0.01,
        "gelu(3) = {}",
        out.data[3]
    );
    // GELU(-3) ~ -0.0040
    assert!(out.data[4].abs() < 0.01, "gelu(-3) = {}", out.data[4]);
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

// test_average_pool
#[test]
fn test_average_pool() {
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);

    let outputs = run_single_op(
        OpKind::AveragePool,
        vec![("input", input)],
        vec![],
        vec!["input"],
        vec!["input"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Avg of each 2x2 block:
    // (0,0): (1+2+5+6)/4 = 3.5
    // (0,1): (3+4+7+8)/4 = 5.5
    // (1,0): (9+10+13+14)/4 = 11.5
    // (1,1): (11+12+15+16)/4 = 13.5
    assert_tensor_approx(out, &[3.5, 5.5, 11.5, 13.5], 1e-5);
}

// test_abs
#[test]
fn test_abs() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Abs,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[3.0, 1.0, 0.0, 1.0, 3.0], 1e-5);
}

// test_exp
#[test]
fn test_exp() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Exp,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[1.0, 1.0_f32.exp(), 2.0_f32.exp()], 1e-4);
}

// test_log
#[test]
fn test_log() {
    let x = Tensor::new(vec![1.0, std::f32::consts::E, 10.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Log,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[0.0, 1.0, 10.0_f32.ln()], 1e-4);
}

// test_clip
#[test]
fn test_clip() {
    // Clip(x, min=2, max=5) on [1, 3, 6]
    let x = Tensor::new(vec![1.0, 3.0, 6.0], vec![3]);
    let min_t = Tensor::new(vec![2.0], vec![1]);
    let max_t = Tensor::new(vec![5.0], vec![1]);

    let outputs = run_single_op(
        OpKind::Clip,
        vec![("x", x)],
        vec![("min", min_t), ("max", max_t)],
        vec!["x"],
        vec!["x", "min", "max"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[2.0, 3.0, 5.0], 1e-5);
}

// test_sin_cos
#[test]
fn test_sin_cos() {
    let x = Tensor::new(
        vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
        vec![3],
    );

    let sin_out = run_single_op(
        OpKind::Sin,
        vec![("x", x.clone())],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let s = sin_out.get("out").unwrap();
    assert_tensor_approx(s, &[0.0, 1.0, 0.0], 1e-5);

    let cos_out = run_single_op(
        OpKind::Cos,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let c = cos_out.get("out").unwrap();
    assert_tensor_approx(c, &[1.0, 0.0, -1.0], 1e-5);
}

// test_leaky_relu
#[test]
fn test_leaky_relu() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".to_string(), 0.1);

    let x = Tensor::new(vec![-10.0, -1.0, 0.0, 1.0, 10.0], vec![5]);
    let outputs = run_single_op(
        OpKind::LeakyRelu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[-1.0, -0.1, 0.0, 1.0, 10.0], 1e-5);
}

// test_reciprocal
#[test]
fn test_reciprocal() {
    let x = Tensor::new(vec![1.0, 2.0, 4.0, 0.5], vec![4]);
    let outputs = run_single_op(
        OpKind::Reciprocal,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[1.0, 0.5, 0.25, 2.0], 1e-5);
}

// test_erf
#[test]
fn test_erf() {
    // erf(0) = 0, erf(inf) = 1, erf(-inf) = -1
    let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Erf,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert!(out.data[0].abs() < 1e-5, "erf(0) = {}", out.data[0]);
    // erf(1) ~ 0.8427
    assert!(
        (out.data[1] - 0.8427).abs() < 0.01,
        "erf(1) = {}",
        out.data[1]
    );
    // erf(-1) ~ -0.8427
    assert!(
        (out.data[2] + 0.8427).abs() < 0.01,
        "erf(-1) = {}",
        out.data[2]
    );
}
