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

// ═══════════════════════════════════════════════════════════════════════════════
// issue #4: the shape-op chains PyTorch's ONNX exporter builds around Pad/Einsum
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a `Constant` node whose `value` attribute is `data` reshaped to `shape`.
fn constant_node(name: &str, output: &str, data: Vec<f32>, shape: Vec<usize>) -> oxionnx::Node {
    let mut attrs = Attributes::default();
    attrs
        .tensors
        .insert("value".to_string(), Tensor::new(data, shape));
    make_node_with_attrs(OpKind::Constant, name, &[], &[output], attrs)
}

fn run_graph(
    nodes: Vec<oxionnx::Node>,
    input_names: &[&str],
    output_names: &[&str],
    feed: Vec<(&str, Tensor)>,
) -> HashMap<String, Tensor> {
    let graph = Graph {
        nodes,
        input_names: input_names.iter().map(|s| s.to_string()).collect(),
        output_names: output_names.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    for (name, tensor) in feed {
        inputs.insert(name, tensor);
    }
    session.run(&inputs).expect("run")
}

/// `nn.ReflectionPad2d` as `torch.onnx.export` actually emits it (opset 11+): `pads` is not a
/// literal at all but the output of `_prepare_onnx_paddings`, which turns torch's
/// `[left, right, top, bottom]` into ONNX's `[begin_0.., end_0..]` with
/// `Concat -> Reshape([-1,2]) -> Slice(reverse) -> Transpose -> Reshape([-1]) -> Cast`.
/// Reproduced here node-for-node from `lama_fp32.onnx` (Carve/LaMa-ONNX), the model in the
/// report, whose 98 `Pad` nodes all take this shape.
///
/// Two properties are asserted: the chain yields the **canonical `2 * rank`** pads
/// `[0,0,1,1, 0,0,1,1]` (the N and C axes zero-padded — it is *not* a short, spatial-only
/// `pads`), and `Pad` then reflects correctly. The reverse step is a `Slice` with
/// `starts = [-1]`, `ends = [INT64_MIN]`, `steps = [-1]`; a `Slice` that ignores negative
/// starts/ends/steps collapses it to zero elements and `Pad` receives an empty `pads`.
#[test]
fn test_issue_4_reflection_pad2d_export_chain_e2e() {
    // x = [[1,2,3],[4,5,6],[7,8,9]] as NCHW [1,1,3,3]; reflect-pad 1 on H and W.
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );

    let mut concat_attrs = Attributes::default();
    concat_attrs.ints.insert("axis".to_string(), 0);
    let mut transpose_attrs = Attributes::default();
    transpose_attrs
        .int_lists
        .insert("perm".to_string(), vec![1, 0]);
    let mut cast_attrs = Attributes::default();
    cast_attrs.ints.insert("to".to_string(), 7); // INT64
    let mut pad_attrs = Attributes::default();
    pad_attrs
        .strings
        .insert("mode".to_string(), "reflect".to_string());

    let nodes = vec![
        // torch's own [left, right, top, bottom] pad list.
        constant_node("torch_pads", "torch_pads", vec![1.0; 4], vec![4]),
        // 2 * rank - len(pad) = 8 - 4 = 4 leading zeros for the N and C axes.
        constant_node("ext_len", "ext_len", vec![4.0], vec![1]),
        make_node_with_attrs(
            OpKind::ConstantOfShape,
            "zeros",
            &["ext_len"],
            &["zeros"],
            Attributes::default(),
        ),
        make_node_with_attrs(
            OpKind::Concat,
            "cat",
            &["torch_pads", "zeros"],
            &["cat"],
            concat_attrs,
        ),
        constant_node("pair_shape", "pair_shape", vec![-1.0, 2.0], vec![2]),
        make_node_with_attrs(
            OpKind::Reshape,
            "to_pairs",
            &["cat", "pair_shape"],
            &["pairs"],
            Attributes::default(),
        ),
        constant_node("flip_start", "flip_start", vec![-1.0], vec![1]),
        constant_node("flip_end", "flip_end", vec![i64::MIN as f32], vec![1]),
        constant_node("flip_axes", "flip_axes", vec![0.0], vec![1]),
        constant_node("flip_steps", "flip_steps", vec![-1.0], vec![1]),
        make_node_with_attrs(
            OpKind::Slice,
            "flip",
            &["pairs", "flip_start", "flip_end", "flip_axes", "flip_steps"],
            &["flipped"],
            Attributes::default(),
        ),
        make_node_with_attrs(
            OpKind::Transpose,
            "collate",
            &["flipped"],
            &["collated"],
            transpose_attrs,
        ),
        constant_node("flat_shape", "flat_shape", vec![-1.0], vec![1]),
        make_node_with_attrs(
            OpKind::Reshape,
            "flatten",
            &["collated", "flat_shape"],
            &["flat"],
            Attributes::default(),
        ),
        make_node_with_attrs(OpKind::Cast, "cast", &["flat"], &["pads"], cast_attrs),
        make_node_with_attrs(OpKind::Pad, "pad", &["x", "pads"], &["y"], pad_attrs),
    ];

    let outputs = run_graph(nodes, &["x"], &["y", "pads"], vec![("x", x)]);

    let pads = outputs.get("pads").expect("output 'pads'");
    assert_eq!(
        pads.data,
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "the exporter's chain must yield ONNX-canonical 2*rank pads"
    );

    // Reflecting index i-1 over 0..2 maps rows/cols to [1,0,1,2,1], so the 5x5 output is
    // built from source rows [4,5,6], [1,2,3], [4,5,6], [7,8,9], [4,5,6], each read with the
    // same column order.
    let y = outputs.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![1, 1, 5, 5]);
    assert_eq!(
        y.data,
        vec![
            5.0, 4.0, 5.0, 6.0, 5.0, //
            2.0, 1.0, 2.0, 3.0, 2.0, //
            5.0, 4.0, 5.0, 6.0, 5.0, //
            8.0, 7.0, 8.0, 9.0, 8.0, //
            5.0, 4.0, 5.0, 6.0, 5.0,
        ]
    );
}

/// The other half of the same report: `Einsum` rejecting an operand whose rank disagrees with
/// its subscript (`input 0 has 3 dims but subscript 'ij' has 2 labels`). The operand rank is
/// not a property of `Einsum` at all — it comes from the `x.reshape(-1, x.shape[-1])` idiom the
/// exporter lowers to `Shape -> Slice(starts=[-1], ends=[INT64_MAX]) -> Concat([-1], ..) ->
/// Reshape`, which is the `Slice` negative-index path again: taking the *whole* shape instead
/// of its last entry makes the reshape target one entry too long and hands `Einsum` a rank-3
/// operand. (216 `Einsum` nodes in `lama_fp32.onnx` are fed exactly this way.)
#[test]
fn test_issue_4_einsum_operand_rank_from_shape_slice_e2e() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);

    let mut concat_attrs = Attributes::default();
    concat_attrs.ints.insert("axis".to_string(), 0);
    let mut einsum_attrs = Attributes::default();
    einsum_attrs
        .strings
        .insert("equation".to_string(), "ij,jk->ik".to_string());

    let nodes = vec![
        make_node_with_attrs(
            OpKind::Shape,
            "shape",
            &["x"],
            &["shape"],
            Attributes::default(),
        ),
        constant_node("last_start", "last_start", vec![-1.0], vec![1]),
        constant_node("last_end", "last_end", vec![i64::MAX as f32], vec![1]),
        constant_node("last_axes", "last_axes", vec![0.0], vec![1]),
        make_node_with_attrs(
            OpKind::Slice,
            "last_dim",
            &["shape", "last_start", "last_end", "last_axes"],
            &["last_dim"],
            Attributes::default(),
        ),
        constant_node("minus_one", "minus_one", vec![-1.0], vec![1]),
        make_node_with_attrs(
            OpKind::Concat,
            "target",
            &["minus_one", "last_dim"],
            &["target"],
            concat_attrs,
        ),
        make_node_with_attrs(
            OpKind::Reshape,
            "flatten",
            &["x", "target"],
            &["flat"],
            Attributes::default(),
        ),
        make_node_with_attrs(
            OpKind::Einsum,
            "einsum",
            &["flat", "w"],
            &["y"],
            einsum_attrs,
        ),
    ];

    let outputs = run_graph(
        nodes,
        &["x", "w"],
        &["y", "target"],
        vec![("x", x), ("w", w)],
    );

    let target = outputs.get("target").expect("output 'target'");
    assert_eq!(
        target.data,
        vec![-1.0, 3.0],
        "Slice must take only the LAST shape entry, keeping the reshape target rank 2"
    );

    // [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]].
    let y = outputs.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![2, 2]);
    assert_eq!(y.data, vec![4.0, 5.0, 10.0, 11.0]);
}

/// issue #4: a pre-opset-11 export (`Pad-2`) reaching the operator through a real `Session`,
/// with optimization (and therefore shape inference) enabled rather than bypassed: the node has
/// `data` as its only input and carries `pads` / `value` as attributes. Both the inference pass
/// and the kernel have to agree on that form; the whole path used to fail with
/// `TensorNotFound: input[1]`.
///
/// x = [[1,2],[3,4]] as [1,1,2,2], pads = [0,0,1,1, 0,0,1,1], value = 7.
#[test]
fn test_issue_4_pad_legacy_attribute_form_session_e2e() {
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("pads".to_string(), vec![0, 0, 1, 1, 0, 0, 1, 1]);
    attrs.floats.insert("value".to_string(), 7.0);

    let node = make_node_with_attrs(OpKind::Pad, "pad0", &["x"], &["y"], attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");

    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    feed.insert("x", Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]));
    let outputs = session.run(&feed).expect("run");

    let y = outputs.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![1, 1, 4, 4]);
    assert_eq!(
        y.data,
        vec![
            7.0, 7.0, 7.0, 7.0, //
            7.0, 1.0, 2.0, 7.0, //
            7.0, 3.0, 4.0, 7.0, //
            7.0, 7.0, 7.0, 7.0,
        ]
    );
}
