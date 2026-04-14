//! Synthetic ResNet-18-like end-to-end inference test.
//!
//! Builds a simplified ResNet architecture from scratch using `Session::from_graph()`
//! and verifies the forward pass produces correctly-shaped, finite outputs.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn det_tensor(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(seed).wrapping_add(17)) as f32;
            (x % 200.0 - 100.0) * 0.01
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

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

fn conv_attrs(strides: &[i64], pads: &[i64]) -> Attributes {
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("strides".to_string(), strides.to_vec());
    attrs.int_lists.insert("pads".to_string(), pads.to_vec());
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);
    attrs
}

/// Shape tensor for Reshape op (ONNX takes shape as second input).
fn shape_tensor(dims: &[i64]) -> Tensor {
    let len = dims.len();
    Tensor::new(dims.iter().map(|&d| d as f32).collect(), vec![len])
}

// ── ResNet builder ───────────────────────────────────────────────────────────

struct ResNetBuilder {
    nodes: Vec<Node>,
    weights: HashMap<String, Tensor>,
    counter: usize,
}

impl ResNetBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            weights: HashMap::new(),
            counter: 0,
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.counter;
        self.counter += 1;
        id
    }

    /// Conv -> output name
    #[allow(clippy::too_many_arguments)]
    fn add_conv(
        &mut self,
        input: &str,
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        stride: i64,
        pad: i64,
        seed: u32,
    ) -> String {
        let id = self.next_id();
        let w_name = format!("conv{id}_w");
        let out_name = format!("conv{id}_out");

        self.weights.insert(
            w_name.clone(),
            det_tensor(&[out_ch, in_ch, kernel, kernel], seed),
        );

        let attrs = conv_attrs(&[stride, stride], &[pad, pad, pad, pad]);
        self.nodes.push(make_node_with_attrs(
            OpKind::Conv,
            &format!("conv{id}"),
            &[input, &w_name],
            &[&out_name],
            attrs,
        ));

        out_name
    }

    /// Relu -> output name
    fn add_relu(&mut self, input: &str) -> String {
        let id = self.next_id();
        let out_name = format!("relu{id}_out");
        self.nodes.push(make_node(
            OpKind::Relu,
            &format!("relu{id}"),
            &[input],
            &[&out_name],
        ));
        out_name
    }

    /// Add -> output name
    fn add_add(&mut self, a: &str, b: &str) -> String {
        let id = self.next_id();
        let out_name = format!("add{id}_out");
        self.nodes.push(make_node(
            OpKind::Add,
            &format!("add{id}"),
            &[a, b],
            &[&out_name],
        ));
        out_name
    }

    /// Residual block: input -> Conv(kxk)->Relu->Conv(kxk) + skip -> Add -> Relu
    /// When channels change or stride > 1, adds a 1x1 conv on the skip path.
    fn add_res_block(
        &mut self,
        input: &str,
        in_ch: usize,
        out_ch: usize,
        stride: i64,
        seed_base: u32,
    ) -> String {
        // Main path
        let pad: i64 = 1;
        let c1 = self.add_conv(input, out_ch, in_ch, 3, stride, pad, seed_base);
        let r1 = self.add_relu(&c1);
        let c2 = self.add_conv(&r1, out_ch, out_ch, 3, 1, pad, seed_base.wrapping_add(100));

        // Skip path
        let skip = if in_ch != out_ch || stride != 1 {
            // 1x1 conv to match dimensions
            self.add_conv(
                input,
                out_ch,
                in_ch,
                1,
                stride,
                0,
                seed_base.wrapping_add(200),
            )
        } else {
            input.to_string()
        };

        // Add + Relu
        let sum = self.add_add(&c2, &skip);
        self.add_relu(&sum)
    }

    /// GlobalAveragePool -> output name
    fn add_global_avg_pool(&mut self, input: &str) -> String {
        let id = self.next_id();
        let out_name = format!("gap{id}_out");
        self.nodes.push(make_node(
            OpKind::GlobalAveragePool,
            &format!("gap{id}"),
            &[input],
            &[&out_name],
        ));
        out_name
    }

    /// Reshape -> output name
    fn add_reshape(&mut self, input: &str, target_shape: &[i64]) -> String {
        let id = self.next_id();
        let shape_name = format!("reshape{id}_shape");
        let out_name = format!("reshape{id}_out");

        self.weights
            .insert(shape_name.clone(), shape_tensor(target_shape));

        self.nodes.push(make_node(
            OpKind::Reshape,
            &format!("reshape{id}"),
            &[input, &shape_name],
            &[&out_name],
        ));

        out_name
    }

    /// MatMul -> output name
    fn add_matmul(&mut self, input: &str, weight_shape: &[usize], seed: u32) -> String {
        let id = self.next_id();
        let w_name = format!("fc{id}_w");
        let out_name = format!("fc{id}_out");

        self.weights
            .insert(w_name.clone(), det_tensor(weight_shape, seed));

        self.nodes.push(make_node(
            OpKind::MatMul,
            &format!("fc{id}"),
            &[input, &w_name],
            &[&out_name],
        ));

        out_name
    }
}

// ── Build full ResNet-like model ─────────────────────────────────────────────

fn build_resnet(num_classes: usize) -> (Graph, HashMap<String, Tensor>) {
    let mut b = ResNetBuilder::new();

    // Input: [1, 3, 32, 32]
    // Stem: Conv(3->16, 3x3, pad=1, stride=1) -> Relu
    let stem_conv = b.add_conv("input", 16, 3, 3, 1, 1, 42);
    let stem_relu = b.add_relu(&stem_conv);

    // ResBlock 1: 16->16, stride=1
    let blk1 = b.add_res_block(&stem_relu, 16, 16, 1, 100);

    // ResBlock 2: 16->32, stride=2 (downsample: 32x32 -> 16x16)
    let blk2 = b.add_res_block(&blk1, 16, 32, 2, 200);

    // ResBlock 3: 32->64, stride=2 (downsample: 16x16 -> 8x8)
    let blk3 = b.add_res_block(&blk2, 32, 64, 2, 300);

    // GlobalAveragePool: [1, 64, 8, 8] -> [1, 64, 1, 1]
    let gap = b.add_global_avg_pool(&blk3);

    // Reshape: [1, 64, 1, 1] -> [1, 64]
    let flat = b.add_reshape(&gap, &[1, 64]);

    // Classifier: MatMul [1,64] x [64, num_classes] -> [1, num_classes]
    let logits = b.add_matmul(&flat, &[64, num_classes], 999);

    let graph = Graph {
        nodes: b.nodes,
        input_names: vec!["input".to_string()],
        output_names: vec![logits],
        ..Default::default()
    };

    (graph, b.weights)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn test_resnet_like_synthetic() {
    let num_classes = 10;
    let (graph, weights) = build_resnet(num_classes);

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build resnet session");

    let input = det_tensor(&[1, 3, 32, 32], 7);
    let outputs = session.run_one("input", input).expect("run resnet");

    // There should be exactly one output
    assert_eq!(outputs.len(), 1);

    let logits = outputs.values().next().expect("output tensor");

    // Shape must be [1, num_classes]
    assert_eq!(logits.shape, vec![1, num_classes]);

    // All values must be finite (no NaN, no Inf)
    for (i, &v) in logits.data.iter().enumerate() {
        assert!(v.is_finite(), "ResNet output[{i}] is not finite: {v}");
    }

    // Values should be in a reasonable range (small weights => bounded output)
    let max_abs = logits.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e6,
        "ResNet output has unexpectedly large values: max_abs = {max_abs}"
    );
}

#[test]
fn test_resnet_output_shape() {
    for &num_classes in &[5, 10, 100] {
        let (graph, weights) = build_resnet(num_classes);
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build resnet session");

        let input = det_tensor(&[1, 3, 32, 32], 13);
        let outputs = session.run_one("input", input).expect("run resnet");

        let logits = outputs.values().next().expect("output tensor");
        assert_eq!(
            logits.shape,
            vec![1, num_classes],
            "ResNet output shape mismatch for num_classes={num_classes}"
        );
    }
}

#[test]
fn test_resnet_deterministic() {
    let (graph1, weights1) = build_resnet(10);
    let (graph2, weights2) = build_resnet(10);

    let session1 = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph1, weights1)
        .expect("build session 1");
    let session2 = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph2, weights2)
        .expect("build session 2");

    let input1 = det_tensor(&[1, 3, 32, 32], 7);
    let input2 = det_tensor(&[1, 3, 32, 32], 7);

    let out1 = session1.run_one("input", input1).expect("run 1");
    let out2 = session2.run_one("input", input2).expect("run 2");

    let v1 = &out1.values().next().expect("out1").data;
    let v2 = &out2.values().next().expect("out2").data;

    assert_eq!(v1.len(), v2.len());
    for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Determinism broken at index {i}: {a} vs {b}"
        );
    }
}
