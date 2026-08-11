//! End-to-end verification of the `fuse_instance_norm` pass against the model
//! it was written for: the inswapper-128 face-swap decoder.
//!
//! That model is a 277 MB download, not a repository fixture, so every test
//! here resolves it at run time and **skips** (returns early) when it is
//! absent. Point `OXIONNX_INSWAPPER_MODEL` at a copy to run them somewhere
//! else; the default location is the one the oxiface model downloader uses,
//! derived from `$HOME` rather than written out.
//!
//! What the model contains (verified against the file, not assumed): 238 nodes,
//! zero `InstanceNormalization`, and twelve AdaIN normalisation chains spelled
//!
//! ```text
//! ReduceMean(axes=[2,3]) → Sub → Mul(diff,diff) → ReduceMean(axes=[2,3])
//!   → Add(1e-8) → Sqrt → Div(1, ·) → Mul(diff, ·)
//! ```
//!
//! each followed by a `Mul`/`Add` pair whose scale and shift come from a Gemm
//! head — runtime tensors, so they stay outside the fused node.

use oxionnx::graph::OpKind;
use oxionnx::optimizer::fusion::fuse_instance_norm;
use oxionnx::optimizer::shape_inference::infer_shapes;
use oxionnx::optimizer::{optimize_with_input_shapes, PassLevel};
use oxionnx::tensor::Tensor;
use oxionnx::{OptLevel, Session};
use std::collections::HashMap;
use std::path::PathBuf;

/// Number of AdaIN normalisation chains in inswapper-128.
const EXPECTED_CHAINS: usize = 12;

/// Nodes each chain contributes before fusion: mean, sub, square, var,
/// add(eps), sqrt, reciprocal-div, normalise-mul.
const NODES_PER_CHAIN: usize = 8;

/// Locate the model, or `None` when it is not installed on this machine.
fn model_path() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("OXIONNX_INSWAPPER_MODEL") {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home)
        .join(".oxiface")
        .join("models")
        .join("web")
        .join("inswapper_128_fp16.onnx");
    path.is_file().then_some(path)
}

/// The graph's declared, fully static input shapes — the same seed
/// `Session::build_from_graph` gives the optimizer.
fn declared_input_shapes(graph: &oxionnx::graph::Graph) -> HashMap<String, Vec<usize>> {
    let mut shapes = HashMap::new();
    for info in &graph.input_infos {
        let dims: Option<Vec<usize>> = info
            .symbolic_shape()
            .iter()
            .map(|d| match d {
                oxionnx::graph::Dim::Static(n) => Some(*n),
                _ => None,
            })
            .collect();
        if let Some(dims) = dims {
            if !dims.is_empty() {
                shapes.insert(info.name.clone(), dims);
            }
        }
    }
    shapes
}

fn count_op(nodes: &[oxionnx::graph::Node], op: &OpKind) -> usize {
    nodes.iter().filter(|n| &n.op == op).count()
}

/// Deterministic pseudo-random tensor: a fixed LCG, so both sessions in the
/// equivalence test see byte-identical input without a dependency on a RNG
/// crate.
fn fixed_input(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut state = seed;
    let data = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

/// Every real-model check, in one test function and strictly in sequence.
///
/// # Why one test and not four
///
/// `cargo test` runs the functions in a test binary on parallel threads, and
/// each check here needs the model's weight map — 277 MB of fp16 on disk
/// becomes ~600 MB of `f32` `Tensor`s in memory, and the equivalence check
/// builds two sessions. Split across four `#[test]` functions this file peaked
/// at 3.9 GB RSS (measured); as one sequential function it peaks at 1.4 GB,
/// because each section drops its parse before the next one starts. The
/// sections are named and reported through `eprintln!`, and every assertion
/// carries its own message, so a failure still says which check broke.
///
/// Skips (returns) when the model is not installed — see the module docs.
#[test]
fn inswapper_adain_fusion_end_to_end() {
    let path = match model_path() {
        Some(p) => p,
        None => return,
    };
    let bytes = std::fs::read(&path).expect("read model");

    check_raw_graph_fusion(&bytes);
    check_full_pipeline(&bytes);
    check_registry_gate(&bytes);
    check_numerical_equivalence(&bytes);
}

/// The pass applied on its own to the raw parsed graph: the isolated
/// node-count measurement, uncontaminated by the other passes in the pipeline.
fn check_raw_graph_fusion(bytes: &[u8]) {
    let (graph, weights) = oxionnx::model::load(bytes).expect("parse model");

    let nodes = graph.nodes.clone();
    let before = nodes.len();
    assert_eq!(
        count_op(&nodes, &OpKind::InstanceNorm),
        0,
        "the model has no InstanceNormalization node; the pattern is decomposed"
    );

    let input_shapes = declared_input_shapes(&graph);
    let shapes = infer_shapes(&nodes, &weights, &input_shapes);
    let fused = fuse_instance_norm(nodes, &weights, &shapes, &graph.output_names);
    let after = fused.len();

    assert_eq!(
        count_op(&fused, &OpKind::OxiInstanceNorm),
        EXPECTED_CHAINS,
        "raw-graph pass: expected {EXPECTED_CHAINS} fused nodes, graph now {after} nodes"
    );
    assert_eq!(
        before - after,
        EXPECTED_CHAINS * (NODES_PER_CHAIN - 1),
        "raw-graph pass: each chain must collapse {NODES_PER_CHAIN} nodes into 1 \
         (before {before}, after {after})"
    );

    // The affine pair the pass must not touch: each fused node's output still
    // feeds the runtime-scale `Mul`.
    for node in fused.iter().filter(|n| n.op == OpKind::OxiInstanceNorm) {
        let out = node.outputs.first().expect("fused node has an output");
        let consumers: Vec<&oxionnx::graph::Node> =
            fused.iter().filter(|n| n.inputs.contains(out)).collect();
        assert_eq!(
            consumers.len(),
            1,
            "raw-graph pass: normalised tensor {out} must feed exactly one node"
        );
        assert!(
            matches!(consumers[0].op, OpKind::Mul),
            "raw-graph pass: the affine scale Mul must survive outside the fused \
             node, got {:?}",
            consumers[0].op
        );
    }

    // Every declared graph output is still produced.
    for name in &graph.output_names {
        assert!(
            fused.iter().any(|n| n.outputs.contains(name)) || weights.contains_key(name),
            "raw-graph pass: graph output {name} lost its producer"
        );
    }

    eprintln!("[raw graph] {before} nodes → {after} ({EXPECTED_CHAINS} OxiInstanceNorm)");
}

/// The same measurement through the real pipeline, at the level
/// `Session::from_bytes` uses — this is what actually ships.
fn check_full_pipeline(bytes: &[u8]) {
    let (graph, mut weights) = oxionnx::model::load(bytes).expect("parse model");
    let registry = oxionnx_ops::default_registry();
    let input_shapes = declared_input_shapes(&graph);
    let before = graph.nodes.len();
    let optimized = optimize_with_input_shapes(
        graph.nodes,
        &mut weights,
        &graph.output_names,
        &registry,
        PassLevel::All,
        &input_shapes,
    );

    assert_eq!(
        count_op(&optimized, &OpKind::OxiInstanceNorm),
        EXPECTED_CHAINS,
        "full pipeline: node count {} (from {before})",
        optimized.len()
    );
    // `fuse_layer_norm` runs first and must not have claimed any of these
    // chains: it would have folded the runtime affine term in, which is wrong
    // here.
    assert_eq!(
        count_op(&optimized, &OpKind::LayerNorm),
        0,
        "full pipeline: LayerNorm must not claim a spatial normalisation chain"
    );
    assert!(
        optimized.len() < before,
        "full pipeline must not grow the graph: {before} → {}",
        optimized.len()
    );
    eprintln!(
        "[full pipeline] {before} nodes → {} ({EXPECTED_CHAINS} OxiInstanceNorm)",
        optimized.len()
    );
}

/// A registry without the fused kernel must leave the graph alone: the pass is
/// gated on `registry.get("OxiInstanceNorm")` precisely so a `with_registry`
/// caller never gets a node nothing can dispatch.
fn check_registry_gate(bytes: &[u8]) {
    // A registry with no operators at all still exercises the gate; at
    // `PassLevel::Extended` constant folding (the pipeline's only other
    // registry consumer) does not run, so nothing else changes behaviour.
    let ungated = {
        let (graph, mut weights) = oxionnx::model::load(bytes).expect("parse model");
        let empty = oxionnx::OperatorRegistry::new();
        let input_shapes = declared_input_shapes(&graph);
        let nodes = optimize_with_input_shapes(
            graph.nodes,
            &mut weights,
            &graph.output_names,
            &empty,
            PassLevel::Extended,
            &input_shapes,
        );
        assert_eq!(
            count_op(&nodes, &OpKind::OxiInstanceNorm),
            0,
            "registry gate: no kernel registered, so no fused node may be emitted"
        );
        nodes.len()
    };

    let gated = {
        let (graph, mut weights) = oxionnx::model::load(bytes).expect("parse model");
        let registry = oxionnx_ops::default_registry();
        let input_shapes = declared_input_shapes(&graph);
        let nodes = optimize_with_input_shapes(
            graph.nodes,
            &mut weights,
            &graph.output_names,
            &registry,
            PassLevel::Extended,
            &input_shapes,
        );
        assert_eq!(
            count_op(&nodes, &OpKind::OxiInstanceNorm),
            EXPECTED_CHAINS,
            "registry gate: the default registry has the kernel, so the pass must fire"
        );
        nodes.len()
    };

    assert_eq!(
        ungated - gated,
        EXPECTED_CHAINS * (NODES_PER_CHAIN - 1),
        "registry gate: the gate is the only difference between these two runs \
         ({ungated} vs {gated} nodes)"
    );
    eprintln!("[registry gate] {ungated} nodes without the kernel, {gated} with it");
}

/// Numerical equivalence on the real model: a fused session and an unoptimized
/// one must agree on the same input.
fn check_numerical_equivalence(bytes: &[u8]) {
    let target = fixed_input(&[1, 3, 128, 128], 0x5eed_1234);
    let source = fixed_input(&[1, 512], 0xfeed_9876);
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("target", target);
    inputs.insert("source", source);

    // Sequential, never concurrent: each session holds ~600 MB of f32 weights,
    // so the two are built, run and dropped one after the other.
    let fused_out = run_once(bytes, OptLevel::All, &inputs);
    let plain_out = run_once(bytes, OptLevel::None, &inputs);

    assert_eq!(
        fused_out.shape, plain_out.shape,
        "equivalence: output shape changed"
    );
    let report =
        oxionnx::tolerance::compare_tensors(&fused_out, &plain_out, 1e-5, 1e-5).expect("compare");
    eprintln!(
        "[equivalence] fused vs unoptimized: max_abs={:.3e} max_rel={:.3e} over {} elements",
        report.max_abs_error, report.max_rel_error, report.num_elements
    );
    assert!(
        report.passed,
        "equivalence: fused output diverged — max_abs={:.3e} max_rel={:.3e} \
         ({} abs / {} rel violations)",
        report.max_abs_error,
        report.max_rel_error,
        report.num_abs_violations,
        report.num_rel_violations
    );
}

/// Build a session at `level`, run it once, and drop everything before
/// returning the output tensor.
fn run_once(bytes: &[u8], level: OptLevel, inputs: &HashMap<&str, Tensor>) -> Tensor {
    let (graph, weights) = oxionnx::model::load(bytes).expect("parse model");
    let session = Session::builder()
        .with_optimization_level(level)
        .build_from_graph(graph, weights)
        .expect("build session");
    let out = session.run(inputs).expect("run");
    out.get("output").expect("output tensor").clone()
}
