//! Wave-1 regression tests for `src/session/run/**` — the session execution paths.
//!
//! Every test here pins a bug that was reachable through the *public* API:
//!
//! | id     | bug                                                             |
//! |--------|-----------------------------------------------------------------|
//! | a4-8   | a node whose operator is not registered was silently skipped     |
//! | a5-0   | ...so `run()` returned `Ok` with graph outputs silently missing   |
//! | a4-4   | subgraph outer-scope captures were not reference-counted         |
//! | a4-6   | the parallel path gave control-flow ops an empty outer scope     |
//! | a3-11  | (same, via `OpContext { outer_scope: None }` in `par_iter`)      |
//! | a4-20  | mixed precision was ignored entirely on the parallel path        |
//! | a4-15  | concurrent runs raced on the shared resolved-shape cache         |
//! | a4-16  | static models never populated it; unnamed dynamic axes went stale|
//! | a4-13  | `run_typed` deep-copied every weight on every call               |

use std::collections::HashMap;
use std::sync::Arc;

use oxionnx::{
    Attributes, DType, Graph, Node, OnnxError, OpContext, OpKind, Operator, OptLevel, Session,
    SessionBuilder, Tensor, TensorInfo, TensorStorage, TypedTensor,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn node_with_attrs(
    op: OpKind,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Attributes,
) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs,
    }
}

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

fn build(graph: Graph, weights: HashMap<String, Tensor>, parallel: bool) -> Session {
    SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(parallel)
        .build_from_graph(graph, weights)
        .expect("session build failed")
}

// ─────────────────────────────────────────────────────────────────────────────
// a4-8 / a5-0 — an operator this engine cannot run must fail the run
// ─────────────────────────────────────────────────────────────────────────────

/// A model containing an operator outside the registry: `y = Frobnicate(x)`.
fn unregistered_op_graph() -> Graph {
    Graph {
        nodes: vec![node(
            OpKind::Unknown("Frobnicate".to_string()),
            "n0",
            &["x"],
            &["y"],
        )],
        input_names: names(&["x"]),
        output_names: names(&["y"]),
        ..Default::default()
    }
}

fn assert_unsupported(err: OnnxError, path: &str) {
    assert!(
        matches!(err, OnnxError::UnsupportedOp(_)),
        "{path}: expected UnsupportedOp, got {err:?}",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Frobnicate"),
        "{path}: must name the op: {msg}"
    );
    assert!(msg.contains("n0"), "{path}: must name the node: {msg}");
}

/// **The bug**: the run loops did `if let OpKind::Unknown(_) = node.op { continue; }`,
/// so loading a model with an operator oxionnx does not implement *succeeded* and
/// `run()` returned `Ok` with the affected outputs silently absent from the map.
#[test]
fn an_unregistered_operator_fails_the_run_on_the_sequential_path() {
    let session = build(unregistered_op_graph(), HashMap::new(), false);
    let err = session
        .run_one("x", Tensor::new(vec![1.0, 2.0], vec![2]))
        .expect_err("a model using an unimplemented operator must not run");
    assert_unsupported(err, "sequential");
}

#[test]
fn an_unregistered_operator_fails_the_run_on_the_parallel_path() {
    // Two independent unknown nodes at depth 0 exercise the *multi-node* branch
    // of the parallel runner, which had its own copy of the skip.
    let graph = Graph {
        nodes: vec![
            node(
                OpKind::Unknown("Frobnicate".to_string()),
                "n0",
                &["x"],
                &["y"],
            ),
            node(OpKind::Relu, "n1", &["x"], &["z"]),
        ],
        input_names: names(&["x"]),
        output_names: names(&["y", "z"]),
        ..Default::default()
    };
    let session = build(graph, HashMap::new(), true);
    let err = session
        .run_one("x", Tensor::new(vec![1.0, 2.0], vec![2]))
        .expect_err("a model using an unimplemented operator must not run");
    assert_unsupported(err, "parallel");
}

#[test]
fn an_unregistered_operator_fails_the_run_on_the_typed_path() {
    let session = build(unregistered_op_graph(), HashMap::new(), false);
    let mut inputs: HashMap<&str, TypedTensor> = HashMap::new();
    inputs.insert(
        "x",
        TypedTensor::new(TensorStorage::F32(vec![1.0, 2.0]), vec![2]),
    );
    let err = session
        .run_typed(&inputs)
        .expect_err("a model using an unimplemented operator must not run");
    assert_unsupported(err, "typed");
}

/// A custom operator registered under a name `OpKind::parse` does not know is
/// `OpKind::Unknown(name)` **and perfectly runnable**.  The unsupported-op gate is
/// therefore the *registry*, never the enum variant: gating on `OpKind::Unknown`
/// would make every custom operator unrunnable.
#[derive(Debug)]
struct Doubler;

impl Operator for Doubler {
    fn op_type(&self) -> &str {
        "Doubler"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v * 2.0).collect(),
            x.shape.clone(),
        )])
    }
}

#[test]
fn a_custom_operator_outside_the_opkind_enum_still_runs() {
    let graph = Graph {
        nodes: vec![node(
            OpKind::Unknown("Doubler".to_string()),
            "n0",
            &["x"],
            &["y"],
        )],
        input_names: names(&["x"]),
        output_names: names(&["y"]),
        ..Default::default()
    };
    let mut registry = oxionnx::default_registry();
    registry.register(Box::new(Doubler));

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_registry(registry)
        .build_from_graph(graph, HashMap::new())
        .expect("session build failed");

    let out = session
        .run_one("x", Tensor::new(vec![1.5, -2.0], vec![2]))
        .expect("a registered custom operator must run");
    assert_eq!(out.get("y").expect("y").data, vec![3.0, -4.0]);
}

/// A graph output that is an *initializer* is never written by any node, yet it is
/// perfectly well defined — the strictness added to `take_outputs` must not reject
/// it.  (Constant folding promotes outputs to initializers, so this is not
/// hypothetical.)
#[test]
fn a_graph_output_that_is_an_initializer_is_returned() {
    let graph = Graph {
        nodes: vec![node(OpKind::Relu, "n0", &["x"], &["y"])],
        input_names: names(&["x"]),
        output_names: names(&["y", "constant"]),
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert("constant".to_string(), Tensor::new(vec![7.0, 8.0], vec![2]));

    let session = build(graph, weights, false);
    let out = session
        .run_one("x", Tensor::new(vec![-1.0, 2.0], vec![2]))
        .expect("an initializer output is not a missing output");
    assert_eq!(out.get("y").expect("y").data, vec![0.0, 2.0]);
    assert_eq!(out.get("constant").expect("constant").data, vec![7.0, 8.0]);
}

/// The same strictness, with the optimizer **on** (`OptLevel::All`, which is
/// `SessionBuilder`'s default): constant folding really does promote a graph
/// output to an initializer and delete the node that produced it, and dead-node
/// elimination really does remove nodes.  `take_outputs` must not mistake either
/// for "an output nothing wrote".
#[test]
fn an_optimized_model_still_returns_every_declared_output() {
    let graph = Graph {
        nodes: vec![
            // Foldable: both inputs are initializers, so this whole node can be
            // evaluated at build time and `folded` becomes a weight.
            node(OpKind::Add, "fold", &["k1", "k2"], &["folded"]),
            node(OpKind::Relu, "live", &["x"], &["y"]),
        ],
        input_names: names(&["x"]),
        output_names: names(&["y", "folded"]),
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert("k1".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
    weights.insert("k2".to_string(), Tensor::new(vec![10.0, 20.0], vec![2]));

    let session = SessionBuilder::new()
        // The builder default, spelled out because it is the point of the test.
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("build");

    let out = session
        .run_one("x", Tensor::new(vec![-3.0, 4.0], vec![2]))
        .expect("an optimized model must still produce all its declared outputs");
    assert_eq!(out.get("y").expect("y").data, vec![0.0, 4.0]);
    assert_eq!(out.get("folded").expect("folded").data, vec![11.0, 22.0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// a4-4 / a4-6 / a3-11 — subgraph outer-scope captures
// ─────────────────────────────────────────────────────────────────────────────

/// ```text
/// t = Add(x, w)          — captured by the If body, consumed by Relu
/// u = Relu(t)            — t's only *counted* consumer
/// c = Sigmoid(u)         — forces the If to be scheduled after Relu
/// y = If(c) { then_branch: Mul(t, s) }
/// ```
///
/// `ref_counts[t]` used to be 1, so `t` was taken out of the run state the moment
/// `Relu` executed — and, because `ReluOp::supports_inplace()`, mutated in place
/// first.  The If body's `Mul(t, s)` then resolved `t` to nothing and the run
/// failed with `TensorNotFound`.
///
/// `sibling` adds `p = Neg(c)` at the *same topological depth* as the If, which is
/// what forces the parallel runner down its multi-node branch — the branch whose
/// `OpContext` carried `outer_scope: None`, emptying the enclosing scope for every
/// control-flow op that happened to have a neighbour.
fn capture_graph(sibling: bool) -> Graph {
    let then_branch = Graph {
        nodes: vec![node(OpKind::Mul, "then_mul", &["t", "s"], &["then_out"])],
        output_names: names(&["then_out"]),
        ..Default::default()
    };
    let else_branch = Graph {
        nodes: vec![node(OpKind::Neg, "else_neg", &["t"], &["else_out"])],
        output_names: names(&["else_out"]),
        ..Default::default()
    };
    let mut if_attrs = Attributes::default();
    if_attrs
        .graphs
        .insert("then_branch".to_string(), then_branch);
    if_attrs
        .graphs
        .insert("else_branch".to_string(), else_branch);

    let mut nodes = vec![
        node(OpKind::Add, "add", &["x", "w"], &["t"]),
        node(OpKind::Relu, "relu", &["t"], &["u"]),
        node(OpKind::Sigmoid, "sigmoid", &["u"], &["c"]),
        node_with_attrs(OpKind::If, "branch", &["c"], &["y"], if_attrs),
    ];
    let mut outputs = names(&["u", "y"]);
    if sibling {
        nodes.push(node(OpKind::Neg, "sibling", &["c"], &["p"]));
        outputs.push("p".to_string());
    }

    Graph {
        nodes,
        input_names: names(&["x"]),
        output_names: outputs,
        ..Default::default()
    }
}

fn capture_weights() -> HashMap<String, Tensor> {
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(vec![10.0, 20.0], vec![2]));
    weights.insert("s".to_string(), Tensor::new(vec![2.0, 3.0], vec![2]));
    weights
}

/// x = [1, 2] → t = [11, 22], u = Relu(t) = [11, 22],
/// c = Sigmoid(u) ≈ [1, 1] (first element ≠ 0 → then_branch),
/// y = Mul(t, s) = [11*2, 22*3] = [22, 66].
fn assert_capture_outputs(out: &HashMap<String, Tensor>, path: &str) {
    let u = out.get("u").unwrap_or_else(|| panic!("{path}: u missing"));
    assert_eq!(u.data, vec![11.0, 22.0], "{path}: u");
    let y = out.get("y").unwrap_or_else(|| panic!("{path}: y missing"));
    assert_eq!(
        y.data,
        vec![22.0, 66.0],
        "{path}: the If body must still see the captured tensor `t`",
    );
}

#[test]
fn a_captured_intermediate_survives_until_the_subgraph_reads_it() {
    let session = build(capture_graph(false), capture_weights(), false);
    let out = session
        .run_one("x", Tensor::new(vec![1.0, 2.0], vec![2]))
        .expect("captured tensors must outlive their counted consumers");
    assert_capture_outputs(&out, "sequential");
}

#[test]
fn a_captured_intermediate_survives_under_parallel_execution() {
    let session = build(capture_graph(true), capture_weights(), true);
    let out = session
        .run_one("x", Tensor::new(vec![1.0, 2.0], vec![2]))
        .expect("control flow must work under with_parallel_execution(true)");
    assert_capture_outputs(&out, "parallel");
    // The sibling that puts the If into a multi-node depth group: Neg(Sigmoid(u)),
    // both elements of which are ≈ -1.
    let p = out.get("p").expect("p missing");
    assert_eq!(p.data.len(), 2);
    for v in &p.data {
        assert!(
            (*v + 1.0).abs() < 1e-4,
            "sibling Neg(Sigmoid(u)) should be ≈ -1, got {v}",
        );
    }
}

/// Both execution paths must agree, node for node — that is the whole point.
#[test]
fn the_two_execution_paths_agree_on_a_graph_with_control_flow() {
    let input = Tensor::new(vec![1.0, 2.0], vec![2]);
    let seq = build(capture_graph(true), capture_weights(), false)
        .run_one("x", input.clone())
        .expect("sequential");
    let par = build(capture_graph(true), capture_weights(), true)
        .run_one("x", input)
        .expect("parallel");
    assert_eq!(seq.len(), par.len());
    for (name, tensor) in &seq {
        let other = par.get(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(&tensor.data, &other.data, "{name} differs between paths");
        assert_eq!(&tensor.shape, &other.shape, "{name} shape differs");
    }
}

/// The typed path had `weights: None` *and* `outer_scope: None`, so an `If` body
/// could resolve neither an initializer nor a captured tensor.
#[test]
fn the_typed_path_resolves_subgraph_captures_and_initializers() {
    let session = build(capture_graph(false), capture_weights(), false);
    let mut inputs: HashMap<&str, TypedTensor> = HashMap::new();
    inputs.insert(
        "x",
        TypedTensor::new(TensorStorage::F32(vec![1.0, 2.0]), vec![2]),
    );
    let out = session.run_typed(&inputs).expect("typed run");
    let y = out.get("y").expect("y missing");
    assert_eq!(
        y.storage.to_f32_vec(),
        vec![22.0, 66.0],
        "the If body must resolve the captured `t` and the initializer `s`",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// a4-20 — mixed precision on the parallel path
// ─────────────────────────────────────────────────────────────────────────────

/// Two independent f16-safe nodes at one depth, so the *multi-node* parallel
/// branch runs them.  `.with_mixed_precision(true).with_parallel_execution(true)`
/// used to return untouched f32 while the same session run sequentially returned
/// f16-rounded values.
#[test]
fn mixed_precision_is_honoured_on_the_parallel_path() {
    // 0.1 is not representable in f16.
    let exact = 0.1_f32;
    let rounded = half::f16::from_f32(exact).to_f32();
    assert_ne!(exact, rounded, "0.1 must lose precision in f16");

    let graph = Graph {
        nodes: vec![
            // Relu HAS a native f16 kernel; Div does not, so it exercises the
            // "run in f32, then round the outputs" half of the policy.
            node(OpKind::Relu, "r0", &["x"], &["a"]),
            node(OpKind::Relu, "r1", &["x"], &["b"]),
            node(OpKind::Div, "d0", &["x", "ones"], &["c"]),
        ],
        input_names: names(&["x"]),
        output_names: names(&["a", "b", "c"]),
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert("ones".to_string(), Tensor::new(vec![1.0; 4], vec![4]));

    for parallel in [false, true] {
        let session = SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(parallel)
            .with_mixed_precision(true)
            .build_from_graph(graph.clone(), weights.clone())
            .expect("build");

        let out = session
            .run_one("x", Tensor::new(vec![exact; 4], vec![4]))
            .expect("run");

        for name in ["a", "b", "c"] {
            for &v in &out.get(name).unwrap_or_else(|| panic!("{name}")).data {
                assert_eq!(
                    v, rounded,
                    "{name}: mixed precision must apply with parallel={parallel}",
                );
            }
        }
    }
}

/// Without the flag the same graph keeps full f32 precision on both paths — the
/// rounding must be caused by `with_mixed_precision(true)`, not by the runner.
#[test]
fn without_the_flag_the_parallel_path_keeps_full_f32_precision() {
    let exact = 0.1_f32;
    let graph = Graph {
        nodes: vec![
            node(OpKind::Relu, "r0", &["x"], &["a"]),
            node(OpKind::Relu, "r1", &["x"], &["b"]),
        ],
        input_names: names(&["x"]),
        output_names: names(&["a", "b"]),
        ..Default::default()
    };
    let session = build(graph, HashMap::new(), true);
    let out = session
        .run_one("x", Tensor::new(vec![exact; 4], vec![4]))
        .expect("run");
    for name in ["a", "b"] {
        for &v in &out.get(name).expect("output").data {
            assert_eq!(v, exact);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// a4-15 / a4-16 — shape resolution
// ─────────────────────────────────────────────────────────────────────────────

/// `x: [batch, 4]` → `y = Relu(x)`, `z = Add(y, bias)`.
///
/// `Relu` and `Add` both support the slot-write path, which allocates each output
/// buffer from the *resolved* shape — so a run that observes another run's shapes
/// mis-sizes its buffers.
fn dynamic_batch_graph() -> Graph {
    Graph {
        nodes: vec![
            node(OpKind::Relu, "relu", &["x"], &["y"]),
            node(OpKind::Add, "add", &["y", "bias"], &["z"]),
        ],
        input_names: names(&["x"]),
        output_names: names(&["z"]),
        input_infos: vec![TensorInfo {
            name: "x".to_string(),
            dtype: DType::F32,
            shape: vec![None, Some(4)],
            dim_params: vec![Some("batch".to_string()), None],
        }],
        ..Default::default()
    }
}

fn dynamic_batch_weights() -> HashMap<String, Tensor> {
    let mut weights = HashMap::new();
    weights.insert(
        "bias".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
    );
    weights
}

/// Row `r` of the input is `[r, r+1, r+2, r+3]`; `Relu` is the identity on it and
/// the bias adds `[1, 2, 3, 4]`, so row `r` of `z` is `[r+1, r+3, r+5, r+7]`.
fn batched_input(batch: usize) -> Tensor {
    let data: Vec<f32> = (0..batch)
        .flat_map(|r| (0..4).map(move |c| (r + c) as f32))
        .collect();
    Tensor::new(data, vec![batch, 4])
}

fn expected_batched_output(batch: usize) -> Vec<f32> {
    (0..batch)
        .flat_map(|r| (0..4).map(move |c| (r + c) as f32 + (c as f32 + 1.0)))
        .collect()
}

/// **The race**: `update_dynamic_dims` wrote the session-wide shape map and the
/// run loops re-read it in a *separate* lock acquisition, so a run could execute
/// against a concurrent run's shapes — mis-sized pool acquisitions, or a spurious
/// `ShapeMismatch` from the validated write-back.
///
/// Two threads, different batch sizes, hundreds of interleaved runs, exact-value
/// assertions on every one.
#[test]
fn concurrent_runs_with_different_batch_sizes_each_get_their_own_shapes() {
    const ITERATIONS: usize = 300;

    let session = Arc::new(build(dynamic_batch_graph(), dynamic_batch_weights(), false));

    let mut handles = Vec::new();
    for batch in [1_usize, 8] {
        let session = Arc::clone(&session);
        handles.push(std::thread::spawn(move || {
            let input = batched_input(batch);
            let expected = expected_batched_output(batch);
            for i in 0..ITERATIONS {
                let out = session
                    .run_one("x", input.clone())
                    .unwrap_or_else(|e| panic!("batch {batch}, iteration {i}: {e}"));
                let z = out
                    .get("z")
                    .unwrap_or_else(|| panic!("batch {batch}, iteration {i}: z missing"));
                assert_eq!(
                    z.shape,
                    vec![batch, 4],
                    "batch {batch}, iteration {i}: wrong output shape",
                );
                assert_eq!(
                    z.data, expected,
                    "batch {batch}, iteration {i}: wrong output values",
                );
            }
        }));
    }
    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
}

/// The same, through the parallel runner.
#[test]
fn concurrent_parallel_runs_with_different_batch_sizes_are_correct() {
    const ITERATIONS: usize = 100;

    let session = Arc::new(build(dynamic_batch_graph(), dynamic_batch_weights(), true));

    let mut handles = Vec::new();
    for batch in [2_usize, 5] {
        let session = Arc::clone(&session);
        handles.push(std::thread::spawn(move || {
            let input = batched_input(batch);
            let expected = expected_batched_output(batch);
            for i in 0..ITERATIONS {
                let out = session
                    .run_one("x", input.clone())
                    .unwrap_or_else(|e| panic!("batch {batch}, iteration {i}: {e}"));
                let z = out.get("z").expect("z missing");
                assert_eq!(z.shape, vec![batch, 4]);
                assert_eq!(z.data, expected);
            }
        }));
    }
    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
}

/// **a4-16, first half**: `update_dynamic_dims` returned early when the model had
/// no *symbolic* dimensions, so a fully static model never populated
/// `resolved_shapes` at all — leaving the slot-write path and the provider shape
/// validation dead for the commonest case.
#[test]
fn a_fully_static_model_populates_the_resolved_shapes() {
    let graph = Graph {
        nodes: vec![
            node(OpKind::Relu, "relu", &["x"], &["y"]),
            node(OpKind::Add, "add", &["y", "bias"], &["z"]),
        ],
        input_names: names(&["x"]),
        output_names: names(&["z"]),
        input_infos: vec![TensorInfo {
            name: "x".to_string(),
            dtype: DType::F32,
            shape: vec![Some(2), Some(4)],
            dim_params: vec![None, None],
        }],
        ..Default::default()
    };
    let session = build(graph, dynamic_batch_weights(), false);
    assert!(
        session.resolved_shapes().is_empty(),
        "nothing is resolved before the first run",
    );

    let out = session.run_one("x", batched_input(2)).expect("run");
    assert_eq!(out.get("z").expect("z").data, expected_batched_output(2));

    let resolved = session.resolved_shapes();
    assert_eq!(
        resolved.get("y"),
        Some(&vec![2, 4]),
        "a static model must still resolve its intermediates: {resolved:?}",
    );
    assert_eq!(resolved.get("z"), Some(&vec![2, 4]));
}

/// **a4-16, second half**: with a named axis *and* an unnamed one
/// (`[batch, ?]` — `Dim::Symbol` plus `Dim::Unknown`), changing only the unnamed
/// axis left the symbolic dim map unchanged, `dims_changed` false, and the
/// previous run's shapes in force.
#[test]
fn changing_only_an_unnamed_dynamic_axis_re_resolves_the_shapes() {
    let graph = Graph {
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["y"])],
        input_names: names(&["x"]),
        output_names: names(&["y"]),
        input_infos: vec![TensorInfo {
            name: "x".to_string(),
            dtype: DType::F32,
            // [batch, ?] — the second axis has neither a value nor a param.
            shape: vec![None, None],
            dim_params: vec![Some("batch".to_string()), None],
        }],
        ..Default::default()
    };
    let session = build(graph, HashMap::new(), false);

    // batch stays 1 throughout; only the unnamed axis moves.
    for width in [3_usize, 5, 3, 7] {
        let data: Vec<f32> = (0..width).map(|i| i as f32 - 1.0).collect();
        let out = session
            .run_one("x", Tensor::new(data.clone(), vec![1, width]))
            .unwrap_or_else(|e| panic!("width {width}: {e}"));
        let y = out.get("y").expect("y missing");
        assert_eq!(y.shape, vec![1, width], "width {width}: output shape");
        let expected: Vec<f32> = data.iter().map(|v| v.max(0.0)).collect();
        assert_eq!(y.data, expected, "width {width}: output values");
        assert_eq!(
            session.resolved_shapes().get("y"),
            Some(&vec![1, width]),
            "width {width}: the resolved shape went stale",
        );
    }
}

/// **The newly-live validation.**
///
/// Populating `resolved_shapes` for static models (a4-16) switches on a check
/// that used to be dead: the parallel runner writes *every* CPU node in a
/// multi-node depth group through `write_node_outputs`, which rejects a tensor
/// whose shape disagrees with shape inference.  For a static model that lookup
/// was previously a guaranteed miss, so the check never fired.
///
/// This runs one depth group full of ops with non-trivial shape inference on
/// both paths and demands identical results.  A `ShapeMismatch` here is a
/// shape-inference bug in the op, surfaced rather than silently absorbed into a
/// mis-sized pool acquisition.
#[test]
fn the_two_paths_agree_across_op_families_with_non_trivial_shape_inference() {
    let mut concat_attrs = Attributes::default();
    concat_attrs.ints.insert("axis".to_string(), 0);
    let mut reduce_attrs = Attributes::default();
    reduce_attrs.int_lists.insert("axes".to_string(), vec![1]);

    let graph = Graph {
        nodes: vec![
            node(OpKind::MatMul, "matmul", &["a", "b"], &["o_matmul"]),
            node(OpKind::Transpose, "transpose", &["a"], &["o_transpose"]),
            node_with_attrs(
                OpKind::Concat,
                "concat",
                &["a", "a2"],
                &["o_concat"],
                concat_attrs,
            ),
            node_with_attrs(
                OpKind::ReduceMean,
                "reduce",
                &["a"],
                &["o_reduce"],
                reduce_attrs,
            ),
            node(OpKind::Softmax, "softmax", &["a"], &["o_softmax"]),
            node(OpKind::Conv, "conv", &["img", "kernel"], &["o_conv"]),
            node(OpKind::Gemm, "gemm", &["a", "b", "c"], &["o_gemm"]),
            node(OpKind::Sqrt, "sqrt", &["a2"], &["o_sqrt"]),
        ],
        input_names: names(&["a", "img"]),
        output_names: names(&[
            "o_matmul",
            "o_transpose",
            "o_concat",
            "o_reduce",
            "o_softmax",
            "o_conv",
            "o_gemm",
            "o_sqrt",
        ]),
        ..Default::default()
    };

    let mut weights = HashMap::new();
    // b: [3, 4]
    weights.insert(
        "b".to_string(),
        Tensor::new((0..12).map(|i| i as f32 * 0.25).collect(), vec![3, 4]),
    );
    // c: Gemm bias, [4]
    weights.insert(
        "c".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
    );
    // a2: [2, 3], all positive so Sqrt is defined
    weights.insert(
        "a2".to_string(),
        Tensor::new((1..7).map(|i| i as f32).collect(), vec![2, 3]),
    );
    // kernel: [1, 1, 3, 3]
    weights.insert(
        "kernel".to_string(),
        Tensor::new(vec![0.5; 9], vec![1, 1, 3, 3]),
    );

    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    // a: [2, 3]
    inputs.insert(
        "a",
        Tensor::new(vec![0.5, -1.0, 2.0, 3.0, -0.25, 1.5], vec![2, 3]),
    );
    // img: [1, 1, 4, 4]
    inputs.insert(
        "img",
        Tensor::new((0..16).map(|i| i as f32 * 0.5).collect(), vec![1, 1, 4, 4]),
    );

    let sequential = build(graph.clone(), weights.clone(), false)
        .run(&inputs)
        .expect("sequential run");
    let parallel = build(graph, weights, true)
        .run(&inputs)
        .expect("parallel run — a ShapeMismatch here is a shape-inference bug in the op");

    assert_eq!(sequential.len(), parallel.len(), "output count");
    for (name, seq) in &sequential {
        let par = parallel
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing from the parallel run"));
        assert_eq!(
            &seq.shape, &par.shape,
            "{name}: shape differs between paths"
        );
        assert_eq!(&seq.data, &par.data, "{name}: values differ between paths");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// a4-13 — run_typed must not deep-copy the weight map on every call
// ─────────────────────────────────────────────────────────────────────────────

/// Weights are no longer seeded into the typed run state, so every way of reading
/// one has to keep working: as a node input, and as a declared graph output.
#[test]
fn the_typed_path_reads_weights_without_seeding_them_into_the_run_state() {
    let graph = Graph {
        nodes: vec![node(OpKind::Add, "add", &["x", "bias"], &["z"])],
        input_names: names(&["x"]),
        // `bias` is an initializer AND a declared output.
        output_names: names(&["z", "bias"]),
        ..Default::default()
    };
    let session = build(graph, dynamic_batch_weights(), false);

    let mut inputs: HashMap<&str, TypedTensor> = HashMap::new();
    inputs.insert(
        "x",
        TypedTensor::new(TensorStorage::F32(vec![10.0, 20.0, 30.0, 40.0]), vec![4]),
    );
    let out = session.run_typed(&inputs).expect("typed run");

    assert_eq!(
        out.get("z").expect("z").storage.to_f32_vec(),
        vec![11.0, 22.0, 33.0, 44.0],
        "an initializer must resolve as a node input",
    );
    assert_eq!(
        out.get("bias").expect("bias").storage.to_f32_vec(),
        vec![1.0, 2.0, 3.0, 4.0],
        "an initializer must resolve as a graph output",
    );
}

/// The typed path must agree with the f32 path on the same model.
#[test]
fn the_typed_path_agrees_with_the_f32_path() {
    let session = build(dynamic_batch_graph(), dynamic_batch_weights(), false);

    let f32_out = session.run_one("x", batched_input(3)).expect("f32 run");

    let mut typed_inputs: HashMap<&str, TypedTensor> = HashMap::new();
    let batched = batched_input(3);
    typed_inputs.insert(
        "x",
        TypedTensor::new(TensorStorage::F32(batched.data.clone()), batched.shape),
    );
    let typed_out = session.run_typed(&typed_inputs).expect("typed run");

    assert_eq!(
        f32_out.get("z").expect("z").data,
        typed_out.get("z").expect("z").storage.to_f32_vec(),
    );
}
