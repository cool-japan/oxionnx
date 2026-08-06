//! Wave-2 engine performance work: fast-path parity and per-node dispatch cost.
//!
//! Two jobs:
//!
//! 1. **Parity.** The parallel runner grew the in-place and output-slot fast
//!    paths that until now only `run/dispatch.rs` (the sequential path) used.
//!    Those paths recycle buffers out of the pool and mutate an input tensor in
//!    place, so the one thing that must never change is the *numbers*: every
//!    test here compares the parallel result against the sequential one
//!    **bit-for-bit**, never approximately.
//! 2. **Measurement.** A ~200-node chain of trivial ops on a 16-element tensor,
//!    where per-node dispatch overhead — not arithmetic — is the whole cost.
//!    The timing tests assert only correctness (a timing assertion in a shared
//!    tree with eleven other builds running is a flake generator); the numbers
//!    are printed for `--nocapture` and recorded in the wave report.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use oxionnx::{
    default_registry, Attributes, Graph, Node, OnnxError, OpContext, OpKind, Operator,
    OperatorRegistry, OptLevel, Session, SessionBuilder, Tensor,
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

fn build(graph: Graph, weights: HashMap<String, Tensor>, parallel: bool) -> Session {
    SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(parallel)
        .with_memory_pool(true)
        .build_from_graph(graph, weights)
        .expect("session build failed")
}

fn build_with_registry(
    graph: Graph,
    weights: HashMap<String, Tensor>,
    parallel: bool,
    registry: OperatorRegistry,
) -> Session {
    SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(parallel)
        .with_memory_pool(true)
        .with_registry(registry)
        .build_from_graph(graph, weights)
        .expect("session build failed")
}

/// `x -> Relu -> Relu -> ... -> y`, `len` nodes deep.  Every level holds exactly
/// one node, so this measures pure per-node dispatch overhead.
fn chain_graph(len: usize) -> Graph {
    let mut nodes = Vec::with_capacity(len);
    for i in 0..len {
        let input = if i == 0 {
            "x".to_string()
        } else {
            format!("t{}", i - 1)
        };
        let output = if i + 1 == len {
            "y".to_string()
        } else {
            format!("t{i}")
        };
        nodes.push(node(
            OpKind::Relu,
            &format!("relu{i}"),
            &[&input],
            &[&output],
        ));
    }
    Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    }
}

/// `width` independent lanes, each `depth` ops long: every topological level
/// holds `width` nodes, which is the shape the rayon multi-node phase handles.
fn wide_graph(width: usize, depth: usize) -> Graph {
    let mut nodes = Vec::with_capacity(width * depth);
    for lane in 0..width {
        for step in 0..depth {
            let input = if step == 0 {
                "x".to_string()
            } else {
                format!("l{lane}_t{}", step - 1)
            };
            let output = if step + 1 == depth {
                format!("y{lane}")
            } else {
                format!("l{lane}_t{step}")
            };
            // Alternate Relu (slot-capable, in-place-capable) and Abs so the
            // level is not a single homogeneous op.
            let op = if step % 2 == 0 {
                OpKind::Relu
            } else {
                OpKind::Abs
            };
            nodes.push(node(op, &format!("n{lane}_{step}"), &[&input], &[&output]));
        }
    }
    Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: (0..width).map(|l| format!("y{l}")).collect(),
        ..Default::default()
    }
}

fn small_input() -> Tensor {
    Tensor::new(
        (0..16).map(|i| i as f32 - 8.0).collect::<Vec<f32>>(),
        vec![16],
    )
}

/// Minimum wall time of `reps` runs, after three warm-up runs.
///
/// Minimum, not mean: this tree is shared with other builds, and the minimum is
/// the statistic least disturbed by a neighbour saturating the machine.
fn min_run_time(session: &Session, inputs: &HashMap<&str, Tensor>, reps: usize) -> Duration {
    for _ in 0..3 {
        session.run(inputs).expect("warm-up run");
    }
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        session.run(inputs).expect("timed run");
        best = best.min(start.elapsed());
    }
    best
}

fn report(label: &str, elapsed: Duration, nodes: usize) {
    eprintln!(
        "[w2-engine-perf] {label:<28} {:>9.1} µs/run  {:>7.0} ns/node",
        elapsed.as_secs_f64() * 1e6,
        elapsed.as_secs_f64() * 1e9 / nodes as f64,
    );
}

// ── instrumented operators ───────────────────────────────────────────────────

/// A `Relu` that writes straight into a caller-provided slot and counts which
/// path the engine took.
struct SlotRelu {
    slot_calls: Arc<AtomicUsize>,
    plain_calls: Arc<AtomicUsize>,
}

impl Operator for SlotRelu {
    fn op_type(&self) -> &str {
        "Relu"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.plain_calls.fetch_add(1, Ordering::Relaxed);
        let x = ctx.input(0)?;
        let mut outputs = vec![Tensor::new(
            x.data.iter().map(|v| v.max(0.0)).collect(),
            x.shape.clone(),
        )];
        // Occupy every declared slot, elided ones included: that is the
        // positional convention `write_node_outputs` enforces, and it keeps this
        // probe usable on both the slot path and the fallback path.
        while outputs.len() < ctx.node.outputs.len() {
            outputs.push(Tensor {
                data: vec![],
                shape: vec![],
            });
        }
        Ok(outputs)
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        self.slot_calls.fetch_add(1, Ordering::Relaxed);
        let x = ctx.input(0)?;
        let slot = slots
            .first_mut()
            .ok_or_else(|| OnnxError::Internal("SlotRelu: no output slot".into()))?;
        slot.shape.clone_from(&x.shape);
        slot.data.resize(x.data.len(), 0.0);
        for (dst, src) in slot.data.iter_mut().zip(&x.data) {
            *dst = src.max(0.0);
        }
        Ok(())
    }
}

/// An `Abs` that mutates its owned input buffer and counts which path ran.
struct InplaceAbs {
    inplace_calls: Arc<AtomicUsize>,
    plain_calls: Arc<AtomicUsize>,
}

impl Operator for InplaceAbs {
    fn op_type(&self) -> &str {
        "Abs"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.plain_calls.fetch_add(1, Ordering::Relaxed);
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.abs()).collect(),
            x.shape.clone(),
        )])
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        self.inplace_calls.fetch_add(1, Ordering::Relaxed);
        for v in input.data.iter_mut() {
            *v = v.abs();
        }
        Ok(vec![input])
    }
}

#[derive(Clone, Default)]
struct PathCounters {
    slot: Arc<AtomicUsize>,
    slot_plain: Arc<AtomicUsize>,
    inplace: Arc<AtomicUsize>,
    inplace_plain: Arc<AtomicUsize>,
}

impl PathCounters {
    fn registry(&self) -> OperatorRegistry {
        let mut registry = default_registry();
        // `register` keys on `op_type()`, so these replace the stock kernels.
        registry.register(Box::new(SlotRelu {
            slot_calls: Arc::clone(&self.slot),
            plain_calls: Arc::clone(&self.slot_plain),
        }));
        registry.register(Box::new(InplaceAbs {
            inplace_calls: Arc::clone(&self.inplace),
            plain_calls: Arc::clone(&self.inplace_plain),
        }));
        registry
    }

    fn slot(&self) -> usize {
        self.slot.load(Ordering::Relaxed)
    }
    fn inplace(&self) -> usize {
        self.inplace.load(Ordering::Relaxed)
    }
}

fn run_map(session: &Session, x: &Tensor) -> HashMap<String, Tensor> {
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("x", x.clone());
    session.run(&inputs).expect("run")
}

fn assert_bit_identical(a: &HashMap<String, Tensor>, b: &HashMap<String, Tensor>, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: output count differs");
    for (name, left) in a {
        let right = b
            .get(name)
            .unwrap_or_else(|| panic!("{what}: output '{name}' missing from the other path"));
        assert_eq!(left.shape, right.shape, "{what}: shape of '{name}'");
        assert_eq!(
            left.data.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            right.data.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            "{what}: '{name}' must be bit-identical between the two paths",
        );
    }
}

// ── parity: the two execution paths must agree exactly ───────────────────────

/// A fusion-heavy graph: wide levels of slot-capable, in-place-capable
/// elementwise ops.  This is exactly the shape the parallel fast paths change,
/// so it is where a divergence would show first.
#[test]
fn the_two_paths_agree_bit_for_bit_on_a_fusion_heavy_graph() {
    let graph = wide_graph(6, 7);
    let x = small_input();

    let sequential = run_map(&build(graph.clone(), HashMap::new(), false), &x);
    let parallel = run_map(&build(graph, HashMap::new(), true), &x);

    assert_eq!(sequential.len(), 6, "every lane must produce its output");
    assert_bit_identical(&sequential, &parallel, "fusion-heavy graph");

    // And the values are the ones the ops define, not merely equal to each other.
    let expected: Vec<f32> = x.data.iter().map(|v| v.max(0.0)).collect();
    for lane in 0..6 {
        let name = format!("y{lane}");
        assert_eq!(
            parallel.get(&name).unwrap_or_else(|| panic!("{name}")).data,
            expected,
            "{name}",
        );
    }
}

/// Multi-input elementwise ops at a shared depth, with a weight and a
/// re-used intermediate, so the ref-count bookkeeping is non-trivial.
#[test]
fn the_two_paths_agree_on_a_diamond_with_shared_intermediates() {
    let graph = Graph {
        nodes: vec![
            node(OpKind::Relu, "r", &["x"], &["a"]),
            node(OpKind::Mul, "m", &["a", "w"], &["b"]),
            node(OpKind::Add, "s", &["a", "w"], &["c"]),
            node(OpKind::Sub, "d", &["b", "c"], &["e"]),
            node(OpKind::Sigmoid, "g", &["e"], &["y"]),
            node(OpKind::Tanh, "t", &["c"], &["z"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string(), "z".to_string()],
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        Tensor::new((0..16).map(|i| 0.5 + i as f32 * 0.25).collect(), vec![16]),
    );

    let x = small_input();
    let sequential = run_map(&build(graph.clone(), weights.clone(), false), &x);
    let parallel = run_map(&build(graph, weights, true), &x);
    assert_bit_identical(&sequential, &parallel, "diamond");
}

// ── the parallel fast paths really are taken ─────────────────────────────────

/// The output-slot path must be live inside the rayon multi-node phase, not
/// only for single-node levels.
#[test]
fn the_parallel_multi_node_phase_uses_the_output_slot_path() {
    // Four independent Relu lanes: one level of four nodes.
    let graph = wide_graph(4, 1);
    let x = small_input();

    let counters = PathCounters::default();
    let parallel = run_map(
        &build_with_registry(graph.clone(), HashMap::new(), true, counters.registry()),
        &x,
    );
    let slot_calls = counters.slot();

    let reference = PathCounters::default();
    let sequential = run_map(
        &build_with_registry(graph, HashMap::new(), false, reference.registry()),
        &x,
    );

    assert_eq!(
        reference.slot(),
        4,
        "the sequential path already writes into slots for all four nodes",
    );
    assert_eq!(
        slot_calls, 4,
        "the parallel multi-node phase must use the slot path for all four nodes, \
         not allocate fresh outputs",
    );
    assert_bit_identical(&sequential, &parallel, "slot path");
}

/// An **elided** (optional, `""`) output must keep its positional slot on the
/// parallel slot path, exactly as it does on the sequential one: the placeholder
/// is acquired, passed to `execute_into_slots`, and then recycled rather than
/// stored under the empty name.
#[test]
fn the_parallel_slot_path_honours_an_elided_output() {
    let mut nodes = Vec::new();
    for lane in 0..3 {
        nodes.push(Node {
            op: OpKind::Relu,
            name: format!("r{lane}"),
            inputs: vec!["x".to_string()],
            // A second, elided output — the ONNX optional-output convention.
            outputs: vec![format!("y{lane}"), String::new()],
            attrs: Attributes::default(),
        });
    }
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: (0..3).map(|l| format!("y{l}")).collect(),
        ..Default::default()
    };

    let x = small_input();
    let counters = PathCounters::default();
    let parallel = run_map(
        &build_with_registry(graph.clone(), HashMap::new(), true, counters.registry()),
        &x,
    );
    let slot_calls = counters.slot();

    let reference = PathCounters::default();
    let sequential = run_map(
        &build_with_registry(graph, HashMap::new(), false, reference.registry()),
        &x,
    );

    assert_eq!(slot_calls, 3, "all three nodes must take the slot path");
    assert_eq!(reference.slot(), 3);
    assert!(
        !parallel.contains_key(""),
        "the elided placeholder must never be stored under the empty name",
    );
    assert_bit_identical(&sequential, &parallel, "elided output");
    let expected: Vec<f32> = x.data.iter().map(|v| v.max(0.0)).collect();
    for lane in 0..3 {
        let name = format!("y{lane}");
        assert_eq!(
            parallel.get(&name).unwrap_or_else(|| panic!("{name}")).data,
            expected,
        );
    }
}

/// The in-place path must be live inside the rayon multi-node phase for a
/// tensor the ref counts prove has exactly one consumer.
#[test]
fn the_parallel_multi_node_phase_uses_the_inplace_path() {
    // `x -> Relu -> mid{lane} -> Abs -> y{lane}`: each `mid` has exactly one
    // consumer and is not a graph output, so `Abs` may take it in place.
    let mut nodes = Vec::new();
    for lane in 0..4 {
        nodes.push(node(
            OpKind::Relu,
            &format!("r{lane}"),
            &["x"],
            &[&format!("mid{lane}")],
        ));
    }
    for lane in 0..4 {
        nodes.push(node(
            OpKind::Abs,
            &format!("a{lane}"),
            &[&format!("mid{lane}")],
            &[&format!("y{lane}")],
        ));
    }
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: (0..4).map(|l| format!("y{l}")).collect(),
        ..Default::default()
    };

    let x = small_input();
    let counters = PathCounters::default();
    let parallel = run_map(
        &build_with_registry(graph.clone(), HashMap::new(), true, counters.registry()),
        &x,
    );
    let inplace_calls = counters.inplace();

    let reference = PathCounters::default();
    let sequential = run_map(
        &build_with_registry(graph, HashMap::new(), false, reference.registry()),
        &x,
    );

    assert_eq!(
        reference.inplace(),
        4,
        "the sequential path already mutates all four intermediates in place",
    );
    assert_eq!(
        inplace_calls, 4,
        "the parallel multi-node phase must mutate the four single-consumer \
         intermediates in place",
    );
    assert_bit_identical(&sequential, &parallel, "in-place path");
}

/// A tensor that is *also* a declared graph output must never be mutated in
/// place: the caller receives the value its producer wrote, not the value a
/// later consumer left behind.
///
/// `Sigmoid` is chosen because it is in-place-capable **and** changes every
/// element it touches — `sigmoid(0) == 0.5`, so a mutation of `mid` is loud.
/// (`Abs` of a `Relu` output would be indistinguishable from no mutation at all.)
#[test]
fn a_graph_output_is_never_mutated_in_place_by_a_later_node() {
    let graph = Graph {
        nodes: vec![
            // `mid` is consumed by `s` AND declared as a graph output.
            node(OpKind::Relu, "r", &["x"], &["mid"]),
            node(OpKind::Sigmoid, "s", &["mid"], &["y"]),
            // A second lane, so this level is a multi-node one on the parallel
            // path (which is where the new in-place claim lives).
            node(OpKind::Relu, "r2", &["x"], &["mid2"]),
            node(OpKind::Sigmoid, "s2", &["mid2"], &["y2"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec![
            "mid".to_string(),
            "y".to_string(),
            "mid2".to_string(),
            "y2".to_string(),
        ],
        ..Default::default()
    };

    let x = small_input();
    let relu: Vec<f32> = x.data.iter().map(|v| v.max(0.0)).collect();
    let sigmoid: Vec<f32> = relu.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();

    for parallel in [false, true] {
        let outputs = run_map(&build(graph.clone(), HashMap::new(), parallel), &x);
        for name in ["mid", "mid2"] {
            assert_eq!(
                outputs.get(name).unwrap_or_else(|| panic!("{name}")).data,
                relu,
                "parallel={parallel}: '{name}' is a declared output and must not be \
                 mutated in place by its consumer",
            );
        }
        for name in ["y", "y2"] {
            assert_eq!(
                outputs.get(name).unwrap_or_else(|| panic!("{name}")).data,
                sigmoid,
                "parallel={parallel}: '{name}'",
            );
        }
    }
}

// ── control flow: the subgraph scope must still resolve ──────────────────────

/// A `Loop` whose body contains an `If` that reads a name from the *enclosing
/// run* scope, and another that reads a name produced earlier **inside the loop
/// body**.  Both must resolve: the merged scope handed to a nested control-flow
/// operator is the outer scope plus the body's intermediates so far.
#[test]
fn a_nested_if_inside_a_loop_body_sees_both_scopes() {
    let mut then_attrs = Attributes::default();
    then_attrs.graphs.insert(
        "then_branch".to_string(),
        Graph {
            // `scale` comes from the enclosing run scope; `stepped` is produced
            // earlier in the loop body.
            nodes: vec![node(OpKind::Mul, "mul", &["stepped", "scale"], &["chosen"])],
            input_names: vec![],
            output_names: vec!["chosen".to_string()],
            ..Default::default()
        },
    );
    then_attrs.graphs.insert(
        "else_branch".to_string(),
        Graph {
            nodes: vec![node(OpKind::Identity, "id", &["stepped"], &["chosen"])],
            input_names: vec![],
            output_names: vec!["chosen".to_string()],
            ..Default::default()
        },
    );

    let mut body_attrs = Attributes::default();
    body_attrs.graphs.insert(
        "body".to_string(),
        Graph {
            nodes: vec![
                node(OpKind::Add, "step", &["accum", "one"], &["stepped"]),
                Node {
                    op: OpKind::If,
                    name: "pick".to_string(),
                    inputs: vec!["cond_in".to_string()],
                    outputs: vec!["accum_out".to_string()],
                    attrs: then_attrs,
                },
                node(OpKind::Identity, "keep", &["cond_in"], &["cond_out"]),
            ],
            input_names: vec![
                "iter".to_string(),
                "cond_in".to_string(),
                "accum".to_string(),
            ],
            output_names: vec!["cond_out".to_string(), "accum_out".to_string()],
            ..Default::default()
        },
    );

    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Loop,
            name: "loop".to_string(),
            inputs: vec!["trip".to_string(), "cond".to_string(), "init".to_string()],
            outputs: vec!["total".to_string()],
            attrs: body_attrs,
        }],
        input_names: vec!["trip".to_string(), "cond".to_string(), "init".to_string()],
        output_names: vec!["total".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert("one".to_string(), Tensor::scalar(1.0));
    weights.insert("scale".to_string(), Tensor::scalar(2.0));

    for parallel in [false, true] {
        let session = build(graph.clone(), weights.clone(), parallel);
        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert("trip", Tensor::scalar(3.0));
        inputs.insert("cond", Tensor::scalar(1.0));
        inputs.insert("init", Tensor::scalar(0.0));
        let outputs = session.run(&inputs).expect("loop with nested If");
        // accum: 0 -> (0+1)*2 = 2 -> (2+1)*2 = 6 -> (6+1)*2 = 14
        assert_eq!(
            outputs.get("total").expect("total").data,
            vec![14.0],
            "parallel={parallel}: the nested If must see both scopes",
        );
    }
}

// ── measurement (printed, never asserted) ────────────────────────────────────

/// Per-node dispatch overhead on a 200-node chain of trivial ops over a
/// 16-element tensor: the arithmetic is ~16 comparisons, so essentially all of
/// the measured time is engine overhead.
///
/// `#[ignore]`: 800 timed inferences do not belong in the permanent gate.  Run
/// with `cargo nextest run --test w2_engine_perf --run-ignored all --no-capture`
/// (or `cargo test -- --ignored --nocapture`) when you want the numbers.
#[test]
#[ignore = "measurement, not a gate — run explicitly with --run-ignored"]
fn microbench_per_node_dispatch_overhead() {
    const NODES: usize = 200;
    const REPS: usize = 200;

    let x = small_input();
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("x", x);

    let sequential = build(chain_graph(NODES), HashMap::new(), false);
    let elapsed = min_run_time(&sequential, &inputs, REPS);
    report("chain-200 sequential", elapsed, NODES);
    let out = sequential.run(&inputs).expect("run");
    assert_eq!(out.get("y").expect("y").data.len(), 16);

    let parallel = build(chain_graph(NODES), HashMap::new(), true);
    let elapsed = min_run_time(&parallel, &inputs, REPS);
    report("chain-200 parallel", elapsed, NODES);

    // 8 lanes x 25 deep: 25 levels of 8 nodes, i.e. the rayon multi-node phase.
    let wide_nodes = 8 * 25;
    let wide_parallel = build(wide_graph(8, 25), HashMap::new(), true);
    let elapsed = min_run_time(&wide_parallel, &inputs, REPS);
    report("wide-8x25 parallel", elapsed, wide_nodes);

    let wide_sequential = build(wide_graph(8, 25), HashMap::new(), false);
    let elapsed = min_run_time(&wide_sequential, &inputs, REPS);
    report("wide-8x25 sequential", elapsed, wide_nodes);
}

/// A `Loop` body of ordinary math, iterated many times: the cost of
/// `execute_subgraph`'s scope handling, which used to deep-copy the enclosing
/// scope once per body node per iteration.
///
/// `#[ignore]` for the same reason as
/// [`microbench_per_node_dispatch_overhead`] — but note that the *correctness*
/// half of what it measures (a loop over a fat enclosing scope returning the
/// right sum) is asserted by `a_nested_if_inside_a_loop_body_sees_both_scopes`,
/// which does run in the gate.
#[test]
#[ignore = "measurement, not a gate — run explicitly with --run-ignored"]
fn microbench_loop_body_scope_overhead() {
    const BODY_NODES: usize = 12;
    const TRIPS: f32 = 64.0;
    const REPS: usize = 50;

    let mut body_nodes = vec![node(OpKind::Identity, "keep", &["cond_in"], &["cond_out"])];
    for i in 0..BODY_NODES {
        let input = if i == 0 {
            "accum".to_string()
        } else {
            format!("s{}", i - 1)
        };
        let output = if i + 1 == BODY_NODES {
            "accum_out".to_string()
        } else {
            format!("s{i}")
        };
        body_nodes.push(node(
            OpKind::Add,
            &format!("add{i}"),
            &[&input, "one"],
            &[&output],
        ));
    }

    let mut attrs = Attributes::default();
    attrs.graphs.insert(
        "body".to_string(),
        Graph {
            nodes: body_nodes,
            input_names: vec![
                "iter".to_string(),
                "cond_in".to_string(),
                "accum".to_string(),
            ],
            output_names: vec!["cond_out".to_string(), "accum_out".to_string()],
            ..Default::default()
        },
    );

    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Loop,
            name: "loop".to_string(),
            inputs: vec!["trip".to_string(), "cond".to_string(), "init".to_string()],
            outputs: vec!["total".to_string()],
            attrs,
        }],
        input_names: vec!["trip".to_string(), "cond".to_string(), "init".to_string()],
        output_names: vec!["total".to_string()],
        ..Default::default()
    };

    // A fat enclosing scope: `execute_subgraph` used to clone all of it, once
    // per body node, per iteration.
    let mut weights = HashMap::new();
    weights.insert("one".to_string(), Tensor::scalar(1.0));
    for i in 0..64 {
        weights.insert(
            format!("ballast{i}"),
            Tensor::new(vec![i as f32; 256], vec![256]),
        );
    }

    let session = build(graph, weights, false);
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("trip", Tensor::scalar(TRIPS));
    inputs.insert("cond", Tensor::scalar(1.0));
    inputs.insert("init", Tensor::scalar(0.0));

    let outputs = session.run(&inputs).expect("loop run");
    assert_eq!(
        outputs.get("total").expect("total").data,
        vec![TRIPS * BODY_NODES as f32],
        "each iteration adds one per body node",
    );

    let elapsed = min_run_time(&session, &inputs, REPS);
    report(
        "loop-64x12 body",
        elapsed,
        TRIPS as usize * (BODY_NODES + 1),
    );
}
