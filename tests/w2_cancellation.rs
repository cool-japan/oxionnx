//! Cooperative cancellation: [`CancellationToken`] bound to a session via
//! [`oxionnx::SessionBuilder::with_session_cancellation`].
//!
//! The claim under test is precise — *the run stops at a **node boundary***, not
//! "somewhere eventually" — so most of these tests are built around a tripwire
//! operator that cancels the token **from inside the graph**, at a known node.
//! What ran and what did not is then an exact, deterministic assertion rather
//! than a race with a background thread.

use oxionnx::{
    Attributes, CancellationToken, Graph, Node, OnnxError, OpContext, OpKind, Operator,
    OperatorRegistry, OptLevel, Session, Tensor,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ── instrumentation operators ───────────────────────────────────────────────

/// Passes its input through and counts that it ran.
struct Tally {
    name: String,
    count: Arc<AtomicUsize>,
}

impl Operator for Tally {
    fn op_type(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(vec![ctx.input(0)?.clone()])
    }
}

/// Passes its input through and cancels the token as a side effect — a
/// "cancel arrives exactly here" event with no thread timing involved.
struct TripWire {
    token: CancellationToken,
}

impl Operator for TripWire {
    fn op_type(&self) -> &str {
        "TripWire"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.token.cancel();
        Ok(vec![ctx.input(0)?.clone()])
    }
}

// ── graph helpers ───────────────────────────────────────────────────────────

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn input_x() -> HashMap<&'static str, Tensor> {
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(vec![-2.0, 0.5, 3.0, -4.0], vec![4]));
    inputs
}

/// `x → Tally(before) → TripWire → Tally(after) → y`
///
/// `before` must run, `after` must not.
fn tripwire_chain(
    token: &CancellationToken,
) -> (Graph, OperatorRegistry, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let before = Arc::new(AtomicUsize::new(0));
    let after = Arc::new(AtomicUsize::new(0));

    let mut registry = oxionnx::default_registry();
    registry.register_as(
        "TallyBefore",
        Box::new(Tally {
            name: "TallyBefore".to_string(),
            count: Arc::clone(&before),
        }),
    );
    registry.register_as(
        "TallyAfter",
        Box::new(Tally {
            name: "TallyAfter".to_string(),
            count: Arc::clone(&after),
        }),
    );
    registry.register_as(
        "TripWire",
        Box::new(TripWire {
            token: token.clone(),
        }),
    );

    let graph = Graph {
        name: "tripwire".to_string(),
        nodes: vec![
            node(OpKind::parse("TallyBefore"), "before", &["x"], &["a"]),
            node(OpKind::parse("TripWire"), "trip", &["a"], &["b"]),
            node(OpKind::parse("TallyAfter"), "after", &["b"], &["y"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    (graph, registry, before, after)
}

// ── the core claim ──────────────────────────────────────────────────────────

#[test]
fn a_run_stops_at_the_first_node_boundary_after_the_token_is_cancelled() {
    let token = CancellationToken::new();
    let (graph, registry, before, after) = tripwire_chain(&token);

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_registry(registry)
        .with_session_cancellation(token.clone())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    match session.run(&input_x()) {
        Err(OnnxError::Cancelled(msg)) => {
            assert!(
                msg.contains("'after'"),
                "the error must name the node the run stopped in front of; got: {msg}"
            );
        }
        other => panic!("expected Cancelled, got {}", describe(&other)),
    }

    assert_eq!(
        before.load(Ordering::SeqCst),
        1,
        "the node before the trip ran"
    );
    assert_eq!(
        after.load(Ordering::SeqCst),
        0,
        "the node after the trip must NOT have run"
    );
}

#[test]
fn the_same_stop_happens_on_the_parallel_execution_path() {
    let token = CancellationToken::new();
    let (graph, registry, before, after) = tripwire_chain(&token);

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_registry(registry)
        .with_parallel_execution(true)
        .with_session_cancellation(token.clone())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
    assert_eq!(before.load(Ordering::SeqCst), 1);
    assert_eq!(after.load(Ordering::SeqCst), 0);
}

#[test]
fn a_token_cancelled_before_the_run_stops_it_at_the_very_first_node() {
    let token = CancellationToken::new();
    let (graph, registry, before, after) = tripwire_chain(&token);

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_registry(registry)
        .with_session_cancellation(token.clone())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    token.cancel();
    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
    assert_eq!(before.load(Ordering::SeqCst), 0, "not a single node ran");
    assert_eq!(after.load(Ordering::SeqCst), 0);
}

#[test]
fn cancelling_from_another_thread_is_observed_by_the_run() {
    let token = CancellationToken::new();
    let (graph, registry, _before, after) = tripwire_chain(&token);
    let session = Arc::new(
        Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_registry(registry)
            .with_session_cancellation(token.clone())
            .build_from_graph(graph, HashMap::new())
            .expect("session builds"),
    );

    let remote = token.clone();
    std::thread::spawn(move || remote.cancel())
        .join()
        .expect("the cancelling thread finishes");

    let inputs: HashMap<String, Tensor> = input_x()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let outputs = oxionnx::block_on(session.run_async(inputs));
    assert!(matches!(outputs, Err(OnnxError::Cancelled(_))));
    assert_eq!(after.load(Ordering::SeqCst), 0);
}

#[test]
fn resetting_the_token_makes_the_session_runnable_again() {
    let token = CancellationToken::new();
    let (graph, registry, before, _after) = tripwire_chain(&token);
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_registry(registry)
        .with_session_cancellation(token.clone())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    token.cancel();
    assert!(session.run(&input_x()).is_err());
    assert_eq!(before.load(Ordering::SeqCst), 0);

    token.reset();
    // The graph's own tripwire cancels again, so the run still stops — but it
    // now gets *past the first node*, which is the point.
    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
    assert_eq!(before.load(Ordering::SeqCst), 1);
}

// ── the guard must be invisible when it does not fire ───────────────────────

/// Every dispatch fast path (in-place, output-slot, plain execute) must survive
/// the wrapping: same numbers, bit for bit.
#[test]
fn an_uncancelled_session_returns_bit_identical_results() {
    let graph = Graph {
        name: "mixed".to_string(),
        nodes: vec![
            // Relu supports in-place; Softmax supports output slots; Transpose
            // takes the plain path. All three are exercised in one run.
            node(OpKind::Relu, "relu", &["x"], &["r"]),
            node(OpKind::Softmax, "softmax", &["r"], &["s"]),
            node(OpKind::Sqrt, "sqrt", &["s"], &["y"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };

    // Everything except the cancellation binding is held equal, so any
    // difference in the numbers can only come from the guard.
    let plain = Session::builder()
        .build_from_graph(graph.clone(), HashMap::new())
        .expect("plain session");
    let guarded = Session::builder()
        .with_session_cancellation(CancellationToken::new())
        .build_from_graph(graph, HashMap::new())
        .expect("guarded session");

    let plain_out = plain.run(&input_x()).expect("plain run");
    let guarded_out = guarded.run(&input_x()).expect("guarded run");

    assert_eq!(
        plain_out["y"].shape, guarded_out["y"].shape,
        "shapes must match"
    );
    assert_eq!(
        plain_out["y"]
            .data
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        guarded_out["y"]
            .data
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "a cancellation guard must not perturb a single bit of the result"
    );
}

/// The zero-copy and in-place fast paths are chosen from operator *predicates*.
/// A guard that forgot to forward them would silently disable both.
#[test]
fn the_guard_forwards_the_dispatch_fast_path_predicates() {
    let inner = oxionnx::default_registry();
    let probes = ["Relu", "Add", "Softmax", "MatMul", "Transpose", "Sqrt"];

    let graph = Graph {
        name: "probe".to_string(),
        nodes: probes
            .iter()
            .enumerate()
            .map(|(i, op)| {
                node(
                    OpKind::parse(op),
                    &format!("n{i}"),
                    &["x", "x"],
                    &[&format!("o{i}")],
                )
            })
            .collect(),
        input_names: vec!["x".to_string()],
        output_names: probes
            .iter()
            .enumerate()
            .map(|(i, _)| format!("o{i}"))
            .collect(),
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_session_cancellation(CancellationToken::new())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");
    let wrapped = session.operator_registry();

    for op in probes {
        let raw = inner
            .get(op)
            .unwrap_or_else(|| panic!("{op} is registered"));
        let guard = wrapped
            .get(op)
            .unwrap_or_else(|| panic!("{op} survives wrapping"));
        assert_eq!(guard.op_type(), raw.op_type(), "{op}: op_type");
        assert_eq!(
            guard.supports_inplace(),
            raw.supports_inplace(),
            "{op}: supports_inplace must be forwarded, or the in-place path dies"
        );
        assert_eq!(
            guard.supports_output_slots(),
            raw.supports_output_slots(),
            "{op}: supports_output_slots must be forwarded, or the zero-copy path dies"
        );
        assert_eq!(
            guard.native_dtypes(),
            raw.native_dtypes(),
            "{op}: native_dtypes must be forwarded, or typed dispatch loses its fast path"
        );
    }
}

// ── the guard must not change error behaviour ───────────────────────────────

/// An operator this engine does not implement must still produce
/// `UnsupportedOp` — the guard registry only wraps names the inner registry
/// actually provides, precisely so this stays true.
#[test]
fn an_unimplemented_operator_still_reports_unsupported_op() {
    let graph = Graph {
        name: "unknown".to_string(),
        nodes: vec![node(
            OpKind::parse("DefinitelyNotAnOnnxOperator"),
            "mystery",
            &["x"],
            &["y"],
        )],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_session_cancellation(CancellationToken::new())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    match session.run(&input_x()) {
        Err(OnnxError::UnsupportedOp(msg)) => {
            assert!(msg.contains("DefinitelyNotAnOnnxOperator"), "got: {msg}");
        }
        other => panic!("expected UnsupportedOp, got {}", describe(&other)),
    }
}

// ── subgraphs ───────────────────────────────────────────────────────────────

/// A subgraph body's operators must be wrapped too — and, critically, must
/// still *resolve*: an unwrapped nested op would turn a working `If` into
/// `UnsupportedOp`.
#[test]
fn a_subgraph_body_still_runs_and_is_itself_a_cancellation_point() {
    let then_branch = Graph {
        // `Neg` appears ONLY inside this branch, never at the top level — the
        // case a non-recursive registry walk would break.
        nodes: vec![node(OpKind::Neg, "neg", &["x"], &["t"])],
        output_names: vec!["t".to_string()],
        ..Default::default()
    };
    let else_branch = Graph {
        nodes: vec![node(OpKind::Sqrt, "sqrt", &["x"], &["e"])],
        output_names: vec!["e".to_string()],
        ..Default::default()
    };
    let mut if_attrs = Attributes::default();
    if_attrs
        .graphs
        .insert("then_branch".to_string(), then_branch);
    if_attrs
        .graphs
        .insert("else_branch".to_string(), else_branch);

    let graph = Graph {
        name: "branchy".to_string(),
        nodes: vec![Node {
            op: OpKind::If,
            name: "if".to_string(),
            inputs: vec!["cond".to_string()],
            outputs: vec!["y".to_string()],
            attrs: if_attrs,
        }],
        input_names: vec!["cond".to_string(), "x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };

    let token = CancellationToken::new();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_session_cancellation(token.clone())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    let mut inputs = HashMap::new();
    inputs.insert("cond", Tensor::new(vec![1.0], vec![1]));
    inputs.insert("x", Tensor::new(vec![1.0, -2.0, 3.0], vec![3]));

    let outputs = session
        .run(&inputs)
        .expect("the branch body must still run");
    assert_eq!(
        outputs["y"].data,
        vec![-1.0, 2.0, -3.0],
        "then_branch = Neg"
    );

    token.cancel();
    assert!(matches!(session.run(&inputs), Err(OnnxError::Cancelled(_))));
}

// ── binding mechanics ───────────────────────────────────────────────────────

#[test]
fn the_bound_token_is_reported_back_and_rebinding_the_same_one_is_a_no_op() {
    let token = CancellationToken::new();
    let graph = Graph {
        name: "relu".to_string(),
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let mut session = Session::from_graph(graph, HashMap::new()).expect("session builds");
    assert!(session.session_cancellation_token().is_none());

    session.set_session_cancellation(token.clone());
    assert!(session.session_cancellation_token().is_some());

    // Re-binding the same token must not stack a second guard layer.
    session.set_session_cancellation(token.clone());
    session.set_session_cancellation(token.clone());

    let before = session.run(&input_x()).expect("still runnable");
    assert_eq!(before["y"].data, vec![0.0, 0.5, 3.0, 0.0]);

    token.cancel();
    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
}

/// `OnnxError::Cancelled` was a declared-but-never-constructed variant before
/// this feature existed; make sure it is now reachable through the public API
/// and renders sensibly.
#[test]
fn the_cancelled_error_variant_is_reachable_and_readable() {
    let token = CancellationToken::new();
    let graph = Graph {
        name: "relu".to_string(),
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let session = Session::builder()
        .with_session_cancellation(token.clone())
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");
    token.cancel();

    let err = session.run(&input_x()).expect_err("cancelled");
    let rendered = err.to_string();
    assert!(rendered.starts_with("Cancelled:"), "got: {rendered}");
    assert!(rendered.contains("relu"), "got: {rendered}");
}

fn describe(result: &Result<HashMap<String, Tensor>, OnnxError>) -> String {
    match result {
        Ok(map) => format!("Ok with outputs {:?}", map.keys().collect::<Vec<_>>()),
        Err(e) => format!("Err({e})"),
    }
}

// ── cost of the guard ───────────────────────────────────────────────────────

/// The guard is one extra registry lookup per node, paid only by sessions that
/// asked for cancellation. This pins that down with a number instead of a
/// hand-wave, and would catch a future change that made it *structurally* more
/// expensive (an extra allocation per node, a lock, a guard-on-guard stack).
///
/// # Why min-of-N and not a single timing
///
/// This runs inside a test suite that saturates every core, so any *single*
/// measurement is really "how much CPU did the scheduler give this thread",
/// which regularly differs by 100x between two adjacent loops. The estimator is
/// therefore the **minimum** over several short, interleaved rounds: preemption
/// can only ever make a round slower, so the minimum converges on the real cost
/// and is not perturbed by a busy machine. Alternating the two sides within each
/// round keeps them under the same conditions. The bound is deliberately an
/// order of magnitude — the purpose is to catch a *structural* regression, not
/// to benchmark.
#[test]
fn the_cancellation_guard_costs_one_lookup_per_node_not_an_order_of_magnitude() {
    const DEPTH: usize = 300;
    const RUNS: usize = 40;
    const ROUNDS: usize = 9;

    let mut nodes = Vec::with_capacity(DEPTH);
    for i in 0..DEPTH {
        let input = if i == 0 {
            "x".to_string()
        } else {
            format!("h{}", i - 1)
        };
        let output = if i + 1 == DEPTH {
            "y".to_string()
        } else {
            format!("h{i}")
        };
        nodes.push(node(
            OpKind::Relu,
            &format!("relu{i}"),
            &[&input],
            &[&output],
        ));
    }
    let graph = Graph {
        name: "deep".to_string(),
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };

    let plain = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph.clone(), HashMap::new())
        .expect("plain session");
    let guarded = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_session_cancellation(CancellationToken::new())
        .build_from_graph(graph, HashMap::new())
        .expect("guarded session");

    let round = |session: &Session| {
        let start = std::time::Instant::now();
        for _ in 0..RUNS {
            session.run(&input_x()).expect("run succeeds");
        }
        start.elapsed().as_secs_f64()
    };

    // Warm-up: first-touch page faults and branch predictors, not the measurement.
    round(&plain);
    round(&guarded);

    let mut plain_best = f64::INFINITY;
    let mut guarded_best = f64::INFINITY;
    for _ in 0..ROUNDS {
        plain_best = plain_best.min(round(&plain));
        guarded_best = guarded_best.min(round(&guarded));
    }

    let per_node_overhead = (guarded_best - plain_best) / (RUNS * DEPTH) as f64;
    println!(
        "cancellation guard, {DEPTH}-node chain, best of {ROUNDS} x {RUNS} runs: \
         plain {:.3} ms, guarded {:.3} ms ({:+.1}%, {:+.1} ns/node)",
        plain_best * 1e3,
        guarded_best * 1e3,
        100.0 * (guarded_best / plain_best - 1.0),
        per_node_overhead * 1e9
    );

    assert!(
        guarded_best < plain_best * 10.0,
        "the guard must stay a per-node lookup: plain {plain_best:.6}s, guarded {guarded_best:.6}s"
    );
}
