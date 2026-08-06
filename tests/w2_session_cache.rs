//! Session caching: `Session::save_optimized` / `load_optimized`.
//!
//! Two claims are under test, and they are different claims:
//!
//! 1. **Round trip fidelity** — a reloaded session produces bit-identical
//!    outputs for the same inputs, including graphs with nested subgraphs and
//!    non-default model metadata.
//! 2. **The optimization pipeline is actually skipped** — not "is fast", but
//!    *does not run*. That is proved by counting operator executions: constant
//!    folding executes ops through the registry, so a registry that counts its
//!    own invocations sees a non-zero count when the model is built from a graph
//!    and **exactly zero** when the same model is loaded from its cache.

use oxionnx::{
    Attributes, Graph, Node, OnnxError, OpContext, OpKind, Operator, OperatorRegistry, OptLevel,
    Session, SessionCacheHeader, Tensor, SESSION_CACHE_FORMAT_VERSION,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ── an operator that reports whether the optimizer ran it ───────────────────

/// A drop-in replacement for `Add` that counts every execution.
///
/// Registered over the real `Add`, it turns "did constant folding run?" into an
/// exact integer instead of a timing guess: `constant_fold` evaluates a foldable
/// node by looking its operator up in the registry and calling `execute`.
struct CountingAdd {
    count: Arc<AtomicUsize>,
}

impl Operator for CountingAdd {
    fn op_type(&self) -> &str {
        "Add"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        if a.shape != b.shape {
            return Err(OnnxError::ShapeMismatch(format!(
                "CountingAdd: {:?} vs {:?}",
                a.shape, b.shape
            )));
        }
        let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        Ok(vec![Tensor::new(data, a.shape.clone())])
    }
}

fn counting_registry(count: &Arc<AtomicUsize>) -> OperatorRegistry {
    let mut registry = oxionnx::default_registry();
    registry.register_as(
        "Add",
        Box::new(CountingAdd {
            count: Arc::clone(count),
        }),
    );
    registry
}

// ── graphs ──────────────────────────────────────────────────────────────────

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// `y = x * (c1 + c2)` — the `Add` is entirely constant, so the optimizer folds
/// it away and the cache should contain one node, not two.
fn foldable_graph(chain: usize) -> (Graph, HashMap<String, Tensor>) {
    let mut weights = HashMap::new();
    weights.insert("c0".to_string(), Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
    let mut nodes = Vec::new();
    for i in 0..chain {
        weights.insert(
            format!("k{i}"),
            Tensor::new(vec![0.5, 0.25, 0.125], vec![3]),
        );
        nodes.push(node(
            OpKind::Add,
            &format!("fold{i}"),
            &[&format!("c{i}"), &format!("k{i}")],
            &[&format!("c{}", i + 1)],
        ));
    }
    nodes.push(node(
        OpKind::Mul,
        "scale",
        &["x", &format!("c{chain}")],
        &["y"],
    ));

    let graph = Graph {
        name: "foldable".to_string(),
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    (graph, weights)
}

fn inputs() -> HashMap<&'static str, Tensor> {
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(vec![2.0, -3.0, 4.0], vec![3]));
    inputs
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "oxionnx_w2_cache_{}_{}_{name}.oxs",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    path
}

fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor.data.iter().map(|v| v.to_bits()).collect()
}

// ── 1. round trip fidelity ──────────────────────────────────────────────────

#[test]
fn a_reloaded_session_produces_bit_identical_outputs() {
    let (graph, weights) = foldable_graph(3);
    let original = Session::from_graph(graph, weights).expect("session builds");
    let expected = original.run(&inputs()).expect("original run");

    let path = temp_path("roundtrip");
    original.save_optimized(&path).expect("cache is written");
    let reloaded = Session::load_optimized(&path).expect("cache loads");
    let actual = reloaded.run(&inputs()).expect("reloaded run");
    let _ = std::fs::remove_file(&path);

    assert_eq!(actual.len(), expected.len(), "same output names");
    for (name, expected_tensor) in &expected {
        let actual_tensor = actual.get(name).expect("output present after reload");
        assert_eq!(actual_tensor.shape, expected_tensor.shape, "{name}: shape");
        assert_eq!(
            bits(actual_tensor),
            bits(expected_tensor),
            "{name}: a reloaded session must reproduce every bit"
        );
    }
}

#[test]
fn the_cache_holds_the_optimized_graph_not_the_original_one() {
    let (graph, weights) = foldable_graph(3);
    let unoptimized = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph.clone(), weights.clone())
        .expect("unoptimized session");
    let optimized = Session::from_graph(graph, weights).expect("optimized session");

    assert_eq!(
        unoptimized.nodes().len(),
        4,
        "3 Adds + 1 Mul before folding"
    );
    assert_eq!(optimized.nodes().len(), 1, "the Adds fold into a constant");

    let bytes = optimized.to_optimized_bytes().expect("serialises");
    let header = Session::peek_optimized_header(&bytes).expect("header decodes");
    assert_eq!(
        header,
        SessionCacheHeader {
            format_version: SESSION_CACHE_FORMAT_VERSION,
            node_count: 1,
            weight_count: optimized.weights().len() as u64,
        }
    );

    let reloaded = Session::from_optimized_bytes(&bytes).expect("cache loads");
    assert_eq!(
        reloaded.nodes().len(),
        1,
        "the reloaded graph is the optimized one"
    );
}

// ── 2. the pipeline is skipped, measured by a counter ───────────────────────

#[test]
fn loading_a_cache_executes_zero_operators_where_building_executes_many() {
    let (graph, weights) = foldable_graph(8);

    let build_count = Arc::new(AtomicUsize::new(0));
    let built = Session::builder()
        .with_registry(counting_registry(&build_count))
        .build_from_graph(graph, weights)
        .expect("session builds");
    let folded = build_count.load(Ordering::SeqCst);
    assert_eq!(
        folded, 8,
        "constant folding must have executed every foldable Add exactly once"
    );
    assert_eq!(built.nodes().len(), 1, "all eight Adds folded away");

    let bytes = built.to_optimized_bytes().expect("serialises");

    let load_count = Arc::new(AtomicUsize::new(0));
    let loaded = Session::builder()
        .with_registry(counting_registry(&load_count))
        .load_optimized_from_bytes(&bytes)
        .expect("cache loads");
    assert_eq!(
        load_count.load(Ordering::SeqCst),
        0,
        "loading a cache must not execute a single operator — that is the whole point"
    );
    assert_eq!(loaded.nodes().len(), 1);
}

/// The same claim as a wall-clock measurement, printed for the record — this
/// test carries **no** pass/fail assertion on the ratio itself, deliberately.
///
/// The estimator is the **minimum** over several rounds, for the reason spelled
/// out in `tests/w2_cancellation.rs`: this suite saturates every core, a single
/// `Instant` pair really measures "how much CPU did the scheduler hand this
/// thread", and preemption can only ever make a round *slower* — so the minimum
/// converges on the real cost while a single sample does not. Unlike a plain
/// build/load call, each round here repeats the operation `RUNS` times and
/// times the batch: a single unlucky context switch then only dilutes one
/// round's total instead of dominating it, the same defense the cancellation
/// test uses.
///
/// Why there is no threshold any more: `Session::build_from_graph` — the
/// function *both* `Session::from_graph` (the "build" side below) and
/// `Session::from_optimized_bytes` (the "load" side) funnel through —
/// unconditionally acquires a full `wgpu::Device`/`Queue`/`Instance` and
/// rebuilds 20+ compute pipelines on *every* call, whenever the `gpu` feature
/// is enabled and a real adapter is reachable (see `TODO.md` §19). That cost is
/// identical on both sides of this comparison, and once a real adapter is
/// present it swamps the thing this test is actually trying to isolate.
/// Measured directly on this exact benchmark: `--no-default-features` (no
/// adapter reachable) gives build ≈23ms / load ≈2.5ms — a ≈9.3x speed-up that
/// reflects the optimizer-skip this test exists to show. `--all-features` with
/// a real adapter present gives build ≈1.70s / load ≈1.78s — ≈0.95x, i.e. load
/// samples marginally *slower* than build, because both numbers are now
/// dominated by the same GPU-acquisition constant and only scheduler/driver
/// jitter separates them. No fixed `MIN_SPEEDUP` is honest across both
/// configurations: one loose enough to survive the GPU-present case is loose
/// enough to be meaningless in the no-GPU case, and one tuned to the no-GPU
/// case is a coin flip once a real adapter shows up — that isn't a threshold
/// to re-tune, it's proof the two configurations are no longer measuring the
/// same thing. So this test now only prints the ratio; the actual,
/// GPU-independent pass/fail proof of "loading skips the optimizer" is
/// `loading_a_cache_executes_zero_operators_where_building_executes_many`
/// above (operator executions: 8 vs 0), which this GPU-acquisition cost does
/// not touch at all.
#[test]
fn loading_a_cache_is_measurably_cheaper_than_optimizing_from_scratch() {
    const RUNS: usize = 10;
    const ROUNDS: usize = 9;
    let (graph, weights) = foldable_graph(400);

    let build_round = || {
        let start = std::time::Instant::now();
        let mut session = None;
        for _ in 0..RUNS {
            session =
                Some(Session::from_graph(graph.clone(), weights.clone()).expect("session builds"));
        }
        (start.elapsed().as_secs_f64(), session.expect("RUNS > 0"))
    };
    let (_, built) = build_round();
    let bytes = built.to_optimized_bytes().expect("serialises");

    let load_round = || {
        let start = std::time::Instant::now();
        let mut session = None;
        for _ in 0..RUNS {
            session = Some(Session::from_optimized_bytes(&bytes).expect("cache loads"));
        }
        (start.elapsed().as_secs_f64(), session.expect("RUNS > 0"))
    };
    // Warm-up: neither side should pay first-touch costs in the measurement.
    let (_, loaded) = load_round();

    let mut build_best = f64::INFINITY;
    let mut load_best = f64::INFINITY;
    for _ in 0..ROUNDS {
        build_best = build_best.min(build_round().0);
        load_best = load_best.min(load_round().0);
    }

    let speed_up = build_best / load_best.max(f64::MIN_POSITIVE);
    println!(
        "session cache, 400-node foldable chain, best of {ROUNDS} x {RUNS} runs: \
         build+optimize {:.3} ms, load from cache {:.3} ms ({} bytes), speed-up {speed_up:.1}x",
        build_best * 1e3,
        load_best * 1e3,
        bytes.len(),
    );

    assert_eq!(built.nodes().len(), loaded.nodes().len());
    // Deliberately no assertion on `speed_up` — see the doc comment above.
    // `loading_a_cache_executes_zero_operators_where_building_executes_many`
    // is the pass/fail signal for this claim.
}

// ── format properties ───────────────────────────────────────────────────────

#[test]
fn serialising_the_same_session_twice_produces_identical_bytes() {
    let (graph, weights) = foldable_graph(3);
    let session = Session::from_graph(graph, weights).expect("session builds");
    let first = session.to_optimized_bytes().expect("serialises");
    let second = session.to_optimized_bytes().expect("serialises");
    assert_eq!(first, second, "the encoding must be deterministic");

    // ...and stable across a round trip, so a cache file can be content-hashed.
    let reloaded = Session::from_optimized_bytes(&first).expect("cache loads");
    assert_eq!(
        reloaded.to_optimized_bytes().expect("re-serialises"),
        first,
        "save → load → save must be a fixed point"
    );
}

#[test]
fn model_metadata_and_value_info_survive_the_round_trip() {
    let graph = Graph {
        name: "meta".to_string(),
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![oxionnx::TensorInfo {
            name: "x".to_string(),
            dtype: oxionnx::DType::F32,
            shape: vec![Some(1), None, Some(3)],
            dim_params: vec![None, Some("seq".to_string()), None],
        }],
        output_infos: vec![oxionnx::TensorInfo {
            name: "y".to_string(),
            dtype: oxionnx::DType::F32,
            shape: vec![Some(1), None, Some(3)],
            dim_params: vec![None, Some("seq".to_string()), None],
        }],
    };
    let mut session = Session::from_graph(graph, HashMap::new()).expect("session builds");
    // Reach the metadata the only way a test can: rebuild the session through
    // the loader, whose metadata comes from the cache.
    let bytes = session.to_optimized_bytes().expect("serialises");
    session = Session::from_optimized_bytes(&bytes).expect("cache loads");

    assert_eq!(
        session.input_info(),
        &[oxionnx::TensorInfo {
            name: "x".to_string(),
            dtype: oxionnx::DType::F32,
            shape: vec![Some(1), None, Some(3)],
            dim_params: vec![None, Some("seq".to_string()), None],
        }]
    );
    assert_eq!(session.output_info().len(), 1);
    assert_eq!(session.input_names(), &["x".to_string()]);
    assert_eq!(session.output_names(), &["y".to_string()]);
}

#[test]
fn a_graph_with_nested_subgraphs_survives_the_round_trip() {
    let then_branch = Graph {
        nodes: vec![node(OpKind::Neg, "neg", &["x"], &["t"])],
        output_names: vec!["t".to_string()],
        ..Default::default()
    };
    let else_branch = Graph {
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["e"])],
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
    if_attrs.floats.insert("unused_float".to_string(), 0.25);
    if_attrs
        .int_lists
        .insert("unused_ints".to_string(), vec![7, -8]);
    if_attrs
        .strings
        .insert("note".to_string(), "hi".to_string());
    if_attrs.tensors.insert(
        "unused_tensor".to_string(),
        Tensor::new(vec![9.0, 8.0], vec![2]),
    );

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

    let original = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");

    let bytes = original.to_optimized_bytes().expect("serialises");
    let reloaded = Session::from_optimized_bytes(&bytes).expect("cache loads");

    let mut inputs = HashMap::new();
    inputs.insert("cond", Tensor::new(vec![0.0], vec![1]));
    inputs.insert("x", Tensor::new(vec![1.0, -2.0, 3.0], vec![3]));

    let expected = original.run(&inputs).expect("original run");
    let actual = reloaded.run(&inputs).expect("reloaded run");
    assert_eq!(bits(&actual["y"]), bits(&expected["y"]));
    assert_eq!(actual["y"].data, vec![1.0, 0.0, 3.0], "else_branch = Relu");
}

#[test]
fn a_custom_operator_name_round_trips_through_the_op_kind_encoding() {
    // `OpKind::Unknown(name)` is how a custom operator is represented; the cache
    // stores the op *string*, so the name must survive verbatim.
    let graph = Graph {
        name: "custom".to_string(),
        nodes: vec![node(
            OpKind::parse("MyCompanyCustomOp"),
            "custom",
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
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");
    let bytes = session.to_optimized_bytes().expect("serialises");
    let reloaded = Session::from_optimized_bytes(&bytes).expect("cache loads");

    let ops: Vec<String> = reloaded.nodes().into_iter().map(|n| n.op_type).collect();
    assert_eq!(ops, vec!["MyCompanyCustomOp".to_string()]);
}

#[test]
fn a_cache_file_survives_the_file_system_round_trip_and_reports_a_missing_one() {
    let (graph, weights) = foldable_graph(2);
    let session = Session::from_graph(graph, weights).expect("session builds");
    let path = temp_path("fs");
    session.save_optimized(&path).expect("cache is written");
    assert!(path.exists());

    let loaded = Session::load_optimized(&path).expect("cache loads");
    assert_eq!(loaded.nodes().len(), session.nodes().len());
    let _ = std::fs::remove_file(&path);

    let missing = Session::load_optimized(&path);
    assert!(
        matches!(missing, Err(OnnxError::Parse(_))),
        "a missing cache file is a typed error, not a panic"
    );
}

/// A cache whose graph references a value nothing produces must load *whole* and
/// then fail loudly at run time — never load a node short and quietly return an
/// incomplete result.
///
/// A cache this crate wrote can never be in that state, so the case is
/// manufactured: a valid file is patched to point one node's input at a name no
/// node produces. `Graph::topological_sort` keeps such a node (appending it in
/// its original position rather than discarding it), which is exactly the
/// property `check_no_nodes_were_dropped` guards.
#[test]
fn a_cache_whose_graph_is_unschedulable_loads_whole_and_fails_at_run() {
    const LINK: &str = "intermediate_value";
    let graph = Graph {
        name: "chain".to_string(),
        nodes: vec![
            node(OpKind::Relu, "first", &["x"], &[LINK]),
            node(OpKind::Sqrt, "second", &[LINK], &["y"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("session builds");
    let mut bytes = session.to_optimized_bytes().expect("serialises");
    assert!(
        Session::from_optimized_bytes(&bytes).is_ok(),
        "baseline loads"
    );

    // The name is written twice: as the first node's output, then as the second
    // node's input. Break the second occurrence only.
    let record: Vec<u8> = (LINK.len() as u64)
        .to_le_bytes()
        .iter()
        .copied()
        .chain(LINK.bytes())
        .collect();
    let occurrences: Vec<usize> = bytes
        .windows(record.len())
        .enumerate()
        .filter(|(_, w)| *w == record.as_slice())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(occurrences.len(), 2, "one producer, one consumer");
    let consumer_text = occurrences[1] + 8;
    bytes[consumer_text] = b'Z';

    let patched = Session::from_optimized_bytes(&bytes)
        .expect("an unschedulable graph still decodes: nothing about the FORMAT is wrong");
    assert_eq!(
        patched.nodes().len(),
        2,
        "no node may be silently dropped on load"
    );

    // ...and the broken edge surfaces as a typed error the first time it matters.
    let outcome = patched.run(&inputs_named("x"));
    assert!(
        outcome.is_err(),
        "a graph reading a value nothing produces must fail, not return a partial result"
    );
}

fn inputs_named(name: &'static str) -> HashMap<&'static str, Tensor> {
    let mut map = HashMap::new();
    map.insert(name, Tensor::new(vec![1.0, 4.0, 9.0], vec![3]));
    map
}
