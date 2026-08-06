//! Wave-3: the static per-run plan (`Session::run_plan`) is equivalent to
//! rebuilding it on every inference.
//!
//! # What moved, and what could break
//!
//! `run_internal` used to rebuild the reference-count seed from scratch on every
//! `run()` — a walk over every node, every input and every subgraph capture —
//! and `run_parallel_inner` used to recompute the whole schedule (depths, depth
//! grouping, critical-path costs, per-level sort) on every `run()` too.  Both are
//! pure functions of data fixed when the session is built, so both now happen
//! once, in `StaticRunPlan::build`.
//!
//! Two things could go wrong, and neither is visible in an ordinary output
//! assertion:
//!
//! * a seed built at the *wrong moment* (before constant folding removes
//!   consumers) over-counts references, so intermediates are never recycled;
//! * a seed **consumed** rather than copied would leave the second run of a
//!   session with counts already at zero, freeing tensors one node early — which
//!   shows up as `TensorNotFound`, or silently as an in-place mutation of a
//!   tensor that still has readers.
//!
//! So the assertions here are about *repetition* and *path agreement*, not about
//! any single output: the same session run many times, on both execution paths,
//! concurrently, with control flow (whose captures are the part of the seed with
//! no other test coverage at the session level).

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor, TensorInfo};
use std::collections::HashMap;
use std::sync::Arc;

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn graph(nodes: Vec<Node>, inputs: &[&str], outputs: &[&str]) -> Graph {
    Graph {
        nodes,
        input_names: inputs.iter().map(|s| (*s).to_string()).collect(),
        output_names: outputs.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    }
}

fn x() -> Tensor {
    Tensor::new(vec![-2.0, 0.5, 3.0, -4.0], vec![4])
}

/// `x → Relu → a`, `a`+`a` → `b`, `a`*`b` → `y`.
///
/// A diamond: `a` has two consumers, so a seed that over- or under-counts it is
/// immediately wrong — under-counting frees (and possibly mutates) it before
/// `Mul` reads it.
fn diamond() -> Graph {
    graph(
        vec![
            node(OpKind::Relu, "n0", &["x"], &["a"]),
            node(OpKind::Add, "n1", &["a", "a"], &["b"]),
            node(OpKind::Mul, "n2", &["a", "b"], &["y"]),
        ],
        &["x"],
        &["y"],
    )
}

fn run_once(session: &Session) -> Vec<f32> {
    let mut inputs = HashMap::new();
    inputs.insert("x", x());
    session
        .run(&inputs)
        .expect("run")
        .get("y")
        .expect("output y")
        .data
        .clone()
}

// ── Repetition: the plan is reused, never consumed ──────────────────────────

/// Twenty runs of one session must produce the identical answer every time.
///
/// A plan that were mutated by a run (counts decremented in place) would fail on
/// run 2, not run 1 — which is exactly why one `run()` per test is not enough.
#[test]
fn a_session_run_many_times_returns_the_same_answer_every_time() {
    for parallel in [false, true] {
        let session = Session::builder()
            .with_parallel_execution(parallel)
            .build_from_graph(diamond(), HashMap::new())
            .expect("build");

        let first = run_once(&session);
        for i in 1..20 {
            assert_eq!(
                run_once(&session),
                first,
                "run {i} disagreed with run 0 (parallel = {parallel})",
            );
        }
        // Relu([-2, 0.5, 3, -4]) = [0, 0.5, 3, 0]; b = 2a; y = a * 2a.
        assert_eq!(first, vec![0.0, 0.5, 18.0, 0.0]);
    }
}

/// The sequential and parallel paths share the seed but not the schedule, so
/// they are the sharpest available check that the precomputed schedule really
/// respects the graph's dependencies.
#[test]
fn both_execution_paths_agree_bit_for_bit() {
    let sequential = Session::builder()
        .with_parallel_execution(false)
        .build_from_graph(diamond(), HashMap::new())
        .expect("build");
    let parallel = Session::builder()
        .with_parallel_execution(true)
        .build_from_graph(diamond(), HashMap::new())
        .expect("build");

    let seq: Vec<u32> = run_once(&sequential).iter().map(|v| v.to_bits()).collect();
    let par: Vec<u32> = run_once(&parallel).iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        seq, par,
        "compared as f32 bit patterns, not within a tolerance"
    );
}

// ── Subgraph captures ───────────────────────────────────────────────────────

/// The part of the seed nothing else covers end-to-end: an `If` body reads `t`
/// out of the enclosing scope, so `t` has a consumer that appears nowhere in
/// `node.inputs`.
///
/// A missing capture reference breaks this **two** ways, and the graph is built
/// so that both are observable:
///
/// * `t`'s count would hit zero at the node that reads it, so its buffer would
///   be recycled before the `If` ran — a `TensorNotFound`;
/// * a count of exactly 1 also unlocks the in-place path, so that node would
///   *mutate* `t` first.  `passthrough` is therefore a `Sigmoid` — which
///   `supports_inplace` and whose output differs from its input everywhere — and
///   not a second `Relu`, whose idempotence would hide the mutation entirely.
#[test]
fn a_subgraph_capture_survives_to_the_control_flow_node() {
    for parallel in [false, true] {
        let then_branch = graph(
            vec![node(OpKind::Mul, "m", &["t", "t"], &["o"])],
            &[],
            &["o"],
        );
        let mut branch = node(OpKind::If, "branch", &["cond"], &["y"]);
        branch
            .attrs
            .graphs
            .insert("then_branch".to_string(), then_branch);
        // An else branch is required for a well-formed If.
        branch.attrs.graphs.insert(
            "else_branch".to_string(),
            graph(vec![node(OpKind::Relu, "r", &["t"], &["o"])], &[], &["o"]),
        );

        let g = graph(
            vec![
                // `t` is produced here, consumed by `passthrough` *and* captured
                // by both branches of `branch`.
                node(OpKind::Relu, "produce", &["x"], &["t"]),
                // Sigmoid, not Relu: it supports in-place, and its output
                // differs from its input, so an unwanted in-place mutation of
                // `t` changes the assertion below instead of hiding in Relu's
                // idempotence.
                node(OpKind::Sigmoid, "passthrough", &["t"], &["unused"]),
                branch,
            ],
            &["x", "cond"],
            &["y"],
        );

        let session = Session::builder()
            .with_parallel_execution(parallel)
            // Keep the graph exactly as written: dead-node elimination would
            // otherwise delete `passthrough`, which is the node that makes `t`
            // reach a count of one before the `If` runs.
            .with_optimization_level(OptLevel::None)
            .build_from_graph(g, HashMap::new())
            .expect("build");

        let mut inputs = HashMap::new();
        inputs.insert("x", x());
        inputs.insert("cond", Tensor::new(vec![1.0], vec![1]));
        let out = session.run(&inputs).expect("run with captures");
        // Relu(x) = [0, 0.5, 3, 0]; then_branch squares it.
        assert_eq!(
            out.get("y").expect("y").data,
            vec![0.0, 0.25, 9.0, 0.0],
            "parallel = {parallel}",
        );

        // And again, to prove the capture reference is re-seeded rather than
        // spent on the first run.
        assert_eq!(
            session
                .run(&inputs)
                .expect("second run")
                .get("y")
                .expect("y")
                .data,
            vec![0.0, 0.25, 9.0, 0.0],
        );
    }
}

// ── Concurrency ─────────────────────────────────────────────────────────────

/// The plan is shared immutably by every concurrent run; nothing may take a
/// `&mut` to it, and no run may observe another run's counts.
#[test]
fn concurrent_runs_on_one_session_do_not_interfere() {
    for parallel in [false, true] {
        let session = Arc::new(
            Session::builder()
                .with_parallel_execution(parallel)
                .build_from_graph(diamond(), HashMap::new())
                .expect("build"),
        );

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let session = Arc::clone(&session);
                std::thread::spawn(move || {
                    let mut all = Vec::new();
                    for _ in 0..25 {
                        all.push(run_once(&session));
                    }
                    all
                })
            })
            .collect();

        for handle in handles {
            let results = handle.join().expect("worker thread panicked");
            for result in results {
                assert_eq!(result, vec![0.0, 0.5, 18.0, 0.0], "parallel = {parallel}");
            }
        }
    }
}

// ── Constant folding: the plan must be built from the FINAL node list ───────

/// The optimizer folds `Add(c0, c1)` away entirely, so the folded node's
/// consumers vanish with it.  A seed built before optimization would still count
/// them, keeping intermediates alive for the whole run — and, more importantly,
/// would key on names the final graph does not contain.
#[test]
fn the_plan_is_built_after_constant_folding_not_before() {
    let mut weights = HashMap::new();
    weights.insert("c0".to_string(), Tensor::new(vec![2.0; 4], vec![4]));
    weights.insert("c1".to_string(), Tensor::new(vec![3.0; 4], vec![4]));

    let g = graph(
        vec![
            node(OpKind::Add, "fold_me", &["c0", "c1"], &["k"]),
            node(OpKind::Mul, "use_it", &["x", "k"], &["y"]),
        ],
        &["x"],
        &["y"],
    );

    for parallel in [false, true] {
        let session = Session::builder()
            .with_parallel_execution(parallel)
            .with_optimization_level(OptLevel::All)
            .build_from_graph(g.clone(), weights.clone())
            .expect("build");

        // `fold_me` is gone; `k` is an initializer now.
        assert_eq!(
            session.nodes().len(),
            1,
            "constant folding should have removed the Add: {:?}",
            session.nodes(),
        );
        let out = run_once(&session);
        assert_eq!(out, vec![-10.0, 2.5, 15.0, -20.0], "parallel = {parallel}");
        // Repeat: a plan keyed on a name the optimized graph dropped would only
        // misbehave once the counts had been decremented.
        assert_eq!(run_once(&session), out);
    }
}

// ── Build-time shape-plan seeding ───────────────────────────────────────────

/// A model whose inputs are all declared statically gets its build-time shape
/// inference seeded into the shape-plan cache, so the first inference does not
/// repeat the pass.  The property that matters is that the seeded plan is the
/// *same* plan a fresh inference would produce.
#[test]
fn a_statically_declared_model_reports_the_same_shapes_seeded_or_not() {
    let mut g = diamond();
    g.input_infos = vec![TensorInfo {
        name: "x".to_string(),
        shape: vec![Some(4)],
        ..Default::default()
    }];

    // With the memory pool on, build-time shape inference runs and seeds the
    // plan cache; with it off, there is no `shape_cache` and hence no seed.
    let seeded = Session::builder()
        .with_memory_pool(true)
        .build_from_graph(g.clone(), HashMap::new())
        .expect("build seeded");
    let unseeded = Session::builder()
        .with_memory_pool(false)
        .build_from_graph(g, HashMap::new())
        .expect("build unseeded");

    assert_eq!(run_once(&seeded), run_once(&unseeded));
    assert_eq!(
        seeded.resolved_shapes(),
        unseeded.resolved_shapes(),
        "the seeded plan must be the plan a fresh inference produces",
    );
    let shapes = seeded.resolved_shapes();
    assert_eq!(shapes.get("x"), Some(&vec![4]));
    assert_eq!(shapes.get("y"), Some(&vec![4]));
}

/// The seed must not become a bypass: a declared static dimension is still
/// validated against the tensor the caller actually supplies.
#[test]
fn a_seeded_plan_does_not_bypass_input_validation() {
    let mut g = diamond();
    g.input_infos = vec![TensorInfo {
        name: "x".to_string(),
        shape: vec![Some(4)],
        ..Default::default()
    }];
    let session = Session::builder()
        .with_memory_pool(true)
        .build_from_graph(g, HashMap::new())
        .expect("build");

    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(vec![1.0, 2.0], vec![2]));
    assert!(
        session.run(&inputs).is_err(),
        "a rank-1 input of the wrong extent must still be rejected",
    );
}

/// A model with a *symbolic* input dimension has no static seed at all, and must
/// still answer correctly for two different concrete shapes in a row.
#[test]
fn a_dynamic_model_still_serves_two_different_shapes() {
    let mut g = diamond();
    g.input_infos = vec![TensorInfo {
        name: "x".to_string(),
        shape: vec![None],
        dim_params: vec![Some("n".to_string())],
        ..Default::default()
    }];
    let session = Session::builder()
        .with_memory_pool(true)
        .build_from_graph(g, HashMap::new())
        .expect("build");

    let mut small = HashMap::new();
    small.insert("x", Tensor::new(vec![1.0, -1.0], vec![2]));
    let mut large = HashMap::new();
    large.insert("x", Tensor::new(vec![1.0, -1.0, 2.0, 3.0], vec![4]));

    for _ in 0..3 {
        assert_eq!(
            session
                .run(&small)
                .expect("small")
                .get("y")
                .expect("y")
                .data,
            vec![2.0, 0.0],
        );
        assert_eq!(
            session
                .run(&large)
                .expect("large")
                .get("y")
                .expect("y")
                .data,
            vec![2.0, 0.0, 8.0, 18.0],
        );
    }
}
