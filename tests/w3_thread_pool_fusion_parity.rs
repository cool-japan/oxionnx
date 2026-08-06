//! Wave-3 (T7-tests-engine): `with_intra_threads(N)` thread-*count* invariance
//! on a fan-in ("fusion-heavy") graph.
//!
//! This is a narrower, sharper claim than the parity work already covered
//! elsewhere in this tree:
//!
//! * `tests/w2_engine_perf.rs` compares the rayon multi-node phase against the
//!   plain sequential dispatcher (`with_parallel_execution(true/false)`) — two
//!   different *code paths*, one of which never touches a thread pool at all.
//! * `src/session/tests/thread_pool.rs` exercises `with_intra_threads(1)` and
//!   `with_intra_threads(4)`, but only on trivial one- and two-node graphs
//!   (a single `Relu`; two independent `Relu`s), which never puts more than
//!   one node on the same topological level through a *shared, multi-threaded*
//!   `rayon::ThreadPool` at once.
//!
//! Neither exercises the thing that can only go wrong with an actual live
//! thread pool of more than one worker: races over the memory pool, the
//! output-slot table, or work-stealing reordering results between sibling
//! nodes at the same depth. This file builds a graph with real width (three
//! independent, *differently-computing* lanes so a scheduling mixup that
//! swapped two lanes' results would be visible) that fan back **in** through
//! two `Add`s (the "fusion" in the name), and checks that `with_intra_threads`
//! set to different worker counts never changes the answer — compared against
//! each other bit-for-bit (same kernels, only the schedule differs, so no
//! tolerance is appropriate) and against a plain-Rust reference computed
//! without going through the engine at all (tolerance noted per assertion,
//! needed only because the reference's `sigmoid`/`exp` may not be bit-identical
//! to whatever kernel the engine dispatches to).

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, Session, SessionBuilder, Tensor};

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// Three lanes computing *different* functions of `x` (so a lane mixup would
/// change the answer), fanning in through two `Add`s, then `Sigmoid` and a
/// self-`Mul`:
///
/// ```text
/// lane0: t0 = Relu(x)        t1 = Neg(t0)
/// lane1: t0 = Abs(x)         t1 = Neg(t0)
/// lane2: t0 = Neg(x)         t1 = Relu(t0)
///
/// f01 = Add(lane0.t1, lane1.t1)
/// f   = Add(f01, lane2.t1)
/// s   = Sigmoid(f)
/// y   = Mul(s, s)
/// ```
fn fusion_graph() -> Graph {
    let nodes = vec![
        // depth 0: width 3
        node(OpKind::Relu, "l0_t0", &["x"], &["l0_t0"]),
        node(OpKind::Abs, "l1_t0", &["x"], &["l1_t0"]),
        node(OpKind::Neg, "l2_t0", &["x"], &["l2_t0"]),
        // depth 1: width 3
        node(OpKind::Neg, "l0_t1", &["l0_t0"], &["l0_t1"]),
        node(OpKind::Neg, "l1_t1", &["l1_t0"], &["l1_t1"]),
        node(OpKind::Relu, "l2_t1", &["l2_t0"], &["l2_t1"]),
        // depth 2 / 3: fan-in
        node(OpKind::Add, "f01", &["l0_t1", "l1_t1"], &["f01"]),
        node(OpKind::Add, "f", &["f01", "l2_t1"], &["f"]),
        // depth 4 / 5
        node(OpKind::Sigmoid, "s", &["f"], &["s"]),
        node(OpKind::Mul, "y", &["s", "s"], &["y"]),
    ];
    Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    }
}

/// The same computation, in plain Rust, independent of the engine entirely.
fn reference(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&xi| {
            let l0_t1 = -xi.max(0.0);
            let l1_t1 = -xi.abs();
            let l2_t1 = (-xi).max(0.0);
            let f = l0_t1 + l1_t1 + l2_t1;
            let s = 1.0 / (1.0 + (-f).exp());
            s * s
        })
        .collect()
}

fn build(intra_threads: usize) -> Session {
    SessionBuilder::new()
        .with_intra_threads(intra_threads)
        .build_from_graph(fusion_graph(), HashMap::new())
        .unwrap_or_else(|e| panic!("build with_intra_threads({intra_threads}) failed: {e}"))
}

fn run_y(session: &Session, x: &[f32]) -> Vec<f32> {
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(x.to_vec(), vec![x.len()]));
    session
        .run(&inputs)
        .expect("run should succeed")
        .get("y")
        .expect("output 'y' should be present")
        .data
        .clone()
}

fn assert_bits_equal(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: index {i} differs bit-for-bit: {x} (0x{:08x}) vs {y} (0x{:08x})",
            x.to_bits(),
            y.to_bits(),
        );
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}: index {i}: engine={a} reference={e} diff={}",
            (a - e).abs()
        );
    }
}

/// One-thread and four-thread pools must produce the identical answer, and
/// both must match a reference computed entirely outside the engine.
#[test]
fn fusion_graph_agrees_across_thread_counts_and_matches_a_plain_rust_reference() {
    let x = vec![-2.0f32, -1.0, 0.5, 3.0, 0.0, -0.25];
    let session_1t = build(1);
    let session_4t = build(4);

    let out_1t = run_y(&session_1t, &x);
    let out_4t = run_y(&session_4t, &x);
    let expected = reference(&x);

    assert_bits_equal(&out_1t, &out_4t, "1 thread vs 4 threads");
    // The reference recomputes sigmoid independently (f32::exp vs whatever the
    // engine dispatches to), so this leg allows a small tolerance; the engine
    // legs above must be exact.
    assert_close(&out_1t, &expected, 1e-6, "1 thread vs plain-Rust reference");
    assert_close(
        &out_4t,
        &expected,
        1e-6,
        "4 threads vs plain-Rust reference",
    );
}

/// A single lucky run does not rule out a race that only shows up
/// intermittently under real multi-threaded scheduling; run several rounds
/// with different inputs (and freshly built sessions, so pool/thread-pool
/// state is not carried over) and require bit-for-bit agreement every time.
#[test]
fn fusion_graph_thread_count_invariance_holds_across_repeated_rounds() {
    for round in 0..15 {
        let base = round as f32 * 0.37 - 2.0;
        let x = vec![base, base + 1.0, -base, base * 0.5, base - 3.0];

        let session_1t = build(1);
        let session_4t = build(4);
        let out_1t = run_y(&session_1t, &x);
        let out_4t = run_y(&session_4t, &x);

        assert_bits_equal(&out_1t, &out_4t, &format!("round {round}"));
    }
}

/// Oversubscription (far more worker threads requested than this machine has
/// cores) must not change the answer either — a plausible real deployment
/// misconfiguration, not just the small thread counts exercised above.
#[test]
fn fusion_graph_agrees_when_the_thread_pool_is_oversubscribed() {
    let x = vec![10.0f32, -10.0, 0.0, 1.5, -1.5, 4.25, -4.25, 100.0];
    let session_1t = build(1);
    let session_oversubscribed = build(64);

    let out_1t = run_y(&session_1t, &x);
    let out_over = run_y(&session_oversubscribed, &x);

    assert_bits_equal(
        &out_1t,
        &out_over,
        "1 thread vs 64 (oversubscribed) threads",
    );
}
