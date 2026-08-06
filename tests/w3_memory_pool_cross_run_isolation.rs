//! Wave-3 (T7-tests-engine): memory-pool buffer reuse must not leak state
//! *across* runs of different shapes, and must not change numerics at all.
//!
//! `tests/w3_pool_acquire_for_overwrite.rs::pooled_inference_is_unaffected`
//! already proves a pooled session gives the same answer on five repeats of
//! the *same* input. That is not the sharpest test available: if a reused
//! buffer were only partially overwritten, feeding the identical input again
//! would still "coincidentally" look correct, because the stale leftover data
//! IS the correct data for that unchanged input. The tests below instead walk
//! a pooled session through a sequence of *differently shaped* inputs
//! (large -> small -> large-with-different-values -> the first large input
//! again) so that any leaked buffer contents would show up as a value
//! mismatch against an independently hand-computed reference, not just a
//! mismatch against a previous run.

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

/// `y = Relu(x) * Relu(x)`: two nodes, so the intermediate `a` is a genuine
/// pooled buffer between two ops, not just a graph output.
fn square_relu_graph() -> Graph {
    Graph {
        nodes: vec![
            node(OpKind::Relu, "relu", &["x"], &["a"]),
            node(OpKind::Mul, "sq", &["a", "a"], &["y"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    }
}

fn reference(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0) * v.max(0.0)).collect()
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

/// Large -> small -> large-with-different-values -> the first large input
/// again, all on one pooled session. Every step is checked against an
/// independent reference, and the shrink-then-regrow step is the one most
/// likely to expose a buffer that was truncated/grown without being fully
/// rewritten.
#[test]
fn pooled_session_does_not_leak_state_across_differently_shaped_runs() {
    let session = SessionBuilder::new()
        .with_memory_pool(true)
        .build_from_graph(square_relu_graph(), HashMap::new())
        .expect("build with memory pool");

    let x_large_1 = vec![-4.0f32, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x_small = vec![5.0f32, -5.0, 0.0];
    let x_large_2 = vec![10.0f32, -10.0, 20.0, -20.0, 0.5, -0.5, 7.0, -7.0];

    let out_large_1 = run_y(&session, &x_large_1);
    assert_eq!(out_large_1, reference(&x_large_1), "large run 1");

    let out_small = run_y(&session, &x_small);
    assert_eq!(
        out_small,
        reference(&x_small),
        "small run: must not carry over any of large run 1's 8 elements"
    );

    let out_large_2 = run_y(&session, &x_large_2);
    assert_eq!(
        out_large_2,
        reference(&x_large_2),
        "large run 2 (regrown after the small run): must not carry over \
         small run's 3 elements, nor large run 1's original 8"
    );

    let out_large_1_repeat = run_y(&session, &x_large_1);
    assert_eq!(
        out_large_1_repeat, out_large_1,
        "repeating large run 1's exact input after two intervening \
         differently-shaped runs must reproduce the exact same output"
    );
}

/// Positive evidence that the pool path was actually exercised (not a vacuous
/// pass because pooling silently never engaged): `pool_stats().reuse_count`
/// must increase as the same-shaped request repeats.
#[test]
fn repeated_same_shape_runs_actually_reuse_pool_buffers() {
    let session = SessionBuilder::new()
        .with_memory_pool(true)
        .build_from_graph(square_relu_graph(), HashMap::new())
        .expect("build with memory pool");

    let x: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();

    let _ = run_y(&session, &x);
    let stats_after_first = session
        .pool_stats()
        .expect("pool_stats must be Some when the memory pool is enabled");

    for _ in 0..10 {
        let _ = run_y(&session, &x);
    }
    let stats_after_many = session.pool_stats().expect("pool_stats after repeats");

    assert!(
        stats_after_many.reuse_count > stats_after_first.reuse_count,
        "reuse_count must climb across repeated same-shape runs \
         (after 1 run: {}, after 11 runs: {}) -- otherwise the pool path in \
         this test is not actually being exercised",
        stats_after_first.reuse_count,
        stats_after_many.reuse_count,
    );
}

/// The memory pool changes *allocation strategy*, never *numerics*: a pooled
/// and a non-pooled session must produce bit-identical outputs across the
/// exact same varying-shape sequence used above.
#[test]
fn pooled_and_unpooled_sessions_agree_bit_for_bit_on_the_same_sequence() {
    let pooled = SessionBuilder::new()
        .with_memory_pool(true)
        .build_from_graph(square_relu_graph(), HashMap::new())
        .expect("build pooled");
    let unpooled = SessionBuilder::new()
        .with_memory_pool(false)
        .build_from_graph(square_relu_graph(), HashMap::new())
        .expect("build unpooled");

    let sequence: Vec<Vec<f32>> = vec![
        vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        vec![5.0, -5.0, 0.0],
        vec![10.0, -10.0, 20.0, -20.0, 0.5, -0.5, 7.0, -7.0],
        vec![1.0],
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
    ];

    for (i, x) in sequence.iter().enumerate() {
        let out_pooled = run_y(&pooled, x);
        let out_unpooled = run_y(&unpooled, x);
        assert_eq!(
            out_pooled,
            out_unpooled,
            "step {i} (len {}): pooled and unpooled sessions must agree bit-for-bit",
            x.len()
        );
    }
}
