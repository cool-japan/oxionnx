//! Regression tests for [a4-5]/[a10-15]: `Graph::topological_sort` used to build its
//! `in_degree` and `dependents` maps with different `known_set` guards, so a node
//! output name that collides with a name already present in the caller-supplied
//! `known` slice (e.g. an outer_scope tensor shadowed by a subgraph body node, or a
//! `Shape` node whose output was hoisted into `weights` while the node itself stayed
//! in the graph) could decrement `in_degree[dep]` more times than it was incremented,
//! underflowing `0usize - 1` and panicking "attempt to subtract with overflow" in any
//! debug/overflow-checked build (including `cargo test`).
//!
//! Reference: hand-traced against Kahn's algorithm by hand (see comments below).

use oxionnx_core::{Attributes, Graph, Node, OpKind};

fn node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: String::new(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// The exact underflow trigger: node 0 produces output "b", but "b" is *also* in the
/// caller's `known` set (simulating a `Shape` output hoisted into `weights`, or a
/// Loop/Scan body node whose output name shadows an outer_scope tensor). Node 1
/// consumes both "a" (a genuine known input) and "b" (the colliding name).
///
/// Before the fix: `in_degree[1]` is computed as 0 (both "a" and "b" are in
/// `known_set`, so neither increments it), but the old `dependents` map still records
/// `0 -> [1]` (built without consulting `known_set`), so popping node 0 during the
/// Kahn's-algorithm sweep runs `in_degree[1] -= 1` on an already-zero count and
/// panics with "attempt to subtract with overflow".
///
/// After the fix: `dependents` uses the same `known_set` guard as `in_degree`, so
/// node 0 -> node 1 is not recorded as a pending edge (node 1 doesn't wait on it),
/// and `topological_sort` returns cleanly.
#[test]
fn topological_sort_does_not_underflow_when_output_name_collides_with_known() {
    let graph = Graph {
        nodes: vec![
            node(OpKind::Shape, &["a"], &["b"]),
            node(OpKind::Reshape, &["a", "b"], &["c"]),
        ],
        ..Default::default()
    };
    let known = vec!["a".to_string(), "b".to_string()];

    // Must not panic (this is the regression check) and must return every node index
    // exactly once.
    let order = graph.topological_sort(&known);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![0, 1],
        "every node index must appear exactly once"
    );
}

/// Same collision shape but with three consumers of the colliding name, to make sure
/// the `saturating_sub` backstop / guarded `dependents` insertion holds under a
/// larger fan-out (not just a single dependent edge).
#[test]
fn topological_sort_handles_fan_out_collision_without_underflow() {
    let graph = Graph {
        nodes: vec![
            node(OpKind::Shape, &["a"], &["shadowed"]),
            node(OpKind::Reshape, &["a", "shadowed"], &["c1"]),
            node(OpKind::Reshape, &["a", "shadowed"], &["c2"]),
            node(OpKind::Reshape, &["a", "shadowed"], &["c3"]),
        ],
        ..Default::default()
    };
    let known = vec!["a".to_string(), "shadowed".to_string()];

    let order = graph.topological_sort(&known);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}

/// Sanity / non-regression check: a normal producer -> consumer chain with no name
/// collisions must still be ordered correctly (producer before consumer).
#[test]
fn topological_sort_orders_simple_chain_correctly() {
    let graph = Graph {
        nodes: vec![
            node(OpKind::Relu, &["x"], &["y"]),
            node(OpKind::Relu, &["y"], &["z"]),
        ],
        ..Default::default()
    };
    let known = vec!["x".to_string()];

    let order = graph.topological_sort(&known);
    assert_eq!(order, vec![0, 1]);
}

/// A diamond dependency (two independent branches merging into one consumer) must
/// place both producers before the merging consumer.
#[test]
fn topological_sort_orders_diamond_dependency_correctly() {
    let graph = Graph {
        nodes: vec![
            node(OpKind::Relu, &["x"], &["a"]),     // 0
            node(OpKind::Identity, &["x"], &["b"]), // 1
            node(OpKind::Add, &["a", "b"], &["y"]), // 2
        ],
        ..Default::default()
    };
    let known = vec!["x".to_string()];

    let order = graph.topological_sort(&known);
    let pos = |i: usize| order.iter().position(|&x| x == i).unwrap();
    assert!(pos(0) < pos(2), "producer of `a` must precede its consumer");
    assert!(pos(1) < pos(2), "producer of `b` must precede its consumer");
}
