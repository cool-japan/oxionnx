//! The **static** part of a run, computed once when the session is built.
//!
//! # What "static" means
//!
//! Everything here is a pure function of `Session::sorted_nodes`,
//! `Session::weights`, `Session::output_names` and `Session::shape_cache` — all
//! four of which are fixed for the life of the session.  Nothing on `Session`
//! mutates them: the only `&mut self` methods are `register_op` and
//! `set_session_cancellation`, and both touch the registry alone.  A plan
//! therefore cannot go stale, which is what makes hoisting it out of `run()`
//! safe rather than merely faster.
//!
//! Two per-run costs move here:
//!
//! * **The reference-count seed.** `run_internal` used to walk every node, every
//!   node input and every subgraph capture on *every inference* to rebuild a map
//!   whose contents never change between runs — including a `weights` probe per
//!   input and a recursive free-name walk per control-flow node.  It is now one
//!   `extend` over a precomputed vector.
//! * **The parallel schedule.** `run_parallel_inner` used to recompute
//!   `compute_node_depths` (a full graph pass that itself calls
//!   `subgraph_captures` per node), `group_by_depth`, the whole
//!   `cost_model::compute_critical_path_costs` pass, and a per-level sort — four
//!   passes over the graph, per inference, all four depending only on data fixed
//!   at build time.
//!
//! # Deliberately *not* done here (named so the next wave does not re-derive it)
//!
//! Wave-2's engine-perf lane proposed going further: intern every tensor name to
//! a `u32` slot id (`slot_ids`, `node_input_slots`, `node_capture_slots`) so the
//! per-run counts become a `Vec<usize>` that is `clone_from_slice`d and indexed
//! without hashing at all.  That is the better end state, and it is also the
//! enabler for moving `SessionRunState.tensors` off the std hasher (which cannot
//! move today because `as_map()` hands it out as `OpContext::outer_scope`).
//!
//! It is not done here because it is not a local change: `decrement_refs_state`
//! is reached from ~10 call sites that hold a `&Node` rather than a node index,
//! and the failure mode of a slot/name desync is an intermediate freed one node
//! early — silent numerical corruption, not a test failure.  It wants its own
//! task with its own differential harness, not a corner of this one.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::scheduling::subgraph_captures;

/// Everything about a run that is decided before the first input arrives.
pub(crate) struct StaticRunPlan {
    /// The seed for this run's [`RefCounts`](super::RefCounts): tensor name →
    /// number of consumers, counting graph outputs as one consumer each.
    ///
    /// Stored as a sorted `Vec` rather than a map because the run needs a *fresh*
    /// map every time (the counts are decremented as the run proceeds), so the
    /// only thing worth precomputing is the contents.  Sorted so a plan is
    /// deterministic — a `hashbrown` iteration order is not, and a plan that
    /// differs run-to-run is one that cannot be compared in a test.
    pub(crate) base_ref_counts: Vec<(String, usize)>,

    /// Node indices grouped by topological depth, each level already sorted by
    /// descending critical-path cost.
    ///
    /// Only the native parallel path reads this; on `wasm32`
    /// `run_parallel_inner` delegates to the sequential path, so building it
    /// there would be pure cost.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) depth_groups: Vec<Vec<usize>>,
}

impl StaticRunPlan {
    /// Compute the plan for a finished graph.
    ///
    /// Must be called **after** optimization and the topological sort: constant
    /// folding removes nodes (and therefore consumers), so a plan built from the
    /// unoptimized node list would over-count references and keep intermediates
    /// alive past their last real consumer.
    pub(crate) fn build(
        sorted_nodes: &[Node],
        weights: &HashMap<String, Tensor>,
        output_names: &[String],
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))] shape_cache: Option<
            &HashMap<String, Vec<usize>>,
        >,
    ) -> Self {
        Self {
            base_ref_counts: base_ref_counts(sorted_nodes, weights, output_names),
            #[cfg(not(target_arch = "wasm32"))]
            depth_groups: depth_groups(sorted_nodes, weights, shape_cache),
        }
    }
}

/// The reference-count seed: how many not-yet-executed consumers each
/// non-initializer tensor has at the start of a run.
///
/// A node "consumes" its `inputs` **and** every outer-scope name its subgraph
/// attributes capture — ONNX subgraphs bind those implicitly by name, so they
/// appear nowhere in `node.inputs` yet an `If`/`Loop`/`Scan` body reads them out
/// of the live run state.  `Session::decrement_refs_state` releases exactly the
/// same set, so the counts stay symmetric; the two sites must also apply the
/// identical weight filter (an initializer is never reference-counted) or a live
/// tensor is freed early.
///
/// Graph outputs get one extra reference so nothing recycles a tensor the caller
/// is about to be handed.
fn base_ref_counts(
    sorted_nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<(String, usize)> {
    let mut counts: hashbrown::HashMap<&str, usize> =
        hashbrown::HashMap::with_capacity(sorted_nodes.len() + output_names.len());
    for node in sorted_nodes {
        for inp in &node.inputs {
            if !inp.is_empty() && !weights.contains_key(inp) {
                *counts.entry(inp.as_str()).or_insert(0) += 1;
            }
        }
        for captured in subgraph_captures(node) {
            if !captured.is_empty() && !weights.contains_key(captured) {
                *counts.entry(captured).or_insert(0) += 1;
            }
        }
    }
    for name in output_names {
        *counts.entry(name.as_str()).or_insert(0) += 1;
    }

    let mut out: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(name, count)| (name.to_string(), count))
        .collect();
    out.sort_unstable();
    out
}

/// Node indices grouped by topological depth, each level ordered heaviest-first.
///
/// Sorting each level by descending critical-path cost starts the longest chain
/// first, which is what keeps the tail of a rayon level short.  The ordering is
/// a scheduling hint only: every node in a level is independent of every other
/// by construction, so the order cannot change any result.
#[cfg(not(target_arch = "wasm32"))]
fn depth_groups(
    sorted_nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    shape_cache: Option<&HashMap<String, Vec<usize>>>,
) -> Vec<Vec<usize>> {
    use super::super::Session;

    let depths = Session::compute_node_depths(sorted_nodes, weights);
    let mut groups = Session::group_by_depth(&depths);
    let critical_costs =
        crate::optimizer::cost_model::compute_critical_path_costs(sorted_nodes, shape_cache);
    for group in &mut groups {
        group.sort_by(|&a, &b| critical_costs[b].cmp(&critical_costs[a]));
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph, OpKind};

    fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op,
            name: name.to_string(),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    fn with_subgraph(mut n: Node, key: &str, sub: Graph) -> Node {
        n.attrs.graphs.insert(key.to_string(), sub);
        n
    }

    /// An **independent** re-implementation of the seed construction that
    /// `run_internal` used to do inline, written from the contract rather than
    /// from `base_ref_counts`.
    ///
    /// This is the whole point of the equality tests below: a green test suite
    /// would not catch a plan that disagrees with the old inline walk, because
    /// nothing else recomputes it to compare against.
    fn reference_seed(
        sorted_nodes: &[Node],
        weights: &HashMap<String, Tensor>,
        output_names: &[String],
    ) -> Vec<(String, usize)> {
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for node in sorted_nodes {
            for inp in &node.inputs {
                if !inp.is_empty() && !weights.contains_key(inp) {
                    *counts.entry(inp.clone()).or_insert(0) += 1;
                }
            }
            for captured in subgraph_captures(node) {
                if !captured.is_empty() && !weights.contains_key(captured) {
                    *counts.entry(captured.to_string()).or_insert(0) += 1;
                }
            }
        }
        for name in output_names {
            *counts.entry(name.clone()).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// A chain plus a fan-out, with one initializer that must **not** be counted.
    #[test]
    fn the_precomputed_seed_matches_the_reference_on_a_plain_graph() {
        let nodes = vec![
            node(OpKind::Add, "n0", &["x", "w"], &["a"]),
            node(OpKind::Relu, "n1", &["a"], &["b"]),
            node(OpKind::Mul, "n2", &["a", "b"], &["y"]),
        ];
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0], vec![1]));
        let outputs = vec!["y".to_string()];

        let plan = base_ref_counts(&nodes, &weights, &outputs);
        assert_eq!(plan, reference_seed(&nodes, &weights, &outputs));
        // Spot-check the values themselves, so a bug shared by both
        // implementations still shows up.
        assert_eq!(
            plan,
            vec![
                ("a".to_string(), 2), // read by n1 and n2
                ("b".to_string(), 1),
                ("x".to_string(), 1),
                ("y".to_string(), 1), // graph output
            ],
            "'w' is an initializer and must not be reference-counted",
        );
    }

    /// The case the seed exists for: a subgraph capture is a consumer even
    /// though it appears nowhere in `node.inputs`.
    #[test]
    fn the_precomputed_seed_counts_subgraph_captures() {
        let body = Graph {
            nodes: vec![node(OpKind::Mul, "m", &["t", "s"], &["o"])],
            input_names: Vec::new(),
            output_names: vec!["o".to_string()],
            ..Default::default()
        };
        let nodes = vec![
            node(OpKind::Relu, "p0", &["x"], &["t"]),
            node(OpKind::Relu, "p1", &["x"], &["s"]),
            with_subgraph(
                node(OpKind::If, "branch", &["cond"], &["y"]),
                "then_branch",
                body,
            ),
        ];
        let weights = HashMap::new();
        let outputs = vec!["y".to_string()];

        let plan = base_ref_counts(&nodes, &weights, &outputs);
        assert_eq!(plan, reference_seed(&nodes, &weights, &outputs));
        let lookup: HashMap<&str, usize> = plan.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(lookup.get("t"), Some(&1), "captured by the If body");
        assert_eq!(lookup.get("s"), Some(&1), "captured by the If body");
        assert_eq!(lookup.get("x"), Some(&2));
        assert_eq!(lookup.get("cond"), Some(&1));
    }

    /// A name that is both a graph output and a node input carries both
    /// references, which is what stops it being recycled mid-run.
    #[test]
    fn a_graph_output_that_is_also_consumed_carries_both_references() {
        let nodes = vec![
            node(OpKind::Relu, "n0", &["x"], &["a"]),
            node(OpKind::Relu, "n1", &["a"], &["b"]),
        ];
        let weights = HashMap::new();
        let outputs = vec!["a".to_string(), "b".to_string()];
        let plan = base_ref_counts(&nodes, &weights, &outputs);
        let lookup: HashMap<&str, usize> = plan.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(lookup.get("a"), Some(&2), "one consumer + one output");
        assert_eq!(lookup.get("b"), Some(&1));
    }

    /// Elided (empty-string) input slots are never counted.
    #[test]
    fn elided_inputs_are_not_counted() {
        let nodes = vec![node(OpKind::Clip, "c", &["x", "", ""], &["y"])];
        let plan = base_ref_counts(&nodes, &HashMap::new(), &["y".to_string()]);
        assert_eq!(
            plan,
            vec![("x".to_string(), 1), ("y".to_string(), 1)],
            "the two elided slots contribute nothing: {plan:?}",
        );
    }

    /// The precomputed schedule must equal the one `run_parallel_inner` used to
    /// build on the fly — same depths, same grouping, same cost ordering.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_precomputed_schedule_matches_computing_it_on_the_fly() {
        use super::super::super::Session;

        // A diamond with an extra-wide middle level, so the cost sort has
        // something to reorder.
        let nodes = vec![
            node(OpKind::Relu, "n0", &["x"], &["a"]),
            node(OpKind::MatMul, "n1", &["a", "a"], &["b"]),
            node(OpKind::Relu, "n2", &["a"], &["c"]),
            node(OpKind::Relu, "n3", &["a"], &["d"]),
            node(OpKind::Add, "n4", &["b", "c"], &["e"]),
            node(OpKind::Add, "n5", &["e", "d"], &["y"]),
        ];
        let weights = HashMap::new();
        let mut shapes = HashMap::new();
        shapes.insert("b".to_string(), vec![64, 64]);
        shapes.insert("c".to_string(), vec![4]);
        shapes.insert("d".to_string(), vec![4]);

        let plan = depth_groups(&nodes, &weights, Some(&shapes));

        // Recompute exactly as the run loop used to.
        let depths = Session::compute_node_depths(&nodes, &weights);
        let mut expected = Session::group_by_depth(&depths);
        let costs =
            crate::optimizer::cost_model::compute_critical_path_costs(&nodes, Some(&shapes));
        for group in &mut expected {
            group.sort_by(|&a, &b| costs[b].cmp(&costs[a]));
        }
        assert_eq!(plan, expected);

        // And it really is a schedule: every node appears exactly once, and no
        // level contains a node that depends on another node in the same level.
        let mut seen: Vec<usize> = plan.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..nodes.len()).collect::<Vec<_>>());
        assert!(plan[1].len() == 3, "n1/n2/n3 all sit at depth 1: {plan:?}",);
        assert_eq!(
            plan[1][0], 1,
            "the 64x64 MatMul is the heaviest node at its level and must run first",
        );
    }

    /// An empty graph is a legal (if useless) session and must not panic.
    #[test]
    fn an_empty_graph_yields_an_empty_plan() {
        let plan = StaticRunPlan::build(&[], &HashMap::new(), &[], None);
        assert!(plan.base_ref_counts.is_empty());
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(
            plan.depth_groups,
            vec![Vec::<usize>::new()],
            "group_by_depth always returns at least one (empty) level",
        );
    }
}
