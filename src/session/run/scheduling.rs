use crate::graph::{Graph, Node};
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::super::Session;
use super::state::SessionRunState;
use super::{CaptureSet, OutputSet, RefCounts};

// ── Subgraph outer-scope captures ───────────────────────────────────────────

/// How deep the capture walk will descend into nested subgraphs.
///
/// ONNX nests subgraphs through `If`/`Loop`/`Scan` attributes, and the walk below
/// is recursive.  A model is not trusted input, so the depth is capped rather
/// than left to blow the stack: 32 levels of nested control flow is already far
/// beyond anything a real exporter emits (PyTorch's `torch.onnx.export` produces
/// at most a handful), and the parser itself had to recurse to build the
/// structure in the first place.
const MAX_SUBGRAPH_NESTING: usize = 32;

/// The names a node's subgraph attributes read from the **enclosing** scope.
///
/// # Why this exists
///
/// ONNX subgraphs capture outer-scope tensors *implicitly, by name*: `IfOp`,
/// `LoopOp` and `ScanOp` all hand `execute_subgraph` an empty explicit-input map
/// and rely entirely on `ctx.outer_scope`, which the run loop fills from the live
/// [`SessionRunState`].  Nothing in `node.inputs` mentions those names.
///
/// The run state is reference-counted (`ref_counts` in `run_internal`), and a
/// tensor is dropped — and its buffer recycled into the pool — the moment its
/// *counted* consumers reach zero.  Counting only `node.inputs` therefore freed
/// captured tensors before the subgraph that reads them ever ran:
///
/// ```text
/// t = Add(x, w)          ref_counts[t] = 1   (only Relu is counted)
/// u = Relu(t)            → t taken out of state here (and, because
///                          `ReluOp::supports_inplace()`, mutated in place first)
/// y = If(cond)  { then_branch: Mul(t, s) }   → t is gone: TensorNotFound
/// ```
///
/// Adding one reference per capturing node closes the hole, and does so
/// *symmetrically*: [`Session::decrement_refs_state`] releases the very same set
/// when the control-flow node has executed, so a captured tensor is freed at
/// exactly the right point rather than being retained until the run ends.
///
/// It also disables the in-place path for a captured tensor for free: with the
/// extra reference, `ref_counts[node.inputs[0]] != 1`, which is precisely the
/// condition `dispatch_node` requires before it may mutate an input buffer.  No
/// second mechanism is needed, and none exists.
///
/// # Result
///
/// A **set** (never a `Vec`): `Attributes::graphs` is a `HashMap`, so its
/// iteration order is nondeterministic, and a name captured by both the
/// `then_branch` and the `else_branch` must contribute exactly one reference on
/// both the increment and the decrement side or the counts desync.
///
/// A name is *free* in a subgraph when some node reads it and it is neither a
/// formal input of that subgraph nor produced by one of its nodes.  Nested
/// subgraphs are walked recursively, with the names bound by each enclosing level
/// subtracted as the walk unwinds.
///
/// The empty-attribute fast path makes this a single `HashMap::is_empty` for
/// every ordinary node, which is what the per-run, per-node call sites need.
///
/// The names are **borrowed from `node`**, not cloned: they are folded straight
/// into [`RefCounts`], whose keys borrow from the same graph, and cloning them
/// allocated a `String` per capture per call — on a path that runs once per node
/// per run *and* once per node in `compute_node_depths`.
pub(crate) fn subgraph_captures(node: &Node) -> CaptureSet<'_> {
    let mut captures = CaptureSet::new();
    if node.attrs.graphs.is_empty() {
        return captures;
    }
    for graph in node.attrs.graphs.values() {
        collect_free_names(graph, 0, &mut captures);
    }
    captures
}

/// Accumulate the free (outer-scope) names of `graph` into `out`.
fn collect_free_names<'g>(graph: &'g Graph, depth: usize, out: &mut CaptureSet<'g>) {
    if depth >= MAX_SUBGRAPH_NESTING {
        return;
    }

    // Everything this subgraph binds itself: its formal inputs and every tensor
    // its own nodes produce.
    let mut bound: CaptureSet<'g> = graph.input_names.iter().map(|name| name.as_str()).collect();
    for sub_node in &graph.nodes {
        for out_name in &sub_node.outputs {
            if !out_name.is_empty() {
                bound.insert(out_name.as_str());
            }
        }
    }

    for sub_node in &graph.nodes {
        for input in &sub_node.inputs {
            if !input.is_empty() && !bound.contains(input.as_str()) {
                out.insert(input.as_str());
            }
        }
        // A nested subgraph's free names are free *here* too, unless this level
        // binds them.
        for nested in sub_node.attrs.graphs.values() {
            let mut nested_free = CaptureSet::new();
            collect_free_names(nested, depth + 1, &mut nested_free);
            for name in nested_free {
                if !bound.contains(name) {
                    out.insert(name);
                }
            }
        }
    }
}

impl Session {
    /// Compute the topological depth for each node in `sorted_nodes`.
    /// Depth 0 = all inputs come from model inputs / weights (no graph predecessors).
    /// For others, depth = max(depth of predecessor nodes) + 1.
    ///
    /// A node's predecessors are its `inputs` **and** the outer-scope tensors its
    /// subgraph attributes capture (see [`subgraph_captures`]).  Captures are real
    /// data dependencies even though they appear nowhere in `node.inputs`, and
    /// omitting them let an `If` whose body reads a deep intermediate land in a
    /// depth group that executes *before* that intermediate is produced.
    pub(crate) fn compute_node_depths(
        sorted_nodes: &[Node],
        weights: &HashMap<String, Tensor>,
    ) -> Vec<usize> {
        // `hashbrown`, not the std default: this map is rebuilt on every
        // `run_parallel_inner` call and is probed once per node input, so it is
        // firmly on the per-run hot path.  Same reasoning as [`RefCounts`].
        let mut tensor_depth: hashbrown::HashMap<&str, usize> =
            hashbrown::HashMap::with_capacity(sorted_nodes.len());
        let mut depths = Vec::with_capacity(sorted_nodes.len());

        for node in sorted_nodes {
            let mut max_pred_depth: Option<usize> = None;
            let captured = subgraph_captures(node);
            let predecessors = node
                .inputs
                .iter()
                .map(String::as_str)
                .chain(captured.iter().copied());
            for name in predecessors {
                if name.is_empty() || weights.contains_key(name) {
                    continue;
                }
                if let Some(&d) = tensor_depth.get(name) {
                    max_pred_depth = Some(match max_pred_depth {
                        Some(cur) => cur.max(d),
                        None => d,
                    });
                }
            }
            let depth = match max_pred_depth {
                Some(d) => d + 1,
                None => 0,
            };
            depths.push(depth);
            for out in &node.outputs {
                if !out.is_empty() {
                    tensor_depth.insert(out.as_str(), depth);
                }
            }
        }
        depths
    }

    /// Group node indices by their topological depth.
    pub(crate) fn group_by_depth(depths: &[usize]) -> Vec<Vec<usize>> {
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut groups = vec![Vec::new(); max_depth + 1];
        for (i, &d) in depths.iter().enumerate() {
            groups[d].push(i);
        }
        groups
    }

    /// Decrement reference counts for everything `node` consumed via
    /// `SessionRunState`, freeing tensors that are no longer needed and returning
    /// their buffers to the pool.
    ///
    /// "Consumed" is `node.inputs` **plus** the outer-scope names its subgraph
    /// attributes capture — the exact same set [`subgraph_captures`] contributed
    /// to `ref_counts` in `run_internal`.  The two sites must apply the identical
    /// weight filter (an initializer is never reference-counted) or the counts
    /// desync and a live tensor is freed early.
    pub(crate) fn decrement_refs_state(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
    ) {
        for inp in &node.inputs {
            self.release_one_ref(inp, state, ref_counts, output_set);
        }
        for captured in subgraph_captures(node) {
            self.release_one_ref(captured, state, ref_counts, output_set);
        }
    }

    /// Drop one reference to `name`, freeing its tensor when the last one goes.
    fn release_one_ref(
        &self,
        name: &str,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
    ) {
        if name.is_empty() || self.weights.contains_key(name) {
            return;
        }
        let Some(count) = ref_counts.get_mut(name) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count != 0 || output_set.contains(name) {
            return;
        }
        // Last consumer: take it out of state and release the buffer to the pool.
        if let Some(mut tensor) = state.take(name) {
            if let Some(ref pool_mutex) = self.pool {
                if let Ok(mut pool) = pool_mutex.lock() {
                    let buf = std::mem::take(&mut tensor.data);
                    if !buf.is_empty() {
                        pool.release(buf);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, OpKind};

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

    fn with_subgraphs(mut node: Node, subgraphs: &[(&str, Graph)]) -> Node {
        for (name, sub) in subgraphs {
            node.attrs.graphs.insert((*name).to_string(), sub.clone());
        }
        node
    }

    // ── subgraph_captures ───────────────────────────────────────────────────

    /// An ordinary node captures nothing, and pays only a `HashMap::is_empty`.
    #[test]
    fn a_node_without_subgraphs_captures_nothing() {
        let n = node(OpKind::Add, "add", &["a", "b"], &["c"]);
        assert!(subgraph_captures(&n).is_empty());
    }

    /// A body reads `t` and `s` from the enclosing scope; `mid` it produces itself.
    #[test]
    fn a_subgraph_body_captures_only_its_free_names() {
        let body = graph(
            vec![
                node(OpKind::Mul, "mul", &["t", "s"], &["mid"]),
                node(OpKind::Relu, "relu", &["mid"], &["out"]),
            ],
            &[],
            &["out"],
        );
        let n = with_subgraphs(
            node(OpKind::If, "branch", &["cond"], &["y"]),
            &[("then_branch", body)],
        );

        let captures = subgraph_captures(&n);
        assert_eq!(captures.len(), 2, "got {captures:?}");
        assert!(captures.contains("t"));
        assert!(captures.contains("s"));
        assert!(
            !captures.contains("mid"),
            "a name the body produces itself is not captured",
        );
        assert!(
            !captures.contains("cond"),
            "`cond` is a node input, not a body free name",
        );
    }

    /// A subgraph's own formal inputs are bound, not captured — that is what
    /// distinguishes `Loop`'s iteration variables from a real capture.
    #[test]
    fn a_subgraph_formal_input_is_not_a_capture() {
        let body = graph(
            vec![node(OpKind::Add, "add", &["iter", "acc", "w"], &["next"])],
            &["iter", "acc"],
            &["next"],
        );
        let n = with_subgraphs(
            node(OpKind::Loop, "loop", &["trip", "cond"], &["final"]),
            &[("body", body)],
        );

        let captures = subgraph_captures(&n);
        assert_eq!(captures.len(), 1, "got {captures:?}");
        assert!(captures.contains("w"));
    }

    /// Both branches contribute, and a name both read counts **once** — the set is
    /// what keeps the increment in `run_internal` and the decrement in
    /// `decrement_refs_state` symmetric despite `Attributes::graphs` being a
    /// `HashMap` with nondeterministic iteration order.
    #[test]
    fn both_branches_contribute_and_a_shared_capture_counts_once() {
        let then_branch = graph(
            vec![node(OpKind::Mul, "m", &["shared", "only_then"], &["o"])],
            &[],
            &["o"],
        );
        let else_branch = graph(
            vec![node(OpKind::Add, "a", &["shared", "only_else"], &["o"])],
            &[],
            &["o"],
        );
        let n = with_subgraphs(
            node(OpKind::If, "branch", &["cond"], &["y"]),
            &[("then_branch", then_branch), ("else_branch", else_branch)],
        );

        let captures = subgraph_captures(&n);
        assert_eq!(captures.len(), 3, "got {captures:?}");
        for name in ["shared", "only_then", "only_else"] {
            assert!(captures.contains(name), "{name} missing from {captures:?}");
        }
    }

    /// A nested subgraph's free names are free in the outer scope too — unless the
    /// enclosing body binds them.
    #[test]
    fn nested_subgraph_free_names_propagate_outwards() {
        let inner = graph(
            vec![node(
                OpKind::Add,
                "inner_add",
                &["outer_intermediate", "deep"],
                &["inner_out"],
            )],
            &[],
            &["inner_out"],
        );
        let outer_body = graph(
            vec![
                node(OpKind::Relu, "produce", &["seed"], &["outer_intermediate"]),
                with_subgraphs(
                    node(OpKind::If, "inner_if", &["c2"], &["z"]),
                    &[("then_branch", inner)],
                ),
            ],
            &[],
            &["z"],
        );
        let n = with_subgraphs(
            node(OpKind::If, "branch", &["cond"], &["y"]),
            &[("then_branch", outer_body)],
        );

        let captures = subgraph_captures(&n);
        assert!(
            captures.contains("deep"),
            "a doubly-nested free name must reach the top: {captures:?}",
        );
        assert!(captures.contains("seed"));
        assert!(captures.contains("c2"));
        assert!(
            !captures.contains("outer_intermediate"),
            "bound by the enclosing body, so not captured: {captures:?}",
        );
    }

    /// Elided (empty-string) input slots are never capture names.
    #[test]
    fn elided_subgraph_inputs_are_not_captured() {
        let body = graph(
            vec![node(OpKind::Clip, "clip", &["t", "", ""], &["o"])],
            &[],
            &["o"],
        );
        let n = with_subgraphs(
            node(OpKind::If, "branch", &["cond"], &["y"]),
            &[("then_branch", body)],
        );
        let captures = subgraph_captures(&n);
        assert_eq!(captures.len(), 1, "got {captures:?}");
        assert!(captures.contains("t"));
    }

    /// A model is untrusted input: pathological nesting must terminate, not
    /// recurse until the stack runs out.
    #[test]
    fn deeply_nested_subgraphs_terminate() {
        let mut inner = graph(
            vec![node(OpKind::Relu, "leaf", &["deepest"], &["o"])],
            &[],
            &["o"],
        );
        for level in 0..(MAX_SUBGRAPH_NESTING * 2) {
            inner = graph(
                vec![with_subgraphs(
                    node(OpKind::If, &format!("if{level}"), &["cond"], &["z"]),
                    &[("then_branch", inner)],
                )],
                &[],
                &["z"],
            );
        }
        let n = with_subgraphs(
            node(OpKind::If, "root", &["cond"], &["y"]),
            &[("then_branch", inner)],
        );
        // The only assertion that matters is that this returns at all.
        let captures = subgraph_captures(&n);
        assert!(captures.contains("cond"));
    }

    // ── compute_node_depths ─────────────────────────────────────────────────

    /// A capture is a real data dependency even though it appears nowhere in
    /// `node.inputs`.  Without counting it, an `If` whose body reads a deep
    /// intermediate lands in a depth group that the parallel runner executes
    /// *before* that intermediate exists.
    #[test]
    fn a_captured_tensor_is_a_depth_predecessor() {
        let body = graph(
            vec![node(OpKind::Relu, "body_relu", &["deep"], &["o"])],
            &[],
            &["o"],
        );
        let nodes = vec![
            node(OpKind::Relu, "n0", &["x"], &["a"]),    // depth 0
            node(OpKind::Relu, "n1", &["a"], &["b"]),    // depth 1
            node(OpKind::Relu, "n2", &["b"], &["deep"]), // depth 2
            // `cond` is a graph input, so without the capture this would be depth 0.
            with_subgraphs(
                node(OpKind::If, "branch", &["cond"], &["y"]),
                &[("then_branch", body)],
            ),
        ];
        let depths = Session::compute_node_depths(&nodes, &HashMap::new());
        assert_eq!(depths[0], 0);
        assert_eq!(depths[1], 1);
        assert_eq!(depths[2], 2);
        assert_eq!(
            depths[3], 3,
            "the If must sit after the node producing its captured tensor",
        );
    }

    /// A capture that resolves to an initializer is not a scheduling dependency —
    /// weights exist before the run starts.
    #[test]
    fn a_captured_initializer_does_not_deepen_a_node() {
        let body = graph(
            vec![node(OpKind::Relu, "body_relu", &["w"], &["o"])],
            &[],
            &["o"],
        );
        let nodes = vec![with_subgraphs(
            node(OpKind::If, "branch", &["cond"], &["y"]),
            &[("then_branch", body)],
        )];
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0], vec![1]));
        assert_eq!(Session::compute_node_depths(&nodes, &weights), vec![0]);
    }
}
