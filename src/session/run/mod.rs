mod dispatch;
mod entry;
mod parallel;
pub(super) mod plan;
pub(crate) mod scheduling;
mod sequential;
/// The asynchronous node loop that lets WebGPU work be awaited rather than
/// blocked on. Compiled only with the `gpu` feature, which is the only backend
/// that has an asynchronous interface at all.
#[cfg(feature = "gpu")]
mod sequential_async;
mod shape_resolution;
mod state;
mod text;
mod typed;

use crate::graph::Node;
use crate::OnnxError;

// ── Per-run bookkeeping maps ────────────────────────────────────────────────

/// Live reference counts for one run: tensor name → number of consumers that
/// have not executed yet.  A tensor whose count reaches zero is taken out of the
/// run state and its buffer recycled into the pool.
///
/// # Why the keys are borrowed, and why the hasher is not SipHash
///
/// This was `std::collections::HashMap<String, usize>`, rebuilt from scratch at
/// the top of every `run()` by walking every node input and calling
/// `entry(inp.clone())`.  Two costs, both paid per inference and both pure
/// overhead:
///
/// * **~1000 `String` allocations for a 500-node model**, every one of them a
///   copy of a name that `self.sorted_nodes` already owns and that outlives the
///   run.  The keys are now `&'g str` borrowed straight out of the graph, so the
///   count map allocates exactly one table and no strings at all.
/// * **SipHash-1-3 on every insert and every lookup.**  These maps are internal,
///   never exposed to a caller, and never fed attacker-chosen keys independent of
///   the model — the model *is* the input, and a model that hash-floods this map
///   has already been parsed and topologically sorted.  `hashbrown`'s default
///   hasher costs a few cycles per short key instead of ~1 ns, and was already a
///   declared (but entirely unused) dependency of this workspace.
///
/// The lifetime `'g` is the graph borrow the run holds (`&self` for the whole of
/// `run_internal`), which is what makes borrowing sound: every key points into
/// `Session::sorted_nodes` or `Session::output_names`, neither of which can be
/// touched during a run.
pub(crate) type RefCounts<'g> = hashbrown::HashMap<&'g str, usize>;

/// The names the graph declares as outputs — the set that keeps a tensor alive
/// past its last consumer and forbids mutating it in place.
///
/// Same reasoning as [`RefCounts`]: borrowed keys, fast hasher, one lookup per
/// node input per run.
pub(crate) type OutputSet<'g> = hashbrown::HashSet<&'g str>;

/// The names a subgraph reads out of its enclosing scope, borrowed from the node
/// that carries the subgraph.
///
/// Returned by [`scheduling::subgraph_captures`] and folded into [`RefCounts`],
/// so it must borrow from the same graph the counts key on.
pub(crate) type CaptureSet<'g> = hashbrown::HashSet<&'g str>;

/// The error every execution path raises for a node whose operator is not in
/// this session's registry.
///
/// # Why the registry — and not `OpKind::Unknown` — is the gate
///
/// `OpKind::parse` maps any op string it does not recognise to
/// `OpKind::Unknown(name)`, and every run loop used to `continue` past such a
/// node.  Loading a model containing one operator this engine does not implement
/// therefore *succeeded*, and `run()` either failed much later with a confusing
/// `TensorNotFound` at some downstream node, or returned a result map silently
/// missing whichever graph outputs depended on it.
///
/// The fix is deliberately **not** "reject `OpKind::Unknown`".  `OpKind` is a
/// fixed enum of the ~167 ops this crate knows by name, while the registry is
/// user-extensible ([`crate::SessionBuilder::with_registry`]): a custom operator
/// registered under a name outside that enum is `OpKind::Unknown(name)` *and*
/// perfectly runnable.  Rejecting the enum variant would make every custom op
/// unrunnable.  So the gate is exactly the one the engine's own subgraph
/// executor already uses (`oxionnx-ops/src/control_flow/functions.rs`): look the
/// op up in the registry, and raise [`OnnxError::UnsupportedOp`] when it is not
/// there.
pub(super) fn unsupported_op_error(node: &Node) -> OnnxError {
    OnnxError::UnsupportedOp(format!(
        "no operator registered for '{}' (node '{}'); this model uses an operator \
         oxionnx does not implement — register one with SessionBuilder::with_registry \
         or re-export the model without it",
        node.op.as_str(),
        node.name,
    ))
}
