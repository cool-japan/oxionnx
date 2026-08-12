//! The name → device-buffer map for one run of [`crate::Session::run_gpu_async`].
//!
//! # Why this lives here and not in `oxionnx-gpu`
//!
//! `oxionnx-gpu` owns a [`DeviceTensor`] — a buffer, a shape and the byte
//! budget's claim on it — and knows nothing else about it. Everything that
//! makes a device tensor *an activation of a graph* is here: which name it
//! answers to, which node produced it, which node is the last one that will
//! read it, and whether it is allowed to stay on the device at all. That split
//! is the same one weight residency drew (`session::gpu_dispatch`'s
//! `initializer_key` against `oxionnx_gpu::context::resident`'s opaque keys),
//! and for the same reason: the GPU crate takes slices and buffers, never
//! `OpKind`s or tensor names.
//!
//! # The lifetime rule
//!
//! Node order is fixed for a run, so the last consumer of every name is known
//! before the first node executes ([`RunActivations::new`] computes it). An
//! activation is released the moment its last consumer's node finishes, with
//! nothing to sweep and no eviction policy to tune.
//!
//! Releasing means handing the allocation to the context's reusable-buffer
//! pool ([`RunActivations::dispose`], which carries the measurements behind
//! that choice), so the very next node that needs an output buffer of that size
//! takes it instead of asking the driver for a new one. The bytes stay in the
//! live total until the pool's own LRU/byte bounds evict them, or until the
//! pool is cleared or reclaimed — see [`oxionnx_gpu::TrackedBuffer`] for what
//! actually frees device memory. A run therefore ends with the live-byte total
//! back at its resident-weight baseline **plus** whatever the pool is holding,
//! which is by construction reusable, bounded and reclaimable.
//!
//! # What may stay on the device
//!
//! Three conditions, all necessary:
//!
//! * it is not a graph output — those are read back by definition, and keeping
//!   one resident would make `take_outputs` fail;
//! * some node consumes it — a dead output kept resident would hold its bytes
//!   until the run ended for nobody's benefit;
//! * **every** consumer can bind it in place, in the slot it consumes it in.
//!   That last condition is what keeps the value from being read back a node
//!   later by an op with no resident-capable arm — which would be the same
//!   round trip, moved.
//!
//! A runtime decline can still strand a resident value in front of a consumer
//! that turns out to need it on the host (a budget refusal, a shape the kernel
//! rejects). That case is handled rather than prevented: the value is read back
//! **once**, memoized into the run state as an ordinary host tensor, and the
//! device copy is kept for any later GPU consumer.

use std::collections::{HashMap, HashSet};

use crate::graph::Node;
use oxionnx_gpu::{DeviceTensor, GpuContext};

/// One device-resident value, plus what the session needs to know about it.
struct ResidentValue {
    tensor: DeviceTensor,
    /// Whether a node in this graph produced it.
    ///
    /// The distinction matters in exactly one place: `initializer_key` must not
    /// hand a name to the weight cache once a node has produced a value under
    /// it, and a *promoted* operand (an initializer this run uploaded so its
    /// consumer could dispatch in place) is not such a name.
    node_output: bool,
}

/// Device-resident activations for one run.
///
/// Constructed empty at the top of the run loop and dropped at the bottom, so
/// its `Drop` is the backstop for the per-node releases: whatever a bug leaves
/// behind is destroyed when the run ends, not leaked into the next frame.
#[derive(Default)]
pub(crate) struct RunActivations {
    /// Whether anything may be kept at all. False makes every method here a
    /// no-op and the whole run behave exactly as it did before residency.
    enabled: bool,
    values: HashMap<String, ResidentValue>,
    /// Index of the last node that consumes each name.
    last_use: HashMap<String, usize>,
    /// Names that may be produced straight onto the device.
    keepable: HashSet<String>,
    /// Largest live activation byte total seen this run.
    peak_bytes: u64,
}

impl RunActivations {
    /// Plan a run's residency from its node order and its declared outputs.
    ///
    /// `slot_accepts_resident` answers, for one consumer, whether its GPU arm
    /// can bind a device buffer in the slot it reads the value from. It is a
    /// parameter rather than a match here because the answer belongs to the
    /// dispatcher, which is the code that would have to change if a kernel
    /// gained the ability.
    pub(crate) fn new(
        enabled: bool,
        nodes: &[Node],
        output_names: &[String],
        slot_accepts_resident: impl Fn(&Node, usize) -> bool,
    ) -> Self {
        if !enabled {
            return Self::default();
        }
        let graph_outputs: HashSet<&str> = output_names.iter().map(String::as_str).collect();
        let mut last_use: HashMap<String, usize> = HashMap::new();
        // A name starts keepable and is disqualified by any consumer that
        // cannot bind it in place — "every consumer can" as a fold rather than
        // a second pass over the graph.
        let mut rejected: HashSet<&str> = HashSet::new();
        let mut consumed: HashSet<&str> = HashSet::new();
        for (index, node) in nodes.iter().enumerate() {
            for (slot, input) in node.inputs.iter().enumerate() {
                if input.is_empty() {
                    continue;
                }
                last_use.insert(input.clone(), index);
                consumed.insert(input.as_str());
                if !slot_accepts_resident(node, slot) {
                    rejected.insert(input.as_str());
                }
            }
            // A name a subgraph closes over is read by an `If`/`Loop` body
            // executing on the CPU, and it does *not* appear in `node.inputs` —
            // which is exactly why it needs naming here. Nothing would reject it
            // otherwise (the slot rule only sees declared inputs) and nothing
            // would materialize it either (the run loop walks `node.inputs`
            // too), so a captured value left on the device would be missing from
            // the run state when the body looked for it. Captures are always
            // host-side, so the rule is simply: never keep one.
            for captured in crate::session::run::scheduling::subgraph_captures(node) {
                last_use.insert(captured.to_string(), index);
                rejected.insert(captured);
            }
        }
        let keepable: HashSet<String> = nodes
            .iter()
            .flat_map(|node| node.outputs.iter())
            .filter(|name| !name.is_empty())
            .filter(|name| !graph_outputs.contains(name.as_str()))
            .filter(|name| consumed.contains(name.as_str()))
            .filter(|name| !rejected.contains(name.as_str()))
            .cloned()
            .collect();
        Self {
            enabled: true,
            values: HashMap::new(),
            last_use,
            keepable,
            peak_bytes: 0,
        }
    }

    /// Whether this run keeps anything on the device.
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether a node's output named `name` may be produced straight into a
    /// device buffer.
    pub(crate) fn may_keep(&self, name: &str) -> bool {
        self.enabled && self.keepable.contains(name)
    }

    /// The device buffer holding `name`, if it has one.
    pub(crate) fn get(&self, name: &str) -> Option<&DeviceTensor> {
        self.values.get(name).map(|value| &value.tensor)
    }

    /// Whether a node in this graph produced `name` onto the device.
    ///
    /// Consulted by `initializer_key`, which must never key a name a node has
    /// written — the weight cache would then serve one tensor's bytes for
    /// another's.
    pub(crate) fn holds_node_output(&self, name: &str) -> bool {
        self.values.get(name).is_some_and(|value| value.node_output)
    }

    /// Record a node output that stayed on the device.
    ///
    /// `ctx` disposes of any value this displaces — see [`Self::dispose`].
    pub(crate) fn insert_output(
        &mut self,
        name: &str,
        tensor: DeviceTensor,
        ctx: Option<&GpuContext>,
    ) {
        self.insert(name, tensor, true, ctx);
    }

    /// Record a host operand this run uploaded so its consumer could dispatch
    /// with every operand in place.
    pub(crate) fn insert_promoted(
        &mut self,
        name: &str,
        tensor: DeviceTensor,
        ctx: Option<&GpuContext>,
    ) {
        self.insert(name, tensor, false, ctx);
    }

    fn insert(
        &mut self,
        name: &str,
        tensor: DeviceTensor,
        node_output: bool,
        ctx: Option<&GpuContext>,
    ) {
        // A name is written twice only when a model reuses one for an
        // initializer *and* a node output — legal ONNX, and explicitly handled
        // by `materialize_resident_inputs`'s `holds_node_output` check. The
        // displaced allocation must go the same way a released one does, or
        // that one graph shape would leak a buffer per frame.
        //
        // `session::tests::gpu_activation` builds that shape and runs it on a
        // device. Three tests there, one claim each:
        //
        //   a_node_output_shadowing_a_promoted_initializer_wins_the_name
        //   displacing_a_promoted_operand_recycles_its_buffer_into_the_pool
        //   repeated_shadowed_runs_neither_grow_the_pool_nor_leak_the_displaced_buffer
        //
        // The first proves a real graph run reaches this branch, the second
        // watches the handoff byte for byte, the third is the per-frame
        // accounting that a leak here would break.
        let displaced = self.values.insert(
            name.to_string(),
            ResidentValue {
                tensor,
                node_output,
            },
        );
        if let Some(old) = displaced {
            Self::dispose(old.tensor, ctx);
        }
        self.peak_bytes = self.peak_bytes.max(self.live_bytes());
    }

    /// Release every activation whose last consumer was node `index`.
    ///
    /// Called once per node, after the node has run — including after a node
    /// that declined and ran on the CPU, because "last consumer" is a property
    /// of the graph, not of where the node executed.
    ///
    /// `ctx` is the session's device, when it has one, and decides *how* the
    /// value is released: see [`Self::dispose`]. It is threaded in per call
    /// rather than held as a field because `ManagedGpuContext` is owned by the
    /// `Session` and only ever borrowed — see `super::gpu_owner` for why that
    /// ownership is not negotiable.
    pub(crate) fn release_after(&mut self, index: usize, ctx: Option<&GpuContext>) {
        if self.values.is_empty() {
            return;
        }
        // `HashMap::retain` — what this used to be — drops the removed values
        // inside its own closure, which is exactly what recycling must not do:
        // there is nowhere in a `retain` predicate to hand a `DeviceTensor` to
        // the pool. So the names are collected first and removed one at a time.
        //
        // The obvious way to avoid the intermediate is `HashMap::extract_if`,
        // which yields the removed `(key, value)` pairs and would need neither
        // the `Vec` nor the `String` clones. It stabilized in Rust 1.88 and this
        // workspace declares `rust-version = "1.75"`, so it is not available
        // here. The cost of not having it is one `String` clone per name
        // released at this node plus one `Vec` allocation on the nodes that
        // release anything — `collect` over a filtered iterator allocates
        // nothing when the filter matches nothing — against a scan of the live
        // map that `retain` performed anyway. Measured inside the frame times
        // in `Self::dispose`, so it is already priced in.
        let doomed: Vec<String> = self
            .values
            .keys()
            .filter(|name| self.last_use.get(*name).copied() == Some(index))
            .cloned()
            .collect();
        for name in doomed {
            if let Some(value) = self.values.remove(&name) {
                Self::dispose(value.tensor, ctx);
            }
        }
    }

    /// Give a finished activation's allocation back to the reusable-buffer
    /// pool, or destroy it when there is no context to give it to.
    ///
    /// # Why recycling, and what it cost to find out
    ///
    /// \[w4\] Wave 1 destroyed here, which returns the bytes to the byte budget
    /// immediately and makes "a finished run is back at its resident-weight
    /// baseline" true without clearing the pool. Recycling instead keeps the
    /// allocation in [`GpuContext::pooled_gpu_bytes`] — still live, still
    /// counted, but idle and reusable — so the next node that needs an output
    /// buffer of that size takes it instead of asking the driver.
    ///
    /// The two were measured against each other as an interleaved, paired A/B
    /// inside one process (`examples/w4_recycle_ab.rs`), 25 pairs per process,
    /// four processes per case with the within-iteration order alternating:
    ///
    /// * **InSwapper-128**, residency + `f16` on: paired median
    ///   recycle/destroy `0.976`–`0.981`, recycling faster in 90 of 100 pairs.
    ///   Pool hit rate 96.1% against 4.9%; 7 driver allocations per frame
    ///   against 86.
    /// * **A 48-node chain of 64 KiB activations**, where nothing is
    ///   compute-bound: paired median `0.724`–`0.889`, recycling faster in 98
    ///   of 100 pairs. Pool hit rate 98.9% against 3.1%.
    /// * Outputs were **byte-identical** in all 200 pairs, which is the
    ///   property that matters: a recycled buffer holds the previous tensor's
    ///   bytes rather than the driver's zeroes, so a kernel that failed to write
    ///   its whole output range would diverge here.
    ///
    /// The cost, which is real and is a *ceiling* rather than a trend:
    /// InSwapper's idle pooled total settles at 84.59 MiB rather than 0.38 MiB,
    /// with the pool holding **64 of its 64 permitted entries** — it walks up to
    /// the count bound and is then held there by LRU eviction, at 84.59 MiB of
    /// a 256 MiB
    /// [`DEFAULT_POOL_BYTE_BUDGET`](oxionnx_gpu::DEFAULT_POOL_BYTE_BUDGET). The
    /// small chain, which returns as many buffers per frame as it takes, needs
    /// only 2 entries and 0.12 MiB, so the ceiling is reached by graphs that
    /// hand the pool more than they ask of it, not by every graph.
    ///
    /// How *large* that ceiling is, when it is reached, is `min(64 entries,
    /// 256 MiB)` — and which of the two binds depends on the activation size
    /// mix. InSwapper's 128x128 activations hit the count bound at 84.59 MiB;
    /// a segmentation-scale graph whose activations run to 4 MiB would hit the
    /// byte bound instead and hold a quarter gibibyte idle. Both are the pool's
    /// documented design, not a consequence of recycling, and both are
    /// reclaimed before a decline — but a caller sizing a device budget should
    /// read the ceiling as 256 MiB, not as the 84.59 MiB this model happens to
    /// produce.
    ///
    /// Either way it is reclaimable: `GpuBufferPool::reclaim_for` empties idle
    /// entries before any allocation is declined, so a pooled buffer can never
    /// be the reason a node falls back to the CPU. In the steady state the pool
    /// serves **100%** of a frame's buffer requests on both graphs.
    ///
    /// `None` is the no-device case, where there is no pool to recycle into and
    /// the value could not have been produced in the first place.
    fn dispose(tensor: DeviceTensor, ctx: Option<&GpuContext>) {
        match ctx {
            Some(ctx) => ctx.recycle_device_tensor(tensor),
            None => drop(tensor),
        }
    }

    /// Device bytes currently held by run-scoped activations.
    ///
    /// The *reserved* size of each allocation, so it is directly comparable
    /// with `GpuContext::live_gpu_bytes`.
    pub(crate) fn live_bytes(&self) -> u64 {
        self.values.values().fold(0u64, |acc, value| {
            acc.saturating_add(value.tensor.reserved_bytes())
        })
    }

    /// The largest [`Self::live_bytes`] this run has reached.
    pub(crate) fn peak_bytes(&self) -> u64 {
        self.peak_bytes
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

    /// Every slot of every op accepts a resident operand — the permissive
    /// baseline, so these tests exercise the graph rules rather than the
    /// dispatcher's capability table.
    fn permissive(_node: &Node, _slot: usize) -> bool {
        true
    }

    fn chain() -> Vec<Node> {
        vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Relu, "relu2", &["h"], &["g"]),
            node(OpKind::Add, "add", &["g", "bias"], &["y"]),
        ]
    }

    #[test]
    fn a_graph_output_is_never_keepable() {
        let outputs = vec!["y".to_string()];
        let plan = RunActivations::new(true, &chain(), &outputs, permissive);
        assert!(plan.may_keep("h"));
        assert!(plan.may_keep("g"));
        assert!(
            !plan.may_keep("y"),
            "a graph output must be read back, or take_outputs cannot find it"
        );
    }

    #[test]
    fn a_dead_output_is_not_keepable() {
        // `d` is produced and never read: keeping it resident would pin its
        // bytes for the whole run for nobody.
        let mut nodes = chain();
        nodes.push(node(OpKind::Relu, "dead", &["g"], &["d"]));
        let plan = RunActivations::new(true, &nodes, &["y".to_string()], permissive);
        assert!(!plan.may_keep("d"));
    }

    /// One consumer that cannot bind the value in place disqualifies it, even
    /// when another consumer could — otherwise the round trip is not removed,
    /// only moved to a later node.
    #[test]
    fn one_incapable_consumer_disqualifies_a_name() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Add, "add", &["h", "b"], &["s"]),
            node(OpKind::Softmax, "softmax", &["h"], &["y"]),
        ];
        let capable = |node: &Node, _slot: usize| !matches!(node.op, OpKind::Softmax);
        let plan = RunActivations::new(true, &nodes, &["y".to_string(), "s".to_string()], capable);
        assert!(!plan.may_keep("h"));
    }

    #[test]
    fn a_disabled_plan_keeps_nothing_at_all() {
        let plan = RunActivations::new(false, &chain(), &["y".to_string()], permissive);
        assert!(!plan.is_enabled());
        assert!(!plan.may_keep("h"));
        assert!(plan.get("h").is_none());
        assert_eq!(plan.live_bytes(), 0);
        assert_eq!(plan.peak_bytes(), 0);
    }

    /// The release index is the *last* consumer, so a value read by two nodes
    /// survives the first one.
    #[test]
    fn last_use_is_the_last_consumer_not_the_first() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Add, "add", &["h", "b"], &["s"]),
            node(OpKind::Mul, "mul", &["h", "s"], &["y"]),
        ];
        let plan = RunActivations::new(true, &nodes, &["y".to_string()], permissive);
        assert_eq!(plan.last_use.get("h").copied(), Some(2));
        assert_eq!(plan.last_use.get("s").copied(), Some(2));
        assert_eq!(plan.last_use.get("x").copied(), Some(0));
    }

    /// An initializer consumed by a node gets a last-use index too, which is
    /// what lets a *promoted* operand be released on the same schedule as a
    /// node output. Initializers are excluded from `base_ref_counts`, so this
    /// map cannot be derived from that one.
    #[test]
    fn initializer_operands_are_tracked_for_release() {
        let plan = RunActivations::new(true, &chain(), &["y".to_string()], permissive);
        assert_eq!(plan.last_use.get("bias").copied(), Some(2));
    }

    /// A value an `If`/`Loop` body closes over never appears in that node's
    /// `inputs`, so nothing else in this function would see it. Keeping such a
    /// value on the device would leave the subgraph's CPU operator looking for a
    /// tensor the run state does not have.
    #[test]
    fn a_subgraph_capture_is_never_keepable() {
        use crate::graph::Graph;

        let body = Graph {
            nodes: vec![node(OpKind::Relu, "inner", &["h"], &["inner_out"])],
            input_names: Vec::new(),
            output_names: vec!["inner_out".to_string()],
            ..Default::default()
        };
        let mut if_attrs = Attributes::default();
        if_attrs.graphs.insert("then_branch".to_string(), body);
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Add, "add", &["h", "b"], &["s"]),
            Node {
                op: OpKind::If,
                name: "cond".to_string(),
                inputs: vec!["s".to_string()],
                outputs: vec!["y".to_string()],
                attrs: if_attrs,
            },
        ];
        let plan = RunActivations::new(true, &nodes, &["y".to_string()], permissive);
        assert!(
            !plan.may_keep("h"),
            "`h` is free in the If body, so the body's CPU operator needs it on \
             the host even though every declared consumer could bind it",
        );
    }
}
