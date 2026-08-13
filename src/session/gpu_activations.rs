//! The name → device-buffer map for one run — of
//! [`crate::Session::run_gpu_async`] on the wgpu backend, and of
//! [`crate::Session::run`] on the CUDA one.
//!
//! # One implementation, two backends
//!
//! Everything in this file is about the *graph*: which name answers to which
//! buffer, which node produced it, which node is the last that will read it,
//! and whether it may stay on the device at all. None of that is
//! backend-specific — only the buffer type is. So [`RunActivations`] is
//! generic over it, bounded by [`DeviceActivation`], and the wgpu and CUDA
//! execution paths drive the *same* code rather than two copies that would
//! drift apart the first time either was fixed.
//!
//! The alternative considered and rejected was a CUDA sibling of this module.
//! The rules below — a graph output is never keepable, a dead output is never
//! keepable, one incapable consumer disqualifies a name, a subgraph capture is
//! never keepable — are subtle, load-bearing and identically true for both
//! backends; two copies of them is two places for a residency bug to hide.
//!
//! # Why this lives here and not in `oxionnx-gpu` / `oxionnx-cuda`
//!
//! Those crates own a device tensor — a buffer, a shape and the byte budget's
//! claim on it — and know nothing else about it. Everything that makes a
//! device tensor *an activation of a graph* is here. That split is the same one
//! weight residency drew (`session::gpu_dispatch`'s `initializer_key` against
//! `oxionnx_gpu::context::resident`'s opaque keys), and for the same reason:
//! the backend crates take slices and buffers, never `OpKind`s or tensor
//! names.
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
//! * its consumers satisfy the run's [`KeepPolicy`] — see that type for the two
//!   answers and the arithmetic that separates them.
//!
//! A runtime decline can still strand a resident value in front of a consumer
//! that turns out to need it on the host (a budget refusal, a shape the kernel
//! rejects). That case is handled rather than prevented: the value is read back
//! **once**, memoized into the run state as an ordinary host tensor, and the
//! device copy is kept for any later GPU consumer.

use std::collections::{HashMap, HashSet};

use crate::graph::Node;

/// How many of a value's consumers must be able to bind it in place for the
/// value to stay on the device.
///
/// # The arithmetic, for a value `V` with one capable consumer and one that
/// must have it on the host
///
/// * [`Self::EveryConsumer`] disqualifies `V`. Its producer reads it back
///   (`|V|` down), the capable consumer uploads it again (`|V|` up), the
///   host-only consumer reads the run state for free. **Two crossings.**
/// * [`Self::AnyCapableConsumer`] keeps it. The producer writes it in place
///   (nothing), the capable consumer binds it (nothing), and the host-only
///   consumer materialises it once (`|V|` down), memoized into the run state so
///   a second host consumer finds it there. **One crossing.**
///
/// So the relaxed policy is never worse and is better by one crossing whenever
/// a value has at least one capable consumer and at least one that does not.
/// Both policies agree on the two extremes: all-capable keeps, none-capable
/// does not (there is no benefit to keeping a value only the host will read,
/// and the read-back happens either way — at the producer, or at the consumer).
///
/// # Then why does the wgpu path use the strict one?
///
/// Because it has no cheap materialisation to fall back on in every case: its
/// read-back is an async fence, and its dispatcher declines on a byte budget it
/// must be able to reclaim. The strict rule was measured there and is left
/// exactly as it was; this parameter exists so the CUDA path can take the
/// relaxed one without either backend inheriting the other's trade. The
/// measured difference on the CUDA path is in this wave's report: SCRFD's
/// `Conv -> Relu -> Add` chains are all-capable and unaffected, while
/// InSwapper — whose every second node is a `Pad`/`Slice`/`Unsqueeze`/
/// broadcasting `Mul` with no CUDA arm — is exactly the graph the strict rule
/// leaves entirely on the transferring path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeepPolicy {
    /// Keep a value only when **every** consumer can bind it in place.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    EveryConsumer,
    /// Keep a value when **at least one** consumer can bind it in place; the
    /// rest materialise it once, through the run's convergence point.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    AnyCapableConsumer,
}

/// A device tensor this map can hold, and the backend that owns its memory.
///
/// The two things the graph logic needs from a buffer type, and nothing else:
/// how big it is (for the live/peak byte accounting) and how to give it back
/// when its last consumer has run.
pub(crate) trait DeviceActivation: Sized {
    /// The backend context that owns the allocator this tensor came from.
    ///
    /// Threaded through [`RunActivations::release_after`] per call rather than
    /// held as a field, because the context is owned by the `Session` and only
    /// ever borrowed.
    type Context: ?Sized;

    /// The *reserved* size of the allocation, so a sum over live activations is
    /// directly comparable with what the backend reports as live device bytes.
    fn reserved_bytes(&self) -> u64;

    /// Release the allocation: recycle it into the backend's reusable-buffer
    /// pool when there is a context to recycle into, destroy it when there is
    /// not.
    fn dispose(self, ctx: Option<&Self::Context>);
}

/// One device-resident value, plus what the session needs to know about it.
struct ResidentValue<T> {
    tensor: T,
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
pub(crate) struct RunActivations<T> {
    /// Whether anything may be kept at all. False makes every method here a
    /// no-op and the whole run behave exactly as it did before residency.
    enabled: bool,
    values: HashMap<String, ResidentValue<T>>,
    /// Index of the last node that consumes each name.
    last_use: HashMap<String, usize>,
    /// Names that may be produced straight onto the device.
    keepable: HashSet<String>,
    /// Largest live activation byte total seen this run.
    peak_bytes: u64,
}

/// The empty plan: residency off, nothing keepable, every method a no-op.
///
/// Hand-written rather than derived because `#[derive(Default)]` would demand
/// `T: Default`, and a device buffer has no meaningful default.
impl<T> Default for RunActivations<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            values: HashMap::new(),
            last_use: HashMap::new(),
            keepable: HashSet::new(),
            peak_bytes: 0,
        }
    }
}

impl<T: DeviceActivation> RunActivations<T> {
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
        policy: KeepPolicy,
        slot_accepts_resident: impl Fn(&Node, usize) -> bool,
    ) -> Self {
        if !enabled {
            return Self::default();
        }
        let graph_outputs: HashSet<&str> = output_names.iter().map(String::as_str).collect();
        let mut last_use: HashMap<String, usize> = HashMap::new();
        // Both halves of the consumer census, in one pass: which names some
        // consumer cannot bind in place, and which names some consumer can.
        // Which of the two the keepable set is built from is the `policy`.
        let mut rejected: HashSet<&str> = HashSet::new();
        let mut capable: HashSet<&str> = HashSet::new();
        // Names no policy may keep, whatever their consumers look like. Only
        // subgraph captures land here — see the loop below.
        let mut forbidden: HashSet<&str> = HashSet::new();
        let mut consumed: HashSet<&str> = HashSet::new();
        for (index, node) in nodes.iter().enumerate() {
            for (slot, input) in node.inputs.iter().enumerate() {
                if input.is_empty() {
                    continue;
                }
                last_use.insert(input.clone(), index);
                consumed.insert(input.as_str());
                if slot_accepts_resident(node, slot) {
                    capable.insert(input.as_str());
                } else {
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
                forbidden.insert(captured);
            }
        }
        let keepable: HashSet<String> = nodes
            .iter()
            .flat_map(|node| node.outputs.iter())
            .filter(|name| !name.is_empty())
            .filter(|name| !graph_outputs.contains(name.as_str()))
            .filter(|name| consumed.contains(name.as_str()))
            .filter(|name| !forbidden.contains(name.as_str()))
            .filter(|name| match policy {
                KeepPolicy::EveryConsumer => !rejected.contains(name.as_str()),
                KeepPolicy::AnyCapableConsumer => capable.contains(name.as_str()),
            })
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
    pub(crate) fn get(&self, name: &str) -> Option<&T> {
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
    pub(crate) fn insert_output(&mut self, name: &str, tensor: T, ctx: Option<&T::Context>) {
        self.insert(name, tensor, true, ctx);
    }

    /// Rebind an already-resident value under a second name and shape.
    ///
    /// The device half of a metadata-only reshape — `Reshape`, `Unsqueeze`,
    /// `Squeeze`, `Flatten`, `Identity` — which on a contiguous row-major
    /// buffer changes nothing but the interpretation. `rebind` builds the new
    /// handle (an `Arc` clone of the same allocation, for a backend whose
    /// tensor supports it) and `None` declines, which leaves the caller to run
    /// the op on the host exactly as it did before.
    ///
    /// Returns whether the alias was installed. A `false` is never an error:
    /// the value stays resident under its original name and the aliasing node
    /// falls back, which is the pre-alias behaviour.
    ///
    /// The alias is recorded as a node output, because that is what it is — the
    /// reshaping node produced it — and so the initializer-identity guard
    /// refuses to key it into the weight cache.
    // No caller yet, by design: the op arms that alias —
    // `Reshape`/`Unsqueeze`/`Squeeze`/`Flatten`, 24 `Unsqueeze` nodes in
    // InSwapper alone — are a later stage, and this is the API they were
    // measured against. The tests in this file are what keep it honest
    // meanwhile; delete the `allow` the moment an arm calls it.
    #[allow(dead_code)]
    pub(crate) fn alias_output(
        &mut self,
        source: &str,
        name: &str,
        rebind: impl FnOnce(&T) -> Option<T>,
        ctx: Option<&T::Context>,
    ) -> bool {
        if !self.enabled || !self.keepable.contains(name) {
            return false;
        }
        let Some(aliased) = self
            .values
            .get(source)
            .and_then(|value| rebind(&value.tensor))
        else {
            return false;
        };
        self.insert(name, aliased, true, ctx);
        true
    }

    /// Record a host operand this run uploaded so its consumer could dispatch
    /// with every operand in place.
    // Operand promotion is a wgpu-path feature (`sequential_async`'s
    // `promote_operands_async`); the CUDA path's two-tier gate reaches the same
    // nodes by pricing them rather than by uploading their small operand, so a
    // `cuda`-only build has no caller.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn insert_promoted(&mut self, name: &str, tensor: T, ctx: Option<&T::Context>) {
        self.insert(name, tensor, false, ctx);
    }

    fn insert(&mut self, name: &str, tensor: T, node_output: bool, ctx: Option<&T::Context>) {
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
            old.tensor.dispose(ctx);
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
    pub(crate) fn release_after(&mut self, index: usize, ctx: Option<&T::Context>) {
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
                value.tensor.dispose(ctx);
            }
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

// ─── per-backend aliases ───────────────────────────────────────────────────

/// The wgpu execution path's activation map.
#[cfg(feature = "gpu")]
pub(crate) type GpuActivations = RunActivations<oxionnx_gpu::DeviceTensor>;

/// The CUDA execution path's activation map.
///
/// The same graph logic, the same last-use schedule, the same keepability
/// rules — only the buffer type differs.
#[cfg(feature = "cuda")]
pub(crate) type CudaActivations = RunActivations<oxionnx_cuda::CudaDeviceTensor>;

// ─── the backends ──────────────────────────────────────────────────────────

/// The wgpu backend. Its device tensor's allocation is owned by the
/// [`GpuContext`]'s buffer pool.
#[cfg(feature = "gpu")]
impl DeviceActivation for oxionnx_gpu::DeviceTensor {
    type Context = oxionnx_gpu::GpuContext;

    fn reserved_bytes(&self) -> u64 {
        oxionnx_gpu::DeviceTensor::reserved_bytes(self)
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
    fn dispose(self, ctx: Option<&Self::Context>) {
        match ctx {
            Some(ctx) => ctx.recycle_device_tensor(self),
            None => drop(self),
        }
    }
}

/// The CUDA backend. Its device tensor's allocation is owned by the
/// [`CudaContext`](oxionnx_cuda::CudaContext)'s scratch pool.
///
/// # Recycling here needs no fence, and that is a property of the stream
/// layout rather than of luck
///
/// The wgpu path recycles a finished activation because a queue submission has
/// already been made and the pool's own bookkeeping tracks completion. The
/// CUDA path recycles a buffer whose *kernel work may still be queued*, which
/// is only sound because `oxicuda-dnn`'s `DnnHandle` now builds its BLAS
/// sub-handle on its own stream: every launch and copy the CUDA execution
/// provider issues rides one queue, so the next borrower's kernel is enqueued
/// behind the read that is still in flight and the device executes the queue
/// in order.
///
/// `CudaContext::recycle_activation` is where that is checked rather than
/// assumed — a context whose streams are *not* unified keeps the conservative
/// behaviour, which costs reuse and not correctness. See
/// `oxionnx_cuda::activation`'s module header for the full argument.
///
/// `None` is the no-device case, where there is no pool to recycle into and
/// the value could not have been produced in the first place.
#[cfg(feature = "cuda")]
impl DeviceActivation for oxionnx_cuda::CudaDeviceTensor {
    type Context = oxionnx_cuda::CudaContext;

    fn reserved_bytes(&self) -> u64 {
        oxionnx_cuda::CudaDeviceTensor::reserved_bytes(self)
    }

    fn dispose(self, ctx: Option<&Self::Context>) {
        match ctx {
            Some(ctx) => ctx.recycle_activation(self),
            None => drop(self),
        }
    }
}

/// The lookup `oxionnx-cuda`'s dispatcher performs against this map.
///
/// The map's *policy* — which names may stay resident, when each is released —
/// belongs here, where the node order is known; the *lookup* has to be
/// reachable from the crate that binds the buffers. This impl is the whole
/// interface between the two, and it is deliberately read-only: nothing inside
/// a dispatch can insert into or release from a run's activation map.
#[cfg(feature = "cuda")]
impl oxionnx_cuda::ResidentActivations for CudaActivations {
    fn resident(&self, name: &str) -> Option<&oxionnx_cuda::CudaDeviceTensor> {
        self.get(name)
    }

    fn holds_node_output(&self, name: &str) -> bool {
        RunActivations::holds_node_output(self, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, OpKind};
    use std::cell::RefCell;

    /// A device tensor with no device behind it.
    ///
    /// The graph rules this module owns — keepability, last use, release order,
    /// aliasing — are the same for every backend and need no hardware to state
    /// or to check. Exercising them against a double rather than against
    /// `oxionnx_gpu::DeviceTensor` is what lets the whole suite run on a
    /// GPU-less host *and* keeps it honest about which half is being tested:
    /// everything that needs a real allocation lives in
    /// `session::tests::gpu_activation`, which requires a device.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct FakeTensor {
        label: &'static str,
        bytes: u64,
    }

    /// Where a disposed [`FakeTensor`] goes, so a test can assert that release
    /// happened at all and in the right order.
    #[derive(Default)]
    struct Recycler {
        released: RefCell<Vec<&'static str>>,
    }

    impl DeviceActivation for FakeTensor {
        type Context = Recycler;

        fn reserved_bytes(&self) -> u64 {
            self.bytes
        }

        fn dispose(self, ctx: Option<&Self::Context>) {
            if let Some(ctx) = ctx {
                ctx.released.borrow_mut().push(self.label);
            }
        }
    }

    fn fake(label: &'static str, bytes: u64) -> FakeTensor {
        FakeTensor { label, bytes }
    }

    /// A plan over [`FakeTensor`]s — the type the graph-rule tests below use.
    type Plan = RunActivations<FakeTensor>;

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
        let plan = Plan::new(
            true,
            &chain(),
            &outputs,
            KeepPolicy::EveryConsumer,
            permissive,
        );
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
        let plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
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
        let plan = Plan::new(
            true,
            &nodes,
            &["y".to_string(), "s".to_string()],
            KeepPolicy::EveryConsumer,
            capable,
        );
        assert!(!plan.may_keep("h"));
    }

    #[test]
    fn a_disabled_plan_keeps_nothing_at_all() {
        let plan = Plan::new(
            false,
            &chain(),
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
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
        let plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
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
        let plan = Plan::new(
            true,
            &chain(),
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
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
        let plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        assert!(
            !plan.may_keep("h"),
            "`h` is free in the If body, so the body's CPU operator needs it on \
             the host even though every declared consumer could bind it",
        );
    }

    // ── the residency bookkeeping itself ────────────────────────────────

    /// A value is released exactly when its last consumer's node has run —
    /// not before, and not left behind at the end of the run.
    #[test]
    fn a_value_is_released_after_its_last_consumer_and_not_before() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Add, "add", &["h", "b"], &["s"]),
            node(OpKind::Mul, "mul", &["h", "s"], &["y"]),
        ];
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        plan.insert_output("h", fake("h", 1024), Some(&recycler));
        plan.insert_output("s", fake("s", 512), Some(&recycler));
        assert_eq!(plan.live_bytes(), 1536);

        plan.release_after(0, Some(&recycler));
        assert!(
            recycler.released.borrow().is_empty(),
            "`h` is read again at node 2; releasing it after node 0 would free a live value",
        );
        plan.release_after(1, Some(&recycler));
        assert!(recycler.released.borrow().is_empty());
        plan.release_after(2, Some(&recycler));
        let mut released = recycler.released.borrow().clone();
        released.sort_unstable();
        assert_eq!(released, vec!["h", "s"]);
        assert_eq!(plan.live_bytes(), 0);
        assert_eq!(plan.peak_bytes(), 1536, "the peak survives the release");
    }

    /// Displacing a name hands the old allocation to the recycler rather than
    /// dropping it on the floor — the one graph shape (an initializer and a
    /// node output sharing a name) that reaches `insert`'s displacement branch.
    #[test]
    fn displacing_a_name_recycles_the_value_it_replaced() {
        let nodes = chain();
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        plan.insert_promoted("h", fake("promoted", 256), Some(&recycler));
        plan.insert_output("h", fake("produced", 256), Some(&recycler));
        assert_eq!(*recycler.released.borrow(), vec!["promoted"]);
        assert!(
            plan.holds_node_output("h"),
            "the node output must win the name, or the weight cache would key an activation",
        );
        assert_eq!(
            plan.live_bytes(),
            256,
            "exactly one value is live under `h`"
        );
    }

    /// A promoted operand is not a node output, so the initializer-identity
    /// guard still lets its name be keyed into the weight cache.
    #[test]
    fn a_promoted_operand_is_not_a_node_output() {
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            true,
            &chain(),
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        plan.insert_promoted("bias", fake("bias", 64), Some(&recycler));
        assert!(plan.get("bias").is_some());
        assert!(!plan.holds_node_output("bias"));
    }

    // ── aliasing ────────────────────────────────────────────────────────

    /// A metadata-only reshape rebinds the same allocation under the new name.
    #[test]
    fn an_alias_binds_the_source_allocation_under_a_second_name() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Unsqueeze, "unsqueeze", &["h"], &["u"]),
            node(OpKind::Relu, "relu2", &["u"], &["y"]),
        ];
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        plan.insert_output("h", fake("h", 1024), Some(&recycler));
        assert!(plan.alias_output("h", "u", |t| Some(t.clone()), Some(&recycler)));
        assert_eq!(plan.get("u"), Some(&fake("h", 1024)));
        assert!(
            plan.holds_node_output("u"),
            "the reshaping node produced this name, so it must not be weight-keyed",
        );
    }

    /// A rebind the backend declines leaves the plan exactly as it was, so the
    /// aliasing node falls back to the host with its source still resident.
    #[test]
    fn a_declined_rebind_changes_nothing() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Unsqueeze, "unsqueeze", &["h"], &["u"]),
            node(OpKind::Relu, "relu2", &["u"], &["y"]),
        ];
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        plan.insert_output("h", fake("h", 1024), Some(&recycler));
        assert!(!plan.alias_output("h", "u", |_| None, Some(&recycler)));
        assert!(plan.get("u").is_none());
        assert!(plan.get("h").is_some(), "the source is untouched");
    }

    /// A name the graph rules refuse to keep cannot be aliased into either —
    /// `u` is a graph output here, so it must be read back.
    #[test]
    fn an_alias_obeys_the_keepability_rules() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Unsqueeze, "unsqueeze", &["h"], &["u"]),
        ];
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            true,
            &nodes,
            &["u".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        plan.insert_output("h", fake("h", 1024), Some(&recycler));
        assert!(!plan.alias_output("h", "u", |t| Some(t.clone()), Some(&recycler)));
    }

    /// A disabled plan aliases nothing, exactly as it keeps nothing.
    #[test]
    fn a_disabled_plan_refuses_to_alias() {
        let recycler = Recycler::default();
        let mut plan = Plan::new(
            false,
            &chain(),
            &["y".to_string()],
            KeepPolicy::EveryConsumer,
            permissive,
        );
        assert!(!plan.alias_output("h", "u", |t| Some(t.clone()), Some(&recycler)));
    }

    // ── the two keep policies ───────────────────────────────────────────

    /// The relaxed policy keeps a value one capable consumer can bind, even
    /// though another consumer will need it on the host.
    ///
    /// The arithmetic is on [`KeepPolicy`]: strict costs a read-back at the
    /// producer *plus* an upload at the capable consumer; relaxed costs one
    /// materialisation and nothing else.
    #[test]
    fn the_relaxed_policy_keeps_a_name_one_consumer_cannot_bind() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Add, "capable", &["h", "b"], &["s"]),
            node(OpKind::Softmax, "host_only", &["h"], &["y"]),
        ];
        let capable = |node: &Node, _slot: usize| !matches!(node.op, OpKind::Softmax);
        let outputs = vec!["y".to_string(), "s".to_string()];

        let strict = Plan::new(true, &nodes, &outputs, KeepPolicy::EveryConsumer, capable);
        assert!(!strict.may_keep("h"));

        let relaxed = Plan::new(
            true,
            &nodes,
            &outputs,
            KeepPolicy::AnyCapableConsumer,
            capable,
        );
        assert!(
            relaxed.may_keep("h"),
            "one capable consumer is enough: the host-only consumer materialises it once, \
             which is the crossing the producer would have paid anyway",
        );
    }

    /// Neither policy keeps a value **no** consumer can bind: the read-back
    /// happens either way, and keeping it would only move it later while
    /// pinning device memory in between.
    #[test]
    fn neither_policy_keeps_a_name_no_consumer_can_bind() {
        let nodes = vec![
            node(OpKind::Relu, "relu", &["x"], &["h"]),
            node(OpKind::Softmax, "host_only", &["h"], &["y"]),
        ];
        let capable = |node: &Node, _slot: usize| !matches!(node.op, OpKind::Softmax);
        for policy in [KeepPolicy::EveryConsumer, KeepPolicy::AnyCapableConsumer] {
            let plan = Plan::new(true, &nodes, &["y".to_string()], policy, capable);
            assert!(
                !plan.may_keep("h"),
                "{policy:?} kept a value nothing can bind"
            );
        }
    }

    /// The two policies agree whenever every consumer is capable — which is
    /// what makes this a widening rather than a different rule.
    #[test]
    fn the_policies_agree_when_every_consumer_is_capable() {
        for policy in [KeepPolicy::EveryConsumer, KeepPolicy::AnyCapableConsumer] {
            let plan = Plan::new(true, &chain(), &["y".to_string()], policy, permissive);
            assert!(plan.may_keep("h"), "{policy:?}");
            assert!(plan.may_keep("g"), "{policy:?}");
            assert!(
                !plan.may_keep("y"),
                "{policy:?}: a graph output is never keepable"
            );
        }
    }

    /// A subgraph capture is forbidden under **both** policies, and that has to
    /// survive the split: the capture rule is not "no consumer can bind it"
    /// (the `If` node's declared inputs may all be bindable), it is "the body
    /// runs on the host and reads this name out of the run state".
    #[test]
    fn a_subgraph_capture_is_forbidden_under_the_relaxed_policy_too() {
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
        let plan = Plan::new(
            true,
            &nodes,
            &["y".to_string()],
            KeepPolicy::AnyCapableConsumer,
            permissive,
        );
        assert!(
            !plan.may_keep("h"),
            "`h` is free in the If body, and the relaxed policy must not reach past that",
        );
    }
}
