use crate::execution_providers::{OpPlacement, ProviderKind};
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod accessors;
// `run_async`/`spawn_run` start inference on a plain `std::thread`, which
// compiles on wasm32-unknown-unknown but panics the instant it is called
// (that target cannot spawn OS threads at all). Nothing on the wasm-facing
// API (`crate::wasm::WasmSession`, which calls `Session::run` synchronously)
// references this module, so it is compiled out entirely on wasm32 rather
// than kept around as dead weight or a fake stub nobody asked for -- a
// caller reaching for `oxionnx::run_async` there gets a compile-time "not
// found" instead of the previous runtime panic.
#[cfg(not(target_arch = "wasm32"))]
pub mod async_run;
mod builder;
pub mod cancellation;
/// The run-scoped map from tensor name to device buffer: which activations stay
/// on the device, and when each one is destroyed.
#[cfg(feature = "gpu")]
pub(crate) mod gpu_activations;
#[cfg(feature = "gpu")]
mod gpu_dispatch;
#[cfg(feature = "gpu")]
mod gpu_owner;
// The summary of this module lives in its own `//!` header rather than here.
// A doc comment at the declaration site and one inside the file are merged,
// and rustdoc then resolves the whole merged block in *this* module's scope —
// which makes every `[`gpu_min_transfer_elements`]`-style link in the header
// unresolvable. Keeping the docs in one place keeps the links working.
pub mod gpu_residency;
mod loading;
pub(crate) mod mixed_precision;
mod run;
pub mod serialize;
mod tests;
pub mod types;

pub use types::{ModelInfo, ModelMetadata, NodeProfile, OptLevel};

#[cfg(not(target_arch = "wasm32"))]
pub use async_run::{block_on, RunFuture, RunHandle};
pub use cancellation::CancellationToken;
pub use serialize::{SessionCacheHeader, SESSION_CACHE_FORMAT_VERSION, SESSION_CACHE_MAGIC};

#[cfg(feature = "gpu")]
pub use gpu_dispatch::GpuExecutionProvider;

pub use builder::SessionBuilder;

/// A loaded ONNX model ready for inference.
pub struct Session {
    pub(crate) sorted_nodes: Vec<crate::graph::Node>,
    pub(crate) weights: HashMap<String, Tensor>,
    pub(crate) input_names: Vec<String>,
    pub(crate) output_names: Vec<String>,
    /// Detailed metadata for graph inputs (from ValueInfoProto).
    pub(crate) input_infos: Vec<oxionnx_core::TensorInfo>,
    /// Detailed metadata for graph outputs (from ValueInfoProto).
    pub(crate) output_infos: Vec<oxionnx_core::TensorInfo>,
    /// Model-level metadata (producer, IR version, opset imports, etc.).
    pub(crate) metadata: ModelMetadata,
    pub(crate) registry: OperatorRegistry,
    pub(crate) profiling_data: Option<Mutex<Vec<NodeProfile>>>,
    pub(crate) pool: Option<Mutex<SizeClassPool>>,
    pub(crate) shape_cache: Option<HashMap<String, Vec<usize>>>,
    /// The parts of a run that do not depend on the inputs, computed once at
    /// build time and reused by every `run()`.
    ///
    /// See [`run::plan::StaticRunPlan`] for what is in it, why it cannot go
    /// stale, and what was deliberately left out of it.
    pub(crate) run_plan: run::plan::StaticRunPlan,
    /// Whether to use rayon-based parallel execution for independent nodes.
    pub(crate) parallel: bool,
    /// Whether to use mixed-precision inference (f16 activations, f32 accumulation).
    pub(crate) mixed_precision: bool,
    /// Operator placement strategy for CPU/GPU routing.
    pub(crate) op_placement: OpPlacement,
    /// Ordered list of execution provider backends to attempt, in priority order.
    ///
    /// When non-empty, the dispatch loop in `run_sequential_inner` iterates this
    /// list for every node and uses the first provider that returns a result.
    /// CPU is always the implicit terminal fallback.
    ///
    /// When empty, the legacy feature-flag heuristic dispatch is used.
    pub(crate) providers: Vec<ProviderKind>,
    /// Current dynamic dimension bindings, updated on each `run()` call.
    /// Maps symbolic dimension names (e.g. "batch_size") to concrete values.
    pub(crate) dynamic_dims: Mutex<HashMap<String, usize>>,
    /// Resolved intermediate tensor shapes for the most recent run.
    ///
    /// Also the memo consulted by `resolve_run_shapes`: it contains the input
    /// shapes it was seeded with, so it is its own cache key.
    pub(crate) resolved_shapes: Mutex<HashMap<String, Vec<usize>>>,
    /// Recently computed shape plans, keyed by the input shapes that produced
    /// them.  See [`ShapePlanCache`].
    pub(crate) shape_plans: ShapePlanCache,
    /// Session-scoped cooperative cancellation token, when one was bound.
    ///
    /// The token is *also* held by every operator in [`Session::registry`] when
    /// this is `Some` — that is where the per-node check actually happens (see
    /// [`cancellation`]).  This field exists so the session can report the
    /// binding back ([`Session::session_cancellation_token`]), so a re-bind can
    /// be recognised as a no-op, and so the streaming generator can poll the
    /// same flag between decode steps.
    pub(crate) cancellation: Option<CancellationToken>,
    /// Per-session rayon thread pool for parallel execution.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) thread_pool: Option<rayon::ThreadPool>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu: Option<gpu_owner::ManagedGpuContext>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<oxionnx_cuda::CudaContext>,
    #[cfg(feature = "directml")]
    pub(crate) dml: Option<oxionnx_directml::DirectMLContext>,
}

// ── ShapePlanCache ──────────────────────────────────────────────────────────

/// How many distinct shape plans one session keeps.
///
/// Small on purpose. The plans this guards against recomputing come from a
/// handful of recurring input shapes (a server alternating batch 1 and batch 8,
/// a pipeline alternating two image sizes); a long history would cost memory
/// proportional to the graph size per entry and buy nothing.
const MAX_CACHED_SHAPE_PLANS: usize = 4;

/// A shape map: tensor name → concrete dimensions.
type ShapeMap = HashMap<String, Vec<usize>>;
/// One cache entry: the input shapes that were inferred from, and the result.
type ShapePlanEntry = (ShapeMap, Arc<ShapeMap>);

/// A tiny MRU cache of shape-inference results, keyed by the *input* shapes
/// that produced them.
///
/// # Why a session needs more than one plan
///
/// `Session` is `Send + Sync` and is routinely parked in an `Arc` behind a web
/// handler, so runs with different batch sizes interleave. The single-slot memo
/// in `resolved_shapes` holds exactly one plan, so two alternating batch sizes
/// miss it on *every* run and each pay a full `infer_shapes` pass over every
/// node — negligible for a toy graph, real for a 1000-node model on a request
/// path. Keeping the last few plans turns that back into a lookup.
///
/// # Correctness
///
/// A plan is a pure function of the input shapes: `infer_shapes` reads only
/// `sorted_nodes` and `weights`, both immutable for the life of the session.
/// The key is therefore the whole input-shape map compared for equality, which
/// is strictly stronger than the single-slot memo's per-input check and cannot
/// return another run's shapes.
///
/// # Contention
///
/// One `Mutex` held for the duration of a `Vec` scan of at most
/// [`MAX_CACHED_SHAPE_PLANS`] entries, each comparison being a map equality on
/// the (few) model inputs. A poisoned lock is not an error — it simply bypasses
/// the cache — so one panicking thread cannot break every later run.
#[derive(Default)]
pub(crate) struct ShapePlanCache {
    /// Most-recently-used first.
    entries: Mutex<Vec<ShapePlanEntry>>,
}

impl ShapePlanCache {
    /// The plan previously computed for exactly these input shapes, if any.
    pub(crate) fn lookup(
        &self,
        input_shapes: &HashMap<String, Vec<usize>>,
    ) -> Option<Arc<HashMap<String, Vec<usize>>>> {
        let mut entries = self.entries.lock().ok()?;
        let hit = entries.iter().position(|(key, _)| key == input_shapes)?;
        // Promote to most-recently-used so a stable alternation never evicts
        // either of its two plans.
        let entry = entries.remove(hit);
        let plan = Arc::clone(&entry.1);
        entries.insert(0, entry);
        Some(plan)
    }

    /// Record `plan` as the result for `input_shapes`, evicting the
    /// least-recently-used entry when full.
    pub(crate) fn store(
        &self,
        input_shapes: &HashMap<String, Vec<usize>>,
        plan: &HashMap<String, Vec<usize>>,
    ) {
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        if let Some(existing) = entries.iter().position(|(key, _)| key == input_shapes) {
            entries.remove(existing);
        }
        entries.insert(0, (input_shapes.clone(), Arc::new(plan.clone())));
        entries.truncate(MAX_CACHED_SHAPE_PLANS);
    }

    /// How many plans are currently cached (tests and diagnostics).
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }
}

// ── Auto-trait invariant ────────────────────────────────────────────────────
//
// `Session: Send + Sync` is a HARD, load-bearing requirement, not a nicety:
//
//   * `run_parallel_inner` (src/session/run/parallel.rs) hands `&self` to
//     rayon's `par_iter()`.  Sharing `&Session` across worker threads requires
//     `Session: Sync`, and moving one into a rayon pool requires `Session: Send`.
//   * Callers routinely park a `Session` in an `Arc` behind a web handler or a
//     thread pool, which requires both.
//
// Today this holds *by accident* on some configurations.  `DirectMLContext` is
// currently a zero-sized struct, so `Session` stays `Send + Sync` for free.  The
// moment it holds a real `ID3D12Device` / `IDMLDevice`, it stops: Windows COM
// interface pointers are apartment-bound and are `!Send + !Sync`.  Without this
// assertion the failure would surface as a wall of inscrutable rayon trait-bound
// errors from *parallel.rs*, far away from the field that actually broke it.
//
// The assertion is enforced under every *native* feature combination — no
// features, `gpu`, `cuda`, `directml`, and all together.  Any provider context
// added to `Session` must be `Send + Sync` (wrap non-thread-safe device handles
// in a `Mutex`, or make the context own a thread-confined worker it talks to
// via channels).
//
// ── Why wasm32 is exempt ────────────────────────────────────────────────────
//
// Every reason above is a *threading* reason, and wasm32-unknown-unknown has no
// threads on any path this crate takes:
//
//   * `run_parallel_inner`'s rayon dependency is declared only for
//     `cfg(not(target_arch = "wasm32"))` (see the root `Cargo.toml`), and
//     `run_internal` forces `use_parallel = false` there, so no `&Session` is
//     ever shared across workers.
//   * `session::async_run` — the only other cross-thread user — is compiled out
//     entirely on wasm32 (see its module gate above).
//   * A wasm session is single-owner by construction: the wasm-bindgen surface
//     that wraps it (`crate::wasm::WasmSession`) is itself `!Send`, because
//     every JS handle it can hold is, so a `Send` bound on `Session` could not
//     buy a caller anything even if it held.
//
// And it cannot hold: with `--features wasm,gpu` the session owns a
// `GpuContext`, whose `wgpu::Device`/`Queue`/`Buffer` are `Rc`-backed
// `WebDevice`/`WebBuffer` handles on the WebGPU backend — `!Send + !Sync` by
// construction, since a `GPUDevice` belongs to the JS agent that created it and
// cannot cross a worker boundary at all. Asserting `Send + Sync` there does not
// catch a bug; it forbids the only GPU backend the browser has.
//
// This is a compile-time check with zero runtime cost: it is evaluated during
// const-checking and produces no code.
#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Session>();
};

#[cfg(test)]
mod shape_plan_cache_tests {
    use super::{ShapePlanCache, MAX_CACHED_SHAPE_PLANS};
    use std::collections::HashMap;

    fn key(batch: usize) -> HashMap<String, Vec<usize>> {
        let mut m = HashMap::new();
        m.insert("x".to_string(), vec![batch, 3]);
        m
    }

    fn plan(marker: usize) -> HashMap<String, Vec<usize>> {
        let mut m = HashMap::new();
        m.insert("y".to_string(), vec![marker]);
        m
    }

    #[test]
    fn a_plan_is_returned_for_the_exact_input_shapes_that_produced_it() {
        let cache = ShapePlanCache::default();
        assert!(cache.lookup(&key(1)).is_none(), "empty cache must miss");

        cache.store(&key(1), &plan(11));
        cache.store(&key(8), &plan(88));

        let hit = cache.lookup(&key(1)).expect("batch 1 must still be cached");
        assert_eq!(hit.get("y"), Some(&vec![11]));
        let hit = cache.lookup(&key(8)).expect("batch 8 must still be cached");
        assert_eq!(hit.get("y"), Some(&vec![88]));
        assert_eq!(cache.len(), 2, "two distinct keys, two entries");
    }

    /// The whole point: two alternating shapes must both stay resident, which is
    /// exactly what the single-slot memo could not do.
    #[test]
    fn a_stable_alternation_never_evicts_either_of_its_two_plans() {
        let cache = ShapePlanCache::default();
        cache.store(&key(1), &plan(11));
        cache.store(&key(8), &plan(88));
        for _ in 0..50 {
            assert!(cache.lookup(&key(1)).is_some());
            assert!(cache.lookup(&key(8)).is_some());
        }
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn the_cache_is_bounded_and_evicts_the_least_recently_used() {
        let cache = ShapePlanCache::default();
        for batch in 0..MAX_CACHED_SHAPE_PLANS {
            cache.store(&key(batch), &plan(batch));
        }
        assert_eq!(cache.len(), MAX_CACHED_SHAPE_PLANS);

        // Touch the oldest so it is no longer the eviction candidate.
        assert!(cache.lookup(&key(0)).is_some());
        cache.store(&key(999), &plan(999));

        assert_eq!(
            cache.len(),
            MAX_CACHED_SHAPE_PLANS,
            "the cache stays bounded"
        );
        assert!(
            cache.lookup(&key(0)).is_some(),
            "a promoted entry must survive the next insertion"
        );
        assert!(
            cache.lookup(&key(1)).is_none(),
            "the least-recently-used entry is the one evicted"
        );
    }

    #[test]
    fn re_storing_a_key_replaces_rather_than_duplicates_it() {
        let cache = ShapePlanCache::default();
        cache.store(&key(1), &plan(11));
        cache.store(&key(1), &plan(22));
        assert_eq!(cache.len(), 1);
        let hit = cache.lookup(&key(1)).expect("still cached");
        assert_eq!(hit.get("y"), Some(&vec![22]), "the newer plan wins");
    }
}
