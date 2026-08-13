//! CUDA context wrapper for oxionnx-cuda.
//!
//! [`CudaContext`] holds a CUDA device context together with a [`DnnHandle`]
//! (which itself contains a `BlasHandle`, PTX cache, and stream).  A single
//! `CudaContext` is created once at `Session` build time and shared across all
//! op dispatches within a session run.
//!
//! ## Activation is OPT-IN, and that is a decision, not an oversight
//!
//! This crate's own CI has no CUDA-capable host: every line under this crate
//! is type-checked and unit-tested there (the pure, allocation-light logic —
//! shape decomposition, broadcast rules, attribute decoding, the
//! [`crate::reference`] oracle) on a machine with no GPU. A `gpu-tests`
//! Cargo feature — off by default, run manually with `cargo test -p
//! oxionnx-cuda --features gpu-tests` on a CUDA-capable machine — covers the
//! rest: real dispatches against a real device, including the cross-thread
//! `CudaContext` regression tests in this crate's own `tests` module.
//! Enabling that feature on a host with no device is harmless: every
//! on-device fixture returns `Option<CudaContext>` and its tests skip, so
//! `--all-features` stays green on a CPU-only machine (see the feature's
//! own comment in `Cargo.toml`). What
//! stays true regardless is the point this section is making: *default*
//! `cargo test`/CI never touches a GPU, so the table below is what actually
//! gates whether a user's inference run does.
//!
//! A GPU kernel bug does not crash.  A transposed index, a truncated
//! reduction, an axis the kernel silently assumes is the last one — each of
//! these returns a buffer of exactly the right *length* and *shape*, full of
//! plausible-looking wrong numbers, which then propagate silently through the
//! rest of the inference graph.  Shipping that **on by default** would mean
//! quietly corrupting the output of every user who happened to build with
//! `--features cuda` on a CUDA-capable machine.  (This is not hypothetical:
//! findings a8-0 through a8-4 fixed in this same wave were exactly that kind
//! of bug, and `try_cuda_dispatch` returned `Ok(Some(wrong_data))` — a
//! successful answer — for every one of them.)
//!
//! So, mirroring `oxionnx-directml`'s identical rationale for the identical
//! reason:
//!
//! | Environment variable | Default | Effect |
//! |---|---|---|
//! | [`ACTIVATION_ENV_VAR`] (`OXIONNX_CUDA`) | **off** | [`CudaContext::try_new`] returns `None` until this is set (or the embedder passes [`Activation::Enabled`]).  `Session` holds `cuda: None`, and every node runs on the CPU. |
//! | [`crate::reference::VERIFY_ENV_VAR`] (`OXIONNX_CUDA_VERIFY`) | off | Shadow-compare every dispatched op against the CPU oracle in [`crate::reference`].  A mismatch is a kernel *failure*: the wrong numbers are thrown away, not returned. |
//! | [`STRICT_ENV_VAR`] (`OXIONNX_CUDA_STRICT`) | off | A shadow-verification *failure* becomes a hard `Err` instead of a silent CPU fallback. |
//! | [`crate::graph_cache::GRAPH_ENV_VAR`] (`OXIONNX_CUDA_GRAPH`) | off | Record each repeated fixed-shape `MatMul`/`Gemm` dispatch into a CUDA graph once and replay it with `cuGraphLaunch` thereafter.  Measured between -10% and +11% per call **depending on the shape** — see [`mod@crate::graph_cache`] for the table and for why the ceiling is that low. |
//!
//! ## `OXIONNX_CUDA_GRAPH` and `OXIONNX_CUDA_VERIFY` do not compose, deliberately
//!
//! Setting both leaves graphs **off** (with a `warn!`), because verification's
//! whole value is that it grades the code path production runs. Grading a
//! replayed graph instead would quietly change what a `VERIFY=1` run is
//! evidence for — it would stop being evidence that the *kernels* are right.
//! `OXIONNX_CUDA_STRICT` is orthogonal to both: it only decides whether a
//! verification mismatch is fatal.
//!
//! The bar for flipping [`ACTIVATION_ENV_VAR`]'s default to on is the same as
//! `oxionnx-directml`'s: run with `OXIONNX_CUDA_VERIFY=1` on real hardware,
//! across more than one input shape per op, and confirm zero mismatches.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use oxicuda_dnn::handle::DnnHandle;
use oxicuda_driver::{Context, Device, Module, Stream};

use crate::error::CudaDispatchError;
use crate::residency::{CacheCounters, DeviceCaches, Operand, PooledBuffer, WeightId};

// ─── activation policy ────────────────────────────────────────────────────

/// Set this to acquire a GPU: `OXIONNX_CUDA=1`.
///
/// Unset — the default — means [`CudaContext::try_new`] returns `None` and
/// this crate does nothing at all.  See the [module docs](self) for why the
/// default is off.
///
/// `1`, `true`, `yes`, `on` (any case) enable it.  Unset, empty, `0`,
/// `false`, `no` and `off` disable it.  Anything else is treated as
/// **enabled**: a user who typed `OXIONNX_CUDA=please` wants the GPU, and
/// silently ignoring them would be a lie.
pub const ACTIVATION_ENV_VAR: &str = "OXIONNX_CUDA";

/// Set this to turn a shadow-verification *failure* into a hard error:
/// `OXIONNX_CUDA_STRICT=1`.
///
/// Same truthiness rules as [`ACTIVATION_ENV_VAR`].  See [`FailurePolicy`]
/// for exactly what it promotes.
pub const STRICT_ENV_VAR: &str = "OXIONNX_CUDA_STRICT";

/// Whether a [`CudaContext`] is permitted to acquire a GPU.
///
/// A future `SessionBuilder::with_cuda(bool)` maps to [`Self::Enabled`] /
/// [`Self::Disabled`]; everything else gets [`Self::EnvOptIn`], which is the
/// [`Default`] and what plain [`CudaContext::try_new`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Acquire a GPU **only** when [`ACTIVATION_ENV_VAR`] is set to a
    /// truthy value.  The default.
    #[default]
    EnvOptIn,
    /// The embedder asked for CUDA explicitly.  Acquire regardless of the
    /// environment — a deliberate bypass of the opt-in gate, exactly as
    /// clear an opt-in as an environment variable.
    Enabled,
    /// Never acquire, whatever the environment says.
    Disabled,
}

impl Activation {
    /// May a context built under this policy go looking for a device?
    ///
    /// Only the *permission*: acquisition can still fail (no driver, no
    /// device 0), in which case [`CudaContext::try_new_with`] returns
    /// `None` anyway.
    #[must_use]
    pub fn permits_acquisition(self) -> bool {
        match self {
            Self::Enabled => true,
            Self::Disabled => false,
            Self::EnvOptIn => env_flag(ACTIVATION_ENV_VAR),
        }
    }
}

/// What CUDA dispatch does when shadow verification (`OXIONNX_CUDA_VERIFY=1`)
/// finds that a kernel's output does **not** match the CPU oracle.
///
/// The distinction from a plain *decline* matters: a decline
/// (`try_cuda_dispatch` returning `Ok(None)`) is a normal, expected outcome
/// for an op/shape this crate does not accelerate — the CPU kernel one line
/// away computes it correctly, no signal needed.  A verify *mismatch* means
/// the GPU was asked to compute something, claimed success, and got the
/// numbers wrong.  That is never silent, whichever policy is active — see
/// `crate::reference::shadow_verify` (crate-private).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailurePolicy {
    /// Log the mismatch at `error!` and fall back to the CPU.  Inference
    /// stays *correct*; the node is merely no longer accelerated.  The
    /// default.
    #[default]
    Fallback,
    /// Return the mismatch as an `Err`.  For CI, for benchmarks, and for
    /// anyone who would rather know that their "GPU-accelerated" run is
    /// silently running on the CPU.
    Strict,
}

impl FailurePolicy {
    /// The policy this process is running under, from [`STRICT_ENV_VAR`].
    ///
    /// Read once and cached: the value cannot change within a process, and
    /// this is consulted on the dispatch path of every claimed node when
    /// verification is on.
    #[must_use]
    pub fn current() -> Self {
        static STRICT: OnceLock<bool> = OnceLock::new();
        if *STRICT.get_or_init(|| env_flag(STRICT_ENV_VAR)) {
            Self::Strict
        } else {
            Self::Fallback
        }
    }
}

/// Read `name` from the environment under this crate's shared truthiness
/// policy.
fn env_flag(name: &str) -> bool {
    parse_env_flag(std::env::var(name).ok().as_deref())
}

/// The pure core of [`env_flag`], so the policy can be tested without
/// touching the process environment — which is global, racy under a
/// threaded test runner, and cached besides.
///
/// This is the crate's single definition of "truthy";
/// [`crate::reference::verify_enabled`] routes through it too, so all three
/// flags (`OXIONNX_CUDA`, `OXIONNX_CUDA_VERIFY`, `OXIONNX_CUDA_STRICT`)
/// answer to exactly the same spellings.
pub(crate) fn parse_env_flag(value: Option<&str>) -> bool {
    match value {
        None => false,
        // Note the direction: anything *unrecognised* is ENABLED.  A user
        // who typed `OXIONNX_CUDA=please` has unambiguously asked for the
        // feature, and quietly handing them the old behaviour because we
        // did not recognise their spelling would be exactly the class of
        // silent, plausible-looking wrongness this crate exists to avoid.
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
    }
}

// ─── the context ───────────────────────────────────────────────────────────

/// Encapsulates the CUDA context and DNN handle used for accelerated dispatch.
///
/// Construction is fallible: if the caller has not opted in (see the
/// [module docs](self)), no CUDA device is available, or initialisation
/// fails, `try_new` returns `None` so the caller can fall back to CPU/wgpu.
///
/// # It is also the session's cache
///
/// One `CudaContext` is built per `oxionnx::Session` and shared by every
/// dispatch that session makes, which makes it the natural — and the only
/// correct — owner of anything that should outlive a single node:
///
/// * a pool of reusable device buffers, so a graph re-run per video frame
///   stops paying `cuMemAlloc`/`cuMemFree` for the same size classes it asked
///   for on the previous frame;
/// * device copies of graph initializers, so invariant weights cross the bus
///   once per session instead of once per node per frame;
/// * compiled PTX modules, so a kernel is JIT-compiled once rather than on
///   every dispatch.
///
/// See [`mod@crate::residency`] for the first two. All three die with the
/// context, on the context they belong to — which is what the field order
/// below is about.
pub struct CudaContext {
    /// Session-lifetime device-buffer pool and weight residency.
    ///
    /// **Declared first on purpose.** Struct fields drop in declaration
    /// order, and every buffer in here frees itself with a `cuMemFree`
    /// against `context`. Dropping the context first would leave those frees
    /// pointing at a destroyed context; `DeviceBuffer`'s own `Drop` has no
    /// guard against that (unlike `Module`'s, which tracks its owning
    /// context). Same reasoning for `modules` below.
    pub(crate) caches: DeviceCaches,
    /// Recorded CUDA graphs for repeated fixed-shape dispatches, and the
    /// device buffers whose addresses those recordings baked in.
    ///
    /// **Declared alongside `caches`, above `context`, for the same reason**:
    /// every buffer it owns frees itself with a `cuMemFree` against `context`,
    /// and every `GraphExec` it owns destroys itself against the same context.
    /// Empty (and untouched on every dispatch) unless
    /// [`GRAPH_ENV_VAR`](crate::graph_cache::GRAPH_ENV_VAR) is set.
    pub(crate) graphs: crate::graph_cache::GraphCache,
    /// Compiled PTX modules for this crate's *own* kernels, keyed by kernel
    /// name.
    ///
    /// Complementary to — not a duplicate of — the caches inside
    /// `oxicuda-blas`'s `BlasHandle`/`GemmDispatcher` and `oxicuda-dnn`'s
    /// convolution engines: those cover kernels *those* crates generate, and
    /// nothing there can see the elementwise and softmax templates this crate
    /// instantiates itself. Without this, `cuda_elementwise` regenerated PTX
    /// and called `cuModuleLoadData` on **every** dispatch — a JIT compile per
    /// node per frame, for the 57 elementwise nodes an InSwapper frame runs.
    modules: RwLock<HashMap<String, Arc<Module>>>,
    /// The underlying CUDA driver context.  Kept alive here so the context
    /// is not dropped while the DNN handle (and kernels it compiles) are in use.
    pub(crate) context: Arc<Context>,
    /// DNN handle (owns stream, BLAS handle, PTX cache, SM version).
    pub(crate) dnn: DnnHandle,
}

impl CudaContext {
    /// Return a reference to the underlying CUDA driver context.
    #[must_use]
    pub fn driver_context(&self) -> &Arc<Context> {
        &self.context
    }

    // ── session-lifetime caches ────────────────────────────────────────────

    /// Borrow an uninitialised device buffer of at least `len` `f32`
    /// elements from this context's pool.
    ///
    /// The contents are whatever the previous borrower left; see
    /// [`PooledBuffer`] for the obligations that creates.
    ///
    /// # Errors
    ///
    /// Propagates an allocation failure when the pool has to grow.
    pub(crate) fn scratch(&self, len: usize) -> Result<PooledBuffer<'_>, CudaDispatchError> {
        self.caches.scratch(len)
    }

    /// Upload `data` onto `stream` for this dispatch, reusing a resident
    /// device copy when `id` names bytes this context has already uploaded.
    ///
    /// Pass `id = None` for anything that is not a graph initializer — an
    /// activation's bytes change every frame, and caching them would serve
    /// the previous frame's numbers.
    ///
    /// # Errors
    ///
    /// Propagates allocation and upload failures.
    pub(crate) fn operand(
        &self,
        id: Option<WeightId<'_>>,
        label: &'static str,
        data: &[f32],
        stream: &Stream,
    ) -> Result<Operand<'_>, CudaDispatchError> {
        self.caches.operand(id, label, data, stream)
    }

    /// A device copy this context already holds under `id`, resolved
    /// **without** the host bytes.
    ///
    /// For a caller that would have to *build* those bytes — a `transB=1`
    /// weight needs a full host transpose before it could be uploaded —
    /// asking first turns an `O(k*n)` per-frame copy into an `O(1)` lookup.
    /// A caller whose bytes are already sitting in the `Tensor` gains nothing
    /// and should go straight to [`Self::operand`].
    ///
    /// `uploaded` is the element count the caller would have uploaded; a
    /// resident copy of any other length is not a match.
    pub(crate) fn resident_operand(
        &self,
        id: WeightId<'_>,
        label: &'static str,
        uploaded: usize,
    ) -> Option<Operand<'_>> {
        self.caches.resident(id, label, uploaded)
    }

    /// A compiled module for `kernel_name`, generating and JIT-compiling its
    /// PTX only on the first request.
    ///
    /// `gen_ptx` runs only on a miss, so a caller may build the PTX string
    /// eagerly in the closure without paying for it on the hot path.
    ///
    /// # Errors
    ///
    /// Propagates `gen_ptx`'s error, or the driver's if the generated PTX
    /// fails to compile.
    pub(crate) fn module(
        &self,
        kernel_name: &str,
        gen_ptx: impl FnOnce() -> Result<String, CudaDispatchError>,
    ) -> Result<Arc<Module>, CudaDispatchError> {
        // Fast path: a read lock and a clone of an `Arc`.
        if let Ok(modules) = self.modules.read() {
            if let Some(module) = modules.get(kernel_name) {
                return Ok(Arc::clone(module));
            }
        }

        // Slow path: generate and compile *outside* the write lock, so a slow
        // JIT never blocks another thread's cache hit. A concurrent compile of
        // the same kernel is harmless — whichever module is inserted last
        // wins, and every `Arc` already handed out stays valid.
        let ptx = gen_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx).map_err(CudaDispatchError::Driver)?);
        if let Ok(mut modules) = self.modules.write() {
            modules.insert(kernel_name.to_string(), Arc::clone(&module));
        }
        Ok(module)
    }

    /// A snapshot of what this context's buffer pool and weight cache have
    /// done since it was built.
    ///
    /// The number to watch is
    /// [`weight_bytes_uploaded`](CacheCounters::weight_bytes_uploaded):
    /// take a snapshot before and after a steady-state frame and it must come
    /// out **zero**. Anything else means some initializer is being
    /// re-uploaded every frame — a cache conflict, or an operand that is not
    /// being keyed at all.
    #[must_use]
    pub fn cache_counters(&self) -> CacheCounters {
        self.caches.counters()
    }

    /// Whether the initializer named `name` currently has a device copy.
    #[must_use]
    pub fn is_weight_resident(&self, name: &str) -> bool {
        self.caches.is_resident(name)
    }

    /// Turn CUDA graph capture/replay on or off for this context, overriding
    /// [`GRAPH_ENV_VAR`](crate::graph_cache::GRAPH_ENV_VAR).
    ///
    /// The embedder-facing counterpart of [`Activation::Enabled`] — an
    /// explicit request is as clear an opt-in as an environment variable, and
    /// this crate already takes that position for GPU acquisition itself.
    ///
    /// Safe to call at any time, including while another thread is
    /// dispatching: it decides only whether *subsequent* dispatches consult
    /// the graph cache. Already-recorded graphs and the buffers they own are
    /// untouched, so switching back on reuses them rather than re-recording.
    ///
    /// Turning it on does **not** override the interaction documented in the
    /// [module docs](self): under `OXIONNX_CUDA_VERIFY=1` a context is built
    /// with graphs off, and turning them on here is exactly as unwise as it
    /// sounds — verification would then grade replays instead of kernels.
    pub fn set_graph_capture(&self, on: bool) {
        self.graphs.set_enabled(on);
    }

    /// Whether this context currently takes the CUDA graph path.
    #[must_use]
    pub fn graph_capture_enabled(&self) -> bool {
        self.graphs.enabled()
    }

    /// How many CUDA graphs this context has recorded, and how many of those
    /// keys are poisoned (a capture that failed and permanently fell back).
    ///
    /// Always `(0, 0)` unless
    /// [`GRAPH_ENV_VAR`](crate::graph_cache::GRAPH_ENV_VAR) is set. A
    /// *poisoned* count above zero is the signal worth watching: it means some
    /// node could not be recorded and is running through ordinary launches, so
    /// a run with graphs "on" is partly not.
    #[must_use]
    pub fn graph_stats(&self) -> (usize, usize) {
        self.graphs.stats()
    }

    /// Device bytes this context is holding in its caches: resident weights
    /// plus buffers sitting idle in the pool.
    ///
    /// Excludes buffers currently lent out to an in-flight dispatch, which
    /// are live working memory rather than cache.
    #[must_use]
    pub fn cached_device_bytes(&self) -> usize {
        self.caches.bytes()
    }

    /// Release every cached device buffer, returning the bytes freed.
    ///
    /// For a caller that has finished with a session's models but is keeping
    /// the session alive, or that needs the memory back for something else.
    /// Correctness is unaffected: the next dispatch re-allocates and
    /// re-uploads exactly as the first one did.
    pub fn release_device_caches(&self) -> usize {
        self.caches.clear()
    }

    /// Attempt to create a `CudaContext` for device 0, honouring
    /// [`ACTIVATION_ENV_VAR`].  **Never panics.**
    ///
    /// Returns `None` unless the user has opted in with `OXIONNX_CUDA=1`.
    /// This is the constructor `oxionnx::Session` calls, so a default build
    /// of a default session on a default CUDA-capable box does **not**
    /// touch the GPU.  See the [module docs](self).
    #[must_use]
    pub fn try_new() -> Option<Self> {
        Self::try_new_with(Activation::default())
    }

    /// Attempt to initialise under an explicit [`Activation`] policy.
    /// **Never panics.**
    ///
    /// This is what a `SessionBuilder::with_cuda(bool)` would map onto:
    /// `true` → [`Activation::Enabled`], `false` → [`Activation::Disabled`].
    ///
    /// Returns `None` on every host with no CUDA driver, on a host whose
    /// device 0 cannot be acquired, and — crucially — whenever the caller
    /// has not opted in.
    #[must_use]
    pub fn try_new_with(activation: Activation) -> Option<Self> {
        if !activation.permits_acquisition() {
            tracing::debug!(
                env = ACTIVATION_ENV_VAR,
                "CUDA not activated; set {ACTIVATION_ENV_VAR}=1 (or call .with_cuda(true)) to \
                 enable it.  Every node will run on the CPU."
            );
            return None;
        }

        match oxicuda_driver::init() {
            Ok(()) => {}
            Err(_) => return None,
        }

        let dev = Device::get(0).ok()?;
        let context = Arc::new(Context::new(&dev).ok()?);

        // Activate the context on the current thread.
        context.set_current().ok()?;

        let dnn = DnnHandle::new(&context).ok()?;

        if crate::reference::verify_enabled() {
            tracing::warn!(
                env = crate::reference::VERIFY_ENV_VAR,
                "CUDA shadow verification is ON: every GPU op will also be computed on the CPU \
                 and compared.  This roughly doubles the cost of every claimed node — it is a \
                 diagnostic mode, not a production one."
            );
        }
        if FailurePolicy::current() == FailurePolicy::Strict {
            tracing::info!(
                env = STRICT_ENV_VAR,
                "CUDA strict mode is ON: a shadow-verification mismatch will abort the run \
                 instead of falling back to the CPU."
            );
        }

        if crate::graph_cache::graph_enabled() {
            tracing::info!(
                env = crate::graph_cache::GRAPH_ENV_VAR,
                "CUDA graph capture is ON for this context."
            );
        }

        Some(Self {
            caches: DeviceCaches::new(),
            graphs: crate::graph_cache::GraphCache::new(),
            modules: RwLock::new(HashMap::new()),
            context,
            dnn,
        })
    }
}

// ── Auto-trait invariant ────────────────────────────────────────────────────
//
// `CudaContext: Send + Sync` is load-bearing, not incidental. `oxionnx::Session`
// holds one in a field and asserts `Session: Send + Sync` itself (see its
// "Auto-trait invariant" block) because its parallel runner hands `&Session` to
// rayon workers and callers park sessions in an `Arc`. A field that stopped
// being `Sync` would surface there as a wall of rayon trait-bound errors in
// `session/run/parallel.rs`, far from the field that broke it.
//
// This became worth asserting *here* the moment this struct grew interior
// mutability: the caches below are the exact kind of field that silently drops
// `Sync` if someone reaches for a `RefCell` or an `Rc` for convenience.
// Compile-time only; it produces no code.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaContext>();
};

#[cfg(test)]
mod tests {
    use super::{parse_env_flag, Activation, CudaContext, FailurePolicy};

    #[test]
    fn try_new_never_panics() {
        // No CUDA host exists in this repository's dev/CI environment; this only asserts
        // acquisition does not panic. `None` is the only outcome this test can observe here,
        // both because activation defaults to off AND because no device exists — the two
        // cannot be distinguished on this host, which is exactly why the pure logic below is
        // tested directly instead.
        let ctx = CudaContext::try_new();
        let _ = ctx.is_some();
    }

    #[test]
    fn disabled_activation_refuses_before_it_ever_touches_a_device() {
        assert!(!Activation::Disabled.permits_acquisition());
        assert!(CudaContext::try_new_with(Activation::Disabled).is_none());
    }

    #[test]
    fn explicit_enable_permits_acquisition_regardless_of_the_environment() {
        assert!(Activation::Enabled.permits_acquisition());
    }

    #[test]
    fn the_default_activation_is_the_env_opt_in() {
        assert_eq!(Activation::default(), Activation::EnvOptIn);
    }

    #[test]
    fn the_default_failure_policy_is_a_cpu_fallback() {
        assert_eq!(FailurePolicy::default(), FailurePolicy::Fallback);
    }

    #[test]
    fn failure_policy_current_never_panics() {
        // Exercises the `OnceLock` read path; the actual value depends on whatever the test
        // runner's environment happens to hold, so only "does not panic" is asserted.
        let _ = FailurePolicy::current();
    }

    #[test]
    fn env_flag_truthiness_treats_unrecognised_values_as_enabled() {
        // Off.
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("no"),
            Some("off"),
        ] {
            assert!(!parse_env_flag(off), "{off:?} must be OFF");
        }
        // Case- and whitespace-insensitive off.
        for off in [Some(" FALSE "), Some("Off"), Some("NO")] {
            assert!(!parse_env_flag(off), "{off:?} must be OFF");
        }
        // On, including the deliberate "anything unrecognised is on" rule.
        for on in [
            Some("1"),
            Some("true"),
            Some("YES"),
            Some("on"),
            Some("please"),
        ] {
            assert!(parse_env_flag(on), "{on:?} must be ON");
        }
    }
}
