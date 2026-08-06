//! CUDA context wrapper for oxionnx-cuda.
//!
//! [`CudaContext`] holds a CUDA device context together with a [`DnnHandle`]
//! (which itself contains a `BlasHandle`, PTX cache, and stream).  A single
//! `CudaContext` is created once at `Session` build time and shared across all
//! op dispatches within a session run.
//!
//! ## Activation is OPT-IN, and that is a decision, not an oversight
//!
//! This repository has no CUDA-capable host: every line under this crate is
//! type-checked and unit-tested (the pure, allocation-light logic — shape
//! decomposition, broadcast rules, attribute decoding, the
//! [`crate::reference`] oracle) on a machine with no GPU, and nothing here
//! has ever been run
//! against real silicon in this codebase's own CI.
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
//!
//! The bar for flipping [`ACTIVATION_ENV_VAR`]'s default to on is the same as
//! `oxionnx-directml`'s: run with `OXIONNX_CUDA_VERIFY=1` on real hardware,
//! across more than one input shape per op, and confirm zero mismatches.

use std::sync::{Arc, OnceLock};

use oxicuda_dnn::handle::DnnHandle;
use oxicuda_driver::{Context, Device};

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
pub struct CudaContext {
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

        Some(Self { context, dnn })
    }
}

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
