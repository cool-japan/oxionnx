//! The DirectML execution context: acquisition policy, thread-safety, and the hardware
//! self-check.
//!
//! # 1. Threading — a compile gate, not advice
//!
//! `oxionnx::Session` stores a [`DirectMLContext`] **by value**, and `Session` must be
//! `Sync`: `session/run/parallel.rs` does
//! `work_items.par_iter().map(|…| OpContext { weights: Some(&self.weights), … })`, which
//! captures `&self` into a rayon closure.  `session/mod.rs` pins that with a
//! `const _: () = { assert_send_sync::<Session>(); }`.
//!
//! The `windows` crate marks every COM interface `!Send + !Sync`.  So the moment a real
//! `ID3D12Device` goes into this struct, **the root crate stops compiling for Windows** —
//! not with a message about COM, but with a wall of rayon trait-bound errors in a file
//! nobody was editing.
//!
//! The fix is the `BackendCell` below: all COM state lives inside a `Mutex`, the
//! `unsafe impl Send` is made on the *cell* and never on the context, and `Sync` is then
//! **earned** from the standard library's `impl<T: Send> Sync for Mutex<T>` rather than
//! asserted.  Read that `SAFETY` block; the mutex is load-bearing, not decorative.
//!
//! Verify with:
//! `cargo clippy --target x86_64-pc-windows-gnu -p oxionnx --features directml --all-targets -- -D warnings`
//!
//! # 2. Activation is OPT-IN, and that is a decision, not an oversight
//!
//! This crate's GPU path **has never been executed**.  This repository has no Windows host
//! and no D3D12 adapter; every line under `backend/d3d12` and `backend/dml` is
//! type-checked and lint-checked by a cross-target `cargo clippy` and by nothing else.
//!
//! A GPU kernel bug does not crash.  A transposed index, a missing UAV barrier, a
//! `GroupsX` that disagrees with the dispatch grid — each of these returns a buffer of
//! exactly the right *length* and *shape*, full of plausible-looking wrong numbers, which
//! then propagate silently through the rest of the inference graph.  Nothing downstream
//! can tell.  Shipping that **on by default** would mean quietly corrupting the output of
//! every user who happened to build with `--features directml` on Windows.
//!
//! So:
//!
//! | Environment variable | Default | Effect |
//! |---|---|---|
//! | [`ACTIVATION_ENV_VAR`] (`OXIONNX_DIRECTML`) | **off** | Nothing else in this crate runs until this is set (or the embedder passes [`Activation::Enabled`]).  [`DirectMLContext::try_new`] returns `None`, `Session` holds `dml: None`, and every node runs on the CPU. |
//! | [`crate::reference::VERIFY_ENV_VAR`] (`OXIONNX_DIRECTML_VERIFY`) | off | Shadow-compare **every** dispatched op against the CPU oracle.  A mismatch is a kernel *failure*: the wrong numbers are thrown away, not returned. |
//! | [`STRICT_ENV_VAR`] (`OXIONNX_DIRECTML_STRICT`) | off | A kernel *failure* becomes a hard `Err` instead of a silent CPU fallback.  A *decline* is still `Ok(None)` — see [`FailurePolicy`]. |
//! | `OXIONNX_DIRECTML_ALLOW_WARP` | off | Permit the software (WARP) adapter.  This is what makes [`DirectMLContext::self_check`] runnable on a Windows VM with no GPU — the **only** environment in which this code can be exercised at all. |
//!
//! The bar for flipping the default to on is: run
//! `cargo run -p oxionnx-directml --example directml_self_check` on real hardware, on more
//! than one vendor's part, and paste the reports.

use std::sync::{Mutex, OnceLock};

use crate::backend::{Backend, BackendKind};
use crate::error::{DirectMLError, Result};
use crate::plan::{BinaryOp, ElementwisePlan, MatMulPlan, UnaryOp};
use crate::reference::{self, ComparisonReport, SelfCheckReport};

// ─── activation policy ───────────────────────────────────────────────────────

/// Set this to acquire a GPU: `OXIONNX_DIRECTML=1`.
///
/// Unset — the default — means [`DirectMLContext::try_new`] returns `None` and this crate
/// does nothing at all.  See this module's documentation for why the default is off and
/// what it would take to change it.
///
/// `1`, `true`, `yes`, `on` (any case) enable it.  Unset, empty, `0`, `false`, `no` and
/// `off` disable it.  Anything else is treated as **enabled**: a user who typed
/// `OXIONNX_DIRECTML=please` wants the GPU, and silently ignoring them would be a lie.
pub const ACTIVATION_ENV_VAR: &str = "OXIONNX_DIRECTML";

/// Set this to turn a GPU *failure* into a hard error: `OXIONNX_DIRECTML_STRICT=1`.
///
/// Same truthiness rules as [`ACTIVATION_ENV_VAR`].  See [`FailurePolicy`] for exactly
/// which errors it promotes — and, just as importantly, which it does not.
pub const STRICT_ENV_VAR: &str = "OXIONNX_DIRECTML_STRICT";

/// Whether a [`DirectMLContext`] is permitted to acquire a GPU.
///
/// The embedder's `SessionBuilder::with_directml(bool)` maps to [`Self::Enabled`] /
/// [`Self::Disabled`]; everything else gets [`Self::EnvOptIn`], which is the [`Default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Acquire a GPU **only** when [`ACTIVATION_ENV_VAR`] is set to a truthy value.
    ///
    /// The default, and what plain [`DirectMLContext::try_new`] uses.
    #[default]
    EnvOptIn,
    /// The embedder asked for DirectML explicitly.  Acquire regardless of the environment.
    ///
    /// This is a *deliberate* bypass of the opt-in gate: a caller who wrote
    /// `.with_directml(true)` in their own source has opted in as clearly as an
    /// environment variable ever could.
    Enabled,
    /// Never acquire, whatever the environment says.  `.with_directml(false)`.
    Disabled,
}

impl Activation {
    /// May a context built under this policy go looking for a device?
    ///
    /// Note that this is only the *permission*.  Acquisition can still fail — no adapter,
    /// no engine — in which case [`DirectMLContext::try_new_with`] returns `None` anyway.
    #[must_use]
    pub fn permits_acquisition(self) -> bool {
        match self {
            Self::Enabled => true,
            Self::Disabled => false,
            Self::EnvOptIn => env_flag(ACTIVATION_ENV_VAR),
        }
    }
}

/// What the router does when a kernel **fails** (as opposed to **declining**).
///
/// The distinction is the whole point of this type, and getting it wrong is how the
/// original code hid a total GPU failure behind a perfectly normal-looking `Ok(None)`:
///
/// * **Declined** ([`DirectMLError::Declined`]) — "this op / shape / dtype is not ours".
///   A normal, expected, *correct* outcome; the CPU kernel one line away computes it
///   properly.  **Never** promoted by [`Self::Strict`].
/// * **Malformed** ([`DirectMLError::ShapeMismatch`]) — "your model is broken".  Also
///   never promoted: the CPU operator will hit the same inputs and raise a far better
///   diagnostic than this crate can, and pre-empting it with a DirectML-flavoured error
///   would just make the user's real bug harder to read.
/// * **Failed** (everything else — `Win32`, `ShaderCompile`, `DispatchFailed`,
///   `TransferError`, `DeviceInitFailed`, `LockPoisoned`, and a
///   `OXIONNX_DIRECTML_VERIFY` mismatch) — "the GPU broke".  This is what
///   [`Self::Strict`] promotes to a hard `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailurePolicy {
    /// Log the failure at `error!` and fall back to the CPU.  Inference stays *correct*;
    /// it is merely no longer accelerated.  The default.
    #[default]
    Fallback,
    /// Return the failure as an `Err`.  For CI, for benchmarks, and for anyone who would
    /// rather know that their "GPU-accelerated" run is silently running on the CPU.
    Strict,
}

impl FailurePolicy {
    /// The policy this process is running under, from [`STRICT_ENV_VAR`].
    ///
    /// Read once and cached: the value cannot change within a process, and this is
    /// consulted on the dispatch path of every claimed node.
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

/// Read `name` from the environment under this crate's shared truthiness policy.
fn env_flag(name: &str) -> bool {
    parse_env_flag(std::env::var(name).ok().as_deref())
}

/// The pure core of [`env_flag`], so the policy can be tested without touching the process
/// environment — which is global, racy under a threaded test runner, and cached besides.
///
/// This is the crate's single definition of "truthy"; [`crate::reference::verify_enabled`]
/// routes through it too, so all three flags answer to exactly the same spellings.
pub(crate) fn parse_env_flag(value: Option<&str>) -> bool {
    match value {
        None => false,
        // Note the direction: anything *unrecognised* is ENABLED.  A user who typed
        // `OXIONNX_DIRECTML=please` has unambiguously asked for the GPU, and quietly
        // handing them a CPU run because we did not recognise their spelling would be the
        // exact class of silent, plausible-looking wrongness this crate exists to avoid.
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
    }
}

// ─── the Send/Sync boundary ──────────────────────────────────────────────────

/// The COM state, and the *only* place in this crate where `Send` is asserted.
///
/// Wrapping `Backend` rather than putting the `unsafe impl` on [`DirectMLContext`] is not
/// a stylistic choice.  A blanket `unsafe impl Send + Sync for DirectMLContext` is a
/// promise about the whole struct **forever**: add a `Rc<Cell<…>>` field to it in two
/// years and the compiler will say nothing, because the blanket impl already blessed it.
/// Narrowing the claim to this newtype means `DirectMLContext` derives its auto-traits
/// from its fields in the ordinary way, so any future non-`Sync` field is a compile error
/// in the crate that adds it — which is where the bug would be.
struct BackendCell(Backend);

// SAFETY: what is being claimed here is `Send` — "a `Backend` may be *moved* to another
// thread" — and nothing more.  `Sync` for `Mutex<BackendCell>` then follows from the
// standard library's `impl<T: ?Sized + Send> Sync for Mutex<T>`; it is earned from the
// mutex, not asserted here.  That distinction is the entire safety argument.
//
// `Backend` transitively owns COM interface pointers (`ID3D12Device`,
// `ID3D12CommandQueue`, `ID3D12CommandAllocator`, `ID3D12GraphicsCommandList`,
// `ID3D12Fence`, `ID3D12PipelineState`, `ID3D12RootSignature`, `ID3D12Resource`,
// `ID3D12DescriptorHeap`, `IDMLDevice`, `IDMLCommandRecorder`, `IDMLCompiledOperator`,
// `IDMLBindingTable`) and non-atomic interior mutability (`D3d12Core::next_fence_value`
// and `::lost` are `Cell`; `GpuBuffer::state` is a `Cell`; `PsoCache` and
// `DmlEngine::cache` are `RefCell`).  The `windows` crate marks every COM interface
// `!Send + !Sync` unconditionally, because a binding generator cannot know an object's
// threading model.  We can, and it differs per object:
//
//   * `ID3D12Device` and `ID3D12CommandQueue` are **free-threaded**: D3D12 guarantees
//     their methods may be called concurrently from any thread.  Moving them is trivial.
//   * `ID3D12CommandAllocator` and `ID3D12GraphicsCommandList` are **not**.  D3D12
//     requires that at most one thread record into a list at a time, and that an
//     allocator not be `Reset` while the GPU still owns a list built from it.  Two
//     `Session::run(&self)` calls from an `Arc<Session>` on two threads, both recording
//     into the single `D3d12Core::list`, is driver-level undefined behaviour: no
//     `HRESULT`, no panic, just a corrupted command stream.  **This is what makes the
//     mutex load-bearing rather than decorative.**
//   * `IDMLBindingTable::Reset` rewrites descriptors into the shader-visible heap from
//     the CPU, immediately; two threads resetting one table race on that heap.
//   * The `Cell`/`RefCell` state is unsynchronised by construction.  A cross-thread
//     `RefCell` double-borrow is a data race, not a tidy panic.
//
// Every one of those objects is reachable **only** through `DirectMLContext::inner`, and
// `DirectMLContext::with_backend` is the only path to it.  It takes the mutex and holds it
// across the entire record → submit → fence-wait sequence (each engine method ends in
// `D3d12Core::submit_and_wait`, which blocks until the GPU has finished with the list), so
// at most one thread is ever inside the D3D12/DirectML recording path and no submission
// outlives the lock.  Nothing escapes: `with_backend`'s closure is
// `for<'a> FnOnce(&'a Backend) -> Result<R, E>` with `R` an early-bound type parameter, so
// `R` *cannot name* `'a` — the compiler itself forbids returning a COM pointer, a `Cell`
// borrow, or a GPU-mapped pointer out of the guarded region.
//
// Finally, `Send` requires that the value may be *dropped* on a thread other than the one
// that created it.  D3D12 and DirectML objects are not apartment-bound OLE objects; they
// are free-threaded C++ objects whose `Release` may be called from any thread, so this
// holds.
unsafe impl Send for BackendCell {}

// ─── the context ─────────────────────────────────────────────────────────────

/// Opaque DirectML execution context.
///
/// Construct with [`Self::try_new`] (environment opt-in) or [`Self::try_new_with`] (the
/// embedder decides).  Both return `None` on every non-Windows target, on Windows without
/// a working D3D12 adapter, and — crucially — whenever the user has not opted in.  See
/// this module's documentation for the activation policy and why it is off by default.
///
/// `Send + Sync`, derived from the fields.  See `BackendCell`.
pub struct DirectMLContext {
    /// All COM state, behind the mutex that makes concurrent `Session::run(&self)` calls
    /// safe.  The parallel session runner already serialises GPU nodes ("Phase 1: serial
    /// GPU dispatch"), so within one `run()` this is uncontended; across two concurrent
    /// `run()`s on one `Arc<Session>` it is exactly what stops two threads recording into
    /// one command allocator.
    inner: Mutex<BackendCell>,
    /// Cached out of the backend so `backend_kind()` needs no lock.  A context's backend
    /// is fixed for its whole life.
    kind: BackendKind,
    /// Cached out of the backend so `adapter_name()` needs neither a lock nor an
    /// allocation.
    adapter: String,
}

impl DirectMLContext {
    /// Attempt to initialise, honouring [`ACTIVATION_ENV_VAR`].  **Never panics.**
    ///
    /// Returns `None` unless the user has opted in with `OXIONNX_DIRECTML=1`.  This is the
    /// constructor `oxionnx::Session` calls, so a default build of a default session on a
    /// default Windows box does **not** touch the GPU.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        Self::try_new_with(Activation::default())
    }

    /// Attempt to initialise under an explicit [`Activation`] policy.  **Never panics.**
    ///
    /// This is what a `SessionBuilder::with_directml(bool)` maps onto:
    /// `true` → [`Activation::Enabled`], `false` → [`Activation::Disabled`].
    ///
    /// Resolution order, once permitted:
    ///
    /// 1. `D3d12Core::try_new()` — device, COMPUTE queue, allocator, list, fence, event.
    /// 2. `DmlEngine::new(&core)` → [`BackendKind::DirectMl`].
    /// 3. else `HlslEngine::new(&core)` → [`BackendKind::Hlsl`].
    /// 4. else `None`.
    ///
    /// `None` — rather than an inactive context — is deliberate on every failure path:
    /// `oxionnx`'s session runner keys "this node is GPU-eligible" off `dml.is_some()`, so
    /// a present-but-declining context would drag every claimed node into the runner's
    /// **serial** GPU phase, watch it decline, and run it on the CPU anyway.  That turns
    /// parallel CPU work into serial CPU work — a GPU provider that makes inference
    /// slower.
    #[must_use]
    pub fn try_new_with(activation: Activation) -> Option<Self> {
        if !activation.permits_acquisition() {
            tracing::debug!(
                env = ACTIVATION_ENV_VAR,
                "DirectML not activated; set {ACTIVATION_ENV_VAR}=1 (or call \
                 .with_directml(true)) to enable it.  Every node will run on the CPU."
            );
            return None;
        }

        let backend = Backend::try_new()?;
        let kind = backend.kind();
        let adapter = backend.adapter_name();

        if reference::verify_enabled() {
            tracing::warn!(
                env = reference::VERIFY_ENV_VAR,
                "DirectML shadow verification is ON: every GPU op will also be computed on \
                 the CPU and compared.  This roughly doubles the cost of every claimed \
                 node — it is a diagnostic mode, not a production one."
            );
        }
        if FailurePolicy::current() == FailurePolicy::Strict {
            tracing::info!(
                env = STRICT_ENV_VAR,
                "DirectML strict mode is ON: a GPU failure will abort the run instead of \
                 falling back to the CPU.  (A declined op still falls back — that is not a \
                 failure.)"
            );
        }

        Some(Self {
            inner: Mutex::new(BackendCell(backend)),
            kind,
            adapter,
        })
    }

    /// Whether this context is backed by a live GPU backend.
    ///
    /// Always `true` for a context that *exists* — [`Self::try_new`] returns `None` rather
    /// than handing back an inactive context, for the reason given there.  It stays an
    /// explicit predicate because [`crate::try_directml_dispatch`] opens with it, and
    /// because on non-Windows it is the monomorphic `false` that lets LLVM fold that
    /// function's entire body away.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.kind.is_gpu()
    }

    /// Which backend this context resolved to.
    #[must_use]
    pub fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    /// The DXGI adapter description, e.g. `"NVIDIA GeForce RTX 4090"`.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter
    }

    /// Run the GPU path on tiny fixed inputs and diff every result against
    /// [`crate::reference`]'s CPU oracle.
    ///
    /// **This is the only mechanism that can validate this crate's Windows-only code.**  It
    /// cannot run in this repository's CI — there is no Windows host and no D3D12 GPU here.
    /// Run it on real hardware:
    ///
    /// ```text
    /// OXIONNX_DIRECTML=1 cargo run -p oxionnx-directml --example directml_self_check
    /// ```
    ///
    /// On a Windows VM with no GPU, add `OXIONNX_DIRECTML_ALLOW_WARP=1` to let it run on
    /// the software adapter.  WARP is a *conformant* D3D12 implementation, so it will
    /// catch a wrong index, a wrong root-signature slot or a wrong tensor descriptor just
    /// as well as silicon will — it will simply do it slowly.  What it cannot catch is the
    /// class of bug that is correct on one IHV and garbage on another (a missing UAV
    /// barrier is the canonical one).
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when no GPU backend is present.  Anything else is a
    /// genuine failure of the GPU path, including a **decline** from a backend on one of
    /// these fixed shapes — every shape below is one this crate claims to support, so
    /// declining it is a bug, not a fallback.
    pub fn self_check(&self, tolerance: f32) -> Result<SelfCheckReport> {
        self.self_check_reports(tolerance).map(|(report, _)| report)
    }

    /// As [`Self::self_check`], but also returns the per-op [`ComparisonReport`]s.
    ///
    /// [`SelfCheckReport`] rolls every op up into one pass/fail and a worst deviation,
    /// which is the right thing to assert on.  It is *not* the right thing to read when
    /// something has gone wrong: a `ComparisonReport` names the op, the number of
    /// mismatched elements, the worst deviation *and the linear index of the first one* —
    /// and that index is the single most diagnostic number available.  A first mismatch at
    /// element 256 says "the second thread group is wrong"; at `N/2` it says "half the
    /// dispatch grid never ran".  `examples/directml_self_check.rs` prints these.
    ///
    /// # Errors
    /// As [`Self::self_check`].
    pub fn self_check_reports(
        &self,
        tolerance: f32,
    ) -> Result<(SelfCheckReport, Vec<ComparisonReport>)> {
        if !self.is_active() {
            return Err(DirectMLError::Declined(
                "self_check: no GPU backend is present".into(),
            ));
        }
        self.with_backend(|backend| run_self_check(backend, tolerance))
    }

    /// Take the backend lock for the duration of one GPU submission.
    ///
    /// The closure holds the mutex across the entire record → submit → fence-wait sequence.
    /// No COM pointer can escape it — see the `SAFETY` block on `BackendCell`, which
    /// explains why the compiler enforces that rather than the programmer.
    ///
    /// # Errors
    /// Whatever `f` returns, plus [`DirectMLError::LockPoisoned`] (mapped through
    /// `E: From<DirectMLError>`) when another thread panicked while holding the lock.
    ///
    /// A poisoned lock is **not** recovered from.  A panic mid-recording leaves the command
    /// list and its allocator in an unknown state, and `ID3D12CommandAllocator::Reset` on an
    /// allocator the GPU may still own is undefined behaviour that nothing would catch.  A
    /// permanently dead context that reports an honest error every time beats a live one
    /// that corrupts memory once.
    pub(crate) fn with_backend<R, E>(
        &self,
        f: impl FnOnce(&Backend) -> core::result::Result<R, E>,
    ) -> core::result::Result<R, E>
    where
        E: From<DirectMLError>,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| E::from(DirectMLError::LockPoisoned))?;
        f(&guard.0)
    }
}

// ─── the self-check program ──────────────────────────────────────────────────

/// Rows of the self-check MatMul / Gemm.
const CHECK_M: usize = 4;
/// Inner dimension of the self-check MatMul / Gemm.
const CHECK_K: usize = 3;
/// Columns of the self-check MatMul / Gemm.
const CHECK_N: usize = 5;
/// Shape of the self-check elementwise operands: 24 elements, rank 3, not a multiple of
/// the 256-wide thread group — so the shader's `if (i >= N) return;` guard is exercised,
/// which is the guard that a "just dispatch `numel` threads" kernel gets wrong.
const CHECK_ELEMENTWISE_SHAPE: [usize; 3] = [2, 3, 4];

/// Every op the router claims, run on fixed inputs and diffed against the oracle.
///
/// The shapes are all ones this crate supports, so a [`DirectMLError::Declined`] from here
/// is a **bug**, not a fallback, and is propagated as an error rather than recorded as a
/// pass.
fn run_self_check(
    backend: &Backend,
    tolerance: f32,
) -> Result<(SelfCheckReport, Vec<ComparisonReport>)> {
    let mut report = SelfCheckReport::new(backend.kind(), backend.adapter_name(), tolerance);
    let mut comparisons: Vec<ComparisonReport> = Vec::new();

    // ── MatMul: [4, 3] · [3, 5] ──────────────────────────────────────────────
    let plan = MatMulPlan::matmul(&[CHECK_M, CHECK_K], &[CHECK_K, CHECK_N])?;
    let a = fixture(CHECK_M * CHECK_K, 0x1234_5678);
    let b = fixture(CHECK_K * CHECK_N, 0x9abc_def0);
    let gpu = backend.matmul(&plan, &a, &b, None)?;
    comparisons.push(reference::verify_matmul(&plan, &a, &b, None, &gpu)?);

    // ── Gemm: alpha · Aᵀ-free · Bᵀ + beta · C, with C broadcast along the rows ──
    //
    // `trans_b` exercises the one place the two engines genuinely diverge: the HLSL path
    // CPU-transposes `B` before upload, while the DirectML path sets
    // `DML_GEMM_OPERATOR_DESC::TransB` and copies nothing.  A `C` of shape `[5]` (not
    // `[4, 5]`) exercises the bias broadcast, which is likewise a CPU epilogue on one
    // engine and a 0-stride tensor descriptor on the other.  If these two agree with the
    // oracle, the two engines agree with each other.
    let gemm = MatMulPlan::gemm(
        &[CHECK_M, CHECK_K],
        &[CHECK_N, CHECK_K],
        Some(&[CHECK_N]),
        0.5,
        2.0,
        false,
        true,
    )?;
    let gemm_a = fixture(CHECK_M * CHECK_K, 0x0f1e_2d3c);
    let gemm_b = fixture(CHECK_N * CHECK_K, 0x4b5a_6978);
    let gemm_c = fixture(CHECK_N, 0x8796_a5b4);
    let gpu = backend.matmul(&gemm, &gemm_a, &gemm_b, Some(&gemm_c))?;
    comparisons.push(reference::verify_matmul(
        &gemm,
        &gemm_a,
        &gemm_b,
        Some(&gemm_c),
        &gpu,
    )?);

    // ── Binary elementwise ───────────────────────────────────────────────────
    let plan = ElementwisePlan::binary(&CHECK_ELEMENTWISE_SHAPE, &CHECK_ELEMENTWISE_SHAPE)?;
    let elems = CHECK_ELEMENTWISE_SHAPE.iter().product::<usize>();
    let lhs = fixture(elems, 0xc3d2_e1f0);
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        // `Div` needs a denominator that is nowhere near zero — not to spare the GPU (it
        // would honestly return ±inf, and `reference::sigmoid`-style saturation rules would
        // accept that) but because a `1/ε` result makes the *relative* tolerance
        // meaningless and turns the check into noise.  Shift into `[0.5, 4.5)`.
        let rhs: Vec<f32> = if op == BinaryOp::Div {
            fixture(elems, 0x1029_3847)
                .into_iter()
                .map(|v| v + 2.5)
                .collect()
        } else {
            fixture(elems, 0x1029_3847)
        };
        let gpu = backend.binary(&plan, op, &lhs, &rhs)?;
        comparisons.push(reference::verify_binary(&plan, op, &lhs, &rhs, &gpu)?);
    }

    // ── Unary elementwise ────────────────────────────────────────────────────
    let plan = ElementwisePlan::unary(&CHECK_ELEMENTWISE_SHAPE)?;
    for op in [UnaryOp::Relu, UnaryOp::Sigmoid, UnaryOp::Tanh] {
        // ×50 puts the inputs in `[-100, 100)`, well past the ±88 where `exp` overflows
        // f32.  `Sigmoid` and `Tanh` must *saturate* there, cleanly, to 0/1 and ∓1 — not
        // produce a NaN.  The oracle is written to do exactly what the shader does, so a
        // NaN on either side is a real disagreement and the comparison will say so.
        let input: Vec<f32> = fixture(elems, 0x5566_7788)
            .into_iter()
            .map(|v| v * 50.0)
            .collect();
        let gpu = backend.unary(&plan, op, &input)?;
        comparisons.push(reference::verify_unary(&plan, op, &input, &gpu)?);
    }

    for comparison in &comparisons {
        report.record(comparison);
    }
    Ok((report, comparisons))
}

/// Deterministic, dependency-free operand data, spread over `[-2, 2)`.
///
/// A plain ramp would be a poor fixture: `A[i] = i` makes a transposed read produce the
/// *same* dot product as a correct one for a square operand, and makes an off-by-one index
/// differ by a constant that a loose tolerance would swallow.  A small LCG gives values
/// with no such structure, while staying reproducible bit-for-bit on every machine that
/// runs the self-check — which matters, because the whole point is to compare *reports*
/// from different people's hardware.
///
/// Values come out of a 12-bit window divided by 1024, so every one is exactly
/// representable in f32 and the oracle's arithmetic is the only source of rounding.
fn fixture(len: usize, seed: u32) -> Vec<f32> {
    // Numerical Recipes' LCG constants; `| 1` keeps the state out of the zero fixed point.
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // `u16 -> f32` is lossless, so there is no `as`-cast rounding anywhere here.
            let bits = u16::try_from((state >> 16) & 0xFFF).unwrap_or(0);
            f32::from(bits) / 1024.0 - 2.0
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        fixture, parse_env_flag, Activation, DirectMLContext, FailurePolicy,
        CHECK_ELEMENTWISE_SHAPE, CHECK_K, CHECK_M, CHECK_N,
    };

    #[test]
    fn try_new_is_none_off_windows_and_never_panics() {
        let ctx = DirectMLContext::try_new();
        #[cfg(not(target_os = "windows"))]
        assert!(
            ctx.is_none(),
            "no non-Windows target has D3D12; a context must never materialise"
        );
        // On Windows this asserts only that acquisition does not panic: a box with no
        // adapter, or a user who has not opted in, legitimately yields `None`.
        let _ = ctx.is_some();
    }

    #[test]
    fn explicit_activation_still_yields_nothing_without_a_device() {
        // `Activation::Enabled` bypasses the *environment* gate, not the *hardware* one.
        let ctx = DirectMLContext::try_new_with(Activation::Enabled);
        #[cfg(not(target_os = "windows"))]
        assert!(
            ctx.is_none(),
            "opting in cannot conjure a D3D12 adapter onto a Linux box"
        );
        let _ = ctx.is_some();
    }

    #[test]
    fn disabled_activation_refuses_before_it_ever_touches_a_device() {
        assert!(!Activation::Disabled.permits_acquisition());
        assert!(DirectMLContext::try_new_with(Activation::Disabled).is_none());
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
        // On, including the deliberate "anything unrecognised is on" rule: a user who
        // typed something has asked for the feature, and silently ignoring them would hand
        // back a false all-clear.
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

    #[test]
    fn fixture_is_deterministic_bounded_and_unstructured() {
        let a = fixture(64, 0xdead_beef);
        let b = fixture(64, 0xdead_beef);
        assert_eq!(
            a, b,
            "two hosts must generate byte-identical self-check data"
        );

        assert!(
            a.iter().all(|v| (-2.0..2.0).contains(v)),
            "values must stay in [-2, 2) so no op saturates by accident"
        );
        assert!(
            a.iter().all(|v| v.is_finite()),
            "a non-finite fixture would make every tolerance meaningless"
        );

        // Not a ramp: a monotone fixture hides transposition and off-by-one bugs.
        let ascending = a.windows(2).filter(|w| w[0] <= w[1]).count();
        assert!(
            ascending > 8 && ascending < a.len() - 8,
            "the fixture must not be monotone in either direction (got {ascending} ascending \
             steps of {})",
            a.len() - 1
        );

        assert_ne!(
            fixture(16, 1),
            fixture(16, 2),
            "different seeds must give different operands, or A and B correlate"
        );
    }

    #[test]
    fn the_self_check_shapes_are_ones_this_crate_actually_accepts() {
        // If any of these ever starts declining, `run_self_check` would return `Err` on
        // real hardware and the acceptance gate would read as a GPU failure when in fact
        // the *shapes* had drifted out of the supported set.  Pin them here, where CI can
        // see it, rather than discovering it on someone else's Windows box.
        use crate::plan::{ElementwisePlan, MatMulPlan};

        assert!(MatMulPlan::matmul(&[CHECK_M, CHECK_K], &[CHECK_K, CHECK_N]).is_ok());
        assert!(MatMulPlan::gemm(
            &[CHECK_M, CHECK_K],
            &[CHECK_N, CHECK_K],
            Some(&[CHECK_N]),
            0.5,
            2.0,
            false,
            true,
        )
        .is_ok());
        assert!(
            ElementwisePlan::binary(&CHECK_ELEMENTWISE_SHAPE, &CHECK_ELEMENTWISE_SHAPE).is_ok()
        );
        assert!(ElementwisePlan::unary(&CHECK_ELEMENTWISE_SHAPE).is_ok());
    }

    #[test]
    fn the_elementwise_self_check_straddles_a_thread_group_boundary() {
        // 24 elements against a 256-wide group: the dispatch launches one group of 256
        // threads, 232 of which must fall out of the `if (i >= N) return;` guard.  A kernel
        // that omits the guard writes 232 elements past the end of the output buffer.
        let elems: usize = CHECK_ELEMENTWISE_SHAPE.iter().product();
        assert!(
            elems % (crate::plan::ELEMENTWISE_THREADS_PER_GROUP as usize) != 0,
            "the self-check must exercise a partial thread group"
        );
    }
}
