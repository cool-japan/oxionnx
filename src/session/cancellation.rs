//! Cooperative cancellation for long-running inference.
//!
//! # What "cooperative" means here
//!
//! There is no way to abort a running operator kernel from the outside without
//! either killing the thread (unsound: the run state is half-written) or
//! polling inside every kernel (a change to all ~165 operators).  What this
//! module provides instead is the standard cooperative contract: a flag the
//! caller can raise from any thread, and a **check between nodes** — the engine
//! finishes the node it is on, sees the flag, and unwinds with
//! [`OnnxError::Cancelled`] instead of starting the next one.
//!
//! # Where the check lives
//!
//! Not in the run loop — in the *registry*.  A session built with a token gets
//! a registry in which every operator its model actually uses is wrapped in a
//! `CancellableOp` that consults the token and then delegates to the real
//! implementation.  This is deliberate and buys three things a check at the top
//! of the sequential loop would not:
//!
//! * it covers the **sequential and the rayon parallel path** with one
//!   implementation, because both reach an operator only through the registry;
//! * it covers **subgraph bodies** (`If` / `Loop` / `Scan` execute their nodes
//!   through `ctx.registry`), so a runaway `Loop` — the single most likely
//!   thing a caller wants to cancel — is interruptible per iteration *and* per
//!   node inside the iteration;
//! * it covers the **typed** (`run_typed`) path, which does not go through
//!   `dispatch_node` at all.
//!
//! # Scope: the token belongs to the session, not to one `run()`
//!
//! [`crate::SessionBuilder::with_session_cancellation`] binds the token for the
//! life of the session.  Cancelling it aborts **every run in flight on that
//! session**, not one request — the names of the API surface say so on purpose.
//! A per-request token would have to be threaded through the run loop, which is
//! not something this module can do from the outside.  Callers that need
//! per-request cancellation should give each request its own `Session` (they
//! are cheap to clone-share only as a whole, so this means one session per
//! concurrent request slot), or cancel at a coarser granularity — the
//! [`crate::streaming`] generator, for instance, checks the token between
//! decode steps, which is per-request by construction.
//!
//! # Not a cancellation point
//!
//! Two things are outside the guard, both by construction:
//!
//! * A node taken by an execution provider (CUDA / DirectML / wgpu) is
//!   dispatched *before* the registry is consulted, so it is not a cancellation
//!   point.  On a graph fully claimed by an accelerator the token is therefore
//!   never observed.  CPU nodes — which is all of them in a default build —
//!   always are.
//! * A node whose operator is *not in the registry at all* never reaches a
//!   guard, but it never reaches an implementation either — it raises
//!   `UnsupportedOp` before any of this matters.
//!
//! An operator installed **after** the session was built, via
//! [`crate::Session::register_op`], *is* a cancellation point: it is wrapped on
//! the way in, by this module's `wrap_owned_op`.  It used to go straight into
//! the already-wrapped registry unguarded, so a model leaning on a
//! late-registered custom operator silently had that many fewer places to stop.

use oxionnx_core::{
    DType, OnnxError, OpContext, Operator, OperatorRegistry, Tensor, TypedOpContext, TypedTensor,
};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A shared flag that asks an in-flight inference to stop at the next node
/// boundary.
///
/// Cloning a token clones the *handle*, not the flag: every clone observes and
/// sets the same underlying state, which is what makes "cancel from another
/// thread" work.
///
/// ```
/// use oxionnx::CancellationToken;
///
/// let token = CancellationToken::new();
/// assert!(!token.is_cancelled());
///
/// let remote = token.clone();
/// std::thread::spawn(move || remote.cancel()).join().ok();
///
/// assert!(token.is_cancelled());
/// token.reset();
/// assert!(!token.is_cancelled());
/// ```
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// A fresh, un-cancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask every run observing this token to stop at its next node boundary.
    ///
    /// Idempotent, callable from any thread, and never blocks.
    pub fn cancel(&self) {
        // `Release` pairs with the `Acquire` load in `is_cancelled`: everything
        // the cancelling thread wrote before this call is visible to the run
        // thread that observes the flag.
        self.flag.store(true, Ordering::Release);
    }

    /// Clear the flag so the token can be reused for the next run.
    ///
    /// Call this **between** runs, never during one: a reset that races a run
    /// simply means that run is no longer cancelled, which is confusing rather
    /// than unsound.
    pub fn reset(&self) {
        self.flag.store(false, Ordering::Release);
    }

    /// Has [`CancellationToken::cancel`] been called (and not reset since)?
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Do the two handles share one flag (i.e. is one a clone of the other)?
    fn same_flag(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.flag, &other.flag)
    }

    /// `Err(OnnxError::Cancelled)` when cancelled, naming the node the run
    /// stopped in front of.
    fn check(&self, node_name: &str, op_type: &str) -> Result<(), OnnxError> {
        if self.is_cancelled() {
            return Err(OnnxError::Cancelled(format!(
                "inference cancelled before node '{node_name}' (op '{op_type}')"
            )));
        }
        Ok(())
    }
}

// ── The guard operator ──────────────────────────────────────────────────────

/// One operator of the wrapped registry: checks the token, then delegates.
///
/// It holds the *whole* inner registry rather than a `Box<dyn Operator>`
/// because [`OperatorRegistry`] hands out only `&dyn Operator` — there is no way
/// to take ownership of a registered operator back out of it.
///
/// # Cost
///
/// One extra hash lookup per *executed* node, paid **only** by sessions that
/// asked for cancellation.  The three dispatch **predicates** are deliberately
/// *not* looked up: the run loop consults `supports_inplace` and
/// `supports_output_slots` before every node, so forwarding those through the
/// registry would have tripled the overhead for no reason.  They are constant
/// for the life of an operator, so they are snapshotted here at wrap time.
/// Measured A/B on a 300-node chain of trivial `Relu`s — the worst case for
/// *relative* overhead, since the operators themselves do almost nothing —
/// best-of-9 × 40 runs on an M-series host:
///
/// | guard | vs unguarded | per node |
/// |---|---|---|
/// | predicates forwarded through the registry | +15.7 % | 26.5 ns |
/// | predicates snapshotted (this code) | **+6.4 %** | **10.9 ns** |
///
/// The measurement lives in
/// `the_cancellation_guard_costs_one_lookup_per_node_not_an_order_of_magnitude`
/// (`tests/w2_cancellation.rs`), which also fails on an order-of-magnitude
/// regression.  Numerical results are unaffected: the guard delegates to the
/// same operator with the same context, and
/// `an_uncancelled_session_returns_bit_identical_results` asserts bit equality.
struct CancellableOp {
    inner: Arc<OperatorRegistry>,
    op_type: String,
    token: CancellationToken,
    /// Snapshot of `inner.get(op_type).supports_inplace()`.
    supports_inplace: bool,
    /// Snapshot of `inner.get(op_type).supports_output_slots()`.
    supports_output_slots: bool,
    /// Snapshot of `inner.get(op_type).native_dtypes()`.
    native_dtypes: &'static [DType],
}

impl CancellableOp {
    /// The real implementation this guard fronts.
    ///
    /// Cannot normally fail: [`wrap_registry`] only ever registers a guard for
    /// a name the inner registry already contains, and neither registry is
    /// mutated afterwards.  The error path exists so a future caller that
    /// unregisters an operator gets a typed error rather than a panic.
    fn inner_op(&self) -> Result<&dyn Operator, OnnxError> {
        self.inner.get(&self.op_type).ok_or_else(|| {
            OnnxError::Internal(format!(
                "cancellation guard for '{}' lost its inner operator",
                self.op_type
            ))
        })
    }
}

// Every defaulted method of `Operator` is forwarded, not just the required
// ones.  A missed `supports_output_slots` / `supports_inplace` would leave the
// zero-copy and in-place fast paths silently switched off for every cancellable
// session — a performance regression no correctness test would catch.  See
// `the_guard_forwards_the_dispatch_fast_path_predicates` in
// `tests/w2_cancellation.rs`.
impl Operator for CancellableOp {
    fn op_type(&self) -> &str {
        &self.op_type
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner_op()?.execute(ctx)
    }

    fn supports_inplace(&self) -> bool {
        self.supports_inplace
    }

    fn execute_inplace(
        &self,
        input: Tensor,
        ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner_op()?.execute_inplace(input, ctx)
    }

    fn native_dtypes(&self) -> &'static [DType] {
        self.native_dtypes
    }

    fn execute_typed(&self, ctx: &TypedOpContext<'_>) -> Result<Vec<TypedTensor>, OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner_op()?.execute_typed(ctx)
    }

    fn supports_output_slots(&self) -> bool {
        self.supports_output_slots
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner_op()?.execute_into_slots(ctx, slots)
    }
}

// ── The owned-operator guard ────────────────────────────────────────────────

/// A guard fronting an operator the caller handed over **by value**.
///
/// [`CancellableOp`] exists because [`OperatorRegistry`] hands out only
/// `&dyn Operator` — there is no way to take a registered operator back out of
/// one, so a guard built from a registry has to hold the whole registry and
/// re-look-up.  [`crate::Session::register_op`] is the one place that is *given*
/// the `Box<dyn Operator>` outright, so its guard can own the implementation
/// directly: no fallback table, no per-call lookup, and no `Internal` error path
/// for "the inner operator went missing".
///
/// The three dispatch predicates are snapshotted at wrap time for the same
/// reason as in [`CancellableOp`]: they are constant for an operator's life and
/// the run loop consults two of them before every node.
struct CancellableBoxedOp {
    inner: Box<dyn Operator>,
    op_type: String,
    token: CancellationToken,
    supports_inplace: bool,
    supports_output_slots: bool,
    native_dtypes: &'static [DType],
}

impl Operator for CancellableBoxedOp {
    fn op_type(&self) -> &str {
        &self.op_type
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner.execute(ctx)
    }

    fn supports_inplace(&self) -> bool {
        self.supports_inplace
    }

    fn execute_inplace(
        &self,
        input: Tensor,
        ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner.execute_inplace(input, ctx)
    }

    fn native_dtypes(&self) -> &'static [DType] {
        self.native_dtypes
    }

    fn execute_typed(&self, ctx: &TypedOpContext<'_>) -> Result<Vec<TypedTensor>, OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner.execute_typed(ctx)
    }

    fn supports_output_slots(&self) -> bool {
        self.supports_output_slots
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        self.token.check(&ctx.node.name, &self.op_type)?;
        self.inner.execute_into_slots(ctx, slots)
    }
}

/// Wrap one owned operator so it consults `token` before it runs.
///
/// Used by [`crate::Session::register_op`] on a session that has a token bound.
/// The guard reports the **inner operator's** `op_type()`, so the registry key
/// is unchanged and a late registration still replaces the operator it is meant
/// to replace.
pub(crate) fn wrap_owned_op(op: Box<dyn Operator>, token: &CancellationToken) -> Box<dyn Operator> {
    // Field order matters: every snapshot borrows `op`, which is moved into
    // `inner` last.
    Box::new(CancellableBoxedOp {
        op_type: op.op_type().to_string(),
        token: token.clone(),
        supports_inplace: op.supports_inplace(),
        supports_output_slots: op.supports_output_slots(),
        native_dtypes: op.native_dtypes(),
        inner: op,
    })
}

// ── Registry wrapping ───────────────────────────────────────────────────────

/// Every distinct `op_type` reachable from `nodes`, including those inside
/// nested subgraph attributes at any depth.
///
/// Missing a nested op type would be a **correctness regression**, not a gap:
/// the wrapped registry is the session's only registry, so an `If` branch whose
/// operator was not wrapped would resolve to `None` and turn a working model
/// into [`OnnxError::UnsupportedOp`].  The walk is therefore explicit and
/// iterative (an arbitrary-depth `Loop`-in-`Scan`-in-`If` nest must not be able
/// to overflow the stack), and `BTreeSet` keeps the result order deterministic.
fn reachable_op_types(nodes: &[crate::graph::Node]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut stack: Vec<&[crate::graph::Node]> = vec![nodes];
    while let Some(batch) = stack.pop() {
        for node in batch {
            out.insert(node.op.as_str().to_string());
            for subgraph in node.attrs.graphs.values() {
                stack.push(&subgraph.nodes);
            }
        }
    }
    out
}

/// A registry in which every operator `nodes` can reach checks `token` before
/// it runs.
///
/// Only names the inner registry actually provides are wrapped.  That is
/// load-bearing: registering a guard for an *unregistered* op would make
/// `registry.get()` return `Some`, which is exactly the lookup the run loop
/// uses to decide "this engine cannot run that operator" — an unsupported op
/// would then fail deep inside the guard with an `Internal` error instead of
/// the actionable [`OnnxError::UnsupportedOp`].
pub(crate) fn wrap_registry(
    inner: OperatorRegistry,
    nodes: &[crate::graph::Node],
    token: &CancellationToken,
) -> OperatorRegistry {
    let opset = inner.model_opset();
    let inner = Arc::new(inner);
    let mut wrapped = OperatorRegistry::new();
    wrapped.set_model_opset(opset);
    for op_type in reachable_op_types(nodes) {
        // `contains` and `get` agree by construction; the `let ... else` keeps
        // the snapshot below infallible without an unwrap.
        let Some(op) = inner.get(&op_type) else {
            continue;
        };
        let guard = CancellableOp {
            supports_inplace: op.supports_inplace(),
            supports_output_slots: op.supports_output_slots(),
            native_dtypes: op.native_dtypes(),
            inner: Arc::clone(&inner),
            op_type: op_type.clone(),
            token: token.clone(),
        };
        wrapped.register_as(op_type, Box::new(guard));
    }
    wrapped
}

// ── Session integration ─────────────────────────────────────────────────────

impl super::Session {
    /// Bind a cancellation token to this **whole session**.
    ///
    /// After this call every operator the model uses consults `token` before it
    /// executes, so `run()` (and `run_typed()`, and any `If`/`Loop`/`Scan`
    /// body) returns [`OnnxError::Cancelled`] at the first node boundary after
    /// [`CancellationToken::cancel`].
    ///
    /// # Scope
    ///
    /// The token is **session-scoped, not run-scoped**: concurrent runs on the
    /// same session share it, and cancelling stops all of them.  See the
    /// [module docs](self) for why, and for the accelerator caveat.
    ///
    /// # Re-binding
    ///
    /// Binding the *same* token twice is a no-op.  Binding a *different* token
    /// layers a second guard over the first: both tokens then stop the session,
    /// and each layer costs one more registry lookup per node.  Prefer
    /// [`crate::SessionBuilder::with_session_cancellation`], which binds once at
    /// construction, and keep one token per session.
    pub fn set_session_cancellation(&mut self, token: CancellationToken) {
        if self
            .cancellation
            .as_ref()
            .is_some_and(|bound| bound.same_flag(&token))
        {
            return;
        }
        // The registry has to move into the guards (they own the fallback
        // lookup table), so it is swapped out for an empty one first.
        let inner = std::mem::replace(&mut self.registry, OperatorRegistry::new());
        self.registry = wrap_registry(inner, &self.sorted_nodes, &token);
        self.cancellation = Some(token);
    }

    /// The token bound to this session, if any.
    #[must_use]
    pub fn session_cancellation_token(&self) -> Option<&CancellationToken> {
        self.cancellation.as_ref()
    }

    /// The operator registry this session dispatches through.
    ///
    /// Normally the one it was built with — but a session with cancellation
    /// bound dispatches through a *wrapped* registry instead (see the
    /// [module docs](self)), and the wrapping is only correct if every
    /// [`Operator`] predicate is forwarded. This accessor exists so that
    /// property can be asserted directly rather than inferred, and so callers
    /// can ask what a session can actually run.
    #[must_use]
    pub fn operator_registry(&self) -> &OperatorRegistry {
        &self.registry
    }
}
