//! Non-blocking inference: [`Session::run_async`], its [`RunFuture`], and a
//! dependency-free [`block_on`] for callers that have no executor.
//!
//! # Executor-agnostic by construction
//!
//! This crate has no async runtime and does not want one — a Pure Rust
//! inference engine has no business dragging tokio into every dependent.  So
//! `run_async` does not *schedule* anything: it starts the inference on a plain
//! [`std::thread`] immediately and hands back a future over a one-shot channel.
//! The returned [`RunFuture`] is an ordinary `Future<Output = Result<…>>` and
//! works unchanged under tokio, async-std, smol, `futures::executor::block_on`,
//! or the [`block_on`] in this module.
//!
//! What that buys, and what it does not:
//!
//! * **Buys:** the calling thread is free while the model runs, `.await` composes
//!   with `select!` / `join!`, and a session already shared as `Arc<Session>`
//!   (which is how sessions are shared behind a web handler) needs no extra
//!   wrapping.
//! * **Does not buy:** thread-per-inference is not free.  For many small
//!   concurrent inferences, spawn cost dominates; run those on your own pool and
//!   call [`Session::run`] directly.  `run_async` is for the case it is named
//!   for — one inference long enough that blocking the caller matters.
//!
//! # Why the receiver is `Arc<Self>`
//!
//! The obvious signature, `fn run_async(&self, …) -> impl Future`, cannot be
//! written soundly: the future outlives the call, so the worker thread would
//! hold a `&Session` with no proof the session is still alive when it runs.
//! [`std::thread::scope`] would prove it, but scopes *block* at their end, which
//! is the exact thing being avoided.  `self: Arc<Self>` is the honest form —
//! it makes the shared ownership the design already requires visible in the
//! signature, and it costs one refcount bump.
//!
//! ```no_run
//! use oxionnx::{Session, Tensor};
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let session = Arc::new(Session::from_file("model.onnx".as_ref())?);
//! let mut inputs = HashMap::new();
//! inputs.insert("x".to_string(), Tensor::new(vec![1.0, 2.0], vec![1, 2]));
//!
//! let future = Arc::clone(&session).run_async(inputs);
//! // ... do other work here; the model is already running ...
//! let outputs = oxionnx::block_on(future)?;
//! # let _ = outputs;
//! # Ok(())
//! # }
//! ```

use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
#[cfg(feature = "gpu")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use super::Session;

/// What a run produces.
type RunOutput = Result<HashMap<String, Tensor>, OnnxError>;

/// How many `run_async`/`spawn_run` worker threads are currently alive, from
/// the moment their thread body starts to the moment it fully returns
/// (including dropping everything it captured).
///
/// Exists purely for `gpu_owner`'s `atexit` hook (hence gated the same way):
/// it needs to know not just "is a `GpuContext` disposal in progress right
/// now" but "*could* one still start" — a worker thread that has not yet
/// reached the point where it drops its `Arc<Session>` is exactly such a
/// case, and checking only the narrower, GPU-specific signal let process
/// exit race ahead of a worker that had not started its teardown yet. See
/// `gpu_owner`'s module docs for the full account of the crash this closes.
#[cfg(feature = "gpu")]
static ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// How many worker threads spawned by `run_async`/`spawn_run` are currently
/// alive, anywhere in the process. `0` is not a promise that none will ever
/// exist again — a caller can always start a new one — only that none is
/// alive *right now*.
#[cfg(feature = "gpu")]
pub(crate) fn active_worker_count() -> usize {
    ACTIVE_WORKERS.load(Ordering::SeqCst)
}

/// RAII: increments [`ACTIVE_WORKERS`] on construction, decrements on drop —
/// including on an early return or a panic unwind, so a worker that panics
/// can never leave the count stuck above zero.
#[cfg(feature = "gpu")]
struct WorkerCountGuard;

#[cfg(feature = "gpu")]
impl WorkerCountGuard {
    fn enter() -> Self {
        ACTIVE_WORKERS.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

#[cfg(feature = "gpu")]
impl Drop for WorkerCountGuard {
    fn drop(&mut self) {
        ACTIVE_WORKERS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The one-shot slot shared by the worker thread and the future.
///
/// Both halves hold an `Arc`, so dropping the future while the worker is still
/// running is safe: the worker writes into a slot that is still alive and simply
/// finds no waker to call.
struct Shared {
    state: Mutex<SharedState>,
    /// Signalled on completion, for [`RunHandle::wait`] — the blocking path must
    /// not spin, and must not need a `Waker`.
    done: Condvar,
    /// A second, independent claim on the session, held for exactly as long as
    /// this `Shared` is — i.e. for as long as *either* the worker thread or the
    /// [`RunFuture`]/[`RunHandle`] is still alive. Never read; it exists purely
    /// for its `Drop`.
    ///
    /// # Why this has to exist
    ///
    /// [`Session::spawn_run`] takes `self: Arc<Self>`, and a caller that does not
    /// keep its own clone (`session.run_async(x)` rather than
    /// `Arc::clone(&session).run_async(x)` — the latter is the pattern the
    /// module docs show) hands the worker thread the *only* surviving
    /// reference. Without this field, that thread's closure ending would run
    /// `Session`'s (and, once a real GPU adapter is behind it, `GpuContext`'s
    /// — a live `wgpu::Device`/`Queue`/`Instance`) full destructor as the
    /// worker's last act before the OS thread itself exits.
    ///
    /// That combination is unsound in practice, not just slow: a native GPU
    /// driver commonly registers its own pthread-TLS destructor on every
    /// thread that ever touches it (here, the thread that ran
    /// `GpuContext::try_new()` while building the session — typically the
    /// *caller's* thread, not this worker). If the device/instance is
    /// destroyed by a *different* thread that then immediately exits, that
    /// other thread's own TLS destructor can run afterwards against
    /// already-freed driver state — this was caught red-handed under gdb as a
    /// `SIGSEGV` inside glibc's `__nptl_deallocate_tsd`, on the thread that
    /// had built the session, well after the worker thread that dropped it
    /// had already finished. With no adapter (`GpuContext::try_new()` ==
    /// `None`), there was nothing to tear down and the whole class of bug was
    /// invisible; a real adapter is the normal case in production, not the
    /// exception.
    ///
    /// Holding a second clone here means the worker's own capture dropping
    /// (whenever that happens) is never the one that brings the session's
    /// reference count to zero *unless every other clone, including this one,
    /// is already gone* — and this one only goes when `Shared` itself does,
    /// i.e. when both the worker and the future/handle have let go. In every
    /// test and every documented usage pattern the future/handle outlives the
    /// worker's own brief post-`complete()` teardown, so the actual drop lands
    /// on whichever thread drops the future/handle — ordinarily the same
    /// thread that built the session in the first place, never a thread that
    /// is also mid-exit.
    ///
    /// # A tempting, and wrong, "improvement" tried and rejected here
    ///
    /// An earlier version of this fix additionally routed this field's drop
    /// through a dedicated, process-lifetime background thread (a "reaper"),
    /// reasoning that *no* ephemeral thread — not even the one that built the
    /// session — is guaranteed to outlive it (a `#[test]` binary's per-test
    /// thread is spawned by `libtest` and joined immediately after the test
    /// returns). That reasoning about the danger was correct, but the fix was
    /// not: it centralizes *every* teardown onto a thread that is, by
    /// construction, never the one that called `GpuContext::try_new()` in
    /// the first place — turning what was previously an occasional
    /// cross-thread create/destroy mismatch into a *guaranteed* one on every
    /// single run. Measured empirically on this crate's own test hardware,
    /// that turned an already-largely-fixed ~2–5% residual failure rate into
    /// a 30-for-30 `SIGSEGV`. The driver's TLS destructor is keyed to *the
    /// thread that created the device*, not to "some safe long-lived
    /// thread" — so the only reliable mitigation is keeping teardown on a
    /// thread that has a real chance of being that one, which is exactly
    /// what this field (without a reaper) already does most of the time.
    _session_keepalive: Arc<Session>,
}

#[derive(Default)]
struct SharedState {
    result: Option<RunOutput>,
    waker: Option<Waker>,
    /// Set once the result has been handed to a consumer, so a second poll or a
    /// second `wait` reports a clear error instead of hanging forever.
    taken: bool,
}

impl Shared {
    /// `session_keepalive` should be an *independent* clone — not the one
    /// about to be moved into the worker thread's closure — so that this
    /// `Shared` keeps the session alive on its own account. See the
    /// `_session_keepalive` field doc for why that matters.
    fn new(session_keepalive: Arc<Session>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SharedState::default()),
            done: Condvar::new(),
            _session_keepalive: session_keepalive,
        })
    }

    /// Publish the run's outcome and release whoever is waiting.
    ///
    /// The waker is taken out *under* the lock and invoked *after* it is
    /// released.  Waking while holding the lock is the classic way to deadlock
    /// against an executor that polls the future synchronously from inside
    /// `wake()`.
    fn complete(&self, result: RunOutput) {
        let waker = match self.state.lock() {
            Ok(mut state) => {
                state.result = Some(result);
                state.waker.take()
            }
            // A poisoned lock means a previous holder panicked mid-update. There
            // is nothing useful left to publish and no one to notify; the
            // waiting side reports the poison itself.
            Err(_) => None,
        };
        self.done.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poisoned() -> OnnxError {
        OnnxError::Internal(
            "run_async: the inference thread panicked while publishing its result".to_string(),
        )
    }

    fn already_taken() -> OnnxError {
        OnnxError::Internal(
            "run_async: this run's result has already been consumed; a RunFuture yields \
             exactly once"
                .to_string(),
        )
    }
}

/// A future over one in-flight [`Session::run_async`] inference.
///
/// Resolves to exactly what [`Session::run`] would have returned. Dropping it
/// does **not** cancel the run — the worker thread finishes and its result is
/// discarded. To actually stop early, bind a
/// [`CancellationToken`](crate::CancellationToken) to the session (see
/// [`crate::SessionBuilder::with_session_cancellation`]) and cancel it.
pub struct RunFuture {
    shared: Arc<Shared>,
}

impl Future for RunFuture {
    type Output = RunOutput;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = match self.shared.state.lock() {
            Ok(state) => state,
            Err(_) => return Poll::Ready(Err(Shared::poisoned())),
        };
        if state.taken {
            return Poll::Ready(Err(Shared::already_taken()));
        }
        match state.result.take() {
            Some(result) => {
                state.taken = true;
                Poll::Ready(result)
            }
            None => {
                // Always re-register: the waker from an earlier poll may belong
                // to a task that has since been moved between executor threads.
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// A blocking handle on the same in-flight inference a [`RunFuture`] represents.
///
/// Returned by [`Session::spawn_run`] for callers with no async context at all:
/// start the inference, do something else, then [`RunHandle::wait`] for it.
///
/// A handle is a *receipt*, not ownership of the thread: dropping it without
/// waiting does **not** cancel anything — the worker finishes and its result is
/// discarded, exactly as for [`RunFuture`]. To stop early, cancel a bound
/// [`CancellationToken`](crate::CancellationToken).
pub struct RunHandle {
    shared: Arc<Shared>,
}

impl RunHandle {
    /// Has the inference finished (successfully or not)?
    ///
    /// Never blocks. A `false` here is a snapshot, not a promise.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.shared
            .state
            .lock()
            .map(|state| state.result.is_some() || state.taken)
            .unwrap_or(true)
    }

    /// Block until the inference finishes and take its result.
    ///
    /// # Errors
    ///
    /// Whatever [`Session::run`] would have returned, or
    /// [`OnnxError::Internal`] if the result was already consumed or the worker
    /// thread panicked.
    pub fn wait(self) -> RunOutput {
        let mut state = match self.shared.state.lock() {
            Ok(state) => state,
            Err(_) => return Err(Shared::poisoned()),
        };
        loop {
            if state.taken {
                return Err(Shared::already_taken());
            }
            if let Some(result) = state.result.take() {
                state.taken = true;
                return result;
            }
            state = match self.shared.done.wait(state) {
                Ok(state) => state,
                Err(_) => return Err(Shared::poisoned()),
            };
        }
    }

    /// Turn this handle into a [`RunFuture`] over the same inference.
    #[must_use]
    pub fn into_future(self) -> RunFuture {
        RunFuture {
            shared: self.shared,
        }
    }
}

impl Session {
    /// Start this inference on its own thread and return a future for its
    /// result.
    ///
    /// The model begins executing before this call returns — the future is a
    /// *receipt*, not a lazily-scheduled task, so a caller that never polls it
    /// still ran the model. See the [module docs](self) for the executor
    /// contract and the cost model.
    ///
    /// Inputs are taken **by value** (`HashMap<String, Tensor>` rather than the
    /// `HashMap<&str, Tensor>` [`Session::run`] takes) because the worker thread
    /// outlives the call: there is no lifetime the borrowed form could have.
    ///
    /// # Errors
    ///
    /// The future resolves to whatever [`Session::run`] would have returned,
    /// plus [`OnnxError::Internal`] if the OS refuses to spawn the thread or the
    /// inference thread panics.
    #[must_use = "the inference has already started; drop this only if the result is unwanted"]
    pub fn run_async(self: Arc<Self>, inputs: HashMap<String, Tensor>) -> RunFuture {
        self.spawn_run(inputs).into_future()
    }

    /// [`Session::run_async`] for callers with no executor: same thread, same
    /// one-shot slot, but a blocking [`RunHandle`] instead of a future.
    #[must_use = "the inference has already started; drop this only if the result is unwanted"]
    pub fn spawn_run(self: Arc<Self>, inputs: HashMap<String, Tensor>) -> RunHandle {
        // An independent claim on the session, held by `Shared` itself rather
        // than by this thread's `self` capture below — see
        // `Shared::_session_keepalive` for why the worker's own capture must
        // not be the last `Arc<Session>` standing.
        let shared = Shared::new(Arc::clone(&self));
        let worker_shared = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("oxionnx-run-async".to_string())
            .spawn(move || {
                // Counted for as long as this closure runs — see
                // `ACTIVE_WORKERS` for why `gpu_owner`'s `atexit` hook needs
                // this signal specifically, not just a narrower "is a
                // disposal in flight right now" one.
                #[cfg(feature = "gpu")]
                let _worker_count_guard = WorkerCountGuard::enter();

                // The slot is filled on *every* exit from this thread, including
                // an unwinding one. Without the guard, an operator that panics
                // would drop the worker's `Arc` with the slot still empty and
                // leave the future — and `RunHandle::wait` — waiting forever.
                let mut guard = CompleteOnDrop {
                    shared: worker_shared,
                    published: false,
                };
                let borrowed: HashMap<&str, &Tensor> =
                    inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();
                let result = self.run_internal(&borrowed);
                guard.published = true;
                guard.shared.complete(result);
            });

        if let Err(e) = spawned {
            // The thread never started, so nothing will ever complete the slot.
            // Fill it here so the future resolves instead of hanging forever.
            shared.complete(Err(OnnxError::Internal(format!(
                "run_async: cannot spawn the inference thread: {e}"
            ))));
        }

        RunHandle { shared }
    }
}

/// Fills the one-shot slot if the worker leaves without having done so.
///
/// The only way that happens is a panic inside an operator (or inside the run
/// loop). A panicking *worker* must not become a hanging *caller*: the caller
/// gets a typed error and the panic message still reaches stderr the usual way.
struct CompleteOnDrop {
    shared: Arc<Shared>,
    published: bool,
}

impl Drop for CompleteOnDrop {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        self.shared.complete(Err(OnnxError::Internal(
            "run_async: the inference thread panicked; see the panic message on stderr".to_string(),
        )));
    }
}

// ── A dependency-free executor for one future ───────────────────────────────

/// Waker that parks and unparks the thread inside [`block_on`].
///
/// Implemented via [`std::task::Wake`], which is the safe route to a `Waker` —
/// no `RawWakerVTable`, no `unsafe`.
struct ThreadWaker {
    thread: std::thread::Thread,
}

impl std::task::Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

/// Drive `future` to completion on the current thread.
///
/// A minimal, correct, single-future executor: it parks between polls rather
/// than spinning, and it is here so that using [`Session::run_async`] does not
/// oblige a caller to take on tokio. Inside an async runtime, `.await` the
/// future instead — calling this from an executor thread blocks that thread.
pub fn block_on<F: Future>(future: F) -> F::Output {
    // `Box::pin` rather than a stack-pinning macro: it needs no `unsafe`, and it
    // pins *any* future, including `!Unpin` ones. One allocation per call is
    // nothing next to an inference.
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(ThreadWaker {
        thread: std::thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            // A spurious unpark just polls again, which is why the loop does not
            // assume `park` returning means the waker fired.
            Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph, Node, OpKind};

    /// `y = x + w`, the smallest graph that actually executes an operator.
    fn add_session() -> Arc<Session> {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![10.0, 20.0], vec![2]));
        let graph = Graph {
            name: "add".to_string(),
            nodes: vec![Node {
                op: OpKind::Add,
                name: "add".to_string(),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attrs: Attributes::default(),
            }],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            input_infos: Vec::new(),
            output_infos: Vec::new(),
        };
        Arc::new(Session::from_graph(graph, weights).expect("session builds"))
    }

    fn one_input() -> HashMap<String, Tensor> {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        inputs
    }

    #[test]
    fn an_awaited_run_returns_exactly_what_the_blocking_run_returns() {
        let session = add_session();
        let sync = session
            .run(&HashMap::from([(
                "x",
                Tensor::new(vec![1.0, 2.0], vec![2]),
            )]))
            .expect("sync run");
        let asynchronous =
            block_on(Arc::clone(&session).run_async(one_input())).expect("async run");
        assert_eq!(asynchronous["y"].data, sync["y"].data);
        assert_eq!(asynchronous["y"].data, vec![11.0, 22.0]);
    }

    #[test]
    fn a_handle_can_be_waited_on_without_any_executor() {
        let session = add_session();
        let handle = session.spawn_run(one_input());
        let outputs = handle.wait().expect("run completes");
        assert_eq!(outputs["y"].data, vec![11.0, 22.0]);
    }

    /// The future is a receipt for work already started, so dropping it must be
    /// harmless — no panic in the worker writing to a dead slot, no hang.
    #[test]
    fn dropping_the_future_before_completion_neither_panics_nor_hangs() {
        let session = add_session();
        for _ in 0..32 {
            drop(Arc::clone(&session).run_async(one_input()));
        }
        // The sessions above are still running; a subsequent run on the same
        // session must be unaffected.
        let outputs = block_on(session.run_async(one_input())).expect("run completes");
        assert_eq!(outputs["y"].data, vec![11.0, 22.0]);
    }

    #[test]
    fn many_concurrent_async_runs_share_one_session() {
        let session = add_session();
        let futures: Vec<RunFuture> = (0..8)
            .map(|_| Arc::clone(&session).run_async(one_input()))
            .collect();
        let expected: Vec<f32> = vec![11.0, 22.0];
        for future in futures {
            let outputs = block_on(future).expect("run completes");
            assert_eq!(outputs["y"].data, expected);
        }
    }

    #[test]
    fn a_second_consumption_of_one_run_is_a_typed_error_not_a_hang() {
        let session = add_session();
        let handle = session.spawn_run(one_input());
        let mut future = handle.into_future();
        assert!(block_on(&mut future).is_ok());
        match block_on(&mut future) {
            Err(OnnxError::Internal(msg)) => assert!(msg.contains("already been consumed")),
            other => panic!("expected an Internal error, got {other:?}"),
        }
    }

    #[test]
    fn a_failing_run_surfaces_its_error_through_the_future() {
        let session = add_session();
        // "x" is missing, so shape resolution / execution must fail.
        let outputs = block_on(session.run_async(HashMap::new()));
        assert!(outputs.is_err(), "a run without its input cannot succeed");
    }
}
