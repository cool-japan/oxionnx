//! `Session::run_async` seen from outside the crate: the public exports, the
//! executor-agnostic `Future`, and the "the calling thread is actually free"
//! property the feature is named for.

use oxionnx::{
    Attributes, CancellationToken, Graph, Node, OnnxError, OpContext, OpKind, Operator, OptLevel,
    RunFuture, Session, Tensor,
};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn relu_session() -> Arc<Session> {
    let graph = Graph {
        name: "relu".to_string(),
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    Arc::new(Session::from_graph(graph, HashMap::new()).expect("session builds"))
}

fn x() -> HashMap<String, Tensor> {
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::new(vec![-1.0, 0.0, 2.5], vec![3]));
    inputs
}

#[test]
fn the_public_api_resolves_to_the_same_answer_as_a_blocking_run() {
    let session = relu_session();
    let outputs = oxionnx::block_on(Arc::clone(&session).run_async(x())).expect("async run");
    assert_eq!(outputs["y"].data, vec![0.0, 0.0, 2.5]);

    let borrowed: HashMap<&str, Tensor> = x().into_iter().map(|(k, v)| (leak(k), v)).collect();
    let sync = session.run(&borrowed).expect("blocking run");
    assert_eq!(outputs["y"].data, sync["y"].data);
}

/// Test-only: `Session::run` borrows its keys, so a test that owns them needs a
/// `&'static str`. Leaking three bytes once per test run is the least noisy way
/// to bridge that and never grows.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// An operator that will not return until the test says so.
///
/// It makes "the inference is genuinely running on another thread" observable
/// without any sleeping: the main thread reaches the barrier *while the model is
/// blocked inside a node*, which can only happen if `run_async` did not block.
struct Gate {
    barrier: Arc<Barrier>,
    entered: Arc<AtomicUsize>,
}

impl Operator for Gate {
    fn op_type(&self) -> &str {
        "Gate"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        self.barrier.wait();
        Ok(vec![ctx.input(0)?.clone()])
    }
}

#[test]
fn the_calling_thread_keeps_running_while_the_model_does() {
    let barrier = Arc::new(Barrier::new(2));
    let entered = Arc::new(AtomicUsize::new(0));

    let mut registry = oxionnx::default_registry();
    registry.register_as(
        "Gate",
        Box::new(Gate {
            barrier: Arc::clone(&barrier),
            entered: Arc::clone(&entered),
        }),
    );

    let graph = Graph {
        name: "gated".to_string(),
        nodes: vec![node(OpKind::parse("Gate"), "gate", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let session = Arc::new(
        Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_registry(registry)
            .build_from_graph(graph, HashMap::new())
            .expect("session builds"),
    );

    let future = session.run_async(x());

    // If `run_async` had blocked, this line would never be reached: the worker
    // is parked inside `Gate::execute` waiting for exactly this barrier.
    barrier.wait();
    assert_eq!(
        entered.load(Ordering::SeqCst),
        1,
        "the model must already be executing on another thread"
    );

    let outputs = oxionnx::block_on(future).expect("run completes");
    assert_eq!(outputs["y"].data, vec![-1.0, 0.0, 2.5]);
}

/// Waker that just counts wakeups — enough to drive a `RunFuture` by hand and
/// prove it behaves like any other `Future` under a foreign executor.
struct CountingWaker {
    wakes: AtomicUsize,
}

impl Wake for CountingWaker {
    fn wake(self: Arc<Self>) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn the_future_can_be_driven_by_a_foreign_executor() {
    let session = relu_session();
    let mut future: Pin<Box<RunFuture>> = Box::pin(session.run_async(x()));

    let waker_state = Arc::new(CountingWaker {
        wakes: AtomicUsize::new(0),
    });
    let waker = Waker::from(Arc::clone(&waker_state));
    let mut cx = Context::from_waker(&waker);

    // Whether the first poll is Pending or Ready is a race with a very fast
    // inference, so both are accepted.
    let outputs = match future.as_mut().poll(&mut cx) {
        Poll::Ready(outputs) => outputs.expect("run completes"),
        Poll::Pending => {
            // The property a foreign executor actually relies on: after a
            // `Pending` poll the future wakes its task *by itself*, with nobody
            // polling it in between. So this waits for the wake instead of
            // re-polling, and that is what makes the check race-free rather
            // than merely lucky.
            //
            // Re-polling in a loop — what this used to do — failed ~50% of runs
            // on this machine, and the implementation was right while the test
            // was wrong. `Shared::complete` publishes the result *under* the
            // state lock but calls `wake()` *after* releasing it, deliberately
            // (see its doc comment: waking under the lock deadlocks against an
            // executor that polls synchronously from inside `wake()`). A second
            // poll landing in that window legally observes `Ready` while the
            // wake count is still zero, so `polls > 1 => wakes >= 1` was never
            // a sound assertion.
            //
            // Waiting is sound in the other direction: reaching this branch
            // means `poll` registered our waker under the lock *before*
            // `complete` took it, so `complete` is guaranteed to find it and
            // call it. A wake that never arrives is a real bug, and the
            // deadline turns that into a failure rather than a hung suite.
            let deadline = Instant::now() + Duration::from_secs(30);
            while waker_state.wakes.load(Ordering::SeqCst) == 0 {
                assert!(
                    Instant::now() < deadline,
                    "a future that returned Pending must wake its task",
                );
                std::thread::yield_now();
            }
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(outputs) => outputs.expect("run completes"),
                // `complete` stores the result before it takes the waker, so a
                // woken task polling `Pending` would mean the two had been
                // reordered.
                Poll::Pending => panic!("the future must be Ready once it has woken its task"),
            }
        }
    };
    assert_eq!(outputs["y"].data, vec![0.0, 0.0, 2.5]);
}

#[test]
fn a_session_scoped_token_cancels_an_async_run() {
    let graph = Graph {
        name: "relu".to_string(),
        nodes: vec![node(OpKind::Relu, "relu", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let token = CancellationToken::new();
    let session = Arc::new(
        Session::builder()
            .with_session_cancellation(token.clone())
            .build_from_graph(graph, HashMap::new())
            .expect("session builds"),
    );
    token.cancel();
    let outcome = oxionnx::block_on(session.run_async(x()));
    assert!(matches!(outcome, Err(OnnxError::Cancelled(_))));
}

#[test]
fn many_sessions_can_run_concurrently_through_handles() {
    let session = relu_session();
    let handles: Vec<_> = (0..16)
        .map(|_| Arc::clone(&session).spawn_run(x()))
        .collect();
    for handle in handles {
        let outputs = handle.wait().expect("run completes");
        assert_eq!(outputs["y"].data, vec![0.0, 0.0, 2.5]);
    }
}

/// An operator that panics.
///
/// The panic is deliberate and is caught by the worker thread's own unwinding —
/// it prints to stderr but does not fail this test.
struct Exploder;

impl Operator for Exploder {
    fn op_type(&self) -> &str {
        "Exploder"
    }

    fn execute(&self, _ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        panic!("deliberate panic from a test operator");
    }
}

/// A panicking worker must become a typed error, never a permanently pending
/// future: the future and the worker share a one-shot slot, and nothing else
/// would ever fill it.
#[test]
fn a_panicking_inference_thread_resolves_the_future_instead_of_hanging() {
    let mut registry = oxionnx::default_registry();
    registry.register_as("Exploder", Box::new(Exploder));

    let graph = Graph {
        name: "boom".to_string(),
        nodes: vec![node(OpKind::parse("Exploder"), "boom", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    };
    let session = Arc::new(
        Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_registry(registry)
            .build_from_graph(graph, HashMap::new())
            .expect("session builds"),
    );

    match oxionnx::block_on(Arc::clone(&session).run_async(x())) {
        Err(OnnxError::Internal(msg)) => assert!(msg.contains("panicked"), "got: {msg}"),
        other => panic!("expected an Internal error, got {other:?}"),
    }

    // The same holds for the blocking handle.
    match session.spawn_run(x()).wait() {
        Err(OnnxError::Internal(msg)) => assert!(msg.contains("panicked"), "got: {msg}"),
        other => panic!("expected an Internal error, got {other:?}"),
    }
}
