//! Wave-3: an operator registered **after** the session was built is a
//! cancellation point too.
//!
//! # The hole this closes
//!
//! `SessionBuilder::with_session_cancellation` wraps every operator the model
//! can reach in a guard that consults the token before delegating.  The wrapping
//! happens once, at construction, from the final node list.  `Session::register_op`
//! then inserted straight into that already-wrapped registry, so a custom
//! operator installed afterwards ran unguarded: correct results, but not a place
//! the run could be stopped.  For a caller whose *whole reason* for a custom
//! operator is that it is the expensive node, that is the one node cancellation
//! needed to cover.
//!
//! The tests are built around operators that count their own invocations, so
//! "did the run stop before this node" is an exact assertion rather than a race
//! with a background thread.

use oxionnx::{
    Attributes, CancellationToken, Graph, Node, OnnxError, OpContext, OpKind, Operator, OptLevel,
    Session, Tensor,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ── instrumentation ─────────────────────────────────────────────────────────

/// Doubles its input and counts that it ran.
struct Counted {
    name: String,
    count: Arc<AtomicUsize>,
    /// Reported through `supports_inplace`, so the guard's predicate forwarding
    /// can be observed from outside.
    inplace: bool,
}

impl Operator for Counted {
    fn op_type(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        let input = ctx.input(0)?;
        Ok(vec![Tensor::new(
            input.data.iter().map(|v| v * 2.0).collect(),
            input.shape.clone(),
        )])
    }

    fn supports_inplace(&self) -> bool {
        self.inplace
    }

    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        for v in input.data.iter_mut() {
            *v *= 2.0;
        }
        Ok(vec![input])
    }
}

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// `x → Custom → y`, where `Custom` is not in the default registry.
fn custom_graph() -> Graph {
    Graph {
        nodes: vec![node(
            OpKind::Unknown("Custom".to_string()),
            "custom",
            &["x"],
            &["y"],
        )],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    }
}

fn input_x() -> HashMap<&'static str, Tensor> {
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
    inputs
}

/// A session with a token bound, plus the late-registered operator's run counter.
fn cancellable_session_with_late_op(
    token: &CancellationToken,
    inplace: bool,
) -> (Session, Arc<AtomicUsize>) {
    let mut session = Session::builder()
        .with_session_cancellation(token.clone())
        // `Custom` is unregistered at build time, so optimization passes that
        // consult the registry must not be allowed to drop the node.
        .with_optimization_level(OptLevel::None)
        .build_from_graph(custom_graph(), HashMap::new())
        .expect("build");

    let count = Arc::new(AtomicUsize::new(0));
    session.register_op(Box::new(Counted {
        name: "Custom".to_string(),
        count: Arc::clone(&count),
        inplace,
    }));
    (session, count)
}

// ── the regression ──────────────────────────────────────────────────────────

/// The point of the whole file: cancel the token, then run.  The late-registered
/// operator must **not** execute, and the run must report `Cancelled`.
#[test]
fn a_late_registered_operator_is_a_cancellation_point() {
    for inplace in [false, true] {
        let token = CancellationToken::new();
        let (session, count) = cancellable_session_with_late_op(&token, inplace);

        token.cancel();
        let result = session.run(&input_x());

        assert!(
            matches!(result, Err(OnnxError::Cancelled(_))),
            "expected Cancelled, got {result:?} (inplace = {inplace})",
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            0,
            "the guard must stop the operator before it runs (inplace = {inplace})",
        );
    }
}

/// The guard must be transparent when the token is *not* cancelled: same result,
/// same number of invocations, on both the allocating and the in-place path.
#[test]
fn an_uncancelled_late_registered_operator_runs_normally() {
    for inplace in [false, true] {
        let token = CancellationToken::new();
        let (session, count) = cancellable_session_with_late_op(&token, inplace);

        let out = session.run(&input_x()).expect("uncancelled run");
        assert_eq!(
            out.get("y").expect("y").data,
            vec![2.0, 4.0, 6.0],
            "inplace = {inplace}",
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

/// A session with **no** token must not pay for a guard, and must behave exactly
/// as before.
#[test]
fn a_session_without_cancellation_registers_the_operator_unwrapped() {
    let mut session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(custom_graph(), HashMap::new())
        .expect("build");
    let count = Arc::new(AtomicUsize::new(0));
    session.register_op(Box::new(Counted {
        name: "Custom".to_string(),
        count: Arc::clone(&count),
        inplace: false,
    }));

    assert!(
        session.session_cancellation_token().is_none(),
        "no token was bound",
    );
    let out = session.run(&input_x()).expect("run");
    assert_eq!(out.get("y").expect("y").data, vec![2.0, 4.0, 6.0]);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

// ── the guard must stay transparent ─────────────────────────────────────────

/// The registry key is the **inner** operator's `op_type()`, so a late
/// registration still replaces a previously registered operator of the same
/// name rather than shadowing it under a different key.
#[test]
fn a_late_registration_replaces_an_earlier_one_of_the_same_name() {
    let token = CancellationToken::new();
    let (mut session, first) = cancellable_session_with_late_op(&token, false);

    let second = Arc::new(AtomicUsize::new(0));
    session.register_op(Box::new(Counted {
        name: "Custom".to_string(),
        count: Arc::clone(&second),
        inplace: false,
    }));

    session.run(&input_x()).expect("run");
    assert_eq!(first.load(Ordering::SeqCst), 0, "the first was replaced");
    assert_eq!(second.load(Ordering::SeqCst), 1, "the second ran");

    // …and the replacement is still a cancellation point.
    token.cancel();
    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
    assert_eq!(second.load(Ordering::SeqCst), 1, "it did not run again");
}

/// `supports_inplace` must be forwarded, or the guard silently switches the
/// in-place fast path off for every late-registered operator.  The registry the
/// session dispatches through is the wrapped one, so this reads the predicate
/// exactly where the run loop reads it.
#[test]
fn the_guard_forwards_the_dispatch_predicates() {
    for inplace in [false, true] {
        let token = CancellationToken::new();
        let (session, _) = cancellable_session_with_late_op(&token, inplace);
        let op = session
            .operator_registry()
            .get("Custom")
            .expect("registered");
        assert_eq!(
            op.supports_inplace(),
            inplace,
            "the guard must report the inner operator's predicate",
        );
        assert_eq!(op.op_type(), "Custom", "and the inner operator's name");
        assert!(
            !op.supports_output_slots(),
            "the inner operator declines slots, so the guard must too",
        );
        assert!(op.native_dtypes().is_empty());
    }
}

/// Resetting the token releases the session again — the guard reads the flag on
/// every call rather than snapshotting it at wrap time.
#[test]
fn resetting_the_token_releases_the_late_registered_operator() {
    let token = CancellationToken::new();
    let (session, count) = cancellable_session_with_late_op(&token, false);

    token.cancel();
    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);

    token.reset();
    let out = session.run(&input_x()).expect("run after reset");
    assert_eq!(out.get("y").expect("y").data, vec![2.0, 4.0, 6.0]);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

/// Binding the token **after** the registration already worked (the registry
/// wrap covers every reachable op type).  That direction must keep working.
#[test]
fn binding_the_token_after_registration_also_guards_the_operator() {
    let mut session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(custom_graph(), HashMap::new())
        .expect("build");
    let count = Arc::new(AtomicUsize::new(0));
    session.register_op(Box::new(Counted {
        name: "Custom".to_string(),
        count: Arc::clone(&count),
        inplace: false,
    }));

    let token = CancellationToken::new();
    session.set_session_cancellation(token.clone());
    token.cancel();

    assert!(matches!(
        session.run(&input_x()),
        Err(OnnxError::Cancelled(_))
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
