//! Wave-3 (T7-tests-engine): `IoBinding`'s "zero-allocation output buffer
//! reuse" claim, checked at the level the existing tests never reach.
//!
//! `tests/io_binding_test.rs::test_io_binding_output_reuse` and
//! `test_bind_output_prealloc_copy_in_place` already prove the *values* are
//! right after a second `run_with_binding` call. Neither one captures
//! `.data.as_ptr()` before and after, so a regression that silently replaced
//! the `buf.data.copy_from_slice(&tensor.data)` fast path in
//! `Session::run_with_binding` (src/session/run/entry.rs) with an
//! always-allocate-fresh path would still pass both of those tests — same
//! values, new address — while quietly reverting the documented guarantee.
//! These tests close that gap by asserting on the pointer directly.
//!
//! Each positive assertion is paired with a negative control: a case in this
//! same file where the pointer is *supposed* to change (a genuine shape
//! mismatch). Without that control, pointer equality could pass for a reason
//! unrelated to the reuse path (e.g. an allocator that happens to return the
//! same address for two same-sized allocations) rather than because the
//! buffer was actually reused.

use oxionnx::{Attributes, Graph, IoBinding, Node, OpKind, Session, Tensor};
use std::collections::HashMap;

/// `y = Relu(x)`, shape-preserving, slot-capable, no declared static shape —
/// so the same session can be run with inputs of different lengths.
fn relu_session() -> Session {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };
    Session::from_graph(graph, HashMap::new()).expect("session creation should succeed")
}

/// Two independent outputs from the same input: `y1 = Relu(x)`, `y2 = Neg(x)`.
fn two_output_session() -> Session {
    let graph = Graph {
        nodes: vec![
            Node {
                op: OpKind::Relu,
                name: "relu".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y1".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Neg,
                name: "neg".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y2".to_string()],
                attrs: Attributes::default(),
            },
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y1".to_string(), "y2".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };
    Session::from_graph(graph, HashMap::new()).expect("session creation should succeed")
}

/// The documented case: pre-allocate the output buffer, run repeatedly with
/// same-shape inputs. The buffer's address must never move, and its contents
/// must always reflect the *latest* run, never a stale earlier one.
#[test]
fn output_buffer_pointer_is_stable_across_repeated_runs_of_the_same_shape() {
    let session = relu_session();
    let mut binding = IoBinding::new();
    binding.bind_output("y", Tensor::new(vec![0.0f32; 4], vec![4]));

    let mut ptr_after_first: Option<*const f32> = None;

    for round in 0..5 {
        let base = round as f32;
        binding.clear_inputs();
        binding.bind_input(
            "x",
            Tensor::new(vec![base - 1.0, -base, base, base + 1.0], vec![4]),
        );
        session
            .run_with_binding(&mut binding)
            .unwrap_or_else(|e| panic!("run_with_binding failed at round {round}: {e}"));

        let out = binding.get_output("y").expect("output 'y' must be present");
        let expected = vec![
            (base - 1.0).max(0.0),
            (-base).max(0.0),
            base.max(0.0),
            (base + 1.0).max(0.0),
        ];
        assert_eq!(out.data, expected, "wrong Relu values at round {round}");

        let ptr = out.data.as_ptr();
        match ptr_after_first {
            None => ptr_after_first = Some(ptr),
            Some(first) => assert_eq!(
                ptr, first,
                "output buffer address moved at round {round}: the zero-allocation \
                 reuse path (Session::run_with_binding's copy_from_slice branch) \
                 appears to have been replaced by an always-reallocate path",
            ),
        }
    }
}

/// Even when the caller never calls `bind_output` up front, the tensor
/// `run_with_binding` inserts after the *first* run becomes the reuse target
/// for every run after that — the doc-example usage pattern
/// (`io_binding.rs`'s module doc loop, which never pre-binds "y").
#[test]
fn an_unbound_output_becomes_pointer_stable_from_the_second_run_onward() {
    let session = relu_session();
    let mut binding = IoBinding::new();
    // Deliberately no bind_output("y", ...) here.

    binding.bind_input("x", Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]));
    session
        .run_with_binding(&mut binding)
        .expect("first run_with_binding");
    let ptr1 = binding
        .get_output("y")
        .expect("output after first run")
        .data
        .as_ptr();

    binding.clear_inputs();
    binding.bind_input("x", Tensor::new(vec![5.0, -6.0, 7.0, -8.0], vec![4]));
    session
        .run_with_binding(&mut binding)
        .expect("second run_with_binding");
    let out2 = binding.get_output("y").expect("output after second run");
    assert_eq!(out2.data, vec![5.0, 0.0, 7.0, 0.0]);
    assert_eq!(
        out2.data.as_ptr(),
        ptr1,
        "the tensor produced by run 1 must become run 2's reused buffer"
    );

    binding.clear_inputs();
    binding.bind_input("x", Tensor::new(vec![-9.0, 10.0, -11.0, 12.0], vec![4]));
    session
        .run_with_binding(&mut binding)
        .expect("third run_with_binding");
    let out3 = binding.get_output("y").expect("output after third run");
    assert_eq!(out3.data, vec![0.0, 10.0, 0.0, 12.0]);
    assert_eq!(
        out3.data.as_ptr(),
        ptr1,
        "the buffer must stay stable through a third run, not just a second"
    );
}

/// Negative control (the case that must NOT be pointer-stable): a shape change
/// makes `run_with_binding`'s `take_output_buffer` match guard
/// (`buf.data.len() == tensor.data.len() && buf.shape == tensor.shape`) fail,
/// so the old buffer is discarded and the new tensor's own allocation takes
/// its place — a different address. Without this test, the pointer-equality
/// assertions above could pass vacuously (e.g. because a bug always reuses the
/// same address regardless of shape, which would be equally wrong).
#[test]
fn output_buffer_pointer_changes_when_the_output_shape_changes_then_restabilizes() {
    let session = relu_session();
    let mut binding = IoBinding::new();
    binding.bind_output("y", Tensor::new(vec![0.0f32; 4], vec![4]));

    binding.bind_input("x", Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]));
    session
        .run_with_binding(&mut binding)
        .expect("run at len 4");
    let out1 = binding.get_output("y").expect("output after len-4 run");
    assert_eq!(out1.shape, vec![4]);
    let ptr_len4 = out1.data.as_ptr();

    // A longer input forces a differently-shaped, differently-sized output.
    binding.clear_inputs();
    binding.bind_input(
        "x",
        Tensor::new(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], vec![6]),
    );
    session
        .run_with_binding(&mut binding)
        .expect("run at len 6");
    let out2 = binding.get_output("y").expect("output after len-6 run");
    assert_eq!(out2.shape, vec![6]);
    assert_eq!(out2.data, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    assert_ne!(
        out2.data.as_ptr(),
        ptr_len4,
        "a genuine shape change must NOT reuse the old (wrong-length) buffer"
    );
    let ptr_len6 = out2.data.as_ptr();

    // Back to a fresh length-6 run: now the *new* buffer is the stable one.
    binding.clear_inputs();
    binding.bind_input(
        "x",
        Tensor::new(vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], vec![6]),
    );
    session
        .run_with_binding(&mut binding)
        .expect("second run at len 6");
    let out3 = binding
        .get_output("y")
        .expect("output after second len-6 run");
    assert_eq!(out3.data, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    assert_eq!(
        out3.data.as_ptr(),
        ptr_len6,
        "once the shape stabilizes, the buffer allocated for that shape must \
         itself become the reuse target"
    );
}

/// Two differently-named outputs bound at once must each be reused
/// independently — one output's buffer identity must not depend on whether a
/// sibling output happened to change shape.
#[test]
fn two_bound_outputs_are_each_independently_pointer_stable() {
    let session = two_output_session();
    let mut binding = IoBinding::new();
    binding.bind_output("y1", Tensor::new(vec![0.0f32; 3], vec![3]));
    binding.bind_output("y2", Tensor::new(vec![0.0f32; 3], vec![3]));

    binding.bind_input("x", Tensor::new(vec![-1.0, 2.0, -3.0], vec![3]));
    session.run_with_binding(&mut binding).expect("first run");
    let ptr_y1 = binding.get_output("y1").expect("y1 present").data.as_ptr();
    let ptr_y2 = binding.get_output("y2").expect("y2 present").data.as_ptr();
    assert_ne!(
        ptr_y1, ptr_y2,
        "the two outputs must not alias the same allocation"
    );

    for round in 0..3 {
        binding.clear_inputs();
        let v = round as f32 + 1.0;
        binding.bind_input("x", Tensor::new(vec![-v, v, -v * 2.0], vec![3]));
        session
            .run_with_binding(&mut binding)
            .unwrap_or_else(|e| panic!("run failed at round {round}: {e}"));

        let y1 = binding.get_output("y1").expect("y1 present");
        let y2 = binding.get_output("y2").expect("y2 present");
        assert_eq!(
            y1.data,
            vec![0.0, v, 0.0],
            "y1 (Relu) wrong at round {round}"
        );
        assert_eq!(
            y2.data,
            vec![v, -v, v * 2.0],
            "y2 (Neg) wrong at round {round}"
        );
        assert_eq!(
            y1.data.as_ptr(),
            ptr_y1,
            "y1's buffer moved at round {round}"
        );
        assert_eq!(
            y2.data.as_ptr(),
            ptr_y2,
            "y2's buffer moved at round {round}"
        );
    }
}
