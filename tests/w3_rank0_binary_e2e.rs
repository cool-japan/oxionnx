//! Wave-3 `T6-tests-ops`: rank-0 (true scalar) binary-op broadcasting through
//! a real `Session::run`, from finding [a11-18].
//!
//! `oxionnx-ops/tests/w3_rank0_binary_matrix.rs` covers breadth across the
//! elementwise-binary op surface at the `Operator` trait layer; this file
//! adds the one thing that layer cannot reach — a genuine `Session` (named
//! graph inputs, shape resolution, output collection) fed a rank-0 tensor.

mod common;

use common::run_op;
use oxionnx::{Attributes, OpKind, Tensor};

/// `Add` with a rank-0 `bias` graph input and a `[2,2]` matrix input, through
/// a real session. The rank-0 operand must broadcast without raising the
/// output rank, exactly as it does at the direct-`Operator` layer.
#[test]
fn add_with_rank0_operand_through_session_e2e() {
    let mat = Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![2, 2]);
    let bias = Tensor::rank0(2.0);

    let out = run_op(
        OpKind::Add,
        vec!["mat", "bias"],
        vec!["y"],
        vec!["mat", "bias"],
        vec![("mat", mat), ("bias", bias)],
        vec![],
        Attributes::default(),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(
        y.shape,
        vec![2, 2],
        "rank-0 operand must not raise the output rank"
    );
    assert_eq!(y.data, vec![3.0, 5.0, 7.0, 9.0]);
}

/// The same check with the rank-0 operand in the *first* position, and a
/// comparison op (a different code path from arithmetic) rather than `Add`.
#[test]
fn greater_with_rank0_operand_first_through_session_e2e() {
    let threshold = Tensor::rank0(4.0);
    let mat = Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![2, 2]);

    let out = run_op(
        OpKind::Greater,
        vec!["threshold", "mat"],
        vec!["y"],
        vec!["threshold", "mat"],
        vec![("threshold", threshold), ("mat", mat)],
        vec![],
        Attributes::default(),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![2, 2]);
    assert_eq!(y.data, vec![1.0, 1.0, 0.0, 0.0], "4 > [1,3,5,7]");
}
