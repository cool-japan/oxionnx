//! Wave-3 `T6-tests-ops`: rank-0 (true scalar, `shape=[]`) broadcasting across
//! the full elementwise binary-op surface, from finding [a11-18].
//!
//! `oxionnx-ops/tests/w2_rank0.rs` (pre-existing) already pins rank-0
//! broadcasting for `Add`/`Div` at the direct-function level
//! (`math::add`/`math::div`), plus unary ops, `Gather`, `Expand`, `Where`,
//! `Clip`, `Identity`/`Cast`. What it does not cover is *breadth*: the other
//! dozen-plus binary ops (arithmetic, comparison, logical, and the variadic
//! `Min`/`Max`) going through the `Operator` trait (the layer a real session
//! actually dispatches through), in **both** operand positions, plus whether
//! rank-0-with-rank-0 stays rank-0 across that wider set rather than just
//! `Add`.
//!
//! This file deliberately stays inside the plain elementwise-binary lane.
//! `oxionnx-ops/tests/w2_rank0.rs`'s Part 2 covers the *reduction*/
//! `Squeeze`-family sites, which the Wave-3 migration has since finished: they
//! all return the emptied shape unchanged rather than promoting it to `[1]`
//! (see `reduce_output_shape`/`reduce_with` in oxionnx-ops/src/math/reduce.rs).
//! Broadcasting — this file's subject — was never on that list and was already
//! correct, so nothing here depends on that edit.
//!
//! Reference values are plain arithmetic on values chosen to avoid ties
//! (comparisons) and sign ambiguity (`Mod`, both operands non-negative so the
//! `fmod`-vs-floor-mod distinction cannot affect the result).

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::Tensor;
use oxionnx_ops::registry::math_ops::{
    AddOp, DivOp, ModOp, MulOp, PowOp, SubOp, VariadicMaxOp, VariadicMinOp,
};
use oxionnx_ops::registry::misc_ops::{
    AndOp, EqualOp, GreaterOp, GreaterOrEqualOp, LessOp, LessOrEqualOp, OrOp, XorOp,
};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn dummy_node(op_type: &str) -> Node {
    Node {
        name: "test".into(),
        op: OpKind::parse(op_type),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn run(op: &dyn Operator, a: &Tensor, b: &Tensor) -> Tensor {
    let node = dummy_node(op.op_type());
    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(a), Some(b)],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let mut out = op
        .execute(&ctx)
        .unwrap_or_else(|e| panic!("{}: {e}", op.op_type()));
    assert_eq!(out.len(), 1, "{}: exactly one output", op.op_type());
    out.remove(0)
}

/// The shared matrix operand: `[1, 3, 5, 7]` as `[2, 2]`, chosen so every
/// comparison below is a strict, tie-free relation against the scalar `2.0`.
fn mat() -> Tensor {
    Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![2, 2])
}

fn scalar() -> Tensor {
    Tensor::rank0(2.0)
}

fn assert_mat_shape(t: &Tensor, label: &str) {
    assert_eq!(
        t.shape,
        vec![2, 2],
        "{label}: rank-0 operand must not raise the output rank"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic: Add, Sub, Mul, Div, Pow, Mod
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn arithmetic_ops_broadcast_rank0_in_both_positions() {
    let (m, s) = (mat(), scalar());

    let out = run(&AddOp, &s, &m);
    assert_mat_shape(&out, "Add(scalar, mat)");
    assert_eq!(out.data, vec![3.0, 5.0, 7.0, 9.0]);
    let out = run(&AddOp, &m, &s);
    assert_mat_shape(&out, "Add(mat, scalar)");
    assert_eq!(out.data, vec![3.0, 5.0, 7.0, 9.0], "commutative");

    let out = run(&SubOp, &s, &m);
    assert_mat_shape(&out, "Sub(scalar, mat)");
    assert_eq!(out.data, vec![1.0, -1.0, -3.0, -5.0]);
    let out = run(&SubOp, &m, &s);
    assert_mat_shape(&out, "Sub(mat, scalar)");
    assert_eq!(out.data, vec![-1.0, 1.0, 3.0, 5.0]);

    let out = run(&MulOp, &s, &m);
    assert_mat_shape(&out, "Mul(scalar, mat)");
    assert_eq!(out.data, vec![2.0, 6.0, 10.0, 14.0]);
    let out = run(&MulOp, &m, &s);
    assert_mat_shape(&out, "Mul(mat, scalar)");
    assert_eq!(out.data, vec![2.0, 6.0, 10.0, 14.0], "commutative");

    let out = run(&DivOp, &s, &m);
    assert_mat_shape(&out, "Div(scalar, mat)");
    for (got, want) in out
        .data
        .iter()
        .zip([2.0 / 1.0, 2.0 / 3.0, 2.0 / 5.0, 2.0 / 7.0])
    {
        assert!(
            (got - want).abs() < 1e-6,
            "Div(scalar,mat): {got} vs {want}"
        );
    }
    let out = run(&DivOp, &m, &s);
    assert_mat_shape(&out, "Div(mat, scalar)");
    assert_eq!(out.data, vec![0.5, 1.5, 2.5, 3.5]);

    let out = run(&PowOp, &s, &m);
    assert_mat_shape(&out, "Pow(scalar, mat)");
    assert_eq!(out.data, vec![2.0, 8.0, 32.0, 128.0], "2^[1,3,5,7]");
    let out = run(&PowOp, &m, &s);
    assert_mat_shape(&out, "Pow(mat, scalar)");
    assert_eq!(out.data, vec![1.0, 9.0, 25.0, 49.0], "[1,3,5,7]^2");

    // All-non-negative operands: fmod (default) and floor-mod agree.
    let out = run(&ModOp, &s, &m);
    assert_mat_shape(&out, "Mod(scalar, mat)");
    assert_eq!(out.data, vec![0.0, 2.0, 2.0, 2.0], "2 mod [1,3,5,7]");
    let out = run(&ModOp, &m, &s);
    assert_mat_shape(&out, "Mod(mat, scalar)");
    assert_eq!(out.data, vec![1.0, 1.0, 1.0, 1.0], "[1,3,5,7] mod 2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Variadic Min/Max, exercised in their 2-input degenerate form
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn min_max_broadcast_rank0_in_both_positions() {
    let (m, s) = (mat(), scalar());

    let out = run(&VariadicMinOp, &s, &m);
    assert_mat_shape(&out, "Min(scalar, mat)");
    assert_eq!(
        out.data,
        vec![1.0, 2.0, 2.0, 2.0],
        "elementwise min(2, [1,3,5,7])"
    );
    let out = run(&VariadicMinOp, &m, &s);
    assert_mat_shape(&out, "Min(mat, scalar)");
    assert_eq!(out.data, vec![1.0, 2.0, 2.0, 2.0], "commutative");

    let out = run(&VariadicMaxOp, &s, &m);
    assert_mat_shape(&out, "Max(scalar, mat)");
    assert_eq!(
        out.data,
        vec![2.0, 3.0, 5.0, 7.0],
        "elementwise max(2, [1,3,5,7])"
    );
    let out = run(&VariadicMaxOp, &m, &s);
    assert_mat_shape(&out, "Max(mat, scalar)");
    assert_eq!(out.data, vec![2.0, 3.0, 5.0, 7.0], "commutative");
}

// ═══════════════════════════════════════════════════════════════════════════
// Comparisons: Greater, Less, Equal, GreaterOrEqual, LessOrEqual
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comparison_ops_broadcast_rank0_in_both_positions() {
    let (m, s) = (mat(), scalar());

    let out = run(&GreaterOp, &s, &m);
    assert_mat_shape(&out, "Greater(scalar, mat)");
    assert_eq!(out.data, vec![1.0, 0.0, 0.0, 0.0], "2 > [1,3,5,7]");
    let out = run(&GreaterOp, &m, &s);
    assert_mat_shape(&out, "Greater(mat, scalar)");
    assert_eq!(out.data, vec![0.0, 1.0, 1.0, 1.0], "[1,3,5,7] > 2");

    let out = run(&LessOp, &s, &m);
    assert_mat_shape(&out, "Less(scalar, mat)");
    assert_eq!(out.data, vec![0.0, 1.0, 1.0, 1.0], "2 < [1,3,5,7]");
    let out = run(&LessOp, &m, &s);
    assert_mat_shape(&out, "Less(mat, scalar)");
    assert_eq!(out.data, vec![1.0, 0.0, 0.0, 0.0], "[1,3,5,7] < 2");

    let out = run(&EqualOp, &s, &m);
    assert_mat_shape(&out, "Equal(scalar, mat)");
    assert_eq!(
        out.data,
        vec![0.0, 0.0, 0.0, 0.0],
        "2 never equals [1,3,5,7]"
    );
    let out = run(&EqualOp, &m, &s);
    assert_mat_shape(&out, "Equal(mat, scalar)");
    assert_eq!(out.data, vec![0.0, 0.0, 0.0, 0.0]);

    let out = run(&GreaterOrEqualOp, &s, &m);
    assert_mat_shape(&out, "GreaterOrEqual(scalar, mat)");
    assert_eq!(
        out.data,
        vec![1.0, 0.0, 0.0, 0.0],
        "no ties: same as Greater"
    );

    let out = run(&LessOrEqualOp, &m, &s);
    assert_mat_shape(&out, "LessOrEqual(mat, scalar)");
    assert_eq!(out.data, vec![1.0, 0.0, 0.0, 0.0], "no ties: same as Less");
}

// ═══════════════════════════════════════════════════════════════════════════
// Logical: And, Or, Xor (0.0/1.0-encoded booleans)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn logical_ops_broadcast_rank0_in_both_positions() {
    let mat_bool = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
    let scalar_true = Tensor::rank0(1.0);

    let out = run(&AndOp, &scalar_true, &mat_bool);
    assert_mat_shape(&out, "And(true, mat)");
    assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
    let out = run(&AndOp, &mat_bool, &scalar_true);
    assert_mat_shape(&out, "And(mat, true)");
    assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0], "commutative");

    let out = run(&OrOp, &scalar_true, &mat_bool);
    assert_mat_shape(&out, "Or(true, mat)");
    assert_eq!(
        out.data,
        vec![1.0, 1.0, 1.0, 1.0],
        "true or anything is true"
    );

    let out = run(&XorOp, &scalar_true, &mat_bool);
    assert_mat_shape(&out, "Xor(true, mat)");
    assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0]);
    let out = run(&XorOp, &mat_bool, &scalar_true);
    assert_mat_shape(&out, "Xor(mat, true)");
    assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0], "commutative");
}

// ═══════════════════════════════════════════════════════════════════════════
// rank-0 with rank-0 stays rank-0, across a wider op set than just Add
// ═══════════════════════════════════════════════════════════════════════════

/// `w2_rank0.rs`'s `elementwise_ops_broadcast_rank0_without_raising_rank`
/// checks this for `Add`/`Div` only, at the direct-function level. Here it is
/// checked through the `Operator` trait (the real dispatch layer) for a wider
/// op sample spanning all three families.
#[test]
fn rank0_rank0_stays_rank0_across_op_families() {
    let a = Tensor::rank0(6.0);
    let b = Tensor::rank0(3.0);
    let empty: Vec<usize> = Vec::new();

    let cases: [(&dyn Operator, f32); 4] = [
        (&SubOp, 3.0),
        (&MulOp, 18.0),
        (&PowOp, 216.0),
        (&VariadicMaxOp, 6.0),
    ];
    for (op, want) in cases {
        let out = run(op, &a, &b);
        assert_eq!(
            out.shape,
            empty,
            "{}: rank0 op rank0 must stay rank0",
            op.op_type()
        );
        assert_eq!(out.data, vec![want], "{}", op.op_type());
    }

    let out = run(&GreaterOp, &a, &b);
    assert_eq!(out.shape, empty, "Greater: rank0 op rank0 must stay rank0");
    assert_eq!(out.data, vec![1.0]);

    let bool_a = Tensor::rank0(1.0);
    let bool_b = Tensor::rank0(0.0);
    let out = run(&AndOp, &bool_a, &bool_b);
    assert_eq!(out.shape, empty, "And: rank0 op rank0 must stay rank0");
    assert_eq!(out.data, vec![0.0]);
}
