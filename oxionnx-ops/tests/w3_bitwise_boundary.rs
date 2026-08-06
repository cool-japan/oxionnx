//! Wave-3 `T6-tests-ops`: `Bitwise{And,Or,Xor,Not}` at the f32
//! integer-precision boundary (2^24), from finding [a11-24].
//!
//! `oxionnx-ops/src/bitwise.rs` already carries its own
//! `test_bitwise_and_documents_f32_precision_limit_beyond_2_pow_24`, which
//! covers exactly this boundary for `BitwiseAnd` at the raw-function layer —
//! far more than the finding's premise of "untested at the integer-precision
//! boundary" (that colocated test predates this session and was found during
//! verification, not written by it). It also corrects the finding's
//! secondary premise: the implementation casts through **i64**, not `u32` —
//! documented in the module's own doc comment as a deliberate choice so a
//! negative (sign-extended two's-complement) operand round-trips correctly,
//! which a naive `u32` cast would not.
//!
//! What is still genuinely uncovered: `BitwiseOr`/`BitwiseXor`/`BitwiseNot`
//! at the same boundary (only `BitwiseAnd` has a precision-limit test), and
//! the `Operator`-trait / registry layer (the raw-function tests never go
//! through `execute`, only `oxionnx-ops::bitwise::bitwise_and` directly).
//! This file adds both, plus one positive control using a mask that is
//! itself exactly representable in f32 (`0xFFFF00`, not `0xFFFFFFFF`, which
//! rounds to `2^32` and would only work by the coincidence of Rust's
//! saturating float->int cast — see the session's final report).

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::Tensor;
use oxionnx_ops::bitwise::{bitwise_and, bitwise_not, bitwise_or, bitwise_xor};
use oxionnx_ops::registry::misc_ops::BitwiseAndOp;

/// `2^24 + 1 = 16_777_217` is the smallest positive integer that does not
/// round-trip through f32 exactly; the literal below already becomes
/// `16_777_216.0` the moment it is parsed, before any bitwise op runs.
const JUST_ABOVE_BOUNDARY: f32 = 16_777_217.0;
const ROUNDED: f32 = 16_777_216.0;

/// `BitwiseOr`/`BitwiseXor` with `0` (the identity element for both) still
/// surface the same f32-storage precision loss `BitwiseAnd`'s existing test
/// documents for `AND` with all-ones: the corruption happens when the
/// *operand* is stored, not inside the bitwise op itself, so every op in the
/// family shows it identically.
#[test]
fn bitwise_or_and_xor_document_the_same_f32_precision_limit() {
    let a = Tensor::new(vec![JUST_ABOVE_BOUNDARY], vec![1]);
    assert_eq!(
        a.data[0], ROUNDED,
        "the f32 literal itself already lost the +1"
    );
    let zero = Tensor::new(vec![0.0], vec![1]);

    let or_out = bitwise_or(&a, &zero).expect("bitwise_or");
    assert_eq!(
        or_out.data[0], ROUNDED,
        "OR with 0 is an identity, still shows the corruption"
    );

    let xor_out = bitwise_xor(&a, &zero).expect("bitwise_xor");
    assert_eq!(
        xor_out.data[0], ROUNDED,
        "XOR with 0 is an identity, still shows the corruption"
    );

    // Below the boundary, both remain exact (mirrors the AND test's own
    // below-boundary check).
    let exact = Tensor::new(vec![ROUNDED], vec![1]); // 2^24 itself, still exact
    assert_eq!(bitwise_or(&exact, &zero).expect("or").data[0], ROUNDED);
    assert_eq!(bitwise_xor(&exact, &zero).expect("xor").data[0], ROUNDED);
}

/// `BitwiseNot` at the boundary: `!16_777_216_i64 == -16_777_217`, which is
/// itself not exactly representable in f32 and rounds to `-16_777_216.0` —
/// the unary op inherits the same limit its binary siblings have. Contrast
/// with `!16_777_215_i64 == -16_777_216`, which **is** exactly representable
/// (a power of two magnitude) despite its operand sitting right at
/// `2^24 - 1`: the boundary is about the *result's* magnitude, not just the
/// operand's.
#[test]
fn bitwise_not_precision_limit_beyond_2_pow_24() {
    let x = Tensor::new(vec![ROUNDED], vec![1]); // 2^24
    let out = bitwise_not(&x);
    // True result is -16_777_217 (!16_777_216 == -16_777_217); f32 cannot
    // represent that exactly and rounds it to -16_777_216.
    assert_eq!(
        out.data[0], -16_777_216.0,
        "!2^24 rounds to -2^24, not the true -16_777_217"
    );

    // One below the boundary: the true result (-2^24) IS exact.
    let x_exact = Tensor::new(vec![16_777_215.0], vec![1]); // 2^24 - 1
    let out_exact = bitwise_not(&x_exact);
    assert_eq!(
        out_exact.data[0], -16_777_216.0,
        "!(2^24 - 1) == -2^24, exactly representable"
    );
    assert_eq!(out_exact.data[0] as i64, !16_777_215_i64);
}

/// The `Operator`/registry layer (`BitwiseAndOp::execute`, what a real
/// session actually dispatches through) must show *exactly* the same
/// boundary behavior as the raw `bitwise_and` function — no accidental extra
/// precision gained or lost by the dispatch wrapper.
#[test]
fn bitwise_and_through_registry_matches_the_raw_function_at_the_boundary() {
    let a = Tensor::new(vec![JUST_ABOVE_BOUNDARY], vec![1]);
    let all_ones = Tensor::new(vec![-1.0], vec![1]);

    let direct = bitwise_and(&a, &all_ones).expect("direct");

    let node = Node {
        name: "test".into(),
        op: OpKind::BitwiseAnd,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    };
    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(&a), Some(&all_ones)],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let via_op = BitwiseAndOp.execute(&ctx).expect("BitwiseAndOp::execute");

    assert_eq!(
        via_op[0].data, direct.data,
        "registry layer must agree with the raw function"
    );
    assert_eq!(
        via_op[0].data[0], ROUNDED,
        "the boundary corruption survives to the operator layer"
    );
}

/// Positive control: a mask that is itself exactly representable in f32
/// (`0xFFFF00 = 16_776_960`, not `0xFFFFFFFF`, which rounds to `2^32` and
/// would only coincidentally still work through Rust's *saturating*
/// float->int cast — see the module doc). Both operands here sit safely
/// below the 2^24 boundary, so the result must be bit-exact integer `AND`.
#[test]
fn bitwise_and_with_an_exactly_representable_mask_is_bit_exact_below_the_boundary() {
    let value = Tensor::new(vec![16_777_215.0], vec![1]); // 2^24 - 1, exact
    let mask = Tensor::new(vec![16_776_960.0], vec![1]); // 0xFFFF00, exact
    assert_eq!(mask.data[0] as u32, 0xFFFF00);

    let out = bitwise_and(&value, &mask).expect("bitwise_and");
    assert_eq!(
        out.data[0] as u32,
        16_777_215u32 & 0xFFFF00,
        "bit-exact within the safe range"
    );
    assert_eq!(out.data[0], 16_776_960.0);
}
