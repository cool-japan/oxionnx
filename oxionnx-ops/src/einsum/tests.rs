//! Tests for [`crate::einsum`].
//!
//! Reference values in [`numpy_cases`] were produced by `numpy.einsum` in
//! float64 over inputs drawn from `{-2.0, -1.75, …, 2.0}` (exact binary
//! fractions, so the f32 evaluation of these small contractions is exact too).
//! The generator lives in the W2-einsum work notes; each case records the
//! equation, the operand shapes and the expected result.
//!
//! Three layers of checking:
//!
//! 1. every case through the public [`einsum`] entry point (whatever strategy
//!    its size heuristic picks),
//! 2. every case through **both** executors explicitly, and
//! 3. the two executors against each other, which is what actually validates
//!    the batch/M/K/N bucketing of the GEMM lowering.
//!
//! Plus [`gemm_path_is_selected_for_attention`], which asserts the *plan* — a
//! GEMM path that silently fell back to the scalar loop would pass every
//! numeric check above.

use super::contract::{plan_contraction, StepKind};
use super::parse::parse_equation;
use super::*;
use oxionnx_core::Tensor;
use std::collections::HashMap;

/// Absolute tolerance for comparisons against the numpy reference and between
/// the two executors. See the module docs of [`crate::einsum`]: lowering to
/// `sgemm` re-associates the contraction sum, so agreement is to a tolerance,
/// not bit-for-bit.
const TOL: f32 = 1e-4;

struct Case {
    equation: &'static str,
    inputs: Vec<(Vec<f32>, Vec<usize>)>,
    expected: Vec<f32>,
    shape: Vec<usize>,
}

impl Case {
    fn tensors(&self) -> Vec<Tensor> {
        self.inputs
            .iter()
            .map(|(data, shape)| Tensor::new(data.clone(), shape.clone()))
            .collect()
    }
}

fn assert_close(actual: &Tensor, expected: &[f32], shape: &[usize], what: &str) {
    assert_eq!(actual.shape, shape, "{what}: wrong output shape");
    assert_eq!(
        actual.data.len(),
        expected.len(),
        "{what}: wrong element count"
    );
    for (index, (&got, &want)) in actual.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "{what}: element {index} is {got}, expected {want}"
        );
    }
}

fn numpy_cases() -> Vec<Case> {
    vec![
        // ellipsis_batch_matmul: numpy.einsum("...ij,...jk->...ik") on shapes [(2, 3, 4), (2, 4, 5)]
        Case {
            equation: "...ij,...jk->...ik",
            inputs: vec![
                (
                    vec![
                        -1.25, 0.0, 1.5, 0.5, 1.25, 0.0, -1.25, -1.25, 0.0, 0.0, -0.75, -1.25, 0.0,
                        -1.0, -1.0, 1.0, -1.25, 0.25, -2.0, 1.5, 1.75, 1.25, 0.75, -1.0,
                    ],
                    vec![2, 3, 4],
                ),
                (
                    vec![
                        1.75, 0.5, 1.25, -2.0, 0.0, -1.25, 1.25, 1.5, -1.25, -1.75, -0.25, -1.25,
                        0.0, 0.5, -1.75, -1.75, 0.75, -1.5, -1.75, -1.25, 2.0, -1.5, 1.25, 2.0,
                        -0.25, 1.25, -0.5, -1.0, -1.75, 0.0, 2.0, -1.25, -1.0, -0.5, -1.0, 1.75,
                        -0.75, -1.75, 0.75, -1.25,
                    ],
                    vec![2, 4, 5],
                ),
            ],
            expected: vec![
                -3.4375, -2.125, -2.3125, 2.375, -3.25, 4.6875, 1.25, 3.4375, -0.9375, 3.75, 2.375,
                0.0, 1.875, 1.8125, 2.875, -1.5, 1.0, 0.25, 3.0, -0.25, -3.5625, 3.125, -2.4375,
                -0.8125, 0.4375, 4.8125, -3.4375, 1.9375, 0.1875, 0.0625,
            ],
            shape: vec![2, 3, 5],
        },
        // ellipsis_broadcast: numpy.einsum("...ij,...jk->...ik") on shapes [(1, 3, 4), (5, 4, 2)]
        Case {
            equation: "...ij,...jk->...ik",
            inputs: vec![
                (
                    vec![
                        2.0, 1.75, -0.25, 1.25, 1.0, 2.0, -1.75, 0.5, -0.25, -1.75, 0.75, -0.25,
                    ],
                    vec![1, 3, 4],
                ),
                (
                    vec![
                        1.75, 0.5, -1.5, -0.25, 1.75, 1.5, 1.0, 1.25, 2.0, 0.0, 0.75, 0.75, 1.5,
                        -1.0, 1.5, 0.5, 0.0, 0.0, -1.75, 0.75, 1.5, 1.25, -2.0, 0.75, 0.25, -1.75,
                        2.0, 0.75, 1.5, -0.75, 1.5, -0.5, -1.75, -2.0, -0.25, -1.75, -0.25, 1.75,
                        -0.5, -0.25,
                    ],
                    vec![5, 4, 2],
                ),
            ],
            expected: vec![
                1.6875, 1.75, -3.8125, -2.0, 3.25, 1.125, 6.8125, 2.1875, 1.625, 3.5, -1.0625,
                -2.1875, -5.9375, 1.9375, -7.125, -0.3125, 4.6875, -0.5625, 5.5, -2.625, 2.375,
                0.8125, -2.8125, -1.3125, -4.5, -7.8125, -2.0625, -8.6875, 0.8125, 4.9375,
            ],
            shape: vec![5, 3, 2],
        },
        // ellipsis_two_leading: numpy.einsum("...ij,...jk->...ik") on shapes [(2, 1, 3, 4), (3, 4, 5)]
        Case {
            equation: "...ij,...jk->...ik",
            inputs: vec![
                (
                    vec![
                        0.5, 2.0, 1.25, -1.5, 1.75, -1.75, 1.75, 1.75, -1.0, 1.5, -1.75, -1.5,
                        -2.0, -1.75, 1.75, -1.5, 1.0, 0.0, 0.75, 0.75, 0.75, 1.75, 1.0, 2.0,
                    ],
                    vec![2, 1, 3, 4],
                ),
                (
                    vec![
                        -1.25, -0.75, 1.25, -0.5, -1.25, 1.25, -0.5, 2.0, 0.0, 0.0, 1.75, -2.0,
                        1.5, 1.5, 1.0, 0.0, 1.5, 1.0, 2.0, -1.25, -0.25, 1.0, 0.75, -0.75, -1.75,
                        -1.0, -0.25, 0.25, 1.0, -2.0, 0.0, 1.5, 0.25, -0.5, -1.75, -1.0, -1.25,
                        -1.0, -0.25, 0.75, 1.0, 0.75, 0.0, 1.75, -2.0, -1.5, 1.75, 1.25, 1.5, 1.75,
                        -0.25, 0.25, -2.0, 1.75, 1.25, -1.5, -1.75, 1.5, -0.75, 1.5,
                    ],
                    vec![3, 4, 5],
                ),
            ],
            expected: vec![
                4.0625, -6.125, 5.0, -1.375, 2.5, -1.3125, -1.3125, 3.0625, 5.25, -2.625, 0.0625,
                1.25, -2.375, -5.125, 1.375, -0.625, 3.75, 2.6875, 1.375, -8.1875, -0.4375, 2.625,
                -0.4375, -4.375, -1.3125, 0.25, -2.125, 0.6875, 3.5, 0.6875, -0.5625, 6.8125,
                -2.25, 7.1875, 1.8125, 1.3125, -4.375, -3.0625, 2.1875, -1.75, -0.5625, 4.0625,
                3.125, -1.4375, 0.1875, 3.375, -3.375, -4.875, 0.625, 6.125, 0.0625, -1.125, 3.125,
                2.125, -1.4375, 3.0, -0.4375, 7.9375, 5.125, -2.4375, 3.75, 2.9375, 0.0, -0.75,
                2.8125, -1.0, 1.1875, 0.1875, -1.3125, -2.5, -3.9375, -0.6875, -0.75, 0.1875,
                -5.0625, 2.4375, -1.5, -7.9375, -1.9375, 0.875, -0.3125, -0.375, -0.375, 2.5,
                0.0625, -5.125, 0.375, 3.1875, 4.1875, 5.8125,
            ],
            shape: vec![2, 3, 3, 5],
        },
        // ellipsis_dot: numpy.einsum("...i,...i->...") on shapes [(2, 3), (2, 3)]
        Case {
            equation: "...i,...i->...",
            inputs: vec![
                (vec![-1.75, 1.25, -2.0, -1.25, -1.0, -1.25], vec![2, 3]),
                (vec![0.5, -1.0, -1.0, 0.0, 2.0, -0.25], vec![2, 3]),
            ],
            expected: vec![-0.125, -1.6875],
            shape: vec![2],
        },
        // ellipsis_middle: numpy.einsum("i...j->...ij") on shapes [(2, 3, 4)]
        Case {
            equation: "i...j->...ij",
            inputs: vec![(
                vec![
                    -1.75, 1.25, 1.25, 1.25, -1.0, -1.5, -0.25, 0.0, 0.25, -1.0, 1.5, -1.5, -0.25,
                    1.25, 1.0, 1.25, 1.25, 1.75, -0.75, 1.25, -0.25, 0.5, 1.5, -0.25,
                ],
                vec![2, 3, 4],
            )],
            expected: vec![
                -1.75, 1.25, 1.25, 1.25, -0.25, 1.25, 1.0, 1.25, -1.0, -1.5, -0.25, 0.0, 1.25,
                1.75, -0.75, 1.25, 0.25, -1.0, 1.5, -1.5, -0.25, 0.5, 1.5, -0.25,
            ],
            shape: vec![3, 2, 4],
        },
        // ellipsis_implicit: numpy.einsum("...ij,...jk") on shapes [(2, 3, 4), (2, 4, 5)]
        Case {
            equation: "...ij,...jk",
            inputs: vec![
                (
                    vec![
                        0.25, 1.5, 0.0, -0.25, 2.0, 1.0, 0.75, 0.0, -1.5, 0.5, -1.25, -1.5, 1.25,
                        -0.25, -1.75, 1.25, -1.5, 0.25, -1.0, -0.5, -1.75, 0.75, -0.25, -2.0,
                    ],
                    vec![2, 3, 4],
                ),
                (
                    vec![
                        1.0, 1.5, -1.0, -1.5, 0.5, 1.25, 1.0, 2.0, 1.25, 0.5, 1.75, 1.5, 1.5, -2.0,
                        0.0, 0.25, 1.5, 0.0, 1.5, -2.0, 0.25, 0.75, -1.0, -0.5, 0.5, -0.75, -0.5,
                        0.25, 1.5, -1.5, 0.5, 0.0, 1.75, -1.0, -1.25, 0.25, 0.75, 1.0, 0.0, -2.0,
                    ],
                    vec![2, 4, 5],
                ),
            ],
            expected: vec![
                2.0625, 1.5, 2.75, 1.125, 1.375, 4.5625, 5.125, 1.125, -3.25, 1.5, -3.4375, -5.875,
                0.625, 3.125, 2.5, -0.0625, 2.0, -3.125, 0.75, 0.6875, -1.1875, -1.625, -0.6875,
                2.125, 1.125, -1.625, -3.1875, -0.5, 2.25, 2.3125,
            ],
            shape: vec![2, 3, 5],
        },
        // ellipsis_one_operand: numpy.einsum("...ij,jk->...ik") on shapes [(3, 2, 3), (3, 4)]
        Case {
            equation: "...ij,jk->...ik",
            inputs: vec![
                (
                    vec![
                        -0.5, 0.25, 1.5, -2.0, -0.75, 1.25, 2.0, -0.75, -0.75, -0.25, 0.0, -1.5,
                        -0.5, 0.25, 0.5, -1.75, 0.75, -1.75,
                    ],
                    vec![3, 2, 3],
                ),
                (
                    vec![
                        -0.25, -1.25, -1.5, 1.0, 0.25, 0.25, -2.0, -1.5, 1.75, 1.5, 1.5, 0.0,
                    ],
                    vec![3, 4],
                ),
            ],
            expected: vec![
                2.8125, 2.9375, 2.5, -0.875, 2.5, 4.1875, 6.375, -0.875, -2.0, -3.8125, -2.625,
                3.125, -2.5625, -1.9375, -1.875, -0.25, 1.0625, 1.4375, 1.0, -0.875, -2.4375,
                -0.25, -1.5, -2.875,
            ],
            shape: vec![3, 2, 4],
        },
        // ellipsis_zero_dims: numpy.einsum("...ij,...jk->...ik") on shapes [(3, 4), (4, 5)]
        Case {
            equation: "...ij,...jk->...ik",
            inputs: vec![
                (
                    vec![
                        1.5, 1.0, -2.0, 1.0, 0.0, 1.75, -1.0, -1.75, 0.0, 1.75, 2.0, -0.75,
                    ],
                    vec![3, 4],
                ),
                (
                    vec![
                        0.5, -1.0, -0.5, 0.5, -1.0, 0.25, 0.25, 0.75, -1.0, -2.0, -0.5, 0.0, 1.25,
                        -1.75, -2.0, 1.75, -1.25, -1.75, -1.75, 1.25,
                    ],
                    vec![4, 5],
                ),
            ],
            expected: vec![
                3.75, -2.5, -4.25, 1.5, 1.75, -2.125, 2.625, 3.125, 3.0625, -3.6875, -1.875, 1.375,
                5.125, -3.9375, -8.4375,
            ],
            shape: vec![3, 5],
        },
        // ellipsis_scalar_only: numpy.einsum("...,...->...") on shapes [(2, 3), (2, 3)]
        Case {
            equation: "...,...->...",
            inputs: vec![
                (vec![0.0, 2.0, 0.5, -1.5, 1.5, 1.5], vec![2, 3]),
                (vec![1.5, -1.5, 2.0, 1.25, 0.25, 0.5], vec![2, 3]),
            ],
            expected: vec![0.0, -3.0, 1.0, -1.875, 0.375, 0.75],
            shape: vec![2, 3],
        },
        // ellipsis_reorder: numpy.einsum("...ij->j...i") on shapes [(2, 3, 4)]
        Case {
            equation: "...ij->j...i",
            inputs: vec![(
                vec![
                    -2.0, -1.25, -0.25, 1.25, 1.75, 1.5, -1.25, 0.25, -1.75, -0.5, -0.25, 1.75,
                    -1.0, -1.5, -0.75, 2.0, 0.0, 1.75, 0.5, -1.25, 0.0, -1.0, -1.75, -0.75,
                ],
                vec![2, 3, 4],
            )],
            expected: vec![
                -2.0, 1.75, -1.75, -1.0, 0.0, 0.0, -1.25, 1.5, -0.5, -1.5, 1.75, -1.0, -0.25,
                -1.25, -0.25, -0.75, 0.5, -1.75, 1.25, 0.25, 1.75, 2.0, -1.25, -0.75,
            ],
            shape: vec![4, 2, 3],
        },
        // diag_extract: numpy.einsum("ii->i") on shapes [(4, 4)]
        Case {
            equation: "ii->i",
            inputs: vec![(
                vec![
                    0.5, -1.25, -1.75, 0.0, -0.25, -0.25, 1.25, -0.75, -1.0, 2.0, -0.75, -1.75,
                    1.0, -0.5, -0.25, 1.25,
                ],
                vec![4, 4],
            )],
            expected: vec![0.5, -0.25, -0.75, 1.25],
            shape: vec![4],
        },
        // diag_trace: numpy.einsum("ii->") on shapes [(4, 4)]
        Case {
            equation: "ii->",
            inputs: vec![(
                vec![
                    -1.75, 1.5, -0.25, -0.75, -1.0, -0.75, -1.5, 1.25, 0.25, 1.25, -1.75, 1.0,
                    -1.75, -0.5, 0.75, 2.0,
                ],
                vec![4, 4],
            )],
            expected: vec![-2.25],
            shape: vec![],
        },
        // diag_then_matmul: numpy.einsum("iij,jk->ik") on shapes [(3, 3, 4), (4, 2)]
        Case {
            equation: "iij,jk->ik",
            inputs: vec![
                (
                    vec![
                        0.75, 0.75, 1.25, -1.5, -2.0, 1.5, -1.0, -0.25, 1.5, 1.75, 1.25, -1.0, 0.0,
                        -1.75, 1.25, 2.0, 1.25, 0.0, -1.75, 1.25, -0.5, 1.75, 0.5, 0.25, 2.0, 2.0,
                        1.0, -1.25, -1.25, 0.75, 0.5, -1.0, 0.25, 1.5, -1.25, -1.0,
                    ],
                    vec![3, 3, 4],
                ),
                (
                    vec![-1.75, 1.25, -0.75, -1.5, 0.0, 1.25, 0.25, -1.5],
                    vec![4, 2],
                ),
            ],
            expected: vec![-2.25, 3.625, -1.875, -2.5, -1.8125, -2.0],
            shape: vec![3, 2],
        },
        // diag_middle: numpy.einsum("iji->ij") on shapes [(3, 4, 3)]
        Case {
            equation: "iji->ij",
            inputs: vec![(
                vec![
                    -0.75, 1.25, -1.5, -0.75, 1.25, -1.5, -0.75, 2.0, -1.75, -1.0, 0.5, -1.0, 1.75,
                    -0.5, -2.0, 1.5, -0.25, 0.0, 0.0, -2.0, 0.25, -1.25, 1.75, -2.0, -1.75, -0.75,
                    1.75, -1.25, -1.0, -1.25, 1.75, 1.75, -1.5, -0.5, -0.75, 0.5,
                ],
                vec![3, 4, 3],
            )],
            expected: vec![
                -0.75, -0.75, -0.75, -1.0, -0.5, -0.25, -2.0, 1.75, 1.75, -1.25, -1.5, 0.5,
            ],
            shape: vec![3, 4],
        },
        // diag_double: numpy.einsum("iijj->ij") on shapes [(2, 2, 3, 3)]
        Case {
            equation: "iijj->ij",
            inputs: vec![(
                vec![
                    0.5, 1.75, -1.75, -2.0, -1.25, 1.25, 0.75, 0.25, -0.5, -1.25, 0.5, -0.5, -0.5,
                    -0.25, 1.25, 0.5, 0.5, -1.75, 0.75, -1.0, -2.0, 1.5, 0.0, -1.75, -1.0, -0.25,
                    -1.25, -0.75, -1.75, 0.25, -0.25, -0.25, -0.25, 1.75, -1.25, 1.0,
                ],
                vec![2, 2, 3, 3],
            )],
            expected: vec![0.5, -1.25, -0.5, -0.75, -0.25, 1.0],
            shape: vec![2, 3],
        },
        // diag_partial_sum: numpy.einsum("iij->j") on shapes [(3, 3, 4)]
        Case {
            equation: "iij->j",
            inputs: vec![(
                vec![
                    0.25, -2.0, 0.75, 0.0, 0.25, -1.5, 0.5, -1.75, 0.75, 0.75, 0.75, 1.0, -1.5,
                    0.25, 0.5, -1.0, 1.0, 1.0, -1.5, -2.0, -1.0, -1.25, 1.75, -0.75, 0.75, 1.5,
                    -2.0, 1.75, -0.5, 0.0, -2.0, 0.25, -1.5, 1.5, 1.0, -1.25,
                ],
                vec![3, 3, 4],
            )],
            expected: vec![-0.25, 0.5, 0.25, -3.25],
            shape: vec![4],
        },
        // diag_implicit_trace: numpy.einsum("ii") on shapes [(4, 4)]
        Case {
            equation: "ii",
            inputs: vec![(
                vec![
                    1.0, 0.0, -0.75, 2.0, -0.25, 1.75, -1.25, -1.25, 2.0, -1.5, -1.5, 0.0, 0.25,
                    -1.75, -1.75, 1.0,
                ],
                vec![4, 4],
            )],
            expected: vec![2.25],
            shape: vec![],
        },
        // implicit_matmul: numpy.einsum("ij,jk") on shapes [(2, 3), (3, 2)]
        Case {
            equation: "ij,jk",
            inputs: vec![
                (vec![-0.25, -1.5, -2.0, 1.75, 1.0, -2.0], vec![2, 3]),
                (vec![0.25, 0.5, -1.25, -0.25, 2.0, 1.25], vec![3, 2]),
            ],
            expected: vec![-2.1875, -2.25, -4.8125, -1.875],
            shape: vec![2, 2],
        },
        // implicit_case_order: numpy.einsum("ab,bC") on shapes [(2, 3), (3, 4)]
        Case {
            equation: "ab,bC",
            inputs: vec![
                (vec![2.0, -1.75, 1.75, -1.25, 1.25, 0.25], vec![2, 3]),
                (
                    vec![
                        -1.5, 1.0, 0.0, 1.0, 2.0, -0.25, 0.0, 0.75, 1.5, -0.25, -2.0, 1.25,
                    ],
                    vec![3, 4],
                ),
            ],
            expected: vec![-3.875, 4.75, 2.0, -1.625, -3.5, -0.5, 2.875, 0.0],
            shape: vec![4, 2],
        },
        // implicit_transpose: numpy.einsum("ji") on shapes [(2, 3)]
        Case {
            equation: "ji",
            inputs: vec![(vec![-2.0, 0.75, -0.5, 1.75, 1.25, 2.0], vec![2, 3])],
            expected: vec![-2.0, 1.75, 0.75, 1.25, -0.5, 2.0],
            shape: vec![3, 2],
        },
        // implicit_outer: numpy.einsum("i,j") on shapes [(3,), (4,)]
        Case {
            equation: "i,j",
            inputs: vec![
                (vec![-2.0, -0.25, 0.0], vec![3]),
                (vec![-0.75, -1.75, -0.25, -1.75], vec![4]),
            ],
            expected: vec![
                1.5, 3.5, 0.5, 3.5, 0.1875, 0.4375, 0.0625, 0.4375, 0.0, 0.0, 0.0, 0.0,
            ],
            shape: vec![3, 4],
        },
        // named_broadcast: numpy.einsum("ij,jk->ik") on shapes [(2, 1), (3, 4)]
        Case {
            equation: "ij,jk->ik",
            inputs: vec![
                (vec![-0.5, -1.0], vec![2, 1]),
                (
                    vec![
                        -1.0, 0.75, 2.0, -1.75, -1.25, 2.0, -0.75, 1.0, 1.25, -1.75, -1.25, 1.75,
                    ],
                    vec![3, 4],
                ),
            ],
            expected: vec![0.5, -0.5, 0.0, -0.5, 1.0, -1.0, 0.0, -1.0],
            shape: vec![2, 4],
        },
        // named_broadcast_rev: numpy.einsum("ij,jk->ik") on shapes [(2, 3), (1, 4)]
        Case {
            equation: "ij,jk->ik",
            inputs: vec![
                (vec![-1.25, 1.75, 1.0, 0.75, -0.5, -1.25], vec![2, 3]),
                (vec![-1.75, -1.5, 0.0, 0.5], vec![1, 4]),
            ],
            expected: vec![-2.625, -2.25, 0.0, 0.75, 1.75, 1.5, 0.0, -0.5],
            shape: vec![2, 4],
        },
        // attention_scores: numpy.einsum("bhqd,bhkd->bhqk") on shapes [(2, 3, 4, 5), (2, 3, 6, 5)]
        Case {
            equation: "bhqd,bhkd->bhqk",
            inputs: vec![
                (
                    vec![
                        0.5, -0.25, 1.25, -1.5, 1.75, -0.5, -1.5, 0.75, 1.0, 1.5, 0.5, 0.75, 0.75,
                        -1.75, 2.0, 1.0, 0.75, 0.25, 0.75, 1.25, 1.0, -1.0, -1.25, 0.0, -0.5, 0.0,
                        1.0, -0.25, 1.25, -1.5, -2.0, 0.75, 1.0, 0.25, 1.5, 2.0, 1.25, 0.75, 0.75,
                        -0.5, -0.5, -1.25, -1.5, -1.0, -1.0, -0.5, -1.0, -0.75, 0.5, -0.5, 0.5,
                        -2.0, 1.25, 1.0, 1.75, 1.25, 0.25, -1.0, -1.25, 1.0, -1.0, 0.25, 1.75, 1.5,
                        -1.0, -1.0, 0.75, 0.0, 1.25, -0.25, 1.75, 2.0, -1.0, -2.0, -1.75, 0.25,
                        1.5, 1.5, -1.5, -0.25, 2.0, -1.5, -0.5, -1.0, 0.5, -1.5, -1.0, 0.25, 0.5,
                        -1.25, 2.0, 1.25, 2.0, -2.0, 2.0, 1.75, 0.25, -1.75, -0.75, 1.0, 1.0, 1.25,
                        0.25, -0.75, -0.5, 0.75, 0.5, 1.75, -1.5, -1.25, -0.25, -0.5, -0.75, 1.25,
                        1.75, -0.75, 0.25, -0.5, -0.75, 2.0,
                    ],
                    vec![2, 3, 4, 5],
                ),
                (
                    vec![
                        -0.5, -0.5, -1.5, -1.75, -0.25, 1.5, -1.25, 0.25, 1.25, -2.0, 1.25, 0.5,
                        1.5, -0.25, 0.25, -1.75, 1.0, 2.0, -0.25, 1.5, 1.0, 0.0, 1.25, -1.75, 0.0,
                        2.0, 1.0, -0.5, 1.75, -1.0, 1.75, -1.0, 1.0, -2.0, -0.75, -1.0, -1.75,
                        -0.25, 1.0, -1.0, 0.25, -1.0, 0.0, 0.25, -1.75, -2.0, -1.75, 0.0, 2.0,
                        -0.25, -0.75, -1.0, -1.25, -1.5, -1.25, -0.75, -0.5, 1.0, 0.5, -1.75, -2.0,
                        1.0, 0.0, -2.0, -1.25, 1.0, 1.5, -2.0, -0.5, 2.0, 0.25, 2.0, -1.75, -2.0,
                        -2.0, 0.5, 1.0, 1.25, -2.0, 0.25, 0.25, 2.0, -1.0, -0.25, 0.75, 1.5, -1.75,
                        1.0, 1.5, 2.0, -1.0, 0.25, -1.75, 1.5, 2.0, -0.5, -1.5, -1.25, 0.0, -1.75,
                        -1.25, 2.0, 1.0, 0.0, 2.0, 1.0, 1.75, 1.5, 0.75, -0.75, 1.0, 0.75, -0.5,
                        -2.0, 1.0, 1.5, -2.0, 0.75, -2.0, -0.25, 0.75, -0.25, 0.5, 1.25, 1.75, 1.0,
                        -2.0, -1.75, -2.0, 2.0, -1.25, -1.0, -1.75, -1.0, 0.5, 1.5, -2.0, 1.5,
                        1.25, -0.25, 0.75, -2.0, 0.0, 0.25, -1.75, -1.75, -0.5, -0.5, 1.5, -0.75,
                        2.0, -0.75, 0.5, 1.75, 0.0, -1.0, 1.0, -0.25, 0.0, -0.75, 1.0, -1.5, 0.0,
                        -1.25, 1.0, -2.0, 1.25, -1.75, 1.0, -1.5, 1.5, 1.75, -1.0, 0.5, -1.25, 0.5,
                        -0.5, -0.75, 0.5, 2.0,
                    ],
                    vec![2, 3, 6, 5],
                ),
            ],
            expected: vec![
                0.1875, -4.0, 3.1875, 4.375, 4.6875, -4.25, -2.25, -0.4375, -0.125, 2.875, -1.3125,
                -2.625, 0.8125, -6.1875, 3.0625, 4.8125, 4.5, -3.6875, -2.875, -0.9375, 2.125,
                1.1875, 0.0, 2.6875, 1.875, 1.5625, 2.125, -0.125, 2.4375, -0.625, -2.625, 1.0625,
                1.9375, 1.125, -0.6875, 2.5, -4.875, -0.8125, -3.8125, 2.8125, -2.75, -0.375,
                1.875, -3.125, 0.3125, -4.5625, -4.1875, -0.125, 3.0, -0.875, 4.0, -1.625, -1.625,
                -3.5625, -0.375, -1.75, -0.8125, -3.3125, -1.875, 0.0, -7.1875, -2.0, -11.5625,
                -1.75, -4.0625, 10.5, -1.0, 6.25, 3.0625, 2.375, 2.875, 0.5625, -1.75, -0.3125,
                1.5, 3.9375, -5.6875, -3.4375, 2.5625, -0.1875, 2.25, 1.4375, -3.1875, -5.4375,
                -6.0, 0.4375, -2.6875, 3.5625, 6.0, 2.3125, -5.25, -3.8125, 3.6875, 4.1875, 3.375,
                1.5625, 1.25, 8.875, 1.125, 3.875, 3.375, -4.375, -2.3125, -3.4375, 1.3125, 1.0625,
                3.1875, 4.6875, 3.1875, 4.0, -4.25, 0.5, -5.0, -9.625, 1.1875, 7.8125, 1.875,
                -1.6875, -1.125, -4.1875, -0.125, 0.5625, -0.4375, -0.875, 3.6875, -1.6875, -0.625,
                0.25, 0.625, -3.5625, 1.0625, -4.4375, 1.6875, -1.375, 0.6875, -0.1875, -2.0625,
                4.8125, -3.25, -0.375, 1.8125, -1.0625, -3.0625, 3.5,
            ],
            shape: vec![2, 3, 4, 6],
        },
        // attention_values: numpy.einsum("bhqk,bhkd->bhqd") on shapes [(2, 3, 4, 6), (2, 3, 6, 5)]
        Case {
            equation: "bhqk,bhkd->bhqd",
            inputs: vec![
                (
                    vec![
                        -1.75, -0.5, -1.0, 1.5, 0.5, -0.5, -0.5, -2.0, -0.25, -1.75, 0.75, 0.5,
                        -1.75, -0.5, 0.0, -1.5, -1.5, -1.0, -0.25, -1.5, -0.25, -0.75, -1.0, 0.0,
                        -1.5, 0.5, -0.25, -1.75, -2.0, -0.25, -0.5, -1.0, -0.75, 0.75, 1.25, 1.0,
                        -2.0, 0.75, -1.25, 0.0, 1.75, -1.75, 0.5, 0.5, 0.0, -0.25, 0.75, 1.25,
                        -0.25, 0.0, -0.5, 1.5, -2.0, -0.5, 0.25, 2.0, -0.75, -1.5, -0.25, 1.75,
                        -0.75, -0.75, -1.0, 2.0, 0.25, -1.25, 2.0, 0.5, 1.25, -2.0, 1.5, 1.0, 0.5,
                        1.25, 1.75, -0.5, -2.0, 2.0, 0.75, 2.0, 2.0, -2.0, -1.5, -1.25, -0.5, 0.75,
                        -2.0, -0.75, 1.5, -1.75, 0.5, -1.5, -2.0, 0.25, 1.0, -0.25, 2.0, -1.25,
                        0.25, -1.0, -2.0, -0.75, -0.75, -1.5, -0.75, 2.0, -2.0, 0.75, -2.0, -1.0,
                        1.5, 0.5, -0.75, 2.0, -1.25, -1.75, -0.25, 1.5, -0.75, -1.25, -2.0, 0.75,
                        1.0, -0.5, 1.5, -0.75, -0.5, -0.75, 0.5, 1.75, 0.25, -1.0, 0.0, 1.75, 1.0,
                        1.75, -0.75, 1.5, -1.5, -0.25, 1.5, 1.75, 1.5, -1.75,
                    ],
                    vec![2, 3, 4, 6],
                ),
                (
                    vec![
                        1.75, 0.25, -0.75, 1.25, -1.0, -1.5, -1.75, -0.5, -1.0, 1.0, -1.0, 1.75,
                        1.75, -1.25, 1.25, -0.25, 0.0, -0.25, 0.25, 1.75, 1.25, 1.75, -1.25, 2.0,
                        -1.75, -1.25, 0.5, 1.0, -2.0, -0.25, -1.5, 0.75, 1.25, -0.25, -0.25, -0.5,
                        -1.5, 1.5, -0.25, -2.0, -0.25, 0.5, -1.5, 1.75, -1.5, -1.5, 0.0, 1.75,
                        -0.75, 1.5, 1.5, 1.0, -2.0, -1.25, -1.5, -0.5, 1.0, -2.0, 1.5, 0.75, 1.25,
                        -0.25, -0.25, -0.75, -2.0, -0.25, 0.5, -1.5, 1.25, 1.75, 2.0, -0.25, -1.0,
                        1.5, 1.25, 0.0, -0.75, 0.75, -1.75, -0.25, -1.0, -1.75, -1.5, 2.0, -0.5,
                        0.5, 0.0, -1.0, 1.25, -1.25, 0.0, -2.0, -1.25, 1.25, 2.0, 0.75, 1.75, 0.0,
                        -1.0, -2.0, 1.0, -1.5, -2.0, 0.25, 1.75, 0.75, 0.75, -0.75, -0.25, 1.0,
                        -2.0, 1.25, -1.5, -1.5, -0.25, 0.0, 0.5, -0.75, 1.5, -1.75, -1.0, -1.25,
                        -1.0, 1.75, 2.0, 0.25, -2.0, -0.75, 1.25, 0.0, -1.25, 1.0, -1.5, -1.25,
                        0.0, -1.0, -1.25, -1.25, -1.0, -0.75, -1.5, 0.75, -0.25, 1.5, -1.75, 1.25,
                        0.75, -0.75, 1.25, 0.5, -2.0, 0.25, 2.0, -1.25, 1.0, -1.75, -0.75, 1.0,
                        -1.75, -2.0, -2.0, -1.5, -0.75, 1.25, 0.5, -1.25, -0.75, 0.75, 0.75, 0.25,
                        2.0, 0.25, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5, -0.75, -1.75,
                    ],
                    vec![2, 3, 6, 5],
                ),
            ],
            expected: vec![
                -0.4375, -0.6875, -1.6875, 1.9375, 1.875, 3.125, 4.5, 0.9375, 1.75, -6.3125,
                -2.5625, -2.6875, 2.8125, -3.0625, 1.5, 1.0, 0.375, 1.9375, -0.6875, -1.125,
                1.8125, -4.25, 0.6875, 3.25, -0.0625, 1.6875, 3.0, -4.1875, -1.5625, 3.25, 6.4375,
                -3.25, 0.5, -6.6875, -3.0625, -0.125, 1.625, -3.0625, 0.875, -1.6875, 0.4375,
                2.5625, 5.1875, -7.8125, 1.125, -0.5625, 2.6875, -4.8125, 5.5, 0.375, -3.625,
                -1.875, 4.6875, -6.4375, -0.125, 3.875, -1.6875, -7.25, 8.75, -3.0625, 6.3125,
                -3.3125, -2.25, 5.9375, -1.9375, 5.0, -5.0, -0.25, 0.3125, 1.5625, -5.0, 5.75,
                4.25, -6.5625, -4.0625, -4.9375, 0.6875, 1.875, -0.3125, 0.9375, 0.4375, -0.5625,
                0.875, -1.3125, 7.875, 3.25, -0.25, 0.4375, -6.3125, 0.875, 3.0, 6.3125, -1.4375,
                -5.75, -2.0625, -0.8125, 1.4375, 2.1875, -8.25, -2.9375, 4.3125, -1.8125, -3.25,
                1.125, -1.8125, -0.375, -1.5625, -1.0625, 4.375, 3.4375, -8.75, -4.3125, 2.3125,
                -0.875, -5.1875, 1.25, -3.375, -2.4375, 5.3125, 3.25,
            ],
            shape: vec![2, 3, 4, 5],
        },
        // chain_three: numpy.einsum("ij,jk,kl->il") on shapes [(2, 3), (3, 4), (4, 2)]
        Case {
            equation: "ij,jk,kl->il",
            inputs: vec![
                (vec![-1.5, -0.75, 1.25, 1.5, -1.25, 0.0], vec![2, 3]),
                (
                    vec![
                        0.75, 1.75, -0.25, -1.75, 0.75, -1.0, -1.0, 0.5, 1.0, -1.0, 0.5, 2.0,
                    ],
                    vec![3, 4],
                ),
                (
                    vec![0.25, -1.25, 1.0, -1.25, -0.5, -0.75, 0.5, -0.75],
                    vec![4, 2],
                ),
            ],
            expected: vec![-1.73438, -0.421875, 1.85938, -3.29688],
            shape: vec![2, 2],
        },
        // outer_three: numpy.einsum("i,j,k->ijk") on shapes [(2,), (3,), (2,)]
        Case {
            equation: "i,j,k->ijk",
            inputs: vec![
                (vec![-0.75, 0.5], vec![2]),
                (vec![0.0, 1.25, -1.25], vec![3]),
                (vec![0.5, -1.5], vec![2]),
            ],
            expected: vec![
                0.0, 0.0, -0.46875, 1.40625, 0.46875, -1.40625, 0.0, 0.0, 0.3125, -0.9375, -0.3125,
                0.9375,
            ],
            shape: vec![2, 3, 2],
        },
        // hadamard: numpy.einsum("ij,ij->ij") on shapes [(3, 4), (3, 4)]
        Case {
            equation: "ij,ij->ij",
            inputs: vec![
                (
                    vec![
                        -1.0, 0.0, 0.75, -2.0, 1.75, 0.75, -1.75, -0.75, 0.5, -0.75, 1.5, 1.25,
                    ],
                    vec![3, 4],
                ),
                (
                    vec![
                        1.25, 0.0, 1.25, -1.0, 1.75, 0.25, 0.75, -0.5, 1.75, -0.75, 1.75, -0.75,
                    ],
                    vec![3, 4],
                ),
            ],
            expected: vec![
                -1.25, 0.0, 0.9375, 2.0, 3.0625, 0.1875, -1.3125, 0.375, 0.875, 0.5625, 2.625,
                -0.9375,
            ],
            shape: vec![3, 4],
        },
        // elementwise_vec: numpy.einsum("i,i->i") on shapes [(5,), (5,)]
        Case {
            equation: "i,i->i",
            inputs: vec![
                (vec![1.75, 0.25, 2.0, 1.0, -0.25], vec![5]),
                (vec![0.0, 0.5, -1.5, -0.5, 0.25], vec![5]),
            ],
            expected: vec![0.0, 0.125, -3.0, -0.5, -0.0625],
            shape: vec![5],
        },
        // sum_all: numpy.einsum("ij->") on shapes [(3, 4)]
        Case {
            equation: "ij->",
            inputs: vec![(
                vec![
                    0.0, -1.25, 0.0, 2.0, 0.75, -1.5, 1.5, 0.75, -0.25, -1.0, 1.5, 1.75,
                ],
                vec![3, 4],
            )],
            expected: vec![4.25],
            shape: vec![],
        },
        // scalar_operand: numpy.einsum(",ij->ij") on shapes [(), (2, 3)]
        Case {
            equation: ",ij->ij",
            inputs: vec![
                (vec![2.0], vec![]),
                (vec![-1.5, -1.25, 0.0, -1.75, 1.75, 0.25], vec![2, 3]),
            ],
            expected: vec![-3.0, -2.5, 0.0, -3.5, 3.5, 0.5],
            shape: vec![2, 3],
        },
        // transpose_only: numpy.einsum("ij->ji") on shapes [(3, 5)]
        Case {
            equation: "ij->ji",
            inputs: vec![(
                vec![
                    -1.5, -2.0, 1.25, 0.0, 1.75, -1.5, 0.5, -0.5, -0.75, 1.75, 0.75, -1.25, -0.25,
                    1.25, -0.5,
                ],
                vec![3, 5],
            )],
            expected: vec![
                -1.5, -1.5, 0.75, -2.0, 0.5, -1.25, 1.25, -0.5, -0.25, 0.0, -0.75, 1.25, 1.75,
                1.75, -0.5,
            ],
            shape: vec![5, 3],
        },
        // bilinear: numpy.einsum("bi,ij,bj->b") on shapes [(4, 3), (3, 5), (4, 5)]
        Case {
            equation: "bi,ij,bj->b",
            inputs: vec![
                (
                    vec![
                        -0.25, 1.0, -1.25, -0.75, -1.75, -1.25, 0.5, 0.0, -0.25, -0.5, 0.75, -2.0,
                    ],
                    vec![4, 3],
                ),
                (
                    vec![
                        -1.5, -1.0, -0.5, -1.0, 0.25, 2.0, -2.0, 0.5, 1.75, -0.25, -0.75, 1.25,
                        1.25, 1.0, 0.5,
                    ],
                    vec![3, 5],
                ),
                (
                    vec![
                        0.25, -1.0, 0.25, -1.0, 2.0, -1.0, -0.75, 1.25, 0.25, -1.5, 1.0, -1.5, 1.0,
                        0.5, -0.25, 2.0, 0.75, -0.25, -0.5, 1.25,
                    ],
                    vec![4, 5],
                ),
            ],
            expected: vec![1.28125, -3.48438, -0.28125, 3.79688],
            shape: vec![4],
        },
        // reduce_then_contract: numpy.einsum("ijk,kl->il") on shapes [(2, 3, 4), (4, 5)]
        Case {
            equation: "ijk,kl->il",
            inputs: vec![
                (
                    vec![
                        -1.25, 0.0, 1.0, -1.25, 0.25, 0.75, -0.25, -0.25, -2.0, -1.5, -0.25, 0.0,
                        0.5, 0.75, -0.25, 1.0, 2.0, 1.5, -1.5, 1.25, 2.0, -1.25, -2.0, -2.0,
                    ],
                    vec![2, 3, 4],
                ),
                (
                    vec![
                        -1.75, -1.75, -0.5, 0.25, -0.25, -1.0, -1.75, 0.25, 0.75, 2.0, -1.5, 0.0,
                        1.5, 0.0, 2.0, 0.0, 0.0, -0.25, -0.5, -2.0,
                    ],
                    vec![4, 5],
                ),
            ],
            expected: vec![
                5.25, 6.5625, 2.4375, -0.5625, 3.25, -3.25, -9.625, -7.6875, 1.75, -7.125,
            ],
            shape: vec![2, 5],
        },
        // batch_gemv: numpy.einsum("bij,bj->bi") on shapes [(3, 4, 5), (3, 5)]
        Case {
            equation: "bij,bj->bi",
            inputs: vec![
                (
                    vec![
                        2.0, -1.0, -1.5, 1.25, 1.75, -0.5, -1.75, 1.75, 0.5, -0.75, -0.5, -0.5,
                        -1.5, -0.5, -0.75, 2.0, 1.75, 0.25, -1.75, -0.5, -1.25, -0.75, -0.5, -1.75,
                        1.5, 1.75, 0.75, 1.0, 2.0, 0.25, -1.75, 1.25, 0.25, -1.5, 1.0, 1.5, -0.75,
                        -1.0, 0.75, 1.0, -1.25, 1.0, -1.5, 1.75, -0.25, -2.0, -1.5, -0.25, 2.0,
                        -2.0, 1.25, 1.75, 1.75, 0.25, -1.5, -0.25, -2.0, -0.5, 1.25, -0.75,
                    ],
                    vec![3, 4, 5],
                ),
                (
                    vec![
                        -1.0, -1.5, -0.5, -1.0, -0.25, 0.0, 1.5, 1.0, -1.0, 1.75, 2.0, -0.75, 0.75,
                        -1.5, -1.0,
                    ],
                    vec![3, 5],
                ),
            ],
            expected: vec![
                -1.4375, 1.9375, 2.6875, -2.875, 2.75, 0.5625, 5.375, -1.125, -6.75, -4.0625,
                3.625, -0.5,
            ],
            shape: vec![3, 4],
        },
        // repeat_and_broadcast: numpy.einsum("...ii->...i") on shapes [(2, 3, 3)]
        Case {
            equation: "...ii->...i",
            inputs: vec![(
                vec![
                    1.25, -0.25, -0.25, 1.25, 0.75, -0.75, 1.0, -2.0, 0.5, 0.5, 1.75, -1.0, 1.25,
                    0.25, -1.75, 1.25, -1.5, -2.0,
                ],
                vec![2, 3, 3],
            )],
            expected: vec![1.25, 0.75, 0.5, 0.5, 0.25, -2.0],
            shape: vec![2, 3],
        },
    ]
}

// ── numpy reference checks ──────────────────────────────────────────────────

#[test]
fn numpy_reference_auto_strategy() {
    for case in numpy_cases() {
        let tensors = case.tensors();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let out = einsum(case.equation, &refs)
            .unwrap_or_else(|e| panic!("einsum('{}') failed: {e}", case.equation));
        assert_close(&out, &case.expected, &case.shape, case.equation);
    }
}

#[test]
fn numpy_reference_general_executor() {
    for case in numpy_cases() {
        let tensors = case.tensors();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let out = einsum_general(case.equation, &refs)
            .unwrap_or_else(|e| panic!("general('{}') failed: {e}", case.equation));
        assert_close(&out, &case.expected, &case.shape, case.equation);
    }
}

#[test]
fn numpy_reference_pairwise_executor() {
    for case in numpy_cases() {
        let tensors = case.tensors();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let out = einsum_pairwise(case.equation, &refs)
            .unwrap_or_else(|e| panic!("pairwise('{}') failed: {e}", case.equation));
        assert_close(&out, &case.expected, &case.shape, case.equation);
    }
}

/// Differential check: the scalar interpreter and the GEMM lowering must agree
/// on every case, which is what validates the batch/M/K/N bucketing.
#[test]
fn general_and_pairwise_executors_agree() {
    for case in numpy_cases() {
        let tensors = case.tensors();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let general = einsum_general(case.equation, &refs)
            .unwrap_or_else(|e| panic!("general('{}') failed: {e}", case.equation));
        let pairwise = einsum_pairwise(case.equation, &refs)
            .unwrap_or_else(|e| panic!("pairwise('{}') failed: {e}", case.equation));
        assert_close(&pairwise, &general.data, &general.shape, case.equation);
    }
}

// ── plan-level checks: the GEMM path must actually be taken ─────────────────

/// Resolve `equation` against zero-filled tensors of `shapes` and return the
/// planned contraction steps.
fn steps_for(equation: &str, shapes: &[Vec<usize>]) -> Vec<super::contract::Step> {
    let tensors: Vec<Tensor> = shapes.iter().map(|s| Tensor::zeros(s)).collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let plan = parse_equation(equation, &refs).expect("parse failed");
    let labels: Vec<Vec<usize>> = plan
        .input_subscripts
        .iter()
        .map(|subs| {
            let mut out: Vec<usize> = Vec::new();
            for &l in subs {
                if !out.contains(&l) {
                    out.push(l);
                }
            }
            out
        })
        .collect();
    plan_contraction(&labels, &plan.output_subscript, &plan.label_sizes)
        .expect("planning failed")
        .steps
}

#[test]
fn gemm_path_is_selected_for_attention() {
    // Q·Kᵀ scores: batch = b·h, m = q, k = d, n = k.
    let steps = steps_for(
        "bhqd,bhkd->bhqk",
        &[vec![2, 12, 128, 64], vec![2, 12, 96, 64]],
    );
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Gemm);
    assert_eq!(
        (steps[0].batch, steps[0].m, steps[0].k, steps[0].n),
        (24, 128, 64, 96)
    );

    // scores·V: batch = b·h, m = q, k = k, n = d.
    let steps = steps_for(
        "bhqk,bhkd->bhqd",
        &[vec![2, 12, 128, 96], vec![2, 12, 96, 64]],
    );
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Gemm);
    assert_eq!(
        (steps[0].batch, steps[0].m, steps[0].k, steps[0].n),
        (24, 128, 96, 64)
    );

    // Ellipsis form of the same contraction resolves to the same GEMM.
    let steps = steps_for(
        "...qd,...kd->...qk",
        &[vec![2, 12, 128, 64], vec![2, 12, 96, 64]],
    );
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Gemm);
    assert_eq!(
        (steps[0].batch, steps[0].m, steps[0].k, steps[0].n),
        (24, 128, 64, 96)
    );
}

#[test]
fn public_entry_point_routes_attention_to_the_gemm_path() {
    let tensors = [
        Tensor::zeros(&[1, 4, 32, 16]),
        Tensor::zeros(&[1, 4, 32, 16]),
    ];
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let plan = parse_equation("bhqd,bhkd->bhqk", &refs).expect("parse failed");
    assert!(
        super::contract::general_path_flops(&plan) > super::contract::GENERAL_PATH_FLOP_LIMIT,
        "attention-shaped contractions must exceed the general-path limit so \
         einsum() dispatches to the GEMM lowering"
    );
}

#[test]
fn outer_product_uses_a_k_equals_one_gemm() {
    let steps = steps_for("ip,jq->ipjq", &[vec![64, 8], vec![32, 4]]);
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Gemm);
    assert_eq!(steps[0].k, 1, "no shared labels means an outer product");
    assert_eq!((steps[0].batch, steps[0].m, steps[0].n), (1, 512, 128));
}

#[test]
fn batched_dot_avoids_a_gemm_call_per_element() {
    let steps = steps_for("bhqd,bhqd->bhq", &[vec![2, 4, 8, 16], vec![2, 4, 8, 16]]);
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Dot);
    assert_eq!(
        (steps[0].batch, steps[0].m, steps[0].k, steps[0].n),
        (64, 1, 16, 1)
    );
}

#[test]
fn three_operand_chain_is_contracted_pairwise() {
    // The greedy heuristic must pick two matmuls, not one giant sweep. The
    // alternative — contracting the two outer operands first — would build a
    // 64×8×4×64 intermediate.
    let steps = steps_for("ij,jk,kl->il", &[vec![64, 8], vec![8, 4], vec![4, 64]]);
    assert_eq!(steps.len(), 2);
    for step in &steps {
        assert_eq!(step.kind, StepKind::Gemm);
    }
    assert_eq!((steps[0].lhs, steps[0].rhs), (0, 1));
    assert_eq!((steps[0].m, steps[0].k, steps[0].n), (64, 8, 4));
    assert_eq!((steps[1].m, steps[1].k, steps[1].n), (64, 4, 64));
}

#[test]
fn contraction_order_follows_the_cost_heuristic_not_operand_order() {
    // `ij,jk,kl->il` with i=j=64, k=l=2. Both left-to-right and right-to-left
    // give a 128-element intermediate, so the flop tie-break decides: pairing
    // (jk,kl) costs 256 multiply-accumulates, pairing (ij,jk) costs 8192.
    let steps = steps_for("ij,jk,kl->il", &[vec![64, 64], vec![64, 2], vec![2, 2]]);
    assert_eq!(steps.len(), 2);
    assert_eq!(
        (steps[0].lhs, steps[0].rhs),
        (1, 2),
        "the two right-hand operands must be contracted first"
    );
    assert_eq!((steps[0].m, steps[0].k, steps[0].n), (64, 2, 2));
    assert_eq!((steps[1].m, steps[1].k, steps[1].n), (64, 64, 2));
}

#[test]
fn single_operand_equations_need_no_contraction_steps() {
    assert!(steps_for("ij->ji", &[vec![3, 4]]).is_empty());
    assert!(steps_for("ii->i", &[vec![4, 4]]).is_empty());
}

// ── legacy scalar oracle + timing note ──────────────────────────────────────

/// The pre-W2 implementation, kept verbatim as an independent oracle and as the
/// "before" number for the GEMM-lowering timing note. It supports neither
/// ellipsis nor broadcasting, so it is only applied to plain equations.
mod legacy {
    use super::{HashMap, Tensor};

    struct LegacyPlan {
        input_subscripts: Vec<Vec<usize>>,
        output_subscript: Vec<usize>,
        label_sizes: Vec<usize>,
        num_labels: usize,
    }

    fn parse(equation: &str, inputs: &[&Tensor]) -> Result<LegacyPlan, String> {
        let eq = equation.replace(' ', "");
        let (lhs, rhs) = if let Some(pos) = eq.find("->") {
            (&eq[..pos], Some(eq[pos + 2..].to_string()))
        } else {
            (eq.as_str(), None)
        };
        let input_strs: Vec<&str> = lhs.split(',').collect();
        if input_strs.len() != inputs.len() {
            return Err("legacy einsum: arity".to_string());
        }
        let mut label_map: HashMap<char, usize> = HashMap::new();
        let mut label_count = 0;
        let mut input_subscripts: Vec<Vec<usize>> = Vec::new();
        for (i, s) in input_strs.iter().enumerate() {
            let chars: Vec<char> = s.chars().collect();
            if chars.len() != inputs[i].ndim() {
                return Err("legacy einsum: rank".to_string());
            }
            let mut subs = Vec::new();
            for &c in &chars {
                let idx = *label_map.entry(c).or_insert_with(|| {
                    let v = label_count;
                    label_count += 1;
                    v
                });
                subs.push(idx);
            }
            input_subscripts.push(subs);
        }
        let mut label_sizes = vec![0usize; label_count];
        for (i, subs) in input_subscripts.iter().enumerate() {
            for (j, &label) in subs.iter().enumerate() {
                let dim = inputs[i].shape[j];
                if label_sizes[label] == 0 {
                    label_sizes[label] = dim;
                } else if label_sizes[label] != dim {
                    return Err("legacy einsum: dim mismatch".to_string());
                }
            }
        }
        let output_subscript = if let Some(ref rhs_str) = rhs {
            rhs_str
                .chars()
                .map(|c| {
                    label_map
                        .get(&c)
                        .copied()
                        .ok_or_else(|| "legacy einsum: output label".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut counts = vec![0usize; label_count];
            for subs in &input_subscripts {
                for &l in subs {
                    counts[l] += 1;
                }
            }
            let mut pairs: Vec<(char, usize)> = label_map.iter().map(|(&c, &l)| (c, l)).collect();
            pairs.sort_by_key(|&(c, _)| c);
            pairs
                .into_iter()
                .filter(|&(_, l)| counts[l] == 1)
                .map(|(_, l)| l)
                .collect()
        };
        Ok(LegacyPlan {
            input_subscripts,
            output_subscript,
            label_sizes,
            num_labels: label_count,
        })
    }

    fn execute(plan: &LegacyPlan, inputs: &[&Tensor]) -> Result<Tensor, String> {
        let out_shape: Vec<usize> = plan
            .output_subscript
            .iter()
            .map(|&l| plan.label_sizes[l])
            .collect();
        let out_numel: usize = if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().product()
        };
        let mut out_data = vec![0.0f32; out_numel];
        let out_set: std::collections::HashSet<usize> =
            plan.output_subscript.iter().copied().collect();
        let contracted: Vec<usize> = (0..plan.num_labels)
            .filter(|l| !out_set.contains(l))
            .collect();
        let contracted_sizes: Vec<usize> =
            contracted.iter().map(|&l| plan.label_sizes[l]).collect();
        let contracted_total: usize = if contracted_sizes.is_empty() {
            1
        } else {
            contracted_sizes.iter().product()
        };
        let input_strides: Vec<Vec<usize>> = inputs
            .iter()
            .map(|t| {
                let ndim = t.ndim();
                let mut strides = vec![1usize; ndim];
                for i in (0..ndim.saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * t.shape[i + 1];
                }
                strides
            })
            .collect();
        let out_strides = {
            let len = out_shape.len();
            let mut s = vec![1usize; len];
            for i in (0..len.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * out_shape[i + 1];
            }
            s
        };
        for (out_flat, out_elem) in out_data.iter_mut().enumerate().take(out_numel) {
            let mut label_values = vec![0usize; plan.num_labels];
            let mut remaining = out_flat;
            for (i, &label) in plan.output_subscript.iter().enumerate() {
                let stride = out_strides[i];
                label_values[label] = remaining / stride;
                remaining %= stride;
            }
            let mut sum = 0.0f32;
            for c_flat in 0..contracted_total {
                let mut c_remaining = c_flat;
                for (ci, &label) in contracted.iter().enumerate() {
                    let stride: usize = if ci + 1 < contracted_sizes.len() {
                        contracted_sizes[ci + 1..].iter().product()
                    } else {
                        1
                    };
                    label_values[label] = c_remaining / stride;
                    c_remaining %= stride;
                }
                let mut product = 1.0f32;
                for (inp_idx, subs) in plan.input_subscripts.iter().enumerate() {
                    let mut flat = 0;
                    for (dim, &label) in subs.iter().enumerate() {
                        flat += label_values[label] * input_strides[inp_idx][dim];
                    }
                    product *= inputs[inp_idx].data[flat];
                }
                sum += product;
            }
            *out_elem = sum;
        }
        Ok(Tensor::new(out_data, out_shape))
    }

    pub(super) fn einsum(equation: &str, inputs: &[&Tensor]) -> Result<Tensor, String> {
        let plan = parse(equation, inputs)?;
        execute(&plan, inputs)
    }
}

/// Every non-ellipsis, non-broadcast case must still match the pre-W2 scalar
/// implementation, so the rewrite is a strict extension.
#[test]
fn matches_the_pre_rewrite_implementation() {
    let plain = [
        "ij,jk->ik",
        "ij->ji",
        "ii->i",
        "ii->",
        "ij->",
        "ij,jk",
        "i,j->ij",
        "i,i->",
        "bij,bjk->bik",
        "bhqd,bhkd->bhqk",
        "ijk,kl->il",
        "ij,jk,kl->il",
        "iij,jk->ik",
    ];
    for case in numpy_cases() {
        if !plain.contains(&case.equation) {
            continue;
        }
        let tensors = case.tensors();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let Ok(oracle) = legacy::einsum(case.equation, &refs) else {
            continue;
        };
        let out = einsum(case.equation, &refs)
            .unwrap_or_else(|e| panic!("einsum('{}') failed: {e}", case.equation));
        assert_close(&out, &oracle.data, &oracle.shape, case.equation);
    }
}

fn filled(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let mut state = seed | 1;
    let data = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) & 0xff) as f32 / 128.0 - 1.0
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

/// Timing note for [a3-12]: the same attention-score contraction through the
/// pre-W2 scalar loop and through the GEMM lowering. The shape is deliberately
/// modest (b=1, h=4, q=k=128, d=64 → 4.2M multiply-accumulates) so the scalar
/// path finishes in a reasonable time inside a unit test; the production shape
/// (h=12, q=k=512) is 48× larger and shows the same ratio.
///
/// The assertion is deliberately loose (2×) because the measured gap is two
/// orders of magnitude — it exists to catch a silent fall-back to the scalar
/// path, not to police a specific ratio.
#[test]
fn gemm_path_timing_note() {
    let q = filled(&[1, 4, 128, 64], 7);
    let k = filled(&[1, 4, 128, 64], 11);
    let inputs = [&q, &k];

    let start = std::time::Instant::now();
    let legacy_out = legacy::einsum("bhqd,bhkd->bhqk", &inputs).expect("legacy einsum failed");
    let legacy_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    let fast_out = einsum("bhqd,bhkd->bhqk", &inputs).expect("einsum failed");
    let fast_elapsed = start.elapsed();

    assert_close(
        &fast_out,
        &legacy_out.data,
        &legacy_out.shape,
        "bhqd,bhkd->bhqk",
    );
    println!(
        "einsum bhqd,bhkd->bhqk [1,4,128,64]: scalar {legacy_elapsed:?} -> gemm {fast_elapsed:?} \
         ({:.1}x)",
        legacy_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(1e-9)
    );
    assert!(
        fast_elapsed.as_nanos() * 2 < legacy_elapsed.as_nanos(),
        "GEMM lowering ({fast_elapsed:?}) should be far faster than the scalar loop \
         ({legacy_elapsed:?}) — a silent fall-back is the likely cause"
    );
}

/// The production-shaped version of [`gemm_path_timing_note`]: the attention
/// score contraction quoted in [a3-12] (b=1, h=12, q=k=512, d=64 → 201M
/// multiply-accumulates). Ignored by default because the scalar reference alone
/// takes seconds; run with
/// `cargo nextest run -p oxionnx-ops -E 'test(full_attention_timing_note)' --run-ignored all --no-capture`.
#[test]
#[ignore = "multi-second scalar reference; run explicitly for the timing note"]
fn full_attention_timing_note() {
    let q = filled(&[1, 12, 512, 64], 7);
    let k = filled(&[1, 12, 512, 64], 11);
    let inputs = [&q, &k];

    let start = std::time::Instant::now();
    let legacy_out = legacy::einsum("bhqd,bhkd->bhqk", &inputs).expect("legacy einsum failed");
    let legacy_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    let fast_out = einsum("bhqd,bhkd->bhqk", &inputs).expect("einsum failed");
    let fast_elapsed = start.elapsed();

    assert_eq!(fast_out.shape, vec![1, 12, 512, 512]);
    let worst = fast_out
        .data
        .iter()
        .zip(legacy_out.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "einsum bhqd,bhkd->bhqk [1,12,512,64]: scalar {legacy_elapsed:?} -> gemm {fast_elapsed:?} \
         ({:.1}x), max |Δ| = {worst:e}",
        legacy_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(1e-9)
    );
    assert!(worst <= TOL, "reassociation drift {worst} exceeds {TOL}");
}

// ── original regression tests ───────────────────────────────────────────────

#[test]
fn test_einsum_matmul() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let out = einsum("ij,jk->ik", &[&a, &b]).expect("einsum matmul failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert!((out.data[0] - 22.0).abs() < 1e-5);
    assert!((out.data[1] - 28.0).abs() < 1e-5);
}

#[test]
fn test_einsum_transpose() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = einsum("ij->ji", &[&a]).expect("einsum transpose failed");
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_einsum_trace() {
    let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
    let out = einsum("ii->", &[&a]).expect("einsum trace failed");
    assert_eq!(out.shape, Vec::<usize>::new());
    assert!((out.data[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_einsum_batch_matmul() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);
    let out = einsum("bij,bjk->bik", &[&a, &b]).expect("einsum batch matmul failed");
    assert_eq!(out.shape, vec![1, 2, 2]);
    assert_eq!(out.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_einsum_dot_product() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let out = einsum("i,i->", &[&a, &b]).expect("einsum dot product failed");
    assert!((out.data[0] - 32.0).abs() < 1e-5);
}

#[test]
fn test_einsum_outer_product() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2]);
    let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3]);
    let out = einsum("i,j->ij", &[&a, &b]).expect("einsum outer product failed");
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
}

#[test]
fn test_einsum_implicit_output() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let out = einsum("ij,jk", &[&a, &b]).expect("einsum implicit failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert!((out.data[0] - 22.0).abs() < 1e-5);
}

#[test]
fn test_einsum_input_count_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2]);
    assert!(einsum("ij,jk->ik", &[&a]).is_err());
}

#[test]
fn test_einsum_dim_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2]);
    assert!(einsum("ij->ji", &[&a]).is_err());
}

#[test]
fn test_einsum_sum_reduction() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = einsum("ij->", &[&a]).expect("einsum sum failed");
    assert_eq!(out.shape, Vec::<usize>::new());
    assert!((out.data[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_einsum_diagonal() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = einsum("ii->i", &[&a]).expect("einsum diagonal failed");
    assert_eq!(out.shape, vec![2]);
    assert_eq!(out.data, vec![1.0, 4.0]);
}

// ── whitespace, spec details ────────────────────────────────────────────────

#[test]
fn whitespace_in_the_equation_is_ignored() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let spaced = einsum(" ... i j , j k -> ... i k ", &[&a, &b]).expect("spaced equation failed");
    let plain = einsum("ij,jk->ik", &[&a, &b]).expect("plain equation failed");
    assert_eq!(spaced.shape, plain.shape);
    assert_eq!(spaced.data, plain.data);
}

#[test]
fn ellipsis_binding_zero_axes_is_a_no_op() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = einsum("...ij->...ji", &[&a]).expect("zero-width ellipsis failed");
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn diagonal_of_a_broadcast_batch() {
    // "...ii->...i" over a batch of 2 matrices, each 3x3.
    let a = filled(&[2, 3, 3], 3);
    let out = einsum("...ii->...i", &[&a]).expect("batched diagonal failed");
    assert_eq!(out.shape, vec![2, 3]);
    for b in 0..2 {
        for i in 0..3 {
            assert_eq!(out.data[b * 3 + i], a.data[b * 9 + i * 3 + i]);
        }
    }
}

/// A size-1 named label broadcasts against a wider one, as numpy does. This
/// exercises the *batch*-label broadcast (stride 0 on a surviving axis) rather
/// than the contracted-label case covered by the `named_broadcast` reference.
///
/// numpy: `einsum('ij,ij->ij', [[2],[-1.5]], [[1,2,3],[4,-1,.5]])`
///   → `[[2,4,6],[-6,1.5,-.75]]`; `einsum('...j,...j->...', …)` → `[12,-5.25]`.
#[test]
fn size_one_named_labels_broadcast_like_numpy() {
    let a = Tensor::new(vec![2.0, -1.5], vec![2, 1]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5], vec![2, 3]);

    let out = einsum("ij,ij->ij", &[&a, &b]).expect("broadcast hadamard failed");
    assert_eq!(out.shape, vec![2, 3]);
    assert_close(
        &out,
        &[2.0, 4.0, 6.0, -6.0, 1.5, -0.75],
        &[2, 3],
        "ij,ij->ij",
    );

    let out = einsum("...j,...j->...", &[&a, &b]).expect("broadcast dot failed");
    assert_close(&out, &[12.0, -5.25], &[2], "...j,...j->...");

    // Both executors must agree on the broadcast, not just the dispatched one.
    for equation in ["ij,ij->ij", "...j,...j->..."] {
        let general = einsum_general(equation, &[&a, &b]).expect("general failed");
        let pairwise = einsum_pairwise(equation, &[&a, &b]).expect("pairwise failed");
        assert_close(&pairwise, &general.data, &general.shape, equation);
    }
}

#[test]
fn zero_length_dimensions_produce_zeros_not_panics() {
    let a = Tensor::new(Vec::new(), vec![2, 0]);
    let b = Tensor::new(Vec::new(), vec![0, 3]);
    let out = einsum("ij,jk->ik", &[&a, &b]).expect("zero-k contraction failed");
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, vec![0.0; 6]);

    let c = Tensor::new(Vec::new(), vec![2, 0, 3]);
    let out = einsum("ijk->ik", &[&c]).expect("zero-axis reduction failed");
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, vec![0.0; 6]);

    let empty_out = einsum("ij,jk->ijk", &[&a, &b]).expect("empty output failed");
    assert_eq!(empty_out.shape, vec![2, 0, 3]);
    assert!(empty_out.data.is_empty());
}

#[test]
fn scalar_operands_are_accepted() {
    let s = Tensor::new(vec![2.0], Vec::new());
    let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = einsum(",ij->ij", &[&s, &m]).expect("scalar operand failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![2.0, 4.0, 6.0, 8.0]);
}

// ── malformed input: typed errors, never panics ─────────────────────────────

fn err_for(equation: &str, shapes: &[Vec<usize>]) -> String {
    let tensors: Vec<Tensor> = shapes.iter().map(|s| Tensor::zeros(s)).collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    match einsum(equation, &refs) {
        Ok(t) => panic!(
            "expected an error for '{equation}', got shape {:?}",
            t.shape
        ),
        Err(e) => e,
    }
}

#[test]
fn malformed_equations_are_rejected() {
    assert!(err_for("i-j->i", &[vec![3, 3]]).contains("invalid character"));
    assert!(err_for("i.j->i", &[vec![3, 3, 3]]).contains("not part of an ellipsis"));
    assert!(err_for("...i...->i", &[vec![3, 3]]).contains("more than one ellipsis"));
    assert!(err_for("ij->ii", &[vec![3, 3]]).contains("more than once"));
    assert!(err_for("ij->ik", &[vec![3, 3]]).contains("does not appear in any input"));
    assert!(err_for("...i->", &[vec![2, 3]]).contains("no '...' ellipsis provided"));
    assert!(err_for("ii->i", &[vec![1, 3]]).contains("don't match"));
    assert!(err_for("ij,jk->ik", &[vec![2, 3], vec![4, 5]]).contains("could not be broadcast"));
    assert!(err_for("ij,jk->ik", &[vec![2, 3]]).contains("2 inputs but got 1"));
    assert!(err_for("...ij->ij", &[vec![3]]).contains("leaving no axes"));
}

#[test]
fn no_inputs_is_an_error() {
    assert!(einsum("->", &[]).is_err());
}

#[test]
fn shape_whose_element_count_overflows_is_an_error_not_a_panic() {
    // Constructed by hand: `Tensor::new` only validates data/shape agreement in
    // debug builds, so a malformed model can hand an operator this pairing.
    let bogus = Tensor {
        data: vec![1.0, 2.0],
        shape: vec![usize::MAX, 4],
    };
    let err = einsum("ij->i", &[&bogus]).expect_err("overflowing shape must error");
    assert!(
        err.contains("overflows usize") || err.contains("elements but shape"),
        "unexpected error: {err}"
    );
}

#[test]
fn tensor_whose_data_disagrees_with_its_shape_is_an_error() {
    let bogus = Tensor {
        data: vec![1.0, 2.0],
        shape: vec![4, 4],
    };
    let err = einsum("ij,jk->ik", &[&bogus, &bogus]).expect_err("short buffer must error");
    assert!(
        err.contains("elements but shape"),
        "unexpected error: {err}"
    );
}
