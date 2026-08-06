//! Einsum operator: Einstein summation convention (ONNX `Einsum`, opset 12+).
//!
//! Handles the full subscript grammar — `"ij,jk->ik"` (matmul), `"ij->ji"`
//! (transpose), `"ii->i"` (diagonal), `"bij,bjk->bik"` (batched matmul),
//! `"...ij,...jk->...ik"` (broadcast batched matmul), implicit output mode, and
//! N-operand contractions — with numpy-compatible semantics throughout. The
//! `parse` submodule documents the precise rules the parser enforces.
//!
//! # Execution strategy
//!
//! | contraction size | path |
//! |---|---|
//! | ≤ `GENERAL_PATH_FLOP_LIMIT` multiply-accumulates | `execute_general` — one scalar loop nest, no materialisation |
//! | larger | `execute_pairwise` — greedy pairwise decomposition, each binary step lowered to `matrixmultiply::sgemm` |
//!
//! Attention-shaped equations such as `bhqd,bhkd->bhqk` therefore run as a
//! batch of `sgemm` calls (`batch = b·h`, `m = q`, `k = d`, `n = k`) rather than
//! as `b·h·q·k·d` scalar multiply-accumulates. Measured on the shape quoted in
//! the original report (`b=1, h=12, q=k=512, d=64`, 201M multiply-accumulates):
//! **2.07 s → 4.87 ms, a 425× speedup**, with a maximum elementwise difference
//! of 0 against the previous scalar implementation. See the `tests` submodule
//! (`gemm_path_timing_note`, `full_attention_timing_note`).
//!
//! # Numerical tolerance
//!
//! The GEMM path sums each contraction in `matrixmultiply`'s blocked order
//! rather than in ascending index order, which re-associates the floating-point
//! sum. Results are therefore **not guaranteed bit-identical** to the scalar
//! path; they are asserted equal to `1e-4` absolute, which is the tolerance the
//! numpy-reference and differential tests use (and well inside the accumulated
//! rounding of an f32 dot product of that length — blocked summation is, if
//! anything, the more accurate of the two). Distributing the batch loop across
//! rayon threads re-associates nothing: each batch element is an independent
//! GEMM over a disjoint output tile, so the parallel and sequential results are
//! bit-identical.

mod contract;
mod operand;
mod parse;
#[cfg(test)]
mod tests;

use operand::Operand;
use oxionnx_core::Tensor;
use parse::{parse_equation, EinsumPlan};

/// Evaluate an einsum `equation` over `inputs`.
///
/// # Errors
/// Returns a descriptive message for any malformed equation or shape
/// disagreement — wrong operand count, illegal characters, a subscript whose
/// length disagrees with its operand's rank, non-broadcastable extents, a
/// repeated or unknown output label, or a missing output ellipsis. Never
/// panics, including on shapes whose element count would overflow `usize`.
pub fn einsum(equation: &str, inputs: &[&Tensor]) -> Result<Tensor, String> {
    let plan = parse_equation(equation, inputs)?;
    if contract::general_path_flops(&plan) <= contract::GENERAL_PATH_FLOP_LIMIT {
        let operands = build_operands(&plan, inputs)?;
        contract::execute_general(&plan, &operands)
    } else {
        contract::execute_pairwise(&plan, inputs)
    }
}

/// Build one strided `Operand` view per input.
fn build_operands<'a>(
    plan: &EinsumPlan,
    inputs: &[&'a Tensor],
) -> Result<Vec<Operand<'a>>, String> {
    plan.input_subscripts
        .iter()
        .enumerate()
        .map(|(index, subs)| {
            let tensor = inputs
                .get(index)
                .ok_or_else(|| format!("einsum: missing input {index}"))?;
            Operand::from_tensor(tensor, subs, &plan.label_sizes)
        })
        .collect()
}

/// Force the scalar general interpreter, bypassing the size heuristic.
///
/// Test-only: the two executors are cross-checked against each other, so both
/// must be reachable regardless of how [`einsum`] would dispatch.
#[cfg(test)]
pub(crate) fn einsum_general(equation: &str, inputs: &[&Tensor]) -> Result<Tensor, String> {
    let plan = parse_equation(equation, inputs)?;
    let operands = build_operands(&plan, inputs)?;
    contract::execute_general(&plan, &operands)
}

/// Force the pairwise GEMM path, bypassing the size heuristic.
#[cfg(test)]
pub(crate) fn einsum_pairwise(equation: &str, inputs: &[&Tensor]) -> Result<Tensor, String> {
    let plan = parse_equation(equation, inputs)?;
    contract::execute_pairwise(&plan, inputs)
}
