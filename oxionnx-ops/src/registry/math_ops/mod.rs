//! Operator trait implementations for math operations.
//!
//! Split into submodules:
//! - `macros`      — shared macro definitions (must be declared first with `#[macro_use]`)
//! - `elementwise` — binary/unary/trig elementwise ops
//! - `matmul_gemm` — MatMul and Gemm ops
//! - `reduce`      — reduction ops (ReduceMean, ArgMax, CumSum, Range, TopK, Mod, BitShift, variadic)

// IMPORTANT: `macros` must be declared before the modules that use its macros,
// because `macro_rules!` definitions in a sibling module are available only
// after the module declaration (in declaration order within the same file).
#[macro_use]
mod macros;

mod elementwise;
mod matmul_gemm;
mod reduce;

// Re-export all public op types, preserving the original public API.
pub use elementwise::{
    AcosOp, AcoshOp, AddOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, DivOp,
    FloorOp, MulOp, NegOp, PowOp, ReciprocalOp, RoundOp, SignOp, SinOp, SinhOp, SqrtOp, SubOp,
    TanOp,
};
pub use matmul_gemm::{GemmOp, MatMulOp};
pub use reduce::{
    ArgMaxOp, ArgMinOp, BitShiftOp, CumSumOp, ModOp, RangeOp, ReduceL1Op, ReduceL2Op,
    ReduceLogSumExpOp, ReduceLogSumOp, ReduceMaxOp, ReduceMeanOp, ReduceMinOp, ReduceProdOp,
    ReduceSumOp, ReduceSumSquareOp, TopKOp, VariadicMaxOp, VariadicMeanOp, VariadicMinOp,
    VariadicSumOp,
};
