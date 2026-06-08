mod argminmax;
mod broadcast;
mod matmul;
mod reduce;
mod topk;
mod unary;
mod variadic;

#[cfg(test)]
mod tests;

// ── Broadcast & elementwise arithmetic ──────────────────────────────────────
pub use broadcast::{add, broadcast_to, div, mul, neg, pow, reciprocal, sqrt, sub};

// ── Reductions ───────────────────────────────────────────────────────────────
pub use reduce::{
    reduce_l1, reduce_l2, reduce_log_sum, reduce_log_sum_exp, reduce_max, reduce_mean, reduce_min,
    reduce_prod, reduce_sum, reduce_sum_square,
};
pub(crate) use reduce::{
    reduce_l1_into, reduce_l2_into, reduce_log_sum_exp_into, reduce_log_sum_into, reduce_max_into,
    reduce_mean_into, reduce_min_into, reduce_output_shape, reduce_prod_into, reduce_sum_into,
    reduce_sum_square_into,
};

// ── ArgMax / ArgMin / CumSum / Range ─────────────────────────────────────────
pub use argminmax::{arg_max, arg_min, cumsum, range};
pub(crate) use argminmax::{arg_output_shape, arg_reduce_into, cumsum_into};

// ── TopK ─────────────────────────────────────────────────────────────────────
pub use topk::top_k;
pub(crate) use topk::{top_k_into, top_k_output_shape};

// ── MatMul / Gemm ─────────────────────────────────────────────────────────────
pub use matmul::{gemm, gemm_into, matmul, matmul_into};

// ── Unary element-wise: trig & rounding ──────────────────────────────────────
pub use unary::{
    acos_op, acosh_op, asin_op, asinh_op, atan_op, atanh_op, ceil, cos_op, cosh_op, floor_op,
    round_op, sign, sin_op, sinh_op, tan_op,
};

// ── Binary element-wise & variadic ───────────────────────────────────────────
pub use variadic::{bit_shift, mod_op, variadic_max, variadic_mean, variadic_min, variadic_sum};
