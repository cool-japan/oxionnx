//! HLSL compute-shader sources for the D3D12 fallback backend.
//!
//! Platform-neutral: these are `&'static str`s, compiled into every build and
//! handed to `D3DCompile` only on Windows.
//!
//! # This is the least-verifiable code in the crate. Read this before editing.
//!
//! `rustc` sees a string.  A typo, a wrong register, a missing semicolon, a
//! transposed index — all invisible to the compiler, to clippy, and to every test
//! that can run on this machine.  The first thing that ever *parses* this text is
//! `D3DCompile`, at run time, on the user's Windows box.  There is no `fxc` and no
//! `dxc` on the Linux host this crate is developed on, so we cannot even shell out
//! to a validator.
//!
//! Two things therefore hold the line, and both are load-bearing:
//!
//! 1. **The shaders are kept trivially simple.**  No shared memory, no tiling
//!    cleverness, no wave intrinsics.  A naive `k`-loop is slower than it could be
//!    and is *reviewable by eye*, which is worth far more here.
//! 2. **`crate::reference` is their executable specification**, and
//!    `DirectMLContext::self_check` diffs the two on real hardware.  If you change
//!    a shader here, change the matching function in `reference.rs` in the same
//!    commit, or the self-check will (correctly) fail.
//!
//! # The two contracts these sources must satisfy
//!
//! * **`cbuffer` layout.**  Each `cbuffer` declares exactly
//!   [`crate::plan::ROOT_CONSTANT_COUNT`] `uint`s, in the same order as the
//!   `#[repr(C)]` fields of [`crate::plan::MatMulConstants`] /
//!   [`crate::plan::ElementwiseConstants`].  One root signature therefore serves
//!   every entry point.  Reordering a field on either side compiles cleanly and
//!   computes garbage.
//! * **Dispatch geometry.**  See each constant's doc comment.  Getting the MatMul
//!   grid backwards is the single easiest mistake in this crate to make, and it
//!   was already made once — see [`MATMUL_HLSL`].

/// Tile-based MatMul: `C = A · B`, one batch slice per dispatch.
///
/// * `[numthreads(16, 16, 1)]` — see [`crate::plan::MATMUL_TILE`].
/// * `cbuffer` order **must** match [`crate::plan::MatMulConstants`]:
///   `M, K, N, AOff, BOff, COff, _p0, _p1`.
///
/// # Dispatch geometry — the transposition trap
///
/// The shader reads `row = tid.y` and `col = tid.x`, so:
///
/// ```text
/// Dispatch( ceil(N / 16), ceil(M / 16), 1 )
///            └─ X = COLUMNS   └─ Y = ROWS
/// ```
///
/// The scaffold this crate grew out of documented the exact opposite
/// (`Dispatch(ceil(M/16), ceil(N/16), 1)`).  Following *that* comment on any
/// non-square matrix dispatches too few groups along one axis and leaves the rest
/// of the output buffer as whatever was there before — a wrong answer, not a
/// crash, and not caught by any shape-only test.
///
/// Do not compute this by hand.  Call [`crate::plan::MatMulPlan::hlsl_grid`],
/// which is unit-tested against exactly this orientation on Linux.
///
/// Batching is expressed entirely through `AOff` / `BOff` / `COff`: the backend
/// records `batch` dispatches with the *same* grid, varying only the root
/// constants.  A zero batch stride leaves an offset at 0, which is how a
/// batch-broadcast operand is read without copying anything.
pub const MATMUL_HLSL: &str = r"
cbuffer Constants : register(b0) {
    uint M; uint K; uint N; uint AOff; uint BOff; uint COff; uint _p0; uint _p1;
}
StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.y;
    uint col = tid.x;
    if (row >= M || col >= N) return;
    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        acc += A[AOff + row * K + k] * B[BOff + k * N + col];
    }
    C[COff + row * N + col] = acc;
}
";

/// Binary elementwise ops.  Entry points: `main_add`, `main_sub`, `main_mul`,
/// `main_div`.
///
/// * `[numthreads(256, 1, 1)]` — see [`crate::plan::ELEMENTWISE_THREADS_PER_GROUP`].
/// * `cbuffer` order **must** match [`crate::plan::ElementwiseConstants`]:
///   `N, GroupsX, _p0 … _p5`.
///
/// # Dispatch geometry
///
/// The grid is **2-D**, not 1-D.  A 1-D grid caps out at
/// `65535 × 256 = 16 776 960` elements — a 4 MiB f32 tensor, which is a perfectly
/// ordinary activation — and D3D12 rejects a larger `Dispatch` outright.  So the
/// linear group index is folded across two dimensions and recovered in the shader:
///
/// ```text
/// i = (gid.y * GroupsX + gid.x) * 256 + lid.x
/// ```
///
/// `GroupsX` **must** equal the `x` actually passed to `Dispatch`.
/// [`crate::plan::ElementwisePlan::constants`] takes it straight from
/// [`crate::plan::ElementwisePlan::hlsl_grid`] so the two cannot drift apart; a
/// mismatch reads the wrong elements *silently*.
///
/// The `if (i >= N) return;` guard is **not optional**: the folded grid
/// deliberately overshoots (`x * y * 256 >= N`), and the guard is what absorbs the
/// surplus threads.
///
/// # Broadcasting
///
/// There is none.  These kernels are pure index-parallel `C[i] = A[i] ⊕ B[i]`; they
/// have no notion of a shape.  [`crate::plan::ElementwisePlan::binary`] declines
/// every non-identical shape pair for exactly this reason — read that function's
/// documentation before "fixing" it.
pub const ELEMENTWISE_BINARY_HLSL: &str = r"
cbuffer Constants : register(b0) {
    uint N; uint GroupsX; uint _p0; uint _p1; uint _p2; uint _p3; uint _p4; uint _p5;
}
StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void main_add(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = A[i] + B[i];
}

[numthreads(256, 1, 1)]
void main_sub(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = A[i] - B[i];
}

[numthreads(256, 1, 1)]
void main_mul(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = A[i] * B[i];
}

[numthreads(256, 1, 1)]
void main_div(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = A[i] / B[i];
}
";

/// Unary elementwise activations.  Entry points: `main_relu`, `main_sigmoid`,
/// `main_tanh`.
///
/// Same `cbuffer` layout and same 2-D group-index recovery as
/// [`ELEMENTWISE_BINARY_HLSL`].
///
/// This source declares **no `t1`** — there is no second operand.  The root
/// signature still has one, because a single root signature serves every entry
/// point, and the D3D12 debug layer errors on an *unset* root parameter even when
/// the bound shader never reads it.  The backend therefore binds `A` to both `t0`
/// and `t1` on this path; the shader simply ignores `t1`.
///
/// `main_sigmoid` computes `1 / (1 + exp(-x))` — the direct form, not the
/// numerically-stable two-branch one.  That is deliberate: parity with
/// [`crate::reference`]'s oracle is the point of these kernels, and the direct form
/// saturates cleanly to 0 and 1 outside roughly ±88 without producing a NaN.  If
/// you change it here, change it there, in the same commit.
pub const ELEMENTWISE_UNARY_HLSL: &str = r"
cbuffer Constants : register(b0) {
    uint N; uint GroupsX; uint _p0; uint _p1; uint _p2; uint _p3; uint _p4; uint _p5;
}
StructuredBuffer<float>   A : register(t0);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void main_relu(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = max(0.0, A[i]);
}

[numthreads(256, 1, 1)]
void main_sigmoid(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = 1.0 / (1.0 + exp(-A[i]));
}

[numthreads(256, 1, 1)]
void main_tanh(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (i >= N) return;
    C[i] = tanh(A[i]);
}
";

/// Numerically-stable single-axis softmax.  Entry point: `main`.
///
/// * `[numthreads(256, 1, 1)]` — see [`crate::plan::REDUCTION_THREADS_PER_GROUP`].
/// * `cbuffer` order **must** match [`crate::plan::SoftmaxConstants`]:
///   `Rows, GroupsX, AxisLen, Inner, _p0 … _p3`.
///
/// # One thread per softmax row
///
/// [`crate::plan::SoftmaxPlan`] collapses the tensor to `outer × axis_len × inner`.
/// One thread owns one **row** `row ∈ [0, Rows)` (`Rows = outer · inner`) and walks
/// the `AxisLen` entries at `base + k · Inner`.  The 2-D group-index folding is
/// identical to the elementwise family, so the same `GroupsX` contract holds:
/// `GroupsX` **must** equal the `x` actually dispatched, and the `if (row >= Rows)`
/// guard absorbs the folded grid's deliberate overshoot.
///
/// # The stabilisation is not optional
///
/// The shader subtracts the row max before every `exp`, exactly as
/// [`crate::reference::ref_softmax`] does:
///
/// ```text
/// m      = max_k A[base + k·Inner]
/// sum    = Σ_k exp(A[base + k·Inner] − m)
/// C[…]   = exp(A[base + k·Inner] − m) / sum
/// ```
///
/// Without the `− m` a row containing a large positive value overflows `exp` to
/// `+inf` and the whole row becomes `NaN`.  The oracle uses the same form, so a
/// shader that dropped it would be *reported* by `compare` on any such row.  If you
/// change one, change the other in the same commit.
pub const SOFTMAX_HLSL: &str = r"
cbuffer Constants : register(b0) {
    uint Rows; uint GroupsX; uint AxisLen; uint Inner; uint _p0; uint _p1; uint _p2; uint _p3;
}
StructuredBuffer<float>   A : register(t0);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (row >= Rows) return;
    uint o = row / Inner;
    uint i = row % Inner;
    uint base = o * AxisLen * Inner + i;

    float m = A[base];
    for (uint k = 1; k < AxisLen; k++) {
        float v = A[base + k * Inner];
        if (v > m) m = v;
    }
    float sum = 0.0;
    for (uint k = 0; k < AxisLen; k++) {
        sum += exp(A[base + k * Inner] - m);
    }
    float inv = 1.0 / sum;
    for (uint k = 0; k < AxisLen; k++) {
        C[base + k * Inner] = exp(A[base + k * Inner] - m) * inv;
    }
}
";

/// Single-axis reduction.  Entry points: `main_sum`, `main_mean`, `main_max`,
/// `main_min`.
///
/// * `[numthreads(256, 1, 1)]` — see [`crate::plan::REDUCTION_THREADS_PER_GROUP`].
/// * `cbuffer` order **must** match [`crate::plan::ReduceConstants`]:
///   `N, GroupsX, AxisLen, Inner, _p0 … _p3`.
///
/// # One thread per output element
///
/// [`crate::plan::ReducePlan`] uses the same `outer × axis_len × inner` collapse as
/// softmax, but the output is smaller: one element per `(outer, inner)` position.
/// Thread `j ∈ [0, N)` (`N = outer · inner = out_count`) writes `C[j]` from the
/// `AxisLen` inputs at `base + k · Inner`, `base = (j / Inner)·AxisLen·Inner +
/// (j % Inner)`.  The 2-D `GroupsX` folding and the `if (j >= N)` guard are the same
/// contract as everywhere else in this file.
///
/// # Accumulation order is the oracle's order
///
/// `main_sum` / `main_mean` accumulate `acc += A[base + k·Inner]` for `k = 0 … AxisLen`
/// in that exact sequential order, which is the order
/// [`crate::reference::ref_reduce`] pins and the order `oxionnx-ops`' `reduce_with`
/// walks (increasing input linear index for a fixed output).  `main_max` / `main_min`
/// merely *select*, doing no arithmetic, so they must reproduce the oracle bit for
/// bit — see [`crate::reference::Tolerance::for_reduce`].  There is no second operand,
/// so this source declares **no `t1`** (like the unary activations); the backend binds
/// `A` to the root signature's `t1` slot, which the shader ignores.
pub const REDUCE_HLSL: &str = r"
cbuffer Constants : register(b0) {
    uint N; uint GroupsX; uint AxisLen; uint Inner; uint _p0; uint _p1; uint _p2; uint _p3;
}
StructuredBuffer<float>   A : register(t0);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void main_sum(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint j = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (j >= N) return;
    uint o = j / Inner;
    uint i = j % Inner;
    uint base = o * AxisLen * Inner + i;
    float acc = 0.0;
    for (uint k = 0; k < AxisLen; k++) {
        acc += A[base + k * Inner];
    }
    C[j] = acc;
}

[numthreads(256, 1, 1)]
void main_mean(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint j = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (j >= N) return;
    uint o = j / Inner;
    uint i = j % Inner;
    uint base = o * AxisLen * Inner + i;
    float acc = 0.0;
    for (uint k = 0; k < AxisLen; k++) {
        acc += A[base + k * Inner];
    }
    C[j] = acc / (float)AxisLen;
}

[numthreads(256, 1, 1)]
void main_max(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint j = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (j >= N) return;
    uint o = j / Inner;
    uint i = j % Inner;
    uint base = o * AxisLen * Inner + i;
    float acc = A[base];
    for (uint k = 1; k < AxisLen; k++) {
        float v = A[base + k * Inner];
        if (v > acc) acc = v;
    }
    C[j] = acc;
}

[numthreads(256, 1, 1)]
void main_min(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint j = (gid.y * GroupsX + gid.x) * 256 + lid.x;
    if (j >= N) return;
    uint o = j / Inner;
    uint i = j % Inner;
    uint base = o * AxisLen * Inner + i;
    float acc = A[base];
    for (uint k = 1; k < AxisLen; k++) {
        float v = A[base + k * Inner];
        if (v < acc) acc = v;
    }
    C[j] = acc;
}
";

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::plan::ROOT_CONSTANT_COUNT;

    /// Count the `uint` declarations inside the first `cbuffer { … }` block.
    fn cbuffer_uints(src: &str) -> usize {
        let start = src.find('{').expect("a cbuffer block");
        let end = src[start..].find('}').expect("a closing brace") + start;
        src[start..end].matches("uint ").count()
    }

    /// The names declared in the first `cbuffer` block, in order.
    fn cbuffer_names(src: &str) -> Vec<String> {
        let start = src.find('{').expect("a cbuffer block");
        let end = src[start..].find('}').expect("a closing brace") + start;
        src[start..end]
            .split("uint ")
            .skip(1)
            .map(|s| {
                s.trim()
                    .trim_end_matches(';')
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn every_cbuffer_is_exactly_root_constant_count_wide() {
        for (name, src) in [
            ("MATMUL_HLSL", MATMUL_HLSL),
            ("ELEMENTWISE_BINARY_HLSL", ELEMENTWISE_BINARY_HLSL),
            ("ELEMENTWISE_UNARY_HLSL", ELEMENTWISE_UNARY_HLSL),
            ("SOFTMAX_HLSL", SOFTMAX_HLSL),
            ("REDUCE_HLSL", REDUCE_HLSL),
        ] {
            assert_eq!(
                cbuffer_uints(src),
                ROOT_CONSTANT_COUNT,
                "{name}'s cbuffer must be exactly {ROOT_CONSTANT_COUNT} uints wide, or the \
                 shared root signature does not fit it"
            );
        }
    }

    #[test]
    fn softmax_and_reduce_cbuffer_field_order_matches_their_constants() {
        // Softmax → plan::SoftmaxConstants { rows, groups_x, axis_len, inner, .. }.
        let names = cbuffer_names(SOFTMAX_HLSL);
        assert_eq!(
            &names[..4],
            &["Rows", "GroupsX", "AxisLen", "Inner"],
            "cbuffer order must match plan::SoftmaxConstants' #[repr(C)] field order"
        );
        // Reduce → plan::ReduceConstants { out_count, groups_x, axis_len, inner, .. }.
        let names = cbuffer_names(REDUCE_HLSL);
        assert_eq!(
            &names[..4],
            &["N", "GroupsX", "AxisLen", "Inner"],
            "cbuffer order must match plan::ReduceConstants' #[repr(C)] field order"
        );
    }

    #[test]
    fn softmax_and_reduce_entry_points_exist() {
        assert!(SOFTMAX_HLSL.contains("void main("));
        for entry in ["main_sum", "main_mean", "main_max", "main_min"] {
            assert!(
                REDUCE_HLSL.contains(&format!("void {entry}(")),
                "missing reduce entry point {entry}"
            );
        }
    }

    #[test]
    fn softmax_subtracts_the_row_max_before_every_exp() {
        // The stabilisation the oracle also performs.  A shader that dropped it would
        // overflow exp() to +inf on any row with a large positive value, and `compare`
        // would (correctly) report the resulting NaNs.  Both exp() calls must be over
        // the max-subtracted argument.
        assert!(
            SOFTMAX_HLSL.contains("float m = A[base];"),
            "must seed the row max"
        );
        assert_eq!(
            SOFTMAX_HLSL.matches("exp(A[base + k * Inner] - m)").count(),
            2,
            "both the sum and the store must exponentiate the max-subtracted value"
        );
        assert!(
            !SOFTMAX_HLSL.contains("exp(A[base + k * Inner])"),
            "a bare exp() without the - m subtraction is the un-stabilised form"
        );
    }

    #[test]
    fn reduce_sum_and_mean_share_the_sequential_axis_order() {
        // Both accumulate `acc += A[base + k * Inner]` for k = 0..AxisLen, which is the
        // order the oracle and oxionnx-ops pin; mean then divides by the axis length.
        assert_eq!(
            REDUCE_HLSL.matches("acc += A[base + k * Inner];").count(),
            2,
            "sum and mean both accumulate in k-major order"
        );
        assert!(
            REDUCE_HLSL.contains("C[j] = acc / (float)AxisLen;"),
            "mean divides by AxisLen"
        );
        // max/min select rather than accumulate.
        assert!(
            REDUCE_HLSL.contains("if (v > acc) acc = v;"),
            "max selects the larger"
        );
        assert!(
            REDUCE_HLSL.contains("if (v < acc) acc = v;"),
            "min selects the smaller"
        );
    }

    #[test]
    fn softmax_and_reduce_recover_the_2d_group_index_and_guard() {
        // Softmax guards `row >= Rows`; reduce guards `j >= N`.  Both fold the group
        // index the same way as the elementwise kernels.
        assert!(SOFTMAX_HLSL.contains("uint row = (gid.y * GroupsX + gid.x) * 256 + lid.x;"));
        assert!(SOFTMAX_HLSL.contains("if (row >= Rows) return;"));
        let reduce_entries = REDUCE_HLSL.matches("[numthreads(256, 1, 1)]").count();
        assert_eq!(reduce_entries, 4, "sum, mean, max, min");
        assert_eq!(
            REDUCE_HLSL
                .matches("uint j = (gid.y * GroupsX + gid.x) * 256 + lid.x;")
                .count(),
            reduce_entries,
        );
        assert_eq!(
            REDUCE_HLSL.matches("if (j >= N) return;").count(),
            reduce_entries,
            "the folded grid overshoots on purpose; the guard is not optional"
        );
    }

    #[test]
    fn softmax_and_reduce_declare_no_second_operand() {
        // Like the unary activations, these read a single input, so they declare no t1.
        assert!(!SOFTMAX_HLSL.contains("register(t1)"));
        assert!(!REDUCE_HLSL.contains("register(t1)"));
    }

    #[test]
    fn matmul_cbuffer_field_order_matches_matmul_constants() {
        let names = cbuffer_names(MATMUL_HLSL);
        assert_eq!(
            &names[..6],
            &["M", "K", "N", "AOff", "BOff", "COff"],
            "cbuffer order must match plan::MatMulConstants' #[repr(C)] field order"
        );
    }

    #[test]
    fn elementwise_cbuffer_field_order_matches_elementwise_constants() {
        for src in [ELEMENTWISE_BINARY_HLSL, ELEMENTWISE_UNARY_HLSL] {
            let names = cbuffer_names(src);
            assert_eq!(&names[..2], &["N", "GroupsX"]);
        }
    }

    #[test]
    fn every_entry_point_exists_in_its_source() {
        assert!(MATMUL_HLSL.contains("void main("));
        for entry in ["main_add", "main_sub", "main_mul", "main_div"] {
            assert!(
                ELEMENTWISE_BINARY_HLSL.contains(&format!("void {entry}(")),
                "missing entry point {entry}"
            );
        }
        for entry in ["main_relu", "main_sigmoid", "main_tanh"] {
            assert!(
                ELEMENTWISE_UNARY_HLSL.contains(&format!("void {entry}(")),
                "missing entry point {entry}"
            );
        }
    }

    #[test]
    fn matmul_honours_the_batch_offsets() {
        // Every operand index must be offset, or batching silently reads slice 0.
        assert!(MATMUL_HLSL.contains("A[AOff + row * K + k]"));
        assert!(MATMUL_HLSL.contains("B[BOff + k * N + col]"));
        assert!(MATMUL_HLSL.contains("C[COff + row * N + col]"));
    }

    #[test]
    fn matmul_maps_rows_to_y_and_cols_to_x() {
        // The invariant that plan::MatMulPlan::hlsl_grid depends on.
        assert!(MATMUL_HLSL.contains("uint row = tid.y;"));
        assert!(MATMUL_HLSL.contains("uint col = tid.x;"));
    }

    #[test]
    fn every_elementwise_entry_point_recovers_the_2d_group_index_and_guards() {
        for src in [ELEMENTWISE_BINARY_HLSL, ELEMENTWISE_UNARY_HLSL] {
            let entries = src.matches("[numthreads(256, 1, 1)]").count();
            assert_eq!(
                src.matches("uint i = (gid.y * GroupsX + gid.x) * 256 + lid.x;")
                    .count(),
                entries,
                "every entry point must recover its linear index the same way"
            );
            assert_eq!(
                src.matches("if (i >= N) return;").count(),
                entries,
                "the bounds guard is NOT optional — the folded grid overshoots on purpose"
            );
        }
    }

    #[test]
    fn unary_source_declares_no_second_operand() {
        assert!(!ELEMENTWISE_UNARY_HLSL.contains("register(t1)"));
        assert!(ELEMENTWISE_BINARY_HLSL.contains("register(t1)"));
    }

    /// The register declarations that `backend::d3d12::pso`'s root signature must match.
    ///
    /// # Why this test lives *here*, in a neutral module
    ///
    /// A mismatch between a `register()` declaration and the root signature is **not** a
    /// compile error. It is a `CreateComputePipelineState` failure, garbage output, or a
    /// device-removal — on a user's GPU.
    ///
    /// `backend::d3d12::pso` is `#[cfg(target_os = "windows")]`, so its own test of this
    /// join (`pso::tests::root_signature_covers_every_register_the_shaders_declare`,
    /// which parses these declarations back out and diffs them against the root
    /// parameters it actually serialises) is type-checked from Linux but can only *run*
    /// on Windows. This module is neutral, so this half of the contract — the shader
    /// side — is pinned on **every** target, in CI, today.
    ///
    /// The shared root signature provides exactly four things:
    ///
    /// | Root param | Type | Register |
    /// |---|---|---|
    /// | 0 | 8 × 32-bit root constants | `b0` |
    /// | 1 | SRV (root descriptor) | `t0` |
    /// | 2 | SRV (root descriptor) | `t1` |
    /// | 3 | UAV (root descriptor) | `u0` |
    ///
    /// A shader that declares anything else has nowhere to be bound.
    #[test]
    fn shader_registers_match_the_root_signature_that_pso_builds() {
        for (name, src) in [
            ("MATMUL_HLSL", MATMUL_HLSL),
            ("ELEMENTWISE_BINARY_HLSL", ELEMENTWISE_BINARY_HLSL),
        ] {
            assert!(
                src.contains("register(b0)"),
                "{name} must bind its cbuffer to b0"
            );
            assert!(src.contains("register(t0)"), "{name} must bind A to t0");
            assert!(src.contains("register(t1)"), "{name} must bind B to t1");
            assert!(src.contains("register(u0)"), "{name} must bind C to u0");
        }

        // The unary, softmax and reduce kernels have no second operand, so they declare
        // no `t1` — the root signature still exposes one, and the backend binds `A` to
        // it, because the D3D12 debug layer errors on an *unset* root parameter even when
        // the bound shader never reads it.
        for (name, src) in [
            ("ELEMENTWISE_UNARY_HLSL", ELEMENTWISE_UNARY_HLSL),
            ("SOFTMAX_HLSL", SOFTMAX_HLSL),
            ("REDUCE_HLSL", REDUCE_HLSL),
        ] {
            assert!(
                src.contains("register(b0)"),
                "{name} must bind its cbuffer to b0"
            );
            assert!(src.contains("register(t0)"), "{name} must bind A to t0");
            assert!(src.contains("register(u0)"), "{name} must bind C to u0");
        }

        // Nothing may reach for a register the root signature does not provide, and
        // nothing may use a non-zero register space: every root parameter is declared in
        // `space0`, which is what an unqualified `register(t0)` resolves to under SM 5.1.
        for (name, src) in [
            ("MATMUL_HLSL", MATMUL_HLSL),
            ("ELEMENTWISE_BINARY_HLSL", ELEMENTWISE_BINARY_HLSL),
            ("ELEMENTWISE_UNARY_HLSL", ELEMENTWISE_UNARY_HLSL),
            ("SOFTMAX_HLSL", SOFTMAX_HLSL),
            ("REDUCE_HLSL", REDUCE_HLSL),
        ] {
            for unprovided in ["register(b1)", "register(t2)", "register(u1)", "space1"] {
                assert!(
                    !src.contains(unprovided),
                    "{name} declares {unprovided}, which the shared root signature does \
                     not provide — this is a device-removal, not a compile error"
                );
            }
        }
    }

    #[test]
    fn thread_group_sizes_match_the_plan_constants() {
        assert!(MATMUL_HLSL.contains(&format!(
            "[numthreads({}, {}, 1)]",
            crate::plan::MATMUL_TILE,
            crate::plan::MATMUL_TILE
        )));
        for src in [ELEMENTWISE_BINARY_HLSL, ELEMENTWISE_UNARY_HLSL] {
            assert!(src.contains(&format!(
                "[numthreads({}, 1, 1)]",
                crate::plan::ELEMENTWISE_THREADS_PER_GROUP
            )));
        }
        for src in [SOFTMAX_HLSL, REDUCE_HLSL] {
            assert!(src.contains(&format!(
                "[numthreads({}, 1, 1)]",
                crate::plan::REDUCTION_THREADS_PER_GROUP
            )));
        }
    }
}
