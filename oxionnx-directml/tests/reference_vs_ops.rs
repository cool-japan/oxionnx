//! Cross-validation of the CPU oracle against the CPU kernels it must agree with.
//!
//! [`oxionnx_directml::reference`] is the specification the GPU path is diffed against —
//! by `DirectMLContext::self_check` on real hardware, and by `OXIONNX_DIRECTML_VERIFY=1`
//! on a real workload.  A wrong oracle is worse than no oracle: it would report a
//! *correct* GPU as broken, and (much worse) it could agree with a broken one.
//!
//! So the oracle is itself checked, here, against `oxionnx-ops` — the tuned CPU kernels
//! that every declined node already falls back to, which have been exercised by every
//! model this project has ever run.  These are two independently-written implementations
//! (`oxionnx-ops`' MatMul is `matrixmultiply`'s blocked SGEMM micro-kernel; the oracle's
//! is a naive `k`-major triple loop), so agreement between them is evidence, not
//! tautology.
//!
//! # `oxionnx-ops` is a `[dev-dependencies]` entry, and must stay one
//!
//! An execution provider must not depend on the operator library it exists to bypass.
//! The dispatch fallback is `Ok(None)` — the *session runner* then calls `oxionnx-ops`.
//! If this crate ever gains a non-dev dependency on `oxionnx-ops`, someone has wired the
//! oracle in as a fallback path, which would be slower than the kernel it replaced and
//! would silently mask a declining backend.
//!
//! # The tolerance is per-op, and that is the point
//!
//! Every comparison below goes through [`Tolerance`], the same policy the GPU is held
//! to.  `Add`/`Sub`/`Mul`/`Relu` must agree with `oxionnx-ops` **bit for bit**; `MatMul`
//! is allowed the documented drift, because `matrixmultiply` reassociates the dot product
//! and the oracle deliberately does not (it reproduces the shader's sequential order).
//! Papering over that difference with one loose tolerance everywhere would hide exactly
//! the bugs this file exists to catch.
//!
//! # `Sigmoid` and `Tanh` are NOT cross-validated against `oxionnx-ops`
//!
//! Under its `simd` feature, `oxionnx-ops` swaps in a *fast approximation* of `sigmoid`
//! and `tanh` (see `simd_ops::functions`, whose own doc comments say "approximation").
//! An approximation is not an oracle.  Those two are therefore cross-validated against an
//! independently-written, numerically-stable formula in this file instead — which is a
//! stronger check anyway, and one that cannot silently change under a feature flag.

use oxionnx_core::Tensor;
use oxionnx_directml::plan::{
    BinaryOp, ConvPlan, ElementwisePlan, MatMulPlan, ReduceKind, ReducePlan, SoftmaxPlan, UnaryOp,
};
use oxionnx_directml::reference::{
    compare, ref_binary, ref_conv, ref_matmul, ref_reduce, ref_softmax, ref_unary, Tolerance,
    TRANSCENDENTAL_ABS_TOLERANCE, TRANSCENDENTAL_REL_TOLERANCE,
};
use proptest::prelude::*;

/// The shapes every MatMul cross-check runs over: square, tall, wide, degenerate
/// (`m == 1`, a row-vector product), a single dot product (`m == n == 1`), and one large
/// enough that `matrixmultiply` takes its blocked path and reassociates in earnest.
const MATMUL_SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 7, 1),
    (2, 3, 2),
    (4, 3, 5),
    (3, 5, 1),
    (1, 5, 3),
    (16, 16, 16),
    (17, 33, 9),
    (64, 128, 32),
    (128, 256, 64),
];

/// Deterministic, reproducible pseudo-random floats in roughly `[-1, 1)`.
///
/// A hand-rolled LCG, not `rand`: this workspace's SciRS2 policy forbids `rand`, and a
/// cross-validation test must be bit-reproducible across runs and machines anyway — a
/// failure that cannot be replayed is not a failure report, it is a rumour.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Top 24 bits → a float in [0, 1), then shifted to [-1, 1).
            let bits = (state >> 40) as u32;
            (bits as f32 / f32::from(1u16 << 12) / 4096.0).mul_add(2.0, -1.0)
        })
        .collect()
}

/// Assert that the oracle's `got` agrees with `oxionnx-ops`' `want` under `tolerance`,
/// and print the full deviation report when it does not.
fn assert_agrees(op: &'static str, got: &[f32], want: &[f32], tolerance: Tolerance, case: &str) {
    let report = compare(op, got, want, tolerance).expect("same length");
    assert!(
        report.passed,
        "the CPU oracle disagrees with oxionnx-ops on {case}: {report}"
    );
}

// ── MatMul ───────────────────────────────────────────────────────────────────

#[test]
fn ref_matmul_agrees_with_oxionnx_ops_matmul_on_every_shape() {
    for (seed, &(m, k, n)) in MATMUL_SHAPES.iter().enumerate() {
        let a = pseudo_random(m * k, seed as u64 + 1);
        let b = pseudo_random(k * n, seed as u64 + 1000);

        let plan = MatMulPlan::matmul(&[m, k], &[k, n]).expect("2-D x 2-D is planable");
        let got = ref_matmul(&plan, &a, &b, None).expect("oracle runs");

        let want = oxionnx_ops::math::matmul(
            &Tensor::new(a.clone(), vec![m, k]),
            &Tensor::new(b.clone(), vec![k, n]),
        )
        .expect("oxionnx-ops matmul runs");

        assert_eq!(plan.output_shape, want.shape, "shape ({m}x{k} · {k}x{n})");
        assert_agrees(
            "MatMul",
            &got,
            &want.data,
            Tolerance::for_matmul(&plan),
            &format!("{m}x{k} · {k}x{n}"),
        );
    }
}

#[test]
fn ref_matmul_is_exact_against_oxionnx_ops_on_small_integer_valued_inputs() {
    // Reassociation only bites when rounding occurs.  On small integer-valued operands
    // every partial sum is exactly representable, so the blocked micro-kernel and the
    // oracle's sequential loop must agree *bit for bit* — which pins the oracle's
    // indexing (row-major A, column-strided B, the `k`-major inner product) far more
    // sharply than a tolerant comparison ever could.  A transposed read of B would pass
    // a 1e-5 check on random data surprisingly often; it cannot pass this.
    for &(m, k, n) in &[(2usize, 3usize, 2usize), (4, 5, 3), (7, 2, 6), (16, 16, 16)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();

        let plan = MatMulPlan::matmul(&[m, k], &[k, n]).expect("planable");
        let got = ref_matmul(&plan, &a, &b, None).expect("oracle runs");
        let want =
            oxionnx_ops::math::matmul(&Tensor::new(a, vec![m, k]), &Tensor::new(b, vec![k, n]))
                .expect("ops matmul runs");

        assert_agrees(
            "MatMul",
            &got,
            &want.data,
            Tolerance::Exact,
            &format!("exact {m}x{k} · {k}x{n}"),
        );
    }
}

#[test]
fn ref_matmul_agrees_with_oxionnx_ops_gemm_across_alpha_beta_and_both_transposes() {
    let (m, k, n) = (5usize, 4usize, 3usize);
    for (case, alpha, beta, trans_a, trans_b, with_c) in [
        ("plain", 1.0f32, 0.0f32, false, false, false),
        ("alpha", 2.5, 0.0, false, false, false),
        ("beta", 1.0, 0.5, false, false, true),
        ("alpha+beta", -1.5, 3.0, false, false, true),
        ("transA", 1.0, 0.0, true, false, false),
        ("transB", 1.0, 0.0, false, true, false),
        ("transA+transB", 1.0, 0.0, true, true, false),
        ("everything", 0.75, -2.0, true, true, true),
    ] {
        // The *stored* shapes: transposing an operand swaps the shape it is stored in.
        let a_shape = if trans_a { vec![k, m] } else { vec![m, k] };
        let b_shape = if trans_b { vec![n, k] } else { vec![k, n] };
        let c_shape = vec![m, n];

        let a = pseudo_random(m * k, 11);
        let b = pseudo_random(k * n, 22);
        let c = pseudo_random(m * n, 33);

        let plan = MatMulPlan::gemm(
            &a_shape,
            &b_shape,
            with_c.then_some(c_shape.as_slice()),
            alpha,
            beta,
            trans_a,
            trans_b,
        )
        .expect("gemm is planable");
        assert_eq!(
            (plan.m, plan.k, plan.n),
            (m as u32, k as u32, n as u32),
            "{case}"
        );

        let got = ref_matmul(&plan, &a, &b, with_c.then_some(c.as_slice())).expect("oracle runs");

        let c_tensor = Tensor::new(c.clone(), c_shape.clone());
        let want = oxionnx_ops::math::gemm(
            &Tensor::new(a.clone(), a_shape.clone()),
            &Tensor::new(b.clone(), b_shape.clone()),
            with_c.then_some(&c_tensor),
            alpha,
            beta,
            trans_a,
            trans_b,
        )
        .expect("oxionnx-ops gemm runs");

        assert_eq!(plan.output_shape, want.shape, "{case}");
        assert_agrees("Gemm", &got, &want.data, Tolerance::for_matmul(&plan), case);
    }
}

#[test]
fn ref_matmul_agrees_with_oxionnx_ops_gemm_on_a_broadcast_bias() {
    // ONNX `Gemm`'s C is broadcast against [M, N]; a row-vector bias is what every dense
    // layer in every model actually carries, so it gets its own case.
    let (m, k, n) = (6usize, 4usize, 3usize);
    let a = pseudo_random(m * k, 101);
    let b = pseudo_random(k * n, 202);

    for c_shape in [vec![n], vec![1, n], vec![m, 1], vec![1, 1], vec![m, n]] {
        let c_elems: usize = c_shape.iter().product();
        let c = pseudo_random(c_elems, 303);

        let plan = MatMulPlan::gemm(&[m, k], &[k, n], Some(&c_shape), 1.0, 1.0, false, false)
            .expect("gemm is planable");
        assert!(plan.has_bias());
        let got = ref_matmul(&plan, &a, &b, Some(&c)).expect("oracle runs");

        let want = oxionnx_ops::math::gemm(
            &Tensor::new(a.clone(), vec![m, k]),
            &Tensor::new(b.clone(), vec![k, n]),
            Some(&Tensor::new(c.clone(), c_shape.clone())),
            1.0,
            1.0,
            false,
            false,
        )
        .expect("oxionnx-ops gemm runs");

        assert_agrees(
            "Gemm",
            &got,
            &want.data,
            Tolerance::for_matmul(&plan),
            &format!("bias {c_shape:?}"),
        );
    }
}

// ── binary elementwise ───────────────────────────────────────────────────────

#[test]
fn ref_binary_is_bit_exact_against_oxionnx_ops() {
    // Add/Sub/Mul are IEEE-exact and order-independent; Div is held to the same ~1 ULP
    // budget the GPU gets (D3D permits 1 ULP there), which `oxionnx-ops` trivially meets
    // on the CPU.  Any disagreement at all on the first three is a bug in one of the two
    // implementations — there is nowhere for it to hide.
    let shape = vec![4usize, 9usize, 3usize];
    let elems: usize = shape.iter().product();
    let a = pseudo_random(elems, 7);
    // Shift B away from zero so that `Div` is well-conditioned; division by a
    // near-denormal is a property of IEEE, not of these two implementations, and it would
    // only measure the LCG.
    let b: Vec<f32> = pseudo_random(elems, 8).iter().map(|v| v + 1.5).collect();

    let plan = ElementwisePlan::binary(&shape, &shape).expect("identical shapes are planable");
    let a_tensor = Tensor::new(a.clone(), shape.clone());
    let b_tensor = Tensor::new(b.clone(), shape.clone());

    for (op, ops_result) in [
        (BinaryOp::Add, oxionnx_ops::math::add(&a_tensor, &b_tensor)),
        (BinaryOp::Sub, oxionnx_ops::math::sub(&a_tensor, &b_tensor)),
        (BinaryOp::Mul, oxionnx_ops::math::mul(&a_tensor, &b_tensor)),
        (BinaryOp::Div, oxionnx_ops::math::div(&a_tensor, &b_tensor)),
    ] {
        let want = ops_result.expect("oxionnx-ops elementwise runs");
        let got = ref_binary(&plan, op, &a, &b).expect("oracle runs");
        assert_eq!(want.shape, plan.output_shape);
        assert_agrees(
            op.as_str(),
            &got,
            &want.data,
            Tolerance::for_binary(op),
            op.as_str(),
        );
        if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) {
            assert_eq!(got, want.data, "{} must be bit-exact", op.as_str());
        }
    }
}

#[test]
fn ref_binary_broadcasting_agrees_with_oxionnx_ops() {
    // `ElementwisePlan::binary` declines non-identical shapes today, so this pair cannot
    // reach the GPU — but the oracle is the *specification* of what it must compute when
    // that restriction lifts, and `oxionnx-ops` is the definition of correct numpy
    // broadcasting.  Building the plan through the (public) struct literal is how the
    // future case gets checked today.
    for (a_shape, b_shape, out_shape) in [
        (vec![2usize, 1usize], vec![3usize], vec![2usize, 3usize]),
        (vec![2, 3, 4], vec![1, 4], vec![2, 3, 4]),
        (vec![1, 3, 1], vec![2, 1, 4], vec![2, 3, 4]),
        (vec![5], vec![1], vec![5]),
    ] {
        let a = pseudo_random(a_shape.iter().product(), 41);
        let b = pseudo_random(b_shape.iter().product(), 42);
        let out_elems: usize = out_shape.iter().product();

        let plan = ElementwisePlan {
            elem_count: u32::try_from(out_elems).expect("fits u32"),
            a_needs_broadcast: a_shape != out_shape,
            b_needs_broadcast: b_shape != out_shape,
            output_shape: out_shape.clone(),
            a_shape: a_shape.clone(),
            b_shape: Some(b_shape.clone()),
        };

        let got = ref_binary(&plan, BinaryOp::Add, &a, &b).expect("oracle runs");
        let want = oxionnx_ops::math::add(
            &Tensor::new(a, a_shape.clone()),
            &Tensor::new(b, b_shape.clone()),
        )
        .expect("oxionnx-ops add runs");

        assert_eq!(want.shape, out_shape, "{a_shape:?} + {b_shape:?}");
        assert_eq!(
            got, want.data,
            "broadcast Add must be bit-exact: {a_shape:?} + {b_shape:?}"
        );
    }
}

// ── unary elementwise ────────────────────────────────────────────────────────

#[test]
fn ref_unary_relu_is_bit_exact_against_oxionnx_ops() {
    let shape = vec![8usize, 8usize];
    let mut a = pseudo_random(64, 9);
    // The values that make Relu interesting, planted where a vectorised kernel's tail
    // handling would miss them.
    a[0] = -0.0;
    a[1] = 0.0;
    a[7] = -f32::MIN_POSITIVE;
    a[63] = f32::MIN_POSITIVE;

    let plan = ElementwisePlan::unary(&shape).expect("planable");
    let got = ref_unary(&plan, UnaryOp::Relu, &a).expect("oracle runs");
    let want = oxionnx_ops::nn::activations::relu(&Tensor::new(a, shape.clone()));

    assert_eq!(want.shape, plan.output_shape);
    assert_agrees("Relu", &got, &want.data, Tolerance::Exact, "relu");
}

#[test]
fn ref_unary_sigmoid_and_tanh_agree_with_an_independent_stable_formula() {
    // NOT cross-validated against `oxionnx-ops`: under its `simd` feature it substitutes
    // a documented *approximation* for both of these, and an approximation cannot be an
    // oracle.  Instead they are checked against independently-written, numerically-stable
    // formulations — which is a stronger test, and immune to a feature flag flipping
    // underneath it.
    let a: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
    let plan = ElementwisePlan::unary(&[a.len()]).expect("planable");

    // The two-branch sigmoid: mathematically identical, computed without ever evaluating
    // exp() of a large positive argument.  The oracle deliberately uses the *shader's*
    // one-branch form, so agreement here says the shader's form is safe over this range.
    let stable_sigmoid: Vec<f32> = a
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
        .collect();
    let got = ref_unary(&plan, UnaryOp::Sigmoid, &a).expect("oracle runs");
    assert_agrees(
        "Sigmoid",
        &got,
        &stable_sigmoid,
        Tolerance::Approx {
            rel: TRANSCENDENTAL_REL_TOLERANCE,
            abs: TRANSCENDENTAL_ABS_TOLERANCE,
        },
        "sigmoid vs the two-branch form",
    );

    // tanh(x) = 2·sigmoid(2x) - 1, an identity that shares no code with `f32::tanh`.
    let via_sigmoid: Vec<f32> = a
        .iter()
        .map(|&x| 2.0 / (1.0 + (-2.0 * x).exp()) - 1.0)
        .collect();
    let got = ref_unary(&plan, UnaryOp::Tanh, &a).expect("oracle runs");
    assert_agrees(
        "Tanh",
        &got,
        &via_sigmoid,
        Tolerance::Approx {
            rel: 1.0e-5,
            abs: 1.0e-6,
        },
        "tanh vs 2·sigmoid(2x) - 1",
    );
}

// ── property tests ───────────────────────────────────────────────────────────

proptest! {
    /// For any 2-D shape triple and any operands, the oracle and `oxionnx-ops` agree on
    /// MatMul within the documented, `K`-scaled budget.
    #[test]
    fn prop_ref_matmul_agrees_with_oxionnx_ops(
        m in 1usize..12,
        k in 1usize..12,
        n in 1usize..12,
        seed in 0u64..2048,
    ) {
        let a = pseudo_random(m * k, seed);
        let b = pseudo_random(k * n, seed ^ 0xDEAD_BEEF);

        let plan = MatMulPlan::matmul(&[m, k], &[k, n]).expect("2-D x 2-D is planable");
        let got = ref_matmul(&plan, &a, &b, None).expect("oracle runs");
        let want = oxionnx_ops::math::matmul(
            &Tensor::new(a, vec![m, k]),
            &Tensor::new(b, vec![k, n]),
        )
        .expect("oxionnx-ops matmul runs");

        let report = compare("MatMul", &got, &want.data, Tolerance::for_matmul(&plan))
            .expect("same length");
        prop_assert!(report.passed, "{m}x{k} · {k}x{n}: {report}");
    }

    /// Add is bit-exact against `oxionnx-ops`, for every shape and every operand.  Not a
    /// tolerance — equality.  If this ever fails, one of the two is indexing wrongly.
    #[test]
    fn prop_ref_binary_add_is_bit_exact_against_oxionnx_ops(
        d0 in 1usize..6,
        d1 in 1usize..6,
        d2 in 1usize..6,
        seed in 0u64..2048,
    ) {
        let shape = vec![d0, d1, d2];
        let elems = d0 * d1 * d2;
        let a = pseudo_random(elems, seed);
        let b = pseudo_random(elems, seed.wrapping_add(1));

        let plan = ElementwisePlan::binary(&shape, &shape).expect("identical shapes");
        let got = ref_binary(&plan, BinaryOp::Add, &a, &b).expect("oracle runs");
        let want = oxionnx_ops::math::add(
            &Tensor::new(a, shape.clone()),
            &Tensor::new(b, shape.clone()),
        )
        .expect("oxionnx-ops add runs");

        prop_assert_eq!(got, want.data);
    }

    /// Relu is bit-exact against `oxionnx-ops`, over the full float range including the
    /// signed zeros and the denormals.
    #[test]
    fn prop_ref_unary_relu_is_bit_exact_against_oxionnx_ops(
        values in prop::collection::vec(-1.0e30f32..1.0e30, 1..64),
    ) {
        let shape = vec![values.len()];
        let plan = ElementwisePlan::unary(&shape).expect("planable");
        let got = ref_unary(&plan, UnaryOp::Relu, &values).expect("oracle runs");
        let want = oxionnx_ops::nn::activations::relu(&Tensor::new(values, shape));
        prop_assert_eq!(got, want.data);
    }
}

// ── Softmax ────────────────────────────────────────────────────────────────────
//
// Cross-validated against `oxionnx_ops::nn::softmax`, which (without the `simd`
// feature this crate's dev build does not enable) uses the same scalar, max-subtracted,
// axis-order formulation the oracle does.  `exp` is `libm`'s on both sides, so agreement
// is essentially exact — well within the transcendental budget the GPU is held to.

#[test]
fn ref_softmax_agrees_with_oxionnx_ops_across_axes_and_shapes() {
    for (shape, axis) in [
        (vec![8usize], 0i64),
        (vec![4, 6], 1),
        (vec![4, 6], 0),
        (vec![2, 3, 5], 2),
        (vec![2, 3, 5], 1),
        (vec![2, 3, 5], 0),
        (vec![3, 7], -1),
        (vec![2, 4, 4, 4], -1),
    ] {
        let elems: usize = shape.iter().product();
        let a = pseudo_random(elems, axis.unsigned_abs().wrapping_add(17));

        let plan = SoftmaxPlan::softmax(&shape, axis).expect("softmax is planable");
        let got = ref_softmax(&plan, &a).expect("oracle runs");

        let want = oxionnx_ops::nn::softmax(&Tensor::new(a.clone(), shape.clone()), axis)
            .expect("oxionnx-ops softmax runs");

        assert_eq!(want.shape, shape, "softmax is shape-preserving");
        assert_agrees(
            "Softmax",
            &got,
            &want.data,
            Tolerance::for_softmax(&plan),
            &format!("softmax {shape:?} axis {axis}"),
        );
    }
}

#[test]
fn ref_softmax_stays_finite_on_a_large_positive_row() {
    // The stabilisation matters: a row full of large values must not overflow to NaN,
    // and both implementations must land on the same normalised distribution.
    let shape = vec![2usize, 5usize];
    let a = vec![
        100.0, 101.0, 102.0, 103.0, 104.0, //
        -50.0, -49.0, -48.0, -47.0, -46.0,
    ];
    let plan = SoftmaxPlan::softmax(&shape, 1).expect("planable");
    let got = ref_softmax(&plan, &a).expect("oracle runs");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "no NaN/Inf from a large row"
    );
    let want = oxionnx_ops::nn::softmax(&Tensor::new(a, shape), 1).expect("ops softmax runs");
    assert_agrees(
        "Softmax",
        &got,
        &want.data,
        Tolerance::for_softmax(&plan),
        "large-row softmax",
    );
}

// ── Reduce ─────────────────────────────────────────────────────────────────────
//
// The oracle accumulates in the same increasing-axis order `oxionnx_ops`' `reduce_with`
// walks for a fixed output, so Sum/Mean are bit-exact on integer-valued inputs and
// Max/Min are exact on any input.  Random-valued Sum/Mean go through the documented
// `√axis_len` budget.

fn reduce_ops(
    kind: ReduceKind,
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor, String> {
    match kind {
        ReduceKind::Sum => oxionnx_ops::math::reduce_sum(x, axes, keepdims),
        ReduceKind::Mean => oxionnx_ops::math::reduce_mean(x, axes, keepdims),
        ReduceKind::Max => oxionnx_ops::math::reduce_max(x, axes, keepdims),
        ReduceKind::Min => oxionnx_ops::math::reduce_min(x, axes, keepdims),
    }
}

#[test]
fn ref_reduce_agrees_with_oxionnx_ops_on_every_kind_axis_and_keepdims() {
    for (shape, axis) in [
        (vec![6usize], 0i64),
        (vec![4, 5], 0),
        (vec![4, 5], 1),
        (vec![2, 3, 5], 0),
        (vec![2, 3, 5], 1),
        (vec![2, 3, 5], 2),
        (vec![3, 4], -1),
    ] {
        let elems: usize = shape.iter().product();
        let a = pseudo_random(elems, axis.unsigned_abs().wrapping_mul(13).wrapping_add(5));

        for kind in [
            ReduceKind::Sum,
            ReduceKind::Mean,
            ReduceKind::Max,
            ReduceKind::Min,
        ] {
            for keepdims in [true, false] {
                let plan = ReducePlan::reduce(kind, &shape, &[axis], keepdims)
                    .expect("single-axis reduce is planable");
                let got = ref_reduce(&plan, &a).expect("oracle runs");
                let want = reduce_ops(
                    kind,
                    &Tensor::new(a.clone(), shape.clone()),
                    &[axis],
                    keepdims,
                )
                .expect("oxionnx-ops reduce runs");

                assert_eq!(
                    plan.output_shape,
                    want.shape,
                    "{} {shape:?} axis {axis} keepdims {keepdims}",
                    kind.as_str()
                );
                assert_agrees(
                    kind.as_str(),
                    &got,
                    &want.data,
                    Tolerance::for_reduce(&plan),
                    &format!(
                        "{} {shape:?} axis {axis} keepdims {keepdims}",
                        kind.as_str()
                    ),
                );
            }
        }
    }
}

#[test]
fn ref_reduce_sum_and_mean_are_bit_exact_on_integer_inputs() {
    // Same accumulation order as `oxionnx-ops`, so on exactly-representable inputs the
    // two must agree to the bit — which pins the oracle's strided indexing far more
    // sharply than a tolerant comparison could.
    let shape = vec![4usize, 5usize, 3usize];
    let a: Vec<f32> = (0..60).map(|i| (i % 9) as f32 - 4.0).collect();
    for axis in 0..3i64 {
        for kind in [ReduceKind::Sum, ReduceKind::Mean] {
            let plan = ReducePlan::reduce(kind, &shape, &[axis], false).expect("planable");
            let got = ref_reduce(&plan, &a).expect("oracle runs");
            let want = reduce_ops(kind, &Tensor::new(a.clone(), shape.clone()), &[axis], false)
                .expect("ops reduce runs");
            assert_eq!(
                got,
                want.data,
                "{} over axis {axis} must be bit-exact on integers",
                kind.as_str()
            );
        }
    }
}

// ── Conv ───────────────────────────────────────────────────────────────────────
//
// Cross-validated against `oxionnx_ops::conv::conv2d`.  Every configuration below
// deliberately avoids that kernel's Winograd F(2,3) fast path (which needs a 3×3
// stride-1 dilation-1 group-1 kernel with output ≥ 4×4 and equal pads), because
// Winograd is a *mathematically* equivalent but numerically different algorithm whose
// drift would exceed the direct-convolution `mad` budget.  What remains is the im2col +
// SGEMM path (and the 1×1 fast path), both of which are ordinary convolution sums that
// agree with the direct oracle within `Tolerance::for_conv` — and bit-exactly on
// integers.

/// `(input, weight, strides, pads, dilations, group, with_bias)`, all Winograd-free.
type ConvCase = (
    Vec<usize>,
    Vec<usize>,
    [usize; 2],
    [usize; 4],
    [usize; 2],
    usize,
    bool,
);

fn conv_cases() -> Vec<ConvCase> {
    vec![
        // 1×1 kernel → the 1×1 matmul fast path.
        (
            vec![1, 3, 5, 5],
            vec![4, 3, 1, 1],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
            true,
        ),
        // 3×3 stride 2 → im2col (stride ≠ 1 disqualifies Winograd).
        (
            vec![1, 2, 8, 8],
            vec![5, 2, 3, 3],
            [2, 2],
            [1, 1, 1, 1],
            [1, 1],
            1,
            true,
        ),
        // 3×3 stride 1 but a 3×3 output (< 4×4) → im2col, not Winograd.
        (
            vec![1, 3, 5, 5],
            vec![6, 3, 3, 3],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
            false,
        ),
        // 2×2 kernel → im2col (kernel ≠ 3×3).
        (
            vec![2, 3, 6, 6],
            vec![4, 3, 2, 2],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
            true,
        ),
        // grouped 3×3 (group ≠ 1 disqualifies Winograd).
        (
            vec![1, 4, 6, 6],
            vec![6, 2, 3, 3],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            2,
            true,
        ),
        // dilated 3×3 (dilation ≠ 1 disqualifies Winograd).
        (
            vec![1, 2, 9, 9],
            vec![3, 2, 3, 3],
            [1, 1],
            [0, 0, 0, 0],
            [2, 2],
            1,
            false,
        ),
        // 5×5 kernel with asymmetric padding → im2col.
        (
            vec![1, 2, 10, 10],
            vec![4, 2, 5, 5],
            [1, 1],
            [2, 1, 2, 1],
            [1, 1],
            1,
            true,
        ),
    ]
}

fn run_conv_case(case: &ConvCase, integer: bool, seed: u64) {
    let (input_shape, weight_shape, strides, pads, dilations, group, with_bias) = case;
    let in_elems: usize = input_shape.iter().product();
    let w_elems: usize = weight_shape.iter().product();
    let c_out = weight_shape[0];

    let (input, weight, bias) = if integer {
        let input: Vec<f32> = (0..in_elems).map(|i| (i % 7) as f32 - 3.0).collect();
        let weight: Vec<f32> = (0..w_elems).map(|i| (i % 5) as f32 - 2.0).collect();
        let bias: Vec<f32> = (0..c_out).map(|i| (i % 3) as f32 - 1.0).collect();
        (input, weight, bias)
    } else {
        (
            pseudo_random(in_elems, seed),
            pseudo_random(w_elems, seed + 100),
            pseudo_random(c_out, seed + 200),
        )
    };

    let bias_shape = if *with_bias { Some(vec![c_out]) } else { None };
    let plan = ConvPlan::conv(
        input_shape,
        weight_shape,
        bias_shape.as_deref(),
        strides,
        pads,
        dilations,
        *group,
    )
    .expect("conv is planable");

    let got = ref_conv(&plan, &input, &weight, with_bias.then_some(bias.as_slice()))
        .expect("oracle runs");

    let bias_tensor = Tensor::new(bias.clone(), vec![c_out]);
    let want = oxionnx_ops::conv::conv2d(
        &Tensor::new(input.clone(), input_shape.clone()),
        &Tensor::new(weight.clone(), weight_shape.clone()),
        with_bias.then_some(&bias_tensor),
        *strides,
        *pads,
        *dilations,
        *group,
    );

    let label = format!(
        "conv in {input_shape:?} w {weight_shape:?} s{strides:?} p{pads:?} d{dilations:?} g{group} \
         integer={integer}"
    );
    assert_eq!(plan.output_shape, want.shape, "{label}");
    if integer {
        assert_eq!(
            got, want.data,
            "conv must be bit-exact on integers: {label}"
        );
    } else {
        assert_agrees("Conv", &got, &want.data, Tolerance::for_conv(&plan), &label);
    }
}

#[test]
fn ref_conv_agrees_with_oxionnx_ops_conv2d_on_random_inputs() {
    for (seed, case) in conv_cases().iter().enumerate() {
        run_conv_case(case, false, seed as u64 * 7 + 1);
    }
}

#[test]
fn ref_conv_is_bit_exact_against_oxionnx_ops_conv2d_on_integer_inputs() {
    for (seed, case) in conv_cases().iter().enumerate() {
        run_conv_case(case, true, seed as u64);
    }
}
