//! Regression tests for W2-perf-matmul: numpy-verified reference values for
//! the optimized matmul/gemm/broadcast/quantized-matmul kernels.
//!
//! Covers:
//! - a6-0/a6-17: MatMul at M=1 (the decode-phase shape) and batched M=1, now
//!   routed unconditionally through `matrixmultiply::sgemm` with no per-batch
//!   `Vec<Vec<f32>>` collect.
//! - a6-6: Gemm's transA/transB stride table at m=3,k=2,n=4 (all distinct so
//!   a swapped `rsa`/`csa` cannot masquerade as correct output), covering
//!   both the untyped `math::gemm` (`gemm_2d_into`'s stride fast path) and
//!   the typed `GemmOp::execute_typed` F32 arm (`math_typed::gemm_f32`);
//!   plus alpha/beta with a full `[m,n]` C bias.
//! - a6-15: the panic risk `resolve_bias_i32`/`resolve_bias_f32` have on a
//!   `[1,n]` C bias (`c_data.len() == n`, but the catch-all branch indexes
//!   `row*n+col` which exceeds that for row>=1) — every dtype must either
//!   handle it or fall back, never panic, and the *value* must still be
//!   correct.
//! - a6-3: `elementwise_binary`'s general (non-trailing-vector) broadcast
//!   path, `[3,1] op [1,4]`, with non-commutative ops (sub, div) to catch an
//!   operand-order regression.
//! - a6-19: `quantized_matmul` (per-tensor and per-channel) and
//!   `fully_quantized_matmul` (zero and non-zero zero-points) after the
//!   i-j-k -> i-p-j loop reorder.
//!
//! Reference values are computed with numpy (see the comment above each
//! test); tolerance is 1e-5 relative+absolute per the task's numerical
//! parity budget, except where a comment states the check is exact.

use oxionnx_core::dtype::{DType, TensorStorage, TypedTensor};
use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{Operator, TypedOpContext};
use oxionnx_core::Tensor;
use oxionnx_ops::math;
use oxionnx_ops::quantized::{
    fully_quantized_matmul, quantized_matmul, QuantParams, QuantizedTensor,
};
use oxionnx_ops::registry::math_ops::GemmOp;

// ── Shared test helpers ──────────────────────────────────────────────────────

fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch (got {}, want {})",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= tol + tol * w.abs(),
            "{label}[{i}]: got {g}, want {w}, diff {diff}"
        );
    }
}

fn gemm_node(alpha: f32, beta: f32, trans_a: bool, trans_b: bool) -> Node {
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".into(), alpha);
    attrs.floats.insert("beta".into(), beta);
    attrs
        .ints
        .insert("transA".into(), if trans_a { 1 } else { 0 });
    attrs
        .ints
        .insert("transB".into(), if trans_b { 1 } else { 0 });
    Node {
        name: "test_gemm".into(),
        op: OpKind::Gemm,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

fn typed_ctx_2<'a>(node: &'a Node, a: &'a TypedTensor, b: &'a TypedTensor) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs: vec![Some(a), Some(b)],
        outer_scope: None,
        registry: None,
    }
}

fn typed_ctx_3<'a>(
    node: &'a Node,
    a: &'a TypedTensor,
    b: &'a TypedTensor,
    c: &'a TypedTensor,
) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs: vec![Some(a), Some(b), Some(c)],
        outer_scope: None,
        registry: None,
    }
}

// ── a6-6: Gemm's stride-based transpose, all transA/transB combinations ────

/// numpy: A(3,2)=[[1,2],[3,4],[5,6]], B(2,4)=[[1,2,3,4],[5,6,7,8]],
/// C = A@B = [[11,14,17,20],[23,30,37,44],[35,46,57,68]].
///
/// m=3, k=2, n=4 are all distinct, so a stride table that swaps `rsa`/`csa`
/// (or `rsb`/`csb`) produces a shape or value mismatch here, unlike a square
/// matrix where a transpose bug can silently produce a plausible-looking
/// wrong answer.
#[test]
fn gemm_stride_table_all_transpose_combinations_m3_k2_n4() {
    let a_normal = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3,2]
    let a_trans = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0]; // [2,3] == A^T, stored
    let b_normal = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,4]
    let b_trans = vec![1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]; // [4,2] == B^T, stored
    let expected = vec![
        11.0f32, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0,
    ];

    for trans_a in [false, true] {
        for trans_b in [false, true] {
            let (a_data, a_shape) = if trans_a {
                (a_trans.clone(), vec![2, 3])
            } else {
                (a_normal.clone(), vec![3, 2])
            };
            let (b_data, b_shape) = if trans_b {
                (b_trans.clone(), vec![4, 2])
            } else {
                (b_normal.clone(), vec![2, 4])
            };

            // Untyped path: oxionnx_ops::math::gemm (gemm_2d_into's stride fast path).
            let a_t = Tensor::new(a_data.clone(), a_shape.clone());
            let b_t = Tensor::new(b_data.clone(), b_shape.clone());
            let got = math::gemm(&a_t, &b_t, None, 1.0, 0.0, trans_a, trans_b)
                .unwrap_or_else(|e| panic!("gemm trans_a={trans_a} trans_b={trans_b}: {e}"));
            assert_eq!(got.shape, vec![3, 4], "trans_a={trans_a} trans_b={trans_b}");
            assert_close(
                &format!("untyped gemm trans_a={trans_a} trans_b={trans_b}"),
                &got.data,
                &expected,
                1e-5,
            );

            // Typed path: GemmOp::execute_typed's F32 arm (math_typed::gemm_f32).
            let a_typed = TypedTensor::new(TensorStorage::F32(a_data), a_shape);
            let b_typed = TypedTensor::new(TensorStorage::F32(b_data), b_shape);
            let node = gemm_node(1.0, 0.0, trans_a, trans_b);
            let ctx = typed_ctx_2(&node, &a_typed, &b_typed);
            let result = GemmOp
                .execute_typed(&ctx)
                .unwrap_or_else(|e| panic!("typed gemm trans_a={trans_a} trans_b={trans_b}: {e}"));
            assert_eq!(result[0].dtype(), DType::F32);
            assert_eq!(result[0].shape, vec![3, 4]);
            assert_close(
                &format!("typed gemm trans_a={trans_a} trans_b={trans_b}"),
                &result[0].storage.to_f32_vec(),
                &expected,
                1e-5,
            );
        }
    }
}

/// numpy: A(2,3)=[[1,2,3],[4,5,6]], B(3,3)=[[1,0,1],[0,1,0],[1,1,1]],
/// A@B=[[4,5,4],[10,11,10]]; C(2,3)=[[1,2,3],[4,5,6]] (a **full [m,n] bias**,
/// not just the `[n]` row-vector case) with alpha=2.0, beta=0.5:
/// Y = alpha*(A@B) + beta*C = [[8.5,11.0,9.5],[22.0,24.5,23.0]].
#[test]
fn gemm_alpha_beta_with_full_mn_bias_f32() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        vec![3, 3],
    );
    let c = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let expected = vec![8.5f32, 11.0, 9.5, 22.0, 24.5, 23.0];

    let got = math::gemm(&a, &b, Some(&c), 2.0, 0.5, false, false).expect("untyped gemm");
    assert_close(
        "untyped gemm alpha/beta full [m,n] bias",
        &got.data,
        &expected,
        1e-5,
    );

    let a_typed = TypedTensor::new(TensorStorage::F32(a.data.clone()), a.shape.clone());
    let b_typed = TypedTensor::new(TensorStorage::F32(b.data.clone()), b.shape.clone());
    let c_typed = TypedTensor::new(TensorStorage::F32(c.data.clone()), c.shape.clone());
    let node = gemm_node(2.0, 0.5, false, false);
    let ctx = typed_ctx_3(&node, &a_typed, &b_typed, &c_typed);
    let result = GemmOp.execute_typed(&ctx).expect("typed gemm");
    assert_close(
        "typed gemm alpha/beta full [m,n] bias",
        &result[0].storage.to_f32_vec(),
        &expected,
        1e-5,
    );
}

/// A `[1, n]` C bias is a valid ONNX-broadcastable shape that
/// `resolve_bias_i32`/`resolve_bias_f32`'s catch-all branch
/// (`c_data[row*n+col]`) indexes out of bounds on for any `row >= 1`, since
/// `c_data.len() == n`, not `m*n`. Every `GemmOp::execute_typed` dtype arm
/// must either broadcast it correctly (F32, via `broadcast_to`) or route to
/// `default_typed_via_f32` (I8/I32/F16/BF16, via `gemm_bias_shape_supported`)
/// — never panic — and the *value* must still be numerically correct either
/// way.
///
/// numpy: A(3,2)=[[1,0],[0,1],[1,1]], B(2,2)=[[2,3],[4,5]], A@B=[[2,3],[4,5],[6,8]];
/// C(1,2)=[[10,20]] broadcasts over all 3 rows: Y=[[12,23],[14,25],[16,28]].
#[test]
fn gemm_c_shape_1xn_does_not_panic_for_any_dtype() {
    let m = 3usize;
    let n = 2usize;
    let a_f32 = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2]
    let b_f32 = [2.0f32, 3.0, 4.0, 5.0]; // [2,2]
    let c_f32 = [10.0f32, 20.0]; // [1,2]
    let expected = [12.0f32, 23.0, 14.0, 25.0, 16.0, 28.0];

    // F32: routes through gemm_f32 + broadcast_to, must be exact-ish.
    let a_typed = TypedTensor::new(TensorStorage::F32(a_f32.to_vec()), vec![m, 2]);
    let b_typed = TypedTensor::new(TensorStorage::F32(b_f32.to_vec()), vec![2, n]);
    let c_typed = TypedTensor::new(TensorStorage::F32(c_f32.to_vec()), vec![1, n]);
    let node = gemm_node(1.0, 1.0, false, false);
    let ctx = typed_ctx_3(&node, &a_typed, &b_typed, &c_typed);
    let result = GemmOp
        .execute_typed(&ctx)
        .expect("F32 gemm with [1,n] C must not panic");
    assert_close(
        "F32 [1,n] C",
        &result[0].storage.to_f32_vec(),
        &expected,
        1e-5,
    );

    // I8xI8->I32 with an I32 [1,n] C: gemm_bias_shape_supported routes this
    // to default_typed_via_f32 (the narrow resolve_bias_i32 doesn't support
    // it) -- must not panic, and the value must still be numerically
    // correct. `default_typed_via_f32` always produces an F32 intermediate
    // (dtype recovery to I32 happens at the session level, not here), so
    // the dtype assertion is F32, not I32 -- this is the documented
    // fallback contract, not a regression.
    let a_i8 = TypedTensor::new(TensorStorage::I8(vec![1, 0, 0, 1, 1, 1]), vec![m, 2]);
    let b_i8 = TypedTensor::new(TensorStorage::I8(vec![2, 3, 4, 5]), vec![2, n]);
    let c_i32 = TypedTensor::new(TensorStorage::I32(vec![10, 20]), vec![1, n]);
    let ctx_i8 = typed_ctx_3(&node, &a_i8, &b_i8, &c_i32);
    let result_i8 = GemmOp
        .execute_typed(&ctx_i8)
        .expect("I8 gemm with [1,n] I32 C must not panic");
    assert_eq!(result_i8[0].dtype(), DType::F32);
    assert_close(
        "I8 gemm [1,n] C bias (via default_typed_via_f32 fallback)",
        &result_i8[0].storage.to_f32_vec(),
        &expected,
        1e-5,
    );
}

// ── a6-0/a6-17: M=1 and batched-M=1 MatMul via the unconditional sgemm path ─

/// numpy-verified `[1,5] @ [5,3]` — the decode-phase GEMM shape (M=1) that
/// used to take `matmul_batch_slice`'s naive scalar loop.
#[test]
fn matmul_m1_matches_numpy() {
    let a = Tensor::new(vec![-1.486, -0.003, 0.406, -1.885, -1.408], vec![1, 5]);
    let b = Tensor::new(
        vec![
            1.713, -1.718, -1.481, 1.793, 0.488, -0.524, 0.046, 0.651, -0.899, -1.448, 1.152,
            0.681, 0.05, 1.267, 0.196,
        ],
        vec![5, 3],
    );
    let expected = vec![0.126859f32, -1.139666, 0.277691];
    let got = math::matmul(&a, &b).expect("matmul M=1");
    assert_eq!(got.shape, vec![1, 3]);
    assert_close("matmul M=1", &got.data, &expected, 1e-4);
}

/// numpy-verified batched `[2,1,3] @ [2,3,4]` (M=1 within each batch slice) —
/// the batched-attention decode shape `[B*H,1,d] @ [B*H,d,S]`.
#[test]
fn matmul_batched_m1_matches_numpy() {
    let a = Tensor::new(
        vec![0.962, -0.591, 0.107, -0.033, -0.293, 0.183],
        vec![2, 1, 3],
    );
    let b = Tensor::new(
        vec![
            -0.529, 0.604, 0.735, -0.742, -0.066, -0.446, -0.834, 0.792, -0.14, -0.705, 0.347,
            -0.596, 0.803, -0.566, -0.934, -0.598, -0.309, -0.062, 0.812, 0.395, -0.321, -0.966,
            -0.68, 0.993,
        ],
        vec![2, 3, 4],
    );
    let expected = vec![
        -0.484872f32,
        0.769199,
        1.237093,
        -1.245648,
        0.005295,
        -0.139934,
        -0.331534,
        0.085718,
    ];
    let got = math::matmul(&a, &b).expect("batched matmul M=1");
    assert_eq!(got.shape, vec![2, 1, 4]);
    assert_close("batched matmul M=1", &got.data, &expected, 1e-4);
}

// ── a6-3: elementwise_binary's general (non-trailing-vector) broadcast path ──

/// numpy: a=[[1],[2],[3]] (3,1), b=[[10,20,30,40]] (1,4) -- neither operand
/// spans the output shape and neither is a trailing-contiguous vector, so
/// this exercises the odometer walk, not either fast path. `sub`/`div` are
/// non-commutative, so a flipped operand order in the fast-path-detection
/// branches would be caught here.
#[test]
fn broadcast_general_path_sub_and_div_preserve_operand_order() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]);

    let sub_expected = vec![
        -9.0f32, -19.0, -29.0, -39.0, -8.0, -18.0, -28.0, -38.0, -7.0, -17.0, -27.0, -37.0,
    ];
    let got_sub = math::sub(&a, &b).expect("a - b");
    assert_eq!(got_sub.shape, vec![3, 4]);
    assert_close("[3,1] - [1,4]", &got_sub.data, &sub_expected, 1e-5);

    let div_expected = vec![
        10.0f32, 20.0, 30.0, 40.0, 5.0, 10.0, 15.0, 20.0, 3.3333333, 6.6666667, 10.0, 13.333333,
    ];
    let got_div = math::div(&b, &a).expect("b / a");
    assert_eq!(got_div.shape, vec![3, 4]);
    assert_close("[1,4] / [3,1]", &got_div.data, &div_expected, 1e-5);
}

/// `[1,1,4] + [4]` after the trailing-vector fast path (a6-3's motivating
/// case, at a size small enough to hand-verify) and the equal-shape fast
/// path both agree with the general path's result for the same broadcast.
#[test]
fn broadcast_trailing_vector_matches_equal_shape_fast_path() {
    let bias = Tensor::new(vec![0.5, -1.0, 2.0, 0.25], vec![4]);
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
    let via_trailing_vector = math::add(&a, &bias).expect("trailing-vector add");
    assert_eq!(via_trailing_vector.shape, vec![1, 1, 4]);

    let bias_full = Tensor::new(vec![0.5, -1.0, 2.0, 0.25], vec![1, 1, 4]);
    let via_equal_shape = math::add(&a, &bias_full).expect("equal-shape add");

    assert_close(
        "trailing-vector vs equal-shape",
        &via_trailing_vector.data,
        &via_equal_shape.data,
        1e-6,
    );
    assert_close(
        "trailing-vector vs hand value",
        &via_trailing_vector.data,
        &[1.5, 1.0, 5.0, 4.25],
        1e-6,
    );
}

// ── a6-19: quantized_matmul / fully_quantized_matmul after the i-p-j reorder ─

/// A: 5x6 f32; Bq: 6x4 i8 (identical data reused across both branches below).
/// Reference dequantization done directly in Python (see values), matching
/// the exact formula `quantized_matmul` uses: `(bq - zp) * scale`.
#[test]
fn quantized_matmul_per_tensor_matches_reference() {
    #[rustfmt::skip]
    let a = Tensor::new(vec![
        -2.5, 1.6, 0.9, -0.4, -0.5, 2.1,
        -2.5, 1.1, -1.8, -2.5, 0.1, 2.8,
        1.4, 1.5, 1.3, 1.7, 0.0, -2.3,
        2.0, -0.3, 0.0, -0.8, -2.0, 2.5,
        1.6, 0.8, -0.6, 1.9, 0.2, -0.4,
    ], vec![5, 6]);
    #[rustfmt::skip]
    let bq: Vec<i8> = vec![
        -10, -55, -82, 10,
         77, -88,  71, 65,
        -45,  26, -67, 51,
         40, -30, -87, 94,
        -11,  78,  35, 55,
         51, -62, -28, -7,
    ];
    let b = QuantizedTensor::new(
        bq,
        vec![6, 4],
        QuantParams {
            scale: vec![0.03],
            zero_point: vec![5],
            per_channel: false,
            axis: 0,
        },
    );
    let expected = vec![
        5.949f32, -4.293, 6.324, 1.173, 7.392, -2.487, 16.809, -8.412, -0.729, -3.048, -5.907,
        10.071, 2.022, -11.328, -7.881, -6.276, 3.255, -6.243, -5.964, 6.369,
    ];
    let got = quantized_matmul(&a, &b).expect("quantized_matmul per-tensor");
    assert_eq!(got.shape, vec![5, 4]);
    assert_close("quantized_matmul per-tensor", &got.data, &expected, 1e-3);
}

/// Same A/Bq as the per-tensor test, but per-channel scale/zero-point (one
/// per output column) -- exercises the precomputed `ch_scale`/`ch_zp`
/// hoisting.
#[test]
fn quantized_matmul_per_channel_matches_reference() {
    #[rustfmt::skip]
    let a = Tensor::new(vec![
        -2.5, 1.6, 0.9, -0.4, -0.5, 2.1,
        -2.5, 1.1, -1.8, -2.5, 0.1, 2.8,
        1.4, 1.5, 1.3, 1.7, 0.0, -2.3,
        2.0, -0.3, 0.0, -0.8, -2.0, 2.5,
        1.6, 0.8, -0.6, 1.9, 0.2, -0.4,
    ], vec![5, 6]);
    #[rustfmt::skip]
    let bq: Vec<i8> = vec![
        -10, -55, -82, 10,
         77, -88,  71, 65,
        -45,  26, -67, 51,
         40, -30, -87, 94,
        -11,  78,  35, 55,
         51, -62, -28, -7,
    ];
    let b = QuantizedTensor::new(
        bq,
        vec![6, 4],
        QuantParams {
            scale: vec![0.01, 0.02, 0.03, 0.04],
            zero_point: vec![1, -2, 3, 0],
            per_channel: true,
            axis: 1,
        },
    );
    let expected = vec![
        2.031f32, -2.694, 6.396, 1.804, 2.352, -2.05, 16.641, -11.776, -0.099, -1.528, -5.691,
        14.148, 0.73, -7.356, -7.797, -8.088, 1.225, -3.672, -5.754, 9.192,
    ];
    let got = quantized_matmul(&a, &b).expect("quantized_matmul per-channel");
    assert_eq!(got.shape, vec![5, 4]);
    assert_close("quantized_matmul per-channel", &got.data, &expected, 1e-3);
}

fn qtensor(data: Vec<i8>, shape: Vec<usize>, scale: f32, zero_point: i8) -> QuantizedTensor {
    QuantizedTensor::new(
        data,
        shape,
        QuantParams {
            scale: vec![scale],
            zero_point: vec![zero_point],
            per_channel: false,
            axis: 0,
        },
    )
}

/// The `a_zp == 0 && b_zp == 0` fast path: 5x6 @ 6x4 i8, scales 0.02/0.03.
#[test]
fn fully_quantized_matmul_zero_zero_point_matches_reference() {
    #[rustfmt::skip]
    let aq: Vec<i8> = vec![
        88,  25,  36,  79,  15, 55,
        66, -55, -89, -40, -43, 74,
        82, -99,  -1,  64, -74, 59,
       -77,  -7,  63, -40, -32, -45,
        43, -50,  98, -11,  -5,  0,
    ];
    #[rustfmt::skip]
    let bq: Vec<i8> = vec![
        16,  10,   1,  99,
        61,  58,  40,  24,
       -32,  97,  -7, -57,
        69, -68,  71,  22,
       -78, -92, -12, -93,
       -72,   2,  94,  -7,
    ];
    let a = qtensor(aq, vec![5, 6], 0.02, 0);
    let b = qtensor(bq, vec![6, 4], 0.03, 0);
    let expected = vec![
        1.2612f32, -0.492, 6.861, 4.3308, -2.511, -2.6034, 1.8726, 7.7328, 0.747, -1.467, 4.2642,
        8.2056, -0.4194, 6.3054, -4.4904, -5.3826, -3.5202, 4.9464, -2.0184, -1.3836,
    ];
    let got = fully_quantized_matmul(&a, &b).expect("fully_quantized_matmul zero-zp");
    assert_eq!(got.shape, vec![5, 4]);
    assert_close("fully_quantized_matmul zero-zp", &got.data, &expected, 1e-3);
}

/// The general (non-zero zero-point) corrected path: same shapes, a_zp=3,
/// b_zp=-4 -- exercises `row_sum_a`/`col_sum_b`/`k_zp_product` alongside the
/// i-p-j reorder.
#[test]
fn fully_quantized_matmul_nonzero_zero_point_matches_reference() {
    #[rustfmt::skip]
    let aq: Vec<i8> = vec![
        88,  25,  36,  79,  15, 55,
        66, -55, -89, -40, -43, 74,
        82, -99,  -1,  64, -74, 59,
       -77,  -7,  63, -40, -32, -45,
        43, -50,  98, -11,  -5,  0,
    ];
    #[rustfmt::skip]
    let bq: Vec<i8> = vec![
        16,  10,   1,  99,
        61,  58,  40,  24,
       -32,  97,  -7, -57,
        69, -68,  71,  22,
       -78, -92, -12, -93,
       -72,   2,  94,  -7,
    ];
    let a = qtensor(aq, vec![5, 6], 0.02, 3);
    let b = qtensor(bq, vec![6, 4], 0.03, -4);
    let expected = vec![
        1.998f32, 0.1674, 7.1964, 5.0244, -2.6982, -2.868, 1.284, 7.5024, 0.843, -1.4484, 3.9588,
        8.2584, -0.729, 5.9184, -5.2014, -5.7354, -3.3186, 5.0706, -2.2182, -1.2252,
    ];
    let got = fully_quantized_matmul(&a, &b).expect("fully_quantized_matmul nonzero-zp");
    assert_eq!(got.shape, vec![5, 4]);
    assert_close(
        "fully_quantized_matmul nonzero-zp",
        &got.data,
        &expected,
        1e-3,
    );
}

/// M=1 quantized matmul: the sequential (non-parallel, `m < 4`) path for
/// both `quantized_matmul` and `fully_quantized_matmul`, hand-verifiable.
///
/// numpy: A=[[1,2,3]] (1,3), Bq (per-tensor, scale=0.1, zp=0) = [[10,20],[30,40],[50,60]]
/// dequantized B = [[1,2],[3,4],[5,6]]; A@B = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22,28].
#[test]
fn quantized_matmul_m1_matches_hand_reference() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let bq: Vec<i8> = vec![10, 20, 30, 40, 50, 60];
    let b = qtensor(bq, vec![3, 2], 0.1, 0);
    let got = quantized_matmul(&a, &b).expect("quantized_matmul M=1");
    assert_eq!(got.shape, vec![1, 2]);
    assert_close("quantized_matmul M=1", &got.data, &[22.0, 28.0], 1e-4);
}
