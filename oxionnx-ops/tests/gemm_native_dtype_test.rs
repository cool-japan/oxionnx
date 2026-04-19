//! Tests for GemmOp native typed dispatch (v0.1.9).
//!
//! Covers F32, I8, I32, F16, BF16 typed dispatch paths with alpha/beta/transA/transB/bias.

use oxionnx_core::{
    dtype::{DType, TensorStorage, TypedTensor},
    graph::{Attributes, Node, OpKind},
    operator::{Operator, TypedOpContext},
};
use oxionnx_ops::registry::math_ops::GemmOp;

// ── Test infrastructure ──────────────────────────────────────────────────────

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

fn make_typed_ctx_2<'a>(
    node: &'a Node,
    a: &'a TypedTensor,
    b: &'a TypedTensor,
) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs: vec![Some(a), Some(b)],
        outer_scope: None,
        registry: None,
    }
}

fn make_typed_ctx_3<'a>(
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

fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

fn f32_to_bf16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

/// Config bundle for `gemm_f32_ref`.
struct GemmRefCfg<'a> {
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    c: Option<&'a [f32]>,
}

/// Reference: naive f32 GEMM used to validate typed kernels.
fn gemm_f32_ref(a: &[f32], b: &[f32], cfg: &GemmRefCfg<'_>) -> Vec<f32> {
    let GemmRefCfg {
        m,
        n,
        k,
        alpha,
        beta,
        trans_a,
        trans_b,
        c,
    } = *cfg;
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = if trans_a {
                    a[p * m + row]
                } else {
                    a[row * k + p]
                };
                let bv = if trans_b {
                    b[col * k + p]
                } else {
                    b[p * n + col]
                };
                acc += av * bv;
            }
            let bias = c.map_or(0.0, |cv| cv[col]);
            out[row * n + col] = acc * alpha + beta * bias;
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// F32 baseline: assert DType::F32 is in native_dtypes(); run execute_typed on 2×2.
#[test]
fn test_gemm_f32_baseline() {
    assert!(
        GemmOp.native_dtypes().contains(&DType::F32),
        "DType::F32 must be in native_dtypes()"
    );

    // [[1,2],[3,4]] @ [[5,6],[7,8]] with alpha=1, beta=0 → [[19,22],[43,50]]
    let a_vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_vals = vec![5.0f32, 6.0, 7.0, 8.0];
    let a = TypedTensor::new(TensorStorage::F32(a_vals), vec![2, 2]);
    let b = TypedTensor::new(TensorStorage::F32(b_vals), vec![2, 2]);

    let node = gemm_node(1.0, 0.0, false, false);
    let ctx = make_typed_ctx_2(&node, &a, &b);
    let result = GemmOp.execute_typed(&ctx).expect("F32 gemm failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::F32);
    assert_eq!(result[0].shape, vec![2, 2]);

    let out = result[0].storage.to_f32_vec();
    assert!(
        (out[0] - 19.0).abs() < 1e-4,
        "[0,0] expected 19, got {}",
        out[0]
    );
    assert!(
        (out[1] - 22.0).abs() < 1e-4,
        "[0,1] expected 22, got {}",
        out[1]
    );
    assert!(
        (out[2] - 43.0).abs() < 1e-4,
        "[1,0] expected 43, got {}",
        out[2]
    );
    assert!(
        (out[3] - 50.0).abs() < 1e-4,
        "[1,1] expected 50, got {}",
        out[3]
    );
}

/// I8×I8→I32 with alpha=1.0, beta=1.0, and I32 bias `[n]` broadcast.
/// M=N=K=4, compare against naive f32 reference (rounded to i32).
#[test]
fn test_gemm_i8_i32_alpha_beta_bias() {
    assert!(
        GemmOp.native_dtypes().contains(&DType::I8),
        "DType::I8 must be in native_dtypes()"
    );

    // M=N=K=4 — non-trivial but small enough to calculate by hand
    // A: identity-like [[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]
    // B: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    // A @ B: each row i of A picks row i of B, scaled by A[i,i]
    //   row0: [1,2,3,4]
    //   row1: [10,12,14,16]
    //   row2: [27,30,33,36]
    //   row3: [52,56,60,64]
    // With bias [10,20,30,40], alpha=1.0, beta=1.0:
    //   row0: [11,22,33,44]
    //   row1: [20,32,44,56]
    //   row2: [37,50,63,76]
    //   row3: [62,76,90,104]
    #[rustfmt::skip]
    let a_vals: Vec<i8> = vec![
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 4,
    ];
    #[rustfmt::skip]
    let b_vals: Vec<i8> = vec![
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
    ];
    let bias_vals: Vec<i32> = vec![10, 20, 30, 40];

    let a = TypedTensor::new(TensorStorage::I8(a_vals.clone()), vec![4, 4]);
    let b = TypedTensor::new(TensorStorage::I8(b_vals.clone()), vec![4, 4]);
    let c = TypedTensor::new(TensorStorage::I32(bias_vals), vec![4]);

    let node = gemm_node(1.0, 1.0, false, false);
    let ctx = make_typed_ctx_3(&node, &a, &b, &c);
    let result = GemmOp.execute_typed(&ctx).expect("I8 gemm failed");

    assert_eq!(result[0].dtype(), DType::I32);
    assert_eq!(result[0].shape, vec![4, 4]);

    let expected_rows: &[&[i32]] = &[
        &[11, 22, 33, 44],
        &[20, 32, 44, 56],
        &[37, 50, 63, 76],
        &[62, 76, 90, 104],
    ];

    if let TensorStorage::I32(ref data) = result[0].storage {
        for (ri, &row) in expected_rows.iter().enumerate() {
            for (ci, &expected) in row.iter().enumerate() {
                let got = data[ri * 4 + ci];
                assert_eq!(
                    got, expected,
                    "I8 gemm [{ri},{ci}]: expected {expected}, got {got}"
                );
            }
        }
    } else {
        panic!("Expected I32 storage, got {:?}", result[0].dtype());
    }
}

/// I32 transposes: exercise transA × transB combinations with M=N=K=3.
/// Uses non-symmetric values so transposes are distinguishable.
#[test]
fn test_gemm_i32_transposes() {
    assert!(
        GemmOp.native_dtypes().contains(&DType::I32),
        "DType::I32 must be in native_dtypes()"
    );

    // A (3×3), non-symmetric:
    // [[1,2,3],[4,5,6],[7,8,9]]
    // A^T = [[1,4,7],[2,5,8],[3,6,9]]
    // B (3×3), non-symmetric:
    // [[9,8,7],[6,5,4],[3,2,1]]
    // B^T = [[9,6,3],[8,5,2],[7,4,1]]
    let a_vals: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let b_vals: Vec<i32> = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];

    // Precompute all 4 combinations via f32 reference
    let cases = [(false, false), (false, true), (true, false), (true, true)];

    for (trans_a, trans_b) in cases {
        let a_f32: Vec<f32> = a_vals.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b_vals.iter().map(|&x| x as f32).collect();
        let ref_out = gemm_f32_ref(
            &a_f32,
            &b_f32,
            &GemmRefCfg {
                m: 3,
                n: 3,
                k: 3,
                alpha: 1.0,
                beta: 0.0,
                trans_a,
                trans_b,
                c: None,
            },
        );

        let a = TypedTensor::new(TensorStorage::I32(a_vals.clone()), vec![3, 3]);
        let b = TypedTensor::new(TensorStorage::I32(b_vals.clone()), vec![3, 3]);
        let node = gemm_node(1.0, 0.0, trans_a, trans_b);
        let ctx = make_typed_ctx_2(&node, &a, &b);
        let result = GemmOp
            .execute_typed(&ctx)
            .unwrap_or_else(|e| panic!("I32 gemm transA={trans_a} transB={trans_b} failed: {e}"));

        assert_eq!(result[0].dtype(), DType::I32);
        assert_eq!(result[0].shape, vec![3, 3]);

        if let TensorStorage::I32(ref data) = result[0].storage {
            for (i, (&got, &expected_f32)) in data.iter().zip(ref_out.iter()).enumerate() {
                let expected = expected_f32.round() as i32;
                assert_eq!(
                    got, expected,
                    "I32 gemm transA={trans_a} transB={trans_b} output[{i}]: expected {expected}, got {got}"
                );
            }
        } else {
            panic!(
                "Expected I32 storage for transA={trans_a} transB={trans_b}, got {:?}",
                result[0].dtype()
            );
        }
    }
}

/// F16 typed GEMM parity: 4×4 A and B with alpha=0.5, beta=0.5, F16 bias [4].
/// Compare typed F16 result against f32 reference; tolerance < 1e-2 absolute.
#[test]
fn test_gemm_f16_parity() {
    assert!(
        GemmOp.native_dtypes().contains(&DType::F16),
        "DType::F16 must be in native_dtypes()"
    );

    let a_f32 = vec![
        1.0f32, 0.5, -0.5, 0.25, 0.25, 1.0, 0.5, -0.5, -0.5, 0.25, 1.0, 0.5, 0.5, -0.5, 0.25, 1.0,
    ];
    let b_f32 = vec![
        1.0f32, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5, 0.5, -0.5, 1.0, 0.0, -0.5, 0.5, 0.0, 1.0,
    ];
    let bias_f32 = vec![0.1f32, 0.2, 0.3, 0.4];

    let alpha = 0.5f32;
    let beta = 0.5f32;
    let m = 4;
    let n = 4;
    let k = 4;

    // f32 reference
    let ref_out = gemm_f32_ref(
        &a_f32,
        &b_f32,
        &GemmRefCfg {
            m,
            n,
            k,
            alpha,
            beta,
            trans_a: false,
            trans_b: false,
            c: Some(&bias_f32),
        },
    );

    // Build typed F16 tensors
    let a = TypedTensor::new(TensorStorage::F16(f32_to_f16_bits(&a_f32)), vec![m, k]);
    let b = TypedTensor::new(TensorStorage::F16(f32_to_f16_bits(&b_f32)), vec![k, n]);
    let c = TypedTensor::new(TensorStorage::F16(f32_to_f16_bits(&bias_f32)), vec![n]);

    let node = gemm_node(alpha, beta, false, false);
    let ctx = make_typed_ctx_3(&node, &a, &b, &c);
    let result = GemmOp.execute_typed(&ctx).expect("F16 gemm failed");

    assert_eq!(result[0].dtype(), DType::F16);
    assert_eq!(result[0].shape, vec![m, n]);

    if let TensorStorage::F16(ref bits) = result[0].storage {
        let got = f16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-2,
                "F16 gemm output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected F16 storage, got {:?}", result[0].dtype());
    }
}

/// BF16 typed GEMM parity: same shape as F16 test, tolerance < 5e-2 absolute.
#[test]
fn test_gemm_bf16_parity() {
    assert!(
        GemmOp.native_dtypes().contains(&DType::BF16),
        "DType::BF16 must be in native_dtypes()"
    );

    let a_f32 = vec![
        1.0f32, 0.5, -0.5, 0.25, 0.25, 1.0, 0.5, -0.5, -0.5, 0.25, 1.0, 0.5, 0.5, -0.5, 0.25, 1.0,
    ];
    let b_f32 = vec![
        1.0f32, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5, 0.5, -0.5, 1.0, 0.0, -0.5, 0.5, 0.0, 1.0,
    ];
    let bias_f32 = vec![0.1f32, 0.2, 0.3, 0.4];

    let alpha = 0.5f32;
    let beta = 0.5f32;
    let m = 4;
    let n = 4;
    let k = 4;

    let ref_out = gemm_f32_ref(
        &a_f32,
        &b_f32,
        &GemmRefCfg {
            m,
            n,
            k,
            alpha,
            beta,
            trans_a: false,
            trans_b: false,
            c: Some(&bias_f32),
        },
    );

    let a = TypedTensor::new(TensorStorage::BF16(f32_to_bf16_bits(&a_f32)), vec![m, k]);
    let b = TypedTensor::new(TensorStorage::BF16(f32_to_bf16_bits(&b_f32)), vec![k, n]);
    let c = TypedTensor::new(TensorStorage::BF16(f32_to_bf16_bits(&bias_f32)), vec![n]);

    let node = gemm_node(alpha, beta, false, false);
    let ctx = make_typed_ctx_3(&node, &a, &b, &c);
    let result = GemmOp.execute_typed(&ctx).expect("BF16 gemm failed");

    assert_eq!(result[0].dtype(), DType::BF16);
    assert_eq!(result[0].shape, vec![m, n]);

    if let TensorStorage::BF16(ref bits) = result[0].storage {
        let got = bf16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (g - r).abs() < 5e-2,
                "BF16 gemm output[{i}]: got {g}, ref {r}, abs diff {}",
                (g - r).abs()
            );
        }
    } else {
        panic!("Expected BF16 storage, got {:?}", result[0].dtype());
    }
}

/// Assert GemmOp.native_dtypes() contains exactly F32, F16, BF16, I8, I32.
#[test]
fn test_gemm_native_dtypes_includes_all_five() {
    let dtypes = GemmOp.native_dtypes();
    let expected = [DType::F32, DType::F16, DType::BF16, DType::I8, DType::I32];
    for &dt in &expected {
        assert!(
            dtypes.contains(&dt),
            "GemmOp.native_dtypes() must contain {dt:?}"
        );
    }
    assert_eq!(
        dtypes.len(),
        expected.len(),
        "GemmOp.native_dtypes() should have exactly {} entries, got {}",
        expected.len(),
        dtypes.len()
    );
}
