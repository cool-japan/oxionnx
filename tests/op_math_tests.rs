//! Math operator integration tests: MatMul, Gemm, and Reduce family.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_tensor_approx, run_single_op};

// ═══════════════════════════════════════════════════════════════════════════════
// MatMul
// ═══════════════════════════════════════════════════════════════════════════════

// 1. test_matmul_2d - [2,3] x [3,4] = [2,4]
#[test]
fn test_matmul_2d() {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  shape [2,3]
    // B = [[1, 0, 1, 0],
    //      [0, 1, 0, 1],
    //      [1, 1, 1, 1]] shape [3,4]
    // C = A @ B =
    //   row0: [1*1+2*0+3*1, 1*0+2*1+3*1, 1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5, 4, 5]
    //   row1: [4*1+5*0+6*1, 4*0+5*1+6*1, 4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11, 10, 11]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![3, 4],
    );
    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    assert_tensor_approx(out, &[4.0, 5.0, 4.0, 5.0, 10.0, 11.0, 10.0, 11.0], 1e-5);
}

// 2. test_matmul_batched - [2,3,4] x [2,4,5] = [2,3,5]
#[test]
fn test_matmul_batched() {
    // batch=2, M=3, K=4, N=5
    // A[0] = ones(3,4), A[1] = 2*ones(3,4)
    let mut a_data = vec![1.0f32; 12]; // batch 0
    a_data.extend(vec![2.0f32; 12]); // batch 1
    let a = Tensor::new(a_data, vec![2, 3, 4]);

    // B[0] = ones(4,5), B[1] = ones(4,5)
    let b_data = vec![1.0f32; 40]; // 2*4*5
    let b = Tensor::new(b_data, vec![2, 4, 5]);

    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3, 5]);

    // batch 0: ones(3,4) @ ones(4,5) = 4*ones(3,5)
    for i in 0..15 {
        assert!(
            (out.data[i] - 4.0).abs() < 1e-5,
            "batch0 idx {}: {} vs 4.0",
            i,
            out.data[i]
        );
    }
    // batch 1: 2*ones(3,4) @ ones(4,5) = 8*ones(3,5)
    for i in 15..30 {
        assert!(
            (out.data[i] - 8.0).abs() < 1e-5,
            "batch1 idx {}: {} vs 8.0",
            i,
            out.data[i]
        );
    }
}

// 16. test_matmul_batch1 - batch=1 MatMul
#[test]
fn test_matmul_batch1() {
    // [1,2,3] @ [1,3,2] = [1,2,2]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    // row0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // row1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]

    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 2, 2]);
    assert_tensor_approx(out, &[4.0, 5.0, 10.0, 11.0], 1e-5);
}

// 17. test_matmul_batch4 - batch=4 MatMul
#[test]
fn test_matmul_batch4() {
    // [4,2,2] @ [4,2,2] = [4,2,2]
    // Use identity matrices for all batches => output = input A
    let eye = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let a_data: Vec<f32> = (0..4)
        .flat_map(|batch| {
            let b = batch as f32;
            vec![b + 1.0, b + 2.0, b + 3.0, b + 4.0]
        })
        .collect();
    let b_data: Vec<f32> = eye.iter().copied().cycle().take(16).collect();

    let a = Tensor::new(a_data.clone(), vec![4, 2, 2]);
    let b = Tensor::new(b_data, vec![4, 2, 2]);

    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![4, 2, 2]);
    // A @ I = A
    assert_tensor_approx(out, &a_data, 1e-5);
}

// 22. test_matmul_small - [1,3] x [3,4] = [1,4] (smallest useful 2D matmul)
#[test]
fn test_matmul_small() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![3, 4],
    );
    // [1,2,3] @ partial-identity => [1, 2, 3, 0]
    let outputs = run_single_op(
        OpKind::MatMul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);
    assert_tensor_approx(out, &[1.0, 2.0, 3.0, 0.0], 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gemm
// ═══════════════════════════════════════════════════════════════════════════════

// 3. test_gemm_transB - Gemm with transB=1, alpha, beta
#[test]
fn test_gemm_trans_b() {
    // A = [[1, 2],
    //      [3, 4]] shape [2,2]
    // B = [[1, 3],
    //      [2, 4]] shape [2,2] => transB => B^T = [[1, 2], [3, 4]]
    // C = [10, 20] shape [2]
    // Y = alpha * A @ B^T + beta * C  with alpha=0.5, beta=2.0
    // A @ B^T = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7, 10], [15, 22]]
    // 0.5 * [[7, 10], [15, 22]] = [[3.5, 5.0], [7.5, 11.0]]
    // + 2.0 * [10, 20] broadcast = + [[20, 40], [20, 40]]
    // = [[23.5, 45.0], [27.5, 51.0]]
    let mut attrs = Attributes::default();
    attrs.ints.insert("transB".to_string(), 1);
    attrs.floats.insert("alpha".to_string(), 0.5);
    attrs.floats.insert("beta".to_string(), 2.0);

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
    let c = Tensor::new(vec![10.0, 20.0], vec![2]);

    let outputs = run_single_op(
        OpKind::Gemm,
        vec![("a", a), ("b", b)],
        vec![("c", c)],
        vec!["a", "b"],
        vec!["a", "b", "c"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert_tensor_approx(out, &[23.5, 45.0, 27.5, 51.0], 1e-4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Reduce ops
// ═══════════════════════════════════════════════════════════════════════════════

// 4. test_reduce_mean_axis - ReduceMean along axis 1
#[test]
fn test_reduce_mean_axis() {
    // x = [[1, 2, 3],
    //      [4, 5, 6]] shape [2,3]
    // ReduceMean axis=1, keepdims=0 => [2, 5]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceMean,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[2.0, 5.0], 1e-5);
}

// 5. test_reduce_sum_keepdims - ReduceSum with keepdims=1
#[test]
fn test_reduce_sum_keepdims() {
    // x = [[1, 2, 3],
    //      [4, 5, 6]] shape [2,3]
    // ReduceSum axis=1, keepdims=1 => [[6], [15]] shape [2,1]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 1);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceSum,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 1]);
    assert_tensor_approx(out, &[6.0, 15.0], 1e-5);
}

// test_reduce_max
#[test]
fn test_reduce_max() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    // x = [[1, 5, 3], [4, 2, 6]] shape [2,3]
    let x = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceMax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[5.0, 6.0], 1e-5);
}

// test_reduce_min
#[test]
fn test_reduce_min() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::ReduceMin,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[1.0, 2.0], 1e-5);
}

// test_reduce_l1
#[test]
fn test_reduce_l1() {
    // input [[1,2],[3,4]] shape [2,2], axes=[1], keepdims=false → [|1|+|2|, |3|+|4|] = [3, 7]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let outputs = run_single_op(
        OpKind::ReduceL1,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[3.0, 7.0], 1e-5);
}

// test_reduce_l2
#[test]
fn test_reduce_l2() {
    // input [3, 4] shape [2], axes=[] (reduce all) → sqrt(9+16) = 5.0
    let attrs = Attributes::default(); // empty axes = reduce all
    let x = Tensor::new(vec![3.0, 4.0], vec![2]);
    let outputs = run_single_op(
        OpKind::ReduceL2,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[5.0], 1e-5);
}

// test_reduce_log_sum
#[test]
fn test_reduce_log_sum() {
    // input [1, 1, 1] shape [3], axes=[] → log(1+1+1) = ln(3) ≈ 1.0986
    let attrs = Attributes::default();
    let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let outputs = run_single_op(
        OpKind::ReduceLogSum,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    let expected = 3.0f32.ln();
    assert!(
        (out.data[0] - expected).abs() < 1e-5,
        "reduce_log_sum: got {}, expected {}",
        out.data[0],
        expected
    );
}

// test_reduce_log_sum_exp
#[test]
fn test_reduce_log_sum_exp() {
    // input [0, 1, 2] shape [3], axes=[] → log(e^0 + e^1 + e^2)
    let attrs = Attributes::default();
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let outputs = run_single_op(
        OpKind::ReduceLogSumExp,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    let expected = (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp()).ln();
    assert!(
        (out.data[0] - expected).abs() < 1e-4,
        "reduce_log_sum_exp: got {}, expected {}",
        out.data[0],
        expected
    );
}

// test_reduce_sum_square
#[test]
fn test_reduce_sum_square() {
    // input [2, 3, 4] shape [3], axes=[] → 4+9+16 = 29.0
    let attrs = Attributes::default();
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let outputs = run_single_op(
        OpKind::ReduceSumSquare,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[29.0], 1e-5);
}
