use super::*;
use oxionnx_core::OnnxError;
use oxionnx_core::Tensor;

#[test]
fn test_gather_embedding() -> Result<(), OnnxError> {
    // Embedding table: 5 tokens × 4 dims
    let table = Tensor::new((0..20).map(|i| i as f32).collect(), vec![5, 4]);
    // Indices: [2, 0, 4]
    let indices = Tensor::new(vec![2.0, 0.0, 4.0], vec![3]);
    let out = gather(&table, &indices, 0)?;
    assert_eq!(out.shape, vec![3, 4]);
    // token 2 → row 2 → [8, 9, 10, 11]
    assert_eq!(out.data[0], 8.0);
    assert_eq!(out.data[1], 9.0);
    // token 0 → [0, 1, 2, 3]
    assert_eq!(out.data[4], 0.0);
    Ok(())
}

#[test]
fn test_scatter_elements() -> Result<(), OnnxError> {
    let data = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
    let indices = Tensor::new(vec![1.0, 0.0], vec![1, 2]);
    let updates = Tensor::new(vec![7.0, 8.0], vec![1, 2]);
    let out = scatter_elements(&data, &indices, &updates, 0)?;
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data[2], 7.0); // row 1, col 0
    assert_eq!(out.data[1], 8.0); // row 0, col 1
    Ok(())
}

#[test]
fn test_scatter_nd() -> Result<(), OnnxError> {
    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let indices = Tensor::new(vec![0.0, 1.0], vec![1, 2]); // index [0][1]
    let updates = Tensor::new(vec![99.0, 100.0], vec![1, 2]); // update 2 values
    let out = scatter_nd(&data, &indices, &updates)?;
    assert_eq!(out.data[2], 99.0);
    assert_eq!(out.data[3], 100.0);
    Ok(())
}

#[test]
fn test_quantize_dequantize_roundtrip() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
    let scale = Tensor::new(vec![0.01], vec![1]);
    let q = quantize_linear(&x, &scale, None)?;
    // 0/0.01=0, 1/0.01=100, -1/0.01=-100, 2/0.01=127 (clamped)
    assert_eq!(q.data[0], 0.0);
    assert_eq!(q.data[1], 100.0);
    assert_eq!(q.data[2], -100.0);
    assert_eq!(q.data[3], 127.0);
    let dq = dequantize_linear(&q, &scale, None)?;
    assert!((dq.data[1] - 1.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_where() {
    let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
    let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let y = Tensor::new(vec![100.0, 200.0, 300.0], vec![3]);
    let out = where_op(&cond, &x, &y).expect("where_op failed");
    assert_eq!(out.data, vec![10.0, 200.0, 30.0]);
}

#[test]
fn test_gather_nd() {
    // data: [[0,1],[2,3],[4,5]], indices: [[0,0],[1,1]] -> [0, 3]
    let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]);
    let indices = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]);
    let out = gather_nd(&data, &indices, 0).expect("gather_nd failed");
    assert_eq!(out.shape, vec![2]);
    assert_eq!(out.data, vec![0.0, 3.0]);
}

#[test]
fn test_gather_nd_slice() {
    // data: [[0,1],[2,3],[4,5]], indices: [[0],[2]] -> [[0,1],[4,5]]
    let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]);
    let indices = Tensor::new(vec![0.0, 2.0], vec![2, 1]);
    let out = gather_nd(&data, &indices, 0).expect("gather_nd slice failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![0.0, 1.0, 4.0, 5.0]);
}

#[test]
fn test_gather_nd_negative_index() {
    let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]);
    // -1 means last row -> [4, 5]
    let indices = Tensor::new(vec![-1.0], vec![1, 1]);
    let out = gather_nd(&data, &indices, 0).expect("gather_nd neg failed");
    assert_eq!(out.data, vec![4.0, 5.0]);
}

#[test]
fn test_one_hot() {
    let indices = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let out = one_hot(&indices, 3, (0.0, 1.0), -1).expect("one_hot failed");
    assert_eq!(out.shape, vec![3, 3]);
    // Should be identity-like matrix
    assert_eq!(out.data[0], 1.0); // [0,0]
    assert_eq!(out.data[1], 0.0); // [0,1]
    assert_eq!(out.data[4], 1.0); // [1,1]
    assert_eq!(out.data[8], 1.0); // [2,2]
}

#[test]
fn test_one_hot_custom_values() {
    let indices = Tensor::new(vec![1.0], vec![1]);
    let out = one_hot(&indices, 3, (5.0, 10.0), 0).expect("one_hot custom failed");
    assert_eq!(out.shape, vec![3, 1]);
    assert_eq!(out.data, vec![5.0, 10.0, 5.0]);
}

#[test]
fn test_one_hot_negative_index() {
    let indices = Tensor::new(vec![-1.0], vec![1]);
    let out = one_hot(&indices, 3, (0.0, 1.0), -1).expect("one_hot neg idx failed");
    assert_eq!(out.shape, vec![1, 3]);
    // -1 + 3 = 2 -> last position
    assert_eq!(out.data, vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_compress() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0], vec![5]);
    let out = compress(&input, &cond, None).expect("compress failed");
    assert_eq!(out.data, vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_compress_with_axis() {
    // 2D input, compress along axis 0
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
    let out = compress(&input, &cond, Some(0)).expect("compress axis failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn test_compress_all_false() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let cond = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
    let out = compress(&input, &cond, None).expect("compress all false failed");
    assert_eq!(out.data.len(), 0);
    assert_eq!(out.shape, vec![0]);
}

#[test]
fn test_unique() {
    let x = Tensor::new(vec![2.0, 1.0, 1.0, 3.0, 2.0], vec![5]);
    let (unique_vals, _indices, inverse, counts) = unique(&x, None, true).expect("unique failed");
    assert_eq!(unique_vals.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(counts.data, vec![2.0, 2.0, 1.0]);
    // Verify inverse mapping
    for (i, &orig) in x.data.iter().enumerate() {
        let inv_idx = inverse.data[i] as usize;
        assert!(
            (unique_vals.data[inv_idx] - orig).abs() < f32::EPSILON,
            "inverse mismatch at {i}"
        );
    }
}

#[test]
fn test_unique_unsorted() {
    let x = Tensor::new(vec![3.0, 1.0, 3.0], vec![3]);
    let (unique_vals, indices, inverse, counts) =
        unique(&x, None, false).expect("unique unsorted failed");
    // First-seen order: 3.0, 1.0
    assert_eq!(unique_vals.data, vec![3.0, 1.0]);
    assert_eq!(indices.data, vec![0.0, 1.0]); // first occurrences
    assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
    assert_eq!(counts.data, vec![2.0, 1.0]);
}

#[test]
fn test_unique_all_same() {
    let x = Tensor::new(vec![5.0, 5.0, 5.0], vec![3]);
    let (unique_vals, _, _, counts) = unique(&x, None, true).expect("unique same failed");
    assert_eq!(unique_vals.data, vec![5.0]);
    assert_eq!(counts.data, vec![3.0]);
}

// ---- Axis-mode unique tests ----

#[test]
fn test_unique_axis0_1d() {
    // 1D with axis=0 should behave like value-level unique
    let x = Tensor::new(vec![3.0, 1.0, 3.0, 2.0, 1.0], vec![5]);
    let (unique_vals, indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique axis0 1d failed");
    // First occurrence order: 3.0, 1.0, 2.0
    assert_eq!(unique_vals.data, vec![3.0, 1.0, 2.0]);
    assert_eq!(unique_vals.shape, vec![3]);
    assert_eq!(indices.data, vec![0.0, 1.0, 3.0]);
    assert_eq!(inverse.data, vec![0.0, 1.0, 0.0, 2.0, 1.0]);
    assert_eq!(counts.data, vec![2.0, 2.0, 1.0]);
}

#[test]
fn test_unique_axis0_2d_rows() {
    // 3x2 matrix with duplicate rows
    // Row 0: [1, 2], Row 1: [3, 4], Row 2: [1, 2]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], vec![3, 2]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique axis0 2d failed");
    // Unique rows (first occurrence): row 0 [1,2], row 1 [3,4]
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(indices.data, vec![0.0, 1.0]);
    assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
    assert_eq!(counts.data, vec![2.0, 1.0]);
}

#[test]
fn test_unique_axis1_2d_columns() {
    // 2x4 matrix, columns: [1,5], [2,6], [1,5], [3,7]
    // Col 0 == Col 2
    let x = Tensor::new(vec![1.0, 2.0, 1.0, 3.0, 5.0, 6.0, 5.0, 7.0], vec![2, 4]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(1), false).expect("unique axis1 2d failed");
    // Unique columns (first occurrence): col 0 [1,5], col 1 [2,6], col 3 [3,7]
    assert_eq!(out.shape, vec![2, 3]);
    // Row 0: [1, 2, 3], Row 1: [5, 6, 7]
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);
    assert_eq!(indices.data, vec![0.0, 1.0, 3.0]);
    assert_eq!(inverse.data, vec![0.0, 1.0, 0.0, 2.0]);
    assert_eq!(counts.data, vec![2.0, 1.0, 1.0]);
}

#[test]
fn test_unique_axis0_sorted() {
    // Rows: [3,4], [1,2], [3,4]
    // Unsorted unique: [3,4] (idx 0), [1,2] (idx 1)
    // Sorted unique (lexicographic): [1,2], [3,4]
    let x = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![3, 2]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(0), true).expect("unique axis0 sorted failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(indices.data, vec![1.0, 0.0]); // row 1 first after sort, then row 0
    assert_eq!(inverse.data, vec![1.0, 0.0, 1.0]); // row 0->[3,4]->idx 1, row 1->[1,2]->idx 0
    assert_eq!(counts.data, vec![1.0, 2.0]);
}

#[test]
fn test_unique_axis0_unsorted() {
    // Same data, but unsorted preserves first-occurrence order
    let x = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![3, 2]);
    let (out, _indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique axis0 unsorted failed");
    assert_eq!(out.shape, vec![2, 2]);
    // First occurrence order: [3,4] then [1,2]
    assert_eq!(out.data, vec![3.0, 4.0, 1.0, 2.0]);
    assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
    assert_eq!(counts.data, vec![2.0, 1.0]);
}

#[test]
fn test_unique_axis_negative() {
    // 2x3 matrix, axis=-1 means axis=1 (columns)
    // Cols: [1,4], [2,5], [1,4] -> col 0 == col 2
    let x = Tensor::new(vec![1.0, 2.0, 1.0, 4.0, 5.0, 4.0], vec![2, 3]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(-1), false).expect("unique neg axis failed");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 4.0, 5.0]);
    assert_eq!(indices.data, vec![0.0, 1.0]);
    assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
    assert_eq!(counts.data, vec![2.0, 1.0]);
}

#[test]
fn test_unique_axis_single_element() {
    // 1x1 tensor with axis=0
    let x = Tensor::new(vec![42.0], vec![1, 1]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique single elem failed");
    assert_eq!(out.shape, vec![1, 1]);
    assert_eq!(out.data, vec![42.0]);
    assert_eq!(indices.data, vec![0.0]);
    assert_eq!(inverse.data, vec![0.0]);
    assert_eq!(counts.data, vec![1.0]);
}

#[test]
fn test_unique_axis_all_same_rows() {
    // 3x2, all rows identical
    let x = Tensor::new(vec![7.0, 8.0, 7.0, 8.0, 7.0, 8.0], vec![3, 2]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique all same rows failed");
    assert_eq!(out.shape, vec![1, 2]);
    assert_eq!(out.data, vec![7.0, 8.0]);
    assert_eq!(indices.data, vec![0.0]);
    assert_eq!(inverse.data, vec![0.0, 0.0, 0.0]);
    assert_eq!(counts.data, vec![3.0]);
}

#[test]
fn test_unique_axis_all_distinct() {
    // 3x2, all rows distinct
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique all distinct failed");
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(indices.data, vec![0.0, 1.0, 2.0]);
    assert_eq!(inverse.data, vec![0.0, 1.0, 2.0]);
    assert_eq!(counts.data, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_unique_axis_3d() {
    // 2x2x2 tensor, unique along axis=0
    // Slice 0: [[1,2],[3,4]], Slice 1: [[1,2],[3,4]] -> identical
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![2, 2, 2]);
    let (out, indices, inverse, counts) =
        unique(&x, Some(0), false).expect("unique 3d axis0 failed");
    assert_eq!(out.shape, vec![1, 2, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(indices.data, vec![0.0]);
    assert_eq!(inverse.data, vec![0.0, 0.0]);
    assert_eq!(counts.data, vec![2.0]);
}

#[test]
fn test_unique_axis_out_of_range() {
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let err = unique(&x, Some(1), false);
    assert!(err.is_err());
    let err_neg = unique(&x, Some(-2), false);
    assert!(err_neg.is_err());
}
