//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{Tensor, TensorLayout};

/// Convert a tensor from NCHW to NHWC layout.
/// Input shape: [N, C, H, W] -> Output shape: [N, H, W, C]
pub fn nchw_to_nhwc(tensor: &Tensor) -> Result<Tensor, String> {
    if tensor.shape.len() != 4 {
        return Err(format!(
            "nchw_to_nhwc: expected 4D tensor, got {}D",
            tensor.shape.len()
        ));
    }
    let (n, c, h, w) = (
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    );
    let mut out = vec![0.0f32; tensor.data.len()];
    for batch in 0..n {
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let src_idx = batch * c * h * w + ch * h * w + row * w + col;
                    let dst_idx = batch * h * w * c + row * w * c + col * c + ch;
                    out[dst_idx] = tensor.data[src_idx];
                }
            }
        }
    }
    Ok(Tensor::new(out, vec![n, h, w, c]))
}
/// Convert a tensor from NHWC to NCHW layout.
/// Input shape: [N, H, W, C] -> Output shape: [N, C, H, W]
pub fn nhwc_to_nchw(tensor: &Tensor) -> Result<Tensor, String> {
    if tensor.shape.len() != 4 {
        return Err(format!(
            "nhwc_to_nchw: expected 4D tensor, got {}D",
            tensor.shape.len()
        ));
    }
    let (n, h, w, c) = (
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    );
    let mut out = vec![0.0f32; tensor.data.len()];
    for batch in 0..n {
        for row in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    let src_idx = batch * h * w * c + row * w * c + col * c + ch;
                    let dst_idx = batch * c * h * w + ch * h * w + row * w + col;
                    out[dst_idx] = tensor.data[src_idx];
                }
            }
        }
    }
    Ok(Tensor::new(out, vec![n, c, h, w]))
}
/// Convert between tensor layouts.
pub fn convert_layout(
    tensor: &Tensor,
    from: TensorLayout,
    to: TensorLayout,
) -> Result<Tensor, String> {
    match (from, to) {
        (TensorLayout::NCHW, TensorLayout::NHWC) => nchw_to_nhwc(tensor),
        (TensorLayout::NHWC, TensorLayout::NCHW) => nhwc_to_nchw(tensor),
        (a, b) if a == b => Ok(tensor.clone()),
        _ => Err(format!(
            "Unsupported layout conversion: {:?} -> {:?}",
            from, to
        )),
    }
}
/// Compute C-order (row-major) strides from shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
/// Build a Tensor from raw f16 little-endian bytes (ONNX `raw_data` with float16 dtype).
pub fn from_f16_bytes(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    let data: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();
    Tensor::new(data, shape)
}
/// Build a Tensor from raw f32 little-endian bytes.
pub fn from_f32_bytes(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Tensor::new(data, shape)
}
/// Build a Tensor from raw i64 little-endian bytes (index tensors).
pub fn from_i64_bytes(bytes: &[u8], shape: Vec<usize>) -> Tensor {
    let data: Vec<f32> = bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
        .collect();
    Tensor::new(data, shape)
}
/// Build a Tensor from repeated float_data values.
pub fn from_f32_vec(floats: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::new(floats, shape)
}
/// Compute broadcast strides: if the original dim is 1 (broadcasted), stride is 0.
pub(super) fn broadcast_strides(original_shape: &[usize], broadcast_shape: &[usize]) -> Vec<usize> {
    let ndim = broadcast_shape.len();
    let pad = ndim - original_shape.len();
    let orig_strides = compute_strides(original_shape);
    (0..ndim)
        .map(|i| {
            if i < pad {
                0
            } else {
                let orig_idx = i - pad;
                if original_shape[orig_idx] == 1 {
                    0
                } else {
                    orig_strides[orig_idx]
                }
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::super::types::BroadcastIter;
    use super::*;
    #[test]
    fn test_broadcast_shape() {
        assert_eq!(
            Tensor::broadcast_shape(&[3, 1], &[1, 4]).expect("broadcast should succeed"),
            vec![3, 4]
        );
        assert_eq!(
            Tensor::broadcast_shape(&[1], &[4, 3]).expect("broadcast should succeed"),
            vec![4, 3]
        );
        assert!(Tensor::broadcast_shape(&[2], &[3]).is_err());
    }
    #[test]
    fn test_reshape() {
        let t = Tensor::zeros(&[2, 3]);
        let r = t.reshape(&[6]);
        assert_eq!(r.shape, vec![6]);
    }
    fn make_seq_tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::new(data, shape.to_vec())
    }
    #[test]
    fn test_view_basic() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.strides(), &[3, 1]);
        assert_eq!(v.ndim(), 2);
        assert_eq!(v.numel(), 6);
    }
    #[test]
    fn test_view_get() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view();
        assert_eq!(v.get(&[0, 0]), Some(0.0));
        assert_eq!(v.get(&[0, 2]), Some(2.0));
        assert_eq!(v.get(&[1, 0]), Some(3.0));
        assert_eq!(v.get(&[1, 2]), Some(5.0));
        assert_eq!(v.get(&[2, 0]), None);
        assert_eq!(v.get(&[0]), None);
    }
    #[test]
    fn test_view_is_contiguous() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view();
        assert!(v.is_contiguous());
        let tv = v.transpose(&[1, 0]);
        assert!(!tv.is_contiguous());
    }
    #[test]
    fn test_view_transpose() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view().transpose(&[1, 0]);
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.get(&[0, 0]), Some(0.0));
        assert_eq!(v.get(&[0, 1]), Some(3.0));
        assert_eq!(v.get(&[1, 0]), Some(1.0));
        assert_eq!(v.get(&[1, 1]), Some(4.0));
        assert_eq!(v.get(&[2, 0]), Some(2.0));
        assert_eq!(v.get(&[2, 1]), Some(5.0));
    }
    #[test]
    fn test_view_transpose_3d() {
        let t = make_seq_tensor(&[2, 3, 4]);
        let v = t.view().transpose(&[2, 0, 1]);
        assert_eq!(v.shape(), &[4, 2, 3]);
        assert_eq!(v.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(v.get(&[2, 0, 1]), Some(6.0));
        assert_eq!(v.get(&[3, 1, 2]), Some(23.0));
    }
    #[test]
    fn test_view_slice() {
        let t = make_seq_tensor(&[4, 3]);
        let v = t.view().slice(0, 1, 3);
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.get(&[0, 0]), Some(3.0));
        assert_eq!(v.get(&[0, 2]), Some(5.0));
        assert_eq!(v.get(&[1, 0]), Some(6.0));
        assert_eq!(v.get(&[1, 2]), Some(8.0));
    }
    #[test]
    fn test_view_select() {
        let t = make_seq_tensor(&[3, 4]);
        let v = t.view().select(0, 1);
        assert_eq!(v.shape(), &[4]);
        assert_eq!(v.get(&[0]), Some(4.0));
        assert_eq!(v.get(&[1]), Some(5.0));
        assert_eq!(v.get(&[2]), Some(6.0));
        assert_eq!(v.get(&[3]), Some(7.0));
    }
    #[test]
    fn test_view_squeeze() {
        let t = make_seq_tensor(&[1, 3, 1, 4]);
        let v = t.view().squeeze(&[0, 2]);
        assert_eq!(v.shape(), &[3, 4]);
        assert_eq!(v.numel(), 12);
        assert_eq!(v.get(&[0, 0]), Some(0.0));
        assert_eq!(v.get(&[2, 3]), Some(11.0));
    }
    #[test]
    fn test_view_unsqueeze() {
        let t = make_seq_tensor(&[3, 4]);
        let v = t.view().unsqueeze(&[0]);
        assert_eq!(v.shape(), &[1, 3, 4]);
        assert_eq!(v.numel(), 12);
        assert_eq!(v.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(v.get(&[0, 2, 3]), Some(11.0));
    }
    #[test]
    fn test_view_to_tensor() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view().transpose(&[1, 0]);
        let mat = v.to_tensor();
        assert_eq!(mat.shape, vec![3, 2]);
        assert_eq!(mat.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
    #[test]
    fn test_view_iter() {
        let t = make_seq_tensor(&[2, 3]);
        let v = t.view().transpose(&[1, 0]);
        let elems: Vec<f32> = v.iter().collect();
        assert_eq!(elems, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
    #[test]
    fn test_view_chained_ops() {
        let t = make_seq_tensor(&[4, 6]);
        let v = t.view().transpose(&[1, 0]).slice(0, 1, 4);
        assert_eq!(v.shape(), &[3, 4]);
        let mat = v.to_tensor();
        assert_eq!(mat.shape, vec![3, 4]);
        assert_eq!(
            mat.data,
            vec![1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0,]
        );
    }
    #[test]
    fn test_broadcast_iter_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[2, 3]);
        assert_eq!(iter.len(), 6);
        assert!(!iter.is_empty());
        let pairs: Vec<(f32, f32)> = iter.collect();
        assert_eq!(
            pairs,
            vec![
                (1.0, 10.0),
                (2.0, 20.0),
                (3.0, 30.0),
                (4.0, 40.0),
                (5.0, 50.0),
                (6.0, 60.0),
            ]
        );
    }
    #[test]
    fn test_broadcast_iter_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![100.0], vec![1]);
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[2, 3]);
        let pairs: Vec<(f32, f32)> = iter.collect();
        for (i, (av, bv)) in pairs.iter().enumerate() {
            assert!((*av - (i as f32 + 1.0)).abs() < 1e-6);
            assert!((*bv - 100.0).abs() < 1e-6);
        }
    }
    #[test]
    fn test_broadcast_iter_row_col() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]);
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[3, 4]);
        assert_eq!(iter.len(), 12);
        let pairs: Vec<(f32, f32)> = iter.collect();
        let expected = vec![
            (1.0, 10.0),
            (1.0, 20.0),
            (1.0, 30.0),
            (1.0, 40.0),
            (2.0, 10.0),
            (2.0, 20.0),
            (2.0, 30.0),
            (2.0, 40.0),
            (3.0, 10.0),
            (3.0, 20.0),
            (3.0, 30.0),
            (3.0, 40.0),
        ];
        assert_eq!(pairs, expected);
    }
    #[test]
    fn test_broadcast_iter_3d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 1, 4]);
        let b = Tensor::new(
            vec![
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
            ],
            vec![1, 3, 4],
        );
        let iter = BroadcastIter::new(&a, &b).expect("should be compatible");
        assert_eq!(iter.output_shape(), &[2, 3, 4]);
        assert_eq!(iter.len(), 24);
        let pairs: Vec<(f32, f32)> = iter.collect();
        assert_eq!(pairs[0], (1.0, 10.0));
        assert_eq!(pairs[4], (1.0, 50.0));
        assert_eq!(pairs[12], (5.0, 10.0));
        assert_eq!(pairs[23], (8.0, 120.0));
    }
    #[test]
    fn test_broadcast_iter_incompatible() {
        let a = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let b = Tensor::new(vec![1.0; 12], vec![4, 3]);
        assert!(BroadcastIter::new(&a, &b).is_none());
    }
    #[test]
    fn test_nchw_to_nhwc() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(data, vec![1, 2, 3, 4]);
        let nhwc = nchw_to_nhwc(&t).expect("conversion should succeed");
        assert_eq!(nhwc.shape, vec![1, 3, 4, 2]);
        assert!((nhwc.data[0] - 0.0).abs() < 1e-6);
        assert!((nhwc.data[1] - 12.0).abs() < 1e-6);
        assert!((nhwc.data[2] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_nhwc_to_nchw() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(data, vec![1, 3, 4, 2]);
        let nchw = nhwc_to_nchw(&t).expect("conversion should succeed");
        assert_eq!(nchw.shape, vec![1, 2, 3, 4]);
    }
    #[test]
    fn test_layout_roundtrip() {
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let original = Tensor::new(data.clone(), vec![2, 3, 2, 4]);
        let nhwc = nchw_to_nhwc(&original).expect("nchw_to_nhwc");
        let back = nhwc_to_nchw(&nhwc).expect("nhwc_to_nchw");
        assert_eq!(back.shape, original.shape);
        assert_eq!(back.data, original.data);
    }
    #[test]
    fn test_convert_layout_same() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let result =
            convert_layout(&t, TensorLayout::NCHW, TensorLayout::NCHW).expect("same layout");
        assert_eq!(result.data, t.data);
        assert_eq!(result.shape, t.shape);
    }
    #[test]
    fn test_non_4d_error() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(nchw_to_nhwc(&t).is_err());
        assert!(nhwc_to_nchw(&t).is_err());
        let t3d = Tensor::new(vec![1.0; 12], vec![2, 3, 2]);
        assert!(nchw_to_nhwc(&t3d).is_err());
    }
}
