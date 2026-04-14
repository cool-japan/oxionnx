//! Bitwise operators: BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot (ONNX opset 18).
//!
//! Tensors are stored as f32; bitwise semantics operate on the u32 bit-pattern.

use oxionnx_core::Tensor;

use crate::math::broadcast_to;

fn bitwise_binary(a: &Tensor, b: &Tensor, op: impl Fn(u32, u32) -> u32) -> Result<Tensor, String> {
    let target = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let ab = broadcast_to(a, &target);
    let bb = broadcast_to(b, &target);
    let data: Vec<f32> = ab
        .data
        .iter()
        .zip(bb.data.iter())
        .map(|(&x, &y)| op(x as u32, y as u32) as f32)
        .collect();
    Ok(Tensor::new(data, target))
}

pub fn bitwise_and(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    bitwise_binary(a, b, |x, y| x & y)
}

pub fn bitwise_or(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    bitwise_binary(a, b, |x, y| x | y)
}

pub fn bitwise_xor(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    bitwise_binary(a, b, |x, y| x ^ y)
}

pub fn bitwise_not(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data.iter().map(|&v| (!(v as u32)) as f32).collect();
    Tensor::new(data, x.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::Tensor;

    #[test]
    fn test_bitwise_and() {
        let a = Tensor::new(vec![5.0, 3.0], vec![2]);
        let b = Tensor::new(vec![3.0, 6.0], vec![2]);
        let out = bitwise_and(&a, &b).unwrap();
        assert_eq!(out.data[0] as u32, 5u32 & 3u32);
        assert_eq!(out.data[1] as u32, 3u32 & 6u32);
    }

    #[test]
    fn test_bitwise_or() {
        let a = Tensor::new(vec![5.0, 3.0], vec![2]);
        let b = Tensor::new(vec![3.0, 6.0], vec![2]);
        let out = bitwise_or(&a, &b).unwrap();
        assert_eq!(out.data[0] as u32, 5u32 | 3u32);
        assert_eq!(out.data[1] as u32, 3u32 | 6u32);
    }

    #[test]
    fn test_bitwise_xor() {
        let a = Tensor::new(vec![5.0], vec![1]);
        let b = Tensor::new(vec![3.0], vec![1]);
        let out = bitwise_xor(&a, &b).unwrap();
        assert_eq!(out.data[0] as u32, 5u32 ^ 3u32);
    }

    #[test]
    fn test_bitwise_not() {
        let x = Tensor::new(vec![0.0], vec![1]);
        let out = bitwise_not(&x);
        assert_eq!(out.data[0] as u32, !0u32);
    }

    #[test]
    fn test_bitwise_broadcast() {
        // Two same-shape tensors — shape is preserved
        let a = Tensor::new(vec![7.0, 7.0], vec![2]);
        let b = Tensor::new(vec![3.0, 5.0], vec![2]);
        let out = bitwise_and(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2]);
    }
}
