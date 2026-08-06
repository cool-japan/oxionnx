//! `CenterCropPad` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

/// ONNX `CenterCropPad` (opset 18+): crop or pad each named axis to a target
/// size, keeping the *original* content centered.
///
/// Inputs: `input_data` (any rank/shape) and `shape` -- a 1-D `int64` tensor
/// giving the target size for each axis named by the `axes` attribute
/// (default: every axis, in order `0..rank`).
///
/// For each such axis, comparing the current size `cur` against the target
/// `tgt`:
/// * `tgt < cur` -- **crop**: `excess = cur - tgt` is split as
///   `floor(excess/2)` removed from the start and the remainder
///   (`ceil(excess/2)`) removed from the end.
/// * `tgt > cur` -- **pad** with zeros: `diff = tgt - cur` is split as
///   `floor(diff/2)` added before and the remainder (`ceil(diff/2)`) added
///   after.
/// * `tgt == cur` -- no-op.
///
/// The "floor before / remainder after" rule is the same split in both
/// directions (`onnx.reference`'s `op_center_crop_pad.py` computes
/// `start = (actual - dim) // 2` for the crop case and
/// `pad_before = diff // 2` for the pad case -- both are a floor division on
/// the smaller/"before" side), which is what this implementation matches
/// exactly (see the module tests, generated from `onnx.reference`).
pub struct CenterCropPadOp;

impl Operator for CenterCropPadOp {
    fn op_type(&self) -> &str {
        "CenterCropPad"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let shape_t = ctx.input(1)?;
        if shape_t.ndim() > 1 {
            return Err(OnnxError::ShapeMismatch(format!(
                "CenterCropPad: 'shape' input must be 1-D, got shape {:?}",
                shape_t.shape
            )));
        }
        let rank = x.ndim();

        let axes_attr = ctx.attrs().ints("axes");
        let axes: Vec<usize> = if axes_attr.is_empty() {
            (0..rank).collect()
        } else {
            axes_attr
                .iter()
                .map(|&a| crate::indexing::normalize_axis(a, rank, "CenterCropPad"))
                .collect::<Result<_, String>>()?
        };
        if shape_t.data.len() != axes.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "CenterCropPad: 'shape' has {} entries but {} axes are named",
                shape_t.data.len(),
                axes.len()
            )));
        }

        let mut target_shape = x.shape.clone();
        for (&axis, &t) in axes.iter().zip(shape_t.data.iter()) {
            if !t.is_finite() || t < 0.0 || t > usize::MAX as f32 {
                return Err(OnnxError::InvalidModel(format!(
                    "CenterCropPad: target size {t} on axis {axis} is not a valid dimension"
                )));
            }
            target_shape[axis] = t as usize;
        }

        // Per-axis: the extent copied from the source (`copy_extent`, always
        // `min(cur, tgt)`), where that extent starts in the source
        // (`src_start`, nonzero only when cropping) and where it lands in
        // the destination (`dst_start`, nonzero only when padding).
        let mut src_start = vec![0usize; rank];
        let mut dst_start = vec![0usize; rank];
        let mut copy_extent = vec![0usize; rank];
        for axis in 0..rank {
            let cur = x.shape[axis];
            let tgt = target_shape[axis];
            copy_extent[axis] = cur.min(tgt);
            if tgt < cur {
                src_start[axis] = (cur - tgt) / 2;
            } else if tgt > cur {
                dst_start[axis] = (tgt - cur) / 2;
            }
        }

        let out_numel: usize = target_shape.iter().product();
        let mut out = vec![0.0_f32; out_numel];
        let copy_total: usize = copy_extent.iter().product();

        if copy_total > 0 {
            let mut coord = vec![0usize; rank];
            for _ in 0..copy_total {
                let mut src_flat = 0usize;
                let mut dst_flat = 0usize;
                for axis in 0..rank {
                    src_flat = src_flat * x.shape[axis] + (src_start[axis] + coord[axis]);
                    dst_flat = dst_flat * target_shape[axis] + (dst_start[axis] + coord[axis]);
                }
                out[dst_flat] = x.data[src_flat];
                // Odometer increment over `copy_extent`.
                for axis in (0..rank).rev() {
                    coord[axis] += 1;
                    if coord[axis] < copy_extent[axis] {
                        break;
                    }
                    coord[axis] = 0;
                }
            }
        }

        Ok(vec![Tensor::new(out, target_shape)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_node() -> oxionnx_core::Node {
        oxionnx_core::Node {
            name: "ccp".into(),
            op: oxionnx_core::OpKind::CenterCropPad,
            inputs: vec!["x".into(), "shape".into()],
            outputs: vec!["y".into()],
            attrs: oxionnx_core::Attributes::default(),
        }
    }

    fn run(x: &Tensor, shape: &Tensor, axes: Option<&[i64]>) -> Result<Tensor, OnnxError> {
        let mut node = dummy_node();
        if let Some(axes) = axes {
            node.attrs.int_lists.insert("axes".into(), axes.to_vec());
        }
        let ctx = OpContext {
            node: &node,
            inputs: vec![Some(x), Some(shape)],
            outer_scope: None,
            weights: None,
            registry: None,
        };
        Ok(CenterCropPadOp.execute(&ctx)?.remove(0))
    }

    fn arange(n: i64) -> Vec<f32> {
        (1..=n).map(|v| v as f32).collect()
    }

    /// Crop both axes, even excess. Reference: `onnx.reference`.
    #[test]
    fn crop_even_matches_onnx_reference() {
        let x = Tensor::new(arange(20), vec![4, 5]);
        let shape = Tensor::new(vec![2.0, 3.0], vec![2]);
        let out = run(&x, &shape, None).expect("run");
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![7.0, 8.0, 9.0, 12.0, 13.0, 14.0]);
    }

    /// Pad both axes, even deficit. Reference: `onnx.reference`.
    #[test]
    fn pad_even_matches_onnx_reference() {
        let x = Tensor::new(arange(6), vec![2, 3]);
        let shape = Tensor::new(vec![4.0, 5.0], vec![2]);
        let out = run(&x, &shape, None).expect("run");
        assert_eq!(out.shape, vec![4, 5]);
        assert_eq!(
            out.data,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 2.0, 3.0, 0.0, //
                0.0, 4.0, 5.0, 6.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
    }

    /// Mixed crop (axis 0, odd excess) + pad (axis 1, odd deficit) in the
    /// same call -- the case that pins "floor before / remainder after" in
    /// both directions simultaneously. Reference: `onnx.reference`.
    #[test]
    fn mixed_crop_and_pad_with_odd_amounts_matches_onnx_reference() {
        let x = Tensor::new(arange(20), vec![4, 5]);
        let shape = Tensor::new(vec![3.0, 8.0], vec![2]);
        let out = run(&x, &shape, None).expect("run");
        assert_eq!(out.shape, vec![3, 8]);
        assert_eq!(
            out.data,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, //
                0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, //
                0.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0,
            ]
        );
    }

    /// `axes = [1]`: axis 0 is left completely untouched.
    #[test]
    fn axes_attribute_restricts_to_named_axis() {
        let x = Tensor::new(arange(20), vec![4, 5]);
        let shape = Tensor::new(vec![3.0], vec![1]);
        let out = run(&x, &shape, Some(&[1])).expect("run");
        assert_eq!(out.shape, vec![4, 3]);
        assert_eq!(
            out.data,
            vec![2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0]
        );
    }

    /// Negative `axes` entries count from the end, and must agree with the
    /// positive spelling of the same axis.
    #[test]
    fn negative_axes_match_positive_equivalent() {
        let x = Tensor::new(arange(20), vec![4, 5]);
        let shape = Tensor::new(vec![3.0], vec![1]);
        let pos = run(&x, &shape, Some(&[1])).expect("run");
        let neg = run(&x, &shape, Some(&[-1])).expect("run");
        assert_eq!(pos.data, neg.data);
        assert_eq!(pos.shape, neg.shape);
    }

    /// 3-D input, crop only the middle axis via `axes`. Reference:
    /// `onnx.reference`.
    #[test]
    fn three_d_crop_middle_axis_matches_onnx_reference() {
        let x = Tensor::new(arange(24), vec![2, 4, 3]);
        let shape = Tensor::new(vec![2.0], vec![1]);
        let out = run(&x, &shape, Some(&[1])).expect("run");
        assert_eq!(out.shape, vec![2, 2, 3]);
        assert_eq!(
            out.data,
            vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
        );
    }

    /// Equal dims on every axis: exact no-op (byte-identical passthrough).
    #[test]
    fn equal_dims_is_a_no_op() {
        let x = Tensor::new(arange(6), vec![2, 3]);
        let shape = Tensor::new(vec![2.0, 3.0], vec![2]);
        let out = run(&x, &shape, None).expect("run");
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, x.data);
    }

    #[test]
    fn rejects_shape_length_mismatch() {
        let x = Tensor::new(arange(6), vec![2, 3]);
        let shape = Tensor::new(vec![2.0], vec![1]);
        let err = run(&x, &shape, None).expect_err("length mismatch must error");
        assert!(format!("{err}").contains("entries"), "got: {err}");
    }
}
