//! `Col2Im` operator implementation.

use oxionnx_core::{Attributes, OnnxError, OpContext, Operator, Tensor};

/// Spatial geometry of one `Col2Im` node.
///
/// `Col2Im` has no `auto_pad`/`ceil_mode`; its geometry attributes
/// (`dilations`, `pads`, `strides`) mirror `Conv`'s but the "spatial rank" is
/// read off the `block_shape` *input* (a runtime tensor, not an attribute)
/// rather than off the data tensor's rank, since `Col2Im`'s data input is
/// always exactly 3-D (`[N, C * block_prod, L]`) regardless of how many
/// spatial dims the reconstructed image has.
struct Col2ImGeometry {
    ndim: usize,
    image_shape: Vec<usize>,
    block_shape: Vec<usize>,
    dilations: Vec<i64>,
    strides: Vec<i64>,
    /// Padding, length `2 * ndim`: `[begin_0..begin_{ndim-1}, end_0..end_{ndim-1}]`.
    pads: Vec<i64>,
    /// Number of sliding-block positions per spatial axis (`L_i`).
    l_dims: Vec<usize>,
}

/// Read a 1-D tensor of positive dimension values (as `usize`).
fn read_positive_dims(t: &Tensor, what: &str) -> Result<Vec<usize>, OnnxError> {
    if t.ndim() > 1 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{what} must be a 1-D tensor, got shape {:?}",
            t.shape
        )));
    }
    t.data
        .iter()
        .map(|&v| {
            if !v.is_finite() || v < 1.0 || v > usize::MAX as f32 {
                Err(OnnxError::InvalidModel(format!(
                    "{what}: entry {v} is not a valid positive dimension"
                )))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

/// Read an `ints` attribute expected to have exactly `ndim` entries, each
/// `>= 1`, defaulting every entry to `default` when the attribute is absent.
fn read_geo_ints(
    attrs: &Attributes,
    name: &str,
    ndim: usize,
    default: i64,
    op: &str,
) -> Result<Vec<i64>, OnnxError> {
    let raw = attrs.ints(name);
    if raw.is_empty() {
        return Ok(vec![default; ndim]);
    }
    if raw.len() != ndim {
        return Err(OnnxError::InvalidModel(format!(
            "{op}: '{name}' has {} entries, expected {ndim}",
            raw.len()
        )));
    }
    for &v in raw {
        if v < 1 {
            return Err(OnnxError::InvalidModel(format!(
                "{op}: '{name}' entries must be >= 1, got {v}"
            )));
        }
    }
    Ok(raw.to_vec())
}

impl Col2ImGeometry {
    fn from_ctx(
        attrs: &Attributes,
        image_shape_t: &Tensor,
        block_shape_t: &Tensor,
    ) -> Result<Self, OnnxError> {
        let image_shape = read_positive_dims(image_shape_t, "Col2Im: image_shape")?;
        let block_shape = read_positive_dims(block_shape_t, "Col2Im: block_shape")?;
        let ndim = image_shape.len();
        if block_shape.len() != ndim {
            return Err(OnnxError::ShapeMismatch(format!(
                "Col2Im: image_shape has {ndim} entries but block_shape has {} \
                 (both must name the same number of spatial axes)",
                block_shape.len()
            )));
        }
        if ndim == 0 {
            return Err(OnnxError::InvalidModel(
                "Col2Im: image_shape/block_shape must have at least one spatial axis".into(),
            ));
        }

        let dilations = read_geo_ints(attrs, "dilations", ndim, 1, "Col2Im")?;
        let strides = read_geo_ints(attrs, "strides", ndim, 1, "Col2Im")?;

        let pads_raw = attrs.ints("pads");
        let pads: Vec<i64> = if pads_raw.is_empty() {
            vec![0; 2 * ndim]
        } else {
            if pads_raw.len() != 2 * ndim {
                return Err(OnnxError::InvalidModel(format!(
                    "Col2Im: 'pads' has {} entries, expected {}",
                    pads_raw.len(),
                    2 * ndim
                )));
            }
            if pads_raw.iter().any(|&v| v < 0) {
                return Err(OnnxError::InvalidModel(
                    "Col2Im: 'pads' entries must be >= 0".into(),
                ));
            }
            pads_raw.to_vec()
        };

        let mut l_dims = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let eff_kernel = dilations[i] * (block_shape[i] as i64 - 1) + 1;
            let numerator = image_shape[i] as i64 + pads[i] + pads[ndim + i] - eff_kernel;
            if numerator < 0 {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Col2Im: axis {i}: image_shape + pads is smaller than the effective \
                     (dilated) block_shape"
                )));
            }
            let l_i = numerator / strides[i] + 1;
            l_dims.push(l_i as usize);
        }

        Ok(Self {
            ndim,
            image_shape,
            block_shape,
            dilations,
            strides,
            pads,
            l_dims,
        })
    }
}

/// Decompose a row-major flat index into per-axis coordinates against `dims`,
/// writing into `out` (`out.len() == dims.len()`).
fn unflatten_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    for (axis, &d) in dims.iter().enumerate().rev() {
        out[axis] = flat % d;
        flat /= d;
    }
}

/// ONNX `Col2Im` (opset 18+): combine sliding local blocks into a large
/// tensor -- the (accumulating) inverse of the sliding-window column
/// extraction `Conv` performs internally (a.k.a. "fold", the operation
/// `torch.nn.Fold` implements).
///
/// Inputs: `X` -- shape `[N, C * prod(block_shape), L]`; `image_shape` and
/// `block_shape` -- 1-D `int64` tensors naming the spatial rank. Output --
/// shape `[N, C, image_shape[0], .., image_shape[ndim-1]]`.
///
/// Overlapping block positions **accumulate** (sum), matching `Fold`/the
/// ONNX spec exactly -- this is what distinguishes `Col2Im` from a plain
/// reshape/scatter, and is exercised by this module's
/// `col2im_overlapping_blocks_accumulate_matches_onnx_reference` test.
pub struct Col2ImOp;

impl Operator for Col2ImOp {
    fn op_type(&self) -> &str {
        "Col2Im"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let image_shape_t = ctx.input(1)?;
        let block_shape_t = ctx.input(2)?;
        let geo = Col2ImGeometry::from_ctx(ctx.attrs(), image_shape_t, block_shape_t)?;

        if x.ndim() != 3 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Col2Im: input must be 3-D [N, C*prod(block_shape), L], got shape {:?}",
                x.shape
            )));
        }
        let n_batch = x.shape[0];
        let block_prod: usize = geo.block_shape.iter().product();
        let l_total: usize = geo.l_dims.iter().product();
        if x.shape[2] != l_total {
            return Err(OnnxError::ShapeMismatch(format!(
                "Col2Im: input's last dimension is {} but the geometry implies L = {l_total}",
                x.shape[2]
            )));
        }
        if x.shape[1] % block_prod != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Col2Im: input's channel dimension {} is not a multiple of prod(block_shape) = {block_prod}",
                x.shape[1]
            )));
        }
        let c = x.shape[1] / block_prod;
        let channel_dim = x.shape[1];

        let mut out_shape = Vec::with_capacity(2 + geo.ndim);
        out_shape.push(n_batch);
        out_shape.push(c);
        out_shape.extend_from_slice(&geo.image_shape);
        let image_total: usize = geo.image_shape.iter().product();
        let mut out = vec![0.0_f32; n_batch * c * image_total];

        let mut k_coord = vec![0usize; geo.ndim];
        let mut l_coord = vec![0usize; geo.ndim];
        let mut out_coord = vec![0usize; geo.ndim];
        for flat_k in 0..block_prod {
            unflatten_into(flat_k, &geo.block_shape, &mut k_coord);
            for flat_l in 0..l_total {
                unflatten_into(flat_l, &geo.l_dims, &mut l_coord);

                let mut in_bounds = true;
                let mut out_spatial_flat = 0usize;
                for ax in 0..geo.ndim {
                    let p = l_coord[ax] as i64 * geo.strides[ax]
                        + k_coord[ax] as i64 * geo.dilations[ax]
                        - geo.pads[ax];
                    if p < 0 || p as usize >= geo.image_shape[ax] {
                        in_bounds = false;
                        break;
                    }
                    out_coord[ax] = p as usize;
                    out_spatial_flat = out_spatial_flat * geo.image_shape[ax] + out_coord[ax];
                }
                if !in_bounds {
                    continue;
                }

                for n in 0..n_batch {
                    for ci in 0..c {
                        let in_channel = ci * block_prod + flat_k;
                        let in_idx = (n * channel_dim + in_channel) * l_total + flat_l;
                        let out_idx = (n * c + ci) * image_total + out_spatial_flat;
                        out[out_idx] += x.data[in_idx];
                    }
                }
            }
        }

        Ok(vec![Tensor::new(out, out_shape)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_node() -> oxionnx_core::Node {
        oxionnx_core::Node {
            name: "col2im".into(),
            op: oxionnx_core::OpKind::Col2Im,
            inputs: vec!["x".into(), "image_shape".into(), "block_shape".into()],
            outputs: vec!["y".into()],
            attrs: Attributes::default(),
        }
    }

    fn run(
        x: &Tensor,
        image_shape: &Tensor,
        block_shape: &Tensor,
        attrs: Attributes,
    ) -> Result<Tensor, OnnxError> {
        let mut node = dummy_node();
        node.attrs = attrs;
        let ctx = OpContext {
            node: &node,
            inputs: vec![Some(x), Some(image_shape), Some(block_shape)],
            outer_scope: None,
            weights: None,
            registry: None,
        };
        Ok(Col2ImOp.execute(&ctx)?.remove(0))
    }

    fn attrs_with_strides(strides: &[i64]) -> Attributes {
        let mut a = Attributes::default();
        a.int_lists.insert("strides".into(), strides.to_vec());
        a
    }

    /// Non-overlapping tiling: 4x4 image, 2x2 blocks, stride 2 -- each output
    /// pixel receives exactly one contribution, so this is Col2Im's simplest
    /// "no accumulation" case. Reference: `onnx.reference` `Col2Im` (opset 21).
    #[test]
    fn col2im_non_overlapping_tiling_matches_onnx_reference() {
        let x = Tensor::new((1..=16).map(|v| v as f32).collect(), vec![1, 4, 4]);
        let image_shape = Tensor::new(vec![4.0, 4.0], vec![2]);
        let block_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        let out = run(&x, &image_shape, &block_shape, attrs_with_strides(&[2, 2])).expect("run");
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
        assert_eq!(
            out.data,
            vec![
                1.0, 5.0, 2.0, 6.0, //
                9.0, 13.0, 10.0, 14.0, //
                3.0, 7.0, 4.0, 8.0, //
                11.0, 15.0, 12.0, 16.0,
            ]
        );
    }

    /// Overlapping blocks (3x3 image, 2x2 blocks, stride 1, L=4): output
    /// positions touched by more than one sliding window must **sum** the
    /// contributions, not overwrite. Reference: `onnx.reference`.
    #[test]
    fn col2im_overlapping_blocks_accumulate_matches_onnx_reference() {
        let x = Tensor::new((1..=16).map(|v| v as f32).collect(), vec![1, 4, 4]);
        let image_shape = Tensor::new(vec![3.0, 3.0], vec![2]);
        let block_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        let out = run(&x, &image_shape, &block_shape, attrs_with_strides(&[1, 1])).expect("run");
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        assert_eq!(
            out.data,
            vec![1.0, 7.0, 6.0, 12.0, 34.0, 22.0, 11.0, 27.0, 16.0]
        );
    }

    /// `pads = [1,1,1,1]` on a 3x3 image / 2x2 block / stride 1 (L=4 per
    /// axis, 16 total). Reference: `onnx.reference`.
    #[test]
    fn col2im_with_padding_matches_onnx_reference() {
        let x = Tensor::new((1..=64).map(|v| v as f32).collect(), vec![1, 4, 16]);
        let image_shape = Tensor::new(vec![3.0, 3.0], vec![2]);
        let block_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        let mut attrs = attrs_with_strides(&[1, 1]);
        attrs.int_lists.insert("pads".into(), vec![1, 1, 1, 1]);
        let out = run(&x, &image_shape, &block_shape, attrs).expect("run");
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        assert_eq!(
            out.data,
            vec![110.0, 114.0, 118.0, 126.0, 130.0, 134.0, 142.0, 146.0, 150.0]
        );
    }

    /// `block_shape = [1, 1]`, `stride = [1, 1]`: every "block" is a single
    /// pixel with no overlap, so with `C = 2` this degenerates to a pure
    /// reshape -- and pins that channel `ci` reads input rows
    /// `[ci*block_prod, (ci+1)*block_prod)`, not an interleaved layout.
    #[test]
    fn col2im_trivial_reshape_pins_channel_layout() {
        let x = Tensor::new((1..=8).map(|v| v as f32).collect(), vec![1, 2, 4]);
        let image_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        let block_shape = Tensor::new(vec![1.0, 1.0], vec![2]);
        let out = run(&x, &image_shape, &block_shape, attrs_with_strides(&[1, 1])).expect("run");
        assert_eq!(out.shape, vec![1, 2, 2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// `dilations = [2, 2]`: the effective (dilated) kernel spans further
    /// than `block_shape` itself, spacing out each block's contributions.
    /// Reference: `onnx.reference`.
    #[test]
    fn col2im_dilation_matches_onnx_reference() {
        let x = Tensor::new((1..=36).map(|v| v as f32).collect(), vec![1, 4, 9]);
        let image_shape = Tensor::new(vec![5.0, 5.0], vec![2]);
        let block_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        let mut attrs = attrs_with_strides(&[1, 1]);
        attrs.int_lists.insert("dilations".into(), vec![2, 2]);
        let out = run(&x, &image_shape, &block_shape, attrs).expect("run");
        assert_eq!(out.shape, vec![1, 1, 5, 5]);
        assert_eq!(
            out.data,
            vec![
                1.0, 2.0, 13.0, 11.0, 12.0, //
                4.0, 5.0, 19.0, 14.0, 15.0, //
                26.0, 28.0, 74.0, 46.0, 48.0, //
                22.0, 23.0, 55.0, 32.0, 33.0, //
                25.0, 26.0, 61.0, 35.0, 36.0,
            ]
        );
    }

    /// **Channel/block ordering discriminator**: `C = 2`, `block = [2,2]`,
    /// `image = [3,3]`, `stride = [1,1]` (input shape `(1, 8, 4)`). Every
    /// other case here has `C == 1` or `block_prod == 1`, so both plausible
    /// input-channel orderings (`ci*block_prod + flat_k` vs.
    /// `flat_k*C + ci`) agree on them; only a genuine `C > 1` with
    /// `block_prod > 1` case can tell them apart. Reference:
    /// `onnx.reference`.
    #[test]
    fn col2im_multi_channel_pins_channel_major_block_layout() {
        let x = Tensor::new((1..=32).map(|v| v as f32).collect(), vec![1, 8, 4]);
        let image_shape = Tensor::new(vec![3.0, 3.0], vec![2]);
        let block_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        let out = run(&x, &image_shape, &block_shape, attrs_with_strides(&[1, 1])).expect("run");
        assert_eq!(out.shape, vec![1, 2, 3, 3]);
        assert_eq!(
            out.data,
            vec![
                1.0, 7.0, 6.0, 12.0, 34.0, 22.0, 11.0, 27.0, 16.0, //
                17.0, 39.0, 22.0, 44.0, 98.0, 54.0, 27.0, 59.0, 32.0,
            ]
        );
    }

    #[test]
    fn col2im_rejects_wrong_l() {
        let x = Tensor::new(vec![0.0; 16], vec![1, 4, 4]);
        let image_shape = Tensor::new(vec![4.0, 4.0], vec![2]);
        let block_shape = Tensor::new(vec![2.0, 2.0], vec![2]);
        // stride=1 on a 4x4 image with 2x2 blocks implies L=9, not 4.
        let err = run(&x, &image_shape, &block_shape, attrs_with_strides(&[1, 1]))
            .expect_err("wrong L must error");
        assert!(format!("{err}").contains('L'), "got: {err}");
    }
}
