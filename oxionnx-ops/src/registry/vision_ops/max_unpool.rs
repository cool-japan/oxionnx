//! `MaxUnpool` operator implementation.

use oxionnx_core::{Attributes, OnnxError, OpContext, Operator, Tensor};

/// Spatial geometry of one `MaxUnpool` node, generalized over the spatial rank.
///
/// `MaxUnpool` has no `auto_pad` / `dilations` / `ceil_mode` attribute, so the
/// only geometry it needs is `kernel_shape` (required), `strides` (default 1)
/// and `pads` (default 0) — read here for an arbitrary number of spatial axes
/// rather than through the 2-D `PoolGeometry`.
struct UnpoolGeometry {
    /// Inferred output extents, one per spatial axis.
    spatial: Vec<usize>,
}

impl UnpoolGeometry {
    fn from_attrs(attrs: &Attributes, input_shape: &[usize]) -> Result<Self, OnnxError> {
        if input_shape.len() < 3 {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxUnpool: input must have rank >= 3 ([N, C, D1, ...]), got {:?}",
                input_shape
            )));
        }
        let rank = input_shape.len() - 2;
        let kernel = attrs.ints("kernel_shape");
        if kernel.len() != rank {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxUnpool: kernel_shape has {} entries, expected {rank} (one per spatial axis)",
                kernel.len()
            )));
        }
        let strides = attrs.ints("strides");
        if !strides.is_empty() && strides.len() != rank {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxUnpool: strides has {} entries, expected {rank}",
                strides.len()
            )));
        }
        let pads = attrs.ints("pads");
        if !pads.is_empty() && pads.len() != 2 * rank {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxUnpool: pads has {} entries, expected {}",
                pads.len(),
                2 * rank
            )));
        }

        let mut spatial = Vec::with_capacity(rank);
        for axis in 0..rank {
            let k = kernel[axis];
            if k < 1 {
                return Err(OnnxError::InvalidModel(format!(
                    "MaxUnpool: kernel_shape[{axis}] must be >= 1, got {k}"
                )));
            }
            let s = strides.get(axis).copied().unwrap_or(1);
            if s < 1 {
                return Err(OnnxError::InvalidModel(format!(
                    "MaxUnpool: strides[{axis}] must be >= 1, got {s}"
                )));
            }
            let pad_begin = pads.get(axis).copied().unwrap_or(0);
            let pad_end = pads.get(rank + axis).copied().unwrap_or(0);
            if pad_begin < 0 || pad_end < 0 {
                return Err(OnnxError::InvalidModel(format!(
                    "MaxUnpool: pads on axis {axis} must be >= 0, got [{pad_begin}, {pad_end}]"
                )));
            }
            // out = (in - 1) * stride - (pad_begin + pad_end) + kernel
            let dim = (input_shape[axis + 2] as i64 - 1) * s - (pad_begin + pad_end) + k;
            if dim < 1 {
                return Err(OnnxError::ShapeMismatch(format!(
                    "MaxUnpool: inferred output extent {dim} on axis {axis} is not positive"
                )));
            }
            spatial.push(dim as usize);
        }
        Ok(Self { spatial })
    }
}

/// Read the optional `output_shape` input (a 1-D int tensor) as dimensions.
fn read_output_shape(t: &Tensor, rank: usize) -> Result<Vec<usize>, OnnxError> {
    if t.data.len() != rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "MaxUnpool: output_shape has {} entries, expected {rank}",
            t.data.len()
        )));
    }
    t.data
        .iter()
        .map(|&v| {
            if !v.is_finite() || v < 1.0 || v > usize::MAX as f32 {
                Err(OnnxError::InvalidModel(format!(
                    "MaxUnpool: output_shape entry {v} is not a valid positive dimension"
                )))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

/// ONNX `MaxUnpool` (opset 9+): the inverse of `MaxPool`, scattering each value
/// back to the position its index names.
///
/// Inputs: `X` (the pooled values), `I` (the flat indices `MaxPool` produced)
/// and an optional `output_shape`.
///
/// The semantics follow the ONNX node test `test_maxunpool_export_with_output_shape`
/// (and `onnx.reference`) exactly: `I` indexes the tensor whose extents are
/// **inferred from `kernel_shape` / `strides` / `pads`**, and an explicit
/// `output_shape` only re-frames that tensor — the scattered block is placed at
/// the origin of the larger output and the remainder stays zero. Interpreting
/// `I` against `output_shape` directly would move every value.
///
/// An index outside the inferred tensor is a malformed model and produces a
/// typed error instead of an out-of-bounds write.
pub struct MaxUnpoolOp;

impl Operator for MaxUnpoolOp {
    fn op_type(&self) -> &str {
        "MaxUnpool"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let indices = ctx.input(1)?;
        if indices.data.len() != x.data.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxUnpool: I has {} elements but X has {}",
                indices.data.len(),
                x.data.len()
            )));
        }
        let geo = UnpoolGeometry::from_attrs(ctx.attrs(), &x.shape)?;

        let mut inferred = vec![x.shape[0], x.shape[1]];
        inferred.extend_from_slice(&geo.spatial);
        let inferred_total: usize = inferred.iter().product();

        let mut scattered = vec![0.0_f32; inferred_total];
        for (flat, (&value, &index)) in x.data.iter().zip(indices.data.iter()).enumerate() {
            if !index.is_finite() || index < 0.0 || index >= inferred_total as f32 {
                return Err(OnnxError::InvalidModel(format!(
                    "MaxUnpool: index {index} at position {flat} is outside the inferred \
                     output tensor of {inferred_total} elements (shape {inferred:?})"
                )));
            }
            scattered[index as usize] = value;
        }

        let Some(shape_input) = ctx.optional_input(2).filter(|t| !t.data.is_empty()) else {
            return Ok(vec![Tensor::new(scattered, inferred)]);
        };
        let requested = read_output_shape(shape_input, inferred.len())?;
        if requested == inferred {
            return Ok(vec![Tensor::new(scattered, inferred)]);
        }

        // Re-frame: copy the inferred block into the origin of the requested
        // tensor, cropping any axis the request makes smaller.
        let requested_total: usize = requested.iter().product();
        let mut out = vec![0.0_f32; requested_total];
        let copy_extent: Vec<usize> = inferred
            .iter()
            .zip(requested.iter())
            .map(|(&a, &b)| a.min(b))
            .collect();
        let copy_total: usize = copy_extent.iter().product();
        let rank = inferred.len();
        let mut coord = vec![0_usize; rank];
        for _ in 0..copy_total {
            let mut src = 0_usize;
            let mut dst = 0_usize;
            for axis in 0..rank {
                src = src * inferred[axis] + coord[axis];
                dst = dst * requested[axis] + coord[axis];
            }
            out[dst] = scattered[src];
            for axis in (0..rank).rev() {
                coord[axis] += 1;
                if coord[axis] < copy_extent[axis] {
                    break;
                }
                coord[axis] = 0;
            }
        }
        Ok(vec![Tensor::new(out, requested)])
    }
}
