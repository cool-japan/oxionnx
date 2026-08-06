//! `LpPool` and `GlobalLpPool` operator implementations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::conv::spatial::{
    odometer_next, parse_auto_pad, read_kernel_shape, read_pads, read_positive_spatial,
    resolve_pads, spatial_rank,
};
use crate::conv::PoolGeometry;

/// Read and validate the `p` attribute shared by `LpPool` / `GlobalLpPool`.
///
/// ONNX declares `p` as an **int** with default `2`. `p == 0` would make the
/// `1/p` exponent infinite, so it is a typed error rather than a NaN result.
fn read_p(ctx: &OpContext<'_>, op: &str) -> Result<f32, OnnxError> {
    let p = ctx.attrs().i("p", 2);
    if p < 1 {
        return Err(OnnxError::InvalidModel(format!(
            "{op}: p must be >= 1, got {p}"
        )));
    }
    Ok(p as f32)
}

/// `(Σ |v|^p)^(1/p)`, with the two overwhelmingly common exponents special-cased
/// so they stay exact instead of round-tripping through `powf`.
#[inline]
fn lp_norm(sum_abs_pow: f32, p: f32) -> f32 {
    if p == 1.0 {
        sum_abs_pow
    } else if p == 2.0 {
        sum_abs_pow.sqrt()
    } else {
        sum_abs_pow.powf(1.0 / p)
    }
}

/// `|v|^p`.
#[inline]
fn abs_pow(v: f32, p: f32) -> f32 {
    let a = v.abs();
    if p == 1.0 {
        a
    } else if p == 2.0 {
        a * a
    } else {
        a.powf(p)
    }
}

/// Row-major strides of the spatial axes of a `[N, C, d_0, …]` tensor.
fn spatial_strides(spatial: &[usize]) -> Vec<usize> {
    let rank = spatial.len();
    let mut strides = vec![1_usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * spatial[d + 1];
    }
    strides
}

// ── LpPool ──────────────────────────────────────────────────────────────────

/// ONNX `LpPool` (opset 1+, `dilations` / `ceil_mode` since 19).
///
/// ```text
/// Y[n, c, o…] = ( Σ_window |X|^p )^(1/p)
/// ```
///
/// Unlike `AveragePool` there is **no** division by the window size, and a
/// padded position contributes `0` (`|0|^p == 0`) — so `count_include_pad` has
/// no analogue here. (`onnx.reference` reaches the same result the long way
/// round, by scaling an average pool back up by the full kernel volume.)
///
/// Geometry (`kernel_shape`, `strides`, `pads`, `auto_pad`, `dilations`,
/// `ceil_mode`) is resolved through the same `PoolGeometry` the
/// `MaxPool`/`AveragePool` operators use, so the three can never disagree on
/// an output extent — and, like them, this operator is rank-generic.
pub struct LpPoolOp;

/// Resolve the pooling geometry of an `LpPool` node.
fn lp_pool_geometry(ctx: &OpContext<'_>, input_shape: &[usize]) -> Result<PoolGeometry, OnnxError> {
    let op = "LpPool";
    let attrs = ctx.attrs();
    let rank = spatial_rank(input_shape, op, "input")?;
    let kernel = read_kernel_shape(attrs.ints("kernel_shape"), rank, op)?;
    let strides = read_positive_spatial(attrs.ints("strides"), rank, 1, "strides", op)?;
    let dilations = read_positive_spatial(attrs.ints("dilations"), rank, 1, "dilations", op)?;
    let explicit = read_pads(attrs.ints("pads"), rank, op)?;
    let auto_pad = parse_auto_pad(attrs.s("auto_pad"), op)?;
    let ceil_mode = attrs.i("ceil_mode", 0) != 0;
    let pads = resolve_pads(
        auto_pad,
        &input_shape[2..],
        &kernel,
        &strides,
        &dilations,
        &explicit,
    );
    PoolGeometry::resolve(
        op,
        &input_shape[2..],
        kernel,
        strides,
        pads,
        dilations,
        ceil_mode,
    )
}

impl LpPoolOp {
    fn run(&self, ctx: &OpContext<'_>, out: &mut Tensor) -> Result<(), OnnxError> {
        let x = ctx.input(0)?;
        let p = read_p(ctx, "LpPool")?;
        let geo = lp_pool_geometry(ctx, &x.shape)?;

        let out_shape = geo.out_shape(&x.shape);
        let total: usize = out_shape.iter().product();
        let expected: usize = x.shape.iter().product();
        if x.data.len() != expected {
            return Err(OnnxError::ShapeMismatch(format!(
                "LpPool: data length {} does not match shape {:?}",
                x.data.len(),
                x.shape
            )));
        }
        if out.data.len() != total {
            out.data.resize(total, 0.0);
        }
        out.shape.clone_from(&out_shape);

        let rank = geo.rank();
        let in_spatial = &x.shape[2..];
        let in_stride = spatial_strides(in_spatial);
        let in_plane: usize = in_spatial.iter().product();
        let out_plane: usize = geo.out.iter().product();
        let k_volume: usize = geo.kernel.iter().product();
        let planes = x.shape[0] * x.shape[1];

        let mut oidx = vec![0_usize; rank];
        let mut kidx = vec![0_usize; rank];

        for nc in 0..planes {
            let plane = nc * in_plane;
            let out_base = nc * out_plane;
            oidx.iter_mut().for_each(|v| *v = 0);
            for o_flat in 0..out_plane {
                let mut sum = 0.0_f32;
                kidx.iter_mut().for_each(|v| *v = 0);
                for _ in 0..k_volume {
                    let mut off = 0_usize;
                    let mut inside = true;
                    for d in 0..rank {
                        let pos = oidx[d] * geo.strides[d] + kidx[d] * geo.dilations[d];
                        match pos.checked_sub(geo.pads[d]) {
                            Some(ip) if ip < in_spatial[d] => off += ip * in_stride[d],
                            _ => {
                                inside = false;
                                break;
                            }
                        }
                    }
                    if inside {
                        sum += abs_pow(x.data[plane + off], p);
                    }
                    odometer_next(&mut kidx, &geo.kernel);
                }
                out.data[out_base + o_flat] = lp_norm(sum, p);
                odometer_next(&mut oidx, &geo.out);
            }
        }
        Ok(())
    }
}

impl Operator for LpPoolOp {
    fn op_type(&self) -> &str {
        "LpPool"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut out = Tensor::zeros(&[0]);
        self.run(ctx, &mut out)?;
        Ok(vec![out])
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let [slot] = slots else {
            return Err(OnnxError::Internal(format!(
                "LpPoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        };
        self.run(ctx, slot)
    }
}

// ── GlobalLpPool ────────────────────────────────────────────────────────────

/// ONNX `GlobalLpPool` (opset 1+): the p-norm over every spatial position.
///
/// `[N, C, D1, …, Dk] -> [N, C, 1, …, 1]`, mirroring the shape convention of
/// `GlobalAveragePool` / `GlobalMaxPool`.
pub struct GlobalLpPoolOp;

impl GlobalLpPoolOp {
    fn run(&self, ctx: &OpContext<'_>, out: &mut Tensor) -> Result<(), OnnxError> {
        let x = ctx.input(0)?;
        let p = read_p(ctx, "GlobalLpPool")?;
        if x.ndim() < 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "GlobalLpPool: input must have rank >= 2 ([N, C, ...]), got {:?}",
                x.shape
            )));
        }
        let n = x.shape[0];
        let c = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        if x.data.len() != n * c * spatial {
            return Err(OnnxError::ShapeMismatch(format!(
                "GlobalLpPool: data length {} does not match shape {:?}",
                x.data.len(),
                x.shape
            )));
        }

        let mut out_shape = vec![n, c];
        out_shape.resize(x.ndim(), 1_usize);
        let total = n * c;
        if out.data.len() != total {
            out.data.resize(total, 0.0);
        }
        out.shape = out_shape;

        for ni in 0..n {
            for ci in 0..c {
                let base = (ni * c + ci) * spatial;
                let mut sum = 0.0_f32;
                for &v in &x.data[base..base + spatial] {
                    sum += abs_pow(v, p);
                }
                out.data[ni * c + ci] = lp_norm(sum, p);
            }
        }
        Ok(())
    }
}

impl Operator for GlobalLpPoolOp {
    fn op_type(&self) -> &str {
        "GlobalLpPool"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut out = Tensor::zeros(&[0]);
        self.run(ctx, &mut out)?;
        Ok(vec![out])
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let [slot] = slots else {
            return Err(OnnxError::Internal(format!(
                "GlobalLpPoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        };
        self.run(ctx, slot)
    }
}
