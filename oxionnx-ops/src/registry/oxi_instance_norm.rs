//! `OxiInstanceNorm` — the optimizer-generated spatial normalization kernel.
//!
//! # What this op is, and why it is not `InstanceNormalization`
//!
//! `OxiInstanceNorm(X) = (X - mean) / sqrt(var + epsilon)`, where `mean` and
//! `var` are taken over `X`'s spatial axes (`shape[2..]`) independently for
//! every `(n, c)` pair. Output shape equals input shape; there is exactly one
//! input and one attribute (`epsilon`).
//!
//! ONNX's `InstanceNormalization` computes the same normalisation but *also*
//! applies a mandatory per-channel affine term (`scale * norm + B`), and its
//! `scale`/`B` operands are required inputs. AdaIN-style generators (the
//! inswapper face-swap decoder is the motivating case) do not fit that shape:
//! their per-channel scale and shift are *runtime* tensors produced by a Gemm
//! head from the identity embedding, not initialisers, and PyTorch exports the
//! normalisation itself as a bare arithmetic chain
//!
//! ```text
//! ReduceMean(axes=[2,3]) → Sub → Mul(diff,diff) → ReduceMean(axes=[2,3])
//!   → Add(eps) → Sqrt → Div(1, ·) → Mul(diff, ·)
//! ```
//!
//! with the affine part following as two ordinary broadcast nodes. Folding the
//! affine into this op would mean either requiring operands the graph does not
//! have in the right form, or duplicating the broadcast semantics of the
//! trailing `Mul`/`Add`; so this op deliberately covers the normalisation only
//! and `crate::optimizer::fusion::fuse_instance_norm` leaves the affine nodes
//! standing.
//!
//! # Numerics
//!
//! Variance is computed two-pass (`mean((x - mean)^2)`), not as
//! `mean(x^2) - mean(x)^2`. That is the arithmetic the graph itself performs,
//! so the fused node agrees with the chain it replaces structurally rather
//! than only within a tolerance — the catastrophic cancellation the one-pass
//! form suffers on large-magnitude activations never enters the picture.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Total element count below which the per-`(n, c)` loop stays serial.
///
/// Splitting a handful of short spatial planes across worker threads costs
/// more in task dispatch than the arithmetic saves; the style blocks this op
/// exists for are `[1, 1024, 32, 32]` and up, three orders of magnitude past
/// this line.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const PARALLEL_MIN_ELEMENTS: usize = 8_192;

/// Normalize one `(n, c)` spatial plane in place.
///
/// Two passes over `plane`: mean, then variance about that mean, then the
/// affine-free rescale. `plane.len()` is the caller-checked spatial size and
/// is never zero here.
#[inline]
fn normalize_plane(plane: &mut [f32], eps: f32) {
    let count = plane.len() as f32;
    let mean = plane.iter().sum::<f32>() / count;
    let var = plane.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / count;
    let inv_std = 1.0 / (var + eps).sqrt();
    for v in plane.iter_mut() {
        *v = (*v - mean) * inv_std;
    }
}

/// Normalize `src` into `dst`, one `(n, c)` plane of `spatial` elements at a
/// time. `dst` must already hold `src.len()` elements.
#[inline]
fn normalize_into(src: &[f32], dst: &mut [f32], spatial: usize, eps: f32) {
    dst.copy_from_slice(src);
    normalize_in_place(dst, spatial, eps);
}

/// Normalize `buf` in place, one `(n, c)` plane of `spatial` elements at a
/// time.
fn normalize_in_place(buf: &mut [f32], spatial: usize, eps: f32) {
    if spatial == 0 || buf.is_empty() {
        return;
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if buf.len() >= PARALLEL_MIN_ELEMENTS {
            buf.par_chunks_mut(spatial)
                .for_each(|plane| normalize_plane(plane, eps));
            return;
        }
        for plane in buf.chunks_mut(spatial) {
            normalize_plane(plane, eps);
        }
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        for plane in buf.chunks_mut(spatial) {
            normalize_plane(plane, eps);
        }
    }
}

/// Spatial size (`product(shape[2..])`) after validating rank and element
/// count.
fn spatial_size(x: &Tensor) -> Result<usize, OnnxError> {
    if x.ndim() < 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "OxiInstanceNorm: input must have rank >= 3 ([N, C, ...]), got shape {:?}",
            x.shape
        )));
    }
    let spatial = x.shape[2..]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d));
    let spatial = match spatial {
        Some(s) => s,
        None => {
            return Err(OnnxError::ShapeMismatch(format!(
                "OxiInstanceNorm: spatial extent of {:?} overflows usize",
                x.shape
            )))
        }
    };
    if x.numel() != x.data.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "OxiInstanceNorm: input has {} elements but shape {:?} implies {}",
            x.data.len(),
            x.shape,
            x.numel()
        )));
    }
    Ok(spatial)
}

/// `OxiInstanceNorm`: spatial mean/variance normalisation, no affine term.
///
/// `x` is `[N, C, d1, d2, …]` (rank >= 3); the reduction runs over
/// `d1, d2, …` for every `(n, c)` pair. See the module docs for why this is a
/// distinct op from `InstanceNormalization`.
pub fn oxi_instance_norm(x: &Tensor, eps: f32) -> Result<Tensor, OnnxError> {
    let spatial = spatial_size(x)?;
    let mut data = vec![0.0f32; x.data.len()];
    normalize_into(&x.data, &mut data, spatial, eps);
    Ok(Tensor::new(data, x.shape.clone()))
}

/// Operator wrapper for [`oxi_instance_norm`].
///
/// Registered under the name the `fuse_instance_norm` optimizer pass emits.
/// The pass checks `registry.get("OxiInstanceNorm")` before rewriting anything,
/// so a session built with a registry that omits this operator simply keeps the
/// unfused chain instead of producing a node nothing can dispatch.
pub struct OxiInstanceNormOp;

impl Operator for OxiInstanceNormOp {
    fn op_type(&self) -> &str {
        "OxiInstanceNorm"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![oxi_instance_norm(x, eps)?])
    }

    /// The normalisation reads a whole `(n, c)` plane (mean, then variance)
    /// before it writes any element of that plane, so rewriting the input
    /// buffer is safe: no element is overwritten while a later reduction still
    /// needs its original value.
    fn supports_inplace(&self) -> bool {
        true
    }

    fn execute_inplace(
        &self,
        mut input: Tensor,
        ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        let spatial = spatial_size(&input)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        normalize_in_place(&mut input.data, spatial, eps);
        Ok(vec![input])
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let slot = match slots.first_mut() {
            Some(slot) => slot,
            None => {
                return Err(OnnxError::Internal(
                    "OxiInstanceNormOp: no output slots".into(),
                ))
            }
        };
        let x = ctx.input(0)?;
        let spatial = spatial_size(x)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        let n = x.data.len();
        if slot.data.len() != n {
            slot.data.resize(n, 0.0_f32);
        }
        slot.shape.clone_from(&x.shape);
        normalize_into(&x.data, &mut slot.data, spatial, eps);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::{Attributes, Node, OpKind};

    fn node_with_eps(eps: f32) -> Node {
        let mut attrs = Attributes::default();
        attrs.floats.insert("epsilon".to_string(), eps);
        Node {
            name: "oxi_in".into(),
            op: OpKind::OxiInstanceNorm,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs,
        }
    }

    fn ctx<'a>(node: &'a Node, x: &'a Tensor) -> OpContext<'a> {
        OpContext {
            node,
            inputs: vec![Some(x)],
            outer_scope: None,
            weights: None,
            registry: None,
        }
    }

    /// The reference the fused op must reproduce: the exact node chain the
    /// optimizer pass deletes, evaluated element by element.
    fn unfused_chain(x: &Tensor, eps: f32) -> Vec<f32> {
        let c = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        let mut out = vec![0.0f32; x.data.len()];
        for plane in 0..(x.data.len() / spatial.max(1)) {
            let _ = c;
            let base = plane * spatial;
            let slice = &x.data[base..base + spatial];
            // ReduceMean(axes=spatial, keepdims=1)
            let mean = slice.iter().sum::<f32>() / spatial as f32;
            // Sub → Mul(diff, diff) → ReduceMean → Add(eps) → Sqrt
            let diff: Vec<f32> = slice.iter().map(|&v| v - mean).collect();
            let var = diff.iter().map(|&d| d * d).sum::<f32>() / spatial as f32;
            let std = (var + eps).sqrt();
            // Div(1, std) → Mul(diff, rstd)
            let rstd = 1.0 / std;
            for (i, &d) in diff.iter().enumerate() {
                out[base + i] = d * rstd;
            }
        }
        out
    }

    fn ramp(shape: &[usize], scale: f32, offset: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|i| offset + scale * ((i % 17) as f32 - 8.0) + 0.25 * (i as f32).sin())
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    #[test]
    fn matches_unfused_chain_nchw() {
        let x = ramp(&[2, 3, 4, 5], 1.5, 3.0);
        let eps = 1e-8;
        let expected = unfused_chain(&x, eps);
        let out = oxi_instance_norm(&x, eps).expect("normalize");
        assert_eq!(out.shape, x.shape);
        for (a, b) in out.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "{a} vs {b}");
        }
    }

    /// Each `(n, c)` plane is normalized on its own: zero mean and unit
    /// variance per plane, not across the whole tensor.
    #[test]
    fn each_plane_has_zero_mean_and_unit_variance() {
        let x = ramp(&[2, 3, 4, 4], 2.0, -7.0);
        let out = oxi_instance_norm(&x, 1e-12).expect("normalize");
        let spatial = 16;
        for plane in out.data.chunks(spatial) {
            let mean = plane.iter().sum::<f32>() / spatial as f32;
            let var = plane.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / spatial as f32;
            assert!(mean.abs() < 1e-4, "plane mean {mean}");
            assert!((var - 1.0).abs() < 1e-3, "plane var {var}");
        }
    }

    /// A constant plane has zero variance, so the result is entirely decided
    /// by `epsilon`: `0 / sqrt(0 + eps) == 0`, finite for every `eps > 0`.
    #[test]
    fn constant_plane_is_zero_not_nan() {
        let x = Tensor::new(vec![5.0; 8], vec![1, 2, 2, 2]);
        let out = oxi_instance_norm(&x, 1e-8).expect("normalize");
        assert!(out.data.iter().all(|v| v.is_finite()), "{:?}", out.data);
        assert!(out.data.iter().all(|v| v.abs() < 1e-6), "{:?}", out.data);
    }

    /// `epsilon` participates as `sqrt(var + eps)`, so a large `eps` visibly
    /// shrinks the output — a kernel that ignored the attribute would not
    /// move here.
    #[test]
    fn epsilon_scales_the_denominator() {
        let x = ramp(&[1, 1, 4, 4], 1.0, 0.0);
        let tight = oxi_instance_norm(&x, 1e-8).expect("normalize");
        let loose = oxi_instance_norm(&x, 100.0).expect("normalize");
        let tight_energy: f32 = tight.data.iter().map(|v| v * v).sum();
        let loose_energy: f32 = loose.data.iter().map(|v| v * v).sum();
        assert!(
            loose_energy < tight_energy * 0.5,
            "eps had no effect: {tight_energy} vs {loose_energy}"
        );
        // And it matches the closed form for the same eps.
        let expected = unfused_chain(&x, 100.0);
        for (a, b) in loose.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "{a} vs {b}");
        }
    }

    /// The parallel path (>= `PARALLEL_MIN_ELEMENTS`) must produce exactly the
    /// same planes as the serial one; chunking bugs show up as planes
    /// normalized against a neighbour's statistics.
    #[test]
    fn parallel_path_matches_reference() {
        // 4 * 16 * 16 * 16 = 16384 elements, past the parallel threshold.
        let x = ramp(&[4, 16, 16, 16], 0.7, 2.0);
        let eps = 1e-5;
        let expected = unfused_chain(&x, eps);
        let out = oxi_instance_norm(&x, eps).expect("normalize");
        for (a, b) in out.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "{a} vs {b}");
        }
    }

    #[test]
    fn rank_3_normalizes_over_the_single_spatial_axis() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![1, 2, 3]);
        let out = oxi_instance_norm(&x, 0.0).expect("normalize");
        let expected = unfused_chain(&x, 0.0);
        for (a, b) in out.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rejects_rank_below_3() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let err = oxi_instance_norm(&x, 1e-5).expect_err("rank 2 must error");
        assert!(format!("{err}").contains("rank >= 3"), "got: {err}");
    }

    #[test]
    fn operator_execute_reads_epsilon_attribute() {
        let x = ramp(&[1, 2, 3, 3], 1.0, 1.0);
        let node = node_with_eps(1e-8);
        let out = OxiInstanceNormOp.execute(&ctx(&node, &x)).expect("execute");
        let expected = unfused_chain(&x, 1e-8);
        for (a, b) in out[0].data.iter().zip(expected.iter()) {
            assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "{a} vs {b}");
        }
    }

    #[test]
    fn operator_default_epsilon_when_attribute_absent() {
        let x = ramp(&[1, 2, 3, 3], 1.0, 1.0);
        let node = Node {
            name: "oxi_in".into(),
            op: OpKind::OxiInstanceNorm,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs: Attributes::default(),
        };
        let out = OxiInstanceNormOp.execute(&ctx(&node, &x)).expect("execute");
        let expected = unfused_chain(&x, 1e-5);
        for (a, b) in out[0].data.iter().zip(expected.iter()) {
            assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "{a} vs {b}");
        }
    }

    /// The three dispatch paths (`execute`, `execute_inplace`,
    /// `execute_into_slots`) must agree element for element — the run loop
    /// picks between them on refcount and shape availability, not on anything
    /// the model says.
    #[test]
    fn inplace_and_slot_paths_agree_with_execute() {
        let x = ramp(&[2, 4, 5, 5], 1.1, -2.0);
        let node = node_with_eps(1e-8);
        let base = OxiInstanceNormOp.execute(&ctx(&node, &x)).expect("execute");

        let inplace_ctx = OpContext {
            node: &node,
            inputs: vec![None],
            outer_scope: None,
            weights: None,
            registry: None,
        };
        let inplace = OxiInstanceNormOp
            .execute_inplace(x.clone(), &inplace_ctx)
            .expect("execute_inplace");
        assert_eq!(inplace[0].shape, base[0].shape);
        assert_eq!(inplace[0].data, base[0].data);

        // A deliberately mis-sized slot: the op must resize and re-shape it
        // rather than write into whatever the pool handed over.
        let mut slots = vec![Tensor::new(vec![0.0f32; 4], vec![4])];
        OxiInstanceNormOp
            .execute_into_slots(&ctx(&node, &x), &mut slots)
            .expect("execute_into_slots");
        assert_eq!(slots[0].shape, base[0].shape);
        assert_eq!(slots[0].data, base[0].data);
    }

    #[test]
    fn slot_path_rejects_empty_slots() {
        let x = ramp(&[1, 2, 2, 2], 1.0, 0.0);
        let node = node_with_eps(1e-8);
        let err = OxiInstanceNormOp
            .execute_into_slots(&ctx(&node, &x), &mut [])
            .expect_err("no slots must error");
        assert!(format!("{err}").contains("no output slots"), "got: {err}");
    }
}
