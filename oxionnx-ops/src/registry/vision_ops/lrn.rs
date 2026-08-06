//! `LRN` (local response normalization) operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

/// ONNX `LRN` (opset 1+), the across-channel normalization of AlexNet /
/// CaffeNet / GoogLeNet.
///
/// For an input `X` shaped `[N, C, D1, …, Dk]`:
///
/// ```text
/// square_sum[n, c, d…] = Σ  X[n, i, d…]^2   for
///                        max(0, c - floor((size - 1) / 2)) ≤ i
///                                          ≤ min(C - 1, c + ceil((size - 1) / 2))
/// Y[n, c, d…] = X[n, c, d…] / (bias + (alpha / size) * square_sum)^beta
/// ```
///
/// Two details the formula hides and that a "symmetric window" implementation
/// gets wrong:
///
/// * `alpha` is divided by `size` before scaling the sum, and
/// * for an **even** `size` the window is asymmetric — one element back and
///   `size / 2` elements forward — because `floor` and `ceil` of
///   `(size - 1) / 2` differ.
///
/// Attribute defaults: `alpha = 0.0001`, `beta = 0.75`, `bias = 1.0`; `size`
/// is required and must be ≥ 1.
pub struct LRNOp;

impl LRNOp {
    fn run(&self, ctx: &OpContext<'_>, out: &mut Tensor) -> Result<(), OnnxError> {
        let x = ctx.input(0)?;
        if x.ndim() < 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "LRN: input must have rank >= 2 ([N, C, ...]), got {:?}",
                x.shape
            )));
        }
        let attrs = ctx.attrs();
        let size = attrs.i("size", 0);
        if size < 1 {
            return Err(OnnxError::InvalidModel(format!(
                "LRN: size must be >= 1, got {size}"
            )));
        }
        let size = size as usize;
        let alpha = attrs.f("alpha", 0.0001);
        let beta = attrs.f("beta", 0.75);
        let bias = attrs.f("bias", 1.0);

        let batch = x.shape[0];
        let channels = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        if x.data.len() != batch * channels * spatial {
            return Err(OnnxError::ShapeMismatch(format!(
                "LRN: data length {} does not match shape {:?}",
                x.data.len(),
                x.shape
            )));
        }

        // Asymmetric window bounds, per the spec's floor/ceil pair.
        let back = (size - 1) / 2;
        let forward = size - 1 - back;
        let scale = alpha / size as f32;

        let total = x.data.len();
        if out.data.len() != total {
            out.data.resize(total, 0.0);
        }
        out.shape.clone_from(&x.shape);

        for n in 0..batch {
            let batch_base = n * channels * spatial;
            for c in 0..channels {
                let begin = c.saturating_sub(back);
                let end = (c + forward + 1).min(channels);
                let dst = batch_base + c * spatial;
                for s in 0..spatial {
                    let mut square_sum = 0.0_f32;
                    for i in begin..end {
                        let v = x.data[batch_base + i * spatial + s];
                        square_sum += v * v;
                    }
                    let denom = (bias + scale * square_sum).powf(beta);
                    out.data[dst + s] = x.data[dst + s] / denom;
                }
            }
        }
        Ok(())
    }
}

impl Operator for LRNOp {
    fn op_type(&self) -> &str {
        "LRN"
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
                "LRNOp: expected 1 output slot, got {}",
                slots.len()
            )));
        };
        self.run(ctx, slot)
    }
}
