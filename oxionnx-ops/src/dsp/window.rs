//! Window function operators: HannWindow, HammingWindow, BlackmanWindow.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::helpers::scalar_i64;

pub(super) enum WindowKind {
    Hann,
    Hamming,
    Blackman,
}

/// Compute a window of `size` samples.
///
/// - `periodic = true`  → denominator = N   (DFT use-case, opset default)
/// - `periodic = false` → denominator = N-1 (filter-design use-case)
pub(super) fn window_generic(size: usize, periodic: bool, kind: &WindowKind) -> Vec<f32> {
    use std::f32::consts::PI;
    if size == 0 {
        return Vec::new();
    }
    let denom = if periodic {
        size as f32
    } else {
        (size - 1) as f32
    };
    (0..size)
        .map(|n| {
            let n_f = n as f32;
            match kind {
                WindowKind::Hann => 0.5 - 0.5 * (2.0 * PI * n_f / denom).cos(),
                WindowKind::Hamming => 0.543_478_26 - 0.456_521_74 * (2.0 * PI * n_f / denom).cos(),
                WindowKind::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * n_f / denom).cos()
                        + 0.08 * (4.0 * PI * n_f / denom).cos()
                }
            }
        })
        .collect()
}

/// Shared logic for all three window-function operators.
fn execute_window_op(
    ctx: &OpContext<'_>,
    kind: &WindowKind,
    op_name: &str,
) -> Result<Vec<Tensor>, OnnxError> {
    let size_t = ctx.input(0)?;
    let size = scalar_i64(size_t, &format!("{op_name}/size"))? as usize;

    // periodic attr: 1 = periodic (DFT), 0 = symmetric (filter). Default = 1.
    let periodic = ctx.attrs().i("periodic", 1) != 0;

    let data = window_generic(size, periodic, kind);
    let out = Tensor::new(data, vec![size]);
    Ok(vec![out])
}

// ── HannWindow ────────────────────────────────────────────────────────────────

pub struct HannWindowOp;
impl Operator for HannWindowOp {
    fn op_type(&self) -> &str {
        "HannWindow"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        execute_window_op(ctx, &WindowKind::Hann, "HannWindow")
    }

    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── HammingWindow ─────────────────────────────────────────────────────────────

pub struct HammingWindowOp;
impl Operator for HammingWindowOp {
    fn op_type(&self) -> &str {
        "HammingWindow"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        execute_window_op(ctx, &WindowKind::Hamming, "HammingWindow")
    }

    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── BlackmanWindow ────────────────────────────────────────────────────────────

pub struct BlackmanWindowOp;
impl Operator for BlackmanWindowOp {
    fn op_type(&self) -> &str {
        "BlackmanWindow"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        execute_window_op(ctx, &WindowKind::Blackman, "BlackmanWindow")
    }

    fn supports_output_slots(&self) -> bool {
        true
    }
}
