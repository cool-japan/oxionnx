//! Small free-standing helpers [`crate::try_cuda_dispatch_resident`]'s match
//! arms share.
//!
//! Split out of `lib.rs` purely for size — mirroring `conv.rs`'s
//! `#[path = "conv_tests.rs"] mod tests;` and `lib.rs`'s own
//! `#[path = "dispatch_tests.rs"] mod dispatch_tests;` (both already-
//! established instances of the same pattern in this crate): a companion
//! file joins the crate root module via `#[path]` rather than becoming a
//! nested module of its own, so every name here is reachable exactly as if
//! it were still defined directly in `lib.rs` (`crate::initializer_id`, not
//! `crate::dispatch_helpers::initializer_id`) once `lib.rs` `use`s it back
//! in. `lib.rs` had grown past this workspace's 2000-line-per-file
//! refactor policy; nothing here changed behaviour, only location.

use std::collections::HashMap;

use oxionnx_core::{OnnxError, Tensor};

use crate::context::FailurePolicy;
use crate::{reference, residency, ResidentActivations};

/// The residency identity for `name`, or `None` when these bytes must not be
/// cached.
///
/// This is the one place that decides whether an operand is a *graph
/// initializer* — bytes that are invariant for the session — as opposed to an
/// activation this run just produced. Getting it wrong in the permissive
/// direction would be a correctness bug of the worst kind: caching an
/// activation means every later frame is computed against the first frame's
/// numbers, silently.
///
/// Three conditions, all necessary:
///
/// * **Not an intermediate.** `resolve` prefers `intermediates` over
///   `weights`, so a name a node has already produced this run is *not* an
///   initializer here whatever the weight map also holds under it. Keying such
///   a name would cache one tensor's bytes and then serve them for another's.
/// * **Not a device-resident node output.** The same rule, for the half of the
///   run state that no longer lives in `intermediates`: once activations can
///   stay on the device, a node output need not appear in the host map at all,
///   so the check above stops covering it. A model that reuses one name for an
///   initializer *and* a node output — legal ONNX — would otherwise have the
///   initializer's bytes cached and then served for the activation's. Mirrors
///   `initializer_key`'s `holds_node_output` guard on the wgpu path.
/// * **Present in `weights`.** That is what "initializer" means at this layer.
///
/// Mirrors `oxionnx::session::gpu_dispatch`'s `initializer_key`, which makes
/// the identical decision for the wgpu backend, for the identical reasons.
pub(crate) fn initializer_id<'a>(
    name: Option<&'a str>,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    activations: &dyn ResidentActivations,
    tensor: &Tensor,
    form: residency::OperandForm,
) -> Option<residency::WeightId<'a>> {
    let name = name?;
    if name.is_empty() || intermediates.contains_key(name) || activations.holds_node_output(name) {
        return None;
    }
    weights
        .contains_key(name)
        .then(|| residency::WeightId::new(name, &tensor.data, form))
}

/// Which derived form of an operand's bytes a GEMM will read.
///
/// A `transA`/`transB` node consumes the *transpose* of its operand, which is
/// a different byte sequence from the operand itself — and one graph can
/// legitimately consume the same initializer both ways from two different
/// nodes. Keeping them in separate cache slots is what stops one node being
/// served the other's bytes.
pub(crate) fn operand_form(transposed: bool) -> residency::OperandForm {
    if transposed {
        residency::OperandForm::Transposed
    } else {
        residency::OperandForm::Raw
    }
}

/// Read the live `OXIONNX_CUDA_VERIFY` / `OXIONNX_CUDA_STRICT` state and
/// delegate to [`reference::shadow_verify`], converting a strict-mode
/// mismatch into an [`OnnxError`].
///
/// Every `try_cuda_dispatch` call site that has just gotten a result back
/// from a GPU kernel routes through this before trusting it — see the
/// [`mod@reference`] module docs and finding a8-5 for why: a wrong GPU kernel
/// returns `Ok(Some(wrong_data))`, a successful-looking answer, and nothing
/// else in this crate's error handling can tell the difference.
///
/// # Returns
/// * `Ok(true)` — the caller should proceed with `gpu`'s numbers (this is
///   the fast path taken unconditionally whenever `OXIONNX_CUDA_VERIFY` is
///   unset, which is every production dispatch by default).
/// * `Ok(false)` — the caller must discard `gpu` and `return Ok(None)`
///   instead, so the real CPU operator recomputes the node. Already logged
///   at `error!`.
///
/// # Errors
/// [`OnnxError::Internal`] (wrapping [`crate::error::CudaDispatchError::Verify`])
/// only when `OXIONNX_CUDA_STRICT=1` and the comparison disagreed.
pub(crate) fn verify_or_fallback(
    op: &str,
    gpu: &[f32],
    oracle: impl FnOnce() -> Option<Vec<f32>>,
) -> Result<bool, OnnxError> {
    reference::shadow_verify(
        op,
        gpu,
        reference::verify_enabled(),
        FailurePolicy::current(),
        oracle,
    )
    .map_err(OnnxError::from)
}

/// Transpose the last two dims of batched 2-D data in-place.
///
/// Input layout: `batch` blocks of `rows * cols` elements (row-major).
/// Output layout: `batch` blocks of `cols * rows` elements (row-major).
pub(crate) fn transpose_2d_batched(
    data: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let slice = rows * cols;
    let mut out = vec![0.0_f32; data.len()];
    for b in 0..batch {
        let base_in = b * slice;
        let base_out = b * slice;
        for r in 0..rows {
            for c in 0..cols {
                out[base_out + c * rows + r] = data[base_in + r * cols + c];
            }
        }
    }
    out
}

/// Apply Gemm bias: `out += beta * bias`, unidirectionally broadcasting
/// `bias` (of `bias_shape`, rank 0/1/2 per the ONNX Gemm spec for `C`)
/// across the `[rows, n]` output, where `rows` is a multiple of `m` (`out`
/// may stack several batch slices, though a spec-conformant `Gemm` node
/// always has `rows == m`).
///
/// Returns `false` — leaving `out` unmodified — when `bias_shape` is not
/// unidirectionally broadcastable to `[m, n]`, or `bias`'s data is shorter
/// than `bias_shape` declares (a malformed model in either case); the
/// caller declines the whole node rather than silently omitting the bias,
/// so the CPU kernel produces the correct result or a proper diagnostic.
pub(crate) fn apply_gemm_bias(
    out: &mut [f32],
    bias: &[f32],
    bias_shape: &[usize],
    m: usize,
    n: usize,
    beta: f32,
) -> bool {
    if m == 0 || n == 0 {
        return false;
    }

    // Right-align `bias_shape` against `[m, n]` (numpy/ONNX unidirectional
    // broadcast): a trailing dim of `1` broadcasts, a trailing dim equal to
    // the target matches, anything else is an illegal (malformed-model) `C`
    // shape. Any dim further left must also be `1` — Gemm's `C` is spec'd
    // as at most 2-D, but this stays defensive rather than assuming it.
    let rank = bias_shape.len();
    let bias_n = if rank >= 1 { bias_shape[rank - 1] } else { 1 };
    let bias_m = if rank >= 2 { bias_shape[rank - 2] } else { 1 };
    let leading_ok = bias_shape[..rank.saturating_sub(2)].iter().all(|&d| d == 1);
    if !leading_ok || !(bias_m == 1 || bias_m == m) || !(bias_n == 1 || bias_n == n) {
        return false;
    }
    let Some(bias_needed) = bias_m.checked_mul(bias_n) else {
        return false;
    };
    if bias.len() < bias_needed {
        return false;
    }

    let total_rows = out.len() / n;
    for row in 0..total_rows {
        let bias_row = if bias_m == 1 { 0 } else { row % m };
        let base = row * n;
        let bias_base = bias_row * bias_n;
        for col in 0..n {
            let bias_col = if bias_n == 1 { 0 } else { col };
            out[base + col] += beta * bias[bias_base + bias_col];
        }
    }
    true
}
