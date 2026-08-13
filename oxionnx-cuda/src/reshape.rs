//! `Reshape` / `Squeeze` / `Unsqueeze` / `Flatten`: zero-cost residency
//! aliases.
//!
//! # No kernel, because none is needed
//!
//! Every op this module covers changes only a tensor's *declared shape*, not
//! its bytes: a row-major, contiguous buffer holding `[2, 3, 4]` elements is
//! bit-for-bit the same allocation as one holding `[4, 6]` or `[1, 2, 3, 4,
//! 1]` of the same 24 elements. [`crate::CudaDeviceTensor::alias`] (built by
//! the residency stage this wave builds on) is exactly that: a second handle
//! to the same device allocation under a different shape, at the cost of an
//! `Arc::clone`. This module supplies the one thing `alias` cannot compute
//! for itself — *which* new shape a given ONNX node asks for — as four pure,
//! GPU-free functions; `lib.rs`'s dispatch arm calls whichever one the node
//! needs and, if it returns `Some`, hands the result straight to `alias`.
//!
//! # Only a device-resident input is claimed
//!
//! A host-resident input has nothing this module can accelerate — a CPU
//! reshape is already an `O(1)` shape relabel (`oxionnx-ops::shape::reshape`
//! clones the data only because `Tensor` does not distinguish "new tensor,
//! shared buffer" from "new tensor, owned buffer"; the FLOPs either way are
//! zero) — so `lib.rs`'s dispatch arm declines outright whenever the input
//! is not already a [`crate::CudaDeviceTensor`]. What this module exists for
//! is the case a CPU reshape cannot help with at all: a device-resident
//! producer feeding a device-resident consumer, where the *only* thing
//! standing between them is a shape node. Before this module, that shape
//! node forced a read-back — a real PCIe round trip and a blocking fence
//! paid for a metadata-only op — and InSwapper's AdaIN chain runs this
//! pattern (`Gemm -> Unsqueeze -> Mul`) 24 times a frame at a small vector's
//! size, so the traffic saved is small but the *fence* count saved is not.
//!
//! # Why this module carries no oracle
//!
//! [`mod@crate::reference`]'s shadow-verification story exists to catch a
//! *kernel* computing the wrong numbers. `alias` runs no kernel and touches
//! no bytes — the numbers it hands back are, by construction, the exact same
//! bytes the producing node already wrote, just under a new shape header.
//! There is nothing here for a CPU oracle to disagree with that a shape
//! mismatch (caught structurally, by [`crate::CudaDeviceTensor::alias`]
//! returning `None` whenever the element count would not match) does not
//! already catch.
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::Reshape`,
//! `OpKind::Squeeze`, `OpKind::Unsqueeze`, and `OpKind::Flatten` — a change
//! from before this wave, when none of the four had any CUDA arm at all (see
//! that function's own doctest, updated alongside it). A host-resident input,
//! or a shape this module cannot resolve, still declines to `Ok(None)`.

/// Normalizes a possibly-negative ONNX axis against `rank`, in the exclusive
/// `[-rank, rank)` range `Squeeze` uses. A local duplicate — see
/// [`crate::pad`]'s identically-named helper's doc comment for why.
#[must_use]
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..r).contains(&a).then_some(a as usize)
}

/// Normalizes a possibly-negative axis against the *inclusive* `[-rank,
/// rank]` range `Flatten` uses (an axis equal to `rank` is legal: "everything
/// goes into the outer dim").
#[must_use]
fn normalize_axis_inclusive(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..=r).contains(&a).then_some(a as usize)
}

/// Resolves ONNX `Reshape`'s target shape, honouring `-1` (infer) and `0`
/// (copy the input dim, unless `allowzero`).
///
/// Mirrors `oxionnx-ops::shape::basic::resolve_reshape` (independently, not
/// by calling it — see the [module docs](self) "why this module carries no
/// oracle" for the layering rule this follows). `numel` is the *actual*
/// element count of the input operand (its `CudaDeviceTensor::len()`), not
/// `input_dims.iter().product()` — the two could differ for a malformed
/// model, and the inferred `-1` dimension must be computed from what is
/// really there.
#[must_use]
pub fn resolve_reshape_shape(
    input_dims: &[usize],
    numel: usize,
    shape: &[i64],
    allowzero: bool,
) -> Option<Vec<usize>> {
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 || shape.iter().any(|&d| d < -1) {
        return None;
    }
    let has_explicit_zero = shape.contains(&0);
    if allowzero && neg_count == 1 && has_explicit_zero {
        // Ambiguous: infer against an explicit literal zero.
        return None;
    }

    let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        let dim = if d == -1 {
            usize::MAX // placeholder, overwritten below
        } else if d == 0 && !allowzero {
            *input_dims.get(i)?
        } else {
            d as usize
        };
        new_shape.push(dim);
    }

    if neg_count == 1 {
        let known: usize = new_shape.iter().filter(|&&d| d != usize::MAX).product();
        if known == 0 {
            return None;
        }
        let inferred = numel / known;
        for d in &mut new_shape {
            if *d == usize::MAX {
                *d = inferred;
            }
        }
    }

    if new_shape.iter().product::<usize>() != numel {
        return None;
    }
    Some(new_shape)
}

/// Resolves ONNX `Squeeze`'s target shape: drop the named (possibly
/// negative) axes, each of which must actually be size-1; if `axes` is
/// empty, drop every size-1 axis.
///
/// Squeezing away every axis yields the empty shape (a genuine rank-0
/// tensor), matching NumPy (`np.squeeze(np.array([5.0])).shape == ()`) and
/// `oxionnx-ops::shape::basic::resolve_squeeze_shape`, which this mirrors.
#[must_use]
pub fn resolve_squeeze_shape(input_shape: &[usize], axes: &[i64]) -> Option<Vec<usize>> {
    let ndim = input_shape.len();
    let resolved: Vec<usize> = if axes.is_empty() {
        (0..ndim).filter(|&i| input_shape[i] == 1).collect()
    } else {
        axes.iter()
            .map(|&a| normalize_axis(a, ndim))
            .collect::<Option<_>>()?
    };
    Some(
        input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (!(resolved.contains(&i) && d == 1)).then_some(d))
            .collect(),
    )
}

/// Resolves ONNX `Unsqueeze`'s target shape: insert size-1 axes at the named
/// (possibly negative) positions, normalized against the **output** rank
/// (`input_shape.len() + axes.len()`) — mirroring
/// `oxionnx-ops::shape::basic::resolve_unsqueeze_shape`'s own doc comment on
/// why the growing shape must not be used for normalization.
#[must_use]
pub fn resolve_unsqueeze_shape(input_shape: &[usize], axes: &[i64]) -> Option<Vec<usize>> {
    let out_rank = input_shape.len().checked_add(axes.len())?;
    let mut normalized: Vec<usize> = axes
        .iter()
        .map(|&a| normalize_axis(a, out_rank))
        .collect::<Option<_>>()?;
    normalized.sort_unstable();
    if normalized.windows(2).any(|w| w[0] == w[1]) {
        return None; // an axis specified more than once (after normalization)
    }

    let mut new_shape = Vec::with_capacity(out_rank);
    let mut axes_iter = normalized.iter().peekable();
    let mut src = input_shape.iter();
    for pos in 0..out_rank {
        if axes_iter.peek() == Some(&&pos) {
            axes_iter.next();
            new_shape.push(1);
        } else {
            new_shape.push(*src.next()?);
        }
    }
    Some(new_shape)
}

/// Resolves ONNX `Flatten`'s target shape: `[prod(shape[..axis]),
/// prod(shape[axis..])]`. Neither product is clamped to a minimum of `1` — a
/// genuinely zero-size input dimension must stay zero, matching
/// `oxionnx-ops::shape::basic::resolve_flatten_shape`.
#[must_use]
pub fn resolve_flatten_shape(input_shape: &[usize], axis: i64) -> Option<Vec<usize>> {
    let ndim = input_shape.len();
    let ax = normalize_axis_inclusive(axis, ndim)?;
    let outer: usize = input_shape[..ax].iter().product();
    let inner: usize = input_shape[ax..].iter().product();
    Some(vec![outer, inner])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_reshape_shape ────────────────────────────────────────────────

    #[test]
    fn reshape_infers_a_single_negative_one() {
        let got = resolve_reshape_shape(&[2, 3, 4], 24, &[-1, 4], false).expect("must resolve");
        assert_eq!(got, vec![6, 4]);
    }

    #[test]
    fn reshape_zero_copies_the_input_dim_by_default() {
        let got = resolve_reshape_shape(&[2, 3, 4], 24, &[0, 12], false).expect("must resolve");
        assert_eq!(got, vec![2, 12]);
    }

    #[test]
    fn reshape_zero_is_literal_under_allowzero() {
        let got = resolve_reshape_shape(&[2, 0, 4], 0, &[2, 0, 4], true).expect("must resolve");
        assert_eq!(got, vec![2, 0, 4]);
    }

    #[test]
    fn reshape_two_negative_ones_decline() {
        assert!(resolve_reshape_shape(&[2, 3, 4], 24, &[-1, -1], false).is_none());
    }

    #[test]
    fn reshape_element_count_mismatch_declines() {
        assert!(resolve_reshape_shape(&[2, 3, 4], 24, &[5, 5], false).is_none());
    }

    #[test]
    fn reshape_allowzero_with_infer_and_explicit_zero_declines() {
        assert!(resolve_reshape_shape(&[2, 3, 4], 24, &[-1, 0], true).is_none());
    }

    // ── resolve_squeeze_shape ────────────────────────────────────────────────

    #[test]
    fn squeeze_drops_every_size_one_axis_by_default() {
        let got = resolve_squeeze_shape(&[1, 3, 1, 4], &[]).expect("must resolve");
        assert_eq!(got, vec![3, 4]);
    }

    #[test]
    fn squeeze_named_axes_only() {
        let got = resolve_squeeze_shape(&[1, 3, 1, 4], &[0]).expect("must resolve");
        assert_eq!(got, vec![3, 1, 4]);
    }

    #[test]
    fn squeeze_negative_axis_normalizes() {
        let got = resolve_squeeze_shape(&[1, 3, 1, 4], &[-2]).expect("must resolve");
        assert_eq!(got, vec![1, 3, 4]);
    }

    #[test]
    fn squeezing_every_axis_yields_the_empty_shape() {
        let got = resolve_squeeze_shape(&[1, 1, 1], &[]).expect("must resolve");
        assert_eq!(got, Vec::<usize>::new());
    }

    #[test]
    fn squeeze_out_of_range_axis_declines() {
        assert!(resolve_squeeze_shape(&[1, 3], &[5]).is_none());
    }

    // ── resolve_unsqueeze_shape ─────────────────────────────────────────────

    #[test]
    fn unsqueeze_inserts_at_the_named_position() {
        // Real InSwapper pattern: [1, 1024] -> [1, 1024, 1, 1] via axes=[2,3].
        let got = resolve_unsqueeze_shape(&[1, 1024], &[2, 3]).expect("must resolve");
        assert_eq!(got, vec![1, 1024, 1, 1]);
    }

    #[test]
    fn unsqueeze_negative_axis_normalizes_against_the_output_rank() {
        let got = resolve_unsqueeze_shape(&[3, 4], &[-1]).expect("must resolve");
        assert_eq!(got, vec![3, 4, 1]);
    }

    #[test]
    fn unsqueeze_duplicate_axis_declines() {
        assert!(resolve_unsqueeze_shape(&[3, 4], &[1, 1]).is_none());
    }

    #[test]
    fn unsqueeze_scalar_to_1d() {
        let got = resolve_unsqueeze_shape(&[], &[0]).expect("must resolve");
        assert_eq!(got, vec![1]);
    }

    // ── resolve_flatten_shape ───────────────────────────────────────────────

    #[test]
    fn flatten_default_axis_is_1() {
        let got = resolve_flatten_shape(&[2, 3, 4], 1).expect("must resolve");
        assert_eq!(got, vec![2, 12]);
    }

    #[test]
    fn flatten_axis_0_is_the_arcface_flatten_pattern() {
        // Real node: w600k_r50.onnx's one Flatten, [1, 25088] -> unchanged
        // (axis default 1); axis 0 here to exercise the "everything in the
        // inner dim" edge instead.
        let got = resolve_flatten_shape(&[1, 512, 7, 7], 0).expect("must resolve");
        assert_eq!(got, vec![1, 25088]);
    }

    #[test]
    fn flatten_axis_equal_to_rank_puts_everything_outer() {
        let got = resolve_flatten_shape(&[2, 3, 4], 3).expect("must resolve");
        assert_eq!(got, vec![24, 1]);
    }

    #[test]
    fn flatten_out_of_range_axis_declines() {
        assert!(resolve_flatten_shape(&[2, 3, 4], 4).is_none());
        assert!(resolve_flatten_shape(&[2, 3, 4], -4).is_none());
    }
}
