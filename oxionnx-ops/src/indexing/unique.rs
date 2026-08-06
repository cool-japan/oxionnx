use oxionnx_core::Tensor;

/// Unique: find unique elements.
/// Returns (unique_values, indices, inverse_indices, counts)
pub fn unique(
    x: &Tensor,
    axis: Option<i64>,
    sorted: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor), String> {
    if let Some(raw_axis) = axis {
        return unique_axis(x, raw_axis, sorted);
    }

    // Flatten approach (axis=None)
    let mut seen: Vec<(f32, usize)> = Vec::new(); // (value, first_index)
    let mut inverse = vec![0.0f32; x.data.len()];

    for (i, &val) in x.data.iter().enumerate() {
        // Exact bit-pattern equality, not an epsilon/tolerance window: ONNX Unique
        // de-duplicates identical values, not "close" ones (an epsilon predicate would wrongly
        // merge two genuinely distinct-but-close floats, e.g. 1.0 and 1.0+f32::EPSILON/2, and
        // is also non-transitive as a dedup key, e.g. a chain of values each within epsilon of
        // the next but not of each other). This mirrors `slices_equal` below (the `axis`-mode
        // path in this same file), so both modes agree on what counts as "equal".
        if let Some(pos) = seen.iter().position(|(v, _)| v.to_bits() == val.to_bits()) {
            inverse[i] = pos as f32;
        } else {
            inverse[i] = seen.len() as f32;
            seen.push((val, i));
        }
    }

    if sorted {
        // Sort by value and remap
        let mut sorted_indices: Vec<usize> = (0..seen.len()).collect();
        sorted_indices.sort_by(|a, b| {
            seen[*a]
                .0
                .partial_cmp(&seen[*b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Remap inverse
        let mut remap = vec![0usize; seen.len()];
        for (new_pos, &old_pos) in sorted_indices.iter().enumerate() {
            remap[old_pos] = new_pos;
        }
        for inv in inverse.iter_mut() {
            *inv = remap[*inv as usize] as f32;
        }
        let sorted_seen: Vec<(f32, usize)> = sorted_indices.iter().map(|&i| seen[i]).collect();
        seen = sorted_seen;
    }

    let unique_vals: Vec<f32> = seen.iter().map(|(v, _)| *v).collect();
    let indices_data: Vec<f32> = seen.iter().map(|(_, i)| *i as f32).collect();
    let mut counts = vec![0.0f32; seen.len()];
    for &inv in &inverse {
        counts[inv as usize] += 1.0;
    }

    let n = unique_vals.len();
    Ok((
        Tensor::new(unique_vals, vec![n]),
        Tensor::new(indices_data, vec![n]),
        Tensor::new(inverse, vec![x.data.len()]),
        Tensor::new(counts, vec![n]),
    ))
}

/// Extract the data of a single slice along `ax` at position `idx`.
/// For shape [d0,..,d_{ax-1}, d_ax, d_{ax+1},..,d_{n-1}], the slice has
/// `outer * inner` elements where outer = product(shape[..ax]), inner = product(shape[ax+1..]).
fn extract_axis_slice(data: &[f32], shape: &[usize], ax: usize, idx: usize) -> Vec<f32> {
    // No `.max(1)`: an empty slice already multiplies to 1 (correct vacuous case). Clamping a
    // genuine zero-size outer/inner dim (e.g. shape [0,3,4] ax=1) to 1 would make the loop below
    // index into `data`, which is correctly zero-length for that shape — an out-of-bounds panic.
    let outer: usize = shape[..ax].iter().product::<usize>();
    let axis_size = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product::<usize>();
    let mut slice_data = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        let base = (o * axis_size + idx) * inner;
        for j in 0..inner {
            slice_data.push(data[base + j]);
        }
    }
    slice_data
}

/// Compare two slices for exact equality (bitwise f32 comparison).
fn slices_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Unique along a specified axis.
/// Returns (unique_tensor, indices, inverse_indices, counts).
fn unique_axis(
    x: &Tensor,
    raw_axis: i64,
    sorted: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor), String> {
    let ndim = x.ndim();
    if ndim == 0 {
        return Err("unique: axis mode requires at least 1D tensor".into());
    }
    let ax = if raw_axis < 0 {
        let a = raw_axis + ndim as i64;
        if a < 0 {
            return Err(format!(
                "unique: axis {raw_axis} out of range for {ndim}D tensor"
            ));
        }
        a as usize
    } else {
        raw_axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "unique: axis {raw_axis} out of range for {ndim}D tensor"
        ));
    }

    let axis_size = x.shape[ax];

    // Extract all slices along the axis
    let all_slices: Vec<Vec<f32>> = (0..axis_size)
        .map(|i| extract_axis_slice(&x.data, &x.shape, ax, i))
        .collect();

    // Find unique slices (first occurrence order)
    // unique_map[i] = index into `unique_indices` that slice i maps to
    let mut unique_indices: Vec<usize> = Vec::new(); // original axis indices of unique slices
    let mut inverse_map: Vec<usize> = vec![0; axis_size];

    for i in 0..axis_size {
        let mut found = None;
        for (uid, &orig_idx) in unique_indices.iter().enumerate() {
            if slices_equal(&all_slices[i], &all_slices[orig_idx]) {
                found = Some(uid);
                break;
            }
        }
        match found {
            Some(uid) => {
                inverse_map[i] = uid;
            }
            None => {
                inverse_map[i] = unique_indices.len();
                unique_indices.push(i);
            }
        }
    }

    // If sorted, sort unique slices lexicographically and remap
    if sorted {
        let mut order: Vec<usize> = (0..unique_indices.len()).collect();
        order.sort_by(|&a, &b| {
            let sa = &all_slices[unique_indices[a]];
            let sb = &all_slices[unique_indices[b]];
            for (va, vb) in sa.iter().zip(sb.iter()) {
                match va.partial_cmp(vb) {
                    Some(std::cmp::Ordering::Equal) | None => continue,
                    Some(ord) => return ord,
                }
            }
            std::cmp::Ordering::Equal
        });

        // Build old-unique-pos -> new-unique-pos mapping
        let mut remap = vec![0usize; unique_indices.len()];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            remap[old_pos] = new_pos;
        }
        // Remap inverse
        for inv in inverse_map.iter_mut() {
            *inv = remap[*inv];
        }
        // Reorder unique_indices
        let sorted_unique: Vec<usize> = order.iter().map(|&i| unique_indices[i]).collect();
        unique_indices = sorted_unique;
    }

    // Build output tensor by stacking unique slices along axis
    let num_unique = unique_indices.len();
    let mut out_shape = x.shape.clone();
    out_shape[ax] = num_unique;

    // No `.max(1)` — see `extract_axis_slice` above: a genuine zero-size outer/inner dim must
    // produce `outer`/`inner == 0` (so the copy loop below correctly does nothing), not a
    // phantom 1 that then slices past the real (zero-length, for that shape) `x.data`/`out_data`.
    let outer: usize = x.shape[..ax].iter().product::<usize>();
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>();

    let total_elems: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; total_elems];

    for (new_idx, &orig_idx) in unique_indices.iter().enumerate() {
        for o in 0..outer {
            let src_base = (o * axis_size + orig_idx) * inner;
            let dst_base = (o * num_unique + new_idx) * inner;
            out_data[dst_base..dst_base + inner]
                .copy_from_slice(&x.data[src_base..src_base + inner]);
        }
    }

    // Build counts
    let mut counts = vec![0.0f32; num_unique];
    for &inv in &inverse_map {
        counts[inv] += 1.0;
    }

    let indices_data: Vec<f32> = unique_indices.iter().map(|&i| i as f32).collect();
    let inverse_data: Vec<f32> = inverse_map.iter().map(|&i| i as f32).collect();

    Ok((
        Tensor::new(out_data, out_shape),
        Tensor::new(indices_data, vec![num_unique]),
        Tensor::new(inverse_data, vec![axis_size]),
        Tensor::new(counts, vec![num_unique]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [Exact bit-pattern equality regression] `1e-7_f32` and `2e-7_f32` are distinct,
    /// independently-representable f32 values (verified via Python:
    /// `struct.pack('<f', 1e-7) = 95bfd633`, `struct.pack('<f', 2e-7) = 95bf5634`), but their
    /// difference (~1.0000000117e-7) is *smaller* than `f32::EPSILON` (1.1920929e-7) — exactly
    /// the case the old `(*v - val).abs() < f32::EPSILON` predicate got wrong, silently merging
    /// two genuinely different values. ONNX Unique must de-duplicate identical values only.
    #[test]
    fn unique_flatten_does_not_merge_close_but_distinct_floats() {
        let x = Tensor::new(vec![1e-7_f32, 2e-7_f32, 1e-7_f32], vec![3]);
        let (vals, indices, inverse, counts) = unique(&x, None, false).expect("unique failed");
        assert_eq!(
            vals.data,
            vec![1e-7_f32, 2e-7_f32],
            "must stay 2 distinct groups"
        );
        assert_eq!(indices.data, vec![0.0, 1.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
        assert_eq!(counts.data, vec![2.0, 1.0]);
    }

    /// Sanity companion to the above: bit-identical values (including `-0.0`/`0.0`, which are
    /// numerically `==` but bit-distinct) still dedup/don't-dedup exactly as bit patterns say.
    #[test]
    fn unique_flatten_bit_identical_values_still_merge() {
        let x = Tensor::new(vec![3.5_f32, 3.5_f32, 3.5_f32], vec![3]);
        let (vals, _, inverse, counts) = unique(&x, None, false).expect("unique failed");
        assert_eq!(vals.data, vec![3.5_f32]);
        assert_eq!(inverse.data, vec![0.0, 0.0, 0.0]);
        assert_eq!(counts.data, vec![3.0]);
    }

    /// [`.max(1)` zero-dim regression] axis-mode: a trailing (inner) dim of size 0 means every
    /// per-axis "row" is itself empty (and therefore all rows compare equal — 0-length slices
    /// are vacuously equal). `outer`/`inner` used to be clamped from a genuine 0 up to 1 by a
    /// stray `.max(1)`, which then sliced the (correctly zero-length) `x.data`/`out_data` out of
    /// bounds — a panic, not just a wrong shape.
    #[test]
    fn unique_axis_zero_size_inner_dim_does_not_panic() {
        let x = Tensor::new(Vec::new(), vec![3, 0]); // 3 empty "rows", 0 elements total
        let (y, indices, inverse, counts) = unique(&x, Some(0), false).expect("unique failed");
        // All 3 empty rows are equal to each other -> exactly 1 unique row.
        assert_eq!(y.shape, vec![1, 0]);
        assert!(y.data.is_empty());
        assert_eq!(indices.data, vec![0.0]);
        assert_eq!(inverse.data, vec![0.0, 0.0, 0.0]);
        assert_eq!(counts.data, vec![3.0]);
    }
}
