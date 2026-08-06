use oxionnx_core::Tensor;

/// Upper bound accepted for [`one_hot`]'s `depth` parameter.
///
/// ONNX `OneHot`'s `depth` is a scalar read straight off a runtime input
/// tensor with no inherent upper bound. Rust's float->usize cast
/// saturates (never wraps), so an adversarial (or merely corrupt) depth
/// like `1e15` becomes a huge `usize` rather than a small nonsense value;
/// without a cap, `vec![off_val; out_numel]` below would try to allocate on
/// the order of petabytes and abort the process with "capacity overflow"
/// instead of failing the inference call gracefully. No real
/// classification/vocabulary size needs anywhere near this many classes.
const MAX_ONE_HOT_DEPTH: usize = 1 << 24; // ~16.8M classes

/// OneHot: create one-hot encoded tensor
/// indices: any shape, depth: number of classes, values: (off_value, on_value), axis: insertion axis
pub fn one_hot(
    indices: &Tensor,
    depth: usize,
    values: (f32, f32),
    axis: i64,
) -> Result<Tensor, String> {
    if depth == 0 {
        return Err("one_hot: depth must be > 0".into());
    }
    if depth > MAX_ONE_HOT_DEPTH {
        return Err(format!(
            "one_hot: depth {depth} exceeds maximum supported depth {MAX_ONE_HOT_DEPTH}"
        ));
    }
    let ndim = indices.ndim() + 1;
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "one_hot: axis {ax} out of range for output ndim {ndim}"
        ));
    }

    let (off_val, on_val) = values;

    // Output shape: insert `depth` at position `ax` in indices.shape
    let mut out_shape = indices.shape.clone();
    out_shape.insert(ax, depth);

    let out_numel: usize = out_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| "one_hot: output tensor size overflows".to_string())?;
    let mut data = vec![off_val; out_numel];

    // For each element in indices, set the corresponding position to on_val
    let outer: usize = if ax < indices.ndim() {
        indices.shape[..ax].iter().product::<usize>().max(1)
    } else {
        indices.numel().max(1)
    };
    let inner: usize = if ax < indices.ndim() {
        indices.shape[ax..].iter().product::<usize>().max(1)
    } else {
        1
    };

    for o in 0..outer {
        for i in 0..inner {
            let idx_flat = o * inner + i;
            if idx_flat >= indices.data.len() {
                continue;
            }
            let mut idx_val = indices.data[idx_flat] as i64;
            if idx_val < 0 {
                idx_val += depth as i64;
            }
            if idx_val >= 0 && (idx_val as usize) < depth {
                let out_flat = (o * depth + idx_val as usize) * inner + i;
                data[out_flat] = on_val;
            }
        }
    }

    Ok(Tensor::new(data, out_shape))
}
