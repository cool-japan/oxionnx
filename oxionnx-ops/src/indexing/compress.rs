use oxionnx_core::Tensor;

/// Compress: select elements based on boolean condition along an axis.
/// If axis is None, flatten input first.
pub fn compress(input: &Tensor, condition: &Tensor, axis: Option<i64>) -> Result<Tensor, String> {
    if let Some(ax_val) = axis {
        let ndim = input.ndim();
        let ax = if ax_val < 0 {
            (ndim as i64 + ax_val) as usize
        } else {
            ax_val as usize
        };
        if ax >= ndim {
            return Err(format!(
                "compress: axis {ax} out of range for {ndim}D tensor"
            ));
        }

        // Count true values in condition
        let true_count = condition.data.iter().filter(|v| **v != 0.0).count();

        let mut out_shape = input.shape.clone();
        out_shape[ax] = true_count;

        let outer: usize = input.shape[..ax].iter().product::<usize>().max(1);
        let axis_size = input.shape[ax];
        let inner: usize = input.shape[ax + 1..].iter().product::<usize>().max(1);

        let mut data = Vec::with_capacity(out_shape.iter().product());

        for o in 0..outer {
            for a in 0..axis_size {
                if a < condition.data.len() && condition.data[a] != 0.0 {
                    for i in 0..inner {
                        data.push(input.data[(o * axis_size + a) * inner + i]);
                    }
                }
            }
        }

        Ok(Tensor::new(data, out_shape))
    } else {
        // Flatten and select
        let mut data = Vec::new();
        for (i, v) in input.data.iter().enumerate() {
            if i < condition.data.len() && condition.data[i] != 0.0 {
                data.push(*v);
            }
        }
        let len = data.len();
        Ok(Tensor::new(data, vec![len]))
    }
}
