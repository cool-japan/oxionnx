use oxionnx_core::Tensor;

/// QuantizeLinear: y = saturate(round(x / scale) + zero_point, int8 range)
/// Result stored as f32 with integer values in [-128, 127].
pub fn quantize_linear(
    x: &Tensor,
    y_scale: &Tensor,
    y_zero_point: Option<&Tensor>,
) -> Result<Tensor, String> {
    let zp = y_zero_point.map(|t| t.data[0]).unwrap_or(0.0);
    let scale_len = y_scale.numel();
    let data: Vec<f32> = x
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = y_scale.data[i % scale_len];
            ((v / scale).round() + zp).clamp(-128.0, 127.0)
        })
        .collect();
    Ok(Tensor::new(data, x.shape.clone()))
}

/// DequantizeLinear: y = (x - zero_point) * scale
pub fn dequantize_linear(
    x: &Tensor,
    x_scale: &Tensor,
    x_zero_point: Option<&Tensor>,
) -> Result<Tensor, String> {
    let zp = x_zero_point.map(|t| t.data[0]).unwrap_or(0.0);
    let scale_len = x_scale.numel();
    let data: Vec<f32> = x
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - zp) * x_scale.data[i % scale_len])
        .collect();
    Ok(Tensor::new(data, x.shape.clone()))
}
