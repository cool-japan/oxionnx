//! Integration tests for GPU shader dispatch functions.

use crate::context::GpuBufferPool;
use crate::context::GpuContext;
use crate::shaders::{
    gpu_batch_norm, gpu_gelu, gpu_layer_norm, gpu_reduce_mean, gpu_relu, gpu_sigmoid, gpu_softmax,
    gpu_transpose,
};

#[test]
fn test_gpu_buffer_pool_basic() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return, // skip if no GPU
    };

    let mut pool = GpuBufferPool::new(16);
    assert_eq!(pool.available_count(), 0);

    // Get a buffer (creates new since pool is empty).
    let buf = pool.get_buffer(
        &ctx.device,
        1024,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    assert_eq!(pool.available_count(), 0);

    // Return it.
    pool.return_buffer(buf, 1024);
    assert_eq!(pool.available_count(), 1);

    // Clear.
    pool.clear();
    assert_eq!(pool.available_count(), 0);
}

#[test]
fn test_gpu_buffer_pool_reuse() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    let mut pool = GpuBufferPool::new(16);
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    // Get and return a 1024-byte buffer.
    let buf = pool.get_buffer(&ctx.device, 1024, usage);
    pool.return_buffer(buf, 1024);
    assert_eq!(pool.available_count(), 1);

    // Request 1024 again — should reuse (count stays 0 after get).
    let _buf2 = pool.get_buffer(&ctx.device, 1024, usage);
    assert_eq!(pool.available_count(), 0);

    // Request something much larger — pool won't have it, creates new.
    let _buf3 = pool.get_buffer(&ctx.device, 1_000_000, usage);
    assert_eq!(pool.available_count(), 0);

    // Return multiple buffers and verify they accumulate.
    let b1 = pool.get_buffer(&ctx.device, 512, usage);
    let b2 = pool.get_buffer(&ctx.device, 2048, usage);
    let b3 = pool.get_buffer(&ctx.device, 4096, usage);
    pool.return_buffer(b1, 512);
    pool.return_buffer(b2, 2048);
    pool.return_buffer(b3, 4096);
    assert_eq!(pool.available_count(), 3);
}

#[test]
fn test_gpu_softmax() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    // Shape: [2, 2000] — last dim > 1000 so GPU should accept.
    let rows = 2usize;
    let cols = 2000usize;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
    let shape = vec![rows, cols];

    let result = gpu_softmax(&ctx, &data, &shape);
    let result = match result {
        Some(r) => r,
        None => return, // GPU declined
    };

    assert_eq!(result.len(), rows * cols);

    // Verify each row sums to ~1.0.
    for row in 0..rows {
        let row_sum: f32 = result[row * cols..(row + 1) * cols].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 0.01,
            "softmax row {row} sum = {row_sum}, expected ~1.0"
        );
    }

    // Verify all values are non-negative.
    for (i, &v) in result.iter().enumerate() {
        assert!(v >= 0.0, "softmax output[{i}] = {v} is negative");
    }
}

#[test]
fn test_gpu_relu() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    let len = 200_000;
    let data: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
        .collect();

    let result = gpu_relu(&ctx, &data);
    let result = match result {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), len);
    for (i, (&out, &inp)) in result.iter().zip(data.iter()).enumerate() {
        let expected = inp.max(0.0);
        assert!(
            (out - expected).abs() < 1e-5,
            "relu mismatch at {i}: got {out}, expected {expected}"
        );
    }
}

#[test]
fn test_gpu_sigmoid() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    let len = 200_000;
    let data: Vec<f32> = (0..len).map(|i| (i as f32 - 100_000.0) * 0.0001).collect();

    let result = gpu_sigmoid(&ctx, &data);
    let result = match result {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), len);
    for (i, (&out, &inp)) in result.iter().zip(data.iter()).enumerate() {
        let expected = 1.0 / (1.0 + (-inp).exp());
        assert!(
            (out - expected).abs() < 1e-4,
            "sigmoid mismatch at {i}: got {out}, expected {expected}"
        );
    }
}

#[test]
fn test_gpu_gelu() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    let len = 200_000;
    let data: Vec<f32> = (0..len).map(|i| (i as f32 - 100_000.0) * 0.00005).collect();

    let result = gpu_gelu(&ctx, &data);
    let result = match result {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), len);
    // Spot-check a few values.
    for &idx in &[0usize, 1000, 50000, 100000, 150000, 199999] {
        if idx >= len {
            continue;
        }
        let x = data[idx];
        let c = 0.797_884_6_f32;
        let inner = c * (x + 0.044715 * x * x * x);
        let expected = 0.5 * x * (1.0 + inner.tanh());
        assert!(
            (result[idx] - expected).abs() < 1e-3,
            "gelu mismatch at {idx}: got {}, expected {expected}",
            result[idx]
        );
    }
}

#[test]
fn test_gpu_layer_norm() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    // [batch=250, n_elements=256] — 64000 > threshold
    let batch = 250usize;
    let n = 256usize;
    let total = batch * n;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let scale: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 - 0.1).collect();
    let shape = vec![batch, n];
    let eps = 1e-5_f32;

    let result = match gpu_layer_norm(&ctx, &data, &shape, &scale, &bias, eps) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), total);

    // CPU reference for each instance
    for b in 0..batch {
        let row = &data[b * n..(b + 1) * n];
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..n {
            let expected = (row[i] - mean) * inv_std * scale[i] + bias[i];
            let got = result[b * n + i];
            assert!(
                (got - expected).abs() < 1e-3,
                "layer_norm mismatch at batch={b}, i={i}: got {got}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_gpu_batch_norm() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    // [N=10, C=2, H=50, W=50] → 50000 elements
    let (nn, c, h, w) = (10, 2, 50, 50);
    let total = nn * c * h * w;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let shape = vec![nn, c, h, w];
    let bn_scale = vec![1.5_f32, 0.8];
    let bn_bias = vec![0.1_f32, -0.2];
    let bn_mean = vec![0.5_f32, -0.3];
    let bn_var = vec![1.0_f32, 2.0];
    let eps = 1e-5_f32;

    let result = match gpu_batch_norm(
        &ctx, &data, &shape, &bn_scale, &bn_bias, &bn_mean, &bn_var, eps,
    ) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), total);

    let spatial = h * w;
    // CPU reference
    for idx in 0..total {
        let ch = (idx / spatial) % c;
        let x = data[idx];
        let expected = bn_scale[ch] * (x - bn_mean[ch]) / (bn_var[ch] + eps).sqrt() + bn_bias[ch];
        assert!(
            (result[idx] - expected).abs() < 1e-3,
            "batch_norm mismatch at {idx}: got {}, expected {expected}",
            result[idx]
        );
    }
}

#[test]
fn test_gpu_transpose() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    // [200, 256] → [256, 200] — 51200 > threshold
    let (rows, cols) = (200usize, 256usize);
    let total = rows * cols;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let shape = vec![rows, cols];
    let perm = vec![1, 0];

    let result = match gpu_transpose(&ctx, &data, &shape, &perm) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), total);

    // CPU reference: output[j*rows+i] = input[i*cols+j]
    for i in 0..rows {
        for j in 0..cols {
            let expected = data[i * cols + j];
            let got = result[j * rows + i];
            assert!(
                (got - expected).abs() < 1e-6,
                "transpose mismatch at ({i},{j}): got {got}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_gpu_reduce_mean() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };

    // [100000, 3] axis=1 → output [100000] (output >= 50000)
    let (d0, d1) = (100_000usize, 3usize);
    let total = d0 * d1;
    let data: Vec<f32> = (0..total).map(|i| (i % 100) as f32 * 0.1).collect();
    let shape = vec![d0, d1];

    let result = match gpu_reduce_mean(&ctx, &data, &shape, &[1], false) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), d0);

    // CPU reference
    for (i, &result_val) in result.iter().enumerate() {
        let start = i * d1;
        let sum: f32 = data[start..start + d1].iter().sum();
        let expected = sum / d1 as f32;
        assert!(
            (result_val - expected).abs() < 1e-4,
            "reduce_mean mismatch at {i}: got {result_val}, expected {expected}",
        );
    }
}
