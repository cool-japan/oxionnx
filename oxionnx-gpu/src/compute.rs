use crate::context::GpuContext;
use oxionnx_core::Tensor;
use wgpu;
use wgpu::util::DeviceExt;

/// Minimum number of FLOPs (M*K*N) before we bother using the GPU.
/// Below this threshold, CPU is faster due to GPU dispatch overhead.
/// GPU is only beneficial for very large GEMMs where compute dominates over
/// CPU-to-GPU transfer overhead.  10M is a conservative threshold.
const GPU_THRESHOLD: usize = 10_000_000;

/// Minimum dimension size for tiled matmul (shared-memory tiles are 16x16).
const TILED_MIN_DIM: usize = 32;

/// Uniform buffer params — must match the WGSL `Params` struct layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32, // align to 16 bytes
}

// ========================================================================
// Helper: read back a staging buffer into Vec<f32>
// ========================================================================

/// Read back GPU staging buffer contents into a `Vec<f32>`.
///
/// On wasm32, blocking device poll is not supported, so this returns `None`.
/// Callers on wasm32 should use async readback instead.
fn read_back_matmul(
    _device: &wgpu::Device,
    _staging: &wgpu::Buffer,
    _count: usize,
) -> Option<Vec<f32>> {
    #[cfg(target_arch = "wasm32")]
    {
        // Cannot block on wasm32 — sync readback is not supported.
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let slice = _staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        _device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if receiver.recv().ok()?.is_err() {
            return None;
        }
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[.._count].to_vec();
        drop(data);
        _staging.unmap();
        Some(result)
    }
}

/// Run matrix multiplication on GPU: C = A * B
/// A: [M, K], B: [K, N] -> C: [M, N]
///
/// Automatically selects tiled (shared memory) kernel for large matrices
/// and falls back to basic kernel for smaller ones.
///
/// Returns `None` if the problem is too small for GPU (caller should use CPU).
pub fn gpu_matmul(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    // Skip GPU for small matrices — overhead not worth it.
    if m * k * n < GPU_THRESHOLD {
        return None;
    }

    // Use tiled kernel for large dimensions, basic for small.
    if m >= TILED_MIN_DIM && n >= TILED_MIN_DIM && k >= TILED_MIN_DIM {
        gpu_matmul_tiled_inner(ctx, a, b, m, k, n)
    } else {
        gpu_matmul_basic(ctx, a, b, m, k, n)
    }
}

/// Tiled matrix multiply using shared memory for improved cache locality.
/// Uses TILE_SIZE x TILE_SIZE tiles loaded into workgroup shared memory.
/// Falls back to the basic kernel for small matrices.
///
/// Returns `None` if the problem is too small for GPU (caller should use CPU).
pub fn gpu_matmul_tiled(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    // Only use GPU for large enough matrices
    let flops = m * k * n;
    if flops < GPU_THRESHOLD {
        return None;
    }

    // Use tiled kernel for large dimensions, basic for small
    if m >= TILED_MIN_DIM && n >= TILED_MIN_DIM && k >= TILED_MIN_DIM {
        gpu_matmul_tiled_inner(ctx, a, b, m, k, n)
    } else {
        gpu_matmul_basic(ctx, a, b, m, k, n)
    }
}

/// Inner implementation of tiled matmul using 16x16 shared-memory tiles.
fn gpu_matmul_tiled_inner(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiled_A"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiled_B"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiled_C"),
        size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = GemmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiled_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiled_staging"),
        size: c_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tiled_matmul_bg"),
        layout: &ctx.tiled_matmul_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tiled_matmul_enc"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tiled_matmul_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.tiled_matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Tiled kernel uses workgroup_size(16, 16): dispatch enough workgroups
        // so global_invocation covers all (col, row) pairs.
        let wg_x = (n as u32).div_ceil(16);
        let wg_y = (m as u32).div_ceil(16);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, c_size);
    queue.submit(std::iter::once(encoder.finish()));

    read_back_matmul(device, &staging_buf, m * n)
}

/// Basic (non-tiled) GPU matmul — used as fallback for matrices with small dimensions.
fn gpu_matmul_basic(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = GemmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: c_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bg"),
        layout: &ctx.matmul_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_enc"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (m as u32).div_ceil(8);
        let wg_y = (n as u32).div_ceil(8);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, c_size);
    queue.submit(std::iter::once(encoder.finish()));

    read_back_matmul(device, &staging_buf, m * n)
}

/// GPU-accelerated Conv2D: im2col on CPU, GEMM on GPU.
///
/// Falls back to `None` if the GEMM is too small for GPU benefit.
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Option<Tensor> {
    let n = input.shape[0];
    let c_in = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let c_out = weight.shape[0];
    let c_per_group = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let oh = (h + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
    let ow = (w + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;

    let c_out_per_group = c_out / group;
    let col_rows = c_per_group * kh * kw;
    let col_cols = oh * ow;

    // Check if the GEMM is large enough for GPU.
    if c_out_per_group * col_rows * col_cols < GPU_THRESHOLD {
        return None;
    }

    let mut out = vec![0.0f32; n * c_out * oh * ow];

    for batch in 0..n {
        for g in 0..group {
            let in_c_start = g * c_per_group;

            // im2col on CPU.
            let mut col = vec![0.0f32; col_rows * col_cols];
            im2col(
                &input.data,
                c_in,
                h,
                w,
                in_c_start,
                c_per_group,
                kh,
                kw,
                strides,
                pads,
                dilations,
                oh,
                ow,
                batch,
                &mut col,
            );

            // GEMM on GPU: weight_slice[c_out_per_group, col_rows] * col[col_rows, col_cols]
            let w_off = g * c_out_per_group * col_rows;
            let w_end = w_off + c_out_per_group * col_rows;
            let weight_slice = &weight.data[w_off..w_end];

            let gemm_result =
                gpu_matmul(ctx, weight_slice, &col, c_out_per_group, col_rows, col_cols)?;

            // Copy result + add bias on CPU.
            let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
            out[o_off..o_off + c_out_per_group * col_cols].copy_from_slice(&gemm_result);

            if let Some(b) = bias {
                for oc in 0..c_out_per_group {
                    let bv = b.data[g * c_out_per_group + oc];
                    let start = o_off + oc * col_cols;
                    for j in 0..col_cols {
                        out[start + j] += bv;
                    }
                }
            }
        }
    }

    Some(Tensor::new(out, vec![n, c_out, oh, ow]))
}

/// Build the im2col column matrix for one (batch, group) slice.
/// Identical logic to the CPU version in `ops/conv.rs`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    oh: usize,
    ow: usize,
    batch: usize,
    col: &mut [f32],
) {
    let col_cols = oh * ow;
    let mut row = 0;
    for ic in 0..c_per_group {
        let in_c = in_c_start + ic;
        let in_plane = &input[(batch * c_in + in_c) * h * w..][..h * w];
        for ky in 0..kh {
            for kx in 0..kw {
                for oy in 0..oh {
                    let iy = (oy * strides[0] + ky * dilations[0]) as isize - pads[0] as isize;
                    if iy < 0 || iy >= h as isize {
                        let base = row * col_cols + oy * ow;
                        for ox in 0..ow {
                            col[base + ox] = 0.0;
                        }
                        continue;
                    }
                    let iy = iy as usize;
                    let base = row * col_cols + oy * ow;
                    for ox in 0..ow {
                        let ix = (ox * strides[1] + kx * dilations[1]) as isize - pads[1] as isize;
                        col[base + ox] = if ix >= 0 && ix < w as isize {
                            in_plane[iy * w + ix as usize]
                        } else {
                            0.0
                        };
                    }
                }
                row += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuTensorTracker;

    /// CPU reference matmul for verification.
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    #[test]
    fn test_gpu_matmul_small_falls_back() {
        // Below threshold: should return None.
        let ctx = GpuContext::try_new();
        if let Some(ref ctx) = ctx {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            let result = gpu_matmul(ctx, &a, &b, 2, 2, 2);
            assert!(result.is_none(), "small matmul should fall back to CPU");
        }
    }

    #[test]
    fn test_gpu_matmul_large() {
        let ctx = GpuContext::try_new();
        if let Some(ref ctx) = ctx {
            let m = 64;
            let k = 64;
            let n = 64;
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();

            let gpu_result = gpu_matmul(ctx, &a, &b, m, k, n);
            let gpu_out = match gpu_result {
                Some(out) => out,
                None => return,
            };
            let cpu_out = cpu_matmul(&a, &b, m, k, n);

            for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
                assert!((g - c).abs() < 1.0, "mismatch at {i}: gpu={g} cpu={c}");
            }
        }
    }

    #[test]
    fn test_tiled_matmul_basic() {
        // 32x32 tiled multiply, verify against CPU
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return, // skip if no GPU
        };

        let m = 32;
        let k = 32;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.3 - 1.5).collect();

        // gpu_matmul_tiled requires flops >= threshold, so use large enough matrices
        // that pass the threshold: 32*32*32 = 32768 < 10M, so it returns None.
        // Use the inner function directly for testing.
        let gpu_out = match gpu_matmul_tiled_inner(&ctx, &a, &b, m, k, n) {
            Some(out) => out,
            None => return,
        };
        let cpu_out = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1.0,
                "tiled basic mismatch at {i}: gpu={g} cpu={c}"
            );
        }
    }

    #[test]
    fn test_tiled_matmul_non_square() {
        // 64x48 * 48x32 non-square tiled multiply
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let m = 64;
        let k = 48;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.2 - 0.8).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.4 - 1.2).collect();

        let gpu_out = match gpu_matmul_tiled_inner(&ctx, &a, &b, m, k, n) {
            Some(out) => out,
            None => return,
        };
        let cpu_out = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu_out.len(), m * n);
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1.0,
                "tiled non-square mismatch at {i}: gpu={g} cpu={c}"
            );
        }
    }

    #[test]
    fn test_tiled_matmul_large() {
        // 256x256 large tiled multiply
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let m = 256;
        let k = 256;
        let n = 256;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.15 - 0.9).collect();

        // 256^3 = 16M > threshold, so gpu_matmul_tiled should work
        let gpu_out = match gpu_matmul_tiled(&ctx, &a, &b, m, k, n) {
            Some(out) => out,
            None => return,
        };
        let cpu_out = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu_out.len(), m * n);
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 2.0,
                "tiled large mismatch at {i}: gpu={g} cpu={c}"
            );
        }
    }

    #[test]
    fn test_gpu_tensor_tracker() {
        // Test tracker store, check, take operations (no GPU needed)
        let mut tracker = GpuTensorTracker::new();
        assert_eq!(tracker.count(), 0);
        assert!(!tracker.is_on_gpu("tensor_a"));

        // We cannot create real wgpu::Buffer without a device, so test
        // the logic with an actual GPU context if available.
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_tracker"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        tracker.store("tensor_a".to_string(), buf, 1024);
        assert!(tracker.is_on_gpu("tensor_a"));
        assert!(!tracker.is_on_gpu("tensor_b"));
        assert_eq!(tracker.count(), 1);

        let taken = tracker.take("tensor_a");
        assert!(taken.is_some());
        let (_, size) = taken.expect("just checked");
        assert_eq!(size, 1024);
        assert!(!tracker.is_on_gpu("tensor_a"));
        assert_eq!(tracker.count(), 0);

        // Test clear
        let buf2 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_tracker2"),
            size: 2048,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        tracker.store("tensor_b".to_string(), buf2, 2048);
        assert_eq!(tracker.count(), 1);
        tracker.clear();
        assert_eq!(tracker.count(), 0);
    }

    #[test]
    fn test_gpu_conv2d_small_falls_back() {
        let ctx = GpuContext::try_new();
        if let Some(ref ctx) = ctx {
            let input = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
            let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
            let result = gpu_conv2d(ctx, &input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
            assert!(result.is_none(), "small conv should fall back to CPU");
        }
    }
}
