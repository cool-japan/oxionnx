//! Correctness tests for the register-blocked rewrite of `TILED_MATMUL_SHADER`
//! (Agent H2, `oxionnx-gpu/src/context/functions.rs`).
//!
//! ## Why these tests bypass `gpu_matmul` / `gpu_matmul_tiled`
//!
//! Those public wrappers (`compute.rs`) decline below a 10M-FLOP threshold,
//! and additionally fall back to the *other* (non-tiled, `matmul_pipeline`)
//! kernel whenever any of M/K/N is below 32. Most of the ragged/small shapes
//! required here (e.g. `1x1x1`, `3x5x7`, `127x129x63` at ~1M FLOPs) sit under
//! one or both of those gates, so the public API would silently test the
//! *wrong* kernel or no kernel at all. Every test below instead drives
//! `ctx.tiled_matmul_pipeline` / `ctx.tiled_matmul_bind_group_layout`
//! directly (both `pub` fields of [`GpuContext`]), using the *exact* dispatch
//! grid formula `compute.rs` uses in production for this pipeline --
//! `wg_x = ceil(N/16)`, `wg_y = ceil(M/16)` (see `gpu_matmul_tiled_inner` and
//! `plan_conv_gemm` there) -- so a bug in the new kernel's 64x64-macro-tile
//! remap against that *specific* over-provisioned grid is exactly the class
//! of bug these tests are positioned to catch. See the doc comment on
//! `TILED_MATMUL_SHADER` in `context/functions.rs` for why that dispatch
//! formula is still correct for the new kernel without `compute.rs` itself
//! changing.
//!
//! ## Tolerance
//!
//! Mixed absolute/relative: `|gpu - cpu| <= 1e-4 * max(1.0, |cpu|)`. A pure
//! relative bound would be spuriously strict wherever a dot product happens
//! to land near zero, which the signed test-data fill below produces often
//! enough to matter.
//!
//! ## Shape coverage
//!
//! `1x1x1` / `3x5x7` -- degenerate and sub-tile. `127x129x63` -- ragged on
//! all three dims, none a multiple of the 64 macro-tile, 16 dispatch-tile, or
//! 4 register-tile. `512^3` / `1024^3` -- macro-tile-aligned, large. `65^3`
//! -- exactly one macro-tile plus one element, the smallest case that forces
//! a *second*, ragged 64-tile along every axis. `12800x9x64` -- the
//! tall-skinny, tiny-K shape typical of an im2col GEMM. `127x128x64` /
//! `130x256x128` -- ragged M with K and N exact multiples of 4, so the
//! aligned-fast-path write-back also gets exercised with a ragged edge on
//! the *other* axis (every other ragged case here has K or N off a multiple
//! of 4 too, which would leave a future vec4 fast path untested).

use std::sync::mpsc;
use std::time::Duration;

use oxionnx_gpu::GpuContext;
use wgpu::util::DeviceExt;

const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Uniform buffer layout mirroring the shader's `Dims` struct / compute.rs's
/// private `GemmParams` (`M, K, N, _pad` -- 16 bytes, std140-compatible).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Deterministic fill with a signed, non-monotonic pattern (unlike a plain
/// `i % small` ramp) so dot products land near zero often enough to exercise
/// the absolute half of `assert_close`'s tolerance, not just the relative
/// half.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

/// Naive CPU reference in `i, k, j` loop order: algebraically the same
/// sequential-k summation the GPU kernel performs per output cell (loop
/// *nesting* differs, accumulation *order* for any fixed cell does not), but
/// with a streaming inner loop over `j` so 1024^3 finishes in a reasonable
/// time in a test binary.
fn cpu_matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            let b_row = &b[kk * n..(kk + 1) * n];
            for j in 0..n {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    c
}

fn assert_close(gpu: &[f32], cpu: &[f32], m: usize, n: usize, label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    assert_eq!(gpu.len(), m * n, "{label}: unexpected length");
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let g = gpu[idx];
            let c = cpu[idx];
            let tol = 1e-4 * c.abs().max(1.0);
            let err = (g - c).abs();
            assert!(
                err <= tol,
                "{label}: mismatch at ({i},{j}) of {m}x{n}: gpu={g} cpu={c} err={err} tol={tol}"
            );
        }
    }
}

/// Decode a mapped staging buffer's bytes to `f32`s, falling back to a
/// manual little-endian decode if the mapped range is not `bytemuck`-aligned
/// (mirrors `device_guard::decode_f32`'s fallback; that function is
/// crate-private so it cannot be reused directly from here).
fn decode_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    let needed = count * std::mem::size_of::<f32>();
    let src = &bytes[..needed];
    if let Ok(values) = bytemuck::try_cast_slice::<u8, f32>(src) {
        return values.to_vec();
    }
    src.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Synchronous read-back: `map_async` + bounded `poll(Wait)` + `recv_timeout`.
/// Reimplemented here (rather than reusing `device_guard::read_back_blocking`)
/// because that helper is crate-private to `oxionnx-gpu`.
fn read_back_f32(ctx: &GpuContext, staging: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(READBACK_TIMEOUT),
        })
        .expect("device.poll failed during test read-back");

    rx.recv_timeout(READBACK_TIMEOUT)
        .expect("read-back did not complete within the timeout")
        .expect("buffer map_async failed");

    let data = slice.get_mapped_range();
    let out = decode_f32(&data, count);
    drop(data);
    staging.unmap();
    out
}

/// Dispatch `ctx.tiled_matmul_pipeline` directly with the exact grid formula
/// `compute.rs` uses in production for this pipeline. See the module doc
/// comment for why the public `gpu_matmul*` wrappers are not used instead.
fn dispatch_tiled_matmul(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "test bug: `a` is not m*k elements");
    assert_eq!(b.len(), k * n, "test bug: `b` is not k*n elements");

    let device = &ctx.device;
    let queue = &ctx.queue;
    let c_len = m * n;
    let c_bytes = (c_len * std::mem::size_of::<f32>()) as u64;

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("h2_tiled_test_a"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("h2_tiled_test_b"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("h2_tiled_test_c"),
        size: c_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dims = Dims {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("h2_tiled_test_dims"),
        contents: bytemuck::bytes_of(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("h2_tiled_test_staging"),
        size: c_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("h2_tiled_test_bg"),
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
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    // Production dispatch formula for `tiled_matmul_pipeline`
    // (oxionnx-gpu/src/compute.rs: `gpu_matmul_tiled_inner`, `plan_conv_gemm`):
    // x <- N, y <- M, both ceil-divided by 16.
    let wg_x = (n as u32).div_ceil(16);
    let wg_y = (m as u32).div_ceil(16);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("h2_tiled_test_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("h2_tiled_test_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.tiled_matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, c_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    read_back_f32(ctx, &staging, c_len)
}

/// Runs one shape end to end: GPU dispatch vs. CPU reference. Skips (rather
/// than fails) when no adapter is reachable, matching every other GPU test
/// file's convention in this crate.
fn check_shape(m: usize, k: usize, n: usize, label: &str) {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[skip] {label}: no wgpu adapter available");
        return;
    };
    let a = fill(m * k, 2_654_435_761);
    let b = fill(k * n, 40_503);
    let gpu = dispatch_tiled_matmul(&ctx, &a, &b, m, k, n);
    let cpu = cpu_matmul_naive(&a, &b, m, k, n);
    assert_close(&gpu, &cpu, m, n, label);
}

#[test]
fn tiled_matmul_1x1x1() {
    check_shape(1, 1, 1, "1x1x1");
}

#[test]
fn tiled_matmul_3x5x7() {
    check_shape(3, 5, 7, "3x5x7");
}

#[test]
fn tiled_matmul_127x129x63() {
    check_shape(127, 129, 63, "127x129x63");
}

/// Exactly one macro-tile plus one element on every axis: the smallest shape
/// that forces a *second*, ragged 64x64 tile along both M and N.
#[test]
fn tiled_matmul_65x65x65() {
    check_shape(65, 65, 65, "65x65x65");
}

#[test]
fn tiled_matmul_512_cubed() {
    check_shape(512, 512, 512, "512x512x512");
}

#[test]
fn tiled_matmul_1024_cubed() {
    check_shape(1024, 1024, 1024, "1024x1024x1024");
}

/// Tall-skinny, tiny-K shape typical of an im2col GEMM (e.g. a 3x3 conv over
/// a handful of input channels): K=9 is below every tile size in the kernel
/// (4, 16, 64), and below `compute.rs`'s `TILED_MIN_DIM` gate, so production
/// never routes this exact shape to the tiled kernel today -- this test
/// pins the kernel's own correctness at it regardless, independent of that
/// (compute.rs-owned, unrelated-to-this-change) selection heuristic.
#[test]
fn tiled_matmul_im2col_skinny_12800x9x64() {
    check_shape(12800, 9, 64, "12800x9x64");
}

/// Ragged M with K, N exact multiples of 4. Every other ragged case above has
/// K or N off a multiple of 4 too; this pins the aligned-write-back fast
/// path (`out_row_base + TM <= M && out_col_base + TN <= N`) against a
/// ragged M edge specifically.
#[test]
fn tiled_matmul_ragged_m_aligned_kn_127x128x64() {
    check_shape(127, 128, 64, "127x128x64");
}

#[test]
fn tiled_matmul_ragged_m_aligned_kn_130x256x128() {
    check_shape(130, 256, 128, "130x256x128");
}

/// Belt-and-suspenders end-to-end check through the *public* API at a shape
/// that clears both of its gates (`TILED_MIN_DIM`, `GPU_THRESHOLD`), so the
/// real `compute.rs` dispatch call (not just this file's reimplementation of
/// it) is exercised at least once against the rewritten kernel.
#[test]
fn tiled_matmul_public_api_1024_cubed() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("[skip] public_api_1024_cubed: no wgpu adapter available");
        return;
    };
    let m = 1024usize;
    let k = 1024usize;
    let n = 1024usize;
    let a = fill(m * k, 2_654_435_761);
    let b = fill(k * n, 40_503);
    let Some(gpu) = oxionnx_gpu::gpu_matmul_tiled(&ctx, &a, &b, m, k, n) else {
        panic!(
            "gpu_matmul_tiled returned None at {m}x{k}x{n}, which clears both its \
             gates (flops={} >= 10_000_000, all dims >= 32) -- expected Some",
            m * k * n
        );
    };
    let cpu = cpu_matmul_naive(&a, &b, m, k, n);
    assert_close(&gpu, &cpu, m, n, "public_api_1024_cubed");
}
