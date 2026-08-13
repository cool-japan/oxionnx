//! Shared WGSL pipeline-building helpers for the standalone kernel batch in
//! this directory (`broadcast_binary`, `prelu`, `pad`, `resize`, `gemm`,
//! `instance_norm`).
//!
//! ## Why these kernels build their pipeline at their own entry point
//!
//! Every other kernel family in this crate (elementwise, reduction, softmax,
//! normalization, transpose, matmul) has its `wgpu::ComputePipeline` built
//! eagerly in `GpuContext::build_from_device_queue` (`context/types.rs`) and
//! cached as a field, so a dispatch only ever creates buffers and a bind group.
//! The kernels below construct theirs lazily instead, inside their own
//! `pub fn gpu_*` entry point, via a [`build_pipeline`] call factored out here.
//!
//! That is now a difference in *when*, not in *where*: since \[w5\] the result
//! is memoized on the [`GpuContext`](crate::GpuContext) itself
//! (`context::pipeline_cache`), exactly like the eager fields, so each
//! `(label, entry_point, src)` is compiled once per context rather than once
//! per call. Lazily is the right time for these: they are the kernels a given
//! graph may never dispatch at all (`instance_norm` and the resize variants in
//! particular), and a context that never runs one never pays for its shader.
//!
//! The cost that memoization removes was real and measured. On an M3
//! (`examples/r3a_cost_breakdown.rs`), a `pad` dispatch cost 0.18-0.61 ms more
//! than a `relu` dispatch moving the same bytes, and that difference *is* the
//! pipeline construction, since `relu`'s pipeline is a cached `GpuContext`
//! field and `pad`'s was not. Across one InSwapper frame the five kernels in
//! this batch are dispatched ~89 times, so it was tens of milliseconds of pure
//! recompilation per frame on native — and browsers, which must translate WGSL
//! to the platform shading language on each `createComputePipeline`, pay
//! substantially more than that.
//!
//! ## Why the cache moved onto the context
//!
//! It used to be a thread-local keyed on the `wgpu::Device` handle. That is
//! unsound on wgpu 29: handle equality is a per-`Instance` id, this crate
//! creates one `Instance` per context, and the second context in a process gets
//! a device that compares *equal* to the first's. The full account, and the
//! `BindGroupLayout[Id(9,1)] does not exist` panic it caused, is in
//! `crate::context::pipeline_cache`.
//!
//! ## No minimum-size threshold
//!
//! Unlike `elementwise.rs` / `common.rs`'s `EW_GPU_THRESHOLD`-style gates,
//! none of the kernels built on top of this file decline purely because a
//! tensor is "too small to bother" — only for inputs that are actually invalid
//! (shape mismatches, buffers the device cannot bind, dispatches wider than the
//! device allows). A CPU/GPU placement heuristic belongs in the session
//! dispatcher that calls these, not baked into the kernel, and folding one in
//! here would make it impossible to verify parity at the 1-element shapes this
//! wave's kernels are required to cover.

use wgpu::BindGroupLayoutEntry;

use crate::context::GpuContext;

/// Workgroup size (threads along X) used by every "one thread per output
/// element" kernel in this batch (`broadcast_binary`, `prelu`, `pad`,
/// `resize`). Mirrors `shaders::common::WG_SIZE`, duplicated here because
/// that constant is private to the `elementwise.rs` / `reduction.rs` group
/// of files and this module cannot add to it (see the module docs above).
pub(super) const WG_SIZE: u32 = 256;

/// A read-only storage buffer binding at `binding`.
pub(super) fn bgl_ro(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A read-write storage buffer binding at `binding`.
pub(super) fn bgl_rw(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A uniform buffer binding at `binding`.
pub(super) fn bgl_uniform(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Build a single-entry-point compute pipeline from WGSL source, or return the
/// one `ctx` already compiled for this `(label, entry_point, src)`.
///
/// `label` names the shader module, pipeline and (suffixed) bind group layout
/// for GPU-debugger visibility; `entry_point` selects which `@compute` function
/// in `wgsl_src` this pipeline runs.
///
/// Taking the whole context rather than `&ctx.device` is the point: the memo
/// lives on the context, so the device that compiled an entry is identified by
/// the entry's *location* and never by comparing handles. See
/// `crate::context::pipeline_cache` for the cache key and for what went wrong
/// when it was the other way round.
pub(super) fn build_pipeline(
    ctx: &GpuContext,
    label: &str,
    wgsl_src: &'static str,
    entry_point: &str,
    bgl_entries: &[BindGroupLayoutEntry],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    ctx.pipelines()
        .get_or_build(&ctx.device, label, wgsl_src, entry_point, bgl_entries)
}

#[cfg(test)]
mod tests {
    use crate::context::GpuContext;
    use crate::shaders::{
        gpu_broadcast_add, gpu_broadcast_div, gpu_broadcast_mul, gpu_broadcast_sub, gpu_pad,
        gpu_resize_bilinear_pytorch_half_pixel, gpu_resize_nearest_asymmetric, PadMode,
    };

    /// Interleaving the four broadcast ops must give four different answers.
    ///
    /// This is the failure the pipeline cache could plausibly introduce, and
    /// it is invisible from the outside. `broadcast_binary` compiles all four
    /// of its entry points from **one** WGSL source under **one** label, and
    /// `pad`/`resize` do the same for two each. A cache keyed on `label`
    /// alone would hand whichever pipeline was compiled first to all of them
    /// — so every `Mul` node in a graph would quietly compute `Add`, with the
    /// correct output shape and no error anywhere in the stack.
    ///
    /// Each op is run twice, on either side of the others, so both the
    /// compile path (first call) and the cache-hit path (second call) are
    /// checked against the same expected values.
    #[test]
    fn interleaved_entry_points_under_one_label_stay_distinct() {
        let Some(ctx) = GpuContext::try_new() else {
            return; // No adapter on this machine.
        };
        let a = [6.0f32, 8.0, 10.0, 12.0];
        let b = [2.0f32, 4.0, 5.0, 3.0];
        let shape = [1usize, 1, 1, 4];

        /// One broadcast entry point plus the answer it must give.
        type BroadcastFn = fn(&GpuContext, &[f32], &[usize], &[f32], &[usize]) -> Option<Vec<f32>>;
        let cases: [(&str, BroadcastFn, [f32; 4]); 4] = [
            ("add", gpu_broadcast_add, [8.0, 12.0, 15.0, 15.0]),
            ("sub", gpu_broadcast_sub, [4.0, 4.0, 5.0, 9.0]),
            ("mul", gpu_broadcast_mul, [12.0, 32.0, 50.0, 36.0]),
            ("div", gpu_broadcast_div, [3.0, 2.0, 2.0, 4.0]),
        ];
        // Two passes: the first compiles each pipeline, the second must be
        // served from the cache and must still be the right one.
        for pass in 0..2 {
            for (name, op, want) in &cases {
                let got = op(&ctx, &a, &shape, &b, &shape)
                    .unwrap_or_else(|| panic!("{name} declined on pass {pass}"));
                assert_eq!(got, want.to_vec(), "{name} wrong on pass {pass}");
            }
        }
    }

    /// `pad`'s two modes share a label and a shader source.
    #[test]
    fn pad_modes_do_not_share_a_cached_pipeline() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        // [1,1,1,3] padded by 1 on the left only.
        let data = [1.0f32, 2.0, 3.0];
        let shape = [1usize, 1, 1, 3];
        for pass in 0..2 {
            let reflect = gpu_pad(&ctx, &data, &shape, 0, 0, 1, 0, PadMode::Reflect, 0.0)
                .unwrap_or_else(|| panic!("reflect declined on pass {pass}"));
            let constant = gpu_pad(&ctx, &data, &shape, 0, 0, 1, 0, PadMode::Constant, -7.0)
                .unwrap_or_else(|| panic!("constant declined on pass {pass}"));
            // reflect mirrors across index 0 -> [2, 1, 2, 3]
            assert_eq!(reflect, vec![2.0, 1.0, 2.0, 3.0], "pass {pass}");
            assert_eq!(constant, vec![-7.0, 1.0, 2.0, 3.0], "pass {pass}");
        }
    }

    /// `resize`'s two kinds share a label and a shader source.
    #[test]
    fn resize_kinds_do_not_share_a_cached_pipeline() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        // [1,1,1,2] -> width 4. Nearest/asymmetric duplicates each sample;
        // bilinear interpolates, so the two must differ in the interior.
        let data = [0.0f32, 4.0];
        let shape = [1usize, 1, 1, 2];
        for pass in 0..2 {
            let nearest = gpu_resize_nearest_asymmetric(&ctx, &data, &shape, 1, 4)
                .unwrap_or_else(|| panic!("nearest declined on pass {pass}"));
            let bilinear = gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &shape, 1, 4)
                .unwrap_or_else(|| panic!("bilinear declined on pass {pass}"));
            assert_eq!(nearest, vec![0.0, 0.0, 4.0, 4.0], "pass {pass}");
            assert_ne!(
                nearest, bilinear,
                "pass {pass}: the two resize kinds returned identical output, \
                 which means one was served the other's cached pipeline",
            );
        }
    }
}
