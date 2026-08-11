//! Shared WGSL pipeline-building helpers for the standalone kernel batch in
//! this directory (`broadcast_binary`, `prelu`, `pad`, `resize`, `gemm`).
//!
//! ## Why these kernels rebuild their pipeline on every call
//!
//! Every other kernel family in this crate (elementwise, reduction, softmax,
//! normalization, transpose, matmul) has its `wgpu::ComputePipeline` built
//! once in `GpuContext::build_from_device_queue` (`context/types.rs`) and
//! cached as a field, so a dispatch only ever creates buffers and a bind
//! group. The five kernels added alongside this file cannot follow that
//! pattern this wave: adding a cached-pipeline field means editing
//! `context/types.rs`, and its WGSL source constant would live in
//! `context/functions.rs` — both owned by this wave's session-integration
//! work landing in the same cycle (see the wave's file-ownership split). So
//! each kernel below builds its shader module, bind group layout and
//! pipeline fresh inside its own `pub fn gpu_*` entry point, via a
//! `build_pipeline` call factored out here for exactly one reason: turning
//! this into a cached-once-on-`GpuContext` field later is then a call-site
//! hoist (call `build_*_pipeline` once in `build_from_device_queue`, store
//! the result, thread it through) rather than a rewrite of the dispatch
//! logic. Each kernel file exposes its own `pub(crate) fn build_*_pipeline`
//! wrapping [`build_pipeline`] for exactly that reason -- `pub(crate)`
//! (rather than `pub(super)`, which [`build_pipeline`] itself uses) because
//! `context/types.rs` calling in from a sibling module is the whole point;
//! `shaders/mod.rs` re-exports each one with `pub(crate) use` so the future
//! caller does not also need `broadcast_binary`/`pad`/etc. to be public
//! submodules.
//!
//! This had a real, known cost — shader compilation on every call. [r3a] That
//! cost is now paid once per `(device, label, entry_point)` instead: see
//! [`build_pipeline`], which memoizes into a thread-local keyed on the device
//! handle, generalizing the one-entry cache `conv2d.rs` already kept for
//! itself. The per-kernel `build_*_pipeline` wrappers and their call sites are
//! unchanged, so the eventual hoist onto `GpuContext` is still the call-site
//! move described above — this is a strictly cheaper stop on the way there,
//! not a replacement for it.
//!
//! Why it was worth doing here rather than deferring again: measured on an M3
//! (`examples/r3a_cost_breakdown.rs`), a `pad` dispatch cost 0.18–0.61 ms more
//! than a `relu` dispatch moving the same bytes, and that difference *is* the
//! pipeline construction, since `relu`'s pipeline is a cached `GpuContext`
//! field and `pad`'s was not. Across one InSwapper frame the five kernels in
//! this batch are dispatched ~89 times, so it was tens of milliseconds of pure
//! recompilation per frame on native — and browsers, which must translate WGSL
//! to the platform shading language on each `createComputePipeline`, pay
//! substantially more than that.
//!
//! ## No minimum-size threshold
//!
//! Unlike `elementwise.rs` / `common.rs`'s `EW_GPU_THRESHOLD`-style gates,
//! none of the five kernels built on top of this file decline purely because
//! a tensor is "too small to bother" — only for inputs that are actually
//! invalid (shape mismatches, buffers the device cannot bind, dispatches
//! wider than the device allows). A CPU/GPU placement heuristic belongs in
//! the session dispatcher that calls these, not baked into the kernel, and
//! folding one in here would make it impossible to verify parity at the
//! 1-element shapes this wave's kernels are required to cover. A future
//! integration wave that wires these into session dispatch should add that
//! heuristic at the call site, the same way `common.rs`'s thresholds gate
//! `elementwise.rs` today.

use wgpu::BindGroupLayoutEntry;

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

/// One memoized pipeline, tagged with everything that identifies it.
///
/// The `wgpu::Device` is **stored, not merely compared**: wgpu handle equality
/// is `Arc` identity, so holding the handle is what stops a later, different
/// device from comparing equal by landing in a freed slot. (Same reasoning as
/// `conv2d.rs`'s `CachedPipeline`, which this generalizes.) Storing it means
/// the entry keeps its device alive, so insertion goes through
/// [`insert_for_current_device`], which evicts the other devices' entries
/// first — see there for why that eviction is what makes dropping a
/// `GpuContext` actually release its device.
struct CachedPipeline {
    device: wgpu::Device,
    label: String,
    entry_point: String,
    /// The WGSL this was compiled from, kept so a `(label, entry_point)`
    /// collision across two different sources is a cache *miss* rather than a
    /// silently wrong pipeline. Every in-crate caller maps one `label` to one
    /// `const` source, so this never actually fires — it is here so that
    /// invariant is enforced rather than assumed.
    src: &'static str,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

thread_local! {
    /// Compiled pipelines for this thread, in insertion order.
    ///
    /// A `Vec` with a linear scan rather than a map: the whole batch is at
    /// most eleven entries (4 broadcast ops, 2 pad modes, 2 resize kinds,
    /// prelu, gemm, instance_norm), so scanning is cheaper than hashing two
    /// strings, and the comparison short-circuits on `label` for all but one
    /// candidate.
    ///
    /// Thread-local rather than a `static Mutex<_>` because `wgpu::Device` is
    /// neither `Send` nor `Sync` on wasm32, where a `static` would not
    /// compile at all. Native worker threads each pay one extra compile per
    /// kernel and nothing else.
    static PIPELINES: std::cell::RefCell<Vec<CachedPipeline>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Push `entry` onto a thread-local device-keyed cache, first dropping every
/// entry that does not belong to the device being inserted for.
///
/// # Why the eviction is not optional
///
/// Every cache in this crate that memoizes per device stores the
/// `wgpu::Device` handle itself, because wgpu handle equality is `Arc`
/// identity and only holding the handle stops a later, different device from
/// comparing equal by landing in a freed slot. The consequence is that each
/// entry keeps its device alive: an append-only cache pins every device the
/// thread has ever touched until the thread exits, so a session that drops its
/// `GpuContext` releases only *its* handle and the driver never tears the
/// device down. Evicting on insert bounds each cache at a single device — the
/// one currently in use — and is what makes dropping a context release its
/// device for real.
///
/// The predicate, not the entry, defines "current": `PIPELINES` keeps every
/// label compiled for that device, and `conv2d`'s cache keeps both the `f32`
/// and the `f16` entry, because both predicates test the device alone.
///
/// # The tradeoff this accepts
///
/// Two live contexts alternating dispatches on one thread now evict each
/// other, so each dispatch recompiles instead of hitting. That is slower, not
/// wrong — a miss rebuilds — and nothing in this crate does it: a session owns
/// one context, and a context is driven from the thread that built it.
pub(super) fn insert_for_current_device<T>(
    cache: &mut Vec<T>,
    belongs_to_current_device: impl Fn(&T) -> bool,
    entry: T,
) {
    cache.retain(belongs_to_current_device);
    cache.push(entry);
}

/// Build a single-entry-point compute pipeline from WGSL source, or return the
/// one already compiled for this `(device, label, entry_point, src)`.
///
/// `label` names the shader module, pipeline and (suffixed) bind group
/// layout for GPU-debugger visibility; `entry_point` selects which
/// `@compute` function in `wgsl_src` this pipeline runs.
///
/// # Cache key
///
/// `bgl_entries` is deliberately **not** part of the key. Every caller in this
/// crate passes a fixed entry list for a given `label`, so including it would
/// add a per-call comparison that can never change the answer. If a future
/// caller ever needs two different layouts under one label, it must use a
/// different label — that is the contract this doc comment establishes.
pub(super) fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl_src: &'static str,
    entry_point: &str,
    bgl_entries: &[BindGroupLayoutEntry],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let hit = PIPELINES.with(|cell| {
        let cache = cell.borrow();
        cache
            .iter()
            .find(|c| {
                c.label == label
                    && c.entry_point == entry_point
                    && &c.device == device
                    // Pointer equality first: every caller passes a `const`
                    // static, so this is the real path. The content compare is
                    // the correctness backstop, not the hot path.
                    && (std::ptr::eq(c.src, wgsl_src) || c.src == wgsl_src)
            })
            .map(|c| (c.pipeline.clone(), c.layout.clone()))
    });
    if let Some(found) = hit {
        return found;
    }

    let built = build_pipeline_uncached(device, label, wgsl_src, entry_point, bgl_entries);
    PIPELINES.with(|cell| {
        insert_for_current_device(
            &mut cell.borrow_mut(),
            |c| &c.device == device,
            CachedPipeline {
                device: device.clone(),
                label: label.to_string(),
                entry_point: entry_point.to_string(),
                src: wgsl_src,
                pipeline: built.0.clone(),
                layout: built.1.clone(),
            },
        );
    });
    built
}

/// The actual `create_shader_module` / `create_compute_pipeline` sequence,
/// with no memoization. Split out so the parity test below can compile a
/// second, independent pipeline and compare results against the cached one.
fn build_pipeline_uncached(
    device: &wgpu::Device,
    label: &str,
    wgsl_src: &str,
    entry_point: &str,
    bgl_entries: &[BindGroupLayoutEntry],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
    });
    let bgl_label = format!("{label}_bgl");
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&bgl_label),
        entries: bgl_entries,
    });
    let pl_label = format!("{label}_pl");
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&pl_label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bgl)
}

#[cfg(test)]
mod tests {
    use crate::context::GpuContext;
    use crate::shaders::{
        gpu_broadcast_add, gpu_broadcast_div, gpu_broadcast_mul, gpu_broadcast_sub, gpu_pad,
        gpu_resize_bilinear_pytorch_half_pixel, gpu_resize_nearest_asymmetric, PadMode,
    };

    /// The eviction contract of [`super::insert_for_current_device`], on a stub
    /// entry type so that it needs no device.
    ///
    /// A `wgpu::Device` cannot be constructed without a device, and a
    /// device-backed assertion about a cache's *contents* would be
    /// order-dependent — the thread-local is shared by every test that runs on
    /// the thread. What is device-independent, and what the leak fix actually
    /// turns on, is the rule itself: everything that does not belong to the
    /// device being inserted for is dropped, everything that does is kept.
    #[test]
    fn insert_evicts_other_devices_and_keeps_the_current_one() {
        // `(device, variant)` pairs, standing in for `CachedPipeline`.
        let mut cache: Vec<(u8, &str)> = vec![(1, "f32"), (1, "f16"), (2, "f32")];

        // Inserting for device 1 keeps *both* of its entries and drops device
        // 2's: the f32/f16 pair of the live device is exactly what must
        // survive, since the predicate tests the device alone.
        super::insert_for_current_device(&mut cache, |e| e.0 == 1, (1, "gemm"));
        assert_eq!(cache, vec![(1, "f32"), (1, "f16"), (1, "gemm")]);

        // A different device evicts all of the previous one's entries — the
        // handle release an append-only cache never performed.
        super::insert_for_current_device(&mut cache, |e| e.0 == 3, (3, "f32"));
        assert_eq!(cache, vec![(3, "f32")]);

        // On an empty cache it degenerates to a push.
        let mut empty: Vec<(u8, &str)> = Vec::new();
        super::insert_for_current_device(&mut empty, |e| e.0 == 7, (7, "f32"));
        assert_eq!(empty, vec![(7, "f32")]);
    }

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
