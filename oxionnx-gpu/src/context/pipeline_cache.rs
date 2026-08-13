//! \[w5\] The compiled-pipeline cache, owned by the [`GpuContext`] whose device
//! compiled every entry in it.
//!
//! # Why this is not a thread-local keyed on the device
//!
//! It was, and that was a crash. The caches this module replaces
//! (`shaders::kernel_support::PIPELINES`, `shaders::conv2d::CONV2D_PIPELINE`
//! and its `f16` negative cache, `shaders::gemm`'s two `f16` verdict caches)
//! were thread-locals whose entries stored a `wgpu::Device` and were looked up
//! with `&cached.device == device`. That key rests on a premise wgpu does not
//! provide:
//!
//! * `wgpu::Device`'s `PartialEq` proxies to `CoreDevice.id`
//!   (`wgpu/src/backend/wgpu_core.rs`, `impl_eq_ord_hash_proxy!(CoreDevice =>
//!   .id)`), an `(index, epoch)` pair — **not** the address of the `Arc` behind
//!   the handle.
//! * Those ids are allocated by the `wgpu_core::Global` inside one
//!   `wgpu::Instance`, and every allocator starts from zero.
//! * [`GpuContext::try_new`](crate::GpuContext::try_new) builds a **fresh
//!   `Instance` per context**.
//!
//! So the second context in a process receives a device whose id equals the
//! first's, the lookup reports a hit, and the dispatch is handed the *other*
//! device's `ComputePipeline` and `BindGroupLayout`. `create_bind_group` then
//! resolves that layout id against the second instance's registry, where it
//! does not exist, and wgpu-core panics — `BindGroupLayout[Id(9,1)] does not
//! exist`, `wgpu-core/src/storage.rs`. Any command that builds two sessions
//! (`oxiface --device gpu detect --embed`, `... swap`) hit it on the second
//! session's first convolution. Dropping the first context first does not help:
//! releasing its ids makes the collision *certain* rather than merely likely.
//!
//! Keying harder cannot fix this, because wgpu exposes no device identity that
//! is unique across instances. Ownership can, and does: the cache is a field of
//! the context, so "which device compiled this" is answered by *where the entry
//! lives* rather than by comparing handles. There is no cross-context lookup to
//! get wrong, no eviction rule to get wrong, and no purge to forget — the cache
//! dies with the context, on the context's own thread (which
//! `oxionnx::session::gpu_owner` guarantees is the thread that created it), for
//! the same reason `super::resident`'s buffers do.
//!
//! # What one entry is keyed on
//!
//! `(label, entry_point, src)`. `label` names the shader module and pipeline,
//! `entry_point` selects which `@compute` function in the source this pipeline
//! runs — `broadcast_binary` compiles four entry points from one source under
//! one label, and `pad`/`resize` two each, so the entry point is load-bearing:
//! keying on `label` alone would make every `Mul` node quietly compute `Add`.
//! `src` is the correctness backstop for a `(label, entry_point)` collision
//! across two different sources; every in-crate caller maps one label to one
//! `const`, so it never fires, but that invariant is enforced here rather than
//! assumed. `bgl_entries` is deliberately *not* part of the key: every caller
//! passes a fixed entry list for a given label, so comparing it could never
//! change the answer. A future caller needing two layouts under one label must
//! use a different label.
//!
//! # Rejected variants live in the same table
//!
//! [w2-f16] The `f16` variants of `conv2d_implicit` and `gemm_nt` are optional
//! fast paths compiled from a derived source. A driver that refuses one must
//! not degrade the context (that would send *every* node of *every* op to the
//! CPU for the rest of the session), so the refusal is remembered as a
//! [`Compiled::Rejected`] slot under the same key and the kernel keeps taking
//! its `f32` path. One table, three answers — ready, refused, never asked —
//! replaces the four separate positive/negative device-keyed caches this
//! module's predecessors kept.

use std::sync::Mutex;

use wgpu::BindGroupLayoutEntry;

/// What a cache slot holds: the compiled pipeline, or the record that this
/// device refused to compile that source.
enum Compiled {
    Ready {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
    },
    /// This device rejected this shader. See the module docs: a decline, not a
    /// dead device.
    Rejected,
}

/// One cache slot, tagged with the identity described in the module docs.
struct Entry {
    label: String,
    entry_point: String,
    src: &'static str,
    compiled: Compiled,
}

impl Entry {
    /// Whether this slot is the one for `(label, entry_point, src)`.
    ///
    /// The `label` comparison short-circuits for all but one candidate, and the
    /// `src` comparison tries pointer equality first because every caller passes
    /// a `const` static — the content compare is the backstop, not the hot path.
    fn matches(&self, label: &str, entry_point: &str, src: &str) -> bool {
        self.label == label
            && self.entry_point == entry_point
            && (std::ptr::eq(self.src, src) || self.src == src)
    }
}

/// What [`PipelineCache::lookup`] found.
pub(crate) enum PipelineLookup {
    /// Compiled and ready.
    Ready(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    /// This device already refused this source once; do not retry it.
    Rejected,
    /// Never compiled on this context.
    Absent,
}

/// Every compute pipeline one [`crate::GpuContext`] has compiled.
///
/// A `Vec` with a linear scan rather than a map: a context holds at most a
/// dozen entries (4 broadcast ops, 2 pad modes, 2 resize kinds, prelu, gemm,
/// instance_norm, conv, plus the two `f16` variants), so scanning is cheaper
/// than hashing two strings.
///
/// The `Mutex` is what keeps `GpuContext: Sync` on native, which
/// `oxionnx::session` asserts for `Session`. It is taken only around a lookup
/// or an insert, never across a compile, so it is never held while the driver
/// is working. A poisoned lock degrades to compiling uncached rather than
/// panicking — the caller gets a correct pipeline either way.
#[derive(Default)]
pub(crate) struct PipelineCache {
    entries: Mutex<Vec<Entry>>,
}

impl PipelineCache {
    /// The pipeline for `(label, entry_point, src)`, compiling it on a miss.
    ///
    /// `device` must be the device this cache's context owns; it is passed in
    /// rather than stored precisely so that no entry can outlive it or be found
    /// by another device's dispatch.
    ///
    /// A `Rejected` slot is treated as a miss and recompiled, because this
    /// entry point is for the kernels whose shader is not optional — there is
    /// no fallback to decline to, so refusing to build would be refusing to run
    /// the node at all. The optional `f16` variants never come through here:
    /// they compile inside their own error scope and record the verdict
    /// themselves ([`Self::lookup`] / [`Self::insert_rejected`]).
    pub(crate) fn get_or_build(
        &self,
        device: &wgpu::Device,
        label: &str,
        src: &'static str,
        entry_point: &str,
        bgl_entries: &[BindGroupLayoutEntry],
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        if let PipelineLookup::Ready(pipeline, layout) = self.lookup(label, entry_point, src) {
            return (pipeline, layout);
        }
        let built = compile(device, label, src, entry_point, bgl_entries);
        self.insert_ready(label, entry_point, src, &built);
        built
    }

    /// Look `(label, entry_point, src)` up without compiling anything.
    ///
    /// The `f16` paths need this separately from [`Self::get_or_build`]: a first
    /// compile there has to happen inside its own error scope, and a hit must
    /// not pay for pushing and popping one.
    pub(crate) fn lookup(&self, label: &str, entry_point: &str, src: &str) -> PipelineLookup {
        let Ok(entries) = self.entries.lock() else {
            return PipelineLookup::Absent;
        };
        match entries.iter().find(|e| e.matches(label, entry_point, src)) {
            Some(Entry {
                compiled: Compiled::Ready { pipeline, layout },
                ..
            }) => PipelineLookup::Ready(pipeline.clone(), layout.clone()),
            Some(Entry {
                compiled: Compiled::Rejected,
                ..
            }) => PipelineLookup::Rejected,
            None => PipelineLookup::Absent,
        }
    }

    /// Record a successfully compiled pipeline under `(label, entry_point, src)`.
    pub(crate) fn insert_ready(
        &self,
        label: &str,
        entry_point: &str,
        src: &'static str,
        built: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    ) {
        self.insert(
            label,
            entry_point,
            src,
            Compiled::Ready {
                pipeline: built.0.clone(),
                layout: built.1.clone(),
            },
        );
    }

    /// Record that this context's device refused to compile `src`.
    pub(crate) fn insert_rejected(&self, label: &str, entry_point: &str, src: &'static str) {
        self.insert(label, entry_point, src, Compiled::Rejected);
    }

    /// Replace, or add, the slot for `(label, entry_point, src)`.
    ///
    /// Replacing rather than pushing matters for one case only: two dispatches
    /// racing the same first compile on two threads. Both compile, both insert,
    /// and without the replace the loser's entry would shadow nothing but would
    /// grow the table without bound over a long session.
    fn insert(&self, label: &str, entry_point: &str, src: &'static str, compiled: Compiled) {
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        let slot = Entry {
            label: label.to_string(),
            entry_point: entry_point.to_string(),
            src,
            compiled,
        };
        match entries
            .iter_mut()
            .find(|e| e.matches(label, entry_point, src))
        {
            Some(existing) => *existing = slot,
            None => entries.push(slot),
        }
    }

    /// Drop every compiled pipeline and every remembered verdict.
    ///
    /// Called from [`crate::GpuContext`]'s `Drop`, for the ordering rather than
    /// for the release: a `ComputePipeline` holds its own handle on the device,
    /// and `GpuContext` declares `device` before this field, so leaving it to
    /// drop glue would destroy the context's device handle *first* and the
    /// pipelines afterwards. Clearing here destroys them while the context
    /// unambiguously still holds its device — the same reason `super::budget`'s
    /// buffers and `super::resident`'s weights are released explicitly there,
    /// and the ordering `oxionnx::session::gpu_owner`'s one-thread rule is
    /// about.
    pub(crate) fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }

    /// How many pipelines this context has compiled. Test-only.
    ///
    /// A statement about *one* context, which is what makes it order
    /// independent: unlike the thread-local counts this replaces, no other test
    /// running on the same thread can move it.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().map_or(0, |entries| entries.len())
    }
}

/// The `create_shader_module` / `create_compute_pipeline` sequence, with no
/// memoization.
///
/// Public to the crate because the `f16` paths compile inside an error scope of
/// their own and record the verdict themselves; everything else goes through
/// [`PipelineCache::get_or_build`]. Also what the cache's parity test compiles a
/// second, independent pipeline with.
pub(crate) fn compile(
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
    use super::{Compiled, Entry, PipelineCache, PipelineLookup};

    /// A slot with no pipeline in it, so the table's bookkeeping can be tested
    /// without a device. `Compiled::Rejected` is the one variant that carries no
    /// wgpu handle, which is what makes this possible.
    fn rejected(label: &str, entry_point: &str, src: &'static str) -> Entry {
        Entry {
            label: label.to_string(),
            entry_point: entry_point.to_string(),
            src,
            compiled: Compiled::Rejected,
        }
    }

    /// The identity rule: label, entry point and source all participate.
    #[test]
    fn a_slot_matches_only_its_own_full_key() {
        let slot = rejected("bcast", "bcast_add", "SOURCE A");
        assert!(slot.matches("bcast", "bcast_add", "SOURCE A"));
        // Same label and source, different entry point -- the `broadcast_binary`
        // case, where four ops share one label and one shader.
        assert!(!slot.matches("bcast", "bcast_mul", "SOURCE A"));
        assert!(!slot.matches("pad", "bcast_add", "SOURCE A"));
        // Same label and entry point, different source: a miss, not a silently
        // wrong pipeline.
        assert!(!slot.matches("bcast", "bcast_add", "SOURCE B"));
    }

    /// The source comparison must be by *content*, not only by address, so two
    /// equal strings at different addresses still hit.
    #[test]
    fn equal_sources_at_different_addresses_still_match() {
        let owned = String::from("SOURCE A");
        let slot = rejected("bcast", "bcast_add", "SOURCE A");
        assert!(!std::ptr::eq(slot.src.as_ptr(), owned.as_ptr()));
        assert!(slot.matches("bcast", "bcast_add", &owned));
    }

    /// A rejection is remembered, distinguishable from "never asked", and
    /// scoped to its own key.
    #[test]
    fn a_rejected_variant_is_remembered_without_shadowing_its_siblings() {
        let cache = PipelineCache::default();
        assert!(matches!(
            cache.lookup("gemm_nt_f16", "gemm_nt", "F16 SRC"),
            PipelineLookup::Absent
        ));

        cache.insert_rejected("gemm_nt_f16", "gemm_nt", "F16 SRC");
        assert!(matches!(
            cache.lookup("gemm_nt_f16", "gemm_nt", "F16 SRC"),
            PipelineLookup::Rejected
        ));
        // The f32 pipeline shares the entry point but not the label, so the
        // rejection must not reach it.
        assert!(matches!(
            cache.lookup("gemm_nt", "gemm_nt", "F32 SRC"),
            PipelineLookup::Absent
        ));
        assert_eq!(cache.len(), 1);

        // Recording the same verdict twice replaces the slot rather than
        // growing the table.
        cache.insert_rejected("gemm_nt_f16", "gemm_nt", "F16 SRC");
        assert_eq!(cache.len(), 1);
    }
}
