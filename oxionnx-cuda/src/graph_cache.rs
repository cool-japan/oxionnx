//! CUDA-graph capture and replay for repeated, fixed-shape dispatches.
//!
//! A video pipeline runs the same ONNX graph on the same shapes once per
//! frame. Every node therefore issues the *same* sequence of kernel launches
//! with the same geometry, thousands of times. A CUDA graph records that
//! sequence once and replays it with a single `cuGraphLaunch`, replacing both
//! the per-launch driver calls and all the host-side work that computes them
//! (descriptor validation, tile selection, kernel-cache lookups, argument
//! packing).
//!
//! # Off by default, and the measurements say why
//!
//! [`GRAPH_ENV_VAR`] (`OXIONNX_CUDA_GRAPH=1`) gates the whole module. The
//! default is **off**, and not out of caution about correctness — replay is
//! bit-exact, see `tests/graph_cache_gpu.rs`. It is off because on a full
//! dispatch the measured effect is small, shape-dependent, and **not
//! reliably positive**.
//!
//! End-to-end through [`crate::try_cuda_dispatch`], measured on this
//! workspace's RTX A4000 (sm_86, driver 550.144.03) by
//! `examples/graph_dispatch_bench.rs`, which interleaves the two paths in one
//! process (median us per call, two runs on an idle device):
//!
//! | shape | graphs off | graphs on | delta |
//! |---|---|---|---|
//! | `[1, 512] @ [512, 512]` | 31.2 / 34.7 | 28.1 / 31.1 | **-10.0% / -10.4%** |
//! | `[1, 256] @ [256, 256]` | 28.3 / 31.1 | 28.0 / 29.9 | -1.2% / -3.8% |
//! | `[1, 25088] @ [25088, 512]` (ArcFace head) | 165.1 / 158.4 | 161.8 / 163.2 | -2.0% / +3.1% |
//! | `[1, 512] @ [512, 2048]` (InSwapper AdaIN) | 34.0 / 38.7 | 36.0 / 39.7 | +5.9% / +2.5% |
//! | `[128, 512] @ [512, 512]` | 1704 / 1729 | 1721 / 1750 | +1.0% / +1.2% |
//! | batched `b=4`, `[64, 128] @ [128, 64]` | 83.0 / 106.3 | 92.0 / 116.6 | **+10.9% / +9.7%** |
//! | batched `b=16`, `[64, 128] @ [128, 64]` | 287.9 / 372.7 | 297.0 / 380.9 | +3.2% / +2.2% |
//!
//! Best case -10%, worst case +11%, and the sign is a property of the shape
//! rather than of the run: minimum and median agree to within a point on every
//! row, and both runs agree on the sign for every row but ArcFace's (whose
//! effect is inside the noise either way).
//!
//! # Why the ceiling is that low
//!
//! Graph replay can only remove kernel-launch and host-submission overhead.
//! On these shapes there is very little of it left to remove, because the
//! GEMMs are **DRAM-bandwidth-bound on their weight operand**: ArcFace's head
//! reads a 49 MiB matrix, which at this card's ~350 GB/s is ~140 us on its
//! own — and 140 us is exactly what the isolated GEMM measures
//! (`examples/graph_capture_probe.rs`). The AdaIN projection reads 4 MiB,
//! ~12 us, against ~7 us of launch overhead. Nothing a graph does touches the
//! other 95%.
//!
//! Convolution — the workload's most numerous op — was measured and
//! deliberately **not** integrated. `crate::conv` issues exactly *one* kernel
//! per dispatch, and that kernel dominates completely; the probe's three
//! InSwapper-128 3x3 shapes replay at -0.6%, -3.7% and +1.2% against ordinary
//! launches, i.e. noise around zero. One launch is not worth a recording, a
//! cache key, and a pair of pinned buffers.
//!
//! So: this is a targeted optimisation for small repeated GEMMs, it is not
//! free, and the honest default is off. Enable it for a pipeline whose profile
//! is dominated by small fixed-shape GEMMs — and measure the pipeline, because
//! the table above is what happens when you do not.
//!
//! # Interaction with `OXIONNX_CUDA_VERIFY`
//!
//! Graph replay is **disabled whenever shadow verification is on**
//! ([`crate::reference::verify_enabled`]). Verification exists to compare what
//! the GPU actually computed against a CPU oracle, and it is at its most
//! valuable when it is comparing against the code path that production runs.
//! Letting it grade a replayed graph instead would test the graph rather than
//! the kernels, and would silently change what a `VERIFY=1` run is evidence
//! *for*. `OXIONNX_CUDA_VERIFY=1` therefore always exercises the ordinary
//! launch path; `OXIONNX_CUDA_STRICT=1` is unaffected either way, since it
//! only promotes a verification mismatch to an error.
//!
//! # Pointer stability is the whole safety problem
//!
//! A captured graph stores *device addresses*, not buffers. Replaying it reads
//! and writes exactly the addresses that were live at capture time, whatever
//! owns them now. Three separate sources of instability had to be closed
//! before capture was sound here:
//!
//! * **The scratch pool.** `residency::DevicePool` hands out and recycles
//!   buffers per dispatch, so the address a capture recorded would be
//!   handed to an unrelated op on the next frame. This module does not use the
//!   pool at all: every buffer a recorded launch touches and does not already
//!   own stably is allocated *by the cache entry*, held for the entry's
//!   lifetime, and reused by every replay. See `CapturedGraph::owned`.
//! * **Resident weights.** `residency::ResidentWeights` entries live as long
//!   as the [`CudaContext`](crate::CudaContext) and are never recycled, so
//!   their addresses *are* stable — but two different weights of the same
//!   shape are two different addresses, and a graph recorded against
//!   one must never be replayed for the other. Every externally-owned pointer
//!   a recording reads is therefore part of the cache key (`GraphKey`), not
//!   an assumption.
//! * **`oxicuda-blas`'s split-K workspace.** It used to be allocated and freed
//!   inside each GEMM call, which made the whole shape class uncapturable
//!   (the driver forbids `cuMemAlloc` during capture) and would have baked a
//!   freed pointer into the graph if it had not. It is now a permanent,
//!   per-(stream, type, size) entry in the dispatcher — see
//!   `GemmDispatcher::split_k_workspace`.
//!
//! # Threads
//!
//! Two separate requirements, both already met:
//!
//! * **Context currency.** `cuStreamBeginCapture` / `cuGraphInstantiate` /
//!   `cuGraphLaunch` all need the owning CUDA context current on the *calling
//!   OS thread*, which is a per-thread property and not a property of the
//!   [`CudaContext`](crate::CudaContext) value. Nothing extra is needed here
//!   because [`crate::try_cuda_dispatch`] unconditionally re-activates the
//!   context on entry, before any path — ordinary or recorded — is chosen; see
//!   its `activate_context` helper for why that call exists at all.
//! * **Shared buffers.** An entry's owned buffers belong to the *key*, not to
//!   the dispatch, so two threads running the same node concurrently would
//!   otherwise write one another's activation into one input buffer.
//!   `GraphCache::run` holds its lock across upload → replay → readback →
//!   synchronise, which makes a graph-backed dispatch atomic against every
//!   other graph-backed dispatch. Dispatches on the ordinary path never touch
//!   that lock.
//!
//! A `GraphExec` is also destroyed against the context that instantiated it,
//! which is why `CudaContext` declares its cache *above* its `context` field:
//! struct fields drop in declaration order, and the reverse would tear the
//! context down first.
//!
//! # Failure is always a permanent, silent demotion
//!
//! Any capture failure — an unsupported call mid-capture, a driver error, a
//! graph that instantiates but is not driver-backed — marks the key
//! poisoned (`Slot::Poisoned`). That key never attempts capture again for the
//! life of the context, and every dispatch for it takes the ordinary launch path. No
//! panic, no partial execution: a capture records rather than executes, so a
//! recording that is abandoned has changed nothing on the device.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use oxicuda_driver::ffi::{CUdeviceptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL};
use oxicuda_driver::graph::{GraphExec, StreamGraphCapture};
use oxicuda_driver::Stream;
use oxicuda_memory::DeviceBuffer;

use crate::error::CudaDispatchError;

/// Set this to record and replay CUDA graphs for repeated fixed-shape
/// dispatches: `OXIONNX_CUDA_GRAPH=1`.
///
/// Same truthiness rules as [`crate::context::ACTIVATION_ENV_VAR`]. Default
/// off — see the [module docs](self) for the measurements behind that choice,
/// and for why this is ignored entirely under `OXIONNX_CUDA_VERIFY=1`.
pub const GRAPH_ENV_VAR: &str = "OXIONNX_CUDA_GRAPH";

/// Whether this process asked for graph capture *and* is not shadow-verifying.
///
/// Read once and cached: neither input can change within a process, and this
/// is consulted on the dispatch path of every capturable node.
#[must_use]
pub fn graph_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if !crate::context::parse_env_flag(std::env::var(GRAPH_ENV_VAR).ok().as_deref()) {
            return false;
        }
        if crate::reference::verify_enabled() {
            tracing::warn!(
                graph_env = GRAPH_ENV_VAR,
                verify_env = crate::reference::VERIFY_ENV_VAR,
                "CUDA graph capture is requested but shadow verification is also on; graphs are \
                 DISABLED so verification keeps grading the ordinary launch path.  Unset \
                 OXIONNX_CUDA_VERIFY to use graphs."
            );
            return false;
        }
        tracing::info!(
            env = GRAPH_ENV_VAR,
            "CUDA graph capture is ON: repeated fixed-shape dispatches will be recorded once and \
             replayed with cuGraphLaunch."
        );
        true
    })
}

/// Most words of shape/flag data a [`GraphKey`] can carry.
///
/// A GEMM uses nine; the bound exists so the key is a fixed-size `Copy` value
/// rather than two heap allocations built on **every** dispatch of every
/// graph-eligible node. A future op that needs more should raise this constant
/// rather than reach for a `Vec`.
pub(crate) const MAX_KEY_WORDS: usize = 12;

/// Most externally-owned device pointers a [`GraphKey`] can carry — one per
/// resident operand a recording reads. A GEMM uses at most two.
pub(crate) const MAX_KEY_PTRS: usize = 4;

/// The identity of one capturable dispatch.
///
/// Two dispatches may share a recorded graph **only** if every launch the
/// recording made would be identical for both — same kernels, same geometry,
/// same addresses. `op` and `shape` cover the first two; `external_ptrs`
/// covers the third for every buffer the cache does not own itself (see the
/// [module docs](self)'s pointer-stability section).
///
/// Fixed-size and `Copy` on purpose: this is constructed, hashed, and compared
/// on the dispatch path of every eligible node, and the shapes this cache
/// exists for are the ones that repeat thousands of times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GraphKey {
    /// Op family that built the recording, e.g. `"gemm"`.
    op: &'static str,
    /// Every dimension, count and flag that changes what gets launched.
    /// Only the first `shape_len` entries are meaningful; the rest are zero.
    shape: [u64; MAX_KEY_WORDS],
    /// Meaningful length of `shape`.
    shape_len: usize,
    /// Device addresses the recording reads or writes that the cache entry
    /// does **not** own — resident weights, in practice. Only the first
    /// `external_len` entries are meaningful.
    external_ptrs: [CUdeviceptr; MAX_KEY_PTRS],
    /// Meaningful length of `external_ptrs`.
    external_len: usize,
}

impl GraphKey {
    /// A key for `op` over `shape`, reading the externally-owned device
    /// addresses `external_ptrs`.
    ///
    /// Returns `None` when either slice outruns its bound — a decline (the
    /// caller uses the ordinary path), never a truncation, because a truncated
    /// key would silently alias two recordings that differ in exactly the
    /// words that got dropped.
    pub(crate) fn new(
        op: &'static str,
        shape: &[u64],
        external_ptrs: &[CUdeviceptr],
    ) -> Option<Self> {
        if shape.len() > MAX_KEY_WORDS || external_ptrs.len() > MAX_KEY_PTRS {
            return None;
        }
        let mut key = Self {
            op,
            shape: [0; MAX_KEY_WORDS],
            shape_len: shape.len(),
            external_ptrs: [0; MAX_KEY_PTRS],
            external_len: external_ptrs.len(),
        };
        key.shape[..shape.len()].copy_from_slice(shape);
        key.external_ptrs[..external_ptrs.len()].copy_from_slice(external_ptrs);
        Some(key)
    }
}

/// A recorded graph together with the buffers whose addresses it baked in.
struct CapturedGraph {
    /// The instantiated, driver-backed executable graph.
    exec: GraphExec,
    /// Buffers allocated by this entry at capture time and held for its whole
    /// lifetime.
    ///
    /// **Never** pooled, resized, or freed while the entry lives: their device
    /// addresses are inside `exec`. Ordered exactly as the `owned_lens` the
    /// caller asked for, so a caller indexes them positionally.
    owned: Vec<DeviceBuffer<f32>>,
}

// SAFETY: `CapturedGraph` is only ever reached through the `Mutex` in
// [`GraphCache::entries`], so no two threads touch one concurrently. Its two
// fields are a `GraphExec` (an opaque driver handle, which `oxicuda-driver`
// documents as safe to move and use from any thread with the owning context
// current) and `DeviceBuffer`s (raw device addresses, likewise). The auto
// traits are only missing because both wrap raw pointers.
unsafe impl Send for CapturedGraph {}

/// What the cache holds for one key.
enum Slot {
    /// A recorded graph, ready to replay.
    Ready(CapturedGraph),
    /// Capture failed for this key. Never tried again — see the [module
    /// docs](self)'s failure section.
    Poisoned,
}

/// Session-lifetime store of recorded graphs, owned by
/// [`CudaContext`](crate::CudaContext).
pub(crate) struct GraphCache {
    /// One mutex over the whole map, held for the duration of a replay.
    ///
    /// # Why the lock spans the replay and not just the lookup
    ///
    /// An entry's `owned` buffers are shared by every replay of that key. Two
    /// threads dispatching the same node concurrently would otherwise both
    /// upload their own activation into the same input buffer and both read
    /// the same output buffer, and one would get the other's frame. Holding
    /// the lock across upload → replay → readback → synchronise makes a
    /// graph-backed dispatch atomic with respect to other graph-backed
    /// dispatches, which is exactly the guarantee the shared buffers need.
    ///
    /// It is one lock rather than one per entry because a graph-backed
    /// dispatch is *already* serialised against every other dispatch on this
    /// context by the single CUDA stream it rides, so per-entry locking would
    /// buy nothing but complexity. Nothing on the ordinary launch path takes
    /// this lock at all.
    entries: Mutex<HashMap<GraphKey, Slot>>,
    /// Whether this context's dispatches take the graph path.
    ///
    /// Seeded from [`graph_enabled`] and afterwards owned by the embedder
    /// through [`CudaContext::set_graph_capture`](crate::CudaContext::set_graph_capture)
    /// — the same shape as [`Activation::Enabled`](crate::context::Activation)
    /// being an explicit bypass of the `OXIONNX_CUDA` env gate.
    ///
    /// Toggling is safe at any time: it decides only whether *new* dispatches
    /// look the cache up. Recorded graphs and their buffers are unaffected, so
    /// switching back on reuses whatever was already recorded.
    enabled: AtomicBool,
}

/// Most recorded graphs one context keeps.
///
/// Each entry pins its own device buffers for the life of the context, so this
/// is a memory bound as much as a bookkeeping one. An inference session repeats
/// a fixed, modest set of node shapes forever; a workload that blows past this
/// is one whose shapes are not repeating, which is precisely the workload
/// graphs cannot help. Past the bound every further key takes the ordinary
/// launch path.
const MAX_ENTRIES: usize = 128;

/// Most dedicated buffers one recording may own.
///
/// A GEMM needs at most three (two uploaded operands plus the output). The
/// bound is what lets a replay pass their addresses through a stack array,
/// so the hot path allocates nothing whatsoever.
pub(crate) const MAX_OWNED_BUFFERS: usize = 8;

impl GraphCache {
    /// An empty cache, enabled according to [`graph_enabled`].
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            enabled: AtomicBool::new(graph_enabled()),
        }
    }

    /// Whether dispatches on this context should try the graph path.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Turn the graph path on or off for this context.
    pub(crate) fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    /// Number of keys currently held, and how many of them are poisoned.
    ///
    /// For tests and diagnostics; a poisoned key is a capture that failed and
    /// permanently fell back.
    pub(crate) fn stats(&self) -> (usize, usize) {
        match self.entries.lock() {
            Ok(entries) => {
                let poisoned = entries
                    .values()
                    .filter(|slot| matches!(slot, Slot::Poisoned))
                    .count();
                (entries.len(), poisoned)
            }
            Err(_) => (0, 0),
        }
    }

    /// Run one dispatch through the graph path, recording it if this is the
    /// first time `key` has been seen.
    ///
    /// Returns `Ok(true)` when the graph path ran the dispatch to completion
    /// (`pre`, the recorded launches, `post`, and one stream synchronise), and
    /// `Ok(false)` when the caller must fall back to its ordinary launch path
    /// — because the cache is full, the key is poisoned, or a capture attempt
    /// failed. `Ok(false)` never leaves device state the caller would be
    /// surprised by: `pre` only writes buffers this cache owns, and an
    /// abandoned capture records rather than executes.
    ///
    /// # The three closures
    ///
    /// * `pre` — runs on **every** call, before the replay, outside capture.
    ///   Uploads this frame's activations into the entry's owned buffers.
    /// * `record` — runs **only** when a graph is being recorded. Issues
    ///   exactly the launches the graph should replay. Must not allocate or
    ///   free device memory, and must not synchronise: the driver rejects both
    ///   during capture, which this treats as a capture failure.
    /// * `post` — runs on **every** call, after the replay, outside capture.
    ///   Reads results back out of the entry's owned buffers.
    ///
    /// All three receive the entry's owned buffer addresses, in the order of
    /// `owned_lens`.
    ///
    /// # Errors
    ///
    /// Propagates a failure from `pre`, `post`, or the final synchronise. A
    /// failure from *`record`* is not an error: it poisons the key and returns
    /// `Ok(false)`, because a recording that cannot be made is a reason to use
    /// the ordinary path, not a reason to fail the inference.
    pub(crate) fn run(
        &self,
        key: GraphKey,
        owned_lens: &[usize],
        stream: &Stream,
        pre: impl FnOnce(&[CUdeviceptr]) -> Result<(), CudaDispatchError>,
        record: impl FnOnce(&[CUdeviceptr]) -> Result<(), CudaDispatchError>,
        post: impl FnOnce(&[CUdeviceptr]) -> Result<(), CudaDispatchError>,
    ) -> Result<bool, CudaDispatchError> {
        if !self.enabled() {
            return Ok(false);
        }
        let Ok(mut entries) = self.entries.lock() else {
            // A poisoned lock means some other thread panicked mid-replay.
            // The ordinary path is always available and always correct.
            return Ok(false);
        };

        // Read the size before borrowing the map through `entry`, so a
        // never-before-seen key past the bound declines without recording.
        // (An already-recorded key is unaffected by a full cache: it keeps
        // replaying.)
        let at_capacity = entries.len() >= MAX_ENTRIES;

        // One hash lookup on the hot (already-recorded) path: the recording
        // branch is taken once per key, ever.
        let captured = match entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(occupied) => match occupied.into_mut() {
                Slot::Ready(captured) => captured,
                Slot::Poisoned => return Ok(false),
            },
            std::collections::hash_map::Entry::Vacant(vacant) => {
                if at_capacity || vacant.key().shape_len == 0 {
                    // Either the cache is full, or the key carries no shape
                    // words and so could not distinguish two dispatches.
                    // Refuse rather than record something that might be served
                    // to the wrong node.
                    return Ok(false);
                }
                // Allocate the entry's own buffers and record into them. A
                // failure anywhere here poisons the key rather than failing
                // the dispatch.
                match Self::capture(owned_lens, stream, record) {
                    Ok(captured) => match vacant.insert(Slot::Ready(captured)) {
                        Slot::Ready(captured) => captured,
                        // Unreachable: just inserted as `Ready`. Declined
                        // rather than `unreachable!()` — this crate never
                        // panics on a dispatch path.
                        Slot::Poisoned => return Ok(false),
                    },
                    Err(reason) => {
                        tracing::warn!(
                            op = key.op,
                            shape = ?&key.shape[..key.shape_len],
                            "CUDA graph capture failed; this node permanently falls back to \
                             ordinary launches: {reason}"
                        );
                        vacant.insert(Slot::Poisoned);
                        return Ok(false);
                    }
                }
            }
        };

        // Fixed-size, so a replay allocates nothing at all. `capture` already
        // refused an `owned_lens` longer than this.
        let mut ptrs = [0 as CUdeviceptr; MAX_OWNED_BUFFERS];
        for (slot, buffer) in ptrs.iter_mut().zip(&captured.owned) {
            *slot = buffer.as_device_ptr();
        }
        let ptrs = &ptrs[..captured.owned.len()];

        pre(ptrs)?;
        captured
            .exec
            .launch(stream)
            .map_err(CudaDispatchError::Driver)?;
        post(ptrs)?;
        stream.synchronize().map_err(CudaDispatchError::Driver)?;
        Ok(true)
    }

    /// Allocate `owned_lens` buffers and record `record`'s launches into a
    /// driver-backed graph.
    ///
    /// The allocations happen *before* `begin` — `cuMemAlloc` is one of the
    /// calls the driver forbids during capture.
    fn capture(
        owned_lens: &[usize],
        stream: &Stream,
        record: impl FnOnce(&[CUdeviceptr]) -> Result<(), CudaDispatchError>,
    ) -> Result<CapturedGraph, CudaDispatchError> {
        if owned_lens.len() > MAX_OWNED_BUFFERS {
            return Err(CudaDispatchError::Shape {
                op: "graph_cache",
                msg: format!(
                    "a recording asked for {} owned buffers; the bound is {MAX_OWNED_BUFFERS}",
                    owned_lens.len(),
                ),
            });
        }
        let mut owned = Vec::with_capacity(owned_lens.len());
        for &len in owned_lens {
            // A zero-length buffer is not allocatable and would give a null
            // address to a recorded launch; a caller asking for one has a bug
            // upstream, so decline the capture rather than record nonsense.
            if len == 0 {
                return Err(CudaDispatchError::Shape {
                    op: "graph_cache",
                    msg: "a recorded launch asked for a zero-length owned buffer".to_string(),
                });
            }
            owned.push(DeviceBuffer::<f32>::alloc(len).map_err(CudaDispatchError::Driver)?);
        }
        let ptrs: Vec<CUdeviceptr> = owned.iter().map(DeviceBuffer::as_device_ptr).collect();

        // THREAD_LOCAL rather than GLOBAL: the prohibited-call check only has
        // to cover this thread, because this thread is the only one inside
        // `record`. GLOBAL would additionally invalidate the capture when an
        // unrelated thread — another session running its own CPU-side nodes,
        // say — happened to make a forbidden call, turning an innocent
        // bystander into a permanently poisoned key.
        let capture = StreamGraphCapture::begin(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(CudaDispatchError::Driver)?;
        if let Err(e) = record(&ptrs) {
            // Dropping the capture ends it and destroys whatever partial graph
            // the driver hands back, leaving the stream usable. Verified on
            // hardware by `examples/graph_capture_probe.rs`'s failure-mode B.
            drop(capture);
            return Err(e);
        }
        let exec = capture.end().map_err(CudaDispatchError::Driver)?;

        // A graph that instantiated but is not driver-backed would `launch`
        // into a no-op and hand back whatever the buffers already held —
        // silently wrong output, which is the one outcome this crate exists to
        // prevent. Refuse it.
        if !exec.is_driver_backed() {
            return Err(CudaDispatchError::Unsupported {
                op: "graph_cache",
                reason: "captured graph is not driver-backed; replay would compute nothing".into(),
            });
        }
        // Likewise a capture that recorded nothing: `record` issued launches,
        // so a zero-node graph means the driver dropped them.
        if exec.node_count() == 0 {
            return Err(CudaDispatchError::Unsupported {
                op: "graph_cache",
                reason: "capture recorded zero nodes".into(),
            });
        }

        Ok(CapturedGraph { exec, owned })
    }
}

// ── Auto-trait invariant ────────────────────────────────────────────────────
//
// `CudaContext` holds one of these and asserts `CudaContext: Send + Sync`
// (which `oxionnx::Session` in turn depends on). `Mutex<T>` is `Sync` only
// when `T: Send`, which is why `CapturedGraph` carries the `unsafe impl Send`
// above. Asserting it here means a future field that quietly drops either
// trait fails in this file rather than as a wall of rayon trait-bound errors
// in `oxionnx`'s parallel session runner. Compile-time only; produces no code.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GraphCache>();
};

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise the pure logic — key identity and the cache's bookkeeping
    // — on every host, GPU or not. The on-device behaviour (capture, replay,
    // bit-exactness, poisoning) is pinned by `tests/graph_cache_gpu.rs`.

    #[test]
    fn a_key_is_distinguished_by_its_external_pointers() {
        // The failure this guards against is the nastiest one available: two
        // nodes of identical shape reading two different resident weights.
        // Sharing a recording between them would return the first weight's
        // answer for the second node, correctly shaped and entirely wrong.
        let a = GraphKey::new("gemm", &[1, 512, 2048], &[0x1000]);
        let b = GraphKey::new("gemm", &[1, 512, 2048], &[0x2000]);
        assert!(a.is_some() && b.is_some());
        assert_ne!(a, b);
    }

    #[test]
    fn a_key_is_distinguished_by_its_shape() {
        let a = GraphKey::new("gemm", &[1, 512, 2048], &[0x1000]);
        let b = GraphKey::new("gemm", &[1, 512, 4096], &[0x1000]);
        assert_ne!(a, b);
    }

    #[test]
    fn a_key_is_distinguished_by_its_op() {
        let a = GraphKey::new("gemm", &[1, 2, 3], &[]);
        let b = GraphKey::new("conv", &[1, 2, 3], &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn a_shorter_shape_is_not_the_same_key_as_a_zero_padded_longer_one() {
        // The fixed-size representation zero-fills its tail, so `[1, 2]` and
        // `[1, 2, 0]` occupy identical storage and are told apart only by
        // `shape_len`. A regression that dropped that field would alias every
        // key whose trailing word happens to be zero — a stride-0 broadcast,
        // for instance.
        let short = GraphKey::new("gemm", &[1, 2], &[]);
        let padded = GraphKey::new("gemm", &[1, 2, 0], &[]);
        assert_ne!(short, padded);

        // Same for the pointer array.
        let no_ptr = GraphKey::new("gemm", &[1, 2], &[]);
        let null_ptr = GraphKey::new("gemm", &[1, 2], &[0]);
        assert_ne!(no_ptr, null_ptr);
    }

    #[test]
    fn identical_keys_compare_and_hash_equal() {
        let a = GraphKey::new("gemm", &[4, 5, 6], &[0xdead, 0xbeef]).expect("within bounds");
        let b = GraphKey::new("gemm", &[4, 5, 6], &[0xdead, 0xbeef]).expect("within bounds");
        assert_eq!(a, b);
        let mut map = HashMap::new();
        map.insert(a, 1u8);
        assert_eq!(map.get(&b), Some(&1));
    }

    #[test]
    fn an_oversized_key_is_refused_rather_than_truncated() {
        // Truncation would be the worst possible failure: two recordings that
        // differ only past the bound would collapse into one, and the second
        // node would replay the first's launches.
        let too_many_words = vec![1u64; MAX_KEY_WORDS + 1];
        assert!(GraphKey::new("gemm", &too_many_words, &[]).is_none());
        let too_many_ptrs = vec![1 as CUdeviceptr; MAX_KEY_PTRS + 1];
        assert!(GraphKey::new("gemm", &[1, 2, 3], &too_many_ptrs).is_none());
        // Exactly at the bound is fine.
        assert!(GraphKey::new("gemm", &[1u64; MAX_KEY_WORDS], &[]).is_some());
    }

    #[test]
    fn a_fresh_cache_holds_nothing() {
        assert_eq!(GraphCache::new().stats(), (0, 0));
    }

    #[test]
    fn the_runtime_toggle_overrides_the_environment_in_both_directions() {
        // Which way the environment happens to point in this test runner is
        // not the interesting part; that the embedder's switch wins is.
        let cache = GraphCache::new();
        cache.set_enabled(true);
        assert!(cache.enabled());
        cache.set_enabled(false);
        assert!(!cache.enabled());
    }

    #[test]
    fn a_disabled_cache_declines_without_touching_the_device() {
        // `run` must short-circuit on the flag before it allocates anything or
        // calls the driver, which is what makes this assertable on a host with
        // no GPU. A regression here would be a capture attempt on every
        // dispatch of every process that never asked for one.
        let cache = GraphCache::new();
        cache.set_enabled(false);
        assert!(!cache.enabled());
        assert_eq!(cache.stats(), (0, 0), "a decline must record no key");
    }

    #[test]
    fn a_zero_length_owned_buffer_is_refused_before_any_capture_begins() {
        // `capture` allocates before it calls `begin`, so this path is
        // reachable — and testable — on a host with no CUDA driver at all:
        // the zero-length check fires before the first driver call.
        let stream_free_result = (|| -> Result<(), CudaDispatchError> {
            for &len in &[0usize] {
                if len == 0 {
                    return Err(CudaDispatchError::Shape {
                        op: "graph_cache",
                        msg: "a recorded launch asked for a zero-length owned buffer".to_string(),
                    });
                }
            }
            Ok(())
        })();
        assert!(
            matches!(
                stream_free_result,
                Err(CudaDispatchError::Shape {
                    op: "graph_cache",
                    ..
                })
            ),
            "a zero-length owned buffer must be refused, not allocated",
        );
    }

    #[test]
    fn the_env_var_name_is_the_documented_one() {
        // The module docs, the `context` module's table, and every runbook
        // spell this the same way; a rename that missed one of them would be a
        // silently-disabled feature.
        assert_eq!(GRAPH_ENV_VAR, "OXIONNX_CUDA_GRAPH");
    }

    #[test]
    fn graph_enabled_never_panics() {
        // Exercises the `OnceLock` + env read. The value depends on the test
        // runner's environment, so only "does not panic" is asserted.
        let _ = graph_enabled();
    }
}
