//! Residency-aware placement and run statistics for
//! [`crate::Session::run_gpu_async`].
//!
//! # What is here, and what is deliberately not
//!
//! This module holds the two things that turned out to matter, both keyed by
//! concepts the session owns and `oxionnx-gpu` deliberately knows nothing
//! about (it takes slices, never tensor names or `OpKind`s):
//!
//! * [`gpu_min_transfer_elements`] — the **two-tier size gate**. Its two
//!   tiers, [`ResidencyTier::Transferred`] and [`ResidencyTier::Resident`],
//!   are two different cost models, and the gap between them is the measured,
//!   quantitative case for building tensor residency at all.
//! * [`GpuRunStats`] — per-run counters (nodes on GPU vs CPU, read-backs and
//!   their bytes, CPU fallback time by op type). These are what the wave's
//!   measurement table is built from; nothing else in the engine records
//!   which provider actually ran a node.
//!
//! The device-resident *activation* map is **not** here — it is
//! `super::gpu_activations`, which owns the name → buffer mapping, the last-use
//! schedule and the "may this value stay on the device" rule. This module keeps
//! only the placement arithmetic and the counters, which is the split that lets
//! the size gate be unit-tested without a device.
//!
//! The measurements that sized every constant below, on an M3 with
//! `examples/r3a_cost_breakdown.rs` and
//! `examples/r3a_inswapper_residency.rs`:
//!
//! * A full InSwapper forward through `run_gpu_async` spends ~75% of its GPU
//!   wall clock inside the `Conv` kernel itself, which C3 already established
//!   runs at 554–692 GFLOP/s — near this codebase's ceiling.
//! * The traffic residency would remove is ~553 MB of weight re-upload plus
//!   ~488 MiB of activation read-back per frame. At the measured ~5 GiB/s
//!   effective round-trip bandwidth that is roughly 95–200 ms of an ~880 ms
//!   frame: real, worth doing, and **not** the multiple the wave assumed.
//! * Meanwhile the same measurement found something with a much better
//!   payoff-to-risk ratio, which *is* implemented here: seven of the ten
//!   GPU-dispatched op types were running slower than their CPU kernels, some
//!   by 19–37x. Fixing that is a gate, not a rewrite.
//!
//! **The weight half of that traffic is now gone.** A graph's initializers are
//! uploaded once per session and bound from a device-side cache thereafter —
//! `oxionnx_gpu::context::resident`, owned by the `GpuContext` because that is
//! what makes the buffers die on the device's own thread (see
//! `super::gpu_owner`). This module contributes the *identity*, which is the
//! part `oxionnx-gpu` must not know: `initializer_key` in `super::gpu_dispatch`
//! (private, so named rather than linked) decides which of a node's inputs are
//! graph initializers,
//! and [`GpuRunStats::weight_cache_hits`] and friends report what the cache
//! did.
//!
//! **The activation half is now gone too.** A value produced by one GPU node and
//! consumed by the next stays in its device buffer for the whole run
//! (`super::gpu_activations`), so the read-back-and-re-upload between them
//! happens only where a CPU-placed consumer or a graph output actually needs
//! host bytes. That is what makes [`ResidencyTier::Resident`] reachable, and
//! with it the memory-bound gate that [`MEMORY_BOUND_TRANSFER_FLOOR`] holds
//! shut for every transferring node.
//!
//! The full reasoning, including why the elementwise ops lose at *every* size
//! while transferring, is on [`gpu_min_transfer_elements`].
//!
//! # The decline-to-CPU contract
//!
//! Unchanged in kind: a node this module declines is never an error. It returns
//! `Ok(None)` from `try_gpu_dispatch_async` and runs on the CPU operator from
//! ordinary host tensors, exactly as an unsupported node always has.
//!
//! What has changed is the *cost* of declining. A declined node whose operands
//! are on the device has to materialize them first — once per tensor per run,
//! memoized into the run state — so the decline side of the resident tier pays
//! a read-back the transferring tier never did. That asymmetry is the whole
//! reason [`RESIDENT_DISPATCH_FLOOR`] is a fill-one-workgroup floor rather than
//! a beat-the-CPU one; see it for the arithmetic.

use crate::graph::OpKind;
use std::collections::HashMap;

/// Minimum GEMM FLOP count (`2 * m * k * n`) for a `Gemm` node to be offered
/// to `gpu_gemm_nt`.
///
/// Mirrors `oxionnx_gpu::compute`'s private `GPU_THRESHOLD`, which
/// `gpu_matmul`/`gpu_conv2d` already apply *inside* their kernels via
/// `gemm_flops`. `gpu_gemm_nt` has no such gate — `kernel_support`'s
/// convention is that the placement heuristic belongs at the session call
/// site — so this is that gate, at that site.
///
/// `u64`, and computed with `checked_mul`, for the same reason `gemm_flops`
/// is: `2 * 2048^3` overflows a 32-bit `usize`, and a wrapped product would
/// compare *below* the threshold and silently route a huge GEMM to the CPU.
///
/// Measured effect: InSwapper's 12 AdaIN heads are `[1,512] x [2048,512]^T` =
/// 2.1 MFLOP each, so all 12 now decline (they were 3.07x slower on the GPU).
/// ArcFace's head is `[1,25088] x [512,25088]^T` = 25.7 MFLOP and still
/// dispatches.
pub const GEMM_GPU_MIN_FLOPS: u64 = 10_000_000;

/// `2 * m * k * n`, or `None` on overflow. See [`GEMM_GPU_MIN_FLOPS`].
#[must_use]
pub fn gemm_flops(m: usize, k: usize, n: usize) -> Option<u64> {
    (m as u64)
        .checked_mul(k as u64)?
        .checked_mul(n as u64)?
        .checked_mul(2)
}

/// Operand-element floor for a memory-bound op under
/// [`ResidencyTier::Transferred`]: `Some(usize::MAX)`, i.e. never dispatch one
/// while its operands have to cross the bus.
///
/// # Native
///
/// Measured (see [`gpu_min_transfer_elements`]): on an M3 the CPU kernels for
/// these ops run rayon-parallel at main-memory bandwidth (~92 GB/s for
/// `Relu`), while a GPU round trip moves the same bytes *twice* at a measured
/// ~5 GiB/s effective and adds ~1.4 ms of fixed dispatch cost. Both sides
/// scale linearly in `n`, so no size makes the GPU win — the floor is a
/// statement of that fact, not a tunable.
///
/// # wasm32
///
/// This arm used to be `None` — no session floor, each kernel's own threshold
/// deciding — on the argument that the native numbers above are measured
/// against a rayon-parallel CPU and must not be extrapolated to a
/// single-threaded browser. That argument still stands, and this constant is
/// **not** a claim that the browser crossover has now been measured: it has
/// not.
///
/// What changed is that the browser's binding constraint turned out not to be
/// throughput. Under `Transferred`, every one of these nodes allocates its
/// operands, its output and a read-back buffer per dispatch, and InSwapper has
/// 57 of them per frame (19 `Add`, 12 `Mul`, 12 `OxiInstanceNorm`, 7
/// `LeakyRelu`, 6 `Relu`, 1 `Div` — the table in
/// [`gpu_min_transfer_elements`]). That is the largest single contributor to
/// the allocation rate on a `GPUDevice` whose memory this crate must keep
/// bounded, and it buys, per the same table, ops that lose to their CPU
/// kernels by 1.1x to 36x natively — with no reason to expect the ordering to
/// invert once the transfer is counted twice. So the floor is set for
/// boundedness, on ops that were never measured to be winners on any target.
///
/// The consequence is explicit: those 57 nodes run on the browser's
/// single-threaded CPU kernels. Reopening this gate needs the browser-side
/// crossover measured *and* the residency work that removes the round trip —
/// at which point the relevant constant is the `Resident` arm below, not this
/// one.
pub const MEMORY_BOUND_TRANSFER_FLOOR: Option<usize> = Some(usize::MAX);

/// Whether a node's operands have to cross the bus for this dispatch.
///
/// This is the "two-tier scheme" the wave was asked for, and the two tiers are
/// genuinely different cost models rather than two constants:
///
/// * [`Self::Transferred`] — every operand is uploaded and every result is
///   read back. The dispatch pays `(inputs + outputs) x 4` bytes of traffic
///   plus a fixed per-dispatch cost, on top of the kernel's own runtime.
/// * [`Self::Resident`] — operands are already in device buffers and the
///   result stays in one. Only the fixed dispatch cost remains.
///
/// The session computes a node's tier from its operands
/// ([`node_residency_tier`]) rather than asserting one. In practice that still
/// answers [`Self::Transferred`] for every node a real graph dispatches, and
/// deliberately so — see [`node_residency_tier`] for why weight residency does
/// **not** promote a node to [`Self::Resident`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyTier {
    /// At least one operand uploads for this dispatch, and the result reads
    /// back. The regime the engine is in for every node with an activation
    /// input — which is every node in a real graph.
    Transferred,
    /// Operands and results live in device buffers across the node boundary.
    Resident,
}

/// Operand-element floor for a memory-bound op under
/// [`ResidencyTier::Resident`]: 256 elements, one full workgroup of the
/// element-wise kernels.
///
/// # Reasoned, not measured — and it is not the same kind of number as
/// [`MEMORY_BOUND_TRANSFER_FLOOR`]
///
/// That constant answers "is a round trip cheaper than the CPU kernel?", and
/// the answer is no at every size, which is why it is `usize::MAX`. This one
/// cannot be that question, because under this tier the alternative to
/// dispatching is not "run the CPU kernel on bytes the host already has". The
/// operands are on the device. Declining means:
///
/// 1. reading every resident operand back (`4n` bytes at the measured ~5 GiB/s
///    effective round-trip bandwidth), then
/// 2. running the CPU kernel, then
/// 3. letting the next GPU node upload the result again (`4·out` bytes).
///
/// Dispatching costs a bind group, an encoder, a pass and a submit, and moves
/// nothing. So the decline side is `>= 8n` bytes of traffic against a fixed
/// cost measured in tens of microseconds — the crossover is not somewhere in
/// the middle of the size range, it is at the bottom of it. The floor's job is
/// therefore not to protect the CPU kernel's wins; it is to refuse a dispatch
/// too small to be a dispatch.
///
/// 256 is that: the element-wise kernels run `@workgroup_size(256)`, so below
/// it not one workgroup is filled, and the tensor is under 1 KiB — cheap to
/// read back whichever way the decision goes. The predecessor value here was an
/// explicitly uncalibrated `4096`, chosen by analogy with a *transferring*
/// threshold; it is replaced rather than inherited because the analogy is to
/// the wrong cost model.
///
/// It is still a reasoned number. What would replace it with a measured one is
/// a browser-side sweep of resident-input elementwise dispatch against
/// read-back-then-CPU at sizes from one workgroup up.
pub const RESIDENT_DISPATCH_FLOOR: usize = 256;

/// The element count a tier's floor is measured against.
///
/// The two tiers count different things, and using one number for both is the
/// bug this function exists to make impossible:
///
/// * [`ResidencyTier::Transferred`] — `transferred`, the elements that actually
///   cross the bus for this dispatch. That is what
///   [`MEMORY_BOUND_TRANSFER_FLOOR`] is a statement about.
/// * [`ResidencyTier::Resident`] — `dispatch`, the largest operand's element
///   count. Under this tier `transferred` is **zero by construction** (every
///   operand is already on the device), so comparing it against any positive
///   floor would decline every node the tier exists to admit — the exact
///   inverse of the intent. What the resident floor asks is whether the
///   dispatch is wide enough to be worth encoding at all, and that is a
///   question about the operands' size, not their provenance.
#[must_use]
pub fn tier_gate_elements(tier: ResidencyTier, transferred: usize, dispatch: usize) -> usize {
    match tier {
        ResidencyTier::Transferred => transferred,
        ResidencyTier::Resident => dispatch,
    }
}

/// The cost model a node's dispatch actually pays, from a census of its
/// operands.
///
/// [`ResidencyTier::Resident`] requires **every** operand to be on the device
/// already. A convolution whose weights are resident but whose input activation
/// still uploads — which is every convolution in InSwapper, SCRFD and ArcFace
/// — pays the transfer cost model, because the transfer it pays is the one that
/// scales with the tensor the floor is measured against.
///
/// That strictness is the point, not a limitation. Weight residency alone must
/// not reopen [`MEMORY_BOUND_TRANSFER_FLOOR`] for InSwapper's 57 elementwise
/// nodes: the measured table in [`gpu_min_transfer_elements`] says they lose to
/// their CPU kernels by 1.1x to 36x *while transferring*, and a `Conv` whose
/// weight is cached but whose activation still uploads is transferring.
///
/// # How a node actually reaches `Resident`
///
/// Two ways, both from `super::gpu_activations`:
///
/// * every operand is a value an earlier GPU node left in a device buffer, or
/// * the large ones are, and `sequential_async`'s operand promotion uploaded
///   the small remainder so the claim is *true* rather than relaxed. That
///   upload is real traffic and is counted as
///   [`GpuRunStats::activation_upload_bytes`].
///
/// A node with no resolvable operands is `Transferred` too: "all zero of its
/// operands are resident" is not a residency claim.
#[must_use]
pub fn node_residency_tier(operands: usize, resident_operands: usize) -> ResidencyTier {
    if operands > 0 && resident_operands == operands {
        ResidencyTier::Resident
    } else {
        ResidencyTier::Transferred
    }
}

/// Minimum operand element count for a GPU dispatch of `op` to be worth it
/// under `tier`, or `None` when this layer imposes no floor and the kernel's
/// own gate decides.
///
/// # These numbers are measured, not estimated
///
/// `examples/r3a_inswapper_residency.rs` runs all 154 InSwapper nodes twice —
/// once with `OpPlacement::CpuOnly`, once with `Auto` — and reports per-op
/// totals for both. On an M3, with every op wired and no size floor beyond
/// each kernel's own, the result was:
///
/// | op | nodes | CPU ms | GPU ms | GPU/CPU |
/// |---|---|---|---|---|
/// | Conv | 20 | 1849.31 | 808.41 | **0.44** |
/// | Pad | 14 | 145.30 | 38.24 | **0.26** |
/// | Resize | 2 | 62.87 | 17.45 | **0.28** |
/// | Mul | 12 | 28.32 | 32.11 | 1.13 |
/// | Add | 19 | 36.83 | 60.83 | 1.65 |
/// | Gemm | 12 | 7.12 | 21.83 | 3.07 |
/// | OxiInstanceNorm | 12 | 16.16 | 53.37 | 3.30 |
/// | LeakyRelu | 7 | 2.76 | 51.88 | 18.82 |
/// | Relu | 6 | 0.52 | 19.07 | 36.60 |
/// | Div | 1 | 0.05 | 1.49 | 28.61 |
///
/// Three ops win; seven lose, several by more than an order of magnitude.
/// That is not a threshold that wants nudging, and it is not a kernel defect
/// — it is the structural consequence of the `Transferred` tier:
///
/// * A memory-bound elementwise op does `O(n)` arithmetic over `O(n)` bytes.
///   The CPU reads and writes those bytes once at main-memory bandwidth
///   (measured here: ~92 GB/s for `Relu`, rayon-parallel). The GPU must move
///   the *same* bytes twice across the bus, and this crate's measured
///   effective round-trip bandwidth is ~5 GiB/s
///   (`examples/r3a_cost_breakdown.rs`) — more than an order of magnitude
///   worse — plus ~1.4 ms of fixed per-dispatch cost. Raising the size floor
///   does not help, because both sides scale linearly in `n`: the GPU loses at
///   every size. Hence `usize::MAX`, which is a claim ("never, while
///   transferring"), not a disabled feature.
/// * `Conv`/`Pad`/`Resize` win because their arithmetic per transferred byte
///   is high enough (`Conv`) or their CPU kernel expensive enough per element
///   (`Pad`'s reflect index math, `Resize`'s bilinear taps) to clear that gap.
///
/// Under [`ResidencyTier::Resident`] the two-times-`n` traffic disappears and
/// only the fixed dispatch cost is left, so the same elementwise kernels
/// become worthwhile at any size that fills the device — hence the small,
/// uniform [`RESIDENT_DISPATCH_FLOOR`] there. **That difference is the entire
/// quantitative case for tensor residency**, and it is why this function takes
/// a tier rather than being a table of constants. Note that the two arms are
/// measured against different quantities; see [`tier_gate_elements`].
///
/// # Why not fold this into each kernel
///
/// `kernel_support`'s module docs state the rule this follows: a CPU/GPU
/// placement heuristic belongs at the session call site, not inside a kernel,
/// so the kernels stay verifiable at 1-element shapes. This is that call site.
#[must_use]
pub fn gpu_min_transfer_elements(op: &OpKind, tier: ResidencyTier) -> Option<usize> {
    match tier {
        ResidencyTier::Resident => match op {
            // Reasoned, not measured, and the reasoning is on
            // `RESIDENT_DISPATCH_FLOOR` — which is *not* the uncalibrated 4096
            // this arm used to carry. That value was chosen by analogy with a
            // transferring threshold; this arm's cost model has no transfer in
            // it, so the analogy was to the wrong thing. Compared against the
            // node's dispatch width, never against its transferred elements
            // (which are zero here by construction) — see `tier_gate_elements`.
            OpKind::Add
            | OpKind::Mul
            | OpKind::Sub
            | OpKind::Div
            | OpKind::Relu
            | OpKind::LeakyRelu
            | OpKind::PRelu
            | OpKind::Sigmoid
            | OpKind::Tanh
            | OpKind::Exp
            | OpKind::Log
            | OpKind::Sqrt
            | OpKind::Abs
            | OpKind::Neg
            | OpKind::SiLU
            | OpKind::Gelu
            | OpKind::OxiInstanceNorm => Some(RESIDENT_DISPATCH_FLOOR),
            _ => None,
        },
        ResidencyTier::Transferred => match op {
            // Measured winners: no session floor, the kernel's own gate rules.
            OpKind::Conv | OpKind::Pad | OpKind::Resize | OpKind::MatMul => None,
            // Memory-bound ops. See `MEMORY_BOUND_TRANSFER_FLOOR` — the value
            // is platform-dependent, because the thing it is measured against
            // (the CPU kernel it must beat) differs by an order of magnitude
            // between a rayon-parallel native build and a browser.
            OpKind::Add
            | OpKind::Mul
            | OpKind::Sub
            | OpKind::Div
            | OpKind::Relu
            | OpKind::LeakyRelu
            | OpKind::PRelu
            | OpKind::Sigmoid
            | OpKind::Tanh
            | OpKind::Exp
            | OpKind::Log
            | OpKind::Sqrt
            | OpKind::Abs
            | OpKind::Neg
            | OpKind::SiLU
            | OpKind::Gelu
            | OpKind::OxiInstanceNorm => MEMORY_BOUND_TRANSFER_FLOOR,
            // `Gemm` is arithmetic-bound like `MatMul`/`Conv`, so it is *not*
            // a structural loser and gets no element floor here. Its 3.07x
            // regression has a different cause and a different fix: unlike
            // those two, `gpu_gemm_nt` carries no FLOP gate at all
            // (`kernel_support`'s "no minimum-size threshold" convention), so
            // every one of InSwapper's 2.1 MFLOP heads dispatched. That is
            // handled by [`GEMM_GPU_MIN_FLOPS`] at the dispatch site, which is
            // the same 10 MFLOP rule `MatMul` and `Conv` already apply inside
            // their kernels.
            OpKind::Gemm => None,
            _ => None,
        },
    }
}

/// Per-run counters describing where a graph actually executed.
///
/// # Why counters and not `NodeProfile`
///
/// `NodeProfile` (src/session/types.rs) already records a duration per node,
/// but it has no provider field and is a public struct shared with the
/// synchronous path — adding one would be a breaking change to a type this
/// wave has no mandate over. These counters answer the questions this wave
/// was asked to answer (how many nodes ran on the GPU, how many read-backs
/// happened, which op types fell back and for how long) without changing any
/// existing public type.
#[derive(Debug, Default, Clone)]
pub struct GpuRunStats {
    /// Nodes the wgpu backend accepted and computed.
    pub gpu_nodes: usize,
    /// Nodes that ran on a CPU operator — whether because they were never
    /// offered to the GPU or because it declined them.
    pub cpu_nodes: usize,
    /// Device→host transfers. One per tensor read back, whether at a graph
    /// output or because the next consumer declined the GPU.
    pub readbacks: usize,
    /// Bytes moved device→host.
    pub readback_bytes: u64,
    /// Wall-clock time attributed to CPU-executed nodes, by op type.
    pub cpu_time_by_op: HashMap<String, std::time::Duration>,
    /// Node count by op type, for nodes that ran on the CPU.
    pub cpu_count_by_op: HashMap<String, usize>,
    /// Wall-clock time attributed to GPU-executed nodes, by op type.
    ///
    /// **Two different quantities live in this map, and residency decides
    /// which.** A node whose output is read back is timed around the
    /// read-back, so its duration covers execution: the map completes only
    /// after the submission retires. A node whose output stays on the device
    /// returns as soon as the work is submitted — `finish_output_async`'s
    /// `Device` arm has no fence, deliberately, because ordering (not elapsed
    /// time) is what the next dispatch needs — so its duration is
    /// **encode-and-submit only**, and the execution it paid for lands in
    /// whichever later node happens to end in a read-back.
    ///
    /// So these per-op numbers are a placement and encode-cost breakdown, not
    /// a kernel-cost one, and turning residency on shifts time between entries
    /// without any kernel getting faster or slower. The only figure comparable
    /// across the residency toggle is whole-run wall clock that ends in a
    /// read-back — the graph outputs — which is what the measurement examples
    /// report.
    pub gpu_time_by_op: HashMap<String, std::time::Duration>,
    /// Initializer operands bound from a buffer that was already on the device.
    pub weight_cache_hits: u64,
    /// Initializer operands that had to be uploaded.
    pub weight_cache_misses: u64,
    /// Bytes those uploads moved host→device. **Zero on every run after the
    /// first** is the whole claim of the weight-residency cache; a run that
    /// dispatches the same graph again and still reports bytes here is a cache
    /// that is thrashing, not one that is working.
    pub weight_upload_bytes: u64,
    /// Node outputs that stayed in a device buffer instead of being read back.
    pub resident_outputs: usize,
    /// Operands a dispatch bound in place, having found them already on the
    /// device. One per (node, slot), so a value read by two nodes counts twice
    /// — which is right, because it is two uploads that did not happen.
    pub resident_operands: usize,
    /// Bytes that did **not** cross the bus because of activation residency:
    /// the uploads [`Self::resident_operands`] avoided plus the read-backs
    /// [`Self::resident_outputs`] avoided.
    ///
    /// Gross, not net — [`Self::activation_readback_bytes`] and
    /// [`Self::activation_upload_bytes`] are the traffic residency *added*, and
    /// the honest figure is the difference. A run where the difference is not
    /// comfortably positive has residency doing nothing but churn.
    pub activation_bytes_saved: u64,
    /// Device→host transfers of a resident activation that a declining consumer
    /// needed on the host after all. At most one per tensor per run: the result
    /// is memoized into the run state.
    pub activation_readbacks: usize,
    /// Bytes those read-backs moved.
    pub activation_readback_bytes: u64,
    /// Host operands uploaded ahead of a dispatch so that every one of its
    /// operands was on the device — see `gpu_dispatch`'s operand promotion.
    pub activation_uploads: usize,
    /// Bytes those uploads moved.
    pub activation_upload_bytes: u64,
    /// Largest device-byte total held by run-scoped activations at once.
    pub activation_peak_bytes: u64,
}

impl GpuRunStats {
    /// Record a node that the GPU accepted.
    pub fn record_gpu_node(&mut self, op: &str, elapsed: std::time::Duration) {
        self.gpu_nodes += 1;
        *self
            .gpu_time_by_op
            .entry(op.to_string())
            .or_insert(std::time::Duration::ZERO) += elapsed;
    }

    /// Fold one dispatch's weight-cache activity in.
    pub fn record_weight_cache(&mut self, hits: u64, misses: u64, uploaded_bytes: u64) {
        self.weight_cache_hits = self.weight_cache_hits.saturating_add(hits);
        self.weight_cache_misses = self.weight_cache_misses.saturating_add(misses);
        self.weight_upload_bytes = self.weight_upload_bytes.saturating_add(uploaded_bytes);
    }

    /// Record a node that ran on a CPU operator.
    pub fn record_cpu_node(&mut self, op: &str, elapsed: std::time::Duration) {
        self.cpu_nodes += 1;
        *self
            .cpu_time_by_op
            .entry(op.to_string())
            .or_insert(std::time::Duration::ZERO) += elapsed;
        *self.cpu_count_by_op.entry(op.to_string()).or_insert(0) += 1;
    }

    /// The `n` op types that cost the most CPU time, slowest first.
    ///
    /// Ties break on op name so the ordering is deterministic — a report that
    /// reshuffles between runs is not a report.
    #[must_use]
    pub fn top_cpu_fallbacks(&self, n: usize) -> Vec<(String, usize, std::time::Duration)> {
        let mut rows: Vec<(String, usize, std::time::Duration)> = self
            .cpu_time_by_op
            .iter()
            .map(|(op, d)| {
                (
                    op.clone(),
                    self.cpu_count_by_op.get(op).copied().unwrap_or(0),
                    *d,
                )
            })
            .collect();
        rows.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        rows.truncate(n);
        rows
    }
}

/// Where the statistics for the currently-running graph accumulate.
///
/// # Why a thread-local rather than a `Session` field
///
/// [`crate::Session::run_gpu_async`] takes `&self`, so any counter it updates
/// needs interior mutability. A `Mutex<GpuRunStats>` field on `Session` would
/// work, but it would also add a public-ish field to a struct this wave does
/// not own, and it would be wrong for the case where one `Session` is shared
/// across threads: two concurrent runs would interleave their counts into one
/// map with no way to tell them apart.
///
/// A thread-local is exactly right instead, because of a contract this module
/// does not get to choose: `run_sequential_async_inner` runs **one node at a
/// time on one thread**, and must (wgpu error scopes are a per-thread LIFO
/// stack — see `sequential_async`'s "Ordering contract"). So "the current
/// thread" and "the current run" are the same scope by construction, and each
/// concurrent run gets its own counters for free.
///
/// Statistics only — nothing about correctness depends on this. A caller that
/// never calls [`take_run_stats`] simply pays a few increments.
#[cfg(feature = "gpu")]
mod current {
    use super::GpuRunStats;
    use std::cell::RefCell;

    thread_local! {
        static STATS: RefCell<GpuRunStats> = RefCell::new(GpuRunStats::default());
    }

    /// Clear the calling thread's counters. Called at the top of each run.
    pub fn reset() {
        STATS.with(|s| *s.borrow_mut() = GpuRunStats::default());
    }

    /// Read and clear the calling thread's counters.
    pub fn take() -> GpuRunStats {
        STATS.with(|s| std::mem::take(&mut *s.borrow_mut()))
    }

    /// Read the calling thread's counters without clearing them.
    pub fn snapshot() -> GpuRunStats {
        STATS.with(|s| s.borrow().clone())
    }

    /// Mutate the calling thread's counters.
    pub fn with_mut<R>(f: impl FnOnce(&mut GpuRunStats) -> R) -> R {
        STATS.with(|s| f(&mut s.borrow_mut()))
    }
}

/// Clear the calling thread's GPU run statistics.
///
/// [`crate::Session::run_gpu_async`] calls this itself at the start of every
/// run, so a caller only needs it when measuring something else.
#[cfg(feature = "gpu")]
pub fn reset_run_stats() {
    current::reset();
}

/// Read and clear the statistics accumulated by the most recent
/// [`crate::Session::run_gpu_async`] on this thread.
#[cfg(feature = "gpu")]
#[must_use]
pub fn take_run_stats() -> GpuRunStats {
    current::take()
}

/// Read the statistics accumulated so far on this thread, leaving them in
/// place.
#[cfg(feature = "gpu")]
#[must_use]
pub fn run_stats() -> GpuRunStats {
    current::snapshot()
}

/// Record a node the GPU accepted. Internal to the async run loop.
#[cfg(feature = "gpu")]
pub(crate) fn note_gpu_node(op: &str, elapsed: std::time::Duration) {
    current::with_mut(|s| s.record_gpu_node(op, elapsed));
}

/// Record a node that ran on a CPU operator. Internal to the async run loop.
#[cfg(feature = "gpu")]
pub(crate) fn note_cpu_node(op: &str, elapsed: std::time::Duration) {
    current::with_mut(|s| s.record_cpu_node(op, elapsed));
}

/// Record a device→host transfer of `elements` f32 values.
#[cfg(feature = "gpu")]
pub(crate) fn note_readback(elements: usize) {
    current::with_mut(|s| {
        s.readbacks += 1;
        s.readback_bytes += (elements as u64).saturating_mul(4);
    });
}

/// Record a dispatch that bound `operands` of its inputs in place, totalling
/// `input_elements`, and left `output_elements` on the device (zero when the
/// result was read back).
#[cfg(feature = "gpu")]
pub(crate) fn note_activation_dispatch(
    operands: usize,
    input_elements: usize,
    output_elements: usize,
) {
    if operands == 0 && output_elements == 0 {
        return;
    }
    current::with_mut(|s| {
        s.resident_operands += operands;
        if output_elements > 0 {
            s.resident_outputs += 1;
        }
        let saved = (input_elements as u64).saturating_add(output_elements as u64);
        s.activation_bytes_saved = s.activation_bytes_saved.saturating_add(saved * 4);
    });
}

/// Record the one read-back a resident activation is allowed per run, taken
/// because a consumer declined and needed it on the host.
#[cfg(feature = "gpu")]
pub(crate) fn note_activation_readback(elements: usize) {
    current::with_mut(|s| {
        s.activation_readbacks += 1;
        s.activation_readback_bytes = s
            .activation_readback_bytes
            .saturating_add((elements as u64).saturating_mul(4));
    });
}

/// Record a host operand uploaded ahead of a dispatch so the node could run
/// with every operand in place.
#[cfg(feature = "gpu")]
pub(crate) fn note_activation_upload(elements: usize) {
    current::with_mut(|s| {
        s.activation_uploads += 1;
        s.activation_upload_bytes = s
            .activation_upload_bytes
            .saturating_add((elements as u64).saturating_mul(4));
    });
}

/// Record the peak device-byte total run-scoped activations reached.
#[cfg(feature = "gpu")]
pub(crate) fn note_activation_peak(bytes: u64) {
    current::with_mut(|s| s.activation_peak_bytes = s.activation_peak_bytes.max(bytes));
}

/// Record one dispatch's weight-cache activity, as the delta between two
/// [`oxionnx_gpu::ResidentCounters`] snapshots taken around it.
///
/// Counted per dispatch rather than per run because the context's own counters
/// are cumulative for the session: differencing them here is what makes
/// "this run uploaded no weights" a statement about *this* run.
#[cfg(feature = "gpu")]
pub(crate) fn note_weight_cache(delta: oxionnx_gpu::ResidentCounters) {
    if delta.is_idle() {
        return;
    }
    current::with_mut(|s| s.record_weight_cache(delta.hits, delta.misses, delta.uploaded_bytes));
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The whole point of the two tiers: an op that must transfer is gated
    /// out at every size, and the *same* op becomes cheap once resident.
    /// If these two ever return the same thing, the tier parameter has
    /// stopped meaning anything and the residency argument has evaporated.
    #[test]
    fn the_memory_bound_floor_is_unreachable_by_construction() {
        assert_eq!(MEMORY_BOUND_TRANSFER_FLOOR, Some(usize::MAX));
    }

    /// The bug this function exists to make impossible: a resident node's
    /// transferred element count is zero by construction, so measuring its
    /// floor against that number would decline every node the tier admits.
    #[test]
    fn the_resident_tier_is_gated_on_width_not_on_transferred_elements() {
        // A 73_728-element resident elementwise node, nothing transferring.
        let measured = tier_gate_elements(ResidencyTier::Resident, 0, 73_728);
        assert_eq!(measured, 73_728);
        let floor = gpu_min_transfer_elements(&OpKind::Relu, ResidencyTier::Resident)
            .expect("memory-bound ops carry a resident floor");
        assert!(
            measured >= floor,
            "a full-size resident elementwise node must dispatch"
        );
        // Compared against the transferred count instead, the same node would
        // decline — which is what a single shared quantity would have done.
        assert!(0 < floor);

        // The transferring arm is unchanged: it still measures the bytes that
        // actually cross the bus.
        assert_eq!(
            tier_gate_elements(ResidencyTier::Transferred, 4096, 73_728),
            4096
        );
    }

    /// The resident floor is one workgroup of the element-wise kernels, and it
    /// is deliberately *small*: the alternative to dispatching a resident node
    /// is a read-back, not a free CPU kernel.
    #[test]
    fn the_resident_floor_is_one_workgroup() {
        assert_eq!(RESIDENT_DISPATCH_FLOOR, 256);
        for op in [OpKind::Relu, OpKind::Add, OpKind::Mul, OpKind::LeakyRelu] {
            assert_eq!(
                gpu_min_transfer_elements(&op, ResidencyTier::Resident),
                Some(RESIDENT_DISPATCH_FLOOR),
            );
        }
        // Below one workgroup nothing dispatches; at one workgroup it does.
        assert!(tier_gate_elements(ResidencyTier::Resident, 0, 255) < RESIDENT_DISPATCH_FLOOR);
        assert!(tier_gate_elements(ResidencyTier::Resident, 0, 256) >= RESIDENT_DISPATCH_FLOOR);
    }

    /// The new counters accumulate the way the browser report reads them.
    #[test]
    fn activation_counters_accumulate_per_run() {
        reset_run_stats();
        note_activation_dispatch(2, 1000, 500);
        note_activation_dispatch(1, 400, 0);
        note_activation_readback(250);
        note_activation_upload(100);
        note_activation_peak(4096);
        note_activation_peak(1024);
        let stats = take_run_stats();
        assert_eq!(stats.resident_operands, 3);
        assert_eq!(
            stats.resident_outputs, 1,
            "only the dispatch that kept its output counts"
        );
        assert_eq!(stats.activation_bytes_saved, (1000 + 500 + 400) * 4);
        assert_eq!(stats.activation_readbacks, 1);
        assert_eq!(stats.activation_readback_bytes, 1000);
        assert_eq!(stats.activation_uploads, 1);
        assert_eq!(stats.activation_upload_bytes, 400);
        assert_eq!(stats.activation_peak_bytes, 4096, "the peak never falls");
    }

    /// [r3b] Weight residency must not promote a node whose activations still
    /// cross the bus. This is the guard on the one place the tier is consumed:
    /// promoting a `Relu` on the strength of a cached weight alone would swap
    /// its `usize::MAX` floor for the resident one and send 6 InSwapper nodes
    /// measured 36.6x slower back to the GPU *while still transferring*.
    #[test]
    fn a_node_is_resident_only_when_every_operand_is() {
        assert_eq!(
            node_residency_tier(2, 1),
            ResidencyTier::Transferred,
            "one resident weight plus one transferred activation still transfers",
        );
        assert_eq!(node_residency_tier(3, 0), ResidencyTier::Transferred);
        assert_eq!(node_residency_tier(2, 2), ResidencyTier::Resident);
        assert_eq!(
            node_residency_tier(0, 0),
            ResidencyTier::Transferred,
            "a node with no resolvable operands claims nothing",
        );

        // The consequence, stated where a future edit would trip over it: the
        // elementwise gate stays shut for the shape weight residency produces.
        assert_eq!(
            gpu_min_transfer_elements(&OpKind::Relu, node_residency_tier(1, 0)),
            MEMORY_BOUND_TRANSFER_FLOOR,
        );
        assert_eq!(
            gpu_min_transfer_elements(&OpKind::Add, node_residency_tier(2, 1)),
            MEMORY_BOUND_TRANSFER_FLOOR,
        );
    }

    /// The weight-cache counters accumulate across the dispatches of one run,
    /// which is what makes "this run uploaded nothing" a per-run claim.
    #[test]
    fn weight_cache_counters_accumulate_per_run() {
        let mut stats = GpuRunStats::default();
        stats.record_weight_cache(0, 2, 4096);
        stats.record_weight_cache(2, 0, 0);
        assert_eq!(stats.weight_cache_hits, 2);
        assert_eq!(stats.weight_cache_misses, 2);
        assert_eq!(stats.weight_upload_bytes, 4096);
    }

    #[test]
    fn the_two_tiers_disagree_for_memory_bound_ops() {
        for op in [
            OpKind::Add,
            OpKind::Mul,
            OpKind::Relu,
            OpKind::LeakyRelu,
            OpKind::OxiInstanceNorm,
        ] {
            let transferred = gpu_min_transfer_elements(&op, ResidencyTier::Transferred);
            let resident = gpu_min_transfer_elements(&op, ResidencyTier::Resident);
            assert_eq!(
                transferred, MEMORY_BOUND_TRANSFER_FLOOR,
                "{op:?} must use the platform's memory-bound floor"
            );
            assert!(
                resident.is_some_and(|r| r < usize::MAX),
                "{op:?} must become dispatchable once resident, got {resident:?}"
            );
        }
    }

    /// The three ops measured to beat their CPU kernels must not acquire a
    /// session-level floor — their own kernels already gate them, and adding
    /// a floor here would silently undo the wave's actual speedup.
    #[test]
    fn measured_winners_keep_no_session_floor() {
        for op in [OpKind::Conv, OpKind::Pad, OpKind::Resize, OpKind::MatMul] {
            assert_eq!(
                gpu_min_transfer_elements(&op, ResidencyTier::Transferred),
                None,
                "{op:?} beat the CPU in measurement and must stay ungated here"
            );
        }
    }

    /// `Gemm` is arithmetic-bound, so it is gated on FLOPs rather than
    /// element count — the same 10 MFLOP rule MatMul/Conv apply internally.
    #[test]
    fn gemm_flop_gate_declines_inswapper_and_admits_arcface() {
        assert_eq!(
            gpu_min_transfer_elements(&OpKind::Gemm, ResidencyTier::Transferred),
            None,
            "Gemm is gated on FLOPs, not on operand elements"
        );
        // InSwapper's AdaIN head: [1,512] x [2048,512]^T.
        let inswapper = gemm_flops(1, 512, 2048).expect("no overflow");
        assert_eq!(inswapper, 2_097_152);
        assert!(
            inswapper < GEMM_GPU_MIN_FLOPS,
            "InSwapper's heads were 3.07x slower on the GPU and must decline"
        );
        // ArcFace's embedding head: [1,25088] x [512,25088]^T.
        let arcface = gemm_flops(1, 25_088, 512).expect("no overflow");
        assert!(
            arcface >= GEMM_GPU_MIN_FLOPS,
            "ArcFace's head is 25.7 MFLOP and must still dispatch, got {arcface}"
        );
    }

    /// `2 * 2048^3` overflows a 32-bit `usize`. Computing it there and
    /// comparing would wrap *below* the threshold and route the largest GEMMs
    /// to the CPU — the exact inverse of what the gate is for.
    #[test]
    fn gemm_flops_is_u64_and_saturates_to_none_on_overflow() {
        assert_eq!(gemm_flops(2048, 2048, 2048), Some(17_179_869_184));
        assert!(gemm_flops(2048, 2048, 2048).is_some_and(|f| f > u64::from(u32::MAX)));
        assert_eq!(gemm_flops(usize::MAX, usize::MAX, 2), None);
        assert_eq!(gemm_flops(0, 1 << 20, 1 << 20), Some(0));
    }

    #[test]
    fn top_cpu_fallbacks_ranks_by_time_then_name() {
        let mut stats = GpuRunStats::default();
        stats.record_cpu_node("Slice", Duration::from_micros(10));
        stats.record_cpu_node("Slice", Duration::from_micros(10));
        stats.record_cpu_node("Unsqueeze", Duration::from_micros(5));
        stats.record_cpu_node("Div", Duration::from_micros(100));

        let top = stats.top_cpu_fallbacks(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "Div");
        assert_eq!(top[0].1, 1);
        assert_eq!(top[0].2, Duration::from_micros(100));
        assert_eq!(top[1].0, "Slice");
        // Both Slice nodes accumulate into one row.
        assert_eq!(top[1].1, 2);
        assert_eq!(top[1].2, Duration::from_micros(20));
    }

    #[test]
    fn gpu_and_cpu_nodes_are_counted_separately() {
        let mut stats = GpuRunStats::default();
        stats.record_gpu_node("Conv", Duration::from_millis(3));
        stats.record_gpu_node("Conv", Duration::from_millis(4));
        stats.record_cpu_node("Slice", Duration::from_micros(1));
        assert_eq!(stats.gpu_nodes, 2);
        assert_eq!(stats.cpu_nodes, 1);
        assert_eq!(
            stats.gpu_time_by_op.get("Conv").copied(),
            Some(Duration::from_millis(7))
        );
    }

    /// The thread-local really is per-thread: two threads must not see each
    /// other's counts. This is the property that makes it a correct
    /// substitute for a per-run field.
    #[test]
    fn statistics_do_not_leak_between_threads() {
        reset_run_stats();
        note_gpu_node("Conv", Duration::from_millis(1));
        let other = std::thread::spawn(|| {
            reset_run_stats();
            note_cpu_node("Slice", Duration::from_millis(1));
            take_run_stats()
        })
        .join()
        .expect("thread joins");
        assert_eq!(other.gpu_nodes, 0, "the other thread saw our GPU node");
        assert_eq!(other.cpu_nodes, 1);

        let ours = take_run_stats();
        assert_eq!(ours.gpu_nodes, 1);
        assert_eq!(ours.cpu_nodes, 0, "we saw the other thread's CPU node");
    }
}
