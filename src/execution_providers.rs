//! Execution Provider compatibility layer for migration from `ort`.
//!
//! `ort` 2.x exposes a rich set of execution provider (EP) types such as
//! `CUDAExecutionProvider`, `CoreMLExecutionProvider`, etc., that allow
//! callers to configure hardware acceleration at session build time.
//!
//! oxionnx selects its backend at compile time via Cargo feature flags
//! (`gpu`, `cuda`).  These stub types mirror the `ort` EP API surface so
//! that code written against `ort` can compile against oxionnx with only
//! a `use` path change — no call-site edits required.
//!
//! Every `build()` call returns an [`ExecutionProviderDispatch`] no-op token.
//! The actual backend selection is governed by the crate's feature flags.

use oxionnx_core::graph::OpKind;
use std::collections::HashMap;

/// Opaque no-op token returned by EP `.build()` calls.
///
/// Passed to [`crate::SessionBuilder::with_execution_providers`], which
/// accepts but ignores the list.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutionProviderDispatch;

// ── CPU ─────────────────────────────────────────────────────────────────────

/// CPU execution provider stub (always active in oxionnx).
#[derive(Debug, Clone, Default)]
pub struct CPUExecutionProvider;

impl CPUExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── CUDA ────────────────────────────────────────────────────────────────────

/// CUDA execution provider stub.
///
/// When the `cuda` feature is enabled, oxionnx routes eligible ops through
/// the CUDA backend automatically.  This type is accepted at the API level
/// for ort-compatible code.
#[derive(Debug, Clone, Default)]
pub struct CUDAExecutionProvider;

impl CUDAExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── CoreML ──────────────────────────────────────────────────────────────────

/// Apple CoreML execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct CoreMLExecutionProvider;

impl CoreMLExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── DirectML ────────────────────────────────────────────────────────────────

/// DirectML (Windows GPU) execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct DirectMLExecutionProvider;

impl DirectMLExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── TensorRT ────────────────────────────────────────────────────────────────

/// NVIDIA TensorRT execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct TensorRTExecutionProvider;

impl TensorRTExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── OpenVINO ────────────────────────────────────────────────────────────────

/// Intel OpenVINO execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct OpenVINOExecutionProvider;

impl OpenVINOExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── Operator Placement ──────────────────────────────────────────────────────

/// Controls how operators are assigned to execution providers.
#[derive(Debug, Clone, Default)]
pub enum OpPlacement {
    /// All ops on CPU (default when no GPU feature).
    #[default]
    CpuOnly,
    /// Auto-select based on op type and tensor size thresholds.
    Auto {
        /// Minimum output tensor bytes for GPU dispatch (default: 65536 = 64KB).
        gpu_threshold_bytes: usize,
    },
    /// Manual per-operator placement.
    Manual(HashMap<OpKind, ProviderKind>),
}

/// Which provider to use for an operator invocation.
///
/// `#[non_exhaustive]`: this enum's variant set is already
/// feature-dependent — `Gpu`/`Cuda`/`DirectMl` only exist when their
/// respective Cargo features are enabled, so no downstream crate can
/// portably `match` it exhaustively today without mirroring these exact
/// `#[cfg]` attributes. Marking it `#[non_exhaustive]` codifies that
/// existing constraint (a wildcard arm was already the only portable way
/// to handle this type) and leaves room for future backends (e.g. ROCm,
/// Vulkan) without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProviderKind {
    /// Pure-Rust CPU execution (always available as fallback).
    Cpu,
    /// wgpu compute backend — Vulkan, Metal or DX12 natively, WebGPU in a
    /// browser (requires the `gpu` feature).
    ///
    /// # wasm32 / WebGPU
    ///
    /// Available, but only through the asynchronous entry points. A browser
    /// thread may not block on a GPU fence, so on `wasm32`:
    ///
    /// * the context must be acquired with
    ///   [`crate::Session::enable_gpu_async`] (session construction cannot
    ///   block on `requestAdapter`, so a session starts without a device), and
    /// * inference must run through [`crate::Session::run_gpu_async`].
    ///
    /// A `wasm32` caller that selects this provider and then calls the
    /// synchronous [`crate::Session::run`] gets correct results computed
    /// entirely on the CPU: the synchronous dispatcher declines every node
    /// there rather than encoding work it could never read back. [a7-10] was
    /// the previous, stronger version of that rule — it declined at context
    /// creation, because no async path existed at all.
    #[cfg(feature = "gpu")]
    Gpu,
    /// NVIDIA CUDA backend (requires `cuda` feature).
    #[cfg(feature = "cuda")]
    Cuda,
    /// Microsoft DirectML backend (requires `directml` feature; no-op off Windows).
    #[cfg(feature = "directml")]
    DirectMl,
}

/// Hard lower bound, in bytes, on the output tensor of any op dispatched to a
/// non-CPU provider via [`OpPlacement::Manual`].
///
/// One 4 KiB memory page = 1024 `f32` elements.
///
/// # Why a floor exists at all
///
/// [`OpPlacement::Manual`] has no size parameter, so without a floor a user who
/// pins `Add -> Cuda` also pins every `[1, 4]` bias-add in the graph to the GPU.
/// Each such dispatch pays the *fixed* cost of a discrete-GPU round trip:
///
/// | stage                                | typical cost |
/// |--------------------------------------|--------------|
/// | host → device copy (PCIe DMA setup)  | ~5–10 µs     |
/// | kernel launch                        | ~3–10 µs     |
/// | fence / stream synchronise           | ~5 µs        |
/// | device → host readback               | ~5–10 µs     |
/// | **round-trip floor (0-byte payload)**| **~20 µs**   |
///
/// That floor is paid *regardless of payload size*.  Meanwhile a single CPU core
/// streams f32 elementwise work at several GB/s, so 4 KiB costs roughly **1 µs**
/// — cache miss included.  Below 4 KiB the GPU therefore loses by more than an
/// order of magnitude no matter which op it is: there is no operator whose
/// arithmetic intensity can amortise a 20 µs fixed cost over fewer than 1024
/// elements.
///
/// # Why 4096 specifically
///
/// - It is one OS page, the granularity at which DMA / pinned-transfer setup
///   cost stops dominating the transfer itself.
/// - It is 1024 `f32`s — below one GPU warp-scheduler's worth of useful work on
///   any modern device (a single SM is idle-ish under 1024 lanes of f32).
/// - It sits an order of magnitude *below* the 64 KiB default of
///   [`OpPlacement::Auto`], so it is a genuine backstop and not a second
///   heuristic competing with the user's `gpu_threshold_bytes`.
///
/// This floor deliberately does **not** apply to [`OpPlacement::Auto`], which has
/// its own explicit, user-supplied `gpu_threshold_bytes` and must honour it
/// exactly.
pub const MIN_GPU_DISPATCH_BYTES: usize = 4096;

// Compile-time invariants on the floor, enforced under every feature combination:
//
//   * It must stay strictly below `OpPlacement::Auto`'s documented 64 KiB default,
//     so that it remains a *backstop* rather than a second heuristic competing
//     with the user's explicit `gpu_threshold_bytes`.
//   * It must be a power of two — the justification above is a page / DMA
//     granularity argument, so a non-power-of-two value would mean the constant
//     was edited without the cost model being revisited.
const _: () = {
    assert!(MIN_GPU_DISPATCH_BYTES < 65_536);
    assert!(MIN_GPU_DISPATCH_BYTES.is_power_of_two());
};

/// Does `provider` have a kernel for `op`?
///
/// This is the op-support predicate of the *actual backend*, not a shared guess:
///
/// (`ProviderKind`'s accelerator variants and the backend crates they name are
/// feature-gated, so these are code spans rather than intra-doc links — they do
/// not exist in every build.)
///
/// - `ProviderKind::Cuda` → `oxionnx_cuda::is_supported_op`
/// - `ProviderKind::DirectMl` → `oxionnx_directml::is_supported_op`
/// - `ProviderKind::Gpu` → [`is_gpu_capable`] (the wgpu op set)
/// - [`ProviderKind::Cpu`] → always `true` (CPU is the terminal fallback and
///   implements every registered operator)
///
/// Cheap and pure: safe to call per-node, on every run, with no device present.
///
/// # Necessary, not sufficient
///
/// `true` means the backend has a kernel for that *op kind*.  The backend may
/// still decline an individual node whose *configuration* is out of range (an
/// oversized softmax row, a broadcasting binary op, a non-flat reduction axis…),
/// in which case dispatch returns `Ok(None)` and the caller falls back.  `false`,
/// by contrast, is a hard guarantee that the backend will never claim the node.
pub fn provider_supports_op(provider: ProviderKind, op: &OpKind) -> bool {
    // `op` is consumed only by the cfg'd accelerator arms below.  With no
    // accelerator feature enabled, `ProviderKind::Cpu` is the enum's sole variant
    // and those arms vanish, leaving `op` genuinely unread.
    let _ = op;

    match provider {
        ProviderKind::Cpu => true,
        #[cfg(feature = "cuda")]
        ProviderKind::Cuda => oxionnx_cuda::is_supported_op(op),
        #[cfg(feature = "directml")]
        ProviderKind::DirectMl => oxionnx_directml::is_supported_op(op),
        #[cfg(feature = "gpu")]
        ProviderKind::Gpu => is_gpu_capable(op),
    }
}

/// The highest-priority *compiled-in* accelerator that has a kernel for `op`,
/// or `None` if every enabled accelerator would decline it.
///
/// Priority order — fixed, and the single source of truth for the whole crate:
///
/// ```text
/// Cuda  >  DirectMl  >  Gpu (wgpu)
/// ```
///
/// CUDA outranks DirectML because it is the more mature and more heavily
/// optimised path; DirectML outranks wgpu because on Windows it talks to D3D12
/// directly rather than through wgpu's portability layer.  A provider that is
/// not compiled in is simply absent from the chain.
///
/// # Policy, not availability
///
/// This function answers *"which backend would be best for this op?"*, not
/// *"is that backend actually alive right now?"*.  It cannot answer the latter:
/// it has no access to the [`crate::Session`]'s device contexts.  In particular,
/// with the `directml` feature enabled on Linux this may return
/// `ProviderKind::DirectMl` even though `DirectMLContext::try_new()` returns
/// `None` there.
///
/// **Callers must therefore treat the result as a preference and still fall
/// through the chain when the chosen provider's context is absent or its
/// dispatch returns `Ok(None)`.** The session dispatch loops already do this via
/// their `if let Some(ctx) = &self.cuda` / `&self.dml` / `&self.gpu` guards.
pub fn select_accelerator(op: &OpKind) -> Option<ProviderKind> {
    #[cfg(feature = "cuda")]
    if oxionnx_cuda::is_supported_op(op) {
        return Some(ProviderKind::Cuda);
    }

    #[cfg(feature = "directml")]
    if oxionnx_directml::is_supported_op(op) {
        return Some(ProviderKind::DirectMl);
    }

    #[cfg(feature = "gpu")]
    if is_gpu_capable(op) {
        return Some(ProviderKind::Gpu);
    }

    // Consumed only by the cfg'd branches above; with no accelerator feature
    // enabled there is nothing to select and `op` is otherwise unused.
    let _ = op;
    None
}

/// Decide placement for a specific operator invocation.
///
/// This is the **single source of truth** for CPU/accelerator routing.  Every
/// dispatch gate in the session layer must agree with it — a gate that bypasses
/// it (for example, "dispatch to CUDA whenever a CUDA context exists") silently
/// re-introduces the size-threshold bug this function exists to prevent.
///
/// * `op` — the operator kind of the node being scheduled.
/// * `output_bytes` — estimated size of the node's output tensor, in bytes.
///   This is what the GPU would have to fence on and read back.
/// * `placement` — the session's configured [`OpPlacement`] strategy.
///
/// # Semantics
///
/// | strategy | behaviour |
/// |----------|-----------|
/// | [`OpPlacement::CpuOnly`] | always [`ProviderKind::Cpu`]. |
/// | [`OpPlacement::Auto`] | `output_bytes < gpu_threshold_bytes` → `Cpu`. Otherwise the highest-priority accelerator that actually has a kernel for `op` ([`select_accelerator`]), else `Cpu`. |
/// | [`OpPlacement::Manual`] | unmapped op → `Cpu`. Pinned to `Cpu` → `Cpu`. Pinned to an accelerator but `output_bytes < `[`MIN_GPU_DISPATCH_BYTES`] → `Cpu` (the floor overrides the pin). Otherwise the pinned provider. |
///
/// `Auto` honours `gpu_threshold_bytes` for **every** provider — CUDA, DirectML
/// and wgpu alike.  Previously the threshold was only consulted on the way to
/// `ProviderKind::Gpu` while the CUDA and DirectML gates ignored it entirely, so
/// `Auto { gpu_threshold_bytes: 1 << 30 }` ("only enormous tensors on the GPU")
/// still shipped a `[1, 4]` bias-add across PCIe.
///
/// `Auto` also consults each backend's **own** op-support predicate rather than
/// the wgpu-flavoured [`is_gpu_capable`].  That matters: `is_gpu_capable` claims
/// `Conv`, but `oxionnx_cuda::is_supported_op` correctly reports `false` for it
/// (there is no CUDA convolution kernel), so `Auto` no longer routes convolutions
/// to CUDA only to have them bounce straight back to the CPU.
///
/// # Availability
///
/// Like [`select_accelerator`], the returned [`ProviderKind`] is a *preference*.
/// Callers must still fall through to the next provider (and ultimately the CPU)
/// when the chosen backend has no live context or its dispatch returns `Ok(None)`.
pub fn decide_placement(op: &OpKind, output_bytes: usize, placement: &OpPlacement) -> ProviderKind {
    match placement {
        OpPlacement::CpuOnly => ProviderKind::Cpu,

        OpPlacement::Auto {
            gpu_threshold_bytes,
        } => {
            // The user's threshold binds every provider, not just wgpu.
            if output_bytes < *gpu_threshold_bytes {
                return ProviderKind::Cpu;
            }
            select_accelerator(op).unwrap_or(ProviderKind::Cpu)
        }

        OpPlacement::Manual(map) => {
            let Some(pinned) = map.get(op).copied() else {
                // Not pinned — CPU is the default for every unmapped op.
                return ProviderKind::Cpu;
            };

            // An explicit CPU pin needs no size check.
            if matches!(pinned, ProviderKind::Cpu) {
                return ProviderKind::Cpu;
            }

            // A pin to an accelerator is still subject to the hard floor: an
            // explicit user preference is not a reason to ship a 16-byte tensor
            // across PCIe.  See MIN_GPU_DISPATCH_BYTES for the cost model.
            if output_bytes < MIN_GPU_DISPATCH_BYTES {
                return ProviderKind::Cpu;
            }

            pinned
        }
    }
}

/// [a7-19] The exact set of [`OpKind`]s that
/// `crate::session::gpu_dispatch::try_gpu_dispatch` has a real match arm for
/// — i.e. the ops the wgpu compute backend can actually claim.
///
/// This is the **single source of truth** for the wgpu op-support surface.
/// [`is_gpu_capable`] below and
/// `crate::session::GpuExecutionProvider::supported_ops` (the third,
/// previously independently hand-maintained list) both derive from this one
/// array, so the three can no longer silently drift apart the way they used
/// to: `is_gpu_capable` claimed `Gemm`/`Sub` (no arm exists — nodes bounced
/// straight back to CPU after a wasted round trip), while `supported_ops`
/// omitted `Gelu`/`ReduceSum`/`ReduceMax`/`ReduceMin`/`Tanh`/`Exp`/`Sqrt`/
/// `Abs`/`Neg`/`Log`/`SiLU`/`LeakyRelu` even though `try_gpu_dispatch`
/// implements all of them, making those kernels unreachable on the default
/// `OpPlacement::Auto` path (which routes through `is_gpu_capable`, not
/// `supported_ops`).
///
/// This array is data, not code, so it still has to be kept in sync with
/// `try_gpu_dispatch`'s match arms **by hand** whenever an arm is added or
/// removed — there is no live GPU available at compile/test time to derive
/// it mechanically. Do that update in the same change that touches the
/// match arms.
pub(crate) const GPU_DISPATCH_OPS: &[OpKind] = &[
    OpKind::MatMul,
    OpKind::Conv,
    OpKind::Softmax,
    OpKind::Relu,
    OpKind::Sigmoid,
    OpKind::Gelu,
    OpKind::ReduceSum,
    OpKind::ReduceMax,
    OpKind::ReduceMin,
    OpKind::Tanh,
    OpKind::Exp,
    OpKind::Sqrt,
    OpKind::Abs,
    OpKind::Neg,
    OpKind::Log,
    OpKind::SiLU,
    OpKind::LeakyRelu,
    OpKind::Add,
    OpKind::Mul,
    OpKind::LayerNorm,
    OpKind::BatchNorm,
    OpKind::Transpose,
    OpKind::ReduceMean,
    // [r3a] Added when `try_gpu_dispatch_async` grew arms for them. Each of
    // these is only reachable *because* it is listed here: `is_gpu_capable`
    // is what `decide_placement`/`gpu_accelerator_gate` consult before the
    // dispatcher is ever called, so a match arm without an entry in this
    // array is unreachable code that no test can distinguish from a kernel
    // that always declines.
    OpKind::Sub,
    OpKind::Div,
    OpKind::PRelu,
    OpKind::Pad,
    OpKind::Resize,
    OpKind::Gemm,
    OpKind::OxiInstanceNorm,
];

/// Check if an operator has a **wgpu** (`gpu` feature) implementation.
///
/// This is the op-support predicate for one specific backend — the wgpu compute
/// path — and is what [`provider_supports_op`] consults for
/// `ProviderKind::Gpu`.  It is *not* a general "can any GPU do this" test:
/// notably it claims `Conv`, which CUDA cannot do.  Use [`select_accelerator`]
/// or [`provider_supports_op`] when you need the answer for a particular backend.
///
/// Backed by `GPU_DISPATCH_OPS` — see its docs for why `Gemm`/`Sub` are
/// deliberately absent (no dispatch arm claims them) even though wgpu is, in
/// principle, capable of both.
pub fn is_gpu_capable(op: &OpKind) -> bool {
    GPU_DISPATCH_OPS.contains(op)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    /// The accelerator that [`select_accelerator`] must pick for a universally
    /// supported op (e.g. `Add`) under the currently-enabled feature set.
    ///
    /// Encodes the mandated priority `Cuda > DirectMl > Gpu`.  Exactly one of the
    /// four `#[cfg]` blocks survives per feature combination, and it is then the
    /// function's tail expression.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    fn highest_priority_accelerator() -> ProviderKind {
        #[cfg(feature = "cuda")]
        {
            ProviderKind::Cuda
        }
        #[cfg(all(not(feature = "cuda"), feature = "directml"))]
        {
            ProviderKind::DirectMl
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "directml"), feature = "gpu"))]
        {
            ProviderKind::Gpu
        }
    }

    // ── MIN_GPU_DISPATCH_BYTES ──────────────────────────────────────────────

    /// The floor is one 4 KiB page = 1024 f32.
    ///
    /// (Its relationship to `Auto`'s 64 KiB default, and its power-of-two-ness,
    /// are enforced at compile time by the `const _` block next to the constant.)
    #[test]
    fn min_gpu_dispatch_bytes_is_one_page() {
        assert_eq!(MIN_GPU_DISPATCH_BYTES, 4096, "one 4 KiB page = 1024 f32");
        assert_eq!(
            MIN_GPU_DISPATCH_BYTES / std::mem::size_of::<f32>(),
            1024,
            "the cost model is stated in f32 elements",
        );
    }

    // ── CpuOnly ─────────────────────────────────────────────────────────────

    #[test]
    fn cpu_only_ignores_size_and_op() {
        let placement = OpPlacement::CpuOnly;
        for op in [
            OpKind::MatMul,
            OpKind::Conv,
            OpKind::Add,
            OpKind::Relu,
            OpKind::Reshape,
        ] {
            assert_eq!(
                decide_placement(&op, usize::MAX, &placement),
                ProviderKind::Cpu,
                "CpuOnly must pin {op:?} to the CPU regardless of size",
            );
        }
    }

    // ── Auto ────────────────────────────────────────────────────────────────

    /// THE regression test for the bug this rework exists to fix.
    ///
    /// `Auto { gpu_threshold_bytes: 1 << 30 }` means "only enormous tensors go to
    /// the GPU".  Before the rework the CUDA and DirectML dispatch gates ignored
    /// `decide_placement` entirely, so a `[1, 4]` f32 bias-add (16 bytes) was
    /// still shipped across PCIe: upload → dispatch → fence-wait → readback, to
    /// replace ~4 ns of f32 addition.
    ///
    /// Must hold under **every** feature combination.
    #[test]
    fn auto_huge_threshold_keeps_a_bias_add_on_the_cpu() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 1 << 30,
        };

        // A [1, 4] f32 bias-add: 16 bytes of output.
        assert_eq!(
            decide_placement(&OpKind::Add, 16, &placement),
            ProviderKind::Cpu,
            "a 16-byte bias-add must never leave the CPU under a 1 GiB threshold",
        );

        // Everything strictly below the threshold stays home, right up to the edge.
        assert_eq!(
            decide_placement(&OpKind::Add, (1 << 30) - 1, &placement),
            ProviderKind::Cpu,
        );

        // MatMul is the most GPU-friendly op there is — the threshold still binds it.
        assert_eq!(
            decide_placement(&OpKind::MatMul, 65_536, &placement),
            ProviderKind::Cpu,
            "gpu_threshold_bytes must bind every provider, not just wgpu",
        );
    }

    #[test]
    fn auto_below_threshold_is_cpu() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 65_536,
        };
        assert_eq!(
            decide_placement(&OpKind::MatMul, 65_535, &placement),
            ProviderKind::Cpu,
        );
    }

    /// At (and above) the threshold, `Auto` selects the highest-priority
    /// accelerator that actually has a kernel for the op.
    #[test]
    fn auto_at_threshold_selects_highest_priority_accelerator() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 4096,
        };
        // `Add` has a kernel in all three backends.
        let at = decide_placement(&OpKind::Add, 4096, &placement);
        let above = decide_placement(&OpKind::Add, 1 << 20, &placement);

        #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
        {
            assert_eq!(
                at,
                highest_priority_accelerator(),
                "priority: Cuda > DirectMl > Gpu"
            );
            assert_eq!(above, highest_priority_accelerator());
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
        {
            assert_eq!(at, ProviderKind::Cpu, "no accelerator compiled in");
            assert_eq!(above, ProviderKind::Cpu);
        }
    }

    /// An op no backend implements stays on the CPU however large it is.
    #[test]
    fn auto_non_accelerable_op_stays_on_cpu() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 0,
        };
        for op in [OpKind::Reshape, OpKind::Shape, OpKind::Gather] {
            assert_eq!(
                decide_placement(&op, 1 << 24, &placement),
                ProviderKind::Cpu,
                "{op:?} has no accelerator kernel",
            );
        }
    }

    /// `Auto` must consult the **backend's own** predicate, not `is_gpu_capable`.
    ///
    /// `oxionnx_cuda::is_supported_op(Conv) == false` — `conv::cuda_conv` always
    /// returns `Ok(None)`.  So `Auto` must never route a `Conv` to CUDA, no matter
    /// which other accelerators are compiled in.  DirectML and wgpu both *do* have a
    /// Conv kernel, and DirectML outranks wgpu, so it is the one that claims Conv when
    /// present.
    #[cfg(feature = "cuda")]
    #[test]
    fn auto_never_routes_conv_to_cuda() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 0,
        };
        let got = decide_placement(&OpKind::Conv, 1 << 20, &placement);
        assert_ne!(
            got,
            ProviderKind::Cuda,
            "CUDA has no Conv kernel; routing Conv there guarantees a wasted round trip",
        );

        // DirectML has a Conv kernel (`DML_CONVOLUTION`) and outranks wgpu, so it
        // claims Conv whenever it is compiled in.
        #[cfg(feature = "directml")]
        assert_eq!(got, ProviderKind::DirectMl);
        // With DirectML off but wgpu on, wgpu's Conv kernel takes the node.
        #[cfg(all(feature = "gpu", not(feature = "directml")))]
        assert_eq!(got, ProviderKind::Gpu);
        // With neither wgpu nor DirectML, Conv has nowhere to go but the CPU.
        #[cfg(all(not(feature = "gpu"), not(feature = "directml")))]
        assert_eq!(got, ProviderKind::Cpu);
    }

    // ── Manual ──────────────────────────────────────────────────────────────

    #[test]
    fn manual_unmapped_op_is_cpu() {
        let placement = OpPlacement::Manual(HashMap::new());
        assert_eq!(
            decide_placement(&OpKind::MatMul, 1 << 24, &placement),
            ProviderKind::Cpu,
        );
    }

    #[test]
    fn manual_explicit_cpu_pin_is_cpu_at_any_size() {
        let mut map = HashMap::new();
        map.insert(OpKind::MatMul, ProviderKind::Cpu);
        let placement = OpPlacement::Manual(map);
        assert_eq!(
            decide_placement(&OpKind::MatMul, 0, &placement),
            ProviderKind::Cpu,
        );
        assert_eq!(
            decide_placement(&OpKind::MatMul, usize::MAX, &placement),
            ProviderKind::Cpu,
        );
    }

    /// A pin to an accelerator is honoured — but only above the hard floor.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn manual_accelerator_pin_enforces_the_size_floor() {
        let accel = highest_priority_accelerator();
        let mut map = HashMap::new();
        map.insert(OpKind::Add, accel);
        let placement = OpPlacement::Manual(map);

        // Below the floor the pin is overridden: nobody wants a 16-byte tensor
        // on a discrete GPU, however emphatically they asked for it.
        assert_eq!(
            decide_placement(&OpKind::Add, 16, &placement),
            ProviderKind::Cpu,
            "a 16-byte tensor must not reach {accel:?} even when explicitly pinned",
        );
        assert_eq!(
            decide_placement(&OpKind::Add, MIN_GPU_DISPATCH_BYTES - 1, &placement),
            ProviderKind::Cpu,
            "the floor is exclusive below MIN_GPU_DISPATCH_BYTES",
        );

        // At and above the floor the pin stands.
        assert_eq!(
            decide_placement(&OpKind::Add, MIN_GPU_DISPATCH_BYTES, &placement),
            accel,
            "the floor is inclusive at MIN_GPU_DISPATCH_BYTES",
        );
        assert_eq!(decide_placement(&OpKind::Add, 1 << 20, &placement), accel,);

        // A different, unpinned op is untouched by the pin.
        assert_eq!(
            decide_placement(&OpKind::Mul, 1 << 20, &placement),
            ProviderKind::Cpu,
        );
    }

    /// With no accelerator feature compiled in, `ProviderKind` has a single
    /// variant, so an accelerator pin is not even expressible: `Manual` can only
    /// ever yield `Cpu`.
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
    #[test]
    fn manual_without_any_accelerator_is_always_cpu() {
        let mut map = HashMap::new();
        map.insert(OpKind::MatMul, ProviderKind::Cpu);
        map.insert(OpKind::Add, ProviderKind::Cpu);
        let placement = OpPlacement::Manual(map);
        for bytes in [
            0,
            MIN_GPU_DISPATCH_BYTES - 1,
            MIN_GPU_DISPATCH_BYTES,
            1 << 24,
        ] {
            assert_eq!(
                decide_placement(&OpKind::MatMul, bytes, &placement),
                ProviderKind::Cpu,
            );
            assert_eq!(
                decide_placement(&OpKind::Reshape, bytes, &placement),
                ProviderKind::Cpu,
            );
        }
    }

    // ── select_accelerator ──────────────────────────────────────────────────

    #[test]
    fn select_accelerator_declines_ops_no_backend_implements() {
        assert_eq!(select_accelerator(&OpKind::Reshape), None);
        assert_eq!(select_accelerator(&OpKind::Shape), None);
        assert_eq!(
            select_accelerator(&OpKind::Unknown("Frobnicate".to_string())),
            None,
            "an unknown op can never be claimed by any backend",
        );
    }

    #[test]
    fn select_accelerator_honours_the_priority_order() {
        // `Add` is implemented by CUDA, DirectML and wgpu alike, so whichever is
        // compiled in with the highest priority must win.
        let got = select_accelerator(&OpKind::Add);

        #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
        assert_eq!(got, Some(highest_priority_accelerator()));
        #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
        assert_eq!(got, None);
    }

    // ── provider_supports_op ────────────────────────────────────────────────

    /// The CPU implements every registered operator; it is the terminal fallback.
    #[test]
    fn provider_supports_op_cpu_is_total() {
        for op in [
            OpKind::MatMul,
            OpKind::Conv,
            OpKind::Reshape,
            OpKind::TreeEnsembleRegressor,
            OpKind::Unknown("Frobnicate".to_string()),
        ] {
            assert!(
                provider_supports_op(ProviderKind::Cpu, &op),
                "CPU must claim {op:?}",
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn provider_supports_op_cuda_delegates_to_the_cuda_crate() {
        assert!(provider_supports_op(ProviderKind::Cuda, &OpKind::MatMul));
        assert!(provider_supports_op(ProviderKind::Cuda, &OpKind::Add));
        assert!(provider_supports_op(ProviderKind::Cuda, &OpKind::Softmax));
        // The whole point: CUDA has no Conv kernel, unlike wgpu.
        assert!(!provider_supports_op(ProviderKind::Cuda, &OpKind::Conv));
        assert!(!provider_supports_op(ProviderKind::Cuda, &OpKind::Reshape));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn provider_supports_op_gpu_is_the_wgpu_op_set() {
        assert!(provider_supports_op(ProviderKind::Gpu, &OpKind::MatMul));
        // wgpu *does* have a Conv kernel — this is where it differs from CUDA.
        assert!(provider_supports_op(ProviderKind::Gpu, &OpKind::Conv));
        assert!(!provider_supports_op(ProviderKind::Gpu, &OpKind::Reshape));
    }

    // ── is_gpu_capable / GPU_DISPATCH_OPS ───────────────────────────────────

    /// [a7-19] `is_gpu_capable` must describe exactly the ops
    /// `try_gpu_dispatch` has a match arm for — no more (that routes a node
    /// to wgpu only for it to bounce straight back to CPU) and no less (that
    /// strands an implemented kernel on the `Auto` path forever).
    ///
    /// `Gemm` and `Sub` are the two ops this test used to (incorrectly)
    /// assert as GPU-capable: `try_gpu_dispatch` has no arm for either, so
    /// under `OpPlacement::Auto` they were being handed to wgpu and declined
    /// every single time — a pure-loss round trip through `select_accelerator`
    /// for a node that could never be accelerated.
    #[test]
    fn is_gpu_capable_matches_try_gpu_dispatch_arms() {
        for op in [
            OpKind::MatMul,
            OpKind::Conv,
            OpKind::Softmax,
            OpKind::Relu,
            OpKind::Sigmoid,
            OpKind::Gelu,
            OpKind::ReduceSum,
            OpKind::ReduceMax,
            OpKind::ReduceMin,
            OpKind::Tanh,
            OpKind::Exp,
            OpKind::Sqrt,
            OpKind::Abs,
            OpKind::Neg,
            OpKind::Log,
            OpKind::SiLU,
            OpKind::LeakyRelu,
            OpKind::Add,
            OpKind::Mul,
            OpKind::LayerNorm,
            OpKind::BatchNorm,
            OpKind::Transpose,
            OpKind::ReduceMean,
            // [r3a] Gained real arms in this wave. `Sub`/`Div`/`Gemm` moved
            // up from the "must NOT be capable" list below, which is the
            // whole point of that list existing: the move is visible here.
            OpKind::Sub,
            OpKind::Div,
            OpKind::PRelu,
            OpKind::Pad,
            OpKind::Resize,
            OpKind::Gemm,
            OpKind::OxiInstanceNorm,
        ] {
            assert!(
                is_gpu_capable(&op),
                "{op:?} has a try_gpu_dispatch match arm and must be GPU-capable",
            );
        }
        for op in [
            OpKind::Reshape,
            OpKind::Squeeze,
            OpKind::Flatten,
            OpKind::Gather,
            OpKind::Shape,
            // [r3a] InSwapper's two remaining CPU op types (48 tiny nodes
            // between them) and SCRFD's shape plumbing. Listed here so that
            // giving either an arm without also listing it above — the exact
            // dead-arm trap this pair of lists guards — fails loudly.
            OpKind::Slice,
            OpKind::Unsqueeze,
            OpKind::Concat,
        ] {
            assert!(
                !is_gpu_capable(&op),
                "{op:?} has no try_gpu_dispatch match arm and must not be GPU-capable",
            );
        }
    }

    /// `GPU_DISPATCH_OPS` must have no duplicate entries — a duplicate would
    /// not break `is_gpu_capable` (a `contains` check), but it would signal
    /// the hand-maintained list has drifted out of careful sync with the
    /// dispatch match arms it is supposed to mirror one-for-one.
    #[test]
    fn gpu_dispatch_ops_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for op in GPU_DISPATCH_OPS {
            assert!(
                seen.insert(op),
                "duplicate entry in GPU_DISPATCH_OPS: {op:?}"
            );
        }
    }
}
