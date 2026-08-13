//! Device-aware dispatch thresholds.
//!
//! # Why these are not constants any more
//!
//! Every `gpu_*` entry point in this crate is allowed to decline, and each one
//! used to decide with a flat compile-time constant: 10 M multiply-accumulates
//! for GEMM, 100 000 elements for element-wise, 50 000 for reduction and
//! normalization. Those numbers answer the question *"is uploading these bytes,
//! computing, and reading the result back cheaper than the CPU kernel?"* — and
//! that question has no adapter-independent answer:
//!
//! * On a **discrete** GPU the operands cross a PCIe-class bus. Measured on
//!   this crate's reference Linux box (RTX A4000 / Vulkan / driver 550.144.03,
//!   `examples/gemm_shape_crossover.rs`), the effective end-to-end host→device
//!   rate is ~4–6 GB/s, which is the same order as the CPU's own main-memory
//!   read rate. So a dispatch whose bytes-moved is comparable to its arithmetic
//!   has no structural advantage at all, however many FLOPs it totals.
//! * On an **integrated** GPU there is no bus: the upload is a copy inside
//!   system RAM. The transfer term collapses, but so does the compute term —
//!   an iGPU's f32 throughput is a fraction of a discrete part's, and it shares
//!   the same memory bandwidth the CPU kernel is already saturating.
//! * On a **software** adapter (Mesa `lavapipe`/`llvmpipe`, SwiftShader,
//!   Direct3D WARP) the "GPU" *is* this CPU, running one shader invocation per
//!   thread with no `matrixmultiply` packing and no rayon fan-out. It cannot
//!   beat the CPU kernel at any size, so the honest threshold is "never".
//!   This is not a hypothetical: a headless Linux container that installs
//!   `mesa-vulkan-drivers` gets `lavapipe` and a perfectly valid
//!   `wgpu::Adapter` with no hardware behind it.
//!
//! [`GpuTuning`] therefore travels with the [`crate::GpuContext`], derived once
//! from the adapter's own [`wgpu::AdapterInfo`], and every kernel asks the
//! context instead of reading a constant.
//!
//! # Shape-awareness, and why a FLOP count alone is the wrong gate
//!
//! Total FLOPs is not sufficient even on one device. `[1, 25088] x [25088, 512]`
//! is 25.7 MFLOP — comfortably past any FLOP floor — and yet it must move a
//! 51.4 MB `B` across the bus to do 25.7 MFLOP of work. Measured on the
//! reference box that dispatch is **1.54x slower** than the CPU kernel, because
//! both sides are bound by streaming `B` exactly once and only the GPU
//! additionally pays a bus crossing, a fence and a read-back.
//!
//! The quantity that separates those cases is the dispatch's *arithmetic
//! intensity* — FLOPs per element that has to cross the bus:
//!
//! ```text
//!            2·m·k·n                  2
//!     I = ─────────────── = ───────────────────────
//!         m·k + k·n + m·n    1/m  +  1/k  +  1/n
//! ```
//!
//! which is dominated by the *smallest* of the three extents: with `k` and `n`
//! large, `I → 2m`, so a small-`m` ("skinny") GEMM has low intensity no matter
//! how large its FLOP count is. That is the shape-awareness
//! [`GpuTuning::gemm_admits`] adds, and it is symmetric — a skinny `n` (a
//! matrix-vector product against a huge weight) and a tiny `k` (a rank-1-ish
//! update writing an enormous output) are declined by the same expression, not
//! by three special cases.
//!
//! # The measured table
//!
//! `gpu_matmul` against `oxionnx_ops::math::matmul` (rayon + `matrixmultiply`),
//! RTX A4000 / Vulkan, 24-thread host, best-of-180 per point after a
//! clock-spin-up burst (this GPU idles at 210 MHz of a 2100 MHz boost and
//! `nvidia-smi -lgc` is denied on it, so an interleaved measurement reports the
//! clock ramp rather than the kernel). Ratios are GPU/CPU; **> 1.00 means the
//! GPU lost**:
//!
//! | m \ k=n | 512 | 1024 | 2048 | 4096 | k=25088,n=512 |
//! |---|---|---|---|---|---|
//! | 1 | — | — | — | 0.81 | **1.54** |
//! | 2 | — | — | — | **1.05** | **1.41** |
//! | 4 | — | — | 0.82 | **1.05** | **1.39** |
//! | 8 | — | — | 0.98 | **1.09** | **1.11** |
//! | 12 | — | **1.27** | 0.73 | 0.95 | **1.16** |
//! | 16 | — | **1.33** | 0.77 | 0.97 | **1.16** |
//! | 24 | — | **1.07** | 0.68 | 0.83 | **1.02** |
//! | 32 | — | 0.88 | 0.53 | 0.66 | 0.79 |
//! | 48 | **1.19** | 0.72 | 0.43 | 0.54 | 0.64 |
//! | 64 | 0.95 | 0.54 | 0.34 | 0.46 | 0.62 |
//! | 96 | 0.69 | 0.38 | 0.26 | 0.36 | 0.44 |
//! | 128 | 0.56 | 0.30 | 0.22 | 0.30 | 0.38 |
//! | 256 | 0.41 | 0.21 | 0.14 | 0.17 | 0.28 |
//!
//! Two independent boundaries fall out of it, and both are needed:
//!
//! * **Intensity.** Every point with `I >= 56` that also cleared the size floor
//!   below won (0.53–0.93). Every measured *loss* has `I <= 47`, except the
//!   three in the 12–17 M band that the size floor removes. `I = 56` with large
//!   `k`/`n` is `m ≈ 28`, which is the directly measured small-`m` crossover:
//!   `m = 32` won at every `(k, n)` tested, `m = 24` lost at two of five.
//! * **Size.** In the 12–17 M band the whole dispatch is under a millisecond,
//!   and the GPU's fixed per-dispatch cost (~0.2 ms here) is the same order as
//!   the entire CPU kernel: `(48, 512, 512)` at 12.6 M lost 1.19x and
//!   `(32, 1024, 512)` at 16.8 M lost 1.20x, both at healthy intensity. The
//!   smallest `m·k·n` that won at *every* shape tested was 25.2 M.
//!
//! The cost of the intensity gate is explicit and was measured, not assumed:
//! it declines `(12, 2048, 2048)` at 0.73 and `(24, 4096, 4096)` at 0.83 —
//! genuine but modest wins — because the same rule is what removes the
//! 1.02–1.54x losses in the `k = 25088` and `k = n = 1024` columns. The
//! principled way to recover them is not a lower threshold: it is residency,
//! which removes `B` from the transfer term entirely (see
//! [`GemmWeightTraffic`]).
//!
//! # The memory-bound kernels lose at *every* size, and that is a measurement
//!
//! The same sweep run against the element-wise, normalization, reduction,
//! transpose and softmax kernels (`examples/kernel_crossover.rs`, same box,
//! same methodology, every operand transferring) produced this — again,
//! **> 1.00 means the GPU lost**:
//!
//! | kernel | 64 Ki | 256 Ki | 1 Mi | 4 Mi | 16 Mi | 64 Mi |
//! |---|---|---|---|---|---|---|
//! | `gpu_relu` (unary EW) | — | 17.9 | 5.6 | 1.9 | 4.2 | 3.7 |
//! | `gpu_add` (binary EW) | — | 37.1 | 20.2 | 13.1 | 5.6 | 4.8 |
//! | `gpu_layer_norm` | 4.1 | 2.1 | 2.3 | 1.8 | 1.8 | — |
//! | `gpu_batch_norm` | 39.9 | 27.0 | 16.1 | 4.2 | — | — |
//! | `gpu_transpose` | 53.9 | 7.3 | 1.10 | 0.91 | 1.55 | — |
//! | `gpu_reduce_sum` (by output count) | — | — | 1.32 (500 Ki) | — | 1.21 (2 Mi) / 0.89 (8 Mi) | — |
//! | `gpu_softmax` (rows × 1024) | 1.55 | 1.02 | **0.62** | **0.52** | 1.06 | — |
//!
//! Run-to-run variance on this clock-unlocked part is roughly +/-30% at the
//! smaller sizes, so the individual entries are indicative and the *orderings*
//! are what the floors are set from: a kernel is treated as having no crossover
//! only when it lost at every size across repeated runs, and softmax — the one
//! that wins — was re-measured to confirm the band rather than a single point.
//!
//! For the first four there is no crossover and there cannot be one, which is
//! why their floors are `usize::MAX` rather than a large number. The argument
//! is structural, not a property of this GPU: an element-wise or per-channel
//! normalization kernel reads `n` and writes `n`. The CPU kernel moves those
//! `2n` elements once, through many-channel DDR at tens of GB/s, rayon-parallel.
//! The GPU must move the same `2n` elements across the bus — measured at 4–6
//! GB/s here — *and then* read and write them again in VRAM, and pay a fence.
//! Both sides are linear in `n` with the GPU's constant an order of magnitude
//! worse, so raising the floor cannot make the GPU win; it can only pick the
//! size at which it starts losing. The session layer reached the identical
//! conclusion from an independent measurement on an Apple M3
//! (`oxionnx::session::gpu_residency::MEMORY_BOUND_TRANSFER_FLOOR`), which is
//! the strongest evidence available that this is not an artifact of one part.
//!
//! **This does not disable those kernels** — it disables them for dispatches
//! that would have to *transfer*. An operand already in a device buffer skips
//! the size gate entirely (`context::activation::skips_size_threshold`), which
//! is the regime where those kernels earn their place and the one activation
//! residency exists to create. What the change removes is the case where the
//! crate's own default gate accepted a dispatch measured 1.8x–45x slower than
//! the CPU call the caller would otherwise have made.
//!
//! `gpu_softmax` is the counter-example that keeps the mechanism honest: it is
//! not purely memory-bound (an `exp` and two reductions per row), it genuinely
//! wins by ~2x in a band, and it keeps a real, finite floor — on the *total*
//! element count as well as the row length, because at 64 rows of 1024 it lost
//! 1.55x while satisfying the row-length gate on its own.
//!
//! # wasm32 is not covered by any of this
//!
//! Every number above was measured against a rayon-parallel native CPU kernel.
//! A browser's is single-threaded, so the ratios do not transfer and this crate
//! has not measured them there. The memory-bound floors therefore keep their
//! historical values on `wasm32` (see [`memory_bound_floor`]) rather than
//! inheriting a conclusion drawn against a 24-thread host. The *GEMM* gates
//! above are applied on every target: they are shape rules whose argument —
//! a dispatch that moves as many elements as it does arithmetic has no
//! advantage to trade — does not depend on how many CPU threads the alternative
//! has.
//!
//! # Overriding
//!
//! [`crate::GpuContext::set_tuning`] replaces the whole struct, which is how
//! the tests pin a threshold without depending on which adapter the machine
//! running them happens to have, and how an embedder that has measured its own
//! target can say so. [`GpuTuning::PARITY`] is the "dispatch anything"
//! preset kernel-numerics tests install.

/// How the adapter behind a [`crate::GpuContext`] is expected to behave, from
/// its own [`wgpu::AdapterInfo`].
///
/// This is a *performance* classification, not a hardware taxonomy: the only
/// thing it is used for is choosing between the threshold sets in
/// [`GpuTuning::for_class`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuPerfClass {
    /// A discrete GPU behind a PCIe-class bus: large compute throughput, and a
    /// real per-dispatch transfer cost to amortize.
    Discrete,
    /// An integrated or virtualized GPU sharing system memory with the host.
    /// Transfers are cheap; compute throughput is far lower, and it competes
    /// with the CPU kernel for the same memory bandwidth.
    Integrated,
    /// A software rasterizer — Mesa `lavapipe`/`llvmpipe`, SwiftShader,
    /// Direct3D WARP. The shader runs on this same CPU. Never worth a
    /// dispatch: see [`GpuTuning::for_class`].
    Software,
    /// The backend did not report a usable device type. Treated as the most
    /// conservative *real* GPU rather than as software, because declining a
    /// working GPU outright is the more expensive mistake and
    /// [`wgpu::DeviceType::Other`] is what several backends report for
    /// perfectly ordinary hardware (this crate's own GL fallback on Linux
    /// reports `Other` for an RTX A4000).
    Unknown,
}

impl GpuPerfClass {
    /// Classify an adapter.
    ///
    /// [`wgpu::DeviceType::VirtualGpu`] is folded into [`Self::Integrated`]
    /// rather than [`Self::Discrete`]: a paravirtualized adapter forwards its
    /// work to a host GPU through a transport this crate cannot measure, and
    /// the conservative side of that uncertainty is the class with the smaller
    /// assumed transfer cost and the smaller assumed compute win.
    #[must_use]
    pub fn from_adapter_info(info: &wgpu::AdapterInfo) -> Self {
        match info.device_type {
            wgpu::DeviceType::DiscreteGpu => Self::Discrete,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => Self::Integrated,
            wgpu::DeviceType::Cpu => Self::Software,
            wgpu::DeviceType::Other => Self::Unknown,
        }
    }

    /// Human-readable name, for diagnostics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Integrated => "integrated",
            Self::Software => "software",
            Self::Unknown => "unknown",
        }
    }
}

/// Whether the `B` operand of a GEMM-shaped dispatch crosses the host→device
/// bus on *every* call, or has a residency identity that makes it cross at most
/// once for the life of the context.
///
/// This is the difference between two cost models, not a hint. With `B`
/// uploaded per dispatch, a skinny GEMM moves `k·n` elements to do `2·m·k·n`
/// FLOPs and loses to the CPU kernel outright — the whole reason
/// [`GpuTuning::gemm_min_intensity`] exists. With `B` cached, those bytes are
/// not in this dispatch's budget at all, and the same shapes win by a wide
/// margin. Measured on the reference box with a warm cache, `[m, 25088] ×
/// [512, 25088]ᵀ` ran at 0.43x the CPU kernel at `m = 1` and 0.30x at `m = 32`,
/// against 1.54x and 0.79x for the same shapes uploading `B`.
///
/// A caller that *has* a cache key reports [`Self::Cached`] even on the first
/// dispatch, when the cache is still cold. That is deliberate: the first call
/// pays one upload and every later call pays none, and a gate keyed on "is it
/// resident *right now*" would decline the very dispatch that would have
/// populated the cache, and so decline every dispatch forever.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmWeightTraffic {
    /// `B` is uploaded for this dispatch and every future one — it has no
    /// residency identity. Every `gpu_matmul*` call is in this regime: the
    /// kernel takes two host slices and has nowhere to cache either.
    PerDispatch,
    /// `B` has a residency identity (`WeightKeys`), so it uploads at most once
    /// per context and binds from the cache afterwards.
    Cached,
}

/// The complete set of size and shape thresholds one [`crate::GpuContext`]
/// dispatches by.
///
/// Every field is public and the struct is `Copy`, so a caller that has
/// measured its own target can build one literally and install it with
/// [`crate::GpuContext::set_tuning`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTuning {
    /// Which class the thresholds below were chosen for.
    pub class: GpuPerfClass,
    /// Minimum `m·k·n` (multiply-accumulates, *not* FLOPs) for a GEMM-shaped
    /// dispatch whose `B` uploads every call. See the module docs for the
    /// measured table behind the discrete value.
    pub gemm_min_mac: u64,
    /// Minimum `m·k·n` for a GEMM-shaped dispatch whose `B` has a residency
    /// identity, and so uploads at most once per context.
    ///
    /// **Lower than [`Self::gemm_min_mac`], and that is the point.** The larger
    /// floor exists to amortize a per-call `k·n` upload; a cached `B` does not
    /// perform one, so applying the same number to it would decline the
    /// dispatches residency was built to make profitable. Concretely, ArcFace's
    /// embedding head — `[1, 25088] × [512, 25088]ᵀ`, 12.8 M multiply-
    /// accumulates — sits *below* the uploading floor and measured **0.43x**
    /// the CPU kernel with a warm cache.
    ///
    /// The value is the session layer's historical `GEMM_GPU_MIN_FLOPS`
    /// (10 MFLOP) expressed in multiply-accumulates, deliberately unchanged:
    /// every cached-weight point this crate has measured, from 12.6 M upward,
    /// won — by 1.3x to 16x — so there is no measurement that would justify
    /// declining more than the engine already did on a path that has only ever
    /// been observed to win.
    pub gemm_min_mac_cached: u64,
    /// Minimum arithmetic intensity — FLOPs per transferred element,
    /// `2·m·k·n / (m·k + k·n + m·n)` — for a GEMM whose `B` uploads per
    /// dispatch. Skipped entirely for [`GemmWeightTraffic::Cached`].
    pub gemm_min_intensity: u64,
    /// Minimum `m·k·n` of the *implied* GEMM inside a convolution.
    ///
    /// Deliberately separate from [`Self::gemm_min_mac`] and left at the value
    /// this crate has always used: `Conv` is the one op the measurements have
    /// always found a clear GPU winner (0.44x the CPU kernel across
    /// InSwapper's 20 convolutions), its im2col/implicit-GEMM cost model is not
    /// the flat GEMM one, and raising a threshold that is not implicated in any
    /// measured regression would trade a real win for nothing.
    pub conv_min_mac: u64,
    /// Minimum element count for a unary element-wise dispatch.
    pub elementwise_min_elements: usize,
    /// Minimum element count for a binary element-wise dispatch.
    pub binary_elementwise_min_elements: usize,
    /// Minimum *output* element count for a reduction dispatch.
    pub reduce_min_output_elements: usize,
    /// Minimum element count for a LayerNorm dispatch.
    pub layer_norm_min_elements: usize,
    /// Minimum element count for a BatchNorm dispatch.
    pub batch_norm_min_elements: usize,
    /// Minimum element count for a Transpose dispatch.
    pub transpose_min_elements: usize,
    /// Minimum row length (last-axis extent) for a Softmax dispatch.
    ///
    /// A row, not a total: the kernel runs one workgroup-level reduction per
    /// row, so a short row wastes the workgroup however many rows there are.
    pub softmax_min_row_len: usize,
    /// Minimum *total* element count for a Softmax dispatch, applied on top of
    /// [`Self::softmax_min_row_len`].
    ///
    /// Both are needed and neither implies the other: `[64, 1024]` satisfies
    /// the row-length gate with room to spare and still measured 1.55x slower
    /// than the CPU kernel, because 64 workgroups do not fill the device and
    /// the whole dispatch is smaller than its own fixed cost. `[1024, 1024]`
    /// — same rows, 16x the work — won by 1.6x.
    pub softmax_min_elements: usize,
}

/// The floor for a kernel whose CPU alternative reads and writes the same bytes
/// once at main-memory bandwidth: `usize::MAX` on native, the historical
/// `legacy` value in a browser.
///
/// See the module docs for the measurement (native: no crossover exists) and
/// for why the browser is deliberately excluded from that conclusion (its CPU
/// kernel is single-threaded, and this crate has not measured the ratio there).
#[must_use]
pub const fn memory_bound_floor(legacy: usize) -> usize {
    if cfg!(target_arch = "wasm32") {
        legacy
    } else {
        usize::MAX
    }
}

/// The floor for a kernel that *does* have a measured crossover: `measured` on
/// native, the historical `legacy` value in a browser.
///
/// Same split, same reason, opposite direction — these are raises to a finite
/// number rather than to "never", so the browser keeps the smaller historical
/// floor instead of inheriting an unmeasured one.
#[must_use]
pub const fn measured_floor(measured: usize, legacy: usize) -> usize {
    if cfg!(target_arch = "wasm32") {
        legacy
    } else {
        measured
    }
}

impl GpuTuning {
    /// The thresholds this crate shipped as compile-time constants before they
    /// became device-aware.
    ///
    /// Kept as a named baseline rather than inlined into
    /// [`Self::for_class`]: three of the four classes below are stated
    /// *relative* to it, and a test can assert that a class has not silently
    /// drifted from the baseline it claims to inherit.
    pub const LEGACY_FLAT: Self = Self {
        class: GpuPerfClass::Unknown,
        gemm_min_mac: 10_000_000,
        // The session layer's historical `GEMM_GPU_MIN_FLOPS` (10 MFLOP) in
        // multiply-accumulates. `gpu_gemm_nt` itself carried no gate at all —
        // `kernel_support`'s convention put the placement heuristic at the
        // session call site — so this is that call site's number, brought here
        // so the two can no longer disagree. They did: the session compared
        // 2·m·k·n against 10 M while `compute.rs` compared m·k·n against the
        // same 10 M, which is a factor of two between the "same" threshold.
        gemm_min_mac_cached: 5_000_000,
        // 0 = no intensity gate, which is exactly what the flat thresholds did.
        gemm_min_intensity: 0,
        conv_min_mac: 10_000_000,
        elementwise_min_elements: 100_000,
        binary_elementwise_min_elements: 100_000,
        reduce_min_output_elements: 50_000,
        layer_norm_min_elements: 50_000,
        batch_norm_min_elements: 50_000,
        transpose_min_elements: 50_000,
        softmax_min_row_len: 1_000,
        // No total-element gate existed; 0 reproduces that exactly.
        softmax_min_elements: 0,
    };

    /// Thresholds that decline nothing.
    ///
    /// For kernel-numerics tests: a parity test wants the shader run at a shape
    /// small enough to check by hand, and that is a different question from
    /// whether a real workload should dispatch it. Install this with
    /// [`crate::GpuContext::set_tuning`] and the shader runs; leave it alone
    /// and the policy above decides. Keeping the two separable is the reason
    /// the policy moved out of the shader modules in the first place.
    pub const PARITY: Self = Self {
        class: GpuPerfClass::Unknown,
        gemm_min_mac: 0,
        gemm_min_mac_cached: 0,
        gemm_min_intensity: 0,
        conv_min_mac: 0,
        elementwise_min_elements: 0,
        binary_elementwise_min_elements: 0,
        reduce_min_output_elements: 0,
        layer_norm_min_elements: 0,
        batch_norm_min_elements: 0,
        transpose_min_elements: 0,
        softmax_min_row_len: 0,
        softmax_min_elements: 0,
    };

    /// Thresholds that decline every dispatch, whatever its size.
    ///
    /// Used for [`GpuPerfClass::Software`]: a `lavapipe`/WARP "GPU" is this
    /// same CPU running one invocation per shader thread, without
    /// `matrixmultiply`'s packing and without rayon, so there is no size at
    /// which dispatching beats calling the CPU operator directly. Declining is
    /// free and total — the contract of every entry point here is already
    /// `None` means "run the CPU kernel" — so this costs nothing but a
    /// comparison, and it is what keeps a headless CI container that happens
    /// to have `mesa-vulkan-drivers` installed from running an entire model
    /// through a software rasterizer.
    pub const NEVER: Self = Self {
        class: GpuPerfClass::Software,
        gemm_min_mac: u64::MAX,
        gemm_min_mac_cached: u64::MAX,
        gemm_min_intensity: u64::MAX,
        conv_min_mac: u64::MAX,
        elementwise_min_elements: usize::MAX,
        binary_elementwise_min_elements: usize::MAX,
        reduce_min_output_elements: usize::MAX,
        layer_norm_min_elements: usize::MAX,
        batch_norm_min_elements: usize::MAX,
        transpose_min_elements: usize::MAX,
        softmax_min_row_len: usize::MAX,
        softmax_min_elements: usize::MAX,
    };

    /// The threshold set for a performance class.
    ///
    /// # Provenance of each arm
    ///
    /// * [`GpuPerfClass::Discrete`] — **measured** on an RTX A4000 over Vulkan;
    ///   see the module docs for the sweep and the two boundaries it produced.
    /// * [`GpuPerfClass::Integrated`] and [`GpuPerfClass::Unknown`] —
    ///   **inherited, not measured.** They take the discrete numbers unchanged.
    ///   That is a deliberate refusal to invent: the two effects that move the
    ///   threshold on an iGPU point in opposite directions (its "upload" is a
    ///   copy in system RAM, which lowers the floor; its f32 throughput and its
    ///   share of the memory bandwidth the CPU kernel is already using are both
    ///   far lower, which raises it), and this crate has no measurement on such
    ///   a part. Replacing that with a guessed number would read as calibration
    ///   while being none. What *is* claimed for these classes is only what the
    ///   arguments behind the discrete numbers already justify without
    ///   reference to the bus: the shape-awareness (intensity) rule, and the
    ///   memory-bound floors, both of which turn on "does this dispatch move as
    ///   many elements as it does arithmetic" rather than on how fast the bus
    ///   is.
    /// * [`GpuPerfClass::Software`] — [`Self::NEVER`]; see it.
    #[must_use]
    pub fn for_class(class: GpuPerfClass) -> Self {
        match class {
            GpuPerfClass::Software => Self::NEVER,
            GpuPerfClass::Discrete | GpuPerfClass::Integrated | GpuPerfClass::Unknown => Self {
                class,
                // Measured: 25.2 M m·k·n is the smallest size that won at every
                // shape tested; 12–17 M lost at three shapes and tied at a
                // fourth. See the module docs.
                gemm_min_mac: 25_000_000,
                // Unchanged from the engine's historical floor; see the field.
                gemm_min_mac_cached: Self::LEGACY_FLAT.gemm_min_mac_cached,
                // Measured: every point at or above 56 FLOP/transferred-element
                // won; every loss outside the sub-millisecond band was at or
                // below 47.
                gemm_min_intensity: 56,
                // Conv is the one measured winner and is deliberately left
                // alone; see the field.
                conv_min_mac: Self::LEGACY_FLAT.conv_min_mac,
                // Measured: no crossover exists while transferring. The
                // residency path bypasses these gates entirely
                // (`skips_size_threshold`), which is where these kernels win.
                elementwise_min_elements: memory_bound_floor(
                    Self::LEGACY_FLAT.elementwise_min_elements,
                ),
                binary_elementwise_min_elements: memory_bound_floor(
                    Self::LEGACY_FLAT.binary_elementwise_min_elements,
                ),
                layer_norm_min_elements: memory_bound_floor(
                    Self::LEGACY_FLAT.layer_norm_min_elements,
                ),
                batch_norm_min_elements: memory_bound_floor(
                    Self::LEGACY_FLAT.batch_norm_min_elements,
                ),
                // Measured: 1.18-1.32x slower up to 2 Mi outputs, 0.89x at
                // 8 Mi. The historical 50 000 was 160x too low.
                reduce_min_output_elements: measured_floor(
                    8_000_000,
                    Self::LEGACY_FLAT.reduce_min_output_elements,
                ),
                // Measured: 53.9x slower at 64 Ki, 7.3x at 256 Ki, 1.10x at
                // 1 Mi, 0.91x at 4 Mi.
                transpose_min_elements: measured_floor(
                    4_000_000,
                    Self::LEGACY_FLAT.transpose_min_elements,
                ),
                // The one memory-adjacent kernel with a real win band; see the
                // field for why the row-length gate alone was not enough.
                softmax_min_row_len: Self::LEGACY_FLAT.softmax_min_row_len,
                softmax_min_elements: measured_floor(
                    262_144,
                    Self::LEGACY_FLAT.softmax_min_elements,
                ),
            },
        }
    }

    /// Derive the tuning for a live adapter.
    #[must_use]
    pub fn from_adapter_info(info: &wgpu::AdapterInfo) -> Self {
        Self::for_class(GpuPerfClass::from_adapter_info(info))
    }

    /// Multiply-accumulate count of an `[m, k] × [k, n]` GEMM, or `None` when
    /// it overflows `u64`.
    ///
    /// `u64`, not `usize`, and that is load-bearing: `usize` is **32 bits** on
    /// wasm32, so a `usize` product overflowed for every GEMM at or above
    /// `m·k·n == 2^32` — a 2048³ multiply is 2x past it. The `checked_mul` that
    /// guards the product then returned `None` and the kernel *declined*, so in
    /// a browser the GPU silently refused exactly the multiplies it exists for
    /// while happily taking the small ones. Measured in Chrome (Apple/metal-3):
    /// 1024³ dispatched in 7.7 ms, 2048³ declined in 0.0 ms with no device
    /// error, which is what put this on the record.
    #[inline]
    #[must_use]
    pub fn gemm_mac(m: usize, k: usize, n: usize) -> Option<u64> {
        (m as u64).checked_mul(k as u64)?.checked_mul(n as u64)
    }

    /// Arithmetic intensity of an `[m, k] × [k, n]` GEMM in FLOPs per
    /// transferred element, rounded down, or `None` on overflow or a degenerate
    /// (zero-extent) shape.
    ///
    /// Computed as `2·m·k·n / (m·k + k·n + m·n)`. The denominator counts every
    /// element that crosses the bus for a fully-transferring dispatch: `A` up,
    /// `B` up, `C` back. See the module docs for why this — and not the
    /// numerator alone — is the quantity that separates the measured wins from
    /// the measured losses.
    #[inline]
    #[must_use]
    pub fn gemm_intensity(m: usize, k: usize, n: usize) -> Option<u64> {
        let (m, k, n) = (m as u64, k as u64, n as u64);
        let mk = m.checked_mul(k)?;
        let kn = k.checked_mul(n)?;
        let mn = m.checked_mul(n)?;
        let moved = mk.checked_add(kn)?.checked_add(mn)?;
        if moved == 0 {
            return None;
        }
        mk.checked_mul(n)?.checked_mul(2).map(|f| f / moved)
    }

    /// Whether a GEMM-shaped dispatch of this shape is worth handing to the
    /// GPU, given how its `B` operand reaches the device.
    ///
    /// Both gates apply to [`GemmWeightTraffic::PerDispatch`]; only the size
    /// gate applies to [`GemmWeightTraffic::Cached`], because the intensity
    /// gate's denominator counts a `k·n` upload that a cached `B` does not
    /// perform. A shape whose `m·k·n` overflows `u64` is nonsensical, not "big
    /// enough for the GPU", and declines.
    #[must_use]
    pub fn gemm_admits(&self, m: usize, k: usize, n: usize, traffic: GemmWeightTraffic) -> bool {
        // A zero extent is a degenerate shape, not a small one: every kernel
        // here rejects it anyway (`gemm_buffer_sizes`), and without this an
        // all-floors-zero tuning would "admit" it and make that rejection look
        // like a kernel failure rather than the gate's answer.
        if m == 0 || k == 0 || n == 0 {
            return false;
        }
        let Some(mac) = Self::gemm_mac(m, k, n) else {
            return false;
        };
        match traffic {
            GemmWeightTraffic::Cached => mac >= self.gemm_min_mac_cached,
            GemmWeightTraffic::PerDispatch => {
                mac >= self.gemm_min_mac
                    && Self::gemm_intensity(m, k, n).is_some_and(|i| i >= self.gemm_min_intensity)
            }
        }
    }

    /// Whether the implied GEMM inside a convolution clears
    /// [`Self::conv_min_mac`].
    #[must_use]
    pub fn conv_admits(&self, m: usize, k: usize, n: usize) -> bool {
        Self::gemm_mac(m, k, n).is_some_and(|mac| mac >= self.conv_min_mac)
    }
}

impl Default for GpuTuning {
    /// [`GpuPerfClass::Unknown`] — the class a context built from a
    /// caller-supplied device/queue pair gets, since no `AdapterInfo` came with
    /// it.
    fn default() -> Self {
        Self::for_class(GpuPerfClass::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_adapters_never_dispatch() {
        let t = GpuTuning::for_class(GpuPerfClass::Software);
        assert_eq!(t, GpuTuning::NEVER);
        // Even a 4096^3 GEMM — 68.7 G multiply-accumulates — declines.
        assert!(!t.gemm_admits(4096, 4096, 4096, GemmWeightTraffic::Cached));
        assert!(!t.gemm_admits(4096, 4096, 4096, GemmWeightTraffic::PerDispatch));
        assert!(!t.conv_admits(4096, 4096, 4096));
        assert_eq!(t.elementwise_min_elements, usize::MAX);
        assert_eq!(t.softmax_min_row_len, usize::MAX);
    }

    #[test]
    fn adapter_device_type_maps_onto_the_perf_classes() {
        let info = |device_type| wgpu::AdapterInfo {
            name: String::new(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        };
        assert_eq!(
            GpuPerfClass::from_adapter_info(&info(wgpu::DeviceType::DiscreteGpu)),
            GpuPerfClass::Discrete
        );
        assert_eq!(
            GpuPerfClass::from_adapter_info(&info(wgpu::DeviceType::IntegratedGpu)),
            GpuPerfClass::Integrated
        );
        // Paravirtualized adapters take the conservative arm — see
        // `GpuPerfClass::from_adapter_info`.
        assert_eq!(
            GpuPerfClass::from_adapter_info(&info(wgpu::DeviceType::VirtualGpu)),
            GpuPerfClass::Integrated
        );
        assert_eq!(
            GpuPerfClass::from_adapter_info(&info(wgpu::DeviceType::Cpu)),
            GpuPerfClass::Software
        );
        // `Other` must NOT be read as software: this crate's own GL fallback
        // reports `Other` for a real RTX A4000.
        assert_eq!(
            GpuPerfClass::from_adapter_info(&info(wgpu::DeviceType::Other)),
            GpuPerfClass::Unknown
        );
    }

    #[test]
    fn intensity_is_dominated_by_the_smallest_extent() {
        // With k and n large, I -> 2m.
        assert_eq!(GpuTuning::gemm_intensity(1, 1 << 20, 1 << 20), Some(1));
        assert_eq!(GpuTuning::gemm_intensity(16, 1 << 20, 1 << 20), Some(31));
        // Symmetric in the three extents: a skinny `n` is just as poor.
        assert_eq!(
            GpuTuning::gemm_intensity(1 << 20, 1 << 20, 1),
            GpuTuning::gemm_intensity(1, 1 << 20, 1 << 20)
        );
        // ...and so is a tiny `k`.
        assert_eq!(
            GpuTuning::gemm_intensity(1 << 20, 1, 1 << 20),
            GpuTuning::gemm_intensity(1, 1 << 20, 1 << 20)
        );
    }

    /// The measured losses in the module-doc table must be declined, and the
    /// measured wins admitted. These are the actual shapes, not stand-ins.
    #[test]
    fn the_gate_reproduces_the_measured_table() {
        let t = GpuTuning::for_class(GpuPerfClass::Discrete);
        let up = GemmWeightTraffic::PerDispatch;
        // Losses: skinny m against a large uploaded B.
        for (m, k, n, ratio) in [
            (1usize, 25088usize, 512usize, 1.54),
            (2, 25088, 512, 1.41),
            (8, 25088, 512, 1.11),
            (16, 25088, 512, 1.16),
            (24, 25088, 512, 1.02),
            (12, 1024, 1024, 1.27),
            (16, 1024, 1024, 1.33),
            (24, 1024, 1024, 1.07),
            (2, 4096, 4096, 1.05),
            (4, 4096, 4096, 1.05),
            (8, 4096, 4096, 1.09),
        ] {
            assert!(
                !t.gemm_admits(m, k, n, up),
                "[{m},{k},{n}] measured {ratio}x SLOWER than the CPU kernel; must decline"
            );
        }
        // Losses in the sub-millisecond band: healthy intensity, too small for
        // the GPU's fixed dispatch cost.
        for (m, k, n, ratio) in [
            (48usize, 512usize, 512usize, 1.19),
            (64, 512, 512, 0.95),
            (32, 512, 1024, 1.02),
            (32, 1024, 512, 1.20),
        ] {
            assert!(
                !t.gemm_admits(m, k, n, up),
                "[{m},{k},{n}] measured {ratio}x against the CPU kernel; must decline"
            );
        }
        // Wins: must still dispatch.
        for (m, k, n, ratio) in [
            (32usize, 1024usize, 1024usize, 0.88),
            (32, 2048, 2048, 0.53),
            (32, 4096, 4096, 0.66),
            (32, 25088, 512, 0.79),
            (48, 512, 1024, 0.80),
            (96, 512, 512, 0.69),
            (256, 1024, 1024, 0.21),
            (384, 256, 256, 0.59),
        ] {
            assert!(
                t.gemm_admits(m, k, n, up),
                "[{m},{k},{n}] measured {ratio}x the CPU kernel; must dispatch"
            );
        }
    }

    /// A cached `B` changes the cost model, not just the constant: the same
    /// skinny shapes that lose while uploading win by 2-16x from the residency
    /// cache, and must not be declined by a rule written for the uploading
    /// case.
    #[test]
    fn a_cached_weight_skips_the_intensity_gate() {
        let t = GpuTuning::for_class(GpuPerfClass::Discrete);
        // ArcFace's embedding head, measured 0.43x the CPU kernel with a warm
        // cache and 1.54x without one.
        assert!(!t.gemm_admits(1, 25088, 512, GemmWeightTraffic::PerDispatch));
        assert!(t.gemm_admits(1, 25088, 512, GemmWeightTraffic::Cached));
        // The size floor still applies to a cached weight: a dispatch too small
        // to amortize the fence is too small either way.
        assert!(!t.gemm_admits(1, 512, 2048, GemmWeightTraffic::Cached));
    }

    #[test]
    fn conv_keeps_the_threshold_it_was_measured_with() {
        let t = GpuTuning::for_class(GpuPerfClass::Discrete);
        assert_eq!(t.conv_min_mac, GpuTuning::LEGACY_FLAT.conv_min_mac);
        // Unchanged from the flat constant: 10 M in, one below out.
        assert!(t.conv_admits(10_000_000, 1, 1));
        assert!(!t.conv_admits(9_999_999, 1, 1));
    }

    #[test]
    fn a_shape_whose_mac_overflows_declines_rather_than_wrapping() {
        let t = GpuTuning::for_class(GpuPerfClass::Discrete);
        let huge = usize::MAX;
        assert_eq!(GpuTuning::gemm_mac(huge, huge, huge), None);
        assert!(!t.gemm_admits(huge, huge, huge, GemmWeightTraffic::Cached));
        assert!(!t.gemm_admits(huge, huge, huge, GemmWeightTraffic::PerDispatch));
        assert!(!t.conv_admits(huge, huge, huge));
    }

    #[test]
    fn a_zero_extent_gemm_declines() {
        let t = GpuTuning::for_class(GpuPerfClass::Discrete);
        assert_eq!(GpuTuning::gemm_intensity(0, 0, 0), None);
        assert!(!t.gemm_admits(0, 4096, 4096, GemmWeightTraffic::PerDispatch));
        assert!(!t.gemm_admits(4096, 0, 4096, GemmWeightTraffic::Cached));
    }

    /// Every non-software class gets the same table: the arguments behind these
    /// numbers are about the *shape* of the traffic, not about how fast the bus
    /// is, so there is nothing to specialize until someone measures an iGPU.
    #[test]
    fn the_non_software_classes_share_one_table() {
        let discrete = GpuTuning::for_class(GpuPerfClass::Discrete);
        for class in [GpuPerfClass::Integrated, GpuPerfClass::Unknown] {
            let t = GpuTuning::for_class(class);
            assert_eq!(t.class, class);
            assert_eq!(GpuTuning { class, ..t }, GpuTuning { class, ..discrete });
        }
    }

    /// Native only: the browser arm keeps the historical floors because the
    /// measurement behind the raise was taken against a rayon-parallel CPU
    /// kernel that a browser does not have.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_memory_bound_kernels_decline_every_transferring_size() {
        let t = GpuTuning::for_class(GpuPerfClass::Discrete);
        // Measured 1.9x-45.7x slower than the CPU kernel at every size from
        // 100 000 to 64 Mi elements; the loss is structural, so the floor is
        // "never" rather than a large number.
        assert_eq!(t.elementwise_min_elements, usize::MAX);
        assert_eq!(t.binary_elementwise_min_elements, usize::MAX);
        assert_eq!(t.layer_norm_min_elements, usize::MAX);
        assert_eq!(t.batch_norm_min_elements, usize::MAX);
        // These two have a measured crossover, so they get a finite raise.
        assert_eq!(t.reduce_min_output_elements, 8_000_000);
        assert_eq!(t.transpose_min_elements, 4_000_000);
        // Softmax genuinely wins in a band and keeps both of its gates.
        assert_eq!(t.softmax_min_row_len, 1_000);
        assert_eq!(t.softmax_min_elements, 262_144);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn the_browser_keeps_the_unmeasured_historical_floors() {
        let t = GpuTuning::for_class(GpuPerfClass::Unknown);
        assert_eq!(
            t.elementwise_min_elements,
            GpuTuning::LEGACY_FLAT.elementwise_min_elements
        );
        assert_eq!(
            t.layer_norm_min_elements,
            GpuTuning::LEGACY_FLAT.layer_norm_min_elements
        );
        assert_eq!(
            t.reduce_min_output_elements,
            GpuTuning::LEGACY_FLAT.reduce_min_output_elements
        );
        assert_eq!(t.softmax_min_elements, 0);
    }

    /// The parity preset must decline nothing, or a kernel-numerics test that
    /// installs it would still be testing the policy.
    #[test]
    fn the_parity_preset_declines_nothing() {
        let t = GpuTuning::PARITY;
        assert!(t.gemm_admits(1, 1, 1, GemmWeightTraffic::PerDispatch));
        assert!(t.gemm_admits(1, 1, 1, GemmWeightTraffic::Cached));
        assert!(t.conv_admits(1, 1, 1));
        assert_eq!(t.elementwise_min_elements, 0);
        assert_eq!(t.binary_elementwise_min_elements, 0);
        assert_eq!(t.reduce_min_output_elements, 0);
        assert_eq!(t.layer_norm_min_elements, 0);
        assert_eq!(t.batch_norm_min_elements, 0);
        assert_eq!(t.transpose_min_elements, 0);
        assert_eq!(t.softmax_min_row_len, 0);
        assert_eq!(t.softmax_min_elements, 0);
        // ...but a degenerate shape is still not a dispatch.
        assert!(!t.gemm_admits(0, 8, 8, GemmWeightTraffic::PerDispatch));
    }
}
