# oxionnx-gpu

GPU compute backend for OxiONNX -- wgpu-based MatMul and Conv2D acceleration.

This crate provides GPU-accelerated implementations of performance-critical ONNX
operators using `wgpu` -- natively via Vulkan, Metal or DX12, and, through a
second async execution path, via WebGPU under `wasm32` -- through a Pure Rust
API. Every entry point returns `Option<_>`: `None` means "this crate
declines, run the CPU operator instead", so a missing adapter, an oversized
tensor or a device error degrades to CPU execution rather than failing. Dispatch
also declines instead of silently producing wrong results on shapes its kernels
don't fully support -- MatMul dropping batch dimensions, Softmax ignoring `axis`,
reduction ignoring `keepdims`, elementwise ops gating on element count instead of
shape equality, and fused Conv discarding the optimizer's activation are all
closed. Every `wgpu` call runs inside an error scope with CPU fallback instead
of panicking on a validation/OOM/device-lost error, dispatch is checked against
the device's own workgroup and buffer-size limits, and blocking readback has a
timeout instead of being able to hang forever. The same decline-rather-than-fail
contract now also covers device memory: `Conv`/`Gemm` weights and inter-node
activations can stay resident in a device buffer instead of crossing the
host/device bus on every dispatch, and every buffer this crate allocates is
checked against a live-byte budget before creation, so a would-be allocation
the device can't afford is a decline rather than an attempted `createBuffer`
that fails.

## Key Types and Functions

### Context and Resource Management

- **`GpuContext`** -- Holds the wgpu `Device` and `Queue`, every cached compute pipeline, the weight-residency cache and the live-byte budget. `GpuContext::try_new()` creates one synchronously via `pollster` and returns `None` on `wasm32` (a browser thread cannot block on adapter acquisition); `GpuContext::try_new_async()` works on both native and `wasm32` and is the only way to get a context in a browser, where it requests a `wgpu::Backends::BROWSER_WEBGPU` adapter. The device is requested with the *adapter's* limits, not the WebGPU baseline, so large tensors are not forced onto the CPU by a 128 MiB binding cap the hardware does not have. `GpuContext::set_f16_compute(bool) -> bool` toggles half-precision compute mid-session, returning the *effective* state (`set_f16_compute(true) == false` means the device lacks the feature and stays on `f32`); `GpuContext::{live_gpu_bytes, gpu_byte_budget, set_gpu_byte_budget}` and `GpuContext::{resident_bytes, resident_counters, resident_len}` introspect the budget and the weight cache below.
- **`GpuBufferPool`** -- Reusable buffer pool that reduces allocation overhead by recycling GPU output buffers. Bounded by both a buffer count and a byte budget (`DEFAULT_POOL_BYTE_BUDGET`, 256 MiB by default), with least-recently-used eviction, so idle buffers cannot pin unbounded VRAM. `get_buffer` returns `Option<TrackedBuffer>` -- `None` when even reclaiming idle entries can't fit the request under the live-byte budget, so the caller declines to the CPU instead of attempting an allocation that would fail.
- **`GpuMemoryBudget`** / **`TrackedBuffer`** -- Live-byte accounting behind every allocation this crate makes. `TrackedBuffer` is the sole owner every buffer goes through: it calls `wgpu::Buffer::destroy` and releases its reservation on drop, which makes "allocated bytes are eventually released" a property of the type rather than of each kernel's error paths -- dropping a bare `wgpu::Buffer` frees nothing on the WebGPU backend (`WebBuffer::drop` is a no-op there), so an untracked pool would otherwise leak device memory until `createBuffer` started failing. `GpuMemoryBudget::admits`/`admits_all` check a prospective allocation against the ceiling (`DEFAULT_LIVE_BYTE_BUDGET`, 1.5 GiB by default) before it is made, not after.

### Weight and Activation Residency

- **Weight residency** -- `Conv`'s weight/bias and `Gemm`'s `B`/`C` are bytes that never change between frames; InSwapper-128 alone re-uploaded 502.7 MB of them on every prior dispatch. `WeightKeys::new(weight, bias)` (`context::resident`) names the two cacheable operand slots, keyed by the caller's own opaque, stable identity (this crate never interprets it) and, independently, by numeric format (`WeightFormat::F32`/`F16`) so a mid-session `set_f16_compute` flip cannot serve a kernel the wrong format's bytes. A lookup is checked, not trusted: a key whose recorded byte length or kernel slot disagrees with the current request uploads for that one dispatch instead of overwriting a possibly-correct entry. `GpuContext::resident_counters()` returns a `ResidentCounters { hits, misses, uploaded_bytes }` (with `since`/`is_idle` for taking deltas between two snapshots); `GpuContext::{resident_bytes, resident_len}` report the cache's current size. Three kernel families expose it, each in two forms -- a `_resident_async` convenience over plain `&Tensor`/`&[f32]` I/O (`gpu_conv2d_fused_resident_async`, `gpu_conv2d_implicit_resident_async`, `gpu_gemm_nt_resident_async`), and a `_placed_async` form that additionally takes an input `TensorSource` and an output `OutputPlacement`, combining weight residency with the activation residency below in one call (`gpu_conv2d_fused_placed_async`, `gpu_conv2d_implicit_placed_async`, `gpu_gemm_nt_placed_async`). The chunked im2col hybrid fallback (below) does not participate: grouped convolutions upload one sliced buffer per group rather than one buffer for the whole weight tensor, so a single caller identity would not name a single buffer there.
- **Activation residency** -- the other half of the transfer problem: a value one GPU node produces can now stay in its device buffer for the next GPU node to consume in place, instead of a read-back and a re-upload at the node boundary. `context::activation` supplies the primitives: `DeviceTensor` (a budget-accounted device buffer plus shape, destroyed -- and its bytes released -- when the caller drops it), `TensorSource` (an operand that is either a host slice or a borrowed device buffer), and `OutputPlacement`/`GpuOutput` -- ask a kernel for `OutputPlacement::Device` and get `GpuOutput::Device(DeviceTensor)` back with no staging buffer, copy or fence wait, instead of `GpuOutput::Host(Tensor)`. `read_device_tensor_async(ctx, &device_tensor)` is the lazy escape hatch for a `DeviceTensor` whose consumer turns out to need it on the host after all; the tensor itself is borrowed, not consumed, so a later GPU consumer can still bind it in place.

### Compute Operations

- **`gpu_matmul`** / **`gpu_matmul_tiled`** -- GPU matrix multiplication with optional tiled variant for better cache behavior.
- **`gpu_gemm_nt`** -- `Gemm` with `transB=1` (`B` supplied and read transposed in-kernel), real `alpha`/`beta` uniforms, and `C` optional, row-broadcast, or a full matrix; `gpu_gemm_nt_resident_async`/`gpu_gemm_nt_placed_async` are its weight- and weight+activation-resident forms (see above).
- **`gpu_conv2d`** / **`gpu_conv2d_async`** -- Convolution without a fused activation; the caller applies any activation itself afterward (avoids double-applying a non-idempotent activation like LeakyRelu when the optimizer's own Conv+activation fusion already folded one into the node).
- **`gpu_conv2d_fused`** / **`gpu_conv2d_fused_async`** -- the same convolution with bias and activation (`ConvActivation`) fused into the kernel epilogue; see "Weight and Activation Residency" above for its `_resident_async`/`_placed_async` forms. Both `gpu_conv2d*` entry points now try the direct implicit-GEMM kernel first (below) and fall back to the chunked im2col-on-CPU/GEMM-on-GPU hybrid only for what it declines -- `group > 1`, or a shape the device can't bind or dispatch -- reusing upload buffers across the chunked dispatch within a byte/iteration budget instead of holding every chunk live at once.
- **Direct implicit-GEMM Conv2D** (`gpu_conv2d_implicit_placed_async` / `gpu_conv2d_implicit_resident_async`) -- what both of the above now dispatch to first: gathers im2col in-register inside the shader as the input tile is staged into workgroup memory, and fuses bias and activation into the epilogue, instead of materializing and re-uploading a column matrix `kH*kW` times the size of the input.
- **`gpu_relu`**, **`gpu_sigmoid`**, **`gpu_gelu`**, **`gpu_silu`**, **`gpu_leaky_relu`** (ONNX default `alpha = 0.01`), **`gpu_leaky_relu_alpha`** (explicit `alpha`, reading the node's own attribute), **`gpu_prelu`** (per-channel slope) -- Element-wise activation functions.
- **`gpu_tanh`**, **`gpu_abs`**, **`gpu_neg`**, **`gpu_exp`**, **`gpu_sqrt`**, **`gpu_log`** -- Unary element-wise math.
- **`gpu_add`**, **`gpu_mul`** -- Element-wise binary operations.
- **`gpu_broadcast_add`**, **`gpu_broadcast_sub`**, **`gpu_broadcast_mul`**, **`gpu_broadcast_div`** -- NumPy-style broadcasting (up to rank 4), covering shapes the flat kernels above decline outright, such as the per-channel `[1,C,H,W] op [1,C,1,1]` case (`BroadcastKind` selects the op in the shared `gpu_broadcast_placed_async` entry point).
- **`gpu_pad`** (`PadMode::Constant` / `PadMode::Reflect`) -- Padding, including reflect-mode boundary handling.
- **`gpu_resize_bilinear_pytorch_half_pixel`**, **`gpu_resize_nearest_asymmetric`** -- Resize kernels matching `oxionnx-ops`'s `pytorch_half_pixel` bilinear and `asymmetric` nearest coordinate-transform formulas.
- **`gpu_softmax`** -- Softmax along the last axis, via a single workgroup-level-reduction pipeline (previously two passes).
- **`gpu_layer_norm`**, **`gpu_layer_norm_axis`**, **`gpu_batch_norm`**, **`gpu_instance_norm`** -- Normalization layers; `gpu_instance_norm` backs the optimizer's fused `OxiInstanceNorm` op (per-`(n, c)`-plane mean/variance, no affine term).
- **`gpu_transpose`** -- Tensor axis transposition on GPU.
- **`gpu_reduce_sum`**, **`gpu_reduce_max`**, **`gpu_reduce_min`**, **`gpu_reduce_mean`** -- Reduction operations.

### Async Execution

Each kernel family above exposes both a sync and an async form generated from one body: `gpu_x_async` **is** the implementation, and the synchronous `gpu_x` is a blocking wrapper around it. On native, either form works -- the synchronous one blocks the calling thread on the GPU fence exactly as it always has, and the async form's own read-back *is* the blocking step, so it exists there for API parity and tests rather than concurrency. On `wasm32`, only `gpu_x_async` runs: the synchronous forms return `None` there instead of attempting a blocking read-back a browser thread cannot perform. The `_placed_async`/`_resident_async` residency entry points above are async-only -- there is no blocking `gpu_conv2d_fused_placed` or `gpu_gemm_nt_resident`. Two `gpu_*_async` calls from this crate must never be in flight on one device at the same time -- wgpu error scopes are a per-thread LIFO stack that the native backend panics on if popped out of order.

The `_placed_async` family -- `gpu_add_placed_async`, `gpu_broadcast_placed_async` (`BroadcastKind`), `gpu_resize_placed_async` (`ResizeKind`), the residency entry points above, and others -- marks the variants that accept an `OutputPlacement` and return `GpuOutput`. `read_device_tensor_async` and `GpuLimits` (the subset of `wgpu::Limits` the dispatch paths need -- `max_storage_buffer_binding_size`, `max_buffer_size`, `max_workgroups_per_dimension` -- cached once at context creation, with `storage_fits`/`buffer_fits`/`all_storage_fit` checking a byte size against them) round out the async-path support types.

## Usage

```toml
[dependencies]
oxionnx-gpu = "0.1.7"
```

```rust
use oxionnx_gpu::{GpuContext, gpu_matmul};
use oxionnx_core::Tensor;

// Initialize GPU (returns None if no suitable adapter is found)
if let Some(ctx) = GpuContext::try_new() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = gpu_matmul(&ctx, &a, &b).expect("GPU matmul");
    println!("Result shape: {:?}", result.shape);
}
```

## Dispatch Thresholds Are Device-Aware, and Shape-Aware

Every entry point may decline. *When* it declines used to be a set of flat
compile-time constants (10 M multiply-accumulates for GEMM, 100 000 elements for
element-wise, 50 000 for reduction/normalization); it is now
`GpuTuning`, derived once from the adapter's own `wgpu::AdapterInfo` and carried
on the `GpuContext` (`GpuContext::tuning()`, `GpuContext::perf_class()`,
`GpuContext::set_tuning()`). Three things that change:

- **A FLOP count alone is the wrong gate.** `[1, 25088] x [25088, 512]` is
  25.7 MFLOP -- past any flat threshold -- and must move a 51.4 MB `B` across the
  bus to do it. Measured on an RTX A4000 over Vulkan, that dispatch is **1.54x
  slower** than `oxionnx-ops`' CPU matmul. `GpuTuning::gemm_admits` therefore also
  gates on arithmetic intensity, `2*m*k*n / (m*k + k*n + m*n)`, which is dominated
  by the *smallest* extent and so declines skinny problems in any of the three
  dimensions rather than only small-`m` ones.
- **Residency changes the answer, so the gate takes it as an input.** The same
  shape through `gpu_gemm_nt_resident_async`, whose `B` has a cache identity and
  crosses the bus once per context rather than once per call, measured **0.43x**
  -- a 2.3x win. `GemmWeightTraffic::{PerDispatch, Cached}` selects between the
  two cost models; the intensity rule applies only to the first.
- **A software adapter never dispatches.** `wgpu::DeviceType::Cpu` (Mesa
  `lavapipe`/`llvmpipe`, SwiftShader, Direct3D WARP) is this same CPU running one
  invocation per shader thread, without `matrixmultiply`'s packing and without
  rayon; it cannot win at any size. A headless container that installs
  `mesa-vulkan-drivers` gets one and a perfectly valid adapter with no hardware
  behind it, which is exactly the case this classification exists for.

Measured on the same box, with every operand transferring, the memory-bound
kernels (`gpu_relu`, `gpu_add`, `gpu_layer_norm`, `gpu_batch_norm`) lose to their
CPU counterparts at **every** size tried, from 64 Ki to 64 Mi elements, by 1.8x to
45x -- so their native floors are `usize::MAX` rather than a large number. That
does not disable them: an operand already in a device buffer skips the size gate
entirely (`skips_size_threshold`), which is the regime activation residency
creates and the one where those kernels earn their place. `GpuTuning::PARITY`
lifts every floor, for tests that want to exercise a shader's numerics rather
than the placement policy. See `context::tuning` for the full measured tables and
the provenance of each number.

## Platform Notes

- **Native** (Linux/macOS/Windows): `GpuContext::try_new()` uses `pollster` for synchronous GPU initialization; backends are selected automatically (Vulkan, Metal, or DX12). Unchanged by this release.
- **Linux needs the Vulkan *loader*, which is a separate package from the driver.** `requested_backends()` asks for `wgpu::Backends::VULKAN` only on Linux, and wgpu reaches a Vulkan GPU through `libvulkan.so.1` -- Debian/Ubuntu `libvulkan1`, Fedora/RHEL/Alpine `vulkan-loader`, Arch `vulkan-icd-loader`. The loader is what reads the ICD manifests in `/usr/share/vulkan/icd.d/` and dlopens the vendor driver behind them; **NVIDIA's driver package installs the manifest and the driver library but not the loader**, and nothing pulls it into a minimal container image. Reproduced on this crate's reference box: with `nvidia-smi` fully working and `/usr/share/vulkan/icd.d/nvidia_icd.json` present, wgpu enumerated *zero* Vulkan adapters and `GpuContext::try_new()` returned a bare `None`; `apt-get install libvulkan1` turned the same call into a working context, with no driver change and no reboot. Use **`GpuContext::try_new_diagnosed()`** (or `try_new_diagnosed_async`) instead of `try_new` anywhere the answer reaches a human: on failure it re-enumerates every *other* backend and, when one of them reports a real GPU, says so and names the package to install, rather than returning the same `None` a machine with no GPU would.
- **WebAssembly**: `wasm32` now has a real async path. `GpuContext::try_new_async()` acquires a `wgpu::Backends::BROWSER_WEBGPU` adapter, and kernel read-backs await a genuine `map_async` promise instead of the native blocking path; `GpuContext::try_new()` still returns `None` here, deliberately -- nothing on that target may block a thread waiting on GPU work, and blocking would deadlock the page. `cargo build -p oxionnx --target wasm32-unknown-unknown --features wasm,gpu` compiles, and the async entry points are exercised by tests, but only on native targets so far (`pollster::block_on`-driven, taking the native `try_new`/blocking-read-back branch, never `BROWSER_WEBGPU` acquisition or the browser read-back path) -- this crate's async path has not been run in an actual browser. This crate is the GPU backend, not the session layer: the entry points that would drive it from a browser are `Session::enable_gpu_async()`/`Session::run_gpu_async()` on the `oxionnx` crate, and as of this release they are not yet wired into `oxionnx`'s wasm-bindgen-exported `WasmSession`, which still runs every GPU-eligible node on the CPU.

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
