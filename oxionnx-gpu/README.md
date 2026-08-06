# oxionnx-gpu

GPU compute backend for OxiONNX -- wgpu-based MatMul and Conv2D acceleration.

This crate provides GPU-accelerated implementations of performance-critical ONNX
operators using `wgpu` on native platforms (Vulkan, Metal, DX12), through a Pure
Rust API. Every entry point returns `Option<_>`: `None` means "this crate
declines, run the CPU operator instead", so a missing adapter, an oversized
tensor or a device error degrades to CPU execution rather than failing. Dispatch
also declines instead of silently producing wrong results on shapes its kernels
don't fully support -- MatMul dropping batch dimensions, Softmax ignoring `axis`,
reduction ignoring `keepdims`, elementwise ops gating on element count instead of
shape equality, and fused Conv discarding the optimizer's activation are all
closed. Every `wgpu` call runs inside an error scope with CPU fallback instead
of panicking on a validation/OOM/device-lost error, dispatch is checked against
the device's own workgroup and buffer-size limits, and blocking readback has a
timeout instead of being able to hang forever.

## Key Types and Functions

### Context and Resource Management

- **`GpuContext`** -- Holds the wgpu `Device` and `Queue`, plus every cached compute pipeline. Created via `GpuContext::try_new()`. The device is requested with the *adapter's* limits, not the WebGPU baseline, so large tensors are not forced onto the CPU by a 128 MiB binding cap the hardware does not have.
- **`GpuBufferPool`** -- Reusable buffer pool that reduces allocation overhead by recycling GPU output buffers. Bounded by both a buffer count and a byte budget (256 MiB by default), with least-recently-used eviction, so idle buffers cannot pin unbounded VRAM.

### Compute Operations

- **`gpu_matmul`** / **`gpu_matmul_tiled`** -- GPU matrix multiplication with optional tiled variant for better cache behavior.
- **`gpu_conv2d`** -- GPU 2D convolution; reuses im2col upload buffers across a chunked dispatch within a byte/iteration budget instead of holding every chunk live at once.
- **`gpu_relu`**, **`gpu_sigmoid`**, **`gpu_gelu`**, **`gpu_silu`**, **`gpu_leaky_relu`** (ONNX default `alpha = 0.01`), **`gpu_leaky_relu_alpha`** (explicit `alpha`, reading the node's own attribute) -- Element-wise activation functions.
- **`gpu_tanh`**, **`gpu_abs`**, **`gpu_neg`**, **`gpu_exp`**, **`gpu_sqrt`**, **`gpu_log`** -- Unary element-wise math.
- **`gpu_add`**, **`gpu_mul`** -- Element-wise binary operations.
- **`gpu_softmax`** -- Softmax along the last axis, via a single workgroup-level-reduction pipeline (previously two passes).
- **`gpu_layer_norm`**, **`gpu_batch_norm`** -- Normalization layers.
- **`gpu_transpose`** -- Tensor axis transposition on GPU.
- **`gpu_reduce_sum`**, **`gpu_reduce_max`**, **`gpu_reduce_min`**, **`gpu_reduce_mean`** -- Reduction operations.

## Usage

```toml
[dependencies]
oxionnx-gpu = "0.1.5"
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

## Platform Notes

- **Native** (Linux/macOS/Windows): Uses `pollster` for synchronous GPU initialization. Backends are selected automatically (Vulkan, Metal, or DX12).
- **WebAssembly**: not supported. `GpuContext::try_new_async()` returns `None` on `wasm32`, so sessions run on the CPU path. The kernels here are synchronous and end in a blocking read-back, which the browser cannot do; before this was gated off, a wasm32 build still uploaded inputs and submitted every dispatch, then discarded the result and recomputed it on the CPU. Browser support needs an `async` variant of every `gpu_*` entry point, not a flag.

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
