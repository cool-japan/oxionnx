# oxionnx-gpu

GPU compute backend for OxiONNX -- wgpu-based MatMul and Conv2D acceleration.

This crate provides GPU-accelerated implementations of performance-critical ONNX
operators using `wgpu`. It works on native platforms (Vulkan, Metal, DX12) and
WebGPU in the browser, all through a single Pure Rust API.

## Key Types and Functions

### Context and Resource Management

- **`GpuContext`** -- Holds the wgpu `Device` and `Queue`. Created via `GpuContext::try_new()` on native or async initialization on wasm32.
- **`GpuBufferPool`** -- Reusable buffer pool that reduces allocation overhead by recycling GPU buffers.
- **`GpuTensorTracker`** -- Tracks which tensors currently reside on GPU to avoid redundant host-device transfers between consecutive GPU operations.

### Compute Operations

- **`gpu_matmul`** / **`gpu_matmul_tiled`** -- GPU matrix multiplication with optional tiled variant for better cache behavior.
- **`gpu_conv2d`** -- GPU 2D convolution.
- **`gpu_relu`**, **`gpu_sigmoid`**, **`gpu_gelu`** -- Element-wise activation functions.
- **`gpu_softmax`** -- Softmax along the last axis.
- **`gpu_reduce_sum`**, **`gpu_reduce_max`** -- Reduction operations.

## Usage

```toml
[dependencies]
oxionnx-gpu = "0.1.0"
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
- **WebAssembly**: `pollster` is not available; use the async API (`GpuContext::try_new_async`) and the `webgpu` wgpu feature.

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
