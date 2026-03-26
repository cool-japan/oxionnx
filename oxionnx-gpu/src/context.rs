use std::collections::HashMap;
use wgpu;

/// Track whether a tensor's data is on GPU to avoid redundant host-device transfers.
///
/// When executing consecutive GPU-capable operations, data can remain on the GPU
/// between operations without being read back to the CPU.
pub struct GpuTensorTracker {
    /// Map from tensor name to its GPU buffer (if currently on GPU).
    gpu_buffers: HashMap<String, (wgpu::Buffer, u64)>,
}

impl GpuTensorTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            gpu_buffers: HashMap::new(),
        }
    }

    /// Check if a tensor is currently on GPU.
    pub fn is_on_gpu(&self, name: &str) -> bool {
        self.gpu_buffers.contains_key(name)
    }

    /// Store a GPU buffer for a tensor.
    pub fn store(&mut self, name: String, buffer: wgpu::Buffer, size: u64) {
        self.gpu_buffers.insert(name, (buffer, size));
    }

    /// Remove and return a GPU buffer.
    pub fn take(&mut self, name: &str) -> Option<(wgpu::Buffer, u64)> {
        self.gpu_buffers.remove(name)
    }

    /// Clear all tracked GPU buffers.
    pub fn clear(&mut self) {
        self.gpu_buffers.clear();
    }

    /// Number of tensors currently tracked on GPU.
    pub fn count(&self) -> usize {
        self.gpu_buffers.len()
    }
}

impl Default for GpuTensorTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool of reusable wgpu::Buffer allocations to reduce allocation overhead.
pub struct GpuBufferPool {
    /// Available buffers sorted by size (ascending).
    buffers: Vec<(u64, wgpu::Buffer)>,
    /// Maximum buffers to retain.
    max_buffers: usize,
}

impl GpuBufferPool {
    /// Create a new buffer pool that retains up to `max_buffers` idle buffers.
    pub fn new(max_buffers: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_buffers,
        }
    }

    /// Get a buffer of at least `min_size` bytes.
    /// Returns a reused buffer if available (within 2x of requested size), or creates a new one.
    pub fn get_buffer(
        &mut self,
        device: &wgpu::Device,
        min_size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Find smallest buffer that is >= min_size and <= 2*min_size (to avoid waste).
        let max_acceptable = min_size.saturating_mul(2);
        let pos = self
            .buffers
            .iter()
            .position(|(sz, _)| *sz >= min_size && *sz <= max_acceptable);
        if let Some(idx) = pos {
            let (_sz, buf) = self.buffers.remove(idx);
            return buf;
        }
        // No suitable buffer found — create a new one.
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_buf"),
            size: min_size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer, size: u64) {
        if self.buffers.len() >= self.max_buffers {
            // Drop the smallest buffer to make room (the new one might be more useful).
            if let Some(min_idx) = self
                .buffers
                .iter()
                .enumerate()
                .min_by_key(|(_, (sz, _))| *sz)
                .map(|(i, _)| i)
            {
                if self.buffers[min_idx].0 < size {
                    self.buffers.remove(min_idx);
                } else {
                    // New buffer is smaller than all existing ones — just drop it.
                    return;
                }
            }
        }
        // Insert sorted by size.
        let insert_pos = self.buffers.partition_point(|(sz, _)| *sz < size);
        self.buffers.insert(insert_pos, (size, buffer));
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Number of buffers currently available for reuse.
    pub fn available_count(&self) -> usize {
        self.buffers.len()
    }
}

/// Holds the wgpu device and queue, plus cached compute pipelines and buffer pool.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub matmul_bind_group_layout: wgpu::BindGroupLayout,
    // Softmax pipeline (two-pass: pass1 = exp, pass2 = normalize)
    pub softmax_pass1_pipeline: wgpu::ComputePipeline,
    pub softmax_pass2_pipeline: wgpu::ComputePipeline,
    pub softmax_bind_group_layout: wgpu::BindGroupLayout,
    // Element-wise pipelines (relu, sigmoid, gelu)
    pub relu_pipeline: wgpu::ComputePipeline,
    pub sigmoid_pipeline: wgpu::ComputePipeline,
    pub gelu_pipeline: wgpu::ComputePipeline,
    pub elementwise_bind_group_layout: wgpu::BindGroupLayout,
    // Reduction pipelines
    pub reduce_sum_pipeline: wgpu::ComputePipeline,
    pub reduce_max_pipeline: wgpu::ComputePipeline,
    pub reduce_bind_group_layout: wgpu::BindGroupLayout,
    // Tiled matmul pipeline (shared memory, 16x16 tiles)
    pub tiled_matmul_pipeline: wgpu::ComputePipeline,
    pub tiled_matmul_bind_group_layout: wgpu::BindGroupLayout,
    // Buffer pool
    pub pool: std::sync::Mutex<GpuBufferPool>,
    // Tensor location tracker for host-device transfer minimization
    pub tracker: std::sync::Mutex<GpuTensorTracker>,
}

impl GpuContext {
    /// Try to create a GPU context. Returns `None` if no GPU is available.
    ///
    /// On native targets, this blocks using `pollster`. On wasm32, this always
    /// returns `None` — use [`try_new_async`] instead.
    pub fn try_new() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::try_new_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            // wasm32 cannot block; callers must use try_new_async().
            None
        }
    }

    /// Async GPU context creation.
    ///
    /// On native targets this is called internally by [`try_new`].
    /// On wasm32 targets this is the only way to create a GPU context (uses WebGPU).
    pub async fn try_new_async() -> Option<Self> {
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else if cfg!(target_os = "linux") {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxionnx"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .ok()?;

        Self::build_from_device_queue(device, queue)
    }

    /// Build the GPU context (pipelines, pool, tracker) from an already-acquired
    /// device and queue. Shared by both synchronous and asynchronous init paths.
    pub fn build_from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Option<Self> {
        // Create the GEMM shader module and pipeline once.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                // A: storage read
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // B: storage read
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // C: storage read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params: uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Softmax pipelines (two-pass) ---
        let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_shader"),
            source: wgpu::ShaderSource::Wgsl(SOFTMAX_SHADER.into()),
        });
        // Softmax BGL: input(read), output(rw), params(uniform)
        let softmax_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("softmax_bgl"),
            entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
        });
        let softmax_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("softmax_pl"),
            bind_group_layouts: &[Some(&softmax_bgl)],
            immediate_size: 0,
        });
        let softmax_pass1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_pass1"),
            layout: Some(&softmax_pl),
            module: &softmax_shader,
            entry_point: Some("pass1_exp"),
            compilation_options: Default::default(),
            cache: None,
        });
        let softmax_pass2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_pass2"),
            layout: Some(&softmax_pl),
            module: &softmax_shader,
            entry_point: Some("pass2_normalize"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Element-wise pipelines ---
        let ew_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_shader"),
            source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_SHADER.into()),
        });
        // EW BGL: input(read), output(rw), params(uniform)
        let ew_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ew_bgl"),
            entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
        });
        let ew_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ew_pl"),
            bind_group_layouts: &[Some(&ew_bgl)],
            immediate_size: 0,
        });
        let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relu_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("relu"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sigmoid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sigmoid_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("sigmoid"),
            compilation_options: Default::default(),
            cache: None,
        });
        let gelu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gelu_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("gelu"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Reduction pipelines ---
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce_shader"),
            source: wgpu::ShaderSource::Wgsl(REDUCE_SHADER.into()),
        });
        // Reduce BGL: input(read), output(rw), params(uniform)
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reduce_bgl"),
            entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
        });
        let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reduce_pl"),
            bind_group_layouts: &[Some(&reduce_bgl)],
            immediate_size: 0,
        });
        let reduce_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reduce_sum_pipeline"),
                layout: Some(&reduce_pl),
                module: &reduce_shader,
                entry_point: Some("reduce_sum"),
                compilation_options: Default::default(),
                cache: None,
            });
        let reduce_max_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reduce_max_pipeline"),
                layout: Some(&reduce_pl),
                module: &reduce_shader,
                entry_point: Some("reduce_max"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Tiled MatMul pipeline (16x16 shared-memory tiles) ---
        let tiled_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tiled_matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(TILED_MATMUL_SHADER.into()),
        });
        let tiled_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tiled_matmul_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let tiled_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tiled_matmul_pl"),
            bind_group_layouts: &[Some(&tiled_bgl)],
            immediate_size: 0,
        });
        let tiled_matmul_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tiled_matmul_pipeline"),
                layout: Some(&tiled_pl),
                module: &tiled_shader,
                entry_point: Some("tiled_matmul"),
                compilation_options: Default::default(),
                cache: None,
            });

        Some(Self {
            device,
            queue,
            matmul_pipeline: pipeline,
            matmul_bind_group_layout: bind_group_layout,
            softmax_pass1_pipeline: softmax_pass1,
            softmax_pass2_pipeline: softmax_pass2,
            softmax_bind_group_layout: softmax_bgl,
            relu_pipeline,
            sigmoid_pipeline,
            gelu_pipeline,
            elementwise_bind_group_layout: ew_bgl,
            reduce_sum_pipeline,
            reduce_max_pipeline,
            reduce_bind_group_layout: reduce_bgl,
            tiled_matmul_pipeline,
            tiled_matmul_bind_group_layout: tiled_bgl,
            pool: std::sync::Mutex::new(GpuBufferPool::new(64)),
            tracker: std::sync::Mutex::new(GpuTensorTracker::new()),
        })
    }
}

// --- Helper functions for bind group layout entries ---

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
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

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
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

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
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

const MATMUL_SHADER: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.M || col >= params.N) { return; }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.K; i++) {
        sum += A[row * params.K + i] * B[i * params.N + col];
    }
    C[row * params.N + col] = sum;
}
"#;

/// Softmax WGSL shader — two-pass approach.
///
/// Pass 1 (`pass1_exp`): For each row, find max, compute exp(x - max), store result.
/// Pass 2 (`pass2_normalize`): For each row, compute sum of exps, divide each element.
///
/// Layout: input[num_rows * row_len], output[num_rows * row_len], params = { num_rows, row_len }.
/// Each workgroup thread handles one row.
const SOFTMAX_SHADER: &str = r#"
struct Params {
    num_rows: u32,
    row_len: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pass1_exp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.num_rows) { return; }
    let base = row * params.row_len;

    // Find row max for numerical stability.
    var max_val: f32 = input[base];
    for (var i: u32 = 1u; i < params.row_len; i++) {
        let v = input[base + i];
        if (v > max_val) { max_val = v; }
    }

    // Compute exp(x - max).
    for (var i: u32 = 0u; i < params.row_len; i++) {
        output[base + i] = exp(input[base + i] - max_val);
    }
}

@compute @workgroup_size(64)
fn pass2_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.num_rows) { return; }
    let base = row * params.row_len;

    // Sum of exps.
    var sum_val: f32 = 0.0;
    for (var i: u32 = 0u; i < params.row_len; i++) {
        sum_val += output[base + i];
    }

    // Normalize.
    let inv_sum = 1.0 / sum_val;
    for (var i: u32 = 0u; i < params.row_len; i++) {
        output[base + i] = output[base + i] * inv_sum;
    }
}
"#;

/// Element-wise WGSL shader — relu, sigmoid, gelu.
///
/// Layout: input[len], output[len], params = { len, _pad }.
const ELEMENTWISE_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }
    output[idx] = max(input[idx], 0.0);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }
    let x = input[idx];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = 0.7978845608; // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"#;

/// Reduction WGSL shader — reduce_sum and reduce_max along an axis.
///
/// The input is conceptualized as [outer_size, axis_len, inner_size].
/// Each thread handles one (outer, inner) pair, reducing over axis_len.
/// Layout: input[total], output[outer_size * inner_size], params = { outer_size, axis_len, inner_size }.
const REDUCE_SHADER: &str = r#"
struct Params {
    outer_size: u32,
    axis_len: u32,
    inner_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let total_out = params.outer_size * params.inner_size;
    if (flat_idx >= total_out) { return; }

    let outer = flat_idx / params.inner_size;
    let inner = flat_idx % params.inner_size;
    let in_base = outer * params.axis_len * params.inner_size + inner;

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.axis_len; i++) {
        acc += input[in_base + i * params.inner_size];
    }
    output[flat_idx] = acc;
}

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let total_out = params.outer_size * params.inner_size;
    if (flat_idx >= total_out) { return; }

    let outer = flat_idx / params.inner_size;
    let inner = flat_idx % params.inner_size;
    let in_base = outer * params.axis_len * params.inner_size + inner;

    var acc: f32 = input[in_base];
    for (var i: u32 = 1u; i < params.axis_len; i++) {
        let v = input[in_base + i * params.inner_size];
        if (v > acc) { acc = v; }
    }
    output[flat_idx] = acc;
}
"#;

/// Tiled matrix multiply WGSL shader using workgroup shared memory.
///
/// Uses 16x16 tiles loaded into shared memory for improved cache locality.
/// Each workgroup computes a 16x16 tile of the output matrix C by iterating
/// over tiles along the K dimension, loading A and B tiles into shared memory,
/// and accumulating partial dot products.
const TILED_MATMUL_SHADER: &str = r#"
struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

const TILE_SIZE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn tiled_matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;
    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    var sum: f32 = 0.0;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            tile_a[local_row][local_col] = a[row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            tile_b[local_row][local_col] = b[b_row * N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row][i] * tile_b[i][local_col];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
"#;
