//! GpuContext — holds the wgpu device/queue and all cached compute pipelines.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::functions::{
    bgl_storage_ro, bgl_storage_rw, bgl_uniform, BATCH_NORM_SHADER, BINARY_ELEMENTWISE_SHADER,
    ELEMENTWISE_SHADER, LAYER_NORM_SHADER, MATMUL_SHADER, REDUCE_SHADER, SOFTMAX_SHADER,
    TILED_MATMUL_SHADER, TRANSPOSE_SHADER,
};
use super::tracker_pool::GpuBufferPool;
use crate::device_guard::GpuLimits;

/// Holds the wgpu device and queue, plus cached compute pipelines and buffer pool.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Device limits that gate every dispatch (cached to keep hot paths cheap).
    pub limits: GpuLimits,
    /// Set once the device has reported an error we could not recover from.
    /// While set, every `gpu_*` entry point declines so work goes to the CPU.
    degraded: Arc<AtomicBool>,
    /// First error observed on this device, kept for diagnostics.
    last_error: Arc<Mutex<Option<String>>>,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub matmul_bind_group_layout: wgpu::BindGroupLayout,
    // Softmax pipeline (one workgroup per row, fused max/exp/sum/normalize)
    pub softmax_pipeline: wgpu::ComputePipeline,
    pub softmax_bind_group_layout: wgpu::BindGroupLayout,
    // Element-wise pipelines (relu, sigmoid, gelu, tanh, exp, sqrt, abs, neg, log, silu, leaky_relu)
    pub relu_pipeline: wgpu::ComputePipeline,
    pub sigmoid_pipeline: wgpu::ComputePipeline,
    pub gelu_pipeline: wgpu::ComputePipeline,
    pub tanh_pipeline: wgpu::ComputePipeline,
    pub exp_pipeline: wgpu::ComputePipeline,
    pub sqrt_pipeline: wgpu::ComputePipeline,
    pub abs_pipeline: wgpu::ComputePipeline,
    pub neg_pipeline: wgpu::ComputePipeline,
    pub log_pipeline: wgpu::ComputePipeline,
    pub silu_pipeline: wgpu::ComputePipeline,
    pub leaky_relu_pipeline: wgpu::ComputePipeline,
    pub elementwise_bind_group_layout: wgpu::BindGroupLayout,
    // Binary element-wise pipelines (add, mul)
    pub add_pipeline: wgpu::ComputePipeline,
    pub mul_pipeline: wgpu::ComputePipeline,
    pub binary_elementwise_bind_group_layout: wgpu::BindGroupLayout,
    // Reduction pipelines
    pub reduce_sum_pipeline: wgpu::ComputePipeline,
    pub reduce_max_pipeline: wgpu::ComputePipeline,
    pub reduce_min_pipeline: wgpu::ComputePipeline,
    pub reduce_mean_pipeline: wgpu::ComputePipeline,
    pub reduce_bind_group_layout: wgpu::BindGroupLayout,
    // Tiled matmul pipeline (shared memory, 16x16 tiles)
    pub tiled_matmul_pipeline: wgpu::ComputePipeline,
    pub tiled_matmul_bind_group_layout: wgpu::BindGroupLayout,
    // LayerNorm pipeline (shared-memory parallel reduction)
    pub layer_norm_pipeline: wgpu::ComputePipeline,
    pub layer_norm_bind_group_layout: wgpu::BindGroupLayout,
    // BatchNorm pipeline (inference-mode per-channel normalization)
    pub batch_norm_pipeline: wgpu::ComputePipeline,
    pub batch_norm_bind_group_layout: wgpu::BindGroupLayout,
    // Transpose pipeline (general permutation)
    pub transpose_pipeline: wgpu::ComputePipeline,
    pub transpose_bind_group_layout: wgpu::BindGroupLayout,
    /// Byte-bounded, LRU-evicting pool of idle output buffers.
    pub pool: std::sync::Mutex<GpuBufferPool>,
}

impl GpuContext {
    /// Try to create a GPU context. Returns `None` if no GPU is available.
    ///
    /// On native targets, this blocks using `pollster`. On wasm32 it returns
    /// `None`, as does [`Self::try_new_async`] — see that method for why the
    /// browser backend is gated off rather than silently discarding results.
    pub fn try_new() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::try_new_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            // wasm32 cannot block, and the async path declines too (a7-10).
            None
        }
    }

    /// Async GPU context creation.
    ///
    /// On native targets this is called internally by [`Self::try_new`].
    ///
    /// # wasm32 / WebGPU
    ///
    /// [a7-10] This returns `None` on wasm32. Every kernel in this crate is a
    /// *synchronous* `Option`-returning function that ends in a blocking
    /// read-back, and blocking is impossible in the browser: `read_back`
    /// (device_guard.rs) therefore cannot complete a `map_async` there. Before
    /// this gate, a wasm32 context still uploaded inputs, allocated buffers,
    /// encoded a pass and called `queue.submit` for every node, then discarded
    /// the result and returned `None` — the CPU recomputed the same op, so the
    /// backend was pure overhead that could never produce a value.
    ///
    /// Declining at context creation makes that honest: no dispatch is ever
    /// encoded, no buffer is ever allocated, and the session runs on the CPU
    /// path directly. Restoring browser acceleration requires an `async`
    /// variant of every `gpu_*` entry point (so the caller can await the
    /// `map_async` future) plus a `wasm-bindgen-futures` bridge — a public API
    /// split, not a local fix.
    pub async fn try_new_async() -> Option<Self> {
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::acquire_device().await
        }
    }

    /// Request an adapter and device, then build the context.
    #[cfg(not(target_arch = "wasm32"))]
    async fn acquire_device() -> Option<Self> {
        let backends = if cfg!(target_os = "linux") {
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

        // Ask for what the adapter actually supports: the defaults cap storage
        // bindings at 128 MiB and buffers at 256 MiB, which forces perfectly
        // ordinary tensors onto the CPU. Fall back to the conservative defaults
        // if the driver refuses the full set.
        let device_queue = match adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxionnx"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
        {
            Ok(pair) => pair,
            Err(_) => adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("oxionnx"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                })
                .await
                .ok()?,
        };
        let (device, queue) = device_queue;

        Self::build_from_device_queue(device, queue)
    }

    /// True once the device has reported an unrecoverable error.
    ///
    /// Every GPU entry point checks this first and declines while it is set, so
    /// a driver reset or an out-of-memory condition degrades the session to CPU
    /// execution instead of aborting the process.
    #[inline]
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded.load(Ordering::Relaxed)
    }

    /// Record an unrecoverable device error and degrade to CPU execution.
    ///
    /// Native-only: every caller sits on a blocking path (`read_back`,
    /// `ErrorScope::finish`) that wasm32 does not compile.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn mark_degraded(&self, reason: impl Into<String>) {
        self.degraded.store(true, Ordering::Relaxed);
        if let Ok(mut slot) = self.last_error.lock() {
            if slot.is_none() {
                *slot = Some(reason.into());
            }
        }
    }

    /// The first device error observed, if any.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|slot| slot.clone())
    }

    /// Build the GPU context (pipelines, pool, tracker) from an already-acquired
    /// device and queue. Shared by both synchronous and asynchronous init paths.
    pub fn build_from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Option<Self> {
        // wgpu turns unhandled device errors into panics by default. This crate
        // is contractually allowed to decline any node, so route them into a
        // flag instead and let every dispatch fall back to the CPU.
        let degraded = Arc::new(AtomicBool::new(false));
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        {
            let degraded = Arc::clone(&degraded);
            let last_error = Arc::clone(&last_error);
            device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
                degraded.store(true, Ordering::Relaxed);
                if let Ok(mut slot) = last_error.lock() {
                    if slot.is_none() {
                        *slot = Some(err.to_string());
                    }
                }
            }));
        }
        let limits = GpuLimits::from_device(&device);

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
        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_pipeline"),
            layout: Some(&softmax_pl),
            module: &softmax_shader,
            entry_point: Some("softmax_rows"),
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
        let tanh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tanh_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("op_tanh"),
            compilation_options: Default::default(),
            cache: None,
        });
        let exp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("exp_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("op_exp"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sqrt_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sqrt_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("op_sqrt"),
            compilation_options: Default::default(),
            cache: None,
        });
        let abs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("abs_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("op_abs"),
            compilation_options: Default::default(),
            cache: None,
        });
        let neg_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("neg_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("op_neg"),
            compilation_options: Default::default(),
            cache: None,
        });
        let log_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("log_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("op_log"),
            compilation_options: Default::default(),
            cache: None,
        });
        let silu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("silu_pipeline"),
            layout: Some(&ew_pl),
            module: &ew_shader,
            entry_point: Some("silu"),
            compilation_options: Default::default(),
            cache: None,
        });
        let leaky_relu_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("leaky_relu_pipeline"),
                layout: Some(&ew_pl),
                module: &ew_shader,
                entry_point: Some("leaky_relu"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Binary element-wise pipelines ---
        let binary_ew_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binary_elementwise_shader"),
            source: wgpu::ShaderSource::Wgsl(BINARY_ELEMENTWISE_SHADER.into()),
        });
        // Binary EW BGL: a(read), b(read), output(rw), params(uniform)
        let binary_ew_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_ew_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });
        let binary_ew_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("binary_ew_pl"),
            bind_group_layouts: &[Some(&binary_ew_bgl)],
            immediate_size: 0,
        });
        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_pipeline"),
            layout: Some(&binary_ew_pl),
            module: &binary_ew_shader,
            entry_point: Some("op_add"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mul_pipeline"),
            layout: Some(&binary_ew_pl),
            module: &binary_ew_shader,
            entry_point: Some("op_mul"),
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
        let reduce_min_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reduce_min_pipeline"),
                layout: Some(&reduce_pl),
                module: &reduce_shader,
                entry_point: Some("reduce_min"),
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

        // --- ReduceMean pipeline (reuses reduce shader + BGL) ---
        let reduce_mean_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reduce_mean_pipeline"),
                layout: Some(&reduce_pl),
                module: &reduce_shader,
                entry_point: Some("reduce_mean"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- LayerNorm pipeline ---
        let ln_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(LAYER_NORM_SHADER.into()),
        });
        let ln_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_norm_bgl"),
            entries: &[
                bgl_storage_ro(0), // input
                bgl_storage_ro(1), // scale
                bgl_storage_ro(2), // bias
                bgl_storage_rw(3), // output
                bgl_uniform(4),    // params
            ],
        });
        let ln_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layer_norm_pl"),
            bind_group_layouts: &[Some(&ln_bgl)],
            immediate_size: 0,
        });
        let layer_norm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("layer_norm_pipeline"),
                layout: Some(&ln_pl),
                module: &ln_shader,
                entry_point: Some("layer_norm"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- BatchNorm pipeline (inference) ---
        let bn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_NORM_SHADER.into()),
        });
        let bn_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_norm_bgl"),
            entries: &[
                bgl_storage_ro(0), // input
                bgl_storage_ro(1), // scale
                bgl_storage_ro(2), // bias
                bgl_storage_ro(3), // mean
                bgl_storage_ro(4), // variance
                bgl_storage_rw(5), // output
                bgl_uniform(6),    // params
            ],
        });
        let bn_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("batch_norm_pl"),
            bind_group_layouts: &[Some(&bn_bgl)],
            immediate_size: 0,
        });
        let batch_norm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("batch_norm_pipeline"),
                layout: Some(&bn_pl),
                module: &bn_shader,
                entry_point: Some("batch_norm"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Transpose pipeline ---
        let tr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transpose_shader"),
            source: wgpu::ShaderSource::Wgsl(TRANSPOSE_SHADER.into()),
        });
        let tr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transpose_bgl"),
            entries: &[
                bgl_storage_ro(0), // input
                bgl_storage_rw(1), // output
                bgl_storage_ro(2), // perm_data (input_strides, output_strides, perm)
                bgl_uniform(3),    // params
            ],
        });
        let tr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transpose_pl"),
            bind_group_layouts: &[Some(&tr_bgl)],
            immediate_size: 0,
        });
        let transpose_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transpose_pipeline"),
            layout: Some(&tr_pl),
            module: &tr_shader,
            entry_point: Some("transpose_op"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            limits,
            degraded,
            last_error,
            matmul_pipeline: pipeline,
            matmul_bind_group_layout: bind_group_layout,
            softmax_pipeline,
            softmax_bind_group_layout: softmax_bgl,
            relu_pipeline,
            sigmoid_pipeline,
            gelu_pipeline,
            tanh_pipeline,
            exp_pipeline,
            sqrt_pipeline,
            abs_pipeline,
            neg_pipeline,
            log_pipeline,
            silu_pipeline,
            leaky_relu_pipeline,
            elementwise_bind_group_layout: ew_bgl,
            add_pipeline,
            mul_pipeline,
            binary_elementwise_bind_group_layout: binary_ew_bgl,
            reduce_sum_pipeline,
            reduce_max_pipeline,
            reduce_min_pipeline,
            reduce_bind_group_layout: reduce_bgl,
            tiled_matmul_pipeline,
            tiled_matmul_bind_group_layout: tiled_bgl,
            reduce_mean_pipeline,
            layer_norm_pipeline,
            layer_norm_bind_group_layout: ln_bgl,
            batch_norm_pipeline,
            batch_norm_bind_group_layout: bn_bgl,
            transpose_pipeline,
            transpose_bind_group_layout: tr_bgl,
            pool: std::sync::Mutex::new(GpuBufferPool::new(64)),
        })
    }
}

/// Quiesce the device before its fields — the underlying `wgpu::Device` /
/// `Queue` / `Instance` and every cached pipeline — run their own automatic
/// drops.
///
/// No `gpu_*` entry point in this crate ever leaves a submission in flight
/// across a call boundary: every dispatch reads its result back (or times
/// out and marks the context degraded) before returning. So in the common
/// case — a context that was actually used for compute — this `poll` finds
/// nothing outstanding and returns immediately.
///
/// It exists for the *uncommon* case: a session built with a live GPU
/// context whose graph never actually dispatched a single node to it —
/// `OpPlacement::CpuOnly` (the crate-wide default) or a graph whose tensors
/// all stayed below the dispatch size floor — where this `drop` may be the
/// *first* explicit synchronization point with the driver since context
/// creation. Native GPU drivers commonly run asynchronous background work
/// per device/instance after creation (shader analysis and pipeline-cache
/// housekeeping); this crate's own pipeline construction above creates over
/// twenty compute pipelines up front, and on this crate's own test hardware
/// that visibly spawns extra driver worker threads (seen under a debugger as
/// `[vkrt] Analysis` / `[vkcf] Analysis` / `[vkps] Update`). Tearing the
/// device down with no synchronization point first gives that work no
/// defined place to finish *before* `vkDestroyDevice` / `vkDestroyInstance`
/// run — and unlike a submitted compute dispatch, nothing else in this
/// crate ever waits on it. A bounded, explicit wait here is cheap when there
/// is nothing to wait for, and is the documented, supported way to bring a
/// `wgpu::Device` to a quiescent point before the handles wrapping it go
/// away.
///
/// Best-effort only, deliberately: a poll error or timeout does not change
/// the fact that every field below is about to run its own drop regardless
/// — `Drop::drop` cannot return a `Result`, and this device is being
/// destroyed either way, successful quiescence or not.
impl Drop for GpuContext {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });
        }
    }
}
