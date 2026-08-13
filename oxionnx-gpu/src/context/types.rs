//! GpuContext — holds the wgpu device/queue and all cached compute pipelines.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::activation::{DeviceTensor, TensorSource};
use super::budget::{GpuMemoryBudget, TrackedBuffer, DEFAULT_LIVE_BYTE_BUDGET};
use super::functions::{
    bgl_storage_ro, bgl_storage_rw, bgl_uniform, BATCH_NORM_SHADER, BINARY_ELEMENTWISE_SHADER,
    ELEMENTWISE_SHADER, LAYER_NORM_SHADER, MATMUL_SHADER, REDUCE_SHADER, SOFTMAX_SHADER,
    TILED_MATMUL_SHADER, TRANSPOSE_SHADER,
};
use super::init_error::{GpuInitDiagnostic, GpuInitError};
use super::pipeline_cache::PipelineCache;
use super::resident::{Lookup, OperandBuffer, ResidentBuffers, ResidentCounters};
use super::tracker_pool::{GpuBufferPool, DEFAULT_POOL_BYTE_BUDGET};
use super::tuning::{GpuPerfClass, GpuTuning};
use super::weight_format::{F16Compute, WeightBytes, WeightFormat};
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
    /// Ceiling on the device memory this context may have allocated at once,
    /// shared with [`Self::pool`]. Every buffer allocated through this context
    /// is counted here and released when it is destroyed.
    budget: Arc<GpuMemoryBudget>,
    /// Buffers for operands whose bytes never change, kept for this context's
    /// lifetime instead of being rebuilt per dispatch. See `super::resident`.
    resident: ResidentBuffers,
    /// Cumulative bytes this context has handed to the driver through
    /// [`Self::upload_buffer`] — every host→device transfer it makes.
    ///
    /// Monotonic and never reset: it exists so a caller can difference two
    /// snapshots and see what one run actually uploaded.
    upload_bytes: AtomicU64,
    /// Whether kernels may leave their results in device buffers for the next
    /// kernel to consume in place. See [`Self::set_activation_residency`].
    activation_residency: AtomicBool,
    /// [w2-f16] Whether the two compute-dominant kernels may take their
    /// half-precision path, and whether this device can at all. Off by default
    /// — see [`Self::set_f16_compute`].
    f16_compute: F16Compute,
    /// The size and shape floors every kernel in this crate declines below,
    /// derived once from this adapter's own [`wgpu::AdapterInfo`]. See
    /// [`super::tuning`] for why these are not compile-time constants.
    tuning: GpuTuning,
    /// \[w5\] Compute pipelines this context's device has compiled, for the
    /// kernels that build their own rather than taking one of the fields above.
    ///
    /// A field rather than a thread-local keyed on `self.device`, because
    /// `wgpu::Device` equality is a per-`Instance` id and this crate creates one
    /// `Instance` per context — so two contexts' devices compare *equal* and a
    /// device-keyed cache serves the second one the first's pipelines. See
    /// [`super::pipeline_cache`] for the crash that produced and why no key can
    /// fix it.
    pipelines: PipelineCache,
}

impl GpuContext {
    /// Try to create a GPU context. Returns `None` if no GPU is available.
    ///
    /// On native targets, this blocks using `pollster`. On wasm32 it returns
    /// `None` — adapter acquisition is inherently a promise in the browser and
    /// nothing there may block a thread waiting on one. Browser callers use
    /// [`Self::try_new_async`], which is fully supported.
    pub fn try_new() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::try_new_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            // A browser thread cannot block on `requestAdapter`'s promise.
            None
        }
    }

    /// Async GPU context creation — the real constructor on every target.
    ///
    /// On native targets this is what [`Self::try_new`] blocks on. On wasm32
    /// this is the *only* way in, and it acquires a WebGPU adapter through
    /// [`wgpu::Backends::BROWSER_WEBGPU`].
    ///
    /// # wasm32 / WebGPU
    ///
    /// This used to return `None` unconditionally [a7-10], because every kernel
    /// in this crate was a synchronous function ending in a blocking read-back
    /// that the browser can never complete — so a wasm32 context uploaded
    /// inputs, allocated buffers, encoded a pass and submitted it for every
    /// node, then threw the result away. Declining up front at least made that
    /// honest.
    ///
    /// The kernels are now written as `async fn`s whose read-back awaits the
    /// `mapAsync` promise (see `device_guard::read_back_web`), with the
    /// synchronous entry points as `pollster::block_on` wrappers that decline
    /// on wasm32 instead of hanging. So the browser path produces real values
    /// and the gate is gone. Callers must use the `*_async` entry points there;
    /// the synchronous ones still return `None`, which keeps a caller that
    /// forgets on the correct-but-slow CPU path rather than deadlocking the
    /// page.
    pub async fn try_new_async() -> Option<Self> {
        Self::acquire_device().await
    }

    /// Which backends to ask for on this target.
    ///
    /// `BROWSER_WEBGPU` on wasm32: it is the only backend wgpu's web build can
    /// serve, and naming it explicitly means a browser without WebGPU (or with
    /// it disabled) fails at `request_adapter` — a clean `None` and a CPU
    /// fallback — rather than wgpu trying to reach for a native backend that
    /// does not exist in a page.
    fn requested_backends() -> wgpu::Backends {
        #[cfg(target_arch = "wasm32")]
        {
            wgpu::Backends::BROWSER_WEBGPU
        }
        #[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
        {
            wgpu::Backends::VULKAN
        }
        #[cfg(all(not(target_arch = "wasm32"), not(target_os = "linux")))]
        {
            wgpu::Backends::all()
        }
    }

    /// Async GPU context creation that **explains** its failure.
    ///
    /// [`Self::try_new_async`] is this with the explanation discarded. Prefer
    /// this one anywhere the answer reaches a human: the most common reason a
    /// Linux server with a perfectly good GPU produces `None` is a missing
    /// Vulkan loader package, and that is not something an operator can guess
    /// from a bare `None`. See [`crate::context::init_error`].
    pub async fn try_new_diagnosed_async() -> Result<Self, GpuInitError> {
        Self::acquire_device_diagnosed().await
    }

    /// Blocking form of [`Self::try_new_diagnosed_async`].
    ///
    /// Returns [`GpuInitError::BlockingUnavailable`] on wasm32, for the same
    /// reason [`Self::try_new`] returns `None` there.
    pub fn try_new_diagnosed() -> Result<Self, GpuInitError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::try_new_diagnosed_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(GpuInitError::BlockingUnavailable)
        }
    }

    /// Request an adapter and device, then build the context.
    async fn acquire_device() -> Option<Self> {
        Self::acquire_device_diagnosed().await.ok()
    }

    /// The real constructor — see [`Self::try_new_diagnosed_async`].
    async fn acquire_device_diagnosed() -> Result<Self, GpuInitError> {
        let backends = Self::requested_backends();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => {
                // Nothing on the requested backends. Before answering "no GPU",
                // ask every *other* backend whether it can see one: a GPU that
                // OpenGL reports and Vulkan does not is a missing Vulkan
                // loader, not missing hardware, and that distinction is the
                // difference between "install one package" and "get a
                // different machine". Only reached on the failure path.
                return Err(GpuInitError::NoAdapter {
                    backends: format!("{backends:?}"),
                    diagnostic: GpuInitDiagnostic::probe(backends).await,
                });
            }
        };

        // [w2-f16] Ask for half-precision shader support *only where the adapter
        // already reports it*. Intersecting rather than naming the feature
        // outright is the whole safety property: `request_device` fails
        // outright when asked for a feature the adapter lacks, so an
        // unconditional `Features::SHADER_F16` would turn every GPU without it
        // — which on the web is any browser that has not enabled `shader-f16`
        // — from "runs f32 kernels" into "has no device at all". The
        // intersection is empty there, which is exactly the descriptor this
        // crate has always sent.
        //
        // Native and wasm32 alike: `Adapter::features()` is served by the
        // browser's `GPUAdapter.features` set on the web, so this is one code
        // path, not two.
        let optional = adapter.features() & wgpu::Features::SHADER_F16;

        // Ask for what the adapter actually supports: the defaults cap storage
        // bindings at 128 MiB and buffers at 256 MiB, which forces perfectly
        // ordinary tensors onto the CPU. Fall back to the conservative defaults
        // if the driver refuses the full set.
        //
        // The feature set is the *outer* fallback and the limits the inner one,
        // so a driver that refuses `SHADER_F16` for a reason its adapter did not
        // advertise still ends up with the exact device this crate requested
        // before half precision existed, rather than with no device at all.
        // Half-precision compute then reports itself unsupported, which is the
        // honest answer.
        let mut device_queue = None;
        for features in [optional, wgpu::Features::empty()] {
            for limits in [adapter.limits(), wgpu::Limits::default()] {
                if let Ok(pair) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("oxionnx"),
                        required_features: features,
                        required_limits: limits,
                        ..Default::default()
                    })
                    .await
                {
                    device_queue = Some(pair);
                    break;
                }
            }
            if device_queue.is_some() {
                break;
            }
            if optional.is_empty() {
                // The two passes would be identical; do not ask twice.
                break;
            }
        }
        let info = adapter.get_info();
        let Some((device, queue)) = device_queue else {
            return Err(GpuInitError::NoDevice {
                adapter: format!("{} ({:?})", info.name, info.backend),
            });
        };

        let mut ctx =
            Self::build_from_device_queue(device, queue).ok_or(GpuInitError::PipelineBuild)?;
        // The adapter's own classification, which `build_from_device_queue`
        // cannot know: it is `pub` and takes an already-acquired device, with no
        // `AdapterInfo` attached. See `super::tuning` for what it selects.
        ctx.set_tuning(GpuTuning::from_adapter_info(&info));
        Ok(ctx)
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
    /// Compiled on every target: `ErrorScope::finish_async` reports browser
    /// validation errors through here too, now that popping a WebGPU error
    /// scope is awaited rather than discarded.
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
        // Captured before `device` is moved into the struct literal below.
        let device_features = device.features();
        let budget = GpuMemoryBudget::new(DEFAULT_LIVE_BYTE_BUDGET);

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
            pool: std::sync::Mutex::new(GpuBufferPool::with_live_budget(
                64,
                DEFAULT_POOL_BYTE_BUDGET,
                Arc::clone(&budget),
            )),
            budget,
            resident: ResidentBuffers::default(),
            upload_bytes: AtomicU64::new(0),
            activation_residency: AtomicBool::new(true),
            // [w2-f16] Read off the *device*, not off whatever `acquire_device`
            // asked for. This constructor is `pub`, so a caller may hand in a
            // device this crate never requested — one built against an existing
            // WebGPU adapter, say — and threading a flag down from
            // `acquire_device` would report that device's capability wrongly.
            // The device's own feature set is the only answer correct for every
            // entry path.
            f16_compute: F16Compute::new(device_features.contains(wgpu::Features::SHADER_F16)),
            // No `AdapterInfo` reaches this constructor — it is `pub` and takes
            // an already-acquired device/queue pair — so the caller's device
            // gets the `Unknown` class. `acquire_device` overwrites this with
            // the real classification immediately afterwards, and any other
            // caller can do the same through `set_tuning`.
            tuning: GpuTuning::default(),
            // [w5] Empty: the kernels that build their own pipelines fill it on
            // first dispatch. It belongs to this context and dies with it.
            pipelines: PipelineCache::default(),
        })
    }

    /// \[w5\] The pipelines this context's device has compiled.
    ///
    /// The kernels in `crate::shaders` that construct their own pipeline — the
    /// standalone batch described in `shaders::kernel_support`, plus `conv2d`
    /// and `gemm` — memoize through here instead of through a thread-local, so
    /// a compiled pipeline can never outlive, or be found by, a device other
    /// than the one that built it. See [`super::pipeline_cache`].
    #[inline]
    pub(crate) fn pipelines(&self) -> &PipelineCache {
        &self.pipelines
    }

    /// The size and shape floors this context's kernels decline below.
    ///
    /// Every `gpu_*` entry point reads its threshold from here rather than from
    /// a compile-time constant; see [`super::tuning`] for what each field means
    /// and how the numbers were arrived at.
    #[inline]
    #[must_use]
    pub fn tuning(&self) -> &GpuTuning {
        &self.tuning
    }

    /// Which performance class this context's adapter was classified as.
    #[inline]
    #[must_use]
    pub fn perf_class(&self) -> GpuPerfClass {
        self.tuning.class
    }

    /// Replace this context's dispatch thresholds wholesale.
    ///
    /// Two callers are expected, and no others:
    ///
    /// * an embedder that has measured its own target and wants to say so, and
    /// * a *kernel* test, which wants the numerics of one shader exercised at a
    ///   size a real workload would decline. Installing [`GpuTuning::PARITY`]
    ///   separates "does this kernel compute the right values" from "is this
    ///   dispatch worth making", which is the whole reason the second question
    ///   moved out of the shader modules and into [`GpuTuning`].
    ///
    /// `&mut self` deliberately: the thresholds are read on every dispatch, so
    /// they must stay a plain field rather than becoming a lock or an atomic.
    /// Configure the context before sharing it.
    pub fn set_tuning(&mut self, tuning: GpuTuning) {
        self.tuning = tuning;
    }
}

// ========================================================================
// Buffer allocation
// ========================================================================

/// Every buffer this crate creates comes from `alloc_buffer`, `staging_buffer`,
/// `pooled_buffer` or `upload_buffer` — nothing calls `Device::create_buffer`
/// outside `TrackedBuffer::create`, which all four go through.
///
/// They exist so two properties hold for the whole crate rather than for each
/// kernel that remembered them: the bytes are reserved against
/// [`GpuMemoryBudget`] before the device is asked for anything, and the handle
/// that comes back destroys itself when it goes out of scope — including on the
/// `?` paths a dispatch is full of. `budget` (module `super::budget`) has the
/// full reasoning; the short version is that dropping a `wgpu::Buffer` frees no
/// device memory at all in a browser.
///
/// `operand_buffer` is the fifth and last entry point, and it is not an
/// exception to any of that: it either returns a buffer the residency cache
/// already holds, or calls `upload_buffer` and keeps the result. Nothing there
/// reaches the device on its own.
impl GpuContext {
    /// Bytes currently allocated on the device by this context (in use plus
    /// pooled).
    #[must_use]
    pub fn live_gpu_bytes(&self) -> u64 {
        self.budget.live_bytes()
    }

    /// The live-byte ceiling. Defaults to [`DEFAULT_LIVE_BYTE_BUDGET`].
    #[must_use]
    pub fn gpu_byte_budget(&self) -> u64 {
        self.budget.limit()
    }

    /// Move the live-byte ceiling — for a page that knows it must share the GPU
    /// process with something else, or for a test that wants to observe the
    /// decline path.
    pub fn set_gpu_byte_budget(&self, bytes: u64) {
        self.budget.set_limit(bytes);
    }

    /// Whether a node needing `byte_sizes` can be dispatched at all.
    ///
    /// Kernels call this once, before their first allocation, so a node that
    /// cannot fit declines as a whole instead of allocating half its operands
    /// and then declining. It is advisory — the reservation inside each
    /// allocation is the authority — but it keeps the failure cheap.
    pub(crate) fn budget_admits(&self, byte_sizes: &[u64]) -> bool {
        let total = byte_sizes
            .iter()
            .fold(0u64, |acc, &bytes| acc.saturating_add(bytes));
        if self.budget.admits(total) {
            return true;
        }
        // Idle pooled buffers are reclaimable: nothing is reading them, and
        // this node fitting matters more than the next one being cheap to
        // allocate. Declining while holding memory nobody is using would be a
        // capability loss with no upside.
        if let Ok(mut pool) = self.pool.lock() {
            pool.reclaim_for(total);
        }
        self.budget.admits(total)
    }

    /// Allocate an uninitialized buffer of exactly `size` bytes.
    pub(crate) fn alloc_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<TrackedBuffer> {
        self.alloc_described(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// The one place a device allocation is actually requested: reserve, and on
    /// a refusal reclaim idle pooled buffers and try once more before
    /// declining.
    fn alloc_described(&self, desc: &wgpu::BufferDescriptor<'_>) -> Option<TrackedBuffer> {
        if let Some(buffer) = TrackedBuffer::create(&self.device, &self.budget, desc) {
            return Some(buffer);
        }
        if let Ok(mut pool) = self.pool.lock() {
            pool.reclaim_for(desc.size);
        }
        TrackedBuffer::create(&self.device, &self.budget, desc)
    }

    /// Allocate a read-back buffer of `size` bytes (`MAP_READ | COPY_DST`).
    pub(crate) fn staging_buffer(&self, label: &str, size: u64) -> Option<TrackedBuffer> {
        self.alloc_buffer(
            label,
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// Take a buffer of at least `size` bytes from the pool, or allocate one.
    pub(crate) fn pooled_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<TrackedBuffer> {
        let mut pool = self.pool.lock().ok()?;
        pool.get_buffer(&self.device, size, usage)
    }

    /// Allocate a buffer holding `contents`.
    ///
    /// # Two upload paths, one for each target
    ///
    /// Native maps the new allocation at creation and memcpys into it — what
    /// `wgpu::util::DeviceExt::create_buffer_init` does, and what every kernel
    /// here used to call directly.
    ///
    /// wasm32 must not: `get_mapped_range_mut` on the WebGPU backend materializes
    /// the mapped range as a `Vec<u8>` *in wasm linear memory*
    /// (`wgpu-29.0.4`, `backend/webgpu.rs`'s `WebBufferMappedRange`), copies the
    /// caller's bytes into it, and copies that Vec into the JS `ArrayBuffer` on
    /// unmap — so uploading a weight tensor costs a full-size temporary in the
    /// module's own heap on top of the device allocation. `GPUQueue.writeBuffer`
    /// takes a view of wasm memory instead (wasm-bindgen passes `&[u8]` as a
    /// subarray of the module's memory), so the bytes go straight from
    /// `contents` to the driver.
    ///
    /// Both arms request the same usage flags, so a buffer allocated on one
    /// target is interchangeable with one allocated on the other.
    pub(crate) fn upload_buffer(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Option<TrackedBuffer> {
        // An empty upload cannot produce a bindable buffer, so it is a decline
        // rather than a zero-sized allocation that fails later at bind time.
        if contents.is_empty() {
            return None;
        }
        // `COPY_DST` is what makes `write_buffer` legal; it is requested on
        // both targets so the two paths produce identical buffers.
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        // Buffer sizes and copy sizes are both multiples of 4 in WebGPU, and
        // [w2-f16] this rounding really does pad: an `f16` weight with an odd
        // element count uploads `2 * len` bytes, i.e. 2 mod 4, so the buffer is
        // two bytes longer than the data. Both consequences are benign.
        //
        // * The shader never reads the padding. Every kernel bounds its weight
        //   indexing by the element counts in its uniform block, which are
        //   derived from the operand shapes, not from the buffer's size.
        // * The residency cache is not confused by it: `resident_upload` keys
        //   and stores the *requested* `byte_len` (`WeightFormat::byte_len`),
        //   never the padded allocation, so a later lookup for the same operand
        //   compares equal.
        //
        // Neither arm below reads the padding back: the native one copies
        // exactly `contents.len()` bytes into the mapped range and leaves the
        // tail as wgpu created it, and the wasm one extends the staging copy
        // with zeroes rather than issue an unaligned `write_buffer`.
        let size = (contents.len() as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let buffer = self.alloc_described(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: true,
            })?;
            // Unmap before deciding: destroying a still-mapped buffer (which is
            // what dropping the handle on a decline would do) is a state wgpu
            // need not be put in, and `size >= contents.len()` by construction
            // anyway.
            let filled = {
                let mut view = buffer.slice(..).get_mapped_range_mut();
                let fits = view.len() >= contents.len();
                if fits {
                    view.slice(..contents.len()).copy_from_slice(contents);
                }
                fits
            };
            buffer.unmap();
            if !filled {
                return None;
            }
            self.note_upload(contents.len() as u64);
            Some(buffer)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let buffer = self.alloc_buffer(label, size, usage)?;
            if size as usize == contents.len() {
                self.queue.write_buffer(&buffer, 0, contents);
            } else {
                let mut padded = contents.to_vec();
                padded.resize(size as usize, 0);
                self.queue.write_buffer(&buffer, 0, &padded);
            }
            self.note_upload(contents.len() as u64);
            Some(buffer)
        }
    }

    /// Add a completed host→device transfer to [`Self::uploaded_bytes`].
    ///
    /// Called only after the transfer has actually happened, so a decline on
    /// the byte budget never shows up as bytes that crossed the bus.
    fn note_upload(&self, bytes: u64) {
        self.upload_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Bytes sitting idle in the reusable-buffer pool.
    ///
    /// A subset of [`Self::live_gpu_bytes`] that no dispatch is reading. Exposed
    /// so a caller measuring residency can separate "still allocated because an
    /// activation is alive" from "still allocated because the pool is holding
    /// it for the next node".
    #[must_use]
    pub fn pooled_gpu_bytes(&self) -> u64 {
        self.pool.lock().map_or(0, |pool| pool.pooled_bytes())
    }

    /// \[w4\] Idle buffers the pool is holding, and the count bound it is holding
    /// them under.
    ///
    /// The pair matters more than either number: with activation recycling a
    /// graph that returns more buffers per frame than it requests walks the
    /// pool up to its retention bound and is then held there by LRU eviction,
    /// so "is this a steady state or a leak" is answered by whether the count
    /// has reached the bound — not by whether it stopped moving.
    #[must_use]
    pub fn pooled_buffers(&self) -> (usize, usize) {
        self.pool
            .lock()
            .map_or((0, 0), |pool| (pool.available_count(), pool.max_buffers()))
    }

    /// \[w4\] The pool's byte retention bound — the other half of
    /// [`Self::pooled_buffers`]'s count bound.
    #[must_use]
    pub fn pool_byte_budget(&self) -> u64 {
        self.pool.lock().map_or(0, |pool| pool.byte_budget())
    }

    /// \[w4\] Buffer requests the pool has served from an idle entry, cumulative.
    ///
    /// With [`Self::pool_allocations`] this is the whole account of what a
    /// change in activation disposition does to allocation churn: the request
    /// count is their sum and is a property of the graph, so only the split
    /// between them can move.
    #[must_use]
    pub fn pool_reuses(&self) -> u64 {
        self.pool.lock().map_or(0, |pool| pool.reuses())
    }

    /// \[w4\] Buffer requests the pool has had to ask the driver for, cumulative.
    /// See [`Self::pool_reuses`].
    #[must_use]
    pub fn pool_allocations(&self) -> u64 {
        self.pool.lock().map_or(0, |pool| pool.allocations())
    }
}

// ========================================================================
// Run-scoped activations
// ========================================================================

/// Consuming a kernel's result in place, instead of reading it back and
/// uploading it again for the next node. See `super::activation`.
impl GpuContext {
    /// Whether kernels may be asked to leave results on the device.
    ///
    /// A context-level switch rather than a session one, because the thing it
    /// controls is a property of the device path: with it off, every entry point
    /// behaves exactly as it did before run-scoped activations existed, which is
    /// what makes an A/B of the two a meaningful test rather than a comparison
    /// of two different sessions.
    #[must_use]
    pub fn activation_residency_enabled(&self) -> bool {
        self.activation_residency.load(Ordering::Relaxed)
    }

    /// Turn run-scoped activation residency on or off. On by default.
    pub fn set_activation_residency(&self, enabled: bool) {
        self.activation_residency.store(enabled, Ordering::Relaxed);
    }
}

// ========================================================================
// Half-precision compute
// ========================================================================

/// [w2-f16] The opt-in `f16` path through the two compute-dominant kernels.
/// See `super::weight_format` for what changes numerically and where.
impl GpuContext {
    /// Whether this device was created with `wgpu::Features::SHADER_F16`.
    ///
    /// Answers "could this context ever run the half-precision kernels?", which
    /// is a property of the adapter and never changes for a given context. A
    /// caller deciding whether to *offer* the mode asks this; a caller deciding
    /// whether a dispatch will *take* it asks [`Self::f16_compute_enabled`].
    #[must_use]
    pub fn f16_compute_supported(&self) -> bool {
        self.f16_compute.supported()
    }

    /// Ask the two compute-dominant kernels to use half precision, and get back
    /// the state that actually took effect.
    ///
    /// **Off by default, on every device.** This mode changes results — see
    /// `super::weight_format` for the three rounding points — and a mode that
    /// changes results must be asked for, never inherited.
    ///
    /// Asking for it on a device without the feature is not an error: the
    /// request is recorded, the effective state stays `false`, and every kernel
    /// keeps taking the `f32` path unchanged. The returned value is the
    /// *effective* state, so `set_f16_compute(true) == false` is the caller's
    /// signal that this device cannot do it.
    ///
    /// Safe to call between runs on a live session. Weights already uploaded in
    /// the other format stay resident and are simply not consulted; the format
    /// is part of the residency cache's key, so no dispatch can be served bytes
    /// in a format its shader does not read (`super::resident`).
    pub fn set_f16_compute(&self, enabled: bool) -> bool {
        self.f16_compute.set(enabled)
    }

    /// Whether dispatches are currently taking the half-precision path.
    ///
    /// The *effective* state: `false` whenever the device lacks the feature, no
    /// matter what was requested.
    #[must_use]
    pub fn f16_compute_enabled(&self) -> bool {
        self.f16_compute.enabled()
    }

    /// The on-device format this context's weight operands should take right
    /// now.
    ///
    /// One place for the "toggle -> format" step so the kernels and the byte
    /// accounting cannot disagree about it.
    #[must_use]
    pub fn weight_format(&self) -> WeightFormat {
        WeightFormat::for_f16(self.f16_compute_enabled())
    }

    /// The buffer to bind for one operand, uploading it only when it is not
    /// already on the device.
    ///
    /// The residency counterpart of [`Self::operand_buffer`]: that one keys
    /// session-lifetime *weights*, this one takes a run-scoped activation the
    /// caller already holds. Both hand back an `OperandBuffer` the kernel binds
    /// identically, which is what keeps one kernel body serving both regimes.
    pub(crate) fn operand_source<'a>(
        &self,
        label: &'static str,
        source: TensorSource<'a>,
        usage: wgpu::BufferUsages,
    ) -> Option<OperandBuffer<'a>> {
        match source {
            TensorSource::Device(tensor) => Some(OperandBuffer::Device(tensor)),
            TensorSource::Host { data, .. } => self
                .upload_buffer(label, bytemuck::cast_slice(data), usage)
                .map(OperandBuffer::Transient),
        }
    }

    /// Bytes an operand still has to allocate — zero when it is already on the
    /// device as a run-scoped activation.
    ///
    /// The activation analogue of [`Self::operand_admission_bytes`], and it
    /// exists for the same reason: a resident operand's bytes are already in the
    /// live total, so counting them again would decline nodes that fit.
    pub(crate) fn source_admission_bytes(&self, source: TensorSource<'_>, bytes: u64) -> u64 {
        if source.is_device() {
            0
        } else {
            bytes
        }
    }

    /// Upload a host tensor into a run-scoped device buffer.
    ///
    /// For a caller that wants an operand on the device *before* a dispatch
    /// rather than as a side effect of one — the small host operand of a node
    /// whose other operands are already resident, whose upload is what makes the
    /// whole node dispatchable in place.
    ///
    /// `COPY_SRC` is requested alongside `STORAGE` so the result can still be
    /// read back like any other activation.
    ///
    /// Declines on a degraded context, like every kernel entry point: once the
    /// device is degraded nothing will dispatch against these bytes, so an
    /// allocation here would only spend budget and bus on an operand whose
    /// consumer is already destined for the CPU.
    pub fn upload_device_tensor(
        &self,
        label: &'static str,
        data: &[f32],
        shape: &[usize],
    ) -> Option<DeviceTensor> {
        if self.is_degraded() || data.is_empty() {
            return None;
        }
        let bytes = crate::device_guard::checked_storage_bytes(&self.limits, data.len())?;
        if !self.budget_admits(&[bytes]) {
            return None;
        }
        let buffer = self.upload_buffer(
            label,
            bytemuck::cast_slice(data),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        Some(DeviceTensor::new(buffer, shape.to_vec(), data.len(), bytes))
    }

    /// Hand a finished activation's allocation back to the reusable-buffer pool
    /// instead of destroying it.
    ///
    /// \[w4\] This is what the session's run loop does at an activation's last
    /// consumer (`session::gpu_activations`'s `RunActivations::dispose`, which
    /// carries the A/B that chose it): measured against destroying, it is
    /// 2% faster on InSwapper-128 and 11–28% faster on a chain of small
    /// activations, byte-identical in 200 of 200 measured pairs.
    ///
    /// The bytes stay in [`Self::live_gpu_bytes`] until the pool evicts them,
    /// so a caller asserting "a finished run is back at its resident-weight
    /// baseline" must clear or reclaim the pool first. Nothing is stranded by
    /// that: [`Self::pooled_gpu_bytes`] is reclaimed before any allocation is
    /// declined, so a pooled buffer can never be the reason a node falls back
    /// to the CPU.
    ///
    /// A poisoned pool mutex costs the recycling and nothing else — the tensor
    /// is destroyed, exactly as it was before this became the default.
    pub fn recycle_device_tensor(&self, tensor: DeviceTensor) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.return_buffer(tensor.into_buffer());
        }
    }
}

// ========================================================================
// Weight residency
// ========================================================================

/// Buffers for operands whose bytes never change, kept for this context's
/// lifetime. See `super::resident` for what that buys and why the cache is
/// owned here rather than by the caller.
impl GpuContext {
    /// The buffer to bind for one operand, uploading it at most once when
    /// `key` names an identity whose bytes are invariant.
    ///
    /// `key` is `None` for anything that changes between dispatches — every
    /// activation, every params block — and those take the ordinary
    /// [`Self::upload_buffer`] path unchanged. `label` names the kernel slot;
    /// it is part of what a cache hit has to agree on, so two kernels cannot
    /// end up sharing bytes they packed differently under one caller key.
    ///
    /// `None` means the byte budget declined the upload, exactly as
    /// [`Self::upload_buffer`] does, and the kernel falls back to the CPU.
    pub(crate) fn operand_buffer(
        &self,
        key: Option<&str>,
        label: &'static str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Option<OperandBuffer<'static>> {
        let byte_len = contents.len() as u64;
        self.resident_upload(key, label, byte_len, WeightFormat::F32, usage, |_| {
            WeightBytes::Borrowed(contents)
        })
    }

    /// [w2-f16] [`Self::operand_buffer`] over `f32` values that may be narrowed
    /// to `f16` on their way to the device.
    ///
    /// # Why this takes `&[f32]` and not bytes
    ///
    /// So that a cache **hit costs nothing**. The lookup needs only the byte
    /// length, which [`WeightFormat::byte_len`] computes from the element count
    /// without touching a single value; the `f32 -> f16` conversion runs inside
    /// the `Vacant`/`Conflict` arms, i.e. once per operand per session, right
    /// before the bytes are handed to the driver.
    ///
    /// Converting at the call site instead would put 9.4M `f32 -> f16`
    /// conversions on the host critical path of *every* frame for InSwapper's
    /// dominant convolution — comfortably more host time than the whole
    /// dispatch saves, and it would have shown up as an f16 mode that is slower
    /// than f32 for reasons nothing in the shader explains.
    pub(crate) fn operand_buffer_typed(
        &self,
        key: Option<&str>,
        label: &'static str,
        data: &[f32],
        format: WeightFormat,
        usage: wgpu::BufferUsages,
    ) -> Option<OperandBuffer<'static>> {
        let byte_len = format.byte_len(data.len());
        self.resident_upload(key, label, byte_len, format, usage, |format| {
            format.convert(data)
        })
    }

    /// The shared body of [`Self::operand_buffer`] and
    /// [`Self::operand_buffer_typed`].
    ///
    /// `materialize` produces the bytes to upload and is called **only** on the
    /// paths that upload — never on a hit. That is the property the whole
    /// half-precision design rests on, so it lives in one place rather than in
    /// each caller.
    fn resident_upload<'m>(
        &self,
        key: Option<&str>,
        label: &'static str,
        byte_len: u64,
        format: WeightFormat,
        usage: wgpu::BufferUsages,
        materialize: impl Fn(WeightFormat) -> WeightBytes<'m>,
    ) -> Option<OperandBuffer<'static>> {
        let Some(key) = key else {
            return self
                .upload_buffer(label, materialize(format).as_bytes(), usage)
                .map(OperandBuffer::Transient);
        };
        // `upload_buffer` adds `COPY_DST` to every buffer it makes, so the
        // usage a cached entry must cover is the same superset — comparing
        // against the caller's bare flags would reject a perfectly good entry.
        let created_usage = usage | wgpu::BufferUsages::COPY_DST;
        match self
            .resident
            .lookup(key, label, byte_len, created_usage, format)
        {
            Lookup::Hit(buffer) => Some(OperandBuffer::Resident(buffer)),
            Lookup::Vacant => {
                let buffer = self.upload_buffer(label, materialize(format).as_bytes(), usage)?;
                Some(OperandBuffer::Resident(
                    self.resident.insert(key, label, byte_len, buffer, format),
                ))
            }
            Lookup::Conflict => {
                let buffer = self.upload_buffer(label, materialize(format).as_bytes(), usage)?;
                self.resident.note_conflict(byte_len);
                Some(OperandBuffer::Transient(buffer))
            }
        }
    }

    /// Bytes an operand still has to allocate — zero once it is resident.
    ///
    /// Kernels feed this to [`Self::budget_admits`] instead of the raw operand
    /// size: a resident weight is *already* counted in the live-byte total, so
    /// asking for room for it a second time would decline nodes that fit.
    /// Advisory in the same way `budget_admits` is; the reservation inside each
    /// allocation remains the authority.
    pub(crate) fn operand_admission_bytes(&self, key: Option<&str>, bytes: u64) -> u64 {
        self.operand_admission_bytes_for(key, WeightFormat::F32, bytes)
    }

    /// [w2-f16] [`Self::operand_admission_bytes`] for a specific on-device
    /// format.
    ///
    /// The format matters and cannot be elided. A weight resident as `f32` has
    /// *not* paid for its `f16` copy: the first `f16` dispatch after a toggle
    /// flip still has to allocate and upload half as many bytes again, and
    /// reporting that as free would admit a node the byte budget has no room
    /// for — which is precisely the accounting this crate declines on rather
    /// than discovers at allocation time.
    pub(crate) fn operand_admission_bytes_for(
        &self,
        key: Option<&str>,
        format: WeightFormat,
        bytes: u64,
    ) -> u64 {
        match key {
            Some(key) if self.resident.contains_format(key, format) => 0,
            _ => bytes,
        }
    }

    /// Whether an operand with this identity is on the device already.
    ///
    /// Answers the caller's question — "will this operand cross the bus?" —
    /// with one qualification: an entry whose slot label or byte length
    /// disagrees with the next request still uploads per dispatch, and this
    /// reports it as resident anyway. That combination means a caller has
    /// reused one identity for two different byte sequences, which the cache
    /// treats as a conflict; [`Self::resident_counters`] is where it shows up
    /// (as upload bytes that never stop growing).
    #[must_use]
    pub fn is_resident(&self, key: &str) -> bool {
        self.resident.contains(key)
    }

    /// Cumulative hit/miss/upload counters for the residency cache.
    ///
    /// The measurement this whole mechanism is judged on: difference two
    /// snapshots around a run, and `uploaded_bytes` must be zero for every run
    /// after the first.
    #[must_use]
    pub fn resident_counters(&self) -> ResidentCounters {
        self.resident.counters()
    }

    /// Device bytes currently pinned by resident operands — a subset of
    /// [`Self::live_gpu_bytes`] that no dispatch will release.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.resident.bytes()
    }

    /// How many distinct operands are resident.
    #[must_use]
    pub fn resident_len(&self) -> usize {
        self.resident.len()
    }

    /// Cumulative bytes uploaded to the device through this context, resident
    /// and transient alike.
    ///
    /// Never falls, and never resets. A run's own upload volume is the
    /// difference between two readings; the residency claim is that the
    /// *initializer* part of that difference goes to zero after the first run,
    /// which [`Self::resident_counters`] reports separately (activations,
    /// params blocks and the bias placeholder keep uploading every dispatch, so
    /// this total does not go to zero and is not expected to).
    #[must_use]
    pub fn uploaded_bytes(&self) -> u64 {
        self.upload_bytes.load(Ordering::Relaxed)
    }

    /// Release every resident buffer.
    ///
    /// Called by `Drop`; also the escape hatch for a caller that wants the
    /// memory back mid-session (a page that has switched models, say). Buffers
    /// a dispatch is still holding survive until that dispatch drops them.
    pub fn clear_resident_buffers(&self) {
        self.resident.clear();
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
        // Idle pooled buffers still hold device memory, and letting the pool's
        // own drop release them would free nothing in a browser (see
        // `super::budget`). Clearing destroys every entry explicitly.
        if let Ok(mut pool) = self.pool.lock() {
            pool.clear();
        }
        // Same for the resident weights, and for the same reason: releasing
        // them here destroys them while the device is unambiguously still
        // alive, rather than leaving it to field drop glue that runs after this
        // function returns.
        self.resident.clear();
        // [w5] And the compiled pipelines, for the ordering rather than the
        // release: a `ComputePipeline` holds its own handle on the device, and
        // `device` is declared before `pipelines`, so drop glue would destroy
        // this context's device handle first and its pipelines afterwards.
        //
        // The thread-local caches this replaced needed a purge here for a
        // different, worse reason: their entries kept a device alive across
        // contexts, the purge could only ever reach the dropping thread's own
        // copy, and — because they identified a device by a handle comparison
        // that is really a per-`Instance` id — it could not reliably tell this
        // context's entries from the next context's in the first place. See
        // `super::pipeline_cache`.
        self.pipelines.clear();
    }
}
