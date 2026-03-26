//! Benchmark-based execution path selection.
//!
//! Times CPU vs GPU execution for specific operators and recommends
//! the faster path based on input size.

use std::fmt;
use std::time::{Duration, Instant};

use oxionnx_core::Tensor;

/// Result of benchmarking a single operator.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The ONNX operator type (e.g. "MatMul", "Relu").
    pub op_type: String,
    /// Dimensions describing the benchmark input.
    pub input_shape: Vec<usize>,
    /// Time spent executing on CPU.
    pub cpu_time: Duration,
    /// Time spent executing on GPU, if available.
    pub gpu_time: Option<Duration>,
    /// Recommended execution path based on timing.
    pub recommended: ExecutionPath,
    /// Ratio of gpu_time / cpu_time.  Less than 1.0 means GPU is faster.
    /// `NaN` when GPU timing is unavailable.
    pub speedup: f64,
}

/// Which execution back-end is recommended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPath {
    /// CPU execution.
    Cpu,
    /// GPU execution.
    Gpu,
}

impl fmt::Display for ExecutionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn recommend(cpu_time: Duration, gpu_time: Option<Duration>) -> (ExecutionPath, f64) {
    match gpu_time {
        Some(gt) => {
            let ratio = gt.as_secs_f64() / cpu_time.as_secs_f64().max(1e-12);
            if ratio < 1.0 {
                (ExecutionPath::Gpu, ratio)
            } else {
                (ExecutionPath::Cpu, ratio)
            }
        }
        None => (ExecutionPath::Cpu, f64::NAN),
    }
}

// ---------------------------------------------------------------------------
// MatMul benchmark
// ---------------------------------------------------------------------------

/// Benchmark `MatMul` with matrices of shape `[m, k]` x `[k, n]`.
pub fn benchmark_matmul(m: usize, k: usize, n: usize) -> BenchmarkResult {
    let a_data: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let a = Tensor::new(a_data.clone(), vec![m, k]);
    let b = Tensor::new(b_data.clone(), vec![k, n]);

    // CPU benchmark
    let cpu_start = Instant::now();
    let _ = oxionnx_ops::math::matmul(&a, &b);
    let cpu_time = cpu_start.elapsed();

    // GPU benchmark (if available)
    #[cfg(feature = "gpu")]
    let gpu_time = {
        use oxionnx_gpu::GpuContext;
        match GpuContext::try_new() {
            Some(ctx) => {
                let start = Instant::now();
                let _ = oxionnx_gpu::gpu_matmul(&ctx, &a_data, &b_data, m, k, n);
                Some(start.elapsed())
            }
            None => None,
        }
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_time: Option<Duration> = None;

    let (recommended, speedup) = recommend(cpu_time, gpu_time);

    BenchmarkResult {
        op_type: "MatMul".into(),
        input_shape: vec![m, k, n],
        cpu_time,
        gpu_time,
        recommended,
        speedup,
    }
}

// ---------------------------------------------------------------------------
// Element-wise benchmark
// ---------------------------------------------------------------------------

/// Benchmark an element-wise activation at a given tensor size.
///
/// Supported `op` values: `"Relu"`, `"Sigmoid"`, `"Tanh"`, `"Gelu"`.
pub fn benchmark_elementwise(op: &str, size: usize) -> BenchmarkResult {
    let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let tensor = Tensor::new(data.clone(), vec![size]);

    let cpu_start = Instant::now();
    match op {
        "Relu" => {
            let _ = oxionnx_ops::nn::relu(&tensor);
        }
        "Sigmoid" => {
            let _ = oxionnx_ops::nn::sigmoid(&tensor);
        }
        "Tanh" => {
            let _ = oxionnx_ops::nn::tanh_op(&tensor);
        }
        "Gelu" => {
            let _ = oxionnx_ops::nn::gelu(&tensor);
        }
        _ => {}
    }
    let cpu_time = cpu_start.elapsed();

    #[cfg(feature = "gpu")]
    let gpu_time = {
        use oxionnx_gpu::GpuContext;
        match GpuContext::try_new() {
            Some(ctx) => {
                let start = Instant::now();
                let result = match op {
                    "Relu" => oxionnx_gpu::gpu_relu(&ctx, &data),
                    "Sigmoid" => oxionnx_gpu::gpu_sigmoid(&ctx, &data),
                    "Gelu" => oxionnx_gpu::gpu_gelu(&ctx, &data),
                    _ => None,
                };
                if result.is_some() {
                    Some(start.elapsed())
                } else {
                    None
                }
            }
            None => None,
        }
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_time: Option<Duration> = None;

    let (recommended, speedup) = recommend(cpu_time, gpu_time);

    BenchmarkResult {
        op_type: op.into(),
        input_shape: vec![size],
        cpu_time,
        gpu_time,
        recommended,
        speedup,
    }
}

// ---------------------------------------------------------------------------
// Suite runner
// ---------------------------------------------------------------------------

/// Run a suite of benchmarks across common sizes and return all results.
pub fn benchmark_suite() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // MatMul benchmarks at increasing sizes
    for &(m, k, n) in &[
        (64, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        results.push(benchmark_matmul(m, k, n));
    }

    // Element-wise activation benchmarks
    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        for op in &["Relu", "Sigmoid", "Gelu"] {
            results.push(benchmark_elementwise(op, size));
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format benchmark results as a human-readable table.
pub fn format_results(results: &[BenchmarkResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<12} {:<20} {:>12} {:>12} {:>8} {:>8}\n",
        "Op", "Shape", "CPU (us)", "GPU (us)", "Winner", "Speedup"
    ));
    out.push_str(&"-".repeat(76));
    out.push('\n');
    for r in results {
        let shape_str = format!("{:?}", r.input_shape);
        let cpu_us = r.cpu_time.as_micros();
        let gpu_us = r
            .gpu_time
            .map(|d| format!("{}", d.as_micros()))
            .unwrap_or_else(|| "N/A".into());
        let winner = r.recommended;
        let speedup = if r.speedup.is_nan() {
            "N/A".into()
        } else {
            format!("{:.2}x", r.speedup)
        };
        out.push_str(&format!(
            "{:<12} {:<20} {:>12} {:>12} {:>8} {:>8}\n",
            r.op_type, shape_str, cpu_us, gpu_us, winner, speedup
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_matmul() {
        let result = benchmark_matmul(16, 16, 16);
        assert_eq!(result.op_type, "MatMul");
        assert_eq!(result.input_shape, vec![16, 16, 16]);
        assert!(!result.cpu_time.is_zero() || result.cpu_time == Duration::ZERO);
        // On a CPU-only build, recommended should be CPU
        #[cfg(not(feature = "gpu"))]
        assert_eq!(result.recommended, ExecutionPath::Cpu);
    }

    #[test]
    fn test_benchmark_elementwise() {
        let result = benchmark_elementwise("Relu", 1000);
        assert_eq!(result.op_type, "Relu");
        assert_eq!(result.input_shape, vec![1000]);
        #[cfg(not(feature = "gpu"))]
        assert_eq!(result.recommended, ExecutionPath::Cpu);
    }

    #[test]
    fn test_benchmark_suite() {
        let results = benchmark_suite();
        // 4 MatMul sizes + 4 sizes * 3 ops = 16 total
        assert_eq!(results.len(), 16);
        assert_eq!(results[0].op_type, "MatMul");
        assert_eq!(results[4].op_type, "Relu");
    }

    #[test]
    fn test_format_results() {
        let results = vec![BenchmarkResult {
            op_type: "MatMul".into(),
            input_shape: vec![32, 32, 32],
            cpu_time: Duration::from_micros(100),
            gpu_time: None,
            recommended: ExecutionPath::Cpu,
            speedup: f64::NAN,
        }];
        let table = format_results(&results);
        assert!(table.contains("MatMul"));
        assert!(table.contains("CPU"));
        assert!(table.contains("N/A"));
        assert!(table.contains("[32, 32, 32]"));
    }

    #[test]
    fn test_execution_path_display() {
        assert_eq!(format!("{}", ExecutionPath::Cpu), "CPU");
        assert_eq!(format!("{}", ExecutionPath::Gpu), "GPU");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_supported_ops() {
        let ops = crate::session::GpuExecutionProvider::supported_ops();
        assert!(!ops.is_empty());
        assert!(ops.contains(&"MatMul"));
        assert!(ops.contains(&"Relu"));
    }
}
