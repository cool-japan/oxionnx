//! Public wrapper around `MLComputeUnits`.
//!
//! Apple's `MLComputeUnits` is an Objective-C `NS_ENUM`; we mirror its four
//! variants behind a plain Rust enum so callers who do not depend on
//! `objc2-core-ml` (and consumers on non-macOS targets) can still configure
//! the runtime.
//!
//! The mapping is:
//!
//! | [`MlComputeUnits`] | Apple constant                |
//! | :----------------- | :---------------------------- |
//! | [`MlComputeUnits::CpuOnly`]   | `MLComputeUnitsCPUOnly`           |
//! | [`MlComputeUnits::CpuAndGpu`] | `MLComputeUnitsCPUAndGPU`         |
//! | [`MlComputeUnits::CpuAndAne`] | `MLComputeUnitsCPUAndNeuralEngine`|
//! | [`MlComputeUnits::All`]       | `MLComputeUnitsAll` (default)     |

/// Hardware classes the CoreML runtime is allowed to schedule the model on.
///
/// Default is [`MlComputeUnits::All`] — the runtime picks the fastest
/// per-op placement (which is the configuration that delivered the
/// 17–25× speedups versus CPU-only on the validation models).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MlComputeUnits {
    /// CPU only.  Useful for apples-to-apples baselining against the CPU
    /// dispatch path.
    CpuOnly,
    /// CPU + GPU; ANE is excluded.
    CpuAndGpu,
    /// CPU + Apple Neural Engine; integrated GPU is excluded.
    CpuAndAne,
    /// All available compute units; runtime decides per op.  This is the
    /// recommended default — the CoreML scheduler is consistently better at
    /// placement than user heuristics.
    #[default]
    All,
}

#[cfg(target_os = "macos")]
impl MlComputeUnits {
    /// Project to the underlying `MLComputeUnits` constant.
    #[inline]
    pub(crate) fn to_native(self) -> objc2_core_ml::MLComputeUnits {
        match self {
            Self::CpuOnly => objc2_core_ml::MLComputeUnits::CPUOnly,
            Self::CpuAndGpu => objc2_core_ml::MLComputeUnits::CPUAndGPU,
            Self::CpuAndAne => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            Self::All => objc2_core_ml::MLComputeUnits::All,
        }
    }
}

/// Per-device counts of program operations after CoreML has lowered the
/// graph for the requested compute-unit policy.  Returned by
/// [`crate::MlPackageModel::compute_plan_summary`].
///
/// `const`-only ops (data placement, no actual compute) are bucketed as
/// `unknown_ops` because the framework does not return a compute device for
/// them.  When evaluating ANE engagement use the ratio of `ane_ops` to the
/// sum of `ane_ops + gpu_ops + cpu_ops`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ComputePlanSummary {
    /// Operations the runtime placed on the Apple Neural Engine.
    pub ane_ops: usize,
    /// Operations the runtime placed on the integrated GPU.
    pub gpu_ops: usize,
    /// Operations the runtime fell back to CPU for.
    pub cpu_ops: usize,
    /// Operations with no reported device — typically `const` data
    /// placement.  Not counted as compute work.
    pub unknown_ops: usize,
}

impl ComputePlanSummary {
    /// Total program ops including `const` / data-placement entries.
    #[inline]
    pub fn total_ops(&self) -> usize {
        self.ane_ops + self.gpu_ops + self.cpu_ops + self.unknown_ops
    }

    /// Total compute ops (excluding `const`-style placements).
    #[inline]
    pub fn compute_ops(&self) -> usize {
        self.ane_ops + self.gpu_ops + self.cpu_ops
    }

    /// Fraction of compute ops placed on the ANE — `0.0..=1.0`.  Returns
    /// `0.0` when there are no compute ops (defensive against empty graphs).
    #[inline]
    pub fn ane_fraction(&self) -> f64 {
        let n = self.compute_ops();
        if n == 0 {
            0.0
        } else {
            (self.ane_ops as f64) / (n as f64)
        }
    }
}
