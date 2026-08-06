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

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]
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

    /// Merge `other`'s per-device counts into `self`, field-by-field.
    ///
    /// A pure, model-free accumulation primitive with no `objc2`/CoreML
    /// dependency — the same routine
    /// [`crate::MlPackageModel::compute_plan_summary`] uses to fold
    /// every `MLProgram` function's classified operations into one
    /// flat histogram, and
    /// [`crate::MlPackageModel::compute_plan_breakdown`] uses to fold
    /// same-named operations (e.g. three separate `"gather"` ops
    /// scattered across the graph) into that operator's single
    /// breakdown entry. Because both entry points build on this exact
    /// merge, summing every entry of `compute_plan_breakdown`'s map
    /// always reconciles with `compute_plan_summary`'s totals for the
    /// same model.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.ane_ops += other.ane_ops;
        self.gpu_ops += other.gpu_ops;
        self.cpu_ops += other.cpu_ops;
        self.unknown_ops += other.unknown_ops;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `merge` must sum every field independently — no cross-field
    /// bleed, no dropped counts.
    #[test]
    fn merge_sums_fields_from_both_operands() {
        let mut a = ComputePlanSummary {
            ane_ops: 3,
            gpu_ops: 1,
            cpu_ops: 0,
            unknown_ops: 2,
        };
        let b = ComputePlanSummary {
            ane_ops: 5,
            gpu_ops: 0,
            cpu_ops: 4,
            unknown_ops: 1,
        };
        a.merge(&b);
        assert_eq!(a.ane_ops, 8);
        assert_eq!(a.gpu_ops, 1);
        assert_eq!(a.cpu_ops, 4);
        assert_eq!(a.unknown_ops, 3);
    }

    /// Merging in a `Default::default()` summary (the zero element)
    /// must be a no-op — required for `HashMap::entry(...).or_default()`
    /// accumulation (as `compute_plan_breakdown` uses) to behave
    /// correctly for an operator name's first-seen occurrence.
    #[test]
    fn merge_with_default_is_identity() {
        let before = ComputePlanSummary {
            ane_ops: 7,
            gpu_ops: 2,
            cpu_ops: 1,
            unknown_ops: 0,
        };
        let mut a = before;
        a.merge(&ComputePlanSummary::default());
        assert_eq!(a.ane_ops, before.ane_ops);
        assert_eq!(a.gpu_ops, before.gpu_ops);
        assert_eq!(a.cpu_ops, before.cpu_ops);
        assert_eq!(a.unknown_ops, before.unknown_ops);
    }

    /// Mirrors how `accumulate_program_operations` (in
    /// `package::macos_impl`) folds one single-field delta per
    /// classified operation into a running total: merging N
    /// single-op deltas must yield `total_ops() == N`, and the flat
    /// total must equal the sum of the same deltas folded into a
    /// per-operator-name breakdown map — the reconciliation
    /// invariant `compute_plan_breakdown`'s model-driven test
    /// exercises end-to-end.
    #[test]
    fn merging_single_op_deltas_reconciles_flat_and_keyed_totals() {
        let classified_ops = [
            (
                "conv",
                ComputePlanSummary {
                    ane_ops: 1,
                    ..ComputePlanSummary::default()
                },
            ),
            (
                "gather",
                ComputePlanSummary {
                    gpu_ops: 1,
                    ..ComputePlanSummary::default()
                },
            ),
            (
                "gather",
                ComputePlanSummary {
                    cpu_ops: 1,
                    ..ComputePlanSummary::default()
                },
            ),
            (
                "const",
                ComputePlanSummary {
                    unknown_ops: 1,
                    ..ComputePlanSummary::default()
                },
            ),
        ];

        let mut flat = ComputePlanSummary::default();
        let mut breakdown: std::collections::HashMap<&str, ComputePlanSummary> =
            std::collections::HashMap::new();
        for (name, delta) in &classified_ops {
            flat.merge(delta);
            breakdown.entry(name).or_default().merge(delta);
        }

        assert_eq!(flat.total_ops(), 4);
        assert_eq!(flat.compute_ops(), 3);
        assert_eq!(flat.ane_ops, 1);
        assert_eq!(flat.gpu_ops, 1);
        assert_eq!(flat.cpu_ops, 1);
        assert_eq!(flat.unknown_ops, 1);

        // "gather" appeared twice (once GPU, once CPU) and must have
        // accumulated into a single entry, not overwritten itself.
        let gather = breakdown.get("gather").copied().unwrap_or_default();
        assert_eq!(gather.gpu_ops, 1);
        assert_eq!(gather.cpu_ops, 1);
        assert_eq!(gather.total_ops(), 2);

        // Summing every breakdown entry must reconcile exactly with
        // the flat total — the core invariant this whole feature
        // exists to guarantee.
        let mut reconciled = ComputePlanSummary::default();
        for per_op in breakdown.values() {
            reconciled.merge(per_op);
        }
        assert_eq!(reconciled.total_ops(), flat.total_ops());
        assert_eq!(reconciled.ane_ops, flat.ane_ops);
        assert_eq!(reconciled.gpu_ops, flat.gpu_ops);
        assert_eq!(reconciled.cpu_ops, flat.cpu_ops);
        assert_eq!(reconciled.unknown_ops, flat.unknown_ops);
    }
}
