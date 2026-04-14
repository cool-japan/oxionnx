//! Opset version coverage report generator.
//!
//! Reports which ONNX operators are supported at which opset versions.

use std::collections::BTreeMap;

/// Information about an operator's support status.
#[derive(Debug, Clone)]
pub struct OpCoverage {
    /// The ONNX operator name.
    pub op_type: String,
    /// Whether the operator is supported in the current registry.
    pub supported: bool,
    /// Minimum opset version where this op was introduced.
    pub since_opset: u32,
    /// Category (math, nn, shape, etc.)
    pub category: String,
}

/// Generate a coverage report for all known ONNX operators.
pub fn generate_coverage_report() -> Vec<OpCoverage> {
    let registry = oxionnx_ops::default_registry();

    // Define all ONNX standard operators with their introduction opset
    let ops: Vec<(&str, u32, &str)> = vec![
        // Math
        ("Add", 1, "math"),
        ("Sub", 1, "math"),
        ("Mul", 1, "math"),
        ("Div", 1, "math"),
        ("MatMul", 1, "math"),
        ("Gemm", 1, "math"),
        ("Pow", 1, "math"),
        ("Sqrt", 1, "math"),
        ("Exp", 1, "math"),
        ("Log", 1, "math"),
        ("Abs", 1, "math"),
        ("Neg", 1, "math"),
        ("Reciprocal", 1, "math"),
        ("Ceil", 1, "math"),
        ("Floor", 1, "math"),
        ("Round", 11, "math"),
        ("Sign", 9, "math"),
        ("Mod", 10, "math"),
        ("Sin", 7, "math"),
        ("Cos", 7, "math"),
        ("Tan", 7, "math"),
        ("Asin", 7, "math"),
        ("Acos", 7, "math"),
        ("Atan", 7, "math"),
        ("Sinh", 9, "math"),
        ("Cosh", 9, "math"),
        ("Asinh", 9, "math"),
        ("Acosh", 9, "math"),
        ("Atanh", 9, "math"),
        ("ReduceMean", 1, "math"),
        ("ReduceSum", 1, "math"),
        ("ReduceMax", 1, "math"),
        ("ReduceMin", 1, "math"),
        ("ReduceProd", 1, "math"),
        ("ArgMax", 1, "math"),
        ("ArgMin", 1, "math"),
        ("CumSum", 11, "math"),
        ("Range", 11, "math"),
        ("TopK", 1, "math"),
        ("BitShift", 11, "math"),
        ("ReduceL1", 1, "math"),
        ("ReduceL2", 1, "math"),
        ("ReduceLogSum", 1, "math"),
        ("ReduceLogSumExp", 1, "math"),
        ("ReduceSumSquare", 1, "math"),
        ("Min", 1, "math"),
        ("Max", 1, "math"),
        ("Mean", 1, "math"),
        ("Sum", 1, "math"),
        // NN
        ("Relu", 1, "nn"),
        ("Sigmoid", 1, "nn"),
        ("Tanh", 1, "nn"),
        ("Softmax", 1, "nn"),
        ("LogSoftmax", 1, "nn"),
        ("Gelu", 20, "nn"),
        ("Erf", 9, "nn"),
        ("LeakyRelu", 1, "nn"),
        ("PRelu", 1, "nn"),
        ("Elu", 1, "nn"),
        ("Selu", 1, "nn"),
        ("Celu", 12, "nn"),
        ("HardSigmoid", 1, "nn"),
        ("HardSwish", 14, "nn"),
        ("Softplus", 1, "nn"),
        ("Softsign", 1, "nn"),
        ("Mish", 18, "nn"),
        ("ThresholdedRelu", 10, "nn"),
        ("Hardmax", 1, "nn"),
        ("Shrink", 9, "nn"),
        ("BatchNormalization", 1, "nn"),
        ("LayerNormalization", 17, "nn"),
        ("GroupNormalization", 18, "nn"),
        ("InstanceNormalization", 1, "nn"),
        ("LpNormalization", 1, "nn"),
        ("MeanVarianceNormalization", 9, "nn"),
        ("Dropout", 1, "nn"),
        // Conv
        ("Conv", 1, "conv"),
        ("ConvTranspose", 1, "conv"),
        ("MaxPool", 1, "conv"),
        ("AveragePool", 1, "conv"),
        ("GlobalAveragePool", 1, "conv"),
        ("GlobalMaxPool", 1, "conv"),
        ("Pad", 1, "conv"),
        ("Resize", 10, "conv"),
        // Shape
        ("Reshape", 1, "shape"),
        ("Transpose", 1, "shape"),
        ("Squeeze", 1, "shape"),
        ("Unsqueeze", 1, "shape"),
        ("Flatten", 1, "shape"),
        ("Concat", 1, "shape"),
        ("Slice", 1, "shape"),
        ("Expand", 8, "shape"),
        ("Split", 1, "shape"),
        ("Tile", 1, "shape"),
        ("DepthToSpace", 1, "shape"),
        ("SpaceToDepth", 1, "shape"),
        ("ReverseSequence", 10, "shape"),
        // Indexing
        ("Gather", 1, "indexing"),
        ("GatherElements", 11, "indexing"),
        ("GatherND", 11, "indexing"),
        ("Where", 9, "indexing"),
        ("ScatterElements", 11, "indexing"),
        ("ScatterND", 11, "indexing"),
        ("OneHot", 9, "indexing"),
        ("Compress", 9, "indexing"),
        ("Unique", 11, "indexing"),
        ("NonZero", 9, "indexing"),
        // Comparison
        ("Equal", 1, "comparison"),
        ("Greater", 1, "comparison"),
        ("GreaterOrEqual", 12, "comparison"),
        ("Less", 1, "comparison"),
        ("LessOrEqual", 12, "comparison"),
        ("And", 1, "logic"),
        ("Or", 1, "logic"),
        ("Xor", 1, "logic"),
        ("Not", 1, "logic"),
        ("BitwiseAnd", 18, "bitwise"),
        ("BitwiseOr", 18, "bitwise"),
        ("BitwiseXor", 18, "bitwise"),
        ("BitwiseNot", 18, "bitwise"),
        ("IsInf", 10, "logic"),
        ("IsNaN", 9, "logic"),
        // Misc
        ("Identity", 1, "misc"),
        ("Cast", 1, "misc"),
        ("Shape", 1, "misc"),
        ("Size", 1, "misc"),
        ("Constant", 1, "misc"),
        ("ConstantOfShape", 9, "misc"),
        ("Clip", 1, "misc"),
        ("EyeLike", 9, "misc"),
        ("Trilu", 14, "misc"),
        // Quantization
        ("QuantizeLinear", 10, "quantization"),
        ("DequantizeLinear", 10, "quantization"),
        // Attention / RNN
        ("LSTM", 1, "rnn"),
        ("GRU", 1, "rnn"),
        ("Einsum", 12, "math"),
        ("NonMaxSuppression", 10, "detection"),
        ("GridSample", 16, "spatial"),
        ("RoiAlign", 10, "spatial"),
    ];

    ops.iter()
        .map(|&(name, since, cat)| OpCoverage {
            op_type: name.to_string(),
            supported: registry.contains(name),
            since_opset: since,
            category: cat.to_string(),
        })
        .collect()
}

/// Format coverage report as a markdown table.
pub fn format_coverage_markdown(report: &[OpCoverage]) -> String {
    let total = report.len();
    let supported = report.iter().filter(|o| o.supported).count();

    let mut out = String::from("# ONNX Operator Coverage Report\n\n");
    out.push_str(&format!(
        "**{}/{} operators supported ({:.0}%)**\n\n",
        supported,
        total,
        100.0 * supported as f64 / total as f64
    ));
    out.push_str("| Operator | Category | Since Opset | Supported |\n");
    out.push_str("|----------|----------|-------------|:---------:|\n");
    for op in report {
        let status = if op.supported { "yes" } else { "no" };
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            op.op_type, op.category, op.since_opset, status
        ));
    }
    out
}

/// Format as a summary grouped by category.
pub fn format_coverage_summary(report: &[OpCoverage]) -> String {
    let mut by_category: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for op in report {
        let entry = by_category.entry(op.category.clone()).or_insert((0, 0));
        entry.0 += 1;
        if op.supported {
            entry.1 += 1;
        }
    }
    let mut out = String::from("Category Coverage:\n");
    for (cat, (total, supported)) in &by_category {
        out.push_str(&format!(
            "  {:<15} {}/{} ({:.0}%)\n",
            cat,
            supported,
            total,
            100.0 * *supported as f64 / *total as f64
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_report() {
        let report = generate_coverage_report();
        assert!(!report.is_empty(), "report should have entries");
        // Verify that at least some operators are marked as supported
        let supported_count = report.iter().filter(|o| o.supported).count();
        assert!(
            supported_count > 0,
            "at least some operators should be supported"
        );
    }

    #[test]
    fn test_coverage_markdown() {
        let report = generate_coverage_report();
        let md = format_coverage_markdown(&report);
        assert!(md.contains("# ONNX Operator Coverage Report"));
        assert!(md.contains("| Operator |"));
        assert!(md.contains("| Add |"));
        assert!(md.contains("yes") || md.contains("no"));
    }

    #[test]
    fn test_coverage_summary() {
        let report = generate_coverage_report();
        let summary = format_coverage_summary(&report);
        assert!(summary.contains("Category Coverage:"));
        assert!(summary.contains("math"));
        assert!(summary.contains("nn"));
        assert!(summary.contains("conv"));
        assert!(summary.contains("shape"));
    }
}
