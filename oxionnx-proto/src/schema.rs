//! Operator schema definitions and validation.
//!
//! Validates that each node's inputs/outputs match the expected ONNX operator spec.

use std::collections::HashMap;

/// Expected input/output specification for an operator.
#[derive(Debug, Clone)]
pub struct OpSchema {
    /// Operator type name.
    pub op_type: String,
    /// Minimum number of required inputs.
    pub min_inputs: usize,
    /// Maximum number of inputs (None = unlimited).
    pub max_inputs: Option<usize>,
    /// Minimum number of outputs.
    pub min_outputs: usize,
    /// Maximum number of outputs (None = unlimited).
    pub max_outputs: Option<usize>,
}

/// Schema validation result for a single node.
#[derive(Debug, Clone)]
pub struct SchemaViolation {
    /// Name of the node that violated its schema.
    pub node_name: String,
    /// Operator type of the node.
    pub op_type: String,
    /// Human-readable description of the violation.
    pub message: String,
}

/// Build the default schema registry covering all standard ONNX operators.
pub fn default_schemas() -> HashMap<String, OpSchema> {
    let mut schemas = HashMap::new();

    // ---- Math ops ----
    add_schema(&mut schemas, "MatMul", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Gemm", 2, Some(3), 1, Some(1));
    add_schema(&mut schemas, "Add", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Sub", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Mul", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Div", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Pow", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Mod", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "BitShift", 2, Some(2), 1, Some(1));

    // ---- Unary ops: 1 input, 1 output ----
    for op in &[
        "Relu",
        "Sigmoid",
        "Tanh",
        "Exp",
        "Log",
        "Sqrt",
        "Abs",
        "Neg",
        "Ceil",
        "Floor",
        "Round",
        "Sign",
        "Erf",
        "Sin",
        "Cos",
        "Tan",
        "Asin",
        "Acos",
        "Atan",
        "Sinh",
        "Cosh",
        "Asinh",
        "Acosh",
        "Atanh",
        "Softplus",
        "Softsign",
        "Mish",
        "Identity",
        "Not",
        "IsInf",
        "IsNaN",
        "Reciprocal",
        "Gelu",
        "SiLU",
        "NonZero",
        "BitwiseNot",
        "Size",
        "Hardmax",
        "Shrink",
    ] {
        add_schema(&mut schemas, op, 1, Some(1), 1, Some(1));
    }

    // ---- Activations with parameters ----
    add_schema(&mut schemas, "LeakyRelu", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "PRelu", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Clip", 1, Some(3), 1, Some(1));
    add_schema(&mut schemas, "Elu", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Selu", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Celu", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "HardSigmoid", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "HardSwish", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "ThresholdedRelu", 1, Some(1), 1, Some(1));

    // ---- Normalization ----
    add_schema(&mut schemas, "BatchNormalization", 5, Some(5), 1, Some(5));
    add_schema(&mut schemas, "LayerNormalization", 2, Some(3), 1, Some(3));
    add_schema(&mut schemas, "GroupNormalization", 3, Some(3), 1, Some(1));
    add_schema(
        &mut schemas,
        "InstanceNormalization",
        3,
        Some(3),
        1,
        Some(1),
    );
    add_schema(&mut schemas, "LpNormalization", 1, Some(1), 1, Some(1));
    add_schema(
        &mut schemas,
        "MeanVarianceNormalization",
        1,
        Some(1),
        1,
        Some(1),
    );

    // ---- Conv / Pool ----
    add_schema(&mut schemas, "Conv", 2, Some(3), 1, Some(1));
    add_schema(&mut schemas, "ConvTranspose", 2, Some(3), 1, Some(1));
    add_schema(&mut schemas, "MaxPool", 1, Some(1), 1, Some(2));
    add_schema(&mut schemas, "AveragePool", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "GlobalAveragePool", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "GlobalMaxPool", 1, Some(1), 1, Some(1));

    // ---- Shape ops ----
    add_schema(&mut schemas, "Reshape", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Transpose", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Concat", 1, None, 1, Some(1)); // variadic inputs
    add_schema(&mut schemas, "Slice", 3, Some(5), 1, Some(1));
    add_schema(&mut schemas, "Squeeze", 1, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Unsqueeze", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Flatten", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Expand", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Split", 1, Some(2), 1, None); // variadic outputs
    add_schema(&mut schemas, "Pad", 2, Some(3), 1, Some(1));
    add_schema(&mut schemas, "Tile", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "DepthToSpace", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "SpaceToDepth", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "ReverseSequence", 2, Some(2), 1, Some(1));

    // ---- Indexing ----
    add_schema(&mut schemas, "Gather", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "GatherElements", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "GatherND", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "ScatterElements", 3, Some(3), 1, Some(1));
    add_schema(&mut schemas, "ScatterND", 3, Some(3), 1, Some(1));
    add_schema(&mut schemas, "Where", 3, Some(3), 1, Some(1));
    add_schema(&mut schemas, "OneHot", 3, Some(3), 1, Some(1));
    add_schema(&mut schemas, "Compress", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Unique", 1, Some(1), 1, Some(4));

    // ---- Softmax family ----
    add_schema(&mut schemas, "Softmax", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "LogSoftmax", 1, Some(1), 1, Some(1));

    // ---- Dropout ----
    add_schema(&mut schemas, "Dropout", 1, Some(3), 1, Some(2));

    // ---- Resize / Shape / Cast / Constant ----
    add_schema(&mut schemas, "Resize", 1, Some(4), 1, Some(1));
    add_schema(&mut schemas, "Shape", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Cast", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Constant", 0, Some(0), 1, Some(1));
    add_schema(&mut schemas, "ConstantOfShape", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "EyeLike", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Trilu", 1, Some(2), 1, Some(1));

    // ---- Reduce ops ----
    for op in &[
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceSumSquare",
    ] {
        add_schema(&mut schemas, op, 1, Some(2), 1, Some(1));
    }
    add_schema(&mut schemas, "ArgMax", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "ArgMin", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "CumSum", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "Range", 3, Some(3), 1, Some(1));
    add_schema(&mut schemas, "TopK", 2, Some(2), 2, Some(2));

    // ---- Quantization ----
    add_schema(&mut schemas, "QuantizeLinear", 2, Some(3), 1, Some(1));
    add_schema(&mut schemas, "DequantizeLinear", 2, Some(3), 1, Some(1));

    // ---- Comparison (binary -> 1 output) ----
    for op in &[
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "And",
        "Or",
        "Xor",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
    ] {
        add_schema(&mut schemas, op, 2, Some(2), 1, Some(1));
    }

    // ---- Variadic math ----
    for op in &["Min", "Max", "Mean", "Sum"] {
        add_schema(&mut schemas, op, 1, None, 1, Some(1));
    }

    // ---- Control flow ----
    add_schema(&mut schemas, "If", 1, Some(1), 1, None);
    add_schema(&mut schemas, "Loop", 2, None, 0, None);
    add_schema(&mut schemas, "Scan", 1, None, 1, None);

    // ---- RNN ----
    add_schema(&mut schemas, "LSTM", 3, Some(8), 0, Some(3));
    add_schema(&mut schemas, "GRU", 3, Some(6), 0, Some(2));

    // ---- Advanced ----
    add_schema(&mut schemas, "Einsum", 1, None, 1, Some(1));
    add_schema(&mut schemas, "NonMaxSuppression", 2, Some(5), 1, Some(1));
    add_schema(&mut schemas, "GridSample", 2, Some(2), 1, Some(1));
    add_schema(&mut schemas, "RoiAlign", 3, Some(3), 1, Some(1));

    // ---- Attention (custom) ----
    add_schema(&mut schemas, "Attention", 3, Some(4), 1, Some(1));
    add_schema(&mut schemas, "MultiHeadAttention", 3, Some(6), 1, Some(1));
    add_schema(&mut schemas, "RotaryEmbedding", 2, Some(3), 1, Some(1));

    // ---- ML domain ----
    add_schema(&mut schemas, "LinearClassifier", 1, Some(1), 2, Some(2));
    add_schema(&mut schemas, "LinearRegressor", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Normalizer", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "Scaler", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "LabelEncoder", 1, Some(1), 1, Some(1));
    add_schema(
        &mut schemas,
        "TreeEnsembleClassifier",
        1,
        Some(1),
        2,
        Some(2),
    );
    add_schema(
        &mut schemas,
        "TreeEnsembleRegressor",
        1,
        Some(1),
        1,
        Some(1),
    );
    add_schema(&mut schemas, "SVMClassifier", 1, Some(1), 2, Some(2));
    add_schema(&mut schemas, "SVMRegressor", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "TfIdfVectorizer", 1, Some(1), 1, Some(1));
    add_schema(&mut schemas, "StringNormalizer", 1, Some(1), 1, Some(1));

    // ---- RMSNorm (SimplifiedLayerNormalization) ----
    add_schema(
        &mut schemas,
        "SimplifiedLayerNormalization",
        2,
        Some(3),
        1,
        Some(1),
    );

    schemas
}

fn add_schema(
    schemas: &mut HashMap<String, OpSchema>,
    op_type: &str,
    min_in: usize,
    max_in: Option<usize>,
    min_out: usize,
    max_out: Option<usize>,
) {
    schemas.insert(
        op_type.to_string(),
        OpSchema {
            op_type: op_type.to_string(),
            min_inputs: min_in,
            max_inputs: max_in,
            min_outputs: min_out,
            max_outputs: max_out,
        },
    );
}

/// Validate all nodes in a graph against the schema registry.
///
/// Unknown operators (those without a schema entry) are silently skipped.
pub fn validate_schemas(
    nodes: &[oxionnx_core::graph::Node],
    schemas: &HashMap<String, OpSchema>,
) -> Vec<SchemaViolation> {
    let mut violations = Vec::new();

    for node in nodes {
        let op_name = node.op.as_str();
        let Some(schema) = schemas.get(op_name) else {
            continue;
        };

        // Count non-empty inputs
        let input_count = node.inputs.iter().filter(|i| !i.is_empty()).count();
        let output_count = node.outputs.iter().filter(|o| !o.is_empty()).count();

        if input_count < schema.min_inputs {
            violations.push(SchemaViolation {
                node_name: node.name.clone(),
                op_type: op_name.to_string(),
                message: format!(
                    "too few inputs: got {}, expected at least {}",
                    input_count, schema.min_inputs
                ),
            });
        }

        if let Some(max) = schema.max_inputs {
            if input_count > max {
                violations.push(SchemaViolation {
                    node_name: node.name.clone(),
                    op_type: op_name.to_string(),
                    message: format!(
                        "too many inputs: got {}, expected at most {}",
                        input_count, max
                    ),
                });
            }
        }

        if output_count < schema.min_outputs {
            violations.push(SchemaViolation {
                node_name: node.name.clone(),
                op_type: op_name.to_string(),
                message: format!(
                    "too few outputs: got {}, expected at least {}",
                    output_count, schema.min_outputs
                ),
            });
        }

        if let Some(max) = schema.max_outputs {
            if output_count > max {
                violations.push(SchemaViolation {
                    node_name: node.name.clone(),
                    op_type: op_name.to_string(),
                    message: format!(
                        "too many outputs: got {}, expected at most {}",
                        output_count, max
                    ),
                });
            }
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_node(op: OpKind, inputs: Vec<&str>, outputs: Vec<&str>) -> Node {
        Node {
            op,
            name: "test_node".to_string(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            attrs: Attributes::default(),
        }
    }

    #[test]
    fn test_schema_valid_relu() {
        let schemas = default_schemas();
        let node = make_node(OpKind::Relu, vec!["x"], vec!["y"]);
        let violations = validate_schemas(&[node], &schemas);
        assert!(violations.is_empty(), "Relu(1 in, 1 out) should be valid");
    }

    #[test]
    fn test_schema_too_few_inputs() {
        let schemas = default_schemas();
        let node = make_node(OpKind::MatMul, vec!["a"], vec!["out"]);
        let violations = validate_schemas(&[node], &schemas);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("too few inputs"));
    }

    #[test]
    fn test_schema_too_many_inputs() {
        let schemas = default_schemas();
        let node = make_node(OpKind::Add, vec!["a", "b", "c"], vec!["out"]);
        let violations = validate_schemas(&[node], &schemas);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("too many inputs"));
    }

    #[test]
    fn test_schema_variadic() {
        let schemas = default_schemas();
        let node = make_node(OpKind::Concat, vec!["a", "b", "c", "d", "e"], vec!["out"]);
        let violations = validate_schemas(&[node], &schemas);
        assert!(
            violations.is_empty(),
            "Concat with 5 inputs should be valid (variadic)"
        );
    }

    #[test]
    fn test_schema_unknown_op() {
        let schemas = default_schemas();
        let node = make_node(
            OpKind::Unknown("CustomOp".to_string()),
            vec!["a"],
            vec!["b"],
        );
        let violations = validate_schemas(&[node], &schemas);
        assert!(
            violations.is_empty(),
            "Unknown ops should be silently skipped"
        );
    }

    #[test]
    fn test_default_schemas_coverage() {
        let schemas = default_schemas();

        // Verify key ops are present
        let expected_ops = [
            "MatMul",
            "Gemm",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Relu",
            "Sigmoid",
            "Tanh",
            "Conv",
            "MaxPool",
            "BatchNormalization",
            "LayerNormalization",
            "Reshape",
            "Transpose",
            "Concat",
            "Softmax",
            "LSTM",
            "GRU",
            "Gather",
            "Where",
            "QuantizeLinear",
            "DequantizeLinear",
            "Einsum",
            "Shape",
            "Cast",
        ];

        for op in &expected_ops {
            assert!(
                schemas.contains_key(*op),
                "Schema registry should contain {}",
                op
            );
        }

        // Verify the registry has a substantial number of entries
        assert!(
            schemas.len() >= 100,
            "Schema registry should have at least 100 entries, got {}",
            schemas.len()
        );
    }
}
