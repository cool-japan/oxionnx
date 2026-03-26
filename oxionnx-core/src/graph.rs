use std::collections::HashMap;

use crate::Tensor;

/// All ONNX operators currently supported by oxionnx.
#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    // Math
    MatMul,
    Gemm,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sqrt,
    Reciprocal,
    Neg,
    ReduceMean,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    ArgMax,
    ArgMin,
    CumSum,
    Range,
    TopK,
    // Neural network
    Softmax,
    LayerNorm,
    GroupNorm,
    BatchNorm,
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    Erf,
    SiLU,
    HardSigmoid,
    HardSwish,
    RMSNorm,
    // Shape
    Reshape,
    Transpose,
    Squeeze,
    Unsqueeze,
    Flatten,
    Concat,
    Slice,
    Expand,
    Split,
    Tile,
    // Indexing
    Gather,
    GatherElements,
    Where,
    ScatterElements,
    ScatterND,
    // CNN
    Conv,
    MaxPool,
    AveragePool,
    Pad,
    LeakyRelu,
    PRelu,
    Resize,
    GlobalAveragePool,
    GlobalMaxPool,
    // Quantization
    QuantizeLinear,
    DequantizeLinear,
    // Passthrough / misc
    Identity,
    Cast,
    Shape,
    Constant,
    Clip,
    Abs,
    Log,
    Exp,
    // Math (new)
    Ceil,
    Floor,
    Round,
    Sign,
    Mod,
    BitShift,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    VariadicMin,
    VariadicMax,
    VariadicMean,
    VariadicSum,
    // Comparison & Logic
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    And,
    Or,
    Xor,
    Not,
    IsInf,
    IsNaN,
    NonZero,
    // Construction
    ConstantOfShape,
    EyeLike,
    Trilu,
    // Neural network (new)
    LogSoftmax,
    Softplus,
    Softsign,
    Mish,
    Celu,
    Elu,
    Selu,
    ThresholdedRelu,
    InstanceNorm,
    LpNorm,
    MeanVarianceNormalization,
    Dropout,
    // Shape (new)
    DepthToSpace,
    SpaceToDepth,
    ReverseSequence,
    // Indexing (new)
    GatherND,
    OneHot,
    Compress,
    Unique,
    // Advanced ops
    Einsum,
    ConvTranspose,
    NonMaxSuppression,
    // RNN ops
    LSTM,
    GRU,
    // Attention ops
    Attention,
    MultiHeadAttention,
    RotaryEmbedding,
    // Spatial ops
    GridSample,
    RoiAlign,
    // Control flow
    If,
    Loop,
    Scan,
    // ML domain ops
    LinearClassifier,
    LinearRegressor,
    Normalizer,
    Scaler,
    LabelEncoder,
    TreeEnsembleClassifier,
    TreeEnsembleRegressor,
    SVMClassifier,
    SVMRegressor,
    TfIdfVectorizer,
    StringNormalizer,
    // Unknown (logged but skipped gracefully)
    Unknown(String),
}

impl OpKind {
    pub fn parse(s: &str) -> Self {
        match s {
            "MatMul" => Self::MatMul,
            "Gemm" => Self::Gemm,
            "Add" => Self::Add,
            "Sub" => Self::Sub,
            "Mul" => Self::Mul,
            "Div" => Self::Div,
            "Pow" => Self::Pow,
            "Sqrt" => Self::Sqrt,
            "Reciprocal" => Self::Reciprocal,
            "Neg" => Self::Neg,
            "ReduceMean" => Self::ReduceMean,
            "ReduceSum" => Self::ReduceSum,
            "ReduceMax" => Self::ReduceMax,
            "ReduceMin" => Self::ReduceMin,
            "ReduceProd" => Self::ReduceProd,
            "ArgMax" => Self::ArgMax,
            "ArgMin" => Self::ArgMin,
            "CumSum" => Self::CumSum,
            "Range" => Self::Range,
            "TopK" => Self::TopK,
            "Softmax" => Self::Softmax,
            "LayerNormalization" | "LayerNorm" => Self::LayerNorm,
            "GroupNormalization" | "GroupNorm" => Self::GroupNorm,
            "BatchNormalization" => Self::BatchNorm,
            "Gelu" => Self::Gelu,
            "Relu" => Self::Relu,
            "Sigmoid" => Self::Sigmoid,
            "Tanh" => Self::Tanh,
            "Erf" => Self::Erf,
            "Silu" | "SiLU" => Self::SiLU,
            "HardSigmoid" => Self::HardSigmoid,
            "HardSwish" => Self::HardSwish,
            "RMSNorm" | "SimplifiedLayerNormalization" => Self::RMSNorm,
            "Reshape" => Self::Reshape,
            "Transpose" => Self::Transpose,
            "Squeeze" => Self::Squeeze,
            "Unsqueeze" => Self::Unsqueeze,
            "Flatten" => Self::Flatten,
            "Concat" => Self::Concat,
            "Slice" => Self::Slice,
            "Expand" => Self::Expand,
            "Split" => Self::Split,
            "Tile" => Self::Tile,
            "Gather" => Self::Gather,
            "GatherElements" => Self::GatherElements,
            "Where" => Self::Where,
            "ScatterElements" => Self::ScatterElements,
            "ScatterND" => Self::ScatterND,
            "Identity" => Self::Identity,
            "Cast" => Self::Cast,
            "Shape" => Self::Shape,
            "Constant" => Self::Constant,
            "Clip" => Self::Clip,
            "Abs" => Self::Abs,
            "Log" => Self::Log,
            "Exp" => Self::Exp,
            "Conv" => Self::Conv,
            "MaxPool" => Self::MaxPool,
            "AveragePool" => Self::AveragePool,
            "Pad" => Self::Pad,
            "LeakyRelu" => Self::LeakyRelu,
            "PRelu" => Self::PRelu,
            "Resize" => Self::Resize,
            "GlobalAveragePool" => Self::GlobalAveragePool,
            "GlobalMaxPool" => Self::GlobalMaxPool,
            "QuantizeLinear" => Self::QuantizeLinear,
            "DequantizeLinear" => Self::DequantizeLinear,
            "Ceil" => Self::Ceil,
            "Floor" => Self::Floor,
            "Round" => Self::Round,
            "Sign" => Self::Sign,
            "Mod" => Self::Mod,
            "BitShift" => Self::BitShift,
            "Sin" => Self::Sin,
            "Cos" => Self::Cos,
            "Tan" => Self::Tan,
            "Asin" => Self::Asin,
            "Acos" => Self::Acos,
            "Atan" => Self::Atan,
            "Sinh" => Self::Sinh,
            "Cosh" => Self::Cosh,
            "Asinh" => Self::Asinh,
            "Acosh" => Self::Acosh,
            "Atanh" => Self::Atanh,
            "Min" => Self::VariadicMin,
            "Max" => Self::VariadicMax,
            "Mean" => Self::VariadicMean,
            "Sum" => Self::VariadicSum,
            "Equal" => Self::Equal,
            "Greater" => Self::Greater,
            "GreaterOrEqual" => Self::GreaterOrEqual,
            "Less" => Self::Less,
            "LessOrEqual" => Self::LessOrEqual,
            "And" => Self::And,
            "Or" => Self::Or,
            "Xor" => Self::Xor,
            "Not" => Self::Not,
            "IsInf" => Self::IsInf,
            "IsNaN" => Self::IsNaN,
            "NonZero" => Self::NonZero,
            "ConstantOfShape" => Self::ConstantOfShape,
            "EyeLike" => Self::EyeLike,
            "Trilu" => Self::Trilu,
            "LogSoftmax" => Self::LogSoftmax,
            "Softplus" => Self::Softplus,
            "Softsign" => Self::Softsign,
            "Mish" => Self::Mish,
            "Celu" | "CeLU" => Self::Celu,
            "Elu" => Self::Elu,
            "Selu" => Self::Selu,
            "ThresholdedRelu" => Self::ThresholdedRelu,
            "InstanceNormalization" => Self::InstanceNorm,
            "LpNormalization" => Self::LpNorm,
            "MeanVarianceNormalization" => Self::MeanVarianceNormalization,
            "Dropout" => Self::Dropout,
            "DepthToSpace" => Self::DepthToSpace,
            "SpaceToDepth" => Self::SpaceToDepth,
            "ReverseSequence" => Self::ReverseSequence,
            "GatherND" => Self::GatherND,
            "OneHot" => Self::OneHot,
            "Compress" => Self::Compress,
            "Unique" => Self::Unique,
            "Einsum" => Self::Einsum,
            "ConvTranspose" => Self::ConvTranspose,
            "NonMaxSuppression" => Self::NonMaxSuppression,
            "LSTM" => Self::LSTM,
            "GRU" => Self::GRU,
            "Attention" => Self::Attention,
            "MultiHeadAttention" => Self::MultiHeadAttention,
            "RotaryEmbedding" => Self::RotaryEmbedding,
            "GridSample" => Self::GridSample,
            "RoiAlign" => Self::RoiAlign,
            "If" => Self::If,
            "Loop" => Self::Loop,
            "Scan" => Self::Scan,
            // ML domain ops
            "LinearClassifier" | "ai.onnx.ml.LinearClassifier" => Self::LinearClassifier,
            "LinearRegressor" | "ai.onnx.ml.LinearRegressor" => Self::LinearRegressor,
            "Normalizer" | "ai.onnx.ml.Normalizer" => Self::Normalizer,
            "Scaler" | "ai.onnx.ml.Scaler" => Self::Scaler,
            "LabelEncoder" | "ai.onnx.ml.LabelEncoder" => Self::LabelEncoder,
            "TreeEnsembleClassifier" | "ai.onnx.ml.TreeEnsembleClassifier" => {
                Self::TreeEnsembleClassifier
            }
            "TreeEnsembleRegressor" | "ai.onnx.ml.TreeEnsembleRegressor" => {
                Self::TreeEnsembleRegressor
            }
            "SVMClassifier" | "ai.onnx.ml.SVMClassifier" => Self::SVMClassifier,
            "SVMRegressor" | "ai.onnx.ml.SVMRegressor" => Self::SVMRegressor,
            "TfIdfVectorizer" | "ai.onnx.ml.TfIdfVectorizer" => Self::TfIdfVectorizer,
            "StringNormalizer" | "ai.onnx.ml.StringNormalizer" => Self::StringNormalizer,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Return the canonical ONNX op_type string for this variant.
    pub fn as_str(&self) -> &str {
        match self {
            Self::MatMul => "MatMul",
            Self::Gemm => "Gemm",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Pow => "Pow",
            Self::Sqrt => "Sqrt",
            Self::Reciprocal => "Reciprocal",
            Self::Neg => "Neg",
            Self::ReduceMean => "ReduceMean",
            Self::ReduceSum => "ReduceSum",
            Self::ReduceMax => "ReduceMax",
            Self::ReduceMin => "ReduceMin",
            Self::ReduceProd => "ReduceProd",
            Self::ArgMax => "ArgMax",
            Self::ArgMin => "ArgMin",
            Self::CumSum => "CumSum",
            Self::Range => "Range",
            Self::TopK => "TopK",
            Self::Softmax => "Softmax",
            Self::LayerNorm => "LayerNormalization",
            Self::GroupNorm => "GroupNormalization",
            Self::BatchNorm => "BatchNormalization",
            Self::Gelu => "Gelu",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::Erf => "Erf",
            Self::SiLU => "SiLU",
            Self::HardSigmoid => "HardSigmoid",
            Self::HardSwish => "HardSwish",
            Self::RMSNorm => "SimplifiedLayerNormalization",
            Self::Reshape => "Reshape",
            Self::Transpose => "Transpose",
            Self::Squeeze => "Squeeze",
            Self::Unsqueeze => "Unsqueeze",
            Self::Flatten => "Flatten",
            Self::Concat => "Concat",
            Self::Slice => "Slice",
            Self::Expand => "Expand",
            Self::Split => "Split",
            Self::Tile => "Tile",
            Self::Gather => "Gather",
            Self::GatherElements => "GatherElements",
            Self::Where => "Where",
            Self::ScatterElements => "ScatterElements",
            Self::ScatterND => "ScatterND",
            Self::Conv => "Conv",
            Self::MaxPool => "MaxPool",
            Self::AveragePool => "AveragePool",
            Self::Pad => "Pad",
            Self::LeakyRelu => "LeakyRelu",
            Self::PRelu => "PRelu",
            Self::Resize => "Resize",
            Self::GlobalAveragePool => "GlobalAveragePool",
            Self::GlobalMaxPool => "GlobalMaxPool",
            Self::QuantizeLinear => "QuantizeLinear",
            Self::DequantizeLinear => "DequantizeLinear",
            Self::Identity => "Identity",
            Self::Cast => "Cast",
            Self::Shape => "Shape",
            Self::Constant => "Constant",
            Self::Clip => "Clip",
            Self::Abs => "Abs",
            Self::Log => "Log",
            Self::Exp => "Exp",
            Self::Ceil => "Ceil",
            Self::Floor => "Floor",
            Self::Round => "Round",
            Self::Sign => "Sign",
            Self::Mod => "Mod",
            Self::BitShift => "BitShift",
            Self::Sin => "Sin",
            Self::Cos => "Cos",
            Self::Tan => "Tan",
            Self::Asin => "Asin",
            Self::Acos => "Acos",
            Self::Atan => "Atan",
            Self::Sinh => "Sinh",
            Self::Cosh => "Cosh",
            Self::Asinh => "Asinh",
            Self::Acosh => "Acosh",
            Self::Atanh => "Atanh",
            Self::VariadicMin => "Min",
            Self::VariadicMax => "Max",
            Self::VariadicMean => "Mean",
            Self::VariadicSum => "Sum",
            Self::Equal => "Equal",
            Self::Greater => "Greater",
            Self::GreaterOrEqual => "GreaterOrEqual",
            Self::Less => "Less",
            Self::LessOrEqual => "LessOrEqual",
            Self::And => "And",
            Self::Or => "Or",
            Self::Xor => "Xor",
            Self::Not => "Not",
            Self::IsInf => "IsInf",
            Self::IsNaN => "IsNaN",
            Self::NonZero => "NonZero",
            Self::ConstantOfShape => "ConstantOfShape",
            Self::EyeLike => "EyeLike",
            Self::Trilu => "Trilu",
            Self::LogSoftmax => "LogSoftmax",
            Self::Softplus => "Softplus",
            Self::Softsign => "Softsign",
            Self::Mish => "Mish",
            Self::Celu => "Celu",
            Self::Elu => "Elu",
            Self::Selu => "Selu",
            Self::ThresholdedRelu => "ThresholdedRelu",
            Self::InstanceNorm => "InstanceNormalization",
            Self::LpNorm => "LpNormalization",
            Self::MeanVarianceNormalization => "MeanVarianceNormalization",
            Self::Dropout => "Dropout",
            Self::DepthToSpace => "DepthToSpace",
            Self::SpaceToDepth => "SpaceToDepth",
            Self::ReverseSequence => "ReverseSequence",
            Self::GatherND => "GatherND",
            Self::OneHot => "OneHot",
            Self::Compress => "Compress",
            Self::Unique => "Unique",
            Self::Einsum => "Einsum",
            Self::ConvTranspose => "ConvTranspose",
            Self::NonMaxSuppression => "NonMaxSuppression",
            Self::LSTM => "LSTM",
            Self::GRU => "GRU",
            Self::Attention => "Attention",
            Self::MultiHeadAttention => "MultiHeadAttention",
            Self::RotaryEmbedding => "RotaryEmbedding",
            Self::GridSample => "GridSample",
            Self::RoiAlign => "RoiAlign",
            Self::If => "If",
            Self::Loop => "Loop",
            Self::Scan => "Scan",
            Self::LinearClassifier => "LinearClassifier",
            Self::LinearRegressor => "LinearRegressor",
            Self::Normalizer => "Normalizer",
            Self::Scaler => "Scaler",
            Self::LabelEncoder => "LabelEncoder",
            Self::TreeEnsembleClassifier => "TreeEnsembleClassifier",
            Self::TreeEnsembleRegressor => "TreeEnsembleRegressor",
            Self::SVMClassifier => "SVMClassifier",
            Self::SVMRegressor => "SVMRegressor",
            Self::TfIdfVectorizer => "TfIdfVectorizer",
            Self::StringNormalizer => "StringNormalizer",
            Self::Unknown(s) => s.as_str(),
        }
    }
}

/// Parsed attributes for a node.
#[derive(Debug, Clone, Default)]
pub struct Attributes {
    pub floats: HashMap<String, f32>,
    pub ints: HashMap<String, i64>,
    pub strings: HashMap<String, String>,
    pub tensors: HashMap<String, Tensor>,
    pub float_lists: HashMap<String, Vec<f32>>,
    pub int_lists: HashMap<String, Vec<i64>>,
    pub string_lists: HashMap<String, Vec<String>>,
    pub graphs: HashMap<String, Graph>,
}

impl Attributes {
    pub fn f(&self, name: &str, default: f32) -> f32 {
        self.floats.get(name).copied().unwrap_or(default)
    }
    pub fn i(&self, name: &str, default: i64) -> i64 {
        self.ints.get(name).copied().unwrap_or(default)
    }
    pub fn ints(&self, name: &str) -> &[i64] {
        self.int_lists
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    pub fn s(&self, name: &str) -> &str {
        self.strings.get(name).map(|s| s.as_str()).unwrap_or("")
    }
    /// Return a string list attribute by name (empty slice if absent).
    pub fn string_list(&self, name: &str) -> &[String] {
        self.string_lists
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    /// Return a reference to a subgraph attribute by name.
    pub fn graph(&self, name: &str) -> Option<&Graph> {
        self.graphs.get(name)
    }
}

/// A single computation node in the ONNX graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: OpKind,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Attributes,
}

/// The full computation graph.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

impl Graph {
    /// Topological sort (Kahn's algorithm).
    /// Returns node indices in execution order given the set of initially-known names.
    pub fn topological_sort(&self, known: &[String]) -> Vec<usize> {
        let n = self.nodes.len();

        // For each node, count how many of its inputs are NOT yet available
        let mut in_degree = vec![0usize; n];
        // Map from output name -> node index that produces it
        let mut producer: HashMap<&str, usize> = HashMap::new();

        for (i, node) in self.nodes.iter().enumerate() {
            for out in &node.outputs {
                producer.insert(out.as_str(), i);
            }
        }

        let known_set: std::collections::HashSet<&str> = known.iter().map(|s| s.as_str()).collect();

        for (i, node) in self.nodes.iter().enumerate() {
            for inp in &node.inputs {
                if inp.is_empty() {
                    continue;
                } // optional input
                if !known_set.contains(inp.as_str()) && producer.contains_key(inp.as_str()) {
                    in_degree[i] += 1;
                }
            }
        }

        // Track which nodes each node's outputs feed into
        let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            for inp in &node.inputs {
                if let Some(&prod_idx) = producer.get(inp.as_str()) {
                    dependents.entry(prod_idx).or_default().push(i);
                }
            }
        }

        let mut queue: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            if let Some(deps) = dependents.get(&idx) {
                for &dep in deps {
                    in_degree[dep] -= 1;
                    if in_degree[dep] == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }

        // If there are remaining nodes (cycles or disconnected), append them in original order
        if order.len() < n {
            for i in 0..n {
                if !order.contains(&i) {
                    order.push(i);
                }
            }
        }

        order
    }
}
