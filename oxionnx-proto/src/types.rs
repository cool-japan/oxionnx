//! Public ONNX protobuf type definitions.

use oxionnx_core::dtype::DType;
use oxionnx_core::graph::TensorInfo;
use oxionnx_core::tensor::{from_f16_bytes, from_f32_bytes, from_f32_vec, from_i64_bytes};
use oxionnx_core::Tensor;

#[derive(Debug, Default, Clone)]
pub struct TensorProto {
    pub dims: Vec<i64>,
    pub data_type: i32, // 1=float32, 10=float16, 7=int64, 6=int32
    pub name: String,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub int64_data: Vec<i64>,
    pub double_data: Vec<f64>,
    pub raw_data: Vec<u8>,
    pub data_location: i32,                   // 0=default (inline), 1=external
    pub external_data: Vec<(String, String)>, // key-value pairs: "location", "offset", "length", "checksum"
}

impl TensorProto {
    /// Convert to our Tensor type (always f32).
    pub fn to_tensor(&self) -> Tensor {
        let shape: Vec<usize> = self.dims.iter().map(|&d| d as usize).collect();
        if !self.raw_data.is_empty() {
            return match self.data_type {
                1 => from_f32_bytes(&self.raw_data, shape),
                10 => from_f16_bytes(&self.raw_data, shape),
                7 => from_i64_bytes(&self.raw_data, shape),
                6 => {
                    let data: Vec<f32> = self
                        .raw_data
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                        .collect();
                    Tensor::new(data, shape)
                }
                dt => {
                    tracing::warn!(
                        dtype = dt,
                        "TensorProto: unsupported dtype, returning zeros"
                    );
                    Tensor::zeros(&shape)
                }
            };
        }
        if !self.float_data.is_empty() {
            return from_f32_vec(self.float_data.clone(), shape);
        }
        if !self.int64_data.is_empty() {
            let data: Vec<f32> = self.int64_data.iter().map(|&v| v as f32).collect();
            return Tensor::new(data, shape);
        }
        if !self.int32_data.is_empty() {
            let data: Vec<f32> = self.int32_data.iter().map(|&v| v as f32).collect();
            return Tensor::new(data, shape);
        }
        if !self.double_data.is_empty() {
            let data: Vec<f32> = self.double_data.iter().map(|&v| v as f32).collect();
            return Tensor::new(data, shape);
        }
        Tensor::zeros(&shape)
    }
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct AttributeValue {
    pub f: f32,
    pub i: i64,
    pub s: String,
    pub t: Option<TensorProto>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<String>,
    pub attr_type: i32,
}

#[derive(Debug, Default, Clone)]
pub struct AttributeProto {
    pub name: String,
    pub value: AttributeValue,
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct NodeProto {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub name: String,
    pub op_type: String,
    pub attributes: Vec<AttributeProto>,
    pub domain: String,
}

// ─────────────────────────────────────────────────────────────────

/// Parsed representation of an ONNX `ValueInfoProto`.
///
/// Holds the tensor name, element type, and optional static shape information
/// extracted from the graph's `input` / `output` fields.
#[derive(Debug, Default, Clone)]
pub struct ValueInfoProto {
    /// Tensor name.
    pub name: String,
    /// ONNX element type (1=float32, 7=int64, 10=float16, …). 0 = unknown.
    pub elem_type: i32,
    /// Per-dimension sizes. `None` = symbolic / dynamic dimension.
    pub shape: Vec<Option<i64>>,
    /// Symbolic dimension parameter names (e.g. "batch_size", "seq_len"). Parallel to `shape`.
    pub dim_params: Vec<Option<String>>,
}

impl ValueInfoProto {
    /// Convert ONNX elem_type integer to our `DType`, defaulting to `F32` for unknown types.
    fn dtype_from_elem_type(elem_type: i32) -> DType {
        match elem_type {
            1 => DType::F32,
            2 => DType::U8,
            3 => DType::I8,
            4 => DType::U16,
            5 => DType::I16,
            6 => DType::I32,
            7 => DType::I64,
            10 => DType::F16,
            11 => DType::F64,
            12 => DType::U32,
            13 => DType::U64,
            16 => DType::BF16,
            _ => DType::F32,
        }
    }

    /// Convert to the runtime `TensorInfo` type used by `Session`.
    pub fn to_tensor_info(&self) -> TensorInfo {
        let shape = self
            .shape
            .iter()
            .map(|dim| match dim {
                Some(d) if *d > 0 => Some(*d as usize),
                _ => None,
            })
            .collect();
        let dim_params = self.dim_params.clone();
        TensorInfo {
            name: self.name.clone(),
            dtype: Self::dtype_from_elem_type(self.elem_type),
            shape,
            dim_params,
        }
    }
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct GraphProto {
    pub nodes: Vec<NodeProto>,
    pub name: String,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Full `ValueInfoProto` data for graph inputs (parallel to `inputs`).
    pub input_value_infos: Vec<ValueInfoProto>,
    /// Full `ValueInfoProto` data for graph outputs (parallel to `outputs`).
    pub output_value_infos: Vec<ValueInfoProto>,
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OpsetImport {
    pub domain: String, // "" = default ONNX domain
    pub version: i64,
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct ModelProto {
    pub ir_version: i64,
    pub graph: GraphProto,
    pub opset_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub opset_imports: Vec<OpsetImport>,
    pub training_info: Vec<TrainingInfo>,
    /// User-defined key-value metadata pairs (from metadata_props field 14).
    pub metadata_props: Vec<(String, String)>,
}

// ─────────────────────────────────────────────────────────────────

/// ONNX training information.
#[derive(Debug, Clone, Default)]
pub struct TrainingInfo {
    /// Training algorithm (e.g., "SGD", "Adam").
    pub algorithm: String,
    /// Learning rate.
    pub learning_rate: f64,
    /// Training graph (if present).
    pub training_graph: Option<GraphProto>,
    /// Initialization bindings: maps parameter name to initializer name.
    pub initialization_bindings: Vec<(String, String)>,
    /// Update bindings: maps parameter name to gradient name.
    pub update_bindings: Vec<(String, String)>,
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct TensorExternalData {
    pub data_location: i32,                   // 0=default (inline), 1=external
    pub external_data: Vec<(String, String)>, // key-value pairs
}
