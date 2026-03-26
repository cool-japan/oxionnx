//! Public ONNX protobuf type definitions.

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
                    eprintln!("TensorProto: unsupported dtype {dt}, returning zeros");
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

#[derive(Debug, Default, Clone)]
pub struct GraphProto {
    pub nodes: Vec<NodeProto>,
    pub name: String,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
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
    pub model_version: i64,
    pub doc_string: String,
    pub opset_imports: Vec<OpsetImport>,
    pub training_info: Vec<TrainingInfo>,
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
