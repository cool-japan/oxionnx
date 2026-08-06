//! Public ONNX protobuf type definitions.

use crate::reader::{bytes_to_f32, dtype_code, dtype_size_bytes, ReaderError};
use oxionnx_core::dtype::DType;
use oxionnx_core::graph::TensorInfo;
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
    /// `string_data` (wire field 6): one raw byte string per repeated occurrence.
    pub string_data: Vec<Vec<u8>>,
    /// `uint64_data` (wire field 11): packed varint u64/u32 payloads.
    pub uint64_data: Vec<u64>,
    pub raw_data: Vec<u8>,
    pub data_location: i32,                   // 0=default (inline), 1=external
    pub external_data: Vec<(String, String)>, // key-value pairs: "location", "offset", "length", "checksum"
}

/// Element count above which the infallible [`TensorProto::to_tensor`] shim
/// refuses to materialize a zero-filled placeholder for an undecodable
/// tensor (16 Mi elements = 64 MiB), so a crafted `dims` on a tiny model
/// cannot be amplified into an unbounded allocation.
const MAX_PLACEHOLDER_ELEMS: usize = 1 << 24;

/// Reinterpret one `int32_data` entry according to the tensor's `data_type`.
///
/// Per the ONNX spec, `FLOAT16`, `BFLOAT16`, the 8/16-bit integer families,
/// and `BOOL` are all stored bit-wise inside `int32_data`, so a raw
/// `value as f32` would turn the FP16 pattern `0x3C00` (= 1.0) into 15360.0.
fn int32_field_to_f32(value: i32, data_type: i32) -> f32 {
    match data_type {
        dtype_code::FLOAT16 => half::f16::from_bits(value as u16).to_f32(),
        dtype_code::BFLOAT16 => half::bf16::from_bits(value as u16).to_f32(),
        dtype_code::INT8 => f32::from(value as i8),
        dtype_code::UINT8 => f32::from(value as u8),
        dtype_code::INT16 => f32::from(value as i16),
        dtype_code::UINT16 => f32::from(value as u16),
        dtype_code::BOOL => {
            if value != 0 {
                1.0
            } else {
                0.0
            }
        }
        _ => value as f32,
    }
}

impl TensorProto {
    /// Validate `dims` and return `(shape, element_count)`.
    ///
    /// `dims` comes straight off the wire, so a negative entry (`-1` encodes
    /// as a 10-byte varint) or an overflowing product must be a typed error
    /// rather than a `usize` wrap that later panics with "capacity overflow".
    fn checked_shape(&self) -> Result<(Vec<usize>, usize), ReaderError> {
        let mut shape = Vec::with_capacity(self.dims.len());
        let mut count: usize = 1;
        for &dim in &self.dims {
            let dim_usize = usize::try_from(dim).map_err(|_| ReaderError::InvalidDim {
                tensor: self.name.clone(),
                dim,
            })?;
            count = count
                .checked_mul(dim_usize)
                .ok_or_else(|| ReaderError::ShapeOverflow {
                    tensor: self.name.clone(),
                    dims: self.dims.clone(),
                })?;
            shape.push(dim_usize);
        }
        Ok((shape, count))
    }

    /// Decode `bytes` (ONNX `raw_data` layout: little-endian, tightly packed)
    /// with this tensor's `data_type` and validate it against `dims`.
    ///
    /// Shared by the inline `raw_data` path and by the external-data loader so
    /// both honour exactly one dtype table.
    pub fn tensor_from_raw_bytes(&self, bytes: &[u8]) -> Result<Tensor, ReaderError> {
        let (shape, count) = self.checked_shape()?;
        let data = bytes_to_f32(bytes, self.data_type, &self.name)?;
        self.finish(data, shape, count)
    }

    /// Fallible conversion into the runtime `Tensor` (always f32).
    ///
    /// Every ONNX numeric dtype is decoded, from `raw_data` as well as from
    /// the typed repeated fields. A dtype without an `f32` representation, a
    /// malformed shape, or a payload whose length disagrees with `dims`
    /// produces a typed [`ReaderError`] — never a silently zero-filled tensor.
    pub fn try_to_tensor(&self) -> Result<Tensor, ReaderError> {
        if !self.raw_data.is_empty() {
            return self.tensor_from_raw_bytes(&self.raw_data);
        }
        let (shape, count) = self.checked_shape()?;
        match self.decode_typed_fields()? {
            Some(data) => self.finish(data, shape, count),
            // A genuinely empty tensor (some dim is 0) legitimately carries no payload.
            None if count == 0 => Ok(Tensor::new(Vec::new(), shape)),
            None => Err(ReaderError::MissingTensorData {
                tensor: self.name.clone(),
                expected: count,
            }),
        }
    }

    /// Convert to our Tensor type (always f32).
    ///
    /// Infallible legacy shim: prefer [`Self::try_to_tensor`], which reports
    /// why an initializer could not be decoded. On failure this logs the error
    /// and yields a bounded placeholder instead of panicking.
    pub fn to_tensor(&self) -> Tensor {
        match self.try_to_tensor() {
            Ok(tensor) => tensor,
            Err(err) => {
                tracing::error!(
                    tensor = %self.name,
                    dtype = self.data_type,
                    error = %err,
                    "TensorProto: cannot decode initializer, substituting a placeholder",
                );
                match self.checked_shape() {
                    Ok((shape, count)) if count <= MAX_PLACEHOLDER_ELEMS => Tensor::zeros(&shape),
                    _ => Tensor::new(Vec::new(), vec![0]),
                }
            }
        }
    }

    /// Validate the decoded element count against `dims` and build the tensor.
    fn finish(
        &self,
        data: Vec<f32>,
        shape: Vec<usize>,
        count: usize,
    ) -> Result<Tensor, ReaderError> {
        if data.len() != count {
            return Err(ReaderError::ElementCountMismatch {
                tensor: self.name.clone(),
                dims: self.dims.clone(),
                expected: count,
                found: data.len(),
            });
        }
        Ok(Tensor::new(data, shape))
    }

    /// Decode the ONNX typed repeated fields (used when `raw_data` is empty).
    ///
    /// Each dtype has exactly one canonical field per the spec; small integer,
    /// boolean, and half-precision dtypes are bit-packed into `int32_data`.
    /// Returns `Ok(None)` when no field carries data.
    fn decode_typed_fields(&self) -> Result<Option<Vec<f32>>, ReaderError> {
        // Dtypes with no f32 representation (STRING, complex, FLOAT8*, 4-bit)
        // must fail loudly. `UNDEFINED` is tolerated: producers of hand-built
        // protos omit `data_type`, and the populated field then implies it.
        if self.data_type != dtype_code::UNDEFINED && dtype_size_bytes(self.data_type).is_none() {
            return Err(ReaderError::UnsupportedDtype {
                tensor: self.name.clone(),
                dtype: self.data_type,
            });
        }

        let canonical: Option<Vec<f32>> = match self.data_type {
            dtype_code::FLOAT32 if !self.float_data.is_empty() => Some(self.float_data.clone()),
            dtype_code::DOUBLE if !self.double_data.is_empty() => {
                Some(self.double_data.iter().map(|&v| v as f32).collect())
            }
            dtype_code::INT64 if !self.int64_data.is_empty() => {
                Some(self.int64_data.iter().map(|&v| v as f32).collect())
            }
            dtype_code::UINT32 | dtype_code::UINT64 if !self.uint64_data.is_empty() => {
                Some(self.uint64_data.iter().map(|&v| v as f32).collect())
            }
            dtype_code::INT32
            | dtype_code::INT8
            | dtype_code::UINT8
            | dtype_code::INT16
            | dtype_code::UINT16
            | dtype_code::BOOL
            | dtype_code::FLOAT16
            | dtype_code::BFLOAT16
                if !self.int32_data.is_empty() =>
            {
                let dt = self.data_type;
                Some(
                    self.int32_data
                        .iter()
                        .map(|&v| int32_field_to_f32(v, dt))
                        .collect(),
                )
            }
            _ => None,
        };
        if canonical.is_some() {
            return Ok(canonical);
        }

        // Fallback for `UNDEFINED` and for producers that used a non-canonical
        // field: take whichever field carries data, converting per `data_type`.
        let inferred: Option<Vec<f32>> = if !self.float_data.is_empty() {
            Some(self.float_data.clone())
        } else if !self.int64_data.is_empty() {
            Some(self.int64_data.iter().map(|&v| v as f32).collect())
        } else if !self.int32_data.is_empty() {
            let dt = self.data_type;
            Some(
                self.int32_data
                    .iter()
                    .map(|&v| int32_field_to_f32(v, dt))
                    .collect(),
            )
        } else if !self.double_data.is_empty() {
            Some(self.double_data.iter().map(|&v| v as f32).collect())
        } else if !self.uint64_data.is_empty() {
            Some(self.uint64_data.iter().map(|&v| v as f32).collect())
        } else {
            None
        };
        if inferred.is_some() && self.data_type != dtype_code::UNDEFINED {
            tracing::debug!(
                tensor = %self.name,
                dtype = self.data_type,
                "TensorProto: dtype stored in a non-canonical typed field",
            );
        }
        Ok(inferred)
    }
}

// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct AttributeValue {
    pub f: f32,
    pub i: i64,
    pub s: String,
    pub t: Option<TensorProto>,
    /// Subgraph attribute (ONNX AttributeType::GRAPH, wire field 6).
    /// Boxed to avoid an unbounded recursive type via GraphProto→NodeProto→AttributeProto→AttributeValue.
    pub g: Option<Box<GraphProto>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<String>,
    /// `tensors` attribute (ONNX AttributeType::TENSORS, wire field 10).
    pub tensors: Vec<TensorProto>,
    /// `graphs` attribute (ONNX AttributeType::GRAPHS, wire field 11).
    pub graphs: Vec<GraphProto>,
    /// `ref_attr_name` (wire field 21): inside a function body, the name of the
    /// enclosing function attribute this attribute takes its value from.
    pub ref_attr_name: String,
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
    /// `overload` (wire field 8, IR ≥ 10): selects which overload of a
    /// same-`(domain, op_type)` model-local function this call site invokes.
    /// `""` means the unnamed (default) overload — see
    /// `crate::model::inline_local_functions`.
    pub overload: String,
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
    /// Map an ONNX `elem_type` code onto our `DType`.
    ///
    /// Returns `None` for codes this engine has no `DType` for — `UNDEFINED`
    /// (0), `STRING` (8), `COMPLEX64/128` (14/15), the `FLOAT8*` family
    /// (17–20) and `UINT4/INT4/FLOAT4` (21–23) — so callers can tell an
    /// actual float32 tensor from one whose declared type is unrecognized.
    fn dtype_from_elem_type(elem_type: i32) -> Option<DType> {
        match elem_type {
            dtype_code::FLOAT32 => Some(DType::F32),
            dtype_code::UINT8 => Some(DType::U8),
            dtype_code::INT8 => Some(DType::I8),
            dtype_code::UINT16 => Some(DType::U16),
            dtype_code::INT16 => Some(DType::I16),
            dtype_code::INT32 => Some(DType::I32),
            dtype_code::INT64 => Some(DType::I64),
            dtype_code::BOOL => Some(DType::Bool),
            dtype_code::FLOAT16 => Some(DType::F16),
            dtype_code::DOUBLE => Some(DType::F64),
            dtype_code::UINT32 => Some(DType::U32),
            dtype_code::UINT64 => Some(DType::U64),
            dtype_code::BFLOAT16 => Some(DType::BF16),
            _ => None,
        }
    }

    /// Convert to the runtime `TensorInfo` type used by `Session`.
    ///
    /// An `elem_type` this engine does not model is reported as `F32` (the
    /// runtime's only computation dtype) after a warning; use
    /// [`Self::try_to_tensor_info`] when the distinction matters.
    pub fn to_tensor_info(&self) -> TensorInfo {
        let dtype = Self::dtype_from_elem_type(self.elem_type).unwrap_or_else(|| {
            if self.elem_type == dtype_code::UNDEFINED {
                tracing::debug!(name = %self.name, "value info has no declared element type");
            } else {
                tracing::warn!(
                    name = %self.name,
                    elem_type = self.elem_type,
                    "unrecognized ONNX elem_type, reporting float32 metadata",
                );
            }
            DType::F32
        });
        self.tensor_info_with_dtype(dtype)
    }

    /// Like [`Self::to_tensor_info`], but reports an unrecognized `elem_type`
    /// as [`ReaderError::UnsupportedDtype`] instead of defaulting to `F32`.
    pub fn try_to_tensor_info(&self) -> Result<TensorInfo, ReaderError> {
        let dtype = Self::dtype_from_elem_type(self.elem_type).ok_or_else(|| {
            ReaderError::UnsupportedDtype {
                tensor: self.name.clone(),
                dtype: self.elem_type,
            }
        })?;
        Ok(self.tensor_info_with_dtype(dtype))
    }

    fn tensor_info_with_dtype(&self, dtype: DType) -> TensorInfo {
        let shape = self
            .shape
            .iter()
            .map(|dim| match dim {
                // An explicit `dim_value: 0` is a legitimate static declaration
                // (a genuinely empty axis), not "no shape info" — only a negative
                // (malformed) or altogether absent dim_value stays dynamic/unknown.
                Some(d) if *d >= 0 => Some(*d as usize),
                _ => None,
            })
            .collect();
        TensorInfo {
            name: self.name.clone(),
            dtype,
            shape,
            dim_params: self.dim_params.clone(),
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
    /// `value_info` (wire field 13): shape/dtype metadata for intermediate values.
    pub value_infos: Vec<ValueInfoProto>,
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
    /// Model-local function library (`functions`, wire field 25). Populated by
    /// the same top-level pass as every other `ModelProto` field, so a caller
    /// that already has a parsed `ModelProto` never needs a second scan of the
    /// raw bytes to find these — see `crate::model::inline_local_functions`.
    pub functions: Vec<crate::parser::FunctionProto>,
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
    /// Initialization graph (if present): computes initial values for trainable tensors.
    pub initialization_graph: Option<GraphProto>,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// [stitch S1-3 / a2-20 tail] An explicit `dim_value: 0` must survive the
    /// `ValueInfoProto -> TensorInfo` conversion as `Some(0)` (a real, static,
    /// zero-sized axis) rather than collapsing into `None` (dynamic/unknown).
    /// The wire-level parser already distinguishes these
    /// (`oxionnx-proto/tests/w1_proto_parser.rs::zero_dim_is_static_not_symbolic`);
    /// this covers the conversion step downstream of that parse.
    #[test]
    fn explicit_zero_dim_is_static_not_dynamic() {
        let vi = ValueInfoProto {
            name: "x".to_string(),
            elem_type: dtype_code::FLOAT32,
            shape: vec![None, Some(0), Some(5)],
            dim_params: vec![Some("batch".to_string()), None, None],
        };
        let info = vi.to_tensor_info();
        assert_eq!(
            info.shape,
            vec![None, Some(0), Some(5)],
            "dim_value=0 must map to Some(0), not None"
        );

        let info = vi.try_to_tensor_info().expect("float32 is representable");
        assert_eq!(info.shape, vec![None, Some(0), Some(5)]);
    }

    /// A negative `dim_value` never legitimately occurs on the wire (the parser
    /// only ever stores what it read from a non-negative varint field), but a
    /// hand-built `ValueInfoProto` could still carry one; it must keep mapping
    /// to `None` (dynamic) rather than being reinterpreted as a huge `usize`
    /// via an `as` cast.
    #[test]
    fn negative_dim_still_maps_to_dynamic() {
        let vi = ValueInfoProto {
            name: "y".to_string(),
            elem_type: dtype_code::FLOAT32,
            shape: vec![Some(-1)],
            dim_params: vec![None],
        };
        let info = vi.to_tensor_info();
        assert_eq!(
            info.shape,
            vec![None],
            "negative dim_value must stay dynamic"
        );
    }

    /// A dimension with no `dim_value` at all (`None` in the parsed `shape`
    /// vec) stays dynamic regardless of the >=0 fix, whether or not a
    /// `dim_param` name is attached.
    #[test]
    fn absent_dim_value_still_maps_to_dynamic() {
        let vi = ValueInfoProto {
            name: "z".to_string(),
            elem_type: dtype_code::FLOAT32,
            shape: vec![None, None],
            dim_params: vec![Some("seq_len".to_string()), None],
        };
        let info = vi.to_tensor_info();
        assert_eq!(info.shape, vec![None, None]);
    }
}
