use crate::parser;
use crate::types::{AttributeProto, OpsetImport, TensorProto};
use oxionnx_core::Tensor;
use oxionnx_core::{Attributes, Graph, Node, OpKind};
use std::collections::HashMap;
use std::path::Path;

/// Raw metadata extracted from a `ModelProto`, before conversion to the session-layer type.
///
/// Returned alongside graph and weights by [`load_with_metadata`] and
/// [`load_with_metadata_and_path`].
pub struct RawModelMeta {
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub graph_name: String,
    pub ir_version: i64,
    pub opset_imports: Vec<(String, i64)>,
    pub metadata_props: Vec<(String, String)>,
}

/// Supported opset version range (inclusive).
pub const SUPPORTED_OPSET_RANGE: (i64, i64) = (7, 21);

/// Validate the opset version and emit a warning if out of supported range.
fn validate_opset(opset_imports: &[OpsetImport]) {
    for import in opset_imports {
        if import.domain.is_empty() {
            let (min, max) = SUPPORTED_OPSET_RANGE;
            if import.version < min || import.version > max {
                tracing::warn!(
                    opset = import.version,
                    min,
                    max,
                    "model uses opset outside supported range",
                );
            }
        }
    }
}

/// Load an ONNX model file and return (Graph, weight_tensors).
/// External data is NOT supported; use `load_with_path` for models with external data.
pub fn load(bytes: &[u8]) -> Result<(Graph, HashMap<String, Tensor>), String> {
    let (_, graph, weights) = load_with_metadata(bytes)?;
    Ok((graph, weights))
}

/// Load an ONNX model from bytes, resolving external data relative to `base_path`.
pub fn load_with_path(
    bytes: &[u8],
    base_path: &Path,
) -> Result<(Graph, HashMap<String, Tensor>), String> {
    let (_, graph, weights) = load_with_metadata_and_path(bytes, base_path)?;
    Ok((graph, weights))
}

/// Load an ONNX model file and return `(RawModelMeta, Graph, weight_tensors)`.
///
/// External data is NOT supported; use [`load_with_metadata_and_path`] for models
/// that store weights in separate external files.
pub fn load_with_metadata(
    bytes: &[u8],
) -> Result<(RawModelMeta, Graph, HashMap<String, Tensor>), String> {
    let model = parser::parse_model(bytes)?;
    validate_opset(&model.opset_imports);

    let meta = RawModelMeta {
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        graph_name: model.graph.name.clone(),
        ir_version: model.ir_version,
        opset_imports: model
            .opset_imports
            .iter()
            .map(|o| (o.domain.clone(), o.version))
            .collect(),
        metadata_props: model.metadata_props.clone(),
    };

    let graph_proto = model.graph;

    // Collect initializer (weight) tensors first (needed for input_names filter and node attrs).
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for init in &graph_proto.initializers {
        if init.data_location == 1 {
            return Err("External data requires load_with_path()".to_string());
        }
        weights.insert(init.name.clone(), init.to_tensor());
    }

    let (graph, weights_out) = build_graph_and_weights(graph_proto, weights)?;
    Ok((meta, graph, weights_out))
}

/// Internal helper: build Graph + return weights map from a GraphProto and pre-collected
/// weights.  The caller hands ownership of `weights` in; this function clones them for
/// attribute resolution and returns the (possibly augmented) map together with the Graph.
fn build_graph_and_weights(
    graph_proto: crate::types::GraphProto,
    weights: HashMap<String, Tensor>,
) -> Result<(Graph, HashMap<String, Tensor>), String> {
    let mut nodes: Vec<Node> = Vec::with_capacity(graph_proto.nodes.len());
    for np in &graph_proto.nodes {
        let op = OpKind::parse(&np.op_type);
        if let OpKind::Unknown(ref name) = op {
            tracing::debug!(op = %name, "unsupported op, will be skipped");
        }
        let attrs = convert_attributes(&np.attributes, &weights)?;
        nodes.push(Node {
            op,
            name: np.name.clone(),
            inputs: np.inputs.clone(),
            outputs: np.outputs.clone(),
            attrs,
        });
    }

    let input_names: Vec<String> = graph_proto
        .inputs
        .iter()
        .filter(|name| !weights.contains_key(name.as_str()))
        .cloned()
        .collect();

    let input_infos = graph_proto
        .input_value_infos
        .iter()
        .map(|vi| vi.to_tensor_info())
        .collect();
    let output_infos = graph_proto
        .output_value_infos
        .iter()
        .map(|vi| vi.to_tensor_info())
        .collect();

    let graph = Graph {
        name: graph_proto.name,
        nodes,
        input_names,
        output_names: graph_proto.outputs,
        input_infos,
        output_infos,
    };

    Ok((graph, weights))
}

/// Load an ONNX model from bytes (resolving external data relative to `base_path`)
/// and return `(RawModelMeta, Graph, weight_tensors)`.
pub fn load_with_metadata_and_path(
    bytes: &[u8],
    base_path: &Path,
) -> Result<(RawModelMeta, Graph, HashMap<String, Tensor>), String> {
    let model = parser::parse_model(bytes)?;
    validate_opset(&model.opset_imports);

    let meta = RawModelMeta {
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        graph_name: model.graph.name.clone(),
        ir_version: model.ir_version,
        opset_imports: model
            .opset_imports
            .iter()
            .map(|o| (o.domain.clone(), o.version))
            .collect(),
        metadata_props: model.metadata_props.clone(),
    };

    let graph_proto = model.graph;

    // Collect initializer (weight) tensors
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for init in &graph_proto.initializers {
        if init.data_location == 1 {
            let tensor = load_external_tensor(init, base_path)?;
            weights.insert(init.name.clone(), tensor);
        } else {
            weights.insert(init.name.clone(), init.to_tensor());
        }
    }

    let (graph, weights_out) = build_graph_and_weights(graph_proto, weights)?;
    Ok((meta, graph, weights_out))
}

/// Load tensor data from an external file referenced by the TensorProto.
fn load_external_tensor(tensor_proto: &TensorProto, base_path: &Path) -> Result<Tensor, String> {
    use oxionnx_core::tensor::{from_f16_bytes, from_f32_bytes, from_i64_bytes};

    let mut location = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for (key, value) in &tensor_proto.external_data {
        match key.as_str() {
            "location" => location = Some(value.clone()),
            "offset" => {
                offset = value
                    .parse::<u64>()
                    .map_err(|e| format!("Invalid offset '{}': {}", value, e))?;
            }
            "length" => {
                length = Some(
                    value
                        .parse::<u64>()
                        .map_err(|e| format!("Invalid length '{}': {}", value, e))?,
                );
            }
            _ => {} // ignore "checksum" and others
        }
    }

    let location = location.ok_or_else(|| {
        format!(
            "External tensor '{}' missing 'location' field",
            tensor_proto.name
        )
    })?;

    let file_path = base_path.join(&location);
    let file_data = std::fs::read(&file_path).map_err(|e| {
        format!(
            "Cannot read external data file '{}': {}",
            file_path.display(),
            e
        )
    })?;

    let start = offset as usize;
    let end = match length {
        Some(len) => start + len as usize,
        None => file_data.len(),
    };

    if end > file_data.len() {
        return Err(format!(
            "External data for '{}': offset {} + length {} exceeds file size {}",
            tensor_proto.name,
            start,
            end - start,
            file_data.len()
        ));
    }

    let raw_bytes = &file_data[start..end];
    let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();

    match tensor_proto.data_type {
        1 => Ok(from_f32_bytes(raw_bytes, shape)),
        10 => Ok(from_f16_bytes(raw_bytes, shape)),
        7 => Ok(from_i64_bytes(raw_bytes, shape)),
        6 => {
            let data: Vec<f32> = raw_bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect();
            Ok(Tensor::new(data, shape))
        }
        11 => {
            // double (float64)
            let data: Vec<f32> = raw_bytes
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect();
            Ok(Tensor::new(data, shape))
        }
        dt => {
            tracing::warn!(
                tensor = %tensor_proto.name,
                dtype = dt,
                "external tensor: unsupported dtype, returning zeros",
            );
            Ok(Tensor::zeros(&shape))
        }
    }
}

/// Build an oxionnx-core `Graph` from a `GraphProto` and pre-extracted weights.
///
/// This is used by both the batch loader (`load`) and the streaming parser path
/// to convert parsed protobuf structures into the runtime graph representation.
pub fn build_graph(
    graph_proto: &crate::types::GraphProto,
    weights: &HashMap<String, Tensor>,
) -> Result<Graph, String> {
    let mut nodes: Vec<Node> = Vec::with_capacity(graph_proto.nodes.len());
    for np in &graph_proto.nodes {
        let op = OpKind::parse(&np.op_type);
        if let OpKind::Unknown(ref name) = op {
            tracing::debug!(op = %name, "unsupported op, will be skipped");
        }
        let attrs = convert_attributes(&np.attributes, weights)?;
        nodes.push(Node {
            op,
            name: np.name.clone(),
            inputs: np.inputs.clone(),
            outputs: np.outputs.clone(),
            attrs,
        });
    }

    let input_names: Vec<String> = graph_proto
        .inputs
        .iter()
        .filter(|name| !weights.contains_key(name.as_str()))
        .cloned()
        .collect();

    let input_infos = graph_proto
        .input_value_infos
        .iter()
        .map(|vi| vi.to_tensor_info())
        .collect();
    let output_infos = graph_proto
        .output_value_infos
        .iter()
        .map(|vi| vi.to_tensor_info())
        .collect();

    Ok(Graph {
        name: graph_proto.name.clone(),
        nodes,
        input_names,
        output_names: graph_proto.outputs.clone(),
        input_infos,
        output_infos,
    })
}

/// Extract training information from model bytes.
///
/// Returns an empty vector if the model contains no training info.
pub fn extract_training_info(bytes: &[u8]) -> Result<Vec<crate::types::TrainingInfo>, String> {
    let model = parser::parse_model(bytes)?;
    Ok(model.training_info)
}

fn convert_attributes(
    attrs: &[AttributeProto],
    weights: &HashMap<String, Tensor>,
) -> Result<Attributes, String> {
    let mut a = Attributes::default();
    for attr in attrs {
        let name = attr.name.clone();
        let v = &attr.value;
        // attr_type: 1=f, 2=i, 3=s, 4=t, 6=floats, 7=ints
        match v.attr_type {
            1 => {
                a.floats.insert(name, v.f);
            }
            2 => {
                a.ints.insert(name.clone(), v.i);
            }
            3 => {
                a.strings.insert(name, v.s.clone());
            }
            4 => {
                if let Some(ref tp) = v.t {
                    a.tensors.insert(name, tp.to_tensor());
                }
            }
            6 => {
                a.float_lists.insert(name, v.floats.clone());
            }
            7 => {
                a.int_lists.insert(name, v.ints.clone());
            }
            0 => {
                // attr_type=0 means unset; infer from which field is populated
                if v.f != 0.0 {
                    a.floats.insert(name.clone(), v.f);
                }
                if v.i != 0 {
                    a.ints.insert(name.clone(), v.i);
                }
                if !v.s.is_empty() {
                    a.strings.insert(name.clone(), v.s.clone());
                }
                if !v.floats.is_empty() {
                    a.float_lists.insert(name.clone(), v.floats.clone());
                }
                if !v.ints.is_empty() {
                    a.int_lists.insert(name.clone(), v.ints.clone());
                }
                if let Some(ref tp) = v.t {
                    a.tensors.insert(name, tp.to_tensor());
                }
            }
            _ => {}
        }
    }
    let _ = weights; // available for future use
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode a varint into bytes.
    fn encode_varint(mut val: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                buf.push(byte);
                break;
            } else {
                buf.push(byte | 0x80);
            }
        }
        buf
    }

    fn encode_varint_field(field: u32, val: u64) -> Vec<u8> {
        let tag = field << 3;
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(val));
        buf
    }

    fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
        let tag = (field << 3) | 2;
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(data.len() as u64));
        buf.extend(data);
        buf
    }

    /// Build a minimal ONNX model binary with one initializer tensor.
    fn build_model_with_initializer(tensor_proto_bytes: &[u8]) -> Vec<u8> {
        // GraphProto: field 5 = initializer (TensorProto)
        let graph_bytes = encode_bytes_field(5, tensor_proto_bytes);
        // ModelProto: field 1 = ir_version, field 7 = graph, field 8 = opset
        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));
        model_bytes
    }

    #[test]
    fn test_load_with_external_data() {
        // Create a temp directory with an external data file
        let tmp_dir = std::env::temp_dir().join("oxionnx_test_ext_data");
        let _ = std::fs::create_dir_all(&tmp_dir);

        // Write 8 floats (2x4 tensor) as raw f32 LE bytes
        let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let raw_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Put 16 bytes of padding then our data
        let offset = 16u64;
        let mut file_data = vec![0u8; offset as usize];
        file_data.extend(&raw_bytes);
        let data_file = tmp_dir.join("weights.bin");
        std::fs::write(&data_file, &file_data).expect("write external data");

        // Build TensorProto with external data
        let mut tensor_bytes = Vec::new();
        // dims packed: [2, 4]
        let mut dims_packed = encode_varint(2);
        dims_packed.extend(encode_varint(4));
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
        // data_type = 1 (float32)
        tensor_bytes.extend(encode_varint_field(2, 1));
        // name = "my_weight"
        tensor_bytes.extend(encode_bytes_field(8, b"my_weight"));
        // external_data entries (field 13, repeated StringStringEntryProto)
        let mut entry_loc = encode_bytes_field(1, b"location");
        entry_loc.extend(encode_bytes_field(2, b"weights.bin"));
        tensor_bytes.extend(encode_bytes_field(13, &entry_loc));

        let mut entry_off = encode_bytes_field(1, b"offset");
        entry_off.extend(encode_bytes_field(2, b"16"));
        tensor_bytes.extend(encode_bytes_field(13, &entry_off));

        let mut entry_len = encode_bytes_field(1, b"length");
        entry_len.extend(encode_bytes_field(2, b"32")); // 8 * 4 bytes
        tensor_bytes.extend(encode_bytes_field(13, &entry_len));
        // data_location = 1 (field 14, enum)
        tensor_bytes.extend(encode_varint_field(14, 1));

        let model_bytes = build_model_with_initializer(&tensor_bytes);

        let (_graph, weights) =
            load_with_path(&model_bytes, &tmp_dir).expect("load_with_path should succeed");

        let tensor = weights.get("my_weight").expect("weight should exist");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data, floats);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_load_rejects_external_data_without_path() {
        // Build a TensorProto with data_location=1
        let mut tensor_bytes = Vec::new();
        let mut dims_packed = encode_varint(2);
        dims_packed.extend(encode_varint(2));
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
        tensor_bytes.extend(encode_varint_field(2, 1));
        tensor_bytes.extend(encode_bytes_field(8, b"ext_weight"));
        tensor_bytes.extend(encode_varint_field(14, 1));

        let model_bytes = build_model_with_initializer(&tensor_bytes);

        let result = load(&model_bytes);
        assert!(result.is_err());
        let err = result.expect_err("should be error");
        assert!(
            err.contains("External data requires load_with_path()"),
            "got: {err}"
        );
    }

    #[test]
    fn test_opset_validation_in_range() {
        // Opset 13 is in supported range, should not error
        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));

        let result = load(&model_bytes);
        assert!(result.is_ok());
    }
}
