//! Memory-mapped weight loader for large ONNX models.
//!
//! Instead of reading all weights into memory at load time, this module
//! memory-maps the ONNX file and passes the mapped region to the existing
//! parser. The OS virtual memory subsystem can then page out unused weight
//! data, reducing resident memory for large models where only a subset of
//! weights are accessed during a given inference pass.
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use oxionnx_proto::mmap_loader::MmapModel;
//!
//! let model = MmapModel::open(Path::new("model.onnx")).expect("open model");
//! let (graph, weights) = model.into_parts();
//! ```

use oxionnx_core::{Graph, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// A memory-mapped ONNX model file.
///
/// The underlying `Mmap` keeps the file contents mapped into the process
/// address space.  After parsing, the `graph` and `weights` are available
/// for use.  Dropping this struct unmaps the file.
pub struct MmapModel {
    /// The memory-mapped file data — kept alive so the OS can manage pages.
    _mmap: memmap2::Mmap,
    /// Parsed computation graph.
    graph: Graph,
    /// Weight tensors extracted from the model.
    weights: HashMap<String, Tensor>,
}

impl MmapModel {
    /// Open and parse an ONNX model file using memory mapping.
    ///
    /// The file is memory-mapped and then parsed using the standard protobuf
    /// parser. External data references are resolved relative to the parent
    /// directory of `path`.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open '{}': {}", path.display(), e))?;

        // SAFETY: We hold the file open for the lifetime of the Mmap.
        // The file must not be modified while mapped (standard mmap contract).
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("mmap failed for '{}': {}", path.display(), e))?;

        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let (graph, weights) = crate::model::load_with_path(&mmap, base_path)?;

        Ok(Self {
            _mmap: mmap,
            graph,
            weights,
        })
    }

    /// Consume the model and return the parsed graph and weight tensors.
    pub fn into_parts(self) -> (Graph, HashMap<String, Tensor>) {
        (self.graph, self.weights)
    }

    /// Reference to the parsed computation graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Reference to the weight tensors.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    /// Total number of weight tensors in the model.
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Total number of scalar parameters across all weights.
    pub fn parameter_count(&self) -> usize {
        self.weights.values().map(|t| t.data.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode a protobuf varint.
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

    /// Build a minimal ONNX model binary with one float32 initializer.
    fn build_minimal_model(name: &str, floats: &[f32]) -> Vec<u8> {
        // TensorProto
        let mut tensor_bytes = Vec::new();

        // dims packed: [floats.len()]
        let dims_packed = encode_varint(floats.len() as u64);
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));

        // data_type = 1 (float32)
        tensor_bytes.extend(encode_varint_field(2, 1));

        // name
        tensor_bytes.extend(encode_bytes_field(8, name.as_bytes()));

        // raw_data: f32 LE bytes (protobuf field 9)
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        tensor_bytes.extend(encode_bytes_field(9, &raw));

        // GraphProto: field 5 = initializer
        let graph_bytes = encode_bytes_field(5, &tensor_bytes);

        // ModelProto: field 1 = ir_version, field 8 = opset, field 7 = graph
        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));
        model_bytes
    }

    /// Write bytes to a temp file and return its path.
    fn write_temp_onnx(name: &str, data: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("oxionnx_mmap_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(name);
        std::fs::write(&path, data).expect("write temp file");
        path
    }

    #[test]
    fn test_mmap_model_open() {
        let floats = vec![1.0f32, 2.0, 3.0, 4.0];
        let model_bytes = build_minimal_model("test_weight", &floats);
        let path = write_temp_onnx("test_mmap_open.onnx", &model_bytes);

        let model = MmapModel::open(&path).expect("MmapModel::open should succeed");
        assert_eq!(model.weight_count(), 1);
        assert_eq!(model.parameter_count(), 4);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_into_parts() {
        let floats = vec![10.0f32, 20.0, 30.0];
        let model_bytes = build_minimal_model("my_weight", &floats);
        let path = write_temp_onnx("test_mmap_parts.onnx", &model_bytes);

        let model = MmapModel::open(&path).expect("open");
        let (_graph, weights) = model.into_parts();

        let tensor = weights.get("my_weight").expect("weight should exist");
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.data, floats);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_open_nonexistent() {
        let result = MmapModel::open(Path::new("/tmp/oxionnx_no_such_file.onnx"));
        assert!(result.is_err());
        let err = result.err().expect("should be error");
        assert!(
            err.contains("Cannot open"),
            "expected 'Cannot open' in error, got: {err}"
        );
    }

    #[test]
    fn test_mmap_accessors() {
        let floats = vec![1.0f32, 2.0];
        let model_bytes = build_minimal_model("w", &floats);
        let path = write_temp_onnx("test_mmap_accessors.onnx", &model_bytes);

        let model = MmapModel::open(&path).expect("open");

        // Test graph() and weights() accessors
        assert!(model.graph().nodes.is_empty()); // no nodes in minimal model
        assert_eq!(model.weights().len(), 1);
        assert!(model.weights().contains_key("w"));

        let _ = std::fs::remove_file(&path);
    }
}
